#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

struct ModArg {
  const float *p;
  int64_t stride;
  int rows_per;
};

constexpr int kWarps = 4;

enum : int { OUT_NONE = 0, OUT_BF16 = 1, OUT_F32 = 2, OUT_FP8 = 3 };

__device__ __forceinline__ void load8(const float *p, float (&v)[8]) {
  const float4 a = *reinterpret_cast<const float4 *>(p), b = *reinterpret_cast<const float4 *>(p + 4);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void ldg8(const float *p, float (&v)[8]) {
  const float4 a = __ldg(reinterpret_cast<const float4 *>(p)), b = __ldg(reinterpret_cast<const float4 *>(p + 4));
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void store8(float *p, const float (&v)[8]) {
  *reinterpret_cast<float4 *>(p) = make_float4(v[0], v[1], v[2], v[3]);
  *reinterpret_cast<float4 *>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
}
__device__ __forceinline__ float warp_sum(float s) {
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  return s;
}

template <int NIT, bool RES, int OUT, bool CAM = false, bool AFF = false>
__global__ void __launch_bounds__(kWarps * 32) fused_mod_kernel(float *__restrict__ x, const __nv_bfloat16 *__restrict__ y,
                                                                ModArg gate, ModArg shift, ModArg scale, ModArg aux,
                                                                void *__restrict__ out, int M, float eps, float inv_scale) {
  constexpr int C = NIT * 256;
  const int lane = threadIdx.x & 31;
  const int m = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (m >= M) return;
  float *xr = x + (size_t)m * C + lane * 8;
  float v[NIT][8];
#pragma unroll
  for (int j = 0; j < NIT; ++j) load8(xr + j * 256, v[j]);

  if constexpr (CAM) {
    const float *ar = gate.p + (size_t)(m / gate.rows_per) * gate.stride + lane * 8;
    const float *tr = aux.p + (size_t)(m / aux.rows_per) * aux.stride + lane * 8;
#pragma unroll
    for (int j = 0; j < NIT; ++j) {
      float a[8], t[8];
      ldg8(ar + j * 256, a);
      ldg8(tr + j * 256, t);
#pragma unroll
      for (int k = 0; k < 8; ++k) v[j][k] = __fadd_rn(__fmul_rn(a[k], v[j][k]), t[k]);
      store8(xr + j * 256, v[j]);
    }
  } else if constexpr (RES) {
    const __nv_bfloat16 *yr = y + (size_t)m * C + lane * 8;
    const float *gr = gate.p ? gate.p + (size_t)(m / gate.rows_per) * gate.stride + lane * 8 : nullptr;
#pragma unroll
    for (int j = 0; j < NIT; ++j) {
      const uint4 raw = __ldcs(reinterpret_cast<const uint4 *>(yr + j * 256));
      const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&raw);
      float yf[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 f = __bfloat1622float2(h[k]);
        yf[2 * k] = f.x; yf[2 * k + 1] = f.y;
      }
      if (gr) {
        float g[8];
        ldg8(gr + j * 256, g);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[j][k] = __fadd_rn(v[j][k], __fmul_rn(yf[k], g[k]));
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) v[j][k] = __fadd_rn(v[j][k], yf[k]);
      }
      store8(xr + j * 256, v[j]);
    }
  }
  if constexpr (OUT != OUT_NONE) {
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < NIT; ++j)
#pragma unroll
      for (int k = 0; k < 8; ++k) s += v[j][k];
    const float mean = warp_sum(s) * (1.f / C);
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < NIT; ++j)
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float d = v[j][k] - mean;
        ss = fmaf(d, d, ss);
      }
    const float rstd = rsqrtf(warp_sum(ss) * (1.f / C) + eps);
    const float *shr = shift.p + (size_t)(m / shift.rows_per) * shift.stride + lane * 8;
    const float *scr = scale.p + (size_t)(m / scale.rows_per) * scale.stride + lane * 8;
#pragma unroll
    for (int j = 0; j < NIT; ++j) {
      float sh[8], sc[8], o[8];
      ldg8(shr + j * 256, sh);
      ldg8(scr + j * 256, sc);
#pragma unroll
      for (int k = 0; k < 8; ++k)
        o[k] = AFF ? __fadd_rn(__fmul_rn(__fmul_rn(v[j][k] - mean, rstd), sc[k]), sh[k])
                   : __fadd_rn(__fmul_rn(__fmul_rn(v[j][k] - mean, rstd), __fadd_rn(1.f, sc[k])), sh[k]);
      const size_t off = (size_t)m * C + j * 256 + lane * 8;
      if constexpr (OUT == OUT_BF16) {
        uint4 pk;
        __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&pk);
#pragma unroll
        for (int k = 0; k < 4; ++k) h[k] = __floats2bfloat162_rn(o[2 * k], o[2 * k + 1]);
        *reinterpret_cast<uint4 *>(reinterpret_cast<__nv_bfloat16 *>(out) + off) = pk;
      } else if constexpr (OUT == OUT_F32) {
        store8(reinterpret_cast<float *>(out) + off, o);
      } else {
        uint2 pk;
        uint16_t *q = reinterpret_cast<uint16_t *>(&pk);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          q[k] = __nv_cvt_float2_to_fp8x2(make_float2(__fmul_rn(o[2 * k], inv_scale), __fmul_rn(o[2 * k + 1], inv_scale)),
                                          __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<uint2 *>(reinterpret_cast<uint8_t *>(out) + off) = pk;
      }
    }
  }
}

template <int NIT>
__global__ void __launch_bounds__(kWarps * 32) rms_bf16_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ w,
                                                               __nv_bfloat16 *__restrict__ out, int M, float eps) {
  constexpr int C = NIT * 256;
  const int lane = threadIdx.x & 31;
  const int m = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (m >= M) return;
  float v[NIT][8];
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < NIT; ++j) {
    const uint4 raw = *reinterpret_cast<const uint4 *>(x + (size_t)m * C + j * 256 + lane * 8);
    const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&raw);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 f = __bfloat1622float2(h[k]);
      v[j][2 * k] = f.x; v[j][2 * k + 1] = f.y;
      s = fmaf(f.x, f.x, s);
      s = fmaf(f.y, f.y, s);
    }
  }
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(warp_sum(s), (float)C), eps));
#pragma unroll
  for (int j = 0; j < NIT; ++j) {
    const uint4 wr = __ldg(reinterpret_cast<const uint4 *>(w + j * 256 + lane * 8));
    const __nv_bfloat162 *g = reinterpret_cast<const __nv_bfloat162 *>(&wr);
    uint4 pk;
    __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&pk);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 gf = __bfloat1622float2(g[k]);
      h[k] = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(v[j][2 * k], r), gf.x), __fmul_rn(__fmul_rn(v[j][2 * k + 1], r), gf.y));
    }
    *reinterpret_cast<uint4 *>(out + (size_t)m * C + j * 256 + lane * 8) = pk;
  }
}

ModArg mod_arg(const c10::optional<torch::Tensor> &t, int64_t C, int64_t M, const char *name) {
  if (!t.has_value()) return {nullptr, 0, 1};
  const auto &a = *t;
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kFloat32, name, " must be a CUDA fp32 tensor");
  TORCH_CHECK(a.dim() >= 1 && a.size(-1) == C && a.stride(-1) == 1, name, " must be [..., C] with unit inner stride");
  const int64_t R = a.numel() / C;
  TORCH_CHECK(R >= 1 && M % R == 0, name, ": rows (", R, ") must divide M (", M, ")");
  int64_t stride = 0, expect = -1;
  for (int d = a.dim() - 2; d >= 0; --d) {
    if (a.size(d) == 1) continue;
    if (expect < 0) { stride = a.stride(d); expect = stride * a.size(d); }
    else { TORCH_CHECK(a.stride(d) == expect, name, ": leading dims must collapse to a single row stride"); expect *= a.size(d); }
  }
  TORCH_CHECK(stride % 4 == 0 && reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0, name, " must be 16-byte aligned per row");
  return {a.data_ptr<float>(), R == 1 ? 0 : stride, (int)(M / R)};
}

template <bool RES, int OUT, bool CAM = false, bool AFF = false>
void launch(torch::Tensor &x, const __nv_bfloat16 *y, ModArg gate, ModArg shift, ModArg scale, void *out, int64_t C, int64_t M,
            double eps, float inv_s, ModArg aux = {nullptr, 0, 1}) {
  const dim3 grid((unsigned)((M + kWarps - 1) / kWarps)), block(kWarps * 32);
  auto st = at::cuda::getCurrentCUDAStream();
  float *xp = x.data_ptr<float>();
#define FM_CASE(N) \
  case N: fused_mod_kernel<N, RES, OUT, CAM, AFF><<<grid, block, 0, st>>>(xp, y, gate, shift, scale, aux, out, (int)M, (float)eps, inv_s); break;
  switch (C / 256) {
    FM_CASE(4) FM_CASE(6) FM_CASE(8) FM_CASE(12) FM_CASE(16) FM_CASE(20)
    default: TORCH_CHECK(false, "unsupported C=", C, " (supported: 1024, 1536, 2048, 3072, 4096, 5120)");
  }
#undef FM_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::pair<int64_t, int64_t> check_x(const torch::Tensor &x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous(), "x must be contiguous CUDA fp32");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");
  const int64_t C = x.size(-1);
  TORCH_CHECK(C % 256 == 0, "C must be a multiple of 256");
  const int64_t M = x.numel() / C;
  TORCH_CHECK(M < (1ll << 31), "too many rows");
  return {C, M};
}

const __nv_bfloat16 *check_y(const torch::Tensor &x, const torch::Tensor &y) {
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kBFloat16 && y.is_contiguous() && y.numel() == x.numel(),
              "y must be contiguous CUDA bf16 with x's shape");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0, "y must be 16-byte aligned");
  return reinterpret_cast<const __nv_bfloat16 *>(y.data_ptr());
}

template <bool RES>
torch::Tensor run_norm(torch::Tensor x, const __nv_bfloat16 *y, ModArg g, const torch::Tensor &shift, const torch::Tensor &scale,
                       double eps, torch::ScalarType dt, double s) {
  const auto [C, M] = check_x(x);
  const ModArg sh = mod_arg(shift, C, M, "shift"), sc = mod_arg(scale, C, M, "scale");
  auto out = torch::empty(x.sizes(), x.options().dtype(dt));
  if (dt == torch::kBFloat16) launch<RES, OUT_BF16>(x, y, g, sh, sc, out.data_ptr(), C, M, eps, 1.0);
  else if (dt == torch::kFloat32) launch<RES, OUT_F32>(x, y, g, sh, sc, out.data_ptr(), C, M, eps, 1.0);
  else if (dt == torch::kFloat8_e4m3fn) launch<RES, OUT_FP8>(x, y, g, sh, sc, out.data_ptr(), C, M, eps, (float)(1.0 / s));
  else TORCH_CHECK(false, "out_dtype must be bfloat16, float32 or float8_e4m3fn");
  return out;
}

}

torch::Tensor mod_norm(torch::Tensor x, torch::Tensor shift, torch::Tensor scale, double eps, c10::optional<at::ScalarType> out_dtype) {
  const auto dt = out_dtype.value_or(torch::kBFloat16);
  TORCH_CHECK(dt != torch::kFloat8_e4m3fn, "use mod_norm_fp8 for e4m3 output");
  return run_norm<false>(x, nullptr, ModArg{nullptr, 0, 1}, shift, scale, eps, dt, 1.0);
}

torch::Tensor mod_norm_fp8(torch::Tensor x, torch::Tensor shift, torch::Tensor scale, double eps, double s) {
  TORCH_CHECK(s > 0, "fp8 scale must be positive");
  return run_norm<false>(x, nullptr, ModArg{nullptr, 0, 1}, shift, scale, eps, torch::kFloat8_e4m3fn, s);
}

torch::Tensor gated_residual(torch::Tensor x, torch::Tensor y, c10::optional<torch::Tensor> gate) {
  const auto [C, M] = check_x(x);
  const __nv_bfloat16 *yp = check_y(x, y);
  const ModArg g = mod_arg(gate, C, M, "gate");
  launch<true, OUT_NONE>(x, yp, g, g, g, nullptr, C, M, 0.0, 1.0);
  return x;
}

torch::Tensor residual_then_mod_norm(torch::Tensor x, torch::Tensor y, c10::optional<torch::Tensor> gate, torch::Tensor shift,
                                     torch::Tensor scale, double eps, c10::optional<at::ScalarType> out_dtype) {
  const auto [C, M] = check_x(x);
  const auto dt = out_dtype.value_or(torch::kBFloat16);
  TORCH_CHECK(dt != torch::kFloat8_e4m3fn, "use residual_then_mod_norm_fp8 for e4m3 output");
  return run_norm<true>(x, check_y(x, y), mod_arg(gate, C, M, "gate"), shift, scale, eps, dt, 1.0);
}

torch::Tensor residual_then_mod_norm_fp8(torch::Tensor x, torch::Tensor y, c10::optional<torch::Tensor> gate, torch::Tensor shift,
                                         torch::Tensor scale, double eps, double s) {
  const auto [C, M] = check_x(x);
  TORCH_CHECK(s > 0, "fp8 scale must be positive");
  return run_norm<true>(x, check_y(x, y), mod_arg(gate, C, M, "gate"), shift, scale, eps, torch::kFloat8_e4m3fn, s);
}

torch::Tensor cam_ln(torch::Tensor x, c10::optional<torch::Tensor> a, c10::optional<torch::Tensor> t, torch::Tensor w,
                     torch::Tensor b, double eps) {
  const auto [C, M] = check_x(x);
  TORCH_CHECK(a.has_value() == t.has_value(), "a and t must both be given or both be None");
  auto out = torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
  const ModArg wa = mod_arg(w, C, M, "w"), ba = mod_arg(b, C, M, "b");
  if (a.has_value())
    launch<false, OUT_BF16, true, true>(x, nullptr, mod_arg(a, C, M, "a"), ba, wa, out.data_ptr(), C, M, eps, 1.0, mod_arg(t, C, M, "t"));
  else
    launch<false, OUT_BF16, false, true>(x, nullptr, ModArg{nullptr, 0, 1}, ba, wa, out.data_ptr(), C, M, eps, 1.0);
  return out;
}

torch::Tensor rms_bf16(torch::Tensor x, torch::Tensor w, double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous(), "x must be contiguous CUDA bf16");
  const int64_t C = x.size(-1), M = x.numel() / C;
  TORCH_CHECK(C % 256 == 0 && M < (1ll << 31), "C must be a multiple of 256");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kBFloat16 && w.is_contiguous() && w.numel() == C, "w must be CUDA bf16 [C]");
  auto out = torch::empty_like(x);
  const dim3 grid((unsigned)((M + kWarps - 1) / kWarps)), block(kWarps * 32);
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), wp = reinterpret_cast<const __nv_bfloat16 *>(w.data_ptr());
  auto op = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr());
#define RB_CASE(N) case N: rms_bf16_kernel<N><<<grid, block, 0, st>>>(xp, wp, op, (int)M, (float)eps); break;
  switch (C / 256) {
    RB_CASE(4) RB_CASE(6) RB_CASE(8) RB_CASE(12) RB_CASE(16) RB_CASE(20)
    default: TORCH_CHECK(false, "unsupported C=", C);
  }
#undef RB_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mod_norm", &mod_norm, "LayerNorm(x, no affine) * (1 + scale) + shift -> bf16 (or fp32)", py::arg("x"), py::arg("shift"),
        py::arg("scale"), py::arg("eps") = 1e-6, py::arg("out_dtype") = py::none());
  m.def("mod_norm_fp8", &mod_norm_fp8, "mod_norm -> e4m3 sat(v / s)", py::arg("x"), py::arg("shift"), py::arg("scale"),
        py::arg("eps"), py::arg("s"));
  m.def("gated_residual", &gated_residual, "x += y * gate in place (gate=None: x += y); returns x", py::arg("x"), py::arg("y"),
        py::arg("gate") = py::none());
  m.def("residual_then_mod_norm", &residual_then_mod_norm, "x += y * gate in place, then mod_norm(x) -> bf16 (or fp32)",
        py::arg("x"), py::arg("y"), py::arg("gate"), py::arg("shift"), py::arg("scale"), py::arg("eps") = 1e-6,
        py::arg("out_dtype") = py::none());
  m.def("residual_then_mod_norm_fp8", &residual_then_mod_norm_fp8, "x += y * gate in place, then mod_norm(x) -> e4m3 sat(v / s)",
        py::arg("x"), py::arg("y"), py::arg("gate"), py::arg("shift"), py::arg("scale"), py::arg("eps"), py::arg("s"));
  m.def("cam_ln", &cam_ln, "x = a * x + t in place (if given), then LayerNorm(x) * w + b -> bf16", py::arg("x"), py::arg("a"),
        py::arg("t"), py::arg("w"), py::arg("b"), py::arg("eps"));
  m.def("rms_bf16", &rms_bf16, "bf16(x * rsqrt(mean(x^2) + eps) * w) with fp32 intermediate", py::arg("x"), py::arg("w"),
        py::arg("eps"));
}
