#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cuda/functional>
#include <cstdint>

namespace {
struct Tensor {
  void *p;
  int type;
  __device__ float load(int64_t i) const {
    if (type == 0) return static_cast<float *>(p)[i];
    if (type == 1) return __bfloat162float(static_cast<__nv_bfloat16 *>(p)[i]);
    if (type == 2) return __half2float(static_cast<half *>(p)[i]);
    if (type == 5) return static_cast<bool *>(p)[i] ? 1.f : 0.f;
    return static_cast<int8_t *>(p)[i];
  }
  __device__ void load8(int64_t i, float *v) const {
    if (type == 0 && (reinterpret_cast<uintptr_t>(static_cast<float *>(p) + i) & 15) == 0) {
      const float4 a = *reinterpret_cast<const float4 *>(static_cast<float *>(p) + i);
      const float4 b = *reinterpret_cast<const float4 *>(static_cast<float *>(p) + i + 4);
      v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
      v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else if ((type == 1 || type == 2) && (reinterpret_cast<uintptr_t>(static_cast<uint16_t *>(p) + i) & 15) == 0) {
      const uint4 raw = *reinterpret_cast<const uint4 *>(static_cast<uint16_t *>(p) + i);
#pragma unroll
      for (int k = 0; k < 8; ++k)
        v[k] = type == 1 ? __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(&raw)[k])
                        : __half2float(reinterpret_cast<const half *>(&raw)[k]);
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) v[k] = load(i + k);
    }
  }
  __device__ float round(float v) const {
    if (type == 1) return __bfloat162float(__float2bfloat16_rn(v));
    if (type == 2) return __half2float(__float2half_rn(v));
    return v;
  }
  __device__ void store(int64_t i, float v) const {
    if (type == 0) static_cast<float *>(p)[i] = v;
    else if (type == 1) static_cast<__nv_bfloat16 *>(p)[i] = __float2bfloat16_rn(v);
    else if (type == 2) static_cast<half *>(p)[i] = __float2half_rn(v);
    else static_cast<uint8_t *>(p)[i] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  }
};
Tensor tensor(const torch::Tensor &t) {
  int type = t.scalar_type() == at::kFloat ? 0 : t.scalar_type() == at::kBFloat16 ? 1 :
      t.scalar_type() == at::kHalf ? 2 : t.scalar_type() == at::kChar ? 3 :
      t.scalar_type() == at::kFloat8_e4m3fn ? 4 : t.scalar_type() == at::kBool ? 5 : -1;
  TORCH_CHECK(t.is_cuda() && type >= 0, "unsupported CUDA glue tensor dtype");
  return {t.data_ptr(), type};
}
void same(const torch::Tensor &x, const torch::Tensor &y) {
  TORCH_CHECK(x.device() == y.device(), "glue tensors must share a CUDA device");
}
void floating(const torch::Tensor &t) {
  TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kHalf,
      "expected float32/bfloat16/float16");
}
void dense(const torch::Tensor &t) {
  floating(t);
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "expected contiguous CUDA tensor");
}
__device__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ float divide(float x, float y) {
  float z;
  asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(z) : "f"(x), "f"(y));
  return z;
}
__device__ float power_scale(float x) {
  float y, z;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  y = ceilf(y);
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(z) : "f"(y));
  return z;
}
template <int N, bool MOD, bool RES>
__global__ void norm_kernel(Tensor x, Tensor w, Tensor a, Tensor y, Tensor residual, Tensor added,
    int c, int rows, int rows_per, int64_t ws, int64_t as, float eps, float inv, bool fp8, bool norm_fp32) {
  using Reduce = cub::WarpReduce<float>;
  __shared__ typename Reduce::TempStorage scratch[4];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, row = blockIdx.x * 4 + warp;
  if (row >= rows) return;
  const int64_t base = int64_t(row) * c;
  float v[N * 8], s = 0.f;
#pragma unroll 1
  for (int j = 0; j < N; ++j) {
    const int d = j * 256 + lane * 8;
    if (d + 7 < c) x.load8(base + d, v + j * 8);
    else {
#pragma unroll
      for (int k = 0; k < 8; ++k) v[j * 8 + k] = d + k < c ? x.load(base + d + k) : 0.f;
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float &f = v[j * 8 + k];
      if constexpr (RES) {
        if (d + k < c) {
          f = x.round(__fadd_rn(f, residual.load(base + d + k)));
          added.store(base + d + k, f);
        }
      }
      s = __fadd_rn(s, MOD ? f : __fmul_rn(f, f));
    }
  }
  s = Reduce(scratch[warp]).Sum(s);
  const float m = __fmul_rn(__shfl_sync(0xffffffff, s, 0), 1.f / c);
  float r;
  if constexpr (MOD) {
    s = 0.f;
#pragma unroll 1
    for (int j = 0; j < N * 8; ++j) {
      const float z = __fsub_rn(v[j], m);
      if ((j / 8) * 256 + lane * 8 + j % 8 < c) s = __fadd_rn(s, __fmul_rn(z, z));
    }
    s = Reduce(scratch[warp]).Sum(s);
    r = rsqrtf(__fadd_rn(__fmul_rn(__shfl_sync(0xffffffff, s, 0), 1.f / c), eps));
  } else r = rsqrtf(__fadd_rn(m, eps));
#pragma unroll 1
  for (int j = 0; j < N * 8; ++j) {
    const int d = (j / 8) * 256 + lane * 8 + j % 8;
    if (d >= c) continue;
    float z = __fmul_rn(MOD ? __fsub_rn(v[j], m) : v[j], r);
    if (!MOD || !norm_fp32) z = x.round(z);
    if constexpr (MOD) {
      const int batch = row / rows_per;
      const float gain = a.round(__fadd_rn(1.f, a.load(int64_t(batch) * as + d)));
      z = __fmul_rn(z, gain);
      if (!norm_fp32 && x.type == 1 && a.type == 1) z = bf(z);
      z = __fadd_rn(z, w.load(int64_t(batch) * ws + d));
      if (!norm_fp32 && x.type == 1 && a.type == 1 && w.type == 1) z = bf(z);
      if (fp8) z = __fmul_rn(bf(z), inv);
    } else z = __fmul_rn(z, w.load(d));
    y.store(base + d, z);
  }
}
template <bool MOD, bool RES>
void norm_launch(torch::Tensor x, torch::Tensor w, torch::Tensor a, torch::Tensor y,
    torch::Tensor residual, torch::Tensor added, double eps, double activation_scale, bool norm_fp32 = false) {
  dense(x); floating(w); floating(a);
  same(x, w); same(x, a); same(x, y); same(x, residual);
  TORCH_CHECK(x.dim() >= 1, "norm input must have a channel dimension");
  const int c = x.size(-1);
  TORCH_CHECK(c > 0 && c <= 8192, "norm channel count must be 1..8192");
  const int rows = x.numel() / c;
  const int rows_per = MOD && x.dim() >= 2 ? x.size(1) : 1;
  TORCH_CHECK(!MOD || (x.dim() == 3 && w.dim() == 2 && a.dim() == 2 && w.size(0) == x.size(0)
      && a.size(0) == x.size(0) && w.size(1) == c && a.size(1) == c && w.stride(1) == 1 && a.stride(1) == 1),
      "modulation requires x[B,L,C], shift/scale[B,C]");
  TORCH_CHECK(MOD || (w.is_contiguous() && w.numel() == c), "RMS weight must contain C contiguous elements");
  TORCH_CHECK(!RES || (residual.sizes() == x.sizes() && residual.is_contiguous()), "residual shape/strides must match x");
  if (!rows) return;
  auto st = at::cuda::getCurrentCUDAStream();
#define NORM(N) norm_kernel<N, MOD, RES><<<(rows + 3) / 4, 128, 0, st>>>(tensor(x), tensor(w), tensor(a), tensor(y), tensor(residual), tensor(added), c, rows, rows_per, w.stride(0), a.stride(0), float(eps), activation_scale ? float(1. / activation_scale) : 1.f, activation_scale != 0, norm_fp32)
  if (c <= 256) { NORM(1); }
  else if (c <= 512) { NORM(2); }
  else if (c <= 1024) { NORM(4); }
  else if (c <= 2048) { NORM(8); }
  else if (c <= 4096) { NORM(16); }
  else { NORM(32); }
#undef NORM
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void rope_kernel(Tensor x, Tensor w, Tensor u, Tensor cos, Tensor sin, Tensor out,
    int h, int l, int rows, float eps, bool dec, bool rotary) {
  using Reduce = cub::WarpReduce<float, 16>;
  __shared__ typename Reduce::TempStorage scratch[8];
  const int lane = threadIdx.x & 15, group = threadIdx.x >> 4, row = blockIdx.x * 8 + group;
  if (row >= rows) return;
  const unsigned mask = (threadIdx.x & 16) ? 0xffff0000u : 0xffffu;
  const int64_t base = int64_t(row) * 128;
  float v[8], n[8], sum = 0.f;
  x.load8(base + lane * 8, v);
#pragma unroll
  for (int k = 0; k < 8; ++k) sum = __fadd_rn(sum, __fmul_rn(v[k], v[k]));
  float rt, rh;
  if (dec) {
    float t = Reduce(scratch[group]).Sum(lane < 8 ? sum : 0.f);
    rt = rsqrtf(__fadd_rn(__fmul_rn(__shfl_sync(mask, t, 0, 16), 1.f / 64), eps));
    float hw = Reduce(scratch[group]).Sum(lane >= 8 ? sum : 0.f);
    rh = rsqrtf(__fadd_rn(__fmul_rn(__shfl_sync(mask, hw, 0, 16), 1.f / 64), eps));
  } else {
    sum = Reduce(scratch[group]).Sum(sum);
    rt = rh = rsqrtf(__fadd_rn(__fmul_rn(__shfl_sync(mask, sum, 0, 16), 1.f / 128), eps));
  }
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int d = lane * 8 + k;
    float z = __fmul_rn(v[k], lane < 8 ? rt : rh);
    if (!dec) z = x.round(z);
    z = __fmul_rn(z, dec && d >= 64 ? u.load(d - 64) : w.load(dec ? d % 64 : d));
    if (!dec) {
      if (w.type != 0) z = x.round(z);
      z = out.round(z);
    }
    n[k] = z;
  }
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int d = lane * 8 + k;
    float z = n[k];
    if (rotary) {
      const int partner_lane = dec ? lane ^ (d < 64 ? 4 : 2) : lane;
      const float partner = __shfl_sync(mask, n[dec ? k : k ^ 1], partner_lane, 16);
      const bool negative = dec ? (d < 64 ? d < 32 : d % 32 < 16) : (d & 1) == 0;
      const int64_t table = int64_t(row / h % l) * 128 + d;
      z = __fadd_rn(__fmul_rn(z, cos.load(table)), __fmul_rn(negative ? -partner : partner, sin.load(table)));
    }
    out.store(base + d, z);
  }
}
struct Source {
  Tensor cache, scale, image, text;
  const int64_t *index;
  int h, hist, ni, nk;
  int64_t cs, ss;
  __device__ float load(int row, int head, int d) const {
    if (row >= nk) return 0.f;
    const int64_t token = index[row];
    if (token < hist) {
      float x = cache.load(token * cs + head * 128 + d);
      return cache.type == 3 ? bf(__fmul_rn(x, scale.load(token * ss + head))) : x;
    }
    if (token < hist + ni) return image.load((token - hist) * h * 128 + head * 128 + d);
    return text.load((token - hist - ni) * h * 128 + head * 128 + d);
  }
};
__global__ void indexed_kernel(Source src, uint8_t *out, float *scales, const float *mean,
    int nb, int first, bool value, int repeat) {
  using Reduce = cub::BlockReduce<unsigned, 256>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ float scale;
  const int t = threadIdx.x, block = blockIdx.x + first, head = blockIdx.y, sh = head / repeat;
  float x[32];
  unsigned mx = 0;
#pragma unroll 1
  for (int j = 0; j < 32; ++j) {
    const int i = t + j * 256, row = block * 64 + i / 128, d = i % 128;
    float v = src.load(row, sh, d);
    if (!value) v = row < src.nk ? __fsub_rn(v, mean[sh * 128 + d]) : 0.f;
    x[j] = v;
    mx = max(mx, __float_as_uint(fabsf(v)));
  }
  mx = Reduce(storage).Reduce(mx, cuda::maximum<>{});
  if (!t) {
    scale = fmaxf(__fmul_rn(__uint_as_float(mx), value ? 1.f / 448 : 1.f / 127), 1.e-12f);
    if (value) scale = power_scale(scale);
    scales[head * nb + block] = scale;
  }
  __syncthreads();
  uint8_t *dst = out + (int64_t(head) * nb + block) * 8192;
#pragma unroll 1
  for (int j = 0; j < 32; ++j) {
    const int i = t + j * 256, row = i / 128, d = i % 128;
    const float q = divide(x[j], scale);
    if (value) {
      const int shuffled = (row & 48) + ((row & 6) << 1) + (row & 1) + ((row & 8) >> 2);
      dst[d * 64 + shuffled] = __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
    } else dst[i] = uint8_t(int8_t(__float2int_rn(fminf(fmaxf(q, -127.f), 127.f))));
  }
}
__global__ void history_kernel(Source src, float *out, int n) {
  const int d = threadIdx.x;
  float sums[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    float sum = 0.f;
#pragma unroll 1
    for (int r = w; r < 128; r += 4) {
      const int row = blockIdx.y * 128 + r;
      float v = 0.f;
      if (row < n) {
        const int64_t token = src.index[row];
        v = src.cache.load(token * src.cs + blockIdx.x * 128 + d);
        if (src.cache.type == 3) v = bf(__fmul_rn(v, src.scale.load(token * src.ss + blockIdx.x)));
      }
      sum = r == w ? v : __fadd_rn(sum, v);
    }
    sums[w] = sum;
  }
  const float v = __fadd_rn(__fadd_rn(sums[0], sums[2]), __fadd_rn(sums[1], sums[3]));
  out[(int64_t(blockIdx.y) * src.h + blockIdx.x) * 128 + d] = divide(v, float(n));
}
__global__ void camera_kernel(Tensor x, Tensor frames, Tensor mask, Tensor out, int c, int t,
    int spatial, int64_t total, bool masked) {
  const int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
  if (i >= total) return;
  const int64_t frame = i / c / spatial % t, batch = i / (int64_t(c) * spatial * t);
  float f = frames.load((batch * t + frame) * c + i % c);
  if (masked) f = __fmul_rn(f, mask.load(batch * t + frame));
  out.store(i, __fadd_rn(x.load(i), f));
}
Source source(torch::Tensor cache, torch::Tensor scale, torch::Tensor image, torch::Tensor text,
    torch::Tensor index, int h, int hist, int ni, int64_t cs, int64_t ss, int nk) {
  TORCH_CHECK(index.is_cuda() && index.scalar_type() == at::kLong && index.is_contiguous(), "indices must be contiguous CUDA int64");
  TORCH_CHECK(index.numel() >= nk && h > 0 && hist >= 0 && ni >= 0 && nk >= 0 && cs >= h * 128,
      "invalid indexed source dimensions");
  for (const auto &t : {scale, image, text, index}) same(cache, t);
  TORCH_CHECK(cache.scalar_type() == at::kChar || cache.scalar_type() == at::kBFloat16 || cache.scalar_type() == at::kFloat || cache.scalar_type() == at::kHalf, "unsupported cache dtype");
  return {tensor(cache), tensor(scale), tensor(image), tensor(text), index.data_ptr<int64_t>(), h, hist, ni, nk, cs, ss};
}
}

torch::Tensor mod(torch::Tensor x, torch::Tensor shift, torch::Tensor scale, double eps, double activation_scale, bool norm_fp32) {
  c10::cuda::CUDAGuard guard(x.device());
  auto norm_dtype = norm_fp32 ? at::kFloat : x.scalar_type();
  auto dt = activation_scale ? at::kFloat8_e4m3fn : c10::promoteTypes(c10::promoteTypes(norm_dtype, shift.scalar_type()), scale.scalar_type());
  auto y = torch::empty(x.sizes(), x.options().dtype(dt));
  norm_launch<true, false>(x, shift, scale, y, x, x, eps, activation_scale, norm_fp32);
  return y;
}
torch::Tensor rms(torch::Tensor x, torch::Tensor weight, double eps) {
  c10::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty(x.sizes(), x.options().dtype(c10::promoteTypes(x.scalar_type(), weight.scalar_type())));
  norm_launch<false, false>(x, weight, weight, y, x, x, eps, 0);
  return y;
}
std::tuple<torch::Tensor, torch::Tensor> add_rms(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
  c10::cuda::CUDAGuard guard(x.device());
  auto added = torch::empty_like(x);
  auto y = torch::empty(x.sizes(), x.options().dtype(c10::promoteTypes(x.scalar_type(), weight.scalar_type())));
  norm_launch<false, true>(x, weight, weight, y, residual, added, eps, 0);
  return {added, y};
}
torch::Tensor norm_rope(torch::Tensor x, torch::Tensor weight, torch::Tensor other, torch::Tensor cos,
    torch::Tensor sin, double eps, bool decoder, bool rotary, at::ScalarType dtype) {
  c10::cuda::CUDAGuard guard(x.device());
  dense(x); dense(weight); dense(other);
  TORCH_CHECK(x.dim() == 4 && x.size(3) == 128 && weight.numel() == (decoder ? 64 : 128)
      && (!decoder || other.numel() == 64), "RoPE requires x[B,L,H,128] and 64/128 norm weights");
  for (const auto &t : {weight, other, cos, sin}) same(x, t);
  if (rotary) {
    dense(cos); dense(sin);
    TORCH_CHECK(cos.numel() >= x.size(1) * 128 && sin.numel() >= x.size(1) * 128, "RoPE tables must contain L*128 elements");
  }
  auto out = torch::empty(x.sizes(), x.options().dtype(dtype));
  dense(out);
  if (x.numel()) rope_kernel<<<(x.numel() / 128 + 7) / 8, 128, 0, at::cuda::getCurrentCUDAStream()>>>(tensor(x), tensor(weight),
      tensor(other), tensor(cos), tensor(sin), tensor(out), x.size(2), x.size(1), x.numel() / 128, float(eps), decoder, rotary);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
void indexed_quant(torch::Tensor cache, torch::Tensor scale, torch::Tensor image, torch::Tensor text,
    torch::Tensor index, torch::Tensor out, torch::Tensor scales, torch::Tensor mean, int64_t h, int64_t hist,
    int64_t ni, int64_t cs, int64_t ss, bool integer, int64_t nk, int64_t nb, int64_t first, bool value, int64_t repeat) {
  c10::cuda::CUDAGuard guard(cache.device());
  auto src = source(cache, scale, image, text, index, h, hist, ni, cs, ss, nk);
  for (const auto &t : {out, scales, mean}) same(cache, t);
  TORCH_CHECK(out.scalar_type() == at::kByte && out.is_contiguous() && out.dim() == 4 && out.size(1) == nb
      && out.numel() == out.size(0) * nb * 8192 && repeat > 0 && out.size(0) == h * repeat
      && first >= 0 && first <= nb && nb * 64 >= nk - 63 && integer == (cache.scalar_type() == at::kChar), "invalid packed output/source");
  TORCH_CHECK(scales.scalar_type() == at::kFloat && scales.is_contiguous() && scales.numel() == out.size(0) * nb
      && mean.scalar_type() == at::kFloat && mean.is_contiguous() && mean.numel() >= h * 128, "invalid quant scales/mean");
  if (first < nb) indexed_kernel<<<dim3(nb - first, out.size(0)), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      src, out.data_ptr<uint8_t>(), scales.data_ptr<float>(), mean.data_ptr<float>(), nb, first, value, repeat);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void history_mean(torch::Tensor cache, torch::Tensor scale, torch::Tensor index, torch::Tensor out,
    int64_t h, int64_t hist, int64_t cs, int64_t ss, bool integer, int64_t n) {
  c10::cuda::CUDAGuard guard(cache.device());
  auto src = source(cache, scale, cache, cache, index, h, hist, 0, cs, ss, n);
  same(cache, out);
  TORCH_CHECK(integer == (cache.scalar_type() == at::kChar) && out.scalar_type() == at::kFloat && out.is_contiguous()
      && out.numel() == ((n + 127) / 128) * h * 128, "invalid history partial output");
  if (n) history_kernel<<<dim3(h, (n + 127) / 128), 128, 0, at::cuda::getCurrentCUDAStream()>>>(src, out.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
torch::Tensor camera_add(torch::Tensor x, torch::Tensor frames, c10::optional<torch::Tensor> mask, int64_t spatial) {
  c10::cuda::CUDAGuard guard(x.device());
  dense(x); dense(frames); same(x, frames);
  TORCH_CHECK(x.dim() == 3 && frames.dim() == 3 && spatial > 0 && frames.size(0) == x.size(0)
      && frames.size(2) == x.size(2) && frames.size(1) * spatial == x.size(1), "invalid camera frame dimensions");
  auto m = mask.value_or(frames);
  if (mask) {
    same(x, m);
    TORCH_CHECK(m.is_contiguous() && m.numel() == frames.size(0) * frames.size(1), "invalid camera mask shape/strides");
  }
  auto dt = mask ? at::kFloat : c10::promoteTypes(x.scalar_type(), frames.scalar_type());
  auto out = torch::empty(x.sizes(), x.options().dtype(dt));
  if (x.numel()) camera_kernel<<<(x.numel() + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      tensor(x), tensor(frames), tensor(m), tensor(out), x.size(2), frames.size(1), spatial, x.numel(), bool(mask));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mod", &mod);
  m.def("rms", &rms);
  m.def("add_rms", &add_rms);
  m.def("norm_rope", &norm_rope);
  m.def("indexed_quant", &indexed_quant, py::arg("cache"), py::arg("scale"), py::arg("image"), py::arg("text"),
      py::arg("index"), py::arg("out"), py::arg("scales"), py::arg("mean"), py::arg("h"), py::arg("hist"), py::arg("ni"),
      py::arg("cs"), py::arg("ss"), py::arg("integer"), py::arg("nk"), py::arg("nb"), py::arg("first"), py::arg("value"), py::arg("repeat") = 1);
  m.def("history_mean", &history_mean);
  m.def("camera_add", &camera_add);
}
