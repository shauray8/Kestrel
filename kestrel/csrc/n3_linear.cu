#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>

namespace {
constexpr int BM = 128, BN = 128, BK = 64, AB = BM * BK, BB = BN * BK;
constexpr int THREADS = 384, MI = 4, NJ = 4;

__device__ __forceinline__ uint32_t sptr(const void *p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ int swz(int row, int chunk) { return row * BK + ((chunk ^ ((row >> 1) & 3)) << 4); }
__device__ __forceinline__ void init_bar(uint64_t *p, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sptr(p)), "r"(count) : "memory");
}
__device__ __forceinline__ void wait_bar(uint64_t *p, int phase) {
  asm volatile("{ .reg .pred done; WAIT: mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1; @!done bra WAIT; }"
               :: "r"(sptr(p)), "r"(phase) : "memory");
}
__device__ __forceinline__ void arrive(uint64_t *p) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(sptr(p)) : "memory");
}
__device__ __forceinline__ void expect(uint64_t *p) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
               :: "r"(sptr(p)), "r"(AB + BB) : "memory");
}
__device__ __forceinline__ void tma(void *dst, const CUtensorMap *map, uint64_t *bar, int k, int row) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(sptr(dst)), "l"(map), "r"(sptr(bar)), "r"(k), "r"(row) : "memory");
}
__device__ __forceinline__ void ldsm(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d, const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(sptr(p)));
}
__device__ __forceinline__ void mma(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void promote(uint32_t (&h)[MI][NJ][2], float (&f)[MI][NJ][4]) {
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      float2 lo = __half22float2(*reinterpret_cast<__half2 *>(&h[i][j][0]));
      float2 hi = __half22float2(*reinterpret_cast<__half2 *>(&h[i][j][1]));
      f[i][j][0] += lo.x; f[i][j][1] += lo.y; f[i][j][2] += hi.x; f[i][j][3] += hi.y;
      h[i][j][0] = h[i][j][1] = 0;
    }
}
__device__ __forceinline__ float gelu(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }
__device__ __forceinline__ uint16_t q2(float a, float b, float inv) {
  __nv_fp8x2_e4m3 q(make_float2(a * inv, b * inv));
  return q.__x;
}
struct Gate { const float *p; int64_t stride; int rows_per; };

__device__ __forceinline__ void residual_pair(float *out, size_t off, float a, float b, const float *g) {
  const __nv_bfloat162 y = __floats2bfloat162_rn(a, b);
  const float2 yf = __bfloat1622float2(y);
  out[off] = __fadd_rn(out[off], g ? __fmul_rn(yf.x, g[0]) : yf.x);
  out[off + 1] = __fadd_rn(out[off + 1], g ? __fmul_rn(yf.y, g[1]) : yf.y);
}

template <int S, int EPI, bool Cont, bool PDL>
__global__ void __maxnreg__(160) gemm(const __grid_constant__ CUtensorMap amap, const __grid_constant__ CUtensorMap bmap,
                                     const float *__restrict__ sw, const __nv_bfloat16 *__restrict__ bias,
                                     void *__restrict__ out, int M, int N, int K, float sa, float out_inv, Gate gate, int split) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t *sA = smem, *sB = smem + S * AB;
  uint64_t *full = reinterpret_cast<uint64_t *>(smem + S * (AB + BB));
  uint64_t *empty = full + S;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int grid_m = (M + BM - 1) / BM, grid_n = N / BN, kt_count = K / BK, tiles = grid_m * grid_n;
  auto tile = [&](int pid, int &m0, int &n0) {
    const int group = pid / (8 * grid_n), first_m = group * 8, gsize = min(grid_m - first_m, 8);
    m0 = (first_m + (pid % (8 * grid_n)) % gsize) * BM;
    n0 = ((pid % (8 * grid_n)) / gsize) * BN;
  };
  auto init = [&](bool reinit) {
    if (tid == 256) {
#pragma unroll
      for (int s = 0; s < S; ++s) {
        if (reinit) {
          asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(sptr(full + s)) : "memory");
          asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(sptr(empty + s)) : "memory");
        }
        init_bar(full + s, 1); init_bar(empty + s, 8);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      if constexpr (PDL)
        if (!reinit) {
          asm volatile("prefetch.tensormap [%0];" :: "l"(&amap) : "memory");
          asm volatile("prefetch.tensormap [%0];" :: "l"(&bmap) : "memory");
        }
    }
    __syncthreads();
  };
  bool dependency_ready = !PDL;
  auto produce = [&](int m0, int n0, int base) {
    if constexpr (PDL)
      if (!dependency_ready) {
        cudaGridDependencySynchronize();
        dependency_ready = true;
      }
    for (int kt = 0; kt < kt_count; ++kt) {
      const int g = base + kt, s = g % S;
      if (g >= S) wait_bar(empty + s, ((g / S) - 1) & 1);
      expect(full + s);
      tma(sA + s * AB, &amap, full + s, kt * BK, m0);
      tma(sB + s * BB, &bmap, full + s, kt * BK, n0);
    }
  };
  auto consume = [&](int m0, int n0, int base, bool last) {
    const int wm = warp / 4, wn = warp % 4, mi_l = lane >> 3, r = lane & 7;
    uint32_t h[MI][NJ][2];
    float f[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        h[i][j][0] = h[i][j][1] = 0;
        f[i][j][0] = f[i][j][1] = f[i][j][2] = f[i][j][3] = 0.f;
      }
    int step = 0;
    for (int kt = 0; kt < kt_count; ++kt) {
      const int g = base + kt, s = g % S;
      wait_bar(full + s, (g / S) & 1);
      const uint8_t *a_tile = sA + s * AB, *b_tile = sB + s * BB;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t a[MI][4], b[NJ][2];
#pragma unroll
        for (int i = 0; i < MI; ++i)
          ldsm(a[i][0], a[i][1], a[i][2], a[i][3], a_tile + swz(wm * 64 + i * 16 + (mi_l & 1) * 8 + r, kk * 2 + (mi_l >> 1)));
#pragma unroll
        for (int j = 0; j < NJ / 2; ++j)
          ldsm(b[j * 2][0], b[j * 2][1], b[j * 2 + 1][0], b[j * 2 + 1][1],
               b_tile + swz(wn * 32 + j * 16 + (mi_l >> 1) * 8 + r, kk * 2 + (mi_l & 1)));
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
          for (int j = 0; j < NJ; ++j) mma(h[i][j], a[i], b[j]);
        if (++step == 16) { step = 0; promote(h, f); }
      }
      __syncwarp();
      if (lane == 0) arrive(empty + s);
    }
    promote(h, f);
    if constexpr (PDL)
      if (last) cudaTriggerProgrammaticLaunchCompletion();
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int col = n0 + wn * 32 + j * 8 + t4 * 2;
      const float s0 = sa * sw[col], s1 = sa * sw[col + 1];
      const float b0 = bias ? __bfloat162float(bias[col]) : 0.f, b1 = bias ? __bfloat162float(bias[col + 1]) : 0.f;
#pragma unroll
      for (int i = 0; i < MI; ++i) {
        const int row = m0 + wm * 64 + i * 16 + g;
        const float v0 = f[i][j][0] * s0 + b0, v1 = f[i][j][1] * s1 + b1;
        const float v2 = f[i][j][2] * s0 + b0, v3 = f[i][j][3] * s1 + b1;
        if constexpr (EPI == 0) {
          auto y = reinterpret_cast<__nv_bfloat16 *>(out);
          auto at = [&](int rr) { return split ? (size_t)(col / split) * M * split + (size_t)rr * split + col % split : (size_t)rr * N + col; };
          if (row < M) *reinterpret_cast<__nv_bfloat162 *>(y + at(row)) = __floats2bfloat162_rn(v0, v1);
          if (row + 8 < M) *reinterpret_cast<__nv_bfloat162 *>(y + at(row + 8)) = __floats2bfloat162_rn(v2, v3);
        } else if constexpr (EPI == 1) {
          auto y = reinterpret_cast<uint8_t *>(out);
          if (row < M) *reinterpret_cast<uint16_t *>(y + (size_t)row * N + col) = q2(gelu(v0), gelu(v1), out_inv);
          if (row + 8 < M) *reinterpret_cast<uint16_t *>(y + (size_t)(row + 8) * N + col) = q2(gelu(v2), gelu(v3), out_inv);
        } else {
          if (row < M) residual_pair(reinterpret_cast<float *>(out), (size_t)row * N + col, v0, v1,
                                    gate.p ? gate.p + (size_t)(row / gate.rows_per) * gate.stride + col : nullptr);
          if (row + 8 < M) residual_pair(reinterpret_cast<float *>(out), (size_t)(row + 8) * N + col, v2, v3,
                                        gate.p ? gate.p + (size_t)((row + 8) / gate.rows_per) * gate.stride + col : nullptr);
        }
      }
    }
  };
  if constexpr (Cont) {
    init(false);
    if (warp >= 8) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
      if (tid == 256)
        for (int pid = blockIdx.x, base = 0; pid < tiles; pid += gridDim.x, base += kt_count) {
          int m0, n0;
          tile(pid, m0, n0);
          produce(m0, n0, base);
        }
      return;
    }
    for (int pid = blockIdx.x, base = 0; pid < tiles; pid += gridDim.x, base += kt_count) {
      int m0, n0;
      tile(pid, m0, n0);
      consume(m0, n0, base, pid + gridDim.x >= tiles);
    }
  } else {
    for (int pid = blockIdx.x; pid < tiles; pid += gridDim.x) {
      int m0, n0;
      tile(pid, m0, n0);
      init(pid != int(blockIdx.x));
      if (warp >= 8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
        if (tid == 256) produce(m0, n0, 0);
        asm volatile("bar.sync 1, 128;" ::: "memory");
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
      } else {
        consume(m0, n0, 0, pid + gridDim.x >= tiles);
      }
      __syncthreads();
    }
  }
}

__global__ void quantize(const __nv_bfloat16 *__restrict__ x, uint8_t *__restrict__ y, size_t count, float inv) {
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= count) return;
  const uint4 raw = *reinterpret_cast<const uint4 *>(x + i);
  const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
  uint16_t p[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&w[j]));
    p[j] = q2(f.x, f.y, inv);
  }
  *reinterpret_cast<uint2 *>(y + i) = make_uint2(p[0] | (uint32_t(p[1]) << 16), p[2] | (uint32_t(p[3]) << 16));
}

CUtensorMap encode(const torch::Tensor &x, int rows, int k) {
  using Encode = CUresult (CUDAAPI *)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
                                    const cuuint64_t *, const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave,
                                    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  static Encode fn = [] {
    void *p = nullptr;
    cudaDriverEntryPointQueryResult status;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &status));
    TORCH_CHECK(status == cudaDriverEntryPointSuccess && p, "cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<Encode>(p);
  }();
  const cuuint64_t dims[2] = {cuuint64_t(k), cuuint64_t(rows)}, strides[1] = {cuuint64_t(k)};
  const cuuint32_t box[2] = {BK, BM}, element_strides[2] = {1, 1};
  alignas(64) CUtensorMap map;
  const auto result = fn(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, x.data_ptr(), dims, strides, box, element_strides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                         CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", int(result));
  return map;
}

template <int S, int EPI, bool Cont, bool PDL = false>
void launch(CUtensorMap a, CUtensorMap b, const torch::Tensor &sw, const __nv_bfloat16 *bias, torch::Tensor &y,
            int m, int n, int k, float sa, float inv, Gate gate, int split) {
  if constexpr (!PDL) {
    const char *pdl = std::getenv("N3_PDL");
    if (pdl && std::atoi(pdl) == 1) {
      launch<S, EPI, Cont, true>(a, b, sw, bias, y, m, n, k, sa, inv, gate, split);
      return;
    }
  }
  constexpr int bytes = S * (AB + BB) + 2 * S * sizeof(uint64_t);
  static thread_local int initialized_device = -1;
  if (initialized_device != sw.get_device()) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(gemm<S, EPI, Cont, PDL>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    initialized_device = sw.get_device();
  }
  const int tiles = ((m + BM - 1) / BM) * (n / BN);
  const int blocks = std::min(tiles, at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
  if constexpr (PDL) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(blocks);
    cfg.blockDim = dim3(THREADS);
    cfg.dynamicSmemBytes = bytes;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, gemm<S, EPI, Cont, PDL>, a, b, sw.data_ptr<float>(), bias,
                                    y.data_ptr(), m, n, k, sa, inv, gate, split));
  } else {
    gemm<S, EPI, Cont, PDL><<<blocks, THREADS, bytes, at::cuda::getCurrentCUDAStream()>>>(a, b, sw.data_ptr<float>(), bias,
        y.data_ptr(), m, n, k, sa, inv, gate, split);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Gate gate_arg(const c10::optional<torch::Tensor> &gate, const torch::Tensor &x, int m, int n) {
  if (!gate) return {nullptr, 0, 1};
  const auto &a = *gate;
  TORCH_CHECK(a.device() == x.device() && a.scalar_type() == torch::kFloat32 && a.dim() >= 1 && a.size(-1) == n && a.stride(-1) == 1, "gate");
  const int64_t rows = a.numel() / n;
  TORCH_CHECK(rows > 0 && m % rows == 0, "gate rows must divide M");
  int64_t stride = 0, expect = -1;
  for (int d = a.dim() - 2; d >= 0; --d) {
    if (a.size(d) == 1) continue;
    if (expect < 0) { stride = a.stride(d); expect = stride * a.size(d); }
    else { TORCH_CHECK(a.stride(d) == expect, "gate leading dimensions"); expect *= a.size(d); }
  }
  return {a.data_ptr<float>(), rows == 1 ? 0 : stride, int(m / rows)};
}

template <bool BF16, int EPI>
torch::Tensor run(torch::Tensor x, torch::Tensor w, torch::Tensor sw, c10::optional<torch::Tensor> bias, double sa,
                  double out_scale, c10::optional<torch::Tensor> residual = c10::nullopt,
                  c10::optional<torch::Tensor> gate = c10::nullopt, int split = 0) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() >= 1 && x.scalar_type() == (BF16 ? torch::kBFloat16 : torch::kFloat8_e4m3fn), "x");
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(w.device() == x.device() && w.dim() == 2 && w.is_contiguous() && w.scalar_type() == torch::kFloat8_e4m3fn, "w");
  TORCH_CHECK(x.size(-1) > 0 && x.size(-1) <= std::numeric_limits<int>::max() && w.size(0) <= std::numeric_limits<int>::max(), "shape range");
  const int k = x.size(-1), n = w.size(0);
  TORCH_CHECK(w.size(1) == k && k % BK == 0 && n > 0 && n % BN == 0 && x.numel() / k <= std::numeric_limits<int>::max() - BM, "shape");
  const int m = x.numel() / k;
  TORCH_CHECK(sw.device() == x.device() && sw.scalar_type() == torch::kFloat32 && sw.is_contiguous() && sw.numel() == n, "sw");
  const __nv_bfloat16 *bp = nullptr;
  if (bias) {
    TORCH_CHECK(bias->device() == x.device() && bias->scalar_type() == torch::kBFloat16 && bias->is_contiguous() && bias->numel() == n, "bias");
    bp = reinterpret_cast<const __nv_bfloat16 *>(bias->data_ptr());
  }
  auto sizes = x.sizes().vec(); sizes.back() = n;
  torch::Tensor y;
  Gate g{nullptr, 0, 1};
  if constexpr (EPI == 2) {
    TORCH_CHECK(residual && residual->device() == x.device() && residual->scalar_type() == torch::kFloat32 && residual->is_contiguous() && residual->sizes() == at::IntArrayRef(sizes), "residual");
    y = *residual;
    g = gate_arg(gate, x, m, n);
  } else if (split) {
    TORCH_CHECK(EPI == 0 && split % BN == 0 && n % split == 0, "split");
    y = torch::empty({n / split, m, split}, x.options().dtype(torch::kBFloat16));
  } else y = torch::empty(sizes, x.options().dtype(EPI == 1 ? torch::kFloat8_e4m3fn : torch::kBFloat16));
  if (!m) return y;
  torch::Tensor a = x;
  if constexpr (BF16) {
    a = torch::empty(x.sizes(), x.options().dtype(torch::kFloat8_e4m3fn));
    quantize<<<(x.numel() + 2047) / 2048, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), reinterpret_cast<uint8_t *>(a.data_ptr()), x.numel(), float(1.0 / sa));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const auto amap = encode(a, m, k), bmap = encode(w, n, k);
  const char *stages = std::getenv("N3_STAGES"), *cont = std::getenv("N3_CONT");
  const int s = stages ? std::atoi(stages) : 5;
  const bool c = cont && std::atoi(cont) == 1;
  TORCH_CHECK(s >= 4 && s <= 6, "N3_STAGES must be 4, 5, or 6");
  const float sf = float(sa), inv = float(1.0 / out_scale);
  if (s == 4) c ? launch<4, EPI, true>(amap, bmap, sw, bp, y, m, n, k, sf, inv, g, split) : launch<4, EPI, false>(amap, bmap, sw, bp, y, m, n, k, sf, inv, g, split);
  else if (s == 5) c ? launch<5, EPI, true>(amap, bmap, sw, bp, y, m, n, k, sf, inv, g, split) : launch<5, EPI, false>(amap, bmap, sw, bp, y, m, n, k, sf, inv, g, split);
  else c ? launch<6, EPI, true>(amap, bmap, sw, bp, y, m, n, k, sf, inv, g, split) : launch<6, EPI, false>(amap, bmap, sw, bp, y, m, n, k, sf, inv, g, split);
  return y;
}

torch::Tensor linear(torch::Tensor x, torch::Tensor w, torch::Tensor sw, c10::optional<torch::Tensor> b, double sa) { return run<true, 0>(x, w, sw, b, sa, 1.0); }
torch::Tensor linear_gelu(torch::Tensor x, torch::Tensor w, torch::Tensor sw, c10::optional<torch::Tensor> b, double sa, double so) { return run<true, 1>(x, w, sw, b, sa, so); }
torch::Tensor linear_f8(torch::Tensor x, torch::Tensor w, torch::Tensor sw, c10::optional<torch::Tensor> b, double sa) { return run<false, 0>(x, w, sw, b, sa, 1.0); }
torch::Tensor linear_f8_gelu(torch::Tensor x, torch::Tensor w, torch::Tensor sw, c10::optional<torch::Tensor> b, double sa, double so) { return run<false, 1>(x, w, sw, b, sa, so); }
torch::Tensor linear_residual(torch::Tensor x, torch::Tensor w, torch::Tensor sw, c10::optional<torch::Tensor> b, double sa, torch::Tensor residual, c10::optional<torch::Tensor> gate) { return run<true, 2>(x, w, sw, b, sa, 1.0, residual, gate); }
torch::Tensor linear_f8_residual(torch::Tensor x, torch::Tensor w, torch::Tensor sw, c10::optional<torch::Tensor> b, double sa, torch::Tensor residual, c10::optional<torch::Tensor> gate) { return run<false, 2>(x, w, sw, b, sa, 1.0, residual, gate); }
__global__ void silu_mul_q(const __nv_bfloat16 *__restrict__ g, const __nv_bfloat16 *__restrict__ u, uint8_t *__restrict__ y, size_t count, float inv) {
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= count) return;
  const uint4 rg = *reinterpret_cast<const uint4 *>(g + i), ru = *reinterpret_cast<const uint4 *>(u + i);
  const uint32_t wg[4] = {rg.x, rg.y, rg.z, rg.w}, wu[4] = {ru.x, ru.y, ru.z, ru.w};
  uint16_t p[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&wg[j]));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&wu[j]));
    const float2 s = __bfloat1622float2(__floats2bfloat162_rn(a.x / (1.f + expf(-a.x)), a.y / (1.f + expf(-a.y))));
    const float2 h = __bfloat1622float2(__floats2bfloat162_rn(s.x * b.x, s.y * b.y));
    p[j] = q2(h.x, h.y, inv);
  }
  *reinterpret_cast<uint2 *>(y + i) = make_uint2(p[0] | (uint32_t(p[1]) << 16), p[2] | (uint32_t(p[3]) << 16));
}

__global__ void silu_mul_q_cols(const __nv_bfloat16 *__restrict__ gu, const float *__restrict__ sw, uint8_t *__restrict__ y,
                                int rows, int inter, float inv) {
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= (size_t)rows * inter) return;
  const size_t r = i / inter; const int c = int(i % inter);
  const __nv_bfloat16 *g = gu + r * 2 * inter + c, *u = g + inter;
  const uint4 rg = *reinterpret_cast<const uint4 *>(g), ru = *reinterpret_cast<const uint4 *>(u);
  const uint32_t wg[4] = {rg.x, rg.y, rg.z, rg.w}, wu[4] = {ru.x, ru.y, ru.z, ru.w};
  uint16_t p[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&wg[j]));
    const float2 b0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&wu[j]));
    const float2 a = __bfloat1622float2(__floats2bfloat162_rn(a0.x * sw[c + 2 * j], a0.y * sw[c + 2 * j + 1]));
    const float2 b = __bfloat1622float2(__floats2bfloat162_rn(b0.x * sw[inter + c + 2 * j], b0.y * sw[inter + c + 2 * j + 1]));
    const float2 s = __bfloat1622float2(__floats2bfloat162_rn(a.x / (1.f + expf(-a.x)), a.y / (1.f + expf(-a.y))));
    const float2 h = __bfloat1622float2(__floats2bfloat162_rn(s.x * b.x, s.y * b.y));
    p[j] = q2(h.x, h.y, inv);
  }
  *reinterpret_cast<uint2 *>(y + i) = make_uint2(p[0] | (uint32_t(p[1]) << 16), p[2] | (uint32_t(p[3]) << 16));
}

torch::Tensor silu_mul_quant_cols(torch::Tensor gu, torch::Tensor sw, double sa) {
  TORCH_CHECK(gu.is_cuda() && gu.is_contiguous() && gu.scalar_type() == torch::kBFloat16 && gu.dim() == 2 && gu.size(1) % 16 == 0, "gu");
  TORCH_CHECK(sw.is_cuda() && sw.scalar_type() == torch::kFloat32 && sw.is_contiguous() && sw.numel() == gu.size(1), "sw");
  const c10::cuda::CUDAGuard guard(gu.device());
  const int rows = gu.size(0), inter = gu.size(1) / 2;
  auto a = torch::empty({rows, inter}, gu.options().dtype(torch::kFloat8_e4m3fn));
  const size_t n = a.numel();
  if (n) silu_mul_q_cols<<<(n + 2047) / 2048, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(gu.data_ptr()), sw.data_ptr<float>(), reinterpret_cast<uint8_t *>(a.data_ptr()), rows, inter, float(1.0 / sa));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return a;
}

torch::Tensor quantize_bf16(torch::Tensor x, double sa) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16 && x.numel() % 8 == 0, "x");
  const c10::cuda::CUDAGuard guard(x.device());
  auto a = torch::empty(x.sizes(), x.options().dtype(torch::kFloat8_e4m3fn));
  if (x.numel()) quantize<<<(x.numel() + 2047) / 2048, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), reinterpret_cast<uint8_t *>(a.data_ptr()), x.numel(), float(1.0 / sa));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return a;
}

torch::Tensor silu_mul_quant(torch::Tensor gu, double sa) {
  TORCH_CHECK(gu.is_cuda() && gu.is_contiguous() && gu.scalar_type() == torch::kBFloat16 && gu.dim() == 3 && gu.size(0) == 2 && gu.size(2) % 8 == 0, "gu");
  const c10::cuda::CUDAGuard guard(gu.device());
  auto a = torch::empty({gu.size(1), gu.size(2)}, gu.options().dtype(torch::kFloat8_e4m3fn));
  const size_t n = a.numel();
  if (n) silu_mul_q<<<(n + 2047) / 2048, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(gu.data_ptr()), reinterpret_cast<const __nv_bfloat16 *>(gu.data_ptr()) + n,
      reinterpret_cast<uint8_t *>(a.data_ptr()), n, float(1.0 / sa));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return a;
}

torch::Tensor linear_split(torch::Tensor x, torch::Tensor w, torch::Tensor sw, c10::optional<torch::Tensor> b, double sa, int64_t split) { return run<true, 0>(x, w, sw, b, sa, 1.0, c10::nullopt, c10::nullopt, int(split)); }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("n3_linear", &linear);
  m.def("n3_linear_gelu_f8", &linear_gelu);
  m.def("n3_linear_f8in", &linear_f8);
  m.def("n3_linear_f8in_gelu_f8", &linear_f8_gelu);
  m.def("n3_linear_residual", &linear_residual);
  m.def("n3_linear_f8in_residual", &linear_f8_residual);
  m.def("n3_linear_split", &linear_split);
  m.def("n3_quantize", &quantize_bf16);
  m.def("n3_silu_mul_quant", &silu_mul_quant);
  m.def("n3_silu_mul_quant_cols", &silu_mul_quant_cols);
}
