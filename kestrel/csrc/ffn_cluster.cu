#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <climits>
#include <cmath>
#include <cstdint>

namespace {
namespace cg = cooperative_groups;
constexpr int BM = 64, BN = 128, BK = 64, STAGES = 2, THREADS = 128, TN = 32, MI = 4, NJ = 4, RANKS = 8;
constexpr int C = 1536, FF = 8960, A_BYTES = BM * BK, B_BYTES = BN * BK;
constexpr int PIPE = STAGES * (A_BYTES + B_BYTES), TILE = BM * BN, OWN = (FF / BN + RANKS - 1) / RANKS, SMEM = PIPE + OWN * TILE;
static_assert(SMEM == 96 * 1024, "shared memory");

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ int swz(int row, int chunk) { return row * BK + ((chunk ^ ((row >> 1) & 3)) << 4); }
__device__ __forceinline__ void cp16(uint32_t dst, const void *src, bool p) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(p ? 16 : 0));
}
__device__ __forceinline__ uint4 ld_cluster(const void *p) {
  uint4 v;
  asm volatile("ld.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(smem_u32(p)) : "memory");
  return v;
}
__device__ __forceinline__ void ldsm_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma8(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float gelu(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }
__device__ __forceinline__ uint16_t q2(float a, float b, float inv) {
  __nv_fp8x2_e4m3 q(make_float2(a * inv, b * inv));
  return q.__x;
}

template <bool DOWN>
__device__ __forceinline__ void tile_mma(const uint8_t *X, const uint8_t *W, int M, int m0, int n0, uint8_t *smem,
                                         cg::cluster_group cluster, float (&f)[MI][NJ][4]) {
  uint8_t *sA = smem, *sB = smem + STAGES * A_BYTES, *resident = smem + PIPE;
  const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5, lr = tid >> 2, lc = tid & 3;
  constexpr int K = DOWN ? FF : C, KT = K / BK;
  auto load = [&](int stage, int kt) {
    if constexpr (DOWN) {
      const int tile = kt / 2;
      const uint8_t *remote = cluster.map_shared_rank(resident, tile % RANKS) + (tile / RANKS) * TILE;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int row = lr + i * 32;
        *reinterpret_cast<uint4 *>(sA + stage * A_BYTES + swz(row, lc)) = ld_cluster(remote + row * BN + (kt & 1) * BK + lc * 16);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int row = lr + i * 32, gm = m0 + row;
        cp16(smem_u32(sA + stage * A_BYTES + swz(row, lc)), X + (size_t)(gm < M ? gm : 0) * K + kt * BK + lc * 16, gm < M);
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = lr + i * 32;
      cp16(smem_u32(sB + stage * B_BYTES + swz(row, lc)), W + (size_t)(n0 + row) * K + kt * BK + lc * 16, true);
    }
  };
  uint32_t h[MI][NJ][2];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j) { h[i][j][0] = h[i][j][1] = 0; f[i][j][0] = f[i][j][1] = f[i][j][2] = f[i][j][3] = 0.f; }
  __syncthreads();
  load(0, 0);
  asm volatile("cp.async.commit_group;");
  auto promote = [&]() {
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        float2 lo = __half22float2(*reinterpret_cast<__half2 *>(&h[i][j][0])), hi = __half22float2(*reinterpret_cast<__half2 *>(&h[i][j][1]));
        f[i][j][0] += lo.x; f[i][j][1] += lo.y; f[i][j][2] += hi.x; f[i][j][3] += hi.y;
        h[i][j][0] = h[i][j][1] = 0;
      }
  };
  int step = 0;
  const int mi = lane >> 3, r = lane & 7;
  for (int kt = 0; kt < KT; ++kt) {
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    const int nk = kt + 1;
    if (nk < KT) load(nk % STAGES, nk);
    asm volatile("cp.async.commit_group;");
    const uint8_t *tA = sA + (kt % STAGES) * A_BYTES, *tB = sB + (kt % STAGES) * B_BYTES;
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t a[MI][4], b[NJ][2];
#pragma unroll
      for (int i = 0; i < MI; ++i) ldsm_x4(a[i][0], a[i][1], a[i][2], a[i][3], smem_u32(tA + swz(i * 16 + (mi & 1) * 8 + r, kk * 2 + (mi >> 1))));
#pragma unroll
      for (int j2 = 0; j2 < NJ / 2; ++j2)
        ldsm_x4(b[j2 * 2][0], b[j2 * 2][1], b[j2 * 2 + 1][0], b[j2 * 2 + 1][1], smem_u32(tB + swz(wn * TN + j2 * 16 + (mi >> 1) * 8 + r, kk * 2 + (mi & 1))));
#pragma unroll
      for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j) mma8(h[i][j], a[i], b[j][0], b[j][1]);
      if (++step == 16) { step = 0; promote(); }
    }
  }
  asm volatile("cp.async.wait_group 0;");
  promote();
}

template <bool RES>
__global__ void __launch_bounds__(THREADS, 1) ffn_kernel(const uint8_t *X, const uint8_t *W0, const float *sw0, const __nv_bfloat16 *b0,
                                                       const uint8_t *W2, const float *sw2, const __nv_bfloat16 *b2, void *Y,
                                                       const float *gate, int M, int rows_per, float sa, float s_mid, float inv_mid) {
  extern __shared__ __align__(128) uint8_t smem[];
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank(), m0 = (blockIdx.x / RANKS) * BM;
  const int lane = threadIdx.x & 31, wn = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  for (int n = rank; n < FF / BN; n += RANKS) {
    float f[MI][NJ][4];
    tile_mma<false>(X, W0, M, m0, n * BN, smem, cluster, f);
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int c = wn * TN + j * 8 + t4 * 2, col = n * BN + c;
      const float s0 = sa * sw0[col], s1 = sa * sw0[col + 1];
      const float b_0 = b0 ? __bfloat162float(b0[col]) : 0.f, b_1 = b0 ? __bfloat162float(b0[col + 1]) : 0.f;
#pragma unroll
      for (int i = 0; i < MI; ++i) {
        const int row = i * 16 + g;
        float v0 = f[i][j][0] * s0 + b_0, v1 = f[i][j][1] * s1 + b_1, v2 = f[i][j][2] * s0 + b_0, v3 = f[i][j][3] * s1 + b_1;
        v0 = gelu(v0); v1 = gelu(v1); v2 = gelu(v2); v3 = gelu(v3);
        uint8_t *dst = smem + PIPE + (n / RANKS) * TILE + c;
        *reinterpret_cast<uint16_t *>(dst + row * BN) = m0 + row < M ? q2(v0, v1, inv_mid) : 0;
        *reinterpret_cast<uint16_t *>(dst + (row + 8) * BN) = m0 + row + 8 < M ? q2(v2, v3, inv_mid) : 0;
      }
    }
  }
  cluster.sync();
  for (int n = rank; n < C / BN; n += RANKS) {
    float f[MI][NJ][4];
    tile_mma<true>(nullptr, W2, M, m0, n * BN, smem, cluster, f);
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int col = n * BN + wn * TN + j * 8 + t4 * 2;
      const float s0 = s_mid * sw2[col], s1 = s_mid * sw2[col + 1];
      const float b_0 = b2 ? __bfloat162float(b2[col]) : 0.f, b_1 = b2 ? __bfloat162float(b2[col + 1]) : 0.f;
      if constexpr (RES) {
        float2 xv[MI][2], gv[MI][2];
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = m0 + i * 16 + g + h * 8;
            if (row < M) {
              xv[i][h] = __ldcg(reinterpret_cast<const float2 *>((const float *)Y + (size_t)row * C + col));
              gv[i][h] = __ldg(reinterpret_cast<const float2 *>(gate + (size_t)(row / rows_per) * C + col));
            }
          }
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = m0 + i * 16 + g + h * 8;
            if (row < M) {
              const float2 y = __bfloat1622float2(__floats2bfloat162_rn(f[i][j][2 * h] * s0 + b_0, f[i][j][2 * h + 1] * s1 + b_1));
              *reinterpret_cast<float2 *>((float *)Y + (size_t)row * C + col) =
                  make_float2(__fadd_rn(xv[i][h].x, __fmul_rn(y.x, gv[i][h].x)), __fadd_rn(xv[i][h].y, __fmul_rn(y.y, gv[i][h].y)));
            }
          }
      } else {
#pragma unroll
        for (int i = 0; i < MI; ++i) {
          const int row = m0 + i * 16 + g;
          const float v0 = f[i][j][0] * s0 + b_0, v1 = f[i][j][1] * s1 + b_1, v2 = f[i][j][2] * s0 + b_0, v3 = f[i][j][3] * s1 + b_1;
          __nv_bfloat16 *dst = (__nv_bfloat16 *)Y + col;
          if (row < M) *reinterpret_cast<__nv_bfloat162 *>(dst + (size_t)row * C) = __floats2bfloat162_rn(v0, v1);
          if (row + 8 < M) *reinterpret_cast<__nv_bfloat162 *>(dst + (size_t)(row + 8) * C) = __floats2bfloat162_rn(v2, v3);
        }
      }
    }
  }
  cluster.sync();
}

template <bool RES>
void launch(torch::Tensor x, torch::Tensor w0, torch::Tensor sw0, const __nv_bfloat16 *b0, torch::Tensor w2, torch::Tensor sw2,
            const __nv_bfloat16 *b2, torch::Tensor y, const float *gate, int rows_per, float sa, float s_mid, float inv_mid) {
  static thread_local int configured_device = -1;
  if (configured_device != x.get_device()) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ffn_kernel<RES>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    configured_device = x.get_device();
  }
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {RANKS, 1, 1};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(((x.size(0) + BM - 1) / BM) * RANKS);
  config.blockDim = dim3(THREADS);
  config.dynamicSmemBytes = SMEM;
  config.stream = at::cuda::getCurrentCUDAStream();
  config.attrs = &attr;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, ffn_kernel<RES>, (const uint8_t *)x.data_ptr(), (const uint8_t *)w0.data_ptr(), sw0.data_ptr<float>(), b0,
                                  (const uint8_t *)w2.data_ptr(), sw2.data_ptr<float>(), b2, y.data_ptr(), gate, int(x.size(0)), rows_per, sa, s_mid, inv_mid));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

torch::Tensor run(torch::Tensor x, double sa, torch::Tensor w0, torch::Tensor sw0, c10::optional<torch::Tensor> b0, double s_mid,
                  torch::Tensor w2, torch::Tensor sw2, c10::optional<torch::Tensor> b2, torch::Tensor y, c10::optional<torch::Tensor> gate) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(0) > 0 && x.size(0) <= INT_MAX && x.size(1) == C, "x must be CUDA [M,1536]");
  c10::cuda::CUDAGuard guard(x.device());
  auto check = [&](const torch::Tensor &t, at::ScalarType dtype) {
    TORCH_CHECK(t.device() == x.device() && t.scalar_type() == dtype && t.is_contiguous(), "device, dtype or layout");
  };
  check(x, torch::kFloat8_e4m3fn); check(w0, torch::kFloat8_e4m3fn); check(w2, torch::kFloat8_e4m3fn);
  check(sw0, torch::kFloat32); check(sw2, torch::kFloat32);
  TORCH_CHECK(w0.dim() == 2 && w0.size(0) == FF && w0.size(1) == C && w2.dim() == 2 && w2.size(0) == C && w2.size(1) == FF, "weight shapes");
  TORCH_CHECK(sw0.dim() == 1 && sw0.numel() == FF && sw2.dim() == 1 && sw2.numel() == C, "scale shapes");
  TORCH_CHECK(std::isfinite(sa) && sa > 0 && std::isfinite(s_mid) && s_mid > 0 && float(sa) > 0 && std::isfinite(float(sa)) &&
              float(s_mid) > 0 && std::isfinite(float(s_mid)), "positive finite scales");
  auto bias = [&](c10::optional<torch::Tensor> b, int n) -> const __nv_bfloat16 * {
    if (!b.has_value()) return nullptr;
    check(*b, torch::kBFloat16);
    TORCH_CHECK(b->dim() == 1 && b->numel() == n, "bias shape");
    return (const __nv_bfloat16 *)b->data_ptr();
  };
  const auto bp0 = bias(b0, FF), bp2 = bias(b2, C);
  check(y, gate.has_value() ? torch::kFloat32 : torch::kBFloat16);
  TORCH_CHECK(y.dim() == 2 && y.sizes() == x.sizes(), "output shape");
  if (gate.has_value()) {
    check(*gate, torch::kFloat32);
    TORCH_CHECK(gate->dim() == 2 && gate->size(0) > 0 && gate->size(1) == C && x.size(0) % gate->size(0) == 0, "gate shape");
    launch<true>(x, w0, sw0, bp0, w2, sw2, bp2, y, gate->data_ptr<float>(), x.size(0) / gate->size(0), float(sa), float(s_mid), 1.f / float(s_mid));
  } else {
    launch<false>(x, w0, sw0, bp0, w2, sw2, bp2, y, nullptr, 1, float(sa), float(s_mid), 1.f / float(s_mid));
  }
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, pybind11::arg("x"), pybind11::arg("sa"), pybind11::arg("w0"), pybind11::arg("sw0"), pybind11::arg("b0"), pybind11::arg("s_mid"),
        pybind11::arg("w2"), pybind11::arg("sw2"), pybind11::arg("b2"), pybind11::arg("out"), pybind11::arg("gate") = pybind11::none());
}
