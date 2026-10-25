#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {
constexpr int BM = 128, BN = 128, BK = 64, STAGES = 3, THREADS = 256, WM = 2, WN = 4, TM = 64, TN = 32, MI = 4, NJ = 4;
constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK;
constexpr int A_CHUNKS = BM * BK / 8 / THREADS;

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ int swz(int row, int chunk) { return row * BK + ((chunk ^ ((row >> 1) & 3)) << 4); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void ldsm_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_f16acc(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint2 cvt8(uint4 v, float s) {
  uint32_t w[4] = {v.x, v.y, v.z, v.w};
  uint16_t p[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&w[i]));
    __nv_fp8x2_e4m3 q(make_float2(f.x * s, f.y * s));
    p[i] = q.__x;
  }
  return make_uint2(p[0] | (uint32_t(p[1]) << 16), p[2] | (uint32_t(p[3]) << 16));
}

__device__ __forceinline__ float gelu_tanh(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }
__device__ __forceinline__ uint16_t q2(float a, float b, float inv) {
  __nv_fp8x2_e4m3 q(make_float2(a * inv, b * inv));
  return q.__x;
}

template <int P, int EPI>
__global__ void __launch_bounds__(THREADS, 1) n1_a16(const __nv_bfloat16 *__restrict__ X, const uint8_t *__restrict__ W,
                                                     const float *__restrict__ sw, const __nv_bfloat16 *__restrict__ bias,
                                                     void *__restrict__ Yv, int M, int N, int K, float inv_sa, float sa, float out_inv) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t *sA = smem, *sB = smem + STAGES * A_BYTES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
  const int grid_m = (M + BM - 1) / BM, grid_n = N / BN, pid = blockIdx.x;
  const int group = pid / (8 * grid_n), first_m = group * 8, gsize = min(grid_m - first_m, 8);
  const int m0 = (first_m + (pid % (8 * grid_n)) % gsize) * BM, n0 = ((pid % (8 * grid_n)) / gsize) * BN;
  const int KT = K / BK;

  uint4 areg[A_CHUNKS];
  auto load_a_regs = [&](int kt) {
#pragma unroll
    for (int i = 0; i < A_CHUNKS; i++) {
      int c = tid + i * THREADS, row = c >> 3, ch = c & 7, gm = m0 + row;
      areg[i] = gm < M ? __ldg(reinterpret_cast<const uint4 *>(X + (size_t)gm * K + kt * BK + ch * 8)) : make_uint4(0, 0, 0, 0);
    }
  };
  auto store_a = [&](int stage) {
#pragma unroll
    for (int i = 0; i < A_CHUNKS; i++) {
      int c = tid + i * THREADS, row = c >> 3, ch = c & 7;
      *reinterpret_cast<uint2 *>(sA + stage * A_BYTES + swz(row, ch >> 1) + (ch & 1) * 8) = cvt8(areg[i], inv_sa);
    }
  };
  auto load_b = [&](int stage, int kt) {
#pragma unroll
    for (int i = 0; i < BN * 4 / THREADS; i++) {
      int c = tid + i * THREADS, row = c >> 2, ch = c & 3;
      cp_async16(smem_u32(sB + stage * B_BYTES + swz(row, ch)), W + (size_t)(n0 + row) * K + kt * BK + ch * 16);
    }
  };

  uint32_t h[MI][NJ][2];
  float f[MI][NJ][4];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++) { h[i][j][0] = h[i][j][1] = 0; f[i][j][0] = f[i][j][1] = f[i][j][2] = f[i][j][3] = 0.f; }

#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KT) { load_a_regs(s); store_a(s); load_b(s, s); }
    asm volatile("cp.async.commit_group;");
  }
  auto promote = [&]() {
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
      for (int j = 0; j < NJ; j++) {
        float2 lo = __half22float2(*reinterpret_cast<__half2 *>(&h[i][j][0]));
        float2 hi = __half22float2(*reinterpret_cast<__half2 *>(&h[i][j][1]));
        f[i][j][0] += lo.x; f[i][j][1] += lo.y; f[i][j][2] += hi.x; f[i][j][3] += hi.y;
        h[i][j][0] = h[i][j][1] = 0;
      }
  };
  int step = 0;
  const int mi_l = lane >> 3, r = lane & 7;
  for (int kt = 0; kt < KT; kt++) {
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));
    __syncthreads();
    const int nk = kt + STAGES - 1;
    const bool more = nk < KT;
    if (more) { load_a_regs(nk); load_b(nk % STAGES, nk); }
    asm volatile("cp.async.commit_group;");
    const uint8_t *tA = sA + (kt % STAGES) * A_BYTES, *tB = sB + (kt % STAGES) * B_BYTES;
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      uint32_t a[MI][4], b[NJ][2];
#pragma unroll
      for (int i = 0; i < MI; i++)
        ldsm_x4(a[i][0], a[i][1], a[i][2], a[i][3], smem_u32(tA + swz(wm * TM + i * 16 + (mi_l & 1) * 8 + r, kk * 2 + (mi_l >> 1))));
#pragma unroll
      for (int j2 = 0; j2 < NJ / 2; j2++)
        ldsm_x4(b[j2 * 2][0], b[j2 * 2][1], b[j2 * 2 + 1][0], b[j2 * 2 + 1][1],
                smem_u32(tB + swz(wn * TN + j2 * 16 + (mi_l >> 1) * 8 + r, kk * 2 + (mi_l & 1))));
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++) mma_f16acc(h[i][j], a[i], b[j][0], b[j][1]);
      if (++step == P) { step = 0; promote(); }
    }

    if (more) store_a(nk % STAGES);
  }
  asm volatile("cp.async.wait_group 0;");
  promote();
  const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
  for (int j = 0; j < NJ; j++) {
    const int col = n0 + wn * TN + j * 8 + t4 * 2;
    const float s0 = sa * sw[col], s1 = sa * sw[col + 1];
    const float b0 = bias ? __bfloat162float(bias[col]) : 0.f, b1 = bias ? __bfloat162float(bias[col + 1]) : 0.f;
#pragma unroll
    for (int i = 0; i < MI; i++) {
      const int row = m0 + wm * TM + i * 16 + g;
      const float v0 = f[i][j][0] * s0 + b0, v1 = f[i][j][1] * s1 + b1, v2 = f[i][j][2] * s0 + b0, v3 = f[i][j][3] * s1 + b1;
      if constexpr (EPI == 0) {
        __nv_bfloat16 *Y = reinterpret_cast<__nv_bfloat16 *>(Yv);
        if (row < M) *reinterpret_cast<__nv_bfloat162 *>(Y + (size_t)row * N + col) = __floats2bfloat162_rn(v0, v1);
        if (row + 8 < M) *reinterpret_cast<__nv_bfloat162 *>(Y + (size_t)(row + 8) * N + col) = __floats2bfloat162_rn(v2, v3);
      } else {
        uint8_t *Y = reinterpret_cast<uint8_t *>(Yv);
        if (row < M) *reinterpret_cast<uint16_t *>(Y + (size_t)row * N + col) = q2(gelu_tanh(v0), gelu_tanh(v1), out_inv);
        if (row + 8 < M) *reinterpret_cast<uint16_t *>(Y + (size_t)(row + 8) * N + col) = q2(gelu_tanh(v2), gelu_tanh(v3), out_inv);
      }
    }
  }
}

__global__ void __launch_bounds__(THREADS, 2) n1_f8(const uint8_t *__restrict__ X, const uint8_t *__restrict__ W,
                                                    const float *__restrict__ sw, const __nv_bfloat16 *__restrict__ bias,
                                                    __nv_bfloat16 *__restrict__ Y, int M, int N, int K, float sa) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t *sA = smem, *sB = smem + STAGES * A_BYTES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
  const int grid_m = (M + BM - 1) / BM, grid_n = N / BN, pid = blockIdx.x;
  const int group = pid / (8 * grid_n), first_m = group * 8, gsize = min(grid_m - first_m, 8);
  const int m0 = (first_m + (pid % (8 * grid_n)) % gsize) * BM, n0 = ((pid % (8 * grid_n)) / gsize) * BN;
  const int KT = K / BK;
  auto load = [&](int stage, int kt) {
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int c = tid + i * THREADS, row = c >> 2, ch = c & 3, gm = m0 + row;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_u32(sA + stage * A_BYTES + swz(row, ch))),
                   "l"(X + (size_t)(gm < M ? gm : 0) * K + kt * BK + ch * 16), "r"(gm < M ? 16 : 0));
      cp_async16(smem_u32(sB + stage * B_BYTES + swz(row, ch)), W + (size_t)(n0 + row) * K + kt * BK + ch * 16);
    }
  };
  uint32_t h[MI][NJ][2];
  float f[MI][NJ][4];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++) { h[i][j][0] = h[i][j][1] = 0; f[i][j][0] = f[i][j][1] = f[i][j][2] = f[i][j][3] = 0.f; }
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KT) load(s, s);
    asm volatile("cp.async.commit_group;");
  }
  int step = 0;
  const int mi_l = lane >> 3, r = lane & 7;
  for (int kt = 0; kt < KT; kt++) {
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));
    __syncthreads();
    const int nk = kt + STAGES - 1;
    if (nk < KT) load(nk % STAGES, nk);
    asm volatile("cp.async.commit_group;");
    const uint8_t *tA = sA + (kt % STAGES) * A_BYTES, *tB = sB + (kt % STAGES) * B_BYTES;
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      uint32_t a[MI][4], b[NJ][2];
#pragma unroll
      for (int i = 0; i < MI; i++)
        ldsm_x4(a[i][0], a[i][1], a[i][2], a[i][3], smem_u32(tA + swz(wm * TM + i * 16 + (mi_l & 1) * 8 + r, kk * 2 + (mi_l >> 1))));
#pragma unroll
      for (int j2 = 0; j2 < NJ / 2; j2++)
        ldsm_x4(b[j2 * 2][0], b[j2 * 2][1], b[j2 * 2 + 1][0], b[j2 * 2 + 1][1],
                smem_u32(tB + swz(wn * TN + j2 * 16 + (mi_l >> 1) * 8 + r, kk * 2 + (mi_l & 1))));
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++) mma_f16acc(h[i][j], a[i], b[j][0], b[j][1]);
      if (++step == 16) {
        step = 0;
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
          for (int j = 0; j < NJ; j++) {
            float2 lo = __half22float2(*reinterpret_cast<__half2 *>(&h[i][j][0])), hi = __half22float2(*reinterpret_cast<__half2 *>(&h[i][j][1]));
            f[i][j][0] += lo.x; f[i][j][1] += lo.y; f[i][j][2] += hi.x; f[i][j][3] += hi.y;
            h[i][j][0] = h[i][j][1] = 0;
          }
      }
    }
  }
  asm volatile("cp.async.wait_group 0;");
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++) {
      float2 lo = __half22float2(*reinterpret_cast<__half2 *>(&h[i][j][0])), hi = __half22float2(*reinterpret_cast<__half2 *>(&h[i][j][1]));
      f[i][j][0] += lo.x; f[i][j][1] += lo.y; f[i][j][2] += hi.x; f[i][j][3] += hi.y;
    }
  const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
  for (int j = 0; j < NJ; j++) {
    const int col = n0 + wn * TN + j * 8 + t4 * 2;
    const float s0 = sa * sw[col], s1 = sa * sw[col + 1];
    const float b0 = bias ? __bfloat162float(bias[col]) : 0.f, b1 = bias ? __bfloat162float(bias[col + 1]) : 0.f;
#pragma unroll
    for (int i = 0; i < MI; i++) {
      const int row = m0 + wm * TM + i * 16 + g;
      if (row < M) *reinterpret_cast<__nv_bfloat162 *>(Y + (size_t)row * N + col) = __floats2bfloat162_rn(f[i][j][0] * s0 + b0, f[i][j][1] * s1 + b1);
      if (row + 8 < M) *reinterpret_cast<__nv_bfloat162 *>(Y + (size_t)(row + 8) * N + col) = __floats2bfloat162_rn(f[i][j][2] * s0 + b0, f[i][j][3] * s1 + b1);
    }
  }
}
}

template <int EPI>
torch::Tensor run_a16(torch::Tensor x, torch::Tensor wq, torch::Tensor sw, c10::optional<torch::Tensor> bias, double sa, double out_scale) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous());
  TORCH_CHECK(wq.scalar_type() == torch::kFloat8_e4m3fn && wq.is_contiguous());
  const int K = x.size(-1), N = wq.size(0);
  TORCH_CHECK(wq.size(1) == K && K % BK == 0 && N % BN == 0, "shape");
  const int M = x.numel() / K;
  auto y = torch::empty({M, N}, EPI == 0 ? x.options() : x.options().dtype(torch::kFloat8_e4m3fn));
  const __nv_bfloat16 *bp = nullptr;
  if (bias.has_value()) bp = reinterpret_cast<const __nv_bfloat16 *>(bias->data_ptr());
  const int smem = STAGES * (A_BYTES + B_BYTES);
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(n1_a16<16, EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
  const int blocks = ((M + BM - 1) / BM) * (N / BN);
  n1_a16<16, EPI><<<blocks, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), reinterpret_cast<const uint8_t *>(wq.data_ptr()), sw.data_ptr<float>(), bp,
      y.data_ptr(), M, N, K, float(1.0 / sa), float(sa), float(1.0 / out_scale));
  auto sizes = x.sizes().vec();
  sizes.back() = N;
  return y.view(sizes);
}
torch::Tensor n1_linear(torch::Tensor x, torch::Tensor wq, torch::Tensor sw, c10::optional<torch::Tensor> bias, double sa) {
  return run_a16<0>(x, wq, sw, bias, sa, 1.0);
}

torch::Tensor n1_linear_gelu_f8(torch::Tensor x, torch::Tensor wq, torch::Tensor sw, c10::optional<torch::Tensor> bias, double sa, double out_scale) {
  return run_a16<1>(x, wq, sw, bias, sa, out_scale);
}

torch::Tensor n1_linear_f8in(torch::Tensor x, torch::Tensor wq, torch::Tensor sw, c10::optional<torch::Tensor> bias, double sa) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat8_e4m3fn && x.is_contiguous());
  const int K = x.size(-1), N = wq.size(0);
  TORCH_CHECK(wq.size(1) == K && K % BK == 0 && N % BN == 0, "shape");
  const int M = x.numel() / K;
  auto y = torch::empty({M, N}, x.options().dtype(torch::kBFloat16));
  const __nv_bfloat16 *bp = nullptr;
  if (bias.has_value()) bp = reinterpret_cast<const __nv_bfloat16 *>(bias->data_ptr());
  const int smem = STAGES * (A_BYTES + B_BYTES);
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(n1_f8, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
  const int blocks = ((M + BM - 1) / BM) * (N / BN);
  n1_f8<<<blocks, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t *>(x.data_ptr()), reinterpret_cast<const uint8_t *>(wq.data_ptr()), sw.data_ptr<float>(), bp,
      reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), M, N, K, float(sa));
  auto sizes = x.sizes().vec();
  sizes.back() = N;
  return y.view(sizes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("n1_linear", &n1_linear, "FP8 ladder linear with fused activation quantization");
  m.def("n1_linear_gelu_f8", &n1_linear_gelu_f8, "FP8 ladder linear + tanh-GELU, e4m3 output");
  m.def("n1_linear_f8in", &n1_linear_f8in, "FP8 ladder linear, e4m3 input");
}
