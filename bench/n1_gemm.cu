#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

constexpr int BK = 64;

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ int swz(int row, int chunk) { return row * BK + ((chunk ^ ((row >> 1) & 3)) << 4); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src, bool pred) {
  int n = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void ldsm_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_f16acc(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f32acc(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int P, int BM, int BN, int WM, int WN, int STAGES, int MINB>
__global__ void __launch_bounds__(WM * WN * 32, MINB) n1_gemm(const uint8_t *__restrict__ A, const uint8_t *__restrict__ B,
                                                       __nv_bfloat16 *__restrict__ C, int M, int N, int K, float scale) {
  constexpr int THREADS = WM * WN * 32, TM = BM / WM, TN = BN / WN, MI = TM / 16, NJ = TN / 8;
  constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK;
  static_assert(NJ % 2 == 0 && (BM * 4) % THREADS == 0 && (BN * 4) % THREADS == 0, "tile config");
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t *sA = smem, *sB = smem + STAGES * A_BYTES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;

  const int grid_m = (M + BM - 1) / BM, grid_n = N / BN, pid = blockIdx.x;
  const int group = pid / (8 * grid_n), first_m = group * 8, gsize = min(grid_m - first_m, 8);
  const int bm = first_m + (pid % (8 * grid_n)) % gsize, bn = (pid % (8 * grid_n)) / gsize;
  const int m0 = bm * BM, n0 = bn * BN;
  const int KT = K / BK;

  auto load_stage = [&](int stage, int kt) {
#pragma unroll
    for (int i = 0; i < BM * 4 / THREADS; i++) {
      int c = tid + i * THREADS, row = c >> 2, ch = c & 3, gm = m0 + row;
      cp_async16(smem_u32(sA + stage * A_BYTES + swz(row, ch)), A + (size_t)(gm < M ? gm : 0) * K + kt * BK + ch * 16, gm < M);
    }
#pragma unroll
    for (int i = 0; i < BN * 4 / THREADS; i++) {
      int c = tid + i * THREADS, row = c >> 2, ch = c & 3, gn = n0 + row;
      cp_async16(smem_u32(sB + stage * B_BYTES + swz(row, ch)), B + (size_t)gn * K + kt * BK + ch * 16, true);
    }
  };

  uint32_t h[MI][NJ][2];
  float f[MI][NJ][4];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++) {
      h[i][j][0] = h[i][j][1] = 0;
      f[i][j][0] = f[i][j][1] = f[i][j][2] = f[i][j][3] = 0.f;
    }

#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KT) load_stage(s, s);
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
    int nk = kt + STAGES - 1;
    if (nk < KT) load_stage(nk % STAGES, nk);
    asm volatile("cp.async.commit_group;");
    const uint8_t *tA = sA + (kt % STAGES) * A_BYTES, *tB = sB + (kt % STAGES) * B_BYTES;
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      uint32_t a[MI][4], b[NJ][2];
#pragma unroll
      for (int i = 0; i < MI; i++)
        ldsm_x4(a[i][0], a[i][1], a[i][2], a[i][3],
                smem_u32(tA + swz(wm * TM + i * 16 + (mi_l & 1) * 8 + r, kk * 2 + (mi_l >> 1))));
#pragma unroll
      for (int j2 = 0; j2 < NJ / 2; j2++)
        ldsm_x4(b[j2 * 2][0], b[j2 * 2][1], b[j2 * 2 + 1][0], b[j2 * 2 + 1][1],
                smem_u32(tB + swz(wn * TN + j2 * 16 + (mi_l >> 1) * 8 + r, kk * 2 + (mi_l & 1))));
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++) {
          if constexpr (P == 0) mma_f32acc(f[i][j], a[i], b[j][0], b[j][1]);
          else mma_f16acc(h[i][j], a[i], b[j][0], b[j][1]);
        }
      if constexpr (P > 0) {
        if (++step == P) { step = 0; promote(); }
      }
    }
  }
  asm volatile("cp.async.wait_group 0;");
  if constexpr (P > 0) promote();
  const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++) {
      int col = n0 + wn * TN + j * 8 + t4 * 2, row = m0 + wm * TM + i * 16 + g;
      if (row < M)
        *reinterpret_cast<__nv_bfloat162 *>(C + (size_t)row * N + col) = __floats2bfloat162_rn(f[i][j][0] * scale, f[i][j][1] * scale);
      if (row + 8 < M)
        *reinterpret_cast<__nv_bfloat162 *>(C + (size_t)(row + 8) * N + col) = __floats2bfloat162_rn(f[i][j][2] * scale, f[i][j][3] * scale);
    }
}

__global__ void ref_gemm(const uint8_t *A, const uint8_t *B, float *C, int M, int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
  if (n >= N) return;
  float s = 0.f;
  for (int k = 0; k < K; k++) {
    __nv_fp8_e4m3 a, b;
    a.__x = A[(size_t)m * K + k]; b.__x = B[(size_t)n * K + k];
    s += float(a) * float(b);
  }
  C[(size_t)m * N + n] = s;
}

template <int P, int BM, int BN, int WM, int WN, int ST, int MINB = 1>
struct Cfg {
  static constexpr int smem = ST * (BM + BN) * BK;
  static void launch(const uint8_t *A, const uint8_t *B, __nv_bfloat16 *C, int M, int N, int K, float scale) {
    static bool init = false;
    if (!init) { CK(cudaFuncSetAttribute(n1_gemm<P, BM, BN, WM, WN, ST, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); init = true; }
    int blocks = ((M + BM - 1) / BM) * (N / BN);
    n1_gemm<P, BM, BN, WM, WN, ST, MINB><<<blocks, WM * WN * 32, smem>>>(A, B, C, M, N, K, scale);
  }
  static double bench(const uint8_t *A, const uint8_t *B, __nv_bfloat16 *C, int M, int N, int K) {
    if (N % BN) return -1;
    for (int i = 0; i < 3; i++) launch(A, B, C, M, N, K, 1.f);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    const int it = 20;
    cudaEventRecord(s);
    for (int i = 0; i < it; i++) launch(A, B, C, M, N, K, 1.f);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms, s, e);
    return 2.0 * M * N * K / (ms / it * 1e-3) / 1e12;
  }
  static void check(const char *name, uint8_t *dA, uint8_t *dB, int M, int N, int K, const std::vector<float> &ref, float tgt) {
    __nv_bfloat16 *dC; CK(cudaMalloc(&dC, (size_t)M * N * 2));
    launch(dA, dB, dC, M, N, K, 1.f);
    CK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> hC((size_t)M * N);
    CK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost));
    double num = 0, den = 0; int bad = 0;
    for (size_t i = 0; i < hC.size(); i++) {
      double c = __bfloat162float(hC[i]), rr = ref[i];
      if (!std::isfinite(c)) { bad++; continue; }
      num += (c - rr) * (c - rr); den += rr * rr;
    }
    printf("{\"kind\":\"n1_check\",\"variant\":\"%s\",\"tile\":\"%dx%d_w%dx%d_s%d\",\"K\":%d,\"amax_target\":%.0f,\"rel_err\":%.3e,\"nonfinite\":%d}\n",
           name, BM, BN, WM, WN, ST, K, tgt, std::sqrt(num / den), bad);
    cudaFree(dC);
  }
};

static uint8_t to_e4m3(float x) { __nv_fp8_e4m3 v(x); return v.__x; }

using T128_F32 = Cfg<0, 128, 128, 2, 4, 4>;
using T128_F32_2cta = Cfg<0, 128, 128, 2, 4, 3, 2>;
using T128x256 = Cfg<16, 128, 256, 2, 4, 3>;
using T128_2cta = Cfg<16, 128, 128, 2, 4, 3, 2>;
using T128_2cta_P32 = Cfg<32, 128, 128, 2, 4, 3, 2>;
using T128_2cta_P8 = Cfg<8, 128, 128, 2, 4, 3, 2>;
using T128_2cta_never = Cfg<1000000, 128, 128, 2, 4, 3, 2>;

int main() {
  {
    const int M = 256, N = 256, K = 8960;
    const float target = 64.f;
    std::vector<uint8_t> hA((size_t)M * K), hB((size_t)N * K);
    srand(1);
    auto rnd = [] { return ((rand() / (float)RAND_MAX) * 2.f - 1.f) + ((rand() / (float)RAND_MAX) * 2.f - 1.f); };
    std::vector<float> fa(hA.size()), fb(hB.size());
    float ma = 0, mb = 0;
    for (auto &v : fa) { float x = rnd() * 1.5f; v = 0.5f * x * (1.f + tanhf(0.79788456f * (x + 0.044715f * x * x * x))); ma = fmaxf(ma, fabsf(v)); }
    for (auto &v : fb) { v = rnd(); mb = fmaxf(mb, fabsf(v)); }
    for (size_t i = 0; i < hA.size(); i++) hA[i] = to_e4m3(fa[i] / ma * target);
    for (size_t i = 0; i < hB.size(); i++) hB[i] = to_e4m3(fb[i] / mb * target);
    uint8_t *dA, *dB; float *dR;
    CK(cudaMalloc(&dA, hA.size())); CK(cudaMalloc(&dB, hB.size())); CK(cudaMalloc(&dR, (size_t)M * N * 4));
    CK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    ref_gemm<<<dim3((N + 127) / 128, M), 128>>>(dA, dB, dR, M, N, K);
    std::vector<float> ref((size_t)M * N);
    CK(cudaMemcpy(ref.data(), dR, ref.size() * 4, cudaMemcpyDeviceToHost));
    T128_F32::check("f32acc", dA, dB, M, N, K, ref, target);
    T128_F32_2cta::check("f32acc", dA, dB, M, N, K, ref, target);
    T128x256::check("ladder_P16", dA, dB, M, N, K, ref, target);
    T128_2cta::check("ladder_P16", dA, dB, M, N, K, ref, target);
    T128_2cta_P32::check("ladder_P32", dA, dB, M, N, K, ref, target);
    T128_2cta_P8::check("ladder_P8", dA, dB, M, N, K, ref, target);
    cudaFree(dA); cudaFree(dB); cudaFree(dR);
  }
  struct S { const char *name; int M, N, K; };
  S shapes[] = {{"lingbot_qkvo_1536", 4680, 1536, 1536}, {"lingbot_qkv_fused", 4680, 4608, 1536},
                {"lingbot_ffn_up", 4680, 8960, 1536}, {"lingbot_ffn_down", 4680, 1536, 8960},
                {"nomas_qkv", 11700, 6144, 2048}, {"nomas_fc1", 11700, 8192, 2048}, {"nomas_fc2", 11700, 2048, 8192},
                {"nomasdec_gate_up", 15600, 24576, 4096}, {"nomasdec_down", 15600, 4096, 12288},
                {"square_8192", 8192, 8192, 8192}};
  for (auto &s : shapes) {
    uint8_t *dA, *dB; __nv_bfloat16 *dC;
    CK(cudaMalloc(&dA, (size_t)s.M * s.K)); CK(cudaMalloc(&dB, (size_t)s.N * s.K)); CK(cudaMalloc(&dC, (size_t)s.M * s.N * 2));
    CK(cudaMemset(dA, 0x38, (size_t)s.M * s.K)); CK(cudaMemset(dB, 0x30, (size_t)s.N * s.K));
    printf("{\"kind\":\"n1_speed\",\"shape\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"f32acc_1cta\":%.1f,\"f32acc_2cta\":%.1f,"
           "\"P16_128x256\":%.1f,\"P8_2cta\":%.1f,\"P16_2cta\":%.1f,\"P32_2cta\":%.1f,\"f16only_2cta\":%.1f}\n",
           s.name, s.M, s.N, s.K, T128_F32::bench(dA, dB, dC, s.M, s.N, s.K), T128_F32_2cta::bench(dA, dB, dC, s.M, s.N, s.K),
           T128x256::bench(dA, dB, dC, s.M, s.N, s.K), T128_2cta_P8::bench(dA, dB, dC, s.M, s.N, s.K),
           T128_2cta::bench(dA, dB, dC, s.M, s.N, s.K), T128_2cta_P32::bench(dA, dB, dC, s.M, s.N, s.K),
           T128_2cta_never::bench(dA, dB, dC, s.M, s.N, s.K));
    fflush(stdout);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
  }
  return 0;
}
