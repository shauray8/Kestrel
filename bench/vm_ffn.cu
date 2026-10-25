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

constexpr int BM = 128, BN = 128, BK = 64, STAGES = 3, THREADS = 256, WN = 4, TM = 64, TN = 32, MI = 4, NJ = 4;
constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK, SMEM = STAGES * (A_BYTES + B_BYTES);

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ int swz(int row, int chunk) { return row * BK + ((chunk ^ ((row >> 1) & 3)) << 4); }
__device__ __forceinline__ void cp16(uint32_t dst, const void *src, bool p) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(p ? 16 : 0));
}
__device__ __forceinline__ void ldsm_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma8(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float gelu(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }

__device__ __forceinline__ void tile_mma(const uint8_t *A, const uint8_t *W, int M, int K, int m0, int n0, uint8_t *smem,
                                         float (&f)[MI][NJ][4]) {
  uint8_t *sA = smem, *sB = smem + STAGES * A_BYTES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
  const int KT = K / BK;
  auto load = [&](int stage, int kt) {
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int c = tid + i * THREADS, row = c >> 2, ch = c & 3, gm = m0 + row;
      cp16(smem_u32(sA + stage * A_BYTES + swz(row, ch)), A + (size_t)(gm < M ? gm : 0) * K + kt * BK + ch * 16, gm < M);
      cp16(smem_u32(sB + stage * B_BYTES + swz(row, ch)), W + (size_t)(n0 + row) * K + kt * BK + ch * 16, true);
    }
  };
  uint32_t h[MI][NJ][2];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++) { h[i][j][0] = h[i][j][1] = 0; f[i][j][0] = f[i][j][1] = f[i][j][2] = f[i][j][3] = 0.f; }
  __syncthreads();
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) { if (s < KT) load(s, s); asm volatile("cp.async.commit_group;"); }
  int step = 0;
  const int mi = lane >> 3, r = lane & 7;
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
      for (int i = 0; i < MI; i++) ldsm_x4(a[i][0], a[i][1], a[i][2], a[i][3], smem_u32(tA + swz(wm * TM + i * 16 + (mi & 1) * 8 + r, kk * 2 + (mi >> 1))));
#pragma unroll
      for (int j2 = 0; j2 < NJ / 2; j2++)
        ldsm_x4(b[j2 * 2][0], b[j2 * 2][1], b[j2 * 2 + 1][0], b[j2 * 2 + 1][1], smem_u32(tB + swz(wn * TN + j2 * 16 + (mi >> 1) * 8 + r, kk * 2 + (mi & 1))));
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++) mma8(h[i][j], a[i], b[j][0], b[j][1]);
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
}

struct Ffn {
  const uint8_t *x, *w0, *w2; uint8_t *h; __nv_bfloat16 *y;
  int M, D, F; float s_up, inv_mid, s_dn;
};

__device__ __forceinline__ void epi_up(const Ffn &p, int m0, int n0, float (&f)[MI][NJ][4]) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wm = warp / WN, wn = warp % WN, g = lane >> 2, t4 = lane & 3;
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++) {
      const int col = n0 + wn * TN + j * 8 + t4 * 2, row = m0 + wm * TM + i * 16 + g;
      __nv_fp8x2_e4m3 q0(make_float2(gelu(f[i][j][0] * p.s_up) * p.inv_mid, gelu(f[i][j][1] * p.s_up) * p.inv_mid));
      __nv_fp8x2_e4m3 q1(make_float2(gelu(f[i][j][2] * p.s_up) * p.inv_mid, gelu(f[i][j][3] * p.s_up) * p.inv_mid));
      if (row < p.M) *reinterpret_cast<uint16_t *>(p.h + (size_t)row * p.F + col) = q0.__x;
      if (row + 8 < p.M) *reinterpret_cast<uint16_t *>(p.h + (size_t)(row + 8) * p.F + col) = q1.__x;
    }
}
__device__ __forceinline__ void epi_dn(const Ffn &p, int m0, int n0, float (&f)[MI][NJ][4]) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wm = warp / WN, wn = warp % WN, g = lane >> 2, t4 = lane & 3;
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++) {
      const int col = n0 + wn * TN + j * 8 + t4 * 2, row = m0 + wm * TM + i * 16 + g;
      if (row < p.M) *reinterpret_cast<__nv_bfloat162 *>(p.y + (size_t)row * p.D + col) = __floats2bfloat162_rn(f[i][j][0] * p.s_dn, f[i][j][1] * p.s_dn);
      if (row + 8 < p.M) *reinterpret_cast<__nv_bfloat162 *>(p.y + (size_t)(row + 8) * p.D + col) = __floats2bfloat162_rn(f[i][j][2] * p.s_dn, f[i][j][3] * p.s_dn);
    }
}

__global__ void __launch_bounds__(THREADS, 2) k_up(Ffn p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int ntn = p.F / BN, m = blockIdx.x / ntn, n = blockIdx.x % ntn;
  float f[MI][NJ][4];
  tile_mma(p.x, p.w0, p.M, p.D, m * BM, n * BN, smem, f);
  epi_up(p, m * BM, n * BN, f);
}
__global__ void __launch_bounds__(THREADS, 2) k_dn(Ffn p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int ntn = p.D / BN, m = blockIdx.x / ntn, n = blockIdx.x % ntn;
  float f[MI][NJ][4];
  tile_mma(p.h, p.w2, p.M, p.F, m * BM, n * BN, smem, f);
  epi_dn(p, m * BM, n * BN, f);
}

__global__ void __launch_bounds__(THREADS, 2) k_vm(Ffn p, const uint32_t *__restrict__ queue, int n_instr, int *head, int *ready) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_idx;
  const int up_per_row = p.F / BN;
  for (;;) {
    if (threadIdx.x == 0) s_idx = atomicAdd(head, 1);
    __syncthreads();
    const int idx = s_idx;
    __syncthreads();
    if (idx >= n_instr) return;
    const uint32_t ins = queue[idx];
    const int op = ins >> 31, m = (ins >> 16) & 0x7fff, n = ins & 0xffff;
    float f[MI][NJ][4];
    if (op == 0) {
      tile_mma(p.x, p.w0, p.M, p.D, m * BM, n * BN, smem, f);
      epi_up(p, m * BM, n * BN, f);
      __syncthreads();
      if (threadIdx.x == 0) { __threadfence(); atomicAdd(&ready[m], 1); }
    } else {
      if (threadIdx.x == 0) {
        while (atomicAdd(&ready[m], 0) < up_per_row) __nanosleep(200);
        __threadfence();
      }
      __syncthreads();
      tile_mma(p.h, p.w2, p.M, p.F, m * BM, n * BN, smem, f);
      epi_dn(p, m * BM, n * BN, f);
    }
  }
}

static uint8_t e4(float v) { __nv_fp8_e4m3 q(v); return q.__x; }

int main() {
  const int M = 4680, D = 1536, F = 8960, mb = (M + BM - 1) / BM, up_n = F / BN, dn_n = D / BN;
  srand(3);
  auto rnd = [] { return (rand() / (float)RAND_MAX) * 2.f - 1.f; };
  std::vector<uint8_t> hx((size_t)M * D), hw0((size_t)F * D), hw2((size_t)D * F);
  for (auto &v : hx) v = e4(rnd() * 8.f);
  for (auto &v : hw0) v = e4(rnd() * 8.f);
  for (auto &v : hw2) v = e4(rnd() * 8.f);
  Ffn p{};
  uint8_t *dx, *dw0, *dw2, *dh, *dh2; __nv_bfloat16 *dy, *dy2;
  CK(cudaMalloc(&dx, hx.size())); CK(cudaMalloc(&dw0, hw0.size())); CK(cudaMalloc(&dw2, hw2.size()));
  CK(cudaMalloc(&dh, (size_t)M * F)); CK(cudaMalloc(&dh2, (size_t)M * F));
  CK(cudaMalloc(&dy, (size_t)M * D * 2)); CK(cudaMalloc(&dy2, (size_t)M * D * 2));
  CK(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw0, hw0.data(), hw0.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw2, hw2.data(), hw2.size(), cudaMemcpyHostToDevice));
  p.x = dx; p.w0 = dw0; p.w2 = dw2; p.M = M; p.D = D; p.F = F;
  p.s_up = 1.f / 1536.f; p.inv_mid = 8.f; p.s_dn = 1.f / 8960.f;
  CK(cudaFuncSetAttribute(k_up, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  CK(cudaFuncSetAttribute(k_dn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  CK(cudaFuncSetAttribute(k_vm, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));

  auto make_queue = [&](int lag) {
    std::vector<uint32_t> q;
    for (int r = 0; r < mb + lag; r++) {
      if (r < mb) for (int n = 0; n < up_n; n++) q.push_back((0u << 31) | (uint32_t(r) << 16) | n);
      int d = r - lag;
      if (lag >= 0 && d >= 0 && d < mb) for (int n = 0; n < dn_n; n++) q.push_back((1u << 31) | (uint32_t(d) << 16) | n);
    }
    return q;
  };
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int *dhead, *dready; uint32_t *dq;
  CK(cudaMalloc(&dhead, 4)); CK(cudaMalloc(&dready, mb * 4)); CK(cudaMalloc(&dq, 4 * (mb * (up_n + dn_n))));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  const double flops = 2.0 * 2 * M * (double)D * F;

  auto run_base = [&]() { p.h = dh; p.y = dy; k_up<<<mb * up_n, THREADS, SMEM>>>(p); k_dn<<<mb * dn_n, THREADS, SMEM>>>(p); };
  for (int i = 0; i < 3; i++) run_base();
  CK(cudaDeviceSynchronize());
  cudaEventRecord(e0); for (int i = 0; i < 20; i++) run_base(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
  float ms_base; cudaEventElapsedTime(&ms_base, e0, e1); ms_base /= 20;
  std::vector<__nv_bfloat16> yb((size_t)M * D);
  CK(cudaMemcpy(yb.data(), dy, yb.size() * 2, cudaMemcpyDeviceToHost));
  printf("{\"kind\":\"vm_ffn\",\"variant\":\"two_launches\",\"ms\":%.4f,\"tf\":%.1f}\n", ms_base, flops / (ms_base * 1e-3) / 1e12);

  for (int lag : {mb, 24, 16, 12, 8, 6, 4}) {
    auto q = make_queue(lag);
    CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    auto run_vm = [&]() {
      p.h = dh2; p.y = dy2;
      cudaMemsetAsync(dhead, 0, 4); cudaMemsetAsync(dready, 0, mb * 4);
      k_vm<<<2 * sms, THREADS, SMEM>>>(p, dq, (int)q.size(), dhead, dready);
    };
    for (int i = 0; i < 3; i++) run_vm();
    CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for (int i = 0; i < 20; i++) run_vm(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 20;
    std::vector<__nv_bfloat16> yv((size_t)M * D);
    CK(cudaMemcpy(yv.data(), dy2, yv.size() * 2, cudaMemcpyDeviceToHost));
    double md = 0; for (size_t i = 0; i < yv.size(); i++) md = fmax(md, fabs((double)__bfloat162float(yv[i]) - __bfloat162float(yb[i])));
    printf("{\"kind\":\"vm_ffn\",\"variant\":\"vm_lag%d\",\"instr\":%zu,\"ms\":%.4f,\"tf\":%.1f,\"speedup_vs_two_launches\":%.3f,\"max_abs_diff_vs_base\":%.3g}\n",
           lag, q.size(), ms, flops / (ms * 1e-3) / 1e12, ms_base / ms, md);
    fflush(stdout);
  }
  return 0;
}
