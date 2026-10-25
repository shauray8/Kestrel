#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {
constexpr int D = 128, BQ = 128, BKV = 64, WARPS = 8, THREADS = WARPS * 32;
constexpr int ROWB = D * 2;

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ int swz(int row, int chunk) { return row * ROWB + ((chunk ^ (row & 7)) << 4); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void ldsm_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma16(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t *>(&h);
}

template <bool HSM>
__global__ void __launch_bounds__(THREADS, 1) n2_attn(const half *__restrict__ Q, const half *__restrict__ K, const half *__restrict__ V,
                                                      half *__restrict__ O, int Lq, int Lk, int H, float scale_log2) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t *sQ = smem, *sK = smem + BQ * ROWB, *sV = sK + 2 * BKV * ROWB;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, head = blockIdx.y, q0 = blockIdx.x * BQ;
  const size_t tok = (size_t)H * D;
  const half *Qh = Q + head * D, *Kh = K + head * D, *Vh = V + head * D;

#pragma unroll
  for (int i = 0; i < BQ * 16 / THREADS; i++) {
    int c = tid + i * THREADS, row = c >> 4, ch = c & 15, gq = q0 + row;
    cp_async16(smem_u32(sQ + swz(row, ch)), Qh + (size_t)(gq < Lq ? gq : 0) * tok + ch * 8, gq < Lq);
  }
  auto load_kv = [&](int buf, int kb) {
#pragma unroll
    for (int i = 0; i < BKV * 16 / THREADS; i++) {
      int c = tid + i * THREADS, row = c >> 4, ch = c & 15, gk = kb * BKV + row;
      bool p = gk < Lk;
      size_t off = (size_t)(p ? gk : 0) * tok + ch * 8;
      cp_async16(smem_u32(sK + buf * BKV * ROWB + swz(row, ch)), Kh + off, p);
      cp_async16(smem_u32(sV + buf * BKV * ROWB + swz(row, ch)), Vh + off, p);
    }
  };
  load_kv(0, 0);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();

  const int mi = lane >> 3, r = lane & 7, g = lane >> 2, t4 = lane & 3;

  uint32_t qa[8][4];
#pragma unroll
  for (int kk = 0; kk < 8; kk++)
    ldsm_x4(qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], smem_u32(sQ + swz(warp * 16 + (mi & 1) * 8 + r, kk * 2 + (mi >> 1))));
  if constexpr (HSM) {
    const __half2 sc = __float2half2_rn(scale_log2);
#pragma unroll
    for (int kk = 0; kk < 8; kk++)
#pragma unroll
      for (int i = 0; i < 4; i++) {
        __half2 t = __hmul2(*reinterpret_cast<__half2 *>(&qa[kk][i]), sc);
        qa[kk][i] = *reinterpret_cast<uint32_t *>(&t);
      }
  }

  float o[16][4];
#pragma unroll
  for (int j = 0; j < 16; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  const int nkb = (Lk + BKV - 1) / BKV;
  for (int kb = 0; kb < nkb; kb++) {
    const int buf = kb & 1;
    if (kb + 1 < nkb) load_kv(buf ^ 1, kb + 1);
    asm volatile("cp.async.commit_group;");
    const uint8_t *tK = sK + buf * BKV * ROWB, *tV = sV + buf * BKV * ROWB;

    uint32_t s[8][2];
#pragma unroll
    for (int j = 0; j < 8; j++) s[j][0] = s[j][1] = 0;
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
#pragma unroll
      for (int j2 = 0; j2 < 4; j2++) {
        uint32_t b0, b1, b2, b3;
        ldsm_x4(b0, b1, b2, b3, smem_u32(tK + swz(j2 * 16 + (mi >> 1) * 8 + r, kk * 2 + (mi & 1))));
        mma16(s[j2 * 2], qa[kk], b0, b1);
        mma16(s[j2 * 2 + 1], qa[kk], b2, b3);
      }
    }
    float al0, al1;
    uint32_t pa[4][4];
    if constexpr (HSM) {

      if (kb * BKV + BKV > Lk) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const int key = kb * BKV + j * 8 + t4 * 2;
          const __half ninf = __ushort_as_half(0xFC00);
          __half2 *a = reinterpret_cast<__half2 *>(&s[j][0]), *b = reinterpret_cast<__half2 *>(&s[j][1]);
          if (key >= Lk) { a->x = ninf; b->x = ninf; }
          if (key + 1 >= Lk) { a->y = ninf; b->y = ninf; }
        }
      }
      __half2 hm0 = *reinterpret_cast<__half2 *>(&s[0][0]), hm1 = *reinterpret_cast<__half2 *>(&s[0][1]);
#pragma unroll
      for (int j = 1; j < 8; j++) {
        hm0 = __hmax2(hm0, *reinterpret_cast<__half2 *>(&s[j][0]));
        hm1 = __hmax2(hm1, *reinterpret_cast<__half2 *>(&s[j][1]));
      }
      float mx0 = fmaxf(m0, fmaxf(__low2float(hm0), __high2float(hm0)));
      float mx1 = fmaxf(m1, fmaxf(__low2float(hm1), __high2float(hm1)));
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
      al0 = exp2f(m0 - mx0); al1 = exp2f(m1 - mx1);
      m0 = mx0; m1 = mx1;
      const __half2 hx0 = __float2half2_rn(mx0), hx1 = __float2half2_rn(mx1);
      __half2 rs0 = __float2half2_rn(0.f), rs1 = rs0;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        __half2 p0 = h2exp2(__hsub2(*reinterpret_cast<__half2 *>(&s[j][0]), hx0));
        __half2 p1 = h2exp2(__hsub2(*reinterpret_cast<__half2 *>(&s[j][1]), hx1));
        rs0 = __hadd2(rs0, p0); rs1 = __hadd2(rs1, p1);
        pa[j >> 1][(j & 1) * 2 + 0] = *reinterpret_cast<uint32_t *>(&p0);
        pa[j >> 1][(j & 1) * 2 + 1] = *reinterpret_cast<uint32_t *>(&p1);
      }
      l0 = l0 * al0 + __low2float(rs0) + __high2float(rs0);
      l1 = l1 * al1 + __low2float(rs1) + __high2float(rs1);
    } else {

      float sf[8][4];
      float mx0 = m0, mx1 = m1;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float2 a = __half22float2(*reinterpret_cast<__half2 *>(&s[j][0])), b = __half22float2(*reinterpret_cast<__half2 *>(&s[j][1]));
        const int key = kb * BKV + j * 8 + t4 * 2;
        sf[j][0] = key < Lk ? a.x * scale_log2 : -INFINITY;
        sf[j][1] = key + 1 < Lk ? a.y * scale_log2 : -INFINITY;
        sf[j][2] = key < Lk ? b.x * scale_log2 : -INFINITY;
        sf[j][3] = key + 1 < Lk ? b.y * scale_log2 : -INFINITY;
        mx0 = fmaxf(mx0, fmaxf(sf[j][0], sf[j][1]));
        mx1 = fmaxf(mx1, fmaxf(sf[j][2], sf[j][3]));
      }
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
      al0 = exp2f(m0 - mx0); al1 = exp2f(m1 - mx1);
      m0 = mx0; m1 = mx1;
      float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float p0 = exp2f(sf[j][0] - mx0), p1 = exp2f(sf[j][1] - mx0), p2 = exp2f(sf[j][2] - mx1), p3 = exp2f(sf[j][3] - mx1);
        rs0 += p0 + p1; rs1 += p2 + p3;
        const int kk = j >> 1, hi = j & 1;
        pa[kk][hi * 2 + 0] = pack_h2(p0, p1);
        pa[kk][hi * 2 + 1] = pack_h2(p2, p3);
      }
      l0 = l0 * al0 + rs0; l1 = l1 * al1 + rs1;
    }

    uint32_t oh[16][2];
#pragma unroll
    for (int j = 0; j < 16; j++) oh[j][0] = oh[j][1] = 0;
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
#pragma unroll
      for (int n2 = 0; n2 < 8; n2++) {
        uint32_t b0, b1, b2, b3;
        ldsm_x4_t(b0, b1, b2, b3, smem_u32(tV + swz(kk * 16 + (mi & 1) * 8 + r, n2 * 2 + (mi >> 1))));
        mma16(oh[n2 * 2], pa[kk], b0, b1);
        mma16(oh[n2 * 2 + 1], pa[kk], b2, b3);
      }
    }
    if (al0 == 1.f && al1 == 1.f) {
#pragma unroll
      for (int j = 0; j < 16; j++) {
        float2 a = __half22float2(*reinterpret_cast<__half2 *>(&oh[j][0])), b = __half22float2(*reinterpret_cast<__half2 *>(&oh[j][1]));
        o[j][0] += a.x; o[j][1] += a.y; o[j][2] += b.x; o[j][3] += b.y;
      }
    } else {
#pragma unroll
      for (int j = 0; j < 16; j++) {
        float2 a = __half22float2(*reinterpret_cast<__half2 *>(&oh[j][0])), b = __half22float2(*reinterpret_cast<__half2 *>(&oh[j][1]));
        o[j][0] = o[j][0] * al0 + a.x; o[j][1] = o[j][1] * al0 + a.y;
        o[j][2] = o[j][2] * al1 + b.x; o[j][3] = o[j][3] * al1 + b.y;
      }
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
  }
  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float i0 = 1.f / l0, i1 = 1.f / l1;
  const int row0 = q0 + warp * 16 + g, row1 = row0 + 8;
  half *Oh = O + head * D;
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const int col = j * 8 + t4 * 2;
    if (row0 < Lq) *reinterpret_cast<__half2 *>(Oh + (size_t)row0 * tok + col) = __floats2half2_rn(o[j][0] * i0, o[j][1] * i0);
    if (row1 < Lq) *reinterpret_cast<__half2 *>(Oh + (size_t)row1 * tok + col) = __floats2half2_rn(o[j][2] * i1, o[j][3] * i1);
  }
}

constexpr int W2 = 4, T2 = W2 * 32;
__global__ void __launch_bounds__(T2, 1) n2_attn2(const half *__restrict__ Q, const half *__restrict__ K, const half *__restrict__ V,
                                                 half *__restrict__ O, int Lq, int Lk, int H, float scale_log2) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t *sQ = smem, *sK = smem + BQ * ROWB, *sV = sK + 2 * BKV * ROWB;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, head = blockIdx.y, q0 = blockIdx.x * BQ;
  const size_t tok = (size_t)H * D;
  const half *Qh = Q + head * D, *Kh = K + head * D, *Vh = V + head * D;
#pragma unroll
  for (int i = 0; i < BQ * 16 / T2; i++) {
    int c = tid + i * T2, row = c >> 4, ch = c & 15, gq = q0 + row;
    cp_async16(smem_u32(sQ + swz(row, ch)), Qh + (size_t)(gq < Lq ? gq : 0) * tok + ch * 8, gq < Lq);
  }
  auto load_kv = [&](int buf, int kb) {
#pragma unroll
    for (int i = 0; i < BKV * 16 / T2; i++) {
      int c = tid + i * T2, row = c >> 4, ch = c & 15, gk = kb * BKV + row;
      bool p = gk < Lk;
      size_t off = (size_t)(p ? gk : 0) * tok + ch * 8;
      cp_async16(smem_u32(sK + buf * BKV * ROWB + swz(row, ch)), Kh + off, p);
      cp_async16(smem_u32(sV + buf * BKV * ROWB + swz(row, ch)), Vh + off, p);
    }
  };
  load_kv(0, 0);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  {
    const __half2 sc = __float2half2_rn(scale_log2);
    for (int i = tid; i < BQ * D / 2; i += T2) {
      __half2 *p = reinterpret_cast<__half2 *>(sQ) + i;
      *p = __hmul2(*p, sc);
    }
  }
  __syncthreads();

  const int mi = lane >> 3, r = lane & 7, g = lane >> 2, t4 = lane & 3;
  float o[2][16][4];
#pragma unroll
  for (int t = 0; t < 2; t++)
#pragma unroll
    for (int j = 0; j < 16; j++) o[t][j][0] = o[t][j][1] = o[t][j][2] = o[t][j][3] = 0.f;
  float m[2][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}}, l[2][2] = {{0.f, 0.f}, {0.f, 0.f}};

  const int nkb = (Lk + BKV - 1) / BKV;
  for (int kb = 0; kb < nkb; kb++) {
    const int buf = kb & 1;
    if (kb + 1 < nkb) load_kv(buf ^ 1, kb + 1);
    asm volatile("cp.async.commit_group;");
    const uint8_t *tK = sK + buf * BKV * ROWB, *tV = sV + buf * BKV * ROWB;
    uint32_t s[2][8][2];
#pragma unroll
    for (int t = 0; t < 2; t++)
#pragma unroll
      for (int j = 0; j < 8; j++) s[t][j][0] = s[t][j][1] = 0;
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      uint32_t qa[2][4];
#pragma unroll
      for (int t = 0; t < 2; t++)
        ldsm_x4(qa[t][0], qa[t][1], qa[t][2], qa[t][3], smem_u32(sQ + swz(warp * 32 + t * 16 + (mi & 1) * 8 + r, kk * 2 + (mi >> 1))));
#pragma unroll
      for (int j2 = 0; j2 < 4; j2++) {
        uint32_t b0, b1, b2, b3;
        ldsm_x4(b0, b1, b2, b3, smem_u32(tK + swz(j2 * 16 + (mi >> 1) * 8 + r, kk * 2 + (mi & 1))));
#pragma unroll
        for (int t = 0; t < 2; t++) { mma16(s[t][j2 * 2], qa[t], b0, b1); mma16(s[t][j2 * 2 + 1], qa[t], b2, b3); }
      }
    }
    float al[2][2];
    uint32_t pa[2][4][4];
#pragma unroll
    for (int t = 0; t < 2; t++) {
      if (kb * BKV + BKV > Lk) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const int key = kb * BKV + j * 8 + t4 * 2;
          const __half ninf = __ushort_as_half(0xFC00);
          __half2 *a = reinterpret_cast<__half2 *>(&s[t][j][0]), *b = reinterpret_cast<__half2 *>(&s[t][j][1]);
          if (key >= Lk) { a->x = ninf; b->x = ninf; }
          if (key + 1 >= Lk) { a->y = ninf; b->y = ninf; }
        }
      }
      __half2 hm0 = *reinterpret_cast<__half2 *>(&s[t][0][0]), hm1 = *reinterpret_cast<__half2 *>(&s[t][0][1]);
#pragma unroll
      for (int j = 1; j < 8; j++) {
        hm0 = __hmax2(hm0, *reinterpret_cast<__half2 *>(&s[t][j][0]));
        hm1 = __hmax2(hm1, *reinterpret_cast<__half2 *>(&s[t][j][1]));
      }
      float mx0 = fmaxf(m[t][0], fmaxf(__low2float(hm0), __high2float(hm0)));
      float mx1 = fmaxf(m[t][1], fmaxf(__low2float(hm1), __high2float(hm1)));
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
      al[t][0] = exp2f(m[t][0] - mx0); al[t][1] = exp2f(m[t][1] - mx1);
      m[t][0] = mx0; m[t][1] = mx1;
      const __half2 hx0 = __float2half2_rn(mx0), hx1 = __float2half2_rn(mx1);
      __half2 rs0 = __float2half2_rn(0.f), rs1 = rs0;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        __half2 p0 = h2exp2(__hsub2(*reinterpret_cast<__half2 *>(&s[t][j][0]), hx0));
        __half2 p1 = h2exp2(__hsub2(*reinterpret_cast<__half2 *>(&s[t][j][1]), hx1));
        rs0 = __hadd2(rs0, p0); rs1 = __hadd2(rs1, p1);
        pa[t][j >> 1][(j & 1) * 2 + 0] = *reinterpret_cast<uint32_t *>(&p0);
        pa[t][j >> 1][(j & 1) * 2 + 1] = *reinterpret_cast<uint32_t *>(&p1);
      }
      l[t][0] = l[t][0] * al[t][0] + __low2float(rs0) + __high2float(rs0);
      l[t][1] = l[t][1] * al[t][1] + __low2float(rs1) + __high2float(rs1);
    }
#pragma unroll
    for (int dh = 0; dh < 2; dh++) {
      uint32_t oh[2][8][2];
#pragma unroll
      for (int t = 0; t < 2; t++)
#pragma unroll
        for (int j = 0; j < 8; j++) oh[t][j][0] = oh[t][j][1] = 0;
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
#pragma unroll
        for (int n2 = 0; n2 < 4; n2++) {
          uint32_t b0, b1, b2, b3;
          ldsm_x4_t(b0, b1, b2, b3, smem_u32(tV + swz(kk * 16 + (mi & 1) * 8 + r, (dh * 4 + n2) * 2 + (mi >> 1))));
#pragma unroll
          for (int t = 0; t < 2; t++) { mma16(oh[t][n2 * 2], pa[t][kk], b0, b1); mma16(oh[t][n2 * 2 + 1], pa[t][kk], b2, b3); }
        }
      }
#pragma unroll
      for (int t = 0; t < 2; t++)
#pragma unroll
        for (int j = 0; j < 8; j++) {
          float2 a = __half22float2(*reinterpret_cast<__half2 *>(&oh[t][j][0])), b = __half22float2(*reinterpret_cast<__half2 *>(&oh[t][j][1]));
          float *oo = o[t][dh * 8 + j];
          oo[0] = oo[0] * al[t][0] + a.x; oo[1] = oo[1] * al[t][0] + a.y;
          oo[2] = oo[2] * al[t][1] + b.x; oo[3] = oo[3] * al[t][1] + b.y;
        }
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
  }
  half *Oh = O + head * D;
#pragma unroll
  for (int t = 0; t < 2; t++) {
    float l0 = l[t][0], l1 = l[t][1];
    l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
    const float i0 = 1.f / l0, i1 = 1.f / l1;
    const int row0 = q0 + warp * 32 + t * 16 + g, row1 = row0 + 8;
#pragma unroll
    for (int j = 0; j < 16; j++) {
      const int col = j * 8 + t4 * 2;
      if (row0 < Lq) *reinterpret_cast<__half2 *>(Oh + (size_t)row0 * tok + col) = __floats2half2_rn(o[t][j][0] * i0, o[t][j][1] * i0);
      if (row1 < Lq) *reinterpret_cast<__half2 *>(Oh + (size_t)row1 * tok + col) = __floats2half2_rn(o[t][j][2] * i1, o[t][j][3] * i1);
    }
  }
}
}

template <bool HSM>
torch::Tensor n2_run(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.scalar_type() == torch::kHalf && k.scalar_type() == torch::kHalf && v.scalar_type() == torch::kHalf);
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && q.size(-1) == D);
  const int Lq = q.size(-3), H = q.size(-2), Lk = k.size(-3);
  auto o = torch::empty_like(q);
  const int smem = BQ * ROWB + 4 * BKV * ROWB;
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(n2_attn<HSM>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
  dim3 grid((Lq + BQ - 1) / BQ, H);
  n2_attn<HSM><<<grid, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half *>(q.data_ptr()), reinterpret_cast<const half *>(k.data_ptr()), reinterpret_cast<const half *>(v.data_ptr()),
      reinterpret_cast<half *>(o.data_ptr()), Lq, Lk, H, float(1.4426950408889634 / std::sqrt((double)D)));
  return o;
}
torch::Tensor n2_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) { return n2_run<true>(q, k, v); }
torch::Tensor n2_attention_v0(torch::Tensor q, torch::Tensor k, torch::Tensor v) { return n2_run<false>(q, k, v); }
torch::Tensor n2_attention_v2(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && q.size(-1) == D);
  const int Lq = q.size(-3), H = q.size(-2), Lk = k.size(-3);
  auto o = torch::empty_like(q);
  const int smem = BQ * ROWB + 4 * BKV * ROWB;
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(n2_attn2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
  dim3 grid((Lq + BQ - 1) / BQ, H);
  n2_attn2<<<grid, T2, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half *>(q.data_ptr()), reinterpret_cast<const half *>(k.data_ptr()), reinterpret_cast<const half *>(v.data_ptr()),
      reinterpret_cast<half *>(o.data_ptr()), Lq, Lk, H, float(1.4426950408889634 / std::sqrt((double)D)));
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("n2_attention", &n2_attention, "FP16 ladder attention, packed-FP16 softmax (non-causal, D=128)");
  m.def("n2_attention_v0", &n2_attention_v0, "v0: FP32 softmax");
  m.def("n2_attention_v2", &n2_attention_v2, "v2: 32 query rows per warp, 4 warps");
}
