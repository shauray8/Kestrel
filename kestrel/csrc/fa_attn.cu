#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <cstring>

namespace {
constexpr int D = 128, BK = 64, T = 256, TILE = BK * D;
__host__ __device__ constexpr int stages(int q_tile) { return q_tile == 128 ? 3 : 2; }

__device__ __forceinline__ uint32_t shared(const void *p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ int sw128(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }
__device__ __forceinline__ int sw64(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }
__device__ __forceinline__ void cp16(void *dst, const void *src, bool pred = true) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(shared(dst)), "l"(src), "r"(pred ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void ld4(uint32_t (&a)[4], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(shared(p)));
}
__device__ __forceinline__ void mma(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_i8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_mx(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sb) {
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%11}, {%12}, {%11,%11};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(127u), "h"(uint16_t(0)), "r"(sb));
}
__device__ __forceinline__ float ceil_pow2(float x) {
  const uint32_t u = __float_as_uint(x);
  return __uint_as_float(((u >> 23) + ((u & 0x7FFFFFu) != 0)) << 23);
}
__device__ __forceinline__ uint32_t quant2(float x, float y) {
  __nv_fp8x2_e4m3 a(make_float2(x, y));
  return a.__x;
}
__device__ __forceinline__ uint32_t quant2h(uint32_t x) {
  uint16_t y;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ __half2 h2(uint32_t x) { return *reinterpret_cast<__half2 *>(&x); }
__device__ __forceinline__ uint32_t u32(__half2 x) { return *reinterpret_cast<uint32_t *>(&x); }
__device__ __forceinline__ uint32_t ex2h(uint32_t x) {
  uint32_t y;
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float poly_exp2(float x) {
  x = fmaxf(x, -127.f);
  const float t = __fadd_rd(x, 12582912.f), f = x - (t - 12582912.f);
  const float p = fmaf(fmaf(fmaf(0.07706769555807114f, f, 0.22764427959918976f), f, 0.6951169371604919f), f, 1.f);
  return __int_as_float(__float_as_int(p) + (__float_as_int(t) << 23));
}
__device__ __forceinline__ void init_bar(uint64_t *p, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(shared(p)), "r"(count) : "memory");
}
__device__ __forceinline__ void arrive(uint64_t *p) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(shared(p)) : "memory");
}
__device__ __forceinline__ void wait_bar(uint64_t *p, int phase) {
  asm volatile("{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@P1 bra DONE;\nbra LAB_WAIT;\nDONE:\n}"
               :: "r"(shared(p)), "r"(phase) : "memory");
}
__device__ __forceinline__ void tma_load(const CUtensorMap *map, void *dst, uint64_t *bar, int y, int head) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {0, %3, %4}], [%2];"
               :: "r"(shared(dst)), "l"(map), "r"(shared(bar)), "r"(y), "r"(head) : "memory");
}
__device__ __forceinline__ void expect(uint64_t *p) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(shared(p)), "r"(2 * TILE) : "memory");
}

__device__ __forceinline__ float2 load2(const half *p) { return __half22float2(*reinterpret_cast<const __half2 *>(p)); }
__device__ __forceinline__ float2 load2(const __nv_bfloat16 *p) { return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(p)); }

template <typename In, bool Transpose>
__global__ void fa_quant_kernel(const In *__restrict__ x, uint8_t *__restrict__ out, float *__restrict__ scales,
                                int L, int H, int nb, int first, float target, bool pow2, bool int8, const float *__restrict__ mean) {
  __shared__ float maxima[8];
  __shared__ uint8_t tile[Transpose ? TILE : 1];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, block = first + blockIdx.x, head = blockIdx.y;
  float2 values[16];
  float mx = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    int pair = tid + j * T, row = pair / 64, col = (pair % 64) * 2, token = block * BK + row;
    values[j] = token < L ? load2(x + (size_t(token) * H + head) * D + col) : make_float2(0.f, 0.f);
    if (mean && token < L) { values[j].x -= mean[head * D + col]; values[j].y -= mean[head * D + col + 1]; }
    mx = fmaxf(mx, fmaxf(fabsf(values[j].x), fabsf(values[j].y)));
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
  if (!lane) maxima[warp] = mx;
  __syncthreads();
  mx = lane < 8 ? maxima[lane] : 0.f;
#pragma unroll
  for (int d = 16; d; d >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
  const float sc = pow2 ? ceil_pow2(fmaxf(mx / target, 1.e-12f)) : fmaxf(mx / target, 1.e-12f);
  if (!tid) scales[head * nb + block] = sc;
  uint8_t *dst = out + (size_t(head) * nb + block) * TILE;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int pair = tid + j * T;
    uint16_t q = int8 ? uint16_t(uint8_t(int8_t(max(-127, min(127, __float2int_rn(values[j].x / sc)))))
                                 | (uint16_t(uint8_t(int8_t(max(-127, min(127, __float2int_rn(values[j].y / sc)))))) << 8))
                      : uint16_t(quant2(values[j].x / sc, values[j].y / sc));
    if constexpr (Transpose) *reinterpret_cast<uint16_t *>(tile + pair * 2) = q;
    else *reinterpret_cast<uint16_t *>(dst + pair * 2) = q;
  }
  if constexpr (Transpose) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int c = tid + j * T, dim = c / 16, p = (c % 16) * 4, key = (p & 48) + ((p >> 2) & 3) * 2;
      uint32_t q = uint32_t(tile[key * D + dim]) | (uint32_t(tile[(key + 1) * D + dim]) << 8)
                 | (uint32_t(tile[(key + 8) * D + dim]) << 16) | (uint32_t(tile[(key + 9) * D + dim]) << 24);
      reinterpret_cast<uint32_t *>(dst)[c] = q;
    }
  }
}

template <int Exp, bool Mx, bool I8 = false>
__device__ __forceinline__ void fa_step(const uint32_t (&qa)[4][4], const uint8_t *tK, const uint8_t *tV, int kb, int Lk, float qs,
                                        const float *ksp, const float *vsp, float threshold, int lane, float (&o)[16][4], float &m0, float &m1,
                                        float &l0, float &l1) {
  const int mi = lane >> 3, r = lane & 7, t4 = lane & 3;
  float al0, al1;
  uint32_t p0[8], p1[8];
  if constexpr (I8) {
    int si[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) si[j][0] = si[j][1] = si[j][2] = si[j][3] = 0;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t b[4];
        ld4(b, tK + sw128(j * 16 + (mi >> 1) * 8 + r, kk * 2 + (mi & 1)));
        mma_i8(si[j * 2], qa[kk], b[0], b[1]);
        mma_i8(si[j * 2 + 1], qa[kk], b[2], b[3]);
      }
    }
    const float sk = qs * *ksp;
    float f[8][4];
    const bool tail = (kb + 1) * BK > Lk;
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int key = kb * BK + j * 8 + t4 * 2;
      const bool k0 = !tail || key < Lk, k1 = !tail || key + 1 < Lk;
      f[j][0] = k0 ? float(si[j][0]) : -INFINITY; f[j][1] = k1 ? float(si[j][1]) : -INFINITY;
      f[j][2] = k0 ? float(si[j][2]) : -INFINITY; f[j][3] = k1 ? float(si[j][3]) : -INFINITY;
      mx0 = fmaxf(mx0, fmaxf(f[j][0], f[j][1])); mx1 = fmaxf(mx1, fmaxf(f[j][2], f[j][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    mx0 *= sk; mx1 *= sk;
    mx0 = mx0 > m0 + threshold ? mx0 : m0;
    mx1 = mx1 > m1 + threshold ? mx1 : m1;
    al0 = ex2(m0 - mx0); al1 = ex2(m1 - mx1);
    m0 = mx0; m1 = mx1;
    const float c0 = mx0 - 8.f, c1 = mx1 - 8.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float e[4] = {fmaf(f[j][0], sk, -c0), fmaf(f[j][1], sk, -c0), fmaf(f[j][2], sk, -c1), fmaf(f[j][3], sk, -c1)};
#pragma unroll
      for (int i = 0; i < 4; ++i) e[i] = Exp == 1 && (j & 3) == 0 ? poly_exp2(e[i]) : ex2(e[i]);
      p0[j] = quant2(e[0], e[1]); p1[j] = quant2(e[2], e[3]);
    }
  } else {
  uint32_t s[8][2];
#pragma unroll
  for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = 0;
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t b[4];
      ld4(b, tK + sw128(j * 16 + (mi >> 1) * 8 + r, kk * 2 + (mi & 1)));
      mma(s[j * 2], qa[kk], b[0], b[1]);
      mma(s[j * 2 + 1], qa[kk], b[2], b[3]);
    }
  }
  const float sk = qs * *ksp;
  if ((kb + 1) * BK > Lk) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int key = kb * BK + j * 8 + t4 * 2;
      if (key >= Lk) s[j][0] = s[j][1] = 0xFC00FC00u;
      else if (key + 1 >= Lk) { s[j][0] = (s[j][0] & 0xFFFFu) | 0xFC000000u; s[j][1] = (s[j][1] & 0xFFFFu) | 0xFC000000u; }
    }
  }
  __half2 x0 = h2(s[0][0]), x1 = h2(s[0][1]);
#pragma unroll
  for (int j = 1; j < 8; ++j) { x0 = __hmax2(x0, h2(s[j][0])); x1 = __hmax2(x1, h2(s[j][1])); }
  float mx0 = fmaxf(__low2float(x0), __high2float(x0)), mx1 = fmaxf(__low2float(x1), __high2float(x1));
  mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
  mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
  mx0 *= sk; mx1 *= sk;
  mx0 = mx0 > m0 + threshold ? mx0 : m0;
  mx1 = mx1 > m1 + threshold ? mx1 : m1;
  al0 = ex2(m0 - mx0); al1 = ex2(m1 - mx1);
  m0 = mx0; m1 = mx1;
  if (Exp == 2 && sk >= 6.2e-5f) {
    const __half2 skh = __float2half2_rn(sk), c0 = __float2half2_rn(8.f - mx0), c1 = __float2half2_rn(8.f - mx1);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      p0[j] = quant2h(ex2h(u32(__hfma2(h2(s[j][0]), skh, c0))));
      p1[j] = quant2h(ex2h(u32(__hfma2(h2(s[j][1]), skh, c1))));
    }
  } else {
    const float c0 = mx0 - 8.f, c1 = mx1 - 8.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float2 a = __half22float2(h2(s[j][0])), b = __half22float2(h2(s[j][1]));
      float e[4] = {fmaf(a.x, sk, -c0), fmaf(a.y, sk, -c0), fmaf(b.x, sk, -c1), fmaf(b.y, sk, -c1)};
#pragma unroll
      for (int i = 0; i < 4; ++i) e[i] = Exp == 1 && (j & 3) == 0 ? poly_exp2(e[i]) : ex2(e[i]);
      p0[j] = quant2(e[0], e[1]); p1[j] = quant2(e[2], e[3]);
    }
  }
  }
  uint32_t pa[2][4], rs[2] = {0, 0};
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    pa[kk][0] = p0[kk * 4] | (p0[kk * 4 + 1] << 16);
    pa[kk][1] = p1[kk * 4] | (p1[kk * 4 + 1] << 16);
    pa[kk][2] = p0[kk * 4 + 2] | (p0[kk * 4 + 3] << 16);
    pa[kk][3] = p1[kk * 4 + 2] | (p1[kk * 4 + 3] << 16);
    mma(rs, pa[kk], 0x38383838u, 0x38383838u);
  }
  l0 = fmaf(l0, al0, __low2float(h2(rs[0]))); l1 = fmaf(l1, al1, __low2float(h2(rs[1])));
  const float sv = *vsp;
  if (al0 != 1.f || al1 != 1.f) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      o[j][0] *= al0; o[j][1] *= al0; o[j][2] *= al1; o[j][3] *= al1;
    }
  }
  if constexpr (Mx) {
    const uint32_t sb = __float_as_uint(sv) >> 23;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t b[4];
        ld4(b, tV + sw64(j * 16 + (mi >> 1) * 8 + r, kk * 2 + (mi & 1)));
        mma_mx(o[j * 2], pa[kk], b[0], b[1], sb);
        mma_mx(o[j * 2 + 1], pa[kk], b[2], b[3], sb);
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t h0[2] = {0, 0}, h1[2] = {0, 0};
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t b[4];
        ld4(b, tV + sw64(j * 16 + (mi >> 1) * 8 + r, kk * 2 + (mi & 1)));
        mma(h0, pa[kk], b[0], b[1]);
        mma(h1, pa[kk], b[2], b[3]);
      }
      const float2 a = __half22float2(h2(h0[0])), b = __half22float2(h2(h0[1]));
      const float2 c = __half22float2(h2(h1[0])), d = __half22float2(h2(h1[1]));
      o[j * 2][0] = fmaf(a.x, sv, o[j * 2][0]); o[j * 2][1] = fmaf(a.y, sv, o[j * 2][1]);
      o[j * 2][2] = fmaf(b.x, sv, o[j * 2][2]); o[j * 2][3] = fmaf(b.y, sv, o[j * 2][3]);
      o[j * 2 + 1][0] = fmaf(c.x, sv, o[j * 2 + 1][0]); o[j * 2 + 1][1] = fmaf(c.y, sv, o[j * 2 + 1][1]);
      o[j * 2 + 1][2] = fmaf(d.x, sv, o[j * 2 + 1][2]); o[j * 2 + 1][3] = fmaf(d.y, sv, o[j * 2 + 1][3]);
    }
  }
}

__device__ __forceinline__ void fa_store(const float (&o)[16][4], float l0, float l1, int row0, int lane, int Lq, int H, int head,
                                         half *__restrict__ O) {
  const float inv0 = 1.f / l0, inv1 = 1.f / l1;
  const int row1 = row0 + 8, t4 = lane & 3;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    int col = j * 8 + t4 * 2;
    if (row0 < Lq) *reinterpret_cast<__half2 *>(O + (size_t(row0) * H + head) * D + col) = __floats2half2_rn(o[j][0] * inv0, o[j][1] * inv0);
    if (row1 < Lq) *reinterpret_cast<__half2 *>(O + (size_t(row1) * H + head) * D + col) = __floats2half2_rn(o[j][2] * inv1, o[j][3] * inv1);
  }
}

template <int Load, int Exp, int QTile, bool Mx = false, bool I8 = false>
__global__ void __launch_bounds__(QTile * 2 + (Load == 3 ? 32 : 0), QTile == 64 ? 2 : 1)
fa_kernel(const uint8_t *__restrict__ Q, const uint8_t *__restrict__ K, const uint8_t *__restrict__ V,
          const float *__restrict__ QS, const float *__restrict__ KS, const float *__restrict__ VS,
          half *__restrict__ O, int Lq, int Lk, int H, float scale_log2, float threshold,
          const __grid_constant__ CUtensorMap kmap, const __grid_constant__ CUtensorMap vmap) {
  constexpr int W = QTile / 16, NT = W * 32, S = stages(QTile);
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t *sQ = smem, *sK = sQ + QTile * D, *sV = sK + S * TILE;
  uint64_t *ready = reinterpret_cast<uint64_t *>(sV + S * TILE), *empty = ready + S;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, head = blockIdx.y, q0 = blockIdx.x * QTile;
  const int nq = (Lq + BK - 1) / BK, nk = (Lk + BK - 1) / BK;
  if (tid < NT) {
#pragma unroll
    for (int j = 0; j < QTile * D / 16 / NT; ++j) {
      int c = tid + j * NT, row = c / 8, ch = c % 8, qr = q0 + row;
      cp16(sQ + sw128(row, ch), Q + (size_t(head) * nq * BK + (qr < Lq ? qr : 0)) * D + ch * 16, qr < Lq);
    }
  }
  if constexpr (Load != 0) {
    if (!tid) {
      for (int i = 0; i < S; ++i) { init_bar(ready + i, 1); init_bar(empty + i, W); }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
  }
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  auto issue_tma = [&](int kb) {
    const int buf = kb % S;
    expect(ready + buf);
    tma_load(&kmap, sK + buf * TILE, ready + buf, kb * BK, head);
    tma_load(&vmap, sV + buf * TILE, ready + buf, kb * D, head);
  };
  auto issue_cp = [&](int kb) {
    const int buf = kb % S;
    const uint8_t *k = K + (size_t(head) * nk + kb) * TILE, *v = V + (size_t(head) * nk + kb) * TILE;
#pragma unroll
    for (int j = 0; j < TILE / 16 / NT; ++j) {
      int c = tid + j * NT;
      cp16(sK + buf * TILE + sw128(c / 8, c % 8), k + c * 16);
      cp16(sV + buf * TILE + sw64(c / 4, c % 4), v + c * 16);
    }
  };
  if constexpr (Load == 3) {
    if (warp == W) {
      if (!lane) {
        for (int kb = 0; kb < nk; ++kb) {
          if (kb >= S) wait_bar(empty + kb % S, (kb / S - 1) & 1);
          issue_tma(kb);
        }
      }
      return;
    }
  } else if constexpr (Load == 1) {
    if (!tid)
      for (int i = 0; i < S - 1 && i < nk; ++i) issue_tma(i);
  } else {
#pragma unroll
    for (int i = 0; i < S - 1; ++i) {
      if (i < nk) issue_cp(i);
      cp_commit();
    }
  }
  const int mi = lane >> 3, r = lane & 7, g = lane >> 2;
  uint32_t qa[4][4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk)
    ld4(qa[kk], sQ + sw128(warp * 16 + (mi & 1) * 8 + r, kk * 2 + (mi >> 1)));
  float o[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const float qs = (q0 + warp * 16 < Lq ? QS[head * nq + (q0 + warp * 16) / BK] : 1.f) * scale_log2;
  for (int kb = 0; kb < nk; ++kb) {
    const int buf = kb % S;
    if constexpr (Load == 0) {
      cp_wait<S - 2>();
      __syncthreads();
      if (kb + S - 1 < nk) issue_cp(kb + S - 1);
      cp_commit();
    } else {
      wait_bar(ready + buf, (kb / S) & 1);
      if constexpr (Load == 1) {
        __syncthreads();
        if (!tid && kb + S - 1 < nk) issue_tma(kb + S - 1);
      }
    }
    fa_step<Exp, Mx, I8>(qa, sK + buf * TILE, sV + buf * TILE, kb, Lk, qs, KS + head * nk + kb, VS + head * nk + kb, threshold, lane,
                     o, m0, m1, l0, l1);
    if constexpr (Load == 3) {
      __syncwarp();
      if (!lane) arrive(empty + buf);
    }
  }
  fa_store(o, l0, l1, q0 + warp * 16 + g, lane, Lq, H, head, O);
}

__device__ __forceinline__ void bar_compute(int n) { asm volatile("bar.sync 1, %0;" :: "r"(n) : "memory"); }
__device__ __forceinline__ int ld_acquire(const int *p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release(int *p, int v) { asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }

template <int Exp, bool Mx, bool I8 = false>
__global__ void __launch_bounds__(288, 1)
fa_bal_kernel(const uint8_t *__restrict__ Q, const float *__restrict__ QS, const float *__restrict__ KS, const float *__restrict__ VS,
              half *__restrict__ O, int Lq, int Lk, int H, float scale_log2, float threshold,
              const __grid_constant__ CUtensorMap kmap, const __grid_constant__ CUtensorMap vmap, float *__restrict__ ws,
              int *__restrict__ flags) {
  constexpr int QTile = 128, W = 8, NT = 256, S = stages(QTile), SLOT = W * 68 * 32;
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t *sQ = smem, *sK = sQ + QTile * D, *sV = sK + S * TILE;
  uint64_t *ready = reinterpret_cast<uint64_t *>(sV + S * TILE), *empty = ready + S;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c = blockIdx.x, G = gridDim.x;
  const int nq = (Lq + BK - 1) / BK, nk = (Lk + BK - 1) / BK, nqt = (Lq + QTile - 1) / QTile;
  const long long work = (long long)nqt * H * nk, w0 = work * c / G, w1 = work * (c + 1) / G;
  if (!tid) {
    for (int i = 0; i < S; ++i) { init_bar(ready + i, 1); init_bar(empty + i, W); }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  if (warp == W) {
    if (!lane) {
      for (long long w = w0; w < w1; ++w) {
        const int it = int(w - w0), buf = it % S, head = int(w / nk) / nqt, kb = int(w % nk);
        if (it >= S) wait_bar(empty + buf, (it / S - 1) & 1);
        expect(ready + buf);
        tma_load(&kmap, sK + buf * TILE, ready + buf, kb * BK, head);
        tma_load(&vmap, sV + buf * TILE, ready + buf, kb * D, head);
      }
    }
    return;
  }
  const int mi = lane >> 3, r = lane & 7, g = lane >> 2;
  for (long long w = w0; w < w1;) {
    const int unit = int(w / nk), kb0 = int(w % nk), head = unit / nqt, q0 = (unit % nqt) * QTile;
    const long long we = w1 < (long long)(unit + 1) * nk ? w1 : (long long)(unit + 1) * nk;
    const int kb1 = kb0 + int(we - w);
    bar_compute(NT);
#pragma unroll
    for (int j = 0; j < QTile * D / 16 / NT; ++j) {
      int ci = tid + j * NT, row = ci / 8, ch = ci % 8, qr = q0 + row;
      cp16(sQ + sw128(row, ch), Q + (size_t(head) * nq * BK + (qr < Lq ? qr : 0)) * D + ch * 16, qr < Lq);
    }
    cp_commit();
    cp_wait<0>();
    bar_compute(NT);
    uint32_t qa[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
      ld4(qa[kk], sQ + sw128(warp * 16 + (mi & 1) * 8 + r, kk * 2 + (mi >> 1)));
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const float qs = (q0 + warp * 16 < Lq ? QS[head * nq + (q0 + warp * 16) / BK] : 1.f) * scale_log2;
    for (int kb = kb0; kb < kb1; ++kb) {
      const int it = int(w - w0) + kb - kb0, buf = it % S;
      wait_bar(ready + buf, (it / S) & 1);
      fa_step<Exp, Mx, I8>(qa, sK + buf * TILE, sV + buf * TILE, kb, Lk, qs, KS + head * nk + kb, VS + head * nk + kb, threshold, lane,
                       o, m0, m1, l0, l1);
      __syncwarp();
      if (!lane) arrive(empty + buf);
    }
    const int row0 = q0 + warp * 16 + g;
    if (kb0 > 0) {
      float *p = ws + size_t(c) * SLOT + warp * 68 * 32 + lane;
#pragma unroll
      for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) p[(j * 4 + i) * 32] = o[j][i];
      p[64 * 32] = m0; p[65 * 32] = m1; p[66 * 32] = l0; p[67 * 32] = l1;
      __threadfence();
      bar_compute(NT);
      if (!tid) st_release(flags + c, 1);
    } else if (kb1 < nk) {
      if (!tid) while (ld_acquire(flags + c + 1) != 1) __nanosleep(64);
      bar_compute(NT);
      const float *p = ws + size_t(c + 1) * SLOT + warp * 68 * 32 + lane;
      const float pm0 = __ldcg(p + 64 * 32), pm1 = __ldcg(p + 65 * 32), pl0 = __ldcg(p + 66 * 32), pl1 = __ldcg(p + 67 * 32);
      const float n0 = fmaxf(m0, pm0), n1 = fmaxf(m1, pm1);
      const float a0 = ex2(m0 - n0), b0 = ex2(pm0 - n0), a1 = ex2(m1 - n1), b1 = ex2(pm1 - n1);
      l0 = l0 * a0 + pl0 * b0; l1 = l1 * a1 + pl1 * b1;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        o[j][0] = o[j][0] * a0 + __ldcg(p + (j * 4) * 32) * b0;
        o[j][1] = o[j][1] * a0 + __ldcg(p + (j * 4 + 1) * 32) * b0;
        o[j][2] = o[j][2] * a1 + __ldcg(p + (j * 4 + 2) * 32) * b1;
        o[j][3] = o[j][3] * a1 + __ldcg(p + (j * 4 + 3) * 32) * b1;
      }
      bar_compute(NT);
      if (!tid) flags[c + 1] = 0;
      fa_store(o, l0, l1, row0, lane, Lq, H, head, O);
    } else {
      fa_store(o, l0, l1, row0, lane, Lq, H, head, O);
    }
    w = we;
  }
}

void check_packed(const torch::Tensor &x, const char *name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kUInt8 && x.is_contiguous() && x.dim() == 4, name, " must be CUDA contiguous uint8 rank4");
}

CUtensorMap make_map(const torch::Tensor &x, bool transposed) {
  static auto encode = [] {
    void *p = nullptr;
    cudaDriverEntryPointQueryResult status;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &status));
    TORCH_CHECK(p && status == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(p);
  }();
  alignas(64) CUtensorMap map{};
  const uint64_t width = transposed ? BK : D, rows = transposed ? D : BK;
  uint64_t dims[3] = {width, uint64_t(x.size(1)) * rows, uint64_t(x.size(0))};
  uint64_t strides[2] = {width, uint64_t(x.size(1)) * TILE};
  uint32_t box[3] = {uint32_t(width), uint32_t(rows), 1}, element[3] = {1, 1, 1};
  CUresult result = encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, x.data_ptr(), dims, strides, box, element,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, transposed ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", int(result));
  return map;
}

template <int Load, int Exp, int QTile, bool Mx = false, bool I8 = false>
void launch(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
            const torch::Tensor &qs, const torch::Tensor &ks, const torch::Tensor &vs,
            torch::Tensor &out, int Lq, int Lk, float scale, float threshold, CUtensorMap km, CUtensorMap vm) {
  constexpr int NT = QTile * 2 + (Load == 3 ? 32 : 0), SMEM = QTile * D + 2 * stages(QTile) * TILE + 64;
  static const bool configured = [] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(fa_kernel<Load, Exp, QTile, Mx, I8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    return true;
  }();
  (void)configured;
  fa_kernel<Load, Exp, QTile, Mx, I8><<<dim3((Lq + QTile - 1) / QTile, q.size(0)), NT, SMEM, at::cuda::getCurrentCUDAStream()>>>(
    q.data_ptr<uint8_t>(), k.data_ptr<uint8_t>(), v.data_ptr<uint8_t>(), qs.data_ptr<float>(), ks.data_ptr<float>(), vs.data_ptr<float>(),
    reinterpret_cast<half *>(out.data_ptr()), Lq, Lk, q.size(0), scale * 1.4426950408889634f, threshold, km, vm);
}

template <int Exp, bool Mx, bool I8 = false>
bool launch_bal(const torch::Tensor &q, const torch::Tensor &qs, const torch::Tensor &ks, const torch::Tensor &vs, torch::Tensor &out,
                int Lq, int Lk, float scale, float threshold, CUtensorMap km, CUtensorMap vm) {
  constexpr int SMEM = 128 * D + 2 * stages(128) * TILE + 64, SLOT = 8 * 68 * 32;
  struct State { int per_sm = -1; torch::Tensor ws, flags; };
  static std::unordered_map<int, State> states;
  State &st = states[q.get_device()];
  if (st.per_sm < 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(fa_bal_kernel<Exp, Mx, I8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&st.per_sm, fa_bal_kernel<Exp, Mx, I8>, 288, SMEM));
  }
  const int H = q.size(0), G = at::cuda::getCurrentDeviceProperties()->multiProcessorCount * st.per_sm;
  if (st.per_sm < 1 || (Lq + 127) / 128 * H < G) return false;
  if (!st.flags.defined()) {
    st.ws = torch::empty({int64_t(G) * SLOT}, q.options().dtype(torch::kFloat32));
    st.flags = torch::zeros({G}, q.options().dtype(torch::kInt32));
    at::cuda::getCurrentCUDAStream().synchronize();
  }
  torch::Tensor &ws = st.ws, &flags = st.flags;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(G);
  cfg.blockDim = dim3(288);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeCooperative;
  attr[0].val.cooperative = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fa_bal_kernel<Exp, Mx, I8>, static_cast<const uint8_t *>(q.data_ptr<uint8_t>()),
                                    static_cast<const float *>(qs.data_ptr<float>()), static_cast<const float *>(ks.data_ptr<float>()),
                                    static_cast<const float *>(vs.data_ptr<float>()), reinterpret_cast<half *>(out.data_ptr()), Lq, Lk,
                                    H, scale * 1.4426950408889634f, threshold, km, vm, ws.data_ptr<float>(), flags.data_ptr<int>()));
  return true;
}
}

void fa_quant(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, int start, int stop, bool transpose, double target, bool pow2,
              bool int8, c10::optional<torch::Tensor> mean) {
  const bool bf = x.scalar_type() == torch::kBFloat16;
  TORCH_CHECK(x.is_cuda() && (bf || x.scalar_type() == torch::kHalf) && x.dim() == 3 && x.size(2) == D && x.is_contiguous(), "x must be contiguous CUDA fp16/bf16 [L,H,128]");
  const c10::cuda::CUDAGuard guard(x.device());
  check_packed(packed, "packed");
  int L = x.size(0), H = x.size(1), nb = (L + BK - 1) / BK;
  TORCH_CHECK(packed.device() == x.device() && packed.size(0) == H && packed.size(1) == nb
              && packed.size(2) == (transpose ? D : BK) && packed.size(3) == (transpose ? BK : D), "invalid packed shape/device");
  TORCH_CHECK(scales.device() == x.device() && scales.scalar_type() == torch::kFloat32 && scales.is_contiguous()
              && scales.dim() == 2 && scales.size(0) == H && scales.size(1) == nb, "invalid scale shape/device/dtype");
  TORCH_CHECK(0 <= start && start <= stop && stop <= nb && target > 0 && std::isfinite(target), "invalid quant block range/target");
  if (start == stop || H == 0) return;
  dim3 grid(stop - start, H);
  auto st = at::cuda::getCurrentCUDAStream();
  uint8_t *po = packed.data_ptr<uint8_t>();
  float *ps = scales.data_ptr<float>();
  TORCH_CHECK(!int8 || !transpose, "int8 quantization is for Q/K only");
  TORCH_CHECK(!mean || (mean->device() == x.device() && mean->scalar_type() == torch::kFloat32 && mean->is_contiguous() && mean->numel() == H * D),
              "mean must be contiguous CUDA fp32 [H, 128]");
  const float *mp = mean ? mean->data_ptr<float>() : nullptr;
  if (bf) {
    auto px = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
    if (transpose) fa_quant_kernel<__nv_bfloat16, true><<<grid, T, 0, st>>>(px, po, ps, L, H, nb, start, target, pow2, int8, mp);
    else fa_quant_kernel<__nv_bfloat16, false><<<grid, T, 0, st>>>(px, po, ps, L, H, nb, start, target, pow2, int8, mp);
  } else {
    auto px = reinterpret_cast<const half *>(x.data_ptr());
    if (transpose) fa_quant_kernel<half, true><<<grid, T, 0, st>>>(px, po, ps, L, H, nb, start, target, pow2, int8, mp);
    else fa_quant_kernel<half, false><<<grid, T, 0, st>>>(px, po, ps, L, H, nb, start, target, pow2, int8, mp);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor fa_plan(torch::Tensor k, torch::Tensor v) {
  check_packed(k, "k"); check_packed(v, "v");
  const c10::cuda::CUDAGuard guard(k.device());
  TORCH_CHECK(k.device() == v.device() && k.size(0) == v.size(0) && k.size(1) == v.size(1)
              && k.size(2) == BK && k.size(3) == D && v.size(2) == D && v.size(3) == BK, "invalid K/V packed layout");
  auto plan = torch::empty({2, int64_t(sizeof(CUtensorMap))}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  CUtensorMap km = make_map(k, false), vm = make_map(v, true);
  std::memcpy(plan.data_ptr<uint8_t>(), &km, sizeof(km));
  std::memcpy(plan.data_ptr<uint8_t>() + sizeof(km), &vm, sizeof(vm));
  return plan;
}

torch::Tensor fa_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor qs, torch::Tensor ks, torch::Tensor vs,
                            int Lq, int Lk, double sm_scale, int load_mode, int exp_mode, double rescale_threshold,
                            c10::optional<torch::Tensor> plan, int q_tile, bool mx, bool bal, bool i8) {
  check_packed(q, "q"); check_packed(k, "k"); check_packed(v, "v");
  const c10::cuda::CUDAGuard guard(q.device());
  int H = q.size(0), nq = (Lq + BK - 1) / BK, nk = (Lk + BK - 1) / BK;
  TORCH_CHECK(Lq > 0 && Lk > 0 && H > 0 && q.size(1) == nq && k.size(1) == nk && v.size(1) == nk, "invalid lengths");
  TORCH_CHECK(q.size(2) == BK && q.size(3) == D && k.size(0) == H && k.size(2) == BK && k.size(3) == D
              && v.size(0) == H && v.size(2) == D && v.size(3) == BK && k.device() == q.device() && v.device() == q.device(), "invalid packed layout/device");
  for (const auto &s : {qs, ks, vs}) TORCH_CHECK(s.device() == q.device() && s.scalar_type() == torch::kFloat32 && s.is_contiguous() && s.dim() == 2 && s.size(0) == H, "invalid scales");
  TORCH_CHECK(qs.size(1) == nq && ks.size(1) == nk && vs.size(1) == nk, "invalid scales lengths");
  TORCH_CHECK(load_mode == 0 || load_mode == 1 || load_mode == 3, "sm_120a does not support TMA multicast; load_mode must be 0, 1 or 3");
  TORCH_CHECK(exp_mode >= 0 && exp_mode <= 2 && std::isfinite(sm_scale) && sm_scale > 0
              && rescale_threshold >= 0 && rescale_threshold <= 0.5, "invalid exp mode, attention scale, or rescale threshold (0..0.5 log2 units)");
  TORCH_CHECK(q_tile == 64 || q_tile == 128, "q_tile must be 64 or 128");
  TORCH_CHECK(!mx || load_mode == 3, "mx requires load_mode 3");
  CUtensorMap km{}, vm{};
  if (load_mode != 0) {
    torch::Tensor maps = plan.has_value() ? *plan : fa_plan(k, v);
    TORCH_CHECK(maps.device().is_cpu() && maps.scalar_type() == torch::kUInt8 && maps.is_contiguous() && maps.numel() == 2 * sizeof(CUtensorMap), "invalid TMA plan");
    std::memcpy(&km, maps.data_ptr(), sizeof(km));
    std::memcpy(&vm, maps.data_ptr<uint8_t>() + sizeof(km), sizeof(vm));
  }
  auto out = torch::empty({Lq, H, D}, q.options().dtype(torch::kHalf));
  TORCH_CHECK(!bal || (load_mode == 3 && q_tile == 128), "bal requires load_mode 3 and q_tile 128");
  TORCH_CHECK(!i8 || (mx && load_mode == 3), "i8 requires mx and load_mode 3");
  if (bal) {
    const float s = sm_scale, t = rescale_threshold;
    const bool done = i8 ? (exp_mode == 0 ? launch_bal<0, true, true>(q, qs, ks, vs, out, Lq, Lk, s, t, km, vm)
                           : exp_mode == 1 ? launch_bal<1, true, true>(q, qs, ks, vs, out, Lq, Lk, s, t, km, vm)
                                           : launch_bal<2, true, true>(q, qs, ks, vs, out, Lq, Lk, s, t, km, vm))
                    : mx ? (exp_mode == 0 ? launch_bal<0, true>(q, qs, ks, vs, out, Lq, Lk, s, t, km, vm)
                           : exp_mode == 1 ? launch_bal<1, true>(q, qs, ks, vs, out, Lq, Lk, s, t, km, vm)
                                           : launch_bal<2, true>(q, qs, ks, vs, out, Lq, Lk, s, t, km, vm))
                         : (exp_mode == 0 ? launch_bal<0, false>(q, qs, ks, vs, out, Lq, Lk, s, t, km, vm)
                           : exp_mode == 1 ? launch_bal<1, false>(q, qs, ks, vs, out, Lq, Lk, s, t, km, vm)
                                           : launch_bal<2, false>(q, qs, ks, vs, out, Lq, Lk, s, t, km, vm));
    if (done) {
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return out;
    }
  }
#define FA_LAUNCH(L, E, M, I) if (q_tile == 64) { launch<L, E, 64, M, I>(q, k, v, qs, ks, vs, out, Lq, Lk, sm_scale, rescale_threshold, km, vm); } else { launch<L, E, 128, M, I>(q, k, v, qs, ks, vs, out, Lq, Lk, sm_scale, rescale_threshold, km, vm); }
#define FA_EXP(L, M, I) if (exp_mode == 0) { FA_LAUNCH(L, 0, M, I); } else if (exp_mode == 1) { FA_LAUNCH(L, 1, M, I); } else { FA_LAUNCH(L, 2, M, I); }
  if (i8) { FA_EXP(3, true, true); } else if (mx) { FA_EXP(3, true, false); } else if (load_mode == 0) { FA_EXP(0, false, false); } else if (load_mode == 1) { FA_EXP(1, false, false); } else { FA_EXP(3, false, false); }
#undef FA_EXP
#undef FA_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa_quant", &fa_quant, pybind11::arg("x"), pybind11::arg("packed"), pybind11::arg("scales"), pybind11::arg("start"),
        pybind11::arg("stop"), pybind11::arg("transpose"), pybind11::arg("target"), pybind11::arg("pow2") = false,
        pybind11::arg("int8") = false, pybind11::arg("mean") = pybind11::none());
  m.def("fa_plan", &fa_plan);
  m.def("fa_attention", &fa_attention, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("qs"), pybind11::arg("ks"), pybind11::arg("vs"), pybind11::arg("Lq"), pybind11::arg("Lk"),
        pybind11::arg("sm_scale"), pybind11::arg("load_mode"), pybind11::arg("exp_mode"), pybind11::arg("rescale_threshold"),
        pybind11::arg("plan") = pybind11::none(), pybind11::arg("q_tile") = 128, pybind11::arg("mx") = false, pybind11::arg("bal") = false,
        pybind11::arg("i8") = false);
}
