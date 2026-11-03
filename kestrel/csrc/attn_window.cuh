#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace kestrel_attn {
constexpr int BQ = 128, BKV = 64, D = 128, ROWB = 256, THREADS = 256;
__device__ __forceinline__ uint32_t addr(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ int swz(int row, int ch) { return row * ROWB + ((ch ^ (row & 7)) << 4); }
__device__ __forceinline__ void cp16(void *dst, const void *src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(addr(dst)), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void ld4(uint32_t (&x)[4], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]) : "r"(addr(p)));
}
__device__ __forceinline__ void ldt4(uint32_t (&x)[4], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]) : "r"(addr(p)));
}
__device__ __forceinline__ void mma(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 to_half(uint4 raw) {
  uint4 out;
  const __nv_bfloat162 *b = reinterpret_cast<const __nv_bfloat162 *>(&raw);
  __half2 *h = reinterpret_cast<__half2 *>(&out);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 x = __bfloat1622float2(b[i]);
    h[i] = __floats2half2_rn(x.x, x.y);
  }
  return out;
}
template <bool BF>
__device__ __forceinline__ void convert(uint8_t *p, int rows) {
  if constexpr (BF) {
    for (int c = threadIdx.x; c < rows * 16; c += THREADS) {
      uint4 *x = reinterpret_cast<uint4 *>(p + swz(c >> 4, c & 15));
      *x = to_half(*x);
    }
  }
}
template <bool BF, bool DB>
__device__ __forceinline__ void run(const void *Q, const void *K, const void *V, void *O, int Lq, int Lk, int H,
                                    int q0, int head, float scale_log2, uint8_t *smem) {
  uint8_t *sQ = smem, *sK = DB ? smem + BQ * ROWB : smem, *sV = sK + (DB ? 2 : 1) * BKV * ROWB;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const size_t tok = (size_t)H * D;
  const half *Qh = (const half *)Q + head * D, *Kh = (const half *)K + head * D, *Vh = (const half *)V + head * D;
#pragma unroll
  for (int i = 0; i < BQ * 16 / THREADS; ++i) {
    const int c = tid + i * THREADS, row = c >> 4, ch = c & 15, gq = q0 + row;
    cp16(sQ + swz(row, ch), Qh + (size_t)(gq < Lq ? gq : 0) * tok + ch * 8, gq < Lq);
  }
  auto load_kv = [&](int buf, int kb) {
#pragma unroll
    for (int i = 0; i < BKV * 16 / THREADS; ++i) {
      const int c = tid + i * THREADS, row = c >> 4, ch = c & 15, gk = kb * BKV + row;
      const bool p = gk < Lk;
      const size_t off = (size_t)(p ? gk : 0) * tok + ch * 8;
      cp16(sK + buf * BKV * ROWB + swz(row, ch), Kh + off, p);
      cp16(sV + buf * BKV * ROWB + swz(row, ch), Vh + off, p);
    }
  };
  if constexpr (DB) load_kv(0, 0);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  convert<BF>(sQ, BQ);
  if constexpr (DB) {
    convert<BF>(sK, BKV);
    convert<BF>(sV, BKV);
  }
  if constexpr (BF) __syncthreads();
  const int mi = lane >> 3, r = lane & 7, g = lane >> 2, t4 = lane & 3;
  uint32_t qa[8][4];
#pragma unroll
  for (int kk = 0; kk < 8; ++kk)
    ld4(qa[kk], sQ + swz(warp * 16 + (mi & 1) * 8 + r, kk * 2 + (mi >> 1)));
  const __half2 sc = __float2half2_rn(scale_log2);
#pragma unroll
  for (int kk = 0; kk < 8; ++kk)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __half2 x = __hmul2(*reinterpret_cast<__half2 *>(&qa[kk][i]), sc);
      qa[kk][i] = *reinterpret_cast<const uint32_t *>(&x);
    }
  if constexpr (!DB) {
    __syncthreads();
    load_kv(0, 0);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    convert<BF>(sK, BKV);
    convert<BF>(sV, BKV);
    if constexpr (BF) __syncthreads();
  }
  float o[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int nkb = (Lk + BKV - 1) / BKV;
  for (int kb = 0; kb < nkb; ++kb) {
    const int buf = DB ? kb & 1 : 0;
    uint4 nextK[4], nextV[4];
    if constexpr (DB) {
      if (kb + 1 < nkb) load_kv(buf ^ 1, kb + 1);
      asm volatile("cp.async.commit_group;");
    } else {
      if (kb + 1 < nkb) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int c = tid + i * THREADS, gk = (kb + 1) * BKV + (c >> 4), ch = c & 15;
          const bool p = gk < Lk;
          const size_t off = (size_t)(p ? gk : 0) * tok + ch * 8;
          nextK[i] = p ? *reinterpret_cast<const uint4 *>(Kh + off) : make_uint4(0, 0, 0, 0);
          nextV[i] = p ? *reinterpret_cast<const uint4 *>(Vh + off) : make_uint4(0, 0, 0, 0);
          if constexpr (BF) { nextK[i] = to_half(nextK[i]); nextV[i] = to_half(nextV[i]); }
        }
      }
    }
    const uint8_t *tK = sK + buf * BKV * ROWB, *tV = sV + buf * BKV * ROWB;
    uint32_t s[8][2];
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = 0;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
#pragma unroll
      for (int j2 = 0; j2 < 4; ++j2) {
        uint32_t b[4];
        ld4(b, tK + swz(j2 * 16 + (mi >> 1) * 8 + r, kk * 2 + (mi & 1)));
        mma(s[j2 * 2], qa[kk], b[0], b[1]);
        mma(s[j2 * 2 + 1], qa[kk], b[2], b[3]);
      }
    if (kb * BKV + BKV > Lk) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int key = kb * BKV + j * 8 + t4 * 2;
        const __half ninf = __ushort_as_half(0xFC00);
        __half2 *a = reinterpret_cast<__half2 *>(&s[j][0]), *b = reinterpret_cast<__half2 *>(&s[j][1]);
        if (key >= Lk) { a->x = ninf; b->x = ninf; }
        if (key + 1 >= Lk) { a->y = ninf; b->y = ninf; }
      }
    }
    __half2 hm0 = *reinterpret_cast<__half2 *>(&s[0][0]), hm1 = *reinterpret_cast<__half2 *>(&s[0][1]);
#pragma unroll
    for (int j = 1; j < 8; ++j) {
      hm0 = __hmax2(hm0, *reinterpret_cast<__half2 *>(&s[j][0]));
      hm1 = __hmax2(hm1, *reinterpret_cast<__half2 *>(&s[j][1]));
    }
    float mx0 = fmaxf(m0, fmaxf(__low2float(hm0), __high2float(hm0)));
    float mx1 = fmaxf(m1, fmaxf(__low2float(hm1), __high2float(hm1)));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float al0 = exp2f(m0 - mx0), al1 = exp2f(m1 - mx1);
    m0 = mx0; m1 = mx1;
    const __half2 hx0 = __float2half2_rn(mx0), hx1 = __float2half2_rn(mx1);
    __half2 rs0 = __float2half2_rn(0.f), rs1 = rs0;
    uint32_t pa[4][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const __half2 p0 = h2exp2(__hsub2(*reinterpret_cast<__half2 *>(&s[j][0]), hx0));
      const __half2 p1 = h2exp2(__hsub2(*reinterpret_cast<__half2 *>(&s[j][1]), hx1));
      rs0 = __hadd2(rs0, p0); rs1 = __hadd2(rs1, p1);
      pa[j >> 1][(j & 1) * 2] = *reinterpret_cast<const uint32_t *>(&p0);
      pa[j >> 1][(j & 1) * 2 + 1] = *reinterpret_cast<const uint32_t *>(&p1);
    }
    l0 = l0 * al0 + __low2float(rs0) + __high2float(rs0);
    l1 = l1 * al1 + __low2float(rs1) + __high2float(rs1);
    uint32_t oh[16][2];
#pragma unroll
    for (int j = 0; j < 16; ++j) oh[j][0] = oh[j][1] = 0;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
#pragma unroll
      for (int n2 = 0; n2 < 8; ++n2) {
        uint32_t b[4];
        ldt4(b, tV + swz(kk * 16 + (mi & 1) * 8 + r, n2 * 2 + (mi >> 1)));
        mma(oh[n2 * 2], pa[kk], b[0], b[1]);
        mma(oh[n2 * 2 + 1], pa[kk], b[2], b[3]);
      }
    if (al0 == 1.f && al1 == 1.f) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const float2 a = __half22float2(*reinterpret_cast<__half2 *>(&oh[j][0])), b = __half22float2(*reinterpret_cast<__half2 *>(&oh[j][1]));
        o[j][0] += a.x; o[j][1] += a.y; o[j][2] += b.x; o[j][3] += b.y;
      }
    } else {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const float2 a = __half22float2(*reinterpret_cast<__half2 *>(&oh[j][0])), b = __half22float2(*reinterpret_cast<__half2 *>(&oh[j][1]));
        o[j][0] = o[j][0] * al0 + a.x; o[j][1] = o[j][1] * al0 + a.y;
        o[j][2] = o[j][2] * al1 + b.x; o[j][3] = o[j][3] * al1 + b.y;
      }
    }
    if constexpr (DB) {
      asm volatile("cp.async.wait_group 0;");
      __syncthreads();
      if (kb + 1 < nkb) {
        convert<BF>(sK + (buf ^ 1) * BKV * ROWB, BKV);
        convert<BF>(sV + (buf ^ 1) * BKV * ROWB, BKV);
      }
      if constexpr (BF) __syncthreads();
    } else {
      __syncthreads();
      if (kb + 1 < nkb) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int c = tid + i * THREADS;
          *reinterpret_cast<uint4 *>(sK + swz(c >> 4, c & 15)) = nextK[i];
          *reinterpret_cast<uint4 *>(sV + swz(c >> 4, c & 15)) = nextV[i];
        }
      }
      __syncthreads();
    }
  }
  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float i0 = 1.f / l0, i1 = 1.f / l1;
  const int row0 = q0 + warp * 16 + g, row1 = row0 + 8;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int col = j * 8 + t4 * 2;
    const __half2 a = __floats2half2_rn(o[j][0] * i0, o[j][1] * i0), b = __floats2half2_rn(o[j][2] * i1, o[j][3] * i1);
    if constexpr (BF) {
      __nv_bfloat16 *Oh = (__nv_bfloat16 *)O + head * D;
      const float2 x = __half22float2(a), y = __half22float2(b);
      if (row0 < Lq) *reinterpret_cast<__nv_bfloat162 *>(Oh + (size_t)row0 * tok + col) = __floats2bfloat162_rn(x.x, x.y);
      if (row1 < Lq) *reinterpret_cast<__nv_bfloat162 *>(Oh + (size_t)row1 * tok + col) = __floats2bfloat162_rn(y.x, y.y);
    } else {
      half *Oh = (half *)O + head * D;
      if (row0 < Lq) *reinterpret_cast<__half2 *>(Oh + (size_t)row0 * tok + col) = a;
      if (row1 < Lq) *reinterpret_cast<__half2 *>(Oh + (size_t)row1 * tok + col) = b;
    }
  }
}
}
