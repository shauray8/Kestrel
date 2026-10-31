#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstring>

namespace {
constexpr int BM = 128, BN = 128, BK = 64, STAGES = 3, THREADS = 256, WN = 4, TM = 64, TN = 32, MI = 4, NJ = 4;
constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK, SMEM = STAGES * (A_BYTES + B_BYTES), RB = THREADS / 32, JQ = 32;
enum : int { OP_HALT, OP_GEMM_F8, OP_GEMM_F8_GELU, OP_MOD_NORM, OP_RESIDUAL_MOD_NORM, OP_GEMM_F8_GATED_RES, OP_JOIN, OP_ROPE_KV_APPEND, OP_KV_ADVANCE, NOP };
enum : int { E_PLAIN, E_GELU, E_RES };
enum : int { DT_F8, DT_BF16, DT_F32 };
enum : int { ERR_TIMEOUT = 1, ERR_OPCODE = 2, ERR_OVERFLOW = 3 };
enum : int { C_HEAD, C_TAIL, C_DONE, C_ERR, C_INFO, C_N };

struct Instr { int op, m, n, t[6]; float f; int n_pred, dep_begin, n_dep, pad[3]; };
struct Tens { void *p; const float *vs; int64_t stride; int rows, cols, dtype; float scale; };
struct Mod { const float *p; int64_t stride; int rows_per; };
constexpr int MAXT = 16;
struct Table { Tens t[MAXT]; };
static_assert(sizeof(Instr) == 64 && sizeof(Tens) == 40, "layout");

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
__device__ __forceinline__ uint16_t q2(float a, float b, float inv) {
  __nv_fp8x2_e4m3 q(make_float2(a * inv, b * inv));
  return q.__x;
}
__device__ __forceinline__ void rotate8(uint4 raw, const float *cs, uint4 &out) {
  const __nv_bfloat162 *in = reinterpret_cast<const __nv_bfloat162 *>(&raw);
  __nv_bfloat162 *dst = reinterpret_cast<__nv_bfloat162 *>(&out);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 x = __bfloat1622float2(in[j]);
    const float c = __ldg(cs + 2 * j), s = __ldg(cs + 2 * j + 1);
    dst[j] = __floats2bfloat162_rn(__fsub_rn(__fmul_rn(x.x, c), __fmul_rn(x.y, s)),
                                   __fadd_rn(__fmul_rn(x.x, s), __fmul_rn(x.y, c)));
  }
}
__device__ __forceinline__ int ld_acq(const int *p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_rel(int *p, int v) { asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ int dec_acq_rel(int *p) {
  int v;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], -1;" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void tile_mma(const uint8_t *A, int64_t lda, const uint8_t *W, int64_t ldw, int M, int K, int m0, int n0,
                                         uint8_t *smem, float (&f)[MI][NJ][4]) {
  uint8_t *sA = smem, *sB = smem + STAGES * A_BYTES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
  const int KT = K / BK, lr = tid >> 2, lc = tid & 3;
  const bool p0 = m0 + lr < M, p1 = m0 + lr + 64 < M;
  const uint8_t *a0 = A + (p0 ? m0 + lr : 0) * lda + lc * 16, *a1 = A + (p1 ? m0 + lr + 64 : 0) * lda + lc * 16;
  const uint8_t *w0 = W + (n0 + lr) * ldw + lc * 16, *w1 = w0 + 64 * ldw;
  const uint32_t so = smem_u32(smem) + swz(lr, lc);
  auto load = [&](int stage, int kt) {
    const int k = kt * BK;
    cp16(so + stage * A_BYTES, a0 + k, p0);
    cp16(so + stage * A_BYTES + 64 * BK, a1 + k, p1);
    cp16(so + STAGES * A_BYTES + stage * B_BYTES, w0 + k, true);
    cp16(so + STAGES * A_BYTES + stage * B_BYTES + 64 * BK, w1 + k, true);
  };
  uint32_t h[MI][NJ][2];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++) { h[i][j][0] = h[i][j][1] = 0; f[i][j][0] = f[i][j][1] = f[i][j][2] = f[i][j][3] = 0.f; }
  __syncthreads();
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) { if (s < KT) load(s, s); asm volatile("cp.async.commit_group;"); }
  auto promote = [&]() {
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
      for (int j = 0; j < NJ; j++) {
        float2 lo = __half22float2(*reinterpret_cast<__half2 *>(&h[i][j][0])), hi = __half22float2(*reinterpret_cast<__half2 *>(&h[i][j][1]));
        f[i][j][0] += lo.x; f[i][j][1] += lo.y; f[i][j][2] += hi.x; f[i][j][3] += hi.y;
        h[i][j][0] = h[i][j][1] = 0;
      }
  };
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
      if (++step == 16) { step = 0; promote(); }
    }
  }
  asm volatile("cp.async.wait_group 0;");
  promote();
}

template <int EPI>
__device__ __forceinline__ void op_gemm(const Instr &I, const Tens *T, uint8_t *smem) {
  float f[MI][NJ][4];
  tile_mma((const uint8_t *)T[0].p, T[0].stride, (const uint8_t *)T[1].p, T[1].stride, T[0].rows, T[0].cols, I.m * BM, I.n * BN, smem, f);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wm = warp / WN, wn = warp % WN, g = lane >> 2, t4 = lane & 3;
  const int m0 = I.m * BM, n0 = I.n * BN, M = T[0].rows;
  const Tens &O = T[3];
  const bool hb = I.t[2] >= 0;
  const float sa = T[0].scale * T[1].scale, inv = 1.f / O.scale, *vs = T[1].vs;
  auto bias = [&](int c) {
    return !hb ? 0.f : T[2].dtype == DT_BF16 ? __bfloat162float(((const __nv_bfloat16 *)T[2].p)[c]) : ((const float *)T[2].p)[c];
  };
#pragma unroll
  for (int j = 0; j < NJ; j++) {
    const int col = n0 + wn * TN + j * 8 + t4 * 2;
    const float s0 = vs ? sa * vs[col] : sa, s1 = vs ? sa * vs[col + 1] : sa, b0 = bias(col), b1 = bias(col + 1);
    if constexpr (EPI == E_RES) {
      const Tens &G = T[4];
      const int rp = M / G.rows;
      float2 xv[MI][2], gv[MI][2];
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int r = m0 + wm * TM + i * 16 + g + h * 8;
          if (r < M) {
            xv[i][h] = __ldcg(reinterpret_cast<const float2 *>((const float *)O.p + r * O.stride + col));
            gv[i][h] = __ldg(reinterpret_cast<const float2 *>((const float *)G.p + (r / rp) * G.stride + col));
          }
        }
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int r = m0 + wm * TM + i * 16 + g + h * 8;
          if (r < M) {
            const float2 y = __bfloat1622float2(__floats2bfloat162_rn(f[i][j][2 * h] * s0 + b0, f[i][j][2 * h + 1] * s1 + b1));
            *reinterpret_cast<float2 *>((float *)O.p + r * O.stride + col) =
                make_float2(__fadd_rn(xv[i][h].x, __fmul_rn(y.x, gv[i][h].x)), __fadd_rn(xv[i][h].y, __fmul_rn(y.y, gv[i][h].y)));
          }
        }
    } else {
#pragma unroll
      for (int i = 0; i < MI; i++) {
        const int row = m0 + wm * TM + i * 16 + g;
        float v0 = f[i][j][0] * s0 + b0, v1 = f[i][j][1] * s1 + b1, v2 = f[i][j][2] * s0 + b0, v3 = f[i][j][3] * s1 + b1;
        if (EPI == E_GELU || O.dtype == DT_F8) {
          if (EPI == E_GELU) { v0 = gelu(v0); v1 = gelu(v1); v2 = gelu(v2); v3 = gelu(v3); }
          uint8_t *Y = (uint8_t *)O.p + col;
          if (row < M) *reinterpret_cast<uint16_t *>(Y + row * O.stride) = q2(v0, v1, inv);
          if (row + 8 < M) *reinterpret_cast<uint16_t *>(Y + (row + 8) * O.stride) = q2(v2, v3, inv);
        } else {
          __nv_bfloat16 *Y = (__nv_bfloat16 *)O.p + col;
          if (row < M) *reinterpret_cast<__nv_bfloat162 *>(Y + row * O.stride) = __floats2bfloat162_rn(v0, v1);
          if (row + 8 < M) *reinterpret_cast<__nv_bfloat162 *>(Y + (row + 8) * O.stride) = __floats2bfloat162_rn(v2, v3);
        }
      }
    }
  }
}

__device__ __forceinline__ void ldcg8(const float *p, float (&v)[8]) {
  const float4 a = __ldcg(reinterpret_cast<const float4 *>(p)), b = __ldcg(reinterpret_cast<const float4 *>(p + 4));
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
__device__ __forceinline__ Mod mod_of(const Instr &I, const Tens *T, int k, int M) {
  if (I.t[k] < 0) return {nullptr, 0, 1};
  return {(const float *)T[k].p, T[k].stride, M / T[k].rows};
}

template <int NIT, bool RES>
__device__ __forceinline__ void op_rows(const Instr &I, const Tens *T) {
  constexpr int C = NIT * 256;
  const Tens X = T[0];
  const int M = X.rows, lane = threadIdx.x & 31;
  const Mod gate = mod_of(I, T, 2, M), shift = mod_of(I, T, 3, M), scale = mod_of(I, T, 4, M);
  const int m = I.m * RB + (threadIdx.x >> 5);
  if (m >= M) return;
  float *xr = (float *)X.p + m * X.stride + lane * 8;
  float v[NIT][8];
#pragma unroll
  for (int j = 0; j < NIT; ++j) ldcg8(xr + j * 256, v[j]);
  if constexpr (RES) {
    const Tens Y = T[1];
    const __nv_bfloat16 *yr = (const __nv_bfloat16 *)Y.p + m * Y.stride + lane * 8;
    const float *gr = gate.p ? gate.p + (m / gate.rows_per) * gate.stride + lane * 8 : nullptr;
#pragma unroll
    for (int j = 0; j < NIT; ++j) {
      const uint4 raw = __ldcg(reinterpret_cast<const uint4 *>(yr + j * 256));
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
  const Tens O = T[5];
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
  const float rstd = rsqrtf(warp_sum(ss) * (1.f / C) + I.f), inv = 1.f / O.scale;
  const float *shr = shift.p + (m / shift.rows_per) * shift.stride + lane * 8;
  const float *scr = scale.p + (m / scale.rows_per) * scale.stride + lane * 8;
#pragma unroll
  for (int j = 0; j < NIT; ++j) {
    float sh[8], sc[8], o[8];
    ldg8(shr + j * 256, sh);
    ldg8(scr + j * 256, sc);
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __fadd_rn(__fmul_rn(__fmul_rn(v[j][k] - mean, rstd), __fadd_rn(1.f, sc[k])), sh[k]);
    const int64_t off = m * O.stride + j * 256 + lane * 8;
    if (O.dtype == DT_BF16) {
      uint4 pk;
      __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&pk);
#pragma unroll
      for (int k = 0; k < 4; ++k) h[k] = __floats2bfloat162_rn(o[2 * k], o[2 * k + 1]);
      *reinterpret_cast<uint4 *>((__nv_bfloat16 *)O.p + off) = pk;
    } else if (O.dtype == DT_F32) {
      store8((float *)O.p + off, o);
    } else {
      uint2 pk;
      uint16_t *q = reinterpret_cast<uint16_t *>(&pk);
#pragma unroll
      for (int k = 0; k < 4; ++k)
        q[k] = __nv_cvt_float2_to_fp8x2(make_float2(__fmul_rn(o[2 * k], inv), __fmul_rn(o[2 * k + 1], inv)), __NV_SATFINITE, __NV_E4M3);
      *reinterpret_cast<uint2 *>((uint8_t *)O.p + off) = pk;
    }
  }
}

template <bool RES>
__device__ __forceinline__ bool op_rows_c(const Instr &I, const Tens *T) {
  switch (T[0].cols) {
    case 1024: op_rows<4, RES>(I, T); return true;
    case 1536: op_rows<6, RES>(I, T); return true;
    case 2048: op_rows<8, RES>(I, T); return true;
    default: return false;
  }
}

__device__ __forceinline__ bool op_kv_advance(const Instr &I, const Tens *T) {
  const Tens K = T[0], V = T[1];
  const int rows = (int)I.f;
  if (rows <= 0 || K.dtype != DT_BF16 || V.dtype != DT_BF16 || K.rows != V.rows || K.cols != V.cols || K.cols % 8 ||
      K.stride % 8 || V.stride % 8 || I.m < 0 || I.n <= I.m || I.n + rows > K.rows) return false;
  const int vecs = K.cols / 8;
  for (int i = threadIdx.x; i < rows * vecs; i += THREADS) {
    const int row = i / vecs, lane = i - row * vecs;
    reinterpret_cast<uint4 *>((__nv_bfloat16 *)K.p + (int64_t)(I.m + row) * K.stride)[lane] =
        reinterpret_cast<const uint4 *>((const __nv_bfloat16 *)K.p + (int64_t)(I.n + row) * K.stride)[lane];
    reinterpret_cast<uint4 *>((__nv_bfloat16 *)V.p + (int64_t)(I.m + row) * V.stride)[lane] =
        reinterpret_cast<const uint4 *>((const __nv_bfloat16 *)V.p + (int64_t)(I.n + row) * V.stride)[lane];
  }
  return true;
}

__device__ __forceinline__ bool op_rope_kv_append(const Instr &I, const Tens *T) {
  const Tens Q = T[0], K = T[1], V = T[2], CS = T[3], KC = T[4], VC = T[5];
  const int base = I.m * BM, rows = min(BM, Q.rows - base), vecs = Q.cols / 8, dvec = CS.cols / 8;
  if (rows <= 0 || Q.dtype != DT_BF16 || K.dtype != DT_BF16 || V.dtype != DT_BF16 || CS.dtype != DT_F32 ||
      KC.dtype != DT_BF16 || VC.dtype != DT_BF16 || Q.rows != K.rows || Q.rows != V.rows || Q.cols != K.cols ||
      Q.cols != V.cols || KC.cols != Q.cols || VC.cols != Q.cols || CS.rows != Q.rows || CS.cols <= 0 || CS.cols % 8 ||
      Q.cols % CS.cols || Q.stride % 8 || K.stride % 8 || V.stride % 8 || KC.stride % 8 || VC.stride % 8 ||
      I.n < 0 || I.n + Q.rows > KC.rows || VC.rows != KC.rows) return false;
  for (int i = threadIdx.x; i < rows * vecs; i += THREADS) {
    const int token = i / vecs, lane = i - token * vecs, g = base + token, pair = (lane % dvec) * 4;
    uint4 qv, kv;
    rotate8(reinterpret_cast<const uint4 *>((const __nv_bfloat16 *)Q.p + (int64_t)g * Q.stride)[lane],
            (const float *)CS.p + (int64_t)g * CS.stride + pair * 2, qv);
    rotate8(reinterpret_cast<const uint4 *>((const __nv_bfloat16 *)K.p + (int64_t)g * K.stride)[lane],
            (const float *)CS.p + (int64_t)g * CS.stride + pair * 2, kv);
    reinterpret_cast<uint4 *>((__nv_bfloat16 *)Q.p + (int64_t)g * Q.stride)[lane] = qv;
    reinterpret_cast<uint4 *>((__nv_bfloat16 *)KC.p + (int64_t)(I.n + g) * KC.stride)[lane] = kv;
    reinterpret_cast<uint4 *>((__nv_bfloat16 *)VC.p + (int64_t)(I.n + g) * VC.stride)[lane] =
        reinterpret_cast<const uint4 *>((const __nv_bfloat16 *)V.p + (int64_t)g * V.stride)[lane];
  }
  return true;
}

__device__ __forceinline__ bool exec(const Instr &I, const Tens *T, uint8_t *smem) {
  switch (I.op) {
    case OP_HALT: return true;
    case OP_GEMM_F8: op_gemm<E_PLAIN>(I, T, smem); return true;
    case OP_GEMM_F8_GELU: op_gemm<E_GELU>(I, T, smem); return true;
    case OP_MOD_NORM: return op_rows_c<false>(I, T);
    case OP_RESIDUAL_MOD_NORM: return op_rows_c<true>(I, T);
    case OP_GEMM_F8_GATED_RES: op_gemm<E_RES>(I, T, smem); return true;
    case OP_ROPE_KV_APPEND: return op_rope_kv_append(I, T);
    case OP_KV_ADVANCE: return op_kv_advance(I, T);
    default: return false;
  }
}

__device__ __forceinline__ void stage(const Instr *ins, const Tens *tt, int idx, Instr &sI, Tens (&sT)[6]) {
  const int tid = threadIdx.x;
  if (tid < 16) reinterpret_cast<int *>(&sI)[tid] = reinterpret_cast<const int *>(ins + idx)[tid];
  else if (tid < 46) {
    const int k = (tid - 16) / 5, w = (tid - 16) % 5, t = ins[idx].t[k];
    if (t >= 0) reinterpret_cast<int64_t *>(sT + k)[w] = reinterpret_cast<const int64_t *>(tt + t)[w];
  }
  __syncthreads();
}

__device__ __forceinline__ void fail(int *ctl, int code, int info) {
  if (atomicCAS(ctl + C_ERR, 0, code) == 0) atomicExch(ctl + C_INFO, info);
}

__global__ void k_reset(const Instr *__restrict__ ins, const int *__restrict__ roots, int n, int nroots, int *ctl, int *pending, int *queue) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { pending[i] = ins[i].n_pred; queue[i] = i < nroots ? roots[i] : -1; }
  if (i < C_N) ctl[i] = i == C_TAIL ? nroots : 0;
}

__global__ void __launch_bounds__(THREADS, 2) k_vm(const Instr *__restrict__ ins, const __grid_constant__ Table tab, const int *__restrict__ deps, int nq,
                                                   int *ctl, int *pending, int *queue, unsigned long long *prof, uint64_t timeout) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_idx, s_cnt, s_base, s_nj, s_join[JQ];
  __shared__ Instr sI;
  __shared__ Tens sT[6];
  __shared__ unsigned long long s_prof[4 * NOP], s_t[2];
  const int tid = threadIdx.x;
  if (prof && tid < 4 * NOP) s_prof[tid] = 0;
  for (;;) {
    if (tid == 0) {
      const uint64_t ta = prof ? gtime() : 0;
      const int t = atomicAdd(ctl + C_HEAD, 1);
      int v = -1;
      if (t < nq) {
        int last = ld_acq(ctl + C_DONE);
        uint64_t t0 = gtime();
        while ((v = ld_acq(queue + t)) < 0) {
          if (ld_acq(ctl + C_ERR)) break;
          const int d = ld_acq(ctl + C_DONE);
          const uint64_t now = gtime();
          if (d != last) { last = d; t0 = now; }
          else if (now - t0 > timeout) { fail(ctl, ERR_TIMEOUT, t); break; }
          __nanosleep(64);
        }
        __threadfence();
      }
      s_idx = v;
      s_nj = 0;
      if (prof) { s_t[0] = ta; s_t[1] = gtime(); }
    }
    __syncthreads();
    const int idx = s_idx;
    if (idx < 0) break;
    stage(ins, tab.t, idx, sI, sT);
    if (!exec(sI, sT, smem)) { if (tid == 0) fail(ctl, ERR_OPCODE, idx); break; }
    __syncthreads();
    const uint64_t tc = prof && tid == 0 ? gtime() : 0;
    int b = sI.dep_begin, c = sI.n_dep, jq = 0;
    for (;;) {
      for (int o = 0; o < c; o += THREADS) {
        if (tid == 0) s_cnt = 0;
        __syncthreads();
        int d = -1, slot = -1;
        if (o + tid < c) {
          d = deps[b + o + tid];
          __threadfence();
          if (dec_acq_rel(pending + d) == 1) {
            if (ins[d].op == OP_JOIN) {
              const int j = atomicAdd(&s_nj, 1);
              if (j < JQ) s_join[j] = d;
              else fail(ctl, ERR_OVERFLOW, d);
            } else slot = atomicAdd(&s_cnt, 1);
          }
        }
        __syncthreads();
        if (tid == 0 && s_cnt) s_base = atomicAdd(ctl + C_TAIL, s_cnt);
        __syncthreads();
        if (slot >= 0) {
          const int s = s_base + slot;
          if (s < nq) st_rel(queue + s, d);
          else fail(ctl, ERR_OVERFLOW, d);
        }
      }
      if (jq >= min(s_nj, JQ)) break;
      const int j = s_join[jq++];
      b = ins[j].dep_begin;
      c = ins[j].n_dep;
    }
    if (tid == 0) {
      atomicAdd(ctl + C_DONE, 1 + min(s_nj, JQ));
      if (prof) {
        const int op = sI.op;
        s_prof[op] += s_t[1] - s_t[0];
        s_prof[NOP + op] += tc - s_t[1];
        s_prof[2 * NOP + op] += gtime() - tc;
        s_prof[3 * NOP + op]++;
      }
    }
  }
  __syncthreads();
  if (prof && tid < 4 * NOP) atomicAdd(prof + tid, s_prof[tid]);
}

__global__ void __launch_bounds__(THREADS, 2) k_tiles(const Instr *__restrict__ ins, const __grid_constant__ Table tab, int begin) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ Instr sI;
  __shared__ Tens sT[6];
  stage(ins, tab.t, begin + blockIdx.x, sI, sT);
  exec(sI, sT, smem);
}

void init_attrs() {
  static bool init = false;
  if (init) return;
  cudaFuncSetAttribute(k_vm, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  cudaFuncSetAttribute(k_tiles, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  init = true;
}

void check_ins(const torch::Tensor &ins) {
  TORCH_CHECK(ins.is_cuda() && ins.scalar_type() == torch::kInt32 && ins.is_contiguous() && ins.dim() == 2 && ins.size(1) == 16, "ins: [n,16] int32");
}

Table table(const std::vector<int64_t> &tt) {
  TORCH_CHECK(tt.size() % 5 == 0 && tt.size() <= 5 * MAXT, "tensors: flat [t*5] int64, t <= 16");
  Table t{};
  std::memcpy(t.t, tt.data(), tt.size() * sizeof(int64_t));
  return t;
}
}

int64_t vm_grid() {
  init_attrs();
  int per = 0, sms = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_vm, THREADS, SMEM);
  return (int64_t)per * sms;
}

void vm_run(torch::Tensor ins, const std::vector<int64_t> &tt, torch::Tensor deps, torch::Tensor roots, torch::Tensor state, torch::Tensor prof,
            int64_t nq, int64_t grid, int64_t timeout_ns) {
  check_ins(ins);
  const int n = ins.size(0), nroots = roots.numel();
  TORCH_CHECK(nq > 0 && nq <= n && nroots <= nq, "nq: instructions that take a queue slot (non-join), 0 < nq <= n");
  TORCH_CHECK(n > 0 && grid > 0 && state.scalar_type() == torch::kInt32 && state.is_contiguous() && state.numel() == C_N + 2 * n, "state: [5+2n] int32");
  TORCH_CHECK(deps.scalar_type() == torch::kInt32 && roots.scalar_type() == torch::kInt32 && nroots <= n, "deps/roots int32");
  TORCH_CHECK(prof.scalar_type() == torch::kInt64 && prof.is_contiguous() && (prof.numel() == 0 || prof.numel() == 4 * NOP), "prof: [] or [4,7] int64");
  init_attrs();
  auto st = at::cuda::getCurrentCUDAStream();
  const Instr *ip = reinterpret_cast<const Instr *>(ins.data_ptr());
  int *ctl = state.data_ptr<int>(), *pending = ctl + C_N, *queue = pending + n;
  k_reset<<<(n + 255) / 256, 256, 0, st>>>(ip, roots.data_ptr<int>(), n, nroots, ctl, pending, queue);
  k_vm<<<(int)grid, THREADS, SMEM, st>>>(ip, table(tt), deps.data_ptr<int>(), (int)nq, ctl, pending, queue,
                                        prof.numel() ? reinterpret_cast<unsigned long long *>(prof.data_ptr()) : nullptr, (uint64_t)timeout_ns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void vm_tiles(torch::Tensor ins, const std::vector<int64_t> &tt, int64_t begin, int64_t count) {
  check_ins(ins);
  TORCH_CHECK(begin >= 0 && count > 0 && begin + count <= ins.size(0), "range");
  init_attrs();
  k_tiles<<<(int)count, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const Instr *>(ins.data_ptr()),
                                                                         table(tt), (int)begin);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grid", &vm_grid, "resident persistent CTAs for k_vm");
  m.def("run", &vm_run, "reset + persistent dependency-counted VM over a tile program");
  m.def("tiles", &vm_tiles, "one launch: instructions [begin, begin+count), one CTA each, no dependency tracking");
}
