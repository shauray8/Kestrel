#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

__device__ __forceinline__ void rotate8(uint4 raw, const float *cos, const float *sin, uint4 &out) {
  const __nv_bfloat162 *in = reinterpret_cast<const __nv_bfloat162 *>(&raw);
  __nv_bfloat162 *dst = reinterpret_cast<__nv_bfloat162 *>(&out);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 x = __bfloat1622float2(in[j]);
    const float c = __ldg(cos + j), s = __ldg(sin + j);
    dst[j] = __floats2bfloat162_rn(__fsub_rn(__fmul_rn(x.x, c), __fmul_rn(x.y, s)),
                                   __fadd_rn(__fmul_rn(x.x, s), __fmul_rn(x.y, c)));
  }
}

__global__ void rope_qk_append_kernel(const uint4 *__restrict__ q, const uint4 *__restrict__ k, const uint4 *__restrict__ v,
                                      const float *__restrict__ cos, const float *__restrict__ sin, uint4 *__restrict__ q_out,
                                      uint4 *__restrict__ k_cache, uint4 *__restrict__ v_cache, int64_t vectors_per_batch,
                                      int64_t cache_token_stride, int heads, int dim, int64_t write_offset) {
  const int64_t within = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (within >= vectors_per_batch) return;
  const int64_t b = blockIdx.y;
  const int64_t i = b * vectors_per_batch + within;
  const int vec_per_token = (heads * dim) >> 3;
  const int64_t token = within / vec_per_token;
  const int lane = (int)(within - token * vec_per_token);
  const int pair = (lane % (dim >> 3)) * 4;
  const int64_t table = token * (dim >> 1) + pair;
  uint4 qv, kv;
  rotate8(q[i], cos + table, sin + table, qv);
  rotate8(k[i], cos + table, sin + table, kv);
  q_out[i] = qv;
  const int64_t dst = b * cache_token_stride + ((write_offset + token) * heads * dim >> 3) + lane;
  k_cache[dst] = kv;
  v_cache[dst] = v[i];
}

__device__ __forceinline__ uint4 rms_apply(uint4 raw, float rstd, uint4 w) {
  const __nv_bfloat162 *x = reinterpret_cast<const __nv_bfloat162 *>(&raw), *g = reinterpret_cast<const __nv_bfloat162 *>(&w);
  uint4 out;
  __nv_bfloat162 *o = reinterpret_cast<__nv_bfloat162 *>(&out);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 a = __bfloat1622float2(x[j]), b = __bfloat1622float2(g[j]);
    const __nv_bfloat162 n = __floats2bfloat162_rn(__fmul_rn(a.x, rstd), __fmul_rn(a.y, rstd));
    const float2 m = __bfloat1622float2(n);
    o[j] = __floats2bfloat162_rn(__fmul_rn(m.x, b.x), __fmul_rn(m.y, b.y));
  }
  return out;
}

__device__ __forceinline__ float sumsq8(uint4 raw) {
  const __nv_bfloat162 *x = reinterpret_cast<const __nv_bfloat162 *>(&raw);
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 a = __bfloat1622float2(x[j]);
    s = fmaf(a.x, a.x, s);
    s = fmaf(a.y, a.y, s);
  }
  return s;
}

template <int VPL>
__global__ void rms_rope_qk_append_kernel(const uint4 *__restrict__ q, const uint4 *__restrict__ k, const uint4 *__restrict__ v,
                                          const uint4 *__restrict__ wq, const uint4 *__restrict__ wk, float eps,
                                          const float *__restrict__ cos, const float *__restrict__ sin, uint4 *__restrict__ q_out,
                                          uint4 *__restrict__ k_cache, uint4 *__restrict__ v_cache, int64_t tokens,
                                          int64_t cache_token_stride, int dim, int64_t write_offset) {
  const int lane = threadIdx.x & 31;
  const int64_t token = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (token >= tokens) return;
  const int64_t b = blockIdx.y;
  constexpr int vec = VPL * 32;
  const int64_t row = (b * tokens + token) * vec;
  uint4 qv[VPL], kv[VPL];
  float sq = 0.f, sk = 0.f;
#pragma unroll
  for (int j = 0; j < VPL; ++j) {
    qv[j] = q[row + lane + j * 32];
    kv[j] = k[row + lane + j * 32];
    sq += sumsq8(qv[j]);
    sk += sumsq8(kv[j]);
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) {
    sq += __shfl_xor_sync(0xffffffffu, sq, d);
    sk += __shfl_xor_sync(0xffffffffu, sk, d);
  }
  const float c = float(vec * 8);
  const float rq = rsqrtf(__fadd_rn(__fdiv_rn(sq, c), eps)), rk = rsqrtf(__fadd_rn(__fdiv_rn(sk, c), eps));
  const int64_t dst = b * cache_token_stride + (write_offset + token) * vec;
#pragma unroll
  for (int j = 0; j < VPL; ++j) {
    const int lv = lane + j * 32, pair = (lv % (dim >> 3)) * 4;
    const int64_t table = token * (dim >> 1) + pair;
    uint4 qo, ko;
    rotate8(rms_apply(qv[j], rq, __ldg(wq + lv)), cos + table, sin + table, qo);
    rotate8(rms_apply(kv[j], rk, __ldg(wk + lv)), cos + table, sin + table, ko);
    q_out[row + lv] = qo;
    k_cache[dst + lv] = ko;
    v_cache[dst + lv] = v[row + lv];
  }
}
void check_nhd(const torch::Tensor &x, const char *name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.dim() == 4,
              name, " must be contiguous CUDA bf16 [B,L,H,D]");
  TORCH_CHECK(x.size(3) % 8 == 0, name, " head dimension must be divisible by 8");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

void check_rope_args(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v, const torch::Tensor &cos,
                     const torch::Tensor &sin, const torch::Tensor &k_cache, const torch::Tensor &v_cache, int64_t write_offset) {
  check_nhd(q, "q");
  check_nhd(k, "k");
  check_nhd(v, "v");
  check_nhd(k_cache, "k_cache");
  check_nhd(v_cache, "v_cache");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "q, k and v must have identical shapes");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache must have identical shapes");
  TORCH_CHECK(k_cache.size(0) == q.size(0) && k_cache.size(2) == q.size(2) && k_cache.size(3) == q.size(3),
              "cache batch/head dimensions must match q");
  TORCH_CHECK(write_offset >= 0 && write_offset + q.size(1) <= k_cache.size(1), "cache write range is out of bounds");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda() && cos.scalar_type() == torch::kFloat32 && sin.scalar_type() == torch::kFloat32,
              "cos and sin must be CUDA fp32");
  TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous() && cos.sizes() == sin.sizes() && cos.dim() == 3 &&
                  cos.size(0) == q.size(1) && cos.size(1) == 1 && cos.size(2) == q.size(3) / 2,
              "cos and sin must be contiguous [L,1,D/2]");
}

}

torch::Tensor rope_qk_append(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor cos, torch::Tensor sin,
                             torch::Tensor k_cache, torch::Tensor v_cache, int64_t write_offset) {
  check_rope_args(q, k, v, cos, sin, k_cache, v_cache, write_offset);
  auto out = torch::empty_like(q);
  const int64_t vec_per_batch = q.numel() / q.size(0) / 8;
  constexpr int threads = 256;
  const dim3 grid((unsigned)((vec_per_batch + threads - 1) / threads), (unsigned)q.size(0));
  rope_qk_append_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4 *>(q.data_ptr()), reinterpret_cast<const uint4 *>(k.data_ptr()),
      reinterpret_cast<const uint4 *>(v.data_ptr()), cos.data_ptr<float>(), sin.data_ptr<float>(),
      reinterpret_cast<uint4 *>(out.data_ptr()), reinterpret_cast<uint4 *>(k_cache.data_ptr()),
      reinterpret_cast<uint4 *>(v_cache.data_ptr()), vec_per_batch, k_cache.numel() / q.size(0) / 8, q.size(2), q.size(3),
      write_offset);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor rms_rope_qk_append(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor wq, torch::Tensor wk, double eps,
                                 torch::Tensor cos, torch::Tensor sin, torch::Tensor k_cache, torch::Tensor v_cache, int64_t write_offset) {
  check_rope_args(q, k, v, cos, sin, k_cache, v_cache, write_offset);
  const int64_t c = q.size(2) * q.size(3);
  for (const auto &w : {wq, wk})
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kBFloat16 && w.is_contiguous() && w.numel() == c &&
                    reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "norm weights must be contiguous CUDA bf16 [H*D]");
  auto out = torch::empty_like(q);
  const int64_t tokens = q.size(1);
  constexpr int threads = 256;
  const dim3 grid((unsigned)((tokens + threads / 32 - 1) / (threads / 32)), (unsigned)q.size(0));
  auto st = at::cuda::getCurrentCUDAStream();
  auto p = [](const torch::Tensor &t) { return reinterpret_cast<const uint4 *>(t.data_ptr()); };
  auto w = [](torch::Tensor &t) { return reinterpret_cast<uint4 *>(t.data_ptr()); };
  const int64_t stride = k_cache.numel() / q.size(0) / 8;
#define RRQ(N) case N: rms_rope_qk_append_kernel<N><<<grid, threads, 0, st>>>(p(q), p(k), p(v), p(wq), p(wk), (float)eps, \
    cos.data_ptr<float>(), sin.data_ptr<float>(), w(out), w(k_cache), w(v_cache), tokens, stride, q.size(3), write_offset); break;
  switch (c % 256 ? 0 : c / 256) {
    RRQ(4) RRQ(6) RRQ(8) RRQ(12) RRQ(16) RRQ(20)
    default: TORCH_CHECK(false, "unsupported hidden size ", c);
  }
#undef RRQ
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_qk_append", &rope_qk_append, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("cos"), py::arg("sin"),
        py::arg("k_cache"), py::arg("v_cache"), py::arg("write_offset"));
  m.def("rms_rope_qk_append", &rms_rope_qk_append, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("wq"), py::arg("wk"),
        py::arg("eps"), py::arg("cos"), py::arg("sin"), py::arg("k_cache"), py::arg("v_cache"), py::arg("write_offset"));
}
