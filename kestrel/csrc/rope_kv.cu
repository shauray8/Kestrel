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
void check_nhd(const torch::Tensor &x, const char *name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.dim() == 4,
              name, " must be contiguous CUDA bf16 [B,L,H,D]");
  TORCH_CHECK(x.size(3) % 8 == 0, name, " head dimension must be divisible by 8");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

}

torch::Tensor rope_qk_append(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor cos, torch::Tensor sin,
                             torch::Tensor k_cache, torch::Tensor v_cache, int64_t write_offset) {
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_qk_append", &rope_qk_append, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("cos"), py::arg("sin"),
        py::arg("k_cache"), py::arg("v_cache"), py::arg("write_offset"));
}
