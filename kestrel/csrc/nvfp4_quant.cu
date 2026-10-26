#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>

namespace {
__device__ __forceinline__ float gelu_tanh(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }

__global__ void quant_nvfp4(const __nv_bfloat16 *__restrict__ x, uint8_t *__restrict__ q, uint8_t *__restrict__ sc,
                            int M, int K, int ncb, float inv_g, float alpha, const __nv_bfloat16 *__restrict__ bias, int gelu) {
  const int kb = K / 16;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)M * kb) return;
  const int r = idx / kb, c = idx % kb;
  const uint4 *src = reinterpret_cast<const uint4 *>(x + (size_t)r * K + c * 16);
  uint4 v0 = src[0], v1 = src[1];
  float f[16];
  const __nv_bfloat162 *h0 = reinterpret_cast<const __nv_bfloat162 *>(&v0), *h1 = reinterpret_cast<const __nv_bfloat162 *>(&v1);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 a = __bfloat1622float2(h0[i]), b = __bfloat1622float2(h1[i]);
    f[2 * i] = a.x; f[2 * i + 1] = a.y; f[8 + 2 * i] = b.x; f[8 + 2 * i + 1] = b.y;
  }
  if (bias != nullptr || alpha != 1.f || gelu) {
#pragma unroll
    for (int i = 0; i < 16; i++) {
      float t = f[i] * alpha + (bias ? __bfloat162float(bias[c * 16 + i]) : 0.f);
      f[i] = gelu ? gelu_tanh(t) : t;
    }
  }
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(f[i]));
  __nv_fp8_e4m3 s8(amax * inv_g / 6.0f);
  float s = float(s8);
  const float inv = s > 0.f ? inv_g / s : 0.f;
  uint32_t packed[2] = {0, 0};
#pragma unroll
  for (int i = 0; i < 8; i++) {
    __nv_fp4x2_storage_t p = __nv_cvt_float2_to_fp4x2(make_float2(f[2 * i] * inv, f[2 * i + 1] * inv), __NV_E2M1, cudaRoundNearest);
    packed[i / 4] |= uint32_t(p) << (8 * (i % 4));
  }
  *reinterpret_cast<uint2 *>(q + (size_t)r * (K / 2) + c * 8) = make_uint2(packed[0], packed[1]);

  const int rb = r >> 7, ri = r & 127, cb = c >> 2, ci = c & 3;
  sc[((size_t)rb * ncb + cb) * 512 + (ri & 31) * 16 + (ri >> 5) * 4 + ci] = s8.__x;
}
}

std::vector<torch::Tensor> nvfp4_quant(torch::Tensor x, double g, double alpha, c10::optional<torch::Tensor> bias, bool gelu) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous());
  const int K = x.size(-1), M = x.numel() / K;
  TORCH_CHECK(K % 64 == 0, "K must be a multiple of 64");
  const int nrb = (M + 127) / 128, ncb = (K / 16 + 3) / 4;
  auto q = torch::empty({M, K / 2}, x.options().dtype(torch::kUInt8));
  auto sc = torch::zeros({(long)nrb * 128, (long)ncb * 4}, x.options().dtype(torch::kUInt8));
  const long long n = (long long)M * (K / 16);
  const __nv_bfloat16 *bp = bias.has_value() ? reinterpret_cast<const __nv_bfloat16 *>(bias->data_ptr()) : nullptr;
  quant_nvfp4<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), q.data_ptr<uint8_t>(), sc.data_ptr<uint8_t>(), M, K, ncb, float(1.0 / g),
      float(alpha), bp, gelu ? 1 : 0);
  return {q, sc};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_quant", &nvfp4_quant, "act(alpha*x + bias) bf16 -> NVFP4 (packed e2m1 + blocked ue4m3 scales)",
        py::arg("x"), py::arg("g"), py::arg("alpha") = 1.0, py::arg("bias") = py::none(), py::arg("gelu") = false);
}
