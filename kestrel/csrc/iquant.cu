#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {
template <bool INT8, bool VALUE, bool FASTDIV>
__global__ void __launch_bounds__(256) iquant_kernel(const void *__restrict__ cache, const float *__restrict__ cscale,
    const __nv_bfloat16 *__restrict__ img, const __nv_bfloat16 *__restrict__ txt, const int64_t *__restrict__ idx,
    uint8_t *__restrict__ out, float *__restrict__ so, const float *__restrict__ mean, int H, int hist, int ni,
    int64_t cs, int64_t ss, int nk, int nb, int first, int repeat) {
  const int block = blockIdx.x + first, head = blockIdx.y, sh = head / repeat, t = threadIdx.x;
  const int row = t >> 2, part = t & 3, r = block * 64 + row;
  float x[32];
  const bool live = r < nk;
  const int64_t token = live ? idx[r] : 0;
  if (!live) {
#pragma unroll
    for (int i = 0; i < 32; ++i) x[i] = 0.f;
  } else if (token < hist) {
    if constexpr (INT8) {
      const int8_t *src = reinterpret_cast<const int8_t *>(cache) + token * cs + sh * 128 + part * 32;
      const float s = cscale[token * ss + sh];
#pragma unroll
      for (int i = 0; i < 32; i += 16) {
        const int4 raw = *reinterpret_cast<const int4 *>(src + i);
        const int8_t *b = reinterpret_cast<const int8_t *>(&raw);
#pragma unroll
        for (int j = 0; j < 16; ++j) x[i + j] = __bfloat162float(__float2bfloat16_rn(float(b[j]) * s));
      }
    } else {
      const __nv_bfloat16 *src = reinterpret_cast<const __nv_bfloat16 *>(cache) + token * cs + sh * 128 + part * 32;
#pragma unroll
      for (int i = 0; i < 32; i += 8) {
        const uint4 raw = *reinterpret_cast<const uint4 *>(src + i);
        const __nv_bfloat162 *b = reinterpret_cast<const __nv_bfloat162 *>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) { const float2 f = __bfloat1622float2(b[j]); x[i + 2 * j] = f.x; x[i + 2 * j + 1] = f.y; }
      }
    }
  } else {
    const __nv_bfloat16 *src = token < hist + ni ? img + (token - hist) * H * 128 : txt + (token - hist - ni) * H * 128;
    src += sh * 128 + part * 32;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
      const uint4 raw = *reinterpret_cast<const uint4 *>(src + i);
      const __nv_bfloat162 *b = reinterpret_cast<const __nv_bfloat162 *>(&raw);
#pragma unroll
      for (int j = 0; j < 4; ++j) { const float2 f = __bfloat1622float2(b[j]); x[i + 2 * j] = f.x; x[i + 2 * j + 1] = f.y; }
    }
  }
  if constexpr (!VALUE) {
    if (live) {
#pragma unroll
      for (int i = 0; i < 32; ++i) x[i] -= mean[sh * 128 + part * 32 + i];
    }
  }
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) m = fmaxf(m, fabsf(x[i]));
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  __shared__ float red[8];
  if ((t & 31) == 0) red[t >> 5] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
  float scale = fmaxf(FASTDIV ? __fdividef(m, VALUE ? 448.f : 127.f) : m / (VALUE ? 448.f : 127.f), 1.e-12f);
  if constexpr (VALUE) { int e; const float f = frexpf(scale, &e); scale = ldexpf(1.f, f == 0.5f ? e - 1 : e); }
  uint8_t *dst = out + (size_t(head) * nb + block) * 8192;
  if constexpr (VALUE) {
    const int sr = (row & 48) + ((row & 6) << 1) + (row & 1) + ((row & 8) >> 2);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const float v = FASTDIV ? __fdividef(x[i], scale) : x[i] / scale;
      dst[(part * 32 + i) * 64 + sr] = __nv_fp8_e4m3(v).__x;
    }
  } else {
    alignas(16) uint8_t q[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const float v = FASTDIV ? __fdividef(x[i], scale) : x[i] / scale;
      q[i] = uint8_t(int8_t(fminf(fmaxf(rintf(v), -127.f), 127.f)));
    }
    *reinterpret_cast<uint4 *>(dst + row * 128 + part * 32) = *reinterpret_cast<uint4 *>(q);
    *reinterpret_cast<uint4 *>(dst + row * 128 + part * 32 + 16) = *reinterpret_cast<uint4 *>(q + 16);
  }
  if (t == 0) so[head * nb + block] = scale;
}
}

void indexed_quant(torch::Tensor cache, torch::Tensor cscale, torch::Tensor img, torch::Tensor txt, torch::Tensor idx,
                   torch::Tensor out, torch::Tensor so, torch::Tensor mean, int64_t heads, int64_t H, int64_t hist, int64_t ni,
                   int64_t cs, int64_t ss, bool int8, int64_t nk, int64_t nb, int64_t first, bool value, int64_t repeat, bool fastdiv) {
  const c10::cuda::CUDAGuard guard(out.device());
  TORCH_CHECK(idx.scalar_type() == torch::kLong && idx.is_contiguous() && out.is_contiguous() && so.is_contiguous(), "iquant args");
  TORCH_CHECK((ni == 0 || img.scalar_type() == torch::kBFloat16) && (nk <= hist + ni || txt.scalar_type() == torch::kBFloat16)
              && (hist == 0 || (int8 ? cache.scalar_type() == torch::kInt8 : cache.scalar_type() == torch::kBFloat16)), "iquant dtypes");
  if (nb - first <= 0) return;
  dim3 grid(nb - first, heads);
  auto st = at::cuda::getCurrentCUDAStream();
  const float *cp = int8 ? cscale.data_ptr<float>() : nullptr;
#define IQ(I, V, F) iquant_kernel<I, V, F><<<grid, 256, 0, st>>>(cache.data_ptr(), cp, \
    reinterpret_cast<const __nv_bfloat16 *>(img.data_ptr()), reinterpret_cast<const __nv_bfloat16 *>(txt.data_ptr()), idx.data_ptr<int64_t>(), \
    out.data_ptr<uint8_t>(), so.data_ptr<float>(), mean.data_ptr<float>(), H, hist, ni, cs, ss, nk, nb, first, repeat)
  if (fastdiv) {
    if (int8) { if (value) IQ(true, true, true); else IQ(true, false, true); }
    else { if (value) IQ(false, true, true); else IQ(false, false, true); }
  } else {
    if (int8) { if (value) IQ(true, true, false); else IQ(true, false, false); }
    else { if (value) IQ(false, true, false); else IQ(false, false, false); }
  }
#undef IQ
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("indexed_quant", &indexed_quant); }
