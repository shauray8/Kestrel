#include "common.cuh"
#include <cuda_fp16.h>

__device__ __forceinline__ float polynomial(float x) {
  float t, j, f, y;
  unsigned bits;
  asm volatile("add.rn.f32 %0, %1, 0f4B400000;" : "=f"(t) : "f"(x));
  asm volatile("sub.rn.f32 %0, %1, 0f4B400000;" : "=f"(j) : "f"(t));
  asm volatile("sub.rn.f32 %0, %1, %2;" : "=f"(f) : "f"(x), "f"(j));
  asm volatile("fma.rn.f32 %0, %1, 0f3AAEC3FF, 0f3C1D955B;" : "=f"(y) : "f"(f));
  asm volatile("fma.rn.f32 %0, %0, %1, 0f3D635847;" : "+f"(y) : "f"(f));
  asm volatile("fma.rn.f32 %0, %0, %1, 0f3E75FDF0;" : "+f"(y) : "f"(f));
  asm volatile("fma.rn.f32 %0, %0, %1, 0f3F317218;" : "+f"(y) : "f"(f));
  asm volatile("fma.rn.f32 %0, %0, %1, 0f3F800000;" : "+f"(y) : "f"(f));
  asm volatile("{ .reg .b32 e; shl.b32 e, %2, 23; add.s32 %0, %1, e; }" : "=r"(bits) : "r"(__float_as_uint(y)), "r"(__float_as_uint(t)));
  return __uint_as_float(bits);
}

__global__ void kernel(float *out, int count) {
  float x = (int(threadIdx.x & 31) - 16) / 4.f;
#if EXP_KIND == 1
  unsigned result[8];
#else
  float result[8] = {};
#endif
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float initial = x + j / 64.f;
#if EXP_KIND == 1
    __half2 input = __float2half2_rn(initial);
    result[j] = *reinterpret_cast<unsigned *>(&input);
#else
    result[j] = initial;
#endif
  }
  for (int i = 0; i < count; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#if EXP_KIND == 0
      asm volatile("ex2.approx.f32 %0, %0; mul.f32 %0, %0, 0f3E800000;" : "+f"(result[j]));
#elif EXP_KIND == 1
      asm volatile("ex2.approx.f16x2 %0, %0; mul.rn.f16x2 %0, %0, %1;" : "+r"(result[j]) : "r"(0x34003400u));
#else
      result[j] = polynomial(result[j]);
      asm volatile("mul.f32 %0, %0, 0f3E800000;" : "+f"(result[j]));
#endif
    }
  }
  float sum = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
#if EXP_KIND == 1
    float2 f = __half22float2(*reinterpret_cast<__half2 *>(&result[j]));
    sum += (f.x + f.y) * 0.5f;
#else
    sum += result[j];
#endif
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum * 0.125f;
}

int main(int argc, char **argv) {
  try {
    int count = iterations(argc, argv), blocks = sms() * 2;
    Buffer<float> out(blocks * 256);
    kernel<<<blocks, 256>>>(out.p, 1); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    std::vector<float> values(out.n); ck(cudaMemcpy(values.data(), out.p, out.n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < values.size(); ++i) {
      float ref = 0;
      for (int j = 0; j < 8; ++j) ref += std::exp2((int(i & 31) - 16) / 4.f + j / 64.f) / 32.f;
      require(std::abs(values[i] / ref - 1) < (EXP_KIND == 1 ? 0.002 : 0.00001), "exp2 accuracy bound failed");
    }
    double seconds = elapsed([&] { kernel<<<blocks, 256>>>(out.p, count); });
    emit("exp2_throughput", double(blocks) * 256 * 8 * count * (EXP_KIND == 1 ? 2 : 1) / seconds / 1e9, "Gexp2/s");
    return 0;
  } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
