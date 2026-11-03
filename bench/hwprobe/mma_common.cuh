#include "common.cuh"
#include <cuda_fp16.h>

#if defined(MMA_HALF)
using Acc = unsigned;
constexpr int REGS = 2;
#elif defined(MMA_INT)
using Acc = int;
constexpr int REGS = 4;
#else
using Acc = float;
constexpr int REGS = 4;
#endif

__device__ __forceinline__ void issue(Acc *d) {
  unsigned a = MMA_OPERAND;
#if defined(MMA_HALF)
  asm volatile(MMA_OPCODE " {%0,%1}, {%2,%2,%2,%2}, {%2,%2}, {%0,%1};" : "+r"(d[0]), "+r"(d[1]) : "r"(a));
#elif defined(MMA_INT)
  asm volatile(MMA_OPCODE " {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a));
#elif defined(MMA_SCALE)
  unsigned scale = MMA_SCALE;
  unsigned short selector = 0;
  asm volatile(MMA_OPCODE " {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3}, {%5}, {%6,%6}, {%5}, {%6,%6};" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a), "r"(scale), "h"(selector));
#else
  asm volatile(MMA_OPCODE " {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a));
#endif
}

__global__ void kernel(float *out, int count) {
  Acc d[8][REGS] = {};
  for (int i = 0; i < count; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) issue(d[j]);
  }
  float sum = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
#pragma unroll
    for (int k = 0; k < REGS; ++k) {
#if defined(MMA_HALF)
      float2 v = __half22float2(*reinterpret_cast<__half2 *>(&d[j][k]));
      sum += v.x + v.y;
#else
      sum += static_cast<float>(d[j][k]);
#endif
    }
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

int main(int argc, char **argv) {
  try {
    int count = iterations(argc, argv), blocks = sms() * 2;
    Buffer<float> out(blocks * 256);
    kernel<<<blocks, 256>>>(out.p, 1); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    std::vector<float> values(out.n); ck(cudaMemcpy(values.data(), out.p, out.n * sizeof(float), cudaMemcpyDeviceToHost));
    for (float v : values) require(v == 8 * 4 * MMA_K, "MMA all-ones correctness failed");
    double seconds = elapsed([&] { kernel<<<blocks, 256>>>(out.p, count); });
    emit("throughput", double(blocks) * 8 * 8 * count * 2 * 16 * 8 * MMA_K / seconds / 1e12, "TOP/s");
    return 0;
  } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
