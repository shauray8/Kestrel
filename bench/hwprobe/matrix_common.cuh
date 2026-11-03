#include "common.cuh"

__global__ void kernel(unsigned *out, int count) {
  __shared__ __align__(128) unsigned data[8 * 128];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < 8 * 128; i += blockDim.x) data[i] = MATRIX_STORE ? 0 : 0x01010101u;
  __syncthreads();
  unsigned addr = shared_addr(data + warp * 128 + lane * 4);
  unsigned a = 0x01010101u, b = a, c = a, d = a;
  for (int i = 0; i < count; ++i) {
#if MATRIX_STORE
    asm volatile(MATRIX_OPCODE " [%0], {%1,%2,%3,%4};" :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
#else
    asm volatile(MATRIX_OPCODE " {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr) : "memory");
#endif
  }
  __syncthreads();
#if MATRIX_STORE
  out[blockIdx.x * blockDim.x + threadIdx.x] = data[warp * 128 + lane] + data[warp * 128 + lane + 32] + data[warp * 128 + lane + 64] + data[warp * 128 + lane + 96];
#else
  out[blockIdx.x * blockDim.x + threadIdx.x] = a + b + c + d;
#endif
}

int main(int argc, char **argv) {
  try {
    int blocks = sms() * 2, count = iterations(argc, argv);
    Buffer<unsigned> out(blocks * 256);
    kernel<<<blocks, 256>>>(out.p, 1); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    std::vector<unsigned> values(out.n); ck(cudaMemcpy(values.data(), out.p, out.n * sizeof(unsigned), cudaMemcpyDeviceToHost));
    for (auto v : values) require(v == 0x04040404u, "matrix transfer correctness failed");
    double seconds = elapsed([&] { kernel<<<blocks, 256>>>(out.p, count); });
    emit("shared_bandwidth", double(blocks) * 8 * 512 * count / seconds / 1e9, "GB/s");
    return 0;
  } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
