#include "common.cuh"

__global__ void kernel(unsigned long long *cycles, unsigned *valid, int count) {
  unsigned x = threadIdx.x, leader = 0, chosen = 0;
  unsigned long long start = clock64();
  for (int i = 0; i < count; ++i) {
#if WARP_ELECT
    asm volatile("{ .reg .pred p; elect.sync %0|p, 0xffffffff; selp.u32 %1, 1, 0, p; }" : "=r"(leader), "=r"(chosen));
#else
    asm volatile("redux.sync.max.u32 %0, %0, 0xffffffff;" : "+r"(x));
#endif
  }
  unsigned long long stop = clock64();
#if WARP_ELECT
  unsigned mask = __ballot_sync(0xffffffff, chosen != 0);
  valid[threadIdx.x] = leader < 32 && mask == (1u << leader) && leader == __shfl_sync(0xffffffff, leader, 0);
#else
  valid[threadIdx.x] = x == 31;
#endif
  if (!threadIdx.x) cycles[0] = stop - start;
}

int main(int argc, char **argv) {
  try {
    int count = iterations(argc, argv);
    Buffer<unsigned long long> cycles(1); Buffer<unsigned> valid(32);
    kernel<<<1, 32>>>(cycles.p, valid.p, 1); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    for (int i = 0; i < 32; ++i) require(valid.get(i) == 1, "warp collective correctness failed");
    std::vector<double> samples;
    for (int i = 0; i < 9; ++i) {
      kernel<<<1, 32>>>(cycles.p, valid.p, count); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
      samples.push_back(double(cycles.get()) / count);
    }
    std::sort(samples.begin(), samples.end());
    emit("warp_instruction_loop", samples[samples.size() / 2], "cycles/iteration");
    return 0;
  } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
