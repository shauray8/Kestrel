#include "cluster_common.cuh"

__global__ void probe_mbarrier(int count, unsigned long long *cycles, unsigned *result) {
  __shared__ unsigned long long barrier;
  __shared__ volatile unsigned data;
  unsigned address = shared_addr(&barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(address) : "memory");
  unsigned completed = 0;
  unsigned long long start = clock64();
  for (int i = 0; i < count; ++i) {
    data = unsigned(i) + 1;
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(address) : "memory");
    unsigned ready = 0;
    for (int attempt = 0; attempt < 1024 && !ready; ++attempt) {
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                   : "=r"(ready) : "r"(address), "r"(unsigned(i) & 1) : "memory");
    }
    if (!ready || data != unsigned(i) + 1) break;
    ++completed;
  }
  cycles[0] = clock64() - start;
  result[0] = completed;
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(address) : "memory");
}

int main(int argc, char **argv) {
  try {
    int count = iterations(argc, argv);
    Buffer<unsigned long long> cycles(1);
    Buffer<unsigned> result(1);
    probe_mbarrier<<<1, 1>>>(17, cycles.p, result.p);
    ck(cudaGetLastError());
    ck(cudaDeviceSynchronize());
    require(result.get() == 17, "mbarrier correctness failed");
    double value = cluster_median_cycles(cycles, count,
      [&] { probe_mbarrier<<<1, 1>>>(count, cycles.p, result.p); },
      [&] { require(result.get() == unsigned(count), "mbarrier phase/data correctness failed"); });
    std::fprintf(stderr, "Single-thread, count-one mbarrier arrive/try_wait completed-phase round trip; includes loop and shared-data check.\n");
    emit("mbarrier_arrive_try_wait_latency", value, "cycles/pair");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
