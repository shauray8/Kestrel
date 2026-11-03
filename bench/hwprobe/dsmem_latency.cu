#include "cluster_common.cuh"

__global__ void probe_dsmem_latency(int count, unsigned long long *cycles, unsigned *result) {
  __shared__ unsigned next[4096];
  unsigned rank = cluster_rank();
  for (unsigned i = threadIdx.x; i < 4096; i += blockDim.x)
    next[i] = cluster_address(&next[(i + 17) & 4095], rank);
  cluster_sync();
  if (threadIdx.x == 0) {
    unsigned base = cluster_address(next, rank ^ 1);
    unsigned address = base;
    unsigned long long start = clock64();
    for (int i = 0; i < count; ++i) address = cluster_load(address);
    unsigned long long stop;
    asm volatile("st.volatile.global.u32 [%1], %2; mov.u64 %0, %%clock64;"
                 : "=l"(stop) : "l"(result + blockIdx.x), "r"((address - base) / unsigned(sizeof(unsigned))) : "memory");
    cycles[blockIdx.x] = stop - start;
  }
  cluster_sync();
}

int main(int argc, char **argv) {
  try {
    int count = iterations(argc, argv);
    Buffer<unsigned long long> cycles(2);
    Buffer<unsigned> result(2);
    auto launch = [&](int n) { ck(cluster_launch(probe_dsmem_latency, 2, 128, 2, 0, n, cycles.p, result.p)); };
    auto validate = [&](int n) {
      unsigned expected = unsigned(17ull * unsigned(n) & 4095);
      require(result.get(0) == expected && result.get(1) == expected, "DSMEM dependent pointer chase failed");
    };
    launch(19);
    ck(cudaGetLastError());
    ck(cudaDeviceSynchronize());
    validate(19);
    double value = cluster_median_cycles(cycles, count, [&] { launch(count); }, [&] { validate(count); });
    std::fprintf(stderr, "One dependent remote ld.shared::cluster chain per CTA in a two-CTA cluster, 16 KiB ring/CTA, stride 17; includes loop overhead.\n");
    emit("dsmem_dependent_load_latency", value, "cycles/load");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
