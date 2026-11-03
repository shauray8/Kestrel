#include "cluster_common.cuh"

__global__ void probe_cluster_barrier(int count, unsigned long long *cycles, unsigned *result) {
  __shared__ unsigned marker;
  if (threadIdx.x == 0) marker = blockIdx.x + 1;
  cluster_sync();
  unsigned peer = cluster_rank() ^ 1;
  unsigned observed = cluster_load(cluster_address(&marker, peer));
  unsigned long long start = clock64();
  for (int i = 0; i < count; ++i) cluster_sync();
  unsigned long long stop = clock64();
  if (threadIdx.x == 0) {
    cycles[blockIdx.x] = stop - start;
    result[blockIdx.x] = observed;
  }
  cluster_sync();
}

int main(int argc, char **argv) {
  try {
    int count = iterations(argc, argv);
    Buffer<unsigned long long> cycles(2);
    Buffer<unsigned> result(2);
    auto launch = [&](int n) { ck(cluster_launch(probe_cluster_barrier, 2, 128, 2, 0, n, cycles.p, result.p)); };
    auto validate = [&] { require(result.get(0) == 2 && result.get(1) == 1, "cluster barrier remote data correctness failed"); };
    launch(17);
    ck(cudaGetLastError());
    ck(cudaDeviceSynchronize());
    validate();
    double value = cluster_median_cycles(cycles, count, [&] { launch(count); }, validate);
    std::fprintf(stderr, "Two-CTA cluster, 128 threads/CTA; median slowest CTA clock interval including loop overhead. Unsupported cluster launch fails independently.\n");
    emit("cluster_barrier_latency", value, "cycles/pair");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
