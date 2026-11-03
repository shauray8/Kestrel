#include "cluster_common.cuh"

__global__ void probe_dsmem_bandwidth(int count, unsigned *result) {
  __shared__ unsigned data[8192];
  for (unsigned i = threadIdx.x; i < 8192; i += blockDim.x) data[i] = blockIdx.x + 1 + i % 128;
  cluster_sync();
  unsigned base = cluster_address(data, cluster_rank() ^ 1);
  unsigned sums[8] = {};
  for (int i = 0; i < count; ++i) {
    unsigned offset = (unsigned(i) & 7) * 1024 + threadIdx.x;
    #pragma unroll
    for (int j = 0; j < 8; ++j) sums[j] += cluster_load(base + 4 * (offset + 128 * j));
  }
  unsigned total = 0;
  #pragma unroll
  for (int j = 0; j < 8; ++j) total += sums[j];
  result[blockIdx.x * blockDim.x + threadIdx.x] = total;
  cluster_sync();
}

int main(int argc, char **argv) {
  try {
    int count = iterations(argc, argv);
    int blocks = std::max(1, sms() / 2) * 2;
    Buffer<unsigned> result(size_t(blocks) * 128);
    std::vector<unsigned> host(result.n);
    auto launch = [&](int n) { ck(cluster_launch(probe_dsmem_bandwidth, blocks, 128, 2, 0, n, result.p)); };
    auto validate = [&](int n) {
      ck(cudaMemcpy(host.data(), result.p, result.n * sizeof(unsigned), cudaMemcpyDeviceToHost));
      for (int block = 0; block < blocks; ++block)
        for (unsigned lane = 0; lane < 128; ++lane) {
          unsigned expected = unsigned(8ull * unsigned(n) * (unsigned(block ^ 1) + 1 + lane));
          require(host[size_t(block) * 128 + lane] == expected, "DSMEM read checksum failed");
        }
    };
    launch(19);
    ck(cudaGetLastError());
    ck(cudaDeviceSynchronize());
    validate(19);
    double seconds = elapsed([&] { launch(count); });
    validate(count);
    std::fprintf(stderr, "Aggregate peer-CTA ld.shared::cluster read bandwidth, %d CTAs, two CTAs/cluster, 32 KiB/CTA; event timing includes initialization and lifecycle barriers.\n", blocks);
    emit("dsmem_read_bandwidth", double(blocks) * 128 * 8 * 4 * count / seconds / 1e9, "GB/s");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
