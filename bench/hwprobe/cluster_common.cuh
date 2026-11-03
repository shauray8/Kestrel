#pragma once
#include "common.cuh"

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ unsigned cluster_rank() {
  unsigned rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
}

__device__ __forceinline__ unsigned cluster_address(const void *p, unsigned rank) {
  unsigned address;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(address) : "r"(shared_addr(p)), "r"(rank));
  return address;
}

__device__ __forceinline__ unsigned cluster_load(unsigned address) {
  unsigned value;
  asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(value) : "r"(address) : "memory");
  return value;
}

template<class Kernel, class... Args>
cudaError_t cluster_launch(Kernel kernel, int blocks, int threads, int cluster, size_t smem, Args... args) {
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = cluster;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(blocks);
  config.blockDim = dim3(threads);
  config.dynamicSmemBytes = smem;
  config.attrs = &attribute;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, kernel, args...);
}

template<class Launch, class Validate>
double cluster_median_cycles(Buffer<unsigned long long> &cycles, int count, Launch launch, Validate validate) {
  std::vector<double> samples;
  std::vector<unsigned long long> host(cycles.n);
  for (int sample = 0; sample < 7; ++sample) {
    launch();
    ck(cudaGetLastError());
    ck(cudaDeviceSynchronize());
    validate();
    ck(cudaMemcpy(host.data(), cycles.p, host.size() * sizeof(host[0]), cudaMemcpyDeviceToHost));
    samples.push_back(double(*std::max_element(host.begin(), host.end())) / count);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}
