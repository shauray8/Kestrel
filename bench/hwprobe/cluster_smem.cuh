#pragma once
#include "cluster_common.cuh"

__global__ void probe_max_smem(size_t bytes, unsigned long long *result) {
  extern __shared__ unsigned char storage[];
  volatile unsigned char *data = storage;
  for (size_t i = threadIdx.x; i < bytes; i += blockDim.x)
    data[i] = static_cast<unsigned char>(i * 17 + blockIdx.x * 29);
  cluster_sync();
  unsigned long long sum = 0;
  for (size_t i = (threadIdx.x + 1) % blockDim.x; i < bytes; i += blockDim.x) sum += data[i];
  atomicAdd(result + blockIdx.x, sum);
  cluster_sync();
}

inline bool cluster_resource_rejection(cudaError_t error) {
  return error == cudaErrorInvalidConfiguration || error == cudaErrorLaunchOutOfResources || error == cudaErrorInvalidValue || error == cudaErrorInvalidClusterSize || error == cudaErrorNotSupported;
}

inline void cluster_clear_rejection(cudaError_t error) {
  cudaError_t pending = cudaGetLastError();
  if (pending != cudaSuccess && pending != error) ck(pending);
}

inline int cluster_smem_main(int argc, char **argv, int cluster) {
  try {
    int repeats = iterations(argc, argv, 1);
    int device = 0;
    ck(cudaGetDevice(&device));
    int cap = 0;
    ck(cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    cudaFuncAttributes attributes{};
    ck(cudaFuncGetAttributes(&attributes, probe_max_smem));
    require(size_t(cap) > attributes.sharedSizeBytes, "no dynamic shared memory available");
    cap -= int(attributes.sharedSizeBytes);
    ck(cudaFuncSetAttribute(probe_max_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, cap));
    Buffer<unsigned long long> result(cluster);
    std::vector<unsigned long long> host(cluster);
    auto attempt = [&](size_t bytes) {
      ck(cudaMemset(result.p, 0, result.n * sizeof(unsigned long long)));
      cudaError_t launch = cluster_launch(probe_max_smem, cluster, 128, cluster, bytes, bytes, result.p);
      if (launch != cudaSuccess) {
        if (cluster_resource_rejection(launch)) {
          cluster_clear_rejection(launch);
          std::fprintf(stderr, "cluster=%d dynamic_smem=%zu launch rejected: %s\n", cluster, bytes, cudaGetErrorString(launch));
          return false;
        }
        ck(launch);
      }
      cudaError_t pending = cudaGetLastError();
      if (pending != cudaSuccess) {
        if (cluster_resource_rejection(pending)) {
          std::fprintf(stderr, "cluster=%d dynamic_smem=%zu launch rejected: %s\n", cluster, bytes, cudaGetErrorString(pending));
          return false;
        }
        ck(pending);
      }
      cudaError_t execution = cudaDeviceSynchronize();
      if (execution != cudaSuccess)
        throw std::runtime_error(std::string("device execution failed, not a capacity rejection: ") + cudaGetErrorString(execution));
      ck(cudaMemcpy(host.data(), result.p, host.size() * sizeof(host[0]), cudaMemcpyDeviceToHost));
      for (int block = 0; block < cluster; ++block) {
        unsigned long long period = 0;
        unsigned long long tail = 0;
        for (size_t i = 0; i < 256; ++i) {
          unsigned value = unsigned((i * 17 + block * 29) & 255);
          period += value;
          if (i < bytes % 256) tail += value;
        }
        require(host[block] == (bytes / 256) * period + tail, "maximum dynamic shared-memory data check failed");
      }
      return true;
    };
    require(attempt(1), "cluster launch rejected even at one dynamic shared-memory byte; cluster is unsupported or unavailable");
    size_t low = 1;
    size_t high = size_t(cap);
    while (low < high) {
      size_t middle = low + (high - low + 1) / 2;
      if (attempt(middle)) low = middle;
      else high = middle - 1;
    }
    for (int i = 0; i < repeats; ++i) require(attempt(low), "maximum shared-memory confirmation launch rejected");
    for (size_t bytes : {size_t(0), low}) {
      cudaLaunchAttribute attribute{};
      attribute.id = cudaLaunchAttributeClusterDimension;
      attribute.val.clusterDim.x = cluster;
      attribute.val.clusterDim.y = 1;
      attribute.val.clusterDim.z = 1;
      cudaLaunchConfig_t config{};
      config.gridDim = dim3(cluster);
      config.blockDim = dim3(128);
      config.dynamicSmemBytes = bytes;
      config.attrs = &attribute;
      config.numAttrs = 1;
      int active = 0;
      cudaError_t error = cudaOccupancyMaxActiveClusters(&active, probe_max_smem, &config);
      if (error != cudaSuccess) cudaGetLastError();
      std::fprintf(stderr, "cluster=%d dynamic_smem=%zu cudaOccupancyMaxActiveClusters=%d (%s)\n", cluster, bytes, active, cudaGetErrorString(error));
    }
    std::fprintf(stderr, "cluster=%d maximum successful dynamic shared memory per CTA, searched byte-exactly through device opt-in limit %d; one cluster of 128-thread CTAs; device faults are fatal.\n", cluster, cap);
    std::string metric = "max_dynamic_smem_cluster" + std::to_string(cluster);
    emit(metric.c_str(), double(low), "bytes/CTA");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
