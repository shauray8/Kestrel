#pragma once
#include "common.cuh"
#include <cstdint>
#if HP_TMA_MODE == 4
#include <cooperative_groups.h>
#endif
#ifndef HP_TMA_RING
#define HP_TMA_RING 8
#endif

namespace hp_tma {
constexpr int width = 64;
constexpr int height = 64;
constexpr int words = width * height;
constexpr int bytes = words * sizeof(unsigned);
constexpr int threads = 128;
constexpr int ring = HP_TMA_RING;
constexpr int stages = 4;

__host__ __device__ inline unsigned pattern(size_t i) { return static_cast<unsigned>(i) * 1664525u + 1013904223u; }

__global__ void initialize(unsigned *data, size_t count) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) data[i] = pattern(i);
}

inline void driver_check(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char *text = nullptr;
    cuGetErrorString(result, &text);
    throw std::runtime_error(text ? text : "CUDA driver error");
  }
}

inline CUtensorMap encode(unsigned *data, int tile_count) {
  alignas(64) CUtensorMap map{};
  require(reinterpret_cast<uintptr_t>(data) % 16 == 0, "tensor data is not 16-byte aligned");
  const cuuint64_t dimensions[2] = {width, cuuint64_t(height) * tile_count};
  const cuuint64_t strides[1] = {width * sizeof(unsigned)};
  const cuuint32_t box[2] = {width, height};
  const cuuint32_t element_strides[2] = {1, 1};
  driver_check(cuInit(0));
  driver_check(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, data, dimensions, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}

__device__ __forceinline__ unsigned long long map_address(const CUtensorMap &map) { return reinterpret_cast<unsigned long long>(&map); }

__device__ __forceinline__ void barrier_init(unsigned long long *barrier) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(shared_addr(barrier)) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void expect(unsigned long long *barrier) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(shared_addr(barrier)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void wait(unsigned long long *barrier, unsigned phase) {
  asm volatile("{ .reg .pred ready;\n"
      "wait_loop: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1;\n"
      "@!ready bra wait_loop; }"
      :: "r"(shared_addr(barrier)), "r"(phase) : "memory");
}

__device__ __forceinline__ void tensor_load(unsigned *tile, const CUtensorMap &map, unsigned long long *barrier, int tile_index) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      :: "r"(shared_addr(tile)), "l"(map_address(map)), "r"(shared_addr(barrier)), "r"(0), "r"(tile_index * height) : "memory");
}

#if HP_TMA_MODE <= 2
template<bool Validate>
__global__ void __launch_bounds__(threads) load(const unsigned *data, const __grid_constant__ CUtensorMap map, int rounds, unsigned *errors, unsigned *sink) {
  extern __shared__ __align__(128) unsigned char raw[];
  unsigned *tiles = reinterpret_cast<unsigned *>(raw);
  unsigned long long *full = reinterpret_cast<unsigned long long *>(raw + stages * bytes);
  if (HP_TMA_MODE != 0 && threadIdx.x == 0)
    for (int s = 0; s < stages; ++s) barrier_init(full + s);
  __syncthreads();
  auto issue = [&](int iteration) {
    const int slot = iteration % stages;
    const int tile_index = blockIdx.x + (iteration % ring) * gridDim.x;
    unsigned *tile = tiles + slot * words;
#if HP_TMA_MODE == 0
    if (iteration < rounds)
      for (int i = threadIdx.x * 4; i < words; i += threads * 4)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared_addr(tile + i)), "l"(data + size_t(tile_index) * words + i) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
#else
    if (iteration < rounds && threadIdx.x == 0) {
      expect(full + slot);
#if HP_TMA_MODE == 1
      asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
          :: "r"(shared_addr(tile)), "l"(data + size_t(tile_index) * words), "r"(bytes), "r"(shared_addr(full + slot)) : "memory");
#else
      tensor_load(tile, map, full + slot, tile_index);
#endif
    }
#endif
  };
  for (int i = 0; i < stages; ++i) issue(i);
  unsigned sum = 0;
  for (int iteration = 0; iteration < rounds; ++iteration) {
    const int slot = iteration % stages;
    const size_t offset = size_t(blockIdx.x + (iteration % ring) * gridDim.x) * words;
#if HP_TMA_MODE == 0
    asm volatile("cp.async.wait_group %0;" :: "n"(stages - 1) : "memory");
#else
    if (threadIdx.x == 0) wait(full + slot, (iteration / stages) & 1);
#endif
    __syncthreads();
    const unsigned *tile = tiles + slot * words;
    if constexpr (Validate) {
      for (int i = threadIdx.x; i < words; i += threads)
        if (tile[i] != pattern(offset + i)) atomicAdd(errors, 1u);
    }
    sum += tile[(iteration * threads + threadIdx.x) % words];
    __syncthreads();
    issue(iteration + stages);
  }
  sink[size_t(blockIdx.x) * threads + threadIdx.x] = sum;
}
#endif

#if HP_TMA_MODE == 3
__global__ void __launch_bounds__(threads) store(const __grid_constant__ CUtensorMap map, int rounds) {
  __shared__ __align__(128) unsigned tile[words];
  for (int i = threadIdx.x; i < words; i += threads) tile[i] = pattern(size_t(blockIdx.x) * words + i);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int iteration = 0; iteration < rounds; ++iteration) {
      const int row = (blockIdx.x + (iteration % ring) * gridDim.x) * height;
      asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];\n"
          "cp.async.bulk.commit_group;\ncp.async.bulk.wait_group.read %4;"
          :: "l"(map_address(map)), "r"(shared_addr(tile)), "r"(0), "r"(row), "n"(stages - 1) : "memory");
    }
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  __syncthreads();
}

__global__ void validate_store(const unsigned *data, int blocks, size_t count, unsigned *errors) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    if (data[i] != pattern(((i / words) % blocks) * words + i % words)) atomicAdd(errors, 1u);
}
#endif

#if HP_TMA_MODE == 4
template<bool Multicast, bool Validate>
__global__ void __launch_bounds__(threads) cluster_load(const __grid_constant__ CUtensorMap map, int rounds, unsigned *errors, unsigned *sink) {
  __shared__ __align__(128) unsigned tile[words];
  __shared__ __align__(8) unsigned long long barrier;
  auto cluster = cooperative_groups::this_cluster();
  if (threadIdx.x == 0) barrier_init(&barrier);
  cluster.sync();
  unsigned sum = 0;
  for (int iteration = 0; iteration < rounds; ++iteration) {
    const int tile_index = blockIdx.x / HP_TMA_CLUSTER + (iteration % ring) * (gridDim.x / HP_TMA_CLUSTER);
    if (threadIdx.x == 0) expect(&barrier);
    cluster.sync();
    if (threadIdx.x == 0) {
      if constexpr (Multicast) {
        if (cluster.block_rank() == 0) {
          const unsigned short mask = (1u << HP_TMA_CLUSTER) - 1;
          asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%4, %5}], [%2], %3;"
              :: "r"(shared_addr(tile)), "l"(map_address(map)), "r"(shared_addr(&barrier)), "h"(mask), "r"(0), "r"(tile_index * height) : "memory");
        }
      } else {
        tensor_load(tile, map, &barrier, tile_index);
      }
      wait(&barrier, iteration & 1);
    }
    __syncthreads();
    if constexpr (Validate) {
      for (int i = threadIdx.x; i < words; i += threads)
        if (tile[i] != pattern(size_t(tile_index) * words + i)) atomicAdd(errors, 1u);
    }
    sum += tile[(iteration * threads + threadIdx.x) % words];
    cluster.sync();
  }
  sink[size_t(blockIdx.x) * threads + threadIdx.x] = sum;
  cluster.sync();
}

template<bool Multicast, bool Validate>
inline void launch_cluster(const CUtensorMap &map, int blocks, int rounds, unsigned *errors, unsigned *sink) {
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = HP_TMA_CLUSTER;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(blocks);
  config.blockDim = dim3(threads);
  config.attrs = &attribute;
  config.numAttrs = 1;
  ck(cudaLaunchKernelEx(&config, cluster_load<Multicast, Validate>, map, rounds, errors, sink));
}
#endif

inline int run(int argc, char **argv) {
  const int rounds = iterations(argc, argv, 256);
#if HP_TMA_MODE <= 2
  const int blocks = sms();
#else
  int blocks = sms() * 2;
#endif
#if HP_TMA_MODE == 4
  blocks = ((blocks + HP_TMA_CLUSTER - 1) / HP_TMA_CLUSTER) * HP_TMA_CLUSTER;
  const int tile_count = (blocks / HP_TMA_CLUSTER) * ring;
  std::fprintf(stderr, "TMA multicast cluster=%d: CUTLASS example 79a documents GeForce RTX 50 as unsupported; attempting a real cluster launch\n", HP_TMA_CLUSTER);
#else
  const int tile_count = blocks * ring;
#endif
  const size_t count = size_t(tile_count) * words;
  Buffer<unsigned> data(count);
  Buffer<unsigned> errors(1);
  Buffer<unsigned> sink(size_t(blocks) * threads);
  initialize<<<blocks, threads>>>(data.p, count);
  ck(cudaGetLastError());
  ck(cudaDeviceSynchronize());
#if HP_TMA_MODE >= 2
  const CUtensorMap map = encode(data.p, tile_count);
#else
  const CUtensorMap map{};
#endif
#if HP_TMA_MODE <= 2
  const int smem = stages * bytes + stages * 8;
  ck(cudaFuncSetAttribute(load<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  ck(cudaFuncSetAttribute(load<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  load<true><<<blocks, threads, smem>>>(data.p, map, ring * 2, errors.p, sink.p);
  ck(cudaGetLastError());
  ck(cudaDeviceSynchronize());
  require(errors.get() == 0, "global-to-shared data validation failed");
  const double seconds = elapsed([&] { load<false><<<blocks, threads, smem>>>(data.p, map, rounds, errors.p, sink.p); });
#if HP_TMA_MODE == 0
  const char *metric = "cp_async16_delivered_gbps";
#elif HP_TMA_MODE == 1
  const char *metric = "cp_bulk_delivered_gbps";
#else
  const char *metric = "tma_load_delivered_gbps";
#endif
  std::fprintf(stderr, "tile=%dx%d u32 (%d bytes), stages=%d, blocks=%d (one per SM), rounds=%d, ring_bytes=%zu l2_resident=%s; logical delivered bytes, not counter-measured L2/DRAM traffic\n", width, height, bytes, stages, blocks, rounds, count * sizeof(unsigned), count * sizeof(unsigned) < (size_t(48) << 20) ? "likely" : "no");
  emit(metric, double(blocks) * rounds * bytes / seconds / 1e9, "GB/s");
#elif HP_TMA_MODE == 3
  store<<<blocks, threads>>>(map, ring * 2);
  ck(cudaGetLastError());
  ck(cudaDeviceSynchronize());
  validate_store<<<blocks, threads>>>(data.p, blocks, count, errors.p);
  ck(cudaGetLastError());
  ck(cudaDeviceSynchronize());
  require(errors.get() == 0, "shared-to-global data validation failed");
  const double seconds = elapsed([&] { store<<<blocks, threads>>>(map, rounds); });
  std::fprintf(stderr, "tile=%dx%d u32 (%d bytes), outstanding_groups<=%d, blocks=%d, rounds=%d, ring_bytes=%zu; logical bytes stored, includes shared initialization and final bulk-group completion\n", width, height, bytes, stages, blocks, rounds, count * sizeof(unsigned));
  emit("tma_store_delivered_gbps", double(blocks) * rounds * bytes / seconds / 1e9, "GB/s");
#else
  launch_cluster<false, true>(map, blocks, ring * 2, errors.p, sink.p);
  ck(cudaDeviceSynchronize());
  require(errors.get() == 0, "independent cluster load validation failed");
  launch_cluster<true, true>(map, blocks, ring * 2, errors.p, sink.p);
  ck(cudaDeviceSynchronize());
  require(errors.get() == 0, "multicast recipient data validation failed");
  const double independent = elapsed([&] { launch_cluster<false, false>(map, blocks, rounds, errors.p, sink.p); });
  const double multicast = elapsed([&] { launch_cluster<true, false>(map, blocks, rounds, errors.p, sink.p); });
  const double delivered = double(blocks) * rounds * bytes;
  std::fprintf(stderr, "cluster=%d, blocks=%d, rounds=%d; independent_delivered_GBps=%.9g multicast_delivered_GBps=%.9g; multicast ideal source bytes are delivered/%d, not measured L2 traffic; both include identical cluster barriers\n", HP_TMA_CLUSTER, blocks, rounds, delivered / independent / 1e9, delivered / multicast / 1e9, HP_TMA_CLUSTER);
  emit("tma_multicast" HP_TMA_CLUSTER_TEXT "_speedup", independent / multicast, "ratio");
#endif
  return 0;
}
}

int main(int argc, char **argv) {
  try { return hp_tma::run(argc, argv); }
  catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
