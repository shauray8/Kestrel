#include "common.cuh"

__global__ void read_data(const uint4 *data, size_t n, unsigned *out, int passes, int policy_kind) {
  unsigned long long policy = 0;
  if (policy_kind == 1) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  if (policy_kind == 2) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
  unsigned sum = 0;
  size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x, stride = size_t(gridDim.x) * blockDim.x;
  for (int p = 0; p < passes; ++p) {
    for (size_t i = tid; i < n; i += stride) {
      uint4 v;
      if (policy_kind) asm volatile("ld.global.cg.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(data + i), "l"(policy) : "memory");
      else asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(data + i) : "memory");
      sum += v.x + v.y + v.z + v.w;
    }
  }
  out[tid] = sum;
}

inline void check_reads(const Buffer<unsigned> &out, size_t n) {
  std::vector<unsigned> values(out.n); ck(cudaMemcpy(values.data(), out.p, out.n * sizeof(unsigned), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < out.n; ++i) {
    unsigned loads = i < n ? static_cast<unsigned>((n - 1 - i) / out.n + 1) : 0;
    require(values[i] == loads * 0x04040404u, "global load correctness failed");
  }
}

int main(int argc, char **argv) {
  try {
    cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, 0));
    require(prop.l2CacheSize > 0, "device reports no L2");
    int blocks = sms() * 4, passes = std::max(1, iterations(argc, argv) / 256);
    size_t resident = (size_t(prop.l2CacheSize) / 3 / 128) * 128;
    size_t bytes = MEMORY_KIND == 1 ? std::max(size_t(512) << 20, size_t(prop.l2CacheSize) * 4) : resident;
    Buffer<uint4> data(bytes / 16); Buffer<unsigned> out(blocks * 256);
    ck(cudaMemset(data.p, 1, bytes));
    read_data<<<blocks, 256>>>(data.p, data.n, out.p, 1, 0); ck(cudaGetLastError()); ck(cudaDeviceSynchronize()); check_reads(out, data.n);
#if MEMORY_KIND >= 2
    size_t cold_bytes = size_t(prop.l2CacheSize) * 2;
    Buffer<uint4> cold(cold_bytes / 16); ck(cudaMemset(cold.p, 1, cold_bytes));
    constexpr int policy = MEMORY_KIND - 2;
    read_data<<<blocks, 256>>>(cold.p, cold.n, out.p, 1, policy); ck(cudaGetLastError()); ck(cudaDeviceSynchronize()); check_reads(out, cold.n);
    cudaEvent_t a, b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
    std::vector<float> samples;
    for (int trial = 0; trial < 9; ++trial) {
      read_data<<<blocks, 256>>>(data.p, data.n, out.p, 1, 0);
      read_data<<<blocks, 256>>>(cold.p, cold.n, out.p, 1, policy);
      ck(cudaGetLastError()); ck(cudaEventRecord(a));
      read_data<<<blocks, 256>>>(data.p, data.n, out.p, 1, 0);
      ck(cudaGetLastError()); ck(cudaEventRecord(b)); ck(cudaEventSynchronize(b));
      float ms; ck(cudaEventElapsedTime(&ms, a, b)); samples.push_back(ms);
    }
    ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b));
    std::sort(samples.begin(), samples.end());
    emit("hot_replay_after_interference", bytes / (samples[4] * 1e-3) / 1e9, "GB/s");
    std::fprintf(stderr, "hot_bytes=%zu interference_bytes=%zu interference_policy=%d\n", bytes, cold_bytes, policy);
#else
    double seconds = elapsed([&] { read_data<<<blocks, 256>>>(data.p, data.n, out.p, passes, 0); });
    emit("logical_read_bandwidth", double(bytes) * passes / seconds / 1e9, "GB/s");
    std::fprintf(stderr, "working_set_bytes=%zu l2_bytes=%d passes=%d\n", bytes, prop.l2CacheSize, passes);
#endif
    return 0;
  } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
