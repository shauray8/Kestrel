#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

inline void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
inline void require(bool v, const char *s) { if (!v) throw std::runtime_error(s); }
inline int iterations(int argc, char **argv, int fallback = 4096) { return argc > 1 ? std::max(1, std::atoi(argv[1])) : fallback; }
inline int sms() { cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p, 0)); return p.multiProcessorCount; }
inline void emit(const char *metric, double value, const char *unit) { require(std::isfinite(value), "nonfinite metric"); std::printf("{\"metric\":\"%s\",\"value\":%.12g,\"unit\":\"%s\"}\n", metric, value, unit); }
template<class T> struct Buffer {
  T *p; size_t n;
  explicit Buffer(size_t count) : n(count) { ck(cudaMalloc(&p, n * sizeof(T))); ck(cudaMemset(p, 0, n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer &) = delete;
  T get(size_t i = 0) const { T v; ck(cudaMemcpy(&v, p + i, sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};
template<class F> double elapsed(F launch, int repeats = 7) {
  cudaEvent_t a, b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
  launch(); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
  std::vector<float> times;
  for (int i = 0; i < repeats; ++i) {
    ck(cudaEventRecord(a)); launch(); ck(cudaGetLastError()); ck(cudaEventRecord(b)); ck(cudaEventSynchronize(b));
    float ms; ck(cudaEventElapsedTime(&ms, a, b)); times.push_back(ms);
  }
  ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b));
  std::sort(times.begin(), times.end()); return times[times.size() / 2] * 1e-3;
}
__device__ __forceinline__ unsigned shared_addr(const void *p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
