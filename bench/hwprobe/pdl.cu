#include "common.cuh"

__device__ __forceinline__ unsigned long long pdl_timer() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}

__device__ __forceinline__ unsigned pdl_work(unsigned value, int count) {
  for (int i = 0; i < count; ++i)
    asm volatile("mad.lo.u32 %0, %0, 1664525, 1013904223;" : "+r"(value));
  return value;
}

__global__ void pdl_primary(int count, unsigned *data, unsigned long long *times) {
  times[0] = pdl_timer();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  unsigned value = pdl_work(7, count);
  data[0] = value;
  times[1] = pdl_timer();
}

__global__ void pdl_secondary(int count, const unsigned *data, unsigned *result, unsigned long long *times) {
  times[2] = pdl_timer();
  unsigned value = pdl_work(19, count);
  times[3] = pdl_timer();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  result[0] = value ^ data[0];
}

int main(int argc, char **argv) {
  try {
    int count = iterations(argc, argv, 65536);
    Buffer<unsigned> data(1);
    Buffer<unsigned> result(1);
    Buffer<unsigned long long> times(4);
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(1);
    config.blockDim = dim3(1);
    config.stream = stream;
    auto launch = [&](bool programmatic) {
      pdl_primary<<<1, 1, 0, stream>>>(count, data.p, times.p);
      ck(cudaGetLastError());
      config.attrs = programmatic ? &attribute : nullptr;
      config.numAttrs = programmatic ? 1 : 0;
      ck(cudaLaunchKernelEx(&config, pdl_secondary, count, data.p, result.p, times.p));
      ck(cudaGetLastError());
    };
    auto host_work = [&](unsigned value) {
      for (int i = 0; i < count; ++i) value = value * 1664525u + 1013904223u;
      return value;
    };
    unsigned primary = host_work(7);
    unsigned expected = primary ^ host_work(19);
    auto validate = [&] {
      require(data.get() == primary && result.get() == expected, "PDL prerequisite/independent work correctness failed");
    };
    for (bool programmatic : {false, true}) {
      launch(programmatic);
      ck(cudaStreamSynchronize(stream));
      validate();
    }
    cudaEvent_t start, stop;
    ck(cudaEventCreate(&start));
    ck(cudaEventCreate(&stop));
    std::vector<double> baseline, dependent;
    std::vector<unsigned long long> overlaps;
    for (int sample = 0; sample < 7; ++sample) {
      for (int order = 0; order < 2; ++order) {
        bool programmatic = ((sample + order) & 1) != 0;
        ck(cudaEventRecord(start, stream));
        launch(programmatic);
        ck(cudaEventRecord(stop, stream));
        ck(cudaEventSynchronize(stop));
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, start, stop));
        validate();
        (programmatic ? dependent : baseline).push_back(ms * 1e-3);
        if (programmatic) {
          unsigned long long host[4];
          ck(cudaMemcpy(host, times.p, sizeof(host), cudaMemcpyDeviceToHost));
          unsigned long long begin = std::max(host[0], host[2]);
          unsigned long long end = std::min(host[1], host[3]);
          overlaps.push_back(end > begin ? end - begin : 0);
        }
      }
    }
    ck(cudaEventDestroy(start));
    ck(cudaEventDestroy(stop));
    ck(cudaStreamDestroy(stream));
    std::sort(baseline.begin(), baseline.end());
    std::sort(dependent.begin(), dependent.end());
    std::sort(overlaps.begin(), overlaps.end());
    require(dependent[3] > 0, "PDL timing resolution insufficient");
    std::fprintf(stderr, "serialized_seconds=%.9g pdl_seconds=%.9g independent_overlap_ns=%llu; identical one-thread kernels in one nonblocking stream, only secondary launch attribute differs. PDL overlap is opportunistic, not guaranteed.\n", baseline[3], dependent[3], overlaps[3]);
    emit("pdl_launch_overlap_speedup", baseline[3] / dependent[3], "ratio");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
