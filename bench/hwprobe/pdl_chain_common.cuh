#include "common.cuh"

constexpr unsigned kernels = 4096;

__global__ void __launch_bounds__(128) chain_step(unsigned *counter, unsigned *scratch, unsigned *errors, unsigned index) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  scratch[size_t(blockIdx.x) * blockDim.x + threadIdx.x] += index;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned seen = *counter;
    if (seen % kernels != index) atomicAdd(errors, 1u);
    *counter = seen + 1;
  }
}

int main(int argc, char **argv) {
  try {
    const int blocks = sms();
    const int graphs = std::max(1, iterations(argc, argv, 2304) / 256);
    Buffer<unsigned> counter(1), errors(1), scratch(size_t(blocks) * 128);
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(blocks);
    config.blockDim = dim3(128);
    config.stream = stream;
    config.attrs = PDL_CHAIN_PROGRAMMATIC ? &attribute : nullptr;
    config.numAttrs = PDL_CHAIN_PROGRAMMATIC ? 1 : 0;
    cudaGraph_t graph;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (unsigned i = 0; i < kernels; ++i) ck(cudaLaunchKernelEx(&config, chain_step, counter.p, scratch.p, errors.p, i));
    ck(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec;
    ck(cudaGraphInstantiate(&exec, graph, 0));
    ck(cudaGraphLaunch(exec, stream));
    ck(cudaStreamSynchronize(stream));
    require(errors.get() == 0 && counter.get() == kernels, "PDL chain ordering check failed");
    cudaEvent_t start, stop;
    ck(cudaEventCreate(&start));
    ck(cudaEventCreate(&stop));
    std::vector<double> samples;
    for (int g = 0; g < graphs; ++g) {
      ck(cudaEventRecord(start, stream));
      ck(cudaGraphLaunch(exec, stream));
      ck(cudaEventRecord(stop, stream));
      ck(cudaEventSynchronize(stop));
      float ms = 0;
      ck(cudaEventElapsedTime(&ms, start, stop));
      samples.push_back(ms * 1e6 / kernels);
    }
    require(errors.get() == 0 && counter.get() == kernels * unsigned(graphs + 1), "PDL chain ordering check failed after timing");
    std::sort(samples.begin(), samples.end());
    std::fprintf(stderr, "%u graph-captured back-to-back kernels, %d CTAs x 128 threads, launch_dependents then griddepcontrol.wait at entry, programmatic=%d; median of %d graph launches\n", kernels, blocks, PDL_CHAIN_PROGRAMMATIC, graphs);
    emit("kernel_to_kernel_period", samples[samples.size() / 2], "ns/kernel");
    ck(cudaGraphExecDestroy(exec));
    ck(cudaGraphDestroy(graph));
    ck(cudaEventDestroy(start));
    ck(cudaEventDestroy(stop));
    ck(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
