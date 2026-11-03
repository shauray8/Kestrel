#include "common.cuh"

__global__ void __maxnreg__(80) probe_setmaxnreg(int count, const unsigned *input, unsigned *output, unsigned *result) {
  size_t offset = size_t(threadIdx.x) * 64;
  const unsigned *src = input + offset;
  unsigned *dst = output + offset;
  asm volatile(
    "{ .reg .u32 x<64>;\n"
    "ld.volatile.global.v4.u32 {x0,x1,x2,x3}, [%0+0];\n"
    "ld.volatile.global.v4.u32 {x4,x5,x6,x7}, [%0+16];\n"
    "ld.volatile.global.v4.u32 {x8,x9,x10,x11}, [%0+32];\n"
    "ld.volatile.global.v4.u32 {x12,x13,x14,x15}, [%0+48];\n"
    "ld.volatile.global.v4.u32 {x16,x17,x18,x19}, [%0+64];\n"
    "ld.volatile.global.v4.u32 {x20,x21,x22,x23}, [%0+80];\n"
    "ld.volatile.global.v4.u32 {x24,x25,x26,x27}, [%0+96];\n"
    "ld.volatile.global.v4.u32 {x28,x29,x30,x31}, [%0+112];\n"
    "ld.volatile.global.v4.u32 {x32,x33,x34,x35}, [%0+128];\n"
    "ld.volatile.global.v4.u32 {x36,x37,x38,x39}, [%0+144];\n"
    "ld.volatile.global.v4.u32 {x40,x41,x42,x43}, [%0+160];\n"
    "ld.volatile.global.v4.u32 {x44,x45,x46,x47}, [%0+176];\n"
    "ld.volatile.global.v4.u32 {x48,x49,x50,x51}, [%0+192];\n"
    "ld.volatile.global.v4.u32 {x52,x53,x54,x55}, [%0+208];\n"
    "ld.volatile.global.v4.u32 {x56,x57,x58,x59}, [%0+224];\n"
    "ld.volatile.global.v4.u32 {x60,x61,x62,x63}, [%0+240];\n"
    "st.volatile.global.v4.u32 [%1+0], {x0,x1,x2,x3};\n"
    "st.volatile.global.v4.u32 [%1+16], {x4,x5,x6,x7};\n"
    "st.volatile.global.v4.u32 [%1+32], {x8,x9,x10,x11};\n"
    "st.volatile.global.v4.u32 [%1+48], {x12,x13,x14,x15};\n"
    "st.volatile.global.v4.u32 [%1+64], {x16,x17,x18,x19};\n"
    "st.volatile.global.v4.u32 [%1+80], {x20,x21,x22,x23};\n"
    "st.volatile.global.v4.u32 [%1+96], {x24,x25,x26,x27};\n"
    "st.volatile.global.v4.u32 [%1+112], {x28,x29,x30,x31};\n"
    "st.volatile.global.v4.u32 [%1+128], {x32,x33,x34,x35};\n"
    "st.volatile.global.v4.u32 [%1+144], {x36,x37,x38,x39};\n"
    "st.volatile.global.v4.u32 [%1+160], {x40,x41,x42,x43};\n"
    "st.volatile.global.v4.u32 [%1+176], {x44,x45,x46,x47};\n"
    "st.volatile.global.v4.u32 [%1+192], {x48,x49,x50,x51};\n"
    "st.volatile.global.v4.u32 [%1+208], {x52,x53,x54,x55};\n"
    "st.volatile.global.v4.u32 [%1+224], {x56,x57,x58,x59};\n"
    "st.volatile.global.v4.u32 [%1+240], {x60,x61,x62,x63}; }"
    :: "l"(src), "l"(dst) : "memory");
  __syncthreads();
  unsigned value = threadIdx.x + 1;
  for (int i = 0; i < count; ++i) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");
    __syncthreads();
    asm volatile("setmaxnreg.inc.sync.aligned.u32 64;" ::: "memory");
    __syncthreads();
    value = value * 1664525u + 1013904223u;
  }
  result[threadIdx.x] = value;
}

int main(int argc, char **argv) {
  try {
    int count = iterations(argc, argv);
    cudaFuncAttributes attributes{};
    ck(cudaFuncGetAttributes(&attributes, probe_setmaxnreg));
    require(attributes.numRegs >= 64, "compiler register allocation below 64: refusing potentially blocking setmaxnreg.inc");
    Buffer<unsigned> input(128 * 64);
    Buffer<unsigned> output(128 * 64);
    Buffer<unsigned> result(128);
    std::vector<unsigned> host(input.n);
    for (size_t i = 0; i < host.size(); ++i) host[i] = unsigned(i * 17 + 3);
    ck(cudaMemcpy(input.p, host.data(), host.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
    std::vector<unsigned> copied(output.n);
    std::vector<unsigned> values(result.n);
    auto run = [&](int n) {
      probe_setmaxnreg<<<1, 128>>>(n, input.p, output.p, result.p);
      ck(cudaGetLastError());
      ck(cudaDeviceSynchronize());
      ck(cudaMemcpy(copied.data(), output.p, copied.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
      require(copied == host, "setmaxnreg live-register prelude data mismatch");
      ck(cudaMemcpy(values.data(), result.p, values.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
      for (unsigned thread = 0; thread < 128; ++thread) {
        unsigned expected = thread + 1;
        for (int i = 0; i < n; ++i) expected = expected * 1664525u + 1013904223u;
        require(values[thread] == expected, "setmaxnreg full-warpgroup execution data mismatch");
      }
    };
    run(1);
    run(count);
    std::fprintf(stderr, "Compiled and correctness-checked %d dec32/inc64 pairs across one complete 128-thread warpgroup; explicit synchronization between every instruction, initial allocation %d registers/thread. This is execution support, not proof the hardware honored the register-allocation hint.\n", count, attributes.numRegs);
    emit("setmaxnreg_compile_run", 1, "boolean");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
