#include <cuda.h>

extern "C" __global__ void __launch_bounds__(128) tma_encode_kernel(const __grid_constant__ CUtensorMap map, unsigned *out) {
  __shared__ __align__(128) unsigned tile[64 * 64];
  __shared__ __align__(8) unsigned long long barrier;
  const unsigned bar = static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(64 * 64 * 4) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(tile))), "l"(reinterpret_cast<unsigned long long>(&map)), "r"(bar), "r"(0), "r"(int(blockIdx.x) * 64) : "memory");
    asm volatile("{ .reg .pred ready;\n"
        "wait_loop: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], 0;\n"
        "@!ready bra wait_loop; }" :: "r"(bar) : "memory");
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) out[size_t(blockIdx.x) * 64 * 64 + i] = tile[i];
}
