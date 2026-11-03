# Kestrel
good world models

## VM attention performance experiment

Defaults remain unchanged. `VM_ATTN=1 VM_ATTN_PERF=48` or `96` selects 128-query, 8-warp VM attention tiles; `VM_ATTN_PERF=0` retains the 64-query, 4-compute-warp implementation. Set the switch before importing `kestrel.vm`; each mode has a separate extension name. The performance modes relax the VM launch bound from two blocks to one block, so measure mixed-op programs as well as attention alone.

The original VM reloads K/V twice as often per query window, converts BF16 vectors synchronously, has no next-tile prefetch, and leaves half the warps idle during MMA/softmax. Both implementations already use the same `ldmatrix` layouts, FP16 MMA ladder, and packed-half softmax reduction. Instruction/table staging happens once per query/head tile, not once per K/V tile, so it is amortized over long windows. Register pressure from forcing two resident VM blocks is another candidate bottleneck; the benchmark reports the compiled register count, local storage, shared memory, and actual occupancy rather than assuming two blocks fit.

The shared `attn_window.cuh` core keeps the N2 arithmetic order, 64-key recurrence, tail masks, and FP16-output rounding before BF16 conversion. The 48-KiB mode reuses Q shared storage after loading Q registers and prefetches the next K/V tile into registers. The 96-KiB mode retains Q storage and uses two `cp.async` K/V buffers; BF16 data is converted in shared memory after each wait, before consumers proceed. `N2_ATTN_SHARED=1` opts standalone N2 into the same core, while the default N2 kernel remains the independent parity reference.

`bench/attn_window_bench.py` times persistent VM, direct instruction tiles, original N2, and shared-core N2 at `(Lq,H,D)=(3120,12,128)` and `Lk=9360,18720,28080`, with three warmups and twenty CUDA-event iterations. It reports per-op `vm.profile` wait/exec/signal averages and checks exact parity. N2 timings use preconverted FP16 tensors; VM timings include BF16 conversion in the VM. Run separate processes with `VM_ATTN_PERF=0`, `48`, and `96`. `tests/test_vm.py --attn-only` covers ragged query/key tails and the full 28,080-token window, checking VM BF16 parity and shared-core FP16 parity against original N2 with zero mismatches required. GPU performance and parity are not yet measured for these variants.
