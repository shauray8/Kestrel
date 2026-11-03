# Kestrel
good world models

## Experimental N3 FP8 GEMM

`kestrel.ops.n3()` exposes `n3_linear`, `n3_linear_gelu_f8`,
`n3_linear_f8in`, and `n3_linear_f8in_gelu_f8` with N1's argument order.
`N3=1` selects these in `lingbot/run.py`'s eligible N1Linear/FusedFFN
paths; calibration and the existing N1 mode toggles still apply. Defaults
are unchanged. `N3_STAGES=4|5|6` selects the TMA pipeline depth (default 5).
The dynamic shared-memory footprints are 65,600 / 82,000 / 98,400 bytes.

N3 uses 64-byte-swizzled 2D TMA loads for both e4m3 operands, transaction
mbarriers, eight consumer warps, and one active producer warp. Three
additional producer-group warps participate only in register reallocation
and synchronization: PTX requires all four warps in a warpgroup to execute
the same `setmaxnreg` instruction. Producers release registers to 24 and
restore 160 before rejoining the block. Consumers use the same
`m16n8k32` f16-accumulating MMA, K order, promotion after every 16 MMA
steps, final promotion, bias arithmetic, tanh-GELU, and `q2` as N1.
BF16 input is first quantized by a separate kernel with N1's float
reciprocal/multiply and e4m3 conversion; FP8 input needs no preprocessing.
This additional BF16-input launch and traffic are included in benchmarks.

The persistent scheduler retains N1's grouped tile ordering, launching at
most one block per SM and stepping through remaining tiles. It amortizes
block setup but does **not** by itself eliminate the final partially
occupied wave. Split-K reductions are deliberately not used because
reassociating the FP32 promotion sums would break bitwise equality.
No speed or bitwise-equivalence result is claimed before GPU testing.

`n3_linear_residual(x, wq, sw, bias, sa, residual, gate)` and the
`n3_linear_f8in_residual` counterpart update contiguous FP32 residuals in
place. Gate layout/broadcasting follows `fused_mod.gated_residual`.
The GEMM result is rounded to BF16 **before** multiplication by the FP32
gate; multiplication and addition are separately rounded, matching
`fused_mod.gated_residual(residual, n1_result, gate)` rather than a
different higher-precision fused expression.

### Hardware constraints and sources

- [PTX ISA 9.x §5.5.7](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-swizzling-modes)
  specifies the 64-byte swizzle and 512-byte alignment.
- [§9.7.10.28.5.3](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
  specifies tensor copies and mbarrier transaction completion; basic TMA
  requires sm_90 or newer.
- [§9.7.15.16.19](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)
  specifies parity waits and acquire ordering.
- [§9.7.21.5](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)
  explicitly supports sm_120a and requires warpgroup-uniform execution.
- [CUTLASS 79a](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
  explicitly states GeForce RTX 50 lacks TMA multicast and requires a
  1x1x1 cluster. Consequently N3 does not emit an unsupported multicast
  instruction or offer a misleading 2-CTA option.
- [CUTLASS's SM120 TMA path](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp)
  uses `.shared::cta` rather than `.shared::cluster` for SM120 loads,
  as does N3.

### GPU checks and benchmarks

Run through the GPU job wrapper, with the branch installed at
`/workspace/wt/gemm-tma`. The scripts emit JSON to stdout; redirect only
under the corresponding `/workspace/samples/<job-name>` directory.

```bash
bash /workspace/kestrel/scripts/gpujob.sh n3-build "PYTHONPATH=/workspace/wt/gemm-tma TORCH_EXTENSIONS_DIR=/workspace/ext/gemm-tma /workspace/venvs/kestrel/bin/python /workspace/wt/gemm-tma/tests/test_n3_linear.py --build-only"
bash /workspace/kestrel/scripts/gpujob.sh n3-test "PYTHONPATH=/workspace/wt/gemm-tma TORCH_EXTENSIONS_DIR=/workspace/ext/gemm-tma /workspace/venvs/kestrel/bin/python /workspace/wt/gemm-tma/tests/test_n3_linear.py"
bash /workspace/kestrel/scripts/gpujob.sh n3-bench "PYTHONPATH=/workspace/wt/gemm-tma TORCH_EXTENSIONS_DIR=/workspace/ext/gemm-tma /workspace/venvs/kestrel/bin/python /workspace/wt/gemm-tma/bench/n3_vs_cublas.py"
bash /workspace/kestrel/scripts/gpujob.sh n3-bench-cold "PYTHONPATH=/workspace/wt/gemm-tma TORCH_EXTENSIONS_DIR=/workspace/ext/gemm-tma /workspace/venvs/kestrel/bin/python /workspace/wt/gemm-tma/bench/n3_vs_cublas.py --cold"
```

Repeat correctness and benchmark jobs with `N3_STAGES=4` and
`N3_STAGES=6` in their command environments. Correctness requires bytewise
equality against N1, including graph capture/replay, both bias choices,
BF16/FP8 input, BF16/GELU-FP8 output, and gated/ungated residual.
Shapes cover M=129/512/3120/4680 and (N,K)=(1536,1536),
(8960,1536),(1536,8960). The benchmark measures captured GPU execution
for N1, N3, and `torch._scaled_mm` with both cuBLASLt accumulation modes,
reporting milliseconds, TFLOPS, and percentage of the 1016-TFLOPS FP8
f16-accumulation peak. cuBLASLt's accumulator is not a bitwise reference
for the N1/N3 ladder.
