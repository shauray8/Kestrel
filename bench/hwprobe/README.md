# sm_120a hardware probes

Each `.cu` builds independently with `nvcc -arch=sm_120a` into its own process, so a rejected instruction, launch error, device fault, or timeout cannot block or poison another probe. `tma_encode.cu` is the one Python-hosted probe: it builds a cubin and `tma_encode.py` launches it through the driver API. Kestrel kernels and defaults are untouched. No GPU results are checked in.

`hwprobe.py` prints one JSON line per probe:

```json
{"feature":"mma_fp8_f16","compiled":true,"ran":true,"metric":"throughput","value":123,"unit":"TOP/s","error":null}
```

The number is a schema illustration, not a measurement. Failures keep `value`/`unit` null, set `metric` to `compile_failed`, `run_failed`, or `invalid_output`, and put the first `error`/`fatal` compiler line or the last stderr line into `error`. `ran` means executed, validated, and parsed. `--build-only` reports `compiled` with `metric: "build_only"`; `--run-only` reuses binaries from the same `--output-dir`. A failure never stops later probes; `--strict` makes the exit status nonzero if anything failed. Rejection of unsupported features is an expected result.

Options: `--list`, `--features 'mma_*' tma_load`, `--iterations N` (overrides every probe default: MMA/matrix/exp/latency 4096, transfers 256 rounds, PDL 65536, global-memory passes `N/256`, PDL-chain graph replays `N/256`, cluster confirmations `N`), `--timeout` (45 s/run), `--compile-timeout` (180 s/build). Logs live in `<output-dir>/<feature>/{compile,run}.log`; stderr carries secondary numbers (e.g. serialized PDL time, multicast rates, occupancy).

## Commands for the parent

```bash
bash /workspace/kestrel/scripts/gpujob.sh hwprobe-build "mkdir -p /workspace/samples/hwprobe-build && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/wt/hw-probe TORCH_EXTENSIONS_DIR=/workspace/ext/hw-probe /workspace/venvs/kestrel/bin/python /workspace/wt/hw-probe/bench/hwprobe/hwprobe.py --build-only --output-dir /workspace/samples/hwprobe-build | tee /workspace/samples/hwprobe-build/results.jsonl"

bash /workspace/kestrel/scripts/gpujob.sh hwprobe "mkdir -p /workspace/samples/hwprobe && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/wt/hw-probe TORCH_EXTENSIONS_DIR=/workspace/ext/hw-probe /workspace/venvs/kestrel/bin/python /workspace/wt/hw-probe/bench/hwprobe/hwprobe.py --output-dir /workspace/samples/hwprobe | tee /workspace/samples/hwprobe/results.jsonl"
```

Driver unit tests (no CUDA): `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s bench/hwprobe -p 'test_*.py'`.

Direct diagnosis on the GPU host: `/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120a bench/hwprobe/tma_load.cu -lcuda -o /workspace/samples/hwprobe-direct/tma_load && /workspace/samples/hwprobe-direct/tma_load 256`.

## Results table

Fill from `results.jsonl`; keep failed rows (with `error`) instead of deleting them.

| Probe | What it measures | Metric / unit | Compiled | Ran | Value | Error |
|---|---|---|---|---|---|---|
| `mma_f16_f16` | m16n8k16 FP16, f16 accumulate (harness calibration) | TOP/s | | | | |
| `mma_fp8_f16` / `mma_fp8_f32` | m16n8k32 e4m3, f16 / f32 accumulate | TOP/s | | | | |
| `mma_f6` / `mma_f6_f16` | kind::f8f6f4 e3m2 (bits 0-5), f32 / f16 accumulate | TOP/s | | | | |
| `mma_f4` / `mma_f4_f16` | kind::f8f6f4 e2m1 (bits 2-5), f32 / f16 accumulate | TOP/s | | | | |
| `mma_mxf8` / `mma_mxf6` / `mma_mxf4_1x` | kind::mxf8f6f4.block_scale 1X ue8m0, e4m3 / e3m2 / e2m1 | TOP/s | | | | |
| `mma_mxf4_kind` | kind::mxf4 2X ue8m0, packed e2m1 k64 | TOP/s | | | | |
| `mma_mxf4_2x_ue8m0` / `mma_mxf4_4x_ue4m3` | kind::mxf4nvf4 valid scale forms | TOP/s | | | | |
| `mma_mxf4_4x_ue8m0` | mxf4nvf4 4X ue8m0 (PTX 9.1; CUDA 13.0 may reject) | TOP/s | | | | |
| `mma_mxf4_2x_ue4m3` | invalid per PTX Table 45; expected compile rejection | TOP/s | | | | |
| `mma_int8` | m16n8k32 s8 | TOP/s | | | | |
| `ldmatrix_b16` / `stmatrix_b16` | m8n8.x4 b16, 512 B/warp/instr | GB/s | | | | |
| `ldmatrix_b8` / `stmatrix_b8` | m16n16.x2.trans / m16n8.x4.trans b8 | GB/s | | | | |
| `cp_async16` / `cp_bulk` / `tma_load` | 16 KiB tiles, 4-stage ring, L2-resident 22 MB set | delivered GB/s | | | | |
| `cp_async16_dram` / `cp_bulk_dram` / `tma_load_dram` | same, 178 MB set (exceeds 96 MB L2) | delivered GB/s | | | | |
| `tma_store` | 2D tensor store, ≤4 bulk groups in flight | GB/s | | | | |
| `tma_encode` | descriptor from ctypes libcuda and cuda-python (if installed), passed as `__grid_constant__` | validated backends | | | | |
| `tma_multicast{2,4,8}` | multicast vs independent TMA (expected rejection) | duration ratio | | | | |
| `cluster_launch{1,2,4,8}` | cluster launch + max dynamic SMEM/CTA (rejection recorded) | bytes/CTA | | | | |
| `cluster_barrier` | barrier.cluster arrive+wait, 2 CTAs | cycles/pair | | | | |
| `dsmem_bandwidth` / `dsmem_latency` | ld.shared::cluster peer reads | GB/s, cycles/load | | | | |
| `mbarrier` | arrive + try_wait.parity round trip | cycles/pair | | | | |
| `setmaxnreg` | dec 32 / inc 64 over a warpgroup | boolean | | | | |
| `pdl` | one-thread primary/secondary overlap (stderr: overlap ns) | serialized/PDL ratio | | | | |
| `pdl_chain` / `pdl_chain_serial` | 4096 graph-captured full-grid kernels with / without PDL | ns/kernel | | | | |
| `ex2_f32` / `ex2_f16x2` / `ex2_polynomial` | MUFU f32, MUFU packed f16, FMA-pipe range-reduced degree-5 | Gexp2/s | | | | |
| `l2_bandwidth` / `dram_bandwidth` | ld.global.cg, L2/3 set vs ≥4×L2 set | GB/s | | | | |
| `cache_normal` / `cache_evict_first` / `cache_evict_last` | hot-set replay after 2×L2 stream with createpolicy hint | GB/s | | | | |
| `redux` / `elect` | warp collectives | cycles/iteration | | | | |

## Measurement definitions

- MMA: eight independent accumulator chains/warp, eight warps/CTA, two CTAs/SM; `2·M·N·K` ops per instruction. All-ones operands (and unit scales) are checked exactly after one iteration on every lane before timing; f16-accumulate values saturate during timed runs, which does not affect throughput. Sub-byte encodings follow the PTX packing: e2m1 at bits 2-5 and e3m2 at bits 0-5 of each byte for kind::f8f6f4/mxf8f6f4; two e2m1 per byte for mxf4/mxf4nvf4.
- Matrix probes: equal bytes per instruction (512 B/warp), not equal shapes; fragments/stored words are checked.
- Transfers: 64×64 u32 tiles, 128 threads, one CTA/SM, 4 stages in flight per CTA (64 KiB), validated across two ring passes before timing. cp.async uses 16-byte `.cg` copies with `wait_group 3`; bulk and tensor loads signal per-stage mbarriers. Descriptors use `cuTensorMapEncodeTiled` (linked `-lcuda`), no swizzle, 256 B L2 promotion, passed as `const __grid_constant__ CUtensorMap`. Rates are logical bytes delivered, not counter-measured traffic.
- `tma_encode.py` loads `libcuda.so.1` through ctypes, encodes with ctypes and, when importable, with `cuda.bindings.driver` (or legacy `cuda.cuda`), reports whether the 128 descriptor bytes match, and validates a TMA load launched with each descriptor. Missing cuda-python is reported, not an error; a present-but-failing backend fails the probe.
- Multicast: every recipient is validated, then compared with independent TMA loads behind identical cluster barriers.
- Latencies include loop overhead (not empty-loop subtracted). `setmaxnreg` success proves execution, not that the allocator honored the hint.
- PDL: `pdl` compares the same one-thread kernel pair with/without `cudaLaunchAttributeProgrammaticStreamSerialization`. `pdl_chain*` captures 4096 launches into one CUDA graph (programmatic edges when enabled), each kernel issuing `griddepcontrol.launch_dependents` then `griddepcontrol.wait`; a serial counter validates ordering. The difference between the two rows is the per-kernel gap PDL hides.
- Exponentials: each element computes `0.25·exp2(x)` per iteration over 8 independent chains, inputs on [-4, 4). The polynomial uses round-to-nearest range reduction via the 1.5·2²³ trick, degree-5 Horner in FMA, and integer exponent insertion (≈10 FP32/INT ops). Accuracy bounds before timing: 1e-5 relative (f32, polynomial), 2e-3 (f16x2); f16 inputs are exactly representable.
- L2/DRAM use `.cg` 16-byte loads; cache-policy probes apply `createpolicy.fractional.L2::evict_{first,last}` to the interfering stream only and time an ordinary replay of the hot set.
- Cluster probes binary-search the largest dynamic SMEM that launches and validates; resource/cluster-size rejections are recorded, device faults are fatal. Stderr reports `cudaOccupancyMaxActiveClusters`.

## Documented availability versus measured acceptance

The probes intentionally distinguish documented instruction forms, assembler acceptance, runtime launch support, validated execution, and throughput. PTX's generic `sm_90 or higher` wording is not sufficient evidence that a GeForce GPU supports every cluster feature.

- [PTX §9.7.16.5.14, mma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma) target notes: e3m2/e2m3/e2m1 and `.kind`/`.block_scale` require sm_120a; `.f16` accumulate with e4m3 since PTX 8.7. [Table 45](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-block-scaling) allows mxf4nvf4 only as 2X/4X ue8m0 or 4X ue4m3, so **2X ue4m3 is invalid** (probe records the rejection). 4X ue8m0 is **PTX 9.1**; [CUTLASS `config.hpp`](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/config.hpp) enables `CUTE_ARCH_MXF4NVF4_4X_UE8M0_MMA_ENABLED` only for CUDA ≥ 13.1, so CUDA 13.0 may reject it. Opcode spellings match [CUTLASS `mma_sm120.hpp`](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm120.hpp), including kind::f8f6f4 `.f16` accumulate forms.
- [PTX §9.7.16.5.15, ldmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix), [§9.7.16.5.16, stmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-stmatrix): `.b8`, `.m16n16` (ldmatrix, `.trans` mandatory, x1/x2 only) and `.m16n8` (stmatrix, `.trans` mandatory) list sm_120a.
- [PTX cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async), [bulk](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk), [bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor), [tiled driver encoding](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html), and [CUTLASS TMA forms](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp). **[CUTLASS example 79a](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu#L56-L58) explicitly says GeForce RTX 50 lacks TMA multicast and requires cluster shape 1x1x1**, and the bulk.tensor target notes advise `.multicast::cluster` only for sm_90a/sm_100f/sm_100a/sm_103*/sm_110* (sm_120 absent). Cluster/multicast probes still attempt real builds/launches and record the outcome.
- [PTX mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier), [cluster barrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster), [mapa](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa), [ld](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld), [setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg), and [griddepcontrol](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol). Setmaxnreg explicitly lists SM120a; see also [CUTLASS register reconfiguration](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/reg_reconfig.h).
- [PTX FP32 ex2](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2), [FP16 ex2](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-ex2), [redux.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-redux-sync), [elect.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync), [createpolicy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-createpolicy).
