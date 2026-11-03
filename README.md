# Kestrel
good world models

## Cluster-resident FFN

`ops.ffn_cluster().run(x8, sa, w0, sw0, b0, s_mid, w2, sw2, b2, out, gate=None)`
computes the 1536→8960→1536 tanh-GELU FFN without a global FP8 intermediate.
Inputs and weights are contiguous E4M3, weight scales are FP32 vectors, and optional
biases are BF16 vectors. `out` is contiguous BF16, or an existing FP32 residual
updated in place when an FP32 `[R,1536]` gate is supplied (`M % R == 0`).
The extension is standalone; existing VM and n1 paths are unchanged.
NVIDIA's [RTX Blackwell architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
lists 170 SMs and 96 MB L2 for RTX 5090; the benchmark also prints the runtime
SM count.

The [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#occupancy)
specifies 128 KiB shared memory per SM but only 99 KiB per block for compute
capability 12.0. The proposed 128-row, eight-block design requires 192 KiB per
block and cannot run on RTX 5090. Instead, each portable eight-block cluster
owns 64 rows: nine resident 64×128 FP8 tiles require at most 72 KiB per block,
and a two-stage A/B pipeline requires 24 KiB, totaling 96 KiB. The guide only
documents nonportable 16-block clusters for B200, so this path does not rely on
them. Up tiles are assigned cyclically to ranks; a cluster barrier publishes
them before every down tile traverses all 8960 K elements in ascending order,
loading DSMEM into local shared stages. A final cluster barrier protects DSMEM
lifetime. Promotion from FP16 to FP32 remains every 16 MMA steps, including a
final promotion; the down accumulator is never split across K partitions.

This is not an established speedup. Smaller row tiles increase weight traffic
and reduce reuse, and 96 KiB permits only one CTA per SM. The existing h8 and
weights can largely reside in RTX 5090 L2, so removed logical h8 traffic should
not be equated with saved DRAM traffic. Compare CUDA-event timings and profiler
L2/DRAM bytes before adopting this path.

From the repository root, with a CUDA 13 toolchain and RTX 5090:

```sh
PYTHONPATH=. python tests/test_ffn_cluster.py --build-only
PYTHONPATH=. python tests/test_ffn_cluster.py
PYTHONPATH=. python bench/ffn_cluster_bench.py
PYTHONPATH=. ncu --metrics lts__t_bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum --kernel-name regex:ffn_kernel --launch-skip 3 --launch-count 1 python bench/ffn_cluster_bench.py --variant cluster --m 3120 --no-gated
PYTHONPATH=. ncu --metrics lts__t_bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum --kernel-name 'regex:k_(reset|vm)' --launch-skip 6 --launch-count 2 python bench/ffn_cluster_bench.py --variant vm --m 3120 --no-gated
PYTHONPATH=. ncu --metrics lts__t_bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum --kernel-name regex:n1_f8 --launch-skip 6 --launch-count 2 python bench/ffn_cluster_bench.py --variant n1 --m 3120 --no-gated
```

The test requires bitwise equality with the VM and n1 paths for plain and gated
residual output at M=3120, M=4680, and a 129-row tail case. The benchmark defaults
to three warmups and twenty timed iterations and accepts `--variant`, `--m`,
and `--gated` to isolate profiler runs.
