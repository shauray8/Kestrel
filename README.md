# Kestrel
good world models

## Sync-free speculative forcing

`SPEC=1 SPEC_SYNCFREE=1` retains the existing speculative acceptance rule and
defaults remain unchanged. After the first denoise step, one batched readback
per chunk obtains the remaining active count and cache end. The remaining steps
use that count rounded up to 128 as a fixed capacity: sorted device compaction,
device counts, masked indexed K/V writes, and inverse-permutation reconstruction
never read the active mask back to Python. Counts can only decrease within a
chunk. An initially empty active set skips the remaining model calls; a later
empty set masks all writes and predictions without a host branch.

The sparse attention path reuses `ops.rope_kv()`'s rotation kernel through
`rope_qk_scatter`. Padding does not write K/V. The original near-dense fallback
(`active > tokens - 16`) still updates every cache row, selected on device.
The RoPE hook bypasses its otherwise redundant Q/K/V projection on sparse calls
and mirrors its cache-end integers on the host for dense and KV-update calls,
avoiding per-layer cache-index readbacks. The mirrors reset at generation start.
Sync-free mode installs this hook even when `ROPE_KV` is unset; the optimized
stack's existing hook can still wrap it for VM attention.
Scheduler lookup uses device `gather`, not scalar-tensor Python indexing.
Smoothing retains the same two convolutions and temporal reduction to preserve
their arithmetic; its Gaussian and temporal weights are cached. Redundancy
caches the frame-bank half of `cdist`'s augmented Euclidean-distance matrix
multiplication and computes the query norm only once across bank tiles.
Calibration and final statistics transfer to the host outside the step loop.

Bit identity is not guaranteed: 128-token capacity replaces 16-token padding,
and later steps retain the first sparse capacity. Linear GEMMs and attention can
therefore select different shape-dependent kernels/reduction orders. Near-dense
fallback queries are reordered active-first rather than raster-order, which can
also change Sage2's query quantization groups. These are numerical changes to
the executed linear/attention operators, not changes to acceptance thresholds,
the active set's ordering, or cache-write semantics. The cached `cdist`
decomposition uses the same FP32 `[-2q, sum(q*q), 1] @
[bank, 1, sum(bank*bank)].T`, clamp, square root, and tiled minimum as PyTorch's
matrix-multiply path; backend GEMM selection still needs GPU comparison.
The fused BF16 RoPE uses separate round-to-nearest FP32 multiplies/adds before
BF16 rounding, matching the previous Python sparse rotation.
Without the reference stack's `ROPE32=1 ROPE_KV=1`, the dense path also switches
from upstream complex-FP64 RoPE to this FP32/BF16 kernel.

CPU checks: `python -m unittest discover -s tests -p test_spec_compact.py`.
GPU indexed-RoPE checks: `python tests/test_rope_kv.py`. The CPU tests cover
bucket boundaries, shrinking/empty masks, legacy-padding equivalence, cached
distance arithmetic, and cache mirrors without scalar reads. Measure GPU
speed and output identity with the A/B commands below.

On the pod, create `/workspace/wt/spec-sync-free` from this branch, then run:

```bash
bash /workspace/kestrel/scripts/gpujob.sh specsf_base "cd /workspace/lingbot-world-v2 && env PYTHONPATH=/workspace/wt/spec-sync-free TORCH_EXTENSIONS_DIR=/workspace/ext/spec-sync-free KESTREL_OUT=/workspace/samples/specsf_base FUSED_GLUE=1 KESTREL_E_BCAST=1 CAM_HOIST=1 ATTN=sage2 FUSED_FFN=1 ROPE32=1 ROPE_KV=1 XATTN=1 CHUNK=2 SEED=42 FREE_BF16_W=0 LB_TIMING=1 TAE_LIVE=1 ONLY=n2_calib_rope32,n1n2_rope32 FRAMES=97 SPEC=1 SPEC_SYNCFREE=0 PROFILE=1 /workspace/venvs/kestrel/bin/python /workspace/wt/spec-sync-free/lingbot/run.py"
bash /workspace/kestrel/scripts/gpujob.sh specsf_syncfree "cd /workspace/lingbot-world-v2 && env PYTHONPATH=/workspace/wt/spec-sync-free TORCH_EXTENSIONS_DIR=/workspace/ext/spec-sync-free KESTREL_OUT=/workspace/samples/specsf_syncfree FUSED_GLUE=1 KESTREL_E_BCAST=1 CAM_HOIST=1 ATTN=sage2 FUSED_FFN=1 ROPE32=1 ROPE_KV=1 XATTN=1 CHUNK=2 SEED=42 FREE_BF16_W=0 LB_TIMING=1 TAE_LIVE=1 ONLY=n2_calib_rope32,n1n2_rope32 FRAMES=97 SPEC=1 SPEC_SYNCFREE=1 PROFILE=1 /workspace/venvs/kestrel/bin/python /workspace/wt/spec-sync-free/lingbot/run.py"
md5sum /workspace/samples/specsf_{base,syncfree}/05_n1n2_rope32_c2.mp4
/workspace/venvs/kestrel/bin/python /workspace/kestrel/eval/quality.py --ref /workspace/samples/specsf_base/05_n1n2_rope32_c2.mp4 /workspace/samples/specsf_syncfree/05_n1n2_rope32_c2.mp4
/workspace/venvs/kestrel/bin/python /workspace/kestrel/eval/drift.py /workspace/samples/specsf_{base,syncfree}/05_n1n2_rope32_c2.mp4 --prompt-file /workspace/lingbot-world-v2/examples/05/prompt.txt
bash /workspace/kestrel/scripts/gpujob.sh specsf_rope "cd /workspace/wt/spec-sync-free && env PYTHONPATH=/workspace/wt/spec-sync-free TORCH_EXTENSIONS_DIR=/workspace/ext/spec-sync-free /workspace/venvs/kestrel/bin/python tests/test_rope_kv.py"
```

Compare `dit_loop` records and the `timeline.idle_gaps` records in
`/workspace/logs/jobs/specsf_{base,syncfree}.log`; also compare the printed
calibrated thresholds and active fractions. The profiler changes timings, so
repeat the two generation commands with `PROFILE=0` and separate
`/workspace/samples/specsf_*` output directories for unprofiled latency.
