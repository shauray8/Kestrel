# Kestrel
good world models

## Attention mass profiling

`ATTN_MASS=1 ATTN_MASS_OUT=<directory>` enables the eager self-attention
instrumentation in `bench/attn_mass.py` through `lingbot/run.py`. Defaults sample
zero-based chunks `1,5,11`, denoise steps `0,1,3` and the clean KV-update forward,
and layers `0,7,15,23,29`. Override those lists with `ATTN_MASS_CHUNKS`,
`ATTN_MASS_STEPS`, and `ATTN_MASS_LAYERS`. Calibration runs are excluded; the
hook leaves the actual attention implementation and its outputs unchanged.
CUDA graphs, speculative forcing, and VM attention are unsupported.

Each sampled layer is processed immediately, one head and 128-query block at a
time, using FP32 dense softmax. `summary.json` contains per-capture means and
metadata; individual capture JSON files contain all per-query-block/per-head
metrics. Query-block means are unweighted, including a potentially shorter last
block. KV block sizes are 4, 16, and 64, including a partial final KV block.
Blocks are ranked by mean probability mass across the query block:

- CDF: mass retained by the top 1/5/10/25/50/75/100% of blocks, rounding block
  counts up.
- Oracle selection: the highest-ranked block prefix whose actual key count fits
  25/50/75% of keys, rounding key budgets down. `key_fraction` records the
  realized budget; a block larger than the budget yields an empty selection.
- `mass`: mean retained softmax probability; `recall`: mean overlap with each
  query's dense top-budget individual keys, divided by that key budget.
- `error` and `error_renorm`: relative Frobenius output error against dense
  attention, without and with per-query probability renormalization.
- `region_mass`: disjoint sink, recent history, and current-chunk probability
  mass. Current-chunk tokens take precedence where early chunks overlap the sink;
  sink offsets account for the actual window passed to attention.

`ATTN_MASS_NPZ=1` additionally saves metric arrays. Add `ATTN_MASS_QKV=1` to
include the sampled rotated Q and actual windowed K/V as lossless FP32 arrays;
this can consume tens of GB of disk, but no layer's Q/K/V is retained afterwards.
These profiling runs are not latency benchmarks.

CPU check: `python -m unittest discover -s tests -p test_attn_mass.py`.

GPU run (from the host after preparing `/workspace/wt/attn-mass`):

```sh
bash /workspace/kestrel/scripts/gpujob.sh attn_mass "cd /workspace/lingbot-world-v2 && env PYTHONPATH=/workspace/wt/attn-mass TORCH_EXTENSIONS_DIR=/workspace/ext/attn-mass FUSED_GLUE=1 KESTREL_E_BCAST=1 CAM_HOIST=1 ATTN=n2 FUSED_FFN=1 ROPE32=1 ROPE_KV=1 XATTN=1 CHUNK=2 SEED=42 FRAMES=97 FREE_BF16_W=0 LB_TIMING=1 TAE_LIVE=1 ONLY=n2_calib_rope32,n1n2_rope32 ATTN_MASS=1 ATTN_MASS_OUT=/workspace/samples/attn_mass KESTREL_OUT=/workspace/samples/attn_mass /workspace/venvs/kestrel/bin/python /workspace/wt/attn-mass/lingbot/run.py"
```
