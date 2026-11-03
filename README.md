# Kestrel
good world models

## AB-Attention

`ATTN=ab` enables a causal-chunk adaptation of [AB-Attention](https://infrec-cardinal.github.io/AB-Attention/) ([paper](https://infrec-cardinal.github.io/AB-Attention/static/papers/ab-attention.pdf), [code](https://github.com/InfRec-Cardinal/AB-Attention)). Defaults for other attention backends are unchanged.

The selector uses 128-query blocks, eight farthest-first-initialized query clusters with one k-means update, and four-key selection blocks represented by two adjacent key-pair means. Cluster-weighted softmax masses feed capped score-proportional inclusion probabilities and systematic sampling without replacement. The N2 sparse kernel gathers those four-key blocks into 64-key execution tiles and adds `-log2(pi)` to the packed-FP16 logits. Attention is recomputed on current activations even when a sampling plan is reused. Text cross-attention remains dense. No token permutation is introduced.

| Variable | Default | Meaning |
| --- | --- | --- |
| `AB_SPARSITY` | `0.9` | Requested fraction of selection blocks skipped; budgets round up to 64-key tiles where possible |
| `AB_DENSE_STEPS` | `1` | Initial dense forwards per chunk, zero-based schedule reset at each `current_start` |
| `AB_DENSE_BLOCKS` | `1` | Initial dense transformer layers |
| `AB_PLAN_CYCLE` | `4` | Rebuild interval in forwards, starting at the first sparse forward |
| `AB_CLUSTERS` | `8` | Query clusters per 128-query block |
| `AB_SEED` | `42` | Sampling seed, independent of the model-generation RNG |
| `AB_KEEP_SINK` | `1` | Force sink-prefix blocks to inclusion probability one |
| `AB_KEEP_CURRENT` | `1` | Force current-chunk suffix blocks to inclusion probability one |

The paper uses 10 initial dense steps out of 50. The causal-fast default maps this to one initial dense forward out of four denoising forwards; the subsequent KV-update forward follows the same per-chunk schedule. Set `AB_KEEP_SINK=0 AB_KEEP_CURRENT=0` to sample the entire KV window as in the paper rather than imposing causal-serving retention anchors. Plans never cross chunk boundaries, and shape changes force rebuilding. `GRAPHS=1` is rejected because its replay would freeze chunk-dependent selections; eager execution and `GRAPHS=0` are supported.

At `M=3120`, `N=28080`, sink `9360`, and requested 90% sparsity:

| Keep sink | Keep current | Selected KV4 blocks | Achieved selection sparsity |
| --- | --- | --- | --- |
| 0 | 0 | 704 / 7020 | 89.9715% |
| 1 | 0 | 2340 / 7020 | 66.6667% |
| 0 | 1 | 780 / 7020 | 88.8889% |
| 1 | 1 | 3120 / 7020 | 55.5556% |

These are budget calculations, not measured quality or speed. Partial final execution tiles add padding work. `PATHS` reports `ab_blocks_kept` / `ab_blocks_total` over self-attention block pairs, including dense forwards/layers, plus dense/sparse call counts and plan builds/reuses.

`ops.n2().n2_attention_sparse(q,k,v,indices,log_bias,selection_block=4)` accepts contiguous CUDA FP16 `[tokens,heads,128]` tensors, contiguous int32 indices and float32 base-two logit biases shaped `[heads,ceil(query_tokens/128),samples]`. Each row must contain sorted, distinct, in-range block IDs with at least one real block; optional `-1` padding comes after real blocks. Zero biases implement ordinary masked attention. Selection blocks of 4 and 64 are supported. Ascending all-ones plans with zero biases preserve dense N2's operation order and are covered by bitwise-parity tests.

### GPU commands

Run from the pod after the parent creates `/workspace/wt/ab-attention`; no GPU results are claimed here.

```bash
bash /workspace/kestrel/scripts/gpujob.sh ab_tests 'cd /workspace/wt/ab-attention && PYTHONPATH=/workspace/wt/ab-attention TORCH_EXTENSIONS_DIR=/workspace/ext/ab-attention /workspace/venvs/kestrel/bin/python -m pytest -q tests/test_ab_attn.py'
bash /workspace/kestrel/scripts/gpujob.sh ab_bench 'cd /workspace/wt/ab-attention && PYTHONPATH=/workspace/wt/ab-attention TORCH_EXTENSIONS_DIR=/workspace/ext/ab-attention /workspace/venvs/kestrel/bin/python bench/ab_attn.py'
bash /workspace/kestrel/scripts/gpujob.sh ab_e2e 'cd /workspace/lingbot-world-v2 && for frames in 97 481; do for mode in dense paper keep; do attn=n2; keep=0; if [ "$mode" != dense ]; then attn=ab; fi; if [ "$mode" = keep ]; then keep=1; fi; env PYTHONPATH=/workspace/wt/ab-attention TORCH_EXTENSIONS_DIR=/workspace/ext/ab-attention KESTREL_OUT=/workspace/samples/ab_${mode}_${frames} ATTN=$attn ONLY=n2 CHUNK=2 WINDOW=18 SINK=6 FRAMES=$frames SEED=42 AB_SPARSITY=0.9 AB_DENSE_STEPS=1 AB_DENSE_BLOCKS=1 AB_PLAN_CYCLE=4 AB_KEEP_SINK=$keep AB_KEEP_CURRENT=$keep /workspace/venvs/kestrel/bin/python /workspace/wt/ab-attention/lingbot/run.py || exit; done; done'
bash /workspace/kestrel/scripts/gpujob.sh ab_eval 'for frames in 97 481; do for mode in paper keep; do /workspace/venvs/kestrel/bin/python /workspace/kestrel/eval/quality.py --ref /workspace/samples/ab_dense_${frames}/05_n2_c2.mp4 /workspace/samples/ab_${mode}_${frames}/05_n2_c2.mp4 || exit; done; /workspace/venvs/kestrel/bin/python /workspace/kestrel/eval/drift.py /workspace/samples/ab_dense_${frames}/05_n2_c2.mp4 /workspace/samples/ab_paper_${frames}/05_n2_c2.mp4 /workspace/samples/ab_keep_${frames}/05_n2_c2.mp4 --prompt-file /workspace/lingbot-world-v2/examples/05/prompt.txt || exit; done'
```

The benchmark separates unconstrained sparse-kernel timing at 50/75/90% sparsity from selector and selector-plus-kernel timing, and reports all four sink/current retention settings. E2E commands hold the prompt, seed, chunk size, window, model path, and N2 numerics fixed; `481` frames is the requested 30-second quality/drift case.
