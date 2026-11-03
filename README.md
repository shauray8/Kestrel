# Kestrel
good world models

`KV_REUSE=1` skips the causal-fast sampler's clean-x0 KV-update forward.
`KV_REUSE_MODE=last` (default) retains each layer's already-appended final
denoise-step self-attention K/V, after normalization and RoPE, without copying.
`KV_REUSE_MODE=extrap` snapshots the penultimate step's appended chunk and
extrapolates the final K/V linearly in timestep to zero. It requires two or
more denoise steps and stores one extra chunk of K/V per layer.
Both modes preserve the sampler's x0 computation, cache indices, cross-attention
cache initialization, speculative history, and live-decoder chunk notifications.
Reused K/V are approximate: subsequent chunks can differ from the baseline.
Defaults are unchanged. `paths.kv_forward_skipped` counts skipped forwards.
CPU bookkeeping coverage: `python -m unittest discover -s tests -p test_kv_reuse.py`.
