# One RTX 5090: resident inference

## Current acceptance status

Use `NOMAS_DEC_GEMM=n3`. Decoder NVFP4 (`NOMAS_DEC_GEMM=nvfp4` / `NOMAS_DEC_FP4=1`) is a rejected quality experiment: the user observed visible degradation in both `f4retry` and `gf4`. Do not promote their sub-10-second timings or use a metric pass to override that observation. An identical-latent comparison also failed the fallback quality gate: PSNR 35.29 dB, SSIM 0.9164 against 0.95, measuring every fourth frame (60 frames). Decoder-only warm times were 5.946 s N3 FP8 versus 4.873 s NVFP4; speed does not establish acceptance.

The corrected Triton glue and synthetic DiT/decoder graph checks passed together on the pod (`glue_graph_parity4`, 165 case records, exit 0). This exercises `glue_check.py --repeats 5` and `graph_check.py --camera --intra --int8-history`; it is not a full-clip visual gate. Keep `NOMAS_GLUE_CUDA=0`: native compilation succeeds, but its strict FP32 ULP parity gate failed. Defaults remain unchanged.

Earlier timing candidates used a fused-modulation autocast precision bug, corrected by preserving FP32 LayerNorm/modulation under CUDA autocast. Keep those historical timings separate from corrected runs. Final corrected N3 FP8 + graphs completed with prompt totals 12.556925991 s first/cold, then 10.321051457 and 10.311258659 s warm (`nomas_gfp8correct`, exit 0). This still misses the 10 s deadline by about 0.32 s; full-clip visual acceptance and repeated matched timing remain open. No accepted new real-time operating point.

## Required launch

Run from kvm-mini through `python -m nomas.run`. The runner must import `hymm.inference`, call `nomas.pipe.install()`, and then call that imported module's `main()` with the normal Hydra argv. Do not execute a second copy with `runpy`: the prompt-loader patch is on the imported module.

```sh
NOMAS_PIPE=1 python -m nomas.run \
  +experiment=single_gpu_low_mem_inference \
  dit_weight=/workspace/models/hy-nomas-int8 \
  model.vae.checkpoint=/workspace/models/pixelumm.safetensors-fp8 \
  model.text_encoder.model_name=/workspace/models/Qwen2.5-VL-7B-Instruct \
  parallelism.sp_size=1 parallelism.shard_text_encoder=false \
  parallelism.text_encoder_cpu_offload=true \
  parallelism.dit_cpu_offload=false parallelism.dit_cpu_offload_for_decode=false \
  model.vae.cpu_offload=false model.vae.parallel_mode=none \
  model.vae.cutlass_fp8=true model.vae.deepgemm_fp8=false \
  model.vae.num_steps=1 model.vae.temporal_window_latents=10 \
  model.vae.temporal_overlap_latents=5 \
  inference.rollout_type=seg_ar inference.infer_steps_video=8 \
  'inference.video_size=[480,832]' inference.video_fps=24 \
  inference.dit_weight_quantization=int8 inference.kv_cache_quantization=int8 \
  inference.sliding_window.enabled=true inference.sliding_window.past_frames=18 \
  inference.sliding_window.sampling=temporal_context \
  inference.torch_compile=true inference.inference_mode=true \
  inference.profile_memory=false inference.output_chunk_frames=8 \
  inference.seg_ar_segment_vae_decode=false model.vae.stable_prefix_streaming=false \
  inference.save_latents=false inference.overlay_keys=false \
  inference.prompt=/workspace/prompts/two_segments.json \
  inference.save_path=/workspace/outputs/nomas-resident
```

Replace checkpoint/prompt/output paths with the deployment's real paths. `inference.max_segments=2` caps longer prompt entries; it does not set each segment's duration. Use two 5-second events in the prompt JSON. Keep the existing third-person prompt processing and camera overrides.

**These residency overrides alone are unsafe without `NOMAS_PIPE=1`.** The reported OOM at 31.34 GiB occurred when the upstream text-encoder hook restored Qwen while DiT and PixelUMM were already on GPU. `text_encoder_cpu_offload=true` means CPU storage between calls, not CPU execution (`hymm/utils/model_cpu_offload.py:70-89,93-117`).

`NOMAS_PIPE=1` temporarily forces all three models to load on CPU, requiring the already serialized FP8 decoder checkpoint. Once `_load_prompt_entries` has applied the exact upstream prompt formatting, it pre-encodes every selected `max_iters` item's segment text with **only Qwen on GPU**, caches the final text states/masks on CPU, deletes `text_encoder.model`, collects/free-caches its GPU allocations, and finally transfers DiT and PixelUMM to GPU once. The text-encoder shell/tokenizer stays for action-token offsets; no text model or its offload controller remains. There is no model reload on subsequent items and no cache eviction. Unlisted/new prompts after encoder release fail explicitly rather than silently loading Qwen into an OOM. All prompts must be in the input suite before startup. Cache lifetime is this process, not a disk cache.

The cold startup still pays checkpoint I/O and one Qwen encode per distinct `(prompt, data_type, is_uncond)`; it is not eliminated or hidden in a per-prompt throughput number. Repeated texts and negative prompts use the cached states. The stock inference loop loads models once (`hymm/inference.py:1676-1691`) and encodes every segment again on each item (`hymm/pipeline/causal_ar.py:2047-2053`).

## Kernel overrides

Keep the existing sparse policy unless deliberately comparing dense Kestrel attention. Its router/layout overrides are:

```text
model.dit.seg_ar_attention_backend=flex
inference.sparse_attention.enabled=true
inference.sparse_attention.router_mode=triton_bf16
inference.sparse_attention.flex_sparse_layout=aligned
inference.sparse_attention.memory_efficient=true
inference.sparse_attention.memory_adaptive_heads=true
```

Kestrel attention/GEMM activation is owned by `nomas.attn` and `nomas.gemm`, not `pipe.py`. The runner installs pipe first so their weight conversions see CPU-loaded tensors and the final device transfer preserves INT8/FP8 payloads and FP32 scales. Do not compare dense FA with sparse flex as if the work and routing were identical.

## Optional overlap

Set `NOMAS_PIPE_ASYNC_DECODE=1` **only after measuring the serial resident peak**. Keep `inference.seg_ar_segment_vae_decode=false`, `model.vae.stable_prefix_streaming=false`, `model.vae.parallel_mode=none`, and `inference.profile_memory=false`. This option supports offline `seg_ar`, no preserved interactive cache, and no segment callback.

The patch inserts a boundary hook immediately after the unique `with stage("cache_write")` scope in `CPTCausalInferencePipeline._process_single_prompt` (`hymm/pipeline/causal_ar.py:2502-2876`), before timing synchronization. This structural anchor does not depend on timing-variable names or values: the earlier assignment matcher incorrectly counted both assignments to `segment_decode_elapsed` and rejected the unchanged checkout. A single worker thread runs PixelUMM's existing `create_streaming_decode_session(..., to_cpu=True, retain_history=False)`, `push`, and final `finish` on a dedicated CUDA stream. GPU events protect each cloned latent handoff; stream ownership protects allocator lifetimes; queued failures propagate at final assembly. Stable pixels are kept on CPU, the withheld cross-boundary window is completed once future latents arrive, and only final stable pixels are returned. No provisional preview is decoded twice. The one-step window protocol is upstream `pixelumm_decoder/streaming.py:483-536,614-646`.

The request-local upstream device-wide timing synchronizations become current-stream synchronizations, otherwise they would serialize the two streams. This is an AST-anchored in-process patch, not an edit to Hy-Nomas. It fails if the unique segment hook cannot be found. Detailed memory profiling is rejected in overlap mode because its device-wide stage synchronizations would destroy overlap. Wall-time sample totals still synchronize at final completion.
The boundary drops generation-only forward tensors before scheduling decode. The session overrides only its window-output placement to use `_predict_x0(..., output_cpu=True)`, the same CPU blender supported by normal resident decoding; this avoids assembling an entire segment's pixels on GPU. Only a temporal decoder window/noise plus the overlap and small latent handoffs need concurrent GPU storage, rather than a GPU video per queued segment. `async_decode_segment` logs each worker segment's time/allocated/reserved GiB. Kestrel FA engines and balanced-kernel scratch must be stream-isolated; a global attention workspace shared by DiT and decoder is not concurrency-safe.


Native demo settings `inference.seg_ar_segment_vae_decode=true model.vae.stable_prefix_streaming=true` publish replace-tail previews but are **blocking**, with `.cpu()` and `torch.cuda.synchronize` at segment boundaries (`causal_ar.py:2947-3064`). They are useful for first-playable-frame latency, not automatically throughput overlap. `output_chunk_frames=8` only chunks conversion of decoded output to uint8 (`hymm/inference.py:2243-2245`); it does not pipeline the decoder.

Overlap can lose on a single saturated GPU through SM/memory-bandwidth contention; expected benefit comes from filling the substantial host/launch gaps, not doubling GPU throughput. Peak includes DiT + decoder activations + persistent KV simultaneously. Do not assume it fits because serial residency fits.

## Compilation and graphs

`inference.torch_compile=true` sets each block's `compile_cache_prepost` (`base.py:141-143`), not a graph of the full Euler step. Upstream pre/post functions are cached once per process with `triton.cudagraphs=false` (`nomas_compile_prepost.py:102-109`), and the block invokes those helpers at `nomas_cpt.py:1324-1368,1410-1437`. Shared compiled callables accept parameter tensors from all 54 blocks. Flex attention and some router helpers have independent compile caches.

Leave `NOMAS_PIPE_COMPILE_MODE` unset to retain upstream behavior. `NOMAS_PIPE_COMPILE_MODE=default` explicitly reinstalls the same no-CUDA-graph pre/post boundary; `max-autotune-no-cudagraphs` is an opt-in benchmark with higher cold-start cost. The patch updates both the defining module and the imported aliases in `nomas_cpt`. Kestrel GEMM may replace these boundaries, in which case its behavior takes precedence.

Compiler-managed `reduce-overhead`/whole-step CUDA graphs are not enabled by the pipe compile-mode override. It rejects graph modes rather than advertising a compiler graph speedup it cannot guarantee. Keep shape, guidance branches, prompt length/template, and inference-mode constant between measured items; inspect upstream `.compile.rank*.json`/sample metrics and `TORCH_LOGS=recompiles` to distinguish first-specialization cost from recurring recompiles. Do not clear compiler caches per prompt.

`NOMAS_GRAPH=1` separately opts into explicit DiT-denoising and PixelUMM-window CUDA graphs; it does not change the pipe compile mode. `NOMAS_GRAPH_DIT=0` leaves DiT and its cache commits eager and graphs only decoder windows (default `NOMAS_GRAPH_DIT=1`). Decoder graphs require `NOMAS_DEC_ATTN=fa`; DiT graphs additionally require `NOMAS_ATTN=fa`, dense resident BF16/INT8 history and precomputed camera embeddings. With `NOMAS_GLUE=1`, one-frame camera chunks accept either segment-local or clip-level keyboards: singleton ranges select the first `tt` keys exactly as upstream; the pipeline normally slices each segment before this call. `camera_rope_time_offset` changes RoPE/gating, not keyboard selection. CPU table transfers, mask compaction, camera projections and history refresh happen before capture; each graph retains its own packed-attention and camera buffers. Cache-write forwards stay eager. Run `nomas/tests/glue_check.py` and `nomas/tests/graph_check.py --camera --intra --int8-history` on the CUDA deployment before enabling this opt-in path.

The camera shortcut also hooks `_blockwise_cross_attn` directly, so upstream-forward calls with singleton ranges do not reach Flex's CPU mask conversion; noncontiguous video inputs are supported by the fused outer shortcut. Graph checks compare eager and captured FA with the same non-balanced reduction mode. FP8 modulation checks bound errors to one E4M3 ULP and 2% global relative L2 instead of applying a fixed absolute tolerance to differently sized FP8 bins.

Fused LayerNorm modulation preserves CUDA autocast's FP32 LayerNorm/output policy even for BF16 inputs, while retaining BF16 rounding of `1 + scale`. Without autocast it preserves the original low-precision intermediate rounding. The modulation check covers both modes and verifies output dtype parity.

## Cache writes and step count

The measured ~3.6 s is a full extra DiT forward to compute history KV, not just a memcpy (`causal_ar.py:2545-2648`). The default `cache_noise_sigma=0.206897` re-noises the finalized latent and uses a matching timestep. Last Euler-step Q/K/V were computed from the **pre-update** latent at a different timestep, so reusing them would change the model's conditioning; fusing the cache write with the last Euler forward is not exact. Keep this pass. The final offline segment already skips it (`causal_ar.py:2177-2178`), so there is no second final cache write to remove. Kestrel attention and GEMM can accelerate this extra forward too. `cache_at_step` also stores a post-scheduler latent and still executes a forward; it is not a free exact reuse path.

Do not change `cache_noise_sigma`, `cache_at_step`, sparse keep fraction, camera conditioning, or masking to obtain a claimed exact pipeline speedup. Six Euler steps (`inference.infer_steps_video=6`) is a separate quality experiment: approximately 25% less denoising work, ~4.7 s at the supplied 18.8 s two-segment denoise baseline, with cache/decode/text costs unchanged. Acceptability is unknown; default stays eight. Score matched seeds for character/identity, action ordering, temporal consistency, and segment transition artifacts before adopting six.

## Memory feasibility

The ~9 GB INT8 DiT + ~9 GB FP8 decoder estimate is ~16.8 GiB combined, leaving ~15 GiB of a nominal 32-GiB device before KV/activations/allocator/driver workspaces. Actual model storage must be measured; this is not a fit guarantee. Video-only 18-frame INT8 K+V is `54 * 2 * 7020 * 2048` bytes, ~1.45 GiB, before scales, stored text, archive working buffers, masks and dequantization. For two segments, the configured 1280-token text capacity can add up to ~0.53 GiB to that allocation. Longer suites/interactive histories can store more text; do not extrapolate this two-segment estimate indefinitely. The CPU archive from `temporal_context` is separate from the bounded GPU history. Concurrent decoding increases the activation peak beyond serial execution.

`NOMAS_PIPE` logs allocated/reserved GiB and synchronized wall time for text H2D, all-text encoding, encoder release, resident-model H2D, and each prompt. Use upstream KV storage audit and one serial `profile_memory=true` run for complete storage/activation evidence. Qwen is deleted **before** either resident model transfers. No allocation-based automatic fallback changes the requested policy; a resident activation OOM remains visible.

## Time budget and expected savings

User-reported baseline is one 10-s clip, two 5-s segments at 480x832. Targets below are **unmeasured engineering goals**, not benchmark results; overlap, kernel, and residency savings are not additive.
Subsequent **parent-reported pod measurements**, distinct from the original approximate baseline: 88.8 s baseline → 54.3 s with resident encode-once pipe → 42.3 s with pipe + Kestrel GEMM/FA, with ~31 s inference per 10-s clip. At four steps, segment-2 denoise was 3.62 s and decoder wall ~12.4 s. These are reported observations, not checks executed by this module's author; four-step quality is not established and eight remains the default. Async overlap has not yet been measured successfully after the anchor repair.


| Stage | Current measured wall time | Resident-pipe target | Reason / expected saving |
| --- | ---: | ---: | --- |
| Per-item offload/text/host overhead | ~50 s | 2–5 s warm item | Remove repeated model transfers and text inference; ~45–48 s if those dominate this bucket. Distinct suite text is paid once at startup. |
| Segment 1 denoise, 8 steps | ~8.3 s | ~8.3 s pipe-only | No math change; Kestrel has a separate target. |
| Segment 1 KV-write forward | ~3.6 s | ~3.6 s pipe-only | Semantically necessary; accelerate underlying kernels instead of skipping. |
| Segment 2 denoise, 8 steps | ~10.5 s | ~10.5 s pipe-only | Longer history; no final offline cache pass. |
| PixelUMM decode | ~30 s | 10–20 s serial resident goal | Remove decoder transfer/host gaps; unchanged FP8 math. ~10–20 s possible, attribution requires profiling. |
| End-to-end | ~100 s | ~35–50 s serial resident goal | Cold model load/text preparation excluded from warm figure and reported separately. |
| Async decode overlap | Not enabled | 0–10 s additional goal | Hidden decode work bounded by segment-2 execution and actual concurrent GPU headroom; may regress. |

The supplied Nsight baseline has only ~31.5 s GPU kernel time in ~100 s wall: flex ~10.28 s, DiT BF16 GEMMs ~6.9 s, decoder CUTLASS GEMMs ~6.78 s, elementwise/copies ~4 s. Thus ~70 s is non-kernel time, but **not all of it is removable offload**: host scheduling, compilation, output conversion, I/O, and transfer time must be attributed separately. In particular the decoder's 30-s wall cannot be equated with its 6.78-s GEMM time alone.

Other low-risk configuration wins: disable latent `.npy` export (`save_latents=false`, upstream defaults true) and optional camera overlay (`overlay_keys=false`) when neither output is required; preserve the video itself and prompt sidecars. Do not globally force FP32 reductions to BF16: RMSNorm, RoPE and cache-noise casts have numerical roles. Dense FA removes sparse router/mask work only on the attention module's supported mask-free shapes; mask-aware paths must preserve semantics. No speculative fp32-cast or mask deletion is included in `pipe.py`.
