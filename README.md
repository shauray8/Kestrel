# Kestrel
good world models

## VM block execution

`VM_BLOCK=1` requires `ATTN=n2 FUSED_GLUE=1 FUSED_FFN=1 ROPE_KV=1 VM_ATTN=1`.
The forward wrapper derives one cache transition for all blocks, using Python
indices attached to the pipeline's zero-initialized caches. The merged programs
write the same `global_end_index`, `local_end_index`, and cross-attention
`is_init` tensors as upstream. Unmanaged/prepopulated caches are imported by
reading block zero once at the forward boundary; the zero-sync contract applies
to caches allocated through the installed pipeline, not externally mutated
index tensors.

Warmup prebuilds the forward's complete chunk-position/eviction/overwrite
schedule after calibration, including the calibration-cache-hit path. Program
tables rebind activation, RoPE, KV, cross-KV, and scalar-index pointers each
forward. Identically shaped blocks must share the local-attention and sink
configuration. A block without camera injection is one program; camera
injection splits it into two programs around the unchanged PyTorch camera
operations. Modulation addition and cache-index commits execute inside the VM.
Affine cross-attention `norm3` uses the same `MOD_NORM` path as the block
implementation: shift is its bias, scale is weight minus one, and the BF16
normalized result feeds FP8 quantization. Identity normalization stays unchanged.
There are no runtime `status()` checks or per-block device-to-host reads on this
path. Embeddings, the output head, diffusion/noise scheduling, and VAE remain
outside the VM. Each VM launch still consists of reset plus persistent kernels.

`Program.compose` relocates tensor/instruction indices, deduplicates identical
tensor descriptors, and adds completion barriers between consecutive programs.
`MAXT=96` makes the read-only `__grid_constant__` table 3,840 bytes; the persistent
kernel's remaining arguments bring it to 3,912 bytes, below even CUDA's original
4,096-byte parameter limit. CUDA 12.1+ supports 32,764 bytes on Volta and newer,
but this implementation does not require the larger limit
([NVIDIA parameter-limit reference](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/)).
The RTX 5090 has 170 SMs
([NVIDIA RTX Blackwell architecture, Appendix A](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)).
Compute capability 12.0 has 128 KB shared memory per SM and 99 KB per block
([NVIDIA tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#occupancy)).
The VM retains its 48 KiB dynamic shared-memory allocation, occupancy-derived
resident grid, and `mma.sync` implementation; no TMEM or `tcgen05` is used.

### Cross-chunk doorbell contract

No spin flag is enabled. All programs and camera/scheduler operations currently
use the current CUDA stream, which already orders a chunk's final KV-update
forward before the next chunk. The next chunk's first program cannot safely be
enqueued ahead of the host-controlled diffusion/noise/input preparation and then
wait for a producer queued behind it on that same stream.

A device doorbell is correct only under all of these conditions:

- One request owns the cache, program state, shared scratch, and a monotonically
  increasing generation counter. Resetting/reusing them requires completion of
  every previous reader; a boolean flag is insufficient across chunk reuse.
- The producer publishes generation `c + 1` with device-scope release semantics
  only after the final KV-update forward of chunk `c`, all 30 blocks' cache
  writes/index commits, and every next-chunk input/conditioning write completes.
  Publishing after the fourth denoise forward alone is incorrect.
- Every consumer CTA acquires the expected generation before loading any of
  those tensors. A single CTA polling without a grid-wide handoff is
  insufficient; timeout/cancellation must propagate to every waiter.
- The producer is independently schedulable. It cannot sit behind the waiter
  on the same stream, depend on the waiter, or require CTAs/resources monopolized
  by the persistent waiting grid. A same-stream event/order dependency is
  preferable until an independent producer and a bounded occupancy reservation
  exist.
- Fresh host tensors stay alive until their last device reader completes, and
  submissions cannot overlap the shared program reset/state or scratch storage.

These prerequisites are not supplied by the current host diffusion loop, so
adding a flag alone would not establish a valid cross-chunk execution protocol.

### Verification commands

CPU index tests mirror the pinned
[self-attention transition](https://github.com/Robbyant/lingbot-world-v2/blob/1895d300d8ac936401689b26389f51cbd36530eb/wan/modules/model_fast.py#L119-L162)
and [four-denoise-plus-KV-update chunk schedule](https://github.com/Robbyant/lingbot-world-v2/blob/1895d300d8ac936401689b26389f51cbd36530eb/wan/image2video.py#L779-L839):

```sh
PYTHONPATH=. python -m unittest discover -s tests -p test_vm_schedule.py
```

GPU parity tests compare two consecutive merged blocks against the separated
programs, including broadcast/per-token modulation, camera boundaries, cache
initialization, rolling eviction, pointer rebinding, and both Identity and affine
LayerNorm cross-attention inputs. GPU runs belong to the
parent; no measured parity or speedup is implied here.

```sh
bash /workspace/kestrel/scripts/gpujob.sh vm-multiblock-tests 'cd /workspace/wt/vm-multiblock && PYTHONPATH=/workspace/wt/vm-multiblock:/workspace/wt/vm-multiblock/lingbot TORCH_EXTENSIONS_DIR=/workspace/ext/vm-multiblock /workspace/venvs/kestrel/bin/python tests/test_vm.py && PYTHONPATH=/workspace/wt/vm-multiblock:/workspace/wt/vm-multiblock/lingbot TORCH_EXTENSIONS_DIR=/workspace/ext/vm-multiblock /workspace/venvs/kestrel/bin/python tests/test_vm_multiblock.py'

for frames in 9 97; do
  for variant in baseline merged; do
    vm=0
    test "$variant" != merged || vm=1
    bash /workspace/kestrel/scripts/gpujob.sh "vm-multiblock-${variant}-${frames}" "cd /workspace/lingbot-world-v2 && env PYTHONPATH=/workspace/wt/vm-multiblock:/workspace/wt/vm-multiblock/lingbot TORCH_EXTENSIONS_DIR=/workspace/ext/vm-multiblock ATTN=n2 CHUNK=2 FRAMES=$frames SEED=42 STEPS_IDX=0,179,358,679 WINDOW=18 SINK=6 FUSED_GLUE=1 FUSED_FFN=1 FP8_GLUE=1 KESTREL_E_BCAST=1 CAM_HOIST=1 ROPE32=1 ROPE_KV=1 VM_ATTN=$vm VM_BLOCK=$vm ONLY=n2_calib_rope32,n1n2_rope32 KESTREL_OUT=/workspace/samples/vm-multiblock-${variant}-${frames} /workspace/venvs/kestrel/bin/python /workspace/wt/vm-multiblock/lingbot/run.py"
  done
  bash /workspace/kestrel/scripts/gpujob.sh "vm-multiblock-quality-${frames}" "cd /workspace/wt/vm-multiblock && md5sum /workspace/samples/vm-multiblock-{baseline,merged}-${frames}/05_n1n2_rope32_c2.mp4 && /workspace/venvs/kestrel/bin/python eval/quality.py --ref /workspace/samples/vm-multiblock-baseline-${frames}/05_n1n2_rope32_c2.mp4 /workspace/samples/vm-multiblock-merged-${frames}/05_n1n2_rope32_c2.mp4"
done
```

The wrapper owns logs under `/workspace/logs`; generated outputs are confined
to the named `/workspace/samples` directories. Compare both the MP4 hashes and
reported decoded-frame LPIPS; hash equality is not a substitute for a quality
report when the encodings differ.
