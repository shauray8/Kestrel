import os
import sys

import torch

HYNOMAS = os.environ.get("HYNOMAS", "/workspace/Hy-Nomas")
MODELS = os.environ.get("NOMAS_MODELS", "/workspace/models/nomas")
DCAE_CKPT = f"{MODELS}/VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt"
PIXELUMM_CKPT = f"{MODELS}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors"
if HYNOMAS not in sys.path:
    sys.path.insert(0, HYNOMAS)


def load_dcae(device="cuda", dtype=torch.bfloat16):
    from hymm.models.vae.autoencoder_dcae_v import AutoencoderDCAEV
    return AutoencoderDCAEV.from_checkpoint(DCAE_CKPT, dtype=dtype, device=device).eval().requires_grad_(False)


def load_pixelumm(device="cuda", dtype=torch.bfloat16, overlap=2, window=10, tile=128, seed=42):
    from hymm.models.vae.pixelumm_decoder.autoencoder import DMD2_DENOISING_STEPS, PixelUMMDCAELatentDecoder
    m = PixelUMMDCAELatentDecoder.from_checkpoint(
        PIXELUMM_CKPT, dtype=dtype, device=device, local_attention_tile_pixels=tile,
        denoising_step_list=list(DMD2_DENOISING_STEPS), num_steps=1, seed=seed,
        temporal_window_latents=window, temporal_overlap_latents=overlap, decode_output_cpu=False)
    return m.eval().requires_grad_(False)


def sample(out):
    return out.sample if hasattr(out, "sample") else out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def dcae_encode(vae, video):
    return vae.encode(video).latent_dist.mode()


@torch.no_grad()
def dcae_decode(vae, z):
    return sample(vae.decode(z))


@torch.no_grad()
def pixelumm_decode(dec, z):
    return sample(dec.decode(z))
