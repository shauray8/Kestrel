#!/bin/bash
set -ex
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=${KESTREL_WORKDIR:-/workspace}
V=$W/venvs/kestrel/bin
mkdir -p $W/models $W/logs $W/venvs
python3 -m venv --system-site-packages $W/venvs/kestrel
$V/pip install -q "huggingface_hub[hf_transfer]" ninja
export HF_HUB_ENABLE_HF_TRANSFER=1
git clone -q --depth 1 https://github.com/robbyant/lingbot-world-v2 $W/lingbot-world-v2 &
git clone -q --depth 1 https://github.com/madebyollin/taehv.git $W/taehv &
git clone -q --depth 1 https://github.com/thu-ml/SageAttention.git $W/SageAttention &
$V/hf download robbyant/lingbot-world-v2-1.3b-causal-fast --local-dir $W/models/lingbot13b > $W/logs/dl_dit.log 2>&1 &
( $V/hf download Wan-AI/Wan2.1-T2V-1.3B models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth --local-dir $W/models/lingbot_assets &&
  $V/hf download Wan-AI/Wan2.1-T2V-1.3B --include "google/*" --local-dir $W/models/lingbot_assets ) > $W/logs/dl_assets.log 2>&1 &
wait
test -s $W/models/lingbot_assets/models_t5_umt5-xxl-enc-bf16.pth
test -s $W/models/lingbot_assets/Wan2.1_VAE.pth
mkdir -p $W/models/lingbot13b/transformers
mv $W/models/lingbot13b/model-0000* $W/models/lingbot13b/model.safetensors.index.json $W/models/lingbot13b/transformers/
cd $W/lingbot-world-v2
git apply $ROOT/lingbot/patches/lingbot_sdpa_timing.diff
git apply $ROOT/lingbot/patches/model_fast_ebcast.diff
grep -v -E "^(torch|torchvision|torchaudio|flash_attn|flash-attn)" requirements.txt > /tmp/req.txt
$V/pip install -q -r /tmp/req.txt einops
$V/pip uninstall -y -q torch torchvision torchaudio || true
$V/pip install -q -e $ROOT --no-deps
cd $W/SageAttention && PATH=$V:/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST="12.0a" MAX_JOBS=32 EXT_PARALLEL=4 \
  NVCC_APPEND_FLAGS=--threads=8 $V/pip install --no-build-isolation -q .
cd $W && $V/python -c "import torch, sageattention, kestrel; print(torch.__version__)"
