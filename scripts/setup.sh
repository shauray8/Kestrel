#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=${KESTREL_WORKDIR:-/workspace}
V="$W/venvs/kestrel/bin"
export OMP_NUM_THREADS=8
export MAX_JOBS=8

clone_once() {
  if [ -e "$2" ]; then
    test -d "$2/.git"
  else
    git clone -q --depth 1 "$1" "$2"
  fi
}

apply_once() {
  if git -C "$1" apply --reverse --check "$2" >/dev/null 2>&1; then
    return
  fi
  git -C "$1" apply --check "$2"
  git -C "$1" apply "$2"
}

mkdir -p "$W/models" "$W/logs" "$W/venvs"
python3 -m venv --system-site-packages "$W/venvs/kestrel"
"$V/pip" install -q "huggingface_hub[hf_transfer]" ninja
export HF_HUB_ENABLE_HF_TRANSFER=1
pids=()
clone_once https://github.com/robbyant/lingbot-world-v2 "$W/lingbot-world-v2" & pids+=("$!")
clone_once https://github.com/madebyollin/taehv.git "$W/taehv" & pids+=("$!")
clone_once https://github.com/thu-ml/SageAttention.git "$W/SageAttention" & pids+=("$!")
if [ ! -s "$W/models/lingbot13b/transformers/model.safetensors.index.json" ]; then
  "$V/hf" download robbyant/lingbot-world-v2-1.3b-causal-fast --local-dir "$W/models/lingbot13b" > "$W/logs/dl_dit.log" 2>&1 & pids+=("$!")
fi
if [ ! -s "$W/models/lingbot_assets/models_t5_umt5-xxl-enc-bf16.pth" ] || [ ! -s "$W/models/lingbot_assets/Wan2.1_VAE.pth" ]; then
  ( "$V/hf" download Wan-AI/Wan2.1-T2V-1.3B models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth --local-dir "$W/models/lingbot_assets" &&
    "$V/hf" download Wan-AI/Wan2.1-T2V-1.3B --include "google/*" --local-dir "$W/models/lingbot_assets" ) > "$W/logs/dl_assets.log" 2>&1 & pids+=("$!")
fi
for pid in "${pids[@]}"; do
  wait "$pid"
done
test -s "$W/models/lingbot_assets/models_t5_umt5-xxl-enc-bf16.pth"
test -s "$W/models/lingbot_assets/Wan2.1_VAE.pth"
mkdir -p "$W/models/lingbot13b/transformers"
shopt -s nullglob
shards=("$W/models/lingbot13b"/model-0000*)
if ((${#shards[@]})); then
  mv "${shards[@]}" "$W/models/lingbot13b/model.safetensors.index.json" "$W/models/lingbot13b/transformers/"
fi
apply_once "$W/lingbot-world-v2" "$ROOT/lingbot/patches/lingbot_sdpa_timing.diff"
apply_once "$W/lingbot-world-v2" "$ROOT/lingbot/patches/model_fast_ebcast.diff"
req=$(mktemp)
trap 'rm -f "$req"' EXIT
grep -v -E "^(torch|torchvision|torchaudio|flash_attn|flash-attn)" "$W/lingbot-world-v2/requirements.txt" > "$req"
"$V/pip" install -q -r "$req" einops
"$V/pip" uninstall -y -q torch torchvision torchaudio || true
"$V/pip" install -q -e "$ROOT" --no-deps
"$V/pip" install -q -r "$ROOT/eval/requirements.txt"
apply_once "$W/SageAttention" "$ROOT/lingbot/patches/sage_stream.diff"
PATH="$V:/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST="12.0a" EXT_PARALLEL=2 NVCC_APPEND_FLAGS=--threads=8 \
  "$V/pip" install --no-build-isolation -q "$W/SageAttention"
cd "$W"
"$V/python" -c "import torch, sageattention, kestrel; print(torch.__version__)"
