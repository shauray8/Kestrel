#!/bin/bash
set -uo pipefail
KESTREL=$(cd "$(dirname "$0")/.." && pwd)
WS=${WORKSPACE:-/workspace}
export OMP_NUM_THREADS=8 MAX_JOBS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$WS/venvs/kestrel/bin/activate"
cd "$WS/lingbot-world-v2"
NAME=${1:-drift_multiseed_$(date -u +%Y%m%d)}
ROOT="$WS/runs/$NAME"
SAMPLES="$WS/samples/$NAME"
mkdir -p "$ROOT/logs" "$SAMPLES"
run_one() {
  local name=$1 seed=$2
  shift 2
  local out="$SAMPLES/${name}_s${seed}"
  if find "$out" -maxdepth 1 -type f -name '*.mp4' ! -name '*_taelive.mp4' 2>/dev/null | grep -q .; then
    echo "DRIFT_SKIP $name $seed existing_sample"
    return 0
  fi
  mkdir -p "$out"
  echo "DRIFT_START $name $seed $(date -u +%FT%TZ)"
  env KESTREL_OUT="$out" CHUNK=2 FRAMES=481 SEED="$seed" TAE_LIVE=0 LB_TIMING=1 CALIB_CACHE="$WS/calib_multiseed" "$@" \
    python "$KESTREL/lingbot/run.py" > "$ROOT/logs/${name}_s${seed}.log" 2>&1
  local rc=$?
  echo "DRIFT_END $name $seed rc=$rc $(date -u +%FT%TZ)"
  return "$rc"
}
failed=0
for seed in ${SEEDS:-42 1 2}; do
  run_one stock "$seed" FREE_BF16_W=0 ONLY=bf16 || failed=1
  run_one fp8 "$seed" FREE_BF16_W=0 ONLY=bf16_calib,n1 || failed=1
  run_one sage "$seed" FREE_BF16_W=0 ATTN=sage2 ONLY=n2 || failed=1
  run_one full "$seed" FREE_BF16_W=0 FUSED_GLUE=1 KESTREL_E_BCAST=1 CAM_HOIST=1 ATTN=sage2 FUSED_FFN=1 ROPE32=1 ROPE_KV=1 XATTN=1 ONLY=n2_calib_rope32,n1n2_rope32 || failed=1
done
mapfile -t videos < <(find "$SAMPLES" -type f -name '*.mp4' ! -name '*_taelive.mp4' ! -name '*calib*' | sort)
if ((${#videos[@]})); then
  python "$KESTREL/eval/drift.py" "${videos[@]}" --prompt-file examples/05/prompt.txt > "$ROOT/drift.jsonl" 2> "$ROOT/drift_eval.log"
  eval_rc=$?
else
  eval_rc=1
fi
printf 'DRIFT_EVAL_RC=%s videos=%s failed=%s\n' "$eval_rc" "${#videos[@]}" "$failed"
exit $((failed || eval_rc))
