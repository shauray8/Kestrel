#!/bin/bash
# usage: sample_batch.sh <config> <batch.json>
set -uo pipefail
W=${KESTREL_WORKDIR:-/workspace}
cfg=$1 batch=$2
STACK="FUSED_GLUE=1 KESTREL_E_BCAST=1 CAM_HOIST=1 ATTN=fa FUSED_FFN=1 ROPE32=1 ROPE_KV=1 XATTN=1 N3=1 FA_LOAD_MODE=3 FA_EXP_MODE=0 FA_Q_TILE=128 FA_RESCALE_THRESHOLD=0.5 RMS_ROPE=1 GLUE2=1 FA_MX=1 FA_BAL=1 N3_CONT=1"
FA="ATTN=fa FA_LOAD_MODE=3 FA_EXP_MODE=0 FA_Q_TILE=128 FA_RESCALE_THRESHOLD=0.5 FA_MX=1 FA_BAL=1"
case $cfg in
  bf16) E="ONLY=bf16 NO_N1=1" ;;
  sage2) E="ATTN=sage2 ONLY=n2" ;;
  kernel) E="$FA ONLY=n2" ;;
  int8) E="$STACK ONLY=n2_calib_rope32,n1n2_rope32 CALIB_CACHE=$W/calib" ;;
  *) echo "bad config $cfg"; exit 2 ;;
esac
cd $W/lingbot-world-v2
env $E CHUNK=${CHUNK:-2} TAE_LIVE=${TAE_LIVE:-1} LB_TIMING=1 MAX_AREA=${MAX_AREA:-399424} CONFIG=$cfg BATCH=$batch \
  SAMPLE_DIR=${SAMPLE_DIR:-$W/samples/lingbot_v2} EXAMPLE=examples/05 IMAGE=$W/scenes/_calib.png FRAMES=97 SEED=42 \
  KESTREL_OUT=$W/scratch_out GIT_HASH=${GIT_HASH:-unknown} GIT_DIRTY=${GIT_DIRTY:-unknown} MODEL=${MODEL:-1.3b} \
  $W/venvs/kestrel/bin/python $W/kestrel/lingbot/run.py
