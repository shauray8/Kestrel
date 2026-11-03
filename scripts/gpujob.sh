#!/bin/bash
set -uo pipefail
W=${KESTREL_WORKDIR:-/workspace}
name=$1
shift
mkdir -p "$W/logs/jobs"
log="$W/logs/jobs/$name.log"
{
  echo "JOB_START $name $(date -u +%FT%TZ)"
  flock "$W/gpu.lock" env OMP_NUM_THREADS=8 MAX_JOBS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PATH="$W/venvs/kestrel/bin:/usr/local/cuda/bin:$PATH" bash -c "$*"
  echo "rc=$?"
  echo "JOB_DONE $name $(date -u +%FT%TZ)"
} > "$log" 2>&1
