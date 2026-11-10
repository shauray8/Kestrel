#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export MODAL_PROFILE=shauray8
for tag in "$@"; do
  for v in $(modal volume ls kestrel-samples "$tag/videos" 2>/dev/null | grep -v '\.'); do
    mkdir -p "videos/$v"
    for f in $(modal volume ls kestrel-samples "$v" 2>/dev/null | grep '\.mp4$'); do
      [ -s "videos/$f" ] || modal volume get kestrel-samples "$f" "videos/$f" --force >/dev/null
    done
  done
  for f in $(modal volume ls kestrel-samples "$tag" 2>/dev/null | grep -E '\.(log|json|jsonl|md)$'); do
    mkdir -p "videos/$tag"; modal volume get kestrel-samples "$f" "videos/$f" --force >/dev/null
  done
done
find videos -name '*.mp4' | sort
