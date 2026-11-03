#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 3 ]]; then
  echo 'Usage: bash run_generation_pipeline.sh SOURCE RUN_DIR GPU [prepare options]' >&2
  exit 2
fi
source_name="$1"
run_dir="$2"
gpu="$3"
shift 3
voice_tau="${PSYMM_VOICE_TAU:-0.3}"
mkdir -p -- "$run_dir"
run_dir="$(cd -- "$run_dir" && pwd)"
if [[ ! -f "$run_dir/plan.meta.json" ]]; then
  bash "$ROOT/run_multimodal.sh" prepare --source "$source_name" --run "$run_dir" "$@"
fi
bash "$ROOT/run_multimodal.sh" audio --run "$run_dir" --gpu "$gpu" --tau "$voice_tau"
bash "$ROOT/run_multimodal.sh" video --run "$run_dir" --gpu "$gpu" --tau "$voice_tau"
bash "$ROOT/run_multimodal.sh" validate --run "$run_dir"
bash "$ROOT/run_multimodal.sh" export --run "$run_dir"
