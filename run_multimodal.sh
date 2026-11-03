#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash run_multimodal.sh {prepare|audio|video|validate|export|preflight} [CLI arguments]' >&2
  exit 2
fi
stage="$1"
shift
case "$stage" in
  video) python_bin="${PSYMM_VIDEO_PYTHON:-python}" ;;
  prepare|audio|validate|export|preflight) python_bin="${PSYMM_AUDIO_PYTHON:-python}" ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
export PSYMM_ROOT="$ROOT"
export PSYMM_WEIGHTS="${PSYMM_WEIGHTS:-$ROOT/weights}"
export PSYMM_DATA="${PSYMM_DATA:-$ROOT/data}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export NLTK_DATA="${NLTK_DATA:-$ROOT/.cache/nltk}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$python_bin" "$ROOT/generate_multimodal.py" "$stage" "$@"
