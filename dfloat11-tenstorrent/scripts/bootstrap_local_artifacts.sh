#!/usr/bin/env bash
# Recreate outputs/extracted + outputs/artifacts after clone (large files are gitignored).
#
# Default: dummy Llama-3.2-1B-shaped weights (no HuggingFace download; still ~525MB BF16 on disk).
# Real checkpoint:
#   ./scripts/bootstrap_local_artifacts.sh --hf meta-llama/Llama-3.2-1B-Instruct
# Gated models: huggingface-cli login && pip install 'transformers>=4.40'
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

USE_DUMMY=1
MODEL=""
EXTRA=( )

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf)
      USE_DUMMY=0
      MODEL="${2:?missing model id after --hf}"
      shift 2
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

mkdir -p outputs/extracted outputs/artifacts outputs/metrics

if [[ "${USE_DUMMY}" -eq 1 ]]; then
  python scripts/extract_weights.py --dummy --layers lm_head --out-dir outputs/extracted "${EXTRA[@]}"
else
  python scripts/extract_weights.py --model "${MODEL}" --layers lm_head --out-dir outputs/extracted "${EXTRA[@]}"
fi

python scripts/compress_weights.py \
  --extracted-dir outputs/extracted \
  --artifact-dir outputs/artifacts \
  --layers lm_head.weight

echo "Done. Artifacts are under outputs/ (ignored by git). See README Quick Start for benchmarks."
