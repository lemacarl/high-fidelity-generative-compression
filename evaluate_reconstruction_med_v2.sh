#!/usr/bin/env bash
# evaluate_reconstruction_med_v2.sh
#
# GPU-side reconstruction-quality evaluation. Runs `python -m tflite.evaluate`
# (restores the TF checkpoint directly, no TFLite conversion) for
# test-image-N (N=1..20) from assets/coffee/test-new/ against the
# med_v2 checkpoint: experiments/tflite_med_v2/final-1500000
#
# Reports PSNR / MS-SSIM / LPIPS and BPP. Side-by-side original|reconstruction
# images are saved under OUT_DIR.
#
# Usage:
#   bash evaluate_reconstruction_med_v2.sh
#   bash evaluate_reconstruction_med_v2.sh --csv my_results.csv   (override output path)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurable paths ──────────────────────────────────────────────────────
VARIANT="med_v2"
CKPT="experiments/tflite_med_v2/final-1500000"
DENSITY_WEIGHTS="experiments/tflite_med_v2/density_weights.npz"
IMG_DIR="assets/coffee/paper"
OUT_DIR="data/reconstructions/eval/${VARIANT}"
CSV_OUT="evaluate_results_${VARIANT}.csv"
NUM_IMAGES=10
# ────────────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv) CSV_OUT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "${CKPT}.index" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi

if [[ ! -f "$DENSITY_WEIGHTS" ]]; then
    echo "ERROR: density weights not found: $DENSITY_WEIGHTS" >&2
    echo "Without them BPP is computed against a random prior." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

# One invocation for all images: the model is built and restored once rather
# than once per image, and metrics come back as CSV instead of being scraped
# out of stdout.
IMAGES=()
for N in $(seq 1 "$NUM_IMAGES"); do
    IMAGES+=("${IMG_DIR}/test_image_${N}.jpg")
done

echo "=== Reconstruction evaluation started at $(date) ==="
echo "Checkpoint: $CKPT"
echo "Results   → $CSV_OUT"
echo ""

python -m tflite.evaluate \
    --checkpoint       "$CKPT" \
    --density_weights  "$DENSITY_WEIGHTS" \
    --images           "${IMAGES[@]}" \
    --out_dir          "$OUT_DIR" \
    --label            "$VARIANT" \
    --csv              "$CSV_OUT"

echo ""
echo "=== Evaluation complete at $(date) ==="
echo "CSV written to: $CSV_OUT"
