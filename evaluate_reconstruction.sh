#!/usr/bin/env bash
# run_evaluate_benchmark.sh
#
# Runs `python -m tflite.evaluate` for test_image_N (N=1..10) on both:
#   hi: experiments/tflite_high_v2/ckpt-1510000   (.jpg images)
#   lo: experiments/tflite_low_v2/final-2000000   (.png images)
#
# Parses PSNR, MS-SSIM, and BPP from stdout and writes them to a CSV.
# Output: evaluate_results.csv
#
# Usage:
#   bash run_evaluate_benchmark.sh
#   bash run_evaluate_benchmark.sh --csv my_results.csv   (override output path)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurable paths ──────────────────────────────────────────────────────
CSV_OUT="reconstruction_benchmark_results.csv"
OUT_DIR="data/reconstructions/eval"

CKPT_HI="experiments/tflite_high_v2/ckpt-1510000"
CKPT_LO="experiments/tflite_low_v2/final-2000000"
CKPT_MED="experiments/tflite_med_v2/final-1500000"

IMG_DIR="assets/coffee/paper"
# ────────────────────────────────────────────────────────────────────────────

# Simple --csv override
while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv) CSV_OUT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

# CSV header
printf "model_variant,image_n,image_path,checkpoint,psnr_db,ms_ssim,bpp,status\n" > "$CSV_OUT"

echo "=== Evaluation benchmark started at $(date) ==="
echo "Results → $CSV_OUT"
echo ""

# ── Helper: run evaluate for one image and parse metrics ────────────────────
# Usage: run_one <variant> <N> <img_path> <ckpt>
#
# Reconstructed images are saved to a variant-specific subdirectory:
#   hi → $OUT_DIR/hi/test_image_N_recon.png
#   lo → $OUT_DIR/lo/test_image_N_recon.png
#   med → $OUT_DIR/med/test_image_N_recon.png

run_one() {
    local variant="$1"
    local n="$2"
    local img_path="$3"
    local ckpt="$4"

    # Each variant gets its own output subdirectory so reconstructions
    # are not overwritten when both hi and lo process the same image name.
    local variant_out_dir="${OUT_DIR}/${variant}"

    printf "  [%-2s N=%02d] %s " "$variant" "$n" "$img_path"

    if [[ ! -f "$img_path" ]]; then
        echo "SKIP (file not found)"
        printf "%s,%s,%s,%s,,,,SKIPPED_NO_INPUT\n" \
            "$variant" "$n" "$img_path" "$ckpt" >> "$CSV_OUT"
        return
    fi

    # Capture stdout+stderr; keep stderr visible on terminal too
    local tmpout
    tmpout=$(mktemp)

    local status="OK"
    if ! python -m tflite.evaluate \
            --checkpoint "$ckpt" \
            --images     "$img_path" \
            --out_dir    "$variant_out_dir" \
            2>&1 | tee "$tmpout"; then
        status="ERROR"
    fi

    # Parse metrics from lines like:
    #   BPP:     0.1234
    #   PSNR:    28.45 dB
    #   MS-SSIM: 0.9876
    local bpp psnr ms_ssim
    bpp=$(grep -oP '(?<=BPP:\s{5})\S+' "$tmpout" | tail -1 || true)
    psnr=$(grep -oP '(?<=PSNR:\s{4})\S+' "$tmpout" | tail -1 || true)
    ms_ssim=$(grep -oP '(?<=MS-SSIM: )\S+' "$tmpout" | tail -1 || true)

    rm -f "$tmpout"

    if [[ -z "$psnr" && "$status" == "OK" ]]; then
        status="PARSE_ERROR"
    fi

    if [[ "$status" == "OK" ]]; then
        echo "PSNR=${psnr} dB  MS-SSIM=${ms_ssim}  BPP=${bpp}"
    else
        echo "$status"
    fi

    printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$variant" "$n" "$img_path" "$ckpt" \
        "${psnr:-}" "${ms_ssim:-}" "${bpp:-}" "$status" \
        >> "$CSV_OUT"
}

# ── hi variant (.jpg) ────────────────────────────────────────────────────────
echo "--- hi  checkpoint: $CKPT_HI ---"
for N in $(seq 1 10); do
    run_one "hi" "$N" "${IMG_DIR}/test_image_${N}.jpg" "$CKPT_HI"
done
echo ""

# ── lo variant (.jpg) ────────────────────────────────────────────────────────
echo "--- lo  checkpoint: $CKPT_LO ---"
for N in $(seq 1 10); do
    run_one "lo" "$N" "${IMG_DIR}/test_image_${N}.jpg" "$CKPT_LO"
done
echo ""

# ── med variant (.jpg) ────────────────────────────────────────────────────────
echo "--- med  checkpoint: $CKPT_MED ---"
for N in $(seq 1 10); do
    run_one "med" "$N" "${IMG_DIR}/test_image_${N}.jpg" "$CKPT_MED"
done
echo ""

echo "=== Benchmark complete at $(date) ==="
echo "CSV written to: $CSV_OUT"
