#!/usr/bin/env bash
# train_gan_v8.sh
#
# Phase 2 GAN fine-tuning, warm-started from the tflite_low_v2 phase-1
# checkpoint. First run with a discriminator that is actually alive: v4-v7 and
# high_gan all collapsed the tower to constant output within ~1000 updates and
# trained as compression-only from then on without reporting it.
#
# Runs the hinge objective (BCE cannot saturate under spectral norm), the
# normalised latent context path, two critic updates per generator update, and
# the collapse detector armed at 10 intervals. --disc_debug prints `sep` and
# `acc` every interval; if `sep` sits at ~0 the run is not adversarial and the
# detector will abort rather than let it finish as a mislabelled model.
#
# Usage:
#   nohup bash train_gan_v8.sh > train_v8.log 2>&1 &
#   tail -f train_v8.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurable ────────────────────────────────────────────────────────────
# get_dataset globs {root}/**/* recursively. test-new/ — the 20 images every
# eval script scores — now lives under assets/coffee/ and is safely outside
# this tree. data/coffee/{test,val} remain in the glob, so 430 held-out images
# are still trained on; set DATASET="data/coffee/train" (1799 images) if those
# splits are meant to stay held out.
DATASET="data/coffee/train"

WARMSTART_CKPT="experiments/tflite_low_v2/final-2000000"
PRIOR_WEIGHTS="experiments/tflite_low_v2/density_weights.npz"
OUT_DIR="experiments/tflite_low_gan_v8/"
REGIME="low"
TARGET_BPP=0.16      # v5 used the regime default of 0.14
N_STEPS=300000
GEN_LR=2e-5
DISC_LR=1e-4
# ────────────────────────────────────────────────────────────────────────────

if [[ ! -d "$DATASET" ]]; then
    echo "ERROR: dataset not found: $DATASET" >&2
    exit 1
fi

if [[ ! -f "${WARMSTART_CKPT}.index" ]]; then
    echo "ERROR: warm-start checkpoint not found: ${WARMSTART_CKPT}" >&2
    exit 1
fi

if [[ ! -f "$PRIOR_WEIGHTS" ]]; then
    echo "ERROR: prior weights not found: $PRIOR_WEIGHTS" >&2
    echo "Without them phase 2 starts from a random entropy model." >&2
    exit 1
fi

N_IMAGES=$(find "$DATASET" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.JPG' \) \
    | wc -l)

echo "=== GAN fine-tuning (v8) started at $(date) ==="
echo "  dataset     : $DATASET  (${N_IMAGES} images)"
echo "  warm-start  : $WARMSTART_CKPT"
echo "  prior       : $PRIOR_WEIGHTS"
echo "  output      : $OUT_DIR"
echo "  steps       : $N_STEPS  (~$((N_STEPS / 2)) generator updates)"
echo "  target bpp  : $TARGET_BPP  (v5 used 0.14 -> coded 0.1426)"
echo "  gen/disc lr : $GEN_LR / $DISC_LR"
echo ""
echo "Confirm the startup line reads 'Factorized prior: absent from checkpoint"
echo "— loaded 12 variables'. A WARNING there means phase 2 is starting from a"
echo "random entropy model and will repeat v4's mistake."
echo ""

# -u disables stdout block buffering. Redirected to a file, python
# otherwise holds print() output in a 4-8KB buffer, so train.log stays
# empty for hours while training runs perfectly well underneath.
python -u -m tflite.training.trainer \
    --dataset_path    "$DATASET" \
    --regime          "$REGIME" \
    --target_bpp      "$TARGET_BPP" \
    --model_type      compression_gan \
    --warmstart \
    --checkpoint      "$WARMSTART_CKPT" \
    --prior_weights   "$PRIOR_WEIGHTS" \
    --gen_lr          "$GEN_LR" \
    --disc_lr         "$DISC_LR" \
    --n_steps         "$N_STEPS" \
    --gan_loss_type   hinge \
    --disc_ctx_norm   layer \
    --n_critic        2 \
    --disc_collapse_patience 10 \
    --disc_debug \
    --checkpoint_dir  "$OUT_DIR"

echo ""
echo "=== GAN fine-tuning complete at $(date) ==="
