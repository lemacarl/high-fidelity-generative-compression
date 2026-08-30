#!/usr/bin/env bash
# train_ft_control.sh
#
# ATTRIBUTION CONTROL for tflite_low_gan_v7.
#
# WHAT QUESTION THIS ANSWERS
#   v7 (low) and high_gan both beat their phase-1 baselines by ~6.5-6.9% LPIPS,
#   with PSNR and MS-SSIM improving alongside. Both runs had a DEAD
#   discriminator — D pinned at 2*ln2 = 1.3863 and G at ln2 = 0.6931, the
#   constant-output equilibrium, for essentially all of training. A generator
#   loss that is constant has zero gradient, so the adversarial term
#   contributed nothing, and the gain should be attributable to fine-tuning on
#   in-domain coffee data instead.
#
#   That is an inference from the loss curves, not a measurement. This run
#   measures it: identical setup, identical data, identical rate target, no
#   discriminator at all. If it reproduces v7's numbers, the GAN machinery can
#   be dropped from the pipeline. If it does not, the adversarial term was
#   doing something after all and the dead-discriminator reading is wrong.
#
# WHAT IS HELD IDENTICAL TO train_gan_v7.sh
#   dataset          data/coffee/train
#   warm start       experiments/tflite_low_v2/final-2000000
#   prior weights    experiments/tflite_low_v2/density_weights.npz
#   regime / target  low / target_bpp=0.16
#   learning rate    2e-5   (see LR NOTE below)
#   Adam beta_1      0.5    (see BETA NOTE below)
#   generator updates 100000 (see STEP NOTE below)
#
#   The only remaining difference is `--model_type compression`, which drops
#   `beta * g_loss` from the generator objective. Compare
#   compression_train_step against generator_train_step in
#   tflite/training/trainer.py: they are otherwise line-for-line the same
#   objective, same variable groups, same gradient guards, same clipping.
#
# STEP NOTE — 100000, NOT 200000
#   GAN mode alternates: odd steps train the generator, even steps train the
#   discriminator. v7's 200000 steps were therefore ~100000 generator updates.
#   Compression mode updates the generator every step, so 100000 here matches
#   v7's actual optimizer work. Passing 200000 would double it and confound
#   the comparison in the direction that flatters this run.
#
# LR NOTE — --lr, NOT --gen_lr
#   trainer.py: `gen_lr = args.gen_lr if (is_gan and args.gen_lr is not None)
#   else args.lr`. --gen_lr is ignored outside GAN mode, so v7's 2e-5 has to
#   be passed as --lr. The default 1e-4 would be 5x too high.
#
# BETA NOTE — why --adam_beta_1 exists
#   trainer.py hardcoded beta_1=0.5 for GAN and 0.9 for compression. That would
#   have left an uncontrolled optimizer difference between v7 and this run, so
#   --adam_beta_1 was added as an explicit override. It defaults to the old
#   behaviour, so no existing script changes.
#
# lr_decay_step defaults to 500000, above both step budgets, so neither run
# decays. Nothing to match there.
#
# READING THE RESULT
#   The comparison is against low_v2 (0.4276 LPIPS) and low_gan_v7 (0.4000) on
#   the same 20 images at matched rate. Three outcomes:
#     ~0.400  -> the GAN contributed nothing; the gain is domain fine-tuning
#     ~0.428  -> the adversarial term was load-bearing despite the flat curves
#     between -> partial, and worth a longer look
#
# Usage:
#   nohup bash train_ft_control.sh > train_ft.log 2>&1 &
#   tail -f train_ft.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurable ────────────────────────────────────────────────────────────
DATASET="data/coffee/train"

WARMSTART_CKPT="experiments/tflite_low_v2/final-2000000"
PRIOR_WEIGHTS="experiments/tflite_low_v2/density_weights.npz"
OUT_DIR="experiments/tflite_low_ft/"
REGIME="low"
TARGET_BPP=0.16
N_STEPS=100000       # == v7's 100k generator updates; see STEP NOTE
LR=2e-5              # == v7's --gen_lr;               see LR NOTE
ADAM_BETA_1=0.5      # == v7's GAN-mode beta_1;        see BETA NOTE
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
    echo "Without them the run starts from a random entropy model." >&2
    exit 1
fi

# An existing output dir means a previous run's checkpoints are in here, and
# the trainer would resume rather than warm-start — silently answering a
# different question than the one in the header.
if [[ -e "${OUT_DIR}checkpoint" ]]; then
    echo "ERROR: ${OUT_DIR} already holds checkpoints." >&2
    echo "Move it aside first; this script is meant to warm-start clean." >&2
    exit 1
fi

N_IMAGES=$(find "$DATASET" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.JPG' \) \
    | wc -l)

echo "=== Attribution control (no GAN) started at $(date) ==="
echo "  dataset     : $DATASET  (${N_IMAGES} images)"
echo "  warm-start  : $WARMSTART_CKPT"
echo "  prior       : $PRIOR_WEIGHTS"
echo "  output      : $OUT_DIR"
echo "  steps       : $N_STEPS generator updates (v7: 200000 alternating = ~100000)"
echo "  target bpp  : $TARGET_BPP  (same as v7)"
echo "  lr          : $LR  (v7 gen_lr, passed as --lr outside GAN mode)"
echo "  adam beta_1 : $ADAM_BETA_1  (v7 GAN-mode value, overridden to match)"
echo "  model_type  : compression   <-- the only intended difference from v7"
echo ""
echo "Confirm the startup line reads 'Factorized prior: absent from checkpoint"
echo "— loaded 12 variables'. A WARNING there means this run started from a"
echo "random entropy model and its bitrate is meaningless."
echo ""

# -u disables stdout block buffering. Redirected to a file, python otherwise
# holds print() output in a 4-8KB buffer, so the log stays empty for hours
# while training runs perfectly well underneath.
python -u -m tflite.training.trainer \
    --dataset_path    "$DATASET" \
    --regime          "$REGIME" \
    --target_bpp      "$TARGET_BPP" \
    --model_type      compression \
    --warmstart \
    --checkpoint      "$WARMSTART_CKPT" \
    --prior_weights   "$PRIOR_WEIGHTS" \
    --lr              "$LR" \
    --adam_beta_1     "$ADAM_BETA_1" \
    --n_steps         "$N_STEPS" \
    --checkpoint_dir  "$OUT_DIR"

echo ""
echo "=== Attribution control complete at $(date) ==="
echo "Next:"
echo "  bash evaluate_reconstruction_low_ft.sh"
echo "  python -m tflite.conversion.export_tflite \\"
echo "      --checkpoint ${OUT_DIR}final-${N_STEPS} --out_dir tflite_models_low_ft/"
echo "  bash evaluate_compression_low_ft.sh"
