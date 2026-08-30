"""
Put the two bitrate measurements side by side on byte-identical inputs.

Two different numbers are in circulation for the same model:

  Path A — checkpoint estimate.  `tflite.evaluate` restores the TF checkpoint
           and reports the entropy model's own likelihood, -log2 P(y_hat).
           Nothing is actually coded.

  Path B — deployed pipeline.  The exported *.tflite models encode the image
           and the NumPy ANS coder writes a real .hfc. bpp is measured from the
           file on disk. This is what the Raspberry Pi reports.

They can disagree for two reasons, and this script separates them:

  1. The weights are not the same. export_tflite.py::_sanitize_model_weights
     rewrites BatchNorm moving statistics before conversion — NaN mean/variance
     replaced with 0/1, and variance clipped up to min_variance=1e-3. Any
     channel that tripped those rules encodes differently after export. The
     --audit_bn pass counts exactly how many would have been rewritten.

  2. Estimate vs reality. A real ANS stream cannot be much *shorter* than the
     model's own entropy without losing information, so Path B << Path A points
     at the coder or the exported weights, not at the estimate.

Both paths are fed the same array from inference.compress.load_image, so
preprocessing is identical by construction.

Usage:
    python -m tflite.compare_paths \
        --checkpoint experiments/tflite_low_v2/final-2000000 \
        --models_dir tflite_models_v2_lo \
        --density_weights experiments/tflite_low_v2/density_weights.npz \
        --images assets/coffee/test-new/*.png
"""

import argparse
import contextlib
import io
import os
import tempfile

import numpy as np
import tensorflow as tf

from tflite.inference.compress import (
    load_image,
    load_compress_interpreters,
    compress as pi_compress,
)
from tflite.compression.entropy_models import FactorizedPriorNumpy
from tflite.compression.prior_model import PriorModel
from tflite.model.compression_model import CompressionModel

PIXELS = 256 * 256
MIN_VARIANCE = 1e-3          # must match export_tflite.py


# ---------------------------------------------------------------------------
# BatchNorm audit — what would export_tflite.py rewrite?
# ---------------------------------------------------------------------------

def audit_batchnorm(model, min_variance=MIN_VARIANCE):
    layers_touched = 0
    n_nan_mean = n_nan_var = n_clipped_var = n_channels = 0

    def walk(m):
        nonlocal layers_touched, n_nan_mean, n_nan_var, n_clipped_var, n_channels
        for layer in getattr(m, "layers", []):
            if hasattr(layer, "layers"):
                walk(layer)
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                mm = layer.moving_mean.numpy()
                mv = layer.moving_variance.numpy()
                bad_mean = int(np.sum(~np.isfinite(mm)))
                bad_var = int(np.sum(~np.isfinite(mv)))
                clipped = int(np.sum(np.isfinite(mv) & (mv < min_variance)))
                n_channels += mm.size
                n_nan_mean += bad_mean
                n_nan_var += bad_var
                n_clipped_var += clipped
                if bad_mean or bad_var or clipped:
                    layers_touched += 1

    walk(model.encoder)
    return dict(layers_touched=layers_touched, n_channels=n_channels,
                n_nan_mean=n_nan_mean, n_nan_var=n_nan_var,
                n_clipped_var=n_clipped_var)


def psnr(x, x_hat):
    mse = float(np.mean((x - x_hat) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--models_dir", required=True)
    p.add_argument("--density_weights", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--fp32", action="store_true",
                   help="Force FP32 TFLite models (default mirrors the Pi, "
                        "which prefers *_int8.tflite when present)")
    p.add_argument("--audit_bn", action="store_true", default=True)
    args = p.parse_args()

    paths = [i for i in args.images if os.path.exists(i)]
    if not paths:
        raise SystemExit("No input images found")

    # ---- Identical inputs for both paths ----
    images = [load_image(q)[0] for q in paths]     # each (1,256,256,3) float32

    # ---- Path A: TF checkpoint ----
    print("Path A — restoring TF checkpoint ...")
    model = CompressionModel()
    _ = model(tf.zeros([1, 256, 256, 3]), training=False)
    tf.train.Checkpoint(model=model).restore(args.checkpoint).expect_partial()

    if args.audit_bn:
        a = audit_batchnorm(model)
        print(f"\nBatchNorm audit of {args.checkpoint}")
        print(f"  encoder BN channels total : {a['n_channels']}")
        print(f"  NaN moving_mean           : {a['n_nan_mean']}")
        print(f"  NaN moving_variance       : {a['n_nan_var']}")
        print(f"  variance < {MIN_VARIANCE:g} (clipped) : {a['n_clipped_var']}")
        print(f"  layers export would alter : {a['layers_touched']}")
        if a["n_nan_mean"] or a["n_nan_var"] or a["n_clipped_var"]:
            print("  => exported weights DIFFER from this checkpoint by design")
        else:
            print("  => export leaves BN untouched; weights should match")

    est = []
    for x in images:
        _, hyper_bpp, latent_bpp = model(tf.constant(x), training=False)
        est.append(float(hyper_bpp + latent_bpp))

    # ---- Path B: exported TFLite + real ANS coding ----
    print(f"\nPath B — {args.models_dir} + NumPy ANS ...")
    fp_weights = np.load(args.density_weights, allow_pickle=True)
    factorized_prior = FactorizedPriorNumpy.from_weights(dict(fp_weights))
    prior_model = PriorModel()
    interpreters = load_compress_interpreters(
        args.models_dir, use_int8=not args.fp32
    )

    real = []
    with tempfile.TemporaryDirectory() as tmp:
        for q in paths:
            out = os.path.join(tmp, os.path.basename(q) + ".hfc")
            # compress() prints a per-stage timing report; silence it here.
            with contextlib.redirect_stdout(io.StringIO()):
                pi_compress(None, interpreters, prior_model,
                            factorized_prior, q, out)
            real.append(8.0 * os.path.getsize(out) / PIXELS)

    # ---- Report ----
    print(f"\n{'image':<22}{'A: estimate':>13}{'B: coded':>11}{'B/A':>8}")
    print("-" * 54)
    for q, a_, b_ in zip(paths, est, real):
        print(f"{os.path.basename(q):<22}{a_:>13.4f}{b_:>11.4f}{b_ / a_:>8.2f}")
    ma, mb = float(np.mean(est)), float(np.mean(real))
    print("-" * 54)
    print(f"{'MEAN':<22}{ma:>13.4f}{mb:>11.4f}{mb / ma:>8.2f}")

    print(f"\nMean coded size: {mb * PIXELS / 8:.0f} bytes/image")
    if mb < 0.5 * ma:
        print("Path B is far below the model's own entropy — the exported\n"
              "encoder or the ANS coder is dropping information, not just\n"
              "coding it more efficiently. Compare reconstructions before\n"
              "trusting the lower number.")
    elif abs(mb - ma) / ma < 0.15:
        print("The two paths agree; the 0.14 figure came from somewhere else.")


if __name__ == "__main__":
    main()
