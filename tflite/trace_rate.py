"""
Locate the 3x gap between the checkpoint's rate estimate and the coded rate.

compare_paths.py established that the deployed pipeline codes ~0.16 bpp while
the checkpoint estimates ~0.52 bpp on the same images, with good
reconstructions — so no information is being lost and the coded length is
legitimate. The estimate is the number that is wrong.

Both paths compute the same discrete Gaussian mass and both derive sigma as
softplus(raw) + MIN_SCALE, so the formulas agree. That leaves the tensors. This
dumps y, mu, sigma and the per-element cost from each path for one image and
reports where they diverge.

Three costs are computed for the TFLite path so the ANS coder can be separated
from the tables it is given:
    exact   -log2 of the continuous Gaussian mass
    table   -log2 of the quantized CDF mass the ANS coder actually uses
    coded   the real .hfc size

If y/mu/sigma agree across paths but the costs differ, the bug is in the rate
computation. If sigma differs, the exported hyper-decoder is the cause.

Usage:
    python -m tflite.trace_rate \
        --checkpoint experiments/tflite_low_v2/final-2000000 \
        --models_dir tflite_models_v2_lo \
        --density_weights experiments/tflite_low_v2/density_weights.npz \
        --image assets/coffee/test-new/test-image-1.png
"""

import argparse
import numpy as np
import tensorflow as tf

from tflite.inference.compress import (
    load_image, load_compress_interpreters, run_interpreter, run_hyper_decoder,
)
from tflite.compression.entropy_models import GaussianPriorNumpy, MIN_SCALE
from tflite.model.compression_model import CompressionModel

PIXELS = 256 * 256
SQRT2 = np.sqrt(2.0)


def exact_bits(symbols, sigma):
    """-log2 of the continuous discrete-Gaussian mass, per element."""
    from scipy.special import erfc
    s = np.asarray(symbols, dtype=np.float64)
    sg = np.maximum(np.asarray(sigma, dtype=np.float64), MIN_SCALE)
    upper = 0.5 * erfc(-((s + 0.5) / sg) / SQRT2)
    lower = 0.5 * erfc(-((s - 0.5) / sg) / SQRT2)
    return -np.log2(np.maximum(upper - lower, 1e-12))


def table_bits(symbols, sigma, precision=16):
    """-log2 of the quantized CDF mass the ANS coder actually charges."""
    gp = GaussianPriorNumpy()
    gp.build_tables()
    idx = gp.compute_indices(np.clip(sigma, MIN_SCALE, None)).ravel()
    sym = np.asarray(symbols, dtype=np.int64).ravel()
    off = gp.CDF_offset.astype(np.int64)[idx]
    length = gp.CDF_length.astype(np.int64)[idx]
    pos = np.clip(sym - off, 0, length - 2)
    cdf = gp.CDF.astype(np.int64)
    mass = cdf[idx, pos + 1] - cdf[idx, pos]
    return -np.log2(np.maximum(mass, 1) / float(2 ** precision))


def describe(name, a):
    a = np.asarray(a, dtype=np.float64)
    print(f"  {name:<10} shape={str(a.shape):<18} "
          f"min={a.min():>9.4f} mean={a.mean():>9.4f} "
          f"max={a.max():>9.4f} std={a.std():>8.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--models_dir", required=True)
    p.add_argument("--density_weights", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--fp32", action="store_true")
    args = p.parse_args()

    x, _ = load_image(args.image)          # (1,256,256,3) float32 — shared input

    # ---------------- Path A: TF checkpoint ----------------
    model = CompressionModel()
    _ = model(tf.zeros([1, 256, 256, 3]), training=False)
    tf.train.Checkpoint(model=model).restore(args.checkpoint).expect_partial()

    xt = tf.constant(x)
    y_tf = tf.clip_by_value(model.encoder(xt, training=False), -20.0, 20.0)
    z_tf = tf.clip_by_value(model.hyper_encoder(y_tf, training=False), -20.0, 20.0)
    z_hat_tf = tf.round(z_tf)
    mu_tf, sigma_tf = model.hyper_decoder(z_hat_tf, training=False)
    y_np, mu_a, sg_a = y_tf.numpy(), mu_tf.numpy(), sigma_tf.numpy()
    sym_a = np.round(y_np - mu_a).astype(np.int64)

    print(f"\nPATH A — checkpoint {args.checkpoint}")
    describe("y", y_np); describe("z", z_tf.numpy())
    describe("mu", mu_a); describe("sigma", sg_a); describe("symbols", sym_a)

    # The model's own reported estimate, for reference
    _, h_bpp, l_bpp = model(xt, training=False)
    print(f"  model-reported bpp: hyper={float(h_bpp):.4f} "
          f"latent={float(l_bpp):.4f} total={float(h_bpp + l_bpp):.4f}")

    # ---------------- Path B: exported TFLite ----------------
    enc, hyp_enc, hyp_dec = load_compress_interpreters(
        args.models_dir, use_int8=not args.fp32
    )
    y_l = run_interpreter(enc, x)
    z_l = run_interpreter(hyp_enc, y_l)
    z_hat_l = np.round(z_l).astype(np.float32)
    mu_b, sg_b = run_hyper_decoder(hyp_dec, z_hat_l)
    y_hat_b = np.round(y_l - mu_b) + mu_b
    sym_b = np.floor(y_hat_b - mu_b + 0.5).astype(np.int64)

    print(f"\nPATH B — {args.models_dir}")
    describe("y", y_l); describe("z", z_l)
    describe("mu", mu_b); describe("sigma", sg_b); describe("symbols", sym_b)

    # ---------------- Divergence ----------------
    print("\nDIVERGENCE (checkpoint vs export, same input)")
    for name, a, b in (("y", y_np, y_l), ("z", z_tf.numpy(), z_l),
                       ("mu", mu_a, mu_b), ("sigma", sg_a, sg_b)):
        if a.shape != b.shape:
            print(f"  {name:<7} SHAPE MISMATCH {a.shape} vs {b.shape}")
            continue
        d = np.abs(a - b)
        denom = np.maximum(np.abs(a).mean(), 1e-9)
        print(f"  {name:<7} max|Δ|={d.max():>9.4f}  mean|Δ|={d.mean():>9.4f}"
              f"  relative={d.mean() / denom:>7.2%}"
              f"  ratio B/A={np.abs(b).mean() / denom:>6.3f}")

    # ---------------- Cost accounting (latents only) ----------------
    print("\nLATENT COST  (bits/pixel, latents only — excludes hyperlatents)")
    for label, sym, sg in (("A checkpoint", sym_a, sg_a),
                           ("B export    ", sym_b, sg_b)):
        e = exact_bits(sym, sg).sum() / PIXELS
        t = table_bits(sym, sg).sum() / PIXELS
        print(f"  {label}  exact={e:.4f}   via ANS table={t:.4f}")

    print("\nCross-check — which tensor drives the gap:")
    e_ab = exact_bits(sym_a, sg_b).sum() / PIXELS
    e_ba = exact_bits(sym_b, sg_a).sum() / PIXELS
    print(f"  A symbols with B sigma: {e_ab:.4f}")
    print(f"  B symbols with A sigma: {e_ba:.4f}")
    print("  (whichever swap moves the number is the tensor at fault)")


if __name__ == "__main__":
    main()
