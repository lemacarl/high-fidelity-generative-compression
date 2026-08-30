"""
Isolate the hyperlatent rate term.

trace_rate.py established that the latents are fine: computed from either
path's own tensors the latent cost is ~0.132 bpp exact / ~0.165 via the ANS
table, and the exported tensors match the checkpoint to <1%. Yet the
checkpoint reports ~0.52 bpp total while the device codes ~0.16. By
subtraction the hyperlatent term carries the whole discrepancy.

This charges the same z_hat two ways:

    TF    CompressionModel.factorized_prior.log_likelihood(z_hat)
          — the differentiable estimate used by evaluate AND by
            total_compression_loss during training

    ANS   FactorizedPriorNumpy CDF tables built from density_weights.npz
          — what the entropy coder actually spends

Both are the same learned density, so they should agree. The report also
breaks out the -10 nat clip in compression_model.py, because a clip that binds
on most elements both caps the reported cost and zeroes the gradient that
would teach the prior to fit z — which would leave the hyperprior permanently
untrained.

Usage:
    python -m tflite.trace_hyper \
        --checkpoint experiments/tflite_low_v2/final-2000000 \
        --density_weights experiments/tflite_low_v2/density_weights.npz \
        --image assets/coffee/test-new/test-image-1.png
"""

import argparse
import numpy as np
import tensorflow as tf

from tflite.inference.compress import load_image
from tflite.compression.entropy_models import FactorizedPriorNumpy, PRECISION
from tflite.model.compression_model import CompressionModel

PIXELS = 256 * 256
LN2 = np.log(2.0)
CLIP_NATS = 10.0          # matches compression_model.py


def ans_bits_per_element(z_hat_nhwc, fp):
    """Bits the ANS coder charges for each hyperlatent, via the CDF tables."""
    z = np.transpose(z_hat_nhwc, (0, 3, 1, 2))          # NHWC -> NCHW
    n, c, h, w = z.shape
    sym = np.round(z).astype(np.int64)
    ch = np.broadcast_to(np.arange(c)[None, :, None, None], z.shape).ravel()
    s = sym.ravel()

    off = fp.CDF_offset.astype(np.int64)[ch]
    length = fp.CDF_length.astype(np.int64)[ch]
    pos = np.clip(s - off, 0, length - 2)
    cdf = fp.CDF.astype(np.int64)
    mass = cdf[ch, pos + 1] - cdf[ch, pos]
    bits = -np.log2(np.maximum(mass, 1) / float(2 ** PRECISION))
    return bits.reshape(n, c, h, w), sym


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--density_weights", required=True)
    p.add_argument("--image", required=True)
    args = p.parse_args()

    x, _ = load_image(args.image)

    model = CompressionModel()
    _ = model(tf.zeros([1, 256, 256, 3]), training=False)
    tf.train.Checkpoint(model=model).restore(args.checkpoint).expect_partial()

    # ---- reach z_hat exactly as CompressionModel.call does ----
    y = tf.clip_by_value(model.encoder(tf.constant(x), training=False), -20., 20.)
    z = tf.clip_by_value(model.hyper_encoder(y, training=False), -20., 20.)
    z_hat = tf.round(z)

    # ---- what the model reports ----
    _, h_bpp, l_bpp = model(tf.constant(x), training=False)
    h_bpp, l_bpp = float(h_bpp), float(l_bpp)
    print(f"\nMODEL-REPORTED  hyper={h_bpp:.4f}  latent={l_bpp:.4f}  "
          f"total={h_bpp + l_bpp:.4f} bpp")

    # ---- TF estimate ----
    log_p = model.factorized_prior(z_hat).numpy()        # nats, (B,H,W,C)
    n_elem = log_p.size
    tf_bits_raw = -log_p / LN2
    log_p_clipped = np.clip(log_p, -CLIP_NATS, 0.0)
    tf_bits_clipped = -log_p_clipped / LN2

    at_clip = float(np.mean(log_p <= -CLIP_NATS))
    ceiling = n_elem * CLIP_NATS / (PIXELS * LN2)

    # ---- ANS reality ----
    fp = FactorizedPriorNumpy.from_weights(dict(np.load(args.density_weights,
                                                        allow_pickle=True)))
    fp.build_tables()
    ans_bits, sym = ans_bits_per_element(z_hat.numpy(), fp)

    print(f"\nHYPERLATENTS  shape={tuple(z_hat.shape)}  n={n_elem}")
    print(f"  z_hat  min={sym.min()}  max={sym.max()}  "
          f"mean={sym.mean():.3f}  std={sym.std():.3f}")

    print(f"\n{'source':<34}{'bpp':>9}{'bits/elem':>12}")
    print("-" * 55)
    print(f"{'TF estimate (no clip)':<34}"
          f"{tf_bits_raw.sum() / PIXELS:>9.4f}{tf_bits_raw.mean():>12.3f}")
    print(f"{'TF estimate (-10 nat clip, shipped)':<34}"
          f"{tf_bits_clipped.sum() / PIXELS:>9.4f}{tf_bits_clipped.mean():>12.3f}")
    print(f"{'ANS CDF tables (what is spent)':<34}"
          f"{ans_bits.sum() / PIXELS:>9.4f}{ans_bits.mean():>12.3f}")

    print(f"\nCLIP  elements at -10 nats: {at_clip:.1%}"
          f"   saturation ceiling: {ceiling:.4f} bpp")
    if at_clip > 0.2:
        print("  A clip binding this often also zeroes the gradient on those\n"
              "  elements, so the factorized prior never learns to fit z.")

    ratio = tf_bits_clipped.sum() / max(ans_bits.sum(), 1e-9)
    print(f"\nTF / ANS ratio: {ratio:.2f}x")

    # ---- worst channels ----
    tf_per_ch = tf_bits_clipped.mean(axis=(0, 1, 2))          # (C,)
    ans_per_ch = ans_bits.mean(axis=(0, 2, 3))                # (C,)
    gap = tf_per_ch - ans_per_ch
    worst = np.argsort(-gap)[:10]
    print(f"\nWorst channels  {'ch':>5}{'TF':>9}{'ANS':>9}{'gap':>9}"
          f"{'z mean':>9}")
    for c in worst:
        print(f"{'':<16}{c:>5}{tf_per_ch[c]:>9.3f}{ans_per_ch[c]:>9.3f}"
              f"{gap[c]:>9.3f}{sym[0, c].mean():>9.2f}")


if __name__ == "__main__":
    main()
