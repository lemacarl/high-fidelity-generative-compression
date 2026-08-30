"""
Isolate the cause of the train/eval bitrate gap.

Training logs report ~0.14 bpp while `tflite.evaluate` reports ~0.53 bpp on the
same checkpoint. Two things change between those two settings, and the model's
`training` flag controls both at once:

  1. quantization  — additive uniform noise (train) vs hard rounding (eval)
  2. normalization — encoder BatchNorm batch statistics (train) vs the stored
                     moving averages (eval); the decoder and hyper-nets use
                     LayerNorm and are unaffected

This script varies the two independently and reports bpp for all four
combinations, so the gap can be attributed to one, the other, or both.

Read the output like this:
  - bpp jumps when norm=batch → norm=moving   ⇒  BatchNorm is the cause
  - bpp jumps when quant=noise → quant=round  ⇒  the rate model is the cause
  - both jump                                 ⇒  both contribute

Images are evaluated as a single batch (default 8, matching the training batch
size) because BatchNorm batch statistics are meaningless at batch size 1.

Usage:
    python -m tflite.diagnose_rate_gap \
        --checkpoint experiments/tflite_low_v2/final-2000000 \
        --images assets/coffee/test-new/*.png
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from tflite.evaluate import load_image, psnr
from tflite.model.compression_model import CompressionModel

# (label, norm_training, quant_mode)
COMBOS = [
    ("norm=batch   quant=noise ", True,  "noise"),
    ("norm=batch   quant=round ", True,  "quantize"),
    ("norm=moving  quant=noise ", False, "noise"),
    ("norm=moving  quant=round ", False, "quantize"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Images per batch; should match the training batch "
                        "size for the BatchNorm comparison to be meaningful")
    args = p.parse_args()

    paths = [i for i in args.images if os.path.exists(i)]
    if not paths:
        raise SystemExit("No input images found")

    print("Loading model ...")
    model = CompressionModel()
    _ = model(tf.zeros([1, args.size, args.size, 3]), training=False)
    tf.train.Checkpoint(model=model).restore(args.checkpoint).expect_partial()
    print(f"Restored: {args.checkpoint}")
    print(f"Images:   {len(paths)}  (batch size {args.batch_size})\n")

    batches = [
        tf.concat([load_image(q, args.size) for q in paths[i:i + args.batch_size]], axis=0)
        for i in range(0, len(paths), args.batch_size)
    ]

    print(f"{'setting':<28}{'bpp':>10}{'PSNR dB':>11}")
    print("-" * 49)

    results = {}
    for label, norm_training, quant_mode in COMBOS:
        bpps, psnrs = [], []
        for x in batches:
            x_hat, hyper_bpp, latent_bpp = model(
                x, training=False,
                norm_training=norm_training, quant_mode=quant_mode,
            )
            bpps.append(float(hyper_bpp + latent_bpp))
            psnrs.append(psnr(x, x_hat))
        results[label] = (float(np.mean(bpps)), float(np.mean(psnrs)))
        print(f"{label:<28}{results[label][0]:>10.4f}{results[label][1]:>11.2f}")

    # Attribute the gap by holding one factor fixed and flipping the other.
    base = results["norm=batch   quant=noise "][0]
    bn_only = results["norm=moving  quant=noise "][0]
    q_only = results["norm=batch   quant=round "][0]
    both = results["norm=moving  quant=round "][0]

    print("\nAttribution (relative to norm=batch quant=noise):")
    print(f"  BatchNorm alone:    {base:.4f} → {bn_only:.4f}   ({bn_only / base:.2f}x)")
    print(f"  Quantization alone: {base:.4f} → {q_only:.4f}   ({q_only / base:.2f}x)")
    print(f"  Both (= eval):      {base:.4f} → {both:.4f}   ({both / base:.2f}x)")


if __name__ == "__main__":
    main()
