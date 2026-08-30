"""
Attribute the training-vs-evaluation bitrate gap to the data pipeline.

Training logs ~0.14 bpp; evaluation reports ~0.53 bpp on the same checkpoint.
`diagnose_rate_gap.py` ruled out quantization and BatchNorm — every mode
combination sits at ~0.53. That leaves the images themselves, which differ in
two independent ways:

  1. PREPROCESSING
       train: mild downscale (0.75-0.95x) then a random 256x256 crop, i.e. a
              near-native-resolution *patch* of the photo
       eval:  centre-crop to square, then LANCZOS-resize the *whole* square to
              256x256, i.e. the entire scene squeezed into 256x256
     The second packs far more high-frequency detail per pixel, which costs
     more bits. Note the on-device path (tflite/inference/compress.py) uses the
     eval-style loader, so eval — not training — reflects deployment.

  2. DOMAIN
       train: OpenImages
       eval:  coffee photographs

This runs the 2x2 so the two can be separated.

Expected if preprocessing dominates:
    openimages + train-loader  ~= 0.14   (reproduces the training logs)
    openimages + eval-loader   >> 0.14   (gap appears without changing domain)

Expected if domain dominates:
    openimages + eval-loader   ~= 0.14
    coffee     + eval-loader   >> 0.14

Usage:
    python -m tflite.diagnose_data_gap \
        --checkpoint experiments/tflite_low_v2/final-2000000
"""

import argparse
import glob
import os
import random

import numpy as np
import tensorflow as tf

from tflite.evaluate import load_image as eval_load_image, psnr
from tflite.model.compression_model import CompressionModel
from tflite.training.data_pipeline import _load_and_preprocess

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG")


def find_images(root, n, seed=0):
    files = []
    for e in EXTS:
        files.extend(glob.glob(os.path.join(root, "**", e), recursive=True))
    files.sort()
    if len(files) > n:
        random.Random(seed).shuffle(files)
        files = files[:n]
    return files


def train_load_image(path, size):
    """Training-style: mild downscale + random crop (stochastic)."""
    img = _load_and_preprocess(tf.constant(path), crop_size=size, training=True)
    return tf.expand_dims(img, 0)


def measure(model, paths, loader, size, batch_size):
    bpps, psnrs = [], []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        try:
            x = tf.concat([loader(q, size) for q in chunk], axis=0)
        except Exception:
            continue                      # unreadable / too-small image
        x_hat, hyper_bpp, latent_bpp = model(x, training=False)
        bpps.append(float(hyper_bpp + latent_bpp))
        psnrs.append(psnr(x, x_hat))
    if not bpps:
        return float("nan"), float("nan")
    return float(np.mean(bpps)), float(np.mean(psnrs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--openimages_dir", default="data/openimages/train")
    p.add_argument("--coffee_dir", default="assets/coffee/test-new")
    p.add_argument("--n_images", type=int, default=40)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    print("Loading model ...")
    model = CompressionModel()
    _ = model(tf.zeros([1, args.size, args.size, 3]), training=False)
    tf.train.Checkpoint(model=model).restore(args.checkpoint).expect_partial()
    print(f"Restored: {args.checkpoint}\n")

    datasets = {
        "openimages": find_images(args.openimages_dir, args.n_images),
        "coffee": find_images(args.coffee_dir, args.n_images),
    }
    loaders = {
        "train-loader (random crop) ": train_load_image,
        "eval-loader  (full resize) ": eval_load_image,
    }

    for name, paths in datasets.items():
        print(f"{name}: {len(paths)} images")
    print()

    print(f"{'dataset':<13}{'loader':<29}{'bpp':>9}{'PSNR dB':>10}")
    print("-" * 61)
    table = {}
    for dname, paths in datasets.items():
        if not paths:
            print(f"{dname:<13}{'(no images found)':<29}")
            continue
        for lname, loader in loaders.items():
            bpp, ps = measure(model, paths, loader, args.size, args.batch_size)
            table[(dname, lname)] = bpp
            print(f"{dname:<13}{lname:<29}{bpp:>9.4f}{ps:>10.2f}")

    tl, el = list(loaders)
    if ("openimages", tl) in table and ("openimages", el) in table:
        base = table[("openimages", tl)]
        print("\nAttribution (baseline = openimages + train-loader, "
              "the setting the training logs measured):")
        print(f"  baseline                  {base:.4f}")
        if ("openimages", el) in table:
            v = table[("openimages", el)]
            print(f"  + preprocessing change    {v:.4f}   ({v / base:.2f}x)")
        if ("coffee", tl) in table:
            v = table[("coffee", tl)]
            print(f"  + domain change           {v:.4f}   ({v / base:.2f}x)")
        if ("coffee", el) in table:
            v = table[("coffee", el)]
            print(f"  + both (= your eval)      {v:.4f}   ({v / base:.2f}x)")


if __name__ == "__main__":
    main()
