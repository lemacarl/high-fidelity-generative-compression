"""
Quick GPU-side evaluation of the trained CompressionModel.

Runs compress → decompress entirely in TF (no TFLite conversion needed)
and reports PSNR, MS-SSIM, and actual BPP for one or more test images.

Usage:
    python -m tflite.evaluate \
        --checkpoint experiments/tflite_low/final-500000 \
        --images assets/camp_jpg_compress.png \
        --out_dir eval_out/
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image

from tflite.model.compression_model import CompressionModel


def load_image(path, target_size=256):
    """Load and centre-crop to target_size × target_size, return [0,1] float32."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # Centre-crop to square then resize to target_size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
    return tf.expand_dims(arr, 0)                    # (1, H, W, 3)


def psnr(x, x_hat):
    mse = tf.reduce_mean(tf.square(x - x_hat))
    if mse == 0:
        return float("inf")
    return float(10.0 * tf.math.log(1.0 / mse) / tf.math.log(10.0))


def ms_ssim(x, x_hat):
    val = tf.image.ssim_multiscale(x * 255.0, x_hat * 255.0, max_val=255.0)
    return float(tf.reduce_mean(val))


def save_comparison(x_np, x_hat_np, out_path):
    """Save original and reconstruction side-by-side."""
    orig  = (np.clip(x_np[0],    0, 1) * 255).astype(np.uint8)
    recon = (np.clip(x_hat_np[0], 0, 1) * 255).astype(np.uint8)
    comparison = np.concatenate([orig, recon], axis=1)
    Image.fromarray(comparison).save(out_path)


def evaluate_image(path, model, out_dir):
    x = load_image(path)

    t0 = time.time()
    # Forward pass with noise quantization (training=True gives noise, False gives round)
    # Use training=False so quantize rounds rather than adds noise — same as inference
    x_hat, hyper_bpp, latent_bpp = model(x, training=False)
    elapsed = time.time() - t0

    bpp = float(hyper_bpp + latent_bpp)
    p   = psnr(x, x_hat)
    s   = ms_ssim(x, x_hat)

    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\n{name}")
    print(f"  BPP:     {bpp:.4f}")
    print(f"  PSNR:    {p:.2f} dB")
    print(f"  MS-SSIM: {s:.4f}")
    print(f"  Time:    {elapsed*1000:.1f} ms")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}_recon.png")
        save_comparison(x.numpy(), x_hat.numpy(), out_path)
        print(f"  Saved:   {out_path}  (original | reconstruction)")

    return {"name": name, "bpp": bpp, "psnr": p, "ms_ssim": s}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="e.g. experiments/tflite_low/final-500000")
    p.add_argument("--images", nargs="+", required=True,
                   help="One or more image paths to evaluate")
    p.add_argument("--out_dir", default="eval_out",
                   help="Directory to save side-by-side comparisons")
    p.add_argument("--size", type=int, default=256,
                   help="Crop/resize to this square size (default 256)")
    args = p.parse_args()

    print("Loading model ...")
    model = CompressionModel()
    # Call once to build all sub-models before restoring
    dummy = tf.zeros([1, args.size, args.size, 3])
    _ = model(dummy, training=False)

    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(args.checkpoint).expect_partial()
    print(f"Restored: {args.checkpoint}")

    results = []
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping")
            continue
        results.append(evaluate_image(img_path, model, args.out_dir))

    if len(results) > 1:
        avg_bpp    = np.mean([r["bpp"]     for r in results])
        avg_psnr   = np.mean([r["psnr"]    for r in results])
        avg_ms_ssim = np.mean([r["ms_ssim"] for r in results])
        print(f"\n{'─'*40}")
        print(f"Average over {len(results)} images:")
        print(f"  BPP:     {avg_bpp:.4f}")
        print(f"  PSNR:    {avg_psnr:.2f} dB")
        print(f"  MS-SSIM: {avg_ms_ssim:.4f}")


if __name__ == "__main__":
    main()
