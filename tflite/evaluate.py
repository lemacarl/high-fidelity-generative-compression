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
import csv
import os
import re
import time

import numpy as np
import tensorflow as tf
from PIL import Image

from tflite.model.compression_model import CompressionModel


def _image_index(path, fallback):
    """Pull the trailing number out of e.g. test-image-12.png."""
    m = re.findall(r"(\d+)", os.path.basename(path))
    return int(m[-1]) if m else fallback


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


def evaluate_image(path, model, out_dir, lpips_metric=None):
    x = load_image(path)

    t0 = time.time()
    # Forward pass with noise quantization (training=True gives noise, False gives round)
    # Use training=False so quantize rounds rather than adds noise — same as inference
    x_hat, hyper_bpp, latent_bpp = model(x, training=False)
    elapsed = time.time() - t0

    bpp = float(hyper_bpp + latent_bpp)
    p   = psnr(x, x_hat)
    s   = ms_ssim(x, x_hat)
    lp  = float(lpips_metric(x, x_hat)[0]) if lpips_metric is not None else None

    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\n{name}")
    print(f"  BPP:     {bpp:.4f}")
    print(f"  PSNR:    {p:.2f} dB")
    print(f"  MS-SSIM: {s:.4f}")
    if lp is not None:
        print(f"  LPIPS:   {lp:.4f}  (lower is better)")
    print(f"  Time:    {elapsed*1000:.1f} ms")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}_recon.png")
        save_comparison(x.numpy(), x_hat.numpy(), out_path)
        print(f"  Saved:   {out_path}  (original | reconstruction)")

    return {"name": name, "path": path, "bpp": bpp, "psnr": p,
            "ms_ssim": s, "lpips": lp}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="e.g. experiments/tflite_low/final-500000")
    p.add_argument("--images", nargs="+", required=True,
                   help="One or more image paths to evaluate")
    p.add_argument("--csv", default=None,
                   help="Write per-image metrics to this CSV. Lets callers "
                        "read results directly instead of scraping stdout.")
    p.add_argument("--label", default="model",
                   help="Value for the model_variant CSV column")
    p.add_argument("--no_lpips", action="store_true",
                   help="Skip LPIPS. It is the only metric here that can tell "
                        "a working GAN from a broken one — PSNR and MS-SSIM "
                        "are distortion measures and penalise a GAN either "
                        "way — so skip it only to save time.")
    p.add_argument("--out_dir", default="eval_out",
                   help="Directory to save side-by-side comparisons")
    p.add_argument("--size", type=int, default=256,
                   help="Crop/resize to this square size (default 256)")
    p.add_argument("--density_weights", default=None,
                   help="Path to density_weights.npz. Required for checkpoints "
                        "written before the factorized prior was tracked in "
                        "the object graph — those contain no prior, so BPP is "
                        "computed against a randomly-initialised density and "
                        "is meaningless. Defaults to density_weights.npz "
                        "beside the checkpoint when that file exists.")
    args = p.parse_args()

    print("Loading model ...")
    model = CompressionModel()
    # Call once to build all sub-models before restoring
    dummy = tf.zeros([1, args.size, args.size, 3])
    _ = model(dummy, training=False)

    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(args.checkpoint).expect_partial()
    print(f"Restored: {args.checkpoint}")

    # The prior is what turns latents into a bitrate. If the checkpoint does
    # not carry it, every BPP below is computed against a random density.
    dw = args.density_weights
    if dw is None:
        guess = os.path.join(os.path.dirname(args.checkpoint),
                             "density_weights.npz")
        dw = guess if os.path.exists(guess) else None

    prior_in_ckpt = any(
        "factorized" in n.lower()
        for n, _ in tf.train.list_variables(args.checkpoint)
    )
    if prior_in_ckpt:
        print("Factorized prior: restored from checkpoint")
    elif dw:
        n_loaded = model.load_factorized_prior_weights(dw)
        print(f"Factorized prior: not in checkpoint — loaded {n_loaded} "
              f"variables from {dw}")
    else:
        print("WARNING: the checkpoint contains no factorized prior and no "
              "density_weights.npz was found. BPP below is computed against a "
              "randomly-initialised density and is NOT meaningful. Pass "
              "--density_weights to fix.")

    lpips_metric = None
    if not args.no_lpips:
        try:
            from tflite.lpips import LPIPS
            lpips_metric = LPIPS()
            print("LPIPS: enabled (VGG16, lower is better)")
        except FileNotFoundError as exc:
            print(f"LPIPS: disabled — {exc}")
        except Exception as exc:                              # noqa: BLE001
            print(f"LPIPS: disabled — {type(exc).__name__}: {exc}")

    results, rows = [], []
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping")
            rows.append(dict(model_variant=args.label,
                             image_n=_image_index(img_path, len(rows) + 1),
                             image_path=img_path, checkpoint=args.checkpoint,
                             psnr_db="", ms_ssim="", lpips="", bpp="",
                             status="SKIPPED_NO_INPUT"))
            continue
        try:
            r = evaluate_image(img_path, model, args.out_dir, lpips_metric)
        except Exception as exc:                              # noqa: BLE001
            print(f"  ERROR on {img_path}: {type(exc).__name__}: {exc}")
            rows.append(dict(model_variant=args.label,
                             image_n=_image_index(img_path, len(rows) + 1),
                             image_path=img_path, checkpoint=args.checkpoint,
                             psnr_db="", ms_ssim="", lpips="", bpp="",
                             status=f"ERROR: {type(exc).__name__}"))
            continue
        results.append(r)
        rows.append(dict(model_variant=args.label,
                         image_n=_image_index(img_path, len(rows) + 1),
                         image_path=img_path, checkpoint=args.checkpoint,
                         psnr_db=f"{r['psnr']:.2f}", ms_ssim=f"{r['ms_ssim']:.4f}",
                         lpips=f"{r['lpips']:.4f}" if r["lpips"] is not None else "",
                         bpp=f"{r['bpp']:.4f}", status="OK"))

    if len(results) > 1:
        avg_bpp    = np.mean([r["bpp"]     for r in results])
        avg_psnr   = np.mean([r["psnr"]    for r in results])
        avg_ms_ssim = np.mean([r["ms_ssim"] for r in results])
        lp_vals = [r["lpips"] for r in results if r["lpips"] is not None]
        print(f"\n{'─'*40}")
        print(f"Average over {len(results)} images:")
        print(f"  BPP:     {avg_bpp:.4f}")
        print(f"  PSNR:    {avg_psnr:.2f} dB")
        print(f"  MS-SSIM: {avg_ms_ssim:.4f}")
        if lp_vals:
            print(f"  LPIPS:   {np.mean(lp_vals):.4f}  (lower is better)")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV written to: {args.csv}")


if __name__ == "__main__":
    main()
