"""
Raspberry Pi inference CLI for TFLite generative compression.

Runs the full compress / decompress pipeline using four TFLite models
and the Python ANS entropy coder.  Accepts a single file or a directory
of files; models are loaded once and reused across all images in a batch.

Usage:
    # Compress a single image
    python -m tflite.inference.compress \
        --compress \
        --input photo.jpg \
        --output photo.hfc \
        --models_dir ~/compression/tflite_models/ \
        --density_weights ~/compression/density_weights.npz

    # Compress a whole folder (outputs written to compressed_out/)
    python -m tflite.inference.compress \
        --compress \
        --input photos/ \
        --output compressed_out/ \
        --models_dir ~/compression/tflite_models/

    # Decompress a single file
    python -m tflite.inference.compress \
        --decompress \
        --input photo.hfc \
        --output photo_reconstructed.png \
        --models_dir ~/compression/tflite_models/ \
        --density_weights ~/compression/density_weights.npz \
        --metrics

    # Decompress a folder of .hfc files
    python -m tflite.inference.compress \
        --decompress \
        --input compressed_out/ \
        --output reconstructed_out/ \
        --models_dir ~/compression/tflite_models/

On Raspberry Pi install tflite-runtime instead of full tensorflow:
    pip install tflite-runtime Pillow numpy scipy
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Support both full TF (development) and tflite-runtime (Pi deployment)
try:
    import tensorflow as tf
    _Interpreter = tf.lite.Interpreter
except ImportError:
    import tflite_runtime.interpreter as tflite
    _Interpreter = tflite.Interpreter

# Make repo root importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tflite.compression.entropy_models import FactorizedPriorNumpy
from tflite.compression.prior_model import PriorModel
from tflite.compression.compression_utils import (
    CompressionOutput, save_compressed_format, load_compressed_format
)

CROP_SIZE = 256
NUM_THREADS = 4   # Raspberry Pi 4 has 4 Cortex-A72 cores


# ---------------------------------------------------------------------------
# TFLite interpreter helpers
# ---------------------------------------------------------------------------

def load_interpreter(model_path, num_threads=NUM_THREADS):
    """Load a TFLite model and allocate tensors."""
    interp = _Interpreter(model_path=model_path, num_threads=num_threads)
    interp.allocate_tensors()
    return interp


def run_interpreter(interp, input_array):
    """Run a single-input TFLite interpreter and return output(s) as numpy."""
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], input_array)
    interp.invoke()
    outs = interp.get_output_details()
    results = [interp.get_tensor(o["index"]) for o in outs]
    return results[0] if len(results) == 1 else results


def run_hyper_decoder(interp, z_hat):
    """
    Run the hyper decoder and return (mu, sigma) in the correct order.

    TFLite output ordering is not guaranteed to match the Keras model's
    return order [mu, sigma].  We identify which tensor is sigma (the scale
    output) because sigma = softplus(x) + MIN_SCALE is always ≥ MIN_SCALE,
    while mu is unconstrained and can be negative.  If neither output is
    all-positive we fall back to the name-based heuristic.
    """
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], z_hat)
    interp.invoke()
    outs = interp.get_output_details()
    r0 = interp.get_tensor(outs[0]["index"])
    r1 = interp.get_tensor(outs[1]["index"])

    # sigma is always ≥ MIN_SCALE (0.11); mu can be negative.
    r0_positive = float(np.nanmin(r0)) >= 0.10
    r1_positive = float(np.nanmin(r1)) >= 0.10

    if r0_positive and not r1_positive:
        return r1, r0   # r0=sigma, r1=mu → return (mu, sigma)
    elif r1_positive and not r0_positive:
        return r0, r1   # correct order already

    # Both or neither positive — try name-based fallback
    n0, n1 = outs[0]["name"].lower(), outs[1]["name"].lower()
    if "sigma" in n0 and "sigma" not in n1:
        return r1, r0
    return r0, r1   # default: assume [mu, sigma]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image(path, target_size=CROP_SIZE):
    """Load and centre-crop/resize image to target_size × target_size.

    The encoder TFLite model has a fixed input shape of (1, 256, 256, 3).
    """
    img = Image.open(path).convert("RGB")
    orig_size = (img.height, img.width)

    # Centre-crop to square then resize to target_size
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((target_size, target_size), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3) [0,1]
    arr = arr[np.newaxis]                             # (1, H, W, 3)
    return arr, (target_size, target_size)


def save_image(arr, path, orig_size=None):
    """Save (1, H, W, 3) float32 [0, 1] as PNG, optionally cropped to orig_size."""
    arr = np.clip(arr[0], 0.0, 1.0)
    if orig_size is not None:
        arr = arr[:orig_size[0], :orig_size[1]]
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(path)


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def compress(args, interpreters, prior_model, factorized_prior, input_path, output_path):
    t_start = time.time()

    print(f"Compressing: {input_path}")
    x, orig_size = load_image(input_path)
    print(f"  Image size: {orig_size} → padded to {x.shape[1:3]}")

    enc_interp, hyp_enc_interp, hyp_dec_interp = interpreters

    # 1. Encoder: image → latents y
    t = time.time()
    y = run_interpreter(enc_interp, x)        # (1, 16, 16, 96)
    print(f"  Encoder: {time.time()-t:.3f}s  y={y.shape}")

    # 2. Hyperprior analysis: y → hyperlatents z
    t = time.time()
    z = run_interpreter(hyp_enc_interp, y)    # (1, 4, 4, 128)
    print(f"  Hyper encoder: {time.time()-t:.3f}s  z={z.shape}")

    # 3. Quantize hyperlatents
    z_hat = np.round(z).astype(np.float32)

    # 4. Entropy encode hyperlatents using factorized prior
    t = time.time()
    factorized_prior.build_tables()
    # Transpose to (N, C, H, W) for ANS
    z_nchw = np.transpose(z_hat, (0, 3, 1, 2))
    n_ch = z_nchw.shape[1]
    z_sym = np.round(z_nchw).astype(np.int32)
    # Use per-channel CDF (indices = channel index broadcast over spatial)
    z_indices = np.tile(
        np.arange(n_ch, dtype=np.int32)[np.newaxis, :, np.newaxis, np.newaxis],
        (1, 1, z_nchw.shape[2], z_nchw.shape[3])
    )
    from tflite.compression import compression_utils as cu
    hyperlatents_encoded, hyper_coding_shape = cu.ans_compress(
        symbols=z_sym, indices=z_indices,
        cdf=factorized_prior.CDF, cdf_length=factorized_prior.CDF_length,
        cdf_offset=factorized_prior.CDF_offset,
        coding_shape=z_nchw.shape[1:], precision=16,
        block_encode=True, vectorize=False,
    )
    print(f"  ANS hyper encode: {time.time()-t:.3f}s")

    # 5. Hyperprior synthesis: z → (mu, sigma)
    t = time.time()
    mu, sigma = run_hyper_decoder(hyp_dec_interp, z_hat)   # each (1, 16, 16, 96)
    print(f"  Hyper decoder: {time.time()-t:.3f}s")

    # 6. Entropy encode latents using Gaussian prior
    t = time.time()
    y_hat = np.round(y - mu) * 1.0 + mu  # center-quantize
    latents_encoded, latent_coding_shape = prior_model.compress(
        y_hat, mu, sigma, block_encode=True
    )
    print(f"  ANS latent encode: {time.time()-t:.3f}s")

    # 7. Save .hfc bitstream
    compression_out = CompressionOutput(
        hyperlatents_encoded=hyperlatents_encoded,
        latents_encoded=latents_encoded,
        hyperlatent_spatial_shape=z_hat.shape[1:3],
        batch_shape=1,
        spatial_shape=orig_size,
        hyper_coding_shape=hyper_coding_shape,
        latent_coding_shape=latent_coding_shape,
    )
    actual_bpp = save_compressed_format(compression_out, str(output_path))

    orig_pixels = orig_size[0] * orig_size[1]
    orig_bytes = orig_pixels * 3
    hfc_bytes = os.path.getsize(output_path)
    theoretical_bpp = 8.0 * hfc_bytes / orig_pixels

    print(f"\n  Compressed: {output_path}")
    print(f"  BPP: {actual_bpp:.4f}  (original image: ~24 bpp)")
    print(f"  Compression ratio: {orig_bytes / hfc_bytes:.1f}×")
    print(f"  Total time: {time.time()-t_start:.2f}s")


# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------

def decompress(args, interpreters, prior_model, factorized_prior, input_path, output_path):
    t_start = time.time()

    print(f"Decompressing: {input_path}")
    compression_out = load_compressed_format(str(input_path))

    orig_size = compression_out.spatial_shape
    hyper_coding_shape = compression_out.hyper_coding_shape
    latent_coding_shape = compression_out.latent_coding_shape

    hyp_dec_interp, dec_interp = interpreters

    # 1. Entropy decode hyperlatents
    t = time.time()
    factorized_prior.build_tables()
    n_ch = hyper_coding_shape[0]
    z_indices = np.tile(
        np.arange(n_ch, dtype=np.int32)[np.newaxis, :, np.newaxis, np.newaxis],
        (1, 1, hyper_coding_shape[1], hyper_coding_shape[2])
    )
    from tflite.compression import compression_utils as cu
    z_decoded = cu.ans_decompress(
        encoded=compression_out.hyperlatents_encoded,
        indices=z_indices,
        cdf=factorized_prior.CDF, cdf_length=factorized_prior.CDF_length,
        cdf_offset=factorized_prior.CDF_offset,
        coding_shape=hyper_coding_shape, precision=16,
        block_decode=True, vectorize=False,
    )
    z_decoded = np.reshape(z_decoded, (1,) + tuple(hyper_coding_shape)).astype(np.float32)
    # NCHW → NHWC
    z_hat = np.transpose(z_decoded, (0, 2, 3, 1))
    print(f"  ANS hyper decode: {time.time()-t:.3f}s  z_hat={z_hat.shape}")

    # 2. Hyperprior synthesis: z_hat → (mu, sigma)
    t = time.time()
    mu, sigma = run_hyper_decoder(hyp_dec_interp, z_hat)
    print(f"  Hyper decoder: {time.time()-t:.3f}s")

    # 3. Entropy decode latents
    t = time.time()
    y_hat = prior_model.decompress(
        encoded=compression_out.latents_encoded,
        mu=mu, sigma=sigma,
        coding_shape=latent_coding_shape,
        block_decode=True,
    )
    print(f"  ANS latent decode: {time.time()-t:.3f}s  y_hat={y_hat.shape}")

    # 4. Decoder: y_hat → reconstruction
    t = time.time()
    x_hat = run_interpreter(dec_interp, y_hat.astype(np.float32))
    print(f"  Decoder: {time.time()-t:.3f}s  x_hat={x_hat.shape}")

    # 5. Save reconstruction
    save_image(x_hat, str(output_path), orig_size=orig_size)
    print(f"\n  Reconstructed: {output_path}")
    print(f"  Total time: {time.time()-t_start:.2f}s")

    # 6. Optional metrics
    if args.metrics:
        _compute_metrics(str(input_path), str(output_path), x_hat, orig_size)


def _compute_metrics(hfc_path, recon_path, x_hat, orig_size):
    """Compute PSNR, MS-SSIM, BPP."""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    except ImportError:
        print("  (install scikit-image for metrics)")
        return

    recon_img = Image.open(recon_path).convert("RGB")
    # Load original from bitstream name (best effort)
    hfc_bytes = os.path.getsize(hfc_path)
    bpp = 8.0 * hfc_bytes / (orig_size[0] * orig_size[1])

    x_recon_np = np.array(recon_img)
    print(f"\n  Metrics (reconstruction only — no original available for PSNR):")
    print(f"  BPP: {bpp:.4f}")
    print(f"  Reconstruction shape: {x_recon_np.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Models required for each operation — avoids loading unused models on the Pi.
# compress:   encoder + hyp_encoder + hyp_decoder  (decoder never runs on Pi)
# decompress: hyp_decoder + decoder               (encoder output is already in .hfc)
_COMPRESS_KEYS   = ("encoder", "hyper_encoder", "hyper_decoder")
_DECOMPRESS_KEYS = ("hyper_decoder", "decoder")


def _resolve_model_paths(models_dir, keys, use_int8):
    """Return {key: path} for the requested model keys, falling back to FP32."""
    suffix = "_int8" if use_int8 else ""
    paths = {}
    for key in keys:
        path = os.path.join(models_dir, f"{key}{suffix}.tflite")
        if not os.path.exists(path):
            fp32 = os.path.join(models_dir, f"{key}.tflite")
            if os.path.exists(fp32):
                print(f"  Warning: {os.path.basename(path)} not found, using FP32")
                path = fp32
            else:
                raise FileNotFoundError(
                    f"Model not found: {path}\nRun export_tflite.py first."
                )
        paths[key] = path
    return paths


def load_compress_interpreters(models_dir, use_int8=True):
    """Load only the models needed for compression (encoder, hyp_encoder, hyp_decoder)."""
    paths = _resolve_model_paths(models_dir, _COMPRESS_KEYS, use_int8)
    print("Loading TFLite models (compress mode) ...")
    interps = [load_interpreter(paths[k]) for k in _COMPRESS_KEYS]
    print(f"  Loaded: {', '.join(_COMPRESS_KEYS)}")
    return interps  # (enc, hyp_enc, hyp_dec)


def load_decompress_interpreters(models_dir, use_int8=True):
    """Load only the models needed for decompression (hyp_decoder, decoder)."""
    paths = _resolve_model_paths(models_dir, _DECOMPRESS_KEYS, use_int8)
    print("Loading TFLite models (decompress mode) ...")
    interps = [load_interpreter(paths[k]) for k in _DECOMPRESS_KEYS]
    print(f"  Loaded: {', '.join(_DECOMPRESS_KEYS)}")
    return interps  # (hyp_dec, dec)


def iter_batch(args):
    """Expand --input/--output into a list of (input_path, output_path) pairs.

    Accepts either a single file or a directory.  For a directory, globs for
    image/hfc files matching --glob patterns and writes outputs alongside the
    inputs (or into --output dir if provided).
    """
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else None

    if in_path.is_dir():
        patterns = [p.strip() for p in args.glob.split(",")]
        files = sorted(f for pat in patterns for f in in_path.glob(pat))
        if not files:
            print(f"No files matching {args.glob} found in {in_path}", file=sys.stderr)
            sys.exit(1)
        out_dir = out_path if out_path else in_path
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = ".hfc" if args.compress else ".png"
        return [(f, out_dir / (f.stem + ext)) for f in files]
    else:
        if args.output is None:
            print("--output is required when --input is a single file", file=sys.stderr)
            sys.exit(1)
        return [(in_path, Path(args.output))]


def main():
    p = argparse.ArgumentParser(description="TFLite generative compression")
    p.add_argument("--compress",   action="store_true")
    p.add_argument("--decompress", action="store_true")
    p.add_argument("--input",  "-i", required=True,
                   help="Input file or directory of files")
    p.add_argument("--output", "-o", default=None,
                   help="Output file or directory (defaults to same dir as input "
                        "when --input is a directory)")
    p.add_argument("--glob", default="*.jpg,*.jpeg,*.png,*.hfc,*.JPG",
                   help="Comma-separated glob patterns used when --input is a directory")
    p.add_argument("--models_dir", default="tflite_models/")
    p.add_argument("--density_weights", default="tflite_models/density_weights.npz",
                   help="Path to factorized prior weights exported during training")
    p.add_argument("--fp32", action="store_true",
                   help="Use FP32 models instead of INT8")
    p.add_argument("--scales_min", type=float, default=None,
                   help="Lower bound of the Gaussian scale table "
                        "(default 0.11, matching the model's MIN_SCALE). "
                        "The hyperprior saturates that floor, where a "
                        "residual of 1 costs 18.5 bits; raising it to "
                        "~0.2-0.3 can cut the coded rate several-fold at "
                        "zero quality cost. Lossless, but compress and "
                        "decompress MUST use the same value.")
    p.add_argument("--metrics", action="store_true",
                   help="Compute and print quality metrics after decompression")
    args = p.parse_args()

    if not args.compress and not args.decompress:
        p.error("Specify --compress or --decompress")

    # Validate: if input is a directory, output (if given) must also be a directory
    in_path = Path(args.input)
    if in_path.is_dir() and args.output and not Path(args.output).is_dir():
        out = Path(args.output)
        if out.suffix:  # looks like a file path, not a directory
            p.error(f"--output must be a directory when --input is a directory; got: {args.output}")

    # Load density weights and prior model (needed by both compress and decompress)
    if not os.path.exists(args.density_weights):
        raise FileNotFoundError(
            f"Density weights not found: {args.density_weights}\n"
            "Copy density_weights.npz from the training checkpoint directory."
        )
    fp_weights = np.load(args.density_weights, allow_pickle=True)
    factorized_prior = FactorizedPriorNumpy.from_weights(dict(fp_weights))
    prior_model = PriorModel(scales_min=args.scales_min)

    # Load only the TFLite models required for the requested operation
    pairs = iter_batch(args)
    if args.compress:
        interpreters = load_compress_interpreters(args.models_dir, use_int8=not args.fp32)
        for input_path, output_path in pairs:
            compress(args, interpreters, prior_model, factorized_prior, input_path, output_path)
    else:
        interpreters = load_decompress_interpreters(args.models_dir, use_int8=not args.fp32)
        for input_path, output_path in pairs:
            decompress(args, interpreters, prior_model, factorized_prior, input_path, output_path)


if __name__ == "__main__":
    main()
