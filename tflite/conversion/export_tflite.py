"""
Export trained Keras models to TFLite FP32 and optional INT8 format.

Produces four .tflite files (FP32), and optionally four *_int8.tflite files:
  encoder.tflite              image (1,256,256,3) → latents (1,16,16,96)
  hyper_encoder.tflite        latents            → hyperlatents (1,4,4,128)
  hyper_decoder.tflite        hyperlatents       → (mu, sigma) (1,16,16,96) each
  decoder.tflite              latents            → image (1,256,256,3)

Usage:
    # FP32 only
    python -m tflite.conversion.export_tflite \
        --checkpoint experiments/tflite_low/final-500000 \
        --out_dir tflite_models/

    # FP32 + INT8 (provide ~100 calibration images)
    python -m tflite.conversion.export_tflite \
        --checkpoint experiments/tflite_low/final-500000 \
        --out_dir tflite_models/ \
        --int8 \
        --image_dir data/train/
"""

import argparse
import glob
import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image

from tflite.model.compression_model import CompressionModel


def _sanitize_model_weights(model, min_variance=1e-3):
    """
    Repair NaN / near-zero values in model variables before TFLite export.

    Two failure modes cause NaN in TFLite (but not TF eager with training=True):

    1. BatchNormalization moving_mean / moving_variance contain NaN.
       These stats become NaN after a single bad batch via EMA:
           moving_stat = 0.99 * NaN + 0.01 * valid = NaN  (stays NaN forever)
       The NaN gradient filter protects Adam but NOT the BN EMA update.
       In training=True mode BN uses fresh batch stats so the model still
       trains and evaluate.py still works; training=False (used by TFLite)
       uses the NaN moving stats → NaN everywhere.

    2. moving_variance near zero → TFLite fuses 1/sqrt(var+eps) into the
       conv kernel, creating overflow weights that become NaN in float32.

    Fix: replace NaN stats with safe defaults (mean=0, var=1) and clip
    near-zero variances to min_variance.
    """
    n_fixed = 0
    for layer in model.layers:
        if hasattr(layer, 'layers'):          # recurse into sub-models
            n_fixed += _sanitize_model_weights(layer, min_variance)
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            mm = layer.moving_mean
            mv = layer.moving_variance

            safe_mean = tf.where(tf.math.is_finite(mm),
                                 mm, tf.zeros_like(mm))
            n_nan_mean = int(tf.reduce_sum(
                tf.cast(~tf.math.is_finite(mm), tf.int32)).numpy())
            if n_nan_mean:
                mm.assign(safe_mean)

            safe_var = tf.where(tf.math.is_finite(mv),
                                tf.maximum(mv, min_variance),
                                tf.ones_like(mv))
            n_nan_var = int(tf.reduce_sum(
                tf.cast(~tf.math.is_finite(mv), tf.int32)).numpy())
            if n_nan_var:
                mv.assign(safe_var)

            n_fixed += n_nan_mean + n_nan_var

    return n_fixed


def _audit_variables(model):
    """Print a NaN/Inf audit of all model variables. Returns (n_nan, n_total)."""
    n_nan_total, n_elem_total = 0, 0
    for v in model.variables:
        n_bad = int(tf.reduce_sum(
            tf.cast(~tf.math.is_finite(v), tf.int32)).numpy())
        n_elem_total += v.shape.num_elements() or 0
        if n_bad:
            n_nan_total += n_bad
            print(f"    NaN/Inf  {v.name}  shape={v.shape}  n_bad={n_bad}")
    return n_nan_total, n_elem_total


def _save_tflite(converter, out_path):
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved {out_path}  ({size_mb:.2f} MB)")
    return tflite_model


def export_fp32(model, out_dir):
    """Convert all four sub-models to FP32 TFLite."""
    os.makedirs(out_dir, exist_ok=True)

    # ── Audit variables before sanitization ─────────────────────────────────
    print("\nAuditing model variables for NaN/Inf ...")
    n_bad, n_total = _audit_variables(model)
    if n_bad:
        print(f"  Found {n_bad} NaN/Inf values in {n_total} parameters.")
    else:
        print(f"  All {n_total} parameters are finite.")

    # ── Sanitize BN moving stats ─────────────────────────────────────────────
    n_fixed = _sanitize_model_weights(model)
    if n_fixed:
        print(f"  Sanitized {n_fixed} NaN BN stat(s). Re-auditing ...")
        n_bad2, _ = _audit_variables(model)
        if n_bad2:
            print(f"  WARNING: {n_bad2} NaN/Inf remain after sanitization "
                  "(likely in trainable weights — TFLite output may still be NaN).")
        else:
            print("  All variables are now finite.")

    # ── Quick Keras inference check ──────────────────────────────────────────
    print("\nKeras model sanity check (training=False) ...")
    dummy_img = np.random.rand(1, 256, 256, 3).astype("float32")
    y = model.encoder(dummy_img, training=False)
    y_nan = np.any(np.isnan(y.numpy()))
    print(f"  Encoder output: shape={y.shape}  has_nan={y_nan}  "
          f"range=[{np.nanmin(y.numpy()):.3f}, {np.nanmax(y.numpy()):.3f}]")
    if y_nan:
        print("  WARNING: Keras encoder still outputs NaN after BN sanitization.")
        print("  Trainable weights may contain NaN — the model may need retraining.")

    # ── TFLite conversion via from_keras_model ──────────────────────────────
    # from_concrete_functions does not embed weights from nested Keras
    # functional models (e.g. the MobileNetV3 backbone inside the encoder),
    # producing tiny flatbuffers where weight buffers are uninitialised → NaN.
    # from_keras_model traverses the full Keras layer/variable hierarchy and
    # embeds all weights correctly.
    tasks = [
        ("encoder",       model.encoder),
        ("hyper_encoder", model.hyper_encoder),
        ("hyper_decoder", model.hyper_decoder),
        ("decoder",       model.decoder),
    ]

    print()
    for name, sub_model in tasks:
        print(f"Exporting {name} ...")
        converter = tf.lite.TFLiteConverter.from_keras_model(sub_model)
        out_path = os.path.join(out_dir, f"{name}.tflite")
        _save_tflite(converter, out_path)

    print("\nFP32 export complete.")


def _make_representative_dataset(image_dir, n=100):
    """Return a generator that yields calibration batches from image_dir."""
    paths = glob.glob(os.path.join(image_dir, "**/*.jpg"), recursive=True)
    paths += glob.glob(os.path.join(image_dir, "**/*.png"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No jpg/png images found under {image_dir!r}")
    random.shuffle(paths)
    paths = paths[:n]

    def gen():
        for p in paths:
            img = np.array(Image.open(p).convert("RGB").resize((256, 256)),
                           dtype=np.float32) / 255.0
            yield [img[np.newaxis]]   # (1, 256, 256, 3)
    return gen


def export_int8(model, out_dir, image_dir):
    """Convert all four sub-models to INT8 TFLite (float32 I/O, int8 ops)."""
    os.makedirs(out_dir, exist_ok=True)

    # Only the encoder has BN/conv layers sensitive to activation range;
    # the other three use random representative data which is sufficient.
    tasks = [
        ("encoder",       model.encoder,       _make_representative_dataset(image_dir)),
        ("hyper_encoder", model.hyper_encoder,  None),
        ("hyper_decoder", model.hyper_decoder,  None),
        ("decoder",       model.decoder,        None),
    ]

    print()
    for name, sub_model, rep_data in tasks:
        print(f"Exporting {name} (INT8) ...")
        converter = tf.lite.TFLiteConverter.from_keras_model(sub_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if rep_data is not None:
            converter.representative_dataset = rep_data
        else:
            # Random calibration for purely linear sub-models
            inp_shape = sub_model.input_shape[1:]
            def _rand_gen(shape=inp_shape):
                for _ in range(50):
                    yield [np.random.rand(1, *shape).astype("float32")]
            converter.representative_dataset = _rand_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,   # FP32 fallback for ops not supported in INT8 (e.g. LOG)
        ]
        converter.inference_input_type  = tf.float32
        converter.inference_output_type = tf.float32
        out_path = os.path.join(out_dir, f"{name}_int8.tflite")
        _save_tflite(converter, out_path)

    print("\nINT8 export complete.")


def verify_tflite(out_dir, latent_channels=96, hyper_channels=192):
    """Shape + NaN sanity check for all four exported models."""
    print("\nVerifying TFLite models ...")

    def _run(interp, dummy):
        inp = interp.get_input_details()[0]
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        return [interp.get_tensor(o["index"]) for o in interp.get_output_details()]

    def _check(name, results, expected_shapes):
        got_shapes = [r.shape for r in results]
        shape_ok  = got_shapes == list(expected_shapes)
        any_nan   = any(np.any(np.isnan(r)) for r in results)
        any_inf   = any(np.any(np.isinf(r)) for r in results)
        ranges    = [(float(np.nanmin(r)), float(np.nanmax(r))) for r in results]
        issues = []
        if not shape_ok: issues.append(f"SHAPE MISMATCH got {got_shapes}")
        if any_nan:      issues.append("NaN IN OUTPUT")
        if any_inf:      issues.append("Inf IN OUTPUT")
        ok = not issues
        print(f"  {name}: shapes={got_shapes}  ranges={[(f'{lo:.3f}',f'{hi:.3f}') for lo,hi in ranges]}  "
              f"{'OK' if ok else '  '.join(issues)}")
        return ok

    dummy_img = np.random.rand(1, 256, 256, 3).astype("float32")
    dummy_lat = np.random.rand(1, 16, 16, latent_channels).astype("float32")
    dummy_hyp = np.random.rand(1, 4, 4, hyper_channels).astype("float32")

    all_ok = True
    for fname, dummy, exp_shapes in [
        ("encoder.tflite",       dummy_img, [(1, 16, 16, latent_channels)]),
        ("hyper_encoder.tflite", dummy_lat, [(1, 4, 4, hyper_channels)]),
        ("decoder.tflite",       dummy_lat, [(1, 256, 256, 3)]),
    ]:
        interp = tf.lite.Interpreter(model_path=os.path.join(out_dir, fname))
        interp.allocate_tensors()
        results = _run(interp, dummy)
        all_ok = _check(fname, results, exp_shapes) and all_ok

    # Hyper decoder: show output names to confirm mu/sigma ordering
    hd_path = os.path.join(out_dir, "hyper_decoder.tflite")
    interp  = tf.lite.Interpreter(model_path=hd_path)
    interp.allocate_tensors()
    results = _run(interp, dummy_hyp)
    ods = interp.get_output_details()
    exp = (1, 16, 16, latent_channels)
    all_ok = _check("hyper_decoder.tflite", results, [exp, exp]) and all_ok
    print("  hyper_decoder output names (for ordering check):")
    for i, od in enumerate(ods):
        r = results[i]
        print(f"    [{i}] {od['name']!r}  "
              f"min={float(np.nanmin(r)):.4f}  max={float(np.nanmax(r)):.4f}")

    print("All models OK." if all_ok else "WARNING: verification failed.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="e.g. experiments/tflite_low/final-500000")
    p.add_argument("--out_dir", default="tflite_models")
    p.add_argument("--int8", action="store_true",
                   help="Also export INT8-quantized models")
    p.add_argument("--image_dir", default="data/train",
                   help="Directory of jpg/png images for INT8 calibration")
    p.add_argument("--verify", action="store_true", default=True)
    args = p.parse_args()

    print("Loading model ...")
    model = CompressionModel()
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(args.checkpoint).expect_partial()
    print(f"Loaded checkpoint: {args.checkpoint}")

    export_fp32(model, args.out_dir)

    if args.int8:
        export_int8(model, args.out_dir, args.image_dir)

    if args.verify:
        verify_tflite(args.out_dir,
                      latent_channels=model.latent_channels,
                      hyper_channels=model.hyper_channels)


if __name__ == "__main__":
    main()
