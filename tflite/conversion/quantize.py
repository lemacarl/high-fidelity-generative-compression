"""
INT8 post-training quantization for all four TFLite models.

Applies TFLite full-integer quantization using a representative dataset
(200 images from the training set). Keeps float32 I/O for compatibility
with the Python entropy coding layer.

Usage:
    python -m tflite.conversion.quantize \
        --models_dir tflite_models/ \
        --calibration_data data/openimages/ \
        [--n_calibration 200]

Output: tflite_models/{encoder,hyper_encoder,hyper_decoder,decoder}_int8.tflite
"""

import argparse
import os
import numpy as np
import tensorflow as tf

from tflite.training.data_pipeline import get_calibration_dataset


def quantize_model(fp32_path, out_path, representative_dataset_gen,
                   n_inputs=1):
    """
    Apply INT8 post-training quantization to a single TFLite model.

    Args:
        fp32_path:   Path to the FP32 .tflite model.
        out_path:    Path for the INT8 output model.
        representative_dataset_gen: callable returning a generator of
                     [input_array] lists (numpy float32).
        n_inputs:    Number of inputs the model expects (1 or 2).
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(
        # We need the SavedModel directories produced by export_tflite.py
        fp32_path.replace(".tflite", "").replace(
            os.path.dirname(fp32_path),
            os.path.join(os.path.dirname(fp32_path), "_saved")
        )
    )
    # Simpler: convert from the FP32 .tflite flatbuffer directly
    converter = tf.lite.TFLiteConverter.from_file(fp32_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen

    # Full INT8 with float32 I/O (easier interop with Python entropy coder)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,   # fallback for unsupported ops
    ]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    fp32_size = os.path.getsize(fp32_path) / 1e6
    int8_size = os.path.getsize(out_path) / 1e6
    reduction = (1 - int8_size / fp32_size) * 100
    print(f"  {os.path.basename(out_path)}: "
          f"{fp32_size:.2f} MB → {int8_size:.2f} MB  ({reduction:.0f}% smaller)")


def make_rep_dataset(data_root, n_samples, input_shape):
    """Build a representative dataset generator for a given input shape."""
    gen = get_calibration_dataset(data_root, n_samples=n_samples)

    def rep_dataset_fn():
        for batch in gen():
            img = batch[0]  # (1, 256, 256, 3)
            if input_shape == (1, 256, 256, 3):
                yield [img.astype(np.float32)]
            else:
                # For intermediate models we don't have real inputs —
                # use random noise within realistic value ranges
                yield [np.random.randn(*input_shape).astype(np.float32)]

    return rep_dataset_fn


def quantize_all(models_dir, calibration_data, n_calibration=200,
                 latent_channels=96, hyper_channels=128):
    """Quantize all four TFLite models."""
    print(f"\nQuantizing models in: {models_dir}")
    print(f"Calibration data: {calibration_data}  ({n_calibration} samples)\n")

    models = [
        ("encoder.tflite",       (1, 256, 256, 3)),
        ("hyper_encoder.tflite", (1, 16, 16, latent_channels)),
        ("hyper_decoder.tflite", (1, 4, 4, hyper_channels)),
        ("decoder.tflite",       (1, 16, 16, latent_channels)),
    ]

    for fname, input_shape in models:
        fp32_path = os.path.join(models_dir, fname)
        if not os.path.exists(fp32_path):
            print(f"  Skipping {fname} (not found)")
            continue

        out_name = fname.replace(".tflite", "_int8.tflite")
        out_path = os.path.join(models_dir, out_name)

        print(f"Quantizing {fname} ...")
        rep_gen = make_rep_dataset(calibration_data, n_calibration, input_shape)
        quantize_model(fp32_path, out_path, rep_gen)

    print("\nINT8 quantization complete.")


def verify_int8(models_dir, latent_channels=96, hyper_channels=128):
    """Sanity check: run inference through all INT8 models."""
    print("\nVerifying INT8 models ...")
    checks = [
        ("encoder_int8.tflite",       np.random.rand(1, 256, 256, 3).astype("float32")),
        ("hyper_encoder_int8.tflite", np.random.rand(1, 16, 16, latent_channels).astype("float32")),
        ("decoder_int8.tflite",       np.random.rand(1, 16, 16, latent_channels).astype("float32")),
    ]
    for fname, dummy in checks:
        path = os.path.join(models_dir, fname)
        if not os.path.exists(path):
            print(f"  {fname} not found, skipping")
            continue
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        outs = [interp.get_tensor(o["index"]) for o in interp.get_output_details()]
        print(f"  {fname}: outputs {[o.shape for o in outs]}  OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="tflite_models/",
                   help="Directory containing FP32 .tflite models")
    p.add_argument("--calibration_data", default="data/openimages/",
                   help="Root directory of calibration images")
    p.add_argument("--n_calibration", type=int, default=200)
    p.add_argument("--verify", action="store_true", default=True)
    args = p.parse_args()

    quantize_all(args.models_dir, args.calibration_data, args.n_calibration)
    if args.verify:
        verify_int8(args.models_dir)


if __name__ == "__main__":
    main()
