"""
Export trained Keras models to TFLite FP32 format.

Produces four .tflite files:
  encoder.tflite              image (1,256,256,3) → latents (1,16,16,96)
  hyper_encoder.tflite        latents            → hyperlatents (1,4,4,128)
  hyper_decoder.tflite        hyperlatents       → (mu, sigma) (1,16,16,96) each
  decoder.tflite              latents            → image (1,256,256,3)

Usage:
    python -m tflite.conversion.export_tflite \
        --checkpoint experiments/tflite_low/ckpt-500000 \
        --out_dir tflite_models/
"""

import argparse
import os
import numpy as np
import tensorflow as tf

from tflite.model.compression_model import CompressionModel


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

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 256, 256, 3], dtype=tf.float32)
    ])
    def encoder_fn(x):
        return model.encoder(x, training=False)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 16, 16, model.latent_channels], dtype=tf.float32)
    ])
    def hyper_encoder_fn(y):
        return model.hyper_encoder(y, training=False)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 4, 4, model.hyper_channels], dtype=tf.float32)
    ])
    def hyper_decoder_fn(z):
        return model.hyper_decoder(z, training=False)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 16, 16, model.latent_channels], dtype=tf.float32)
    ])
    def decoder_fn(y_hat):
        return model.decoder(y_hat, training=False)

    tasks = [
        ("encoder",       encoder_fn,       model.encoder),
        ("hyper_encoder", hyper_encoder_fn, model.hyper_encoder),
        ("hyper_decoder", hyper_decoder_fn, model.hyper_decoder),
        ("decoder",       decoder_fn,       model.decoder),
    ]

    for name, fn, sub_model in tasks:
        print(f"\nExporting {name} ...")
        # Pass sub_model as trackable_obj so TF can locate all variables;
        # avoids the "untracked resource" error from using tf.Module().
        concrete_fn = fn.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [concrete_fn], sub_model
        )
        out_path = os.path.join(out_dir, f"{name}.tflite")
        _save_tflite(converter, out_path)

    print("\nFP32 export complete.")


def verify_tflite(out_dir, latent_channels=96, hyper_channels=128):
    """Quick shape check for all four exported models."""
    print("\nVerifying TFLite models ...")
    checks = [
        ("encoder.tflite",      np.random.rand(1, 256, 256, 3).astype("float32"),  (1, 16, 16, latent_channels)),
        ("hyper_encoder.tflite", np.random.rand(1, 16, 16, latent_channels).astype("float32"), (1, 4, 4, hyper_channels)),
        ("decoder.tflite",      np.random.rand(1, 16, 16, latent_channels).astype("float32"),  (1, 256, 256, 3)),
    ]
    for fname, dummy_input, expected_shape in checks:
        path = os.path.join(out_dir, fname)
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        interp.set_tensor(inp["index"], dummy_input)
        interp.invoke()
        result = interp.get_tensor(out["index"])
        status = "OK" if result.shape == expected_shape else f"FAIL (got {result.shape})"
        print(f"  {fname}: output {result.shape}  {status}")

    # Hyper decoder has two outputs
    path = os.path.join(out_dir, "hyper_decoder.tflite")
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    dummy = np.random.rand(1, 4, 4, hyper_channels).astype("float32")
    interp.set_tensor(inp["index"], dummy)
    interp.invoke()
    shapes = [interp.get_tensor(o["index"]).shape for o in outs]
    exp = (1, 16, 16, latent_channels)
    ok = all(s == exp for s in shapes)
    print(f"  hyper_decoder.tflite: outputs {shapes}  {'OK' if ok else 'FAIL'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to tf.train.Checkpoint (e.g. experiments/tflite_low/ckpt-500000)")
    p.add_argument("--out_dir", default="tflite_models")
    p.add_argument("--verify", action="store_true", default=True)
    args = p.parse_args()

    print("Loading model ...")
    model = CompressionModel()
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(args.checkpoint).expect_partial()
    print(f"Loaded checkpoint: {args.checkpoint}")

    export_fp32(model, args.out_dir)

    if args.verify:
        verify_tflite(args.out_dir)


if __name__ == "__main__":
    main()
