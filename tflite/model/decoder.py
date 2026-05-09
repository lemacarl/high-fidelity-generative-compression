"""
Lightweight decoder for TFLite generative compression.

Replaces the original 9-ResBlock + 4-transposed-conv generator with a
depthwise-separable design that is far cheaper on ARM Cortex-A72:

  Latents (B, 16, 16, C)
    → Entry 1×1 Conv (C → 256)
    → 3 Inverted-residual blocks at 16×16
    → 4× Bilinear upsample + DW-sep convolution stages
  Reconstruction (B, 256, 256, 3)  sigmoid → [0, 1]

Bilinear upsampling is preferred over ConvTranspose2D on ARM because it
avoids checkerboard artefacts and irregular memory access patterns.
"""

import tensorflow as tf

ENTRY_CHANNELS = 256
STAGE_CHANNELS = [256, 128, 64, 32]  # channel count after each upsample
EXPANSION_FACTOR = 4                 # inverted-residual expansion ratio


def _dw_sep_block(x, out_channels, name_prefix):
    """Depthwise-separable conv block: DW 3×3 → BN → ReLU6 → PW 1×1 → BN → ReLU6."""
    x = tf.keras.layers.DepthwiseConv2D(
        3, padding="same", use_bias=False, name=f"{name_prefix}_dw"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    x = tf.keras.layers.ReLU(6.0, name=f"{name_prefix}_dw_relu")(x)
    x = tf.keras.layers.Conv2D(
        out_channels, 1, padding="same", use_bias=False, name=f"{name_prefix}_pw"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_pw_bn")(x)
    x = tf.keras.layers.ReLU(6.0, name=f"{name_prefix}_pw_relu")(x)
    return x


def _inverted_residual(x, channels, expansion, name_prefix):
    """
    MobileNetV3-style inverted residual block with skip connection.
    Expansion → DW → Squeeze → skip add (only when in/out channels match).
    """
    in_channels = x.shape[-1]
    expanded = channels * expansion

    # Expand
    y = tf.keras.layers.Conv2D(
        expanded, 1, padding="same", use_bias=False, name=f"{name_prefix}_expand"
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_expand_bn")(y)
    y = tf.keras.layers.ReLU(6.0, name=f"{name_prefix}_expand_relu")(y)

    # Depthwise
    y = tf.keras.layers.DepthwiseConv2D(
        3, padding="same", use_bias=False, name=f"{name_prefix}_dw"
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(y)
    y = tf.keras.layers.ReLU(6.0, name=f"{name_prefix}_dw_relu")(y)

    # Project
    y = tf.keras.layers.Conv2D(
        channels, 1, padding="same", use_bias=False, name=f"{name_prefix}_project"
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_project_bn")(y)

    if in_channels == channels:
        y = tf.keras.layers.Add(name=f"{name_prefix}_add")([x, y])
    return y


def build_decoder(latent_channels=96, output_channels=3):
    """
    Build and return the Keras decoder model.

    Args:
        latent_channels: Must match the encoder's output depth.
        output_channels: 3 for RGB.

    Returns:
        tf.keras.Model  input:  (B, 16, 16, latent_channels) float32
                        output: (B, 256, 256, 3) float32 [0, 1]
    """
    inputs = tf.keras.Input(
        shape=(16, 16, latent_channels), name="latents_in"
    )

    # Entry projection
    x = tf.keras.layers.Conv2D(
        ENTRY_CHANNELS, 1, padding="same", use_bias=False, name="entry_conv"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="entry_bn")(x)
    x = tf.keras.layers.ReLU(6.0, name="entry_relu")(x)

    # Bottleneck residual blocks at 16×16
    for i in range(3):
        x = _inverted_residual(x, ENTRY_CHANNELS, EXPANSION_FACTOR, f"irb_{i}")

    # 4 upsample stages: 16→32→64→128→256
    for i, out_ch in enumerate(STAGE_CHANNELS):
        x = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear", name=f"upsample_{i}"
        )(x)
        x = _dw_sep_block(x, out_ch, name_prefix=f"up_block_{i}")

    # Output head: 7×7 conv → sigmoid, same style as original generator
    x = tf.keras.layers.Conv2D(
        output_channels, 7, padding="same", use_bias=True, name="output_conv"
    )(x)
    outputs = tf.keras.layers.Activation("sigmoid", name="output_sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="decoder")
    return model


if __name__ == "__main__":
    dec = build_decoder()
    dec.summary(line_length=100)
    import numpy as np
    dummy = np.random.rand(1, 16, 16, 96).astype("float32")
    out = dec(dummy, training=False)
    assert out.shape == (1, 256, 256, 3), f"Shape mismatch: {out.shape}"
    print(f"Decoder output shape: {out.shape}  OK")
