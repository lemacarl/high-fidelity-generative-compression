"""
Full-capacity decoder for TFLite generative compression.

Matches the original HIFIC Generator (~25 M parameters) translated from
PyTorch to TF/Keras with two ARM/INT8-safe substitutions:

  • ConvTranspose2D  →  UpSampling2D (bilinear) + Conv2D
    Avoids checkerboard artefacts in INT8 TFLite on ARM.

  • ReflectionPad2d  →  "same" padding
    Native TFLite support; marginal quality difference.

  • ChannelNorm2D / InstanceNorm2D  →  LayerNormalization
    Closest functional equivalent without track_running_stats state.
    TFLite-supported since v2.5.

Architecture (default, C=96 latent channels):

  Latents (B, 16, 16, 96)
    → LayerNorm + Conv2D(96→384, 3×3)          # entry projection
    → 9× ResidualBlock(384)                    # bottleneck at 16×16
    → UpSample→Conv2D→LN→ReLU  (16→32,  384→192)
    → UpSample→Conv2D→LN→ReLU  (32→64,  192→96)
    → UpSample→Conv2D→LN→ReLU  (64→128, 96→64)
    → UpSample→Conv2D→LN→ReLU  (128→256, 64→32)
    → Conv2D(32→3, 7×7) + sigmoid
  Reconstruction (B, 256, 256, 3)

Approximate parameter count: ~25.5 M
"""

import tensorflow as tf

ENTRY_CHANNELS = 384
N_RESIDUAL_BLOCKS = 9
STAGE_CHANNELS = [192, 96, 64, 32]   # channel count after each upsample stage


def _residual_block(x, channels, name_prefix):
    """
    Standard residual block: Conv2D → LayerNorm → ReLU → Conv2D → LayerNorm + skip.

    Mirrors PyTorch ResidualBlock in src/network/generator.py.
    Uses LayerNorm (norm over channel axis per spatial position) instead of
    ChannelNorm2D / InstanceNorm2D(track_running_stats=False).
    """
    y = tf.keras.layers.Conv2D(
        channels, 3, padding="same", use_bias=False, name=f"{name_prefix}_c1"
    )(x)
    y = tf.keras.layers.LayerNormalization(
        axis=-1, name=f"{name_prefix}_ln1"
    )(y)
    y = tf.keras.layers.ReLU(name=f"{name_prefix}_relu")(y)

    y = tf.keras.layers.Conv2D(
        channels, 3, padding="same", use_bias=False, name=f"{name_prefix}_c2"
    )(y)
    y = tf.keras.layers.LayerNormalization(
        axis=-1, name=f"{name_prefix}_ln2"
    )(y)

    return tf.keras.layers.Add(name=f"{name_prefix}_add")([x, y])


def build_decoder(latent_channels=96, output_channels=3):
    """
    Build and return the full-capacity Keras decoder model.

    Args:
        latent_channels: Must match the encoder's output depth (default 96).
        output_channels: 3 for RGB.

    Returns:
        tf.keras.Model  input:  (B, 16, 16, latent_channels) float32
                        output: (B, 256, 256, 3) float32 in [0, 1]

    Parameter count (latent_channels=96):
        Entry conv:       96×384×9 + 384            ≈   332 K
        9× ResBlock:  9 × 2 × (384×384×9)           ≈  23.8 M
        Upsample convs:   384×192×9 + 192×96×9 + … ≈   0.8 M
        Output head:      32×3×49  + 3              ≈     5 K
        LayerNorm params: negligible
        Total                                       ≈  25.0 M
    """
    inputs = tf.keras.Input(
        shape=(16, 16, latent_channels), name="latents_in"
    )

    # ── Entry projection ────────────────────────────────────────────────────
    x = tf.keras.layers.LayerNormalization(
        axis=-1, name="entry_ln"
    )(inputs)
    x = tf.keras.layers.Conv2D(
        ENTRY_CHANNELS, 3, padding="same", use_bias=False, name="entry_conv"
    )(x)
    x = tf.keras.layers.LayerNormalization(
        axis=-1, name="entry_conv_ln"
    )(x)

    # ── Bottleneck residual blocks at 16×16 ─────────────────────────────────
    for i in range(N_RESIDUAL_BLOCKS):
        x = _residual_block(x, ENTRY_CHANNELS, f"rb_{i}")

    # ── 4 bilinear upsample stages: 16→32→64→128→256 ────────────────────────
    for i, out_ch in enumerate(STAGE_CHANNELS):
        x = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear", name=f"up_{i}"
        )(x)
        x = tf.keras.layers.Conv2D(
            out_ch, 3, padding="same", use_bias=False, name=f"up_conv_{i}"
        )(x)
        x = tf.keras.layers.LayerNormalization(
            axis=-1, name=f"up_ln_{i}"
        )(x)
        x = tf.keras.layers.ReLU(name=f"up_relu_{i}")(x)

    # ── Output head ─────────────────────────────────────────────────────────
    # 7×7 conv matches the original generator's post_pad + Conv2d(filters[-1], 3, 7×7)
    x = tf.keras.layers.Conv2D(
        output_channels, 7, padding="same", use_bias=True, name="output_conv"
    )(x)
    outputs = tf.keras.layers.Activation("sigmoid", name="output_sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="decoder")
    return model


if __name__ == "__main__":
    import numpy as np

    dec = build_decoder()
    dec.summary(line_length=100)

    total_params = sum(
        np.prod(v.shape) for v in dec.trainable_variables
    )
    print(f"\nTrainable parameters: {total_params:,}  (~{total_params/1e6:.1f} M)")

    dummy = np.random.rand(1, 16, 16, 96).astype("float32")
    out = dec(dummy, training=False)
    assert out.shape == (1, 256, 256, 3), f"Shape mismatch: {out.shape}"
    print(f"Decoder output shape: {out.shape}  OK")
