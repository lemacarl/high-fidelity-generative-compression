"""
MobileNetV3 Small-based encoder for TFLite generative compression.

Architecture:
  MobileNetV3Small (pretrained ImageNet) → 16×16 feature map
  → Conv2D(1×1) projection → latents y of shape (B, 16, 16, C)

The backbone is fine-tuned at a lower LR (1/10th) while the projection
head is trained from scratch at the base LR.
"""

import tensorflow as tf

LATENT_CHANNELS = 96


def _find_last_16x16_layer(base_model):
    """Return the last layer that outputs a 16×16 spatial map for 256×256 input."""
    target = None
    for layer in base_model.layers:
        try:
            shape = layer.output.shape
            if len(shape) == 4 and shape[1] == 16 and shape[2] == 16:
                target = layer
        except AttributeError:
            pass
    if target is None:
        raise RuntimeError(
            "Could not find a 16×16 feature layer in MobileNetV3Small. "
            "Check that input_shape=(256,256,3) is set correctly."
        )
    return target


def build_encoder(latent_channels=LATENT_CHANNELS,
                  input_shape=(256, 256, 3),
                  freeze_backbone=False):
    """
    Build and return the Keras encoder model.

    Args:
        latent_channels: Number of output channels for the latent code.
        input_shape: HWC input dimensions (must be 256×256×3).
        freeze_backbone: If True, MobileNetV3 weights are frozen.

    Returns:
        tf.keras.Model  input: (B,H,W,3) float32 [0,1]
                        output: (B,16,16,latent_channels) float32
    """
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,
    )

    feature_layer = _find_last_16x16_layer(base)
    features = feature_layer.output  # (None, 16, 16, C_backbone)

    # Project backbone features to latent space — no activation so latents
    # can take any real value before quantization.
    latents = tf.keras.layers.Conv2D(
        latent_channels,
        kernel_size=1,
        padding="same",
        use_bias=True,
        name="latent_projection",
    )(features)

    if freeze_backbone:
        for layer in base.layers:
            layer.trainable = False

    model = tf.keras.Model(inputs=base.input, outputs=latents, name="encoder")
    return model


def get_backbone_vars(encoder_model):
    """Return variables belonging to the MobileNetV3 backbone (for separate LR)."""
    proj_name = "latent_projection"
    return [v for v in encoder_model.trainable_variables
            if proj_name not in v.name]


def get_projection_vars(encoder_model):
    """Return variables belonging only to the projection head."""
    proj_name = "latent_projection"
    return [v for v in encoder_model.trainable_variables
            if proj_name in v.name]


if __name__ == "__main__":
    enc = build_encoder()
    enc.summary(line_length=100)
    import numpy as np
    dummy = np.random.rand(1, 256, 256, 3).astype("float32")
    out = enc(dummy, training=False)
    assert out.shape == (1, 16, 16, LATENT_CHANNELS), f"Shape mismatch: {out.shape}"
    print(f"Encoder output shape: {out.shape}  OK")
