"""
Full Keras compression model for training.

Combines encoder, hyperprior, and decoder into a single model whose
call() returns (reconstruction, hyperlatent_bpp, latent_bpp).

Quantization strategy:
  - Training:   additive uniform noise  ∈ [-0.5, 0.5]  (continuous relaxation)
  - Evaluation: hard rounding  (straight-through estimator via @tf.custom_gradient)

Rate estimation:
  - Hyperlatents: factorized non-parametric density (FactorizedPrior layer)
  - Latents:      conditional Gaussian p(y | μ, σ)  from hyperprior synthesis
"""

import tensorflow as tf

from tflite.model.encoder import build_encoder, get_backbone_vars, get_projection_vars
from tflite.model.decoder import build_decoder
from tflite.model.hyperprior import (
    build_hyper_encoder, build_hyper_decoder, FactorizedPrior, MIN_SCALE
)

LATENT_CHANNELS = 96
HYPER_CHANNELS = 128


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

@tf.custom_gradient
def _round_ste(x):
    """Hard rounding with straight-through gradient estimator."""
    def grad(dy):
        return dy
    return tf.round(x), grad


def quantize(x, mode="noise", means=None):
    """
    Quantize tensor x.

    mode='noise':    add U[-0.5, 0.5] (used during training)
    mode='quantize': hard round (used during evaluation / export)
    """
    if mode == "noise":
        noise = tf.random.uniform(tf.shape(x), -0.5, 0.5)
        return x + noise
    elif mode == "quantize":
        if means is not None:
            return _round_ste(x - means) + means
        return _round_ste(x)
    raise ValueError(f"Unknown quantize mode: {mode}")


# ---------------------------------------------------------------------------
# Rate helpers
# ---------------------------------------------------------------------------

def gaussian_log_likelihood(y, mu, sigma):
    """
    log P(y | μ, σ) under a unit-width discrete Gaussian (integral over ±0.5).

    Returns: per-element log-probability tensor (same shape as y).
    """
    half = 0.5
    sigma = tf.maximum(sigma, MIN_SCALE)
    upper = (y - mu + half) / sigma
    lower = (y - mu - half) / sigma
    # log(Φ(upper) - Φ(lower)) stably via log_ndtr if available, else manual
    log_upper = _log_ndtr(upper)
    log_lower = _log_ndtr(lower)
    # log(exp(log_upper) - exp(log_lower))  — numerically stable version
    log_prob = log_upper + tf.math.log(
        tf.maximum(1.0 - tf.exp(log_lower - log_upper), 1e-9)
    )
    return log_prob


def _log_ndtr(x):
    """log(Φ(x)) — log of the standard Gaussian CDF."""
    # For large positive x: log_ndtr ≈ log(1) = 0
    # For large negative x: log_ndtr ≈ -x²/2 - log(-x) - 0.5*log(2π)
    # Use half-erfc for numerical stability
    return tf.math.log(0.5) + tf.math.log(
        tf.maximum(1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)), 1e-9)
    )


def bits_per_pixel(log_likelihood_sum, spatial_pixels):
    """Convert sum of log-likelihoods (nats) to bits per pixel."""
    return -log_likelihood_sum / (spatial_pixels * tf.math.log(2.0))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CompressionModel(tf.keras.Model):
    """
    End-to-end trainable compression model.

    call(x, training=True)  → x_hat, hyper_bpp, latent_bpp
    """

    def __init__(self,
                 latent_channels=LATENT_CHANNELS,
                 hyper_channels=HYPER_CHANNELS,
                 freeze_backbone=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_channels = latent_channels
        self.hyper_channels = hyper_channels

        self.encoder = build_encoder(
            latent_channels=latent_channels,
            freeze_backbone=freeze_backbone,
        )
        self.decoder = build_decoder(latent_channels=latent_channels)
        self.hyper_encoder = build_hyper_encoder(
            latent_channels=latent_channels, hyper_channels=hyper_channels
        )
        self.hyper_decoder = build_hyper_decoder(
            latent_channels=latent_channels, hyper_channels=hyper_channels
        )
        self.factorized_prior = FactorizedPrior(
            n_channels=hyper_channels, name="factorized_prior"
        )

    def call(self, x, training=True):
        """
        Args:
            x:        (B, 256, 256, 3) float32 in [0, 1]
            training: if True, use noise quantization; else hard quantize

        Returns:
            x_hat:      (B, 256, 256, 3) reconstruction
            hyper_bpp:  scalar — hyperlatent bits per pixel
            latent_bpp: scalar — latent bits per pixel
        """
        quant_mode = "noise" if training else "quantize"
        spatial_pixels = tf.cast(
            tf.shape(x)[1] * tf.shape(x)[2], tf.float32
        )

        # --- Encoder ---
        y = self.encoder(x, training=training)  # (B,16,16,C)

        # --- Hyperprior analysis ---
        z = self.hyper_encoder(y, training=training)  # (B,4,4,N)

        # --- Quantize hyperlatents ---
        z_hat = quantize(z, mode=quant_mode)

        # --- Hyperlatent rate ---
        log_p_z = self.factorized_prior(z_hat)  # (B,4,4,N) log-probs (nats)
        hyper_bpp = bits_per_pixel(
            tf.reduce_sum(log_p_z), spatial_pixels * tf.cast(tf.shape(x)[0], tf.float32)
        )

        # --- Hyperprior synthesis: predict latent distribution ---
        mu, sigma = self.hyper_decoder(z_hat, training=training)  # (B,16,16,C) each

        # --- Quantize latents (means-centred) ---
        y_hat = quantize(y, mode=quant_mode, means=mu)  # (B,16,16,C)

        # --- Latent rate ---
        log_p_y = gaussian_log_likelihood(y_hat, mu, sigma)  # (B,16,16,C)
        latent_bpp = bits_per_pixel(
            tf.reduce_sum(log_p_y), spatial_pixels * tf.cast(tf.shape(x)[0], tf.float32)
        )

        # --- Decoder ---
        x_hat = self.decoder(y_hat, training=training)  # (B,256,256,3)

        return x_hat, hyper_bpp, latent_bpp

    def get_variable_groups(self):
        """
        Return (amortization_vars, entropy_vars) for separate optimizers.

        amortization_vars: encoder (backbone + projection), decoder, hyper nets
        entropy_vars:      factorized prior density parameters
        """
        amort = (
            self.encoder.trainable_variables
            + self.decoder.trainable_variables
            + self.hyper_encoder.trainable_variables
            + self.hyper_decoder.trainable_variables
        )
        entropy = self.factorized_prior.trainable_variables
        return amort, entropy

    def get_encoder_subgroups(self):
        """Return (backbone_vars, projection_vars) for two-LR backbone fine-tuning."""
        return (
            get_backbone_vars(self.encoder),
            get_projection_vars(self.encoder),
        )

    def export_factorized_prior_weights(self):
        """Return density weights as numpy dict for ANS table building at inference."""
        return self.factorized_prior.export_weights()


if __name__ == "__main__":
    import numpy as np

    model = CompressionModel()
    x = np.random.rand(2, 256, 256, 3).astype("float32")

    # Training mode
    x_hat, h_bpp, l_bpp = model(x, training=True)
    assert x_hat.shape == (2, 256, 256, 3), f"x_hat shape: {x_hat.shape}"
    total_bpp = (h_bpp + l_bpp).numpy()
    print(f"Training forward pass OK. x_hat: {x_hat.shape}  total bpp: {total_bpp:.4f}")

    # Evaluation mode
    x_hat_eval, h_bpp_e, l_bpp_e = model(x, training=False)
    print(f"Eval forward pass OK. x_hat: {x_hat_eval.shape}  total bpp: {(h_bpp_e+l_bpp_e).numpy():.4f}")

    amort, entropy = model.get_variable_groups()
    print(f"Amortization params: {sum(v.shape.num_elements() for v in amort):,}")
    print(f"Entropy params:      {sum(v.shape.num_elements() for v in entropy):,}")
