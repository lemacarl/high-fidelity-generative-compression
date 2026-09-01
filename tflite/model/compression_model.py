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

import os

import numpy as np
import tensorflow as tf

from tflite.model.encoder import build_encoder, get_backbone_vars, get_projection_vars
from tflite.model.decoder import build_decoder
from tflite.model.hyperprior import (
    build_hyper_encoder, build_hyper_decoder, FactorizedPrior, MIN_SCALE
)

LATENT_CHANNELS = 96
HYPER_CHANNELS = 192  # scaled up from 128 to support full-capacity 25M decoder


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

    Uses erfc for numerical stability — avoids log(0) and log(negative) entirely.
    """
    sigma = tf.maximum(sigma, MIN_SCALE)
    # Φ(x) = 0.5 * erfc(-x / sqrt(2))  — stable for all x
    upper = _gaussian_cdf((y - mu + 0.5) / sigma)
    lower = _gaussian_cdf((y - mu - 0.5) / sigma)
    likelihood = tf.maximum(upper - lower, 1e-9)
    return tf.math.log(likelihood)


def _gaussian_cdf(x):
    """Φ(x) via erfc — numerically stable for large |x|."""
    return 0.5 * tf.math.erfc(-x / tf.cast(tf.sqrt(2.0), x.dtype))


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

    def call(self, x, training=True, return_latents=False,
             quant_mode=None, norm_training=None):
        """
        Args:
            x:              (B, 256, 256, 3) float32 in [0, 1]
            training:       if True, use noise quantization; else hard quantize
            return_latents: if True, also return y_hat for discriminator conditioning
            quant_mode:     override quantization ("noise" | "quantize"); defaults
                            to the value implied by `training`
            norm_training:  override the `training` flag passed to the sub-networks
                            (BatchNorm batch-stats vs moving-average mode);
                            defaults to `training`. Kept separate from
                            `quant_mode` so the two can be measured
                            independently.

        Returns:
            x_hat:      (B, 256, 256, 3) reconstruction
            hyper_bpp:  scalar — hyperlatent bits per pixel
            latent_bpp: scalar — latent bits per pixel
            y_hat:      (B, 16, 16, C) quantized latents  [only when return_latents=True]
        """
        quant_mode = quant_mode or ("noise" if training else "quantize")
        norm_training = training if norm_training is None else norm_training
        spatial_pixels = tf.cast(
            tf.shape(x)[1] * tf.shape(x)[2], tf.float32
        )

        # --- Encoder ---
        y = self.encoder(x, training=norm_training)  # (B,16,16,C)
        y = tf.clip_by_value(y, -20.0, 20.0)

        # --- Hyperprior analysis ---
        z = self.hyper_encoder(y, training=norm_training)  # (B,4,4,N)
        z = tf.clip_by_value(z, -20.0, 20.0)

        # --- Quantize hyperlatents ---
        z_hat = quantize(z, mode=quant_mode)

        # --- Hyperlatent rate ---
        log_p_z = self.factorized_prior(z_hat)  # (B,4,4,N) log-probs (nats)
        # Clip floor is -20 nats = 28.9 bits/symbol, not -10 (14.43 bits).
        # The ANS coder charges 18.48 bits for a |residual|=1 at sigma=0.11 and
        # 17.48 for a tail escape, both ABOVE the old -10 floor — and
        # clip_by_value zeroes the gradient outside its range, so the encoder
        # and hyperprior received no signal at all to avoid the only symbols
        # that actually cost bits. Measured on low_gan_v8: ~4.3% of latents sat
        # in that dead zone and accounted for ~19.4k of the 19.4k bits in the
        # coded latent stream, giving coded/model = 1.92x against v7's 1.17x.
        # It also let sigma park permanently on the MIN_SCALE floor, since
        # overconfidence carried no penalty. -20 is just inside the 1e-9
        # likelihood guard in gaussian_log_likelihood (log(1e-9) = -20.72), so
        # the numerical protection that clip was there for is unchanged.
        log_p_z = tf.clip_by_value(log_p_z, -20.0, 0.0)
        hyper_bpp = bits_per_pixel(
            tf.reduce_sum(log_p_z), spatial_pixels * tf.cast(tf.shape(x)[0], tf.float32)
        )

        # --- Hyperprior synthesis: predict latent distribution ---
        mu, sigma = self.hyper_decoder(z_hat, training=norm_training)  # (B,16,16,C) each

        # --- Quantize latents (means-centred) ---
        y_hat = quantize(y, mode=quant_mode, means=mu)  # (B,16,16,C)

        # --- Latent rate ---
        log_p_y = gaussian_log_likelihood(y_hat, mu, sigma)  # (B,16,16,C)
        # Clip floor is -20 nats = 28.9 bits/symbol, not -10 (14.43 bits).
        # The ANS coder charges 18.48 bits for a |residual|=1 at sigma=0.11 and
        # 17.48 for a tail escape, both ABOVE the old -10 floor — and
        # clip_by_value zeroes the gradient outside its range, so the encoder
        # and hyperprior received no signal at all to avoid the only symbols
        # that actually cost bits. Measured on low_gan_v8: ~4.3% of latents sat
        # in that dead zone and accounted for ~19.4k of the 19.4k bits in the
        # coded latent stream, giving coded/model = 1.92x against v7's 1.17x.
        # It also let sigma park permanently on the MIN_SCALE floor, since
        # overconfidence carried no penalty. -20 is just inside the 1e-9
        # likelihood guard in gaussian_log_likelihood (log(1e-9) = -20.72), so
        # the numerical protection that clip was there for is unchanged.
        log_p_y = tf.clip_by_value(log_p_y, -20.0, 0.0)
        latent_bpp = bits_per_pixel(
            tf.reduce_sum(log_p_y), spatial_pixels * tf.cast(tf.shape(x)[0], tf.float32)
        )

        # --- Decoder ---
        x_hat = self.decoder(y_hat, training=norm_training)  # (B,256,256,3)

        if return_latents:
            return x_hat, hyper_bpp, latent_bpp, y_hat
        return x_hat, hyper_bpp, latent_bpp

    def freeze_batchnorm(self):
        """
        Put every encoder BatchNorm layer into inference mode permanently.

        The MobileNetV3 backbone is the only part of this model whose behavior
        depends on the `training` flag; the decoder and hyper-nets use
        LayerNorm, which does not. Setting `trainable = False` makes Keras run
        the layer against its moving statistics in both modes.

        Note this locks in whatever statistics are currently stored, so
        recalibrate first if the running averages may be stale.

        Returns:
            int — number of BatchNorm layers frozen
        """
        n = 0
        for layer in self.encoder.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                n += 1
        return n

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

    def load_factorized_prior_weights(self, source):
        """
        Load the factorized prior from a density_weights.npz path or dict.

        Checkpoints written before the prior was tracked in the object graph do
        not contain it, so restoring one leaves the prior at random
        initialisation — which makes the rate estimate meaningless while
        leaving reconstructions untouched. Call this after restore when
        evaluating such a checkpoint.

        Returns:
            int — number of variables assigned
        """
        if isinstance(source, (str, bytes, os.PathLike)):
            source = dict(np.load(source, allow_pickle=True))
        return self.factorized_prior.load_exported_weights(source)

    def export_factorized_prior_weights(self):
        """Return density weights as numpy dict for ANS table building at inference."""
        return self.factorized_prior.export_weights()


if __name__ == "__main__":
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
