"""
Full-capacity hyperprior for TFLite generative compression.

Mirrors the Ballé et al. (2018) scale hyperprior architecture but uses
bilinear upsample + Conv2D in the synthesis path (instead of ConvTranspose2D)
so that INT8 TFLite quantization works correctly on ARM.

Scaled to HYPER_CHANNELS = 192 (up from 128) to better support the
full-capacity 25 M-param decoder by providing sharper (mu, sigma)
predictions over the 96-channel latents.

Three Keras models are exposed:
  build_hyper_encoder  – latents y  → hyperlatents z   (for compression)
  build_hyper_decoder  – hyperlatents z* → (mu, sigma)  (for decompression)

Additionally, a FactorizedPrior layer (Keras Layer) is provided for in-graph
rate estimation during training.
"""

import tensorflow as tf
import numpy as np

HYPER_CHANNELS = 192    # N in analysis / synthesis networks (was 128)
MIN_SCALE = 0.11        # minimum sigma for numerical stability


# ---------------------------------------------------------------------------
# Analysis network: latents → hyperlatents
# ---------------------------------------------------------------------------

def build_hyper_encoder(latent_channels=96, hyper_channels=HYPER_CHANNELS):
    """
    HyperpriorAnalysis: y (B,16,16,C) → z (B,4,4,N)

    Two stride-2 convolutions produce a 4× spatial downsampling.
    """
    inputs = tf.keras.Input(shape=(16, 16, latent_channels), name="hyper_enc_in")

    x = tf.keras.layers.Conv2D(
        hyper_channels, 3, padding="same", activation="relu", name="ha_conv1"
    )(inputs)
    x = tf.keras.layers.Conv2D(
        hyper_channels, 5, strides=2, padding="same", activation="relu", name="ha_conv2"
    )(x)
    x = tf.keras.layers.Conv2D(
        hyper_channels, 5, strides=2, padding="same", name="ha_conv3"
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="hyper_encoder")


# ---------------------------------------------------------------------------
# Synthesis network: hyperlatents → (mu, sigma)
# ---------------------------------------------------------------------------

def build_hyper_decoder(latent_channels=96, hyper_channels=HYPER_CHANNELS):
    """
    HyperpriorSynthesis: z* (B,4,4,N) → mu (B,16,16,C), sigma (B,16,16,C)

    Bilinear upsample + Conv2D used instead of ConvTranspose2D for INT8
    TFLite compatibility on ARM.

    Shared trunk, two separate output heads (saves ~50 % params vs dual nets).
    """
    inputs = tf.keras.Input(shape=(4, 4, hyper_channels), name="hyper_dec_in")

    # Shared trunk: 4×4 → 8×8 → 16×16
    x = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear", name="hs_up1"
    )(inputs)
    x = tf.keras.layers.Conv2D(
        hyper_channels, 5, padding="same", activation="relu", name="hs_conv1"
    )(x)
    x = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear", name="hs_up2"
    )(x)
    x = tf.keras.layers.Conv2D(
        hyper_channels, 5, padding="same", activation="relu", name="hs_conv2"
    )(x)

    # Mean head (unconstrained)
    mu = tf.keras.layers.Conv2D(
        latent_channels, 3, padding="same", name="hs_mu"
    )(x)

    # Scale head: softplus ensures sigma > 0; offset by MIN_SCALE
    sigma_raw = tf.keras.layers.Conv2D(
        latent_channels, 3, padding="same", name="hs_sigma_raw"
    )(x)
    sigma = tf.keras.layers.Lambda(
        lambda s: tf.nn.softplus(s) + MIN_SCALE, name="hs_sigma"
    )(sigma_raw)

    return tf.keras.Model(
        inputs=inputs, outputs=[mu, sigma], name="hyper_decoder"
    )


# ---------------------------------------------------------------------------
# Factorized prior (non-parametric density over hyperlatents)
# ---------------------------------------------------------------------------

class FactorizedPrior(tf.keras.layers.Layer):
    """
    Trainable factorized density model for hyperlatents.

    Approximates the marginal distribution p(z) with a product of
    per-channel non-parametric densities using a learned CDF network
    (4 layers of linear + tanh gating), following Ballé et al. (2018).

    Used in-graph during training for rate estimation.
    At inference time, the trained weights are exported as numpy arrays
    and used to build ANS CDF tables in tflite/compression/entropy_models.py.
    """

    def __init__(self, n_channels, n_filters=3, init_scale=10.0, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.init_scale = init_scale

    def build(self, input_shape):
        scale = self.init_scale ** (1.0 / (self.n_filters + 1))
        self._H = []
        self._a = []
        self._b = []

        sizes = [1] + [self.n_filters] * self.n_filters + [1]
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            H_init = np.random.uniform(-limit, limit,
                                       size=(self.n_channels, fan_in, fan_out)).astype("float32")
            self._H.append(
                self.add_weight(name=f"H_{i}", shape=H_init.shape,
                                initializer=tf.constant_initializer(H_init),
                                trainable=True)
            )
            self._a.append(
                self.add_weight(name=f"a_{i}", shape=(self.n_channels, 1, fan_out),
                                initializer="zeros", trainable=True)
            )
            self._b.append(
                self.add_weight(name=f"b_{i}", shape=(self.n_channels, 1, fan_out),
                                initializer=tf.constant_initializer(
                                    np.log(np.expm1(1.0 / scale / fan_out))
                                ),
                                trainable=True)
            )
        super().build(input_shape)

    def _logits_cumulative(self, x):
        """Evaluate logit(CDF(x)) for each channel independently."""
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], self.n_channels
        x_t = tf.transpose(x, [3, 0, 1, 2])          # (C, B, H, W)
        x_t = tf.reshape(x_t, [C, -1, 1])             # (C, N, 1)

        logits = x_t
        for H_k, a_k, b_k in zip(self._H, self._a, self._b):
            logits = tf.linalg.matmul(logits, tf.nn.softplus(H_k))
            logits = logits + b_k
            logits = logits + tf.tanh(a_k) * tf.tanh(logits)
            logits = tf.clip_by_value(logits, -30.0, 30.0)
        return logits  # (C, N, 1)

    def log_likelihood(self, x):
        """
        Estimate log P(x) for rate computation.
        Uses sigmoid(upper) - sigmoid(lower) — numerically stable.

        Returns: (B, H, W, C) tensor of per-element log probabilities (nats).
        """
        shape = tf.shape(x)
        B, H, W = shape[0], shape[1], shape[2]
        C = self.n_channels

        upper = self._logits_cumulative(x + 0.5)   # (C, N, 1)
        lower = self._logits_cumulative(x - 0.5)   # (C, N, 1)

        likelihood = tf.maximum(tf.sigmoid(upper) - tf.sigmoid(lower), 1e-9)
        log_prob = tf.math.log(likelihood)          # (C, N, 1)

        log_prob = tf.reshape(log_prob, [C, B, H, W])
        log_prob = tf.transpose(log_prob, [1, 2, 3, 0])   # (B, H, W, C)
        return log_prob

    def call(self, x):
        return self.log_likelihood(x)

    def export_weights(self):
        """Return density parameters as a dict of numpy arrays for ANS table building."""
        weights = {}
        for i, (H_k, a_k, b_k) in enumerate(zip(self._H, self._a, self._b)):
            weights[f"H_{i}"] = tf.nn.softplus(H_k).numpy()
            weights[f"a_{i}"] = a_k.numpy()
            weights[f"b_{i}"] = b_k.numpy()
        weights["n_channels"] = self.n_channels
        weights["n_filters"] = self.n_filters
        return weights


if __name__ == "__main__":
    import numpy as np

    he = build_hyper_encoder()
    hd = build_hyper_decoder()

    dummy_y = np.random.randn(1, 16, 16, 96).astype("float32")
    z = he(dummy_y)
    assert z.shape == (1, 4, 4, HYPER_CHANNELS), f"Hyper-encoder shape: {z.shape}"

    mu, sigma = hd(z)
    assert mu.shape == (1, 16, 16, 96), f"mu shape: {mu.shape}"
    assert sigma.shape == (1, 16, 16, 96), f"sigma shape: {sigma.shape}"
    assert (sigma.numpy() > 0).all(), "sigma must be positive"
    print(f"Hyper-encoder out: {z.shape}  mu: {mu.shape}  sigma: {sigma.shape}  OK")

    fp = FactorizedPrior(n_channels=HYPER_CHANNELS, name="factorized_prior")
    dummy_z = np.random.randn(2, 4, 4, HYPER_CHANNELS).astype("float32")
    log_p = fp(dummy_z)
    assert log_p.shape == dummy_z.shape, f"FactorizedPrior shape: {log_p.shape}"
    print(f"FactorizedPrior log_likelihood shape: {log_p.shape}  OK")
