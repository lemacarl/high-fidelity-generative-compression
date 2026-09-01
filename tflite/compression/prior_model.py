"""
Numpy-only prior model — wraps GaussianPriorNumpy with compress/decompress
methods that interface directly with the ANS codec.

This is the TFLite-path equivalent of src/compression/prior_model.py.
"""

import numpy as np
from tflite.compression.entropy_models import GaussianPriorNumpy, MIN_SCALE
from tflite.compression import compression_utils, entropy_coding


class PriorModel:
    """
    Compress/decompress latents using a conditional Gaussian entropy model.

    The scale (sigma) parameters are predicted by the hyperprior synthesis
    TFLite model and passed in at compress/decompress time.
    """

    def __init__(self, precision=16, scales_min=None):
        """
        scales_min: lower bound of the scale table. Defaults to the module's
            SCALES_MIN (0.11), which matches MIN_SCALE in model/hyperprior.py.

            Worth tuning. The hyperprior saturates that floor — measured
            median sigma is 0.110 for both low_gan_v7 and low_gan_v8 — and at
            sigma=0.11 the PMF over the +/-1 support is so peaked that a
            single residual of 1 costs 18.5 bits and an escape costs 17.5.
            Raising the floor widens every PMF: zeros stop being free, but
            nonzero residuals stop being catastrophic. For v8 that trade is
            heavily favourable (0.2959 -> ~0.109 bpp re-priced at 0.25), and
            it is LOSSLESS: the quantized latents are untouched, so the
            reconstruction is bit-identical. Only the bitstream shrinks.

            The optimum is model-dependent — a model with almost no nonzero
            residuals pays for the wider PMF and gains nothing — so sweep it
            per checkpoint rather than assuming a global best.

            The value is NOT recorded in the .hfc container, so compress and
            decompress must be given the same one or decoding will desync.
        """
        self.precision = precision
        self.scales_min = scales_min
        self._gaussian = (GaussianPriorNumpy() if scales_min is None
                          else GaussianPriorNumpy(scales_min=scales_min))
        self._gaussian.build_tables()

    def compress(self, y_hat, mu, sigma, vectorize=False, block_encode=True):
        """
        Entropy-encode quantized latents y_hat.

        Args:
            y_hat:  (1, H, W, C) float32  — already quantized (rounded)
            mu:     (1, H, W, C) float32  — predicted means
            sigma:  (1, H, W, C) float32  — predicted scales (> 0)
            vectorize, block_encode: passed through to ANS coder

        Returns:
            encoded:       ANS bitstring (numpy uint32 array)
            coding_shape:  tuple used by decoder
        """
        # Convert to (N, C, H, W) for ANS coder (channel-first)
        y_hat = np.transpose(y_hat, (0, 3, 1, 2))   # NHWC → NCHW
        mu    = np.transpose(mu,    (0, 3, 1, 2))
        sigma = np.transpose(sigma, (0, 3, 1, 2))

        sigma = np.clip(sigma, MIN_SCALE, None)

        # Map sigma to scale-table indices
        indices = self._gaussian.compute_indices(sigma)

        # Symbols: integer offsets from mean (the quantised residual)
        symbols = np.floor(y_hat - mu + 0.5).astype(np.int32)

        input_shape = y_hat.shape          # (N, C, H, W)
        coding_shape = input_shape[1:]     # (C, H, W)

        encoded, coding_shape = compression_utils.ans_compress(
            symbols=symbols,
            indices=indices,
            cdf=self._gaussian.CDF,
            cdf_length=self._gaussian.CDF_length,
            cdf_offset=self._gaussian.CDF_offset,
            coding_shape=coding_shape,
            precision=self.precision,
            vectorize=vectorize,
            block_encode=block_encode,
        )
        return encoded, coding_shape

    def decompress(self, encoded, mu, sigma, coding_shape,
                   vectorize=False, block_decode=True):
        """
        Entropy-decode latents.

        Args:
            encoded:       ANS bitstring
            mu:            (1, H, W, C) float32
            sigma:         (1, H, W, C) float32
            coding_shape:  tuple from compress()

        Returns:
            y_hat: (1, H, W, C) float32 — dequantized latents
        """
        mu_nchw    = np.transpose(mu,    (0, 3, 1, 2))
        sigma_nchw = np.transpose(sigma, (0, 3, 1, 2))
        sigma_nchw = np.clip(sigma_nchw, MIN_SCALE, None)

        indices = self._gaussian.compute_indices(sigma_nchw)

        symbols_shape = mu_nchw.shape  # (N, C, H, W)

        decoded = compression_utils.ans_decompress(
            encoded=encoded,
            indices=indices,
            cdf=self._gaussian.CDF,
            cdf_length=self._gaussian.CDF_length,
            cdf_offset=self._gaussian.CDF_offset,
            coding_shape=coding_shape,
            precision=self.precision,
            vectorize=vectorize,
            block_decode=block_decode,
        )
        decoded = np.reshape(decoded, symbols_shape).astype(np.float32)
        # Dequantize: y_hat = symbol + mean
        y_hat_nchw = decoded + mu_nchw
        # Back to NHWC
        y_hat = np.transpose(y_hat_nchw, (0, 2, 3, 1))
        return y_hat
