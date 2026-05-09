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

    def __init__(self, precision=16):
        self.precision = precision
        self._gaussian = GaussianPriorNumpy()
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
