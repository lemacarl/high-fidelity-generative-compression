"""
Numpy-only entropy models for TFLite inference on Raspberry Pi.

Replaces src/compression/hyperprior_model.py and prior_model.py.
These classes build ANS CDF tables from either:
  - Exported factorized prior weights (from trained FactorizedPrior Keras layer)
  - Gaussian scale parameters (predicted by the hyperprior synthesis network)

No PyTorch, no TensorFlow — pure numpy/scipy.
"""

import numpy as np
import scipy.special
import scipy.optimize

PRECISION = 16
TAIL_MASS = 2 ** -8
MIN_SCALE = 0.11
SCALES_MIN = 0.11
SCALES_MAX = 256.0
SCALES_LEVELS = 64


# ---------------------------------------------------------------------------
# CDF / quantile helpers (numpy equivalents of src/helpers/maths.py)
# ---------------------------------------------------------------------------

def _gaussian_cdf(x):
    """Standard Gaussian CDF Φ(x)."""
    return 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))


def _gaussian_quantile(p):
    """Inverse of Φ(p)."""
    return scipy.special.ndtri(p)


def pmf_to_quantized_cdf(pmf, precision=PRECISION):
    """
    Convert a PMF to an integer CDF suitable for ANS coding.

    Mirrors src/helpers/maths.pmf_to_quantized_cdf (numpy version).

    Args:
        pmf:       1-D float32 array summing to ≈ 1.0
        precision: number of bits (CDF values are integers in [0, 2**precision])

    Returns:
        1-D int32 array of length len(pmf)+1 where cdf[-1] == 2**precision
    """
    pmf = np.asarray(pmf, dtype=np.float64)
    # Normalise
    pmf = pmf / pmf.sum()
    # Scale to integer range, keeping at least 1 per symbol
    cdf_float = np.concatenate([[0.0], np.cumsum(pmf)])
    cdf_int = np.floor(cdf_float * (1 << precision) + 0.5).astype(np.int32)
    # Force strictly monotone
    for i in range(1, len(cdf_int)):
        if cdf_int[i] <= cdf_int[i - 1]:
            cdf_int[i] = cdf_int[i - 1] + 1
    # Force last entry to equal 2**precision
    cdf_int[-1] = 1 << precision
    return cdf_int


# ---------------------------------------------------------------------------
# Factorized prior entropy model (for hyperlatents)
# ---------------------------------------------------------------------------

class FactorizedPriorNumpy:
    """
    Numpy inference implementation of the FactorizedPrior Keras layer.

    Weights are loaded from a dict exported by
    `tflite.model.hyperprior.FactorizedPrior.export_weights()`.

    Usage:
        fp = FactorizedPriorNumpy.from_weights(weights_dict)
        # OR
        fp = FactorizedPriorNumpy.from_file("density_weights.npz")
        fp.build_tables()
        # Then pass fp.CDF, fp.CDF_length, fp.CDF_offset to ans_compress/decompress
    """

    def __init__(self, n_channels, H_list, a_list, b_list):
        self.n_channels = n_channels
        self.H = H_list   # list of (C, in, out) arrays — already softplus'd
        self.a = a_list   # list of (C, 1, out)
        self.b = b_list   # list of (C, 1, out)

        self.CDF = None
        self.CDF_length = None
        self.CDF_offset = None

    @classmethod
    def from_weights(cls, weights_dict):
        n_channels = int(weights_dict["n_channels"])
        n_filters = int(weights_dict["n_filters"])
        H_list = [weights_dict[f"H_{i}"] for i in range(n_filters + 1)]
        a_list = [weights_dict[f"a_{i}"] for i in range(n_filters + 1)]
        b_list = [weights_dict[f"b_{i}"] for i in range(n_filters + 1)]
        return cls(n_channels, H_list, a_list, b_list)

    @classmethod
    def from_file(cls, path):
        data = np.load(path, allow_pickle=True)
        return cls.from_weights(dict(data))

    def save(self, path):
        weights = {}
        for i, (H, a, b) in enumerate(zip(self.H, self.a, self.b)):
            weights[f"H_{i}"] = H
            weights[f"a_{i}"] = a
            weights[f"b_{i}"] = b
        weights["n_channels"] = self.n_channels
        weights["n_filters"] = len(self.H) - 1
        np.savez(path, **weights)

    def _cdf_channel(self, x, c):
        """CDF at values x (shape (N,)) for channel c.

        H[i] is (C, in, out); H[i][c] gives (in, out) — 2D — so that
        the matmul stays 2D and logits remains (N, out) throughout,
        avoiding the (1, N, out) shape that makes brentq receive an array
        instead of a scalar.
        """
        logits = np.asarray(x, dtype=np.float64)[:, np.newaxis]  # (N, 1)
        for H_k, a_k, b_k in zip(self.H, self.a, self.b):
            # h[c] → (in, out); a[c] → (1, out); b[c] → (1, out)
            logits = logits @ H_k[c]            # (N, out) — 2-D matmul
            logits = logits + b_k[c]            # broadcast (N, out) + (1, out)
            logits = logits + np.tanh(a_k[c]) * np.tanh(logits)
        return scipy.special.expit(logits[:, 0])  # (N,)

    def build_tables(self, precision=PRECISION, tail_mass=TAIL_MASS):
        """
        Build integer CDF tables for all channels.

        After calling this, `self.CDF`, `self.CDF_length`, `self.CDF_offset`
        are ready to pass to ans_compress / ans_decompress.
        """
        C = self.n_channels
        multiplier = -_gaussian_quantile(tail_mass / 2)

        pmf_centers = []
        for c in range(C):
            # Find tail quantile for this channel using scalar bisection
            try:
                lower_tail = scipy.optimize.brentq(
                    lambda x: self._cdf_channel(np.array([x]), c)[0] - tail_mass / 2,
                    a=-1000.0, b=0.0, xtol=1e-4
                )
                upper_tail = scipy.optimize.brentq(
                    lambda x: self._cdf_channel(np.array([x]), c)[0] - (1 - tail_mass / 2),
                    a=0.0, b=1000.0, xtol=1e-4
                )
                center = int(np.ceil(max(abs(lower_tail), abs(upper_tail))))
            except ValueError:
                center = 10
            pmf_centers.append(center)

        pmf_centers = np.array(pmf_centers, dtype=np.int32)
        pmf_lengths = 2 * pmf_centers + 1
        max_length = int(pmf_lengths.max())

        CDF = np.zeros((C, max_length + 2), dtype=np.int32)
        CDF_length = np.zeros(C, dtype=np.int32)
        CDF_offset = -pmf_centers.astype(np.int32)

        for c in range(C):
            center = pmf_centers[c]
            length = pmf_lengths[c]

            samples = np.arange(-center, center + 1, dtype=np.float64)
            upper = self._cdf_channel(samples + 0.5, c)
            lower = self._cdf_channel(samples - 0.5, c)
            pmf = np.maximum(upper - lower, 0.0)

            # Append tail mass at both ends
            tail_pmf = np.array([2.0 * lower[0]], dtype=np.float64)
            pmf = np.concatenate([pmf, tail_pmf])

            cdf_row = pmf_to_quantized_cdf(pmf, precision)
            # Pad to max_length + 2
            padded = np.zeros(max_length + 2, dtype=np.int32)
            padded[:len(cdf_row)] = cdf_row
            CDF[c] = padded
            CDF_length[c] = length + 2

        self.CDF = CDF.astype(np.uint32)
        self.CDF_length = CDF_length
        self.CDF_offset = CDF_offset


# ---------------------------------------------------------------------------
# Gaussian prior entropy model (for latents, indexed by scale)
# ---------------------------------------------------------------------------

class GaussianPriorNumpy:
    """
    Indexed Gaussian entropy model for latents.

    Precomputes CDF tables for a logarithmic scale table, then selects the
    nearest table entry for each scale value predicted by the hyperprior
    synthesis network.

    Usage:
        gp = GaussianPriorNumpy()
        gp.build_tables()
        indices = gp.compute_indices(sigma_array)  # (B, H, W, C) → (B, H, W, C) int
        # pass gp.CDF, gp.CDF_length, gp.CDF_offset to ans_compress
    """

    def __init__(self, scales_min=SCALES_MIN, scales_max=SCALES_MAX,
                 levels=SCALES_LEVELS, precision=PRECISION, tail_mass=TAIL_MASS):
        self.scales_min = scales_min
        self.scales_max = scales_max
        self.levels = levels
        self.precision = precision
        self.tail_mass = tail_mass

        self.scale_table = np.exp(
            np.linspace(np.log(scales_min), np.log(scales_max), levels)
        ).astype(np.float32)

        self.CDF = None
        self.CDF_length = None
        self.CDF_offset = None

    def build_tables(self):
        multiplier = float(-_gaussian_quantile(self.tail_mass / 2))
        pmf_centers = np.ceil(self.scale_table * multiplier).astype(np.int32)
        pmf_lengths = 2 * pmf_centers + 1
        max_length = int(pmf_lengths.max())

        n_scales = len(self.scale_table)
        CDF = np.zeros((n_scales, max_length + 2), dtype=np.int32)
        CDF_length = np.zeros(n_scales, dtype=np.int32)
        CDF_offset = (-pmf_centers).astype(np.int32)

        for i, (scale, center, length) in enumerate(
            zip(self.scale_table, pmf_centers, pmf_lengths)
        ):
            samples = np.arange(-center, center + 1, dtype=np.float64)
            upper = _gaussian_cdf((0.5 - samples) / float(scale))
            lower = _gaussian_cdf((-0.5 - samples) / float(scale))
            pmf = np.maximum(upper - lower, 0.0)

            tail = 2.0 * _gaussian_cdf(-0.5 / float(scale))
            pmf = np.concatenate([pmf, [tail]])

            cdf_row = pmf_to_quantized_cdf(pmf, self.precision)
            padded = np.zeros(max_length + 2, dtype=np.int32)
            padded[:len(cdf_row)] = cdf_row
            CDF[i] = padded
            CDF_length[i] = int(length) + 2

        self.CDF = CDF.astype(np.uint32)
        self.CDF_length = CDF_length
        self.CDF_offset = CDF_offset

    def compute_indices(self, scales):
        """
        Map predicted sigma values to nearest scale-table index.

        scales: numpy array, any shape, float32
        Returns: int32 array, same shape
        """
        scales = np.clip(np.asarray(scales, dtype=np.float32),
                         self.scales_min, self.scales_max)
        # For each scale find the first table entry >= scale
        indices = np.ones(scales.shape, dtype=np.int32) * (self.levels - 1)
        for j, s in enumerate(self.scale_table[:-1]):
            indices = np.where(scales <= s, np.minimum(indices, j), indices)
        # Ensure valid range
        indices = np.clip(indices, 0, self.levels - 1).astype(np.int32)
        return indices


if __name__ == "__main__":
    gp = GaussianPriorNumpy()
    gp.build_tables()
    print(f"GaussianPrior CDF shape: {gp.CDF.shape}")
    scales = np.random.uniform(0.1, 10.0, (1, 4, 4, 128)).astype(np.float32)
    idx = gp.compute_indices(scales)
    print(f"Indices range: [{idx.min()}, {idx.max()}]  shape: {idx.shape}  OK")
