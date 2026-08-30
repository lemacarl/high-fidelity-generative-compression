"""
LPIPS (Zhang et al. 2018) in pure TensorFlow — no PyTorch at evaluation time.

Why this exists: PSNR and MS-SSIM are both full-reference DISTORTION measures.
At a fixed bitrate the bits needed to reproduce the original exactly are not
there, so a non-GAN decoder emits the conditional mean of everything the
latents could have come from — which is blur, because averaging minimises
squared error under uncertainty. A GAN decoder emits a sample instead: sharp,
plausible texture that will not align pixel for pixel. Rate-distortion-
perception theory makes that a hard tradeoff, so a GAN that is working
correctly scores WORSE on PSNR and MS-SSIM — and so does a GAN that is broken.
The two are indistinguishable on distortion metrics.

LPIPS is still full-reference, but it compares deep features instead of local
windows, so it does not collapse to "blur wins" and tracks human preference on
generative output far better. Lower is better.

Weights come from tflite/lpips_weights/vgg16_lpips.npz, produced once by
tflite/convert_lpips_weights.py from torchvision VGG16 plus the LPIPS linear
weights vendored at src/loss/perceptual_similarity/weights/v0.1/vgg.pth. The
backbone must be torchvision's: LPIPS feeds the network through a scaling
layer that, for input 2x-1, computes (x - 0.485) / 0.229 — precisely
torchvision's ImageNet normalisation. tf.keras.applications.VGG16 expects
Caffe-style BGR preprocessing and would produce features the linear weights
were never calibrated against.

Usage:
    from tflite.lpips import LPIPS
    metric = LPIPS()                      # builds once, reuse across images
    d = metric(x, x_hat)                  # (B,H,W,3) float32 in [0,1] -> (B,)
"""

import os

import numpy as np
import tensorflow as tf

DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "lpips_weights", "vgg16_lpips.npz"
)

# VGG16 `features`: conv+ReLU counts per block, 2x2 maxpool between blocks.
# LPIPS taps the final ReLU of each block (relu1_2, relu2_2, relu3_3,
# relu4_3, relu5_3).
BLOCKS = (2, 2, 3, 3, 3)

# ScalingLayer constants from the reference implementation.
_SHIFT = np.array([-0.030, -0.088, -0.188], dtype=np.float32)
_SCALE = np.array([0.458, 0.448, 0.450], dtype=np.float32)


class LPIPS:
    """Callable LPIPS metric. Build once; calling is cheap."""

    def __init__(self, weights_path=DEFAULT_WEIGHTS):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"LPIPS weights not found: {weights_path}\n"
                "Generate them once with PyTorch installed:\n"
                "  python -m tflite.convert_lpips_weights"
            )
        w = np.load(weights_path, allow_pickle=False)
        self._convs = []
        i = 0
        for n_conv in BLOCKS:
            block = []
            for _ in range(n_conv):
                block.append((tf.constant(w[f"conv{i}_w"]),
                              tf.constant(w[f"conv{i}_b"])))
                i += 1
            self._convs.append(block)
        self._lins = [tf.constant(w[f"lin{k}"]) for k in range(len(BLOCKS))]
        self.shift = tf.constant(_SHIFT.reshape(1, 1, 1, 3))
        self.scale = tf.constant(_SCALE.reshape(1, 1, 1, 3))

    # -- internals ----------------------------------------------------------

    def _features(self, x):
        """x: (B,H,W,3) already scaled. Returns the 5 tapped feature maps."""
        taps = []
        h = x
        for b, block in enumerate(self._convs):
            for kernel, bias in block:
                h = tf.nn.conv2d(h, kernel, strides=1, padding="SAME")
                h = tf.nn.relu(tf.nn.bias_add(h, bias))
            taps.append(h)                       # post-ReLU, before pooling
            if b < len(self._convs) - 1:
                h = tf.nn.max_pool2d(h, ksize=2, strides=2, padding="VALID")
        return taps

    @staticmethod
    def _unit_normalize(feat, eps=1e-10):
        """L2-normalise across channels, matching perceptual_loss.normalize_tensor."""
        norm = tf.sqrt(tf.reduce_sum(feat ** 2, axis=-1, keepdims=True) + eps)
        return feat / norm

    # -- public -------------------------------------------------------------

    def __call__(self, x, x_hat):
        """
        Args:
            x, x_hat: (B,H,W,3) float32 in [0, 1]
        Returns:
            (B,) float32 — LPIPS distance, lower is better
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x_hat = tf.convert_to_tensor(x_hat, dtype=tf.float32)

        # [0,1] -> [-1,1], then the LPIPS scaling layer.
        a = ((2.0 * x - 1.0) - self.shift) / self.scale
        b = ((2.0 * x_hat - 1.0) - self.shift) / self.scale

        total = None
        for fa, fb, lin in zip(self._features(a), self._features(b), self._lins):
            diff = (self._unit_normalize(fa) - self._unit_normalize(fb)) ** 2
            # 1x1 conv with the learned per-channel weights == weighted sum
            # over channels; then average spatially.
            weighted = tf.reduce_sum(diff * lin, axis=-1)          # (B,H,W)
            layer = tf.reduce_mean(weighted, axis=[1, 2])          # (B,)
            total = layer if total is None else total + layer
        return total


_DEFAULT = None


def lpips(x, x_hat):
    """Convenience wrapper using a lazily-built shared metric."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LPIPS()
    return _DEFAULT(x, x_hat)
