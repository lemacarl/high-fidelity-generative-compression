"""
Thin adapter over src/compression/entropy_coding.py.

Re-exports all public symbols but replaces the torch-dependent pad_factor
call with a pure numpy equivalent.  The original entropy_coding.py is left
completely untouched.
"""

import numpy as np
import sys
import os

# Make root of repo importable when running from tflite/ sub-directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Re-export everything we need from the original module
from src.compression.ans import (                     # noqa: F401
    empty_message, push, pop, flatten,
    unflatten, unflatten_scalar, RANS_L,
)
from src.compression.entropy_coding import (          # noqa: F401
    OVERFLOW_WIDTH, OVERFLOW_CODE, PATCH_SIZE, Codec,
    base_codec,
    ans_index_encoder, ans_index_decoder,
    vec_ans_index_encoder, vec_ans_index_decoder,
    ans_encode_decode_test,
)


# ---------------------------------------------------------------------------
# Pure-numpy pad_factor (replaces utils.pad_factor which calls torch.Tensor)
# ---------------------------------------------------------------------------

def pad_factor(arr, spatial_dims, factor):
    """
    Pad the last two dimensions of `arr` so they are both divisible by `factor`.

    Args:
        arr:           numpy array of shape (..., H, W)
        spatial_dims:  (H, W) tuple (used to infer current spatial size)
        factor:        int or (int, int) padding factor

    Returns:
        Padded numpy array (reflect padding, same as original).
    """
    arr = np.asarray(arr)
    H, W = arr.shape[-2], arr.shape[-1]
    if np.isscalar(factor):
        fh, fw = int(factor), int(factor)
    else:
        fh, fw = int(factor[0]), int(factor[1])

    pad_h = (fh - (H % fh)) % fh
    pad_w = (fw - (W % fw)) % fw

    if pad_h == 0 and pad_w == 0:
        return arr

    pad_width = [(0, 0)] * (arr.ndim - 2) + [(0, pad_h), (0, pad_w)]
    return np.pad(arr, pad_width, mode="reflect")
