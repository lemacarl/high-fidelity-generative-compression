"""
Compression utilities for TFLite path — torch/autograd dependencies removed.

Reuses the binary .hfc format from src/compression/compression_utils.py so
compressed files are readable by both codecs.
"""

import functools
import os
import numpy as np
from collections import namedtuple

from tflite.compression import entropy_coding

_MAGIC_VALUE_SEP = b'\x46\xE2\x84\x92'

PATCH_SIZE = entropy_coding.PATCH_SIZE

CompressionOutput = namedtuple("CompressionOutput", [
    "hyperlatents_encoded",
    "latents_encoded",
    "hyperlatent_spatial_shape",
    "batch_shape",
    "spatial_shape",
    "hyper_coding_shape",
    "latent_coding_shape",
])


# ---------------------------------------------------------------------------
# Patch decomposition helpers (pure numpy, replaces torch.unfold)
# ---------------------------------------------------------------------------

def decompose(x, n_channels, patch_size=PATCH_SIZE):
    """Decompose (B, C, H, W) into spatial patches of patch_size."""
    x = np.asarray(x, dtype=np.int32)
    B, C, H, W = x.shape
    ph, pw = patch_size

    # Ensure divisible
    assert H % ph == 0 and W % pw == 0, (
        f"Spatial dims ({H},{W}) not divisible by patch_size {patch_size}"
    )

    # (B, C, H/ph, ph, W/pw, pw)
    y = x.reshape(B, C, H // ph, ph, W // pw, pw)
    # (B, H/ph, W/pw, C, ph, pw)
    y = y.transpose(0, 2, 4, 1, 3, 5)
    unfolded_shape = y.shape
    # (n_patches_total, C, ph, pw)
    y = y.reshape(-1, C, ph, pw)
    return y.astype(np.int32), unfolded_shape


def reconstitute(x, original_shape, unfolded_shape, patch_size=PATCH_SIZE):
    """Reconstitute patches back into (B, C, H, W)."""
    x = np.asarray(x, dtype=np.int32)
    B, C, H, W = original_shape
    ph, pw = patch_size
    # unfolded_shape: (B, H/ph, W/pw, C, ph, pw)
    x = x.reshape(unfolded_shape)
    # (B, C, H/ph, ph, W/pw, pw)
    x = x.transpose(0, 3, 1, 4, 2, 5)
    x = x.reshape(original_shape)
    return x.astype(np.int32)


# ---------------------------------------------------------------------------
# Tail estimation (pure numpy/scipy, replaces torch Adam loop)
# ---------------------------------------------------------------------------

def estimate_tails(cdf_fn, target, shape, extra_counts=24):
    """
    Estimate tail quantiles via a simple Adam loop (numpy version).

    cdf_fn:  callable float32 → float32  (must be vectorised over shape)
    target:  scalar target CDF value
    shape:   int or tuple — shape of the output tensor
    """
    from scipy.optimize import brentq

    if np.isscalar(shape):
        shape = (shape,)

    tails = np.zeros(shape, dtype=np.float64)
    for idx in np.ndindex(*shape):
        # find x such that cdf_fn(x) == target using Brent's method
        try:
            tails[idx] = brentq(lambda x: float(cdf_fn(np.array([x]))[0]) - target,
                                a=-1000.0, b=1000.0, xtol=1e-6, maxiter=500)
        except ValueError:
            tails[idx] = 0.0
    return tails.astype(np.float32)


# ---------------------------------------------------------------------------
# ANS compress / decompress (thin wrapper over entropy_coding)
# ---------------------------------------------------------------------------

def ans_compress(symbols, indices, cdf, cdf_length, cdf_offset, coding_shape,
                 precision, vectorize=False, block_encode=True):
    if vectorize:
        encoded = entropy_coding.vec_ans_index_encoder(
            symbols=symbols, indices=indices, cdf=cdf,
            cdf_length=cdf_length, cdf_offset=cdf_offset,
            precision=precision, coding_shape=coding_shape,
        )
    else:
        if block_encode:
            encoded = entropy_coding.ans_index_encoder(
                symbols=symbols, indices=indices, cdf=cdf,
                cdf_length=cdf_length, cdf_offset=cdf_offset,
                precision=precision, coding_shape=coding_shape,
            )
        else:
            encoded = []
            for i in range(symbols.shape[0]):
                coded = entropy_coding.ans_index_encoder(
                    symbols=symbols[i], indices=indices[i], cdf=cdf,
                    cdf_length=cdf_length, cdf_offset=cdf_offset,
                    precision=precision, coding_shape=coding_shape,
                )
                encoded.append(coded)
    return encoded


def ans_decompress(encoded, indices, cdf, cdf_length, cdf_offset, coding_shape,
                   precision, vectorize=False, block_decode=True):
    if vectorize:
        decoded = entropy_coding.vec_ans_index_decoder(
            encoded, indices=indices, cdf=cdf,
            cdf_length=cdf_length, cdf_offset=cdf_offset,
            precision=precision, coding_shape=coding_shape,
        )
    else:
        if block_decode:
            decoded = entropy_coding.ans_index_decoder(
                encoded, indices=indices, cdf=cdf,
                cdf_length=cdf_length, cdf_offset=cdf_offset,
                precision=precision, coding_shape=coding_shape,
            )
        else:
            decoded = []
            for i in range(len(encoded)):
                d = entropy_coding.ans_index_decoder(
                    encoded[i], indices=indices[i], cdf=cdf,
                    cdf_length=cdf_length, cdf_offset=cdf_offset,
                    precision=precision, coding_shape=coding_shape,
                )
                decoded.append(d)
            decoded = np.stack(decoded, axis=0)
    return decoded


def check_argument_shapes(cdf, cdf_length, cdf_offset):
    if cdf.ndim != 2 or cdf.shape[1] < 3:
        raise ValueError(f"'cdf' should be 2-D with dim >= 3: {cdf.shape}")
    if cdf_length.ndim != 1 or cdf_length.shape[0] != cdf.shape[0]:
        raise ValueError(f"'cdf_length' 1-D length mismatch: {cdf_length.shape}")
    if cdf_offset.ndim != 1 or cdf_offset.shape[0] != cdf.shape[0]:
        raise ValueError(f"'cdf_offset' 1-D length mismatch: {cdf_offset.shape}")


# ---------------------------------------------------------------------------
# Binary file I/O helpers (identical format to src/compression/compression_utils.py)
# ---------------------------------------------------------------------------

def compose(*args):
    def compose2(f1, f2):
        def composed(*a, **kw):
            return f1(f2(*a, **kw))
        return composed
    return functools.reduce(compose2, args)

def return_list(f):
    return compose(list, f)

def write_bytes(f, ts, xs):
    for t, x in zip(ts, xs):
        f.write(t(x).tobytes())

@return_list
def read_bytes(f, ts):
    for t in ts:
        yield np.frombuffer(f.read(t().itemsize), t, count=1)[0]

def write_shapes(shape, fout):
    for s in shape:
        assert s < 2**16, s
    write_bytes(fout, [np.uint16] * len(shape), shape)
    return 2 * len(shape)

def read_shapes(fin, shape_len):
    return tuple(map(int, read_bytes(fin, [np.uint16] * shape_len)))

def write_num_bytes_encoded(num_bytes, fout):
    assert num_bytes < 2**32
    write_bytes(fout, [np.uint32], [num_bytes])
    return 4

def read_num_bytes_encoded(fin):
    return int(read_bytes(fin, [np.uint32])[0])

def message_to_bytes(f_out, message):
    f_out.write(message.tobytes())

def message_from_bytes(f_in, num_bytes, t=np.uint32):
    return np.frombuffer(f_in.read(num_bytes), t, count=-1)


# ---------------------------------------------------------------------------
# Save / load compressed .hfc format
# ---------------------------------------------------------------------------

def save_compressed_format(compression_output, out_path):
    with open(out_path, "wb") as f:
        write_shapes(compression_output.hyperlatent_spatial_shape, f)
        write_shapes(compression_output.spatial_shape, f)
        write_shapes(compression_output.hyper_coding_shape, f)
        write_shapes(compression_output.latent_coding_shape, f)
        write_shapes([compression_output.batch_shape], f)
        f.write(_MAGIC_VALUE_SEP)

        enc_hyp = compression_output.hyperlatents_encoded
        write_num_bytes_encoded(len(enc_hyp) * 4, f)
        message_to_bytes(f, enc_hyp)
        f.write(_MAGIC_VALUE_SEP)

        enc_lat = compression_output.latents_encoded
        write_num_bytes_encoded(len(enc_lat) * 4, f)
        message_to_bytes(f, enc_lat)
        f.write(_MAGIC_VALUE_SEP)

    actual_bpp = (8.0 * os.path.getsize(out_path)
                  / float(np.prod(compression_output.spatial_shape)))
    return actual_bpp


def load_compressed_format(in_path):
    with open(in_path, "rb") as f:
        hyperlatent_spatial_shape = read_shapes(f, 2)
        spatial_shape = read_shapes(f, 2)
        hyper_coding_shape = read_shapes(f, 3)
        latent_coding_shape = read_shapes(f, 3)
        batch_shape = read_shapes(f, 1)
        assert f.read(4) == _MAGIC_VALUE_SEP

        num_bytes = read_num_bytes_encoded(f)
        hyperlatents_encoded = message_from_bytes(f, num_bytes)
        assert f.read(4) == _MAGIC_VALUE_SEP

        num_bytes = read_num_bytes_encoded(f)
        latents_encoded = message_from_bytes(f, num_bytes)
        assert f.read(4) == _MAGIC_VALUE_SEP

    return CompressionOutput(
        hyperlatents_encoded=hyperlatents_encoded,
        latents_encoded=latents_encoded,
        hyperlatent_spatial_shape=hyperlatent_spatial_shape,
        spatial_shape=spatial_shape,
        hyper_coding_shape=hyper_coding_shape,
        latent_coding_shape=latent_coding_shape,
        batch_shape=batch_shape[0],
    )
