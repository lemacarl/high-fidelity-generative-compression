"""
Self-contained ANS entropy coding for the TFLite path.

All torch dependencies removed. Uses tflite/compression/ans.py for the
raw rANS primitives and inlines the decompose/reconstitute/view_update
helpers to avoid a circular import with compression_utils.py.
"""

import numpy as np
from collections import namedtuple

import tflite.compression.ans as vrans

OVERFLOW_WIDTH = 4
OVERFLOW_CODE = 1 << (1 << OVERFLOW_WIDTH)
PATCH_SIZE = (1, 1)

Codec = namedtuple('Codec', ['push', 'pop'])
cast2u64 = lambda x: np.array(x, dtype=np.uint64)


# ---------------------------------------------------------------------------
# Inlined spatial helpers (avoids circular import with compression_utils)
# ---------------------------------------------------------------------------

def _decompose(x, n_channels, patch_size=PATCH_SIZE):
    x = np.asarray(x, dtype=np.int32)
    B, C, H, W = x.shape
    ph, pw = patch_size
    assert H % ph == 0 and W % pw == 0
    y = x.reshape(B, C, H // ph, ph, W // pw, pw)
    y = y.transpose(0, 2, 4, 1, 3, 5)
    unfolded_shape = y.shape
    y = y.reshape(-1, C, ph, pw)
    return y.astype(np.int32), unfolded_shape


def _reconstitute(x, original_shape, unfolded_shape, patch_size=PATCH_SIZE):
    x = np.asarray(x, dtype=np.int32)
    x = x.reshape(unfolded_shape)
    x = x.transpose(0, 3, 1, 4, 2, 5)
    x = x.reshape(original_shape)
    return x.astype(np.int32)


def _view_update(head, view_fun):
    subhead = view_fun(head)
    indices = view_fun(np.arange(head.size).reshape(head.shape))
    def update(new_subhead):
        new_head = head.copy()
        new_head.flat[indices.flat] = new_subhead
        return new_head
    return subhead, update


# ---------------------------------------------------------------------------
# CDF helpers
# ---------------------------------------------------------------------------

def _indexed_cdf_to_enc_statfun(cdf_i):
    def _enc_statfun(value):
        lower = cdf_i[value]
        return lower, cdf_i[int(value + np.uint64(1))] - lower
    return _enc_statfun


def _vec_indexed_cdf_to_enc_statfun(cdf_i):
    def _enc_statfun(value):
        lower = np.take_along_axis(cdf_i, np.expand_dims(value, -1), axis=-1)[..., 0]
        upper = np.take_along_axis(cdf_i, np.expand_dims(value + 1, -1), axis=-1)[..., 0]
        return lower, upper - lower
    return _enc_statfun


def _indexed_cdf_to_dec_statfun(cdf_i, cdf_i_length):
    cdf_i = cdf_i[:cdf_i_length]
    def _dec_statfun(cum_freq):
        return np.searchsorted(cdf_i, cum_freq, side='right') - 1
    return _dec_statfun


def _vec_indexed_cdf_to_dec_statfun(cdf_i, cdf_i_length):
    *coding_shape, max_cdf_length = cdf_i.shape
    coding_shape = tuple(coding_shape)
    cdf_i_flat = np.reshape(cdf_i, (-1, max_cdf_length))
    cdf_i_flat_ragged = [c[:l] for (c, l) in zip(cdf_i_flat, cdf_i_length.flatten())]

    def _dec_statfun(value):
        assert value.shape == coding_shape
        sym_flat = np.array(
            [np.searchsorted(cb, v_i, 'right') - 1
             for (cb, v_i) in zip(cdf_i_flat_ragged, value.flatten())])
        return np.reshape(sym_flat, coding_shape)
    return _dec_statfun


# ---------------------------------------------------------------------------
# Base codec
# ---------------------------------------------------------------------------

def base_codec(enc_statfun, dec_statfun, precision, log=False):
    def push(message, symbol):
        start, freq = enc_statfun(symbol)
        return vrans.push(message, start, freq, precision)

    def pop(message, log=log):
        cf, pop_fun = vrans.pop(message, precision)
        symbol = dec_statfun(cf)
        start, freq = enc_statfun(symbol)
        assert np.all(start <= cf) and np.all(cf < start + freq)
        return pop_fun(start, freq), symbol

    return Codec(push, pop)


def overflow_view(value, mask):
    return value[mask]


def substack(codec, view_fun):
    def push(message, start, freq, precision, mask):
        head, tail = message
        view_fun_ = lambda x: view_fun(x, mask)
        subhead, update = _view_update(head, view_fun_)
        subhead, tail = vrans.push((subhead, tail), start, freq, precision)
        return update(subhead), tail

    def pop(message, precision, mask, *args, **kwargs):
        head, tail = message
        view_fun_ = lambda x: view_fun(x, mask)
        subhead, update = _view_update(head, view_fun_)
        cf, pop_fun = vrans.pop((subhead, tail), precision)
        symbol = cf
        start, freq = symbol, 1
        assert np.all(start <= cf) and np.all(cf < start + freq)
        (subhead, tail), data = pop_fun(start, freq), symbol
        return (update(subhead), tail), data

    return Codec(push, pop)


# ---------------------------------------------------------------------------
# Scalar ANS encode / decode
# ---------------------------------------------------------------------------

def ans_index_buffered_encoder(symbols, indices, cdf, cdf_length, cdf_offset,
                                precision, overflow_width=OVERFLOW_WIDTH, **kwargs):
    instructions = []
    coding_shape = symbols.shape[1:]
    symbols = symbols.astype(np.int32).flatten()
    indices = indices.astype(np.int32).flatten()

    max_overflow = (1 << overflow_width) - 1
    overflow_cdf_size = (1 << overflow_width) + 1
    overflow_cdf = np.arange(overflow_cdf_size, dtype=np.uint64)
    enc_statfun_overflow = _indexed_cdf_to_enc_statfun(overflow_cdf)
    dec_statfun_overflow = _indexed_cdf_to_dec_statfun(overflow_cdf, len(overflow_cdf))
    overflow_push, overflow_pop = base_codec(enc_statfun_overflow, dec_statfun_overflow,
                                             overflow_width)

    for i in range(len(indices)):
        cdf_index = indices[i]
        cdf_i = cdf[cdf_index]
        cdf_length_i = int(cdf_length[cdf_index])
        max_value = cdf_length_i - 2

        # Use plain Python ints to avoid numpy int32 overflow in shift ops
        value = int(symbols[i]) - int(cdf_offset[cdf_index])
        overflow = 0
        if value < 0:
            overflow = -2 * value - 1
            value = max_value
        elif value >= max_value:
            overflow = 2 * (value - max_value)
            value = max_value

        enc_statfun = _indexed_cdf_to_enc_statfun(cdf_i)
        start, freq = enc_statfun(value)
        instructions.append((start, freq, False))

        if value == max_value:
            widths = 0
            while (overflow >> (widths * overflow_width)) != 0:
                widths += 1
            val = widths
            while val >= max_overflow:
                start, freq = enc_statfun_overflow(cast2u64(max_overflow))
                instructions.append((start, freq, True))
                val -= max_overflow
            start, freq = enc_statfun_overflow(cast2u64(val))
            instructions.append((start, freq, True))
            for j in range(widths):
                val = (overflow >> (j * overflow_width)) & max_overflow
                start, freq = enc_statfun_overflow(cast2u64(val))
                instructions.append((start, freq, True))

    return instructions, coding_shape


def ans_index_encoder_flush(instructions, precision, overflow_width=OVERFLOW_WIDTH, **kwargs):
    message = vrans.empty_message(())
    for i in reversed(range(len(instructions))):
        start, freq, flag = instructions[i]
        if not flag:
            message = vrans.push(message, start, freq, precision)
        else:
            message = vrans.push(message, start, freq, overflow_width)
    encoded = vrans.flatten(message)
    print('Symbol compressed to {:.3f} bits.'.format(32 * len(encoded)))
    return encoded


def ans_index_encoder(symbols, indices, cdf, cdf_length, cdf_offset,
                      precision, overflow_width=OVERFLOW_WIDTH, **kwargs):
    instructions, coding_shape = ans_index_buffered_encoder(
        symbols, indices, cdf, cdf_length, cdf_offset, precision, overflow_width)
    encoded = ans_index_encoder_flush(instructions, precision, overflow_width)
    return encoded, coding_shape


def ans_index_decoder(encoded, indices, cdf, cdf_length, cdf_offset,
                      precision, coding_shape, overflow_width=OVERFLOW_WIDTH, **kwargs):
    message = vrans.unflatten_scalar(encoded)
    decoded = np.empty(indices.shape).flatten()
    indices = indices.astype(np.int32).flatten()

    max_overflow = (1 << overflow_width) - 1
    overflow_cdf_size = (1 << overflow_width) + 1
    overflow_cdf = np.arange(overflow_cdf_size, dtype=np.uint64)
    enc_statfun_overflow = _indexed_cdf_to_enc_statfun(overflow_cdf)
    dec_statfun_overflow = _indexed_cdf_to_dec_statfun(overflow_cdf, len(overflow_cdf))
    overflow_push, overflow_pop = base_codec(enc_statfun_overflow, dec_statfun_overflow,
                                             overflow_width)

    for i in range(len(indices)):
        cdf_index = int(indices[i])
        cdf_i = cdf[cdf_index]
        cdf_length_i = int(cdf_length[cdf_index])
        max_value = cdf_length_i - 2

        enc_statfun = _indexed_cdf_to_enc_statfun(cdf_i)
        dec_statfun = _indexed_cdf_to_dec_statfun(cdf_i, cdf_length_i)
        symbol_push, symbol_pop = base_codec(enc_statfun, dec_statfun, precision)
        message, value = symbol_pop(message)
        value = int(value)

        if value == max_value:
            message, val = overflow_pop(message)
            val = int(val)
            widths = val
            while val == max_overflow:
                message, val = overflow_pop(message)
                val = int(val)
                widths += val
            overflow = 0
            for j in range(widths):
                message, val = overflow_pop(message)
                val = int(val)
                overflow |= val << (j * overflow_width)
            value = overflow >> 1
            if overflow & 1:
                value = -value - 1
            else:
                value += max_value

        decoded[i] = value + int(cdf_offset[cdf_index])

    return decoded


# ---------------------------------------------------------------------------
# Vectorised ANS encode / decode
# ---------------------------------------------------------------------------

def vec_ans_index_buffered_encoder(symbols, indices, cdf, cdf_length, cdf_offset,
                                    precision, coding_shape,
                                    overflow_width=OVERFLOW_WIDTH, **kwargs):
    instructions = []
    symbols_shape = symbols.shape
    B, n_channels = symbols_shape[:2]
    symbols = symbols.astype(np.int32)
    indices = indices.astype(np.int32)
    cdf_index = indices

    max_overflow = (1 << overflow_width) - 1
    overflow_cdf_size = (1 << overflow_width) + 1
    overflow_cdf = np.arange(overflow_cdf_size, dtype=np.uint64)[None, None, None, :]
    enc_statfun_overflow = _vec_indexed_cdf_to_enc_statfun(overflow_cdf)
    dec_statfun_overflow = _vec_indexed_cdf_to_dec_statfun(
        overflow_cdf, np.ones_like(overflow_cdf) * len(overflow_cdf))
    overflow_push, overflow_pop = base_codec(enc_statfun_overflow, dec_statfun_overflow,
                                             overflow_width)

    max_value = cdf_length[cdf_index] - 2
    values = symbols - cdf_offset[cdf_index]

    overflow = np.zeros_like(values)
    of_mask_lower = values < 0
    overflow = np.where(of_mask_lower, -2 * values - 1, overflow)
    of_mask_upper = values >= max_value
    overflow = np.where(of_mask_upper, 2 * (values - max_value), overflow)
    values = np.where(np.logical_or(of_mask_lower, of_mask_upper), max_value, values)

    if B == 1:
        # PATCH_SIZE=(1,1) so spatial dims are always divisible; no padding needed
        values, _ = _decompose(values, n_channels)
        overflow, _ = _decompose(overflow, n_channels)
        cdf_index, unfolded_shape = _decompose(indices, n_channels)
        coding_shape = values.shape[1:]

    for i in range(len(cdf_index)):
        value_i = values[i]
        cdf_index_i = cdf_index[i]
        cdf_i = cdf[cdf_index_i]
        cdf_length_i = cdf_length[cdf_index_i]
        max_value_i = cdf_length_i - 2

        enc_statfun = _vec_indexed_cdf_to_enc_statfun(cdf_i)
        start, freq = enc_statfun(value_i)
        instructions.append((start, freq, False, precision, 0))

        overflow_i = overflow[i]
        of_mask = value_i == max_value_i

        if np.any(of_mask):
            widths = np.zeros_like(value_i)
            cond_mask = (overflow_i >> (widths * overflow_width)) != 0
            while np.any(cond_mask):
                widths = np.where(cond_mask, widths + 1, widths)
                cond_mask = (overflow_i >> (widths * overflow_width)) != 0

            val = widths
            val_push = cast2u64(val)
            overflow_start, overflow_freq = enc_statfun_overflow(val_push)
            start = overflow_start[of_mask]
            freq = overflow_freq[of_mask]
            instructions.append((start, freq, True, int(overflow_width), of_mask))

            cond_mask = widths != 0
            counter = 0
            while np.any(cond_mask):
                encoding = (overflow_i >> (counter * overflow_width)) & max_overflow
                val = np.where(cond_mask, encoding, val)
                val_push = cast2u64(val)
                overflow_start, overflow_freq = enc_statfun_overflow(val_push)
                start = overflow_start[of_mask]
                freq = overflow_freq[of_mask]
                instructions.append((start, freq, True, int(overflow_width), of_mask))
                widths = np.where(cond_mask, widths - 1, widths)
                cond_mask = widths != 0
                counter += 1

    return instructions, coding_shape


def vec_ans_index_encoder_flush(instructions, precision, coding_shape,
                                 overflow_width=OVERFLOW_WIDTH, **kwargs):
    message = vrans.empty_message(coding_shape)
    overflow_push, _ = substack(codec=None, view_fun=overflow_view)
    for i in reversed(range(len(instructions))):
        start, freq, flag, precision_i, mask = instructions[i]
        if not flag:
            message = vrans.push(message, start, freq, precision)
        else:
            message = overflow_push(message, start, freq, precision_i, mask)
    encoded = vrans.flatten(message)
    print('Symbol compressed to {:.3f} bits.'.format(32 * len(encoded)))
    return encoded


def vec_ans_index_encoder(symbols, indices, cdf, cdf_length, cdf_offset,
                           precision, coding_shape,
                           overflow_width=OVERFLOW_WIDTH, **kwargs):
    instructions, coding_shape = vec_ans_index_buffered_encoder(
        symbols, indices, cdf, cdf_length, cdf_offset,
        precision, coding_shape, overflow_width)
    encoded = vec_ans_index_encoder_flush(instructions, precision, coding_shape, overflow_width)
    return encoded, coding_shape


def vec_ans_index_decoder(encoded, indices, cdf, cdf_length, cdf_offset,
                           precision, coding_shape,
                           overflow_width=OVERFLOW_WIDTH, **kwargs):
    original_shape = indices.shape
    B, n_channels, *_ = original_shape
    message = vrans.unflatten(encoded, coding_shape)
    indices = indices.astype(np.int32)
    cdf_index = indices

    max_overflow = (1 << overflow_width) - 1
    overflow_cdf_size = (1 << overflow_width) + 1
    overflow_cdf = np.arange(overflow_cdf_size, dtype=np.uint64)[None, :]
    enc_statfun_overflow = _vec_indexed_cdf_to_enc_statfun(overflow_cdf)
    dec_statfun_overflow = _vec_indexed_cdf_to_dec_statfun(
        overflow_cdf, np.ones_like(overflow_cdf) * len(overflow_cdf))
    overflow_codec = base_codec(enc_statfun_overflow, dec_statfun_overflow, overflow_width)

    if B == 1:
        cdf_index, unfolded_shape = _decompose(indices, n_channels)
        padded_shape = indices.shape
        coding_shape = cdf_index.shape[1:]

    symbols = []
    _, overflow_pop = substack(codec=overflow_codec, view_fun=overflow_view)

    for i in range(len(cdf_index)):
        cdf_index_i = cdf_index[i]
        cdf_i = cdf[cdf_index_i]
        cdf_length_i = cdf_length[cdf_index_i]

        enc_statfun = _vec_indexed_cdf_to_enc_statfun(cdf_i)
        dec_statfun = _vec_indexed_cdf_to_dec_statfun(cdf_i, cdf_length_i)
        symbol_push, symbol_pop = base_codec(enc_statfun, dec_statfun, precision)
        message, value = symbol_pop(message)

        max_value_i = cdf_length_i - 2
        of_mask = value == max_value_i

        if np.any(of_mask):
            message, val = overflow_pop(message, overflow_width, of_mask)
            val = cast2u64(val)
            widths = val
            cond_mask = val == max_overflow
            while np.any(cond_mask):
                message, val = overflow_pop(message, overflow_width, of_mask)
                val = cast2u64(val)
                widths = np.where(cond_mask, widths + val, widths)
                cond_mask = val == max_overflow

            overflow = np.zeros_like(val)
            cond_mask = widths != 0
            counter = 0
            while np.any(cond_mask):
                message, val = overflow_pop(message, overflow_width, of_mask)
                val = cast2u64(val)
                op = overflow | (val << (counter * overflow_width))
                overflow = np.where(cond_mask, op, overflow)
                widths = np.where(cond_mask, widths - 1, widths)
                cond_mask = widths != 0
                counter += 1

            overflow_broadcast = value.copy()
            overflow_broadcast[of_mask] = overflow
            overflow = overflow_broadcast
            value = np.where(of_mask, overflow >> 1, value)
            cond_mask = np.logical_and(of_mask, overflow & 1)
            value = np.where(cond_mask, -value - 1, value)
            cond_mask = np.logical_and(of_mask, ~(overflow & 1).astype(bool))
            value = np.where(cond_mask, value + max_value_i, value)

        symbol = value + cdf_offset[cdf_index_i]
        symbols.append(symbol)

    if B == 1:
        decoded = _reconstitute(np.stack(symbols, axis=0), padded_shape, unfolded_shape)
        if tuple(decoded.shape) != tuple(original_shape):
            decoded = decoded[:, :, :original_shape[2], :original_shape[3]]
    else:
        decoded = np.stack(symbols, axis=0)
    return decoded


def ans_encode_decode_test(symbols, decompressed_symbols):
    return np.testing.assert_almost_equal(symbols, decompressed_symbols)
