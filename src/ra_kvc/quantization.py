"""Affine fake quantization. Tensors retain their original floating dtype."""

import torch


def fake_quantize(x, bits, dim):
    """Quantize/dequantize along dim, preserving constant groups exactly."""
    if bits == 16:
        return x.clone()
    if bits not in (2, 4):
        raise ValueError("Supported simulated widths are 2, 4 and 16 bits")
    work = x.float()
    lo = work.amin(dim=dim, keepdim=True)
    span = work.amax(dim=dim, keepdim=True) - lo
    levels = 2**bits - 1
    scale = torch.where(span > 0, span / levels, torch.ones_like(span))
    codes = torch.round((work - lo) / scale).clamp(0, levels)
    return (codes * scale + lo).to(x.dtype)


def quantize_block(keys, values, boosted_channels=0, uniform_bits=None):
    """Input shape [batch, KV heads, tokens, head dimension].

    K scales are per channel across the token block. V scales are per token
    across the head dimension. The largest mean-absolute K channels get INT4.
    """
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("Expected equally shaped 4D K and V blocks")
    if uniform_bits is not None:
        return (fake_quantize(keys, uniform_bits, -2),
                fake_quantize(values, uniform_bits, -1))
    d = keys.shape[-1]
    if not 0 <= boosted_channels <= d:
        raise ValueError("Invalid channel count")
    k2 = fake_quantize(keys, 2, -2)
    v2 = fake_quantize(values, 2, -1)
    if boosted_channels == 0:
        return k2, v2
    score = keys.float().abs().mean(dim=-2, keepdim=True)
    indices = score.topk(boosted_channels, dim=-1).indices
    mask = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, indices, True)
    k4 = fake_quantize(keys, 4, -2)
    return torch.where(mask, k4, k2), v2
