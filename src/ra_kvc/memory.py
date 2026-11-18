"""Projected packed KV storage; this is not measured tensor allocation."""

import math


def cache_storage(layers, kv_heads, head_dim, tokens, records, fp_bits=16):
    """Model a concrete packed format with FP32 scale/offset metadata.

    Each quantized block stores K and V integer payload, two FP32 numbers
    per K channel and per V token, a channel bitmap for mixed precision,
    and an 8-byte block directory entry. Payloads round up to bytes per head.
    Incomplete/recent/sink tokens retain fp_bits. Batch size is one.
    """
    if tokens < 0 or any(v <= 0 for v in (layers, kv_heads, head_dim)):
        raise ValueError("Invalid cache dimensions")
    multiplier = layers * kv_heads
    quantized = sum(r["end"] - r["start"] for r in records)
    if quantized > tokens:
        raise ValueError("More quantized tokens than cached tokens")
    payload = 0
    metadata = 0
    for r in records:
        g = r["end"] - r["start"]
        if r["mode"] == "uniform4":
            k_bits, v_bits = 4 * g * head_dim, 4 * g * head_dim
            mask = 0
        else:
            k_bits = 2 * g * head_dim + 2 * g * r["boosted"]
            v_bits = 2 * g * head_dim
            mask = math.ceil(head_dim / 8) if r["mode"] in (
                "fixed", "entropy", "margin", "random") else 0
        payload += multiplier * (math.ceil(k_bits/8) + math.ceil(v_bits/8))
        metadata += multiplier * (8 * head_dim + 8 * g + mask + 8)
    floating = multiplier * (tokens-quantized) * head_dim * 2 * fp_bits / 8
    reference = multiplier * tokens * head_dim * 2 * fp_bits / 8
    total = payload + metadata + floating
    elements = multiplier * tokens * head_dim * 2
    return {"packed_payload_bytes": payload, "metadata_bytes": metadata,
            "floating_buffer_bytes": floating, "estimated_total_bytes": total,
            "reference_cache_bytes": reference,
            "estimated_compression": reference/total if total else 1.0,
            "effective_bits_including_overhead": 8*total/elements if elements else 0,
            "quantized_tokens": quantized, "cached_tokens": tokens,
            "projection_reference_bits": fp_bits}
