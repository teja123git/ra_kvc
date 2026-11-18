"""An explicit cache adapter for Transformers 4.46.3 Qwen2 models."""

from .controller import BudgetController
from .quantization import quantize_block

MODES = ("fp16", "uniform2", "uniform4", "fixed", "entropy", "margin", "random")


def as_legacy(cache):
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    if isinstance(cache, (tuple, list)):
        return tuple(cache)
    raise TypeError("Unsupported cache API; install transformers==4.46.3")


class CacheQuantizer:
    def __init__(self, mode, prompt_length, tracker, calibration, group_size=32,
                 sink_tokens=8, recent_tokens=32, target_fraction=0.125, seed=17):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if group_size < 2 or min(sink_tokens, recent_tokens) < 0:
            raise ValueError("Invalid buffer dimensions")
        self.mode, self.prompt_length = mode, prompt_length
        self.tracker, self.calibration = tracker, calibration
        self.group_size, self.recent_tokens = group_size, recent_tokens
        self.next_start = sink_tokens
        self.target_fraction = target_fraction
        self.controller = BudgetController(target_fraction, seed=seed)
        self.records = []

    def apply(self, cache):
        legacy = as_legacy(cache)
        if self.mode == "fp16":
            return legacy
        length, d = legacy[0][0].shape[-2:]
        while self.next_start + self.group_size <= length-self.recent_tokens:
            start, end = self.next_start, self.next_start+self.group_size
            phase = "generated" if start >= self.prompt_length else "prompt_or_boundary"
            kind = self.mode if self.mode in ("entropy", "margin") else "entropy"
            score = self.tracker.block_score(start, end, kind)
            if self.mode in ("uniform2", "uniform4"):
                n = 0
            elif self.mode == "fixed" or phase != "generated":
                n = round(self.target_fraction*d)
            else:
                n = self.controller.allocate(d, score, self.calibration, self.mode)
            for keys, values in legacy:
                if keys.shape[-1] != d:
                    raise ValueError("Different head dimensions across layers unsupported")
                k, v = quantize_block(keys[..., start:end, :], values[..., start:end, :],
                                      boosted_channels=n,
                                      uniform_bits=4 if self.mode == "uniform4" else None)
                keys[..., start:end, :] = k
                values[..., start:end, :] = v
            self.records.append({"start": start, "end": end, "boosted": n,
                                 "fraction": n/d, "signal": score, "phase": phase,
                                 "mode": self.mode, "credit": self.controller.credit})
            self.next_start = end
        return legacy
