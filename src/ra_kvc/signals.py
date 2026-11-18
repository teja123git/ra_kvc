"""Uncertainty proxies. Neither proxy is a direct measure of reasoning."""

import math
import torch


def uncertainty(logits):
    z = logits.float().reshape(-1)
    log_p = torch.log_softmax(z, dim=-1)
    entropy = -(log_p.exp() * log_p).sum() / math.log(z.numel())
    top = z.topk(2).values
    return {"entropy": float(entropy.item()),
            "margin": float((top[0] - top[1]).item())}


class SignalTracker:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.previous = None
        self.by_position = {}

    def observe(self, position, values):
        if self.previous is None:
            current = dict(values)
        else:
            current = {k: self.alpha * values[k] + (1-self.alpha) * self.previous[k]
                       for k in values}
        self.previous = current
        self.by_position[int(position)] = current

    def block_score(self, start, end, name):
        vals = [self.by_position[t][name] for t in range(start, end)
                if t in self.by_position]
        return sum(vals) / len(vals) if vals else None
