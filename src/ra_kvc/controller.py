"""Causal precision allocation with a capped credit balance."""

import math
import random


class BudgetController:
    """One credit purchases one boosted channel in one equal-size block.

    Each arriving generated block earns target_fraction * head_dim credits.
    The controller cannot borrow from future blocks. Unused credit is bounded
    by one nominal block budget plus rounding. Exact terminal equality is not
    assumed; callers report the realized allocation.
    """
    def __init__(self, target_fraction=0.125, gain=1.0, seed=17):
        if not 0 <= target_fraction <= 0.5:
            raise ValueError("target_fraction must lie in [0, 0.5]")
        self.target = target_fraction
        self.gain = gain
        self.earned = 0.0
        self.spent = 0
        self.count = 0
        self.rng = random.Random(seed)

    def allocate(self, d, signal, calibration, kind="entropy"):
        target_count = self.target * d
        self.earned += target_count
        self.count += 1
        available = max(0, int(math.floor(self.earned + 1e-9)) - self.spent)
        if kind == "random":
            z = self.rng.uniform(-1, 1)
        elif signal is None:
            z = 0.0
        else:
            centre = calibration[kind]["mean"]
            spread = max(calibration[kind]["std"], 1e-6)
            z = (signal - centre) / spread
            if kind == "margin":
                z = -z
        z = min(1.0, max(-1.0, z))
        requested = round(target_count * (1 + self.gain * z))
        max_count = min(d, int(math.ceil(2 * target_count)))
        # Bound underspending without using future uncertainty.
        minimum = max(0, int(math.floor(available - target_count)))
        chosen = min(max_count, available, max(minimum, requested))
        self.spent += chosen
        return chosen

    @property
    def credit(self):
        return self.earned - self.spent
