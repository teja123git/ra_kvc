"""Numeric answer extraction and paired small-sample statistics."""

import math
import re
from decimal import Decimal, InvalidOperation
import numpy as np

NUMBER = r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def canonical_number(text):
    try:
        value = Decimal(text.replace(",", "").strip())
        return format(value.normalize(), "f") if value.is_finite() else None
    except InvalidOperation:
        return None


def extract_answer(text, allow_fallback=True):
    # Prefer the last explicit answer marker, then boxed answers.
    markers = re.findall(r"####\s*(" + NUMBER + r")", text)
    if markers:
        return canonical_number(markers[-1]), "marker"
    boxed = re.findall(r"\\boxed\{\s*(" + NUMBER + r")\s*\}", text)
    if boxed:
        return canonical_number(boxed[-1]), "boxed"
    final = re.findall(r"(?:final answer|answer is|answer:)\s*\$?\s*(" + NUMBER + r")",
                       text, flags=re.I)
    if final:
        return canonical_number(final[-1]), "answer_phrase"
    numbers = re.findall(NUMBER, text)
    if allow_fallback and numbers:
        return canonical_number(numbers[-1]), "last_number"
    return None, "missing"


def wilson(correct, total, z=1.96):
    if not total:
        return [None, None]
    p = correct/total
    denom = 1 + z*z/total
    centre = (p + z*z/(2*total))/denom
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denom
    return [max(0.0, centre-half), min(1.0, centre+half)]


def paired_bootstrap(a, b, seed=17, replicates=5000):
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if not len(delta):
        return {"difference": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), (replicates, len(delta)))].mean(axis=1)
    wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
    discordant = wins + losses
    pvalue = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses)+1)) / 2**discordant) if discordant else 1.0
    return {"difference": float(delta.mean()),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist(), "n": len(delta),
            "wins": wins, "losses": losses, "discordant_pairs": discordant,
            "mcnemar_exact_p_two_sided": pvalue,
            "note": "Resampling cannot reveal unobserved disagreements. A degenerate interval does not establish equivalence." if not discordant else "Exploratory paired comparison; no multiplicity correction."}
