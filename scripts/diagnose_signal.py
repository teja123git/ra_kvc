"""Causal block sensitivity diagnostic on held-out calibration trajectories.

For a completed generated block, preserve all other cache blocks, quantize
this one block, teacher-force the same next token and measure the next-logit
KL divergence from the unquantized model. No diagnostic is used to tune on
GSM8K test questions. This is a local sensitivity proxy, not a quality oracle.
"""

import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ra_kvc.data import gsm8k
from ra_kvc.generation import load_model, math_prompt
from ra_kvc.cache import as_legacy
from ra_kvc.quantization import quantize_block


def divergence(a, b):
    la, lb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
    return max(0.0, float((la.exp()*(la-lb)).sum().item()))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/local_pilot")
    parser.add_argument("--model-path")
    args = parser.parse_args()
    run = Path(args.run)
    manifest = json.loads((run/"manifest.json").read_text())
    cfg = manifest["config"]
    model, tokenizer = load_model(args.model_path or cfg["model"], cfg["device"],
                                  None if args.model_path else cfg.get("revision"), cfg["threads"])
    device = next(model.parameters()).device
    train, _ = gsm8k("train", cfg["calibration_samples"], cfg["seed"], ROOT/"data/cache")
    questions = {r["id"]: r["question"] for r in train}
    samples = [json.loads(s) for s in (run/"calibration_predictions.jsonl").read_text().splitlines()]
    records = []
    for row in samples:
        prompt = tokenizer.apply_chat_template(math_prompt(questions[row["id"]]), tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        p_len = prompt_ids.shape[1]
        signals = {s["position"]: s for s in row["signals"]}
        g, sink, recent = cfg["group_size"], cfg["sink_tokens"], cfg["recent_tokens"]
        start = sink + max(0, (p_len-sink+g-1)//g)*g
        while start+g+recent < p_len+len(row["token_ids"]):
            end = start+g
            prefix_generated = end+recent-p_len
            prefix = torch.cat([prompt_ids, torch.tensor([row["token_ids"][:prefix_generated]], device=device)], 1)
            out = model(prefix, use_cache=True)
            full_cache = as_legacy(out.past_key_values)
            token = torch.tensor([[row["token_ids"][prefix_generated]]], device=device)
            ref = model(token, past_key_values=full_cache, use_cache=True).logits[0, -1]
            del out
            changes = {}
            for label, ratio in [("uniform2", 0), ("boost25", 0.25)]:
                modified = []
                for k, v in full_cache:
                    kcopy, vcopy = k.clone(), v.clone()
                    a, b = quantize_block(k[..., start:end, :], v[..., start:end, :], round(ratio*k.shape[-1]))
                    kcopy[..., start:end, :], vcopy[..., start:end, :] = a, b
                    modified.append((kcopy, vcopy))
                logits = model(token, past_key_values=tuple(modified), use_cache=True).logits[0, -1]
                changes[label] = divergence(ref, logits)
            entropy = np.mean([signals[t]["entropy"] for t in range(start, end)])
            margin = np.mean([signals[t]["margin"] for t in range(start, end)])
            records.append({"id": row["id"], "start": start, "end": end,
                            "entropy": float(entropy), "margin": float(margin),
                            "kl_uniform2": changes["uniform2"], "kl_boost25": changes["boost25"],
                            "kl_reduction": changes["uniform2"]-changes["boost25"]})
            print(f"{row['id']} block {start}:{end}: KL={changes['uniform2']:.6g}", flush=True)
            start += g
    def corr(name):
        if len(records)<3:
            return None
        a = np.array([r[name] for r in records]); b = np.array([r["kl_reduction"] for r in records])
        if a.std()==0 or b.std()==0:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    result = {"n_blocks": len(records), "n_questions": len(set(r["id"] for r in records)),
              "pearson_entropy_vs_kl_reduction": corr("entropy"),
              "pearson_margin_vs_kl_reduction": corr("margin"), "records": records,
              "limitation": "Small, correlated block sample; Pearson correlations are descriptive. Local one-token perturbation does not measure final answer accuracy or establish reasoning detection."}
    (run/"signal_diagnostic.json").write_text(json.dumps(result, indent=2))
    if records:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, field in zip(axes, ["entropy", "margin"]):
            ax.scatter([r[field] for r in records], [r["kl_reduction"] for r in records], color="#008576", s=35)
            ax.axhline(0, color="gray", linewidth=0.7); ax.set_xlabel(field.capitalize())
            ax.set_ylabel("KL reduction from 25% channel boost")
        fig.suptitle(f"Calibration diagnostic: {len(records)} cache blocks")
        fig.tight_layout(); (run/"figures").mkdir(exist_ok=True)
        fig.savefig(run/"figures/signal_sensitivity.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
