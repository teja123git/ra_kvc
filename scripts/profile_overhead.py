"""Measure individual prototype operations on an actual model cache/logit sample."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from ra_kvc.cache import as_legacy
from ra_kvc.controller import BudgetController
from ra_kvc.data import gsm8k
from ra_kvc.generation import load_model, math_prompt
from ra_kvc.quantization import quantize_block
from ra_kvc.signals import uncertainty


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/local_pilot")
    parser.add_argument("--model-path")
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args()
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text())
    cfg = manifest["config"]
    calibration = json.loads((run / "calibration.json").read_text())
    model, tokenizer = load_model(args.model_path or cfg["model"], cfg["device"],
                                  None if args.model_path else manifest["model_revision"], cfg["threads"])
    device = next(model.parameters()).device
    questions, _ = gsm8k("train", 1, cfg["seed"], ROOT / "data/cache")
    ids = tokenizer.apply_chat_template(math_prompt(questions[0]["question"]),
                                        add_generation_prompt=True, return_tensors="pt").to(device)
    output = model(ids, use_cache=True)
    cache = as_legacy(output.past_key_values)
    logits = output.logits[:, -1, :]
    first, last = cfg["sink_tokens"], cfg["sink_tokens"]+cfg["group_size"]
    blocks = [(k[..., first:last, :], v[..., first:last, :]) for k, v in cache]
    d = blocks[0][0].shape[-1]
    controller = BudgetController(cfg["target_fraction"])

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def measure(fn):
        for _ in range(5):
            fn()
        times = []
        for _ in range(args.repeats):
            synchronize()
            start = time.perf_counter_ns()
            fn()
            synchronize()
            times.append((time.perf_counter_ns()-start)/1e6)
        return {"median_ms": float(np.median(times)), "p10_ms": float(np.quantile(times, 0.1)),
                "p90_ms": float(np.quantile(times, 0.9)), "samples_ms": times}

    operations = {
        "entropy_and_margin_per_token": lambda: uncertainty(logits),
        "controller_per_block": lambda: controller.allocate(d, calibration["entropy"]["mean"], calibration),
        "uniform2_one_block_all_layers": lambda: [quantize_block(k, v, 0) for k, v in blocks],
        "fixed_one_block_all_layers": lambda: [quantize_block(k, v, round(cfg["target_fraction"]*d)) for k, v in blocks],
    }
    results = {name: measure(fn) for name, fn in operations.items()}
    result = {"model": cfg["model"], "model_revision": manifest["model_revision"],
              "device": str(device), "dtype": str(logits.dtype), "threads": torch.get_num_threads(),
              "question_id": questions[0]["id"], "vocabulary_size": logits.shape[-1],
              "group_size": cfg["group_size"], "layers": len(cache), "repeats": args.repeats,
              "operations": results,
              "note": "Measured microbenchmarks on real model tensors, with no dedicated hardware isolation. Costs have different frequencies (per token versus per block). They are not additive end-to-end speedup estimates, and GPU timings require an actual GPU run."}
    (run / "overhead_profile.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({name: data["median_ms"] for name, data in results.items()}, indent=2))


if __name__ == "__main__":
    main()
