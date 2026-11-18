"""Run calibration once and resume a paired GSM8K comparison."""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
import torch
import transformers
from ra_kvc.data import gsm8k
from ra_kvc.generation import generate, load_model, math_prompt
from ra_kvc.scoring import extract_answer


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False)+"\n")
        stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fast_gpu.json")
    parser.add_argument("--output", default="results/gpu_run")
    parser.add_argument("--model-path", help="Optional local copy of the configured model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    for argument, field in [(args.limit, "samples"), (args.modes, "modes"),
                            (args.max_new_tokens, "max_new_tokens")]:
        if argument is not None:
            cfg[field] = argument
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    manifest_path = out/"manifest.json"
    old = None
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["config_sha256"] != signature:
            raise RuntimeError("Output directory has a different configuration. Choose a new directory.")
    rows, source = gsm8k("test", cfg["samples"], cfg["seed"], ROOT/"data/cache")
    if old and old["dataset"] != source:
        raise RuntimeError("Dataset contents or selected IDs changed. Use a new output directory.")
    revision = cfg.get("revision") or (old.get("model_revision") if old else None)
    model, tokenizer = load_model(args.model_path or cfg["model"], cfg["device"],
                                  None if args.model_path else revision, cfg["threads"])
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
                "config_sha256": signature, "dataset": source,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "python": platform.python_version(), "platform": platform.platform(),
                "device": str(next(model.parameters()).device),
                "dtype": str(next(model.parameters()).dtype),
                "model_revision": revision or getattr(model.config, "_commit_hash", None),
                "model_dimensions": {"layers": model.config.num_hidden_layers,
                                     "kv_heads": model.config.num_key_value_heads,
                                     "head_dim": model.config.hidden_size//model.config.num_attention_heads},
                "notes": "Fake quantization. Estimated packed storage is separate from actual memory. CPU unquantized reference uses float32."}
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2))
    elif any(old.get(field) != manifest.get(field) for field in ("device", "dtype", "torch", "transformers", "model_revision")):
        raise RuntimeError("Execution device, dtype, dependencies or model revision changed. Use a new output directory.")
    (out/"sample_ids.json").write_text(json.dumps(source["sample_ids"], indent=2))
    cal_path = out/"calibration.json"
    if cal_path.exists():
        calibration = json.loads(cal_path.read_text())
    else:
        train, train_source = gsm8k("train", cfg["calibration_samples"], cfg["seed"], ROOT/"data/cache")
        calibration = {"entropy": {"mean": 0.2, "std": 0.1}, "margin": {"mean": 2.0, "std": 2.0}}
        calibration_cfg = {**cfg, "max_new_tokens": cfg["calibration_max_new_tokens"]}
        values = {"entropy": [], "margin": []}
        for i, row in enumerate(train):
            result = generate(model, tokenizer, math_prompt(row["question"]), "fp16", calibration_cfg, calibration)
            append(out/"calibration_predictions.jsonl", {"id": row["id"], **result})
            for name in values:
                values[name].extend(s[name] for s in result["signals"])
            print(f"Calibration {i+1}/{len(train)}: {result['generated_tokens']} tokens in {result['seconds']:.1f}s", flush=True)
        for name, vals in values.items():
            calibration[name] = {"mean": float(np.mean(vals)), "std": max(float(np.std(vals)), 1e-5), "n_tokens": len(vals)}
        calibration["source"] = train_source
        calibration["note"] = "Statistics of smoothed per-token uncertainty from disjoint GSM8K training examples"
        cal_path.write_text(json.dumps(calibration, indent=2))
    output_path = out/"predictions.jsonl"
    done = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["id"], r["mode"]))
    for i, row in enumerate(rows):
        gold, _ = extract_answer(row["answer"], allow_fallback=False)
        for mode in cfg["modes"]:
            if (row["id"], mode) in done:
                continue
            result = generate(model, tokenizer, math_prompt(row["question"]), mode, cfg, calibration, cfg["seed"]+row["dataset_index"])
            predicted, rule = extract_answer(result["text"])
            # A truncated chain without an explicit final answer is not scored
            # using its last intermediate number.
            if result["truncated"] and rule == "last_number":
                predicted, rule = None, "truncated_without_final_answer"
            record = {"id": row["id"], "question": row["question"], "gold": gold,
                      "mode": mode, "predicted": predicted, "extraction_rule": rule,
                      "correct": predicted == gold and gold is not None, **result}
            append(output_path, record)
            print(f"{i+1}/{len(rows)} {mode:8s} correct={record['correct']} tokens={result['generated_tokens']} time={result['seconds']:.1f}s blocks={len(result['blocks'])}", flush=True)
    print(f"Completed. Results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
