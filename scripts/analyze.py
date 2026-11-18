"""Summarize only paired completed examples and rebuild all main figures."""

import argparse
import csv
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ra_kvc.scoring import paired_bootstrap, wilson
from ra_kvc.memory import cache_storage

LABELS = {"fp16": "Unquantized", "uniform2": "Uniform 2-bit", "uniform4": "Uniform 4-bit",
          "fixed": "Fixed boost", "entropy": "Entropy-adaptive", "margin": "Margin-adaptive", "random": "Random allocation"}
COLORS = {"fp16": "#667085", "uniform2": "#CE6458", "uniform4": "#BBA34C",
          "fixed": "#4678B7", "entropy": "#008576", "margin": "#845BB0", "random": "#C48D4A"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/local_pilot")
    args = p.parse_args()
    run = Path(args.run)
    manifest = json.loads((run/"manifest.json").read_text())
    cfg = manifest["config"]
    rows = [json.loads(line) for line in (run/"predictions.jsonl").read_text().splitlines()]
    by_mode = {mode: {} for mode in cfg["modes"]}
    for row in rows:
        by_mode[row["mode"]][row["id"]] = row
    common = set.intersection(*(set(v) for v in by_mode.values()))
    ids = [x for x in manifest["dataset"]["sample_ids"] if x in common]
    if not ids:
        raise RuntimeError("No questions have completed every configured mode")
    d = manifest["model_dimensions"]["head_dim"]
    summary_rows = []
    for mode in cfg["modes"]:
        batch = [by_mode[mode][i] for i in ids]
        correct = sum(r["correct"] for r in batch)
        blocks = [b for r in batch for b in r["blocks"]]
        generated_blocks = [b for b in blocks if b["phase"] == "generated"]
        nquant = sum(b["end"]-b["start"] for b in blocks)
        ngenquant = sum(b["end"]-b["start"] for b in generated_blocks)
        fraction = sum((b["end"]-b["start"])*b["boosted"] for b in blocks)/(nquant*d) if nquant else 0
        gfraction = sum((b["end"]-b["start"])*b["boosted"] for b in generated_blocks)/(ngenquant*d) if ngenquant else 0
        bits = 16*sum(r["memory"]["estimated_total_bytes"] for r in batch)/sum(r["memory"]["reference_cache_bytes"] for r in batch)
        lo, hi = wilson(correct, len(batch))
        summary_rows.append({"mode": mode, "label": LABELS[mode], "n": len(batch),
                             "correct": correct, "accuracy": correct/len(batch), "ci95_low": lo, "ci95_high": hi,
                             "effective_bits_including_overhead": bits,
                             "mean_estimated_cache_KiB": np.mean([r["memory"]["estimated_total_bytes"]/1024 for r in batch]),
                             "mean_actual_cache_KiB": np.mean([r["actual_cache_tensor_bytes"]/1024 for r in batch]),
                             "mean_payload_KiB": np.mean([r["memory"]["packed_payload_bytes"]/1024 for r in batch]),
                             "mean_metadata_KiB": np.mean([r["memory"]["metadata_bytes"]/1024 for r in batch]),
                             "mean_floating_buffers_KiB": np.mean([r["memory"]["floating_buffer_bytes"]/1024 for r in batch]),
                             "mean_generated_tokens": np.mean([r["generated_tokens"] for r in batch]),
                             "truncated": sum(r["truncated"] for r in batch),
                             "missing_answer": sum(r["predicted"] is None for r in batch),
                             "mean_seconds": np.mean([r["seconds"] for r in batch]),
                             "boosted_fraction_all_quantized": fraction,
                             "boosted_fraction_generated": gfraction,
                             "generated_blocks": len(generated_blocks),
                             "requests_with_generated_blocks": sum(any(b["phase"] == "generated" for b in r["blocks"]) for r in batch)})
    comparisons = {}
    if "fixed" in by_mode:
        baseline = [by_mode["fixed"][i]["correct"] for i in ids]
        for mode in ("entropy", "margin", "random"):
            if mode in by_mode:
                comparisons[mode+"_minus_fixed"] = paired_bootstrap([by_mode[mode][i]["correct"] for i in ids], baseline)
    summary = {"status": "complete" if len(ids)==cfg["samples"] else "partial_paired_subset",
               "completed_paired_questions": len(ids), "planned_questions": cfg["samples"],
               "manifest": manifest, "methods": summary_rows, "paired_comparisons": comparisons,
               "scoring": "Numeric exact match; explicit answer preferred. Truncated outputs without explicit final answers count as incorrect.",
               "interpretation": "Exploratory evidence. Storage is projected, runtime is simulation overhead. Differences in realized budgets and lengths prevent an automatic equal-memory superiority claim."}
    (run/"summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(run/"metrics.csv", summary_rows)
    write_csv(run/"answers.csv", [{"id": i, "mode": m, "gold": by_mode[m][i]["gold"],
                                  "predicted": by_mode[m][i]["predicted"], "correct": by_mode[m][i]["correct"],
                                  "truncated": by_mode[m][i]["truncated"]} for i in ids for m in cfg["modes"]])
    figdir = run/"figures"
    figdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    for index, row in enumerate(summary_rows):
        ax.bar(index, 100*row["accuracy"], color=COLORS[row["mode"]], width=0.65)
        ax.errorbar(index, 100*row["accuracy"], yerr=[[100*(row["accuracy"]-row["ci95_low"])], [100*(row["ci95_high"]-row["accuracy"])]], color="#253147", capsize=4)
        ax.text(index, min(97, 100*row["ci95_high"]+3), f"{row['correct']}/{row['n']}", ha="center", fontsize=9)
    ax.set_xticks(range(len(summary_rows)), [r["label"].replace(" ", "\n", 1) for r in summary_rows], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("GSM8K exact-match accuracy (%)")
    ax.set_title(f"Paired pilot: {len(ids)} questions; 95% Wilson intervals")
    fig.tight_layout(); fig.savefig(figdir/"accuracy.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for j, r in enumerate(summary_rows):
        ax.scatter(r["effective_bits_including_overhead"], r["accuracy"]*100,
                   color=COLORS[r["mode"]], marker=["o", "s", "^", "D", "P", "X", "v"][j],
                   s=75, facecolors="none", linewidths=1.7,
                   label=f"{r['label']}: {r['effective_bits_including_overhead']:.3f} bits, {100*r['accuracy']:.1f}%")
    ax.set_xlabel("Projected bits per KV element, including buffers and metadata")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title("Observed storage/accuracy trade-off; generation lengths vary")
    ax.set_ylim(max(0, min(r["accuracy"]*100 for r in summary_rows)-12), min(100, max(r["accuracy"]*100 for r in summary_rows)+15))
    ax.margins(x=.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(figdir/"accuracy_storage.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    bottom = np.zeros(len(summary_rows))
    for field, label, color in [("mean_payload_KiB", "Integer payload", "#008576"),
                                 ("mean_metadata_KiB", "Scales, offsets and metadata", "#4678B7"),
                                 ("mean_floating_buffers_KiB", "16-bit floating region", "#BAC5D0")]:
        vals = np.array([r[field] for r in summary_rows])
        ax.bar(range(len(summary_rows)), vals, bottom=bottom, color=color, label=label)
        bottom += vals
    ax.set_xticks(range(len(summary_rows)), [r["label"].replace(" ", "\n", 1) for r in summary_rows], fontsize=9)
    ax.set_ylabel("Mean projected cache storage (KiB)")
    ax.set_title("Measured lengths with an analytical storage format")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir/"memory_breakdown.png"); plt.close(fig)

    if "entropy" in by_mode:
        chosen = max((by_mode["entropy"][i] for i in ids), key=lambda r: len(r["blocks"]))
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), sharex=True)
        signals = chosen["signals"]
        axes[0].plot([s["position"] for s in signals], [s["entropy"] for s in signals], color=COLORS["entropy"])
        axes[0].set_ylabel("Smoothed entropy")
        blocks = chosen["blocks"]
        for b in blocks:
            axes[1].plot([b["start"], b["end"]], [b["fraction"]*100]*2, color=COLORS["entropy"], linewidth=3)
        axes[1].axhline(cfg["target_fraction"]*100, linestyle="--", color=COLORS["fixed"], label="Fixed target")
        axes[1].set_ylabel("Protected K channels (%)"); axes[1].set_xlabel("Token position in prompt + response")
        axes[1].legend(); axes[0].set_title(f"Allocation trace: {chosen['id']} (longest logged cache trace)")
        fig.tight_layout(); fig.savefig(figdir/"allocation_trace.png"); plt.close(fig)

    dims = manifest["model_dimensions"]
    projections = []
    for length in (256, 512, 1024, 2048, 4096, 8192, 16384):
        for mode in ("fp16", "uniform2", "fixed", "uniform4"):
            recs = []
            if mode != "fp16":
                for start in range(cfg["sink_tokens"], length-cfg["recent_tokens"]-cfg["group_size"]+1, cfg["group_size"]):
                    recs.append({"start": start, "end": start+cfg["group_size"], "mode": mode,
                                 "boosted": round(d*cfg["target_fraction"]) if mode=="fixed" else 0})
            projections.append({"mode": mode, "tokens": length,
                                **cache_storage(dims["layers"], dims["kv_heads"], dims["head_dim"], length, recs)})
    write_csv(run/"memory_projection.csv", projections)
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    for mode in ("fp16", "uniform2", "fixed", "uniform4"):
        subset = [r for r in projections if r["mode"]==mode]
        ax.plot([r["tokens"] for r in subset], [r["estimated_total_bytes"]/2**20 for r in subset], marker="o", color=COLORS[mode], label=LABELS[mode])
    ax.set_xlabel("Context length (tokens)"); ax.set_ylabel("Projected packed KV cache (MiB)")
    ax.set_title("Analytical projection with metadata and FP16 buffers; batch size 1")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir/"memory_projection.png"); plt.close(fig)
    print(json.dumps({"paired_questions": len(ids), "accuracy": {r["mode"]: r["accuracy"] for r in summary_rows}}, indent=2))


if __name__ == "__main__":
    main()
