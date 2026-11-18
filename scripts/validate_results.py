"""Audit saved experiment evidence without downloading or running a model."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ra_kvc.memory import cache_storage
from ra_kvc.scoring import extract_answer


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/local_pilot")
    parser.add_argument("--humaneval")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", default="results/validation/evidence_audit.json")
    args = parser.parse_args()
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text())
    cfg, dims = manifest["config"], manifest["model_dimensions"]
    rows = read_rows(run / "predictions.jsonl")
    expected = {(i, m) for i in manifest["dataset"]["sample_ids"] for m in cfg["modes"]}
    seen = set()
    d = dims["head_dim"]
    for row in rows:
        identity = (row["id"], row["mode"])
        require(identity in expected and identity not in seen, f"Unexpected/duplicate record: {identity}")
        seen.add(identity)
        require(row["generated_tokens"] == len(row["token_ids"]), f"Token count: {identity}")
        require(len(row["signals"]) == row["generated_tokens"], f"Signal count: {identity}")
        n = row["prompt_tokens"] + row["generated_tokens"]
        computed = cache_storage(dims["layers"], dims["kv_heads"], d, n, row["blocks"])
        require(computed == row["memory"], f"Storage mismatch: {identity}")
        scalar_bytes = {"torch.float32": 4, "torch.float16": 2, "torch.bfloat16": 2}[row["actual_cache_dtype"]]
        require(row["actual_cache_tensor_bytes"] == 2*dims["layers"]*dims["kv_heads"]*d*n*scalar_bytes,
                f"Actual floating tensor size: {identity}")
        end = cfg["sink_tokens"]
        earned, spent = 0, 0
        for block in row["blocks"]:
            require(block["start"] == end and block["end"] - end == cfg["group_size"],
                    f"Cache block alignment: {identity}")
            require(block["end"] <= n - cfg["recent_tokens"], f"Recent buffer: {identity}")
            end = block["end"]
            require(block["mode"] == row["mode"], f"Block method: {identity}")
            require(block["fraction"] == block["boosted"]/d, f"Channel fraction: {identity}")
            require((block["phase"] == "generated") == (block["start"] >= row["prompt_tokens"]),
                    f"Prompt/response phase: {identity}")
            if row["mode"] in ("entropy", "margin", "random") and block["phase"] == "generated":
                earned += cfg["target_fraction"]*d
                spent += block["boosted"]
                require(0 <= earned-spent <= cfg["target_fraction"]*d+1, f"Causal credit bound: {identity}")
                require(abs(block["credit"]-(earned-spent)) < 1e-7, f"Recorded credit: {identity}")
        answer, rule = extract_answer(row["text"])
        if row["truncated"] and rule == "last_number":
            answer, rule = None, "truncated_without_final_answer"
        require(answer == row["predicted"] and rule == row["extraction_rule"], f"Answer extraction: {identity}")
        require(row["correct"] == (answer == row["gold"] and row["gold"] is not None), f"Correctness: {identity}")
    if not args.allow_partial:
        require(seen == expected, f"Missing {len(expected-seen)} method/question records")
    paired = [i for i in manifest["dataset"]["sample_ids"] if all((i, m) in seen for m in cfg["modes"])]
    summary = json.loads((run / "summary.json").read_text())
    require(summary["completed_paired_questions"] == len(paired), "Stale summary: rerun analyze.py")
    for entry in summary["methods"]:
        selected = [r for r in rows if r["mode"] == entry["mode"] and r["id"] in paired]
        correct = sum(r["correct"] for r in selected)
        require(entry["n"] == len(paired) and entry["correct"] == correct, "Summary counts differ from raw answers")
        require(entry["accuracy"] == correct/len(selected), "Summary accuracy differs from raw answers")
    calibration = json.loads((run / "calibration.json").read_text())
    require(calibration["source"]["split"] == "train" and manifest["dataset"]["split"] == "test",
            "Calibration/test split mismatch")
    human_count = None
    if args.humaneval:
        human = Path(args.humaneval)
        programs = read_rows(human / "programs.jsonl")
        scores = read_rows(human / "scores.jsonl")
        by_id = {(s["task_id"], s["mode"]): s for s in scores}
        require(len(by_id) == len(scores) == len(programs), "HumanEval duplicate/missing scores")
        for record in programs:
            code = record["program"] + "\n\n" + record["test"] + f"\ncheck({record['entry_point']})\n"
            digest = hashlib.sha256(code.encode()).hexdigest()
            score = by_id[(record["task_id"], record["mode"])]
            require(digest == record["execution_sha256"] == score["execution_sha256"], "HumanEval hash mismatch")
        hs = json.loads((human / "summary.json").read_text())
        for entry in hs["methods"]:
            selected = [r for r in scores if r["mode"] == entry["mode"]]
            require(entry["passed"] == sum(r["passed"] for r in selected) and entry["n"] == len(selected),
                    "HumanEval summary differs from scores")
        human_count = len(programs)
    result = {"status": "passed", "audited_utc": datetime.now(timezone.utc).isoformat(),
              "gsm8k_records": len(rows), "paired_questions": len(paired),
              "missing_records": len(expected-seen), "humaneval_programs_and_scores": human_count,
              "checks": ["unique predetermined examples", "token counts", "cache dimensions and buffers",
                         "recomputed storage", "causal allocation budget", "answer rescoring",
                         "summary counts", "disjoint calibration split", "program/test execution hashes"],
              "prediction_sha256": hashlib.sha256((run / "predictions.jsonl").read_bytes()).hexdigest()}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
