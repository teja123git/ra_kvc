"""Generate HumanEval programs. Scoring is a separate, isolated step."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from ra_kvc.data import humaneval
from ra_kvc.generation import generate, load_model


def extract_program(text, problem):
    opening = re.search(r"```(?:python|py)?[ \t]*\r?\n", text, flags=re.I)
    if opening:
        code = text[opening.end():].split("```", 1)[0]
    else:
        code = text.strip()
    name = problem["entry_point"]
    if re.search(r"\bdef\s+"+re.escape(name)+r"\s*\(", code):
        preamble = problem["prompt"].split("def "+name)[0]
        return preamble + code
    return problem["prompt"] + code


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/local_pilot")
    p.add_argument("--output", default="results/humaneval_pilot")
    p.add_argument("--model-path")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--modes", nargs="+", default=["fp16", "uniform2", "fixed", "entropy"])
    args = p.parse_args()
    source = Path(args.run)
    cfg = json.loads((source/"manifest.json").read_text())["config"]
    cal = json.loads((source/"calibration.json").read_text())
    cfg = {**cfg, "max_new_tokens": args.max_new_tokens, "threads": args.threads}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    problems = humaneval(args.limit, cfg["seed"], ROOT/"data/cache")
    manifest = {"config": cfg, "task_ids": [r["task_id"] for r in problems], "modes": args.modes,
                "prompt_style": "Chat, complete Python function, code only", "calibration": "GSM8K train signals reused without HumanEval tuning"}
    manifest_file = out/"manifest.json"
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
        raise RuntimeError("Existing HumanEval output has a different configuration")
    manifest_file.write_text(json.dumps(manifest, indent=2))
    rows_file = out/"programs.jsonl"
    done = set()
    if rows_file.exists():
        done = {(r["task_id"], r["mode"]) for r in map(json.loads, rows_file.read_text().splitlines())}
    model, tokenizer = load_model(args.model_path or cfg["model"], cfg["device"],
                                  None if args.model_path else cfg.get("revision"), cfg["threads"])
    for index, problem in enumerate(problems):
        messages = [{"role": "system", "content": "You write correct Python functions. Return Python code only."},
                    {"role": "user", "content": "Complete the following function. Return the complete function definition and any necessary imports. Do not include tests or explanations.\n\n"+problem["prompt"]}]
        for mode in args.modes:
            if (problem["task_id"], mode) in done:
                continue
            result = generate(model, tokenizer, messages, mode, cfg, cal, cfg["seed"]+index)
            program = extract_program(result["text"], problem)
            combined = program+"\n\n"+problem["test"]+f"\ncheck({problem['entry_point']})\n"
            row = {"task_id": problem["task_id"], "mode": mode, "program": program,
                   "test": problem["test"], "entry_point": problem["entry_point"],
                   "execution_sha256": hashlib.sha256(combined.encode()).hexdigest(), **result}
            with rows_file.open("a") as f:
                f.write(json.dumps(row)+"\n")
            print(f"HumanEval {index+1}/{len(problems)} {mode} {result['generated_tokens']} tokens {result['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
