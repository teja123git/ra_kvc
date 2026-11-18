"""Download official datasets and save deterministic sample identities."""

import gzip
import hashlib
import json
import random
import urllib.request
from pathlib import Path

GSM_ROOT = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
HUMAN_EVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with urllib.request.urlopen(url, timeout=90) as response:
            content = response.read()
        temporary = destination.with_suffix(destination.suffix+".part")
        temporary.write_bytes(content)
        temporary.replace(destination)
    return destination


def gsm8k(split, limit, seed=17, cache_dir="data/cache"):
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    path = download(f"{GSM_ROOT}/{split}.jsonl", Path(cache_dir)/f"gsm8k_{split}.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected = [{**rows[i], "id": f"gsm8k_{split}_{i}", "dataset_index": i}
                for i in indices[:limit]]
    return selected, {"source_url": f"{GSM_ROOT}/{split}.jsonl", "split": split,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "seed": seed, "sample_ids": [r["id"] for r in selected]}


def humaneval(limit, seed=17, cache_dir="data/cache"):
    path = download(HUMAN_EVAL_URL, Path(cache_dir)/"HumanEval.jsonl.gz")
    rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
    random.Random(seed).shuffle(rows)
    return rows[:limit]
