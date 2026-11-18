"""Execute generated functions in Docker, or after explicit code review.

Docker is the default. The reviewed mode requires a manifest containing the
exact SHA256 of every reviewed program AND its test harness. The resource
limits in reviewed mode are time/memory guards, not a security sandbox.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from ra_kvc.scoring import wilson, paired_bootstrap


def limits():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024*1024, 1024*1024))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/humaneval_pilot")
    p.add_argument("--reviewed-manifest", help="Exact hashes approved after inspecting code and tests")
    args = p.parse_args()
    run = Path(args.run)
    records = [json.loads(line) for line in (run/"programs.jsonl").read_text().splitlines()]
    reviewed = None
    if args.reviewed_manifest:
        reviewed = set(json.loads(Path(args.reviewed_manifest).read_text())["reviewed_execution_hashes"])
        if os.name != "posix":
            raise RuntimeError("Reviewed execution with resource limits requires POSIX")
    elif shutil.which("docker") is None:
        raise RuntimeError("Docker unavailable. Generate programs now and score on a Docker host, or inspect every program/test and supply its exact review manifest.")
    if reviewed is None:
        image_check = subprocess.run(["docker", "image", "inspect", "python:3.11-slim"], capture_output=True)
        if image_check.returncode:
            raise RuntimeError("Prepare the evaluation image first: docker pull python:3.11-slim")
    results = []
    for record in records:
        code = record["program"]+"\n\n"+record["test"]+f"\ncheck({record['entry_point']})\n"
        digest = hashlib.sha256(code.encode()).hexdigest()
        if digest != record["execution_sha256"]:
            raise RuntimeError("Program hash mismatch")
        if reviewed is not None and digest not in reviewed:
            raise RuntimeError(f"Unreviewed program: {record['task_id']} {record['mode']}")
        try:
            with tempfile.TemporaryDirectory(prefix="ra_kvc_eval_") as tmp:
                path = Path(tmp)/"check.py"; path.write_text(code)
                if reviewed is None:
                    container_id = Path(tmp)/"container_id"
                    command = ["docker", "run", "--rm", "--network", "none", "--read-only",
                               "--pids-limit", "32", "--memory", "512m", "--cpus", "1",
                               "--ulimit", "cpu=4:4", "--cidfile", str(container_id),
                               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                               "--user", "65534:65534", "--mount", f"type=bind,src={tmp},dst=/work,readonly",
                               "python:3.11-slim", "python", "-I", "-S", "/work/check.py"]
                    os.chmod(tmp, 0o755); os.chmod(path, 0o644)
                    try:
                        proc = subprocess.run(command, capture_output=True, text=True, timeout=20)
                    finally:
                        if container_id.exists():
                            subprocess.run(["docker", "rm", "-f", container_id.read_text().strip()],
                                           capture_output=True, timeout=10)
                    if proc.returncode in (125, 126, 127):
                        raise RuntimeError("Docker infrastructure failed: "+proc.stderr[-500:])
                else:
                    proc = subprocess.run([sys.executable, "-I", "-S", str(path)], cwd=tmp,
                                          env={"PATH": os.defpath}, capture_output=True, text=True,
                                          timeout=6, preexec_fn=limits)
            passed, error = proc.returncode==0, proc.stderr[-1200:]
        except subprocess.TimeoutExpired:
            passed, error = False, "timeout"
        results.append({"task_id": record["task_id"], "mode": record["mode"], "passed": passed,
                        "error": error, "execution_sha256": digest,
                        "execution_method": "manually_reviewed_with_resource_limits" if reviewed is not None else "docker"})
    (run/"scores.jsonl").write_text("".join(json.dumps(r)+"\n" for r in results))
    manifest = json.loads((run/"manifest.json").read_text())
    modes = manifest["modes"]
    by_mode = {m: {r["task_id"]: r for r in results if r["mode"]==m} for m in modes}
    ids = sorted(set.intersection(*(set(v) for v in by_mode.values())))
    stats = []
    for mode in modes:
        count = sum(by_mode[mode][i]["passed"] for i in ids)
        stats.append({"mode": mode, "n": len(ids), "passed": count,
                      "pass_at_1": count/len(ids) if ids else None, "ci95": wilson(count,len(ids))})
    summary = {"n_tasks": len(ids), "methods": stats, "manifest": manifest,
               "note": "One greedy program per task. Small subset with a documented chat prompt. No claim of full HumanEval performance."}
    (run/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
