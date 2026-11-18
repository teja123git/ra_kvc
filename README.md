# Reasoning-Adaptive Key-Value Cache Compression (RA-KVC)

DA4022 Hardware-Aware Deep Learning course project, IIT Guwahati.


The experiment changes the **fraction** of key-cache channels protected at
4 bits across generated token blocks. Other eligible key channels and all
eligible values use 2 bits. A causal credit controller limits the cumulative
precision allocation. Channel selection uses mean absolute key magnitude.

This is an accuracy simulation using fake quantization. Floating tensors
hold the rounded/dequantized values. Packed storage is an analytical
projection, and actual process/GPU memory is logged separately.

## What is included

| Folder | Purpose |
|---|---|
| `src/ra_kvc/` | Quantizer, cache adapter, signals, controller, scorer and memory model |
| `configs/` | CPU pilot, rapid GPU study and extended study |
| `scripts/` | Benchmarking, signal diagnostic, plotting, report generation and code evaluation |
| `tests/` | Quantizer, budget, cache integration and scorer checks |
| `notebooks/` | Colab notebook that accepts the project ZIP |
| `results/` | Actual pilot outputs, metrics, plots and validation records |
| `report/` | Editable Markdown report and PDF |
| `docs/` | Method, reproduction guidance and source references |

## Fastest route

The archive includes saved pilot outputs. Reading the report and inspecting
those results requires no model download. To run a larger experiment, open
`notebooks/Run_RA_KVC_Colab.ipynb` in Colab, choose a GPU runtime, upload the
project ZIP and run the cells. The notebook rebuilds plots and the report
from the newly generated outputs. It does not depend on a GitHub repository.

The rapid configuration evaluates 50 predetermined GSM8K test questions on
Qwen2.5-1.5B-Instruct with six methods and a 512-token generation cap. Runtime depends on the GPU, generated
lengths and downloads. The notebook checks for a GPU before starting.

## Local installation

Python 3.10 or newer is required. The delivered CPU run used Python 3.12,
PyTorch 2.6.0+cpu and Transformers 4.46.3. Keep Transformers pinned because
the cache adapter uses its legacy tuple interface. PyTorch 2.5+ is supported
by the code, but other versions/devices require the included integration test.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

Use the PyTorch installation appropriate for your GPU. A CUDA build is
required for `configs/fast_gpu.json`; CPU-only PyTorch remains sufficient
for the tests and pilot configuration.

## Reproduce the CPU pilot

```bash
python scripts/run_benchmark.py --config configs/pilot.json --output results/reproduced_pilot
python scripts/analyze.py --run results/reproduced_pilot
python scripts/diagnose_signal.py --run results/reproduced_pilot
python scripts/profile_overhead.py --run results/reproduced_pilot
python scripts/validate_results.py --run results/reproduced_pilot --output results/reproduced_pilot/evidence_audit.json
python scripts/build_report.py --run results/reproduced_pilot
```

The unquantized reference is FP32 on CPU. The mode identifier `fp16` denotes
the unquantized reference for compatibility across configurations; on GPU
the reference uses BF16 where supported and otherwise FP16. The storage
projection always uses a specified 16-bit reference and states this clearly.

## Run the rapid GPU study

```bash
python scripts/run_benchmark.py --config configs/fast_gpu.json --output results/gpu_run
python scripts/analyze.py --run results/gpu_run
python scripts/diagnose_signal.py --run results/gpu_run
python scripts/build_report.py --run results/gpu_run
```

The runner resumes completed `(question_id, method)` pairs. It refuses to
mix configurations in the same output directory. Changing the model,
sample count or decoding settings requires a new output directory. Analysis
uses only questions completed by every configured method and labels partial
runs. Do not compare subsets with different question identities.

The delivered `results/length_check/` is a separate unquantized-reference run
on the same 20 questions with a 512-token cap. It was added after the pilot
showed frequent truncation. It is a generation-length diagnostic, not a
longer-cap adaptive comparison. `configs/length_check.json` reproduces it.

## HumanEval

```bash
python scripts/run_humaneval.py --run results/gpu_run --output results/humaneval_gpu --limit 40
docker pull python:3.11-slim
python scripts/score_humaneval.py --run results/humaneval_gpu
python scripts/build_report.py --run results/gpu_run --humaneval results/humaneval_gpu
```

Generation does not execute programs. Scoring defaults to Docker with no
network and a read-only filesystem. Colab normally does not provide Docker;
download the generated programs and score them on a Docker host. The saved
small pilot may instead use explicitly reviewed code with exact SHA256
approval records and time/memory limits. That mode is not a general sandbox
and must never be enabled for unreviewed generated programs.

## How the method works

1. The model generates one token with cached K/V tensors.
2. We obtain normalized output entropy and the top-two logit margin.
3. A moving average smooths both signals. Calibration uses disjoint training examples.
4. Newly eligible 32-token cache blocks leave the recent-token buffer.
5. A fixed or adaptive policy chooses how many key channels receive 4-bit precision.
6. Both methods rank channels with the same mean-magnitude rule.
7. The block is quantized once, and those values are used on subsequent steps.

Prompt blocks and prompt/response boundary blocks use the same fixed rule
in fixed and adaptive methods. Only wholly generated blocks adapt. Initial
8 tokens and at least the most recent 32 tokens stay unquantized. An
incomplete group also stays unquantized. This is a simplified Kitty-inspired
layout, not a reproduction of Kitty's production kernels or exact defaults.

## Results and interpretation

Read `results/local_pilot/summary.json`, `metrics.csv`, `answers.csv` and
`predictions.jsonl`. Every accuracy number comes from saved generated text.
Read the manifest for the model revision, dataset checksum, sample IDs,
dtype, dependency versions and settings.

The pilot is exploratory. Twenty examples are too few for broad performance
claims, and the small model's baseline capability and truncation matter.
Higher uncertainty does not establish that a token contains reasoning.
We report realized budget differences and answer lengths. An adaptive policy
cannot claim an equal-memory win just because its nominal target equals the
fixed target. The comparison also needs comparable realized storage.

No real INT2 packing, custom GPU kernels, speculative decoding or model
training is claimed. Larger-model configurations are provided as runnable
extensions; a configuration file alone is not evidence that it was run.


## Included CPU results

Qwen2.5-0.5B-Instruct, 20 predetermined GSM8K questions, greedy decoding, 256-token cap.

| Method | Correct | Accuracy | Truncated |
|---|---:|---:|---:|
| Unquantized | 4/20 | 20% | 14/20 |
| Uniform 2-bit | 4/20 | 20% | 15/20 |
| Fixed boost | 6/20 | 30% | 13/20 |
| Entropy-adaptive | 6/20 | 30% | 13/20 |
| Margin-adaptive | 6/20 | 30% | 13/20 |
| Random allocation | 6/20 | 30% | 13/20 |

Fixed, entropy, margin and random allocation have identical correctness on this cohort. These measurements do not establish an adaptive accuracy gain.

A separate unquantized length diagnostic reaches **10/20 (50%)** at a 512-token cap, with only **1/20** truncated responses. It uses the same question IDs and demonstrates a major length-limit confounder. Other policies were not rerun at that longer cap. HumanEval yields **0/8** for each of four tested methods and cannot distinguish retention quality in this slice.

The package includes 16 passing unit/integration tests and result audits with no missing method/question records. Storage numbers are analytical projections; floating tensors retain the real cache. See `report/RA_KVC_Report.pdf` and `docs/DEFENCE_QA.md` for interpretation.

## Team responsibilities

See `CONTRIBUTIONS.md`. The table defines proposed verification/ownership
areas. Replace pending entries with work actually completed by each member
before submitting an individual-contribution statement.

## References

- Kitty paper: https://arxiv.org/abs/2511.18643
- Official Kitty implementation: https://github.com/Summer-Summer/Kitty
- SEQUOIA: https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract-Conference.html
- Qwen2.5 models: https://huggingface.co/Qwen
- GSM8K: https://github.com/openai/grade-school-math
- HumanEval: https://github.com/openai/human-eval

Downloaded model weights and dataset caches are not included in this
archive and retain their upstream licenses. This project implements its
own small simulation and credits the research ideas above.
