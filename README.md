# Reasoning-Adaptive Key-Value Cache Compression (RA-KVC)

**DA4022 · Hardware-Aware Deep Learning · IIT Guwahati**

## Overview

Autoregressive language models generate responses one token at a time. During generation, each attention layer maintains a **key-value (KV) cache** containing representations of previously processed tokens. Keys determine how new tokens attend to earlier context, while values provide the information retrieved through attention.

Caching these representations avoids repeatedly computing them, but memory consumption grows with every generated token. For long sequences, the KV cache can become a significant memory bottleneck, motivating compression techniques that reduce storage without substantially degrading generation quality .

**Reasoning-Adaptive Key-Value Cache Compression (RA-KVC)** investigates whether a language model's own uncertainty can guide the allocation of cache precision. The central idea is simple: not every cached representation necessarily requires the same numerical precision. If certain blocks are more sensitive to quantization errors, allocating additional bits to them could potentially preserve more useful information.

We implement and evaluate a block-wise mixed-precision quantization framework that uses either next-token entropy or the margin between the two highest token scores to guide precision allocation. Our objective is to determine whether these uncertainty signals can improve the effectiveness of a fixed cache-precision budget.

## Methodology

RA-KVC combines block-wise KV-cache quantization, channel-level precision allocation, uncertainty estimation, and a budget-constrained controller.

### 1. Block-wise cache organization

We divide the generated KV cache into blocks of 32 tokens. The initial eight tokens and at least the most recent 32 tokens remain in floating-point precision to preserve the initial context and recent information.

Only complete blocks consisting entirely of generated tokens are eligible for adaptive quantization. This provides a consistent unit for precision allocation and budget accounting.

### 2. Channel-wise mixed-precision quantization

Within each eligible block, key channels are ranked according to their average absolute magnitude.

- **4-bit precision:** Assigned to selected key channels considered more important by the magnitude-based ranking.
- **2-bit precision:** Assigned to the remaining eligible key channels and all eligible value channels.

A 2-bit representation provides four quantization levels, whereas a 4-bit representation provides sixteen. The additional precision increases storage cost but can retain more numerical detail.

This design keeps the channel-selection strategy consistent across policies while allowing the allocation of higher precision to vary.

### 3. Uncertainty-guided precision allocation

We investigate two signals derived from the model's next-token predictions:

- **Entropy-adaptive allocation:** Uses next-token distribution entropy to estimate prediction uncertainty.
- **Margin-adaptive allocation:** Uses the difference between the highest and second-highest token scores as an alternative indicator of prediction confidence.

Both signals are calibrated using four separate training questions. We investigate whether these prediction-level signals can identify blocks that are more sensitive to quantization.

For comparison, we also implement a fixed allocation policy that protects the same fraction of channels in every block, alongside a random allocation baseline.

### 4. Budget-constrained adaptive controller

Precision allocation is governed by a credit-based budget.

For a configuration with 64 key channels and a target 4-bit allocation of 12.5%, each eligible block earns eight credits. Each credit allows one additional key channel to be upgraded from 2-bit to 4-bit precision.

Unused credits carry forward to subsequent blocks. For example, if a block uses no credits, the next block can spend sixteen. However, the controller cannot spend credits that have not yet been accumulated.

Once a block is quantized, its rounded values are used in subsequent attention computations. This makes the allocation decisions persistent rather than treating each block as an independent, reversible experiment.

We also track actual credit expenditure, since the final allocation may leave some of the available budget unused.

## Experimental Setup

We evaluate RA-KVC using **Qwen2.5-0.5B-Instruct** on CPU with greedy decoding.

| Configuration | Setting |
|---|---|
| Model | Qwen2.5-0.5B-Instruct |
| Hardware | CPU |
| Decoding | Greedy |
| Primary benchmark | GSM8K |
| Evaluation sample | 20 test questions |
| Generation limit | 256 new tokens |
| KV block size | 32 tokens |
| Protected initial context | 8 tokens |
| Minimum recent FP buffer | 32 tokens |
| Quantization precision | 2-bit / 4-bit |
| Key channels | 64 |
| Target upgrade fraction | 12.5% |

All mixed-precision policies use the same model, prompts, channel ranking, and evaluation questions. This allows us to isolate the effect of changing the allocation strategy.

## Results and Observations

### GSM8K evaluation

The following table reports the number of correctly solved questions and the number of responses that reached the generation limit.

| Policy | Correct answers | Reached token limit |
|---|---:|---:|
| Unquantized FP32 | 4/20 | 14 |
| Uniform 2-bit | 4/20 | 15 |
| Fixed mixed precision | 6/20 | 13 |
| Entropy-adaptive | 6/20 | 13 |
| Margin-adaptive | 6/20 | 13 |
| Random allocation | 6/20 | 13 |

The mixed-precision policies achieved the same result of **6/20 correct answers**, compared with 4/20 for both the unquantized reference and uniform 2-bit quantization.

However, all four mixed-precision policies solved exactly the same six questions. Neither entropy-based nor margin-based allocation produced an observable accuracy improvement over fixed or random allocation.

This is an important finding: although model uncertainty provides a convenient signal for adaptive allocation, it does not necessarily identify the cache blocks that are most sensitive to quantization.

The results therefore do not establish an advantage for uncertainty-guided allocation on this evaluation sample.

### Generation-length investigation

We also investigated whether the generation limit influenced the observed performance.

When the unquantized reference was rerun with a limit of 512 new tokens, it solved **10/20 questions**, compared with 4/20 under the original 256-token limit. Only one response reached the longer generation limit.

This suggests that generation length and answer completion are important confounding factors when interpreting the initial accuracy results. The longer-limit experiment also highlights why token-limit statistics should be considered alongside exact-match accuracy.

### HumanEval evaluation

We additionally evaluated four policies on eight HumanEval programming tasks.

All four policies obtained **0/8 functional correctness scores**. Consequently, this experiment does not provide evidence of useful performance on code-generation tasks and should be treated as a preliminary evaluation rather than a meaningful benchmark comparison.

## Implementation and Measurement Scope

RA-KVC implements the quantization logic, uncertainty estimators, allocation controller, cache integration, and storage accounting.

An important distinction concerns the execution and storage models:

- **Prediction simulation:** Quantization and rounding effects are simulated to study their influence on subsequent model predictions.
- **CPU execution:** The underlying CPU tensors remain floating point; this is not an end-to-end packed low-bit inference implementation.
- **Storage estimation:** Memory calculations account for quantized payloads, scales, offsets, block metadata, and floating-point buffers.

The internal experiment label `fp16` refers to the unquantized reference policy, which actually executes in FP32 on CPU. Storage estimates use a 16-bit reference representation.

Therefore, the reported experiments evaluate the effects of quantization-induced rounding and allocation strategies, rather than demonstrating actual low-bit CPU inference acceleration or measured hardware memory-bandwidth improvements.

See the [project report](report/RA_KVC_Report.pdf) for the derivations, detailed storage calculations, and measured CPU operation times.

## Repository Structure

| Path | Description |
|---|---|
| `src/ra_kvc/` | Quantizer, uncertainty signals, allocation controller, cache integration, and storage accounting |
| `results/local_pilot/` | Main predictions, accuracy results, allocation traces, and figures |
| `results/length_check/` | Extended generation-length experiment |
| `results/humaneval_pilot/` | Generated programs and functional evaluation results |
| `results/validation/` | Test records and result-audit artifacts |
| `scripts/` | Benchmark execution, scoring, validation, and analysis scripts |
| `configs/` | Experiment configurations |
| `tests/` | Tests for quantization, cache behavior, budget accounting, and scoring |
| `report/` | Detailed project report |

Generation-length plots are available under `results/length_check/figures/`.

## Reproducing the Experiments

The experiments were conducted using:

- Python 3.12
- PyTorch 2.6.0+cpu
- Transformers 4.46.3

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

# Run unit tests
python -m unittest discover -s tests -v

# Validate saved experimental results
python scripts/validate_results.py \
    --run results/local_pilot \
    --humaneval results/humaneval_pilot
```

To rerun the main benchmark and regenerate its analysis:

```bash
python scripts/run_benchmark.py \
    --config configs/pilot.json \
    --output results/reproduced_pilot

python scripts/analyze.py --run results/reproduced_pilot
```

Saved experiment manifests record the model revision, dataset, and question IDs to support reproducibility.

## Team

- Kondapi Venkata Dhruv
- Mathangi Venkatraman Srinivasan
- Venkat Sai Teja Kalluri
- Sanjay G
- Vipul Sharma

**Reference:** Our quantization design draws inspiration from [Kitty](https://arxiv.org/abs/2511.18643).

Dataset acknowledgements and licensing information are available in [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).

---

*DA4022 · Hardware-Aware Deep Learning · Indian Institute of Technology Guwahati*
