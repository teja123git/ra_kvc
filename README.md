# Reasoning-Adaptive Key-Value Cache Compression (RA-KVC)

**DA4022 · Hardware-Aware Deep Learning · IIT Guwahati**

When a language model writes an answer, it generates one token at a time. At each attention layer, it saves **keys** that help it find relevant earlier tokens and **values** that carry information from those tokens. Reusing these vectors saves computation, but every new token makes the key-value (KV) cache larger. A long answer therefore needs more cache memory even though the model weights stay the same.

We studied whether the model could use its own uncertainty to decide **when extra cache precision is worth spending**. Storing a number with two bits gives it only four possible levels because there are 4 permutations. Four bits gives it sixteen, so it can retain more detail, at a higher storage cost. Our project tested whether changing which blocks receive that extra precision helps the model finish a math problem correctly.

## How RA-KVC works

1. **Form a block.** We kept the first eight tokens and at least the latest 32 tokens in floating point. We compressed older tokens in complete blocks of 32. Only blocks made entirely of generated tokens used adaptive allocation.
2. **Choose the channels.** Within each block, we ranked key channels by average absolute magnitude. Selected channels received four bits. The other eligible keys and all eligible values received two.
3. **Choose how many to protect.** The fixed policy protected the same fraction in each block. Our adaptive policies used either next-token entropy or the gap between the two highest token scores. We calibrated these signals on four separate training questions. We tested uncertainty as a possible indicator of sensitivity to rounding.
4. **Enforce the budget.** With 64 key channels and a 12.5% target, a generated block earns eight credits. Each credit buys one channel an upgrade from two bits to four. If one block spends zero, the next can spend sixteen. The controller can use only credits earned so far. Once it rounds a block, later attention steps use those values.

We held the model, prompts and channel ranking fixed across the mixed-precision policies. This let us ask whether *when* we spent the credits changed the final answers. We also tracked actual spending because an answer can end with unused credits.

## What we found

We ran Qwen2.5-0.5B-Instruct on CPU with greedy decoding. Each policy answered the same 20 GSM8K test questions with a limit of new 256 tokens.

| Policy | Correct answers | Responses that reached the token limit |
|---|---:|---:|
| Unquantized FP32 | 4/20 | 14 |
| Uniform 2-bit | 4/20 | 15 |
| Fixed mixed precision | 6/20 | 13 |
| Entropy-adaptive | 6/20 | 13 |
| Margin-adaptive | 6/20 | 13 |
| Random allocation | 6/20 | 13 |

All four mixed-precision policies solved exactly the same six questions. We observed no accuracy gain from using uncertainty to allocate precision on this sample. When we reran the unquantized reference with a 512-token limit, it solved 10/20 questions and only one response reached the limit. We also evaluated four policies on eight HumanEval tasks. Each scored 0/8.

We simulated the effect of rounding on subsequent model predictions. The CPU tensors remained floating point. Our storage estimates include payloads, scales, offsets, block metadata and floating buffers. The [report](report/RA_KVC_Report.pdf) gives the derivation and the measured CPU operation times.

## Explore the work

| Path | Contents |
|---|---|
| [Source](src/ra_kvc/) | Quantizer, signals, controller, cache integration and storage accounting |
| [Main results](results/local_pilot/) | Predictions, accuracy table, allocation traces and figures |
| [Length experiment](results/length_check/) | The same questions with a longer generation limit |
| [HumanEval](results/humaneval_pilot/) | Generated programs and functional test scores |
| [Validation](results/validation/) | Test and result-audit records |
| [Experiments](scripts/) and [configurations](configs/) | Benchmark, scoring and analysis commands |
| [Tests](tests/) | Checks for quantization, cache behavior, budget and scoring |

The [generation-length plots](results/length_check/figures/) accompany the 512-token reference run.

## Reproduce the results

Our CPU run used Python 3.12, PyTorch 2.6.0+cpu and Transformers 4.46.3. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python scripts/validate_results.py --run results/local_pilot --humaneval results/humaneval_pilot
```

To repeat model inference and regenerate the main plots:

```bash
python scripts/run_benchmark.py --config configs/pilot.json --output results/reproduced_pilot
python scripts/analyze.py --run results/reproduced_pilot
```

The saved manifests identify the model revision, dataset and question IDs. The internal label `fp16` denotes the unquantized policy. It ran in FP32 on CPU. The packed-storage calculations use a 16-bit reference.

**Team:** Kondapi Venkata Dhruv, Mathangi Venkatraman Srinivasan, Venkat Sai Teja Kalluri, Sanjay G and Vipul Sharma. Our quantization design draws on [Kitty](https://arxiv.org/abs/2511.18643). Dataset credits and licenses are in [docs](docs/THIRD_PARTY.md).
