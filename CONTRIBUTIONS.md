# Individual contributions

Each member is assigned 20% of the planned workload across the five technical areas below. It does not assert that the named
members already performed these tasks. Each person should inspect the code,
run or verify their component, write their section and record the evidence.

| Member | Roll number | Assigned work | Evidence to record | Status |
|---|---|---|---|---|
| Kondapi Venkata Dhruv | 230122024 | Quantizer and uniform/fixed baselines | Tests, quantizer review, baseline experiment | Pending member verification |
| Mathangi Venkatraman Srinivasan | 230121037 | Entropy/margin signals and sensitivity diagnostic | Calibration review, diagnostic interpretation | Pending member verification |
| Venkat Sai Teja Kalluri | 230121032 | Adaptive controller and cache integration | Budget tests, allocation trace, algorithm write-up | Pending member verification |
| Sanjay G | 230121049 | Experiment runner, scoring and statistics | Saved predictions, scoring audit, plots | Pending member verification |
| Vipul Sharma | 230103117 | Storage model and hardware analysis | Memory derivation, buffer/metadata checks, limitations | Pending member verification |

For the final report, use concrete wording such as: "Verified the controller
credit bound on the saved allocation traces and wrote the budget-accounting
section." Use that wording only after completing the stated work.

Shared work can include result interpretation, report editing, repository
packaging and presentation recording. Record shared work separately from
each member's main responsibility. The initial implementation and draft
artifacts were prepared with AI assistance and require team review.

## Files and review steps

| Member | Files to review | Concrete check | Report section |
|---|---|---|---|
| Dhruv | `src/ra_kvc/quantization.py`, `tests/test_core.py` | Explain K/V quantization axes and constant-group preservation | 2, quantization design |
| Srinivasan | `src/ra_kvc/signals.py`, `scripts/diagnose_signal.py`, `results/local_pilot/signal_diagnostic.json` | Trace one block's uncertainty and explain why eight diagnostic blocks are inconclusive | 2 and 8, signals and diagnostic |
| Teja | `src/ra_kvc/controller.py`, `cache.py`, `generation.py` | Check earned/spent credits and explain why finalized blocks cannot regain precision | 3, causal controller |
| Sanjay | `scripts/run_benchmark.py`, `analyze.py`, `validate_results.py`, `results/local_pilot/answers.csv` | Audit predictions, regenerate the table and explain truncation and paired uncertainty | 4, 5 and 5.1, evaluation |
| Vipul | `src/ra_kvc/memory.py`, `scripts/profile_overhead.py`, `results/local_pilot/memory_projection.csv` | Derive payload plus metadata bytes and distinguish FP32 tensors from projected storage | 6 and 7, hardware analysis |

After doing the work, add a dated entry below with the actual change, check or
interpretation completed. A commit or saved audit output is useful evidence.

## Completed member work

Pending team entries.
