# Execution notes

The delivered 20-question pilot resumes predictions saved before the conversation
was interrupted. Completed question/method pairs were retained, and remaining
pairs use the same configuration, model checkpoint and CPU floating dtype.

CPU inference shares the host with artifact processing and, during part of the
run, a separate generation-length diagnostic. Per-request wall times are real
observations, but not controlled latency benchmarks. They should not be used to
claim throughput gains or precise relative implementation overhead.

`predictions.jsonl` is the measurement record. `summary.json`, CSVs and figures
are derived from it. `scripts/validate_results.py` checks the derived counts,
budget bounds, cache sizes and answer extraction without running the model.

The operation microbenchmark uses actual model tensors and records individual
timings. It is descriptive and also has no dedicated hardware isolation.
