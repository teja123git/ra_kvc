# Sources and third-party material

The simulation is a small independent implementation inspired by KV-cache
quantization research. It does not copy Kitty's production kernels or claim to
reproduce Kitty's published accuracy and throughput configuration.

| Resource | Use in this project | Source |
|---|---|---|
| Kitty | Mixed key-channel precision motivation | https://arxiv.org/abs/2511.18643 |
| Kitty implementation | Reference implementation, not bundled | https://github.com/Summer-Summer/Kitty |
| SEQUOIA | Related speculative-decoding work, future signal option | https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract-Conference.html |
| Qwen2.5-0.5B-Instruct | Model used for the saved CPU experiments | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct |
| GSM8K | Question text, gold answers and deterministic evaluation IDs | https://github.com/openai/grade-school-math |
| HumanEval | Function prompts and original tests in saved evaluation records | https://github.com/openai/human-eval |
| Transformers 4.46.3 | Model and legacy cache interface | https://github.com/huggingface/transformers/tree/v4.46.3 |

The upstream GSM8K and HumanEval license notices are preserved in `licenses/`.
Saved result files contain selected dataset questions/tests for reproducibility.
Downloaded model weights are excluded and remain subject to their own model
licenses. Larger-model configuration files do not imply that those models were
executed in the delivered study.
