"""Greedy batch-one generation with post-forward cache finalization."""

import time
import torch
from .cache import CacheQuantizer, as_legacy
from .memory import cache_storage
from .signals import SignalTracker, uncertainty


def load_model(model_id, device="auto", revision=None, threads=4):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(threads)
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.startswith("cuda") else torch.float32
    kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa", **kwargs).to(device).eval()
    if model.config.model_type != "qwen2":
        raise ValueError("This release validates Qwen2/Qwen2.5 only")
    return model, tokenizer


def math_prompt(question):
    return [{"role": "system", "content": "You are a helpful mathematics assistant."},
            {"role": "user", "content": question + "\nSolve step by step, concisely. End with #### followed by the final numeric answer."}]


@torch.inference_mode()
def generate(model, tokenizer, messages, mode, config, calibration, seed=17, raw_prompt=None):
    device = next(model.parameters()).device
    if raw_prompt is None:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = raw_prompt
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_length = ids.shape[1]
    if prompt_length + config["max_new_tokens"] > model.config.max_position_embeddings:
        raise ValueError("Prompt plus generation exceeds model context")
    tracker = SignalTracker(config.get("smoothing", 0.3))
    quantizer = CacheQuantizer(mode, prompt_length, tracker, calibration,
                              group_size=config["group_size"], sink_tokens=config["sink_tokens"],
                              recent_tokens=config["recent_tokens"], target_fraction=config["target_fraction"], seed=seed)
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = model(input_ids=ids, use_cache=True, return_dict=True)
    past = quantizer.apply(output.past_key_values)
    logits = output.logits[:, -1, :]
    del output
    generated = []
    stopped_eos = False
    for _ in range(config["max_new_tokens"]):
        token = int(torch.argmax(logits, dim=-1).item())
        if token in eos:
            stopped_eos = True
            break
        position = prompt_length + len(generated)
        tracker.observe(position, uncertainty(logits))
        generated.append(token)
        output = model(input_ids=torch.tensor([[token]], device=device),
                       past_key_values=past, use_cache=True, return_dict=True)
        past = quantizer.apply(output.past_key_values)
        logits = output.logits[:, -1, :]
        del output
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    shape = past[0][0].shape
    actual_bytes = sum(k.numel()*k.element_size() + v.numel()*v.element_size() for k, v in past)
    memory = cache_storage(len(past), shape[1], shape[-1], shape[-2], quantizer.records)
    floating_dtype = str(past[0][0].dtype)
    result = {"text": tokenizer.decode(generated, skip_special_tokens=True),
              "prompt_tokens": prompt_length, "generated_tokens": len(generated),
              "token_ids": generated, "stopped_eos": stopped_eos,
              "truncated": not stopped_eos, "seconds": elapsed,
              "tokens_per_second": len(generated)/elapsed if elapsed else 0,
              "actual_cache_tensor_bytes": actual_bytes, "actual_cache_dtype": floating_dtype,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "memory": memory, "blocks": quantizer.records,
              "signals": [{"position": p, **v} for p, v in tracker.by_position.items()],
              "budget_credit_channels": quantizer.controller.credit}
    del past, logits
    return result
