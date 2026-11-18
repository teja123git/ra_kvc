# Method and design decisions

## Research question

Does output uncertainty help allocate a limited amount of extra key-cache
precision across generated token blocks? The operational proxy is uncertainty,
not an observed label of reasoning difficulty.

## Quantizer

For a group of values x and b in {2, 4}, let l=min(x), u=max(x),
s=(u-l)/(2^b-1). Reconstruction is
`x_hat = l + s * clip(round((x-l)/s), 0, 2^b-1)`.
Constant groups are preserved exactly. K reduces over block tokens and V
reduces over channels. The quantizer calculates in FP32 and casts back to
the cache dtype. No full-precision historical master copy is retained for
future promotion. Only the recent buffer is eligible for first-time quantization.

Channels are ranked independently for each layer and KV head by their mean
absolute value in the new block. The highest-ranked n receive 4 bits. All
policies use the same ranker, grouping and buffer sizes.

## Signals and calibration

Normalized entropy is `-sum(p*log(p))/log(vocabulary_size)`. Margin is the
largest minus second-largest raw logit. A moving average uses alpha=0.3.
Statistics come from smoothed signals on GSM8K training examples. A generated
block uses its average smoothed signal. The margin direction is reversed
when converting to an uncertainty score. The controller clips the standardized
score to [-1,1]. The Python implementation measures this overhead as part of
runtime; it does not claim it is free.

## Controller

For D channels and target r, each newly finalized generated block earns rD
credits. Requested protection is round(rD*(1+z)), clipped to [0, 2rD].
Protection cannot exceed available credits. The controller also keeps the
remaining balance no larger than approximately one nominal block budget.
Therefore cumulative allocation is at most the target and its shortfall is
bounded, apart from integer rounding. Early EOS can leave unused credit.
The completed run reports this shortfall; exact per-request equality is not
assumed. Prompt and boundary blocks use fixed allocation and are accounted
for separately.

## Cache timing

The uncertainty that predicts token t is recorded at t's absolute position.
After consuming a token, cache blocks become eligible only when they are
outside the shared recent buffer. A newly completed block is quantized once
after that forward pass and before the next forward pass. Already finalized
blocks are never requantized. Precision decisions thus affect future attention
and do not use future tokens. Changing a 2-bit representation to a larger dtype
would not restore information, so retroactive promotion is deliberately absent.

## Packed storage projection

The storage model describes a hypothetical packed format: integer K/V payloads,
two FP32 parameters per K channel and V token, one key-channel bitmap for each
mixed-precision block, an eight-byte per-head block descriptor, and 16-bit
sink/recent/incomplete buffers. Payload sizes round up to bytes. This is a
specified projection, not a packed allocator implemented by this code.

Ignoring overhead and buffers, K uses 2+2r bits per scalar and V uses 2. With
equal K/V sizes, their average is 2+r. Metadata is significant for short
groups, so the full projected compression is smaller than the payload-only
ratio 16/(2+r). All dimensions count KV heads rather than query heads.

## Experimental controls

The run uses deterministic greedy decoding and identical prompts/sample IDs.
Calibration and evaluation use separate dataset splits. Values stay at the
same precision across fixed/adaptive methods. Small prompt differences,
generation limits, extraction failures, unequal lengths and residual-credit
differences are all reported. Random allocation is one seeded control, not
an average across many seeds in the rapid pilot. The diagnostic measures local
next-logit sensitivity, not an oracle of final answer accuracy.

## Scope refinements from the earlier proposal

The supplied run uses a smaller model for CPU feasibility. GPU configurations
retain the original single-model approach. HumanEval is treated as coding,
not a non-reasoning benchmark. SEQUOIA's draft/target signal remains future
work. The baseline is Kitty-inspired and does not claim reproduction of its
optimized memory layout or throughput. Individual ownership remains subject
to actual member verification.
