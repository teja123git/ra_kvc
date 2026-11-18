# Explaining the project

## A 30-second introduction

We study how to allocate precision in an LLM's KV cache. Most eligible keys
and values use simulated 2-bit quantization, while selected key channels use
4 bits. A fixed policy protects the same fraction in every block. Our adaptive
policy changes that fraction using output entropy or the top-two logit margin,
subject to a causal credit budget. We implemented the method, measured a small
CPU pilot, and modeled a packed storage format. The evidence does not establish
an adaptive accuracy gain or actual GPU-memory savings.

## What exactly changes?

Only eligible KV-cache values change. Model weights stay unchanged. There is
no fine-tuning. The shared quantizer rounds cache values and dequantizes them
back into floating tensors. Future model steps attend to those changed values.

## How does this relate to Kitty?

Kitty motivates mixed precision for key channels. Our simplified baseline ranks
channels by mean absolute key magnitude and always protects a fixed fraction.
The adaptive method keeps that ranker and varies the number of protected
channels across generated blocks. It is a Kitty-inspired simulation, not a
reproduction of Kitty's optimized implementation or published benchmark scores.

## Why use entropy or margin?

High entropy means the next-token distribution is spread out. A small top-two
logit gap means the leading choices are close. These are uncertainty proxies.
They do not prove that a token represents difficult reasoning or that its cache
will matter later. That relationship is a hypothesis tested by the diagnostic.

## How does a budget work with 64 channels?

At a target fraction of 12.5%, each finalized generated block earns 8 credits.
One credit protects one channel at 4 bits instead of 2 bits. A block can save
credits for a later block, but cannot borrow from the future. For example,
allocations of 0, 16, 8 and 8 channels spend 32 credits across four blocks,
matching a fixed allocation of 8 each. This example is illustrative.

Real generations may end with unused credit. The implemented rule provides
an upper bound and a bounded shortfall, not exact equality at every endpoint.
The results therefore report realized fractions and storage.

## Can a compressed old block be promoted later?

Not in this implementation. A block is finalized only once after it leaves the
recent buffer. Its precision decision is permanent. Increasing a datatype later
does not restore information already discarded by 2-bit rounding.

## Which cache regions remain floating?

The first 8 tokens, at least the most recent 32 tokens, and an incomplete final
group remain floating. Other groups contain 32 tokens. Prompt and boundary
groups use the same fixed rule in every mixed-precision policy. Only wholly
generated blocks use adaptive allocation.

## Why is average KV payload precision 2.125 bits?

For protected key fraction r, K needs 2+2r bits per scalar. V remains at 2.
K and V have equal element counts, so the combined average is 2+r. At
r=0.125 this is 2.125. It excludes metadata and floating regions.

## How much does metadata matter?

For one layer and one KV head, a 32-token group with 64 channels and 8 protected
key channels has the following analytical cost:

| Item | Bytes |
|---|---:|
| Key payload | 576 |
| Value payload | 512 |
| Two FP32 parameters for each key channel | 512 |
| Two FP32 parameters for each value token | 256 |
| Key-channel bitmap | 8 |
| Block descriptor | 8 |
| **Total** | **1872** |
| The same K/V group at 16 bits | 8192 |

The group contains 4096 K/V scalars, so the complete quantized representation
averages 3.65625 bits per scalar before floating buffers. Its projected
compression against a 16-bit cache is about 4.38x, rather than the 7.53x
payload-only ratio. The complete request includes additional floating regions.
These are format calculations, not measured allocated bytes.

## Why does the CPU run still use FP32 cache memory?

Fake quantization simulates lost numerical precision using floating tensors.
It tests accuracy effects. Real compression would require packing integer codes
and integrating dequantization with attention. Those kernels are outside the
implemented scope. The saved logs separately contain actual tensor bytes and
projected packed bytes against a stated 16-bit reference.

## Why can quantization sometimes increase the score?

A small numerical change can redirect a greedy generation, including changing
its length. It may turn an incorrect answer into a correct one by chance, or
allow a final answer to appear before the cap. A small pilot cannot establish
that the quantized model is generally better. The longer-cap reference check
examines one of these confounders.

## What does the length check prove?

It measures the same unquantized reference on the same test IDs with a larger
generation cap. It was added after observing truncation, so it is a diagnostic.
It does not test adaptive policies at that larger cap and does not justify
cross-cap method comparisons.

## What does the sensitivity diagnostic measure?

It replays fixed calibration trajectories, changes one cache block, and
teacher-forces the same following token. It measures how a 25% key-channel
boost changes the next-logit KL distortion relative to uniform 2-bit precision.
Eight blocks from four questions are only a local, correlated sample. The
diagnostic does not measure final-answer benefit or establish a reasoning detector.

## What does HumanEval contribute here?

It verifies the generation and functional-test pipeline on a small secondary
coding sample. Every evaluated policy passed zero of the eight tasks. The saved
programs contain wrong behavior, wrong function names, recursion errors or
incomplete answers. This floor effect provides no evidence of relative
quantization quality or equivalence. Coding is not a non-reasoning control.

## Why include a random policy?

It changes protection amounts under the same credit rule without using output
uncertainty. This checks whether any benefit might come from arbitrary timing.
The pilot uses one seeded schedule per question, not a multi-seed study.

## Which statements are supported?

The cache modification affects future model computation. The controller and
memory accounting have checks. The saved predictions support the reported
subset scores. A fully specified format supports the analytical storage
projection. Broad adaptive superiority, true INT2 allocation and production
speedup are not demonstrated.

## What should the next experiment be?

Evaluate every policy on a larger paired cohort with enough tokens to finish
answers. Match realized storage, repeat random schedules and sweep allocation
budgets. After the algorithm is useful, implement real packing and measure
GPU throughput at equal memory capacity. A speculative draft-model signal is
future work, not part of the delivered implementation.
