# ACE-3: selected engineering highlights

**Completed-result cutoff: 2026-09-06 10:40:51 UTC.**
This is a selection of reviewed, bounded engineering milestones from local
experiments, not a comprehensive quality evaluation or full-model certification.
The [numeric summary](ACE3_ENGINEERING_HIGHLIGHTS_20260906.json) records the
coverage, precision policies, and selected output identities.

## Completed component milestones

| Boundary | Demonstrated coverage |
| --- | --- |
| Indexed decoder checks, layers16-23 | Each retained two-position fixture produced 46,676 trace rows and 1,792 final rows, bit-exact against its freshly materialized integer oracle |
| FP16-interstage component comparisons, layers21-23 | Maximum absolute errors of 0.015625, 0.015625, and 0.0234375, respectively, across the recorded stage/trace comparisons |
| Final RMSNorm | 1,792 outputs and both RMS roots match the integer oracle; maximum absolute difference from its FP16-boundary reference is 0.03125 |
| Streaming tied head / Top-K | All 151,936 vocabulary logits compared, using 136,134,656 streamed FP16 weights; ordered Top-10 and selected token agree with the component oracle |
| Tokenizer / host integration | One selected token is authenticated, appended to the fixed input history, and decoded as the complete one-token continuation below |

The decoder geometry is hidden width 896, with the official
`Qwen/Qwen2.5-0.5B-Instruct-AWQ` checkpoint pinned to revision
`db09cd27ead7fee40cdee309693cf83601b9c899`.
These are separately scoped component runs connected by authenticated
predecessor handoffs, not one newly demonstrated end-to-end process.
Layer22 acceptance reused its completed retained diagnostic execution; the
acceptance step is not counted as an additional RTL run.

## Precision and comparison policy

Decoder projections use native asymmetric packed INT4 AWQ with group size 128,
FP16 scales, and FP16 activations/KV. The layer21-23 floating component reference
uses dequantized AWQ with **round-to-nearest-even FP16 at implemented interstage
boundaries**. Its acceptance rule is:

```text
abs_error = abs(produced - reference)
relative_error = abs_error / max(abs(reference), 2^-14)
accept = abs_error <= 0.125
         OR (relative_error < 0.001 AND ULP_distance <= 1)
```

This is **not a global one-ULP bound**. The reported layer21-23 values satisfy
the absolute branch. The policy concerns the implemented FP16 boundaries,
not unrestricted float64 arithmetic or independently maintained whole-model
agreement.

For final RMSNorm, the component reference uses float32 mean-square/rsqrt
with epsilon `1e-6`, FP16 rounding of the normalized activation, and FP16
multiplication by official `model.norm.weight`. All 1,792 retained output
differences also satisfy `abs_error <= 0.125`.

The tied head consumes token position 1 of the accepted final-RMSNorm fixture.
Its official tied embedding/head weights are **FP16, not INT4**. FP16 operands
are decoded exactly to fixed point, products are accumulated without
intermediate rounding, and each logit receives one FP16 round-to-nearest-even
conversion. Top-K uses descending rounded logits, with ascending token ID
for ties. Agreement is with this component policy on the supplied hidden state.

## One complete, bounded host continuation

```text
plain-text fixture:    Hello world
input token history:  [9707, 1879]
selected token:       0  ->  !
selected logit bits:  0x4c1d
result token history: [9707, 1879, 0]
decoded fragment:     Hello world!
```

This shows the entire **one-token append** of the experiment, not an excerpt
presented as a complete dialogue answer. The selected token comes from the
reviewed tied-head RTL result; appending and decoding it are **host software**
operations with authenticated official tokenizer assets. The seed is plain
text, not a chat-formatted request.

## Scope

The decoder, final-norm and tied-head results above are actual **local RTL
simulation** results. The host step consumes the completed result; it does not
rerun RTL. No software-only continuation preview is included in this snapshot.

Component agreement and one readable fragment do not establish general model
quality, independently maintained full-model reference equivalence, accepted
persistent-KV second-token execution, or full dialogue. No latency, throughput,
PPA, synthesis, FPGA, or hardware result is claimed. Hardware work is outside
this milestone's active scope.

This publication contains highlights and a compact numeric summary only.
It does not distribute model data, runtime source changes, execution archives,
or a self-contained reproduction environment.
