# ACE-3: bounded RTL token-feedback example

**ACTUAL LOCAL RTL EXECUTION / FEEDBACK EXAMPLE**

Date: 2026-09-06. This completed execution example records an **RTL-selected
second generated token**. It is not full independent numerical certification
or an accepted continuous-generation seed. At the publication handoff,
independent whole-trajectory certification was **pending / not established**,
and formal mission review was pending.

The [compact summary](ACE3_RTL_TOKEN_FEEDBACK_20260906.json) contains only the
bounded result, component coverage, reuse facts, selected artifact identities,
and these necessary scope labels.

## Exact token boundary

```text
input history:       [9707, 1879, 0]
input text:          Hello world!
consumed token:      0 -> !, at zero-based position 2
RTL-selected token:  358 -> " I"
selected logit:      FP16 bits 0x4c1b = 16.421875
output history:      [9707, 1879, 0, 358]
```

The **entire bounded RTL transcript** is:

```text
Hello world! I
```

The consumed `!` is the first generated token, following the two-token plain-text
seed `Hello world`. Token `358` is the **second** generated token and remains
pending for possible input at zero-based position **3**. This example does not
process that next position or select a third generated token. It is neither a
chat answer nor an EOS-complete response.

## Completed local component coverage

| Boundary | Recorded execution result |
| --- | --- |
| Target decoder transactions, layers0-23 | All 24 reached natural terminals; local exact-oracle trace and final-output comparisons agree |
| Final RMSNorm tail | Natural terminal; 1,792 serialized output values and RMS roots agree with its own-input exact oracle |
| Full tied head | Natural terminal; all 151,936 vocabulary logits and accumulators agree with its own-input exact oracle |
| Ordered Top-10 | Token IDs agree with the independent software comparison |
| Host decode | The official tokenizer maps selected token `358` to the leading-space piece ` I` |

**Local exact-oracle agreement is conditioned on the supplied RTL inputs.**
It does not establish independently maintained whole-trajectory numerical
agreement. The independent software Top-10 observation concerns token-ID
ordering, not bit-identical logits or whole-model certification.

The model remains `Qwen/Qwen2.5-0.5B-Instruct-AWQ`, revision
`db09cd27ead7fee40cdee309693cf83601b9c899`. Decoder projections use native
asymmetric packed INT4 AWQ/G128 with FP16 activations and K/V. The tied
embedding/head weights are **FP16, not INT4**; the head covers token IDs
`0` through `151935`.

## Reuse without another decoder replay

The completed decoder states were retained and reused unchanged.
**Decoder transactions reexecuted during this tail recovery: zero.**
Final RMSNorm and the full tied-head tail completed in a fresh execution;
this was not another 24-layer or prefill replay. The original source artifacts
were not modified.

No timing or performance claim is attached to this recovery example.

## Selected artifact identities

| Artifact | SHA256 |
| --- | --- |
| Retained execution evidence | `d322ef47a3918e752bf79230866a42401a5ae5aaad15f09d9c21733b119ad58e` |
| Sealed manifest | `9f7bb8f1112e5243cdd72ec28d5ce2005dccb6271e7fbe18fff43e88c37c45b4` |
| Actual final-RMSNorm raw output | `37f6518d82684de5bb954b8b861c7f53706dbaf046b22f745a975ebcd1dd28b1` |
| Full actual RTL head-logit stream | `f6c21a374dcf95d031d22c91e22c5928726d7381663ee790fa366b42020b88bc` |

Only identities and whitelisted aggregate facts are published, not the internal
evidence document, model data, raw streams, execution archives, or source code.

## Scope relative to other examples

The earlier [32-token software preview](ACE3_SOFTWARE_CONTINUATION_PREVIEW_20260906.md)
remains **software-only**. Its longer continuation must not be attributed to
this RTL execution: the actual bounded RTL text here is only `Hello world! I`.

This result does not certify a continuous-generation seed, deployment of a
continuous RTL driver, general dialogue quality, or whole-model numerical
correctness. It is computer-local RTL simulation, not synthesis, PPA, FPGA,
silicon, or a hardware benchmark. Public runtime/code boundaries are unchanged;
source and self-contained reproduction releases remain separate.
