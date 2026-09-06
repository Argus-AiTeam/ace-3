# ACE-3: reviewed layer22 continuation through one host token

**Evidence cutoff: 2026-09-06 10:40:51 UTC.** This separate, evidence-only
milestone extends the [layers16-21 report](ACE3_LAYERS16_21_20260906.md).
It records bounded component acceptance and a single tokenizer/host append,
**not independently maintained full-model reference equivalence, persistent-KV
generation, or dialogue completion**. No runtime source is released here.

The [machine-readable summary](ACE3_LAYER22_TO_HOST_20260906.json) contains exact
input/output/source/manifest hashes, predecessor edges, historical command
arguments, timings, and reviewer-event receipts. Paths use relative names or
logical aliases; hashes always describe original bytes.

## What was completed

| Component | Actual retained artifact directory | Independent reviewer `done` UTC | Bounded result |
| --- | --- | --- | --- |
| Layer22 | `build/diagnose_layer22_from_layer21` | 09:36:56 | Accepted the existing sealed diagnostic; no duplicate execution |
| Layer23 | `build/model24_layer23_attempt001` | 09:53:22 | Fresh RTL run; integer trace/final exact; no FP16-policy failures |
| Final RMSNorm | `build/model24_final_rmsnorm_attempt001` | 10:05:04 | 1,792 outputs and RMS roots exact to integer oracle; no policy failures |
| Tied head / Top-K | `build/model24_tied_lm_head_topk_attempt002` | 10:30:54 | All 151,936 logits compared; ordered Top-10 and selected token match |
| Tokenizer / host | `build/model24_tokenizer_host_integration_attempt001` | 10:40:42 | Append selected token `0` to `[9707,1879]`, yielding `Hello world!` |

Reviewer task IDs, respectively: `2200a1d4460b`, `584e7ea085a7`,
`2ba4205c8938`, `0f784a0c042c`, and `177b77ce57da`.
These are actual `round.review.completed` events with `review_source=reviewer`
and `review_skipped=false`, followed by `life.mission.completed` / `done`.
The final host mission completed at **10:40:51.127588 UTC**.
Task outcomes retain `stage_certification=deferred`: bounded reviewer completion
is not a whole-project or full-model certification.

The summary records event line numbers, timestamps, and SHA256 of each original
line including its newline, associated through the surrounding task lifecycle.
Raw private events, prompts, session logs, and machine paths are not published.

### Reuse, reviews, and historical status

Layer22 acceptance reused the already fresh cycle-zero diagnostic, sealed at
09:24:15. A 09:27:19 operator event superseded pending duplicate task
`0c6f4745fefb`; acceptance task `2200a1d4460b` then reviewed the retained
evidence read-only. **There was no new official layer22 rerun to count.**

Tied-head task round 1 received **`continue` at 10:22:08**: functional output
was correct, but the new RTL/contract lacked the required design-manifest and
traceability records. Attempt001 was retained; fresh attempt002 bound those
records and subsequently received `done`. This is not a retrospective
promotion of attempt001.

Final RMSNorm and tied-head seals still say
`ENGINEERING_PASS_REVIEW_REQUIRED`, and their status files still say
`PENDING_INDEPENDENT_REVIEW`. They were sealed **before** the later reviewer
events. Host lineage faithfully preserves those historical strings.
No seal, status, or review field was rewritten to manufacture acceptance.
Likewise, inherited `official_attempt: not run` text in layer metadata is
preserved rather than silently relabelled.

## Numerical and execution boundaries

The decoder checkpoint is `Qwen/Qwen2.5-0.5B-Instruct-AWQ`, revision
`db09cd27ead7fee40cdee309693cf83601b9c899`. Decoder projections use asymmetric
packed INT4/G128 with FP16 scales, activations and KV. Layer22 consumes accepted
layer21 attempt002's complete handoff; layer23 consumes that layer22 diagnostic.
Both runs reached their natural terminal and produced **46,676 trace rows plus
1,792 final rows**, each bit-exact to freshly materialized integer-oracle vectors.

Their independent component-policy reference is PyTorch CPU float64
dequantized AWQ with **round-to-nearest-even FP16 at implemented boundaries**.
The acceptance rule is:

```text
abs_error = abs(produced - reference)
relative_error = abs_error / max(abs(reference), 2^-14)
accept = abs_error <= 0.125
         OR (relative_error < 0.001 AND ULP_distance <= 1)
```

ULP distance compares ordered FP16 encodings against the FP16-rounded reference.
This is **not globally <=1 ULP**:

| Component comparison | Maximum absolute error | Maximum observed ULP distance | Failing coordinates |
| --- | ---: | ---: | ---: |
| Layer22, 38 stage/token groups | 0.015625 | 7,564 | 0 |
| Layer23, 38 stage/token groups | 0.0234375 | 10,985 | 0 |
| Final RMSNorm, 1,792 values | 0.03125 | 1 | 0 |

**RMSNorm comparator caveat:** its archived report declares failure only when
absolute, relative, **and** ULP thresholds are all exceeded, which is looser than
the decoder rule above. Its current producer also uses a `2^-24` relative-error
denominator floor. This difference is not concealed or rewritten. A read-only
comparison of all retained raw/reference values confirmed that every RMSNorm
error also passes the **stricter absolute branch** (`max_abs_error=0.03125`).
The reference uses NumPy float32 mean-square/rsqrt with epsilon `1e-6`, then
FP16 normalized activation and FP16 multiplication by official `model.norm.weight`.

The tied head consumes **token index 1** from accepted final RMSNorm, without
replaying decoder layers. Its weights are **FP16, not INT4**: distinct official
`model.embed_tokens.weight` and `lm_head.weight` tensors have identical
authenticated values. The run streams **136,134,656 weights**, covers all
**151,936 vocabulary logits**, and returns ordered Top-10. The independent
component oracle decodes FP16 exactly to Q16.24, accumulates exact Q32.48
products without intermediate rounding, and rounds each logit once to FP16 RNE.
Equal rounded logits are ordered by ascending token ID.

All accumulator/logit/Top-K comparisons report zero mismatches. Selected token
is **`0`**, logit bits **`0x4c1d`**. This verifies the implemented component policy
**on the accepted upstream hidden state**, not the entire model against a
separately maintained full-model implementation.

## One host append, not a dialogue claim

The retained host result is exactly:

```text
plain-text seed:       "Hello world"
input token history:   [9707, 1879]
selected token:       0 -> "!"
result token history: [9707, 1879, 0]
decoded fragment:     "Hello world!"
```

Official tokenizer and tokenizer-config bytes are hash-bound. A separate chat
template serialization probe is recorded, but **that probe was not the
generation prompt**. The demonstrated seed is plain text, not a chat-formatted
request. Host generation/validation command receipts both record exit zero;
this integration did not rerun RTL or execute persistent-KV feedback.
The host manifest binds `integration.json`; it is not a fabricated
`sealed_output_manifest.json` or a new RTL-run receipt.

## Timing and immutable provenance

| Scope | Recorded host wall seconds |
| --- | ---: |
| Layer22 original diagnostic, total | 516.387791958 |
| Layer22 simulation only | 463.491288694 |
| Layer23 attempt001, total | 384.428518227 |
| Layer23 simulation only | 346.110946546 |
| Final RMSNorm compilation | 2.432635361 |
| Final RMSNorm simulation | 0.004217461 |
| Tied-head attempt002 official simulation | 122.020757175982 |

Tied-head simulation reports 136,287,502 simulated cycles; RMSNorm reports
3,681. These are **simulation receipts, not hardware timing or PPA**.
The host integration receipts do not record elapsed wall time; none is inferred
from file modification times or reviewer duration. Do not sum these reused,
separate components into an end-to-end latency.

| Manifest | Original SHA256 |
| --- | --- |
| Layer22 diagnostic seal | `1832f299e4b1ea242dfaa83ff945970b83ef697c5249cb43cd2e531163a82c0c` |
| Layer23 attempt001 seal | `2c7adde9827e16ddd9f495df502788aa2d3ef8a09fc307723281bf43a49c3646` |
| Final RMSNorm attempt001 seal | `f5cfac5693b75bb2a570ff2ce90c8fcfa68b3e5c8c86bb71f73b8c8618f78a67` |
| Tied-head attempt002 seal | `c701c116343609d84c894b9c14fbeca3d37075d527c9a28bcaccb196a8557190` |
| Tokenizer/host manifest | `5c48df9ff106ebac5bed326694f77bda84c4c7996e69d74acb2a758827fdf201` |

The publication audit rehashed **260 entries** across the four component seals,
**330 predecessor-baseline entries** (not necessarily distinct files), and the
host manifest's integration artifact. All assigned output seals matched.
Official final-norm/tied-weight tensor slices and tokenizer assets were also
checked against recorded hashes, without publishing their contents.

Provenance limitations remain explicit:

- The historical layer22 driver digest is retained, but its working-tree path
  now contains the layer23-capable version. Do not treat that current file as
  the original layer22 execution source. The shared frozen RTL/model hashes
  still match; no live source was copied into this milestone.
- The host manifest does not bind the executed host-runtime source hash.
  The JSON labels its current source digest as an observation only, **not**
  an immutable execution-source receipt.
- Tied-head input metadata has a copied `prior_official_attempt.attempt_id`
  saying attempt002 although that entry's root and seal identify attempt001.
  The summary flags this rather than silently correcting the original.
- The prior report's **layer19 seal-receipt discrepancy** and **layer17
  preservation-baseline coverage limitation** remain in force. Layer21
  attempt001 remains FAIL under its original unrounded-interstage policy.

## Audit and release limits

For an archive holder, the JSON's `artifact_receipts` give relative filenames,
byte lengths and SHA256 values. Check them with `sha256sum` against the original
attempt directories; then check every original seal entry and the host
manifest's `integration.json` entry. Reviewer-event hashes require the retained
private event lines, including newlines. The prior report also contains a
standard-library read-only seal-checking example.

Recorded command arguments in this summary replace private roots with
`$ARCHIVE_REPOSITORY` and `$FROZEN_R20_SOURCE`. They document what ran; they are
**not instructions to rerun into immutable directories**. No weights, vectors,
builds, logits dump, private logs, or execution environment are published.
Without the retained archives, these documents are not a self-contained
reproduction package.

The full-24-layer persistent-KV **second-token task is excluded and not accepted
by this milestone**; its active attempt was not inspected. W4A16 persistent
generation and full dialogue remain incomplete at this cutoff. Future
precisions are separate later work. Hardware, synthesis, PPA, and FPGA work
remain cancelled and are not claimed.
