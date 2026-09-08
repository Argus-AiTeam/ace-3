# ACE-3: bounded uniform residual-association RTL agreement

**INDEPENDENTLY REVIEWED CANDIDATE / BOUNDED RTL SIMULATION**

Date: 2026-09-08. An isolated uniform residual-association candidate completed
actual RTL execution and **538 independent FP16-interstage comparisons** under
the unchanged numerical gate, with zero material failures. Separate
**Icarus/vvp operator simulation passed 28,682 cases**, including directed
cases checked against an exact Fraction oracle.

This is distinct from the earlier
[Q-only RoPE milestone](ACE3_Q_ONLY_ROPE_BOUNDED_RTL_20260907.md).
The [machine-readable summary](ACE3_UNIFORM_RESIDUAL_BOUNDED_RTL_20260908.json)
records this new scope. This release publishes documentation only, not
candidate source, production adoption, or an executable reproduction.

## Uniform mechanism and experimental lineage

The changed layer-final residual association is:

```text
RNE16(RNE16(attention_O + MLP_down) + incoming_hidden)
```

`RNE16` means FP16 round-to-nearest-even. The association is uniform, not a
coordinate patch; it uses no reference injection or extra FP16 companion.
Stage 12 and the MLP path remain unchanged. The experimental base retains
Q-only fused-Q48 RoPE, the general exponential candidate, native asymmetric
G128 AWQ, and FP16 activations and K/V. This is not a claim that the original
production baseline has been fully repaired.

## Exact executed scope

All layer, position, stage, and element indices below are zero-based.

| Actual candidate comparison scope | Passing comparisons |
| --- | ---: |
| Layer 0, positions 0–3: changed stage-18 residual outputs | 4 |
| Layers 1–7, positions 0–3: 28 actual Verilator transactions, 19 stages each | 532 |
| Layer 8 endpoint: RMSNorm and Q projection only, stages 0–1 | 2 |
| **Total** | **538** |

Layer-0 positions 0–2 reused authenticated, unchanged stages 0–17. The
unchanged base binary executed a new layer-0 position-3 prefix from its own
position-2 state; its old-association stage-18 output was **not** used as
candidate hidden. Icarus computed all four changed layer-0 stage-18 outputs
from actual hidden, attention-O, and MLP-down operands. Those actual outputs
fed layer 1.

The 28 later transactions used their own actual candidate hidden and K/V.
Retained records bind **21 same-binary preceding-position state links** and
**24 interlayer hidden links**. Position 0 starts empty in each changed
binary. All 28 transactions also passed separate candidate-local exact
stage checks; that does not mean bit-exact agreement with the independently
propagated FP16-interstage reference. Software outputs were not relabeled RTL.

The layer-8 endpoint consumes actual layer-7 position-3 hidden. Its logical
position is 3, but it physically runs at **position 0 in a fresh engine**,
only through position-independent RMSNorm and Q projection. It establishes
**no full layer-8 position-3 transaction, layer-8 K/V, RoPE, or stages 2–18**.
The original layer-8 position-3 stage-8 indices 13 and 15 were not executed.
Layer-8 positions 0–2 are not imported into this new comparison total.

## Unchanged numerical gate and bounded observations

Both values must be finite, and each scalar must satisfy:

```text
abs_error <= 0.125
OR (relative_error < 0.001 AND ordered_FP16_ULP <= 1)

relative_error = abs_error / max(abs(reference), 2^-14)
```

- **Layer 7 / position 0 / stage 18 / index 62:** actual `1580`, reference
  `1581`, absolute error `1`, ULP distance `1`; accepted by the unchanged
  relative/ULP alternative. The gate is not an absolute-error-only rule.
- **Logical layer 8 / position 3 / Q index 223:** actual `-13.7421875`,
  reference `-13.7109375`, absolute error `0.03125`. Across all **896 Q
  outputs**, maximum absolute error was **0.03125**, with zero material
  failures, at the restricted endpoint described above.

The run completed naturally with exit code 0 at **2026-09-08 09:33:21 UTC**.
Recorded experimental elapsed time was **8,523.217445007991 seconds**
(approximately 2 hours 22 minutes). This is not a performance-bottleneck
finding, hardware latency, or tokens-per-second measurement.

## Independent review and immutable identities

A later, non-skipped independent reviewer accepted the completed bounded
scope with status **`done` at 2026-09-08 09:39:58 UTC**, including the separate
software eligibility boundary below. The frozen result and checkpoint
contain earlier pending-review annotations; those predate the actual review
and were not rewritten to manufacture acceptance.

Before publication, retained manifest hashes, all 538 gate records, and
candidate hidden/state links were authenticated read-only. All **1,859**
artifacts in the RTL output manifest matched their recorded hashes; no
simulator, model, numerical gate, or scalar screen was replayed.

| Artifact | SHA256 |
| --- | --- |
| Bounded terminal result | `6ecd10dfee4b1ae4d65de2eb0541545347ffcc831c6ce580553b149ecca7bbed` |
| Frozen manifest | `2edead446d7437758a65c6805d7fe736defc02bc3c596ee52761f12ed5b76ad8` |
| RTL output manifest | `19b431cdec71e0a4bb9b84e1577b5ecdf1706146ae13e51cba02d91483159c2c` |
| Operator result | `b0fcbf94f6530e83c46b6a5a361607d440c469e8aba2270493ac2fadd1fb0650` |
| Later independent review | `b13013f5364a4396025fc5077e561234b79832025b8c079a27c61b6c067d0ae3` |
| Separate software eligibility result | `28a4838a6a3f5ebed708ce400a07de988ebb492558fb17667be88389377f87fa` |

These are artifact fingerprints, not a publication of weights, numerical
arrays, raw logs, simulator state, candidate source, or review transcripts.

## Eligibility and claim boundary

**Separate binary64-v1 eligibility remains unsatisfied.** The independently
reviewed software-candidate screen evaluated 24 retained layer-final outputs
at positions 0–2 (21,504 coordinates) and did not meet that separate profile.
It is a **software** screen, not an RTL-output admission test. Position 3
remains unbound in its authenticated binary64 reference set, not a pass.
The positive FP16-interstage RTL result does not waive this known limitation
or establish binary64-v1 admission, whole-trajectory repair, or promotion.

Fixed input token IDs **`[9707, 1879, 0, 358]`** are not candidate-native
greedy choices. No new full-model, final norm/head/tail, tokenizer, or token
generation result is established. The earlier
[two-generated-token text](ACE3_RTL_TOKEN_FEEDBACK_20260906.md),
**`Hello world! I`**, remains a different execution lineage, unchanged by
this candidate. This is not a full nine-layer position-3 pass or complete
attention-blocker repair, and makes no dialogue-quality claim.

No synthesis, PPA, FPGA/U280, XRT/HBM execution, Vivado/Vitis implementation,
bitstream, deployment, hardware timing, or throughput result is claimed.
