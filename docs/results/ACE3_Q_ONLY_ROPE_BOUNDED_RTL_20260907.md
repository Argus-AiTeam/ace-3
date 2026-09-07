# ACE-3: bounded Q-only RoPE RTL agreement

**INDEPENDENTLY REVIEWED CANDIDATE / BOUNDED RTL SIMULATION**

Date: 2026-09-07. An actual **Verilator RTL run completed 27 transactions**:
zero-based layers **0–8**, positions **0–2**, with **19 independent
FP16-interstage comparisons per transaction**. All **513 stage comparisons**
met the unchanged numerical gate, with zero material failures. Separate
**Icarus/vvp RTL simulation passed 5,640 operator cases**, covering Q arithmetic
boundaries and preservation of baseline K rotation.

The [machine-readable summary](ACE3_Q_ONLY_ROPE_BOUNDED_RTL_20260907.json)
records the bounded result and selected immutable identities. This is a
documentation release, not source promotion or production adoption.

## Arithmetic and experimental lineage

The candidate applies a general **Q-only exact Q48 rotary multiply-add,
followed by one FP16 round-to-nearest-even conversion**. The exact products
are sign-extended before summation; signed nonzero underflow is preserved,
and exact cancellation yields positive zero.

This experiment starts from the **earlier experimental general
softmax-exponential candidate configuration**, not the original production
baseline. Relative to that configuration, the change is Q rotation only:
K rotation retains its two-round path, while V, SiLU, the public interface,
and handshake remain unchanged. This result therefore does not establish
that a Q-only edit repairs every original production-baseline behavior.
It is also distinct from the earlier
[Q-projection dot-plus-bias component milestone](ACE3_Q_PROJECTION_RTL_20260906.md).

## Actual execution and independent reference

The frozen model checkpoint matches the official
[`Qwen/Qwen2.5-0.5B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ)
revision `db09cd27ead7fee40cdee309693cf83601b9c899`: native packed asymmetric
INT4 weights, group size 128, with FP16 activations, scales, and K/V.
The tied embedding/head is FP16, **not INT4**. The model has 24 layers and
hidden size 896; this experiment covers only its first nine layers.

Candidate hidden states and K/V were regenerated in RTL from fixed token
roots **`[9707, 1879, 0]`**. Each next layer consumed the preceding layer's
actual output. Each layer started position 0 with empty state, and positions
1–2 consumed that same rebuilt binary's preceding candidate state.
No old binary/state was imported into the changed simulation.

The comparison reference remained **independently propagated FP16-interstage
history**. Reference Q, hidden states, and K/V were not substituted for
candidate execution inputs. Separate candidate-local exact trace/final-output
checks do not imply bit-exact agreement with that independent reference.

### Unchanged numerical gate

Both values must be finite, and each scalar must satisfy:

```text
abs_error <= 0.125
OR (relative_error < 0.001 AND ordered_FP16_ULP <= 1)

relative_error = abs_error / max(abs(reference), 2^-14)
```

At **layer 8 / position 2 / stage 8**, all **42/42 scores were finite**,
with zero material failures and maximum absolute error **0.09375**.
Exactly **18/42** scores were bit-exact; **full bit-exactness is not claimed**.
Indices **9, 10, and 11** meet this same unchanged gate.

The recorded experimental elapsed time was approximately **7,550 seconds**
(7,549.627699077013 seconds in the bounded result). It includes setup,
compilation, reference work, and comparison. It is **not** per-token latency,
hardware speed, or a throughput benchmark.

## Later independent review and immutable identities

An independent reviewer accepted this **completed bounded RTL candidate**
with status **`done` at 2026-09-07 17:41:06 UTC**. This later review supersedes
the earlier pending-review annotation in the immutable execution result;
the original result was not rewritten.

| Artifact | SHA256 |
| --- | --- |
| Bounded terminal result | `d7c1b382baaa232d04a2ccacf49309d0645d390e8cb9025d14e443bd62d11c19` |
| Frozen manifest | `03ab838054fe8318d369e8f260801c1e0193ad1c43720133afb96fbb5b90f785` |
| Operator result | `7795baf65472dbd0f1d0753253af3d429017fb1c10d48ee7922dfbe28f4eac9c` |
| Layer 8 / position 2 result | `b3147bf59e4835fe83aea607af6229597e4746de1e818e6033eb13c8f887fc5e` |
| Later independent review | `62283b01b32f9152c7c00bf99d0e285f6aa6f0374dfab49697aae9b79437299c` |

Only selected summary facts and artifact identities are published. The
candidate source, weights, raw traces, simulator state, and review documents
are not included. The existing public source tree alone is not presented as
an executable reproduction of this frozen candidate.

## Claim boundary

- **Established:** reviewed actual RTL candidate agreement for layers 0–8,
  positions 0–2, under the stated FP16-interstage gate.
- **Not established:** repaired full-24-layer generation, candidate-native
  greedy-history approval, position 3 execution, a third generated token,
  useful dialogue, driver-seed certification, or production adoption.
- **Not established:** binary64 or final-layer fidelity/acceptance. The
  separately approved quantization-aware binary64 profile remains separate;
  its outstanding retrospective acceptance requirements are **not waived**
  by this FP16-interstage result.
- **Not established:** synthesis, PPA, FPGA or other hardware execution,
  timing closure, or measured throughput.

No new token selection is claimed here: token `0` is a fixed input root,
not a newly generated result of this nine-layer run. The historical
[two-generated-token RTL text](ACE3_RTL_TOKEN_FEEDBACK_20260906.md),
**`Hello world! I`**, belongs to a **different earlier complete execution
lineage**, not this candidate. Joining this candidate's nine-layer output
to that earlier layer-23 result, tail, or K/V would not establish whole-model
success and is not part of this milestone.
