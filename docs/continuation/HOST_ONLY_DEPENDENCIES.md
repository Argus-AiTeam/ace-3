# ACE-3 host-only dependencies

This repository is a curated source snapshot, not a full host image.

The [V10 evidence adapter](progress-2026-10-12-v10-evidence-adapter.json) publishes
only reviewed Python source, fixture tests and sanitized provenance. Its original
independent reviews, retained traversal/model payloads, proposal archives and
execution receipts stay host-only. Format compatibility does not supply missing
scientific evidence or authorize a new execution.

The [October 11 increment](progress-2026-10-11.json) publishes one reviewed
SiLU RTL repair and original evidence hashes. Its authoritative review, raw
development logs and live execution archives remain host-only. The table below
retains the dated September inventory; discover current runtime state separately.

The [October 12 increment](progress-2026-10-12.json) adds three reviewed
Makefile/testbench source files. Original reviews, focused simulation logs,
proposal archives and active bindings remain host-only. Publication verifies
the copied source hashes without rerunning science or publishing raw logs.

The subsequent [file-gate increment](progress-2026-10-12-file-gates.json) updates
the same two testbenches with exact reviewed bytes. Only source and sanitized
provenance are published; original export reviews, proposals, runtime state and
execution receipts remain on the host. It does not certify the full regression.

The [checkpoint caller integration](progress-2026-10-12-bound-checkpoint-caller.json)
publishes the Makefile, validator and focused tests together. The explicit
checkpoint/tokenizer payloads remain external, licensed host dependencies.
Private proposal producers, source archives, native reviews and execution
authority are not included; a source-only clone does not authorize execution.

| Dependency | Location / provenance (updated September 28) | Published here | Recovery consequence |
| --- | --- | --- | --- |
| Live Argus state, backlog, handoffs, native events, registry and claims | `/home/argustest/.argus-skill-ace3/projects/s-62150b05` | No | Same-host continuation can inspect it. Disk loss removes lifecycle and authoritative receipts not separately archived. |
| Active runtime source | `build/argus-planner-structured-empty-hold-release-input-fcd4bb2c7c09-attempt001/snapshot`; release `0.1.2+e69764dbdb9bd222` at observation | Runtime identity/progress only; older published patches are historical, not this complete runtime | Discover current runtime from live status. Do not copy the public `argus_skill` snapshot over this sealed release or replay its deployment. |
| Latest exact-integer Stage11 software | `d12e372181f0` and its 64-source execution dependency set | Four exact historical implementation/test files, dependency sources and companion tests; hashes in September 28 manifest | Subsequent native-schema/source-census defects prevent execution as published. Repair owner is active; no new numerical result exists. |
| Latest measured S13-S17 composition | `build/stage16_silu_product_execution_v1-92e4223ef310-attempt001` | Small REJECTED summary and original result/review hashes | All nine S18 and eighteen direction checks failed. The unchanged composition is consumed and must not be replayed. |
| Current Stage11 handoff and source lineage | Native state and original retained Stage10/tensor/state/KV payloads | No authoritative handoff, claim, arrays or active authority | The first science owner stopped before claim. Public source does not supply missing Manager inputs or grant authority. |
| Reviewed scientific release | `build/argus-stage11-reviewed-interface-release-287d994f2fe4-attempt001` | Six exact source snapshots at normal source paths; full seal/authority/state remain host-only | Source publication does not grant execution authority or permit replay. |
| Actual rejected scientific evidence | `build/stage11-attention-output-suffix-authorized-b4978b898600-attempt001` and `build/stage11_s18_boundary_localizer_v2_attempt001` | Sanitized numerical rows, counts and original hashes | Both are consumed REJECTED results with normal review; raw arrays and original receipts remain on the host. |
| Accepted `04c` root | `build/argus-stage11-current-runtime-lineage-schema-04c9ee209234-attempt001` | No, hash-linked summary only | Operational acceptance cannot be recovered from the summary alone. |
| Failed `e2ba` root | `build/argus-stage11-emitter-candidate-lineage-interface-e2ba4dbc8651-attempt001` | No | Preserve host-local root unchanged. It must not be sealed, retried, or presented as accepted. |
| Absent `609` fresh root | Authorized path for attempt001 was absent at snapshot | No | Do not invent or reconstruct a PASS; native mission is terminal FAILED. |
| Full model weights/tokenizer/cache | Official Qwen2.5/AWQ or other approved host-local sources, governed by upstream licenses | No | Only small repository fixtures are included. Reacquire full payloads from approved provenance and verify license/hash. |
| Raw build roots, provider/worker logs, `.argus_subagents` | Live repository/state directories | No | Recompute only under new lawful authority; never replay closed one-shot science merely to recreate evidence. |
| GitHub credential | User-supplied `GH_CONFIG_DIR=/home/argustest/argustest2/.gh-config` | Never | Supply externally; expected GitHub login is `aHappend`. |
| Copilot profile/credentials | `/home/argustest/.copilot` | Never | Supply/restore locally; it is independent of GitHub identity. |
| Scheduled supervision | External/session-scoped registration | No | Re-establish explicitly from current intent. Old registration/text is not evidence of ongoing checks. |

The public progress JSON records original evidence hashes but is deliberately not
an unchanged copy of private authoritative receipts.
