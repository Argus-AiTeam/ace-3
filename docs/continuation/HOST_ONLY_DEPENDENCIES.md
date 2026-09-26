# ACE-3 host-only dependencies

This repository is a curated source snapshot, not a full host image.

| Dependency | Location / provenance (updated September 26) | Published here | Recovery consequence |
| --- | --- | --- | --- |
| Live Argus state, backlog, handoffs, native events, registry and claims | `/home/argustest/.argus-skill-ace3/projects/s-62150b05` | No | Same-host continuation can inspect it. Disk loss removes lifecycle and authoritative receipts not separately archived. |
| Active runtime source | `build/argus-planner-recovery-20260925-attempt001/candidate`; release `0.1.2+f0d9ffb2b15a2dbb` at observation | Only the reviewed Planner format-retry diff, not the whole runtime | Discover current runtime from live status. The old attempt007 runtime is not the active launch instruction. |
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
