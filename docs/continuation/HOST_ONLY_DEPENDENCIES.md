# ACE-3 host-only dependencies

This repository is a curated source snapshot, not a full host image.

| Dependency | 2026-09-24 location / provenance | Published here | Recovery consequence |
| --- | --- | --- | --- |
| Live Argus state, backlog, handoffs, native events, registry and claims | `/home/argustest/.argus-skill-ace3/projects/s-62150b05` | No | Same-host continuation can inspect it. Disk loss removes lifecycle and authoritative receipts not separately archived. |
| Active runtime source | `build/argus-stage11-terminal-replan-ledger-repair-attempt007/candidate`; release/digest `5e0e6fe7...11aefe1` | No | Reconstruct/replace Argus runtime separately and revalidate import identity. Never use the ambient Manager checkout as the runtime. |
| Accepted `04c` root | `build/argus-stage11-current-runtime-lineage-schema-04c9ee209234-attempt001` | No, hash-linked summary only | Operational acceptance cannot be recovered from the summary alone. |
| Failed `e2ba` root | `build/argus-stage11-emitter-candidate-lineage-interface-e2ba4dbc8651-attempt001` | No | Preserve host-local root unchanged. It must not be sealed, retried, or presented as accepted. |
| Absent `609` fresh root | Authorized path for attempt001 was absent at snapshot | No | Do not invent or reconstruct a PASS; native mission is terminal FAILED. |
| Full model weights/tokenizer/cache | Official Qwen2.5/AWQ or other approved host-local sources, governed by upstream licenses | No | Only small repository fixtures are included. Reacquire full payloads from approved provenance and verify license/hash. |
| Raw build roots, provider/worker logs, `.argus_subagents` | Live repository/state directories | No | Recompute only under new lawful authority; never replay closed one-shot science merely to recreate evidence. |
| GitHub credential | User-supplied `GH_CONFIG_DIR=/home/argustest/argustest2/.gh-config` | Never | Supply externally; expected GitHub login is `aHappend`. |
| Copilot profile/credentials | `/home/argustest/.copilot` | Never | Supply/restore locally; it is independent of GitHub identity. |
| Scheduled supervision, including retired schedule126 | External/session-scoped registration | No | Re-establish explicitly from current intent. Old registration/text is not evidence of ongoing checks. |

The public progress JSON records original evidence hashes but is deliberately not
an unchanged copy of private authoritative receipts.
