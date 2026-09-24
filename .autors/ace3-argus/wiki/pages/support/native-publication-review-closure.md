---
title: Native publication review closure
description: Standard reviewed-handoff bindings, immutable proposals and the resident source boundary.
---

# Native publication review closure

For the retained idle-health candidate's schema-v3 review API, Engineer proposals
are not approval. Publication consumption requires a current indexed
`round_reviewed_handoff`, `producer_role=reviewer`, standard `review.status=done`,
and exact `review.artifact_bindings`. `latest.json`, kind `handoff_ref`, selects
the current handoff; an earlier round is superseded, not a fallback approval.

`life.context_packet.record_reviewed_handoff` accepts keyword arguments
`mission_context_path`, `round_index`, `engineer_summary`, `review`, and
`checkpoint_path`. Only a done decision persists typed bindings. The standard
Reviewer parser is `reviewer._parsing.parse_decision_text`, or
`decision_from_payload` for the actual structured verdict. The binding parser
allows at most 40 unique `{ref, sha256}` objects. A proposal format must not
introduce a new Reviewer-authored top-level decision schema.

The pure `daemon.reviewed_snapshot._reviewed_artifact` verifies bound JSON
objects. Raw source/log artifacts need the same standard typed binding
membership and exact raw-byte hash verification; the JSON-object helper cannot
parse Python source. `validate_reviewed_snapshot` authenticates the complete
native source manifest and current review. Callers must separately require
the exact manifest binding; the snapshot helper alone does not enforce it.

The retained kind-bearing tree manifest and the native path/size/hash manifest
have different digest representations. Applicability across these formats
requires equality of every normalized file row, not equality of digest strings
or a relabeled candidate.

On-disk writer support does not establish capabilities already loaded in a
resident process. A genuine standard raw Reviewer verdict can be finalized by
Manager/Host through the exact candidate writer in a fresh handoff mission.
An incumbent-produced handoff missing bindings is not a publication seal.
The handoff directory is a Host-controlled filesystem trust boundary, not a
cryptographic independent-review identity service. Disposable interface-test
seals are never production approval.
