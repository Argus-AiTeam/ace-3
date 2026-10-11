# ACE-3 continuation handoff

**October 12 source follow-up:** the independently reviewed
[Icarus file-open gates](progress-2026-10-12-file-gates.json) are copied exactly
into the FP16 adaptation and QKV testbenches, retaining the prior path, PASS
sentinel, Makefile and SiLU fixes. Missing vector inputs or unwritable waveform
paths now fail explicitly. This is development/not-certified source publication,
not V1 acceptance, execution authority or an overwrite of a running workspace.
The new checkpoint is `checkpoint-2026-10-12-icarus-file-gates`; old tags stay fixed.

Snapshot date: **2026-09-28**. Exact observation time and live work are in
[progress-2026-09-28.json](progress-2026-09-28.json). Live state can advance.
The operational guide is [HANDOFF.zh-CN.md](HANDOFF.zh-CN.md).

The equal mirrors are <https://github.com/aHappend/ace-3> and
<https://github.com/Argus-AiTeam/ace-3>, branch `main`, checkpoint ref
`refs/tags/checkpoint-2026-09-28-source`. Publish as `aHappend` to both and verify
identical commit/tree IDs; see [DUAL_MIRROR_PUBLICATION.md](DUAL_MIRROR_PUBLICATION.md).

**Latest measured comparator:** `92e4223ef310` executed the nine-control
S13-S17 composition once and received normal independent review/native DONE.
Its scientific result is **REJECTED**: all nine S18 checks and all eighteen
paired-reference direction checks failed. See the
[sanitized summary](../results/STAGE16_REJECTED_20260927.json).
Do not rerun that unchanged composition under a new wrapper or identity.

**Current mechanism:** exact-integer G128 accumulation at the L23/P0
`self_attn.o_proj` boundary, using unchanged retained Stage10 FP16 inputs.
Software owner `d12e372181f0` completed with independent review. Its successor
`7a2022cef21c` discovered incompatible native node-key/dependency requirements
and import-order-sensitive source membership. It ended with Reviewer
`replan_requested` and native `failed`, with **zero claims and zero numerical
invocations**. This is an unmeasured hypothesis, not a numerical REJECTED result.
Native software-repair owner `aba2b6c0f3d0` was running at capture.

The same scoped campaign was recovered through normal lifecycle APIs on release
`0.1.2+e69764dbdb9bd222`. The legal-empty-plan fix and reviewed deployment remain
complete. Do not recreate packaging or restart an already live daemon.

This publication captures 69 source/test files in its
[manifest](source-manifest-2026-09-28.json), including unchanged dependencies.
The four new implementation/test files match the historical software checkpoint
exactly; the subsequently discovered preclaim defects are deliberately disclosed.
Ongoing repair edits, authoritative handoffs and raw input arrays are excluded.
This is not an executable or operationally accepted release.
The original September 26 basis remains at `checkpoint-2026-09-26-source`;
later dependency snapshots must not be mistaken for that older experiment seal.
Never replace a running host workspace with this snapshot.

On the original host, run `python tools/argus_continuation.py inspect`.
Discover the actual daemon/runtime/backlog/review and terminal artifacts.
Do not duplicate a running task, replay a consumed experiment, or restore old
PID/claim records. Communicate through the normal Manager route.

Historical results remain in
[STAGE11_REJECTED_20260926.json](../results/STAGE11_REJECTED_20260926.json);
the [current boundary](../results/STAGE11_EXACT_O_PROJ_20260928.md) distinguishes
reviewed software, entry failure and unmeasured science.
These are sanitized summaries, not original receipts. Models, raw evidence,
authority, runtime state and credentials remain host-only; see
[HOST_ONLY_DEPENDENCIES.md](HOST_ONLY_DEPENDENCIES.md). The public repository is
not a standalone replay package or a full host-loss backup.
