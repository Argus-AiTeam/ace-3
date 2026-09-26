# ACE-3 continuation handoff

Snapshot date: **2026-09-26**. Exact observation time and live work are in
[progress-2026-09-26.json](progress-2026-09-26.json). Live state can advance.
The operational guide is [HANDOFF.zh-CN.md](HANDOFF.zh-CN.md).

The equal mirrors are <https://github.com/aHappend/ace-3> and
<https://github.com/Argus-AiTeam/ace-3>, branch `main`, checkpoint ref
`refs/tags/checkpoint-2026-09-26-source`. Publish as `aHappend` to both and verify
identical commit/tree IDs; see [DUAL_MIRROR_PUBLICATION.md](DUAL_MIRROR_PUBLICATION.md).

**Completed, not instructions to repeat:** release `287d994f2fe4`, continuation
`0426fd273809`, actual CPU check `b4978b898600`, and v2 S18 boundary experiment
under `6738b75d0cc8` all received normal independent review and native DONE.
Both scientific results are **REJECTED**. CPU execution recovered; neither
result supports the tested hypothesis or establishes model/hardware admission.
The v2 experiment recovered zero of eighteen terminal directions.

Afterward, stale temporary recovery steering caused unnecessary release
preparation. That steering was withdrawn, and a genuine retained-evidence
analysis task started at 13:00 UTC. Do not recreate the obsolete fresh-v6
checklist. Select unfinished research from current evidence. New execution
preparation is justified by a concrete new task, not by an already completed
recovery instruction.

This publication restores the exact reviewed source basis of the two results,
plus the reviewed Argus Planner format-retry patch as a separate patch artifact.
It deliberately does not overwrite that basis with later live release-module
changes. See [source-manifest-2026-09-26.json](source-manifest-2026-09-26.json).
Never replace a running host workspace with this snapshot. The patch is not
automatically applied to a different Argus release.

On the original host, run `python tools/argus_continuation.py inspect`.
Discover the actual daemon/runtime/backlog/review and terminal artifacts.
Do not duplicate a running task, replay a consumed experiment, or restore old
PID/claim records. Communicate through the normal Manager route.

Results and original evidence hashes are in
[STAGE11_REJECTED_20260926.json](../results/STAGE11_REJECTED_20260926.json).
These are sanitized summaries, not original receipts. Models, raw evidence,
authority, runtime state and credentials remain host-only; see
[HOST_ONLY_DEPENDENCIES.md](HOST_ONLY_DEPENDENCIES.md). The public repository is
not a standalone replay package or a full host-loss backup.
