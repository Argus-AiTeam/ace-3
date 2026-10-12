# ACE-3 continuation entry

Read `docs/continuation/HANDOFF.zh-CN.md` and
`docs/continuation/progress-2026-10-12-bound-checkpoint.json` before changing anything.
Keep `docs/continuation/progress-2026-10-12-decoder-failure-probe.json` as the earlier decoder record.
Keep `docs/continuation/progress-2026-10-12-file-gates.json` as the earlier file-open gate record.
Keep `docs/continuation/progress-2026-10-12.json` as the earlier path/PASS repair record.
Keep the October 11 SiLU source record as an accepted development increment.
Keep `docs/continuation/progress-2026-09-28.json` as the historical baseline.

The October 12 increment adds the independently reviewed Makefile and FP16/QKV
testbench path and PASS-sentinel repairs, retaining the October 11 SiLU repair.
These are development-level source increments, not V1 full-regression acceptance.
Keep these repaired bytes when assembling new execution archives; do not silently
restore the pre-repair file from an older checkout.

The subsequent October 12 file-gate increment preserves those repairs and adds
the reviewed fail-closed checks for unreadable vector files and unwritable VCD
paths in the same two testbenches. Copy these exact bytes into future archives;
source publication does not certify the full V1 regression or authorize replay.

The decoder failure-probe increment copies one reviewed C++ file. Only its
injected-failure branch avoids reading expected trace/final records before
recording the first DUT output and failing. Normal numerical comparisons remain.
Do not import the separate private-path Makefile, frozen-hash replacements or
LD_PRELOAD tracing wrapper. Their provenance remains unresolved; this source
increment is development/not_certified, not full V2/model acceptance.

The bound-checkpoint increment copies the independently reviewed position-3
LM-head validator and its focused test. Python callers must supply the
keyword-only `checkpoint`; CLI validation must supply `--checkpoint`. The
validator no longer silently recomputes with a different default checkpoint.
No oracle arithmetic, numerical tolerance or model data changed. The isolated
test checks argument forwarding with internal evidence/model fixtures; it is
not a full producer-to-execution regression or V3 acceptance. Do not copy the
private proposal producer or raw execution state into this public snapshot.

At the September 28 snapshot, S13-S17 composition `92e4223ef310` was reviewed REJECTED.
The new exact-integer Stage11 o_proj software `d12e372181f0` received review,
but science owner `7a2022cef21c` stopped before any claim or numerical run:
the wrapper disagrees with its native mission schema and its source census
depends on import order. Repair owner `aba2b6c0f3d0` was running at the snapshot.
Do not call this a completed or rejected Stage11 numerical experiment.
Preserve the failure, complete independent repair review, then use a coherent
fresh science owner. Do not duplicate a running repair or replay consumed work.

Published sources are the historical reviewed implementation and its dependency
snapshot, including the subsequently discovered defects, plus the one RTL file
identified by the October 11 source manifest, the three files identified by
the October 12 path/PASS manifest, the subsequent two-file file-gate manifest,
the single C++ file in the decoder failure-probe manifest, and the validator/test
pair in the bound-checkpoint manifest.
Other later repair edits are not included. The original September 26 source
basis remains at its dated tag.
Never copy this snapshot over a running workspace.

On the original host, run `python tools/argus_continuation.py inspect` first.
Discover the live runtime from `daemon.status.json`; never import the ambient
checkout as a substitute. Do not restart a live daemon or duplicate a running
mission. Communicate through the normal Manager route. Read-only observation is
not authorization, and acceptance requires the real issuer binding, normal
Reviewer verdict, and native terminal.

Argus owns implementation, science, operation, and independent review. A new
conversation should communicate objectives and verify receipts, not replay a
one-shot Engineer operation or seize a claim. Keep rejected and failed evidence.

GitHub is a curated source and continuation snapshot, not a complete backup of
model weights, raw evidence, active state, credentials, or scheduled supervision.
See `docs/continuation/HOST_ONLY_DEPENDENCIES.md`.
