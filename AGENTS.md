# ACE-3 continuation entry

Read `docs/continuation/HANDOFF.zh-CN.md` and
`docs/continuation/progress-2026-10-11.json` before changing anything.
Keep `docs/continuation/progress-2026-09-28.json` as the historical baseline.

The October 11 increment publishes the independently reviewed SiLU WIDTH repair.
It is development-level source acceptance, not V1 full-regression acceptance.
Keep these repaired bytes when assembling new execution archives; do not silently
restore the pre-repair file from an older checkout.

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
identified by the October 11 source manifest. Other later repair edits are not
included. The original September 26 source basis remains at its dated tag.
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
