# Parent-Host capture for the single-round P0/L5-L8 cone

`ace3.model.candidates.host_capture_v3` separates preparation, actual capture,
and externally trusted admission. It does not issue a Host identity, Reviewer
verdict, numerical waiver, or runtime trust input.

`prepare --out ABSOLUTE_FRESH_BUILD_PATH` authenticates the retained v3 software
support, genuine software/L4 reviews, exact retained RTL source/header,
canonical tensors, original independent references, actual L4 input, empty P0
prior state, and host tools. It compiles the exact public contract, not the
decoder implementation. `freeze.json` contains exact compiler/simulator argv,
bindings and separate planned paths for layers 5 through 8.
`launch_action.json` names the runnable first capture. No future raw output,
state, completed runtime receipt or trusted runtime digest is required or
created during preparation.

The parent independently inspects the launch action, its command file and
source/input checksums, then invokes `capture-layer05.command.sh` using the
existing durable runner, serialized with other local decoder executions.
Capture reuses the established vector serializer and an exactly source-derived
capture-only mode of the C++ transaction harness,
Verilator compile command and saved-state ABI. Command sidecars precede
subprocess creation; real PID, timing, exit status, logs, generated ABI, binary,
input, trace, final output and own-layer state are recorded. Failed/partial
attempts remain immutable and cannot create admission.

The frozen harness retains expectation files from reviewed software archives as
**coordinate/count monitors**, never activations, Q, hidden feedback or KV
supplied to RTL. Explicit `--capture-only` disables only the two comparisons
of trace/final FP16 values with those files. Ordering, indices, position, token,
last/done metadata, stall stability, count, fault, timeout, file-write and
saved-state paths remain intact. All actual trace/final values are recorded;
successful capture logs `CAPTURED`, not numerical `PASS`. Without the flag,
the original strict behavior remains. Preparation authenticates the retained
software input chain. At execution, a passing actual predecessor may differ
from that software chain: it supplies the real hidden input, while monitor
values neither constrain it nor replace it.

The derivative is generated under the fresh build path by
`capture_harness_v3.capture_source` from the authenticated retained harness.
No RTL, port, parameter, arithmetic, tensor drive or state serialization code
is changed. Runtime source admission accepts either the exact legacy closure
or precisely this derivative with the explicit capture flag; arbitrary harness
edits and all arithmetic source changes still fail. This narrow source
compatibility addition does not change v3 numerical/reference/trust/state
requirements and requires independent review before launch.

After observing and authenticating its actual capture, the parent supplies the
manifest digest separately:

```text
python3 -B -m ace3.model.candidates.host_capture_v3 admit \
  --plan ABSOLUTE_PLAN_PATH --layer 5 --trusted-manifest-sha256 HOST_OBSERVED_DIGEST
```

Never obtain that trust input by copying a candidate's claimed digest. The
capture result deliberately contains only a manifest path and
`CAPTURED_NOT_ADMITTED`, not a trusted digest. Admission calls the existing v3
source/state validator and independent local/global numerical evaluator.
Only a passing actual admission may be the next hidden parent:

Both `admit` and `admit-recovery` accept an optional `--out
ABSOLUTE_FRESH_BUILD_ROOT`. An explicit admission root must be a resolved,
nonexistent **direct child** of the repository's `build/` directory, not a
symlink, an existing file/directory, or a path nested inside a retained attempt.
All stage reports, local references, the command receipt and the result go
there. This permits additive retained L5 evaluation without adding or replacing
any file in the original attempt003 or recovered attempt004 inventories:

```text
python3 -B -m ace3.model.candidates.host_capture_v3 admit-recovery \
  --plan ABSOLUTE_RECOVERY_FREEZE --layer 5 \
  --trusted-manifest-sha256 HOST_OBSERVED_DERIVED_MANIFEST_SHA256 \
  --out ABSOLUTE_FRESH_BUILD_ROOT
```

The result's `admission_invocation` authenticates `command.json`, which binds
the exact output root, plan, manifest, separately supplied Host digest and argv.
An absent/malformed digest or digest mismatch still fails before numerical
evaluation; selecting an output path grants no manifest or source authority.
All plan/source/derivation/state checks and numerical gates still run unchanged.
Source drift remains a rejection, not permission to edit old freezes or bypass
validation. Omitting `--out` retains `<transaction>/admission` with exclusive
creation; an existing admission is never overwritten. `capture` cannot use
`--out` to override its frozen runtime path.

An external admission is not automatically substituted into the frozen
next-layer parent lookup. Consuming it requires a separately compatible,
source-bound continuation; this option neither launches RTL nor authorizes L6.
For the original in-plan capture/admission sequence:

```text
python3 -B -m ace3.model.candidates.host_capture_v3 capture \
  --plan ABSOLUTE_PLAN_PATH --layer 6 --parent-result-sha256 HOST_OBSERVED_ADMISSION_DIGEST
```

Apply capture/admit in order through L8, without another permission mission;
stop at any failure. Each layer resets its own P0 cache and saves its own state;
no preceding layer opaque state is restored. Each actual capture is a durable
runner command, not a foreground long-running shell. A retry needs a new
preparation output path. No L0-L4 replay, model/tail/token generation, synthesis,
PPA or hardware result follows.

The runtime trace reader decodes the retained writer's actual
`token[2] position[4] stage[2] index[4] fp16[4]` hexadecimal ABI and requires
transaction token ordinal zero. The earlier synthetic stage-prefix fixture
did not match that writer. This parser correction changes no RTL arithmetic,
ports, thresholds, reference inputs or admission requirements; its acceptance
still requires normal independent review.

Capture and admission share coordinate-aware P0 tensor assembly. Vector stages
use the explicit index, including RoPE's low/high pair emission. S8/S9 use
context index zero and fourteen head occurrences; each score/probability pair
must precede that head's 64 S10 dimensions. Missing, duplicate, out-of-range,
wrong-position and invalid coordinates fail. Canonical tensor indexing does
not replace the separately authenticated raw emission/causal-order check.

For a completed capture rejected by an older decoder, `recover --plan
ORIGINAL_FREEZE --out FRESH_BUILD_PATH --historical-sources HOST_SOURCE_SNAPSHOT`
creates only additive derived operands. It authenticates historical producer
helpers against the original freeze, binds the corrected sources in the new
freeze and vector boundary manifest, and retains the original execution
receipts, raw trace/final/state, wrong-order archive and failed admission.
No simulator is launched or state restored. `host_action.json` supplies the
exact `admit-recovery` command; the parent must independently authenticate the
derived manifest and supply its digest. Recovery's layout/lineage/state checks
are not runtime or numerical admission. All original v3 gates remain mandatory.
This recovery interface is L5/P0-only and cannot launch L6.

## Continuation from an externally admitted retained L5

`ace3.model.candidates.prepare_l6_l8_continuation_v3` prepares a fresh plan
with `--out`, `--parent-result`, `--recovery-plan`, and
`--historical-admission-sources`. It reuses the supported capture/admit
interfaces without rerunning L5 admission or changing either original freeze.
The plan retains the original four-layer compatibility scope; its explicit
`continuation.layers` is `[6, 7, 8]`. Transaction 5 points read-only to the
existing recovered admission directory, so the existing parent lookup consumes
the actual L5 final hidden. L5 saved state is evidence only, never an L6 restore.
All three new runtime directories are fresh and each starts with empty P0 KV.

Preparation authenticates original runtime receipts and preserved historical
producer revisions separately from current capture/decoder sources, preserves
the exact capture-only harness and arithmetic closure, binds canonical
weights/control and unchanged global references, and compiles only the public
interface. It emits exact `capture-layerNN.command.sh` and
`admit-layerNN.command.sh` files with preexecution checks. The parent supplies
`HOST_PARENT_RESULT_SHA256` for capture and `HOST_MANIFEST_SHA256` for admission
from its own observations, not from a candidate-issued trust token.
Normal independent acceptance of both actual L5 and the new bindings is required
before L6. Invoke each capture through the durable runner; authenticate and
admit its result before the next capture. Stop on failure. Preparation is not a
numerical evaluation, reviewer verdict, new simulator run, or generated token.
