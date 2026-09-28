"""Exact native gate/up accumulation from retained Q24-state S13 FP16 words.

Only S14-S18 and the final norm/two-row head execute. Integer AWQ grouping
replaces binary64 reduction at gate/up, not the FP16 operator boundary or
the wider Q24 residual state. References cannot enter the bit computation.
The fixed one-use output requires the normal Manager command handoff.
"""

import argparse
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ace3.model.candidates import stage13_q24_state_rmsnorm_contrast_v1 as parent


base, retained, prior = parent.base, parent.retained, parent.prior
native, layer, require, pin = parent.native, parent.layer, parent.require, parent.pin
ROOT, PREFIX, CONTROLS, PAIR = parent.ROOT, parent.PREFIX, parent.CONTROLS, parent.PAIR
SOURCE = Path(__file__).resolve()
NAME = SOURCE.stem
TEST = ROOT / f"tests/test_{NAME}.py"
MISSION = "1e2d468a4723"
OUTPUT = ROOT / "build" / f"{NAME}-{MISSION}-attempt001"
RETAINED_RESULT = parent.OUTPUT / "result.json"
RETAINED_HASH = "b6b6c4fbb75aca902353f05b3e71644f8bc2da26003d5e4c6abac88cf11dddf4"
BOUNDARY = (
    "CPU-only width896/L23/P0/pair[34319,13], nine controls, non-admission. "
    "Exact native integer G128 gate/up reduction from retained Q24-state S13 "
    "FP16 words; native GEMM nibble order, no qzero plus-one, FP16 scales, "
    "S16 RTZ, FP16 operator boundaries/KV and wider Q24 residual state. "
    "Only S14-S18/final RMSNorm/two-row head; S10-S13 gates without replay. "
    "Original references are comparators only. No prefix, admission, reference, "
    "closed-producer or Stage13 replay; no output patching, strict-FP16-state "
    "W4A16, new-token, full-model, hardware or precision/scale expansion claim."
)


def project_sources():
    return parent.project_sources() | {str(SOURCE), str(TEST)}


def historical_record(record):
    """Resolve only the consumed Stage13 release pin, never execution sources."""
    if record == {
            "path": "/home/argustest/ace3-argus/ace3/model/stage11_current_runtime_release.py",
            "bytes": 130008,
            "sha256": "aa8a02de4d174a9c8101881e81a026341ee50cb461beac499944d9fbc1d0d115"}:
        record = {
            **record,
            "path": str(ROOT / "build" /
                        "argus-stage11-value-content-current-census-release-f0385ca1ad3a-attempt001" /
                        "source.py")}
    return retained.selection.bound(record)


def projection(tensors, name, words):
    require(name in {PREFIX + "mlp.gate_proj", PREFIX + "mlp.up_proj"},
            "gate/up projection scope")
    require(isinstance(words, np.ndarray) and words.shape == (896,)
            and words.dtype.str == "<u2"
            and not np.any((words & 0x7c00) == 0x7c00), "stage13 finite FP16 words")
    for suffix, shape, dtype in (
            ("qweight", (896, 608), "<i4"), ("qzeros", (7, 608), "<i4"),
            ("scales", (7, 4864), "<f2")):
        value = tensors.get(name + "." + suffix)
        require(isinstance(value, np.ndarray) and value.shape == shape
                and value.dtype.str == dtype, "canonical gate/up tensor " + suffix)
        if suffix == "scales":
            require(np.all(np.isfinite(value)), "nonfinite gate/up scales")
    require(name + ".bias" not in tensors, "canonical gate/up has no bias")
    # single_bias also preserves negative underflow zero in this bias-free path.
    return native.state.project(tensors, name, words, single_bias=True)


def compute_suffix(raw, tensors, operands):
    prior.check_state(raw)
    arrays, timings = dict(raw), {}
    for stage, name, source in ((14, "gate_proj", 13), (15, "up_proj", 13),
                                (16, None, None), (17, "down_proj", 16), (18, None, None)):
        began = time.monotonic()
        if stage in (14, 15):
            words = projection(tensors, PREFIX + "mlp." + name, raw["stage13"])
        elif stage == 17:
            words = native.projection(tensors, PREFIX + "mlp." + name, arrays[f"stage{source:02d}"])
        elif stage == 16:
            value = (native.functional.silu(native.decoded(arrays["stage14"]))
                     * native.decoded(arrays["stage15"]))
            arrays["s16_unrounded_binary64"] = value.numpy().copy()
            words = native.toward_zero(arrays["s16_unrounded_binary64"])
        else:
            state = native.state.add(
                {"i": raw["scratch_i"], "z": raw["scratch_z"], "h": raw["stage12"]},
                arrays["stage17"])
            arrays["output_i"], arrays["output_z"] = state["i"], state["z"]
            words = state["h"]
        arrays[f"stage{stage:02d}"] = words
        timings[f"stage{stage}"] = time.monotonic() - began
    began = time.monotonic()
    arrays["final_rmsnorm"], detail = retained.suffix.rmsnorm(arrays["stage18"], operands[0])
    timings["final_rmsnorm"] = time.monotonic() - began
    began = time.monotonic()
    arrays["final_logits"] = retained.suffix.logits(arrays["final_rmsnorm"], operands[1])
    timings["pair_head"] = time.monotonic() - began
    prior.check_state(arrays)
    return arrays, detail, timings


@contextmanager
def guarded(out, audit):
    require(prior._active is None and sys.getprofile() is None, "nested execution guard")
    if not prior._hook_installed:
        sys.addaudithook(prior.audit_event)
        prior._hook_installed = True
    allowed = project_sources()
    forbidden = {
        "check", "main", "execute", "run", "diagnose", "stages", "_stages",
        "continuation_stages", "native_layer", "execute_layer", "execute_layers",
        "original_branches", "reference_arrays", "build_report", "intervene",
        "adjust", "nearest", "finite_table", "run_control", "guarded", "suffix_stages",
        "compute_suffix", "stage13", "evaluate_control",
        "evaluate_actual_rtl_result", "evaluate_p0_transaction",
    }

    def profile(frame, event, arg):
        if event != "call":
            return
        file, name = frame.f_code.co_filename, frame.f_code.co_name
        if not file.startswith(str(ROOT) + "/"):
            return
        key = file.removeprefix(str(ROOT) + "/") + ":" + name
        audit["project_calls"][key] = audit["project_calls"].get(key, 0) + 1
        denied = file not in allowed or (file != str(SOURCE) and name in forbidden)
        if file == str(Path(native.__file__).resolve()):
            denied |= name not in {"projection", "decoded", "rne", "toward_zero", "<genexpr>"}
            if name == "projection":
                denied |= frame.f_locals["name"] != PREFIX + "mlp.down_proj"
        if file == str(Path(native.state.__file__).resolve()) and name == "project":
            denied |= frame.f_locals["prefix"] not in {
                PREFIX + "mlp.gate_proj", PREFIX + "mlp.up_proj"}
        if name in {"expected_stage", "stage_report", "local_reference", "check_stage_state",
                    "evaluate_decoder_stage"}:
            denied |= frame.f_locals.get("stage") not in range(10, 19)
        if denied:
            audit["forbidden_events"].append(key)
            raise retained.bridge.BindingError("unapproved project dispatch: " + key)

    prior._active = (out, audit)
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(None)
        prior._active = None


def evaluate_control(control, raw, original, classified, tensors, trajectory,
                     reference, refs, operands, oracle, audit):
    require(control == classified["control"], "retained control lineage splice")
    protected = {k: retained.digest(v) for k, v in raw.items()}
    arrays, detail, timings = compute_suffix(raw, tensors, operands)
    audit["controls"].append(control)
    audit["stage_dispatches"].extend([[control, 23, 0, s] for s in range(14, 19)])
    audit["final_rmsnorm_invocations"] += 1
    audit["pair_head_invocations"] += 1
    began = time.monotonic()
    for stage, name in ((14, "gate_proj"), (15, "up_proj")):
        oracle.verify_projection(tensors, PREFIX + "mlp." + name,
                                 raw["stage13"], arrays[f"stage{stage}"])
    oracle.suffix_oracle.verify_suffix(
        arrays["stage18"], operands, arrays["final_rmsnorm"], detail, arrays["final_logits"])
    require(np.array_equal(layer.prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                           arrays["stage16"]), "native S16 RTZ mismatch")
    reports = []
    state = {"h": raw["input_hidden"], "i": raw["input_i"], "z": raw["input_z"]}
    for stage in range(10, 19):
        expected = layer.expected_stage(stage, arrays, state, tensors)
        reports.append(layer.stage_report(stage, arrays, trajectory, reference, expected))
        audit["stage_gate_checks"].append([control, 23, 0, stage])
    after = prior.compare(arrays["stage18"], reference)
    require(after["status"] == reports[-1]["status"], "S18 comparator disagreement")
    margins = retained.paired_margins(
        arrays["final_logits"], original["final_logits"], classified["paired_margins"], refs)
    require(protected == {k: retained.digest(v) for k, v in raw.items()}, "retained bytes changed")
    for key in tuple(f"stage{s:02d}" for s in range(14)) + (
            "input_hidden", "input_i", "input_z", "scratch_i", "scratch_z",
            "input_cache_k", "input_cache_v", "output_cache_k", "output_cache_v"):
        require(retained.digest(arrays[key]) == protected[key], "protected prefix/state/KV: " + key)
    timings["oracles_and_gates"] = time.monotonic() - began
    return arrays, {
        "control": control, "stages": reports, "S18": after, "paired_margins": margins,
        "component_oracles": {"status": "PASS", "errors": []},
        "protected_arrays": protected, "protected_bytes_unchanged": True,
        "changed_words": {str(s): int(np.count_nonzero(arrays[f"stage{s}"] != raw[f"stage{s}"]))
                          for s in (14, 15)},
        "final_rmsnorm_integer_details": detail, "timing_seconds": timings,
    }


def scientific_status(rows):
    require([r["control"] for r in rows] == list(CONTROLS), "result control census")
    for row in rows:
        require([s["stage"] for s in row["stages"]] == list(range(10, 19))
                and all(s["status"] in {"PASS", "FAIL"} for s in row["stages"])
                and row["S18"]["status"] in {"PASS", "FAIL"}
                and row["component_oracles"]["status"] in {"PASS", "FAIL"}
                and [m["reference_branch"] for m in row["paired_margins"]]
                == ["fp16", "binary64"]
                and all(type(m["direction_pass"]) is bool for m in row["paired_margins"])
                and row["protected_bytes_unchanged"] is True, "result schema/integrity defect")
    return base.scientific_status(rows)


def retained_input(document, row):
    require(document["mission_id"] == parent.MISSION and document["status"] == "REJECTED"
            and document["science_invocations"] == 1 and document["non_admission"] is True
            and document["audit"]["forbidden_events"] == []
            and [r["control"] for r in document["controls"]] == list(CONTROLS),
            "retained Stage13 execution identity")
    with np.load(io.BytesIO(historical_record(row["payload"])), allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate archive keys")
        original = {k: archive[k] for k in archive.files}
    raw = dict(original)
    for key, value in original.items():
        if key.startswith("q24_state_contrast_"):
            raw[key.removeprefix("q24_state_contrast_")] = value
    require("q24_state_contrast_stage13" in original, "missing retained Q24-state S13")
    prior.check_state(raw)
    for value in raw.values():
        value.flags.writeable = False
    return original, raw


def dispatch_census(audit):
    def count(module, name):
        key = str(Path(module.__file__).resolve().relative_to(ROOT)) + ":" + name
        return audit["project_calls"].get(key, 0)
    require(audit["forbidden_events"] == [] and audit["controls"] == list(CONTROLS)
            and audit["stage_dispatches"] == [[c, 23, 0, s] for c in CONTROLS for s in range(14, 19)]
            and audit["stage_gate_checks"] == [[c, 23, 0, s] for c in CONTROLS for s in range(10, 19)]
            and audit["final_rmsnorm_invocations"] == audit["pair_head_invocations"] == 9,
            "ordered dispatch/gate census defect")
    require(count(native.state, "project") == 18 and count(native, "projection") == 9
            and count(native, "toward_zero") == count(native.state, "add") == 9
            and count(retained.fp16_adaptation_oracle, "rmsnorm") == 9,
            "observed arithmetic dispatch census defect")
    require(audit["writes"] == [str(OUTPUT / (c + ".npz")) for c in CONTROLS], "write census defect")


def save(name, value):
    with (OUTPUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def check():
    require(Path.cwd() == ROOT and os.getuid() == 1000 and sys.dont_write_bytecode
            and not sys.flags.optimize, "isolated source/account/bytecode context")
    require(subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
            == "argus/full-projection", "isolated branch")
    require(subprocess.run(["git", "check-ignore", "--quiet", str(OUTPUT)]).returncode == 0,
            "output must be ignored")
    require(OUTPUT.resolve() == OUTPUT and OUTPUT.is_dir()
            and {p.name for p in OUTPUT.iterdir()} == {"command.sh"},
            "fresh Manager-provisioned output missing or occupied")
    command = (OUTPUT / "command.sh").read_text()
    require(command == f"python3 -B -m ace3.model.candidates.{NAME} --check\n",
            "command sidecar mismatch")
    mission_path = retained.HANDOFFS / MISSION / "mission.json"
    mission = json.loads(mission_path.read_bytes())
    require(mission["mission_id"] == MISSION and mission["scope"] == "bounded"
            and mission["node_key"] == "stage14-stage15-q24-gate-up-projection-contrast"
            and mission["execution_workdir"] == str(ROOT) and mission["deps"] == []
            and "review:required" in mission["tags"], "current science owner/dependency preflight")
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec")
    oracle = retained.module_from(TEST, NAME + "_oracle")
    sources = [pin(Path(p)) for p in sorted(project_sources())]
    save("command.json", {
        "mission_id": MISSION, "mission": pin(mission_path), "command_bytes": command,
        "argv": sys.argv, "cwd": str(ROOT), "uid": os.getuid(), "pid": os.getpid(),
        "executable": sys.executable, "timestamp": time.time(), "science_budget": 1,
        "authorization_id": "", "action": "", "source_pins": sources,
        "claim": "exclusive command.json; fixed fresh root; no retries or alternate roots"})
    audit, rows = base.new_audit(), []
    result = {
        "schema_version": 1, "mission_id": MISSION, "status": "UNKNOWN",
        "controls": rows, "audit": audit, "science_budget": 1, "science_invocations": 0,
        "non_admission": True, "normal_independent_review": "REQUIRED",
        "automatic_replay_authorized": False, "boundary": BOUNDARY, "source_pins": sources,
        "flags": base.closure_source.FLAGS}
    try:
        native.torch.set_num_threads(1)
        with guarded(OUTPUT, audit):
            closure, _, bridge, ref, refs, operands, tensors, trajectory, records = base.inputs()
            document = json.loads(retained.bridge.authenticate(RETAINED_RESULT, RETAINED_HASH))
            records.extend([pin(RETAINED_RESULT), *document["input_pins"], *document["source_pins"],
                            *(r["payload"] for r in document["controls"]),
                            pin(parent.OUTPUT / "command.json"), pin(mission_path),
                            pin(OUTPUT / "command.sh")])
            for record in records:
                historical_record(record)
            require(document["thresholds"] == closure["thresholds"]
                    and [r["control"] for r in bridge["controls"]] == list(CONTROLS),
                    "retained thresholds/control lineage")
            prepared = [retained_input(document, row) for row in document["controls"]]
            result.update(input_pins=records, thresholds=closure["thresholds"],
                          historical_failure=closure["historical_failure"],
                          retained_stage13=pin(RETAINED_RESULT),
                          operand_pins=[retained.digest(v) for v in operands],
                          tensor_pins={k: retained.digest(v) for k, v in tensors.items()})
            result["science_invocations"] = 1
            for control, (original, raw), classified in zip(
                    CONTROLS, prepared, bridge["controls"], strict=True):
                arrays, row = evaluate_control(
                    control, raw, original, classified, tensors, trajectory, ref, refs,
                    operands, oracle, audit)
                path = OUTPUT / (control + ".npz")
                with path.open("xb") as stream:
                    np.savez(stream, **arrays)
                row["payload"] = pin(path)
                rows.append(row)
            for record in records:
                historical_record(record)
            for record in sources:
                retained.selection.bound(record)
            require(result["operand_pins"] == [retained.digest(v) for v in operands]
                    and result["tensor_pins"] == {k: retained.digest(v) for k, v in tensors.items()},
                    "authenticated operands changed")
            dispatch_census(audit)
            result["status"] = scientific_status(rows)
            result["protected_input_files_unchanged"] = True
    except Exception as error:
        # Preserve a consumed execution/integrity failure, not a numerical verdict.
        result.update(status="UNKNOWN", error_type=type(error).__name__, error=str(error))
    result["evidence_write_census"] = sorted(p.name for p in OUTPUT.iterdir()) + ["result.json"]
    save("result.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 1 if result["status"] == "UNKNOWN" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", required=True, action="store_true")
    parser.parse_args()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
