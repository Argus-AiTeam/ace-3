"""One-use CPU S13 integer RMSNorm contrast on retained value-content inputs.

Run the fixed ignored attempt's command.sh. In-memory compilation, isolated
pinned-review checks and synthetic integration through the real S13-S18 entry
precede the sole retained-data run.
S10-S12 are retained, not dispatched; their unchanged gates are reevaluated.
References are visible only after the suffix output bits have been computed.
An UNKNOWN consumes the attempt and cannot authorize a retry.
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
import unittest

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_value_content_s18_rtz_rounding_contrast_v1 as prior
from ace3.model.candidates import q24_s16_l23_p0_value_content_sufficiency_v1 as closure_source


retained = prior.retained
layer, native = closure_source.layer, closure_source.native
ROOT = retained.ROOT
SOURCE = Path(__file__).resolve()
NAME = SOURCE.stem
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "-782b265df4c5-attempt001")
MISSION = "782b265df4c5"
CONTROLS, PAIR = retained.CONTROLS, retained.PAIR
PREFIX = "model.layers.23."
RTZ_RESULT = prior.OUTPUT / "result.json"
RTZ_HASH = "9049b093ea96f4c54be566de5cc7625c7b0f1809cd0cb917c8bf4fd3f68755f0"
RTZ_REVIEW_HASH = "6685445bd183b639824a91d3fa960732649e3a01e05fc2e2cab9861751464e63"
require, pin = retained.require, retained.pin
BOUNDARY = (
    "CPU-only width896/L23/P0/pair[34319,13], isolated non-admission. "
    "S13 derives solely from retained S12 FP16 words and authenticated post-attention "
    "norm weights via integer FP16 RMSNorm. Only S13-S18, final integer RMSNorm "
    "and two tied-head rows execute. S10-S12 gates are reevaluated without native "
    "S10-S12 replay. No original prefix/admission/reference/closed-producer replay, "
    "reference-selected output words, coordinate search or margin targeting. "
    "Original-input global FP16/binary64 references remain comparators only. "
    "Retained value-substitution ancestry is not native-V repair. Native S16 RTZ, "
    "G128 asymmetric packed INT4/GEMM nibble ordering/no qzero plus-one, FP16 "
    "scales/operators/KV and wider Q24 residual state are unchanged. No strict-"
    "FP16-state W4A16, new-token/full-model admission, hardware or scale expansion."
)


def project_sources():
    paths = {str(SOURCE), str(TEST), str(retained.ORACLE)}
    paths.update(str(Path(m.__file__).resolve()) for n, m in tuple(sys.modules.items())
                 if n.startswith("ace3.") and getattr(m, "__file__", None))
    return paths


def new_audit():
    return {"scope": "process-local Python audit/project-call profile; not an OS sandbox",
            "project_calls": {}, "forbidden_events": [], "writes": [], "controls": [],
            "stage_dispatches": [], "stage_gate_checks": [],
            "final_rmsnorm_invocations": 0, "pair_head_invocations": 0}


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
                denied |= frame.f_locals["name"] not in {
                    PREFIX + "mlp." + s for s in ("gate_proj", "up_proj", "down_proj")}
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


def stage13(words, gamma):
    require(words.shape == gamma.shape == (896,) and words.dtype.str == "<u2"
            and gamma.dtype.str == "<f2" and np.all(np.isfinite(gamma)),
            "retained stage13 entry width/dtype/weight")
    layer.prior.local.finite_words(words, (896,))
    return retained.suffix.rmsnorm(words, gamma)


def compute_suffix(raw, tensors, operands):
    """No reference, coordinate list, acceptance interval or margin is an input."""
    prior.check_state(raw)
    arrays = dict(raw)
    timings, began = {}, time.monotonic()
    arrays["stage13"], detail = stage13(raw["stage12"], tensors[PREFIX + "post_attention_layernorm.weight"])
    timings["stage13"] = time.monotonic() - began
    for stage, name, source in ((14, "gate_proj", 13), (15, "up_proj", 13),
                                (16, None, None), (17, "down_proj", 16), (18, None, None)):
        began = time.monotonic()
        if stage in (14, 15, 17):
            words = native.projection(tensors, PREFIX + "mlp." + name, arrays[f"stage{source:02d}"])
        elif stage == 16:
            value = (native.functional.silu(native.decoded(arrays["stage14"]))
                     * native.decoded(arrays["stage15"]))
            arrays["s16_unrounded_binary64"] = value.numpy().copy()
            words = native.toward_zero(arrays["s16_unrounded_binary64"])
        else:
            successor = native.state.add(
                {"i": raw["scratch_i"], "z": raw["scratch_z"], "h": raw["stage12"]},
                arrays["stage17"])
            arrays["output_i"], arrays["output_z"] = successor["i"], successor["z"]
            words = successor["h"]
        arrays[f"stage{stage:02d}"] = words
        timings[f"stage{stage}"] = time.monotonic() - began
    began = time.monotonic()
    arrays["final_rmsnorm"], final_detail = retained.suffix.rmsnorm(arrays["stage18"], operands[0])
    timings["final_rmsnorm"] = time.monotonic() - began
    began = time.monotonic()
    arrays["final_logits"] = retained.suffix.logits(arrays["final_rmsnorm"], operands[1])
    timings["pair_head"] = time.monotonic() - began
    prior.check_state(arrays)
    return arrays, detail, final_detail, timings


def scientific_status(rows):
    require(len(rows) == 9 and [r["control"] for r in rows] == list(CONTROLS), "result control census")
    return "SUPPORTED" if all(
        len(r["stages"]) == 9 and [s["stage"] for s in r["stages"]] == list(range(10, 19))
        and all(s["status"] == "PASS" for s in r["stages"])
        and r["S18"]["status"] == "PASS" and r["component_oracles"]["status"] == "PASS"
        and len(r["paired_margins"]) == 2 and all(m["direction_pass"] for m in r["paired_margins"])
        for r in rows) else "REJECTED"


def run_control(control, raw, old, classified, tensors, trajectory, reference, refs, operands, oracle, audit):
    require(control == old["control"] == classified["control"], "retained control lineage splice")
    protected = {k: retained.digest(v) for k, v in raw.items()}
    arrays, detail, final_detail, timings = compute_suffix(raw, tensors, operands)
    audit["controls"].append(control)
    audit["stage_dispatches"].extend([[control, 23, 0, s] for s in range(13, 19)])
    audit["final_rmsnorm_invocations"] += 1
    audit["pair_head_invocations"] += 1
    began = time.monotonic()
    component_errors = []
    try:
        oracle.verify_norm(raw["stage12"], tensors[PREFIX + "post_attention_layernorm.weight"],
                           arrays["stage13"], detail)
        oracle.suffix_oracle.verify_suffix(
            arrays["stage18"], operands, arrays["final_rmsnorm"], final_detail, arrays["final_logits"])
    except AssertionError as error:
        component_errors.append(str(error) or "independent integer RMSNorm/head mismatch")
    if not np.array_equal(layer.prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                          arrays["stage16"]):
        component_errors.append("native S16 RTZ mismatch")
    reports = []
    parent = {"h": raw["input_hidden"], "i": raw["input_i"], "z": raw["input_z"]}
    for stage in range(10, 19):
        expected = layer.expected_stage(stage, arrays, parent, tensors)
        reports.append(layer.stage_report(stage, arrays, trajectory, reference, expected))
        if expected is not None:
            arrays[f"local_reference_stage{stage:02d}"] = expected
        audit["stage_gate_checks"].append([control, 23, 0, stage])
    before, after = prior.compare(raw["stage18"], reference), prior.compare(arrays["stage18"], reference)
    require(before["failure_count"] == old["failure_count"], "retained S18 comparator splice")
    require(after["status"] == reports[-1]["status"], "S18 gate comparator disagreement")
    margins = retained.paired_margins(arrays["final_logits"], raw["final_logits"],
                                     classified["paired_margins"], refs)
    require(protected == {k: retained.digest(v) for k, v in raw.items()},
            "protected retained bytes changed")
    for key in tuple(f"stage{s:02d}" for s in range(13)) + (
            "input_hidden", "input_i", "input_z", "scratch_i", "scratch_z",
            "input_cache_k", "input_cache_v", "output_cache_k", "output_cache_v"):
        require(retained.digest(arrays[key]) == protected[key], "protected prefix/state/KV: " + key)
    timings["oracles_and_gates"] = time.monotonic() - began
    changed = {k: v for k, v in arrays.items() if k not in raw or v is not raw[k]}
    payload = {**raw, **{"integer_contrast_" + k: v for k, v in changed.items()}}
    return payload, {
        "control": control, "stages": reports, "S18_before": before, "S18": after,
        "paired_margins": margins, "protected_arrays": protected, "protected_bytes_unchanged": True,
        "component_oracles": {"status": "FAIL" if component_errors else "PASS", "errors": component_errors},
        "stage13_integer_oracle": {
            **detail, "epsilon_q48": 281474977, "width": 896,
            "semantics": "RNE mean Q48; floor integer sqrt; signed magnitude RNE division Q24; FP16 RNE",
            "input_fields": ["stage12", PREFIX + "post_attention_layernorm.weight"],
            "reference_selected_output_words": False,
            "independent_check": "rational FP16 decode, Python rational ties-even and isqrt; binary-search FP16 RNE",
            "changed_words": int(np.count_nonzero(arrays["stage13"] != raw["stage13"]))},
        "final_rmsnorm_integer_details": final_detail, "timing_seconds": timings}


def rtz_review_pins():
    review = retained.HANDOFFS / "9602adc7f96e/round-0001.json"
    record = {"path": str(review), "sha256": RTZ_REVIEW_HASH}
    accepted = json.loads(retained.selection.bound(record))
    latest = retained.HANDOFFS / "9602adc7f96e/latest.json"
    require(accepted["mission_id"] == "9602adc7f96e" and accepted["producer_role"] == "reviewer"
            and accepted["review"]["status"] == "done"
            and json.loads(latest.read_bytes())["handoff"]["path"] == str(review),
            "independent RTZ review missing")
    return [record, pin(latest)]


def inputs():
    closure, failure, bridge, reference, refs, operands, records = prior.inputs()
    contrast = json.loads(retained.bridge.authenticate(RTZ_RESULT, RTZ_HASH))
    require(contrast["mission_id"] == "9602adc7f96e" and contrast["status"] == "REJECTED"
            and contrast["non_admission"] and contrast["science_invocations"] == 1
            and contrast["audit"]["forbidden_events"] == []
            and contrast["thresholds"] == closure["thresholds"], "historical RTZ contrast identity")
    records.extend([pin(RTZ_RESULT), *rtz_review_pins(),
                    *contrast["input_pins"], *contrast["source_pins"],
                    *(r["payload"] for r in contrast["controls"]),
                    pin(prior.OUTPUT / "command.json"), pin(prior.OUTPUT / "validation.json")])
    baseline = ROOT / "build/q24_s16_final_head_from_l23_coordinate62_suffix_execution_v1_attempt001/result.json"
    summary = json.loads(retained.bridge.authenticate(
        baseline, closure["preservation"]["sha256_before"][str(baseline)]))["preflight"]
    tensors, trajectory, second_reference, checkpoint = layer.load_inputs({"summary": summary})
    require(np.array_equal(second_reference, reference), "original global reference lineage changed")
    records.append(checkpoint)
    for array in (*tensors.values(), *trajectory.values(), *operands):
        array.flags.writeable = False
    unique = {}
    for record in records:
        require(record["path"] not in unique
                or unique[record["path"]]["sha256"] == record["sha256"], "conflicting immutable pins")
        unique[record["path"]] = record
    records = list(unique.values())
    for record in records:
        retained.selection.bound(record)
    return closure, failure, bridge, reference, refs, operands, tensors, trajectory, records


def save(name, value):
    with (OUTPUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def dispatch_census(audit):
    def count(module, function):
        key = str(Path(module.__file__).resolve().relative_to(ROOT)) + ":" + function
        return audit["project_calls"].get(key, 0)
    require(audit["forbidden_events"] == [] and audit["controls"] == list(CONTROLS)
            and audit["stage_dispatches"] == [[c, 23, 0, s] for c in CONTROLS for s in range(13, 19)]
            and audit["stage_gate_checks"] == [[c, 23, 0, s] for c in CONTROLS for s in range(10, 19)]
            and audit["final_rmsnorm_invocations"] == audit["pair_head_invocations"] == 9,
            "ordered dispatch/gate census defect")
    require(count(native, "projection") == 27 and count(native, "toward_zero") == 9
            and count(native.state, "add") == 9
            and count(retained.fp16_adaptation_oracle, "rmsnorm") == 18,
            "observed arithmetic dispatch census defect")
    require(audit["writes"] == [str(OUTPUT / (c + ".npz")) for c in CONTROLS], "write census defect")


def execute():
    require(Path.cwd() == ROOT and os.getuid() == 1000 and sys.dont_write_bytecode
            and not sys.flags.optimize, "isolated source/account/bytecode context")
    require(subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
            == "argus/full-projection", "isolated branch")
    require(subprocess.run(["git", "check-ignore", "--quiet", str(OUTPUT)]).returncode == 0,
            "output must be ignored")
    require(OUTPUT.resolve() == OUTPUT and OUTPUT.is_dir()
            and {p.name for p in OUTPUT.iterdir()} == {"command.sh"},
            "one-use output occupied or command sidecar missing")
    command = (OUTPUT / "command.sh").read_text()
    require(command == f"python3 -B -m ace3.model.candidates.{NAME} --execute\n", "command sidecar mismatch")
    mission_path = retained.HANDOFFS / MISSION / "mission.json"
    mission = json.loads(mission_path.read_bytes())
    require(mission["mission_id"] == MISSION and mission["scope"] == "bounded"
            and mission["node_key"] == "value-content-stage13-rmsnorm-integer-contrast"
            and mission["execution_workdir"] == str(ROOT)
            and "review:required" in mission["tags"] and mission["deps"] == [],
            "current native science owner/dependency preflight")
    save("command.json", {
        "mission_id": MISSION, "command_bytes": command, "argv": sys.argv, "cwd": str(ROOT),
        "uid": os.getuid(), "pid": os.getpid(), "executable": sys.executable,
        "timestamp": time.time(), "mission": pin(mission_path), "science_budget": 1,
        "authorization_id": "", "action": "",
        "claim": "exclusive command.json; fixed fresh root; no retries or alternate roots",
        "host_controls": "Host-owned account/role/model/access and locks unchanged",
        "direction": "(new_margin-original_actual_margin)*(original_reference_margin-original_actual_margin)>0",
        "validation": "in-memory compile and focused no-science unittest integration once before science"})
    audit, rows = new_audit(), []
    result = {"mission_id": MISSION, "status": "UNKNOWN", "controls": rows, "audit": audit,
              "science_budget": 1, "science_invocations": 0, "non_admission": True,
              "normal_independent_review": "REQUIRED", "automatic_replay_authorized": False,
              "boundary": BOUNDARY, "flags": closure_source.FLAGS}
    try:
        native.torch.set_num_threads(1)
        tests = retained.module_from(TEST, NAME + "_tests")
        sources = [pin(Path(p)) for p in sorted(project_sources())]
        for path in (SOURCE, TEST):
            compile(path.read_bytes(), str(path), "exec")
        log = io.StringIO()
        checked = unittest.TextTestRunner(stream=log, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(tests))
        save("validation.json", {
            "compiled": [str(SOURCE), str(TEST)], "tests": checked.testsRun,
            "failures": len(checked.failures), "errors": len(checked.errors),
            "skips": len(checked.skipped), "log": log.getvalue(), "source_pins": sources,
            "actual_retained_stage13_entry": "compute_suffix; synthetic native S14-S18; no retained reads"})
        require(checked.wasSuccessful() and checked.testsRun > 0 and not checked.skipped,
                "focused compile/tests failed; no retained-data execution")
        with guarded(OUTPUT, audit):
            began = time.monotonic()
            closure, failure, bridge, ref, refs, operands, tensors, trajectory, records = inputs()
            records.extend([pin(mission_path), pin(OUTPUT / "command.sh")])
            result.update({
                "input_pins": records, "source_pins": sources, "thresholds": closure["thresholds"],
                "historical_failure": closure["historical_failure"],
                "historical_science_status": closure["science_status"],
                "historical_family_status": "REJECTED", "historical_rtz_status": "REJECTED",
                "operand_pins": [retained.digest(v) for v in operands],
                "tensor_pins": {k: retained.digest(v) for k, v in tensors.items()}})
            prepared = []
            require([r["control"] for r in failure["controls"]] == list(CONTROLS), "retained control order")
            for row in failure["controls"]:
                with np.load(io.BytesIO(retained.selection.bound(row["raw_payload"])),
                             allow_pickle=False) as archive:
                    require(len(archive.files) == len(set(archive.files)), "duplicate archive keys")
                    arrays = {k: archive[k] for k in archive.files}
                for array in arrays.values():
                    array.flags.writeable = False
                prior.check_state(arrays)
                prepared.append(arrays)
            result["authentication_seconds"] = time.monotonic() - began
            result["science_invocations"] = 1
            for control, raw, old, classified in zip(
                    CONTROLS, prepared, closure["controls"], bridge["controls"], strict=True):
                payload, row = run_control(control, raw, old, classified, tensors, trajectory,
                                           ref, refs, operands, tests, audit)
                path = OUTPUT / (control + ".npz")
                with path.open("xb") as stream:
                    np.savez(stream, **payload)
                row["payload"] = pin(path)
                rows.append(row)
            for record in [*records, *sources]:
                retained.selection.bound(record)
            require(result["operand_pins"] == [retained.digest(v) for v in operands]
                    and result["tensor_pins"] == {k: retained.digest(v) for k, v in tensors.items()},
                    "authenticated operands changed")
            dispatch_census(audit)
            result["status"] = scientific_status(rows)
            result["protected_input_files_unchanged"] = True
    except Exception as error:
        # Authentication/execution defects consume the attempt, with their failure retained.
        result.update(status="UNKNOWN", error_type=type(error).__name__, error=str(error))
    result["evidence_write_census"] = sorted(p.name for p in OUTPUT.iterdir()) + ["result.json"]
    save("result.json", result)
    print(json.dumps({"status": result["status"], "output": str(OUTPUT),
                      "controls": len(rows), "error": result.get("error")}))
    return 1 if result["status"] == "UNKNOWN" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", required=True, action="store_true")
    parser.parse_args()
    return execute()


if __name__ == "__main__":
    raise SystemExit(main())
