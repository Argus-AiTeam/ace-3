"""Exactly nine non-admitting raw-parent L22/P0 CPU-software consumers."""

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l22_from_l21_coordinate62_suffix_preflight_v1 as preflight
from ace3.model.candidates import diagnose_q24_s16_l12_coordinate62_producer_cone_v3 as producer


ROOT = preflight.ROOT
NAME = "q24_s16_l22_from_l21_coordinate62_suffix_execution_v1"
ID = NAME.replace("_", "-")
MODULE = "ace3.model.candidates.diagnose_" + NAME
SOURCE = ROOT / f"ace3/model/candidates/diagnose_{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE}"
           f" --execute --out build/{OUTPUT.name}")
CONTROLS = preflight.census.CONTROLS
require = preflight.require
record = preflight.matrix.record
prior = producer.prior
EXPECTED_TESTS = 24
PINS = {
    "preflight_source": {
        "path": str(preflight.SOURCE),
        "sha256": "46d8ea4e6df531886e1c98ef9d7fe8440b8487500ed3beb7485f7ac4912994e4",
    },
    "preflight_contract": {
        "path": str(preflight.CONTRACT),
        "sha256": "500c35f99c6b36251dfd9aadc777431575ce206f86f6d4f42c6abf467d6fc7fc",
    },
    "preflight_review": {
        "path": str(preflight.matrix.HANDOFFS / "b9de1724bcd7/round-0001.json"),
        "sha256": "6438d89bbd2539d3c5bf001a6ce46233826065749beac89b0488942bcab5c366",
    },
}
FLAGS = {
    "native_L0_L21_invocations": 0, "native_L23_invocations": 0,
    "rtl_invocations": 0, "hardware_invocations": 0, "gpu_invocations": 0,
    "simulation_invocations": 0, "external_service_invocations": 0,
    "accepted_prefix_replay": False, "original_prefix_replay": False,
    "admission_replay": False, "reference_recomputation": False,
    "reference_reanchoring": False, "candidate_admitted": False,
    "policy_adopted": False, "successor_published": False,
    "strict_FP16_state_claim": False, "new_token_claim": False, "full_model_claim": False,
}
BOUNDARY = (
    "Nine isolated CPU-software L22/P0 counterfactual consumers of the complete raw "
    "failed L21 states, not admission parents or sparse-cut replacements. No L0-L21 "
    "or accepted/original-prefix/admission replay. Q24 residual state is wider than "
    "FP16; G128 asymmetric packed INT4, native GEMM nibble ordering, no qzero plus-one, "
    "FP16 scales/operator boundaries/KV and native S16 RTZ remain unchanged. "
    "Original-input global references, exact thresholds and all historical failures "
    "remain unchanged. No strict-FP16-state W4A16, new-token or full-model admission. "
    "Independent Host Reviewer validation of this new evidence is required."
)
EXPECTED_CONTRACT = {
    "diagnostic_id": ID, "version": 1, "cwd": str(ROOT), "evidence": PINS,
    "controls": list(CONTROLS), "parent_node": [21, 0, 18], "consumer_node": [22, 0],
    "history": [9707], "stages": list(range(19)), "invocations_per_control": 1,
    "state": preflight.STATE, "consumer_kv": preflight.KV,
    "policy_id": preflight.census.POLICY, "profile_id": preflight.census.PROFILE,
    "reference_policy": preflight.census.REFERENCE_POLICY, "excess_budget": "1/8",
    "local_threshold": preflight.EXPECTED_CONTRACT["local_threshold"],
    "output": str(OUTPUT), "fresh_ignored_direct_child": True, "overwrite": False,
    "interface": ["--execute", "--validate", "--out"], "execution_command": COMMAND,
    "expected_tests": EXPECTED_TESTS, "flags": FLAGS, "claim_boundary": BOUNDARY,
    "normal_host_review": "REQUIRED",
}


def same(left, right, message):
    require(json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True), message)


def output_path(value, fresh=True):
    value = Path(value)
    out = value if value.is_absolute() else ROOT / value
    require(out.resolve() == out and out.parent == ROOT / "build"
            and re.fullmatch(NAME + r"_attempt[0-9]{3,}", out.name) is not None
            and int(out.name.rsplit("attempt", 1)[1]) > 0, "noncanonical execution output")
    require(not out.is_symlink(), "symlink execution output")
    require(not out.exists() if fresh else out.is_dir(), "output already exists or is missing")
    ignored = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--", str(out)],
        capture_output=True, check=False)
    require(ignored.returncode == 0, "output is not confirmed ignored: " +
            ignored.stderr.decode(errors="replace"))
    return out


def write(path, document):
    with path.open("x") as stream:
        json.dump(document, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def archive(document):
    with np.load(io.BytesIO(preflight.census.read_bound(document)), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def source_context():
    require(Path.cwd() == ROOT and sys.executable == PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT) and sys.dont_write_bytecode
            and not sys.flags.optimize and Path(__file__).resolve() == SOURCE,
            "disclosed repository-bound unoptimized -B command required")
    require(os.getuid() == 1000, "execution account changed")
    branch = subprocess.run(
        ["git", "-C", str(ROOT), "branch", "--show-current"],
        check=True, capture_output=True, text=True).stdout.strip()
    require(branch == "argus/full-projection", "isolated worktree branch changed")
    origins = {}
    for name, module in sorted(sys.modules.items()):
        if name == "ace3" or name.startswith("ace3."):
            path = getattr(module, "__file__", None)
            if path:
                path = Path(path)
                require(path.is_relative_to(ROOT) and path.resolve() == path,
                        "imported source outside isolated repository")
                origins[name] = record(path)
    origins[MODULE] = record(SOURCE)
    origins["focused_test"] = record(TEST)
    origins["execution_contract"] = record(CONTRACT)
    return origins


def authenticate():
    same(json.loads(CONTRACT.read_bytes()), EXPECTED_CONTRACT, "execution contract mismatch")
    for pin in PINS.values():
        preflight.census.read_bound(pin)
    preflight.matrix.check_review(
        json.loads(preflight.census.read_bound(PINS["preflight_review"])), "b9de1724bcd7", 1)
    evidence = preflight.authenticate()
    require(evidence["summary"]["status"] == "PREFLIGHT_READY",
            "BLOCKED_MISSING_L22_REFERENCE")
    for name, pin in evidence["retained"]["result"]["origins_after_execution"].items():
        module = sys.modules.get(name)
        if module is not None and "path" in pin:
            require(Path(module.__file__).resolve() == Path(pin["path"]),
                    "reviewed native source origin changed")
            same(record(Path(pin["path"])), pin, "reviewed native source changed")
    return evidence


def parents_from(evidence):
    return {label: {key: state[field].copy() for key, field in preflight.STATE["fields"].items()}
            for label, state in evidence["states"].items()}


@contextmanager
def suffix_only(parents, audit):
    require(list(parents) == list(CONTROLS), "missing/reordered native control schedule")
    raw = producer.native.candidate._stages

    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("prefix, admission, external or unbounded native execution forbidden")

    def guarded(tensors, layer, state, arrays):
        index = len(audit["native_controls"])
        require(type(layer) is int and layer == 22 and index < 9,
                "native dispatch outside nine L22/P0 controls")
        producer.paired.same_arrays(state, parents[CONTROLS[index]])
        prior.retained.verify_parent(state, state)
        audit["native_controls"].append(CONTROLS[index])
        yield from raw(tensors, layer, state, arrays)

    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates.") and Path(
                    getattr(module, "__file__", SOURCE)).resolve() != SOURCE:
                for attribute in ("stages", "continuation_stages", "native_layer", "run",
                                  "_stages", "execute_layers", "execute_layer",
                                  "diagnose", "original_branches"):
                    if callable(getattr(module, attribute, None)):
                        replacement = (guarded if module is producer.native.candidate
                                       and attribute == "_stages" else forbidden)
                        stack.enter_context(patch.object(module, attribute, replacement))
        stack.enter_context(patch.object(subprocess, "Popen", forbidden))
        stack.enter_context(patch.object(os, "system", forbidden))
        yield


def stage_report(stage, arrays, state, trajectory, reference, expected):
    report = prior.gates.evaluate_decoder_stage(
        stage=stage, actual=arrays[f"stage{stage:02d}"],
        reference=trajectory[f"stage{stage:02d}"], policy=prior.gates.POLICY_ID,
        local_reference=expected, reference_binary64=reference if stage == 18 else None)
    require(report["status"] in ("PASS", "FAIL"), "missing mandatory L22 gate")
    report.update(node=[22, 0, stage], residual_state_lineage="PASS", kv_lineage="PASS")
    return report


def expected_stage(stage, arrays, state, tensors):
    expected = prior.retained.check_stage_state(stage, arrays, state)
    if stage < 18 and stage != 12:
        expected = prior.local.local_reference(
            stage, {key: arrays[key].copy() for key in prior.local.OPERANDS[stage]}, tensors, 22)
    return expected


def execute_layer(tensors, state, trajectory, reference):
    arrays, locals_, reports = {}, {}, []
    timings = {"native_seconds": 0.0, "oracle_gate_seconds": 0.0}
    stages = producer.native.candidate._stages(tensors, 22, state, arrays)
    while True:
        began = time.monotonic()
        try:
            stage = next(stages)
        except StopIteration:
            break
        timings["native_seconds"] += time.monotonic() - began
        require(type(stage) is int and stage == len(reports) and stage < 19,
                "out-of-order native L22 stage")
        began = time.monotonic()
        expected = expected_stage(stage, arrays, state, tensors)
        if stage < 18:
            locals_[f"stage{stage:02d}"] = expected
        reports.append(stage_report(stage, arrays, state, trajectory, reference, expected))
        timings["oracle_gate_seconds"] += time.monotonic() - began
    require(len(reports) == 19, "incomplete native L22 stage sequence")
    require(np.array_equal(prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                           arrays["stage16"]), "native S16 RTZ operand mismatch")
    return arrays, locals_, reports, timings


def load_inputs(evidence):
    binding = evidence["summary"]["L22_original_reference"]
    extension = json.loads(preflight.census.read_bound(binding["manifest"]))
    item = binding["reference"]
    checkpoint = extension["checkpoint"]
    path = Path(checkpoint["path"])
    require(path.is_relative_to(ROOT / "build") and path.resolve() == path,
            "checkpoint outside isolated build")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    require(digest.hexdigest() == checkpoint["sha256"] and size == checkpoint["bytes"],
            "official checkpoint hash mismatch")
    with producer.legacy.safe_open(str(path), framework="numpy") as model:
        tensors = {name: np.ascontiguousarray(model.get_tensor(name))
                   for name in prior.local.tensor_shapes(22)}
    prior.local.authenticate_tensors(tensors, item["canonical"], 22)
    trajectory = archive(item["fp16"])
    reference = np.load(io.BytesIO(preflight.census.read_bound(item["binary64"])),
                        allow_pickle=False)
    require(reference.shape == (896,) and reference.dtype.str == "<f8"
            and np.all(np.isfinite(reference)), "invalid original-input L22 reference")
    for stage, size in prior.local.SIZES.items():
        prior.local.finite_words(trajectory[f"stage{stage:02d}"], (size,))
    return tensors, trajectory, reference, checkpoint


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(record(path))
    spec = importlib.util.spec_from_file_location("l22_execution_focused_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader missing")
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    tests.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    require(suite.countTestCases() == EXPECTED_TESTS, "focused test collection changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.testsRun == EXPECTED_TESTS and result.wasSuccessful() and not result.skipped,
            "focused tests failed, errored or skipped")
    return {"compiled": compiled, "collected": suite.countTestCases(), "executed": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "native_layer_invocations": 0}


def check_result(result, evidence, out):
    same(result["preflight"], evidence["summary"], "reviewed preflight lineage changed")
    same(result["preflight_pins"], PINS, "preflight pins changed")
    same(result["flags"], FLAGS, "execution boundary changed")
    same({key: result[key] for key in FLAGS}, FLAGS, "top-level execution boundary changed")
    require(result["diagnostic_id"] == ID and result["status"] == "DIAGNOSED"
            and result["output"] == str(out) and result["output_created_exclusively"] is True
            and result["native_layer_invocations"] == result["native_L22_P0_invocations"] == 9
            and result["audit"] == {"native_controls": list(CONTROLS), "forbidden_calls": 0},
            "execution identity/count/output mismatch")
    require([row["control"] for row in result["controls"]] == list(CONTROLS),
            "missing/reordered L22 results")
    for row, plan in zip(result["controls"], evidence["summary"]["controls"], strict=True):
        same(row["retained_L21"], plan["retained_L21"], "raw L21 failure changed")
        same(row["parent_archive"], plan["parent_archive"], "raw L21 parent substituted")
        require(len(row["mandatory_statuses"]) == 19
                and set(row["mandatory_statuses"]) <= {"PASS", "FAIL"}
                and row["L22_status"] == ("PASS" if all(
                    s == "PASS" for s in row["mandatory_statuses"]) else "FAIL")
                and row["candidate_admitted"] is False
                and row["policy_adopted"] is False and row["successor_published"] is False
                and row["source_operand_state_KV_RTZ_checks"] == "PASS",
                "invalid L22 result boundary")
    require(result["claim_boundary"] == BOUNDARY and result["normal_host_review"] == "REQUIRED",
            "result claim changed")


def validate_evidence(out, evidence=None, inputs=None):
    output_path(out, fresh=False)
    evidence = authenticate() if evidence is None else evidence
    inputs = load_inputs(evidence) if inputs is None else inputs
    tensors, trajectory, reference, checkpoint = inputs
    result = json.loads((out / "result.json").read_bytes())
    check_result(result, evidence, out)
    same(result["origins"], source_context(), "source changed during execution")
    same(result["checkpoint"], checkpoint, "checkpoint identity changed")
    tests = result["tests"]
    require(tests["collected"] == tests["executed"] == EXPECTED_TESTS
            and all(tests[key] == 0 for key in (
                "failures", "errors", "skipped", "native_layer_invocations")),
            "invalid compile/test receipt")
    same(tests["compiled"], [record(SOURCE), record(TEST)], "compiled source changed")
    expected_paths = {str(out / "command.json")}
    for label in CONTROLS:
        expected_paths.update(str(out / f"{label}_L22{suffix}") for suffix in (
            ".npz", "_parent.npz", "_local_references.npz", "_gates.json"))
    expected_paths.add(str(out / "interventions.json"))
    records = result["artifacts"]
    require(len(records) == len(expected_paths)
            and {r["path"] for r in records} == expected_paths, "artifact set changed")
    for item in records:
        preflight.census.read_bound(item)
    bound = {r["path"]: r for r in records}
    same(json.loads(preflight.census.read_bound(bound[str(out / "interventions.json")])),
         result["controls"], "intervention rows changed")
    command = json.loads(preflight.census.read_bound(bound[str(out / "command.json")]))
    require(command["cwd"] == str(ROOT) and command["executable"] == PYTHON
            and command["uid"] == 1000 and command["dont_write_bytecode"] is True
            and command["PYTHONPATH"] == str(ROOT)
            and command["output_existed_before_creation"] is False
            and command["command"] == COMMAND, "execution context changed")
    parents = parents_from(evidence)
    audit = {"native_controls": [], "forbidden_calls": 0}
    with suffix_only(parents, audit):
        for row in result["controls"]:
            label = row["control"]
            get = lambda suffix: bound[str(out / f"{label}_L22{suffix}")]
            state = archive(get("_parent.npz"))
            producer.paired.same_arrays(state, parents[label])
            arrays, locals_ = archive(get(".npz")), archive(get("_local_references.npz"))
            reports = json.loads(preflight.census.read_bound(get("_gates.json")))
            same(row["gates"], get("_gates.json"), "report binding changed")
            require(len(reports) == 19 and set(locals_) == {
                f"stage{s:02d}" for s in range(18)}, "incomplete stage/local-reference archive")
            for stage in range(19):
                expected = expected_stage(stage, arrays, state, tensors)
                if stage < 18:
                    require(np.array_equal(expected, locals_[f"stage{stage:02d}"]),
                            "independent local operand reference changed")
                same(reports[stage], stage_report(
                    stage, arrays, state, trajectory, reference, expected), "stored gate changed")
            require(np.array_equal(prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                                   arrays["stage16"]), "stored S16 RTZ operand mismatch")
            same(row["mandatory_statuses"], [r["status"] for r in reports], "stage status changed")
            same(row["S18_failure_indices"], [r["index"] for r in
                 reports[18]["binary64_v1"]["failures"]], "global failure indices changed")
    require(audit == {"native_controls": [], "forbidden_calls": 0}, "validation dispatched native")
    return {
        "diagnostic_id": ID, "status": "VALIDATED_EXECUTION_EVIDENCE",
        "result": record(out / "result.json"), "controls": list(CONTROLS),
        "stage_reports_verified": 171, "native_layer_invocations": 0,
        "forbidden_calls": 0, "source_operand_state_KV_lineage_checks": "PASS",
        "original_input_L22_reference_binding": evidence["summary"]["L22_original_reference"],
        "tests": tests, "flags": FLAGS, "normal_host_review": "REQUIRED",
        **FLAGS,
    }


def execute(out):
    started = time.monotonic()
    out = output_path(out)
    evidence = authenticate()
    tests = focused_tests(evidence)
    origins = source_context()
    inputs = load_inputs(evidence)
    tensors, trajectory, reference, checkpoint = inputs
    parents = parents_from(evidence)
    producer.legacy.torch.set_num_threads(1)
    require(str(producer.legacy.torch.tensor(0).device) == "cpu", "non-CPU default device")
    output_path(out)
    out.mkdir(exist_ok=False)
    command = {
        "command": COMMAND, "cwd": str(ROOT), "executable": sys.executable,
        "argv": sys.argv, "uid": os.getuid(), "PYTHONPATH": os.environ["PYTHONPATH"],
        "dont_write_bytecode": sys.dont_write_bytecode, "output_existed_before_creation": False,
        "host_managed_controls": "unchanged role/model/account/budget/access/locks; no service calls",
    }
    write(out / "command.json", command)
    artifacts, rows = [record(out / "command.json")], []
    audit = {"native_controls": [], "forbidden_calls": 0}
    try:
        with suffix_only(parents, audit):
            for plan in evidence["summary"]["controls"]:
                label = plan["control"]
                arrays, locals_, reports, timings = execute_layer(
                    tensors, parents[label], trajectory, reference)
                for suffix, values in (("_parent.npz", parents[label]), (".npz", arrays),
                                       ("_local_references.npz", locals_)):
                    path = out / f"{label}_L22{suffix}"
                    with path.open("xb") as stream:
                        np.savez(stream, **values)
                    artifacts.append(record(path))
                path = out / f"{label}_L22_gates.json"
                write(path, reports)
                artifacts.append(record(path))
                statuses = [r["status"] for r in reports]
                row = {
                    "control": label, "parent_archive": plan["parent_archive"],
                    "retained_L21": plan["retained_L21"], "mandatory_statuses": statuses,
                    "L22_status": "PASS" if all(s == "PASS" for s in statuses) else "FAIL",
                    "S18_failure_indices": [r["index"] for r in reports[18]["binary64_v1"]["failures"]],
                    "source_operand_state_KV_RTZ_checks": "PASS", "gates": record(path),
                    "candidate_admitted": False, "policy_adopted": False,
                    "successor_published": False, "timing_seconds": timings,
                }
                rows.append(row)
                print(json.dumps({"control": label, "L22_status": row["L22_status"],
                                  "S18_failure_indices": row["S18_failure_indices"]}), flush=True)
        require(audit == {"native_controls": list(CONTROLS), "forbidden_calls": 0},
                "incomplete execution schedule")
        same(source_context(), origins, "source changed during suffix")
        write(out / "interventions.json", rows)
        artifacts.append(record(out / "interventions.json"))
        result = {
            "diagnostic_id": ID, "status": "DIAGNOSED", "preflight_pins": PINS,
            "preflight": evidence["summary"], "controls": rows, "audit": audit,
            "native_layer_invocations": len(audit["native_controls"]),
            "native_L22_P0_invocations": len(audit["native_controls"]),
            "flags": FLAGS, **FLAGS, "origins": origins, "checkpoint": checkpoint,
            "artifacts": artifacts, "tests": tests, "output": str(out),
            "output_created_exclusively": True, "normal_host_review": "REQUIRED",
            "claim_boundary": BOUNDARY, "total_seconds_before_validation": time.monotonic() - started,
        }
        write(out / "result.json", result)
        validation = validate_evidence(out)
        write(out / "validation.json", validation)
    except Exception as error:
        write(out / "failure.json", {
            "status": "FAILED", "error_type": type(error).__name__, "error": str(error),
            "audit": audit, "completed_controls": [r["control"] for r in rows], "flags": FLAGS,
        })
        raise
    print(json.dumps({"status": validation["status"], "output": str(out),
                      "native_L22_P0_invocations": len(audit["native_controls"]),
                      "stage_reports_verified": validation["stage_reports_verified"]}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    source_context()
    if args.execute:
        execute(args.out)
    else:
        out = output_path(args.out, fresh=False)
        print(json.dumps(validate_evidence(out), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
