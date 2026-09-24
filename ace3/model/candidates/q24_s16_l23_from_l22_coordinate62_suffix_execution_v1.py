"""Nine raw failed-parent L23/P0 CPU consumers, never admission or prefix replay.

--execute compiles both new files, runs non-native tests, exclusively creates
one ignored build output, and validates the stored numerical evidence once.
--validate repeats the non-native checks for independent Host review only.
Q24 residual state is wider than FP16; INT4 weights and FP16 operators/KV
are unchanged. Numerical results do not authorize a successor or token.
"""

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

from ace3.model.candidates import diagnose_q24_s16_l23_from_l22_coordinate62_suffix_preflight_v1 as preflight
from ace3.model.candidates import diagnose_q24_s16_l22_from_l21_coordinate62_suffix_execution_v1 as previous


ROOT = preflight.ROOT
NAME = "q24_s16_l23_from_l22_coordinate62_suffix_execution_v1"
ID = NAME.replace("_", "-")
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
PYTHON = preflight.PYTHON
CONTROLS = preflight.CONTROLS
EXPECTED_TESTS = 20
REVIEW_MISSION = "f013e4a7c490"
require, record, same = preflight.require, preflight.record, preflight.same
write, archive = previous.write, previous.archive
producer, prior = previous.producer, previous.prior
PINS = {
    "preflight_source": {
        "path": str(preflight.SOURCE),
        "sha256": "ff6b4fd49241cdfa4f0215ec1c76b946a8309bb3d09d64b4faf4162ba58d00f6",
    },
    "preflight_contract": {
        "path": str(preflight.CONTRACT),
        "sha256": "36d41eeafcbfad1131d902f706b5cb25e40d8e999cc3b13132a25b3dd5877891",
    },
    "preflight_review": {
        "path": str(preflight.HANDOFFS / REVIEW_MISSION / "round-0001.json"),
        "sha256": "4c2250b3db2391ecf37ca0fae6b963c1125b130241ec3b22e353248c994e5570",
    },
}
FLAGS = {
    "native_L0_L22_invocations": 0, "prefix_invocations": 0, "admission_invocations": 0,
    "rtl_invocations": 0, "hardware_invocations": 0, "gpu_invocations": 0,
    "simulation_invocations": 0, "external_service_invocations": 0,
    "accepted_prefix_replay": False, "original_prefix_replay": False,
    "admission_replay": False, "reference_recomputation": False,
    "reference_reanchoring": False, "candidate_admitted": False,
    "policy_adopted": False, "successor_published": False,
    "strict_FP16_state_claim": False, "new_token_claim": False, "full_model_claim": False,
}
BOUNDARY = (
    "Exactly nine isolated native S16 RTZ Q24 CPU-software L23/P0 S0-S18 consumers "
    "of complete raw failed L22 I/Z/H states, not admission parents or sparse-cut "
    "replacements. All L21/L22 failures and independently propagated original-input "
    "global references remain unchanged. No L0-L22 prefix/admission replay, RTL, "
    "hardware, GPU or simulation. Q24 residual state is wider than FP16; native G128 "
    "asymmetric packed INT4 GEMM nibble ordering, no qzero plus-one adjustment, FP16 "
    "scales/operator boundaries and own empty P0 FP16 KV remain unchanged. Exact "
    "source/operand/state/KV/lineage gates and thresholds remain mandatory. No "
    "strict-FP16-state W4A16, new-token or full-model admission claim. Independent "
    "Host Reviewer validation of this new execution evidence is required."
)


def command_for(out):
    return (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE}"
            f" --execute --out build/{out.name}")


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
    origins[MODULE], origins["focused_test"] = record(SOURCE), record(TEST)
    return origins


def authenticate():
    for pin in PINS.values():
        preflight.read_bound(pin)
    review = json.loads(preflight.read_bound(PINS["preflight_review"]))
    preflight.parent.matrix.check_review(review, REVIEW_MISSION, 1)
    latest = json.loads((preflight.HANDOFFS / REVIEW_MISSION / "latest.json").read_bytes())
    require(latest["kind"] == "handoff_ref"
            and latest["handoff"]["path"] == PINS["preflight_review"]["path"],
            "preparation review is no longer terminal")
    rows = [json.loads(line) for line in preflight.BACKLOG.read_text().splitlines()
            if line.strip()]
    rows = [row for row in rows if row["id"] == REVIEW_MISSION]
    require(len(rows) == 1 and rows[0]["status"] == "done"
            and rows[0]["outcome"]["review_status"] == "done"
            and rows[0]["finished_ts"] >= review["created_at"],
            "preparation review lacks terminal backlog corroboration")
    evidence = preflight.authenticate()
    require(evidence["summary"]["status"] == "PREFLIGHT_READY",
            "BLOCKED_MISSING_L23_REFERENCE")
    for name, pin in evidence["result"]["origins"].items():
        module = sys.modules.get(name)
        if module is not None:
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
        raise RuntimeError("prefix, admission, external or unbounded execution forbidden")

    def guarded(tensors, layer, state, arrays):
        index = len(audit["native_controls"])
        require(type(layer) is int and layer == 23 and index < 9,
                "native dispatch outside nine L23/P0 controls")
        producer.paired.same_arrays(state, parents[CONTROLS[index]])
        prior.retained.verify_parent(state, state)
        require(arrays == {}, "consumer stage/cache arrays must start empty")
        audit["native_controls"].append(CONTROLS[index])
        yield from raw(tensors, layer, state, arrays)

    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates.") and Path(
                    getattr(module, "__file__", SOURCE)).resolve() != SOURCE:
                for attribute in ("stages", "continuation_stages", "native_layer", "run",
                                  "_stages", "execute_layers", "execute_layer",
                                  "diagnose", "original_branches", "execute"):
                    if callable(getattr(module, attribute, None)):
                        replacement = (guarded if module is producer.native.candidate
                                       and attribute == "_stages" else forbidden)
                        stack.enter_context(patch.object(module, attribute, replacement))
        stack.enter_context(patch.object(subprocess, "Popen", forbidden))
        stack.enter_context(patch.object(os, "system", forbidden))
        yield


def expected_stage(stage, arrays, state, tensors):
    expected = prior.retained.check_stage_state(stage, arrays, state)
    if stage < 18 and stage != 12:
        expected = prior.local.local_reference(
            stage, {key: arrays[key].copy() for key in prior.local.OPERANDS[stage]}, tensors, 23)
    return expected


def stage_report(stage, arrays, trajectory, reference, expected):
    report = prior.gates.evaluate_decoder_stage(
        stage=stage, actual=arrays[f"stage{stage:02d}"],
        reference=trajectory[f"stage{stage:02d}"], policy=prior.gates.POLICY_ID,
        local_reference=expected, reference_binary64=reference if stage == 18 else None)
    require(report["status"] in ("PASS", "FAIL"), "missing mandatory L23 gate")
    report.update(node=[23, 0, stage], residual_state_lineage="PASS", kv_lineage="PASS")
    return report


def execute_layer(tensors, state, trajectory, reference):
    arrays, locals_, reports = {}, {}, []
    timings = {"native_seconds": 0.0, "oracle_gate_seconds": 0.0}
    stages = producer.native.candidate._stages(tensors, 23, state, arrays)
    while True:
        began = time.monotonic()
        try:
            stage = next(stages)
        except StopIteration:
            break
        timings["native_seconds"] += time.monotonic() - began
        require(type(stage) is int and stage == len(reports) and stage < 19,
                "out-of-order native L23 stage")
        began = time.monotonic()
        expected = expected_stage(stage, arrays, state, tensors)
        if stage < 18:
            locals_[f"stage{stage:02d}"] = expected
        reports.append(stage_report(stage, arrays, trajectory, reference, expected))
        timings["oracle_gate_seconds"] += time.monotonic() - began
    require(len(reports) == 19, "incomplete native L23 stage sequence")
    require(np.array_equal(prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                           arrays["stage16"]), "native S16 RTZ operand mismatch")
    return arrays, locals_, reports, timings


def load_inputs(evidence):
    binding = evidence["summary"]["L23_original_reference"]
    extension = json.loads(preflight.read_bound(binding["manifest"]))
    item, checkpoint = binding["reference"], extension["checkpoint"]
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
                   for name in prior.local.tensor_shapes(23)}
    prior.local.authenticate_tensors(tensors, item["canonical"], 23)
    trajectory = archive(item["fp16"])
    reference = np.load(io.BytesIO(preflight.read_bound(item["binary64"])), allow_pickle=False)
    require(reference.shape == (896,) and reference.dtype.str == "<f8"
            and np.all(np.isfinite(reference)), "invalid original-input L23 reference")
    for stage, size in prior.local.SIZES.items():
        prior.local.finite_words(trajectory[f"stage{stage:02d}"], (size,))
    return tensors, trajectory, reference, checkpoint


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(record(path))
    spec = importlib.util.spec_from_file_location("l23_execution_focused_tests", TEST)
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
            and result["native_layer_invocations"] == result["native_L23_P0_invocations"] == 9
            and result["audit"] == {"native_controls": list(CONTROLS), "forbidden_calls": 0},
            "execution identity/count/output mismatch")
    require([row["control"] for row in result["controls"]] == list(CONTROLS),
            "missing/reordered L23 results")
    for row, plan in zip(result["controls"], evidence["summary"]["controls"], strict=True):
        for key in ("retained_L21", "retained_L22", "L22_failure", "parent_archive"):
            same(row[key], plan[key], "raw parent history/lineage changed: " + key)
        statuses = row["mandatory_statuses"]
        require(len(statuses) == 19 and set(statuses) <= {"PASS", "FAIL"}
                and row["L23_status"] == ("PASS" if all(s == "PASS" for s in statuses) else "FAIL")
                and row["source_operand_state_KV_RTZ_checks"] == "PASS"
                and all(row[key] is False for key in (
                    "candidate_admitted", "policy_adopted", "successor_published")),
                "invalid L23 result boundary")
    require(result["claim_boundary"] == BOUNDARY and result["normal_host_review"] == "REQUIRED",
            "result claim changed")


def validate_evidence(out, evidence, inputs):
    output_path(out, fresh=False)
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
    expected_paths = {str(out / name) for name in ("command.json", "interventions.json")}
    expected_paths.update(str(out / f"{label}_L23{suffix}") for label in CONTROLS
                          for suffix in (".npz", "_parent.npz", "_local_references.npz",
                                         "_gates.json"))
    records = result["artifacts"]
    require(len(records) == len(expected_paths)
            and {r["path"] for r in records} == expected_paths, "artifact set changed")
    for item in records:
        preflight.read_bound(item)
    bound = {r["path"]: r for r in records}
    same(json.loads(preflight.read_bound(bound[str(out / "interventions.json")])),
         result["controls"], "intervention rows changed")
    command = json.loads(preflight.read_bound(bound[str(out / "command.json")]))
    require(command["command"] == command_for(out) and command["cwd"] == str(ROOT)
            and command["executable"] == PYTHON and command["uid"] == 1000
            and command["dont_write_bytecode"] is True and command["PYTHONPATH"] == str(ROOT)
            and command["output_existed_before_creation"] is False,
            "execution context changed")
    parents, audit = parents_from(evidence), {"native_controls": [], "forbidden_calls": 0}
    with suffix_only(parents, audit):
        for row in result["controls"]:
            label = row["control"]
            get = lambda suffix: bound[str(out / f"{label}_L23{suffix}")]
            state = archive(get("_parent.npz"))
            producer.paired.same_arrays(state, parents[label])
            prior.retained.verify_parent(state, parents[label])
            arrays, locals_ = archive(get(".npz")), archive(get("_local_references.npz"))
            reports = json.loads(preflight.read_bound(get("_gates.json")))
            same(row["gates"], get("_gates.json"), "report binding changed")
            require(len(reports) == 19 and set(locals_) == {
                f"stage{s:02d}" for s in range(18)}, "incomplete stage/local-reference archive")
            for stage in range(19):
                expected = expected_stage(stage, arrays, state, tensors)
                if stage < 18:
                    require(np.array_equal(expected, locals_[f"stage{stage:02d}"]),
                            "independent local operand reference changed")
                same(reports[stage], stage_report(stage, arrays, trajectory, reference, expected),
                     "stored gate changed")
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
        "original_input_L23_reference_binding": evidence["summary"]["L23_original_reference"],
        "tests": tests, "flags": FLAGS, "normal_host_review": "REQUIRED", **FLAGS,
    }


def execute(out):
    started = time.monotonic()
    out = output_path(out)
    # The sidecar precedes every result-bearing check; exclusive creation also
    # makes a partial attempt permanently occupied rather than retryable.
    out.mkdir(exist_ok=False)
    write(out / "command.json", {
        "command": command_for(out), "cwd": str(ROOT), "executable": sys.executable,
        "argv": sys.argv, "uid": os.getuid(), "PYTHONPATH": os.environ["PYTHONPATH"],
        "dont_write_bytecode": sys.dont_write_bytecode, "output_existed_before_creation": False,
        "host_managed_controls": "unchanged role/model/account/budget/access/locks; no service calls",
    })
    audit, rows = {"native_controls": [], "forbidden_calls": 0}, []
    try:
        evidence = authenticate()
        tests = focused_tests(evidence)
        origins = source_context()
        inputs = load_inputs(evidence)
        tensors, trajectory, reference, checkpoint = inputs
        parents = parents_from(evidence)
        producer.legacy.torch.set_num_threads(1)
        require(str(producer.legacy.torch.tensor(0).device) == "cpu", "non-CPU default device")
        artifacts = [record(out / "command.json")]
        with suffix_only(parents, audit):
            for plan in evidence["summary"]["controls"]:
                label = plan["control"]
                arrays, locals_, reports, timings = execute_layer(
                    tensors, parents[label], trajectory, reference)
                for suffix, values in (("_parent.npz", parents[label]), (".npz", arrays),
                                       ("_local_references.npz", locals_)):
                    path = out / f"{label}_L23{suffix}"
                    with path.open("xb") as stream:
                        np.savez(stream, **values)
                    artifacts.append(record(path))
                path = out / f"{label}_L23_gates.json"
                write(path, reports)
                artifacts.append(record(path))
                statuses = [r["status"] for r in reports]
                row = {
                    **{key: plan[key] for key in (
                        "control", "parent_archive", "retained_L21", "retained_L22", "L22_failure")},
                    "mandatory_statuses": statuses,
                    "L23_status": "PASS" if all(s == "PASS" for s in statuses) else "FAIL",
                    "S18_failure_indices": [r["index"] for r in reports[18]["binary64_v1"]["failures"]],
                    "source_operand_state_KV_RTZ_checks": "PASS", "gates": record(path),
                    "candidate_admitted": False, "policy_adopted": False,
                    "successor_published": False, "timing_seconds": timings,
                }
                rows.append(row)
                print(json.dumps({"control": label, "L23_status": row["L23_status"],
                                  "S18_failure_indices": row["S18_failure_indices"]}), flush=True)
        require(audit == {"native_controls": list(CONTROLS), "forbidden_calls": 0},
                "incomplete execution schedule")
        same(source_context(), origins, "source changed during suffix")
        write(out / "interventions.json", rows)
        artifacts.append(record(out / "interventions.json"))
        write(out / "result.json", {
            "diagnostic_id": ID, "status": "DIAGNOSED", "preflight_pins": PINS,
            "preflight": evidence["summary"], "controls": rows, "audit": audit,
            "native_layer_invocations": len(audit["native_controls"]),
            "native_L23_P0_invocations": len(audit["native_controls"]),
            "flags": FLAGS, **FLAGS, "origins": origins, "checkpoint": checkpoint,
            "artifacts": artifacts, "tests": tests, "output": str(out),
            "output_created_exclusively": True, "normal_host_review": "REQUIRED",
            "claim_boundary": BOUNDARY, "total_seconds_before_validation": time.monotonic() - started,
        })
        validation = validate_evidence(out, evidence, inputs)
        write(out / "validation.json", validation)
    except Exception as error:
        write(out / "failure.json", {
            "status": "FAILED", "error_type": type(error).__name__, "error": str(error),
            "audit": audit, "completed_controls": [r["control"] for r in rows], "flags": FLAGS,
        })
        raise
    print(json.dumps({"status": validation["status"], "output": str(out),
                      "native_L23_P0_invocations": len(audit["native_controls"]),
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
        evidence = authenticate()
        focused_tests(evidence)
        print(json.dumps(validate_evidence(out, evidence, load_inputs(evidence)), sort_keys=True),
              flush=True)


if __name__ == "__main__":
    main()
