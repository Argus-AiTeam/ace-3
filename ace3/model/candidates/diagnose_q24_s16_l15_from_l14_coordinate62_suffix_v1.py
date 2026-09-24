"""Isolated non-admission L15/P0 suffix from frozen reviewed L14 outputs."""

import argparse
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l14_from_l12_v3_coordinate62_suffix_v1 as parent


producer = parent.producer
ROOT = parent.ROOT
NAME = "q24_s16_l15_from_l14_coordinate62_suffix_v1"
ID = "ace3-q24-s16-l15-from-l14-coordinate62-suffix-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
SELECTED = ROOT / "build/q24_s16_l14_from_l12_v3_coordinate62_suffix_v1_c55e934b5d70_attempt001"
SELECTED_SHA = "a84511c4d558929d9e434d6762e0fb8509ab0e1a1288e53d7c5afb3b0b33e8dd"
PARENT_REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/c55e934b5d70/round-0001.json")
SOURCE_PINS = {
    "ace3/model/candidates/diagnose_q24_s16_l14_from_l12_v3_coordinate62_suffix_v1.py":
        "37b918b3df144b89927ce8f89aa4b6af99a703633d956f4109681ff254d725f4",
    "ace3/contracts/candidates/q24_s16_l14_from_l12_v3_coordinate62_suffix_v1.json":
        "ef0716ea6d49c4a418313b4deda41063fe03fc82e329ca59fa42df889222392b",
}
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 "
           f"/home/argustest/miniconda3/bin/python -B -m {MODULE} --check")
OUTPUT_PREFIX = NAME + "_"
EXECUTION_COMMAND = COMMAND.removesuffix("--check") + (
    f"--execute --out build/{OUTPUT_PREFIX}fresh")
EXPECTED_TESTS = 40
CONTROLS = parent.CONTROLS
FLAGS = {
    **parent.FLAGS, "native_L12_invocations": 0, "native_L13_invocations": 0,
    "native_L14_invocations": 0, "native_L15_P0_invocations": 0,
}
require, record, equal = parent.require, parent.record, parent.equal
np = parent.np
consumer_parent = parent.consumer_parent


def execution_bounds():
    return {
        **FLAGS, "native_layer_invocations": 13, "native_L15_P0_invocations": 13,
        "position": 0, "layers": [15], "stages": list(range(19)),
        "schedule": [{"control": label, "layer": 15, "position": 0} for label in CONTROLS],
        "invocations_per_control": 1, "output_parent": str(ROOT / "build"),
        "output_name_prefix": OUTPUT_PREFIX, "fresh_directory_required": True,
        "reference_recomputation": False, "accepted_prefix_replay": False,
    }


def history_records():
    records = parent.history_records()
    for relative, digest in SOURCE_PINS.items():
        path = ROOT / relative
        require(path.resolve() == path, "frozen L14 source origin mismatch")
        item = record(path)
        require(item["sha256"] == digest, f"frozen L14 source changed: {relative}")
        records.append(item)
    return records


def check_contract(contract):
    history_records()
    base = json.loads(parent.CONTRACT.read_text())
    parent.check_contract(base)
    expected = {
        "diagnostic_id": ID, "inherits": str(parent.CONTRACT.relative_to(ROOT)),
        "inherits_sha256": SOURCE_PINS[str(parent.CONTRACT.relative_to(ROOT))],
        "selected_result": str(SELECTED / "result.json"),
        "selected_result_sha256": SELECTED_SHA, "parent_review": str(PARENT_REVIEW),
        "policy_id": base["policy_id"], "thresholds": base["thresholds"],
        "controls": list(CONTROLS), "retained_L14_gate_passing_controls": list(CONTROLS),
        "consumer": {"layer": 15, "position": 0, "stages": list(range(19))},
        "interface": ["--check", "--execute", "--out"], "cwd": str(ROOT),
        "command": COMMAND, "execution_command": EXECUTION_COMMAND,
        "execution_bounds": execution_bounds(), "expected_tests": EXPECTED_TESTS,
        **FLAGS, "native_control_dispatch_authorized": True,
        "accepted_prefix_replay": False, "scientific_result_claim": False,
        "attribution_question": "Do the 13 frozen reviewed L14 controls, including the "
        "actual baseline, retain their all-stage gate acceptance at L15/P0?",
        "layer_necessity": "Exactly one complete L15/P0 S0-S18 consumer per frozen "
        "L14 control, preserving each control's distinct lineage and the actual "
        "baseline. L0-L14 outputs are authenticated only, never recomputed.",
        "state_boundary": "Complete frozen L14 output I/Z/H is the immediate L15 "
        "parent. Each control has own-empty-P0 FP16 KV, never L14 KV. Q24 residual "
        "state is wider than FP16; native G128 asymmetric packed INT4 weights, "
        "FP16 scales/operator boundaries/KV and S16 RTZ remain unchanged.",
        "reference_boundary": "Use only the frozen independently propagated "
        "original-input L15 binary64 global reference and FP16 trajectory, linked "
        "through original L14 references. No recomputation or control-derived reference.",
        "claim_boundary": "Isolated versioned CPU-software non-admission surface. "
        "--check authenticates frozen evidence and executes only focused mocked-dispatch "
        "tests; it does not produce an L15 numerical result. --execute is bounded to "
        "13 L15/P0 invocations into a fresh versioned build directory. Independent "
        "Host review is required. No repair, unique cause, admission, strict-FP16-state "
        "W4A16, new-token or full-model PASS follows. Historical failures and accepted "
        "evidence remain unchanged; no successor is published or queued.",
    }
    equal(contract, expected, "L15 bounded contract mismatch")


def check_result(result):
    expected = {
        "diagnostic_id": parent.ID, "status": "DIAGNOSED", "node": [14, 0, 18],
        **parent.FLAGS, "native_layer_invocations": 13, "native_L14_P0_invocations": 13,
        "native_L12_invocations": 0, "native_L13_invocations": 0,
        "execution_bounds": parent.execution_bounds(), "control_count": 13,
        "retained_status": "FAIL", "all_L14_gate_passing_controls": list(CONTROLS),
        "retained_L13_gate_passing_controls": list(parent.PASSING),
        "conditional_rescue_retained_controls": list(parent.PASSING),
        "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS",
        "accepted_prefix_replay": False, "unique_upstream_producer_attributed": False,
    }
    for key, value in expected.items():
        equal(result[key], value, f"frozen L14 result mismatch: {key}")


def check_rows(rows):
    equal([row["control"] for row in rows], list(CONTROLS), "missing/reordered L14 controls")
    for row in rows:
        for key, value in {
            "node": [14, 0], "mandatory_statuses": ["PASS"] * 19,
            "L14_status": "PASS", "all_L14_gates_pass": True, "L14_S18_failure_indices": [],
            "L14_source_operand_state_KV_RTZ_checks": "PASS", "candidate_admitted": False,
            "prior_kv": "own empty P0", "prior_layer_kv_consumed": False,
            "conditional_rescue_retained": row["control"] in parent.PASSING,
        }.items():
            equal(row[key], value, f"frozen L14 row mismatch: {key}")


def check_reports(reports, row):
    require(len(reports) == 19, "incomplete frozen L14 stage evidence")
    for stage, report in enumerate(reports):
        equal(report["node"], [14, 0, stage], "wrong frozen L14 stage")
        require(report["policy_id"] == producer.prior.gates.POLICY_ID
                and report["residual_state_lineage"] == report["kv_lineage"] == "PASS"
                and report["status"] == row["mandatory_statuses"][stage],
                "frozen L14 policy/state/KV/status mismatch")
    equal([item["index"] for item in reports[18]["binary64_v1"]["failures"]],
          row["L14_S18_failure_indices"], "frozen L14 global failure set mismatch")


def check_reference(extension):
    previous = parent.check_reference(extension)
    current = extension["layers"]["15"]
    equal(current["input_binary64"], previous["binary64"], "re-anchored L15 global reference")
    equal(current["input_fp16"], previous["fp16"], "re-anchored L15 FP16 trajectory")
    require(current["prior_kv"] == "own empty P0", "L15 reference KV mismatch")
    return current


def source_context():
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
            and CONTRACT.resolve() == ROOT / f"ace3/contracts/candidates/{NAME}.json",
            "L15 source/contract origin mismatch")
    importlib.import_module(MODULE)
    importlib.import_module(TEST_MODULE)
    origins = parent.source_context()
    require(MODULE in origins and TEST_MODULE in origins, "missing L15 candidate/test origins")
    return origins


def check_parent_review(review):
    require(review["kind"] == "round_reviewed_handoff" and review["producer_role"] == "reviewer"
            and review["mission_id"] == "c55e934b5d70" and review["review"]["status"] == "done",
            "frozen L14 execution lacks independent acceptance")


def authenticate(origins):
    selected = record(SELECTED / "result.json")
    require(selected["sha256"] == SELECTED_SHA, "selected frozen L14 result mismatch")
    inherited = parent.authenticate(origins)
    bound = {item["path"]: item for item in inherited["input_bindings"]}
    bind = lambda item: producer.upstream.bind_input(item, bound)
    read = lambda item: json.loads(bind(item).read_text())
    archive = lambda item: producer.native.load_state(item, bind)
    review_record = record(PARENT_REVIEW)
    # Host review metadata is not a repository source, tensor or scientific input.
    check_parent_review(json.loads(PARENT_REVIEW.read_text()))
    equal(record(PARENT_REVIEW), review_record, "L14 review changed during authentication")
    result = read(selected)
    check_result(result)
    equal(result["selected_result"], inherited["selected_result"], "spliced L14 ancestry")
    for item in result["input_bindings"] + result["artifacts"]:
        bind(item)
    for name, item in result["origins_after_execution"].items():
        equal(origins[name], item, f"frozen L14 source changed: {name}")
        if "path" in item:
            bind(item)
    validation = read(result["validation"])
    for key, value in {
        "diagnostic_id": parent.ID, "status": "VALIDATED_SOFTWARE_SURFACE",
        "collected": 40, "executed": 40, "failures": 0, "errors": 0, "skipped": 0,
        "command": parent.COMMAND, "cwd": str(ROOT),
        "executable": "/home/argustest/miniconda3/bin/python", "PYTHONPATH": str(ROOT),
        **parent.FLAGS,
    }.items():
        equal(validation[key], value, f"frozen L14 validation mismatch: {key}")
    equal(validation["origins"], result["origins_after_execution"], "L14 source lineage mismatch")
    equal(validation["consumers"], inherited["consumers"], "L14 validation parent splice")

    def artifact(name):
        matches = [item for item in result["artifacts"] if item["path"] == str(SELECTED / name)]
        require(len(matches) == 1, f"missing/duplicate L14 artifact: {name}")
        return matches[0]

    rows = read(artifact("interventions.json"))
    check_rows(rows)
    equal(result["retained_actual_L14_baseline"], rows[0], "frozen actual baseline changed")
    reference14 = np.load(bind(inherited["L14_original_reference"]["binary64"]), allow_pickle=False)
    equal(result["L14_original_reference"], inherited["L14_original_reference"],
          "frozen L14 reference substitution")
    consumers = []
    baseline = None
    for row, frozen in zip(rows, inherited["consumers"], strict=True):
        label = row["control"]
        for key, value in frozen.items():
            if key != "L14_status":
                equal(row[key], value, f"L14 parent lineage mismatch: {label}/{key}")
        previous = consumer_parent(archive(row["parent_stages"]))
        saved_parent = archive(artifact(label + "_L14_parent.npz"))
        producer.paired.same_arrays(saved_parent, previous)
        arrays_record = artifact(label + "_L14.npz")
        arrays = archive(arrays_record)
        equal(row["gates"], artifact(label + "_L14_gates.json"), "spliced L14 reports")
        check_reports(read(row["gates"]), row)
        for stage in range(19):
            producer.prior.retained.check_stage_state(stage, arrays, previous)
        require(np.array_equal(producer.prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                               arrays["stage16"]), "frozen L14 S16 RTZ operand mismatch")
        equal(producer.prior.measure(int(arrays["stage18"][62]), float(reference14[62])),
              row["index62"], "frozen L14 scalar/reference mismatch")
        if baseline is None:
            baseline = arrays["stage18"]
        equal(np.flatnonzero(arrays["stage18"] != baseline).tolist(),
              row["L14_S18_changed_indices"], "frozen L14 baseline comparison changed")
        output = consumer_parent(arrays)
        consumers.append({
            "control": label, "node": [15, 0], "parent_stages": arrays_record,
            "parent_fields": {"i": "output_i", "z": "output_z", "h": "stage18"},
            "parent_Q24_units62": int(output["i"][62]),
            "parent_FP16_bits62": f"{int(output['h'][62]):04x}",
            "prior_kv": "own empty P0", "prior_layer_kv_consumed": False,
            "retained_L14_all_gates_pass": True, "L15_status": "NOT_EXECUTED",
            "candidate_admitted": False,
        })
    freeze = read(inherited["retained_freeze"])
    extension = read(freeze["reference_extension"])
    reference = check_reference(extension)
    for key in ("input_binary64", "input_fp16", "binary64", "fp16"):
        bind(reference[key])
    return {
        "selected_result": selected, "parent_review": review_record,
        "input_bindings": list(bound.values()), "retained_freeze": inherited["retained_freeze"],
        "consumers": consumers, "L15_original_reference": reference,
    }


def validate():
    origins = source_context()
    contract_record = record(CONTRACT)
    contract = json.loads(CONTRACT.read_text())
    check_contract(contract)
    history = history_records()
    compiled = []
    for item in origins.values():
        if "path" in item and Path(item["path"]).suffix == ".py":
            path = Path(item["path"])
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
            compiled.append(item)
    with ExitStack() as stack:
        guards = [stack.enter_context(patch.object(owner, name, side_effect=AssertionError(
            "native/RTL dispatch forbidden in L15 check"))) for owner, name in (
                (parent, "diagnose"), (parent, "execute_layer"), (parent.NativeControls, "run"),
                (producer, "diagnose"), (producer.NativeControls, "run"),
                (producer.upstream, "execute_layer"), (producer.native, "stages"),
                (producer.native.candidate, "_stages"), (producer.native, "execute_layers"),
                (producer.coordinate, "execute_layer"), (producer.legacy, "diagnose"),
                (producer.legacy, "original_branches"))]
        evidence = authenticate(origins)
        tests = importlib.import_module(TEST_MODULE)
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        require(count == EXPECTED_TESTS == tests.EXPECTED_TESTS and count > 0,
                "focused test collection mismatch")
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
        require(not any(guard.called for guard in guards), "forbidden dispatch attempted")
    require(result.testsRun == count and result.wasSuccessful() and not result.skipped,
            "focused tests failed or did not execute exactly")
    producer.coordinate.recheck(evidence["input_bindings"])
    producer.coordinate.recheck(history)
    equal(record(PARENT_REVIEW), evidence["parent_review"], "L14 review changed during check")
    equal(source_context(), origins, "source changed during L15 check")
    equal(record(CONTRACT), contract_record, "contract changed during L15 check")
    return {
        "diagnostic_id": ID, "status": "VALIDATED_SOFTWARE_SURFACE",
        "command": COMMAND, "cwd": str(ROOT), "executable": sys.executable,
        "PYTHONPATH": str(ROOT), "origins": origins, "compiled": compiled,
        "contract": contract_record, "history": history, **evidence, **FLAGS,
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "policy_id": contract["policy_id"], "thresholds": contract["thresholds"],
        "native_control_dispatch_authorized": False, "scientific_result_claim": False,
        "accepted_L0_L8_tests_executed": False, "accepted_prefix_replay": False,
        "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "AUTHENTICATED_RETAINED_ONLY",
        "L15_status": "NOT_EXECUTED", "execution_surface_declared": True,
        "execution_command": EXECUTION_COMMAND, "execution_bounds": execution_bounds(),
    }


def execute_layer(tensors, state, trajectory, reference):
    prior = producer.prior
    prior.retained.verify_parent(state, state)
    arrays, locals_, reports = {}, {}, []
    for stage in producer.native.stages(tensors, 15, state, arrays):
        require(type(stage) is int and stage == len(reports) and stage < 19,
                "out-of-order L15 stage")
        expected = prior.retained.check_stage_state(stage, arrays, state)
        if stage < 18 and stage != 12:
            expected = prior.local.local_reference(
                stage, {key: arrays[key].copy() for key in prior.local.OPERANDS[stage]},
                tensors, 15)
        if stage < 18:
            locals_[f"stage{stage:02d}"] = expected
        report = prior.gates.evaluate_decoder_stage(
            stage=stage, actual=arrays[f"stage{stage:02d}"],
            reference=trajectory[f"stage{stage:02d}"], policy=prior.gates.POLICY_ID,
            local_reference=expected, reference_binary64=reference if stage == 18 else None)
        require(report["status"] in ("PASS", "FAIL"), "missing mandatory L15 gate")
        report.update(node=[15, 0, stage], residual_state_lineage="PASS", kv_lineage="PASS")
        reports.append(report)
    require(len(reports) == 19, "incomplete native L15 suffix")
    require(np.array_equal(prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                           arrays["stage16"]), "L15 S16 RTZ operand mismatch")
    return arrays, locals_, reports


class NativeControls:
    def __init__(self, parents, tensors, trajectory, reference):
        equal(list(parents), list(CONTROLS), "missing/reordered L15 parents")
        self.parents = {label: {key: value.copy() for key, value in state.items()}
                        for label, state in parents.items()}
        self.tensors, self.trajectory, self.reference = tensors, trajectory, reference
        self.count = 0

    def run(self, label, state):
        require(self.count < len(CONTROLS) and label == CONTROLS[self.count],
                "native L15 control outside bounded schedule")
        producer.paired.same_arrays(state, self.parents[label])
        producer.prior.retained.verify_parent(state, state)
        self.count += 1
        return execute_layer(self.tensors, state, self.trajectory, self.reference)

    def finish(self):
        require(self.count == len(CONTROLS) == 13, "incomplete L15 control schedule")


def diagnose(out, log, contract, validation):
    check_contract(contract)
    require(validation["diagnostic_id"] == ID
            and validation["status"] == "VALIDATED_SOFTWARE_SURFACE"
            and validation["collected"] == validation["executed"] == EXPECTED_TESTS
            and all(validation[key] == 0 for key in (
                "failures", "errors", "skipped", "native_layer_invocations", "rtl_invocations"))
            and validation["L15_status"] == "NOT_EXECUTED", "invalid pre-dispatch validation")
    equal(source_context(), validation["origins"], "source changed before L15 execution")
    equal(record(CONTRACT), validation["contract"], "contract changed before L15 execution")
    evidence = authenticate(validation["origins"])
    equal(evidence["input_bindings"], validation["input_bindings"], "frozen inputs changed")
    equal(evidence["parent_review"], validation["parent_review"], "L14 review changed before dispatch")
    bound = {item["path"]: item for item in evidence["input_bindings"]}
    bind = lambda item: producer.upstream.bind_input(item, bound)
    archive = lambda item: producer.native.load_state(item, bind)
    producer.coordinate.recheck(bound.values())
    freeze = json.loads(bind(evidence["retained_freeze"]).read_text())
    extension = json.loads(bind(freeze["reference_extension"]).read_text())
    item = check_reference(extension)
    equal(item, evidence["L15_original_reference"], "L15 reference changed")
    reference = np.load(bind(item["binary64"]), allow_pickle=False)
    require(reference.dtype == np.dtype("<f8") and reference.shape == (896,)
            and np.all(np.isfinite(reference)), "invalid original-input L15 reference")
    trajectory = archive(item["fp16"])
    for stage, size in producer.prior.local.SIZES.items():
        producer.prior.local.finite_words(trajectory[f"stage{stage:02d}"], (size,))
    with producer.legacy.safe_open(str(bind(extension["checkpoint"])), framework="numpy") as model:
        tensors = {key: np.ascontiguousarray(model.get_tensor(key))
                   for key in producer.prior.local.tensor_shapes(15)}
    producer.prior.local.authenticate_tensors(tensors, item["canonical"], 15)
    parents = {row["control"]: consumer_parent(archive(row["parent_stages"]))
               for row in evidence["consumers"]}
    controls = NativeControls(parents, tensors, trajectory, reference)
    producer.legacy.torch.set_num_threads(1)
    artifacts, rows = [], []
    started = time.monotonic()
    baseline = None
    for frozen in evidence["consumers"]:
        label = frozen["control"]
        began = time.monotonic()
        arrays, locals_, reports = controls.run(label, parents[label])
        if label == "actual":
            baseline = arrays["stage18"].copy()
        require(baseline is not None, "missing retained actual control")
        for name, values in ((label + "_L15_parent.npz", parents[label]),
                             (label + "_L15.npz", arrays),
                             (label + "_L15_local_references.npz", locals_)):
            np.savez(out / name, **values)
            artifacts.append(record(out / name))
        gates_path = out / (label + "_L15_gates.json")
        producer.write(gates_path, reports)
        artifacts.append(record(gates_path))
        failures = [failure["index"] for failure in reports[18]["binary64_v1"]["failures"]]
        scalar = producer.prior.measure(int(arrays["stage18"][62]), float(reference[62]))
        require(scalar["accepted"] == (62 not in failures), "L15 scalar/full-vector disagreement")
        passing = all(report["status"] == "PASS" for report in reports)
        row = {
            **frozen, "L15_status": "PASS" if passing else "FAIL",
            "mandatory_statuses": [report["status"] for report in reports],
            "all_L15_gates_pass": passing, "L15_S18_failure_indices": failures,
            "conditional_acceptance_retained": frozen["retained_L14_all_gates_pass"] and passing,
            "L15_S18_changed_indices": np.flatnonzero(arrays["stage18"] != baseline).tolist(),
            "index62": scalar, "gates": record(gates_path),
            "L15_source_operand_state_KV_RTZ_checks": "PASS",
            "seconds": time.monotonic() - began,
        }
        rows.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
    controls.finish()
    producer.write(out / "interventions.json", rows)
    artifacts.append(record(out / "interventions.json"))
    producer.coordinate.recheck(bound.values())
    producer.coordinate.recheck(validation["history"])
    equal(record(PARENT_REVIEW), evidence["parent_review"], "L14 review changed during execution")
    equal(source_context(), validation["origins"], "source changed during L15 execution")
    equal(record(CONTRACT), validation["contract"], "contract changed during L15 execution")
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [15, 0, 18], **FLAGS,
        "native_layer_invocations": controls.count, "native_L15_P0_invocations": controls.count,
        "execution_bounds": execution_bounds(), "control_count": len(rows),
        "selected_result": evidence["selected_result"], "parent_review": evidence["parent_review"],
        "retained_status": "FAIL", "retained_L14_gate_passing_controls": list(CONTROLS),
        "all_L15_gate_passing_controls": [row["control"] for row in rows if row["all_L15_gates_pass"]],
        "retained_actual_L15_baseline": rows[0],
        "original_global_reference_unchanged": True, "L15_original_reference": item,
        "source_operand_state_KV_lineage_checks": "PASS",
        "origins_after_execution": validation["origins"],
        "input_bindings": list(bound.values()), "artifacts": artifacts,
        "accepted_prefix_replay": False, "unique_upstream_producer_attributed": False,
        "timing_seconds": {"total": time.monotonic() - started},
        "claim_boundary": contract["claim_boundary"],
    }


def output_path(value):
    out = value.absolute()
    require(out == out.resolve() and out.parent == ROOT / "build"
            and out.name.startswith(OUTPUT_PREFIX) and len(out.name) > len(OUTPUT_PREFIX),
            "output outside fresh versioned L15 build scope")
    require(not out.exists() and not out.is_symlink(), "output already exists")
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.check and args.out is not None or args.execute and args.out is None:
        parser.error("--out is required only with --execute")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.check:
        print(json.dumps(validate(), sort_keys=True), flush=True)
        return 0
    out = output_path(args.out)
    validation = validate()
    out.mkdir(exist_ok=False)
    producer.write(out / "validation.json", validation)
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({"cwd": str(Path.cwd()), "executable": sys.executable,
                              "argv": sys.argv if argv is None else argv,
                              "PYTHONPATH": os.environ.get("PYTHONPATH")}) + "\n")
        log.flush()
        result = diagnose(out, log, json.loads(CONTRACT.read_text()), validation)
    result["validation"] = record(out / "validation.json")
    producer.write(out / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
