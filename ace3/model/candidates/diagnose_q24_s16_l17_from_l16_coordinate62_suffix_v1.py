"""Bounded non-admission L17/P0 surface from authenticated reviewed L16 controls."""

import argparse
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l16_from_l15_coordinate62_suffix_v1 as parent


producer = parent.producer
ROOT = parent.ROOT
NAME = "q24_s16_l17_from_l16_coordinate62_suffix_v1"
ID = "ace3-q24-s16-l17-from-l16-coordinate62-suffix-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
SELECTED = ROOT / "build/q24_s16_l16_from_l15_coordinate62_suffix_v1_64818362a12d_attempt001"
SELECTED_SHA = "0b77cc2dcc3a3d2d218aaf707debe5fd45b8bb837da3845c1ecd8c49b480e491"
PARENT_REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/64818362a12d/round-0002.json")
SOURCE_PINS = {
    "ace3/model/candidates/diagnose_q24_s16_l16_from_l15_coordinate62_suffix_v1.py":
        "a3d6eca36d0341798ba70b992a65f8d4573f4fce2107b36a6babd029a9e65620",
    "ace3/contracts/candidates/q24_s16_l16_from_l15_coordinate62_suffix_v1.json":
        "66875921e8af5884506ee63ccd4ef7384bbf67e989a3e0e85560b5535b678408",
    "tests/test_q24_s16_l16_from_l15_coordinate62_suffix_v1.py":
        "963a8e1314291a07de68032bfd81e9d8c0a0651f4f63ef051512e02ba487737b",
}
REFERENCE_PINS = {
    "binary64": "e98f346398479158130aa834d6b2a8aabe7c23c7ddf741e346395bbdb8ba6c62",
    "fp16": "d2d8a1bbcb502f9b3fa13df481b10b2e1ba1d33b6c37cd6d4f0b6ea5df90759d",
}
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 "
           f"/home/argustest/miniconda3/bin/python -B -m {MODULE} --check")
OUTPUT_PREFIX = NAME + "_"
EXECUTION_COMMAND = COMMAND.removesuffix("--check") + (
    f"--execute --out build/{OUTPUT_PREFIX}fresh")
EXPECTED_TESTS = 32
CONTROLS = parent.CONTROLS
FLAGS = {**parent.FLAGS, "native_L17_P0_invocations": 0, "native_L0_L16_invocations": 0}
require, record, equal = parent.require, parent.record, parent.equal
np, consumer_parent = parent.np, parent.consumer_parent


def planned_bounds():
    return {
        "planned_native_layer_invocations": 13, "planned_native_L17_P0_invocations": 13,
        "layers": [17], "position": 0, "stages": list(range(19)),
        "schedule": [{"control": label, "layer": 17, "position": 0} for label in CONTROLS],
        "invocations_per_control": 1, "native_L0_L16_invocations": 0,
        "rtl_invocations": 0, "reference_recomputation": False,
        "accepted_prefix_replay": False, "separate_execution_task_required": True,
    }


def execution_bounds():
    return {
        **FLAGS, "native_layer_invocations": 13, "native_L17_P0_invocations": 13,
        "layers": [17], "position": 0, "stages": list(range(19)),
        "schedule": planned_bounds()["schedule"], "invocations_per_control": 1,
        "output_parent": str(ROOT / "build"), "output_name_prefix": OUTPUT_PREFIX,
        "fresh_directory_required": True, "ignored_directory_required": True,
        "symlinks_forbidden": True, "reference_recomputation": False,
        "accepted_prefix_replay": False, "separate_execution_task_required": True,
    }


def repository_record(path):
    require(path.is_relative_to(ROOT) and path.resolve() == path,
            "L17 input/source/contract must resolve inside the isolated repository")
    return record(path)


def history_records():
    records = parent.history_records()
    for relative, digest in SOURCE_PINS.items():
        item = repository_record(ROOT / relative)
        require(item["sha256"] == digest, f"frozen L16 source changed: {relative}")
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
        "controls": list(CONTROLS), "retained_L16_gate_passing_controls": list(CONTROLS),
        "retained_L15_gate_passing_controls": list(parent.PASSING),
        "consumer": {"layer": 17, "position": 0, "stages": list(range(19))},
        "reference_sha256": REFERENCE_PINS, "interface": ["--check", "--execute", "--out"],
        "cwd": str(ROOT), "command": COMMAND, "expected_tests": EXPECTED_TESTS,
        "execution_command": EXECUTION_COMMAND, "execution_bounds": execution_bounds(),
        "planned_bounds": planned_bounds(), **FLAGS,
        "native_control_dispatch_authorized": True, "execution_surface_declared": True,
        "accepted_prefix_replay": False, "scientific_result_claim": False,
        "attribution_question": "Do the 13 frozen reviewed L16 controls, including the "
        "actual baseline and the four historical L15 failures now passing at L16, "
        "retain their gate classifications at L17/P0?",
        "layer_necessity": "Plan one complete L17/P0 S0-S18 immediate consumer per "
        "frozen L16 control. L0-L16 are authenticated only, never recomputed.",
        "state_boundary": "Complete frozen L16 output I/Z/H, own-empty-P0 FP16 KV, "
        "never L16 KV. Q24 residual state is wider than FP16. Native G128 asymmetric "
        "packed INT4 weights, native GEMM nibble ordering, no qzero plus-one, FP16 "
        "scales/operator boundaries/KV and S16 RTZ remain unchanged.",
        "reference_boundary": "Only the frozen independently propagated original-input "
        "L17 binary64 global reference and FP16 trajectory linked through original L16 "
        "references. No recomputation, re-anchoring or control-derived reference.",
        "claim_boundary": "Isolated versioned CPU-software non-admission surface. "
        "--check authenticates frozen evidence and runs focused mocked-dispatch tests "
        "with zero native/RTL dispatch and L17_status=NOT_EXECUTED. A separate execution "
        "task may use --execute --out for exactly 13 L17/P0 invocations in a fresh "
        "ignored versioned build directory. No admission, repair, unique cause, "
        "strict-FP16-state W4A16, new-token or full-model PASS. Preserve all historical "
        "failures and accepted evidence. Independent Host review is required; "
        "no successor is published or queued.",
    }
    equal(contract, expected, "L17 non-admission execution-surface contract mismatch")


def check_parent_review(review):
    require(review["kind"] == "round_reviewed_handoff" and review["producer_role"] == "reviewer"
            and review["mission_id"] == "64818362a12d" and review["round"] == 2
            and review["review"]["status"] == "done",
            "frozen L16 execution lacks independent acceptance")


def check_result(result):
    for key, value in {
        "diagnostic_id": parent.ID, "status": "DIAGNOSED", "node": [16, 0, 18],
        **parent.FLAGS, "native_layer_invocations": 13, "native_L16_P0_invocations": 13,
        "native_L0_L15_invocations": 0,
        "execution_bounds": parent.execution_bounds(), "control_count": 13,
        "retained_status": "FAIL", "all_L16_gate_passing_controls": list(CONTROLS),
        "retained_L15_gate_passing_controls": list(parent.PASSING),
        "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS",
        "accepted_prefix_replay": False, "unique_upstream_producer_attributed": False,
    }.items():
        equal(result[key], value, f"frozen L16 result mismatch: {key}")


def check_rows(rows):
    equal([row["control"] for row in rows], list(CONTROLS), "missing/reordered L16 controls")
    for row in rows:
        prior_passing = row["control"] in parent.PASSING
        for key, value in {
            "node": [16, 0], "mandatory_statuses": ["PASS"] * 19,
            "L16_status": "PASS", "all_L16_gates_pass": True,
            "L16_S18_failure_indices": [], "L16_source_operand_state_KV_RTZ_checks": "PASS",
            "candidate_admitted": False, "prior_kv": "own empty P0",
            "prior_layer_kv_consumed": False, "retained_L15_all_gates_pass": prior_passing,
            "retained_L15_S18_failure_indices": [] if prior_passing else [62],
            "conditional_acceptance_retained": prior_passing,
        }.items():
            equal(row[key], value, f"frozen L16 row mismatch: {key}")


def check_reports(reports, row):
    require(len(reports) == 19, "incomplete frozen L16 stage evidence")
    for stage, report in enumerate(reports):
        equal(report["node"], [16, 0, stage], "wrong frozen L16 stage")
        require(report["policy_id"] == producer.prior.gates.POLICY_ID
                and report["residual_state_lineage"] == report["kv_lineage"] == "PASS"
                and report["status"] == row["mandatory_statuses"][stage],
                "frozen L16 policy/state/KV/status mismatch")
    equal([item["index"] for item in reports[18]["binary64_v1"]["failures"]],
          row["L16_S18_failure_indices"], "frozen L16 global failure set mismatch")


def check_reference(extension):
    previous = parent.check_reference(extension)
    current = extension["layers"]["17"]
    equal(current["input_binary64"], previous["binary64"], "re-anchored L17 global reference")
    equal(current["input_fp16"], previous["fp16"], "re-anchored L17 FP16 trajectory")
    require(current["prior_kv"] == "own empty P0", "L17 reference KV mismatch")
    for key, digest in REFERENCE_PINS.items():
        equal(current[key]["sha256"], digest, f"substituted L17 reference: {key}")
    return current


def source_context():
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "L17 candidate origin mismatch")
    repository_record(CONTRACT)
    importlib.import_module(MODULE)
    importlib.import_module(TEST_MODULE)
    origins = parent.source_context()
    require(MODULE in origins and TEST_MODULE in origins, "missing L17 candidate/test origins")
    return origins


def authenticate(origins):
    selected = repository_record(SELECTED / "result.json")
    require(selected["sha256"] == SELECTED_SHA, "selected frozen L16 result mismatch")
    inherited = parent.authenticate(origins)
    bound = {item["path"]: item for item in inherited["input_bindings"]}

    def bind(item):
        equal(repository_record(Path(item["path"])), item, "L16/L17 artifact binding mismatch")
        return producer.upstream.bind_input(item, bound)

    read = lambda item: json.loads(bind(item).read_text())
    archive = lambda item: producer.native.load_state(item, bind)
    review_record = record(PARENT_REVIEW)
    check_parent_review(json.loads(PARENT_REVIEW.read_text()))
    equal(record(PARENT_REVIEW), review_record, "L16 review changed during authentication")
    result = read(selected)
    check_result(result)
    equal(result["selected_result"], inherited["selected_result"], "spliced L16 ancestry")
    equal(result["parent_review"], inherited["parent_review"], "spliced L16 review lineage")
    # Official model inputs are authenticated by the inherited closure.
    for item in result["input_bindings"]:
        producer.upstream.bind_input(item, bound)
    for item in result["artifacts"]:
        bind(item)
    for name, item in result["origins_after_execution"].items():
        equal(origins[name], item, f"frozen L16 source changed: {name}")
        if "path" in item:
            bind(item)
    validation = read(result["validation"])
    for key, value in {
        "diagnostic_id": parent.ID, "status": "VALIDATED_SOFTWARE_SURFACE",
        "collected": parent.EXPECTED_TESTS, "executed": parent.EXPECTED_TESTS,
        "failures": 0, "errors": 0, "skipped": 0, "command": parent.COMMAND,
        "cwd": str(ROOT), "executable": "/home/argustest/miniconda3/bin/python",
        "PYTHONPATH": str(ROOT), "L16_status": "NOT_EXECUTED", **parent.FLAGS,
    }.items():
        equal(validation[key], value, f"frozen L16 validation mismatch: {key}")
    equal(validation["origins"], result["origins_after_execution"], "L16 source lineage mismatch")
    equal(validation["consumers"], inherited["consumers"], "L16 validation parent splice")

    def artifact(name):
        matches = [item for item in result["artifacts"] if item["path"] == str(SELECTED / name)]
        require(len(matches) == 1, f"missing/duplicate L16 artifact: {name}")
        return matches[0]

    rows = read(artifact("interventions.json"))
    check_rows(rows)
    equal(result["retained_actual_L16_baseline"], rows[0], "frozen actual baseline changed")
    equal(result["L16_original_reference"], inherited["L16_original_reference"],
          "frozen L16 reference substitution")
    reference16 = np.load(bind(inherited["L16_original_reference"]["binary64"]), allow_pickle=False)
    consumers, baseline = [], None
    for row, frozen in zip(rows, inherited["consumers"], strict=True):
        label = row["control"]
        for key, value in frozen.items():
            if key != "L16_status":
                equal(row[key], value, f"L16 parent lineage mismatch: {label}/{key}")
        previous = consumer_parent(archive(row["parent_stages"]))
        producer.paired.same_arrays(archive(artifact(label + "_L16_parent.npz")), previous)
        arrays_record = artifact(label + "_L16.npz")
        arrays = archive(arrays_record)
        equal(row["gates"], artifact(label + "_L16_gates.json"), "spliced L16 reports")
        check_reports(read(row["gates"]), row)
        for stage in range(19):
            producer.prior.retained.check_stage_state(stage, arrays, previous)
        require(np.array_equal(producer.prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                               arrays["stage16"]), "frozen L16 S16 RTZ operand mismatch")
        equal(producer.prior.measure(int(arrays["stage18"][62]), float(reference16[62])),
              row["index62"], "frozen L16 scalar/reference mismatch")
        if baseline is None:
            baseline = arrays["stage18"]
        equal(np.flatnonzero(arrays["stage18"] != baseline).tolist(),
              row["L16_S18_changed_indices"], "frozen L16 baseline comparison changed")
        output = consumer_parent(arrays)
        consumers.append({
            "control": label, "node": [17, 0], "parent_stages": arrays_record,
            "parent_fields": {"i": "output_i", "z": "output_z", "h": "stage18"},
            "parent_Q24_units62": int(output["i"][62]),
            "parent_FP16_bits62": f"{int(output['h'][62]):04x}",
            "prior_kv": "own empty P0", "prior_layer_kv_consumed": False,
            "retained_L15_all_gates_pass": row["retained_L15_all_gates_pass"],
            "retained_L15_S18_failure_indices": row["retained_L15_S18_failure_indices"],
            "retained_L16_all_gates_pass": row["all_L16_gates_pass"],
            "retained_L16_S18_failure_indices": row["L16_S18_failure_indices"],
            "L17_status": "NOT_EXECUTED", "candidate_admitted": False,
        })
    freeze = read(inherited["retained_freeze"])
    reference = check_reference(read(freeze["reference_extension"]))
    for key in ("input_binary64", "input_fp16", "binary64", "fp16"):
        bind(reference[key])
    global_reference = np.load(bind(reference["binary64"]), allow_pickle=False)
    require(global_reference.dtype == np.dtype("<f8") and global_reference.shape == (896,)
            and np.all(np.isfinite(global_reference)), "invalid original-input L17 reference")
    trajectory = archive(reference["fp16"])
    for stage, size in producer.prior.local.SIZES.items():
        producer.prior.local.finite_words(trajectory[f"stage{stage:02d}"], (size,))
    return {
        "selected_result": selected, "parent_review": review_record,
        "input_bindings": list(bound.values()), "retained_freeze": inherited["retained_freeze"],
        "consumers": consumers, "L17_original_reference": reference,
    }


def validate():
    origins = source_context()
    contract_record = repository_record(CONTRACT)
    contract = json.loads(CONTRACT.read_text())
    check_contract(contract)
    history, compiled = history_records(), []
    for item in origins.values():
        if "path" in item and Path(item["path"]).suffix == ".py":
            path = Path(item["path"])
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
            compiled.append(item)
    with ExitStack() as stack:
        targets = [
            (sys.modules[__name__], "diagnose"), (sys.modules[__name__], "execute_layer"),
            (producer, "diagnose"), (producer.NativeControls, "run"),
            (producer.upstream, "execute_layer"), (producer.native, "stages"),
            (producer.native.candidate, "_stages"), (producer.native, "execute_layers"),
            (producer.coordinate, "execute_layer"), (producer.legacy, "diagnose"),
            (producer.legacy, "original_branches"),
        ]
        for ancestor in (parent, parent.parent, parent.parent.parent):
            targets.extend(((ancestor, "diagnose"), (ancestor, "execute_layer"),
                            (ancestor.NativeControls, "run")))
        guards = [stack.enter_context(patch.object(owner, name, side_effect=AssertionError(
            "native/RTL dispatch forbidden in L17 check"))) for owner, name in targets]
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
    equal(record(PARENT_REVIEW), evidence["parent_review"], "L16 review changed during check")
    equal(source_context(), origins, "source changed during L17 check")
    equal(repository_record(CONTRACT), contract_record, "contract changed during L17 check")
    return {
        "diagnostic_id": ID, "status": "VALIDATED_SOFTWARE_SURFACE",
        "command": COMMAND, "cwd": str(ROOT), "executable": sys.executable,
        "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1",
        "origins": origins, "compiled": compiled, "contract": contract_record,
        "history": history, **evidence, **FLAGS,
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "policy_id": contract["policy_id"], "thresholds": contract["thresholds"],
        "native_control_dispatch_authorized": False, "scientific_result_claim": False,
        "accepted_L0_L8_tests_executed": False, "accepted_prefix_replay": False,
        "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "AUTHENTICATED_RETAINED_ONLY",
        "L17_status": "NOT_EXECUTED", "execution_surface_declared": True,
        "execution_command": EXECUTION_COMMAND, "execution_bounds": execution_bounds(),
        "planned_bounds": planned_bounds(), "dispatch_guard_calls": sum(g.call_count for g in guards),
    }


def execute_layer(tensors, state, trajectory, reference):
    prior = producer.prior
    prior.retained.verify_parent(state, state)
    arrays, locals_, reports = {}, {}, []
    for stage in producer.native.stages(tensors, 17, state, arrays):
        require(type(stage) is int and stage == len(reports) and stage < 19,
                "out-of-order L17 stage")
        expected = prior.retained.check_stage_state(stage, arrays, state)
        if stage < 18 and stage != 12:
            expected = prior.local.local_reference(
                stage, {key: arrays[key].copy() for key in prior.local.OPERANDS[stage]},
                tensors, 17)
        if stage < 18:
            locals_[f"stage{stage:02d}"] = expected
        report = prior.gates.evaluate_decoder_stage(
            stage=stage, actual=arrays[f"stage{stage:02d}"],
            reference=trajectory[f"stage{stage:02d}"], policy=prior.gates.POLICY_ID,
            local_reference=expected, reference_binary64=reference if stage == 18 else None)
        require(report["status"] in ("PASS", "FAIL"), "missing mandatory L17 gate")
        report.update(node=[17, 0, stage], residual_state_lineage="PASS", kv_lineage="PASS")
        reports.append(report)
    require(len(reports) == 19, "incomplete native L17 suffix")
    require(np.array_equal(prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                           arrays["stage16"]), "L17 S16 RTZ operand mismatch")
    return arrays, locals_, reports


class NativeControls:
    def __init__(self, parents, tensors, trajectory, reference):
        equal(list(parents), list(CONTROLS), "missing/reordered L17 parents")
        self.parents = {label: {key: value.copy() for key, value in state.items()}
                        for label, state in parents.items()}
        self.tensors, self.trajectory, self.reference = tensors, trajectory, reference
        self.count = 0

    def run(self, label, state):
        require(self.count < len(CONTROLS) and label == CONTROLS[self.count],
                "native L17 control outside bounded schedule")
        producer.paired.same_arrays(state, self.parents[label])
        producer.prior.retained.verify_parent(state, state)
        self.count += 1
        return execute_layer(self.tensors, state, self.trajectory, self.reference)

    def finish(self):
        require(self.count == len(CONTROLS) == 13, "incomplete L17 control schedule")


def diagnose(out, log, contract, validation):
    check_contract(contract)
    require(validation["diagnostic_id"] == ID
            and validation["status"] == "VALIDATED_SOFTWARE_SURFACE"
            and validation["collected"] == validation["executed"] == EXPECTED_TESTS
            and all(validation[key] == 0 for key in (
                "failures", "errors", "skipped", "native_layer_invocations", "rtl_invocations"))
            and validation["L17_status"] == "NOT_EXECUTED", "invalid pre-dispatch validation")
    equal(source_context(), validation["origins"], "source changed before L17 execution")
    equal(repository_record(CONTRACT), validation["contract"], "contract changed before L17 execution")
    evidence = authenticate(validation["origins"])
    equal(evidence["input_bindings"], validation["input_bindings"], "frozen inputs changed")
    equal(evidence["parent_review"], validation["parent_review"], "L16 review changed before dispatch")
    bound = {item["path"]: item for item in evidence["input_bindings"]}
    bind = lambda item: producer.upstream.bind_input(item, bound)
    archive = lambda item: producer.native.load_state(item, bind)
    producer.coordinate.recheck(bound.values())
    freeze = json.loads(bind(evidence["retained_freeze"]).read_text())
    extension = json.loads(bind(freeze["reference_extension"]).read_text())
    item = check_reference(extension)
    equal(item, evidence["L17_original_reference"], "L17 reference changed")
    reference = np.load(bind(item["binary64"]), allow_pickle=False)
    require(reference.dtype == np.dtype("<f8") and reference.shape == (896,)
            and np.all(np.isfinite(reference)), "invalid original-input L17 reference")
    trajectory = archive(item["fp16"])
    for stage, size in producer.prior.local.SIZES.items():
        producer.prior.local.finite_words(trajectory[f"stage{stage:02d}"], (size,))
    with producer.legacy.safe_open(str(bind(extension["checkpoint"])), framework="numpy") as model:
        tensors = {key: np.ascontiguousarray(model.get_tensor(key))
                   for key in producer.prior.local.tensor_shapes(17)}
    producer.prior.local.authenticate_tensors(tensors, item["canonical"], 17)
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
        for name, values in ((label + "_L17_parent.npz", parents[label]),
                             (label + "_L17.npz", arrays),
                             (label + "_L17_local_references.npz", locals_)):
            np.savez(out / name, **values)
            artifacts.append(record(out / name))
        gates_path = out / (label + "_L17_gates.json")
        producer.write(gates_path, reports)
        artifacts.append(record(gates_path))
        failures = [failure["index"] for failure in reports[18]["binary64_v1"]["failures"]]
        scalar = producer.prior.measure(int(arrays["stage18"][62]), float(reference[62]))
        require(scalar["accepted"] == (62 not in failures), "L17 scalar/full-vector disagreement")
        passing = all(report["status"] == "PASS" for report in reports)
        row = {
            **frozen, "L17_status": "PASS" if passing else "FAIL",
            "mandatory_statuses": [report["status"] for report in reports],
            "all_L17_gates_pass": passing, "L17_S18_failure_indices": failures,
            "conditional_acceptance_retained": frozen["retained_L15_all_gates_pass"]
            and frozen["retained_L16_all_gates_pass"] and passing,
            "L17_S18_changed_indices": np.flatnonzero(arrays["stage18"] != baseline).tolist(),
            "index62": scalar, "gates": record(gates_path),
            "L17_source_operand_state_KV_RTZ_checks": "PASS",
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
    equal(record(PARENT_REVIEW), evidence["parent_review"], "L16 review changed during execution")
    equal(source_context(), validation["origins"], "source changed during L17 execution")
    equal(repository_record(CONTRACT), validation["contract"], "contract changed during L17 execution")
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [17, 0, 18], **FLAGS,
        "native_layer_invocations": controls.count, "native_L17_P0_invocations": controls.count,
        "execution_bounds": execution_bounds(), "control_count": len(rows),
        "selected_result": evidence["selected_result"], "parent_review": evidence["parent_review"],
        "retained_status": "FAIL", "retained_L15_gate_passing_controls": list(parent.PASSING),
        "retained_L16_gate_passing_controls": list(CONTROLS),
        "all_L17_gate_passing_controls": [row["control"] for row in rows if row["all_L17_gates_pass"]],
        "retained_actual_L17_baseline": rows[0],
        "original_global_reference_unchanged": True, "L17_original_reference": item,
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
            "output outside fresh versioned L17 build scope")
    require(not out.exists() and not out.is_symlink(), "output already exists")
    ignored = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--", str(out)],
        capture_output=True, text=True, check=False)
    require(ignored.returncode == 0,
            f"output must be git-ignored (exit {ignored.returncode}): {ignored.stderr}")
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
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
    output_path(out)
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
