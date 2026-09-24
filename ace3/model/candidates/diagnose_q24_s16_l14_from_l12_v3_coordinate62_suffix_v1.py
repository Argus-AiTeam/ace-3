"""Bounded non-admission L14/P0 suffix; --check never dispatches native work."""

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

from ace3.model.candidates import diagnose_q24_s16_l12_coordinate62_producer_cone_v3 as producer


ROOT = producer.ROOT
NAME = "q24_s16_l14_from_l12_v3_coordinate62_suffix_v1"
ID = "ace3-q24-s16-l14-from-l12-v3-coordinate62-suffix-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
SELECTED = ROOT / "build/q24_s16_l12_coordinate62_producer_cone_v3_a739fabe5157_attempt001"
SELECTED_SHA = "5d1c2c710728eeef2be6813a4f018de050e9ef5693846f7a5180bee193959898"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 "
           f"/home/argustest/miniconda3/bin/python -B -m {MODULE} --check")
OUTPUT_PREFIX = NAME + "_"
EXECUTION_COMMAND = COMMAND.removesuffix("--check") + (
    f"--execute --out build/{OUTPUT_PREFIX}fresh")
EXPECTED_TESTS = 40
SOURCE_PINS = {
    "ace3/model/candidates/diagnose_q24_s16_l12_coordinate62_producer_cone_v3.py":
        "5d6a197cfd703a0f7b9baa7ddec20b2724182d6d41e0ec4ebaf39ed5bc7f4a98",
    "ace3/contracts/candidates/q24_s16_l12_coordinate62_producer_cone_v3.json":
        "3c5ec1809f80a92aa07454746ff05ac668b0804a8a514f0d0e0f0904e42aef2e",
    "tests/test_q24_s16_l12_coordinate62_producer_cone_v3.py":
        "e19935604fff11ce854e91795af3464af2f8fc8c482c22bca0a702a357d1005f",
}
FLAGS = {
    "native_layer_invocations": 0, "rtl_invocations": 0, "native_L0_L8_invocations": 0,
    "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
    "normal_host_review": "REQUIRED",
}
CONTROLS = tuple(label for label, _ in producer.plan())
PASSING = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62", "mapped_all",
    "inherited_native",
)
require, record = producer.require, producer.record
np = producer.legacy.np


def execution_bounds():
    return {
        "native_layer_invocations": 13, "native_L14_P0_invocations": 13,
        "native_L12_invocations": 0, "native_L13_invocations": 0,
        "native_L0_L8_invocations": 0, "rtl_invocations": 0,
        "position": 0, "layers": [14], "stages": list(range(19)),
        "schedule": [{"control": label, "layer": 14, "position": 0} for label in CONTROLS],
        "invocations_per_control": 1,
        "output_parent": str(ROOT / "build"), "output_name_prefix": OUTPUT_PREFIX,
        "fresh_directory_required": True, "reference_recomputation": False,
    }


def equal(actual, expected, message):
    require(json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True), message)


def history_records():
    records = producer.history_records()
    for relative, digest in SOURCE_PINS.items():
        path = ROOT / relative
        require(path.resolve() == path, "producer source origin mismatch")
        item = record(path)
        require(item["sha256"] == digest, f"producer source mismatch: {relative}")
        records.append(item)
    return records


def check_contract(contract):
    history_records()
    base = json.loads(producer.CONTRACT.read_text())
    producer.check_contract(base)
    expected = {
        "diagnostic_id": ID, "inherits": str(producer.CONTRACT.relative_to(ROOT)),
        "inherits_sha256": SOURCE_PINS[str(producer.CONTRACT.relative_to(ROOT))],
        "selected_result": str(SELECTED / "result.json"),
        "selected_result_sha256": SELECTED_SHA,
        "policy_id": base["policy_id"], "thresholds": base["thresholds"],
        "controls": list(CONTROLS), "retained_L13_gate_passing_controls": list(PASSING),
        "consumer": {"layer": 14, "position": 0, "stages": list(range(19))},
        "interface": ["--check", "--execute", "--out"], "cwd": str(ROOT), "command": COMMAND,
        "execution_command": EXECUTION_COMMAND, "execution_bounds": execution_bounds(),
        "expected_tests": EXPECTED_TESTS, **FLAGS,
        "native_control_dispatch_authorized": True, "accepted_prefix_replay": False,
        "scientific_result_claim": False,
        "attribution_question": "Do the frozen L12-v3 counterfactual L13 controls retain "
        "their conditional rescue at the immediate L14/P0 consumer? This check "
        "authenticates inputs only and does not answer that execution question.",
        "state_boundary": "Use each complete L13 output I/Z/H as its own L14 parent; "
        "own-empty-P0 FP16 KV, never L13 KV. Q24 residual state is wider than FP16. "
        "Native G128 asymmetric packed INT4 weights, FP16 scales and operator "
        "boundaries, and S16 RTZ are unchanged.",
        "reference_boundary": "The original-input independently propagated binary64 "
        "global reference is unchanged and must never be replaced with a control, "
        "mapped state or local reference. This check does not construct or evaluate "
        "an L14 reference or numerical result.",
        "claim_boundary": "Isolated versioned CPU-software-only non-admission surface. "
        "--check performs zero native dispatch; --execute runs exactly one L14/P0 "
        "consumer for each of the 13 frozen L13 parents, without L12/L13 recomputation. "
        "Execution needs its own bounded Planner selection and independent Host review. "
        "No repair, unique cause, strict-FP16-state W4A16, new-token or full-model PASS "
        "follows. All historical failures and accepted evidence remain unchanged; "
        "no successor is published or queued.",
    }
    equal(contract, expected, "L14 bounded execution-surface contract mismatch")


def check_result(result):
    expected = {
        "diagnostic_id": producer.ID, "status": "DIAGNOSED", "node": [12, 0, 18],
        "index": 62, "control_count": 13, "native_layer_invocations": 15,
        "retained_status": "FAIL", "native_retained_bitwise_reproduction": True,
        "reviewed_coordinate_rescue_reproduced": True,
        "original_L12_S18_bitwise_reproduction": True, "unchanged_baseline_gates": True,
        "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS",
        "all_L13_gate_passing_controls": list(PASSING),
        "unique_upstream_producer_attributed": False,
        **{key: value for key, value in FLAGS.items() if key != "native_layer_invocations"},
    }
    for key, value in expected.items():
        equal(result[key], value, f"frozen L12-v3 result mismatch: {key}")


def check_rows(rows):
    equal([row["label"] for row in rows], list(CONTROLS), "missing/reordered controls")
    for row in rows:
        passing = row["label"] in PASSING
        equal(row["mandatory_statuses"], ["PASS"] * 18 + ["PASS" if passing else "FAIL"],
              "retained L13 mandatory statuses changed")
        equal(row["L13_S18_failure_indices"], [] if passing else [62],
              "retained L13 failure coordinates changed")
        equal(row["all_L13_gates_pass"], passing, "retained control gate result changed")
        require(row["L13_source_operand_state_KV_RTZ_checks"] == "PASS"
                and row["candidate_admitted"] is False, "control lineage/admission changed")


def check_reports(reports, row):
    require(len(reports) == 19, "incomplete L13 stage evidence")
    for stage, report in enumerate(reports):
        equal(report["node"], [13, 0, stage], "wrong L13 stage identity")
        require(report["policy_id"] == producer.prior.gates.POLICY_ID
                and report["residual_state_lineage"] == report["kv_lineage"] == "PASS"
                and report["status"] == row["mandatory_statuses"][stage],
                "L13 policy/lineage/status mismatch")
    equal([item["index"] for item in reports[18]["binary64_v1"]["failures"]],
          row["L13_S18_failure_indices"], "L13 full-vector failures mismatch")


def consumer_parent(arrays):
    parent = producer.prior.retained.state_from(arrays, "output", "stage18")
    producer.prior.retained.verify_parent(parent, parent)
    return {key: value.copy() for key, value in parent.items()}


def source_context():
    require(sys.dont_write_bytecode and not sys.flags.optimize
            and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
            "use the published bytecode-disabled command context")
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
            and CONTRACT.resolve() == ROOT / f"ace3/contracts/candidates/{NAME}.json",
            "L14 source/contract origin mismatch")
    importlib.import_module(MODULE)
    importlib.import_module(TEST_MODULE)
    origins = producer.source_context()
    require(all(name in origins for name in (
        MODULE, TEST_MODULE, "tests.test_q24_s16_toward_zero_l3_l8_v1")),
        "missing required candidate/test origins")
    return origins


def check_reference(extension):
    producer.coordinate.check_reference_chain(extension)
    previous, current = extension["layers"]["13"], extension["layers"]["14"]
    equal(current["input_binary64"], previous["binary64"], "re-anchored L14 global reference")
    equal(current["input_fp16"], previous["fp16"], "re-anchored L14 FP16 trajectory")
    require(current["prior_kv"] == "own empty P0", "L14 reference KV is not own empty P0")
    return current


def authenticate(origins):
    selected = record(SELECTED / "result.json")
    require(selected["sha256"] == SELECTED_SHA, "selected frozen L12-v3 result mismatch")
    data = producer.authenticate()
    bound = data["bound"]
    bind = lambda item: producer.upstream.bind_input(item, bound)
    read = lambda item: json.loads(bind(item).read_text())
    archive = lambda item: producer.native.load_state(item, bind)
    result = read(selected)
    check_result(result)
    for item in result["input_bindings"] + result["artifacts"]:
        bind(item)
    for name, item in result["origins_after_execution"].items():
        equal(origins[name], item, f"frozen source origin changed: {name}")
        if "path" in item:
            bind(item)
    validation = read(result["validation"])
    for key, value in {
        "diagnostic_id": producer.ID, "collected": 30, "executed": 30,
        "failures": 0, "errors": 0, "skipped": 0, "command": producer.COMMAND,
        "cwd": str(ROOT), "executable": "/home/argustest/miniconda3/bin/python",
        "PYTHONPATH": str(ROOT), **FLAGS,
    }.items():
        equal(validation[key], value, f"frozen validation mismatch: {key}")
    equal(validation["origins"], result["origins_after_execution"],
          "frozen validation/execution source mismatch")

    def artifact(name):
        matches = [item for item in result["artifacts"]
                   if item["path"] == str(SELECTED / name)]
        require(len(matches) == 1, f"missing/duplicate frozen artifact: {name}")
        return matches[0]

    rows = read(artifact("interventions.json"))
    check_rows(rows)
    for key, value in producer.classify(rows).items():
        equal(result[key], value, f"frozen classification mismatch: {key}")
    consumers = []
    for row in rows:
        label = row["label"]
        equal(row["parent"], artifact(label + "_L12_parent.npz"), "spliced L12 parent")
        equal(row["gates"], artifact(label + "_L13_gates.json"), "spliced L13 gates")
        parent = archive(row["parent"])
        producer.prior.retained.verify_parent(parent, parent)
        stages_record = artifact(label + "_L13.npz")
        arrays = archive(stages_record)
        reports = read(row["gates"])
        check_reports(reports, row)
        for stage in range(19):
            producer.prior.retained.check_stage_state(stage, arrays, parent)
        require(np.array_equal(producer.prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                               arrays["stage16"]), "frozen L13 S16 RTZ operand mismatch")
        equal(producer.prior.measure(int(arrays["stage18"][62]), float(data["reference"][62])),
              row["index62"], "frozen original-input reference/scalar mismatch")
        if label in ("actual", "mapped62", "mapped_all"):
            key = {"actual": "actual_arrays", "mapped62": "single_062_reviewed",
                   "mapped_all": "mapped_reviewed"}[label]
            producer.paired.same_arrays(arrays, data[key])
        output = consumer_parent(arrays)
        consumers.append({
            "control": label, "node": [14, 0], "parent_stages": stages_record,
            "parent_fields": {"i": "output_i", "z": "output_z", "h": "stage18"},
            "parent_Q24_units62": int(output["i"][62]),
            "parent_FP16_bits62": f"{int(output['h'][62]):04x}",
            "prior_kv": "own empty P0", "prior_layer_kv_consumed": False,
            "retained_L13_all_gates_pass": row["all_L13_gates_pass"],
            "L14_status": "NOT_EXECUTED", "candidate_admitted": False,
        })
    reference = check_reference(data["extension"])
    for key in ("input_binary64", "input_fp16", "binary64", "fp16"):
        bind(reference[key])
    return {
        "selected_result": selected, "input_bindings": list(bound.values()),
        "retained_freeze": data["authentication"]["retained_freeze"],
        "consumers": consumers, "L14_original_reference": reference,
    }


def validate():
    origins = source_context()
    contract_record = record(CONTRACT)
    check_contract(json.loads(CONTRACT.read_text()))
    history = history_records()
    compiled = []
    for item in origins.values():
        if "path" in item and Path(item["path"]).suffix == ".py":
            path = Path(item["path"])
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
            compiled.append(item)
    with ExitStack() as stack:
        guards = [stack.enter_context(patch.object(owner, name, side_effect=AssertionError(
            "native/RTL dispatch forbidden in L14 check"))) for owner, name in (
                (producer, "diagnose"), (producer.NativeControls, "run"),
                (producer.upstream, "execute_layer"), (producer.native, "stages"),
                (producer.native.candidate, "_stages"),
                (producer.native, "execute_layers"), (producer.coordinate, "execute_layer"),
                (producer.legacy, "diagnose"), (producer.legacy, "original_branches"))]
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
    equal(source_context(), origins, "source changed during check")
    equal(record(CONTRACT), contract_record, "contract changed during check")
    return {
        "diagnostic_id": ID, "status": "VALIDATED_SOFTWARE_SURFACE",
        "command": COMMAND, "cwd": str(ROOT), "executable": sys.executable,
        "PYTHONPATH": str(ROOT), "origins": origins, "compiled": compiled,
        "contract": contract_record, "history": history, **evidence, **FLAGS,
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "native_control_dispatch_authorized": False, "scientific_result_claim": False,
        "accepted_L0_L8_tests_executed": False, "accepted_prefix_replay": False,
        "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "AUTHENTICATED_RETAINED_ONLY",
        "L14_status": "NOT_EXECUTED", "execution_surface_declared": True,
        "execution_command": EXECUTION_COMMAND, "execution_bounds": execution_bounds(),
    }


def execute_layer(tensors, parent, trajectory, reference):
    # The inherited evaluator is scoped to L9-L13; only its unchanged gate recipe is reused.
    prior = producer.prior
    prior.retained.verify_parent(parent, parent)
    arrays, locals_, reports = {}, {}, []
    for stage in producer.native.stages(tensors, 14, parent, arrays):
        require(type(stage) is int and stage == len(reports) and stage < 19,
                "out-of-order L14 stage")
        expected = prior.retained.check_stage_state(stage, arrays, parent)
        if stage < 18 and stage != 12:
            expected = prior.local.local_reference(
                stage, {key: arrays[key].copy() for key in prior.local.OPERANDS[stage]},
                tensors, 14)
        if stage < 18:
            locals_[f"stage{stage:02d}"] = expected
        report = prior.gates.evaluate_decoder_stage(
            stage=stage, actual=arrays[f"stage{stage:02d}"],
            reference=trajectory[f"stage{stage:02d}"], policy=prior.gates.POLICY_ID,
            local_reference=expected, reference_binary64=reference if stage == 18 else None)
        require(report["status"] in ("PASS", "FAIL"), "missing mandatory L14 gate evidence")
        report.update(node=[14, 0, stage], residual_state_lineage="PASS", kv_lineage="PASS")
        reports.append(report)
    require(len(reports) == 19, "incomplete native L14 suffix")
    require(np.array_equal(prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                           arrays["stage16"]), "L14 S16 RTZ operand mismatch")
    return arrays, locals_, reports


class NativeControls:
    def __init__(self, parents, tensors, trajectory, reference):
        equal(list(parents), list(CONTROLS), "missing/reordered L14 parents")
        self.parents = {
            label: {key: value.copy() for key, value in parent.items()}
            for label, parent in parents.items()
        }
        self.tensors, self.trajectory, self.reference = tensors, trajectory, reference
        self.count = 0

    def run(self, label, parent):
        require(self.count < len(CONTROLS) and label == CONTROLS[self.count],
                "native L14 control outside bounded schedule")
        producer.paired.same_arrays(parent, self.parents[label])
        producer.prior.retained.verify_parent(parent, parent)
        self.count += 1
        return execute_layer(self.tensors, parent, self.trajectory, self.reference)

    def finish(self):
        require(self.count == len(CONTROLS) == 13, "incomplete L14 control schedule")


def diagnose(out, log, contract, validation):
    check_contract(contract)
    require(validation["diagnostic_id"] == ID
            and validation["status"] == "VALIDATED_SOFTWARE_SURFACE"
            and validation["collected"] == validation["executed"] == EXPECTED_TESTS
            and all(validation[key] == 0 for key in (
                "failures", "errors", "skipped", "native_layer_invocations", "rtl_invocations"))
            and validation["L14_status"] == "NOT_EXECUTED",
            "invalid pre-dispatch validation")
    equal(source_context(), validation["origins"], "source changed before L14 execution")
    equal(record(CONTRACT), validation["contract"], "contract changed before L14 execution")
    evidence = authenticate(validation["origins"])
    equal(evidence["input_bindings"], validation["input_bindings"], "frozen inputs changed")
    bound = {item["path"]: item for item in evidence["input_bindings"]}
    producer.coordinate.recheck(bound.values())
    bind = lambda item: producer.upstream.bind_input(item, bound)
    archive = lambda item: producer.native.load_state(item, bind)
    freeze = json.loads(bind(evidence["retained_freeze"]).read_text())
    extension = json.loads(bind(freeze["reference_extension"]).read_text())
    item = check_reference(extension)
    equal(item, evidence["L14_original_reference"], "L14 reference changed")
    reference = np.load(bind(item["binary64"]), allow_pickle=False)
    require(reference.dtype == np.dtype("<f8") and reference.shape == (896,)
            and np.all(np.isfinite(reference)), "invalid original-input L14 reference")
    trajectory = archive(item["fp16"])
    for stage, size in producer.prior.local.SIZES.items():
        producer.prior.local.finite_words(trajectory[f"stage{stage:02d}"], (size,))
    with producer.legacy.safe_open(str(bind(extension["checkpoint"])), framework="numpy") as model:
        tensors = {key: np.ascontiguousarray(model.get_tensor(key))
                   for key in producer.prior.local.tensor_shapes(14)}
    producer.prior.local.authenticate_tensors(tensors, item["canonical"], 14)
    parents = {
        row["control"]: consumer_parent(archive(row["parent_stages"]))
        for row in evidence["consumers"]
    }
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
        for name, values in ((label + "_L14_parent.npz", parents[label]),
                             (label + "_L14.npz", arrays),
                             (label + "_L14_local_references.npz", locals_)):
            np.savez(out / name, **values)
            artifacts.append(record(out / name))
        gates_path = out / (label + "_L14_gates.json")
        producer.write(gates_path, reports)
        artifacts.append(record(gates_path))
        failures = [failure["index"] for failure in reports[18]["binary64_v1"]["failures"]]
        scalar = producer.prior.measure(int(arrays["stage18"][62]), float(reference[62]))
        require(scalar["accepted"] == (62 not in failures), "L14 scalar/full-vector disagreement")
        passing = all(report["status"] == "PASS" for report in reports)
        row = {
            **frozen, "L14_status": "PASS" if passing else "FAIL",
            "mandatory_statuses": [report["status"] for report in reports],
            "all_L14_gates_pass": passing, "L14_S18_failure_indices": failures,
            "conditional_rescue_retained": frozen["retained_L13_all_gates_pass"] and passing,
            "L14_S18_changed_indices": np.flatnonzero(arrays["stage18"] != baseline).tolist(),
            "index62": scalar, "gates": record(gates_path),
            "L14_source_operand_state_KV_RTZ_checks": "PASS",
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
    equal(source_context(), validation["origins"], "source changed during L14 execution")
    equal(record(CONTRACT), validation["contract"], "contract changed during L14 execution")
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [14, 0, 18], **FLAGS,
        "native_layer_invocations": controls.count, "native_L14_P0_invocations": controls.count,
        "native_L12_invocations": 0, "native_L13_invocations": 0,
        "execution_bounds": execution_bounds(), "control_count": len(rows),
        "selected_result": evidence["selected_result"], "retained_status": "FAIL",
        "retained_L13_gate_passing_controls": list(PASSING),
        "all_L14_gate_passing_controls": [row["control"] for row in rows if row["all_L14_gates_pass"]],
        "conditional_rescue_retained_controls": [
            row["control"] for row in rows if row["conditional_rescue_retained"]],
        "retained_actual_L14_baseline": rows[0],
        "original_global_reference_unchanged": True, "L14_original_reference": item,
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
            "output outside fresh versioned L14 build scope")
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
