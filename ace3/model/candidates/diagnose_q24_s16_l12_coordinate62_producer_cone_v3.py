"""Bounded non-admission CPU producer controls; --check never dispatches them."""

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

from ace3.model.candidates import diagnose_q24_s16_l12_coordinate62_producer_cone_v2 as v2


ROOT = v2.ROOT
ID = "ace3-q24-s16-l12-coordinate62-producer-cone-v3"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l12_coordinate62_producer_cone_v3"
TEST_MODULE = "tests.test_q24_s16_l12_coordinate62_producer_cone_v3"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l12_coordinate62_producer_cone_v3.json"
OUTPUT_PREFIX = "q24_s16_l12_coordinate62_producer_cone_v3_"
CONTEXT = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 "
           f"/home/argustest/miniconda3/bin/python -B -m {MODULE}")
COMMAND = CONTEXT + " --check"
EXECUTION_COMMAND = CONTEXT + f" --execute --out build/{OUTPUT_PREFIX}fresh"
EXPECTED_TESTS = 30
HISTORY = {
    **v2.HISTORY,
    "ace3/model/candidates/diagnose_q24_s16_l12_coordinate62_producer_cone_v2.py":
        "c1569932cc3784d069a0329c707975fda00b18b6166a40aeceb0c0e5c7a38a4c",
    "ace3/contracts/candidates/q24_s16_l12_coordinate62_producer_cone_v2.json":
        "46d378c90626ace8555055a3ab453056a3a2ff60ebc5a3288922a4d2fd2bc2d6",
    "tests/test_q24_s16_l12_coordinate62_producer_cone_v2.py":
        "57844f8adca8a2490391aba76dac312e84773a693fcbb592aeaa923f0979a503",
}
legacy, coordinate = v2.legacy, v2.coordinate
upstream, prior, native, paired = v2.upstream, v2.prior, v2.native, v2.paired
require, record, write = v2.require, v2.record, legacy.write
plan, compose, frozen_parent = v2.plan, v2.compose, v2.frozen_parent
classify, original_branches, authenticate = v2.classify, v2.original_branches, v2.authenticate


def history_records():
    records = [record(ROOT / relative) for relative in HISTORY]
    for item, digest in zip(records, HISTORY.values(), strict=True):
        require(item["sha256"] == digest, f"producer-cone history changed: {item['path']}")
    return records


def check_contract(contract):
    history_records()
    base = json.loads(v2.CONTRACT.read_text())
    v2.check_contract(base)
    coordinate.check_contract(json.loads(coordinate.CONTRACT.read_text()))
    expected = {
        "diagnostic_id": ID, "inherits": str(v2.CONTRACT.relative_to(ROOT)),
        "inherits_sha256": HISTORY[str(v2.CONTRACT.relative_to(ROOT))],
        "preserved_fields": list(v2.PRESERVED_FIELDS),
        "reviewed_result": str(v2.REVIEWED / "result.json"),
        "reviewed_result_sha256": v2.REVIEWED_SHA,
        "retained_freeze_sha256": prior.FREEZE_SHA,
        "inputs": base["inputs"], "policy_id": base["policy_id"],
        "thresholds": base["thresholds"], "controls": base["controls"],
        "candidate_admitted": False, "policy_adopted": False,
        "successor_published": False, "normal_host_review": "REQUIRED",
        "rtl_invocations": 0, "native_L0_L8_invocations": 0,
        "accepted_prefix_replay": False, "scientific_result_claim": False,
        "cwd": str(ROOT), "command": COMMAND, "execution_command": EXECUTION_COMMAND,
        "interface": ["--check", "--execute --out FRESH_DIRECTORY"],
        "expected_tests": EXPECTED_TESTS,
        "native_control_dispatch_authorized": True,
        "dispatch_purpose": "isolated_non_admission_counterfactual_suffix",
        "attribution_question": "Which frozen L12 coordinate-62 inherited/O/down branches "
        "are conditionally sufficient to change the retained L13 S18 rejection, and "
        "does recomputing the inherited-coordinate L12 suffix preserve that rescue?",
        "suffix_extent": {"position": 0, "L12": 2, "L13": 13},
        "maximum_native_layer_invocations": 15,
        "layer_necessity": {
            "L12": "One isolated baseline to authenticate exact producer operands and "
            "gates, one inherited_native control to recompute reachable nonlinear effects.",
            "L13": "One unchanged 19-stage suffix with all gates for each of the 13 "
            "frozen producer, scratch, reviewed endpoint and inherited-native controls.",
        },
        "execution_boundary": "--check compiles actual repo sources, authenticates the "
        "pinned v2 closure and reviewed v3/L12 artifacts read-only, and executes exactly "
        "the focused v3 tests with zero native/RTL dispatch. --execute additionally "
        "permits only the declared 15 native layer invocations into a fresh versioned "
        "build directory. This delivery does not execute controls. A separate bounded "
        "Planner execution task and independent Host review are required.",
        "reproduction": "Run command once from cwd with the absolute interpreter and "
        "visible command-local PYTHONPATH. Exactly 30 tests must execute with zero "
        "failures/errors/skips. Historical suites are imported for origin checks only. "
        "The execution command is a fresh-output template, not queued work; use the "
        "operator-required durable runner for execution. No historical bare-command PASS.",
        "authentication": "Inherit every pinned v2 source/current-update, input, artifact, "
        "operand, I/Z/H state, own-empty-P0 FP16 KV, receipt and lineage check unchanged. "
        "Inherit v1 branch definitions, mapping, classification and claim boundary "
        "through the hash-pinned v2 contract. Original-input independently propagated "
        "binary64 global reference and all 19 stage gates remain unchanged. Recheck "
        "bound inputs, source origins, contracts and history at completion.",
    }
    require(type(contract) is dict and set(contract) == set(expected), "v3 contract fields changed")
    for key, value in expected.items():
        require(type(contract[key]) is type(value)
                and json.dumps(contract[key], sort_keys=True) == json.dumps(value, sort_keys=True),
                f"v3 contract mismatch: {key}")


def source_context():
    require(Path.cwd() == ROOT and os.environ.get("PYTHONPATH") == str(ROOT)
            and Path(sys.executable) == Path("/home/argustest/miniconda3/bin/python"),
            "use the published repo-bound command context")
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "v3 source origin mismatch")
    require(CONTRACT.resolve() == ROOT / "ace3/contracts/candidates/q24_s16_l12_coordinate62_producer_cone_v3.json",
            "v3 contract origin mismatch")
    importlib.import_module(MODULE)
    importlib.import_module(TEST_MODULE)
    origins = v2.source_context()
    require(MODULE in origins and TEST_MODULE in origins, "missing v3 module origins")
    return origins


def validate():
    origins = source_context()
    contract_record = record(CONTRACT)
    check_contract(json.loads(CONTRACT.read_text()))
    compiled = []
    for item in origins.values():
        if "path" in item and Path(item["path"]).suffix == ".py":
            path = Path(item["path"])
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
            compiled.append(item)
    with ExitStack() as stack:
        guards = [stack.enter_context(patch.object(owner, name, side_effect=AssertionError(
            "native dispatch forbidden during v3 check"))) for owner, name in (
                (upstream, "execute_layer"), (native, "stages"), (native, "execute_layers"),
                (coordinate, "execute_layer"), (legacy, "diagnose"), (legacy, "original_branches"))]
        data = authenticate()
        tests = importlib.import_module(TEST_MODULE)
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        require(count == EXPECTED_TESTS == tests.EXPECTED_TESTS, "focused test count mismatch")
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
        require(not any(guard.called for guard in guards), "native dispatch attempted during check")
    require(result.testsRun == count and result.wasSuccessful() and not result.skipped,
            "focused validation failed")
    coordinate.recheck(data["bound"].values())
    require(source_context() == origins and record(CONTRACT) == contract_record,
            "source/contract changed during check")
    return {
        "diagnostic_id": ID, "status": "VALIDATED_SOFTWARE_SURFACE",
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "command": COMMAND, "origins": origins, "compiled": compiled,
        "contract": contract_record, "history": history_records(),
        "reviewed_result": data["reviewed_result"],
        "retained_freeze": data["authentication"]["retained_freeze"],
        "authenticated_binding_count": len(data["bound"]),
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "native_control_dispatch_authorized": True, "native_layer_invocations": 0,
        "rtl_invocations": 0, "native_L0_L8_invocations": 0,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "normal_host_review": "REQUIRED", "accepted_L0_L8_tests_executed": False,
        "scientific_result_claim": False, "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "AUTHENTICATED_RETAINED_ONLY",
    }


class NativeControls:
    """Enforce the exact control order, suffix extent and invocation budget."""

    def __init__(self, data):
        self.data = data
        self.schedule = [("baseline", 12)]
        for label, _ in plan():
            if label == "inherited_native":
                self.schedule.append((label, 12))
            self.schedule.append((label, 13))
        self.count = 0

    def run(self, label, layer, parent):
        require(type(layer) is int and self.count < 15
                and self.schedule[self.count] == (label, layer), "native control outside bounded schedule")
        self.count += 1
        data = self.data
        source = data["layers"][12] if layer == 12 else data
        tensors = data["tensors12"] if layer == 12 else data["tensors"]
        return upstream.execute_layer(tensors, layer, parent, source["trajectory"], source["reference"])

    def finish(self):
        require(self.count == len(self.schedule) == 15, "incomplete native control schedule")


def diagnose(out, log, contract, validation):
    check_contract(contract)
    require(validation["diagnostic_id"] == ID
            and validation["status"] == "VALIDATED_SOFTWARE_SURFACE"
            and validation["collected"] == validation["executed"] == EXPECTED_TESTS
            and all(validation[key] == 0 for key in ("failures", "errors", "skipped",
                                                    "native_layer_invocations", "rtl_invocations"))
            and source_context() == validation["origins"]
            and record(CONTRACT) == validation["contract"], "invalid pre-dispatch validation")
    started = time.monotonic()
    legacy.torch.set_num_threads(1)
    data = authenticate()
    controls = NativeControls(data)
    artifacts, rows = [], []

    def document(name, value):
        write(out / name, value)
        item = record(out / name)
        artifacts.append(item)
        return item

    def save(name, arrays):
        item = prior.retained.save(out / name, arrays)
        artifacts.append(item)
        return item

    document("retained_authentication.json", data["authentication"])
    document("L9_L12_coordinate62.json", [
        {"layer": layer, "position": 0, **coordinate.compare_parents(
            item["output"], paired.mapped_parent(item["reference"]), item["reference"])[62]}
        for layer, item in data["layers"].items()])
    original = original_branches(data)
    save("L12_original_branches.npz", original)
    layer = data["layers"][12]
    actual, locals_, reports = controls.run("baseline", 12, layer["parent"])
    paired.same_arrays(actual, layer["arrays"])
    paired.same_arrays(locals_, layer["locals"])
    coordinate.check_reports(reports, layer["reports"])
    save("actual_L12.npz", actual)
    document("actual_L12_gates.json", reports)
    document("L12_coordinate62_decomposition.json", {
        **paired.parts(layer["parent"], actual, original, 62),
        "S12_actual_vs_original": coordinate.compare_parents(
            prior.retained.state_from(actual, "scratch", "stage12"),
            paired.mapped_parent(original["residual"]), original["residual"])[62],
        "branch_definitions": json.loads(v2.CONTRACT.read_text())["branch_definitions"],
        "original_values_hex": {key: float(value[62]).hex() for key, value in original.items()},
    })
    baseline_scratch = prior.retained.state_from(actual, "scratch", "stage12")
    for label, parts in plan():
        began = time.monotonic()
        l12_reports = None
        if label in ("mapped62", "mapped_all"):
            parent = coordinate.legacy.intervene(data["actual"], data["mapped"],
                                                [62] if label == "mapped62" else list(range(896)))
            cut = {"kind": "reviewed_output_coordinate_control", "parts": list(parts)}
        elif label == "inherited_native":
            incoming = coordinate.legacy.intervene(
                layer["parent"], paired.mapped_parent(original["input"]), [62])
            generated, _, l12_reports = controls.run(label, 12, incoming)
            save(label + "_L12.npz", generated)
            document(label + "_L12_gates.json", l12_reports)
            parent = prior.retained.state_from(generated, "output", "stage18")
            cut = {"kind": "native_L12_recomputed_from_inherited_coordinate62", "parts": list(parts)}
        else:
            if "scratch" in parts:
                scratch = coordinate.legacy.intervene(
                    baseline_scratch, paired.mapped_parent(original["residual"]), [62])
                down = actual["stage17"].copy()
                if "down" in parts:
                    down[62] = native.candidate.rne(legacy.torch.from_numpy(original["s17"].copy()))[62]
                parent = native.candidate.state.add(scratch, down)
                prior.retained.verify_parent(parent, prior.retained.transition_reference(scratch, down))
                vectors = {"scratch_" + key: value for key, value in scratch.items()}
            else:
                incoming, o, down, scratch, parent = frozen_parent(layer["parent"], actual, original, parts)
                vectors = {"input_" + key: value for key, value in incoming.items()}
                vectors.update({"scratch_" + key: value for key, value in scratch.items()})
                vectors["o"] = o
                if label == "actual":
                    paired.same_arrays(scratch, baseline_scratch)
                    prior.retained.verify_parent(parent, data["actual"])
            vectors["down"] = down
            save(label + "_L12_cut_operands.npz", vectors)
            for key in parent:
                unchanged = legacy.np.arange(896) != 62
                require(legacy.np.array_equal(parent[key][unchanged], data["actual"][key][unchanged]),
                        "frozen cut changed unselected coordinates")
            cut = {"kind": "exogenous_frozen_scalar_branch_cut", "parts": list(parts),
                   "exact_residual_transitions": "PASS", "L12_operator_repair_or_gate_pass_claimed": False}
        parent_record = save(label + "_L12_parent.npz", parent)
        arrays, _, suffix_reports = controls.run(label, 13, parent)
        if label == "actual":
            paired.same_arrays(arrays, data["actual_arrays"])
            paired.same_arrays(arrays, data["actual_reviewed"])
            coordinate.check_reports(suffix_reports, data["actual_reports"])
        elif label in ("mapped62", "mapped_all"):
            paired.same_arrays(arrays, data["single_062_reviewed" if label == "mapped62" else "mapped_reviewed"])
            if label == "mapped_all":
                coordinate.check_reports(suffix_reports, data["mapped_reports"])
        save(label + "_L13.npz", arrays)
        gates = document(label + "_L13_gates.json", suffix_reports)
        scalar = prior.measure(int(arrays["stage18"][62]), float(data["reference"][62]))
        failures = [item["index"] for item in suffix_reports[18]["binary64_v1"]["failures"]]
        require(scalar["accepted"] == (62 not in failures), "scalar/full-vector gate disagreement")
        if label == "actual":
            require(failures == [62], "retained failure set changed")
        row = {
            "label": label, "cut": cut, "parent": parent_record, "gates": gates, "index62": scalar,
            "L12_output_Q24_units62": int(parent["i"][62]),
            "L12_output_minus_actual_Q24_units62": int(parent["i"][62]) - int(data["actual"]["i"][62]),
            "L12_output_FP16_bits62": f"{int(parent['h'][62]):04x}",
            "L13_S18_failure_indices": failures,
            "L13_S18_changed_indices": legacy.np.flatnonzero(
                arrays["stage18"] != data["actual_arrays"]["stage18"]).tolist(),
            "mandatory_statuses": [report["status"] for report in suffix_reports],
            "all_L13_gates_pass": all(report["status"] == "PASS" for report in suffix_reports),
            "native_L12_mandatory_statuses": None if l12_reports is None else [
                report["status"] for report in l12_reports],
            "L13_source_operand_state_KV_RTZ_checks": "PASS", "candidate_admitted": False,
            "seconds": time.monotonic() - began,
        }
        rows.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
    controls.finish()
    document("interventions.json", rows)
    by_name = {row["label"]: row for row in rows}
    base = by_name["actual"]["L12_output_Q24_units62"]
    interactions = {}
    for label, parts in plan()[:8]:
        if len(parts) > 1:
            value = by_name[label]["L12_output_Q24_units62"] - base
            value -= sum(by_name["frozen_" + branch]["L12_output_Q24_units62"] - base for branch in parts)
            require(value == 0, "frozen exact additive decomposition did not close")
            interactions[label] = value
    document("frozen_Q24_additive_interactions.json", interactions)
    coordinate.recheck(data["bound"].values())
    coordinate.recheck(validation["history"])
    require(source_context() == validation["origins"] and record(CONTRACT) == validation["contract"],
            "source/contract changed during execution")
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [12, 0, 18], "index": 62,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "rtl_invocations": 0, "native_L0_L8_invocations": 0, "native_layer_invocations": controls.count,
        "normal_host_review": "REQUIRED", "retained_status": "FAIL",
        "control_count": len(rows), "native_retained_bitwise_reproduction": True,
        "reviewed_coordinate_rescue_reproduced": True, "original_L12_S18_bitwise_reproduction": True,
        "unchanged_baseline_gates": True, "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS", "origins_after_execution": validation["origins"],
        "input_bindings": list(data["bound"].values()), "artifacts": artifacts,
        "all_L13_gate_passing_controls": [row["label"] for row in rows if row["all_L13_gates_pass"]],
        "timing_seconds": {"total": time.monotonic() - started},
        "claim_boundary": json.loads(v2.CONTRACT.read_text())["claim_boundary"], **classify(rows),
    }


def output_path(value):
    out = value.absolute()
    require(out == out.resolve() and out.parent == ROOT / "build"
            and out.name.startswith(OUTPUT_PREFIX) and len(out.name) > len(OUTPUT_PREFIX),
            "output outside fresh v3 build scope")
    require(not out.exists() and not out.is_symlink(), "output already exists")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.check and args.out is not None or args.execute and args.out is None:
        parser.error("--out is required only with --execute")
    if args.check:
        print(json.dumps(validate(), sort_keys=True), flush=True)
        return 0
    out = output_path(args.out)
    validation = validate()
    out.mkdir(exist_ok=False)
    write(out / "validation.json", validation)
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({"cwd": str(Path.cwd()), "executable": sys.executable,
                              "argv": sys.argv, "PYTHONPATH": os.environ.get("PYTHONPATH")}) + "\n")
        log.flush()
        result = diagnose(out, log, json.loads(CONTRACT.read_text()), validation)
    result["validation"] = record(out / "validation.json")
    write(out / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
