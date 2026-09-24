"""Non-admitting, input-only Q24 coordinate-62 sensitivity at L9/P0.

From /home/argustest/ace3-argus use PYTHONPATH=/home/argustest/ace3-argus
PYTHONDONTWRITEBYTECODE=1 /home/argustest/miniconda3/bin/python -B -m
ace3.model.candidates.diagnose_q24_s16_l9_coordinate62_input_threshold_v1.
--check compiles and tests only. For a separately authorized sweep, --out
accepts build/q24_s16_l9_coordinate62_input_threshold_... or
build/q24_s16_l9_coordinate62_wire_selected_parent_... and validates once
before the fixed non-admitting L9-L13 CPU sweep in a fresh exclusive directory.
Runs over two minutes require the durable runner.
"""

import argparse
from contextlib import redirect_stderr
from fractions import Fraction
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_runtime_v1 as runtime


entry = runtime.entry
require = entry.require
ROOT = runtime.ROOT
ID = "ace3-q24-s16-l9-coordinate62-input-threshold-v1"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l9_coordinate62_input_threshold_v1"
TEST_MODULE = "tests.test_q24_s16_l9_coordinate62_input_threshold_v1"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l9_coordinate62_input_threshold_v1.json"
REVIEWED = (
    ("q24_s16_l9_coordinate62_producer_cone_a38731cbbd8f_attempt001",
     "c36ed0cb236b06b2fd72e2f73502f38f022e55c7fb67bf0f0b2fc705915539d4", entry.ID),
    ("q24_s16_l10_coordinate62_producer_cone_3e4cf1c3fcd2_attempt001",
     "8830027d6e37ec77ad6d0a9b9815cedef0925d962b747fd38f84071e32f415cd",
     "ace3-q24-s16-l10-coordinate62-producer-cone-v1"),
)
LAYERS = (9, 10, 11, 12, 13)
OUTPUT_PREFIXES = (
    "q24_s16_l9_coordinate62_input_threshold_",
    "q24_s16_l9_coordinate62_wire_selected_parent_",
)


def plan():
    return [
        {"label": f"input_{step}_of_8", "step": step, "offset_q24": 0}
        for step in range(9)
    ] + [
        {"label": f"input_{step}_of_8_{label}", "step": step, "offset_q24": offset}
        for step in (0, 8) for label, offset in (("minus_unit", -1), ("plus_unit", 1))
    ]


def input_parent(actual, mapped, control):
    require(control in plan(), "unknown input-Q24 control")
    entry.prior.retained.verify_parent(actual, actual)
    entry.prior.retained.verify_parent(mapped, mapped)
    step, offset = control["step"], control["offset_q24"]
    value = Fraction((8 - step) * int(actual["i"][62]) + step * int(mapped["i"][62]), 8)
    integer = round(value) + offset
    require(-(1 << 63) <= integer < (1 << 63), "input Q24 integer overflow")
    if step in (0, 8):
        endpoint = actual if step == 0 else mapped
        negative = int(endpoint["i"][62]) < 0 or bool(endpoint["z"][62])
    else:
        negative = value < 0 or (value == 0 and bool(actual["z"][62]) and bool(mapped["z"][62]))
    tag = int(integer == 0 and negative)
    word = entry.prior.rational.project(integer, tag)
    require(word & 0x7c00 != 0x7c00, "nonfinite input projection")
    result = {key: values.copy() for key, values in actual.items()}
    result["i"][62], result["z"][62], result["h"][62] = integer, tag, word
    entry.prior.retained.verify_parent(result, result)
    return result


def authenticate():
    data = entry.authenticate()
    historical = []
    for directory, digest, diagnostic_id in REVIEWED:
        binding = entry.record(ROOT / "build" / directory / "result.json")
        require(binding["sha256"] == digest, "wrong reviewed input-threshold evidence")
        path = entry.upstream.bind_input(binding, data["bound"])
        reviewed = json.loads(path.read_text())
        require(reviewed["diagnostic_id"] == diagnostic_id and reviewed["status"] == "DIAGNOSED"
                and reviewed["native_retained_bitwise_reproduction"] is True
                and reviewed["unchanged_baseline_gates"] is True
                and reviewed["original_global_reference_unchanged"] is True
                and reviewed["source_operand_state_KV_lineage_checks"] == "PASS"
                and reviewed["rtl_invocations"] == 0
                and all(reviewed[key] is False for key in
                        ("candidate_admitted", "policy_adopted", "successor_published")),
                "reviewed input-threshold evidence scope mismatch")
        for item in reviewed["origins_after_execution"].values():
            if "path" in item:
                entry.upstream.bind_input(item, data["bound"])
        for item in reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]:
            entry.upstream.bind_input(item, data["bound"])
        policy = authenticate_policy(reviewed, data["bound"])
        historical.append({"result": binding, "read_only": True, "policy_binding": policy})
    data["input_threshold_evidence"] = historical
    return data


def authenticate_policy(reviewed, bound):
    """Missing legacy metadata needs a frozen validation-to-contract binding."""
    expected = entry.prior.gates.POLICY_ID
    if "policy_id" in reviewed:
        require(reviewed["policy_id"] == expected, "reviewed policy mismatch")
    validation_path = entry.upstream.bind_input(reviewed["validation"], bound)
    validation = json.loads(validation_path.read_text())
    contract_binding = validation.get("contract")
    require("policy_id" in reviewed or contract_binding is not None,
            "missing reviewed policy contract binding")
    if contract_binding is not None:
        contract_path = entry.upstream.bind_input(contract_binding, bound)
        contract = json.loads(contract_path.read_text())
        require(contract["diagnostic_id"] == reviewed["diagnostic_id"]
                and contract["policy_id"] == expected, "reviewed contract policy mismatch")
    gate_bindings = [item for item in reviewed["input_bindings"]
                     if item["path"] == str(entry.prior.gates.CONTRACT)]
    require(len(gate_bindings) == 1, "missing or duplicate reviewed gate policy binding")
    gate_path = entry.upstream.bind_input(gate_bindings[0], bound)
    require(json.loads(gate_path.read_text())["policy_id"] == expected,
            "reviewed gate policy mismatch")
    return {
        "policy_id": expected, "top_level_present": "policy_id" in reviewed,
        "validation": reviewed["validation"], "contract": contract_binding,
        "gate": gate_bindings[0],
    }


def execute_layer(layer, position, parent, data):
    require(type(layer) is int and layer in LAYERS
            and type(position) is int and position == 0,
            "native dispatch restricted to L9-L13/P0; L0-L8 forbidden")
    item = data["layers"][layer]
    if "native_dispatches" in data:
        data["native_dispatches"].append({"layer": layer, "position": position})
    return entry.upstream.execute_layer(
        item["tensors"], layer, parent, item["trajectory"], item["reference"])


def execute_control(control, actual, mapped, data):
    started = time.monotonic()
    parent = input_parent(actual, mapped, control)
    rows = []
    for layer in LAYERS:
        began = time.monotonic()
        arrays, locals_, reports = execute_layer(layer, 0, parent, data)
        seconds = time.monotonic() - began
        if control == plan()[0]:
            retained = data["layers"][layer]
            entry.paired.same_arrays(arrays, retained["arrays"])
            entry.paired.same_arrays(locals_, retained["locals"])
            entry.coordinate.check_reports(reports, retained["reports"])
        output = entry.prior.retained.state_from(arrays, "output", "stage18")
        rows.append({
            "layer": layer, "input_parent": parent, "output_parent": output,
            "arrays": arrays, "local_references": locals_, "reports": reports,
            "index62": entry.prior.measure(
                int(output["h"][62]), float(data["layers"][layer]["reference"][62])),
            "candidate_admitted": False,
            "native_and_gates_seconds": seconds,
        })
        parent = output
    return {"control": control, "layers": rows, "candidate_admitted": False,
            "seconds": time.monotonic() - started}


def threshold_intervals(rows, layer):
    require(type(layer) is int and layer in LAYERS, "threshold layer outside L9-L13")
    require([row["control"] for row in rows] == plan(), "incomplete or reordered sweep")
    sampled = []
    for row in rows[:9]:
        require([item["layer"] for item in row["layers"]] == list(LAYERS),
                "incomplete threshold layer evidence")
        accepted = row["layers"][layer - 9]["index62"]["accepted"]
        require(type(accepted) is bool, "missing scalar threshold decision")
        sampled.append(accepted)
    return [
        {"left_step": step, "right_step": step + 1, "denominator": 8,
         "left_accepted": sampled[step], "right_accepted": sampled[step + 1]}
        for step in range(8) if sampled[step] != sampled[step + 1]
    ]


def execute_sweep(out=None, log=None):
    """Fixed CPU control execution; never called by --check."""
    started = time.monotonic()
    origins = source_context()
    contract_binding = entry.record(CONTRACT)
    entry.torch.set_num_threads(1)
    data = authenticate()
    timing = {"authentication": time.monotonic() - started}
    data["native_dispatches"] = []
    began = time.monotonic()
    original = entry.original_branches(data)
    timing["original_input_L9_reference"] = time.monotonic() - began
    actual = data["layers"][9]["parent"]
    mapped = entry.paired.mapped_parent(original["input"])
    rows, artifacts = [], []
    for control in plan():
        row = execute_control(control, actual, mapped, data)
        incoming = row["layers"][0]["input_parent"]
        row["input_Q24_units62"] = int(incoming["i"][62])
        row["input_delta_Q24_units62"] = int(incoming["i"][62]) - int(actual["i"][62])
        row["input_FP16_bits62"] = f"{int(incoming['h'][62]):04x}"
        for item in row["layers"]:
            require(len(item["reports"]) == 19 and all(
                report["node"] == [item["layer"], 0, stage]
                and report["policy_id"] == entry.prior.gates.POLICY_ID
                and report["residual_state_lineage"] == report["kv_lineage"] == "PASS"
                for stage, report in enumerate(item["reports"])), "incomplete native gate evidence")
            item["all_gates_pass"] = all(r["status"] == "PASS" for r in item["reports"])
            if out is not None:
                path = out / f"{control['label']}_L{item['layer']}.npz"
                arrays = {f"{kind}__{key}": value for kind in
                          ("input_parent", "output_parent", "arrays", "local_references")
                          for key, value in item[kind].items()}
                with path.open("xb") as stream:
                    entry.np.savez_compressed(stream, **arrays)
                binding = entry.record(path)
                artifacts.append(binding)
                for kind in ("input_parent", "output_parent", "arrays", "local_references"):
                    del item[kind]
                item["state_operand_KV_local_archive"] = binding
        row["all_L9_L13_gates_pass"] = all(item["all_gates_pass"] for item in row["layers"])
        rows.append(row)
        if log is not None:
            log.write(json.dumps({
                "control": control, "input_delta_Q24_units62": row["input_delta_Q24_units62"],
                "L13_index62": row["layers"][-1]["index62"], "seconds": row["seconds"],
                "native_layer_count": len(data["native_dispatches"]),
            }) + "\n")
            log.flush()
    baseline = rows[0]["layers"][-1]["index62"]
    require(baseline["actual_fp16_bits"] == "6630" and baseline["accepted"] is False,
            "retained L13 coordinate62 rejection changed")
    expected_dispatches = [{"layer": layer, "position": 0} for _ in plan() for layer in LAYERS]
    require(data["native_dispatches"] == expected_dispatches, "native dispatch count/scope mismatch")
    began = time.monotonic()
    runtime.reauthenticate(data)
    require(source_context() == origins, "source changed during input sweep")
    require(entry.record(CONTRACT) == contract_binding, "contract changed during input sweep")
    timing["reauthentication"] = time.monotonic() - began
    timing["native_controls_and_gates"] = sum(
        item["native_and_gates_seconds"] for row in rows for item in row["layers"])
    timing["total"] = time.monotonic() - started
    flips = [row for row in rows if row["layers"][-1]["index62"]["accepted"]
             and row["layers"][-1]["index62"]["actual_fp16_bits"] == "662f"]
    ordered = sorted(rows, key=lambda row: row["input_Q24_units62"])
    brackets = [
        {"left_control": left["control"]["label"], "right_control": right["control"]["label"],
         "left_Q24_units": left["input_Q24_units62"], "right_Q24_units": right["input_Q24_units62"],
         "width_Q24_units": right["input_Q24_units62"] - left["input_Q24_units62"],
         "left_index62": left["layers"][-1]["index62"],
         "right_index62": right["layers"][-1]["index62"]}
        for left, right in zip(ordered, ordered[1:])
        if left["layers"][-1]["index62"]["accepted"] != right["layers"][-1]["index62"]["accepted"]
    ]
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "controls": rows,
        "policy_id": entry.prior.gates.POLICY_ID, "contract": contract_binding,
        "control_count": len(rows), "native_layer_count": len(data["native_dispatches"]),
        "native_execution_records": data["native_dispatches"], "timing_seconds": timing,
        "origins_after_execution": origins, "artifacts": artifacts,
        "reviewed_policy_bindings": data["input_threshold_evidence"],
        "native_retained_bitwise_reproduction": True,
        "unchanged_baseline_gates": True, "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS",
        "smallest_observed_flip_absolute_Q24_units": min(
            (abs(row["input_delta_Q24_units62"]) for row in flips), default=None),
        "L13_all_sample_transition_brackets": brackets,
        "smallest_observed_transition_width_Q24_units": min(
            (row["width_Q24_units"] for row in brackets), default=None),
        "all_L9_L13_gate_passing_controls": [
            row["control"]["label"] for row in rows if row["all_L9_L13_gates_pass"]],
        "sampled_threshold_intervals": {
            str(layer): threshold_intervals(rows, layer) for layer in LAYERS},
        "input_bindings": list(data["bound"].values()),
        "metadata_only_external_review": data["L8_review_binding"],
        "normal_host_review": "REQUIRED", "candidate_admitted": False,
        "policy_adopted": False, "successor_published": False,
        "native_L0_L8_invocations": 0, "rtl_invocations": 0,
        "accepted_L0_L8_execution": False,
        "claim_boundary": (
            "One fixed 13-control L9-L13/P0 CPU-software sensitivity sweep; all failures retained. "
            "Observed brackets are not exact thresholds; no monotonicity or unique cause claimed. "
            "Accepted L0-L8 is read-only. Original-input global references and numerical gates "
            "are unchanged. Q24 residuals are wider than FP16; INT4 weights, FP16 operator/KV "
            "boundaries and S16 RTZ are unchanged. No admission, policy adoption, successor, "
            "strict-FP16-state W4A16, L14+, new-token, full-model PASS, RTL, hardware, GPU, FPGA, "
            "simulation, synthesis or PPA claim. Independent Host review remains required."
        ),
    }


def source_context():
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "input-threshold source origin mismatch")
    importlib.import_module(TEST_MODULE)
    origins = runtime.source_context()
    require(MODULE in origins and TEST_MODULE in origins, "missing input-threshold origins")
    return origins


def validate():
    origins = source_context()
    compiled = []
    for name in (MODULE, TEST_MODULE):
        path = Path(origins[name]["path"])
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(str(path))
    require(CONTRACT.resolve().is_relative_to(ROOT), "contract outside repository")
    contract = json.loads(CONTRACT.read_text())
    require(contract["diagnostic_id"] == ID and contract["controls"] == plan()
            and contract["native_layers"] == list(LAYERS)
            and contract["policy_id"] == entry.prior.gates.POLICY_ID
            and contract["normal_host_review"] == "REQUIRED"
            and contract["output_prefixes"] == list(OUTPUT_PREFIXES)
            and all(contract["inputs"][key] == f"{directory}/result.json; sha256 {digest}"
                    for key, (directory, digest, _) in
                    zip(("reviewed_L9", "reviewed_L10"), REVIEWED))
            and contract["max_native_layer_evaluations"] == len(plan()) * len(LAYERS)
            and all(contract[key] is False for key in
                    ("candidate_admitted", "policy_adopted", "successor_published",
                     "scientific_result_claim", "accepted_L0_L8_execution")),
            "input-threshold contract mismatch")
    tests = sys.modules[TEST_MODULE]
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    with patch.object(entry.native, "stages",
                      side_effect=AssertionError("native execution forbidden during validation")) as stages:
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
        require(not stages.called, "native execution attempted during validation")
    require(count == tests.EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")
    require(source_context() == origins, "source changed during validation")
    return {
        "status": "VALIDATED_SOFTWARE_ONLY", "diagnostic_id": ID,
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": str(ROOT),
        "origins": origins, "compiled": compiled, "collected": count,
        "contract": entry.record(CONTRACT),
        "reviewed_L9": entry.record(ROOT / "build" / REVIEWED[0][0] / "result.json"),
        "authentication_scope": "Focused selected L9 evidence; not full sweep readiness",
        "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "native_layer_invocations": 0, "accepted_L0_L8_tests_executed": False,
        "native_L0_L8_invocations": 0, "rtl_invocations": 0,
        "scientific_result_claim": False, "normal_host_review": "REQUIRED",
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
    }


def output_directory(path):
    out = path.resolve()
    require(out.parent == ROOT / "build" and out.name.startswith(OUTPUT_PREFIXES),
            "output outside bounded build scope")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.check:
        print(json.dumps(validate()))
        return 0
    out = output_directory(args.out)
    out.mkdir(exist_ok=False)
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({
            "cwd": str(Path.cwd()), "executable": sys.executable, "argv": sys.argv,
            "PYTHONPATH": entry.os.environ.get("PYTHONPATH"),
            "PYTHONDONTWRITEBYTECODE": entry.os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "loadavg": entry.os.getloadavg(), "cpu_affinity": sorted(entry.os.sched_getaffinity(0)),
            "torch_threads_before": entry.torch.get_num_threads(),
        }) + "\n")
        log.flush()
        try:
            with redirect_stderr(log):
                validation = validate()
            entry.write(out / "validation.json", validation)
            native_stages = entry.native.stages
            native_records = []

            def guarded_stages(tensors, layer, parent, arrays):
                require(type(layer) is int and layer in LAYERS,
                        "native stage execution outside L9-L13")
                native_records.append({"layer": layer, "position": 0})
                return native_stages(tensors, layer, parent, arrays)

            with patch.object(entry.native, "stages", side_effect=guarded_stages):
                result = execute_sweep(out, log)
            require(native_records == result["native_execution_records"],
                    "native stage/dispatch accounting mismatch")
            result["native_stage_execution_records"] = native_records
            require(result["origins_after_execution"] == validation["origins"]
                    and result["contract"] == validation["contract"],
                    "source/contract changed after validation")
            validation["sweep_execution"] = {
                "native_L0_L8_invocations": result["native_L0_L8_invocations"],
                "rtl_invocations": result["rtl_invocations"],
                "native_layer_count": result["native_layer_count"],
                "native_execution_records": result["native_execution_records"],
                "native_stage_execution_records": native_records,
            }
            entry.write(out / "validation.json", validation)
            result["validation"] = entry.record(out / "validation.json")
            entry.write(out / "result.json", result)
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, AssertionError):
            traceback.print_exc(file=log)
            raise
        summary = {"status": result["status"], "result": str(out / "result.json")}
        log.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
