"""Tighter L9/P0 coordinate-62 refinement of frozen v2 evidence.

Run COMMAND from ROOT. This compiles repository sources and runs only the v3
tests; accepted L0-L8 tests are imported for origin binding, never executed.
Separately authorized --out runs the 17 native_cpu controls at L9-L13/P0 in a
fresh versioned build directory. Use the durable runner for long commands.
Implementation validation alone publishes no scientific result or admission.
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

from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_threshold_refine_v2 as prior


entry = prior.entry
base = prior.base
require = entry.require
ROOT = prior.ROOT
ID = "ace3-q24-s16-l9-coordinate62-threshold-refine-v3"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l9_coordinate62_threshold_refine_v3"
TEST_MODULE = "tests.test_q24_s16_l9_coordinate62_threshold_refine_v3"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l9_coordinate62_threshold_refine_v3.json"
OBSERVED = (
    "q24_s16_l9_coordinate62_threshold_refine_v2_e665bc3c75ac_publication001",
    "97453fef54f86025970da406b0bc5c6bd9c13bdd4643531fdbc48db4da08eead",
    prior.ID,
)
PASSING, FAILING = 26501847707, 26501851672
LAYERS = (9, 10, 11, 12, 13)
EXPECTED_TESTS = 25
OUTPUT_PREFIX = "q24_s16_l9_coordinate62_threshold_refine_v3_"
COMMAND = (
    f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 "
    f"/home/argustest/miniconda3/bin/python -B -m {MODULE} --check"
)
zero_dispatch = prior.zero_dispatch
native_execution = prior.native_execution


def plan():
    return [
        {"label": f"refine_{step}_of_16",
         "input_Q24_units62": round(Fraction((16 - step) * PASSING + step * FAILING, 16))}
        for step in range(17)
    ]


def execution_plan(layers=LAYERS, position=0, backend="native_cpu"):
    require(backend == "native_cpu", "execution backend must be native_cpu; RTL forbidden")
    require(type(position) is int and position == 0, "execution position restricted to P0")
    require(isinstance(layers, (list, tuple))
            and all(type(layer) is int for layer in layers) and tuple(layers) == LAYERS,
            "execution layers restricted to L9-L13; native L0-L8 forbidden")
    return [{"layer": layer, "position": position} for _ in plan() for layer in layers]


def execution_contract():
    return {
        "backend": "native_cpu", "layers": list(LAYERS), "position": 0,
        "native_layer_evaluations": 85, "output_prefix": OUTPUT_PREFIX,
        "fresh_exclusive_directory": True,
        "command": COMMAND.removesuffix("--check") + f"--out build/{OUTPUT_PREFIX}<name>",
    }


def input_parent(actual, control):
    require(control in plan() and type(control["input_Q24_units62"]) is int,
            "unknown refinement control")
    entry.prior.retained.verify_parent(actual, actual)
    parent = {key: value.copy() for key, value in actual.items()}
    parent["i"][62], parent["z"][62] = control["input_Q24_units62"], 0
    parent["h"][62] = entry.prior.rational.project(control["input_Q24_units62"], 0)
    entry.prior.retained.verify_parent(parent, parent)
    return parent


def check_contract(contract):
    values = [row["input_Q24_units62"] for row in plan()]
    require(len(set(values)) == 17 and all(
        0 < b - a <= 256 for a, b in zip(values, values[1:])), "invalid refinement spacing")
    expected = {
        "diagnostic_id": ID, "policy_id": entry.prior.gates.POLICY_ID,
        "cwd": str(ROOT), "command": COMMAND, "observed_sweep": list(OBSERVED),
        "observed_passing_Q24_units62": PASSING, "observed_failing_Q24_units62": FAILING,
        "coordinate": 62, "position": 0, "prospective_layers": list(LAYERS),
        "controls": plan(), "maximum_adjacent_spacing_Q24_units": 256,
        "expected_focused_tests": EXPECTED_TESTS, "check_only": True,
        "native_L0_L8_invocations": 0, "native_L9_L13_invocations": 0, "rtl_invocations": 0,
        "planned_native_L0_L8_invocations": 0, "planned_native_L9_L13_invocations": 85,
        "planned_rtl_invocations": 0, "accepted_L0_L8_execution": False,
        "scientific_result_claim": False, "candidate_admitted": False,
        "policy_adopted": False, "successor_published": False,
        "normal_host_review": "REQUIRED", "execution": execution_contract(),
        "thresholds": json.loads(prior.CONTRACT.read_text())["thresholds"],
    }
    for key, value in expected.items():
        require(key in contract and type(contract[key]) is type(value)
                and contract[key] == value, f"refinement contract mismatch: {key}")


def source_context():
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "refinement source origin mismatch")
    require(CONTRACT.resolve() == ROOT / "ace3/contracts/candidates"
            / "q24_s16_l9_coordinate62_threshold_refine_v3.json",
            "refinement contract origin mismatch")
    importlib.import_module(TEST_MODULE)
    origins = prior.source_context()
    require(MODULE in origins and TEST_MODULE in origins, "missing refinement origins")
    return origins


def check_observation(observed, data, origins):
    expected = {
        "status": "DIAGNOSED", "policy_id": entry.prior.gates.POLICY_ID,
        "control_count": 17, "native_layer_count": 85, "native_L9_L13_invocations": 85,
        "observed_endpoint_bitwise_reproduction": True,
        "source_operand_state_KV_lineage_checks": "PASS",
        "native_L0_L8_invocations": 0, "rtl_invocations": 0,
        "accepted_L0_L8_execution": False, "candidate_admitted": False,
        "policy_adopted": False, "successor_published": False,
        "normal_host_review": "REQUIRED", "exact_threshold_claim": False,
        "monotonicity_claim": False,
    }
    for key, value in expected.items():
        require(key in observed and type(observed[key]) is type(value)
                and observed[key] == value, f"observed metadata mismatch: {key}")
    saved_origins = observed["origins_after_execution"]
    for name, item in saved_origins.items():
        require(name in origins and item == origins[name], f"observed source mismatch: {name}")
    require(all(name in saved_origins for name in (
        prior.MODULE, prior.TEST_MODULE, base.MODULE, entry.MODULE,
        "tests.test_q24_s16_toward_zero_l3_l8_v1")), "missing observed source bindings")
    require(observed["contract"] == entry.record(prior.CONTRACT), "observed contract mismatch")
    require(observed["reviewed_policy_bindings"] == data["input_threshold_evidence"]
            and observed["v1_refinement_evidence"] == data["v1_refinement_evidence"],
            "observed selected-parent/history linkage mismatch")
    reference = base.validation_evidence(json.loads(prior.CONTRACT.read_text()), data)
    for key, value in reference.items():
        require(observed[key] == value, f"observed reference binding mismatch: {key}")
    require(observed["native_execution_records"] == prior.execution_plan()
            and observed["native_stage_execution_records"] == prior.execution_plan(),
            "observed dispatch accounting mismatch")
    require([row["control"] for row in observed["controls"]] == prior.plan(),
            "observed controls mismatch")
    for row in observed["controls"]:
        require(row["input_Q24_units62"] == row["control"]["input_Q24_units62"],
                "observed control input mismatch")
        require([item["layer"] for item in row["layers"]] == list(LAYERS),
                "observed layer sequence mismatch")
        for item in row["layers"]:
            reports, layer = item["reports"], item["layer"]
            require(len(reports) == 19 and all(
                report["node"] == [layer, 0, stage]
                and report["policy_id"] == entry.prior.gates.POLICY_ID
                and report["kv_lineage"] == report["residual_state_lineage"] == "PASS"
                for stage, report in enumerate(reports)), "observed state/KV/gate mismatch")
            require(item["all_gates_pass"] is all(r["status"] == "PASS" for r in reports),
                    "observed layer gate summary mismatch")
        require(row["all_L9_L13_gates_pass"] is all(
            item["all_gates_pass"] for item in row["layers"]), "observed gate summary mismatch")
    transitions = [
        {"left_Q24_units": left["input_Q24_units62"],
         "right_Q24_units": right["input_Q24_units62"],
         "width_Q24_units": right["input_Q24_units62"] - left["input_Q24_units62"],
         "left_index62": left["layers"][-1]["index62"],
         "right_index62": right["layers"][-1]["index62"]}
        for left, right in zip(observed["controls"], observed["controls"][1:])
        if left["layers"][-1]["index62"]["accepted"] != right["layers"][-1]["index62"]["accepted"]
    ]
    require(observed["L13_all_sample_transition_brackets"] == transitions
            and len(transitions) == 1
            and transitions[0]["left_Q24_units"] == PASSING
            and transitions[0]["right_Q24_units"] == FAILING
            and observed["smallest_observed_transition_width_Q24_units"] == FAILING - PASSING,
            "observed transition bracket mismatch")


def authenticate_observation(data):
    binding, observed = base.producer.read_result(OBSERVED, data["bound"])
    check_observation(observed, data, source_context())
    bind = lambda item: entry.upstream.bind_input(item, data["bound"])
    for item in list(observed["origins_after_execution"].values()) + observed[
            "input_bindings"] + observed["artifacts"] + [observed["contract"]]:
        if "path" in item:
            bind(item)
    # The reviewed publication uses size; native evidence uses bytes.
    publication_bindings = {}
    for key in ("validation", "raw_result", "raw_validation", "raw_command_log", "publication_source"):
        item = observed[key]
        publication_bindings[key] = {
            "path": item["path"], "bytes": item["size"], "sha256": item["sha256"]}
        bind(publication_bindings[key])
    validation = json.loads(bind(publication_bindings["validation"]).read_text())
    for key, value in base.validation_evidence(json.loads(prior.CONTRACT.read_text()), data).items():
        require(validation[key] == value, f"observed validation reference mismatch: {key}")
    require(validation["collected"] == validation["executed"] == prior.EXPECTED_TESTS
            and validation["failures"] == validation["errors"] == validation["skipped"] == 0,
            "observed focused validation mismatch")
    policy = base.legacy.authenticate_policy(
        {**observed, "validation": publication_bindings["validation"]}, data["bound"])
    endpoints = []
    for control, accepted, bits in ((plan()[0], True, "662f"), (plan()[-1], False, "6630")):
        integer = control["input_Q24_units62"]
        matches = [row for row in observed["controls"] if row["input_Q24_units62"] == integer]
        require(len(matches) == 1, "missing or duplicate observed endpoint")
        row = matches[0]
        previous = input_parent(data["layers"][9]["parent"], control)
        for item in row["layers"]:
            archive = item["state_operand_KV_local_archive"]
            require(archive in observed["artifacts"], "unbound endpoint archive")
            arrays = entry.native.load_state(archive, bind)
            parent = {key: arrays[f"input_parent__{key}"] for key in ("i", "z", "h")}
            output = {key: arrays[f"output_parent__{key}"] for key in ("i", "z", "h")}
            entry.prior.retained.verify_parent(parent, previous)
            entry.prior.retained.verify_parent(output, output)
            metric = entry.prior.measure(
                int(output["h"][62]), float(data["layers"][item["layer"]]["reference"][62]))
            require(item["index62"] == metric, "observed scalar reference mismatch")
            previous = output
        metric = row["layers"][-1]["index62"]
        require(metric["accepted"] is accepted and metric["actual_fp16_bits"] == bits
                and row["all_L9_L13_gates_pass"] is accepted, "observed endpoint decision mismatch")
        endpoints.append({"input_Q24_units62": integer, "observed_accepted": accepted,
                          "control": row["control"], "read_only": True})
    return {
        "result": binding, "read_only": True, "historical_status": observed["status"],
        "policy_binding": policy, "endpoints": endpoints,
        "publication_bindings": publication_bindings,
        "historical_v1_evidence": data["v1_refinement_evidence"],
    }, observed


def authenticate():
    data = prior.authenticate()
    data["v2_refinement_evidence"], observed = authenticate_observation(data)
    return data, observed


def execute_control(control, data, observed):
    started = time.monotonic()
    parent = input_parent(data["layers"][9]["parent"], control)
    endpoint = next((row for row in observed["controls"]
                     if row["input_Q24_units62"] == control["input_Q24_units62"]), None)
    rows = []
    for layer in LAYERS:
        began = time.monotonic()
        arrays, locals_, reports = base.legacy.execute_layer(layer, 0, parent, data)
        seconds = time.monotonic() - began
        output = entry.prior.retained.state_from(arrays, "output", "stage18")
        entry.prior.retained.verify_parent(output, output)
        require(len(reports) == 19 and all(
            report["node"] == [layer, 0, stage]
            and report["policy_id"] == entry.prior.gates.POLICY_ID
            and report["residual_state_lineage"] == report["kv_lineage"] == "PASS"
            for stage, report in enumerate(reports)), "incomplete native gate evidence")
        if endpoint is not None:
            frozen = endpoint["layers"][layer - 9]
            saved = entry.native.load_state(frozen["state_operand_KV_local_archive"],
                                           lambda item: entry.upstream.bind_input(item, data["bound"]))
            for kind, values in (("input_parent", parent), ("output_parent", output),
                                 ("arrays", arrays), ("local_references", locals_)):
                prefix = kind + "__"
                entry.paired.same_arrays(values, {
                    key.removeprefix(prefix): value for key, value in saved.items()
                    if key.startswith(prefix)})
            entry.coordinate.check_reports(reports, frozen["reports"])
        rows.append({
            "layer": layer, "input_parent": parent, "output_parent": output,
            "arrays": arrays, "local_references": locals_, "reports": reports,
            "index62": entry.prior.measure(
                int(output["h"][62]), float(data["layers"][layer]["reference"][62])),
            "all_gates_pass": all(report["status"] == "PASS" for report in reports),
            "native_and_gates_seconds": seconds, "candidate_admitted": False,
        })
        parent = output
    return {"control": control, "input_Q24_units62": control["input_Q24_units62"],
            "layers": rows, "all_L9_L13_gates_pass": all(row["all_gates_pass"] for row in rows),
            "seconds": time.monotonic() - started, "candidate_admitted": False}


def execute_sweep(out=None, log=None, *, layers=LAYERS, position=0, backend="native_cpu",
                  validation=None):
    records = execution_plan(layers, position, backend)
    started = time.monotonic()
    origins, contract_binding = source_context(), entry.record(CONTRACT)
    contract = json.loads(CONTRACT.read_text())
    check_contract(contract)
    if validation is not None:
        require(origins == validation["origins"] and contract_binding == validation["contract"],
                "source/contract changed before execution")
    with zero_dispatch():
        data, observed = authenticate()
    evidence = base.validation_evidence(contract, data)
    timing = {"authentication": time.monotonic() - started}
    entry.torch.set_num_threads(1)
    rows, artifacts = [], []
    with native_execution(records) as dispatches:
        for control in plan():
            row = execute_control(control, data, observed)
            for item in row["layers"]:
                if out is not None:
                    path = out / f"{control['label']}_L{item['layer']}.npz"
                    arrays = {f"{kind}__{key}": value for kind in (
                        "input_parent", "output_parent", "arrays", "local_references")
                        for key, value in item[kind].items()}
                    with path.open("xb") as stream:
                        entry.np.savez_compressed(stream, **arrays)
                    binding = entry.record(path)
                    artifacts.append(binding)
                    for kind in ("input_parent", "output_parent", "arrays", "local_references"):
                        del item[kind]
                    item["state_operand_KV_local_archive"] = binding
            rows.append(row)
            if log is not None:
                log.write(json.dumps({"control": control, "seconds": row["seconds"],
                                      "L13_index62": row["layers"][-1]["index62"]}) + "\n")
                log.flush()
        began = time.monotonic()
        base.legacy.runtime.reauthenticate(data)
        require(source_context() == origins and entry.record(CONTRACT) == contract_binding,
                "source/contract changed during refinement")
        timing["reauthentication"] = time.monotonic() - began
    brackets = [
        {"left_Q24_units": left["input_Q24_units62"],
         "right_Q24_units": right["input_Q24_units62"],
         "width_Q24_units": right["input_Q24_units62"] - left["input_Q24_units62"],
         "left_index62": left["layers"][-1]["index62"],
         "right_index62": right["layers"][-1]["index62"]}
        for left, right in zip(rows, rows[1:])
        if left["layers"][-1]["index62"]["accepted"] != right["layers"][-1]["index62"]["accepted"]
    ]
    timing["native_controls_and_gates"] = sum(
        item["native_and_gates_seconds"] for row in rows for item in row["layers"])
    timing["total"] = time.monotonic() - started
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "policy_id": entry.prior.gates.POLICY_ID,
        "controls": rows, "control_count": len(rows), "native_layer_count": len(records),
        **dispatches, **evidence, "contract": contract_binding,
        "origins_after_execution": origins, "input_bindings": list(data["bound"].values()),
        "artifacts": artifacts, "v2_refinement_evidence": data["v2_refinement_evidence"],
        "v1_refinement_evidence": data["v1_refinement_evidence"],
        "reviewed_policy_bindings": data["input_threshold_evidence"],
        "observed_endpoint_bitwise_reproduction": True,
        "source_operand_state_KV_lineage_checks": "PASS",
        "L13_all_sample_transition_brackets": brackets,
        "smallest_observed_transition_width_Q24_units": min(
            (row["width_Q24_units"] for row in brackets), default=None),
        "all_L9_L13_gate_passing_controls": [
            row["control"]["label"] for row in rows if row["all_L9_L13_gates_pass"]],
        "timing_seconds": timing, "native_L0_L8_invocations": 0,
        "native_L9_L13_invocations": len(records), "rtl_invocations": 0,
        "exact_threshold_claim": False, "monotonicity_claim": False,
        "accepted_L0_L8_execution": False, "candidate_admitted": False,
        "policy_adopted": False, "successor_published": False,
        "normal_host_review": "REQUIRED", "claim_boundary": contract["claim_boundary"],
    }


def validate():
    with zero_dispatch() as guards:
        origins = source_context()
        compiled = []
        for name in (MODULE, TEST_MODULE):
            path = Path(origins[name]["path"])
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
            compiled.append(origins[name])
        contract_binding = entry.record(CONTRACT)
        contract = json.loads(CONTRACT.read_text())
        check_contract(contract)
        tests = sys.modules[TEST_MODULE]
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        require(count == EXPECTED_TESTS, "unexpected focused test count")
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
        require(result.testsRun == count and result.wasSuccessful() and not result.skipped,
                "focused refinement validation failed")
        data = tests.ThresholdRefineTests.data
        evidence = base.validation_evidence(contract, data)
        require(source_context() == origins and entry.record(CONTRACT) == contract_binding,
                "source/contract changed during preflight")
    records = execution_plan()
    return {
        "status": "VALIDATED_SOFTWARE_PREFLIGHT_ONLY", "diagnostic_id": ID,
        "cwd": str(Path.cwd()), "executable": sys.executable,
        "PYTHONPATH": entry.os.environ["PYTHONPATH"], "command": COMMAND,
        "origins": origins, "compiled": compiled, "contract": contract_binding,
        "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "v2_refinement_evidence": data["v2_refinement_evidence"],
        "input_bindings": list(data["bound"].values()), **evidence,
        "controls": plan(), "execution": execution_contract(), "execution_plan": records,
        "maximum_observed_plan_spacing_Q24_units": max(
            b["input_Q24_units62"] - a["input_Q24_units62"] for a, b in zip(plan(), plan()[1:])),
        "planned_native_L0_L8_invocations": sum(row["layer"] <= 8 for row in records),
        "planned_native_L9_L13_invocations": len(records), "planned_rtl_invocations": 0,
        "native_L0_L8_invocations": guards[0].call_count,
        "native_L9_L13_invocations": guards[0].call_count,
        "native_stage_invocations": guards[1].call_count,
        "rtl_invocations": guards[2].call_count + guards[3].call_count,
        "accepted_L0_L8_tests_executed": False, "scientific_result_claim": False,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "normal_host_review": "REQUIRED", "claim_boundary": contract["claim_boundary"],
    }


def output_directory(path):
    require(not path.is_symlink(), "output must not be a symlink")
    out = path.resolve()
    require(out.parent == ROOT / "build" and out.name.startswith(OUTPUT_PREFIX)
            and len(out.name) > len(OUTPUT_PREFIX), "output outside versioned bounded build scope")
    require(not out.exists(), "output already exists")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(validate(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    out = output_directory(args.out)
    execution_plan()
    out.mkdir(exist_ok=False)
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({"cwd": str(Path.cwd()), "executable": sys.executable,
                              "argv": sys.argv if argv is None else [MODULE, *argv],
                              "PYTHONPATH": entry.os.environ.get("PYTHONPATH")}) + "\n")
        log.flush()
        try:
            with redirect_stderr(log):
                validation = validate()
            entry.write(out / "preflight_validation.json", validation)
            result = execute_sweep(out, log, validation=validation)
            validation["sweep_execution"] = {
                key: result[key] for key in (
                    "native_L0_L8_invocations", "native_L9_L13_invocations", "rtl_invocations",
                    "native_layer_count", "native_execution_records", "native_stage_execution_records")
            }
            entry.write(out / "validation.json", validation)
            result["validation"] = entry.record(out / "validation.json")
            entry.write(out / "result.json", result)
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, AssertionError):
            traceback.print_exc(file=log)
            raise
        print(json.dumps({"status": result["status"], "result": str(out / "result.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
