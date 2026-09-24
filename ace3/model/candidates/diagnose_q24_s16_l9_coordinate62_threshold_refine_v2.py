"""Versioned L9/P0 coordinate-62 refinement; --check never dispatches numerics.

The pinned v1 result remains BLOCKED at its original 63441-unit resolution.
Seventeen exact-integer controls refine its observed pass/fail interval, not
an assumed monotone or exact threshold. --out requires separate authorization,
a fresh versioned build directory and the durable runner for long commands.
Neither mode admits a policy, successor, strict-FP16 state or full-model PASS.
"""

import argparse
from contextlib import ExitStack, contextmanager, redirect_stderr
from fractions import Fraction
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_threshold_refine_v1 as prior


entry = prior.entry
base = prior.prior
require = entry.require
ROOT = Path("/home/argustest/ace3-argus")
ID = "ace3-q24-s16-l9-coordinate62-threshold-refine-v2"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l9_coordinate62_threshold_refine_v2"
TEST_MODULE = "tests.test_q24_s16_l9_coordinate62_threshold_refine_v2"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l9_coordinate62_threshold_refine_v2.json"
OBSERVED = (
    "q24_s16_l9_coordinate62_threshold_refine_v1_0b88a5ee2bc1_attempt001",
    "e60d80c0b9653bcb0e8b72a68007b8e29c1724b9e3889c08c10c5b2ffb2a5688",
    prior.ID,
)
PASSING, FAILING = 26501812021, 26501875462
LAYERS = (9, 10, 11, 12, 13)
EXPECTED_TESTS = 24
OUTPUT_PREFIX = "q24_s16_l9_coordinate62_threshold_refine_v2_"
COMMAND = (
    f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 "
    f"/home/argustest/miniconda3/bin/python -B -m {MODULE} --check"
)


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


def check_contract(contract):
    values = [row["input_Q24_units62"] for row in plan()]
    require(len(set(values)) == 17 and all(
        0 < b - a <= 4096 for a, b in zip(values, values[1:])), "invalid refinement spacing")
    expected = {
        "diagnostic_id": ID, "policy_id": entry.prior.gates.POLICY_ID,
        "cwd": str(ROOT), "command": COMMAND, "observed_sweep": list(OBSERVED),
        "observed_passing_Q24_units62": PASSING, "observed_failing_Q24_units62": FAILING,
        "position": 0, "coordinate": 62, "prospective_layers": list(LAYERS),
        "controls": plan(), "maximum_adjacent_spacing_Q24_units": 4096,
        "expected_focused_tests": EXPECTED_TESTS, "check_only": True,
        "native_L0_L8_invocations": 0, "native_L9_L13_invocations": 0, "rtl_invocations": 0,
        "scientific_result_claim": False, "candidate_admitted": False,
        "policy_adopted": False, "successor_published": False,
        "normal_host_review": "REQUIRED", "execution": execution_contract(),
        "thresholds": json.loads(prior.CONTRACT.read_text())["thresholds"],
    }
    for key, value in expected.items():
        require(key in contract and type(contract[key]) is type(value)
                and contract[key] == value, f"refinement contract mismatch: {key}")


def input_parent(actual, control):
    require(control in plan() and type(control["input_Q24_units62"]) is int,
            "unknown refinement control")
    entry.prior.retained.verify_parent(actual, actual)
    parent = {key: value.copy() for key, value in actual.items()}
    parent["i"][62], parent["z"][62] = control["input_Q24_units62"], 0
    parent["h"][62] = entry.prior.rational.project(control["input_Q24_units62"], 0)
    entry.prior.retained.verify_parent(parent, parent)
    return parent


def source_context():
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "refinement source origin mismatch")
    require(CONTRACT.resolve() == ROOT / "ace3/contracts/candidates" / (ID[5:].replace("-", "_") + ".json"),
            "refinement contract origin mismatch")
    importlib.import_module(TEST_MODULE)
    origins = prior.source_context()
    require(MODULE in origins and TEST_MODULE in origins, "missing refinement origins")
    return origins


def authenticate_observation(data):
    binding, observed = base.producer.read_result(OBSERVED, data["bound"])
    expected = {
        "status": "BLOCKED", "policy_id": entry.prior.gates.POLICY_ID,
        "control_count": 9, "native_layer_count": 45, "native_L9_L13_invocations": 45,
        "observed_endpoint_bitwise_reproduction": True,
        "unchanged_baseline_gates": True, "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS",
        "native_L0_L8_invocations": 0, "rtl_invocations": 0,
        "accepted_L0_L8_execution": False, "candidate_admitted": False,
        "policy_adopted": False, "successor_published": False, "normal_host_review": "REQUIRED",
    }
    for key, value in expected.items():
        require(key in observed and type(observed[key]) is type(value)
                and observed[key] == value, f"observed sweep metadata mismatch: {key}")
    require(observed["blocker"]["code"] == "fixed_plan_resolution_exceeds_objective",
            "historical blocker mismatch")
    origins = observed["origins_after_execution"]
    current = source_context()
    for name, item in origins.items():
        require(name in current and item == current[name], f"observed source mismatch: {name}")
    require(all(module.MODULE in origins for module in (prior, base, base.legacy, entry))
            and prior.TEST_MODULE in origins, "missing observed source bindings")
    bind = lambda item: entry.upstream.bind_input(item, data["bound"])
    for item in list(origins.values()) + observed["input_bindings"] + observed["artifacts"] + [
            observed["contract"], observed["validation"]]:
        if "path" in item:
            bind(item)
    require(observed["contract"] == entry.record(prior.CONTRACT), "observed contract mismatch")
    require(observed["reviewed_policy_bindings"] == data["input_threshold_evidence"]
            and observed["observed_bracket_evidence"] == data["observed_bracket_evidence"],
            "observed selected-parent linkage mismatch")
    policy = base.legacy.authenticate_policy(observed, data["bound"])
    validation = json.loads(bind(observed["validation"]).read_text())
    reference = base.validation_evidence(json.loads(prior.CONTRACT.read_text()), data)
    for key, value in reference.items():
        require(validation[key] == observed[key] == value,
                f"observed reference binding mismatch: {key}")
    require(observed["native_execution_records"] == prior.execution_plan()
            and observed["native_stage_execution_records"] == prior.execution_plan(),
            "observed dispatch accounting mismatch")
    require([row["control"] for row in observed["controls"]] == prior.plan(),
            "observed controls mismatch")
    endpoints = []
    for integer, accepted, bits in ((PASSING, True, "662f"), (FAILING, False, "6630")):
        matches = [row for row in observed["controls"] if row["input_Q24_units62"] == integer]
        require(len(matches) == 1, "missing or duplicate observed endpoint")
        row = matches[0]
        require([item["layer"] for item in row["layers"]] == list(LAYERS),
                "observed endpoint layer sequence mismatch")
        previous = input_parent(data["layers"][9]["parent"], plan()[0 if accepted else -1])
        for item in row["layers"]:
            reports, layer = item["reports"], item["layer"]
            require(len(reports) == 19 and all(
                report["node"] == [layer, 0, stage]
                and report["policy_id"] == entry.prior.gates.POLICY_ID
                and report["kv_lineage"] == report["residual_state_lineage"] == "PASS"
                for stage, report in enumerate(reports)), "observed state/KV/gate mismatch")
            require(item["all_gates_pass"] is all(r["status"] == "PASS" for r in reports),
                    "observed layer gate summary mismatch")
            archive = item["state_operand_KV_local_archive"]
            require(archive in observed["artifacts"], "unbound endpoint archive")
            arrays = entry.native.load_state(archive, bind)
            parent = {key: arrays[f"input_parent__{key}"] for key in ("i", "z", "h")}
            output = {key: arrays[f"output_parent__{key}"] for key in ("i", "z", "h")}
            entry.prior.retained.verify_parent(parent, previous)
            entry.prior.retained.verify_parent(output, output)
            previous = output
        metric = row["layers"][-1]["index62"]
        require(metric == entry.prior.measure(
            int(previous["h"][62]), float(data["layers"][13]["reference"][62])),
            "observed scalar reference mismatch")
        require(metric["accepted"] is accepted and metric["actual_fp16_bits"] == bits
                and row["all_L9_L13_gates_pass"] is accepted
                and row["all_L9_L13_gates_pass"] is all(
                    item["all_gates_pass"] for item in row["layers"]),
                "observed endpoint decision mismatch")
        endpoints.append({"input_Q24_units62": integer, "observed_accepted": accepted,
                          "control": row["control"], "read_only": True})
    brackets = observed["L13_all_sample_transition_brackets"]
    require(len(brackets) == 1 and brackets[0]["left_Q24_units"] == PASSING
            and brackets[0]["right_Q24_units"] == FAILING
            and brackets[0]["width_Q24_units"] == FAILING - PASSING,
            "observed transition bracket mismatch")
    return {"result": binding, "read_only": True, "historical_status": observed["status"],
            "historical_blocker": observed["blocker"], "policy_binding": policy,
            "endpoints": endpoints}


def authenticate():
    data = prior.authenticate()
    data["v1_refinement_evidence"] = authenticate_observation(data)
    return data


@contextmanager
def zero_dispatch():
    with ExitStack() as stack:
        guards = [
            stack.enter_context(patch.object(owner, name, side_effect=AssertionError(
                f"{name} forbidden during refinement preflight")))
            for owner, name in (
                (entry.upstream, "execute_layer"), (entry.native, "stages"),
                (subprocess, "Popen"), (entry.os, "system"), (Path, "mkdir"), (entry, "write"),
            )
        ]
        yield guards
        require(not any(guard.called for guard in guards),
                "native/RTL execution or publication attempted during preflight")


@contextmanager
def native_execution(records):
    require(records == execution_plan(), "execution plan mismatch")
    dispatches, stages = [], []
    native_layer, native_stages = entry.upstream.execute_layer, entry.native.stages

    def check_next(layer, seen):
        require(type(layer) is int and len(seen) < len(records)
                and records[len(seen)] == {"layer": layer, "position": 0},
                "native execution outside planned L9-L13/P0 route; L0-L8 forbidden")
        seen.append({"layer": layer, "position": 0})

    def guarded_layer(tensors, layer, parent, trajectory, reference):
        check_next(layer, dispatches)
        return native_layer(tensors, layer, parent, trajectory, reference)

    def guarded_stages(tensors, layer, parent, arrays):
        check_next(layer, stages)
        return native_stages(tensors, layer, parent, arrays)

    with patch.object(entry.upstream, "execute_layer", side_effect=guarded_layer), \
            patch.object(entry.native, "stages", side_effect=guarded_stages), \
            patch.object(subprocess, "Popen", side_effect=AssertionError("RTL/process forbidden")) as process, \
            patch.object(entry.os, "system", side_effect=AssertionError("RTL/shell forbidden")) as shell:
        yield {"native_execution_records": dispatches, "native_stage_execution_records": stages}
        require(dispatches == stages == records, "native stage/dispatch accounting mismatch")
        require(not process.called and not shell.called, "RTL/process execution attempted")


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
        data = authenticate()
    evidence = base.validation_evidence(contract, data)
    observed = json.loads(Path(data["v1_refinement_evidence"]["result"]["path"]).read_text())
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
        "artifacts": artifacts, "v1_refinement_evidence": data["v1_refinement_evidence"],
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
    return {
        "status": "VALIDATED_SOFTWARE_PREFLIGHT_ONLY", "diagnostic_id": ID,
        "cwd": str(Path.cwd()), "executable": sys.executable,
        "PYTHONPATH": entry.os.environ["PYTHONPATH"], "command": COMMAND,
        "origins": origins, "compiled": compiled, "contract": contract_binding,
        "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "v1_refinement_evidence": data["v1_refinement_evidence"],
        "input_bindings": list(data["bound"].values()), **evidence,
        "controls": plan(), "execution": execution_contract(), "execution_plan": execution_plan(),
        "maximum_observed_plan_spacing_Q24_units": max(
            b["input_Q24_units62"] - a["input_Q24_units62"] for a, b in zip(plan(), plan()[1:])),
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
    execution_plan()
    out.mkdir(exist_ok=False)
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({"cwd": str(Path.cwd()), "executable": sys.executable,
                              "argv": sys.argv, "PYTHONPATH": entry.os.environ.get("PYTHONPATH")}) + "\n")
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
    raise SystemExit(main())
