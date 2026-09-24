"""Repository-bound L21-only retained/canonical producer-cone diagnostic.

--check preserves the read-only preflight. --execute additionally runs bounded
CPU controls, emitting non-admitting evidence on stdout without publishing state.
"""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import (
    diagnose_q24_s16_l21_coordinate62_adjusted_l22_l23_suffix_v1 as dependency,
)


margin = dependency.margin
retained, require = margin.retained, margin.require
ROOT, PYTHON = margin.ROOT, margin.PYTHON
NAME = "q24_s16_l21_coordinate62_producer_cone_v1"
ID = "ace3-q24-s16-l21-coordinate62-producer-cone-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
SOURCE = ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
INPUT, PINS = dependency.INPUT, dict(dependency.PINS)
EXPECTED_TESTS = 24
FLAGS = {**margin.FLAGS, "scientific_result_claim": False}
native, local, gates = dependency.native, dependency.local, dependency.gates
CONES = (17, 16, 15, 14, 13, 11, 10, 7, 3, 0)
EXECUTION = {
    "version": 1, "native_layers": [21], "maximum_native_layer_invocations": 11,
    "canonical_cone_order": list(CONES),
    "parent": "authenticated frozen L20 output I/Z/H; never the adjusted L21 output",
    "S16": "independent same-operand binary64 SiLU/product with unchanged FP16 RTZ",
    "stop": "first new mandatory gate failure or full-vector S18 pass; otherwise INCONCLUSIVE",
    "output": "stdout only; no state archive, successor or admission",
}
LOCAL_GATE = {
    "absolute_budget": "1/8", "relative_strict_bound": "1/1000",
    "ordered_FP16_ULP": 1, "relative_denominator_floor": "1/16384",
}
LINEAGE = {
    "L15_cut_to_L21_input_IZH": "PASS", "own_empty_P0_FP16_KV": "PASS",
    "S12_S18_exact_IZH": "PASS", "S16_retained_operand_RTZ": "PASS",
    "original_input_reference_recurrence": "PASS",
}
no_execution = margin.no_execution


def context():
    require(Path.cwd() == ROOT and Path(sys.executable) == PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "published repo-bound command required")
    return {
        "cwd": str(ROOT), "executable": sys.executable,
        "PYTHONPATH": os.environ["PYTHONPATH"], "dont_write_bytecode": True,
    }


def origins():
    require(Path(__file__).resolve() == SOURCE and CONTRACT.resolve() == CONTRACT,
            "candidate/contract origin mismatch")
    records = margin.margin.origins()
    for name in (MODULE, TEST_MODULE, margin.margin.prior.LEGACY_TEST):
        require(name in records
                and records[name]["path"] == str(ROOT.joinpath(*name.split("."))
                                                .with_suffix(".py")),
                f"required repository origin missing: {name}")
    return records


def check_contract(contract):
    expected = {
        "diagnostic_id": ID, "version": 1, "input": str(INPUT.relative_to(ROOT)),
        "upstream_sha256": PINS, "suffix_sha256": margin.PINS,
        "node": [21, 0, 18], "index": 62, "focused_tests": EXPECTED_TESTS,
        "policy_id": margin.gates.POLICY_ID, "local_gate": LOCAL_GATE,
        "excess_budget": "1/8", "reference_policy": dependency.REFERENCE_POLICY,
        "native_layers": [], "interface": ["--check", "--execute"],
        "executor": EXECUTION, **FLAGS,
    }
    for key, value in expected.items():
        require(type(contract[key]) is type(value) and contract[key] == value,
                f"versioned contract mismatch: {key}")


def check_reconstruction(document, frozen, failure, analysis):
    require(document["first_failure"] == failure == frozen["first_failure"]
            and failure["node"] == [21, 0, 18] and failure["index"] == 62
            and failure["gate"] == "binary64_v1"
            and document["lineage"] == LINEAGE
            and all(document[key] == value for key, value in analysis.items()),
            "frozen L21 failure/margin/lineage reconstruction mismatch")


def authenticate():
    with no_execution():
        inputs = margin.margin.prior.BoundInputs()
        documents = {name: margin.scalar_math.pinned(inputs, INPUT / name, digest)
                     for name, digest in PINS.items()}
        document, validation = documents["result.json"], documents["validation.json"]
        dependency.check_margin(document, validation, documents["command.json"])
        for record in document["authenticated_inputs"]:
            inputs.bind(record)
        margin.upstream.preflight.bind_origins(inputs, validation["origins"])
        for record in (*validation["compiled"], validation["contract"]):
            inputs.bind(record)
        verified, frozen, failure, analysis = margin.authenticate()
        for record in verified.records.values():
            margin.margin.bind_closure(inputs, record)
        check_reconstruction(document, frozen, failure, analysis)
        return inputs, document, frozen


def validate():
    command = context()
    tests = importlib.import_module(TEST_MODULE)
    importlib.import_module(margin.margin.prior.LEGACY_TEST)
    source_origins = origins()
    contract_record = retained.record(CONTRACT)
    check_contract(json.loads(CONTRACT.read_text()))
    compiled = []
    for path in (SOURCE, ROOT / "tests" / (TEST_MODULE.split(".")[-1] + ".py")):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(retained.record(path))
    with no_execution():
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        require(count == EXPECTED_TESTS, "focused test collection mismatch")
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.testsRun == EXPECTED_TESTS and result.wasSuccessful()
            and not result.skipped, "focused tests failed")
    require(origins() == source_origins and retained.record(CONTRACT) == contract_record,
            "source/contract changed during validation")
    return {
        **command, "origins": source_origins, "compiled": compiled,
        "contract": contract_record, "contract_parsed": True,
        "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "legacy_19_tests_executed": False,
        "upstream_tests_executed": False,
    }


def prepare():
    validation = validate()
    with no_execution():
        inputs, document, frozen = authenticate()
        for record in (*validation["origins"].values(), validation["contract"]):
            inputs.bind(record)
        require(origins() == validation["origins"], "module origins changed during preflight")
        for record in inputs.records.values():
            require(retained.record(record["path"]) == record,
                    f"input/source changed during preflight: {record['path']}")
    result = {
        "diagnostic_id": ID, "status": "PREFLIGHT_VALIDATED", **FLAGS,
        "validation": validation, "upstream_sha256": PINS,
        "suffix_sha256": margin.PINS,
        "authenticated_inputs": list(inputs.records.values()),
        "first_failure": document["first_failure"],
        "retained_status": document["retained_status"],
        "accepted_frozen_layers": [15, 16, 17, 18, 19, 20],
        "frozen_native_layers": frozen["audit"]["native_layers"],
        "native_layers": [], "lineage": document["lineage"],
        "review_provenance": document["review_provenance"],
        "reference_policy": document["reference_policy"],
        "claim_boundary": (
            "Read-only CPU-software producer-cone runtime preflight only; no new "
            "producer attribution, native intervention, state publication or numerical PASS. "
            "Historical failures and bounded Reviewer acceptances remain unchanged. "
            "Q24 state is wider than FP16; not strict-FP16-state W4A16, new-token, "
            "full-model, RTL, simulation, synthesis, PPA, GPU, FPGA or hardware evidence."
        ),
    }
    return result, inputs, frozen


def preflight():
    return prepare()[0]


def controls():
    yield "retained_native", ()
    for index, stage in enumerate(CONES):
        yield f"canonical_S{stage:02d}_suffix", tuple(sorted(CONES[:index + 1]))


@contextmanager
def l21_only(audit):
    raw = native.candidate._stages

    def forbidden(*args, **kwargs):
        raise RuntimeError("native replay, external execution or state publication forbidden")

    def guarded(tensors, layer, parent, arrays):
        require(type(layer) is int and layer == 21, "only native L21 permitted")
        require(len(audit["native_layers"]) < EXECUTION["maximum_native_layer_invocations"],
                "L21 control execution budget exceeded")
        audit["native_layers"].append(layer)
        yield from raw(tensors, layer, parent, arrays)

    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates."):
                for attribute in ("stages", "continuation_stages", "native_layer",
                                  "run", "_stages", "save"):
                    if callable(getattr(module, attribute, None)):
                        replacement = (guarded if module is native.candidate
                                       and attribute == "_stages" else forbidden)
                        stack.enter_context(patch.object(module, attribute, replacement))
        stack.enter_context(patch.object(subprocess, "Popen", forbidden))
        stack.enter_context(patch.object(os, "system", forbidden))
        yield


def canonical_s16(arrays):
    gate = local.finite_words(arrays["stage14"], (4864,))
    up = local.finite_words(arrays["stage15"], (4864,))
    sigmoid = np.empty_like(gate)
    positive = gate >= 0
    sigmoid[positive] = 1 / (1 + np.exp(-gate[positive]))
    exponential = np.exp(gate[~positive])
    sigmoid[~positive] = exponential / (1 + exponential)
    unrounded = gate * sigmoid * up
    return unrounded, margin.margin.prior.rtz_reference(unrounded)


def drive_control(tensors, parent, trajectory, reference, cut):
    require(cut in [parts for _, parts in controls()], "unregistered producer-cone cut")
    arrays, reports, changes, timings = {}, [], [], []
    producer = native.candidate._stages(tensors, 21, parent, arrays)
    try:
        for stage in range(19):
            started = time.monotonic()
            require(next(producer) == stage, "out-of-order L21 native stage")
            native_seconds = time.monotonic() - started
            started = time.monotonic()
            expected = retained.check_stage_state(stage, arrays, parent)
            if stage < 18 and stage != 12:
                expected = local.local_reference(
                    stage, {key: arrays[key].copy() for key in local.OPERANDS[stage]},
                    tensors, 21)
            if stage in cut:
                key = f"stage{stage:02d}"
                original = arrays[key]
                if stage == 16:
                    arrays["s16_unrounded_binary64"], replacement = canonical_s16(arrays)
                else:
                    replacement = expected.copy()
                arrays[key] = replacement
                changes.append({
                    "stage": stage, "operands": list(local.OPERANDS[stage]),
                    "changed_indices": np.flatnonzero(original != replacement).tolist(),
                })
                retained.check_stage_state(stage, arrays, parent)
            if stage == 16:
                require(np.array_equal(
                    margin.margin.prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                    arrays["stage16"]), "changed S16 operand RTZ")
            report = gates.evaluate_decoder_stage(
                stage=stage, actual=arrays[f"stage{stage:02d}"],
                reference=trajectory[f"stage{stage:02d}"], policy=gates.POLICY_ID,
                local_reference=expected, reference_binary64=reference if stage == 18 else None)
            report.update(node=[21, 0, stage], residual_state_lineage="PASS",
                          kv_lineage="PASS", local_reference_independent=stage < 18)
            reports.append(report)
            timings.append({"stage": stage, "native_seconds": native_seconds,
                            "oracle_and_cut_seconds": time.monotonic() - started})
            require(report["status"] in ("PASS", "FAIL"), "L21 mandatory gate blocked")
            if report["status"] == "FAIL":
                break
    finally:
        producer.close()
    return arrays, reports, changes, timings


def same_arrays(actual, expected):
    require(actual.keys() == expected.keys(), "native retained array coverage mismatch")
    for key, value in actual.items():
        require(value.dtype == expected[key].dtype and value.shape == expected[key].shape
                and value.tobytes() == expected[key].tobytes(),
                f"native retained bitwise mismatch: {key}")


def decision(reports):
    require(reports and all(row["status"] == "PASS" for row in reports[:-1]),
            "control continued after mandatory failure")
    last = reports[-1]
    require(last["stage"] == len(reports) - 1 and last["status"] in ("PASS", "FAIL"),
            "invalid control report sequence")
    if last["status"] == "PASS":
        require(len(reports) == 19, "incomplete passing control")
        return "PASS_BOUNDARY"
    if last["stage"] != 18:
        return "LOCAL_GATE_BOUNDARY"
    failures = last["binary64_v1"]["failures"]
    require(failures, "missing global failure evidence")
    return ("GLOBAL_GATE_BOUNDARY" if any(row["index"] != 62 for row in failures)
            else None)


def sample(arrays, reports, reference):
    result = {"last_stage": reports[-1]["stage"], "status": reports[-1]["status"]}
    if "stage18" not in arrays:
        return result
    index = 62
    values = {
        "input_Q24": Fraction(int(arrays["input_i"][index]), margin.Q),
        "O": margin.rational.fp16_value(int(arrays["stage11"][index])),
        "down": margin.rational.fp16_value(int(arrays["stage17"][index])),
        "output_Q24": Fraction(int(arrays["output_i"][index]), margin.Q),
    }
    require(values["input_Q24"] + values["O"] + values["down"] == values["output_Q24"],
            "coordinate-62 residual decomposition mismatch")
    result.update(
        residual_contributions={key: str(value) for key, value in values.items()},
        index62=margin.scalar_math.scalar(
            int(arrays["stage18"][index]), Fraction.from_float(float(reference[index]))))
    return result


def execute(audit):
    started = time.monotonic()
    result, inputs, frozen = prepare()
    result["phase_seconds"] = {"compile_test_authentication": time.monotonic() - started}
    started = time.monotonic()
    entry = frozen["layers"][-1]
    arrays = inputs.archive(entry["actual_stages"])
    parent = inputs.archive(entry["input_state_evidence"])
    retained.verify_parent(retained.state_from(arrays, "input", "input_hidden"), parent)
    item = entry["original_reference"]
    trajectory = inputs.archive(item["fp16"])
    upstream = margin.upstream.upstream
    reference = upstream.global_reference(inputs, item)
    freeze = inputs.read(retained.record(upstream.margin.INPUT / "freeze.json"))
    extension = inputs.read(freeze["reference_extension"])
    require(extension["layers"]["21"] == item, "L21 original reference substitution")
    expected_reports = inputs.read(entry["reports"])
    expected_locals = inputs.archive(entry["local_references"])
    result.update(status="INCONCLUSIVE", native_layers=audit["native_layers"],
                  controls=[], executor=EXECUTION, parent_binding=entry["input_state_evidence"],
                  original_reference=item, native_retained_bitwise_reproduction=False)
    native.candidate.torch.set_num_threads(1)
    with upstream.safe_open(str(inputs.bind(extension["checkpoint"])), framework="numpy") as model:
        tensors = {name: model.get_tensor(name) for name in local.tensor_shapes(21)}
    local.authenticate_tensors(tensors, item["canonical"], 21)
    result["phase_seconds"]["L21_operand_loading"] = time.monotonic() - started
    with l21_only(audit):
        for label, cut in controls():
            generated, reports, changes, timings = drive_control(
                tensors, parent, trajectory, reference, cut)
            if not cut:
                same_arrays(generated, arrays)
                require(reports == expected_reports, "native retained gate mismatch")
                for stage in range(18):
                    expected = (retained.transition_reference(parent, generated["stage11"])["h"]
                                if stage == 12 else local.local_reference(
                                    stage, {key: generated[key].copy()
                                            for key in local.OPERANDS[stage]}, tensors, 21))
                    require(np.array_equal(expected, expected_locals[f"stage{stage:02d}"]),
                            "native retained local reference mismatch")
                result["native_retained_bitwise_reproduction"] = True
            boundary = decision(reports)
            row = {"label": label, "canonical_stages": list(cut), "changes": changes,
                   "phase_timings": timings, "reports": reports,
                   **sample(generated, reports, reference)}
            result["controls"].append(row)
            if boundary:
                require(bool(cut), "retained baseline boundary changed")
                result.update(status=boundary, decisive_control=label)
                break
    started = time.monotonic()
    require(origins() == result["validation"]["origins"], "module origins changed during execution")
    for record in inputs.records.values():
        require(retained.record(record["path"]) == record,
                f"input/source changed during execution: {record['path']}")
    result["phase_seconds"]["final_binding_check"] = time.monotonic() - started
    result.update(
        native_invocations=len(audit["native_layers"]),
        native_layer_invocations=len(audit["native_layers"]),
        authenticated_inputs=list(inputs.records.values()),
        interpretation=(
            "All registered same-input canonical producer-cone controls still fail only "
            "L21/P0/S18 coordinate 62. These controls do not isolate inherited drift from "
            "accumulated legal FP16 rounding; producer cause remains inconclusive."
            if result["status"] == "INCONCLUSIVE" else
            "First registered control reaching a different mandatory gate boundary. "
            "This conditional boundary is not unique root-cause attribution or admission."),
        claim_boundary=(
            "Non-admitting L21/P0 CPU controls from frozen L20 I/Z/H only. Original-input "
            "references and all gates are unchanged. Canonical S16 retains RTZ; exact "
            "Q24 residual state is wider than FP16. No native L0-L8, other native layer, "
            "RTL, state publication, strict-FP16-state W4A16, token, full-model or hardware "
            "claim. Historical failures and bounded Reviewer acceptances remain unchanged."))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    audit = {"native_layers": []}
    try:
        result = preflight() if args.check else execute(audit)
    except (ValueError, OSError, KeyError, TypeError, IndexError,
            ArithmeticError, RuntimeError, SyntaxError, ImportError) as exc:
        result = {"diagnostic_id": ID, "status": "BLOCKED", **FLAGS,
                  "error": f"{type(exc).__name__}: {exc}",
                  "native_layers": audit["native_layers"],
                  "native_invocations": len(audit["native_layers"]),
                  "native_layer_invocations": len(audit["native_layers"])}
        print(result["error"], file=sys.stderr)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in (
        "PREFLIGHT_VALIDATED", "INCONCLUSIVE", "PASS_BOUNDARY",
        "LOCAL_GATE_BOUNDARY", "GLOBAL_GATE_BOUNDARY") else 1


if __name__ == "__main__":
    raise SystemExit(main())
