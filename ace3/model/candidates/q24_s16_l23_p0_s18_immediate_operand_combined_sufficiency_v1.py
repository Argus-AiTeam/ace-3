"""One combined original-input FP16 S12/S17 operand intervention at L23/P0 S18.

Execute once with the pinned Python, PYTHONPATH and bytecode disabled. Only
S18 and the existing final-head diagnostics run; the exclusive ignored attempt
retains compilation, focused tests, raw payloads and full coordinate reports.
An occupied or failed attempt must not be replayed. Independent Host review
is required; final-head rescue has no defined predicate.
"""

import argparse
from contextlib import ExitStack, contextmanager
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_mlp_down_output_sufficiency_v1 as prior


base, final, layer, native = prior.base, prior.final, prior.layer, prior.native
ROOT = prior.ROOT
NAME = "q24_s16_l23_p0_s18_immediate_operand_combined_sufficiency_v1"
TASK = "4024c64f610d"
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
CONTROLS = prior.CONTROLS
require, same, record, read_bound = prior.require, prior.same, prior.record, prior.read_bound
scientific_status, suffix_only = prior.scientific_status, prior.suffix_only
EXPECTED_TESTS = 20
OPERAND_FIELDS = tuple("stage18_operand_" + key for key in ("i", "z", "h"))
FLAGS = {
    **prior.FLAGS, "stage12_native_operator_validated": False,
    "unique_root_cause_localized": False,
}
BOUNDARY = (
    "Conditional all-nine rescue applies only to the combined original-input FP16 "
    "L23 stage12/stage17 immediate-operand intervention at S18. Actual retained "
    "input and scratch Q24 I/Z/H, S0-S16, KV/source identity and all model operands "
    "remain unchanged. The separate S12 operand is lifted exactly to Q24 I/Z/H; "
    "no actual scratch state is rounded or replaced. Final RMSNorm/tied-head "
    "metrics are diagnostic-only, final_rescued=NOT_DEFINED. Any numeric S18 "
    "failure rejects all-nine rescue and leaves the S18 local integration/"
    "reference-boundary hypothesis unresolved; no unique root cause is localized. "
    "All historical failures remain unchanged. Q24 residual state is wider than "
    "FP16; native G128 asymmetric INT4 GEMM nibble order, no qzero plus-one, FP16 "
    "scales/operators/KV remain fixed. No prefix/admission/reference replay, "
    "strict-FP16-state W4A16, new-token, full-model, hardware/GPU/RTL or ACE2 claim."
)


def fresh_output():
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "occupied output; zero dispatch")
    require(OUTPUT.resolve() == OUTPUT and OUTPUT.parent == ROOT / "build"
            and Path(__file__).resolve() == SOURCE, "noncanonical source/output")
    ignored = final.subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--", str(OUTPUT)],
        capture_output=True, check=False)
    require(ignored.returncode == 0, "output not confirmed ignored")


def source_context():
    return {**final.source_context(), "intervention_source": record(SOURCE),
            "intervention_test": record(TEST)}


def prepare(actual, reference):
    require(not set(OPERAND_FIELDS).intersection(actual), "reserved S18 operand fields")
    layer.prior.local.finite_words(reference["stage12"], (896,))
    arrays = prior.prepare(actual, reference)
    operand = native.state.lift(reference["stage12"])
    layer.prior.retained.verify_parent(operand, operand, embedding=reference["stage12"])
    for key, value in operand.items():
        arrays["stage18_operand_" + key] = value
        value.flags.writeable = False
    return arrays


def protected(actual, arrays, reference):
    checks = prior.protected(actual, arrays, reference)
    operand = {key: arrays["stage18_operand_" + key] for key in ("i", "z", "h")}
    layer.prior.retained.verify_parent(operand, operand, embedding=reference["stage12"])
    return {**checks, "original_input_fp16_stage12_operand": True,
            "original_input_fp16_stage17_operand": True}


def new_audit():
    return {**prior.new_audit(), "stage12_operand_lifts": 0}


def check_counts(audit):
    prior.check_counts(audit)
    same(audit["stage12_operand_lifts"], 9, "S12 operand encoding census mismatch")


@contextmanager
def write_scope(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_mutations"] += 1
        raise RuntimeError("filesystem mutation outside exclusive output writes forbidden")

    with base.write_scope(OUTPUT, audit), ExitStack() as stack:
        for name in ("mkdir", "makedirs", "unlink", "remove", "rmdir", "rename",
                     "replace", "link", "symlink", "chmod", "truncate", "utime"):
            stack.enter_context(patch.object(os, name, forbidden))
        yield


def run_control(label, actual, arrays, trajectory, binary64, operands, references,
                baseline, audit):
    index = len(audit["controls"])
    require(index < 9 and label == CONTROLS[index], "extra or reordered control")
    checks = protected(actual, arrays, trajectory)
    audit["controls"].append(label)
    operand = {key: arrays["stage18_operand_" + key] for key in ("i", "z", "h")}
    started = time.monotonic()
    successor = native.state.add(operand, arrays["stage17"])
    arrays["output_i"], arrays["output_z"], arrays["stage18"] = (
        successor[key] for key in ("i", "z", "h"))
    native_end = time.monotonic()
    audit["stage18_oracle_invocations"] += 1
    expected = layer.prior.retained.transition_reference(operand, trajectory["stage17"])
    layer.prior.retained.verify_parent(successor, expected)
    for key, value in expected.items():
        arrays["stage18_oracle_" + key] = value
    report = layer.stage_report(18, arrays, trajectory, binary64, None)
    report["residual_state_lineage"] = "PASS_explicit_combined_S12_S17_operand_intervention"
    gate_end = time.monotonic()
    arrays["final_rmsnorm"], details = final.rmsnorm(arrays["stage18"], operands[0])
    norm_end = time.monotonic()
    arrays["final_logits"] = final.logits(arrays["final_rmsnorm"], operands[1])
    head_end = time.monotonic()
    metrics = final.comparisons(
        {"rmsnorm": arrays["final_rmsnorm"], "logits": arrays["final_logits"]}, references)
    margins = base.margin_rows(baseline["logits"], arrays["final_logits"], references)
    same(protected(actual, arrays, trajectory), checks, "post-execution protection changed")
    return {
        "control": label, "status": report["status"], "reports": [report],
        "final_rescued": "NOT_DEFINED", "final_comparisons": metrics,
        "paired_final_margins": margins, "preservation_checks": checks,
        "source_operand_state_KV_lineage_checks": "PASS",
        "rmsnorm_integer_details": details,
        "timing_seconds": {"stage18_native": native_end - started,
                           "stage18_oracle_gate": gate_end - native_end,
                           "final_rmsnorm": norm_end - gate_end,
                           "lm_head": head_end - norm_end,
                           "comparisons": time.monotonic() - head_end},
    }


def report_summary(report):
    return {**prior.report_summary(report),
            "failed_coordinates": [row for row in report["binary64_v1"]["rows"]
                                   if not row["accepted"]],
            "fp16_diagnostic_failures": report["fp16"]["failures"]}


def focused_tests(origins):
    compiled = []
    for pin in {p["path"]: p for p in origins.values()}.values():
        compile(read_bound(pin), pin["path"], "exec", dont_inherit=True)
        compiled.append(pin)
    spec = importlib.util.spec_from_file_location(NAME + "_tests", TEST)
    require(spec is not None and spec.loader is not None, "test loader missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    require(suite.countTestCases() == EXPECTED_TESTS, "focused test census changed")
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "focused tests failed:\n" + output.getvalue())
    return {"compiled": compiled, "executed": result.testsRun, "failures": 0, "errors": 0,
            "skipped": 0, "raw_output": output.getvalue(), "retained_control_dispatches": 0}


def execute():
    fresh_output()
    native.torch.set_num_threads(1)
    require(str(native.torch.tensor(0).device) == "cpu", "non-CPU default device")
    origins = source_context()
    OUTPUT.mkdir(exist_ok=False)
    audit, rows, artifacts = new_audit(), [], []

    def save_json(name, value):
        with (OUTPUT / name).open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        return record(OUTPUT / name)

    try:
        with write_scope(audit):
            artifacts.append(save_json("command.json", {
                "task_id": TASK, "revision": 1, "argv": sys.argv, "cwd": str(ROOT),
                "uid": os.getuid(), "executable": sys.executable, "origins": origins,
                "host_managed_controls": "unchanged role/model/account/budget/access/concurrency locks",
                "execution_allowance": "one; exclusive output; no automatic replay",
            }))
            tests = focused_tests(origins)
            artifacts.append(save_json("compile_tests.json", tests))
            started = time.monotonic()
            baseline, baseline_arrays, references, pins = base.retained.authenticate()
            summary = baseline["preflight"]
            tensors, trajectory, binary64, checkpoint = layer.load_inputs({"summary": summary})
            same(final.preflight.bind_assets(checkpoint), summary["assets"], "asset identity changed")
            operands = final.load_operands(summary)
            require(summary["L23_original_reference"]["reference"]["prior_kv"] == "own empty P0",
                    "reference KV/source scope changed")
            for branch in ("fp16", "binary64"):
                same(summary["L23_original_reference"]["reference"][branch],
                     summary["final_reference"]["reference"]["input_" + branch],
                     "original-input reference lineage splice")
            for value in (*tensors.values(), *trajectory.values(), binary64, *references.values()):
                value.flags.writeable = False
            prepared = []
            for row in baseline["controls"]:
                actual = base.archive(row["parent"]["terminal_archive"])
                arrays = prepare(actual, trajectory)
                audit["stage12_operand_lifts"] += 1
                prepared.append((row, actual, arrays))
            same([row["control"] for row, _, _ in prepared], list(CONTROLS),
                 "retained nine-control identity changed")
            auth_seconds = time.monotonic() - started
            with suffix_only(audit):
                for row, actual, arrays in prepared:
                    label = row["control"]
                    entry = run_control(label, actual, arrays, trajectory, binary64, operands,
                                        references, baseline_arrays[label], audit)
                    with (OUTPUT / f"{label}.npz").open("xb") as stream:
                        np.savez(stream, **arrays)
                    pin = record(OUTPUT / f"{label}.npz")
                    artifacts.append(pin)
                    entry.update({
                        "parent": row["parent"], "raw_payload": pin,
                        "S18_acceptance": report_summary(entry["reports"][0]),
                        "substitution": {
                            "source": summary["L23_original_reference"]["reference"]["fp16"],
                            "fields": ["stage12", "stage17"],
                            "boundary": "stage18 immediate operands only",
                            "shape_each": [896], "dtype_each": "<u2",
                            "stage12_encoding": "exact FP16 lift into stage18_operand_i/z/h",
                            "changed_fp16_components": {
                                key: int(np.count_nonzero(actual[key] != trajectory[key]))
                                for key in ("stage12", "stage17")},
                            "retained_input_and_scratch_Q24_state_unchanged": True,
                            "retained_S0_S16_unchanged": True, "retained_KV_source_unchanged": True,
                            "original_stage17_preserved_as": "retained_stage17",
                            "original_output_preserved_as": "retained_stage18/output_i/output_z",
                        },
                    })
                    rows.append(entry)
                    print(json.dumps({k: v for k, v in entry.items()
                                      if k not in ("parent", "reports")}), flush=True)
            same(source_context(), origins, "source changed during execution")
            for pin in [*pins, checkpoint]:
                read_bound(pin)
            same(final.preflight.bind_assets(checkpoint), summary["assets"],
                 "operands/assets changed during execution")
            check_counts(audit)
            status = scientific_status(rows)
            result = {
                "task_id": TASK, "revision": 1, "diagnostic_id": NAME, "status": status,
                "execution_status": "COMPLETE", "execution_count": 1,
                "delivery_status": "AWAITING_INDEPENDENT_REVIEW", "final_rescued": "NOT_DEFINED",
                "controls": rows, "audit": {**audit, "file_write_opens": 12},
                "artifacts": artifacts, "origins": origins, "input_pins": pins,
                "checkpoint": checkpoint, "preflight": summary, "thresholds": summary["thresholds"],
                "final_reference": summary["final_reference"], "tests": tests,
                "output": str(OUTPUT), "output_created_exclusively": True,
                "authentication_seconds": auth_seconds, "arithmetic": final.ARITHMETIC,
                "flags": FLAGS, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
                "automatic_replay_authorized": False, "expected_final_file_count": 12,
                "S18_local_integration_reference_boundary_hypothesis": "UNRESOLVED",
                "decision": ("conditional all-nine combined immediate-operand intervention rescue only"
                             if status == "PASS" else
                             "reject all-nine rescue for combined immediate-operand intervention; "
                             "S18 local integration/reference-boundary hypothesis unresolved"),
            }
            save_json("result.json", result)
            same(audit, result["audit"], "final write census mismatch")
            same(sorted(path.name for path in OUTPUT.iterdir()),
                 sorted(["command.json", "compile_tests.json", "result.json"]
                        + [label + ".npz" for label in CONTROLS]), "artifact census mismatch")
            print(json.dumps({key: result[key] for key in (
                "task_id", "status", "execution_status", "execution_count", "decision",
                "final_rescued", "audit", "output", "flags", "claim_boundary")}), flush=True)
    except Exception as error:
        with write_scope(audit):
            save_json("failure.json", {
                "task_id": TASK, "status": "UNKNOWN", "execution_status": "DEFECT",
                "error_type": type(error).__name__, "error": str(error),
                "audit": {**audit, "file_write_opens": audit["file_write_opens"] + 1},
                "controls": rows, "flags": FLAGS, "automatic_replay_authorized": False,
            })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    execute()


if __name__ == "__main__":
    from importlib import import_module

    import_module("ace3.model.candidates." + NAME).main()
