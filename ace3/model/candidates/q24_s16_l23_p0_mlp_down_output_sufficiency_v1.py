"""One L23/P0 stage17 operand intervention; only S18 and final diagnostics run.

Run --execute once from the isolated worktree with its pinned Python, PYTHONPATH
and bytecode disabled. The exclusive attempt retains compile/tests, nine raw
payloads and complete metrics. Never replay an occupied attempt, including a
failed one. S18 uses the unchanged original-input binary64 gate; final-head
comparisons have no rescue threshold. Host independent review is mandatory.
"""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
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

from ace3.model.candidates import q24_s16_l23_p0_value_content_sufficiency_v1 as base


ROOT = base.ROOT
NAME = "q24_s16_l23_p0_mlp_down_output_sufficiency_v1"
TASK = "d09e8e7a98af"
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
CONTROLS = base.CONTROLS
final, layer, native = base.final, base.layer, base.native
require, same, record, read_bound = base.require, base.same, base.record, base.read_bound
AFFECTED = frozenset(("stage17", "stage18", "output_i", "output_z"))
EXPECTED_TESTS = 24
FLAGS = {
    **base.FLAGS, "stage17_native_operator_validated": False,
    "attention_recomputed": False, "token_published": False,
    "original_trajectory_rescued": False, "stage_certification": "not_certified",
}
BOUNDARY = (
    "Conditional all-nine S18 rescue for the original-input FP16 L23 stage17 "
    "MLP-down output substitution only. Actual input/scratch Q24 I/Z/H, inherited "
    "residual, S0-S16, attention/KV/source identity and operands remain fixed. "
    "Final RMSNorm/tied-head comparisons are diagnostic-only, rescued=NOT_DEFINED "
    "there. All historical failures remain unchanged. Q24 residual state is wider "
    "than FP16; native G128 asymmetric INT4 GEMM nibble order, no qzero plus-one, "
    "FP16 scales/operators/KV remain unchanged. No original-prefix/admission/"
    "reference replay, unique-root-cause, strict-FP16-state W4A16, new-token, "
    "full-model, hardware/GPU/RTL or ACE2 claim/change."
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
    # Reuse authentication only, not the other experiment's value substitution.
    parent, _ = base.prepare(actual, reference)
    layer.prior.retained.verify_parent(parent, parent)
    layer.prior.retained.check_stage_state(18, actual, parent)
    layer.prior.local.finite_words(reference["stage17"], (896,))
    reserved = {"retained_" + key for key in AFFECTED} | {"final_rmsnorm", "final_logits"}
    require(not reserved.intersection(actual), "unexpected retained payload fields")
    arrays = {key: value.copy() for key, value in actual.items()}
    for key in AFFECTED:
        arrays["retained_" + key] = actual[key].copy()
    arrays["stage17"] = reference["stage17"].copy()
    for value in arrays.values():
        value.flags.writeable = False
    return arrays


def protected(actual, arrays, reference):
    checks = {}
    for key, value in actual.items():
        other = arrays["retained_" + key if key in AFFECTED else key]
        checks[key] = (value.dtype == other.dtype and value.shape == other.shape
                       and value.tobytes() == other.tobytes())
        require(checks[key], "protected retained field changed: " + key)
    words = arrays["stage17"]
    require(words.dtype == reference["stage17"].dtype and words.shape == (896,)
            and words.tobytes() == reference["stage17"].tobytes(),
            "stage17 substitution lineage changed")
    return checks


def new_audit():
    return {
        "controls": [], "stage_dispatches": [], "stage18_oracle_invocations": 0,
        "final_rmsnorm_invocations": 0, "lm_head_invocations": 0, "top_k_invocations": 0,
        "native_L0_L22_invocations": 0, "native_S0_S17_invocations": 0,
        "reference_producer_invocations": 0, "external_invocations": 0,
        "retained_evidence_writes": 0, "forbidden_calls": 0,
        "forbidden_mutations": 0, "file_write_opens": 0,
    }


@contextmanager
def suffix_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("non-S18 decoder arithmetic forbidden")

    def counted(function, counter, limit):
        def call(*args, **kwargs):
            require(audit[counter] < limit, "extra dispatch: " + counter)
            audit[counter] += 1
            return function(*args, **kwargs)
        return call

    add = native.state.add

    def stage18(state, words):
        index = len(audit["stage_dispatches"])
        require(index < 9 and audit["controls"] == list(CONTROLS[:index + 1]),
                "extra or reordered S18 dispatch")
        audit["stage_dispatches"].append([CONTROLS[index], 23, 0, 18])
        return add(state, words)

    with final.no_dispatch(audit), ExitStack() as stack:
        for module, name in ((base, "suffix_stages"), (native, "projection"),
                             (native, "rne"), (native, "toward_zero"),
                             (layer, "expected_stage"), (layer.prior.local, "local_reference")):
            stack.enter_context(patch.object(module, name, forbidden))
        stack.enter_context(patch.object(native.state, "add", stage18))
        for name, counter, limit in (
            ("rmsnorm", "final_rmsnorm_invocations", 9),
            ("logits", "lm_head_invocations", 9), ("top_k", "top_k_invocations", 27),
        ):
            stack.enter_context(patch.object(
                final, name, counted(getattr(final, name), counter, limit)))
        yield


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
    scratch = layer.prior.retained.state_from(arrays, "scratch", "stage12")
    started = time.monotonic()
    successor = native.state.add(scratch, arrays["stage17"])
    arrays["output_i"], arrays["output_z"], arrays["stage18"] = (
        successor[key] for key in ("i", "z", "h"))
    native_end = time.monotonic()
    audit["stage18_oracle_invocations"] += 1
    expected = layer.prior.retained.transition_reference(scratch, arrays["stage17"])
    layer.prior.retained.verify_parent(successor, expected)
    for key, value in expected.items():
        arrays["stage18_oracle_" + key] = value
    report = layer.stage_report(18, arrays, trajectory, binary64, None)
    report["residual_state_lineage"] = "PASS_explicit_stage17_operand_intervention"
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


def scientific_status(rows):
    same([row["control"] for row in rows], list(CONTROLS), "incomplete/reordered controls")
    for row in rows:
        same([r["stage"] for r in row["reports"]], [18], "incorrect affected cone")
        require(row["status"] == row["reports"][0]["status"]
                and row["status"] in ("PASS", "FAIL"), "invalid S18 acceptance")
        require(row["preservation_checks"] and all(row["preservation_checks"].values())
                and row["source_operand_state_KV_lineage_checks"] == "PASS",
                "retained-state/source/KV/lineage defect")
        require(row["final_rescued"] == "NOT_DEFINED", "invented final-head predicate")
    return "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"


def check_counts(audit):
    same(audit["controls"], list(CONTROLS), "control dispatch census mismatch")
    same(audit["stage_dispatches"], [[label, 23, 0, 18] for label in CONTROLS],
         "stage dispatch census mismatch")
    expected = new_audit()
    expected.update(stage18_oracle_invocations=9, final_rmsnorm_invocations=9,
                    lm_head_invocations=9, top_k_invocations=27, file_write_opens=11)
    for key in expected.keys() - {"controls", "stage_dispatches"}:
        same(audit[key], expected[key], "dispatch/write census mismatch: " + key)


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


def report_summary(report):
    gate = report["binary64_v1"]
    worst = max(gate["rows"], key=lambda row: Fraction(row["excess_error"]))
    return {"status": report["status"], "policy_id": report["policy_id"],
            "coordinates": gate["coordinates"], "failure_count": gate["failure_count"],
            "fp16_diagnostic_failure_count": report["fp16"]["failure_count"],
            "worst_index": worst["index"], "max_excess_error": worst["excess_error"],
            "excess_budget": worst["excess_budget"],
            "threshold_margin": str(Fraction(worst["excess_budget"])
                                    - Fraction(worst["excess_error"]))}


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
                prepared.append((row, actual, prepare(actual, trajectory)))
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
                            "field": "stage17", "boundary": "stage18 MLP-down operand only",
                            "shape": [896], "dtype": "<u2",
                            "changed_fp16_components": int(np.count_nonzero(
                                actual["stage17"] != arrays["stage17"])),
                            "retained_input_and_scratch_Q24_state_unchanged": True,
                            "retained_S0_S16_unchanged": True, "retained_KV_source_unchanged": True,
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
                "decision": ("conditional all-nine MLP-down output intervention rescue only"
                             if status == "PASS" else
                             "reject all-nine rescue for this MLP-down output intervention only"),
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
