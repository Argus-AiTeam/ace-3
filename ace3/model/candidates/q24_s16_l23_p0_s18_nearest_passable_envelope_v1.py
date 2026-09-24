"""One nearest-passable S18 word intervention; only the final diagnostic head runs."""

import argparse
from bisect import bisect_left, bisect_right
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

from ace3.model.candidates import diagnose_q24_s16_l23_p0_s18_binary64_reference_boundary_v1 as diagnostic


ROOT, final, base, layer = diagnostic.ROOT, diagnostic.final, diagnostic.base, diagnostic.layer
require, same, record, read_bound = diagnostic.require, diagnostic.same, diagnostic.record, diagnostic.read_bound
policy, rational = diagnostic.policy, diagnostic.rational
NAME = "q24_s16_l23_p0_s18_nearest_passable_envelope_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
TASK = "8e448d376f47"
CONTROLS = diagnostic.CONTROLS
EXPECTED_TESTS = 22
RESULT_PIN = {
    "path": str(diagnostic.OUTPUT / "result.json"), "bytes": 19109472,
    "sha256": "366d9595ced4c7d9038bb00ce32854f472aec68b9d12947f5d38bb227380afc6",
}
FLAGS = {
    **base.FLAGS, "token_published": False, "original_trajectory_rescued": False,
    "decoder_recomputed": False, "intervened_stage18_native_lineage_validated": False,
    "intervened_stage18_is_Q24_state_projection": False,
}
BOUNDARY = (
    "Nearest-passable finite FP16 word intervention on the reviewed combined-operand "
    "S18 vector, not the original native trajectory. Only failing coordinates change. "
    "Retained L0-S17, S18, Q24 I/Z, operands, FP16 KV/source/lineage and all historical "
    "FAILs remain unchanged; intervened_stage18 is a separate diagnostic boundary, "
    "not a new Q24 state or an unchanged native operator. The original scalar "
    "binary64 excess budget remains 1/8; S18 PASS is scalar-output-only. Final "
    "RMSNorm/tied-head comparisons are diagnostic-only; final_rescued=NOT_DEFINED. "
    "Q24 residual state is wider than FP16. Native S16 RTZ, G128 asymmetric packed "
    "INT4 GEMM nibble order, no qzero plus-one, FP16 scales/operators/KV unchanged. "
    "No prefix/native/attention/MLP/reference/admission replay, hardware/RTL/GPU/FPGA/"
    "simulation, ACE2 changes, precision/scale expansion, strict-FP16-state W4A16, "
    "new-token or full-model admission. Normal independent Host review required."
)


def source_context():
    return {**final.source_context(), "intervention_source": record(SOURCE),
            "intervention_test": record(TEST)}


def fresh_output():
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "occupied output; zero dispatch")
    require(OUTPUT.resolve() == OUTPUT and OUTPUT.parent == ROOT / "build"
            and Path(__file__).resolve() == SOURCE, "noncanonical source/output")
    check = final.subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--", str(OUTPUT)],
        capture_output=True, check=False)
    require(check.returncode == 0, "output not confirmed ignored")


def terminal_review():
    directory = layer.preflight.HANDOFFS / diagnostic.TASK
    latest = json.loads((directory / "latest.json").read_bytes())
    require(latest["kind"] == "handoff_ref", "S18 diagnostics lack terminal review")
    path = Path(latest["handoff"]["path"])
    require(path.parent == directory and path.resolve() == path, "review path escaped")
    pin = record(path)
    review = json.loads(read_bound(pin))
    backlog = [json.loads(line) for line in layer.preflight.BACKLOG.read_text().splitlines()
               if line.strip()]
    final.preflight.check_review(review, latest, backlog, mission=diagnostic.TASK,
                                 round_number=2, pin=pin)
    mission_path = directory / "mission.json"
    same(latest["mission"]["path"], str(mission_path), "review mission path changed")
    mission = json.loads(mission_path.read_bytes())
    require(mission["mission_id"] == diagnostic.TASK
            and mission["execution_workdir"] == str(ROOT), "review scope changed")
    return {"pin": pin, "record": review, "mission": record(mission_path),
            "backlog_status": "done", "outcome_review_status": "done"}


def authenticate():
    review = terminal_review()
    retained = json.loads(read_bound(RESULT_PIN))
    require(retained["task_id"] == diagnostic.TASK and retained["revision"] == 2
            and retained["diagnostic_id"] == diagnostic.NAME
            and retained["status"] == "COMPLETE" and retained["execution_count"] == 1
            and retained["output"] == str(diagnostic.OUTPUT)
            and retained["final_rescued"] == "NOT_DEFINED", "diagnostic identity defect")
    same(retained["flags"], diagnostic.parent.FLAGS, "retained boundary flags changed")
    for pin in retained["origins"].values():
        read_bound(pin)
    combined, trajectory, reference64, payloads, authentication = diagnostic.authenticate()
    same(retained["authentication"], authentication, "diagnostic parent authentication splice")
    same(retained["thresholds"], combined["thresholds"], "threshold splice")
    same([row["control"] for row in retained["controls"]], list(CONTROLS),
         "missing/reordered diagnostic controls")
    for row, parent_row, arrays in zip(
            retained["controls"], combined["controls"], payloads, strict=True):
        require(row["status"] == "COMPLETE"
                and row["source_operand_state_KV_lineage"] == "AUTHENTICATED",
                "diagnostic source/lineage defect")
        same(row["parent"], parent_row["parent"], "diagnostic parent splice")
        same(row["raw_input"], parent_row["raw_payload"], "diagnostic input splice")
        same(row["substitution"], parent_row["substitution"], "operand lineage splice")
        same(row["raw_payload"]["path"], str(diagnostic.OUTPUT / (row["control"] + ".npz")),
             "diagnostic payload path splice")
        saved = base.archive(row["raw_payload"])
        preserve(arrays, saved)
        same(sorted(saved), sorted(arrays), "diagnostic payload field splice")
        for value in arrays.values():
            value.flags.writeable = False
    summary = combined["preflight"]
    for branch in ("fp16", "binary64"):
        same(summary["L23_original_reference"]["reference"][branch],
             summary["final_reference"]["reference"]["input_" + branch],
             "original-input final reference reanchored")
    references = final.load_references(summary)
    return retained, combined, trajectory, reference64, payloads, references, {
        "diagnostic_result": RESULT_PIN, "diagnostic_review": review,
        "parent_authentication": authentication,
        "source_operand_state_KV_lineage": "AUTHENTICATED",
    }


def select_word(word, row, reference64, table):
    gate = policy.evaluate_layer_final_output(
        actual_fp16_bits=word, reference_binary64_hex=float(reference64).hex())
    for key, field in (
        ("actual_fp16_bits", "actual_substituted_fp16_word"),
        ("reference_binary64_hex", "reference_binary64_hex"),
        ("q", "representation_floor"), ("actual_error", "actual_error"),
        ("excess_error", "excess_error"), ("excess_budget", "excess_budget"),
        ("accepted", "accepted"),
    ):
        same(gate[key], row[field], "recorded scalar authentication defect: " + field)
    value = rational.fp16_value(word)
    reference = Fraction.from_float(float(reference64))
    radius = Fraction(gate["q"]) + policy.EXCESS_BUDGET
    values, words = table
    low, high = bisect_left(values, reference - radius), bisect_right(values, reference + radius)
    require(low < high, "empty finite passable interval")
    same(row["passable_interval"], {
        "order": "all finite FP16 words ascending exact value then unsigned word",
        "first_index": low, "last_index": high - 1, "count": high - low,
        "first_word": f"{words[low]:04x}", "last_word": f"{words[high - 1]:04x}",
        "lower_inclusive": str(reference - radius), "upper_inclusive": str(reference + radius),
    }, "recorded passable interval defect")
    nearest, distance = diagnostic.nearest_words(value, table, low, high)
    same(row["nearest_passable_words_to_actual"], [f"{w:04x}" for w in nearest],
         "recorded nearest-passable tie set defect")
    same(row["minimum_passable_word_change"], str(distance), "recorded minimum delta defect")
    same(row["passable_signed_changes"],
         [str(rational.fp16_value(w) - value) for w in nearest], "recorded signed delta defect")
    selected = word if gate["accepted"] else min(
        nearest, key=lambda w: (abs(rational.fp16_value(w) - value), rational.fp16_value(w), w))
    accepted = policy.evaluate_layer_final_output(
        actual_fp16_bits=selected, reference_binary64_hex=float(reference64).hex())
    require(accepted["accepted"], "nearest-passable intervention failed original scalar gate")
    return selected, {
        "index": row["index"], "retained_word": f"{word:04x}",
        "selected_word": f"{selected:04x}", "changed": selected != word,
        "selected_value": str(rational.fp16_value(selected)),
        "signed_delta": str(rational.fp16_value(selected) - value),
        "absolute_delta": str(abs(rational.fp16_value(selected) - value)),
        "nearest_passable_ties": [f"{w:04x}" for w in nearest],
        "tie_present": len(nearest) > 1, "passing_word_preserved": not gate["accepted"] or selected == word,
        "selection_order": "absolute signed change; exact FP16 value; unsigned word",
        "passable_interval": row["passable_interval"],
        "retained_gate": gate, "intervened_gate": accepted,
        "threshold_margin": str(policy.EXCESS_BUDGET - Fraction(accepted["excess_error"])),
    }


def intervene(actual, row, reference64, table):
    layer.prior.local.finite_words(actual["stage18"], (896,))
    require(reference64.shape == (896,) and reference64.dtype.str == "<f8"
            and np.all(np.isfinite(reference64)), "invalid independent S18 reference")
    same([r["index"] for r in row["rows"]], list(range(896)), "coordinate census changed")
    require(not any(key.startswith("intervened_") for key in actual),
            "reserved intervention payload fields")
    arrays = {key: value.copy() for key, value in actual.items()}
    selected, reports = [], []
    for word, scalar_row, reference in zip(actual["stage18"], row["rows"], reference64, strict=True):
        new_word, report = select_word(int(word), scalar_row, reference, table)
        selected.append(new_word)
        reports.append(report)
    arrays["intervened_stage18"] = np.asarray(selected, dtype="<u2")
    for value in arrays.values():
        value.flags.writeable = False
    return arrays, reports


def preserve(actual, arrays):
    checks = {}
    for key, value in actual.items():
        other = arrays[key]
        checks[key] = (value.dtype == other.dtype and value.shape == other.shape
                       and value.tobytes() == other.tobytes())
        require(checks[key], "protected retained field changed: " + key)
    return checks


@contextmanager
def final_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("only the intervened final diagnostic suffix may execute")

    native = diagnostic.parent.native
    with final.no_dispatch(audit), ExitStack() as stack:
        for module, names in (
            (native, ("projection", "rne", "toward_zero")),
            (native.state, ("add", "lift")),
            (base, ("suffix_stages", "prepare")),
            (diagnostic.parent, ("prepare", "run_control")),
            (diagnostic, ("control_report", "scalar")),
            (layer, ("stage_report", "expected_stage")),
            (layer.prior.local, ("local_reference",)),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden))
        yield


@contextmanager
def write_scope(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_mutations"] += 1
        raise RuntimeError("mutation outside exclusive intervention writes forbidden")

    with base.write_scope(OUTPUT, audit), ExitStack() as stack:
        for name in ("mkdir", "makedirs", "unlink", "remove", "rmdir", "rename",
                     "replace", "link", "symlink", "chmod", "truncate", "utime"):
            stack.enter_context(patch.object(os, name, forbidden))
        yield


def new_audit():
    return {
        "controls": [], "s18_word_interventions": 0, "scalar_gate_evaluations": 0,
        "final_rmsnorm_invocations": 0, "lm_head_invocations": 0, "top_k_invocations": 0,
        "native_L0_L23_invocations": 0, "attention_MLP_invocations": 0,
        "reference_producer_invocations": 0, "prefix_admission_invocations": 0,
        "external_invocations": 0, "retained_evidence_writes": 0,
        "forbidden_calls": 0, "forbidden_mutations": 0, "file_write_opens": 0,
    }


def run_control(label, actual, arrays, reports, operands, references, audit):
    require(len(audit["controls"]) < 9 and label == CONTROLS[len(audit["controls"])],
            "extra/reordered intervention control")
    checks = preserve(actual, arrays)
    require(arrays["intervened_stage18"].dtype.str == "<u2"
            and arrays["intervened_stage18"].shape == (896,)
            and [f"{int(w):04x}" for w in arrays["intervened_stage18"]]
            == [r["selected_word"] for r in reports]
            and all(r["intervened_gate"]["accepted"] and r["passing_word_preserved"]
                    and r["changed"] == (not r["retained_gate"]["accepted"]) for r in reports),
            "intervention vector/acceptance lineage defect")
    audit["controls"].append(label)
    audit["s18_word_interventions"] += 1
    audit["scalar_gate_evaluations"] += 2 * len(reports)
    start = time.monotonic()
    audit["final_rmsnorm_invocations"] += 1
    arrays["intervened_final_rmsnorm"], details = final.rmsnorm(arrays["intervened_stage18"], operands[0])
    norm_end = time.monotonic()
    audit["lm_head_invocations"] += 1
    arrays["intervened_final_logits"] = final.logits(arrays["intervened_final_rmsnorm"], operands[1])
    head_end = time.monotonic()
    audit["top_k_invocations"] += 3
    comparisons = final.comparisons(
        {"rmsnorm": arrays["intervened_final_rmsnorm"], "logits": arrays["intervened_final_logits"]},
        references)
    margins = base.margin_rows(actual["final_logits"], arrays["intervened_final_logits"], references)
    same(preserve(actual, arrays), checks, "post-execution protection defect")
    changed = [r for r in reports if r["changed"]]
    return {
        "control": label, "S18_status": "PASS", "coordinates": len(reports),
        "retained_S18_failures": sum(not r["retained_gate"]["accepted"] for r in reports),
        "changed_coordinates": len(changed), "passing_coordinates_preserved": len(reports) - len(changed),
        "S18_binary64_failures": sum(not r["intervened_gate"]["accepted"] for r in reports),
        "maximum_excess": str(max(Fraction(r["intervened_gate"]["excess_error"]) for r in reports)),
        "minimum_threshold_margin": str(min(Fraction(r["threshold_margin"]) for r in reports)),
        "maximum_absolute_word_delta": str(max(Fraction(r["absolute_delta"]) for r in reports)),
        "tie_coordinates": [r["index"] for r in reports if r["tie_present"]],
        "coordinate_reports": reports, "final_rescued": "NOT_DEFINED",
        "final_comparisons": comparisons, "paired_final_margins": margins,
        "preservation_checks": checks, "source_operand_state_KV_lineage": "AUTHENTICATED",
        "rmsnorm_integer_details": details,
        "timing_seconds": {"final_rmsnorm": norm_end - start, "lm_head": head_end - norm_end,
                           "comparisons": time.monotonic() - head_end},
    }


def check_delivery(rows, audit):
    same([row["control"] for row in rows], list(CONTROLS), "incomplete/reordered controls")
    same(audit["controls"], list(CONTROLS), "dispatch order defect")
    for row in rows:
        require(row["S18_status"] == "PASS" and row["S18_binary64_failures"] == 0
                and row["coordinates"] == 896 and row["final_rescued"] == "NOT_DEFINED",
                "invalid scalar acceptance or invented final-head predicate")
        require(row["preservation_checks"] and all(row["preservation_checks"].values()),
                "retained-state preservation defect")
        same(row["changed_coordinates"], row["retained_S18_failures"],
             "changed coordinates differ from retained failure set")
    expected = new_audit()
    expected.update(controls=list(CONTROLS), s18_word_interventions=9,
                    scalar_gate_evaluations=16128, final_rmsnorm_invocations=9,
                    lm_head_invocations=9, top_k_invocations=27, file_write_opens=11)
    same(audit, expected, "dispatch/write census defect")


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
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    require(result.wasSuccessful() and not result.skipped, "focused tests failed:\n" + output.getvalue())
    return {"compiled": compiled, "executed": result.testsRun, "failures": 0, "errors": 0,
            "skipped": 0, "raw_output": output.getvalue(), "retained_control_dispatches": 0}


def execute():
    fresh_output()
    origins = source_context()
    diagnostic.parent.native.torch.set_num_threads(1)
    require(str(diagnostic.parent.native.torch.tensor(0).device) == "cpu", "non-CPU default device")
    OUTPUT.mkdir(exist_ok=False)
    audit, rows, artifacts = new_audit(), [], []

    def save(name, value):
        with (OUTPUT / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return record(OUTPUT / name)

    try:
        with write_scope(audit):
            artifacts.append(save("command.json", {
                "task_id": TASK, "revision": 1, "argv": sys.argv, "cwd": str(ROOT),
                "uid": os.getuid(), "executable": sys.executable, "origins": origins,
                "host_managed_controls": "unchanged role/model/account/budget/access/concurrency locks",
                "execution_allowance": "one; exclusive output; no automatic replay",
            }))
            tests = focused_tests(origins)
            artifacts.append(save("compile_tests.json", tests))
            started = time.monotonic()
            with final_only(audit):
                retained, combined, trajectory, reference64, payloads, references, authentication = authenticate()
                operands = final.load_operands(combined["preflight"])
                table = diagnostic.finite_table()
                prepared = []
                for row, actual in zip(retained["controls"], payloads, strict=True):
                    arrays, reports = intervene(actual, row, reference64, table)
                    prepared.append((row, actual, arrays, reports))
                auth_end = time.monotonic()
                for row, actual, arrays, reports in prepared:
                    entry = run_control(row["control"], actual, arrays, reports, operands, references, audit)
                    with (OUTPUT / (row["control"] + ".npz")).open("xb") as stream:
                        np.savez(stream, **arrays)
                    pin = record(OUTPUT / (row["control"] + ".npz"))
                    artifacts.append(pin)
                    entry.update({
                        "raw_payload": pin, "parent": row["parent"],
                        "intervention_input": row["raw_payload"],
                        "baseline_final_margins_source": row["raw_input"],
                        "historical_S18_status": row["historical_S18_status"],
                        "intervention_field": "intervened_stage18",
                        "retained_stage18_and_Q24_state_unchanged": True,
                    })
                    rows.append(entry)
                    print(json.dumps({k: v for k, v in entry.items()
                                      if k not in ("coordinate_reports", "parent", "preservation_checks")}),
                          flush=True)
            same(source_context(), origins, "source changed during execution")
            for pin in [RESULT_PIN, *authentication["parent_authentication"]["baseline_authenticated_pins"]]:
                read_bound(pin)
            for row in retained["controls"]:
                read_bound(row["raw_payload"])
                read_bound(row["raw_input"])
            same(final.preflight.bind_assets(authentication["parent_authentication"]["checkpoint"]),
                 combined["preflight"]["assets"], "operands changed during execution")
            check_delivery(rows, audit)
            result = {
                "task_id": TASK, "revision": 1, "diagnostic_id": NAME,
                "execution_status": "COMPLETE", "execution_count": 1,
                "delivery_status": "AWAITING_INDEPENDENT_REVIEW", "S18_status": "PASS",
                "final_rescued": "NOT_DEFINED", "controls": rows,
                "audit": {**audit, "file_write_opens": audit["file_write_opens"] + 1},
                "artifacts": artifacts, "authentication": authentication, "origins": origins,
                "preflight": combined["preflight"], "thresholds": combined["thresholds"],
                "final_reference": combined["preflight"]["final_reference"], "tests": tests,
                "output": str(OUTPUT), "output_created_exclusively": True,
                "expected_file_count": 12, "arithmetic": final.ARITHMETIC,
                "permitted_read_only_identity_subprocesses": 3,
                "external_invocation_audit_scope": "guarded authentication/intervention/final suffix; excludes git identity probes",
                "authentication_and_selection_seconds": auth_end - started,
                "flags": FLAGS, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
                "automatic_replay_authorized": False,
            }
            save("result.json", result)
            same(audit, result["audit"], "persisted write census mismatch")
            same(sorted(path.name for path in OUTPUT.iterdir()),
                 sorted(["command.json", "compile_tests.json", "result.json"]
                        + [label + ".npz" for label in CONTROLS]), "artifact census defect")
            print(json.dumps({
                "task_id": TASK, "execution_status": "COMPLETE", "S18_status": "PASS",
                "final_rescued": "NOT_DEFINED", "audit": audit, "output": str(OUTPUT),
                "compiled": len(tests["compiled"]), "tests_executed": tests["executed"],
                "flags": FLAGS, "claim_boundary": BOUNDARY,
            }), flush=True)
            return result
    except Exception as error:
        with write_scope(audit):
            save("failure.json", {
                "task_id": TASK, "status": "UNKNOWN", "error_type": type(error).__name__,
                "error": str(error), "controls": rows,
                "audit": {**audit, "file_write_opens": audit["file_write_opens"] + 1},
                "flags": FLAGS, "scientific_result_emitted": False, "automatic_replay_authorized": False,
            })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    execute()


if __name__ == "__main__":
    from importlib import import_module

    import_module(MODULE).main()
