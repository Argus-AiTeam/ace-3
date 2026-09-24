"""Task 8cc9ae757dd1: one final-input intervention, diagnostic-only.

--execute exclusively creates attempt001, compiles the bound sources and runs
focused synthetic tests before the nine retained controls. Never rerun it.
The raw result awaits Host independent review; rescue is NOT_DEFINED because
there is no predeclared final-head predicate. Decoder gates are history only.
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

from ace3.model.candidates import q24_s16_l23_p0_value_content_sufficiency_v1 as base


ROOT = base.ROOT
NAME = "q24_s16_l23_p0_final_rmsnorm_input_rescue_v1"
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
TASK = "8cc9ae757dd1"
CONTROLS = base.CONTROLS
final, layer, native = base.final, base.layer, base.native
require, same, record, read_bound = base.require, base.same, base.record, base.read_bound
EXPECTED_TESTS = 20
FLAGS = {
    **base.FLAGS, "token_published": False, "original_trajectory_rescued": False,
    "decoder_recomputed": False, "final_input_native_lineage_validated": False,
}
BOUNDARY = (
    "Fixed original-input FP16 L23 stage18 substitution only at final RMSNorm input. "
    "All retained decoder fields, Q24 I/Z/H, FP16 KV, source identities and "
    "historical FAILs remain unchanged. Only final RMSNorm/tied-head/top-k execute. "
    "Existing comparisons are diagnostic-only; rescued=NOT_DEFINED because no "
    "predeclared final-head predicate exists. No decoder S18 gate is applied to "
    "this intervention. Q24 state is wider than FP16; native G128 asymmetric INT4 "
    "GEMM nibble order, no qzero plus-one, FP16 scales/operators/KV remain unchanged. "
    "No prefix/admission/reference replay, trajectory rescue, strict-FP16-state "
    "W4A16, new-token, full-model, hardware, GPU, RTL or ACE2 claim/change."
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
    # Reuse the P0 source/KV gates, discarding the other experiment's substitution.
    parent, _ = base.prepare(actual, reference)
    layer.prior.retained.verify_parent(parent, parent)
    layer.prior.retained.check_stage_state(18, actual, parent)
    layer.prior.local.finite_words(reference["stage18"], (896,))
    require("final_input_hidden" not in actual and "final_rmsnorm" not in actual
            and "final_logits" not in actual, "unexpected retained final fields")
    arrays = {key: value.copy() for key, value in actual.items()}
    arrays["final_input_hidden"] = reference["stage18"].copy()
    for value in arrays.values():
        value.flags.writeable = False
    return arrays


def protected(actual, arrays, reference):
    checks = {}
    for key, value in actual.items():
        other = arrays[key]
        checks[key] = (value.dtype == other.dtype and value.shape == other.shape
                       and value.tobytes() == other.tobytes())
        require(checks[key], "protected retained field changed: " + key)
    words = arrays["final_input_hidden"]
    require(words.dtype == reference["stage18"].dtype
            and words.shape == (896,)
            and words.tobytes() == reference["stage18"].tobytes(),
            "final-input substitution lineage changed")
    return checks


@contextmanager
def final_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("decoder arithmetic forbidden at final-input boundary")

    with final.no_dispatch(audit), ExitStack() as stack:
        for module, name in ((base, "suffix_stages"), (native, "projection"),
                             (native.state, "add"), (layer, "expected_stage"),
                             (layer, "stage_report")):
            stack.enter_context(patch.object(module, name, forbidden))
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


def run_control(label, actual, arrays, reference, operands, references, baseline, audit):
    index = len(audit["controls"])
    require(index < 9 and label == CONTROLS[index], "extra or reordered control")
    checks = protected(actual, arrays, reference)
    audit["controls"].append(label)
    start = time.monotonic()
    audit["final_rmsnorm_invocations"] += 1
    arrays["final_rmsnorm"], details = final.rmsnorm(arrays["final_input_hidden"], operands[0])
    norm_end = time.monotonic()
    audit["lm_head_invocations"] += 1
    arrays["final_logits"] = final.logits(arrays["final_rmsnorm"], operands[1])
    head_end = time.monotonic()
    audit["top_k_invocations"] += 3
    metrics = final.comparisons(
        {"rmsnorm": arrays["final_rmsnorm"], "logits": arrays["final_logits"]}, references)
    margins = base.margin_rows(baseline["logits"], arrays["final_logits"], references)
    same(protected(actual, arrays, reference), checks, "post-execution preservation changed")
    return {
        "control": label, "rescued": "NOT_DEFINED", "final_comparisons": metrics,
        "paired_final_margins": margins, "preservation_checks": checks,
        "rmsnorm_integer_details": details,
        "timing_seconds": {"rmsnorm": norm_end - start, "lm_head": head_end - norm_end,
                           "comparisons": time.monotonic() - head_end},
    }


def check_delivery(rows, audit):
    same([row["control"] for row in rows], list(CONTROLS), "incomplete/reordered controls")
    same(audit["controls"], list(CONTROLS), "dispatch order changed")
    for row in rows:
        require(row["rescued"] == "NOT_DEFINED", "invented scientific classification")
        require(row["preservation_checks"] and all(row["preservation_checks"].values()),
                "retained-state preservation defect")
    for key, expected in (
        ("final_rmsnorm_invocations", 9), ("lm_head_invocations", 9),
        ("top_k_invocations", 27), ("native_L0_L23_invocations", 0),
        ("reference_producer_invocations", 0), ("external_invocations", 0),
        ("retained_evidence_writes", 0), ("forbidden_calls", 0),
        ("forbidden_mutations", 0), ("file_write_opens", 11),
    ):
        same(audit[key], expected, "dispatch/write census mismatch: " + key)


def check_artifact_census():
    same(sorted(path.name for path in OUTPUT.iterdir()),
         sorted(["command.json", "compile_tests.json", "result.json"]
                + [label + ".npz" for label in CONTROLS]), "final artifact census mismatch")


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
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "focused tests failed:\n" + stream.getvalue())
    return {"compiled": compiled, "executed": result.testsRun, "failures": 0, "errors": 0,
            "skipped": 0, "raw_output": stream.getvalue(), "retained_control_dispatches": 0}


def execute():
    fresh_output()
    native.torch.set_num_threads(1)
    require(str(native.torch.tensor(0).device) == "cpu", "non-CPU default device")
    origins = source_context()
    OUTPUT.mkdir(exist_ok=False)
    audit = {
        "controls": [], "final_rmsnorm_invocations": 0, "lm_head_invocations": 0,
        "top_k_invocations": 0, "native_L0_L23_invocations": 0,
        "reference_producer_invocations": 0, "external_invocations": 0,
        "retained_evidence_writes": 0, "forbidden_calls": 0,
        "forbidden_mutations": 0, "file_write_opens": 0,
    }
    rows, artifacts = [], []

    def save_json(name, value):
        with (OUTPUT / name).open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        return record(OUTPUT / name)

    try:
        with write_scope(audit):
            artifacts.append(save_json("command.json", {
                "task_id": TASK, "revision": 2, "argv": sys.argv, "cwd": str(ROOT),
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
            with final_only(audit):
                for row, actual, arrays in prepared:
                    label = row["control"]
                    entry = run_control(label, actual, arrays, trajectory, operands, references,
                                        baseline_arrays[label], audit)
                    with (OUTPUT / f"{label}.npz").open("xb") as stream:
                        np.savez(stream, **arrays)
                    pin = record(OUTPUT / f"{label}.npz")
                    artifacts.append(pin)
                    entry.update({
                        "parent": row["parent"], "raw_payload": pin,
                        "substitution": {
                            "source": summary["L23_original_reference"]["reference"]["fp16"],
                            "field": "stage18", "boundary": "final RMSNorm input only",
                            "stored_field": "final_input_hidden", "shape": [896], "dtype": "<u2",
                            "changed_fp16_components": int(np.count_nonzero(
                                actual["stage18"] != arrays["final_input_hidden"])),
                            "retained_stage18_unchanged": True, "retained_Q24_state_unchanged": True,
                            "retained_KV_source_identity_unchanged": True,
                        },
                    })
                    rows.append(entry)
                    print(json.dumps({k: v for k, v in entry.items() if k != "parent"}), flush=True)
            same(source_context(), origins, "source changed during execution")
            for pin in [*pins, checkpoint]:
                read_bound(pin)
            same(final.preflight.bind_assets(checkpoint), summary["assets"],
                 "operands/assets changed during execution")
            check_delivery(rows, audit)
            result = {
                "task_id": TASK, "revision": 2, "diagnostic_id": NAME,
                "delivery_status": "AWAITING_INDEPENDENT_REVIEW",
                "execution_status": "COMPLETE", "execution_count": 1,
                "rescued": "NOT_DEFINED",
                "classification_reason": "No predeclared final-head rescue predicate exists.",
                "controls": rows, "audit": audit, "artifacts": artifacts, "origins": origins,
                "input_pins": pins, "checkpoint": checkpoint, "preflight": summary,
                "thresholds": summary["thresholds"], "final_reference": summary["final_reference"],
                "tests": tests, "output": str(OUTPUT), "output_created_exclusively": True,
                "authentication_seconds": auth_seconds, "arithmetic": final.ARITHMETIC,
                "flags": FLAGS, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
                "automatic_replay_authorized": False, "expected_final_file_count": 12,
            }
            # The final exclusive result open is included in its own persisted census.
            result["audit"] = {**audit, "file_write_opens": audit["file_write_opens"] + 1}
            save_json("result.json", result)
            same(audit, result["audit"], "final write census mismatch")
            check_artifact_census()
            print(json.dumps({
                "task_id": TASK, "delivery_status": result["delivery_status"],
                "execution_status": "COMPLETE", "rescued": "NOT_DEFINED", "audit": audit,
                "output": str(OUTPUT), "compiled": len(tests["compiled"]),
                "tests": {key: tests[key] for key in ("executed", "failures", "errors", "skipped")},
                "flags": FLAGS, "claim_boundary": BOUNDARY,
            }), flush=True)
    except Exception as error:
        with write_scope(audit):
            save_json("failure.json", {
                "task_id": TASK, "delivery_status": "DEFECT", "error_type": type(error).__name__,
                "error": str(error), "audit": audit, "controls": rows, "flags": FLAGS,
                "scientific_result_emitted": False, "automatic_replay_authorized": False,
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
