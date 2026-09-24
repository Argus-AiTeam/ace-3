"""One retained middle-pair MLP hotspot intervention; CPU, stdout-only, non-admission."""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from ace3.model.candidates import diagnostic_capture_v1 as capture
from ace3.model.candidates import diagnose_q24_s16_final_head_impact_census_v1 as retained
from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_component_hotspot_audit_v1 as direct
from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_hotspot_stability_classifier_v1 as hotspot
from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_value_content_suffix_intervention_v1 as guards


ROOT = hotspot.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mlp_stage17_hotspot_suffix_intervention_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = hotspot.PYTHON
CONTROLS = hotspot.CONTROLS
BRANCHES = ("fp16", "binary64")
PAIR = (34319, 13)
COMPONENT = "mlp_stage17"
parent, layer, native = guards.parent, guards.layer, guards.native
require, same, bound_bytes = hotspot.require, hotspot.same, hotspot.bound_bytes
ENVIRONMENT = {
    "HOME": "/home/argustest", "PATH": "/home/argustest/miniconda3/bin:/usr/bin:/bin",
    "LC_ALL": "C.UTF-8", "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
}
FLAGS = {
    **layer.FLAGS, "projection_invocations": 0, "reference_producer_invocations": 0,
    "retained_evidence_writes": 0, "full_vocabulary_head_recomputation": False,
    "token_selection": False,
}
COUNTS = {"forbidden_calls": 0, "s18_adds": 9, "state_oracles": 9,
          "final_norms": 9, "pair_heads": 9}
BOUNDARY = (
    "One coordinate, L23/P0, pair (34319,13), nine retained controls. Only S18, "
    "final RMSNorm and two tied-head rows execute. Original-input FP16/binary64 "
    "references, native S16 RTZ, G128 asymmetric INT4 GEMM nibble ordering without "
    "qzero plus-one, FP16 scales/operator boundaries/KV and wider-than-FP16 Q24 "
    "residual state are unchanged. No prefix/admission/reference/closed producer "
    "replay, strict-FP16-state W4A16, new-token or full-model admission. Stop this "
    "branch after this observation and independent Host review; no new ranking."
)
PREREGISTRATION = {
    "selection": "Within the retained middle pair only, maximize the sum of exact "
                 "absolute mlp_stage17 coordinate masses across all nine controls "
                 "and both reference branches; ties choose ascending coordinate. "
                 "Selection precedes all suffix calls.",
    "replacement": "The same working stage17[k] becomes original-input FP16 stage17[k] "
                   "in all controls. Every other prefix/MLP value and input/scratch/KV "
                   "byte remains unchanged.",
    "prediction": "Negative retained weighted mlp_stage17 contribution at k. Both "
                  "fixed-weight hotspot movement and actual pair-margin movement "
                  "must strictly agree with its nonzero sign in every row.",
    "SUPPORTED": "All integrity and existing S18 numerical gates pass, and all 18 "
                 "rows have both strict directional agreements.",
    "REJECTED": "An authenticated zero/opposite movement, prediction mismatch or "
                "existing numerical failure rejects bounded local sufficiency. "
                "Nonlinear S18/final-boundary/cancellation remains a possible "
                "obstruction, not a uniquely established causal mechanism.",
    "UNKNOWN": "Authentication, source/test/capture, state/KV/lineage, zero prediction "
               "or execution-integrity defect.",
}


def exact(word):
    value = float(np.asarray(word, dtype="<u2").view("<f2"))
    require(np.isfinite(value), "nonfinite FP16 operand")
    return Fraction.from_float(value)


def middle(control):
    pairs = [p for p in control["pairs"] if (p["left_id"], p["right_id"]) == PAIR]
    same(len(pairs), 1, "middle pair missing or duplicated")
    return pairs[0]


def preregister(evidence):
    controls = evidence["report"]["controls"]
    raw = evidence["retained_50f"]["report"]["controls"]
    same([c["control"] for c in controls], list(CONTROLS), "hotspot control census")
    same([c["control"] for c in raw], list(CONTROLS), "direct-hidden control census")
    masses, accounts = {}, []
    for control, original in zip(controls, raw, strict=True):
        for branch in BRANCHES:
            table = middle(control)["branches"][branch]
            source = middle(original)["branches"][branch]
            same(source["residual_internal_reference"], "original_input_L23_fp16",
                 "internal reference reanchored")
            same(source["hidden_reference"], "original_input_L23_" + branch,
                 "global reference reanchored")
            same(source["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION",
                 "binary64 internal reconstruction")
            ranking = table["per_component_coordinate_hotspots"][COMPONENT]
            cells = source["selected_coordinates"]
            indices = [r["coordinate"] for r in cells]
            require(indices == sorted(set(indices)) and all(type(k) is int and 0 <= k < 896
                                                           for k in indices),
                    "coordinate census")
            ranked = {r["coordinate"]: r for r in ranking["ranking"]}
            same(len(ranked), len(ranking["ranking"]), "duplicate hotspot")
            same(set(ranked), set(indices), "hotspot/direct-hidden splice")
            for cell in cells:
                k = cell["coordinate"]
                value = Fraction(cell["weighted_components"][COMPONENT])
                factor = Fraction(cell["weight_times_reference_anchor_times_row_difference"])
                same(value, factor * Fraction(cell["hidden_components"][COMPONENT]),
                     "weighted component closure")
                same(Fraction(ranked[k]["signed"]), value, "hotspot signed mass splice")
                same(Fraction(ranked[k]["absolute"]), abs(value), "hotspot mass splice")
                masses[k] = masses.get(k, Fraction()) + abs(value)
            accounts.append({"control": control["control"], "branch": branch,
                             "cells": cells, "retained_maximum_ties": ranking["maximum_ties"]})
    require(bool(masses), "empty retained selection")
    coordinate = min(masses, key=lambda k: (-masses[k], k))
    rows = []
    for account in accounts:
        cells = [c for c in account["cells"] if c["coordinate"] == coordinate]
        same(len(cells), 1, "selected hotspot absent from retained row")
        cell = cells[0]
        prediction = -Fraction(cell["weighted_components"][COMPONENT])
        require(prediction != 0, "zero retained prediction")
        rows.append({k: v for k, v in account.items() if k != "cells"} | {
            "predicted_delta": str(prediction), "retained_cell": cell})
    return {**PREREGISTRATION, "coordinate": coordinate, "component": COMPONENT,
            "pair": list(PAIR), "source_position": 0, "source_token_id": 9707,
            "selection_masses": {str(k): str(masses[k]) for k in sorted(masses)}, "rows": rows}


def prepare(actual, reference, coordinate):
    require(type(coordinate) is int and 0 <= coordinate < 896, "invalid coordinate")
    arrays = {k: v.copy() for k, v in actual.items()}
    require(actual["stage17"].dtype.str == reference["stage17"].dtype.str == "<u2"
            and actual["stage17"].shape == reference["stage17"].shape == (896,),
            "stage17 operand shape/dtype")
    arrays["stage17"][coordinate] = reference["stage17"][coordinate]
    for value in arrays.values():
        value.flags.writeable = False
    protected(actual, arrays, reference, coordinate)
    return arrays


def protected(actual, arrays, reference, coordinate):
    for key, value in actual.items():
        if key in ("stage17", "stage18", "output_i", "output_z"):
            continue
        other = arrays[key]
        require(value.dtype == other.dtype and value.shape == other.shape
                and value.tobytes() == other.tobytes(), "protected operand/state/KV: " + key)
    expected = actual["stage17"].copy()
    expected[coordinate] = reference["stage17"][coordinate]
    require(expected.dtype == arrays["stage17"].dtype
            and expected.shape == arrays["stage17"].shape
            and expected.tobytes() == arrays["stage17"].tobytes(), "replacement operand changed")
    for kind, stage in (("k", "05"), ("v", "03")):
        same(actual["output_cache_" + kind].shape, (1, 128), "P0 KV shape")
        require(np.array_equal(actual["output_cache_" + kind][0], actual["stage" + stage]),
                "retained KV lineage")


@contextmanager
def suffix_only(audit, active, operands):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("prefix/reference/producer/write/out-of-suffix dispatch forbidden")

    raw_add, raw_expected = native.state.add, layer.expected_stage
    raw_norm, raw_head = parent.rmsnorm, parent.logits

    def add(state, words):
        require(active.get("stage") == 18 and state is active["scratch"]
                and words is active["arrays"]["stage17"], "S18 operand scope")
        audit["s18_adds"] += 1
        return raw_add(state, words)

    def expected(stage, arrays, state, tensors):
        require(stage == active.get("stage") == 18 and arrays is active["arrays"]
                and state is active["input"] and tensors is active["tensors"], "state oracle scope")
        audit["state_oracles"] += 1
        return raw_expected(stage, arrays, state, tensors)

    def norm(words, weights):
        require(active.get("stage") == 19 and words is active["arrays"]["stage18"]
                and weights is operands[0], "RMSNorm operand scope")
        audit["final_norms"] += 1
        return raw_norm(words, weights)

    def head(words, weights):
        require(active.get("stage") == 20 and words is active["arrays"]["final_rmsnorm"]
                and weights is operands[1] and weights.shape == (2, 896), "two-row head scope")
        audit["pair_heads"] += 1
        return raw_head(words, weights)

    with parent.no_dispatch(audit), guards.no_writes(audit), ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates.") and name != MODULE:
                for attribute in ("check", "run_tests", "focused_tests", "collect", "report",
                                  "measure", "validate", "run_control", "suffix_stages"):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        for module, name, function in (
            (native, "projection", forbidden), (native.state, "add", add),
            (layer, "expected_stage", expected), (parent, "rmsnorm", norm),
            (parent, "logits", head), (parent, "top_k", forbidden),
        ):
            stack.enter_context(patch.object(module, name, function))
        yield


def classify(rows, reports):
    same([(r["control"], r["branch"]) for r in rows],
         [(c, b) for c in CONTROLS for b in BRANCHES], "directional census")
    same(list(reports), list(CONTROLS), "numerical census")
    require(all(r["stage"] == 18 and r["status"] in ("PASS", "FAIL")
                for r in reports.values()), "S18 gate integrity")
    agreements = []
    for row in rows:
        p, m, delta = (Fraction(row[k]) for k in
                       ("predicted_delta", "mlp_maxima_delta", "margin_delta"))
        require(p != 0, "zero retained prediction")
        agreement = p * m > 0 and p * delta > 0 and m == p
        same(row["direction_observed"], agreement, "direction result splice")
        agreements.append(agreement)
    return ("SUPPORTED" if all(agreements) and all(r["status"] == "PASS" for r in reports.values())
            else "REJECTED")


def capture_preflight():
    require(Path.cwd() == ROOT and sys.executable == PYTHON and os.getuid() == 1000
            and sys.dont_write_bytecode and not sys.flags.optimize, "source/account/Python gate")
    same({k: os.environ.get(k) for k in ENVIRONMENT}, ENVIRONMENT, "environment gate")
    out = Path(os.readlink("/proc/self/fd/1"))
    require(out.name == "check.stdout" and out.parent.name == "run"
            and out.parent.parent.parent == ROOT / "build" and out.resolve() == out,
            "byte-exact shared capture required")
    same(os.readlink("/proc/self/fd/2"), str(out.with_name("check.stderr")), "stderr capture gate")
    preflight = json.loads(out.with_name("check.environment.json").read_bytes())
    same(preflight["sources"], [capture.binding(SOURCE), capture.binding(TEST)], "source/test pin gate")
    same(preflight["environment"], ENVIRONMENT, "capture environment")
    same(preflight["capture_implementation"], capture.implementation_pins(), "shared sealer drift")
    same(preflight["model_or_service_calls_authorized"], 0, "service authority")
    same(preflight["independent_host_review"], "REQUIRED", "review authority")
    argv = [PYTHON, "-B", "-m", MODULE, "--check"]
    same(json.loads(out.with_name("check.argv.json").read_bytes()), argv, "check argv")
    same(out.with_name("check.command.txt").read_bytes(),
         (capture.environment_command(ENVIRONMENT, argv) + "\n").encode(), "check command")
    for label in ("compile", "pytest"):
        same(out.with_name(label + ".stderr").read_bytes(), b"", "validation stderr")
        require(out.with_name(label + ".whole-command.log").read_bytes().endswith(
            b"\nEXIT_STATUS=0\nTIMED_OUT=False\n"), "validation did not complete successfully")
    suites = ET.fromstring(out.with_name("pytest.xml").read_bytes()).findall("testsuite")
    require(suites and sum(int(s.attrib["tests"]) for s in suites) >= 12
            and not any(int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped")),
            "task-native tests failed or incomplete")
    return preflight, {"compiled": preflight["sources"],
                       "pytest": capture.binding(out.with_name("pytest.xml")),
                       "executed": sum(int(s.attrib["tests"]) for s in suites)}


def check():
    preflight, tests = capture_preflight()
    native.torch.set_num_threads(1)
    same(str(native.torch.tensor(0).device), "cpu", "CPU-only execution")
    sources = parent.source_context()
    sources[MODULE], sources["task_tests"] = capture.binding(SOURCE), capture.binding(TEST)
    spec = importlib.util.spec_from_file_location(NAME + "_tests", TEST)
    require(spec is not None and spec.loader is not None, "independent oracle loader")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    audit = dict.fromkeys(COUNTS, 0)
    began = time.monotonic()
    with guards.no_writes(audit), parent.no_dispatch(audit):
        evidence = hotspot.authenticate()
        original = direct.authenticate()
        same(evidence["retained_50f"], original, "reviewed parent splice")
        for pin in (*original["diagnostic_sources"].values(), *original["authenticated_files"],
                    *original["hidden_pins"]):
            bound_bytes(pin)
        plan = preregister(evidence)
        result, baseline, references, files = retained.authenticate()
        summary = result["preflight"]
        tensors, trajectory, binary64, checkpoint = layer.load_inputs({"summary": summary})
        same(parent.preflight.bind_assets(checkpoint), summary["assets"], "source/asset identity")
        full = parent.load_operands(summary)
        operands = (full[0], full[1][list(PAIR)].copy())
        for value in (*tensors.values(), *operands, *trajectory.values(), binary64,
                      *references.values()):
            value.flags.writeable = False
        archives = {r["control"]: layer.archive(r["parent"]["terminal_archive"])
                    for r in result["controls"]}
        same(list(archives), list(CONTROLS), "retained archive census")
        k = plan["coordinate"]
        # Bind the proposed replacement to the reviewed original-input boundary, not a local oracle.
        for row in plan["rows"]:
            cell = row["retained_cell"]
            boundary = cell["retained_fp16_residual_account"]["boundaries"]["stage17"]
            same(exact(archives[row["control"]]["stage17"][k]),
                 Fraction(boundary["actual_exact"]), "retained stage17 operand splice")
            same(exact(trajectory["stage17"][k]), Fraction(boundary["reference_exact"]),
                 "original-input stage17 operand splice")
        authentication_seconds = time.monotonic() - began
        outputs, reports, timings, active = {}, {}, {}, {}
        with suffix_only(audit, active, operands):
            for control in CONTROLS:
                actual = archives[control]
                arrays = prepare(actual, trajectory, k)
                state = {"i": arrays["input_i"], "z": arrays["input_z"], "h": arrays["input_hidden"]}
                scratch = {"i": arrays["scratch_i"], "z": arrays["scratch_z"], "h": arrays["stage12"]}
                active.update(stage=18, arrays=arrays, input=state, scratch=scratch, tensors=tensors)
                start = time.monotonic()
                changed = native.state.add(scratch, arrays["stage17"])
                arrays.update(output_i=changed["i"], output_z=changed["z"], stage18=changed["h"])
                expected = layer.expected_stage(18, arrays, state, tensors)
                reports[control] = layer.stage_report(18, arrays, trajectory, binary64, expected)
                protected(actual, arrays, trajectory, k)
                stage_end = time.monotonic()
                active["stage"] = 19
                arrays["final_rmsnorm"], norm_account = parent.rmsnorm(arrays["stage18"], operands[0])
                active["stage"] = 20
                arrays["pair_logits"] = parent.logits(arrays["final_rmsnorm"], operands[1])
                final_end = time.monotonic()
                oracle.verify_suffix(actual, arrays, trajectory, k, operands)
                outputs[control] = {key: arrays[key].tolist() for key in (
                    "stage17", "output_i", "output_z", "stage18", "final_rmsnorm", "pair_logits")}
                outputs[control]["norm_account"] = norm_account
                timings[control] = {"s18_and_gate": stage_end - start,
                                    "final_norm_and_two_rows": final_end - stage_end,
                                    "independent_oracle": time.monotonic() - final_end}
        rows = []
        for prediction in plan["rows"]:
            control, branch = prediction["control"], prediction["branch"]
            cell = prediction["retained_cell"]
            factor = Fraction(cell["weight_times_reference_anchor_times_row_difference"])
            mlp = factor * (exact(outputs[control]["stage17"][k]) - exact(archives[control]["stage17"][k]))
            old = exact(baseline[control]["logits"][PAIR[0]]) - exact(baseline[control]["logits"][PAIR[1]])
            new = exact(outputs[control]["pair_logits"][0]) - exact(outputs[control]["pair_logits"][1])
            ref = references["logits_" + branch]
            ref_margin = (exact(ref[PAIR[0]]) - exact(ref[PAIR[1]]) if branch == "fp16" else
                          Fraction.from_float(float(ref[PAIR[0]])) - Fraction.from_float(float(ref[PAIR[1]])))
            p, delta = Fraction(prediction["predicted_delta"]), new - old
            rows.append({"control": control, "branch": branch, "predicted_delta": str(p),
                         "mlp_maxima_delta": str(mlp), "mlp_prediction_remainder": str(mlp - p),
                         "retained_margin": str(old), "intervened_margin": str(new),
                         "independent_original_reference_margin": str(ref_margin),
                         "retained_margin_error": str(old - ref_margin),
                         "intervened_margin_error": str(new - ref_margin),
                         "margin_delta": str(delta), "margin_prediction_remainder": str(delta - p),
                         "direction_observed": p * mlp > 0 and p * delta > 0 and mlp == p})
        decision = classify(rows, reports)
        same(audit, COUNTS, "suffix/forbidden dispatch census")
        for pin in (*sources.values(), *files, *original["hidden_pins"],
                    *hotspot.PINS.values(), *direct.PINS.values()):
            bound_bytes(pin)
    return {
        "diagnostic_id": NAME, "version": 1, "status": decision,
        "artifact_authentication": "AUTHENTICATED", "preregistration": plan,
        "parent_pins": {"hotspot": hotspot.PINS, "direct_hidden": direct.PINS},
        "diagnostic_sources": sources, "authenticated_files": files,
        "capture_preflight": preflight, "tests": tests, "flags": FLAGS,
        "dispatch_and_write_audit": audit, "stage_reports": reports, "rows": rows,
        "outputs": outputs, "original_thresholds": summary["thresholds"],
        "L23_reference_authority": summary["L23_original_reference"],
        "final_reference_authority": summary["final_reference"],
        "retained_controls_and_failure_gates": original["retained_controls_and_failure_gates"],
        "retained_lineage_separation": original["report"]["lineage_separation"],
        "historical_failures_preserved": True, "original_global_reference_unchanged": True,
        "authentication_seconds": authentication_seconds, "timing_seconds": timings,
        "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    try:
        result = check()
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, AssertionError, ET.ParseError) as error:
        print(json.dumps({"diagnostic_id": NAME, "status": "UNKNOWN", "flags": FLAGS,
                          "error_type": type(error).__name__, "error": str(error),
                          "normal_host_review": "REQUIRED"}, sort_keys=True, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    from importlib import import_module

    raise SystemExit(import_module(MODULE).main())
