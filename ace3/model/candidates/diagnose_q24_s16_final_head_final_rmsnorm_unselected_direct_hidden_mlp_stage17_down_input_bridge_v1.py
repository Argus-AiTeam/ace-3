"""Stdout-only exact down-input accounts for retained middle-pair MLP maxima."""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_coordinate_hotspot_audit_v1 as hotspot
from ace3.model.candidates import diagnose_q24_s16_final_head_l23_mlp_down_projection_input_coordinates_v1 as down


bridge, base, parent = hotspot.bridge, hotspot.base, hotspot.parent
require, same = base.require, base.same
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_down_input_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 12
FLAGS = {**hotspot.FLAGS, "mlp_down_projection_replay": 0,
         "down_input_causal_allocation": False}
BOUNDARY = (
    "Only the retained middle-pair MLP-stage17 maximum coordinates, including "
    "all maximum ties, are bridged through all 4864 L23 stage16 FP16 inputs. "
    "This is not the earlier final-head-channel down-input selection. Native "
    "G128 asymmetric INT4 GEMM ordering, no qzero plus-one, and FP16 scales "
    "are unchanged. Both branches use original-input FP16 internal stages, "
    "with their own retained final-RMSNorm/tied-head scalar factor. No binary64 "
    "internal stage is reconstructed. The explicit projection-boundary remainder "
    "is not solely rounding. Absolute input mass need not equal absolute retained "
    "output: cancellation with the boundary is reported separately. Maximum "
    "localization and this isolated accounting do not override the common "
    "UNKNOWN stop gate or authorize further expansion. " + hotspot.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("down-input bridge forbids predecessor execution and writes")

    with hotspot.read_only(audit), ExitStack() as stack:
        for module, names in (
            (hotspot, ("check", "run_tests")),
            (down, ("check", "focused_tests", "measure", "report")),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden))
        yield


def load_tensors(evidence):
    canonical = evidence["result"]["preflight"]["L23_original_reference"]["reference"]["canonical"]
    same(sorted(k for k in canonical if k.startswith(down.PREFIX)),
         sorted(down.PREFIX + k for k in down.SPECS), "down tensor census changed")
    with parent.preflight.parent.producer.legacy.safe_open(
            evidence["assets"]["checkpoint"]["path"], framework="numpy") as model:
        return {k: down.bind_tensor(model.get_tensor(down.PREFIX + k),
                                    canonical[down.PREFIX + k], k) for k in down.SPECS}


def bind_inputs(evidence, tensors):
    reference, _ = hotspot.terminal.bound_vectors(evidence)
    binding = evidence["result"]["preflight"]["L23_original_reference"]["reference"]
    same(sorted(tensors), sorted(down.SPECS), "down tensor set changed")
    for suffix in down.SPECS:
        down.bind_tensor(tensors[suffix], binding["canonical"][down.PREFIX + suffix], suffix)
    actual = {}
    sources = [(None, evidence["reference_archive"], binding["fp16"])]
    sources.extend((r["control"], evidence["archives"][r["control"]],
                    r["parent"]["terminal_archive"]) for r in evidence["result"]["controls"])
    for control, archive, pin in sources:
        authenticated = down.residual.read_archive(pin)
        for stage in ("stage16", "stage17"):
            require(archive[stage].dtype == authenticated[stage].dtype
                    and np.array_equal(archive[stage], authenticated[stage]),
                    "retained down-input archive binding changed")
        values = {
            "stage16": down.residual.words(archive["stage16"], (down.WIDTH,)),
            "stage17": down.residual.words(archive["stage17"], (down.OUTPUT_WIDTH,)),
        }
        if control is None:
            require(values["stage17"] == reference["stage17"], "reference stage17 splice")
            reference = values
        else:
            require(values["stage17"] == evidence["actual"][control]["stage17"], "actual stage17 splice")
            actual[control] = values
    same(list(actual), list(hotspot.census.CONTROLS), "input control order changed")
    return actual, reference


def input_order(terms):
    ordered = sorted(range(len(terms)), key=lambda i: (-abs(terms[i]), i))
    groups = {}
    for i in ordered:
        groups.setdefault(str(abs(terms[i])), []).append(i)
    return {
        "absolute_coordinate_order": ordered,
        "selected_coordinates": ordered[:down.LIMIT],
        "maximum_magnitude_ties": (
            groups[str(abs(terms[ordered[0]]))] if ordered and terms[ordered[0]] else []),
        "magnitude_tie_groups": [
            {"absolute_contribution": value, "coordinates": ids}
            for value, ids in groups.items() if len(ids) > 1],
        "hotspot_order_meaningful": any(terms),
    }


def account(actual, reference, weights, retained):
    coordinate = retained["coordinate"]
    raw = down.account(actual, reference, weights, coordinate)
    same(raw["retained_projection_delta"], retained["mlp_stage17_hidden_delta"],
         "retained MLP coordinate delta changed")
    factor = Fraction(retained["retained_weighted_row_factor"])
    require(factor != 0, "retained maximum has zero scalar factor")
    deltas = [a-r for a, r in zip(actual["stage16"], reference["stage16"], strict=True)]
    terms = [a*w for a, w in zip(deltas, weights, strict=True)]
    weighted = [factor*v for v in terms]
    order = input_order(terms)
    same(order["selected_coordinates"],
         [r["coordinate"] for r in raw["largest_absolute_coordinates"]],
         "down helper input selection changed")
    selected = set(order["selected_coordinates"])
    exact = sum(weighted, Fraction())
    remainder = factor*Fraction(raw["down_projection_boundary_remainder"])
    retained_signed = Fraction(retained["mlp_stage17_signed_contribution"])
    require(exact+remainder == retained_signed, "weighted projection closure changed")
    require(abs(retained_signed) == Fraction(retained["mlp_stage17_absolute_contribution"]),
            "retained absolute coordinate contribution changed")
    return {
        "output_coordinate": coordinate, "retained_hotspot": retained,
        "down_projection": raw, "input_order": order,
        "input_coordinates": [
            {"coordinate": i, "input_group": i//128,
             "input_delta": str(deltas[i]), "weight": str(weights[i]),
             "signed_contribution": str(terms[i]), "absolute_contribution": str(abs(terms[i])),
             "weighted_signed_contribution": str(weighted[i]),
             "weighted_absolute_contribution": str(abs(weighted[i]))}
            for i in range(down.WIDTH)],
        "weighted_input_total": bridge.mass(weighted),
        "weighted_selected_total": bridge.mass([weighted[i] for i in range(down.WIDTH) if i in selected]),
        "weighted_unselected_total": bridge.mass([weighted[i] for i in range(down.WIDTH) if i not in selected]),
        "weighted_groups_in_input_order": [
            {"input_group": g, **bridge.mass(weighted[g*128:(g+1)*128])}
            for g in range(down.GROUPS)],
        "weighted_projection_boundary_remainder": str(remainder),
        "weighted_projection_boundary_absolute": str(abs(remainder)),
        "retained_coordinate_signed_contribution": str(retained_signed),
        "retained_coordinate_absolute_contribution": str(abs(retained_signed)),
        "input_and_boundary_cancellation_mass": str(
            sum(map(abs, weighted), Fraction())+abs(remainder)-abs(retained_signed)),
        "weighted_projection_closure_residual": str(exact+remainder-retained_signed),
        "exact_signed_and_absolute_closure": True,
    }


def report(evidence, census, dominance, observed, tensors):
    same(observed, hotspot.audit_coordinates(evidence, census, dominance),
         "retained hotspot audit binding changed")
    actual, reference = bind_inputs(evidence, tensors)
    rows, columns = [], {}
    for row in observed["pairs"][0]["rows"]:
        maxima = row["top_mlp_stage17_magnitude_ties"]
        require(maxima and maxima == sorted(set(maxima)), "invalid retained maximum ties")
        entries = {r["coordinate"]: r for r in row["coordinates"]}
        accounts = []
        for coordinate in maxima:
            if coordinate not in columns:
                columns[coordinate] = down.weight_column(tensors, coordinate)
            accounts.append(account(actual[row["control"]], reference, columns[coordinate],
                                    entries[coordinate]))
        rows.append({
            "control": row["control"], "branch": row["branch"],
            "hidden_reference": row["hidden_reference"],
            "rmsnorm_reference": row["rmsnorm_reference"],
            "binary64_internal_stages": row["binary64_internal_stages"],
            "excluded_selected_coordinates": row["excluded_selected_coordinates"],
            "retained_maximum_coordinate_ties": maxima,
            "retained_gate_values": row["retained_gate_values"],
            "hotspots": accounts,
        })
    return {
        "left_id": hotspot.PAIR[0], "right_id": hotspot.PAIR[1],
        "layer": 23, "position": 0, "row_count": len(rows),
        "projection_account_count": sum(len(r["hotspots"]) for r in rows),
        "input_coordinate_order_rule": "descending absolute contribution, then ascending input coordinate",
        "selection_scope": "retained middle-pair MLP-stage17 maxima, including every maximum tie",
        "rows": rows, "retained_hotspot_audit": observed,
        "common_component": observed["common_component"],
        "stop_nested_bridge_expansion": observed["stop_nested_bridge_expansion"],
        "prior_pair_and_common_gates_unchanged": True,
        "down_projection_tensor_bindings": {
            k: evidence["result"]["preflight"]["L23_original_reference"]["reference"]["canonical"][down.PREFIX+k]
            for k in down.SPECS},
    }


def validate_report(evidence, census, dominance, observed, tensors, candidate):
    same(candidate, report(evidence, census, dominance, observed, tensors),
         "down-input/group/gate report binding changed")


def run_tests(evidence, census, dominance, observed, tensors, result):
    spec = importlib.util.spec_from_file_location("mlp_stage17_down_input_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = (evidence, census, dominance, observed, tensors, result)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "down-input tests failed, errored or skipped")
    return {"executed": outcome.testsRun, "failures": len(outcome.failures),
            "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               hotspot.MODULE: parent.record(hotspot.SOURCE),
               down.MODULE: parent.record(down.SOURCE),
               "mlp_stage17_down_input_tests": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    with bridge.read_only(audit):
        evidence = bridge.measure()
    with read_only(audit):
        census = hotspot.census.census(evidence["report"])
        dominance = hotspot.dominance.audit_census(census, evidence["report"])
        observed = hotspot.audit_coordinates(evidence, census, dominance)
        tensors = load_tensors(evidence)
        result = report(evidence, census, dominance, observed, tensors)
        tests = run_tests(evidence, census, dominance, observed, tensors, result)
        for pin in (*origins.values(), *base.PINS.values(), *bridge.margin.rows.PINS.values(),
                    *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "down-input bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_MIDDLE_PAIR_MLP_STAGE17_DOWN_INPUT_BRIDGE",
        "command": COMMAND, "diagnostic_sources": origins,
        "pins": {**base.PINS, **bridge.margin.rows.PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "authenticated_files": evidence["files"], "hidden_pins": evidence["hidden_pins"],
        "assets": evidence["assets"],
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "retained_thresholds": evidence["result"]["preflight"]["thresholds"],
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": {"compiled": compiled, **tests},
        "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        "input_mode": "authenticated retained in-memory hotspot audit and original L23 operands",
        "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
