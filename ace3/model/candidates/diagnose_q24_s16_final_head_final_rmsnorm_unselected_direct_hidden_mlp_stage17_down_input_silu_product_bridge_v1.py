"""Stdout-only retained middle-pair stage16 gate/up/raw-product accounts."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_down_input_bridge_v1 as previous
from ace3.model.candidates import diagnose_q24_s16_final_head_l23_mlp_silu_product_bridge_v1 as silu


bridge, hotspot, down = previous.bridge, previous.hotspot, previous.down
base, parent = previous.base, previous.parent
require, same = base.require, base.same
ROOT = previous.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_down_input_silu_product_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 14
FLAGS = {**previous.FLAGS, "silu_replay": 0, "sigmoid_replay": 0,
         "product_operator_replay": 0, "stage16_causal_allocation": False}
BOUNDARY = (
    "Only the retained selected stage16 down inputs of the middle-pair "
    "(34319,13) MLP-stage17 maxima are decomposed, not the earlier final-head "
    "channel selections. Both branches use authenticated original-input L23 "
    "FP16 stage14 gate g, stage15 up u and stage16 y operands. The exact raw "
    "baseline g*u is NOT SiLU. With b=y-g*u separately on actual and reference, "
    "delta_y=delta_g*u_ref+g_ref*delta_u+delta_g*delta_u+delta_b. "
    "The boundary combines nonlinear SiLU and implementation/product/conversion "
    "differences; separate effects are NOT_IDENTIFIABLE_WITHOUT_REPLAY. "
    "Rational multiplication is retained-operand accounting, not evaluation "
    "of sigmoid, SiLU or product operators or an operator precision change. "
    "Each four-term sum closes through the authenticated down weight and retained "
    "coordinate factor to its selected contribution. Unselected contributions "
    "and the distinct down-projection remainder close the stage17 coordinate. "
    "Repeated selections are logical accounts, not independent samples, causal "
    "shares or permission to pass the common UNKNOWN stop gate. " + previous.BOUNDARY
)


@contextmanager
def nonlinear_read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("stage16 bridge forbids nonlinear/product operator replay")

    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.") and callable(getattr(module, "_stages", None)):
                stack.enter_context(patch.object(module, "_stages", forbidden))
        for name in ("numpy", "math", "torch", "torch.nn.functional"):
            module = sys.modules.get(name)
            if module is not None:
                for attribute in ("silu", "sigmoid", "exp", "expm1"):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        yield


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("stage16 bridge forbids predecessor execution and writes")

    with previous.read_only(audit), nonlinear_read_only(audit), ExitStack() as stack:
        for module, names in (
            (previous, ("check", "run_tests")),
            (silu, ("check", "focused_tests", "measure", "report", "hotspot_summary")),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden))
        yield


def bind_inputs(evidence, tensors):
    previous.bind_inputs(evidence, tensors)
    same(list(evidence["archives"]), list(hotspot.census.CONTROLS),
         "stage16 archive control order changed")
    binding = evidence["result"]["preflight"]["L23_original_reference"]["reference"]
    sources = [(None, evidence["reference_archive"], binding["fp16"])]
    sources.extend((row["control"], evidence["archives"][row["control"]],
                    row["parent"]["terminal_archive"])
                   for row in evidence["result"]["controls"])
    actual, reference = {}, None
    for control, archive, pin in sources:
        authenticated = down.residual.read_archive(pin)
        for stage in ("stage14", "stage15", "stage16"):
            require(archive[stage].dtype == authenticated[stage].dtype
                    and archive[stage].shape == authenticated[stage].shape
                    and np.array_equal(archive[stage], authenticated[stage]),
                    "retained gate/up/stage16 archive binding changed")
        values = silu.operands(archive)
        if control is None:
            reference = values
        else:
            actual[control] = values
    same(list(actual), list(hotspot.census.CONTROLS), "stage16 operand census changed")
    require(reference is not None, "original L23 stage16 reference missing")
    return actual, reference


def account(actual, reference, retained):
    factor = Fraction(retained["retained_hotspot"]["retained_weighted_row_factor"])
    coordinates = []
    for rank, coordinate in enumerate(retained["input_order"]["selected_coordinates"], 1):
        pin = retained["input_coordinates"][coordinate]
        same(pin["coordinate"], coordinate, "selected down-input coordinate splice")
        a, r = ({key: values[coordinate] for key, values in source.items()}
                for source in (actual, reference))
        local = silu.account(a, r, coordinate, Fraction(pin["weight"]))
        same(local["retained_stage16_delta"], pin["input_delta"], "stage16 delta splice")
        same(local["weighted_stage16_delta"], pin["signed_contribution"],
             "selected down-input contribution splice")
        weighted = [
            factor*Fraction(term["signed_contribution"])
            for term in local["weighted_terms_in_operand_order"]]
        require(sum(weighted, Fraction()) == Fraction(pin["weighted_signed_contribution"]),
                "selected coordinate-factor closure changed")
        coordinates.append({
            "stage16_coordinate_rank": rank, "selected_down_input": pin, "bridge": local,
            "coordinate_weighted_terms_in_operand_order": [
                {"term_index": i, "term": name, "signed_contribution": str(value),
                 "absolute_contribution": str(abs(value))}
                for i, (name, value) in enumerate(zip(silu.TERM_NAMES, weighted, strict=True))],
            "coordinate_weighted_term_totals": bridge.mass(weighted),
            "weighted_selected_input_closure_residual": str(
                sum(weighted, Fraction())-Fraction(pin["weighted_signed_contribution"])),
        })
    same(len(coordinates), down.LIMIT, "retained stage16 selection count changed")
    selected = [Fraction(c["selected_down_input"]["weighted_signed_contribution"])
                for c in coordinates]
    same(bridge.mass(selected), retained["weighted_selected_total"],
         "weighted selected input totals changed")
    terms = [Fraction(t["signed_contribution"]) for c in coordinates
             for t in c["coordinate_weighted_terms_in_operand_order"]]
    unselected = Fraction(retained["weighted_unselected_total"]["signed"])
    boundary = Fraction(retained["weighted_projection_boundary_remainder"])
    signed = Fraction(retained["retained_coordinate_signed_contribution"])
    closure = sum(terms, Fraction())+unselected+boundary-signed
    require(closure == 0, "selected stage16 to retained stage17 coordinate closure changed")
    return {
        "output_coordinate": retained["output_coordinate"],
        "retained_weighted_row_factor": str(factor),
        "selected_stage16_coordinates": coordinates,
        "summary": {
            "selected_coordinate_count": len(coordinates),
            "weighted_selected_total": retained["weighted_selected_total"],
            "coordinate_weighted_operand_terms_in_order": [
                {"term": name, "totals": bridge.mass([
                    Fraction(c["coordinate_weighted_terms_in_operand_order"][i]["signed_contribution"])
                    for c in coordinates])}
                for i, name in enumerate(silu.TERM_NAMES)],
            "coordinate_weighted_term_totals": bridge.mass(terms),
            "weighted_unselected_total": retained["weighted_unselected_total"],
            "weighted_projection_boundary_remainder": str(boundary),
            "retained_coordinate_signed_contribution": str(signed),
            "retained_coordinate_absolute_contribution": str(abs(signed)),
            "term_expansion_cancellation_mass": str(
                sum(map(abs, terms), Fraction())-sum(map(abs, selected), Fraction())),
            "expanded_input_and_boundary_cancellation_mass": str(
                sum(map(abs, terms), Fraction())
                + Fraction(retained["weighted_unselected_total"]["absolute"])
                + abs(boundary)-abs(signed)),
            "weighted_stage17_coordinate_closure_residual": str(closure),
            "exact_selected_to_stage17_identity": True,
        },
    }


def report(evidence, census, dominance, observed, tensors, retained):
    previous.validate_report(evidence, census, dominance, observed, tensors, retained)
    actual, reference = bind_inputs(evidence, tensors)
    rows = [
        {**{key: value for key, value in row.items() if key != "hotspots"},
         "hotspots": [account(actual[row["control"]], reference, entry)
                      for entry in row["hotspots"]]}
        for row in retained["rows"]]
    count = sum(len(h["selected_stage16_coordinates"]) for row in rows for h in row["hotspots"])
    return {
        "left_id": hotspot.PAIR[0], "right_id": hotspot.PAIR[1], "layer": 23, "position": 0,
        "row_count": len(rows), "projection_account_count": retained["projection_account_count"],
        "bridge_account_count": count, "closure_term_count": count*len(silu.TERM_NAMES),
        "rows": rows, "retained_down_input_bridge": retained,
        "common_component": retained["common_component"],
        "stop_nested_bridge_expansion": retained["stop_nested_bridge_expansion"],
        "prior_pair_and_common_gates_unchanged": True,
        "internal_reference": "original_input_L23_fp16",
        "binary64_internal_stages": "NOT_RETAINED_NO_RECONSTRUCTION",
    }


def validate_report(inputs, candidate):
    same(candidate, report(*inputs), "stage16 operand/term/closure/gate report binding changed")


def collect(audit):
    with bridge.read_only(audit), nonlinear_read_only(audit):
        evidence = bridge.measure()
    with read_only(audit):
        census = hotspot.census.census(evidence["report"])
        dominance = hotspot.dominance.audit_census(census, evidence["report"])
        observed = hotspot.audit_coordinates(evidence, census, dominance)
        tensors = previous.load_tensors(evidence)
        retained = previous.report(evidence, census, dominance, observed, tensors)
    return evidence, census, dominance, observed, tensors, retained


def run_tests(inputs, result):
    spec = importlib.util.spec_from_file_location("mlp_stage17_stage16_bridge_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = inputs, result
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "stage16 bridge tests failed, errored or skipped")
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
    origins = {
        **parent.source_context(), MODULE: parent.record(SOURCE),
        "mlp_stage17_stage16_bridge_tests": parent.record(TEST),
        **{module.MODULE: parent.record(module.SOURCE)
           for module in (previous, hotspot, down, silu, silu.gate_up)},
    }
    audit = {"forbidden_calls": 0}
    inputs = collect(audit)
    evidence = inputs[0]
    with read_only(audit):
        result = report(*inputs)
        tests = run_tests(inputs, result)
        for pin in (*origins.values(), *base.PINS.values(), *bridge.margin.rows.PINS.values(),
                    *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "stage16 bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_MIDDLE_PAIR_MLP_STAGE17_DOWN_INPUT_SILU_PRODUCT_BRIDGE",
        "command": COMMAND, "diagnostic_sources": origins,
        "pins": {**base.PINS, **bridge.margin.rows.PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "authenticated_files": evidence["files"], "hidden_pins": evidence["hidden_pins"],
        "assets": evidence["assets"],
        "L23_reference_authority": evidence["result"]["preflight"]["L23_original_reference"],
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "retained_thresholds": evidence["result"]["preflight"]["thresholds"],
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": {"compiled": compiled, **tests},
        "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        "input_mode": "authenticated retained in-memory down-input bridge and original L23 operands",
        "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
