"""Read-only retained scalar-scale residual margin accounting; stdout JSON only."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_direct_hidden_residual_margin_bridge_v1 as direct


margin, hidden, residual = direct.margin, direct.hidden, direct.residual
base, parent = direct.base, direct.parent
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_scalar_scale_residual_margin_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, EXPECTED_TESTS = 896, 20
BRANCHES = direct.BRANCHES
ENERGY_COMPONENTS = direct.COMPONENTS
DEFECT_COMPONENTS = ("actual_scalar_anchor_defect", "negative_reference_scalar_anchor_defect")
COMPONENTS = (*ENERGY_COMPONENTS, *DEFECT_COMPONENTS)
require, same, mass = base.require, base.same, direct.mass
FLAGS = {**direct.FLAGS, "scalar_split_causal_allocation": False}
BOUNDARY = (
    "Only the selected weighted global_scale term is split. The symmetric "
    "secant residual-energy allocation is an exact accounting convention, not "
    "a unique causal attribution. Each selected output coordinate weights the "
    "global 896-coordinate energy account, not just its own residual energy. "
    "Anchor defects remain explicit, including at zero mean-square change. "
    "Direct hidden, interaction, RMSNorm boundary, unselected-coordinate and "
    "head-boundary terms remain unchanged. " + direct.REFERENCE_SCOPE + " "
    + margin.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("scalar-scale bridge forbids operators, prior checks and writes")

    with margin.hotspots.read_only(audit), ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.") and name != MODULE:
                for attribute in (
                    "rmsnorm", "_torch_rmsnorm", "logits", "decode_array_q24",
                    "load_operands", "reference_suffix", "compute", "projection",
                    "local_reference", "attention_value", "rne", "toward_zero",
                    "check", "focused_tests", "measure",
                ):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        yield


def energy_account(actual, reference, terminal, actual_norm, reference_norm):
    require(len(terminal) == WIDTH and all(isinstance(v, Fraction) for v in terminal),
            "invalid independent branch terminal")
    a = actual["stage18"]
    for values, norm in ((a, actual_norm), (terminal, reference_norm)):
        mean = sum((v*v for v in values), Fraction()) / WIDTH
        require(Fraction(norm["mean_square"]) == mean, "mean-square operand splice")
        require(Fraction(norm["epsilon"]) == hidden.EPSILON, "scalar epsilon changed")
        require(Fraction(norm["radicand"]) == mean + hidden.EPSILON, "scalar radicand splice")
        anchor = Fraction(norm["inverse_norm_anchor"])
        require(anchor > 0, "nonpositive scalar anchor")
        require(Fraction(norm["inverse_square_identity_defect"])
                == anchor*anchor*(mean + hidden.EPSILON) - 1, "scalar defect splice")
    coordinates = []
    for i, (av, rv) in enumerate(zip(a, terminal, strict=True)):
        local = residual.account(actual, reference, i)
        parts = local["actual_q24_boundary_parts"]
        components = {
            key: Fraction(local["boundaries"][stage]["actual_minus_reference"])
            for key, stage in zip(ENERGY_COMPONENTS[:3], residual.STAGES[:3], strict=True)
        }
        components.update({
            "actual_residual_boundary": sum(
                (Fraction(parts[k]) for k in residual.ACTUAL_PARTS[:-1]), Fraction()),
            "negative_reference_residual_boundary": -Fraction(local["reference_boundary_remainder"]),
            "q24_to_fp16_conversion": Fraction(parts["output_fp16_boundary"]),
            "fp16_to_branch_terminal_remainder": reference["stage18"][i] - rv,
        })
        delta = av - rv
        require(sum(components.values(), Fraction()) == delta, "residual hidden closure failed")
        linear = {k: 2*rv*v/WIDTH for k, v in components.items()}
        quadratic = {k: delta*v/WIDTH for k, v in components.items()}
        energy = {k: linear[k]+quadratic[k] for k in ENERGY_COMPONENTS}
        require(sum(energy.values(), Fraction()) == (av*av-rv*rv)/WIDTH,
                "residual energy closure failed")
        coordinates.append({
            "coordinate": i,
            "hidden_components": {k: str(v) for k, v in components.items()},
            "linear_energy_components": {k: str(v) for k, v in linear.items()},
            "quadratic_energy_components": {k: str(v) for k, v in quadratic.items()},
            "energy_components": {k: str(v) for k, v in energy.items()},
            "mean_square_change": str(sum(energy.values(), Fraction())),
        })
    totals = {
        field: {key: mass(Fraction(row[field][key]) for row in coordinates)
                for key in ENERGY_COMPONENTS}
        for field in ("linear_energy_components", "quadratic_energy_components", "energy_components")
    }
    delta = Fraction(actual_norm["mean_square"]) - Fraction(reference_norm["mean_square"])
    require(sum((Fraction(v["signed"]) for v in totals["energy_components"].values()), Fraction())
            == delta, "global residual energy closure failed")
    return {"coordinates": coordinates, "totals": totals, "global_mean_square_delta": str(delta)}


def scalar_account(energy, actual_norm, reference_norm):
    qa, qr = (Fraction(n["radicand"]) for n in (actual_norm, reference_norm))
    sa, sr = (Fraction(n["inverse_norm_anchor"]) for n in (actual_norm, reference_norm))
    require(qa > 0 and qr > 0 and sa > 0 and sr > 0, "invalid scalar accounting anchors")
    da, dr = sa*sa*qa-1, sr*sr*qr-1
    require(Fraction(actual_norm["inverse_square_identity_defect"]) == da, "actual anchor defect changed")
    require(Fraction(reference_norm["inverse_square_identity_defect"]) == dr, "reference anchor defect changed")
    change = Fraction(energy["global_mean_square_delta"])
    require(qa-qr == change, "energy/radicand delta splice")
    denominator = qa*qr*(sa+sr)
    factor = -1/denominator
    values = {k: Fraction(energy["totals"]["energy_components"][k]["signed"])*factor
              for k in ENERGY_COMPONENTS}
    values.update({
        "actual_scalar_anchor_defect": qr*da/denominator,
        "negative_reference_scalar_anchor_defect": -qa*dr/denominator,
    })
    require(sum(values.values(), Fraction()) == sa-sr, "scalar energy/defect closure failed")
    return {
        "actual_norm": actual_norm, "reference_norm": reference_norm,
        "global_mean_square_delta": str(change),
        "energy_to_scale_factor": str(factor),
        "inverse_norm_anchor_delta": str(sa-sr),
        "scale_components": {k: str(v) for k, v in values.items()},
        "component_mass": mass(values.values()),
        "exact_scalar_identity": True,
    }


def split_coordinate(row, direct_row, scalar, energy_row, terminal, weight):
    i = row["coordinate"]
    require(type(i) is int and 0 <= i < WIDTH, "coordinate outside retained width")
    same(direct_row["coordinate"], i, "direct-hidden coordinate splice")
    same(energy_row["coordinate"], i, "energy coordinate splice")
    same(direct_row["hidden_components"], energy_row["hidden_components"],
         "direct-hidden/residual energy splice")
    h = row["hidden_bridge"]
    same(h["coordinate"], i, "hidden coordinate splice")
    require(Fraction(h["reference_hidden"]) == terminal, "independent terminal splice")
    require(Fraction(h["weight"]) == weight, "norm weight splice")
    difference = Fraction(row["left_weight"]) - Fraction(row["right_weight"])
    require(Fraction(row["row_difference"]) == difference, "tied-row difference splice")
    scale_delta = Fraction(scalar["inverse_norm_anchor_delta"])
    require(Fraction(h["inverse_norm_anchor_delta"]) == scale_delta, "scalar anchor delta splice")
    require(Fraction(h["global_scale"]) == weight*terminal*scale_delta, "hidden scalar term splice")
    factor = difference*weight*terminal
    weighted = {k: factor*Fraction(v) for k, v in scalar["scale_components"].items()}
    target = Fraction(row["weighted_terms"]["global_scale"])
    require(target == factor*scale_delta, "weighted scalar term splice")
    require(sum(weighted.values(), Fraction()) == target, "weighted scalar closure failed")
    return {
        "coordinate": i, "selection_reasons": row["selection_reasons"],
        "row_difference_times_weight_times_reference_hidden": str(factor),
        "weighted_components": {k: str(v) for k, v in weighted.items()},
        "global_scale_weighted_term": str(target),
        "weighted_component_mass": mass(weighted.values()),
        "selected_residual_energy": energy_row,
        "exact_global_scale_identity": True,
    }


def selected_account(rows, upstream):
    totals = {k: mass(Fraction(row["weighted_components"][k]) for row in rows)
              for k in COMPONENTS}
    scale = mass(Fraction(row["global_scale_weighted_term"]) for row in rows)
    same({k: scale[k] for k in ("signed", "absolute")},
         upstream["selected_term_totals"]["global_scale"], "selected global-scale totals changed")
    require(sum((Fraction(v["signed"]) for v in totals.values()), Fraction())
            == Fraction(scale["signed"]), "selected scalar component closure failed")
    absolute = sum((Fraction(v["absolute"]) for v in totals.values()), Fraction())
    return {
        "component_totals": totals, "selected_global_scale": scale,
        "component_absolute_sum": str(absolute),
        "within_coordinate_cancellation_mass": str(absolute-Fraction(scale["absolute"])),
        "across_coordinate_cancellation_mass": scale["cancellation_absolute_mass"],
        "total_component_cancellation_mass": str(absolute-abs(Fraction(scale["signed"]))),
        "exact_selected_identity": True,
    }


def report(evidence):
    upstream = evidence["margin_report"]
    retained = evidence["direct_report"]
    same(retained["retained_final_rmsnorm_logit_margin_bridge"], upstream,
         "direct-hidden/margin report splice")
    same([c["control"] for c in upstream["controls"]], list(parent.CONTROLS),
         "scalar control order changed")
    same([c["control"] for c in retained["controls"]], list(parent.CONTROLS),
         "direct-hidden control order changed")
    base.check_history(evidence["result"])
    reference = residual.operands(evidence["reference_archive"], actual=False)
    controls, count = [], 0
    for control, dc, hc in zip(upstream["controls"], retained["controls"],
                               evidence["hidden_report"]["controls"], strict=True):
        label = control["control"]
        same(hc["control"], label, "hidden/scalar control splice")
        energies, scalars = {}, {}
        for branch in BRANCHES:
            rn = hc["branches"][branch]["reference_norm"]
            energy = energy_account(evidence["actual"][label], reference,
                                    evidence["hidden"][branch], hc["actual_norm"], rn)
            same(energy["global_mean_square_delta"],
                 hc["branches"][branch]["global_mean_square_delta"], "hidden energy total splice")
            energies[branch] = energy
            scalars[branch] = scalar_account(energy, hc["actual_norm"], rn)
        pairs = []
        for pair, dp in zip(control["pairs"], dc["pairs"], strict=True):
            same(tuple(pair[k] for k in ("left_id", "right_id", "roles")),
                 tuple(dp[k] for k in ("left_id", "right_id", "roles")), "diagnostic pair splice")
            same(list(pair["branches"]), list(BRANCHES), "reference branch order changed")
            branches = {}
            for branch in BRANCHES:
                original = pair["branches"][branch]
                old_direct = dp["branches"][branch]
                same(original["hidden_reference"], "original_input_L23_"+branch,
                     "independent branch identity changed")
                same(old_direct["unchanged_margin_accounting"], original["accounting"],
                     "direct-hidden margin remainder splice")
                energy = energies[branch]
                selected = [
                    split_coordinate(row, old, scalars[branch],
                                     energy["coordinates"][row["coordinate"]],
                                     evidence["hidden"][branch][row["coordinate"]],
                                     evidence["weights"][row["coordinate"]])
                    for row, old in zip(original["selected_coordinates"],
                                        old_direct["selected_coordinates"], strict=True)
                ]
                indices = [r["coordinate"] for r in selected]
                require(indices == sorted(set(indices)) and 62 in indices
                        and 8 <= len(indices) <= 25, "selected coordinate census changed")
                unselected = {}
                for key in ENERGY_COMPONENTS:
                    chosen = mass(Fraction(row["selected_residual_energy"]["energy_components"][key])
                                  for row in selected)
                    total = energy["totals"]["energy_components"][key]
                    signed = Fraction(total["signed"])-Fraction(chosen["signed"])
                    absolute = Fraction(total["absolute"])-Fraction(chosen["absolute"])
                    require(absolute >= abs(signed), "unselected residual energy remainder failed")
                    unselected[key] = {"selected": chosen, "unselected_signed_remainder": str(signed),
                                       "unselected_absolute_remainder": str(absolute)}
                branches[branch] = {
                    "hidden_reference": original["hidden_reference"],
                    "residual_internal_reference": "original_input_L23_fp16",
                    "binary64_internal_stages": "NOT_RETAINED_NO_RECONSTRUCTION",
                    "selected_coordinates": selected,
                    "global_scale_accounting": selected_account(selected, original["accounting"]),
                    "residual_energy_selected_and_unselected": unselected,
                    "unchanged_margin_accounting": original["accounting"],
                }
                count += len(selected)
            pairs.append({"left_id": pair["left_id"], "right_id": pair["right_id"],
                          "roles": pair["roles"], "branches": branches})
        controls.append({
            "control": label, "pairs": pairs,
            "global_scalar_accounts": {
                b: {**scalars[b], "residual_energy_totals": energies[b]["totals"]}
                for b in BRANCHES},
        })
    same(count, upstream["selected_coordinate_accounts"], "scalar account census changed")
    return {
        **{k: upstream[k] for k in ("control_count", "diagnostic_pair_count", "pair_branch_count",
                                   "coordinate62_accounts", "lineage_separation")},
        "selected_coordinate_accounts": count, "weighted_component_count": count*len(COMPONENTS),
        "global_energy_coordinate_count": len(controls)*len(BRANCHES)*WIDTH,
        "reference_scope": direct.REFERENCE_SCOPE,
        "energy_identity": "E_k=mean((a+r)*c_k)=mean(2*r*c_k)+mean((a-r)*c_k); sum(c_k)=a-r",
        "scalar_identity": "s_a-s_r=(-delta_mean_square+q_r*d_a-q_a*d_r)/(q_a*q_r*(s_a+s_r)); d=s*s*q-1",
        "allocation_scope": "symmetric secant accounting, not causal percentages; no division by energy delta",
        "retained_final_rmsnorm_direct_hidden_residual_margin_bridge": retained,
        "controls": controls,
    }


def measure():
    for pin in margin.rows.PINS.values():
        base.read_bound(pin)
    evidence = hidden.authenticate()
    evidence["hidden_report"] = hidden.report(evidence)
    evidence["geometry"] = margin.cutoff.report(
        evidence["result"], evidence["arrays"], evidence["references"])
    ids = {i for control in evidence["geometry"]["controls"]
           for pair in margin.contributions.pairs_for(control) for i in pair}
    evidence["rows"] = margin.contributions.load_rows(evidence["assets"], ids)
    evidence["margin_report"] = margin.report(evidence)
    evidence["direct_report"] = direct.report(evidence)
    evidence["report"] = report(evidence)
    return evidence


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("scalar_scale_residual_margin_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "scalar-scale bridge focused tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "scalar_scale_residual_margin_test": parent.record(TEST)}
    for module in (direct, margin, hidden, residual):
        origins[module.MODULE] = parent.record(module.SOURCE)
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *margin.rows.PINS.values(), *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "scalar-scale bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_SCALAR_SCALE_RESIDUAL_MARGIN_BRIDGE",
        "command": COMMAND, "diagnostic_sources": origins,
        "pins": {**base.PINS, **margin.rows.PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "authenticated_files": evidence["files"], "hidden_pins": evidence["hidden_pins"],
        "assets": evidence["assets"],
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": tests, "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        "report": evidence["report"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
