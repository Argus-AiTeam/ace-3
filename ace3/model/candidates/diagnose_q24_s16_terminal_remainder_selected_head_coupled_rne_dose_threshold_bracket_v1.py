"""Exact retained-scalar dose intervals; stdout-only CPU non-admission diagnostic."""

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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_rne_crossing_attribution_v1 as crossing


scalar, base, parent, closed, bridge = (
    crossing.scalar, crossing.base, crossing.parent, crossing.closed, crossing.bridge)
require, same, ROOT = base.require, base.same, base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_selected_head_coupled_rne_dose_threshold_bracket_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS, POLARITIES, BRANCHES = crossing.IDS, crossing.PAIRS, crossing.POLARITIES, crossing.BRANCHES
LOW, HIGH = Fraction(1, 32), Fraction(3, 64)
DOSES = (LOW, HIGH, Fraction(1, 16), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(1))
DOSE_FIELDS = (
    "same_polarity_thirtysecond_dose_delta", "observed_delta",
    "same_polarity_sixteenth_dose_delta", "same_polarity_eighth_dose_delta",
    "same_polarity_quarter_dose_delta", "same_polarity_half_dose_delta",
    "same_polarity_full_dose_delta")
EXPECTED_TESTS = 14
MISSION = "34314ced1916"
PINS = (
    {"path": str(crossing.SOURCE), "sha256": "d46046636a31271415f76ffe082add5cbc721de4d7307f48fd742577220b84cc"},
    {"path": str(crossing.TEST), "sha256": "8f87c8de854668a03c6ba6a5b82ca6ccc7450dbdf13c8fc61939e128b2e3bd75"},
    {"path": str(closed.HANDOFFS / MISSION / "round-0001.json"),
     "sha256": "84668299441bed1fc766d42e980dd885bd5dabf75d281423d8bda69d881deed0"},
)
RECEIPT = {
    "path": "/tmp/1789476663516-copilot-tool-output-4176695-de9478b8-8d9a-41a3-bf5b-b717cd039586.txt",
    "sha256": "d2dad7f5eb79608a54dcea46dc5c00cf9a1e3ce90e5cca8b9da8aa479694f0a7",
}
JOINT = "minimal joint-dose bracket localization"
ASYMMETRIC = "asymmetric/single-row threshold successor"
PREREGISTRATION = {
    "operand": "Only authenticated retained baseline/exact-target/working accumulator "
               "scalars, FP16 words, frozen row dose slopes, closed dose receipts and "
               "fixed original-input references. No hidden operands or row dots are computed.",
    "prediction": "The four mapped_all reverse coupled contrasts require BOTH row "
                  "first-midpoint thresholds strictly after 1/32 and reached at/before "
                  "3/64 with nearest-even endpoint ownership; all other contrasts remain "
                  "single-row, without contradictory closed-dose threshold outcomes.",
    "supported": JOINT,
    "only_one_new_row_or_closed_threshold_ordering_contradiction": ASYMMETRIC,
    "authentication_arithmetic_or_contrast_closure_failure": "UNKNOWN/integrity",
    "comparison": "Exact-target precedence. At 1/32 and 3/64 compare individual-word "
                  "margin deltas exactly; at larger closed doses compare only first-"
                  "crossing/nonzero and direction, not uncomputed rounded magnitudes. "
                  "A valid 1/32 model/closed-dose discrepancy is scientific rejection, "
                  "not corrupted evidence. Working scalars are retained only at 3/64; "
                  "never interpolate hidden-RNE remainders or infer working-dose slopes.",
    "boundary": crossing.PREREGISTRATION["boundary"] + " Analytical dose-to-midpoint "
                "intervals only. Ordered/reference duplicates are not independent samples. "
                "Q24 residual state is wider than FP16; INT4 weights, native S16 RTZ, "
                "FP16 operator boundaries and KV are unchanged. No precision/scale expansion.",
}
FLAGS = {**crossing.FLAGS, "closed_crossing_diagnostic_replay": False,
         "working_remainder_interpolation": False, "new_dose_operand_execution": False}
AUDIT_KEYS = (
    "forbidden_calls", "prefix_replays", "admission_replays", "reference_producer_replays",
    "final_rmsnorm_invocations", "selected_row_head_invocations", "full_vocabulary_replays",
    "closed_stage18_replays", "closed_headonly_replays", "closed_scalar_diagnostic_replays",
    "closed_crossing_diagnostic_replays", "exact_selected_row_accumulations",
    "hidden_rne_scalars", "writes", "artifact_overwrites")


def selection_gate():
    crossing.selection_gate()
    scalar.selection_gate()
    same((IDS, PAIRS, POLARITIES, BRANCHES),
         ((319, 34319), ((319, 34319), (34319, 319)),
          ("forward", "reverse"), ("fp16", "binary64")), "threshold selection changed")
    require(DOSES == tuple(Fraction(n, d) for n, d in
                          ((1, 32), (3, 64), (1, 16), (1, 8), (1, 4), (1, 2), (1, 1))),
            "closed dose selection changed")
    require(type(LOW) is Fraction and LOW == Fraction(1, 32)
            and type(HIGH) is Fraction and HIGH == Fraction(3, 64)
            and all(type(d) is Fraction for d in DOSES), "threshold dose type/value changed")


@contextmanager
def read_only(audit):
    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("dose thresholds forbid closed crossing/scalar/operator replay and writes")

    with crossing.read_only(audit), ExitStack() as stack:
        for name in ("check", "run_tests", "attribute", "classify"):
            stack.enter_context(patch.object(crossing, name, refuse))
        yield


def authenticate_crossing():
    for pin in (*PINS, RECEIPT):
        base.read_bound(pin)
    review = json.loads(base.read_bound(PINS[-1]))
    same([review[k] for k in ("kind", "mission_id", "producer_role", "round")],
         ["round_reviewed_handoff", MISSION, "reviewer", 1], "crossing review identity changed")
    same(review["review"]["status"], "done", "independent crossing review incomplete")
    latest_pin = parent.record(Path(PINS[-1]["path"]).with_name("latest.json"))
    latest = json.loads(base.read_bound(latest_pin))
    same((latest["kind"], latest["handoff"]["path"]),
         ("handoff_ref", PINS[-1]["path"]), "crossing review lineage changed")
    text = base.read_bound(RECEIPT).decode()
    marker = '{"validation":'
    same(text.count(marker), 1, "crossing receipt missing or duplicated")
    receipt, _ = json.JSONDecoder().raw_decode(text[text.index(marker):])
    require(set(receipt) == {"validation", "native_result"}, "crossing receipt envelope changed")
    return receipt, [*PINS, RECEIPT, latest_pin]


def bind_crossing(receipt, evidence, summary, rows, summaries, head_summary):
    result, validation = receipt["native_result"], receipt["validation"]
    same([validation[k] for k in ("native_command", "native_exit", "stdout_json_documents")],
         [crossing.COMMAND, 0, 1], "crossing execution receipt changed")
    same((result["diagnostic_id"], result["version"], result["status"], result["command"]),
         (crossing.NAME, 1, "supported", crossing.COMMAND), "crossing result identity changed")
    same(result["flags"], crossing.FLAGS, "crossing boundaries changed")
    same(result["preregistration"], crossing.PREREGISTRATION, "crossing contract changed")
    same(result["tests"], validation["tests"], "crossing test receipt mismatch")
    same([result["tests"][k] for k in ("executed", "errors", "failures", "skipped")],
         [12, 0, 0, 0], "crossing test outcome changed")
    same(result["tests"]["compiled"], [parent.record(crossing.SOURCE), parent.record(crossing.TEST)],
         "crossing source/test identity drift")
    old_audit = dict.fromkeys((k for k in AUDIT_KEYS if k != "closed_crossing_diagnostic_replays"), 0)
    same(result["dispatch_and_write_audit"], old_audit, "crossing dispatch/write drift")
    same(validation["dispatch_and_write_audit"], old_audit, "crossing audit receipt drift")
    same(result["protected_input_identity"], closed.digest(evidence),
         "crossing source/token/Q24/state/KV/lineage/reference/head-row drift")
    same(validation["protected_input_identity"], result["protected_input_identity"],
         "crossing validation identity drift")
    for key, value in (
        ("retained_scalar_receipt", summary), ("retained_dose_summaries", list(summaries)),
        ("retained_headonly_summary", head_summary), ("authenticated_files", evidence["files"]),
        ("hidden_pins", evidence["hidden_pins"]), ("assets", evidence["assets"]),
        ("selected_head_row_identity", closed.digest(evidence["rows"])),
        ("final_reference_authority", evidence["result"]["preflight"]["final_reference"]),
        ("retained_controls_and_failure_gates", evidence["result"]["controls"]),
        ("retained_thresholds", evidence["result"]["preflight"]["thresholds"]),
        ("retained_common_component", "UNKNOWN"), ("counterfactual_common_component", "UNKNOWN"),
    ):
        same(result[key], value, f"crossing retained {key} changed")
    report = result["report"]
    for key, value in (
        ("classification", "supported"), ("successor", crossing.COUPLED),
        ("selected_branch", "exact_target"), ("contrast_count", 72),
        ("single_row_contrasts", 68), ("two_row_contrasts", 4),
        ("opposite_compensation_contrasts", 0), ("individual_midpoint_closures", 108),
        ("individual_slack_closures", 72), ("target_contrast_closures", 72),
        ("working_contrast_closures", 72), ("integrity_error", None),
    ):
        same((report[key], validation[key]), (value, value), f"crossing {key} changed")
    same([{k: r[k] for k in crossing.FIELDS} for r in report["rows"]], rows,
         "crossing retained individual scalar binding changed")
    for pin in (*result["reviewed_parent_pins"], *result["diagnostic_sources"].values(),
                *result["retained_dose_receipts"]):
        base.read_bound(pin)
    same(validation["authenticated_pin_count"], len(result["reviewed_parent_pins"]),
         "crossing authenticated pin census changed")
    return report


def exact(text):
    require(type(text) is str, "non-string exact dose rational")
    value = Fraction(text)
    require(str(value) == text, "noncanonical exact dose rational")
    return value


def reached(threshold, dose):
    require(type(dose) is Fraction and dose >= 0, "invalid analytical dose")
    boundary = exact(threshold["dose_to_midpoint"])
    return dose > boundary or (dose == boundary and threshold["crossing_at_equality"])


def threshold(baseline, word, slope):
    require(type(baseline) is Fraction and type(slope) is Fraction and slope != 0,
            "invalid exact accumulator/slope")
    cell = crossing.cell(baseline, word)
    direction = 1 if slope > 0 else -1
    midpoint = exact(cell["upper_midpoint" if direction > 0 else "lower_midpoint"])
    inclusive = not cell["even_significand"]
    neighbor = word + (direction if not word & 0x8000 else -direction)
    boundary_kind = "adjacent_value_midpoint"
    if word in (0, 0x8000):
        toward_other_sign = (word == 0 and direction < 0) or (word == 0x8000 and direction > 0)
        if toward_other_sign:
            # Signed zero changes its word before reaching a nonzero-value midpoint.
            midpoint, neighbor, inclusive = Fraction(), word ^ 0x8000, word == 0x8000
            boundary_kind = "signed_zero_transition"
        else:
            neighbor = 1 if direction > 0 else 0x8001
    slack = direction * (midpoint - baseline)
    dose = slack / abs(slope)
    require(dose >= 0 and baseline + slope * dose == midpoint, "dose threshold closure failed")
    crossing.decode(neighbor)
    return {
        "direction": direction, "boundary_kind": boundary_kind,
        "midpoint": str(midpoint), "initial_accumulator_slack": str(slack),
        "dose_to_midpoint": str(dose), "crossing_at_equality": inclusive,
        "tie_winner_word": neighbor if inclusive else word, "neighbor_word": neighbor,
        "baseline_dose_interval": {"lower": "0", "lower_inclusive": True,
                                   "upper": str(dose), "upper_inclusive": not inclusive},
        "crossed_dose_interval": {"lower": str(dose), "lower_inclusive": inclusive,
                                  "upper": None, "upper_inclusive": False},
        "threshold_equation_residual": "0",
        "bracket": {
            str(d): {"accumulator_slack": str(direction * (midpoint - baseline - slope * d)),
                     "dose_slack": str(dose - d),
                     "crossed": d > dose or (d == dose and inclusive)}
            for d in (LOW, HIGH)},
    }


def solve_rows(retained_rows, evidence):
    same([(r["control"], r["polarity"], r["row_id"]) for r in retained_rows],
         [(c, p, i) for c in parent.CONTROLS for p in POLARITIES for i in IDS],
         "threshold row census changed")
    rows, slopes, baselines = [], {}, {}
    for retained in retained_rows:
        c, p, i = (retained[k] for k in crossing.FIELDS[:3])
        b, target, working = (crossing.rational(retained[n + "_exact"])
                              for n in ("baseline", "target", "working"))
        word = retained["baseline_word"]
        for name, value in (("baseline", b), ("target", target), ("working", working)):
            same(crossing.cell(value, retained[name + "_word"]), retained["cells"][name],
                 "retained scalar cell changed")
        same(word, int(evidence["arrays"][c]["logits"][i]), "baseline logit word changed")
        baseline = (b, word)
        require(baseline == baselines.setdefault((c, i), baseline), "polarity baseline changed")
        frozen_slope = crossing.rational(retained["frozen_row_dose_slope"])
        require(frozen_slope == slopes.setdefault(i, frozen_slope), "frozen row dose slope changed")
        slope = -frozen_slope if p == "forward" else frozen_slope
        same(retained["signed_dose"], str(-HIGH if p == "forward" else HIGH), "signed dose changed")
        require(b + HIGH * slope == target, "exact-target affine dose closure failed")
        same(str(working - target), retained["hidden_rne_dot_remainder"],
             "working scalar remainder changed")
        account = threshold(b, word, slope)
        target_word = account["neighbor_word"] if reached(account, HIGH) else word
        same(target_word, retained["target_word"], "first-midpoint target word closure failed")
        lower_word = account["neighbor_word"] if reached(account, LOW) else word
        crossing.cell(b + LOW * slope, lower_word)
        for name, endpoint in (("target", target), ("working", working)):
            old = retained[name]
            same(old["signed_movement"], str(endpoint - b), "retained scalar movement changed")
            same(old["word_changed"], retained[name + "_word"] != word, "retained row crossing changed")
            same(old["individual_rounded_delta"],
                 str(crossing.decode(retained[name + "_word"]) - crossing.decode(word)),
                 "retained individual word delta changed")
        rows.append({**{k: retained[k] for k in crossing.FIELDS},
                     "frozen_row_dose_slope": str(frozen_slope), "signed_row_dose_slope": str(slope),
                     "hidden_rne_dot_remainder": str(working - target),
                     "working_dose_threshold": None, "threshold": account,
                     "lower_word": lower_word, "upper_word": target_word,
                     "affine_target_closure_residual": "0"})
    return rows


def decide(coupled, other_single, lower_agreements, closed_agreements):
    require(type(coupled) is list and len(coupled) == 4, "coupled contrast census changed")
    joint_rows = all(all(row["both_new_in_bracket"]) for row in coupled)
    joint = joint_rows and other_single == 68 and lower_agreements == 72 and closed_agreements == 504
    return {
        "classification": "supported" if joint else "rejected",
        "joint_threshold_prediction": "supported" if joint else "rejected",
        "successor": JOINT if joint else ASYMMETRIC,
        "both_thresholds_new_in_all_four_coupled_contrasts": joint_rows,
        "integrity_error": None,
    }


def solve(retained, evidence, summaries):
    selection_gate()
    rows = solve_rows(retained["rows"], evidence)
    lookup = {(r["control"], r["polarity"], r["row_id"]): r for r in rows}
    contrasts = retained["contrasts"]
    same([(r["control"], r["polarity"], r["left_id"], r["right_id"], r["branch"]) for r in contrasts],
         [(c, p, l, r, b) for c in parent.CONTROLS for p in POLARITIES for l, r in PAIRS for b in BRANCHES],
         "threshold contrast census changed")
    dose_rows = [dict(zip(summaries[-1]["contrast_fields"], v, strict=True))
                 for v in summaries[-1]["contrast_values"]]
    same([(r["control"], r["polarity"], r["left_id"], r["right_id"]) for r in dose_rows],
         [(c, p, l, r) for c in parent.CONTROLS for p in POLARITIES for l, r in PAIRS],
         "closed dose contrast census changed")
    doses = {(r["control"], r["polarity"], r["left_id"], r["right_id"]): r for r in dose_rows}
    output, coupled, consistency = [], [], []
    lower_agreements = other_single = 0
    for old in contrasts:
        c, p, l, r, branch = (old[k] for k in ("control", "polarity", "left_id", "right_id", "branch"))
        left, right = lookup[c, p, l], lookup[c, p, r]
        decode = crossing.decode
        baseline = decode(left["baseline_word"]) - decode(right["baseline_word"])
        refs = evidence["references"]["logits_" + branch]
        reference = (decode(int(refs[l])) - decode(int(refs[r])) if branch == "fp16"
                     else Fraction(float(refs[l])) - Fraction(float(refs[r])))
        same((old["baseline_rounded_margin"], old["fixed_reference_margin"],
              old["retained_margin_change"]),
             (str(baseline), str(reference), str(baseline - reference)),
             "fixed original-reference closure failed")
        for name in ("target", "working"):
            ld = decode(left[name + "_word"]) - decode(left["baseline_word"])
            rd = decode(right[name + "_word"]) - decode(right["baseline_word"])
            margin = baseline + ld - rd
            account = old[name]
            same((account["rounded_margin"], account["margin_change"], account["delta"],
                  account["left_contribution"], account["right_contribution"],
                  account["crossing_rows"], account["opposite_row_compensation"]),
                 (str(margin), str(margin - reference), str(ld - rd), str(ld), str(-rd),
                  [i for i, delta in ((l, ld), (r, rd)) if delta], ld * -rd < 0),
                 "retained individual-row/contrast closure failed")
            same(account["delta"], doses[c, p, l, r]["observed_delta"],
                 "closed 3/64 contrast closure failed")
        low_delta = decode(left["lower_word"]) - decode(right["lower_word"]) - baseline
        high_delta = decode(left["upper_word"]) - decode(right["upper_word"]) - baseline
        same(str(high_delta), old["closed_bracket_delta"], "upper-dose exact contrast closure failed")
        lower_closed = exact(doses[c, p, l, r][DOSE_FIELDS[0]])
        lower_match = low_delta == lower_closed
        lower_agreements += lower_match
        newly_crossed = [LOW < exact(x["threshold"]["dose_to_midpoint"]) <= HIGH
                         and reached(x["threshold"], HIGH) for x in (left, right)]
        first = min(exact(x["threshold"]["dose_to_midpoint"]) for x in (left, right))
        second = max(exact(x["threshold"]["dose_to_midpoint"]) for x in (left, right))
        item = {k: old[k] for k in ("control", "polarity", "left_id", "right_id", "branch")}
        item.update(
            fixed_reference_margin=str(reference), baseline_rounded_margin=str(baseline),
            lower_exact_delta=str(low_delta), lower_closed_delta=str(lower_closed),
            lower_consistent=lower_match, upper_exact_delta=str(high_delta),
            upper_closed_delta=old["closed_bracket_delta"], upper_consistent=True,
            lower_margin_change=str(baseline + low_delta - reference),
            upper_margin_change=str(baseline + high_delta - reference),
            target_crossing_rows=old["target"]["crossing_rows"],
            both_new_in_bracket=newly_crossed,
            row_threshold_bracket={"first_row_infimum": str(first), "both_rows_infimum": str(second)},
            both_rows_crossed_dose_interval={
                "lower": str(second), "lower_inclusive": all(reached(x["threshold"], second)
                                                            for x in (left, right)),
                "upper": None, "upper_inclusive": False},
            single_row_dose_interval=None if first == second else {
                "lower": str(first), "lower_inclusive": any(reached(x["threshold"], first)
                                                          for x in (left, right)),
                "upper": str(second), "upper_inclusive": not all(reached(x["threshold"], second)
                                                               for x in (left, right))},
            target_contrast_closure_residual="0", working_contrast_closure_residual="0")
        if len(old["target"]["crossing_rows"]) == 2:
            require(c == "mapped_all" and p == "reverse", "reviewed coupled case changed")
            coupled.append(item)
        else:
            require(len(old["target"]["crossing_rows"]) == 1, "reviewed single-row count changed")
            other_single += 1
        output.append(item)
        for dose, field in zip(DOSES, DOSE_FIELDS, strict=True):
            active = [x for x in (left, right) if reached(x["threshold"], dose)]
            observed = exact(doses[c, p, l, r][field])
            # Oppositely directed row slopes make the ordered margin contributions agree.
            signed_slopes = (exact(left["signed_row_dose_slope"]), -exact(right["signed_row_dose_slope"]))
            require(signed_slopes[0] * signed_slopes[1] > 0, "nonzero inference has row compensation")
            direction = (signed_slopes[0] > 0) - (signed_slopes[0] < 0)
            nonzero_match = bool(active) == bool(observed)
            direction_match = not observed or (observed > 0) - (observed < 0) == direction
            consistency.append({
                **{k: item[k] for k in ("control", "polarity", "left_id", "right_id", "branch")},
                "dose": str(dose), "closed_delta": str(observed),
                "first_threshold_crossed_rows": [x["row_id"] for x in active],
                "nonzero_consistent": nonzero_match, "direction_consistent": direction_match,
                "consistent": nonzero_match and direction_match,
                "larger_dose_rounded_magnitude_computed": False})
    counts = {str(d): {
        "contrasts": 72,
        "agreements": sum(x["consistent"] for x in consistency if x["dose"] == str(d))}
        for d in DOSES}
    agreements = sum(x["consistent"] for x in consistency)
    return {
        **decide(coupled, other_single, lower_agreements, agreements),
        "selected_branch": "exact_target", "rows": rows, "contrasts": output,
        "coupled_contrasts": coupled, "other_single_row_contrasts": other_single,
        "row_threshold_closures": len(rows), "retained_scalar_cell_closures": 3 * len(rows),
        "target_contrast_closures": len(output), "working_contrast_closures": len(output),
        "lower_exact_margin_agreements": lower_agreements, "upper_exact_margin_agreements": len(output),
        "closed_dose_consistency_count": len(consistency), "closed_dose_consistency_agreements": agreements,
        "closed_dose_counts": counts, "closed_dose_consistency": consistency,
    }


def run_tests(evidence, report, retained, receipt, summary, rows, summaries, head_summary):
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    spec = importlib.util.spec_from_file_location("coupled_threshold_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE, module.REPORT, module.RETAINED = evidence, report, retained
    module.RECEIPT, module.SUMMARY, module.ROWS = receipt, summary, rows
    module.SUMMARIES, module.HEAD_SUMMARY = summaries, head_summary
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "focused tests failed, errored or skipped")
    return {"executed": result.testsRun, "errors": len(result.errors), "failures": len(result.failures),
            "skipped": len(result.skipped), "compiled": [parent.record(SOURCE), parent.record(TEST)]}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter gate failed")
    selection_gate()
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "coupled_threshold_tests": parent.record(TEST)}
    audit = dict.fromkeys(AUDIT_KEYS, 0)
    with read_only(audit):
        receipt, pins = authenticate_crossing()
        summary, scalar_pins = crossing.authenticate_receipt()
        chain_pins, head_summary = scalar.authenticate_parent()
        summaries = scalar.retained.retained_doses()
        for pin in bridge.margin.rows.PINS.values():
            base.read_bound(pin)
        evidence = bridge.hidden.authenticate()
        evidence["rows"] = bridge.margin.contributions.load_rows(evidence["assets"], IDS)
        base.check_history(evidence["result"])
        comparisons = scalar.retained.bind_doses(evidence, summaries)
        scalar.bind_parent(evidence, head_summary, comparisons)
        rows = crossing.bind_receipt(summary, evidence, chain_pins)
        retained = bind_crossing(receipt, evidence, summary, rows, summaries, head_summary)
        identity = closed.digest(evidence)
        frozen = closed.digest((receipt, summary, summaries, head_summary, rows, PREREGISTRATION, FLAGS))
        report = solve(retained, evidence, summaries)
        report_identity = closed.digest(report)
        tests = run_tests(evidence, report, retained, receipt, summary, rows, summaries, head_summary)
        closed.protect(evidence, identity)
        same(closed.digest(report), report_identity, "threshold result mutation")
        same(closed.digest((receipt, summary, summaries, head_summary, rows, PREREGISTRATION, FLAGS)),
             frozen, "retained evidence/contract mutation")
        all_pins = [*pins, *scalar_pins, *chain_pins, *origins.values(), *base.PINS.values(),
                    *bridge.margin.rows.PINS.values(), *evidence["hidden_pins"],
                    *scalar.retained.RECEIPTS, scalar.PARENT_RECEIPT]
        for pin in all_pins:
            base.read_bound(pin)
    same(audit, dict.fromkeys(AUDIT_KEYS, 0), "forbidden dispatch/write census changed")
    return {
        "diagnostic_id": NAME, "version": 1, "status": report["classification"],
        "classification": report["classification"], "successor": report["successor"],
        "command": COMMAND, "report": report, "preregistration": PREREGISTRATION, "tests": tests,
        "authenticated_pins": all_pins, "reviewed_crossing_receipt": RECEIPT,
        "retained_scalar_receipt": summary, "protected_input_identity": identity,
        "selected_head_row_identity": closed.digest(evidence["rows"]),
        "authenticated_files": evidence["files"], "assets": evidence["assets"],
        "retained_dose_summaries": summaries,
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "retained_thresholds": evidence["result"]["preflight"]["thresholds"],
        "retained_common_component": "UNKNOWN", "counterfactual_common_component": "UNKNOWN",
        "mutation_guards": {"protected_inputs_unchanged": True, "retained_receipts_unchanged": True,
                            "report_unchanged": True, "source_pins_reauthenticated": True},
        "flags": FLAGS, "dispatch_and_write_audit": audit, "normal_host_review": "REQUIRED",
        "claim_boundary": PREREGISTRATION["boundary"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", required=True, action="store_true")
    parser.parse_args(argv)
    try:
        result = check()
    except (ValueError, RuntimeError, OSError, ArithmeticError, LookupError, TypeError, ImportError) as error:
        print(json.dumps({"diagnostic_id": NAME, "status": "UNKNOWN", "classification": "UNKNOWN/integrity",
                          "successor": None, "integrity_error": f"{type(error).__name__}: {error}",
                          "flags": FLAGS, "normal_host_review": "REQUIRED"}, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
