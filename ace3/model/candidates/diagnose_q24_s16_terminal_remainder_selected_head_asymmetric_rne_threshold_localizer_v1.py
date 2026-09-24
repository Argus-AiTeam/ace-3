"""Retained-scalar rowwise bracket increments; stdout-only CPU non-admission."""

import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_coupled_rne_dose_threshold_bracket_v1 as bracket


crossing, scalar, base, parent, closed, bridge = (
    bracket.crossing, bracket.scalar, bracket.base, bracket.parent, bracket.closed, bracket.bridge)
require, same, ROOT = base.require, base.same, base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_selected_head_asymmetric_rne_threshold_localizer_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS, POLARITIES, BRANCHES = bracket.IDS, bracket.PAIRS, bracket.POLARITIES, bracket.BRANCHES
LOW, HIGH, EXPECTED_TESTS = Fraction(1, 32), Fraction(3, 64), 14
MISSION = "599d2716fb3c"
PINS = (
    {"path": str(bracket.SOURCE), "sha256": "d2094b56695bdedbe3f04a174f1cf26350d94ab02122a5a3f810935d6032aee8"},
    {"path": str(bracket.TEST), "sha256": "5463bc55323851ae172a7aabdc4e33a6edb78465c92ec828f3b81251da68c975"},
    {"path": str(closed.HANDOFFS / MISSION / "round-0001.json"),
     "sha256": "4d7530bafd37aa1bc0fefe3467bf2ea5d11d09b4bd8b2496b1f1c8ad21a9a179"},
)
RECEIPT = {
    "path": "/tmp/1789477661941-copilot-tool-output-6665-24bfb927-3dae-4892-8266-b517194eabfc.txt",
    "sha256": "0cffed891c2ca99cc81d0bfc570586c983f929cdaacf814dd5558b493bbbb09a",
}
LATE = "row319-late-threshold localization"
EARLY = "row34319/early-active-dominance successor"
MIXED = "both-row/asymmetric-mixed successor"
PREREGISTRATION = {
    "operand": "Reviewed retained crossing/bracket scalar receipts only; no row dots, "
               "hidden operands, new doses or working-remainder interpolation.",
    "prediction": "In all four mapped_all reverse ordered/reference accounts, row 319 "
                  "is inactive at 1/32 and active at 3/64, row 34319 is already active "
                  "before 1/32 and unchanged across the bracket, and row 319 alone "
                  "equals the entire nonzero closed 3/64-minus-1/32 margin increment.",
    "row319_necessary_and_sufficient_for_every_bracket_increment": LATE,
    "early_active_row34319_alone_explains_every_bracket_increment": EARLY,
    "another_valid_row_state_or_both_row_increment_explanation": MIXED,
    "authentication_arithmetic_or_closure_failure": "UNKNOWN/integrity; no scientific successor",
    "comparison": "Exact-target precedence. Separate baseline-to-1/32 activation, "
                  "1/32-to-3/64 increment and baseline-to-3/64 total. Holding one "
                  "retained rounded row at its lower endpoint is scalar accounting, "
                  "not a new hidden/operator intervention. Dominance means exclusive "
                  "bracket-increment accounting, not runtime or full-model causality.",
    "boundary": bracket.PREREGISTRATION["boundary"] + " No closed bracket diagnostic "
                "replay. The four ordered/reference accounts are not independent samples.",
}
FLAGS = {**bracket.FLAGS, "closed_bracket_diagnostic_replay": False,
         "row_operand_intervention": False}
AUDIT_KEYS = (*bracket.AUDIT_KEYS, "closed_bracket_diagnostic_replays")


def selection_gate():
    bracket.selection_gate()
    same((IDS, PAIRS, POLARITIES, BRANCHES),
         ((319, 34319), ((319, 34319), (34319, 319)),
          ("forward", "reverse"), ("fp16", "binary64")), "localizer selection changed")
    require(type(LOW) is Fraction and LOW == Fraction(1, 32)
            and type(HIGH) is Fraction and HIGH == Fraction(3, 64), "localizer bracket changed")


@contextmanager
def read_only(audit):
    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("localizer forbids closed bracket computation and artifact writes")

    with bracket.read_only(audit), ExitStack() as stack:
        for name in ("check", "run_tests", "solve", "solve_rows", "decide"):
            stack.enter_context(patch.object(bracket, name, refuse))
        yield


def bind_bracket(receipt, evidence, summary, summaries):
    result, validation = receipt["native_result"], receipt["validation"]
    same((result["diagnostic_id"], result["version"], result["command"],
          result["status"], result["classification"], result["successor"]),
         (bracket.NAME, 1, bracket.COMMAND, "rejected", "rejected", bracket.ASYMMETRIC),
         "reviewed bracket identity/outcome changed")
    same([validation[k] for k in ("native_command", "native_exit", "stdout_json_documents")],
         [bracket.COMMAND, 0, 1], "reviewed bracket execution changed")
    for key, expected in (
        ("flags", bracket.FLAGS), ("preregistration", bracket.PREREGISTRATION),
        ("retained_scalar_receipt", summary), ("retained_dose_summaries", summaries),
        ("protected_input_identity", closed.digest(evidence)),
        ("selected_head_row_identity", closed.digest(evidence["rows"])),
        ("authenticated_files", evidence["files"]), ("assets", evidence["assets"]),
        ("final_reference_authority", evidence["result"]["preflight"]["final_reference"]),
        ("retained_controls_and_failure_gates", evidence["result"]["controls"]),
        ("retained_thresholds", evidence["result"]["preflight"]["thresholds"]),
        ("retained_common_component", "UNKNOWN"), ("counterfactual_common_component", "UNKNOWN"),
        ("reviewed_crossing_receipt", bracket.RECEIPT),
        ("dispatch_and_write_audit", dict.fromkeys(bracket.AUDIT_KEYS, 0)),
    ):
        same(result[key], expected, f"reviewed bracket {key} changed")
    same(result["tests"], validation["tests"], "bracket test receipt changed")
    same([result["tests"][k] for k in ("executed", "errors", "failures", "skipped")],
         [14, 0, 0, 0], "reviewed bracket tests failed")
    same(result["tests"]["compiled"], [parent.record(bracket.SOURCE), parent.record(bracket.TEST)],
         "reviewed bracket source/test drift")
    same(validation["protected_input_identity"], result["protected_input_identity"],
         "bracket validation input identity drift")
    same(validation["dispatch_and_write_audit"], result["dispatch_and_write_audit"],
         "bracket validation audit drift")
    same(validation["authenticated_pin_count"], len(result["authenticated_pins"]),
         "bracket pin census changed")
    for key, expected in (
        ("classification", "rejected"), ("successor", bracket.ASYMMETRIC),
        ("selected_branch", "exact_target"), ("integrity_error", None),
        ("row_threshold_closures", 36), ("retained_scalar_cell_closures", 108),
        ("target_contrast_closures", 72), ("working_contrast_closures", 72),
        ("lower_exact_margin_agreements", 72), ("upper_exact_margin_agreements", 72),
        ("other_single_row_contrasts", 68), ("closed_dose_consistency_count", 504),
        ("closed_dose_consistency_agreements", 504),
    ):
        same((result["report"][key], validation[key]), (expected, expected),
             f"reviewed bracket {key} drift")
    return result["report"]


def authenticate():
    selection_gate()
    for pin in (*PINS, RECEIPT):
        base.read_bound(pin)
    review = json.loads(base.read_bound(PINS[-1]))
    same([review[k] for k in ("kind", "mission_id", "producer_role", "round")],
         ["round_reviewed_handoff", MISSION, "reviewer", 1], "bracket review identity changed")
    same(review["review"]["status"], "done", "independent bracket review incomplete")
    latest_pin = parent.record(Path(PINS[-1]["path"]).with_name("latest.json"))
    latest = json.loads(base.read_bound(latest_pin))
    same((latest["kind"], latest["handoff"]["path"]),
         ("handoff_ref", PINS[-1]["path"]), "bracket review lineage changed")
    text, marker = base.read_bound(RECEIPT).decode(), '{"validation":'
    same(text.count(marker), 1, "bracket receipt missing or duplicated")
    receipt, _ = json.JSONDecoder().raw_decode(text[text.index(marker):])
    require(set(receipt) == {"validation", "native_result"}, "bracket receipt envelope changed")
    crossing_receipt, crossing_pins = bracket.authenticate_crossing()
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
    retained_crossing = bracket.bind_crossing(
        crossing_receipt, evidence, summary, rows, summaries, head_summary)
    retained_bracket = bind_bracket(receipt, evidence, summary, summaries)
    pins = [*PINS, RECEIPT, latest_pin, *crossing_pins, *scalar_pins, *chain_pins,
            *receipt["native_result"]["authenticated_pins"]]
    for pin in pins:
        base.read_bound(pin)
    return {"evidence": evidence, "receipt": receipt, "summary": summary,
            "summaries": summaries, "crossing": retained_crossing,
            "bracket": retained_bracket, "authenticated_pins": pins}


def row_account(row, retained, evidence):
    same({k: row[k] for k in crossing.FIELDS},
         {k: retained[k] for k in crossing.FIELDS}, "retained scalar row drift")
    c, p, i = (row[k] for k in crossing.FIELDS[:3])
    b, target, working = (crossing.rational(row[n + "_exact"])
                          for n in ("baseline", "target", "working"))
    slope = crossing.rational(row["frozen_row_dose_slope"])
    same(row["frozen_row_dose_slope"], retained["frozen_row_dose_slope"], "frozen slope drift")
    signed = -slope if p == "forward" else slope
    same(row["signed_row_dose_slope"], str(signed), "signed slope drift")
    require(target == b + HIGH * signed, "affine target closure failed")
    same(row["hidden_rne_dot_remainder"], str(working - target), "working remainder drift")
    same(row["baseline_word"], int(evidence["arrays"][c]["logits"][i]), "retained logit drift")
    for name, value in (("baseline", b), ("target", target), ("working", working)):
        same(crossing.cell(value, row[name + "_word"]), retained["cells"][name],
             "retained individual RNE cell drift")
    threshold = bracket.threshold(b, row["baseline_word"], signed)
    same(row["threshold"], threshold, "retained midpoint/threshold arithmetic drift")
    crossing.cell(b + LOW * signed, row["lower_word"])
    same(row["upper_word"], row["target_word"], "upper row word drift")
    low_active, high_active = (bracket.reached(threshold, d) for d in (LOW, HIGH))
    same((low_active, high_active),
         (row["lower_word"] != row["baseline_word"], row["upper_word"] != row["baseline_word"]),
         "row activation closure failed")
    require(not low_active or high_active, "nonmonotonic row activation")
    state = "early_active" if low_active else "late_threshold" if high_active else "inactive"
    decode = crossing.decode
    low_delta = decode(row["lower_word"]) - decode(row["baseline_word"])
    high_delta = decode(row["upper_word"]) - decode(row["baseline_word"])
    return {**row, "state": state, "lower_exact": str(b + LOW * signed),
            "lower_rounded_delta": str(low_delta), "upper_rounded_delta": str(high_delta),
            "bracket_increment": str(high_delta - low_delta),
            "threshold_equation_residual": str(b + signed * bracket.exact(threshold["dose_to_midpoint"])
                                               - bracket.exact(threshold["midpoint"]))}


def decide(coupled, selected_rows):
    same([(c["left_id"], c["right_id"], c["branch"]) for c in coupled],
         [(l, r, b) for l, r in PAIRS for b in BRANCHES], "localizer coupled census changed")
    same([r["row_id"] for r in selected_rows], list(IDS), "localizer selected row census changed")
    states = {r["row_id"]: r["state"] for r in selected_rows}
    require(all(s in ("early_active", "late_threshold", "inactive") for s in states.values()),
            "invalid row partition")
    increments = []
    for item in coupled:
        parts = {i: bracket.exact(item["row_increment_contributions"][str(i)]) for i in IDS}
        total = bracket.exact(item["closed_bracket_increment"])
        require(total != 0 and sum(parts.values()) == total, "coupled increment closure failed")
        increments.append((parts, total))
    late = (states == {319: "late_threshold", 34319: "early_active"}
            and all(v[319] == total and v[34319] == 0 for v, total in increments)
            and bracket.exact(selected_rows[1]["threshold"]["dose_to_midpoint"]) < LOW)
    early = (states[34319] == "early_active"
             and all(v[34319] == total and v[319] == 0 for v, total in increments))
    successor = LATE if late else EARLY if early else MIXED
    return {"classification": successor, "successor": successor,
            "row319_late_prediction": "supported" if late else "rejected",
            "integrity_error": None}


def localize(context):
    selection_gate()
    old, retained, evidence = context["bracket"], context["crossing"], context["evidence"]
    keys = ("control", "polarity", "row_id")
    census = [(c, p, i) for c in parent.CONTROLS for p in POLARITIES for i in IDS]
    for values in (old["rows"], retained["rows"]):
        same([tuple(r[k] for k in keys) for r in values], census, "localizer row census changed")
    rows = [row_account(r, s, evidence) for r, s in zip(old["rows"], retained["rows"], strict=True)]
    lookup = {(r["control"], r["polarity"], r["row_id"]): r for r in rows}
    contrast_keys = ("control", "polarity", "left_id", "right_id", "branch")
    expected = [(c, p, l, r, b) for c in parent.CONTROLS for p in POLARITIES
                for l, r in PAIRS for b in BRANCHES]
    for values in (old["contrasts"], retained["contrasts"]):
        same([tuple(r[k] for k in contrast_keys) for r in values], expected,
             "localizer contrast census changed")
    summaries = context["summaries"]
    dose_rows = [dict(zip(summaries[-1]["contrast_fields"], v, strict=True))
                 for v in summaries[-1]["contrast_values"]]
    same([(r["control"], r["polarity"], r["left_id"], r["right_id"]) for r in dose_rows],
         [(c, p, l, r) for c in parent.CONTROLS for p in POLARITIES for l, r in PAIRS],
         "closed dose census changed")
    doses = {(r["control"], r["polarity"], r["left_id"], r["right_id"]): r for r in dose_rows}
    contrasts, coupled = [], []
    decode = crossing.decode
    for item, cross in zip(old["contrasts"], retained["contrasts"], strict=True):
        c, p, l, r, branch = (item[k] for k in contrast_keys)
        left, right = lookup[c, p, l], lookup[c, p, r]
        refs = evidence["references"]["logits_" + branch]
        reference = (decode(int(refs[l])) - decode(int(refs[r])) if branch == "fp16"
                     else Fraction(float(refs[l])) - Fraction(float(refs[r])))
        baseline = decode(left["baseline_word"]) - decode(right["baseline_word"])
        same((item["baseline_rounded_margin"], item["fixed_reference_margin"]),
             (str(baseline), str(reference)), "original reference/baseline drift")
        same((cross["baseline_rounded_margin"], cross["fixed_reference_margin"],
              cross["retained_margin_change"]), (str(baseline), str(reference), str(baseline - reference)),
             "crossing original reference drift")
        parts = {}
        for name, field in (("lower", bracket.DOSE_FIELDS[0]), ("upper", "observed_delta")):
            values = {str(l): decode(left[name + "_word"]) - decode(left["baseline_word"]),
                      str(r): -(decode(right[name + "_word"]) - decode(right["baseline_word"]))}
            delta = sum(values.values())
            same((item[name + "_exact_delta"], item[name + "_closed_delta"],
                  item[name + "_margin_change"]),
                 (str(delta), str(delta), str(baseline + delta - reference)),
                 "bracket endpoint/reference closure failed")
            same(str(delta), doses[c, p, l, r][field], "closed endpoint outcome drift")
            parts[name] = values
        for name in ("target", "working"):
            ld = decode(left[name + "_word"]) - decode(left["baseline_word"])
            rd = -(decode(right[name + "_word"]) - decode(right["baseline_word"]))
            account = cross[name]
            same([account[k] for k in ("rounded_margin", "margin_change", "delta",
                                       "left_contribution", "right_contribution")],
                 list(map(str, (baseline + ld + rd, baseline + ld + rd - reference, ld + rd, ld, rd))),
                 "crossing individual/reference contrast closure failed")
            same(account["delta"], item["upper_closed_delta"], "retained working/target closure failed")
            same(account["crossing_rows"], [i for i, v in ((l, ld), (r, rd)) if v],
                 "crossing row membership drift")
        increments = {str(i): parts["upper"][str(i)] - parts["lower"][str(i)] for i in IDS}
        closed_increment = bracket.exact(item["upper_closed_delta"]) - bracket.exact(item["lower_closed_delta"])
        require(sum(increments.values()) == closed_increment, "rowwise increment closure failed")
        output = {**item, "row_states": {str(i): lookup[c, p, i]["state"] for i in IDS},
                  "lower_row_contributions": {k: str(v) for k, v in parts["lower"].items()},
                  "upper_row_contributions": {k: str(v) for k, v in parts["upper"].items()},
                  "row_increment_contributions": {k: str(v) for k, v in increments.items()},
                  "closed_bracket_increment": str(closed_increment),
                  "row319_held_at_lower_increment": str(increments["34319"]),
                  "row34319_held_at_lower_increment": str(increments["319"]),
                  "increment_closure_residual": str(sum(increments.values()) - closed_increment)}
        if len(cross["target"]["crossing_rows"]) == 2:
            require((c, p) == ("mapped_all", "reverse"), "coupled case drift")
            coupled.append(output)
        else:
            same(len(cross["target"]["crossing_rows"]), 1, "single-row case drift")
        contrasts.append(output)
    same(old["coupled_contrasts"],
         [r for r in old["contrasts"] if r["control"] == "mapped_all" and r["polarity"] == "reverse"],
         "retained coupled partition drift")
    selected = [lookup["mapped_all", "reverse", i] for i in IDS]
    consistency = old["closed_dose_consistency"]
    same([tuple(r[k] for k in contrast_keys) + (r["dose"],) for r in consistency],
         [key + (str(d),) for key in expected for d in bracket.DOSES], "closed dose account census drift")
    for item in consistency:
        c, p, l, r, branch = (item[k] for k in contrast_keys)
        dose = bracket.exact(item["dose"])
        field = bracket.DOSE_FIELDS[bracket.DOSES.index(dose)]
        same(item["closed_delta"], doses[c, p, l, r][field], "closed dose receipt drift")
        active = [i for i in (l, r) if bracket.reached(lookup[c, p, i]["threshold"], dose)]
        same(item["first_threshold_crossed_rows"], active, "closed threshold state drift")
        observed = bracket.exact(item["closed_delta"])
        signed = bracket.exact(lookup[c, p, l]["signed_row_dose_slope"])
        other = -bracket.exact(lookup[c, p, r]["signed_row_dose_slope"])
        require(signed * other > 0, "first-crossing inference has compensation")
        require(bool(active) == bool(observed) and (not observed or observed * signed > 0),
                "closed dose activation/direction closure failed")
        require(item["consistent"] is True and item["nonzero_consistent"] is True
                and item["direction_consistent"] is True, "closed consistency flag drift")
    same(old["closed_dose_counts"],
         {str(d): {"contrasts": 72, "agreements": 72} for d in bracket.DOSES},
         "closed dose count drift")
    partitions = {state: [{k: r[k] for k in keys} for r in rows if r["state"] == state]
                  for state in ("early_active", "late_threshold", "inactive")}
    return {**decide(coupled, selected), "selected_branch": "exact_target",
            "rows": rows, "row_state_partitions": partitions,
            "row_state_counts": {k: len(v) for k, v in partitions.items()},
            "selected_rows": selected, "contrasts": contrasts, "coupled_contrasts": coupled,
            "row_threshold_closures": len(rows), "retained_scalar_cell_closures": 3 * len(rows),
            "lower_exact_margin_agreements": len(contrasts), "upper_exact_margin_agreements": len(contrasts),
            "target_contrast_closures": len(contrasts), "working_contrast_closures": len(contrasts),
            "increment_closures": len(contrasts), "coupled_increment_closures": len(coupled),
            "other_single_row_contrasts": len(contrasts) - len(coupled),
            "closed_dose_consistency_count": len(consistency),
            "closed_dose_consistency_agreements": len(consistency),
            "closed_dose_counts": old["closed_dose_counts"],
            "larger_dose_rounded_magnitudes_computed": False}


def run_tests(context, report):
    import pytest

    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)

    class Results:
        def __init__(self):
            self.executed = self.failures = self.errors = self.skipped = self.collected = 0

        def pytest_configure(self, config):
            config._ace3_asymmetric_context = (context, report)

        def pytest_collection_finish(self, session):
            self.collected = len(session.items)

        def pytest_runtest_logreport(self, report):
            self.executed += report.when == "call"
            self.failures += report.failed and report.when == "call"
            self.errors += report.failed and report.when != "call"
            self.skipped += report.skipped

    results = Results()
    with patch.dict(os.environ, {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}), redirect_stdout(sys.stderr):
        code = pytest.main([str(TEST), "-q", "-s", "--noconftest",
                            "-p", "no:cacheprovider", "-p", "no:stepwise",
                            "-p", "no:logging"], plugins=[results])
    require(code == 0 and results.collected == results.executed == EXPECTED_TESTS
            and not (results.failures or results.errors or results.skipped),
            "focused pytest failed, errored, skipped or changed census")
    return {"runner": "pytest", "executed": results.executed, "failures": results.failures,
            "errors": results.errors, "skipped": results.skipped,
            "compiled": [parent.record(SOURCE), parent.record(TEST)]}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "asymmetric_threshold_tests": parent.record(TEST)}
    audit = dict.fromkeys(AUDIT_KEYS, 0)
    with read_only(audit):
        context = authenticate()
        evidence = context["evidence"]
        identity, frozen = closed.digest(evidence), closed.digest((context, PREREGISTRATION, FLAGS))
        report = localize(context)
        report_identity = closed.digest(report)
        tests = run_tests(context, report)
        closed.protect(evidence, identity)
        same(closed.digest((context, PREREGISTRATION, FLAGS)), frozen, "retained evidence/contract mutation")
        same(closed.digest(report), report_identity, "localizer report mutation")
        pins = [*context["authenticated_pins"], *origins.values()]
        for pin in pins:
            base.read_bound(pin)
    same(audit, dict.fromkeys(AUDIT_KEYS, 0), "forbidden dispatch/write census changed")
    return {
        "diagnostic_id": NAME, "version": 1, "status": "supported" if report["successor"] == LATE else "rejected",
        "classification": report["classification"], "successor": report["successor"],
        "command": COMMAND, "report": report, "preregistration": PREREGISTRATION, "tests": tests,
        "authenticated_pins": pins, "reviewed_bracket_receipt": RECEIPT,
        "reviewed_crossing_receipt": bracket.RECEIPT, "retained_scalar_receipt": context["summary"],
        "retained_dose_summaries": context["summaries"], "protected_input_identity": identity,
        "selected_head_row_identity": closed.digest(evidence["rows"]),
        "authenticated_files": evidence["files"], "assets": evidence["assets"],
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "retained_thresholds": evidence["result"]["preflight"]["thresholds"],
        "retained_common_component": "UNKNOWN", "counterfactual_common_component": "UNKNOWN",
        "closed_crossing_classification": context["crossing"]["classification"],
        "closed_bracket_classification": context["bracket"]["classification"],
        "mutation_guards": {"protected_inputs_unchanged": True, "retained_receipts_unchanged": True,
                            "report_unchanged": True, "source_pins_reauthenticated": True},
        "dispatch_and_write_audit": audit, "flags": FLAGS, "normal_host_review": "REQUIRED",
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
