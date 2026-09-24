"""Two preregistered terminal-vector suffix doses; stdout-only CPU non-admission."""

import argparse
import builtins
from contextlib import ExitStack, contextmanager, redirect_stdout
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_asymmetric_rne_threshold_localizer_v1 as localizer


bracket, crossing, scalar, base, parent, closed = (
    localizer.bracket, localizer.crossing, localizer.scalar,
    localizer.base, localizer.parent, localizer.closed)
stage = scalar.retained.bracket
require, same, ROOT = base.require, base.same, base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_selected_head_row319_dyadic_straddle_suffix_contrast_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS, POLARITIES, BRANCHES = localizer.IDS, localizer.PAIRS, localizer.POLARITIES, localizer.BRANCHES
DOSES = (Fraction(17, 512), Fraction(9, 256))
EXPECTED_TESTS = 14
MISSION = "3b360c74aee3"
PINS = (
    {"path": str(localizer.SOURCE), "sha256": "6e59d918a400f216ffcc3084d75fe9592e4648addd76732b00fd14307f0e4ce7"},
    {"path": str(localizer.TEST), "sha256": "9ae94e5c1238b07b444792c097336d78d97d4ee4fd3920cdd302c9b524723b55"},
    {"path": str(closed.HANDOFFS / MISSION / "round-0001.json"),
     "sha256": "2b8d5b9422b1bd80a509cd21aa0136ac11662d358fffa5b1762d40f00e9c11c1"},
)
RECEIPT = {
    "path": "/tmp/1789478626568-copilot-tool-output-43501-62a34ca8-c46b-4138-8f06-6b1f6069a8ef.txt",
    "sha256": "ed7487e96bbec042757b68783b1b59b1f454f4973ccdff1e95be734bff7a5ac2",
}
SUPPORT = "dyadic-straddle threshold support"
LOW_MOVES = "lower-dose non-threshold/rounding successor"
HIGH_STATIC = "upper-dose non-threshold/rounding successor"
CONTROL = "control-specific suffix/rounding successor"
ROUNDING = "selected-row non-threshold/rounding successor"
PREREGISTRATION = {
    "operand": "Freeze original-input FP16 stage18 minus original-input binary64 terminal, "
               "all 896 coordinates in input order. Construct only doses 17/512 and 9/256: "
               "RNE_FP16(actual_FP16_stage18 +/- dose * frozen_terminal_vector), minus "
               "forward, plus reverse. Run final RMSNorm and only rows 319,34319; use "
               "retained baseline logits, never a baseline or closed intervention replay.",
    "prediction": "The exact retained final-hidden accumulator slopes predict all 72 "
                  "ordered/reference contrasts at each new terminal-vector dose. For "
                  "mapped_all reverse, 17/512 leaves row319 inactive and 9/256 activates "
                  "it, while row34319 stays early-active. Require individual row-word "
                  "agreement as well as rounded-margin agreement; references remain fixed.",
    "supported": SUPPORT,
    "lower_dose_already_moves_row319": LOW_MOVES,
    "upper_dose_fails_to_move_row319": HIGH_STATIC,
    "other_controls_or_row34319_diverge": CONTROL,
    "remaining_selected_row_model_disagreement": ROUNDING,
    "precedence": "Authentication, arithmetic, census, dispatch and algebraic closure "
                  "failures are UNKNOWN/integrity. Otherwise test control divergence, "
                  "lower-dose movement, upper-dose inactivity, remaining scalar-model "
                  "disagreement, then support. A measured model disagreement is a "
                  "scientific rejection, not an algebraic closure failure.",
    "boundary": "CPU-only reference-guided non-admission suffix counterfactual, not a "
                "reference-independent repair or causal attribution. Retained head-only "
                "slopes are predictions, not interpolated working-dose remainders. "
                "Historical FAIL/UNKNOWN, closed dose outcomes, exact thresholds, "
                "original-input global references and source/token/operand/Q24/state/KV/"
                "lineage gates stay fixed. Q24 state is wider than FP16; native S16 RTZ, "
                "INT4 weights and FP16 operator/KV boundaries are unchanged. No prefix, "
                "admission, reference-producer, full-vocabulary, closed-intervention, "
                "hardware/GPU/RTL/ACE2, production, precision/scale expansion, strict-"
                "FP16-state W4A16, new-token or full-model admission claim.",
}
FLAGS = {
    **localizer.FLAGS, "stage18_operand_construction": True,
    "rmsnorm_recomputation": True, "rmsnorm_operator_replay": True,
    "new_dose_operand_execution": True, "selected_row_accumulator_recomputation": True,
    "closed_localizer_diagnostic_replay": False, "baseline_suffix_replay": False,
}
AUDIT_KEYS = (*localizer.AUDIT_KEYS, "closed_localizer_diagnostic_replays",
              "baseline_suffix_replays", "new_operand_constructions", "operand_rne_scalars")


def selection_gate():
    localizer.selection_gate()
    same((IDS, PAIRS, POLARITIES, BRANCHES),
         ((319, 34319), ((319, 34319), (34319, 319)),
          ("forward", "reverse"), ("fp16", "binary64")), "dyadic selection changed")
    require(DOSES == (Fraction(17, 512), Fraction(9, 256))
            and all(type(d) is Fraction for d in DOSES), "dyadic dose selection changed")


@contextmanager
def no_replay(audit):
    def refusal(counter):
        def refuse(*args, **kwargs):
            audit["forbidden_calls"] += 1
            audit[counter] += 1
            raise RuntimeError("dyadic suffix forbids " + counter)
        return refuse

    with scalar.retained.no_closed_replay(audit), ExitStack() as stack:
        for module, names, counter in (
            (localizer, ("check", "run_tests", "localize", "row_account", "decide"),
             "closed_localizer_diagnostic_replays"),
            (bracket, ("check", "run_tests", "solve", "solve_rows", "decide"),
             "closed_bracket_diagnostic_replays"),
            (crossing, ("check", "run_tests", "attribute", "classify"),
             "closed_crossing_diagnostic_replays"),
            (scalar, ("check", "run_tests", "measure", "accumulate", "classify"),
             "closed_scalar_diagnostic_replays"),
            (scalar.retained, ("check", "run_tests", "prepare", "dots"),
             "closed_headonly_replays"),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, refusal(counter)))

        def opening(original):
            def checked(file, mode="r", *args, **kwargs):
                if not isinstance(mode, str) or any(c in mode for c in "wax+"):
                    if isinstance(mode, str) and any(c in mode for c in "wa+"):
                        audit["artifact_overwrites"] += 1
                    return refusal("writes")()
                return original(file, mode, *args, **kwargs)
            return checked

        for module in (builtins, io):
            stack.enter_context(patch.object(module, "open", opening(module.open)))
        original_open = os.open

        def os_open(path, flags, *args, **kwargs):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                if flags & (os.O_TRUNC | os.O_APPEND):
                    audit["artifact_overwrites"] += 1
                return refusal("writes")()
            return original_open(path, flags, *args, **kwargs)

        stack.enter_context(patch.object(os, "open", os_open))
        for name in ("mkdir", "makedirs", "unlink", "remove", "rmdir", "rename",
                     "replace", "link", "symlink", "chmod", "truncate", "utime"):
            stack.enter_context(patch.object(os, name, refusal("writes")))
        yield


def bind_localizer(receipt, context):
    result, validation = receipt["native_result"], receipt["validation"]
    evidence = context["evidence"]
    same((result["diagnostic_id"], result["version"], result["command"], result["status"],
          result["classification"], result["successor"]),
         (localizer.NAME, 1, localizer.COMMAND, "supported", localizer.LATE, localizer.LATE),
         "reviewed localizer identity/outcome changed")
    same([validation[k] for k in ("native_command", "native_exit", "stdout_json_documents")],
         [localizer.COMMAND, 0, 1], "localizer execution receipt changed")
    for key, expected in (
        ("preregistration", localizer.PREREGISTRATION), ("flags", localizer.FLAGS),
        ("retained_scalar_receipt", context["summary"]),
        ("retained_dose_summaries", context["summaries"]),
        ("protected_input_identity", closed.digest(evidence)),
        ("selected_head_row_identity", closed.digest(evidence["rows"])),
        ("authenticated_files", evidence["files"]), ("assets", evidence["assets"]),
        ("final_reference_authority", evidence["result"]["preflight"]["final_reference"]),
        ("retained_controls_and_failure_gates", evidence["result"]["controls"]),
        ("retained_thresholds", evidence["result"]["preflight"]["thresholds"]),
        ("reviewed_crossing_receipt", bracket.RECEIPT),
        ("reviewed_bracket_receipt", localizer.RECEIPT),
        ("retained_common_component", "UNKNOWN"), ("counterfactual_common_component", "UNKNOWN"),
        ("closed_crossing_classification", context["crossing"]["classification"]),
        ("closed_bracket_classification", context["bracket"]["classification"]),
        ("dispatch_and_write_audit", dict.fromkeys(localizer.AUDIT_KEYS, 0)),
    ):
        same(result[key], expected, "reviewed localizer " + key + " changed")
    same(result["tests"], validation["tests"], "localizer tests receipt drift")
    same([result["tests"][k] for k in ("runner", "executed", "errors", "failures", "skipped")],
         ["pytest", 14, 0, 0, 0], "localizer test gate failed")
    same(result["tests"]["compiled"], [parent.record(localizer.SOURCE), parent.record(localizer.TEST)],
         "localizer compiled source/test drift")
    for key in ("protected_input_identity", "dispatch_and_write_audit", "classification", "successor"):
        same(result[key], validation[key], "localizer validation drift: " + key)
    report = result["report"]
    for key, expected in (
        ("classification", localizer.LATE), ("successor", localizer.LATE),
        ("integrity_error", None), ("selected_branch", "exact_target"),
        ("row_threshold_closures", 36), ("retained_scalar_cell_closures", 108),
        ("increment_closures", 72), ("coupled_increment_closures", 4),
        ("lower_exact_margin_agreements", 72), ("upper_exact_margin_agreements", 72),
        ("target_contrast_closures", 72), ("working_contrast_closures", 72),
        ("other_single_row_contrasts", 68), ("closed_dose_consistency_agreements", 504),
    ):
        same(report[key], expected, "localizer report drift: " + key)
    same([{k: r[k] for k in crossing.FIELDS} for r in report["rows"]],
         [{k: r[k] for k in crossing.FIELDS} for r in context["bracket"]["rows"]],
         "localizer retained scalar splice")
    return report


def authenticate():
    selection_gate()
    for pin in (*PINS, RECEIPT):
        base.read_bound(pin)
    review = json.loads(base.read_bound(PINS[-1]))
    same([review[k] for k in ("kind", "mission_id", "producer_role", "round")],
         ["round_reviewed_handoff", MISSION, "reviewer", 1], "localizer review identity changed")
    same(review["review"]["status"], "done", "independent localizer review incomplete")
    latest_pin = parent.record(Path(PINS[-1]["path"]).with_name("latest.json"))
    latest = json.loads(base.read_bound(latest_pin))
    same((latest["kind"], latest["handoff"]["path"]),
         ("handoff_ref", PINS[-1]["path"]), "localizer review lineage changed")
    text, marker = base.read_bound(RECEIPT).decode(), '{"validation":'
    same(text.count(marker), 1, "localizer receipt missing or duplicated")
    receipt, _ = json.JSONDecoder().raw_decode(text[text.index(marker):])
    require(set(receipt) == {"validation", "native_result"}, "localizer envelope changed")
    context = localizer.authenticate()
    report = bind_localizer(receipt, context)
    delta, _ = stage.bind_doses(context["evidence"], *context["summaries"][:-1])
    pins = [*PINS, RECEIPT, latest_pin, *context["authenticated_pins"],
            *receipt["native_result"]["authenticated_pins"]]
    for pin in pins:
        base.read_bound(pin)
    return {**context, "localizer": report, "localizer_receipt": receipt,
            "delta": delta, "authenticated_pins": pins}


def round_scalar(value):
    require(type(value) is Fraction and value.denominator & (value.denominator - 1) == 0,
            "new-dose RNE requires an exact dyadic scalar")
    bits, saturated = parent.head.fixed_to_f16(value.numerator, value.denominator.bit_length() - 1)
    require(not saturated, "new-dose scalar saturated")
    rounded = crossing.decode(bits)
    return {"exact": str(value), "fp16_word": bits, "rounded": str(rounded),
            "rne_remainder": str(rounded - value)}


def derive(context):
    selection_gate()
    retained = context["localizer"]["rows"]
    same([(r["control"], r["polarity"], r["row_id"]) for r in retained],
         [(c, p, i) for c in parent.CONTROLS for p in POLARITIES for i in IDS],
         "retained scalar census changed")
    rows = []
    for row in retained:
        b, s = (bracket.exact(row[k]) for k in ("baseline_exact", "signed_row_dose_slope"))
        require(s == (-1 if row["polarity"] == "forward" else 1)
                * bracket.exact(row["frozen_row_dose_slope"]), "signed scalar slope drift")
        threshold = bracket.threshold(b, row["baseline_word"], s)
        same(threshold, row["threshold"], "retained threshold equation drift")
        same(row["baseline_word"],
             int(context["evidence"]["arrays"][row["control"]]["logits"][row["row_id"]]),
             "baseline scalar/logit binding drift")
        predictions = {}
        for dose in DOSES:
            rounded = round_scalar(b + dose * s)
            active = bracket.reached(threshold, dose)
            same(active, rounded["fp16_word"] != row["baseline_word"],
                 "first-threshold scalar activation closure failed")
            predictions[str(dose)] = {**rounded, "active": active,
                                      "dose_slack": str(bracket.exact(threshold["dose_to_midpoint"]) - dose)}
        rows.append({**row, "dyadic_predictions": predictions})
    selected = {r["row_id"]: r for r in rows
                if (r["control"], r["polarity"]) == ("mapped_all", "reverse")}
    t319, t34319 = (bracket.exact(selected[i]["threshold"]["dose_to_midpoint"]) for i in IDS)
    require(DOSES[0] < t319 < DOSES[1] and 0 < t34319 < Fraction(1, 32),
            "exact dyadic straddle/early-active derivation failed")
    inequalities = []
    for left, right in ((DOSES[0], t319), (t319, DOSES[1]), (t34319, Fraction(1, 32))):
        lhs, rhs = left.numerator * right.denominator, right.numerator * left.denominator
        require(lhs < rhs, "integer cross-product inequality failed")
        inequalities.append({"left": str(left), "right": str(right),
                             "left_cross_product": lhs, "right_cross_product": rhs,
                             "positive_difference": rhs - lhs})
    return {"rows": rows, "row319_threshold": str(t319), "row34319_threshold": str(t34319),
            "strict_inequalities": inequalities,
            "derivation": "threshold = (directed FP16 midpoint - retained baseline accumulator) / signed retained slope",
            "working_remainder_interpolated": False}


def prepare(words, delta, polarity, dose):
    selection_gate()
    require(type(dose) is Fraction and dose in DOSES, "unregistered or closed dose")
    require(polarity in POLARITIES, "invalid polarity")
    require(isinstance(words, np.ndarray) and words.dtype == np.dtype("<u2")
            and words.shape == (896,) and np.isfinite(words.view("<f2")).all(),
            "invalid actual FP16 terminal operand")
    require(type(delta) is tuple and len(delta) == 896
            and all(type(v) is Fraction and v.denominator & (v.denominator - 1) == 0 for v in delta),
            "invalid frozen dyadic terminal vector")
    signed = -dose if polarity == "forward" else dose
    working = words.copy()
    for i in range(896):
        target = Fraction(float(words.view("<f2")[i])) + signed * delta[i]
        working[i] = round_scalar(target)["fp16_word"]
    working.flags.writeable = False
    return working, {"dose": str(dose), "signed_dose": str(signed),
                     "coordinate_order": "all_896_input_order_including_zeros",
                     "working_fp16_words": working.tolist(),
                     "changed_coordinates": np.flatnonzero(words != working).tolist(),
                     "frozen_terminal_vector_identity": closed.digest(delta)}


def decide(tables):
    selection_gate()
    same(list(tables), list(map(str, DOSES)), "dose table census changed")
    expected = [(c, p, l, r, b) for c in parent.CONTROLS for p in POLARITIES
                for l, r in PAIRS for b in BRANCHES]
    agreements, selected, control_diverged = {}, {}, False
    for dose, rows in tables.items():
        same([tuple(r[k] for k in ("control", "polarity", "left_id", "right_id", "branch"))
              for r in rows], expected, "contrast census changed")
        for row in rows:
            old, new, ref, observed, predicted = (bracket.exact(row[k]) for k in (
                "baseline_margin", "measured_margin", "fixed_reference_margin",
                "observed_delta", "predicted_delta"))
            decode = crossing.decode
            require(old == decode(row["baseline_words"][0]) - decode(row["baseline_words"][1]),
                    "baseline word/margin closure failed")
            require(new == decode(row["measured_words"][0]) - decode(row["measured_words"][1]),
                    "measured word/margin closure failed")
            require(predicted == decode(row["predicted_words"][0]) - decode(row["predicted_words"][1]) - old,
                    "prediction word/margin closure failed")
            require((observed, bracket.exact(row["retained_margin_change"]),
                     bracket.exact(row["intervened_margin_change"]))
                    == (new - old, old - ref, new - ref), "original-reference/algebraic closure failed")
            same(row["prediction_agrees"], observed == predicted, "prediction closure flag drift")
            same(row["individual_words_agree"], row["predicted_words"] == row["measured_words"],
                 "individual prediction closure flag drift")
            is_selected = (row["control"], row["polarity"]) == ("mapped_all", "reverse")
            early_index = (row["left_id"], row["right_id"]).index(34319)
            control_diverged |= ((not is_selected and not row["individual_words_agree"])
                                 or (is_selected and row["measured_words"][early_index]
                                     != row["predicted_words"][early_index]))
        selected[dose] = [r for r in rows if (r["control"], r["polarity"]) == ("mapped_all", "reverse")]
        agreements[dose] = {
            "contrasts": len(rows), "scalar_threshold_agreements": sum(r["prediction_agrees"] for r in rows),
            "individual_word_agreements": sum(r["individual_words_agree"] for r in rows),
            "mapped_all_reverse_agreements": sum(r["individual_words_agree"] for r in selected[dose]),
        }
    low, high = (selected[str(d)] for d in DOSES)

    def active(row):
        i = (row["left_id"], row["right_id"]).index(319)
        return row["measured_words"][i] != row["baseline_words"][i]

    if control_diverged:
        outcome = CONTROL
    elif any(active(r) for r in low):
        outcome = LOW_MOVES
    elif any(not active(r) for r in high):
        outcome = HIGH_STATIC
    elif any(v["individual_word_agreements"] != 72 or v["scalar_threshold_agreements"] != 72
             for v in agreements.values()):
        outcome = ROUNDING
    else:
        outcome = SUPPORT
    return {"status": "supported" if outcome == SUPPORT else "rejected",
            "classification": outcome, "successor": outcome, "integrity_error": None,
            "prediction_closure_counts": agreements}


def measure(context, derivation, audit):
    evidence, delta = context["evidence"], context["delta"]
    rows = np.stack([evidence["rows"][i] for i in IDS])
    lookup = {(r["control"], r["polarity"], r["row_id"]): r for r in derivation["rows"]}
    prepared = {}
    with stage.hotspot.read_only(audit), no_replay(audit):
        for dose in DOSES:
            for control in parent.CONTROLS:
                for polarity in POLARITIES:
                    prepared[dose, control, polarity] = prepare(
                        evidence["archives"][control]["stage18"], delta, polarity, dose)
                    audit["new_operand_constructions"] += 1
                    audit["operand_rne_scalars"] += 896
    jobs = [value[0] for value in prepared.values()]
    tables, outputs = {str(d): [] for d in DOSES}, {str(d): [] for d in DOSES}
    with closed.suffix_only(audit, jobs, evidence["weight_array"], rows), no_replay(audit):
        for (dose, control, polarity), (words, operand) in prepared.items():
            normalized, norm_scalars = parent.rmsnorm(words, evidence["weight_array"])
            measured = parent.logits(normalized, rows)
            audit["exact_selected_row_accumulations"] += len(IDS)
            require(np.isfinite(normalized.view("<f2")).all() and np.isfinite(measured.view("<f2")).all(),
                    "nonfinite suffix result")
            outputs[str(dose)].append({
                "control": control, "polarity": polarity, "operand": operand,
                "rmsnorm_words": normalized.tolist(), "rmsnorm_scalars": norm_scalars,
                "selected_logit_words": measured.tolist(),
            })
            for left, right in PAIRS:
                old_words = [int(evidence["arrays"][control]["logits"][i]) for i in (left, right)]
                measured_words = [int(measured[IDS.index(i)]) for i in (left, right)]
                predicted_words = [lookup[control, polarity, i]["dyadic_predictions"][str(dose)]["fp16_word"]
                                   for i in (left, right)]
                old, new, predicted = (crossing.decode(w[0]) - crossing.decode(w[1])
                                       for w in (old_words, measured_words, predicted_words))
                for branch in BRANCHES:
                    refs = evidence["references"]["logits_" + branch]
                    ref = (crossing.decode(int(refs[left])) - crossing.decode(int(refs[right]))
                           if branch == "fp16" else Fraction(float(refs[left])) - Fraction(float(refs[right])))
                    tables[str(dose)].append({
                        "control": control, "polarity": polarity, "left_id": left,
                        "right_id": right, "branch": branch, "dose": str(dose),
                        "baseline_words": old_words, "measured_words": measured_words,
                        "predicted_words": predicted_words, "baseline_margin": str(old),
                        "measured_margin": str(new), "predicted_delta": str(predicted - old),
                        "observed_delta": str(new - old), "fixed_reference_margin": str(ref),
                        "retained_margin_change": str(old - ref),
                        "intervened_margin_change": str(new - ref),
                        "prediction_agrees": new == predicted,
                        "individual_words_agree": measured_words == predicted_words,
                    })
    return {**decide(tables), "dose_tables": tables, "suffix_outputs": outputs,
            "dyadic_threshold_derivation": derivation,
            "frozen_original_terminal_vector": list(map(str, delta)),
            "frozen_original_terminal_vector_identity": closed.digest(delta)}


def experiment(audit):
    selection_gate()
    with stage.hotspot.read_only(audit), no_replay(audit):
        context = authenticate()
        derivation = derive(context)
    identity = closed.digest(context["evidence"])
    frozen = closed.digest((context, derivation, PREREGISTRATION, FLAGS))
    report = measure(context, derivation, audit)
    closed.protect(context["evidence"], identity)
    same(closed.digest((context, derivation, PREREGISTRATION, FLAGS)), frozen,
         "frozen evidence/derivation/contract mutation")
    return context, report


def run_tests(context, report):
    import pytest

    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)

    class Results:
        def __init__(self):
            self.collected = self.executed = self.errors = self.failures = self.skipped = 0

        def pytest_configure(self, config):
            config._ace3_dyadic_context = (context, report)

        def pytest_collection_finish(self, session):
            self.collected = len(session.items)

        def pytest_runtest_logreport(self, report):
            self.executed += report.when == "call"
            self.failures += report.failed and report.when == "call"
            self.errors += report.failed and report.when != "call"
            self.skipped += report.skipped

    results = Results()
    with patch.dict(os.environ, {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}), redirect_stdout(sys.stderr):
        code = pytest.main([str(TEST), "-q", "-s", "--noconftest", "-p", "no:cacheprovider",
                            "-p", "no:stepwise", "-p", "no:logging"], plugins=[results])
    require(code == 0 and results.collected == results.executed == EXPECTED_TESTS
            and not (results.errors or results.failures or results.skipped),
            "focused pytest failed, errored, skipped or changed census")
    return {"runner": "pytest", "executed": results.executed, "errors": results.errors,
            "failures": results.failures, "skipped": results.skipped,
            "compiled": [parent.record(SOURCE), parent.record(TEST)]}


def check(audit):
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE), "dyadic_tests": parent.record(TEST)}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("ace3.") and path and Path(path).suffix == ".py":
            require(Path(path).resolve().is_relative_to(ROOT), "foreign source origin")
            origins[name] = parent.record(Path(path).resolve())
    context, report = experiment(audit)
    frozen = closed.digest((context, report, PREREGISTRATION, FLAGS))
    with stage.hotspot.read_only(audit), no_replay(audit):
        tests = run_tests(context, report)
        same(closed.digest((context, report, PREREGISTRATION, FLAGS)), frozen, "test mutation")
        pins = [*context["authenticated_pins"], *origins.values()]
        for pin in pins:
            base.read_bound(pin)
    expected = dict.fromkeys(AUDIT_KEYS, 0)
    expected.update(final_rmsnorm_invocations=36, selected_row_head_invocations=36,
                    exact_selected_row_accumulations=72,
                    new_operand_constructions=36, operand_rne_scalars=32256)
    same(audit, expected, "suffix-only dispatch/write census changed")
    evidence = context["evidence"]
    return {
        "diagnostic_id": NAME, "version": 1, "command": COMMAND,
        "status": report["status"], "classification": report["classification"],
        "successor": report["successor"], "report": report, "tests": tests,
        "preregistration": PREREGISTRATION, "flags": FLAGS, "authenticated_pins": pins,
        "reviewed_localizer_receipt": RECEIPT, "reviewed_bracket_receipt": localizer.RECEIPT,
        "reviewed_crossing_receipt": bracket.RECEIPT,
        "retained_dose_summaries": context["summaries"],
        "closed_crossing_classification": context["crossing"]["classification"],
        "closed_bracket_classification": context["bracket"]["classification"],
        "closed_localizer_classification": context["localizer"]["classification"],
        "protected_input_identity": closed.digest(evidence),
        "selected_head_row_identity": closed.digest(evidence["rows"]),
        "authenticated_files": evidence["files"], "assets": evidence["assets"],
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "retained_thresholds": evidence["result"]["preflight"]["thresholds"],
        "retained_common_component": "UNKNOWN", "counterfactual_common_component": "UNKNOWN",
        "mutation_guards": {"protected_inputs_unchanged": True, "retained_receipts_unchanged": True,
                            "report_unchanged": True, "source_pins_reauthenticated": True},
        "dispatch_and_write_audit": audit, "normal_host_review": "REQUIRED",
        "claim_boundary": PREREGISTRATION["boundary"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", required=True, action="store_true")
    parser.parse_args(argv)
    audit = dict.fromkeys(AUDIT_KEYS, 0)
    try:
        result = check(audit)
    except (ValueError, RuntimeError, OSError, ArithmeticError, LookupError, TypeError, ImportError) as error:
        print(json.dumps({"diagnostic_id": NAME, "status": "UNKNOWN", "classification": "UNKNOWN/integrity",
                          "successor": None, "integrity_error": f"{type(error).__name__}: {error}",
                          "dispatch_and_write_audit": audit, "preregistration": PREREGISTRATION,
                          "flags": FLAGS, "normal_host_review": "REQUIRED"}, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
