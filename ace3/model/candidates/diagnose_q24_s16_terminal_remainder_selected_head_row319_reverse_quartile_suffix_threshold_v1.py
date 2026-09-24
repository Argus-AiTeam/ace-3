"""Three new reverse terminal-vector doses; stdout-only CPU non-admission."""

import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_row319_dyadic_straddle_suffix_contrast_v1 as prior


base, parent, closed, stage = prior.base, prior.parent, prior.closed, prior.stage
bracket, crossing = prior.bracket, prior.crossing
require, same, ROOT = prior.require, prior.same, prior.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_selected_head_row319_reverse_quartile_suffix_threshold_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS, BRANCHES = prior.IDS, prior.PAIRS, prior.BRANCHES
DOSES = (Fraction(65, 2048), Fraction(33, 1024), Fraction(67, 2048))
LOWER, UPPER = Fraction(1, 32), Fraction(17, 512)
EXPECTED_TESTS = 15
MISSION = "e762f68f110e"
PINS = (
    {"path": str(prior.SOURCE), "sha256": "5e057f85efefb6d12472830f1ea4c9da811c4ceedcc470f48ebdf2b065fba811"},
    {"path": str(prior.TEST), "sha256": "3b09bbc7cabfb31a52e2f00588be2e7298eb765cc0ab527dc5377931129b56c5"},
    {"path": str(closed.HANDOFFS / MISSION / "round-0001.json"),
     "sha256": "cd1a90cd7e7e92014ec75b795b3b64eb0b7a0d2f3943ac21d3e50001b5553183"},
)
RECEIPT = {
    "path": "/tmp/1789479529804-copilot-tool-output-60331-a69f400e-7a89-4126-b6ad-fe9c7506341c.txt",
    "sha256": "6f80c5a4e8b3afee0797e4665476ba126780f34d4b0712a63adf6247591b3c34",
}
NEAR, MIDDLE, UPPER_ONLY, MIXED = (
    "near closed 1/32 boundary", "middle quartile",
    "upper quartile by retained 17/512", "mixed by control family")
SUCCESSORS = {
    NEAR: "dose search closed; may motivate a distinct control-specific rounding-mechanism "
          "question using retained bytes, subject to new preregistration and independent review",
    MIDDLE: "dose search closed; may motivate a distinct control-specific rounding-mechanism "
            "question using retained bytes, subject to new preregistration and independent review",
    UPPER_ONLY: "dose search closed; may motivate a distinct control-specific rounding-mechanism "
                "question using retained bytes, subject to new preregistration and independent review",
    MIXED: "dose search and global-threshold model closed; may motivate at most one "
           "preregistered family-contrast question using retained bytes, subject to independent review",
}
FAMILIES = {
    "frozen_inherited": parent.CONTROLS[:4],
    "scratch": parent.CONTROLS[4:6],
    "mapped62": ("mapped62",), "mapped_all": ("mapped_all",),
    "inherited_native": ("inherited_native",),
}
PREREGISTRATION = {
    "operand": "Only reverse RNE_FP16(actual_FP16_stage18 + dose * frozen_terminal_vector), "
               "all 896 coordinates in input order, doses 65/2048, 33/1024, 67/2048. "
               "Freeze original-input FP16 stage18 minus independent binary64 terminal. "
               "Nine existing controls, final RMSNorm and selected rows 319/34319 only; "
               "retain baseline logits and both original-input global references.",
    "prediction": "Retained exact scalar transfer predicts row319 inactivity at all "
                  "three new doses in every control, while reviewed suffix evidence "
                  "already measures row319 activity at 17/512 in all nine controls.",
    "classification": "Per control, first measured activity at 65/2048 selects quartile 1 "
                      "(near closed 1/32); at 33/1024 or 67/2048 selects quartile 2 or 3 "
                      "(middle); none at the new doses selects quartile 4 using retained "
                      "17/512 activity. Different quartiles across controls select mixed. "
                      "Record reversions; do not assume monotonicity or claim the "
                      "continuous onset lies in a sampled interval.",
    "successors": SUCCESSORS,
    "branch_closure": "This is the terminal dose/threshold-localization experiment. "
                      "After normal independent review, every outcome closes dose search, "
                      "including rejection and UNKNOWN. No further dose subdivision, "
                      "mini-bisection, coordinate choice, reference mutation or "
                      "scalar-threshold transfer claim is permitted. Possible distinct "
                      "mechanism/family questions are not authorized executions or causal claims.",
    "integrity": "Authentication, exact arithmetic, reference/word closure, source or "
                 "census/dispatch/write failures select UNKNOWN/integrity, no successor. "
                 "No scientific successor is permitted until the exact missing integrity "
                 "prerequisite is supplied and independently reviewed; dose search stays closed. "
                 "A valid early activation rejects scalar transfer, not integrity.",
    "boundary": prior.PREREGISTRATION["boundary"] + " No prior 17/512 or 9/256 suffix "
                "replay, no new forward doses, no continuous-threshold or bottleneck claim.",
}
FLAGS = {
    **prior.FLAGS, "closed_dyadic_diagnostic_replay": False,
    "prior_17_512_replay": False, "new_forward_dose_execution": False,
    "continuous_threshold_claim": False, "further_dose_search_permitted": False,
    "scalar_threshold_transfer_claim": False,
}
AUDIT_KEYS = (*prior.AUDIT_KEYS, "closed_dyadic_diagnostic_replays", "prior_17_512_replays")


def selection_gate():
    prior.selection_gate()
    require(DOSES == (Fraction(65, 2048), Fraction(33, 1024), Fraction(67, 2048))
            and all(type(d) is Fraction for d in DOSES), "quartile dose selection changed")
    require((LOWER, UPPER) == (Fraction(1, 32), Fraction(17, 512)), "endpoint selection changed")
    same([c for controls in FAMILIES.values() for c in controls], list(parent.CONTROLS),
         "control family census changed")


def expected_audit():
    audit = dict.fromkeys(AUDIT_KEYS, 0)
    audit.update(final_rmsnorm_invocations=27, selected_row_head_invocations=27,
                 exact_selected_row_accumulations=54, new_operand_constructions=27,
                 operand_rne_scalars=24192)
    return audit


@contextmanager
def no_replay(audit):
    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        audit["closed_dyadic_diagnostic_replays"] += 1
        if kwargs.get("dose") == UPPER or (len(args) == 4 and args[3] == UPPER):
            audit["prior_17_512_replays"] += 1
        raise RuntimeError("closed dyadic diagnostic/dose replay forbidden")

    with prior.no_replay(audit), ExitStack() as stack:
        for name in ("check", "run_tests", "experiment", "derive", "prepare", "measure", "decide"):
            stack.enter_context(patch.object(prior, name, refuse))
        yield


def fixed_margin(evidence, branch, left, right):
    refs = evidence["references"]["logits_" + branch]
    if branch == "fp16":
        return crossing.decode(int(refs[left])) - crossing.decode(int(refs[right]))
    require(branch == "binary64", "unregistered reference branch")
    return Fraction(float(refs[left])) - Fraction(float(refs[right]))


def bind_prior(receipt, context):
    result, validation, evidence = receipt["native_result"], receipt["validation"], context["evidence"]
    same([result[k] for k in ("diagnostic_id", "version", "command", "status", "classification", "successor")],
         [prior.NAME, 1, prior.COMMAND, "rejected", prior.CONTROL, prior.CONTROL],
         "reviewed dyadic identity/outcome changed")
    same([validation[k] for k in ("native_command", "native_exit", "stdout_json_documents")],
         [prior.COMMAND, 0, 1], "reviewed dyadic execution changed")
    bindings = {
        "preregistration": prior.PREREGISTRATION, "flags": prior.FLAGS,
        "retained_dose_summaries": context["summaries"],
        "protected_input_identity": closed.digest(evidence),
        "selected_head_row_identity": closed.digest(evidence["rows"]),
        "authenticated_files": evidence["files"], "assets": evidence["assets"],
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "retained_thresholds": evidence["result"]["preflight"]["thresholds"],
        "reviewed_localizer_receipt": prior.RECEIPT,
        "reviewed_bracket_receipt": prior.localizer.RECEIPT,
        "reviewed_crossing_receipt": bracket.RECEIPT,
        "closed_crossing_classification": context["crossing"]["classification"],
        "closed_bracket_classification": context["bracket"]["classification"],
        "closed_localizer_classification": context["localizer"]["classification"],
        "retained_common_component": "UNKNOWN", "counterfactual_common_component": "UNKNOWN",
    }
    for key, expected in bindings.items():
        same(result[key], expected, "reviewed dyadic binding changed: " + key)
    audit = dict.fromkeys(prior.AUDIT_KEYS, 0)
    audit.update(final_rmsnorm_invocations=36, selected_row_head_invocations=36,
                 exact_selected_row_accumulations=72, new_operand_constructions=36,
                 operand_rne_scalars=32256)
    same(result["dispatch_and_write_audit"], audit, "reviewed dyadic dispatch changed")
    same(result["tests"], validation["tests"], "reviewed dyadic tests drift")
    same([result["tests"][k] for k in ("runner", "executed", "errors", "failures", "skipped")],
         ["pytest", 14, 0, 0, 0], "reviewed dyadic test gate failed")
    same(result["tests"]["compiled"], [parent.record(prior.SOURCE), parent.record(prior.TEST)],
         "reviewed dyadic compiled pins changed")
    for key in ("classification", "successor", "protected_input_identity", "dispatch_and_write_audit"):
        same(result[key], validation[key], "reviewed dyadic validation drift: " + key)
    report = result["report"]
    same([report[k] for k in ("status", "classification", "successor", "integrity_error")],
         ["rejected", prior.CONTROL, prior.CONTROL, None], "reviewed dyadic report changed")
    same(report["prediction_closure_counts"], validation["prediction_closure_counts"],
         "reviewed agreement receipt drift")
    same(report["frozen_original_terminal_vector"], list(map(str, context["delta"])),
         "reviewed terminal operand changed")
    same(report["frozen_original_terminal_vector_identity"], closed.digest(context["delta"]),
         "reviewed terminal identity changed")
    same(list(report["dose_tables"]), ["17/512", "9/256"], "reviewed dose census changed")
    endpoints = {}
    for dose, rows in report["dose_tables"].items():
        same([tuple(r[k] for k in ("control", "polarity", "left_id", "right_id", "branch")) for r in rows],
             [(c, p, l, r, b) for c in parent.CONTROLS for p in prior.POLARITIES
              for l, r in PAIRS for b in BRANCHES], "reviewed contrast census changed")
        agreements = 36 if dose == "17/512" else 40
        same(report["prediction_closure_counts"][dose], {
            "contrasts": 72, "individual_word_agreements": agreements,
            "scalar_threshold_agreements": agreements,
            "mapped_all_reverse_agreements": 0 if dose == "17/512" else 4,
        }, "reviewed rejection counts changed")
        for row in rows:
            c, l, r, b = (row[k] for k in ("control", "left_id", "right_id", "branch"))
            same(row["baseline_words"], [int(evidence["arrays"][c]["logits"][i]) for i in (l, r)],
                 "reviewed baseline binding changed")
            old, new = (crossing.decode(words[0]) - crossing.decode(words[1])
                        for words in (row["baseline_words"], row["measured_words"]))
            ref = fixed_margin(evidence, b, l, r)
            require(tuple(Fraction(row[k]) for k in (
                "baseline_margin", "measured_margin", "fixed_reference_margin",
                "observed_delta", "retained_margin_change", "intervened_margin_change"))
                == (old, new, ref, new - old, old - ref, new - ref), "reviewed reference closure failed")
            if dose == "17/512" and row["polarity"] == "reverse" and (l, r, b) == (319, 34319, "fp16"):
                require(row["measured_words"][0] != row["baseline_words"][0],
                        "reviewed row319 endpoint is inactive")
                endpoints[c] = {"dose": dose, "baseline_words": row["baseline_words"],
                                "measured_words": row["measured_words"], "row319_active": True}
    same(list(endpoints), list(parent.CONTROLS), "reviewed endpoint control census changed")
    return endpoints


def authenticate():
    selection_gate()
    for pin in (*PINS, RECEIPT):
        base.read_bound(pin)
    review = json.loads(base.read_bound(PINS[-1]))
    same([review[k] for k in ("kind", "mission_id", "producer_role", "round")],
         ["round_reviewed_handoff", MISSION, "reviewer", 1], "dyadic review identity changed")
    same(review["review"]["status"], "done", "independent dyadic review incomplete")
    latest_pin = parent.record(Path(PINS[-1]["path"]).with_name("latest.json"))
    latest = json.loads(base.read_bound(latest_pin))
    same((latest["kind"], latest["handoff"]["path"]), ("handoff_ref", PINS[-1]["path"]),
         "dyadic review lineage changed")
    text, marker = base.read_bound(RECEIPT).decode(), '{"validation":'
    same(text.count(marker), 1, "dyadic receipt missing or duplicated")
    receipt, _ = json.JSONDecoder().raw_decode(text[text.index(marker):])
    same(sorted(receipt), ["native_result", "validation"], "dyadic receipt envelope changed")
    context = prior.authenticate()
    endpoints = bind_prior(receipt, context)
    pins = [*PINS, RECEIPT, latest_pin, *context["authenticated_pins"],
            *receipt["native_result"]["authenticated_pins"]]
    for pin in pins:
        base.read_bound(pin)
    return {**context, "prior_receipt": receipt, "prior_endpoints": endpoints, "authenticated_pins": pins}


def derive(context):
    selection_gate()
    bounds = (LOWER, *DOSES, UPPER)
    inequalities = []
    for left, right in zip(bounds, bounds[1:]):
        lhs, rhs = left.numerator * right.denominator, right.numerator * left.denominator
        require(lhs < rhs, "exact dose ordering failed")
        inequalities.append({"left": str(left), "right": str(right),
                             "left_cross_product": lhs, "right_cross_product": rhs,
                             "positive_difference": rhs - lhs})
    retained = [r for r in context["localizer"]["rows"] if r["polarity"] == "reverse"]
    same([(r["control"], r["row_id"]) for r in retained],
         [(c, i) for c in parent.CONTROLS for i in IDS], "retained reverse scalar census changed")
    rows = []
    for row in retained:
        b, s = (bracket.exact(row[k]) for k in ("baseline_exact", "signed_row_dose_slope"))
        require(s == bracket.exact(row["frozen_row_dose_slope"]), "reverse scalar slope drift")
        same(bracket.threshold(b, row["baseline_word"], s), row["threshold"], "scalar threshold drift")
        same(row["baseline_word"], int(context["evidence"]["arrays"][row["control"]]["logits"][row["row_id"]]),
             "scalar baseline/logit drift")
        predictions = {}
        for dose in DOSES:
            rounded = prior.round_scalar(b + dose * s)
            active = rounded["fp16_word"] != row["baseline_word"]
            same(active, bracket.reached(row["threshold"], dose), "scalar first-crossing closure failed")
            if row["row_id"] == 319:
                require(not active, "preregistered scalar inactivity derivation failed")
            predictions[str(dose)] = {**rounded, "active": active,
                "dose_slack": str(bracket.exact(row["threshold"]["dose_to_midpoint"]) - dose)}
        rows.append({**row, "quartile_predictions": predictions})
    return {"rows": rows, "dose_order": list(map(str, bounds)), "strict_inequalities": inequalities,
            "row319_inactive_predictions": 27, "prior_row319_active_controls": 9,
            "working_remainder_interpolated": False}


def prepare(words, delta, polarity, dose):
    selection_gate()
    require(type(dose) is Fraction and dose in DOSES, "unregistered or closed dose")
    require(polarity == "reverse", "only reverse polarity is authorized")
    require(isinstance(words, np.ndarray) and words.dtype == np.dtype("<u2")
            and words.shape == (896,) and np.isfinite(words.view("<f2")).all(),
            "invalid actual FP16 terminal operand")
    require(type(delta) is tuple and len(delta) == 896
            and all(type(v) is Fraction and v.denominator & (v.denominator - 1) == 0 for v in delta),
            "invalid frozen dyadic terminal vector")
    working = np.asarray([prior.round_scalar(Fraction(float(w)) + dose * v)["fp16_word"]
                          for w, v in zip(words.view("<f2"), delta, strict=True)], dtype="<u2")
    working.flags.writeable = False
    return working, {"dose": str(dose), "signed_dose": str(dose),
                     "coordinate_order": "all_896_input_order_including_zeros",
                     "working_fp16_words": working.tolist(),
                     "changed_coordinates": np.flatnonzero(words != working).tolist(),
                     "frozen_terminal_vector_identity": closed.digest(delta)}


def decide(tables):
    selection_gate()
    same(list(tables), list(map(str, DOSES)), "new dose table census changed")
    observations, counts, baselines = {c: [] for c in parent.CONTROLS}, {}, {}
    for dose, table in tables.items():
        same([tuple(r[k] for k in ("control", "polarity", "left_id", "right_id", "branch")) for r in table],
             [(c, "reverse", l, r, b) for c in parent.CONTROLS for l, r in PAIRS for b in BRANCHES],
             "new contrast census changed")
        canonical = {}
        for row in table:
            same(row["dose"], dose, "contrast dose drift")
            old, new, predicted = (crossing.decode(w[0]) - crossing.decode(w[1])
                                   for w in (row["baseline_words"], row["measured_words"], row["predicted_words"]))
            ref = Fraction(row["fixed_reference_margin"])
            require(tuple(Fraction(row[k]) for k in (
                "baseline_margin", "measured_margin", "predicted_delta", "observed_delta",
                "retained_margin_change", "intervened_margin_change"))
                == (old, new, predicted - old, new - old, old - ref, new - ref),
                "new reference/algebraic closure failed")
            same(row["prediction_agrees"], new == predicted, "margin agreement flag drift")
            same(row["individual_words_agree"], row["measured_words"] == row["predicted_words"],
                 "word agreement flag drift")
            words = tuple(tuple(row[k][::1 if row["left_id"] == 319 else -1])
                          for k in ("baseline_words", "measured_words", "predicted_words"))
            c = row["control"]
            if c in canonical:
                same(words, canonical[c], "ordered-pair/reference word consistency failed")
            canonical[c] = words
            require(words[0][0] == words[2][0], "scalar row319 inactivity prediction drift")
            if c in baselines:
                same(words[0], baselines[c], "cross-dose baseline drift")
            baselines[c] = words[0]
        for c, words in canonical.items():
            observations[c].append({"dose": dose, "row319_active": words[1][0] != words[0][0],
                                    "row34319_active": words[1][1] != words[0][1],
                                    "measured_words": list(words[1])})
        counts[dose] = {"contrasts": 36,
                        "scalar_threshold_agreements": sum(r["prediction_agrees"] for r in table),
                        "individual_word_agreements": sum(r["individual_words_agree"] for r in table),
                        "row319_active_controls": sum(v[-1]["row319_active"] for v in observations.values())}
    controls = []
    for c, measured in observations.items():
        active = [r["row319_active"] for r in measured]
        first = next((i for i, value in enumerate(active) if value), 3)
        classification = NEAR if first == 0 else UPPER_ONLY if first == 3 else MIDDLE
        controls.append({
            "control": c, "family": next(f for f, members in FAMILIES.items() if c in members),
            "classification": classification, "first_activation_quartile": first + 1,
            "first_measured_active_dose": str((*DOSES, UPPER)[first]),
            "first_active_source": "new_suffix" if first < 3 else "reviewed_prior_suffix",
            "sampling_interval": {"lower_exclusive": str((LOWER, *DOSES)[first]),
                                  "upper_inclusive": str((*DOSES, UPPER)[first])},
            "activity_reversion_observed": any(a and not b for a, b in zip(active, active[1:])),
            "new_measurements": measured,
        })
    quartiles = {r["first_activation_quartile"] for r in controls}
    outcome = controls[0]["classification"] if len(quartiles) == 1 else MIXED
    return {"status": "supported" if quartiles == {4} else "rejected",
            "classification": outcome, "successor": SUCCESSORS[outcome], "integrity_error": None,
            "dose_search_closed": True, "global_threshold_model_closed": outcome == MIXED,
            "per_control_first_activation": controls, "prediction_closure_counts": counts,
            "continuous_onset_in_sampling_interval_claimed": False}


def measure(context, derivation, audit):
    evidence, delta = context["evidence"], context["delta"]
    rows = np.stack([evidence["rows"][i] for i in IDS])
    lookup = {(r["control"], r["row_id"]): r for r in derivation["rows"]}
    prepared = {}
    with stage.hotspot.read_only(audit), no_replay(audit):
        for dose in DOSES:
            for c in parent.CONTROLS:
                prepared[dose, c] = prepare(evidence["archives"][c]["stage18"], delta, "reverse", dose)
                audit["new_operand_constructions"] += 1
                audit["operand_rne_scalars"] += 896
    tables, outputs = {str(d): [] for d in DOSES}, {str(d): [] for d in DOSES}
    with closed.suffix_only(audit, [v[0] for v in prepared.values()], evidence["weight_array"], rows), no_replay(audit):
        for (dose, c), (words, operand) in prepared.items():
            normalized, scalars = parent.rmsnorm(words, evidence["weight_array"])
            measured = parent.logits(normalized, rows)
            audit["exact_selected_row_accumulations"] += 2
            require(np.isfinite(normalized.view("<f2")).all() and np.isfinite(measured.view("<f2")).all(),
                    "nonfinite suffix result")
            outputs[str(dose)].append({"control": c, "polarity": "reverse", "operand": operand,
                "rmsnorm_words": normalized.tolist(), "rmsnorm_scalars": scalars,
                "selected_logit_words": measured.tolist()})
            for l, r in PAIRS:
                old_words = [int(evidence["arrays"][c]["logits"][i]) for i in (l, r)]
                new_words = [int(measured[IDS.index(i)]) for i in (l, r)]
                predicted_words = [lookup[c, i]["quartile_predictions"][str(dose)]["fp16_word"] for i in (l, r)]
                old, new, predicted = (crossing.decode(w[0]) - crossing.decode(w[1])
                                       for w in (old_words, new_words, predicted_words))
                for branch in BRANCHES:
                    ref = fixed_margin(evidence, branch, l, r)
                    tables[str(dose)].append({"control": c, "polarity": "reverse", "dose": str(dose),
                        "left_id": l, "right_id": r, "branch": branch, "baseline_words": old_words,
                        "measured_words": new_words, "predicted_words": predicted_words,
                        "baseline_margin": str(old), "measured_margin": str(new),
                        "predicted_delta": str(predicted - old), "observed_delta": str(new - old),
                        "fixed_reference_margin": str(ref), "retained_margin_change": str(old - ref),
                        "intervened_margin_change": str(new - ref), "prediction_agrees": new == predicted,
                        "individual_words_agree": new_words == predicted_words})
    return {**decide(tables), "dose_tables": tables, "suffix_outputs": outputs,
            "scalar_prediction_derivation": derivation, "reviewed_upper_endpoints": context["prior_endpoints"],
            "frozen_original_terminal_vector": list(map(str, delta)),
            "frozen_original_terminal_vector_identity": closed.digest(delta)}


def experiment(audit):
    with stage.hotspot.read_only(audit), no_replay(audit):
        context = authenticate()
        derivation = derive(context)
    frozen = closed.digest((context, derivation, PREREGISTRATION, FLAGS))
    report = measure(context, derivation, audit)
    same(closed.digest((context, derivation, PREREGISTRATION, FLAGS)), frozen, "frozen evidence mutation")
    same(audit, expected_audit(), "suffix-only dispatch/write census changed")
    return context, report


def run_tests(context, report):
    import pytest

    class Results:
        collected = executed = errors = failures = skipped = 0

        def pytest_configure(self, config):
            config._ace3_reverse_quartile_context = (context, report)

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
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE), "quartile_tests": parent.record(TEST)}
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
    same(audit, expected_audit(), "final dispatch/write census changed")
    retained = context["prior_receipt"]["native_result"]
    preserved = {k: retained[k] for k in (
        "retained_dose_summaries", "closed_crossing_classification", "closed_bracket_classification",
        "closed_localizer_classification", "protected_input_identity", "selected_head_row_identity",
        "authenticated_files", "assets", "final_reference_authority", "retained_controls_and_failure_gates",
        "retained_thresholds", "retained_common_component", "counterfactual_common_component")}
    return {**preserved, "diagnostic_id": NAME, "version": 1, "command": COMMAND,
        "status": report["status"], "classification": report["classification"], "successor": report["successor"],
        "dose_search_closed": report["dose_search_closed"],
        "global_threshold_model_closed": report["global_threshold_model_closed"],
        "report": report, "tests": tests, "preregistration": PREREGISTRATION, "flags": FLAGS,
        "authenticated_pins": pins, "reviewed_dyadic_receipt": RECEIPT,
        "prior_result_bindings": {k: retained[k] for k in (
            "diagnostic_id", "status", "classification", "successor", "reviewed_localizer_receipt",
            "reviewed_bracket_receipt", "reviewed_crossing_receipt", "dispatch_and_write_audit", "tests")},
        "preserved_dyadic_rejection_counts": retained["report"]["prediction_closure_counts"],
        "dispatch_and_write_audit": audit, "normal_host_review": "REQUIRED",
        "mutation_guards": {"protected_inputs_unchanged": True, "retained_receipts_unchanged": True,
                            "report_unchanged": True, "source_pins_reauthenticated": True},
        "claim_boundary": PREREGISTRATION["boundary"]}


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
                          "dose_search_closed": True,
                          "dispatch_and_write_audit": audit, "preregistration": PREREGISTRATION,
                          "flags": FLAGS, "normal_host_review": "REQUIRED"}, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
