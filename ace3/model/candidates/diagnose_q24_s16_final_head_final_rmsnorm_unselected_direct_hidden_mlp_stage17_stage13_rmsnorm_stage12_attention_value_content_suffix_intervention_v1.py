"""One preregistered middle-pair value intervention; stdout-only, never admission."""

import argparse
import builtins
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import hashlib
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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_score_value_contrast_bridge_v1 as previous
from ace3.model.candidates import q24_s16_l23_p0_value_content_sufficiency_v1 as suffix


base, parent, source = previous.base, previous.parent, previous.source
layer, native = suffix.layer, suffix.native
require, same = base.require, base.same
ROOT = previous.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_value_content_suffix_intervention_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
CONTROLS, PAIR = parent.CONTROLS, (34319, 13)
REVIEW_DIR = layer.preflight.HANDOFFS / "773f3ee8859c"
PARENT_PINS = (
    {"path": str(previous.SOURCE), "bytes": 18179,
     "sha256": "51a3bd13f8aadaca651d7b7bbaf6707821ad93c74754850175842c1d282ba745"},
    {"path": str(previous.TEST), "bytes": 22859,
     "sha256": "4be2a8dd40b4d5b4bd5700699d7080574b05f007285581b54976e08cd4c3fa43"},
    {"path": str(REVIEW_DIR / "round-0001.json"), "bytes": 690,
     "sha256": "7c93b5a97728567689204495697819e6160f097edc15865dc7727252c6529511"},
)
SUFFIX_PINS = (
    {"path": str(suffix.SOURCE), "bytes": 20705,
     "sha256": "9477e86df6a8c40a24ec23197721b12c8aecc7bf556c8d04cb9695de5a3ed0ae"},
    {"path": str(suffix.TEST), "bytes": 4562,
     "sha256": "f32447270a3fa4aae9369b0b13270d4f3bd950dcab74906358d37216bea16155"},
)
EXPECTED_TESTS = 12
FLAGS = {**layer.FLAGS, "score_replay": 0, "softmax_replay": 0,
         "v_projection_replay": 0, "retained_evidence_writes": 0,
         "reference_producer_invocations": 0, "full_vocabulary_head_recomputation": False,
         "token_selection": False, "historical_failures_preserved": True,
         "original_global_reference_unchanged": True}
PREREGISTRATION = {
    "question": "Does replacing one retained P0 value component by its original-input "
                "FP16 operand change retained middle-pair MLP-stage17 maxima and final "
                "margin in the score/value bridge-predicted direction?",
    "selection": "For each of 128 value coordinates, sum its signed downstream "
                 "coordinate_linear_operand_contribution within each retained row. "
                 "Choose the coordinate with greatest sum of absolute row totals; "
                 "ties choose the lowest coordinate. No intervention outputs enter selection.",
    "replacement": "Only working stage07[k] becomes original-input FP16 stage07[k], "
                   "for the same single k in all nine controls. No stage03/cache update.",
    "prediction": "Each row's predicted delta is the negative selected-component "
                  "bridge total. Both the fixed-weight retained MLP maxima delta and "
                  "the actual FP16 pair-margin delta must have that strict nonzero sign.",
    "supported": "Authenticated execution, every unchanged S10-S18 gate PASS, and "
                 "both strict directional contrasts in all 18 control/reference rows.",
    "rejected": "Authenticated execution with any absent/opposite directional contrast "
                "or any existing numerical gate FAIL.",
    "UNKNOWN": "Authentication, source/operand/state/KV/lineage, zero prediction, "
               "nonfinite output, test, dispatch or execution-integrity defect.",
    "scope": "One intervention; no retuning, no prefix/admission/reference replay. "
             "Q24 state is wider than FP16; INT4 and FP16 operator/KV boundaries fixed. "
             "Neither outcome is admission, strict-FP16-state W4A16, new-token or "
             "full-model evidence. Stop this decomposition branch after classification.",
}


def authenticate_parent():
    for pin in (*PARENT_PINS, *SUFFIX_PINS):
        base.read_bound(pin)
    latest = json.loads((REVIEW_DIR / "latest.json").read_bytes())
    same(latest["kind"], "handoff_ref", "score/value parent is not reviewed")
    same(latest["handoff"]["path"], PARENT_PINS[-1]["path"], "latest review changed")
    review = json.loads(base.read_bound(PARENT_PINS[-1]))
    layer.preflight.parent.matrix.check_review(review, "773f3ee8859c", 1)
    previous.authenticate_parent()
    return parent.record(REVIEW_DIR / "latest.json")


def digest_report(report):
    digest = hashlib.sha256()
    for part in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(report):
        digest.update(part.encode())
    return digest.hexdigest()


def preregister(report):
    same((report["left_id"], report["right_id"]), PAIR, "pair changed")
    same(report["attention_score_value_summary"]["classification"],
         "SINGLETON_PROBABILITY_INVARIANT", "score/value contradiction")
    same(report["common_component"], "UNKNOWN", "historical common gate changed")
    same(report["stop_nested_bridge_expansion"], True, "historical stop gate changed")
    same([(r["control"], r["branch"]) for r in report["rows"]],
         [(c, b) for c in CONTROLS for b in ("fp16", "binary64")], "row census changed")
    for control in report["attention_score_value_controls"]:
        same(control["source_position"], 0, "source position changed")
        same(control["source_token_id"], 9707, "source token changed")
        for row in control["scores_by_absolute_delta"]:
            same((row["actual_probability"], row["reference_probability"],
                  row["probability_delta"]), ("1", "1", "0"), "singleton changed")
    totals = [[Fraction() for _ in range(128)] for _ in report["rows"]]
    for account in report["attention_score_value_accounts"]:
        require(all(v == "0" for v in account["closure_residuals"].values()),
                "downstream closure changed")
        core = report["attention_score_value_local_accounts"][account["local_account_key"]]
        multiplier = Fraction(account["multipliers"][previous.FIELDS[-1]])
        for item in core["value_component_contrasts"]:
            previous.singleton_identity(item["contrasts"])
            totals[account["row_index"]][item["value_coordinate"]] += (
                Fraction(item["contrasts"]["signed_contribution"])*multiplier)
    masses = [sum((abs(row[k]) for row in totals), Fraction()) for k in range(128)]
    coordinate = min(range(128), key=lambda k: (-masses[k], k))
    predictions = [-row[coordinate] for row in totals]
    require(all(predictions), "zero bridge prediction prevents directional classification")
    return {
        **PREREGISTRATION, "value_coordinate": coordinate, "source_position": 0,
        "source_token_id": 9707, "pair": list(PAIR),
        "coordinate_selection_masses": list(map(str, masses)),
        "predicted_row_deltas": list(map(str, predictions)),
    }


def prepare(actual, reference, coordinate):
    require(type(coordinate) is int and 0 <= coordinate < 128, "invalid value coordinate")
    state, arrays = suffix.prepare(actual, reference)
    arrays["stage07"] = actual["stage07"].copy()
    arrays["stage07"][coordinate] = reference["stage07"][coordinate]
    arrays["stage07"].flags.writeable = False
    for value in state.values():
        value.flags.writeable = False
    protected(actual, arrays, reference, coordinate)
    return state, arrays


def protected(actual, arrays, reference, coordinate):
    for key in suffix.PROTECTED:
        a, b = actual[key], arrays[key]
        require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                "protected state/KV/source/score changed: " + key)
    expected = actual["stage07"].copy()
    expected[coordinate] = reference["stage07"][coordinate]
    require(arrays["stage07"].dtype == expected.dtype
            and arrays["stage07"].shape == expected.shape
            and arrays["stage07"].tobytes() == expected.tobytes(),
            "exact replacement operand changed")


@contextmanager
def no_writes(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("suffix diagnostic forbids filesystem mutation")

    def wrap(original):
        def opening(path, mode="r", *args, **kwargs):
            if not isinstance(mode, str) or any(c in mode for c in "wax+"):
                return forbidden()
            return original(path, mode, *args, **kwargs)
        return opening

    original = os.open

    def os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            return forbidden()
        return original(path, flags, *args, **kwargs)

    with ExitStack() as stack:
        for module in (builtins, io):
            stack.enter_context(patch.object(module, "open", wrap(module.open)))
        stack.enter_context(patch.object(os, "open", os_open))
        for name in ("mkdir", "makedirs", "unlink", "remove", "rmdir", "rename",
                     "replace", "link", "symlink", "chmod", "truncate", "utime"):
            stack.enter_context(patch.object(os, name, forbidden))
        yield


@contextmanager
def suffix_only(audit, active, tensors, operands):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("predecessor, prefix/admission/reference or out-of-cone dispatch")

    raw_projection, raw_norm, raw_logits = native.projection, parent.rmsnorm, parent.logits
    raw_expected = layer.expected_stage
    projections = {11: ("self_attn.o_proj", 10), 14: ("mlp.gate_proj", 13),
                   15: ("mlp.up_proj", 13), 17: ("mlp.down_proj", 16)}

    def projection(actual_tensors, name, words):
        stage = active.get("stage")
        if stage not in projections:
            return forbidden()
        kind, input_stage = projections[stage]
        require(actual_tensors is tensors and name == "model.layers.23." + kind
                and words is active["arrays"][f"stage{input_stage:02d}"],
                "projection operand/scope changed")
        audit["projections"] += 1
        return raw_projection(actual_tensors, name, words)

    def expected(stage, arrays, state, actual_tensors):
        require(stage == active.get("stage") and 10 <= stage <= 18
                and arrays is active["arrays"] and actual_tensors is tensors
                and state is active["state"], "local oracle outside active suffix")
        audit["local_oracles"] += 1
        return raw_expected(stage, arrays, state, actual_tensors)

    def norm(words, weights):
        require(active.get("stage") == 19 and words is active["arrays"]["stage18"]
                and weights is operands[0], "final norm outside active suffix")
        audit["final_norms"] += 1
        return raw_norm(words, weights)

    def logits(words, weights):
        require(active.get("stage") == 20 and words is active["arrays"]["final_rmsnorm"]
                and weights is operands[1] and weights.shape == (2, 896),
                "only the two retained pair head rows may run")
        audit["pair_heads"] += 1
        return raw_logits(words, weights)

    with parent.no_dispatch(audit), no_writes(audit), ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates.") and module is not sys.modules[MODULE]:
                for attribute in ("check", "run_tests", "focused_tests", "collect",
                                  "measure", "validate", "run_control"):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        for module, name, function in (
            (native, "projection", projection), (layer, "expected_stage", expected),
            (parent, "rmsnorm", norm), (parent, "logits", logits),
            (parent, "top_k", forbidden),
        ):
            stack.enter_context(patch.object(module, name, function))
        yield


def exact(word):
    return Fraction.from_float(float(np.asarray(word, dtype="<u2").view("<f2")))


def contrast_rows(report, plan, evidence, outputs):
    rows = []
    for index, row in enumerate(report["rows"]):
        control, branch = row["control"], row["branch"]
        actual, changed = evidence["archives"][control], outputs[control]
        mlp = Fraction()
        coordinates = []
        for hotspot in row["hotspots"]:
            k = hotspot["output_coordinate"]
            factor = Fraction(hotspot["retained_weighted_row_factor"])
            delta = exact(changed["stage17"][k])-exact(actual["stage17"][k])
            mlp += factor*delta
            coordinates.append({"coordinate": k, "delta": str(delta),
                                "fixed_row_factor": str(factor), "weighted_delta": str(factor*delta)})
        baseline = evidence["arrays"][control]["logits"]
        old_margin = exact(baseline[PAIR[0]])-exact(baseline[PAIR[1]])
        new_margin = exact(changed["pair_logits"][0])-exact(changed["pair_logits"][1])
        ref = evidence["references"]["logits_"+branch]
        reference_margin = (exact(ref[PAIR[0]])-exact(ref[PAIR[1]]) if branch == "fp16"
                            else Fraction.from_float(float(ref[PAIR[0]]))
                            - Fraction.from_float(float(ref[PAIR[1]])))
        prediction = Fraction(plan["predicted_row_deltas"][index])
        delta = new_margin-old_margin
        rows.append({
            "control": control, "branch": branch, "predicted_delta": str(prediction),
            "mlp_maxima": coordinates, "mlp_maxima_delta": str(mlp),
            "mlp_prediction_remainder": str(mlp-prediction),
            "retained_margin": str(old_margin), "intervened_margin": str(new_margin),
            "independent_original_reference_margin": str(reference_margin),
            "retained_margin_error": str(old_margin-reference_margin),
            "intervened_margin_error": str(new_margin-reference_margin),
            "margin_delta": str(delta), "margin_prediction_remainder": str(delta-prediction),
            "direction_observed": prediction*mlp > 0 and prediction*delta > 0,
        })
    return rows


def classify(rows, reports):
    same(len(rows), 18, "incomplete directional census")
    same(list(reports), list(CONTROLS), "incomplete control schedule")
    for control in CONTROLS:
        same([r["stage"] for r in reports[control]], list(range(10, 19)),
             "incomplete suffix gate census")
        require(all(r["status"] in ("PASS", "FAIL") for r in reports[control]),
                "invalid numerical gate result")
    for row in rows:
        p, m, d = map(Fraction, (row["predicted_delta"], row["mlp_maxima_delta"], row["margin_delta"]))
        require(p != 0, "zero prediction")
        same(row["direction_observed"], p*m > 0 and p*d > 0, "direction result changed")
    return ("supported" if all(row["direction_observed"] for row in rows)
            and all(r["status"] == "PASS" for rs in reports.values() for r in rs) else "rejected")


def run_tests(evidence):
    spec = importlib.util.spec_from_file_location(NAME + "_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "focused tests failed, errored or skipped")
    return {"executed": outcome.testsRun, "failures": len(outcome.failures),
            "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT) and sys.dont_write_bytecode
            and not sys.flags.optimize, "isolated source/account/interpreter gate failed")
    native.torch.set_num_threads(1)
    require(str(native.torch.tensor(0).device) == "cpu", "non-CPU default device")
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               NAME + "_tests": parent.record(TEST)}
    audit = {"forbidden_calls": 0, "stages": [], "projections": 0, "local_oracles": 0,
             "final_norms": 0, "pair_heads": 0}
    began = time.monotonic()
    with no_writes(audit):
        review_pin = authenticate_parent()
        inputs = previous.collect(audit)
        with previous.read_only(audit):
            retained = previous.report(*inputs)
            plan = preregister(retained)
            parent_digest = digest_report(retained)
        attention_inputs = inputs[0][0][0][0]
        e = source.evidence(attention_inputs[0])
        summary = e["result"]["preflight"]
        tensors, trajectory, binary64, checkpoint = layer.load_inputs({"summary": summary})
        same(parent.preflight.bind_assets(checkpoint), summary["assets"], "asset binding changed")
        full_operands = parent.load_operands(summary)
        operands = (full_operands[0], full_operands[1][list(PAIR)].copy())
        for value in (*tensors.values(), *operands):
            value.flags.writeable = False
        authentication_seconds = time.monotonic()-began
        outputs, reports, timings = {}, {}, {}
        active = {}
        with suffix_only(audit, active, tensors, operands):
            for control in CONTROLS:
                actual = e["archives"][control]
                state, arrays = prepare(actual, trajectory, plan["value_coordinate"])
                active.update(arrays=arrays, state=state)
                generator = suffix.suffix_stages(tensors, state, arrays)
                reports[control] = []
                start = time.monotonic()
                for stage in range(10, 19):
                    active["stage"] = stage
                    audit["stages"].append([control, 23, 0, stage])
                    same(next(generator), stage, "out-of-cone stage")
                    oracle = layer.expected_stage(stage, arrays, state, tensors)
                    reports[control].append(layer.stage_report(stage, arrays, trajectory, binary64, oracle))
                    if oracle is not None:
                        arrays[f"local_reference_stage{stage:02d}"] = oracle
                require(next(generator, None) is None, "extra suffix stage")
                protected(actual, arrays, trajectory, plan["value_coordinate"])
                require(np.array_equal(layer.prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                                       arrays["stage16"]), "S16 RTZ mismatch")
                stage_end = time.monotonic()
                active["stage"] = 19
                arrays["final_rmsnorm"], _ = parent.rmsnorm(arrays["stage18"], operands[0])
                active["stage"] = 20
                arrays["pair_logits"] = parent.logits(arrays["final_rmsnorm"], operands[1])
                outputs[control] = arrays
                timings[control] = {"suffix_and_local_oracles": stage_end-start,
                                    "final_norm_and_pair_head": time.monotonic()-stage_end}
        rows = contrast_rows(retained, plan, e, outputs)
        status = classify(rows, reports)
        tests = run_tests((inputs, retained, plan, e, outputs, rows, reports, tensors, operands))
        same(audit["stages"], [[c, 23, 0, s] for c in CONTROLS for s in range(10, 19)],
             "dispatch census changed")
        same([audit[k] for k in ("forbidden_calls", "projections", "local_oracles",
                                "final_norms", "pair_heads")], [0, 36, 81, 9, 9],
             "suffix arithmetic/write census changed")
        authenticate_parent()
        for pin in (*origins.values(), review_pin, *e["files"], *e["hidden_pins"]):
            base.read_bound(pin)
    return {
        "diagnostic_id": NAME, "version": 1, "status": status, "command": COMMAND,
        "preregistration": plan, "parent_source_test_review_pins": PARENT_PINS,
        "suffix_helper_pins": SUFFIX_PINS,
        "parent_report_sha256": parent_digest,
        "parent_output_binding": "Fresh pure report from pinned reviewed source and authenticated "
                                 "retained inputs; no saved 773 stdout artifact exists.",
        "diagnostic_sources": origins, "latest_parent_review": review_pin,
        "authenticated_files": e["files"], "hidden_pins": e["hidden_pins"],
        "assets": e["assets"],
        "flags": FLAGS, "audit": audit, "tests": {"compiled": compiled, **tests},
        "original_thresholds": summary["thresholds"],
        "L23_reference_authority": summary["L23_original_reference"],
        "final_reference_authority": summary["final_reference"],
        "historical_controls": e["result"]["controls"],
        "rows": rows, "stage_reports": reports, "normal_host_review": "REQUIRED",
        "parent_common_component": retained["common_component"],
        "parent_stop_nested_bridge_expansion": retained["stop_nested_bridge_expansion"],
        "authentication_seconds": authentication_seconds, "timing_seconds": timings,
        "outputs": {c: {k: v.tolist() for k, v in arrays.items()} for c, arrays in outputs.items()},
        "operands": {c: {"retained_value_words": e["archives"][c]["stage07"].tolist(),
                        "original_reference_value_words": trajectory["stage07"].tolist()}
                     for c in CONTROLS},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    try:
        result = check()
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, AssertionError) as error:
        print(json.dumps({"diagnostic_id": NAME, "status": "UNKNOWN", "flags": FLAGS,
                          "error_type": type(error).__name__, "error": str(error),
                          "normal_host_review": "REQUIRED"}, sort_keys=True, allow_nan=False))
        raise
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    from importlib import import_module

    import_module(MODULE).main()
