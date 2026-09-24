"""Classify retained dose-response boundaries without constructing or executing a suffix."""

import argparse
from contextlib import redirect_stdout
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_row319_retained_byte_family_contrast_v1 as retained


ROOT, PYTHON = retained.ROOT, retained.PYTHON
NAME = "diagnose_q24_s16_terminal_remainder_selected_head_row319_control_specific_rounding_boundary_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = retained.COMMAND.replace(retained.NAME, NAME)
SEAL = {
    "path": str(ROOT / "build/diagnostics/row319-retained-byte-family-contrast/412ec79f743b-round1-attempt001/capture-bindings.json"),
    "sha256": "4ef977631a8f93ff9efa164301dda163c1a8ec2267dfa61bcbe9139451b72c7b",
    "bytes": 9171,
}
REVIEW = {
    "path": str(retained.HANDOFFS / "412ec79f743b/round-0001.json"),
    "sha256": "ad0b72e8f5a983a7a3cb1c0f39031f946cb48cce4cf2ea8d830196356ac9f3c9",
    "bytes": 690,
}
CONTROLS, DOSES, ROWS = retained.CONTROLS, retained.DOSES, retained.ROWS
OPERAND = "pre-RMSNorm operand FP16 words"
RMSNORM = "RMSNorm scalar/output rounding"
HEAD = "selected-head accumulator/RNE"
MIXED = "mixed/inconsistent fields"
UNKNOWN = retained.UNKNOWN
FIELDS = ("operand_delta", "rmsnorm_scalar_delta", "rmsnorm_output_delta", "head_rne_delta")
SUCCESSORS = {
    OPERAND: "preregister_operand_rounding_cell_question",
    RMSNORM: "preregister_rmsnorm_scalar_vs_output_question",
    HEAD: "preregister_selected_head_rne_cell_question",
    MIXED: "preregister_control_family_consistency_question",
    UNKNOWN: None,
}
PREREGISTRATION = {
    "inputs": retained.PREREGISTRATION["inputs"],
    "authority": "Reviewed 412ec79f743b and its reviewed b2e10d85b210/e762f68f110e parents.",
    "rule": "Compare exact retained after-minus-baseline numeric responses, not absolute "
            "profiles or changed-coordinate lists. At the earliest nonuniform stage, the "
            "same field must separate mapped_all alone at ALL FOUR fixed doses to localize "
            "the descriptive discriminator; any partial/shared/rotating split is mixed. "
            "Order: operand FP16 deltas; RMSNorm mean/root or output deltas; individual "
            "selected-head RNE residual deltas. Equal upstream responses do not imply "
            "equal absolute inputs. Downstream differences cannot erase an earlier split.",
    "head_prerequisite": "Both baseline and dose must retain individual exact_accumulator "
                         "and post_hidden_rne_accumulator rational scalars in ROWS order "
                         "under selected_head_accumulators. Verify post-hidden scalars "
                         "lie in their retained logit RNE cells. Never derive individual "
                         "scalars from margins, selected logits, or a new dot product.",
    "missing_fields": "Absent head scalars make that stage unavailable, not equal. A valid "
                      "earlier descriptive split can still be reported with this explicit "
                      "limitation. If localization reaches an unavailable stage, select "
                      "UNKNOWN/integrity with no scientific successor. Malformed supplied "
                      "fields or failed authentication always select UNKNOWN/integrity.",
    "interpretation": "Earliest retained response separation is not rounding-only causal "
                      "attribution, a dominant stage, actual-suffix threshold proof, or "
                      "repair. Scalar floor-root cell slacks are observations, not RMSNorm "
                      "recomputation. Selected-logit activation is an outcome, not a feature.",
    "successors": SUCCESSORS,
    "closure": retained.PREREGISTRATION["closure"],
    "boundary": retained.PREREGISTRATION["boundary"],
}
AUDIT_KEYS = retained.AUDIT_KEYS + ("dose_subdivisions", "coordinate_reselections",
                                  "production_repairs", "precision_expansions", "scale_expansions")
EXPECTED_TESTS = 18
require, IntegrityError = retained.require, retained.IntegrityError


def sealed_stdout(seal, bank):
    raw = retained.capture_bytes(seal, bank)
    streams = {name: bank.read(seal["artifacts"][name]["destination"])
               for name in ("stdout", "stderr", "whole_capture")}
    positions = {"stdout": 0, "stderr": 0}
    offset = 0
    require(bool(seal["stream_segments"]), "missing whole-capture stream segments")
    for segment in seal["stream_segments"]:
        name, size = segment["stream"], segment["size_bytes"]
        require(name in positions and type(size) is int and size > 0,
                "invalid capture segment")
        require(type(segment["stream_offset"]) is int
                and type(segment["whole_offset"]) is int
                and segment["stream_offset"] == positions[name]
                and segment["whole_offset"] == offset, "noncontiguous capture segment")
        chunk = streams[name][positions[name]:positions[name] + size]
        require(len(chunk) == size and streams["whole_capture"][offset:offset + size] == chunk,
                "whole-capture segment/stream mismatch")
        positions[name] += size
        offset += size
    require(offset == len(streams["whole_capture"])
            and all(positions[k] == len(streams[k]) for k in positions),
            "incomplete whole-capture segmentation")
    return raw


def authenticate(bank):
    retained.review_gate(retained.document(bank.read(REVIEW)), "412ec79f743b", 1)
    seal = retained.document(bank.read(SEAL))
    require((seal["task_id"], seal["attempt_id"], seal["account_uid"], seal["working_directory"]) ==
            ("412ec79f743b", "412ec79f743b-round1-attempt001", 1000, str(ROOT)),
            "family task/attempt/account/worktree mismatch")
    require(seal["command"] == retained.COMMAND
            and seal["argv"] == [PYTHON, "-B", "-m", retained.MODULE, "--check"],
            "family command mismatch")
    family = retained.document(sealed_stdout(seal, bank))
    require((family["diagnostic_id"], family["version"], family["command"], family["status"],
             family["classification"]) ==
            (retained.NAME, 1, retained.COMMAND, "measured", retained.UNIQUE),
            "reviewed family identity/classification mismatch")
    retained.tests_gate(family, retained.EXPECTED_TESTS)
    require(seal["source_test_pins_before"] == seal["source_test_pins_after"],
            "family source/test changed during capture")
    compiled = family["tests"]["compiled"]
    require([p["path"] for p in compiled] == [str(retained.SOURCE), str(retained.TEST)]
            and family["source_test_pins"] == compiled, "family compiled paths mismatch")
    require(len(seal["source_test_pins_before"]) == 2, "family source/test census mismatch")
    for before, pin in zip(seal["source_test_pins_before"], compiled, strict=True):
        require((before["path"], before["sha256"], before["size_bytes"]) ==
                (pin["path"], pin["sha256"], pin["bytes"]), "family source/test pin mismatch")
        bank.read(pin)
    for pin in (*family["authenticated_pins"], *family["authenticated_files"]):
        bank.read(pin)
    require(family["dispatch_and_write_audit"] == dict.fromkeys(retained.AUDIT_KEYS, 0),
            "family forbidden operation census changed")
    require(family["dose_search_closed"] is True and family["global_threshold_model_closed"] is True
            and family["flags"] == {
                "candidate_admitted": False, "dose_search_closed": True,
                "global_threshold_model_closed": True, "retained_bytes_only": True,
                "rounding_causal_attribution": False, "scalar_threshold_transfer_claim": False,
                "stdout_only": True}, "family closure/non-admission gate changed")
    # Authentication helpers only: never call any parent's analyze/check/compare or suffix.
    q, p, b2_seal = retained.authenticate(bank)
    require(family["b2_capture_bindings"] == b2_seal, "family/b2 capture mismatch")
    for key in retained.PRESERVED:
        require(family[key] == q[key], "family/parent retained binding mismatch: " + key)
    require(family["historical_flags"] == {"b2": q["flags"], "dyadic": p["flags"]}
            and family["preserved_dyadic_rejection_counts"] == q["preserved_dyadic_rejection_counts"]
            and family["prior_attempts"] == b2_seal["prior_attempts"],
            "historical failure/rejection binding changed")
    report = family["report"]
    require(set(report["retained_observations"]) == set(DOSES)
            and set(report["retained_baselines"]) == set(CONTROLS), "retained dose/control census")
    for dose in DOSES:
        parent = p if dose == "17/512" else q
        require(report["retained_observations"][dose] ==
                [o for o in parent["report"]["suffix_outputs"][dose] if o["polarity"] == "reverse"],
                "family/parent suffix byte mismatch: " + dose)
        require(report["retained_reference_closures"][dose] ==
                [r for r in parent["report"]["dose_tables"][dose] if r["polarity"] == "reverse"],
                "family/parent reference closure mismatch: " + dose)
    for control in q["retained_controls_and_failure_gates"]:
        baseline = report["retained_baselines"][control["control"]]
        for field, pin, shape, member in (
            ("working_fp16_words", control["parent"]["terminal_archive"], (896,), "stage18"),
            ("rmsnorm_words", control["arrays"]["rmsnorm"], (896,), None),
            ("selected_logit_words", control["arrays"]["logits"], (151936,), None),
        ):
            array = retained.array(bank, pin, "<u2", shape, member)
            values = [int(array[i]) for i in ROWS] if field == "selected_logit_words" else array.tolist()
            require(baseline[field] == values, "retained baseline byte mismatch: " + field)
        require(baseline["rmsnorm_scalars"] == control["rmsnorm_integer_details"],
                "retained baseline scalar mismatch")
    return family, seal


def word_delta(before, after, length):
    retained.words(before, length)
    retained.words(after, length)
    return [str(retained.decode(b) - retained.decode(a))
            for a, b in zip(before, after, strict=True)]


def root_cell(scalars):
    require(set(scalars) == {"mean_q48", "root_q24"}
            and all(type(v) is int and v > 0 for v in scalars.values()),
            "missing/invalid RMSNorm integer scalars")
    mean, root = scalars["mean_q48"], scalars["root_q24"]
    lower, upper = mean - root * root, (root + 1) * (root + 1) - mean
    require(lower >= 0 and upper > 0, "retained RMSNorm floor-root cell mismatch")
    return {"lower_squared_slack_q48": lower, "upper_squared_slack_q48": upper}


def rne_residual(word, accumulator):
    value = Fraction(accumulator)
    magnitude = word & 0x7fff
    retained.words([word], 1)
    require(0 < magnitude < 0x7bff, "head RNE cell requires finite nonzero interior word")
    center = retained.decode(word)
    lower = retained.decode(word + 1 if word & 0x8000 else word - 1)
    upper = retained.decode(word - 1 if word & 0x8000 else word + 1)
    low_mid, high_mid = (lower + center) / 2, (upper + center) / 2
    require(low_mid < value < high_mid
            or (word & 1 == 0 and value in (low_mid, high_mid)),
            "retained selected-head accumulator/logit RNE cell mismatch")
    return center - value


def head_response(before, after):
    old = before.get("selected_head_accumulators")
    new = after.get("selected_head_accumulators")
    if old is None and new is None:
        return None
    require(old is not None and new is not None, "partial selected-head accumulator binding")
    responses = []
    for record in (old, new):
        require(set(record) == {"exact_accumulator", "post_hidden_rne_accumulator"}
                and all(isinstance(v, list) and len(v) == 2
                        and all(isinstance(x, str) for x in v) for v in record.values()),
                "missing individual selected-head accumulator scalars")
    for i in range(2):
        residuals, hidden_rounding = [], []
        for observation, record in ((before, old), (after, new)):
            exact = Fraction(record["exact_accumulator"][i])
            post = Fraction(record["post_hidden_rne_accumulator"][i])
            residuals.append(rne_residual(observation["selected_logit_words"][i], post))
            hidden_rounding.append(post - exact)
        responses.append({"row": ROWS[i],
                          "hidden_rne_effect_delta": str(hidden_rounding[1] - hidden_rounding[0]),
                          "head_rne_residual_delta": str(residuals[1] - residuals[0])})
    return responses


def classify(profiles):
    require(set(profiles) == set(DOSES), "response dose census changed")
    groups, availability = {}, {}
    for field in FIELDS:
        groups[field] = {}
        present = []
        for dose in DOSES:
            require(set(profiles[dose]) == set(CONTROLS), "response control census changed")
            buckets = {}
            for control in CONTROLS:
                value = profiles[dose][control][field]
                present.append(value is not None)
                if value is not None:
                    buckets.setdefault(retained.encoded(value), []).append(control)
            groups[field][dose] = list(buckets.values())
        require(all(present) or not any(present), "partial response field: " + field)
        availability[field] = all(present)
    classification, reason, features = UNKNOWN, "No separating retained response field.", []
    for label, fields in ((OPERAND, FIELDS[:1]), (RMSNORM, FIELDS[1:3]), (HEAD, FIELDS[3:])):
        if not all(availability[f] for f in fields):
            reason = "Missing retained response prerequisite: " + ", ".join(
                f for f in fields if not availability[f])
            break
        split = any(len(groups[f][dose]) > 1 for f in fields for dose in DOSES)
        if not split:
            continue
        features = [f for f in fields
                    if all(["mapped_all"] in groups[f][dose] for dose in DOSES)]
        classification = label if features else MIXED
        reason = ("Earliest stable mapped_all dose-response separator; descriptive only."
                  if features else "Earliest response split is shared, partial, or inconsistent by field/dose.")
        break
    return {"classification": classification, "reason": reason,
            "localizing_fields": features, "field_partitions": groups,
            "field_availability": availability, "successor": SUCCESSORS[classification],
            "successor_flags": {flag: SUCCESSORS[classification] == flag
                                for flag in SUCCESSORS.values() if flag},
            "integrity_error": reason if classification == UNKNOWN else None}


def compare(report):
    require(set(report["retained_observations"]) == set(DOSES)
            and set(report["retained_baselines"]) == set(CONTROLS), "retained census changed")
    profiles, observations, missing = {}, [], []
    for dose in DOSES:
        outputs = report["retained_observations"][dose]
        require([o["control"] for o in outputs] == list(CONTROLS), "retained output census changed")
        profiles[dose] = {}
        for output in outputs:
            control, operand = output["control"], output["operand"]
            before = report["retained_baselines"][control]
            require(output["polarity"] == "reverse" and operand["dose"] == operand["signed_dose"] == dose
                    and operand["coordinate_order"] == "all_896_input_order_including_zeros",
                    "retained dose/polarity/coordinate identity mismatch")
            working = operand["working_fp16_words"]
            delta = word_delta(before["working_fp16_words"], working, 896)
            require(operand["changed_coordinates"] == [
                i for i, (a, b) in enumerate(zip(before["working_fp16_words"], working, strict=True)) if a != b],
                "retained operand changed-coordinate closure")
            old_cell, new_cell = root_cell(before["rmsnorm_scalars"]), root_cell(output["rmsnorm_scalars"])
            normalized = word_delta(before["rmsnorm_words"], output["rmsnorm_words"], 896)
            head = head_response(before, output)
            if head is None:
                missing.append({"dose": dose, "control": control, "rows": list(ROWS),
                                "field": "baseline/dose.selected_head_accumulators",
                                "prerequisite": PREREGISTRATION["head_prerequisite"]})
            profiles[dose][control] = {
                "operand_delta": delta,
                "rmsnorm_scalar_delta": {k: output["rmsnorm_scalars"][k] - before["rmsnorm_scalars"][k]
                                        for k in ("mean_q48", "root_q24")},
                "rmsnorm_output_delta": normalized, "head_rne_delta": head}
            logits = word_delta(before["selected_logit_words"], output["selected_logit_words"], 2)
            activity = [Fraction(v) != 0 for v in logits]
            require(activity[0] == (control == "mapped_all" or dose == "17/512"),
                    "reviewed row319 early-activation outcome mismatch")
            observations.append({"dose": dose, "control": control, "rows": list(ROWS),
                                 "selected_logit_delta": logits, "selected_row_activity": activity,
                                 "baseline_root_cell": old_cell, "dose_root_cell": new_cell})
    return {**classify(profiles), "response_profiles": profiles, "observations": observations,
            "observation_count": len(observations), "selected_row_observation_count": 2 * len(observations),
            "missing_bindings": missing, "absolute_baselines_are_not_discriminator_features": True}


def base_result():
    return {"diagnostic_id": NAME, "version": 1, "command": COMMAND,
            "dose_search_closed": True, "global_threshold_model_closed": True,
            "normal_host_review": "REQUIRED", "preregistration": PREREGISTRATION,
            "claim_boundary": PREREGISTRATION["boundary"],
            "flags": {"stdout_only": True, "retained_bytes_only": True,
                      "candidate_admitted": False, "rounding_causal_attribution": False,
                      "scalar_threshold_transfer_claim": False, "actual_suffix_threshold_proof": False,
                      "strict_fp16_state_w4a16": False, "new_token_admitted": False,
                      "full_model_admitted": False, "production_repair": False,
                      "dose_search_closed": True, "global_threshold_model_closed": True}}


def unknown(result, error):
    result.update(status="UNKNOWN", classification=UNKNOWN, successor=None,
                  successor_flags={flag: False for flag in SUCCESSORS.values() if flag},
                  integrity_error=f"{type(error).__name__}: {error}")
    return result


def analyze(bank):
    result = base_result()
    try:
        family, seal = authenticate(bank)
        result.update({key: family[key] for key in retained.PRESERVED})
        result.update(family_capture_bindings=seal, parent_classification=family["classification"],
                      historical_flags=family["historical_flags"], prior_attempts=family["prior_attempts"],
                      preserved_dyadic_rejection_counts=family["preserved_dyadic_rejection_counts"],
                      retained_family_report=family["report"])
        report = compare(family["report"])
        result.update(report=report, classification=report["classification"], successor=report["successor"],
                      successor_flags=report["successor_flags"], integrity_error=report["integrity_error"],
                      status="UNKNOWN" if report["classification"] == UNKNOWN else "measured")
    except (IntegrityError, OSError, LookupError, TypeError, ValueError, ArithmeticError) as error:
        unknown(result, error)
    result["authenticated_pins"] = list(bank.pins.values())
    return result


def run_tests(result):
    import pytest

    class Results:
        collected = executed = errors = failures = skipped = 0

        def pytest_configure(self, config):
            config._ace3_rounding_boundary_result = result

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
            "failures": results.failures, "skipped": results.skipped}


def check(audit):
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT) and sys.dont_write_bytecode
            and not sys.flags.optimize, "isolated workdir/account/interpreter/source gate failed")
    pins = []
    for path in (SOURCE, TEST):
        raw = path.read_bytes()
        compile(raw, str(path), "exec", dont_inherit=True)
        pins.append({"path": str(path), "bytes": len(raw), "sha256": retained.digest(raw)})
    with retained.read_only(audit):
        result = analyze(retained.BoundBytes())
        frozen = retained.encoded(result)
        tests = run_tests(result)
        require(retained.encoded(result) == frozen, "tests mutated authenticated evidence")
        for pin in (*pins, *result["authenticated_pins"]):
            require(retained.digest(Path(pin["path"]).read_bytes()) == pin["sha256"],
                    "authenticated source/evidence changed during check: " + pin["path"])
    require(audit == dict.fromkeys(AUDIT_KEYS, 0), "forbidden dispatch/write census changed")
    tests["compiled"] = pins
    result.update(tests=tests, source_test_pins=pins, dispatch_and_write_audit=audit)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", required=True, action="store_true")
    parser.parse_args(argv)
    audit = dict.fromkeys(AUDIT_KEYS, 0)
    try:
        result = check(audit)
    except (IntegrityError, OSError, ValueError, ImportError, SyntaxError) as error:
        result = unknown(base_result(), error)
        result["dispatch_and_write_audit"] = audit
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.modules[MODULE] = sys.modules[__name__]
    sys.exit(main())
