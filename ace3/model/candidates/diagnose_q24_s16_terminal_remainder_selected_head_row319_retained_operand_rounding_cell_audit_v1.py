"""Audit sealed operand words and rounding-cell availability without rerounding."""

import argparse
from contextlib import redirect_stdout
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_row319_control_specific_rounding_boundary_classifier_v1 as boundary


retained = boundary.retained
ROOT, PYTHON = retained.ROOT, retained.PYTHON
NAME = "diagnose_q24_s16_terminal_remainder_selected_head_row319_retained_operand_rounding_cell_audit_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = retained.COMMAND.replace(retained.NAME, NAME)
SEAL = {
    "path": str(ROOT / "build/diagnostics/row319-control-specific-rounding-boundary/4b02378f0989-round1-attempt001/capture-bindings.json"),
    "sha256": "c84b381836ffe33715de1e7eb22ea67ff7e7a8c91feeb4f1df2603017d0b42d9",
    "bytes": 11121,
}
REVIEW = {
    "path": str(retained.HANDOFFS / "4b02378f0989/round-0001.json"),
    "sha256": "b18f2f07aa6a61574fa7d69a3ce99e4de7a45acfed01a1000afc05a34b5a7e37",
    "bytes": 690,
}
CONTROLS, DOSES, ROWS = retained.CONTROLS, retained.DOSES, retained.ROWS
SUPPORTED = "supported retained operand cell signature"
WORDS_ONLY = "word-move-only/no cell binding"
MIXED = "mixed/inconsistent controls"
UNKNOWN = retained.UNKNOWN
SUCCESSORS = {
    SUPPORTED: "preregister_retained_cell_signature_vs_selected_row_outcome",
    WORDS_ONLY: "preregister_existing_unrounded_operand_provenance_availability",
    MIXED: "preregister_fixed_control_family_word_vs_cell_consistency",
    UNKNOWN: None,
}
PREREGISTRATION = {
    "authority": "Reviewed 4b02378f0989, 412ec79f743b, b2e10d85b210 and e762f68f110e.",
    "inputs": retained.PREREGISTRATION["inputs"],
    "rule": "Audit all 896 coordinates in input order, for all nine controls at the four "
            "fixed doses. Exact after-minus-baseline word values must separate mapped_all "
            "alone at every dose. Otherwise classify mixed/inconsistent controls. Complete "
            "authenticated unrounded baseline/dose bindings whose cell-response signature "
            "also separates mapped_all alone at every dose support a retained operand cell "
            "signature; inconsistent complete cell signatures select mixed. If all unrounded "
            "bindings are absent, report word-move-only/no cell binding, not a causal success. "
            "Partial, malformed or out-of-cell supplied bindings select UNKNOWN/integrity.",
    "cell_binding": "Decode finite IEEE binary16 word geometry exactly as rationals, including "
                    "sign, significand parity, numeric neighbors, midpoint endpoints and RNE "
                    "tie ownership. This is not proof of an actual rounding event. The optional "
                    "unrounded_operand_values field, when retained on BOTH baseline and dose "
                    "operand records, must be 896 canonical rational strings in input order. "
                    "Check membership only; never form a corrected operand or reround it. "
                    "Signed-zero unrounded values need additional sign authority and remain "
                    "UNKNOWN with this rational-only schema. Other supplied unrounded/cell/tie "
                    "fields without this recognized binding are an explicit integrity gap.",
    "signature": "For every input coordinate, use baseline/dose word sign and parity, numeric "
                 "neighbor step, actual retained lower/interior/upper tie labels, and the "
                 "change in exact word-minus-unrounded residual. No coordinate selection, "
                 "ranking, logits, reference values or control labels are feature inputs.",
    "successors": SUCCESSORS,
    "successor_boundaries": {
        SUPPORTED: "A separately reviewed retained-cell/outcome association question, not "
                   "rounding-only causality or execution authority.",
        WORDS_ONLY: "A distinct provenance availability question for already existing sealed "
                    "unrounded operand bytes. No reconstruction, producer replay or new dose; "
                    "if absent, report the missing prerequisite, not a synthetic binding.",
        MIXED: "A separately preregistered fixed-family consistency question using retained "
               "word/cell accounts only, not another global threshold or dose search.",
        UNKNOWN: "No scientific successor until the exact integrity gap is supplied and reviewed.",
    },
    "closure": retained.PREREGISTRATION["closure"],
    "boundary": retained.PREREGISTRATION["boundary"],
}
AUDIT_KEYS = boundary.AUDIT_KEYS + (
    "suffix_constructions", "suffix_reroundings", "bisections", "coordinate_interventions",
    "hardware_dispatch", "ACE2_modifications",
)
EXPECTED_TESTS = 16
require, IntegrityError = retained.require, retained.IntegrityError


def authenticate(bank):
    retained.review_gate(retained.document(bank.read(REVIEW)), "4b02378f0989", 1)
    seal = retained.document(bank.read(SEAL))
    require((seal["task_id"], seal["attempt_id"], seal["account_uid"], seal["working_directory"]) ==
            ("4b02378f0989", "4b02378f0989-round1-attempt001", 1000, str(ROOT)),
            "boundary task/attempt/account/worktree mismatch")
    require(seal["command"] == boundary.COMMAND
            and seal["argv"] == [PYTHON, "-B", "-m", boundary.MODULE, "--check"],
            "boundary command mismatch")
    parent = retained.document(boundary.sealed_stdout(seal, bank))
    require((parent["diagnostic_id"], parent["version"], parent["command"], parent["status"],
             parent["classification"]) ==
            (boundary.NAME, 1, boundary.COMMAND, "measured", boundary.OPERAND),
            "reviewed boundary identity/classification mismatch")
    retained.tests_gate(parent, boundary.EXPECTED_TESTS)
    require(seal["source_test_pins_before"] == seal["source_test_pins_after"],
            "boundary source/test changed during capture")
    pins = parent["tests"]["compiled"]
    require([p["path"] for p in pins] == [str(boundary.SOURCE), str(boundary.TEST)]
            and parent["source_test_pins"] == pins
            and len(seal["source_test_pins_before"]) == 2, "boundary compiled pins mismatch")
    for before, pin in zip(seal["source_test_pins_before"], pins, strict=True):
        require((before["path"], before["sha256"], before["size_bytes"]) ==
                (pin["path"], pin["sha256"], pin["bytes"]), "boundary source/test pin mismatch")
        bank.read(pin)
    for pin in parent["authenticated_pins"]:
        bank.read(pin)
    require(parent["flags"] == boundary.base_result()["flags"]
            and parent["dose_search_closed"] is True
            and parent["global_threshold_model_closed"] is True
            and parent["dispatch_and_write_audit"] == dict.fromkeys(boundary.AUDIT_KEYS, 0),
            "boundary closure/non-admission/dispatch gate changed")
    # Only authentication helpers run; no parent's compare, check or scientific execution.
    family, family_seal = boundary.authenticate(bank)
    require(parent["family_capture_bindings"] == family_seal
            and parent["retained_family_report"] == family["report"],
            "boundary/family retained byte mismatch")
    for key in (*retained.PRESERVED, "historical_flags", "prior_attempts",
                "preserved_dyadic_rejection_counts"):
        require(parent[key] == family[key], "boundary/family preserved binding: " + key)
    return parent, seal


def ordinal(word):
    retained.words([word], 1)
    magnitude = word & 0x7fff
    return -magnitude if word & 0x8000 else magnitude


def cell(word):
    retained.words([word], 1)
    center = retained.decode(word)
    magnitude, negative = word & 0x7fff, bool(word & 0x8000)
    if magnitude == 0:
        lower_word, upper_word = (0x8001, 0x8000) if negative else (0, 1)
        low, high = (Fraction(-1, 2**25), Fraction(0)) if negative else (
            Fraction(0), Fraction(1, 2**25))
    else:
        lower_word = word + 1 if negative else word - 1
        upper_word = word - 1 if negative else word + 1
        # Virtual overflow neighbors define the finite extreme's open midpoint.
        lower = Fraction(-65536) if lower_word == 0xfc00 else retained.decode(lower_word)
        upper = Fraction(65536) if upper_word == 0x7c00 else retained.decode(upper_word)
        low, high = (lower + center) / 2, (upper + center) / 2
    return {
        "word": word, "value": str(center), "sign_bit": int(negative),
        "significand_parity": word & 1, "lower_neighbor_word": lower_word,
        "upper_neighbor_word": upper_word, "lower_midpoint": str(low),
        "upper_midpoint": str(high), "lower_inclusive": not bool(word & 1),
        "upper_inclusive": not bool(word & 1),
        "zero_sign_binding_required": magnitude == 0,
        "overflow_neighbor_is_virtual": magnitude == 0x7bff,
    }


def bind(word, raw):
    require(isinstance(raw, str), "unrounded operand must be a retained rational string")
    value = Fraction(raw)
    require(str(value) == raw, "noncanonical unrounded operand rational")
    geometry = cell(word)
    require(word & 0x7fff != 0 or value != 0,
            "unrounded signed-zero sign authority unavailable")
    low, high = Fraction(geometry["lower_midpoint"]), Fraction(geometry["upper_midpoint"])
    require((low < value < high) or
            (value == low and geometry["lower_inclusive"]) or
            (value == high and geometry["upper_inclusive"]),
            "retained unrounded operand outside its FP16 RNE cell")
    return {
        "unrounded_value": raw, "lower_slack": str(value - low),
        "upper_slack": str(high - value),
        "tie": "lower" if value == low else "upper" if value == high else "interior",
        "rounding_residual": str(retained.decode(word) - value),
    }


def prerequisites(record):
    names = sorted(k for k in record if any(s in k.lower() for s in ("unrounded", "cell", "tie")))
    require(not set(names) - {"unrounded_operand_values"},
            "unrecognized retained operand cell prerequisites: " + ", ".join(names))
    if "unrounded_operand_values" not in record:
        return None
    values = record["unrounded_operand_values"]
    require(isinstance(values, list) and len(values) == 896,
            "partial/invalid unrounded operand census")
    return values


def partition(values):
    buckets = {}
    for control in CONTROLS:
        buckets.setdefault(retained.encoded(values[control]), []).append(control)
    return list(buckets.values())


def classify(word_groups, cell_groups, present):
    require(set(word_groups) == set(DOSES) and set(cell_groups) == set(DOSES)
            and len(present) == 72, "word/cell availability census changed")
    require(all(present) or not any(present), "partial retained unrounded cell bindings")
    if not all(["mapped_all"] in word_groups[dose] for dose in DOSES):
        classification = MIXED
    elif not any(present):
        classification = WORDS_ONLY
    elif all(["mapped_all"] in cell_groups[dose] for dose in DOSES):
        classification = SUPPORTED
    else:
        classification = MIXED
    return classification


def compare(source, parent_report):
    require(set(source["retained_baselines"]) == set(CONTROLS)
            and set(source["retained_observations"]) == set(DOSES), "operand census changed")
    observations, cells, missing, present = [], {}, [], []
    word_groups, cell_groups = {}, {}
    for dose in DOSES:
        outputs = source["retained_observations"][dose]
        require([o["control"] for o in outputs] == list(CONTROLS), "control order changed")
        deltas, signatures = {}, {}
        for output in outputs:
            control, operand = output["control"], output["operand"]
            baseline = source["retained_baselines"][control]
            require(output["polarity"] == "reverse"
                    and operand["dose"] == operand["signed_dose"] == dose
                    and operand["coordinate_order"] == "all_896_input_order_including_zeros",
                    "retained operand dose/polarity/order changed")
            before = retained.words(baseline["working_fp16_words"], 896)
            after = retained.words(operand["working_fp16_words"], 896)
            require(operand["changed_coordinates"] ==
                    [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b],
                    "changed-coordinate byte closure failed")
            old_values, new_values = prerequisites(baseline), prerequisites(operand)
            present.extend((old_values is not None, new_values is not None))
            for label, record, values in (("baseline", baseline, old_values),
                                          ("dose.operand", operand, new_values)):
                if values is None:
                    missing.append({"dose": dose, "control": control, "record": label,
                                    "retained_fields": sorted(record),
                                    "missing": "unrounded_operand_values",
                                    "reason": "Word-derived cell geometry cannot bind an actual rounding event."})
            moves, deltas[control], signatures[control] = [], [], []
            for i, (a, b) in enumerate(zip(before, after, strict=True)):
                for word in (a, b):
                    if str(word) not in cells:
                        cells[str(word)] = cell(word)
                delta, step = str(retained.decode(b) - retained.decode(a)), ordinal(b) - ordinal(a)
                old_binding = None if old_values is None else bind(a, old_values[i])
                new_binding = None if new_values is None else bind(b, new_values[i])
                moves.append({
                    "coordinate": i, "baseline_word": a, "dose_word": b, "word_changed": a != b,
                    "numeric_delta": delta, "neighbor_steps": step,
                    "neighbor_relation": "same_numeric_value" if step == 0 else (
                        "adjacent" if abs(step) == 1 else "multiple_cells"),
                    "baseline_binding": old_binding, "dose_binding": new_binding,
                })
                deltas[control].append(delta)
                if old_binding is not None and new_binding is not None:
                    signatures[control].append([
                        a >> 15, b >> 15, a & 1, b & 1, step,
                        old_binding["tie"], new_binding["tie"],
                        str(Fraction(new_binding["rounding_residual"]) -
                            Fraction(old_binding["rounding_residual"])),
                    ])
            require(deltas[control] == parent_report["response_profiles"][dose][control]["operand_delta"],
                    "reviewed operand_delta separator binding changed")
            observations.append({
                "dose": dose, "control": control, "coordinates": moves,
                "changed_word_count": sum(m["word_changed"] for m in moves),
                "baseline_unrounded_available": old_values is not None,
                "dose_unrounded_available": new_values is not None,
            })
        word_groups[dose] = partition(deltas)
        cell_groups[dose] = partition(signatures) if all(signatures.values()) else []
    classification = classify(word_groups, cell_groups, present)
    return {
        "classification": classification, "observations": observations, "word_cells": cells,
        "word_delta_partitions": word_groups, "bound_cell_signature_partitions": cell_groups,
        "availability": {
            "retained_baseline_vectors": 9, "retained_dose_vectors": 36,
            "retained_baseline_fp16_bytes": 9 * 896 * 2,
            "retained_dose_fp16_bytes": 36 * 896 * 2,
            "coordinate_pairs": 36 * 896, "word_cell_geometry": True,
            "unrounded_baseline_dose_record_occurrences": sum(present),
            "actual_cell_binding": all(present), "actual_tie_binding": all(present),
        },
        "missing_bindings": missing,
        "frozen_terminal_vector_is_not_an_unrounded_operand_binding": True,
        "no_causal_attribution": True,
    }


def base_result():
    result = boundary.base_result()
    result.update(diagnostic_id=NAME, command=COMMAND, preregistration=PREREGISTRATION)
    return result


def unknown(result, error):
    result.update(status="UNKNOWN", classification=UNKNOWN, successor=None,
                  successor_flags={flag: False for flag in SUCCESSORS.values() if flag},
                  integrity_error=f"{type(error).__name__}: {error}",
                  availability={"retained_bytes": "unconfirmed", "actual_cell_binding": "unconfirmed"})
    return result


def analyze(bank):
    result = base_result()
    try:
        parent, seal = authenticate(bank)
        result.update({key: parent[key] for key in retained.PRESERVED})
        for key in ("historical_flags", "prior_attempts", "preserved_dyadic_rejection_counts"):
            result[key] = parent[key]
        result.update(parent_capture_bindings=seal, parent_classification=parent["classification"],
                      retained_family_report=parent["retained_family_report"])
        report = compare(parent["retained_family_report"], parent["report"])
        classification = report["classification"]
        result.update(report=report, classification=classification, status="measured",
                      availability=report["availability"], integrity_error=None,
                      successor=SUCCESSORS[classification],
                      successor_flags={flag: SUCCESSORS[classification] == flag
                                       for flag in SUCCESSORS.values() if flag})
    except (IntegrityError, OSError, LookupError, TypeError, ValueError, ArithmeticError) as error:
        unknown(result, error)
    result["authenticated_pins"] = list(bank.pins.values())
    return result


def run_tests(result):
    import pytest

    class Results:
        collected = executed = errors = failures = skipped = 0

        def pytest_configure(self, config):
            config._ace3_operand_cell_result = result

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
