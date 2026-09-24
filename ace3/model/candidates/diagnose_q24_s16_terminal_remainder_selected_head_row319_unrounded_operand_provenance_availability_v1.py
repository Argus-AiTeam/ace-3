"""Inspect authenticated retained operand provenance; never construct an operand."""

import argparse
from contextlib import redirect_stdout
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_row319_retained_operand_rounding_cell_audit_v1 as cells


retained, boundary = cells.retained, cells.boundary
ROOT, PYTHON = retained.ROOT, retained.PYTHON
NAME = "diagnose_q24_s16_terminal_remainder_selected_head_row319_unrounded_operand_provenance_availability_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = retained.COMMAND.replace(retained.NAME, NAME)
SEAL = {
    "path": str(ROOT / "build/diagnostics/row319-retained-operand-rounding-cell-audit/c150a7e4c29e-round1-attempt001/capture-bindings.json"),
    "sha256": "1c6d4acc7cc3c342f2bc9ffd49217d080d0dca209c2f7ac44e3f3964aff66fb9",
    "bytes": 25471,
}
REVIEW = {
    "path": str(retained.HANDOFFS / "c150a7e4c29e/round-0001.json"),
    "sha256": "0c00961597be690abeaf3619f8bace4567e22571f270a0434242936d47d76d9d",
    "bytes": 690,
}
CONTROLS, DOSES = retained.CONTROLS, retained.DOSES
COMPLETE = "complete availability"
ABSENT = "absent availability"
PARTIAL = "partial/malformed provenance"
UNKNOWN = retained.UNKNOWN
SUCCESSORS = {
    COMPLETE: "preregister_fixed_control_retained_exact_cell_association",
    ABSENT: "close_retained_cell_mechanism_route_select_distinct_cpu_question",
    PARTIAL: "repair_only_identified_existing_provenance_bindings_then_independent_review",
    UNKNOWN: None,
}
PREREGISTRATION = {
    "inputs": "Reviewed c150a7e4c29e and reviewed 4b02378f0989, 412ec79f743b, "
              "b2e10d85b210, e762f68f110e; row319, nine fixed controls, reverse "
              "65/2048, 33/1024, 67/2048 and 17/512, with retained baselines.",
    "search_scope": "The authenticated inherited pin census: inspect build JSON/NPY/NPZ, "
                    "sealed stdout, capture-bindings and pinned JSON-line tool receipts. "
                    "Follow only explicitly named operand-provenance/unrounded/cell-binding "
                    "links, requiring path/hash/size. Source/tests, review/session history "
                    "and raw stream duplicates are authenticated, not operand evidence. "
                    "No directory crawl, weight/tokenizer loading or historical producer execution. "
                    "The source-bound 4864-element S16 SiLU/product unrounded field is explicitly "
                    "non-target, not malformed 896-coordinate final-RMSNorm operand provenance.",
    "payloads": "unrounded_operand_values: 896 canonical rational strings attached to the "
                "exact retained working_fp16_words and control/dose identity. "
                "exact_cell_bindings: 896 records containing word, sign_bit, lower_slack, "
                "upper_slack and tie; exact rational slacks must close the word's RNE cell "
                "and respect endpoint ownership. No operand is recovered from these slacks. "
                "Detached JSON payloads inherit an explicit pinned operand record; provenance "
                "manifests must carry their own fixed control/dose/word identity. Unrecognized "
                "payload schemas, incomplete vectors or conflicting bindings are partial/malformed.",
    "rule": "Complete requires consistent bindings for all 45 distinct operand records "
            "(72 baseline/dose occurrences, 36 matrix rows). Absent requires zero candidate "
            "payloads after the entire bounded search. Any supplied but incomplete/malformed "
            "provenance is partial/malformed, never absence. Authentication, missing linked "
            "bytes, unpinned links or unreadable artifacts are UNKNOWN/integrity.",
    "successors": SUCCESSORS,
    "successor_gates": {
        COMPLETE: "Only a new preregistered retained-cell association question after independent "
                  "review; not rounding causality, repair or suffix execution authority.",
        ABSENT: "The searched artifacts cannot support this retained-cell mechanism route. "
                "Choose a genuinely different CPU question; do not reconstruct missing bytes, "
                "rerun an old producer, search doses or repeat an availability record-check.",
        PARTIAL: "Only supply/correct the specifically identified already-existing provenance "
                 "through new immutable bindings and independent review; no scientific successor.",
        UNKNOWN: "No scientific successor until the exact integrity prerequisite is supplied "
                 "and independently reviewed; never downgrade UNKNOWN to absence.",
    },
    "closure": retained.PREREGISTRATION["closure"],
    "boundary": retained.PREREGISTRATION["boundary"],
}
AUDIT_KEYS = cells.AUDIT_KEYS + ("operand_reconstructions", "reference_mutations")
EXPECTED_TESTS = 19
require, IntegrityError = retained.require, retained.IntegrityError
PAYLOADS = ("unrounded_operand_values", "exact_cell_bindings")
LINKS = ("operand_provenance", "operand_provenance_manifest", "unrounded_operand_provenance",
         "unrounded_operand_manifest", "exact_cell_binding_manifest")


class SearchBytes(retained.BoundBytes):
    def __init__(self, loader=Path.read_bytes):
        super().__init__(loader)
        self.attempts = []

    def read(self, pin):
        fact = {"expected": dict(pin)}
        try:
            raw = super().read(pin)
        except (OSError, ValueError, LookupError, TypeError) as error:
            raw = self.data.get(pin.get("path"))
            if raw is not None:
                fact["actual"] = {"sha256": retained.digest(raw), "bytes": len(raw)}
            fact["error"] = f"{type(error).__name__}: {error}"
            self.attempts.append(fact)
            raise
        if not any(p["expected"] == pin for p in self.attempts):
            self.attempts.append({"expected": dict(pin), "actual": self.pins[pin["path"]]})
        return raw


def authenticate(bank):
    retained.review_gate(retained.document(bank.read(REVIEW)), "c150a7e4c29e", 1)
    seal = retained.document(bank.read(SEAL))
    require((seal["task_id"], seal["attempt_id"], seal["account_uid"], seal["working_directory"]) ==
            ("c150a7e4c29e", "c150a7e4c29e-round1-attempt001", 1000, str(ROOT)),
            "cell-audit task/attempt/account/worktree mismatch")
    require(seal["command"] == cells.COMMAND
            and seal["argv"] == [PYTHON, "-B", "-m", cells.MODULE, "--check"],
            "cell-audit command mismatch")
    parent = retained.document(boundary.sealed_stdout(seal, bank))
    require((parent["diagnostic_id"], parent["version"], parent["command"], parent["status"],
             parent["classification"]) ==
            (cells.NAME, 1, cells.COMMAND, "measured", cells.WORDS_ONLY),
            "reviewed cell-audit identity/classification mismatch")
    retained.tests_gate(parent, cells.EXPECTED_TESTS)
    pins = parent["tests"]["compiled"]
    require(parent["source_test_pins"] == pins
            and [p["path"] for p in pins] == [str(cells.SOURCE), str(cells.TEST)]
            and seal["source_test_pins_before"] == seal["source_test_pins_after"]
            and len(seal["source_test_pins_before"]) == 2, "cell-audit compiled pins mismatch")
    for captured, pin in zip(seal["source_test_pins_before"], pins, strict=True):
        require((captured["path"], captured["sha256"], captured["size_bytes"]) ==
                (pin["path"], pin["sha256"], pin["bytes"]), "cell-audit source/test mismatch")
        bank.read(pin)
    for pin in parent["authenticated_pins"]:
        bank.read(pin)
    require(parent["flags"] == cells.base_result()["flags"]
            and parent["dose_search_closed"] is True
            and parent["global_threshold_model_closed"] is True
            and parent["dispatch_and_write_audit"] == dict.fromkeys(cells.AUDIT_KEYS, 0),
            "cell-audit closure/non-admission/zero-dispatch mismatch")
    previous, previous_seal = cells.authenticate(bank)
    require(parent["parent_capture_bindings"] == previous_seal
            and parent["retained_family_report"] == previous["retained_family_report"],
            "cell-audit retained parent bytes mismatch")
    for key in (*retained.PRESERVED, "historical_flags", "prior_attempts",
                "preserved_dyadic_rejection_counts"):
        require(parent[key] == previous[key], "preserved parent binding: " + key)
    return parent, seal


def expected_operands(report):
    expected = {(control, "baseline"): report["retained_baselines"][control]["working_fp16_words"]
                for control in CONTROLS}
    for dose in DOSES:
        observations = report["retained_observations"][dose]
        require([o["control"] for o in observations] == list(CONTROLS), "fixed control order changed")
        for item in observations:
            require(item["polarity"] == "reverse" and item["operand"]["dose"] == dose,
                    "fixed dose/polarity changed")
            expected[item["control"], dose] = item["operand"]["working_fp16_words"]
    for words in expected.values():
        retained.words(words, 896)
    return expected


def rational(value):
    require(isinstance(value, str), "cell slack must be an exact rational string")
    result = Fraction(value)
    require(str(result) == value, "noncanonical exact cell slack")
    return result


def payload_signature(kind, values, words):
    require(isinstance(values, list) and len(values) == 896, "incomplete 896-coordinate payload")
    signature = []
    for word, value in zip(words, values, strict=True):
        if kind == "unrounded_operand_values":
            binding = cells.bind(word, value)
        else:
            require(isinstance(value, dict)
                    and set(value) == {"word", "sign_bit", "lower_slack", "upper_slack", "tie"},
                    "malformed exact cell binding")
            require(type(value["word"]) is int and value["word"] == word
                    and type(value["sign_bit"]) is int and value["sign_bit"] == word >> 15,
                    "exact cell word/sign identity mismatch")
            geometry = cells.cell(word)
            low, high = rational(value["lower_slack"]), rational(value["upper_slack"])
            require(low >= 0 and high >= 0
                    and low + high == rational(geometry["upper_midpoint"]) -
                    rational(geometry["lower_midpoint"]), "exact cell slack closure failed")
            require((low > 0 or geometry["lower_inclusive"])
                    and (high > 0 or geometry["upper_inclusive"]), "unowned exact cell tie")
            require(value["tie"] == ("lower" if low == 0 else "upper" if high == 0 else "interior"),
                    "exact cell tie/slack mismatch")
            binding = value
        signature.append({k: binding[k] for k in ("lower_slack", "upper_slack", "tie")})
    return retained.digest(retained.encoded(signature))


def classify(states, issues):
    if issues:
        return PARTIAL
    count = sum(bool(v) for v in states.values())
    if count == len(CONTROLS) * (len(DOSES) + 1):
        return COMPLETE
    return PARTIAL if count else ABSENT


def matrix(states, unknown=False, issues=()):
    def entry(key):
        bindings = states.get(key, [])
        errors = [issue for issue in issues if (issue.get("control"), issue.get("dose")) == key]
        return {"availability": "unconfirmed" if unknown else "partial/malformed" if errors else
                "available" if bindings else "absent", "bindings": bindings, "issues": errors}
    return [{"row": 319, "control": control, "dose": dose, "polarity": "reverse",
             "baseline": entry((control, "baseline")), "operand": entry((control, dose))}
            for dose in DOSES for control in CONTROLS]


def walk(value, pointer="", context=None):
    context = dict(context or {})
    if isinstance(value, dict):
        context.update({k: value[k] for k in ("control", "dose", "polarity", "row") if k in value})
        yield pointer, value, context
        for key, child in value.items():
            child_context = dict(context)
            if key in CONTROLS:
                child_context["control"] = key
            if key in DOSES:
                child_context["dose"] = key
            if key == "retained_baselines":
                child_context.update(dose="baseline", polarity="reverse")
            escaped = key.replace("~", "~0").replace("/", "~1")
            yield from walk(child, pointer + "/" + escaped, child_context)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                yield from walk(child, pointer + "/" + str(index), context)


def artifact_kind(path):
    p = Path(path)
    if p.is_relative_to(ROOT / "build"):
        if p.name in ("raw-command.bin", "stderr.bin"):
            return "authenticated stream duplicate"
        return {".json": "json", ".npy": "npy", ".npz": "npz", ".bin": "json"}.get(
            p.suffix, "authentication only")
    if p.parent == Path("/tmp") and p.suffix == ".txt":
        return "json-line receipt"
    return "authentication only"


def inspect_artifacts(bank, expected, result):
    states = {key: [] for key in expected}
    issues = result["provenance_issues"]
    queue = [(pin, None) for pin in bank.pins.values()]
    searched = set()
    references = {}

    def linked(pin, source):
        require(isinstance(pin, dict) and {"path", "sha256"} <= pin.keys()
                and ("bytes" in pin or "size_bytes" in pin), "unpinned provenance link: " + source)
        require(Path(pin["path"]).is_relative_to(ROOT / "build")
                or Path(pin["path"]).parent == Path("/tmp"), "out-of-scope provenance link: " + source)
        raw = bank.read(pin)
        references.setdefault(pin["path"], []).append(source)
        return raw

    for pin, inherited in queue:
        path = pin["path"]
        if path in searched:
            continue
        searched.add(path)
        raw, kind = bank.read(pin), artifact_kind(path)
        fact = {**bank.pins[path], "format": kind, "candidate_fields": [], "array_members": [],
                "non_target_unrounded_fields": []}
        result["searched_artifacts"].append(fact)
        if kind in ("authentication only", "authenticated stream duplicate"):
            fact["operand_search"] = False
            continue
        fact["operand_search"] = True
        if kind in ("npy", "npz"):
            archive = retained.np.load(io.BytesIO(raw), allow_pickle=False)
            try:
                arrays = [(name, archive[name]) for name in archive.files] if kind == "npz" else [
                    ("array", archive)]
                for name, array in arrays:
                    fact["array_members"].append({"name": name, "dtype": array.dtype.str,
                                                  "shape": list(array.shape)})
                    if any(s in name.lower() for s in ("unrounded", "cell_binding", "operand_provenance")):
                        if (kind == "npz" and name == "s16_unrounded_binary64"
                                and array.dtype.str == "<f8" and array.shape == (4864,)
                                and "stage16" in archive.files
                                and archive["stage16"].dtype.str == "<u2"
                                and archive["stage16"].shape == (4864,)):
                            source = str(ROOT / "ace3/model/candidates/q24_s16_toward_zero_native_v1.py")
                            require(source in bank.pins, "missing authenticated S16 field declaration")
                            declaration = bank.data[source].decode().splitlines()[126:130]
                            require(declaration[0].strip() == "elif stage == 16:"
                                    and 'arrays["stage14"]' in declaration[1]
                                    and 'arrays["stage15"]' in declaration[1]
                                    and 'arrays["s16_unrounded_binary64"]' in declaration[2],
                                    "S16 field source declaration changed")
                            fact["non_target_unrounded_fields"].append({
                                "name": name, "dtype": array.dtype.str, "shape": list(array.shape),
                                "source": bank.pins[source], "source_lines": [127, 130],
                                "declaration": declaration,
                                "reason": "S16 SiLU/product operand; not the 896-coordinate "
                                          "baseline/reverse-dose pre-final-RMSNorm operand",
                            })
                            continue
                        issues.append({"source": path, "pointer": name,
                                       "error": "array payload lacks authenticated exact control/dose/cell schema"})
            finally:
                if kind == "npz":
                    archive.close()
            continue
        if kind == "json-line receipt":
            lines = [(i, line) for i, line in enumerate(raw.splitlines()) if line.startswith(b"{")]
            require(bool(lines), "no retained JSON document in receipt: " + path)
            documents = [(f"/line/{i + 1}", retained.document(line)) for i, line in lines]
        else:
            documents = [("", retained.document(raw))]
        for prefix, document in documents:
            for pointer, record, context in walk(document, prefix, inherited):
                if "path" in record and "sha256" in record:
                    references.setdefault(record["path"], []).append(path + "#" + pointer)
                for name in LINKS:
                    if name in record and record[name] is not None:
                        link = record[name]
                        linked(link, path + "#" + pointer + "/" + name)
                        queue.append((link, context))
                candidates = [key for key, value in record.items()
                              if key in PAYLOADS or
                              (key not in LINKS and isinstance(value, (dict, list))
                               and any(s in key.lower() for s in ("unrounded", "cell_binding")))]
                for name in candidates:
                    location = pointer + "/" + name
                    fact["candidate_fields"].append(location)
                    if context.get("polarity", "reverse") != "reverse" or context.get("row", 319) != 319:
                        continue
                    key = (context.get("control"), context.get("dose"))
                    values = record[name]
                    payload_pin = None
                    if isinstance(values, dict) and "path" in values:
                        payload_pin = values
                        values = retained.document(linked(payload_pin, path + "#" + location))
                        queue.append((payload_pin, context))
                    try:
                        require(name in PAYLOADS, "unrecognized operand provenance payload: " + name)
                        require(key in expected and record.get("working_fp16_words") == expected[key],
                                "missing/mismatched fixed control/dose/word identity")
                        signature = payload_signature(name, values, expected[key])
                        require(all(b["cell_signature_sha256"] == signature for b in states[key]),
                                "conflicting retained cell bindings")
                        states[key].append({"path": path, "sha256": fact["sha256"], "bytes": fact["bytes"],
                                            "pointer": location, "kind": name,
                                            "payload_pin": payload_pin,
                                            "cell_signature_sha256": signature})
                    except (ValueError, TypeError, LookupError, ArithmeticError) as error:
                        issues.append({"source": path, "pointer": location, "control": key[0],
                                       "dose": key[1], "error": f"{type(error).__name__}: {error}"})
    for fact in result["searched_artifacts"]:
        fact["referenced_by"] = sorted(set(references.get(fact["path"], [])))
        if fact["path"] == REVIEW["path"]:
            fact["referenced_by"].append("diagnostic.REVIEW (reviewed root)")
        if fact["path"] == SEAL["path"]:
            fact["referenced_by"].append("diagnostic.SEAL (reviewed root)")
    return states


def base_result():
    result = cells.base_result()
    result.update(diagnostic_id=NAME, command=COMMAND, preregistration=PREREGISTRATION,
                  searched_artifacts=[], provenance_issues=[], source_test_pins=[],
                  classification=UNKNOWN, status="UNKNOWN", successor=None,
                  availability_matrix=matrix({}, unknown=True))
    return result


def unknown(result, error):
    result.update(status="UNKNOWN", classification=UNKNOWN, successor=None,
                  successor_flags={flag: False for flag in SUCCESSORS.values() if flag},
                  integrity_error=f"{type(error).__name__}: {error}",
                  availability_matrix=matrix({}, unknown=True))
    return result


def analyze(bank):
    result = base_result()
    try:
        parent, seal = authenticate(bank)
        for key in (*retained.PRESERVED, "historical_flags", "prior_attempts",
                    "preserved_dyadic_rejection_counts"):
            result[key] = parent[key]
        result.update(parent_capture_bindings=seal, parent_classification=parent["classification"],
                      authenticated_review_missions=["c150a7e4c29e", "4b02378f0989", "412ec79f743b",
                                                     "b2e10d85b210", "e762f68f110e"])
        expected = expected_operands(parent["retained_family_report"])
        states = inspect_artifacts(bank, expected, result)
        classification = classify(states, result["provenance_issues"])
        result.update(status="measured", classification=classification, integrity_error=None,
                      successor=SUCCESSORS[classification],
                      successor_flags={flag: SUCCESSORS[classification] == flag
                                       for flag in SUCCESSORS.values() if flag},
                      availability_matrix=matrix(states, issues=result["provenance_issues"]),
                      distinct_operand_records=len(expected),
                      available_operand_records=sum(bool(v) for v in states.values()),
                      absence_scope="Only the enumerated authenticated artifacts and named provenance links; "
                                    "not a filesystem-wide or physical impossibility claim.",
                      frozen_terminal_vector_is_not_an_unrounded_operand=True,
                      rounded_words_state_and_reference_arrays_are_not_unrounded_operands=True)
    except (IntegrityError, OSError, LookupError, TypeError, ValueError, ArithmeticError) as error:
        unknown(result, error)
    result["authenticated_pins"] = list(bank.pins.values())
    result["artifact_read_facts"] = bank.attempts
    return result


def run_tests(result):
    import pytest

    class Results:
        collected = executed = errors = failures = skipped = 0

        def pytest_configure(self, config):
            config._ace3_operand_availability_result = result

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
    return {"runner": "pytest", "executed": results.executed, "errors": results.errors,
            "failures": results.failures, "skipped": results.skipped,
            "collected": results.collected, "returncode": int(code)}


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
        result = analyze(SearchBytes())
        frozen = retained.encoded(result)
        tests = run_tests(result)
        require(retained.encoded(result) == frozen, "tests mutated authenticated evidence")
        for pin in (*pins, *result["authenticated_pins"]):
            require(retained.digest(Path(pin["path"]).read_bytes()) == pin["sha256"],
                    "source/evidence changed during check: " + pin["path"])
    require(audit == dict.fromkeys(AUDIT_KEYS, 0), "forbidden dispatch/write census changed")
    tests["compiled"] = pins
    result.update(tests=tests, source_test_pins=pins, dispatch_and_write_audit=audit)
    if (tests["returncode"] != 0 or tests["collected"] != tests["executed"]
            or tests["executed"] != EXPECTED_TESTS
            or tests["errors"] or tests["failures"] or tests["skipped"]):
        result["measurement_classification_before_failed_validation"] = result["classification"]
        unknown(result, IntegrityError("focused pytest failed, errored, skipped or changed census"))
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
    return int(result["classification"] == UNKNOWN)


if __name__ == "__main__":
    sys.modules[MODULE] = sys.modules[__name__]
    sys.exit(main())
