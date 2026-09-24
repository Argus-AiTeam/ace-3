"""Read-only retained-byte contrast; no dose, RMSNorm, or head execution."""

import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from unittest.mock import patch

import numpy as np


ROOT = Path("/home/argustest/ace3-argus")
PYTHON = "/home/argustest/miniconda3/bin/python"
NAME = "diagnose_q24_s16_terminal_remainder_selected_head_row319_retained_byte_family_contrast_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
HANDOFFS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
SEAL = {
    "path": str(ROOT / "build/diagnostics/row319-reverse-quartile/b2e10d85b210-round2-attempt001/capture-bindings.json"),
    "sha256": "50702d615e9af20d826384b3b6b4d8dbee6d25f557e12d1c9d5fdc8dc03d2f19",
}
REVIEWS = (
    {"path": str(HANDOFFS / "b2e10d85b210/round-0002.json"),
     "sha256": "e8117fc112531458f05452b9a00f5504e47a21fcb6cbbc5f9ac7f102c831ebd7"},
    {"path": str(HANDOFFS / "e762f68f110e/round-0001.json"),
     "sha256": "cd1a90cd7e7e92014ec75b795b3b64eb0b7a0d2f3943ac21d3e50001b5553183"},
)
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
    "mapped_all", "inherited_native",
)
DOSES = ("65/2048", "33/1024", "67/2048", "17/512")
ROWS = (319, 34319)
FIELDS = ("working_fp16_words", "changed_coordinates", "rmsnorm_words", "rmsnorm_scalars")
UNIQUE = "unique mapped_all retained-byte discriminator"
FAMILY = "family/inconsistent retained-byte discriminator"
UNKNOWN = "UNKNOWN/integrity"
SUCCESSORS = {
    UNIQUE: "One control-specific question: which rounding boundary distinguishes the "
            "retained mapped_all operand/RMSNorm profile from the eight controls?",
    FAMILY: "One family-specific question: does the reported shared/inconsistent "
            "operand family, rather than control identity, distinguish the rounding response?",
    UNKNOWN: None,
}
PREREGISTRATION = {
    "inputs": "Only authenticated b2e10d85b210 retained reverse 65/2048, 33/1024, "
              "67/2048 and reviewed e762f68f110e reverse 17/512 bytes, fixed nine controls "
              "and rows 319/34319. No operand construction or suffix execution.",
    "rule": "Group exact equality separately for the four preregistered operand/RMSNorm "
            "fields at every retained dose. A field whose mapped_all group is singleton "
            "at ALL FOUR doses is a unique discriminator. Otherwise any nontrivial "
            "partition is family/inconsistent. All fields identical selects UNKNOWN. "
            "Selected-logit words, references, control names and historical gate outcomes "
            "are NOT discriminator features. No coordinate ranking or reselection.",
    "interpretation": "A retained-byte discriminator is descriptive separation, not "
                      "rounding-only causality, an actual-suffix threshold proof, or repair. "
                      "Baseline differences are disclosed; separation need not be dose-induced.",
    "successors": SUCCESSORS,
    "integrity": "Missing bytes, pins, fields, reference/word closure or review authority "
                 "select UNKNOWN/integrity and no scientific successor until the exact "
                 "prerequisite is supplied and independently reviewed.",
    "closure": "dose_search_closed and global_threshold_model_closed remain true for "
               "every outcome. No new doses, subdivision, scalar-threshold transfer, "
               "coordinate choice or reference mutation. Successor questions require "
               "new preregistration and independent review, not automatic execution.",
    "boundary": "CPU-only native-S16-RTZ non-admission evidence; INT4 weights, FP16 "
                "operators/KV and wider Q24 residual state unchanged. Preserve original-input "
                "independently propagated global references, exact thresholds, source/operand/"
                "state/KV/lineage gates, all historical FAIL/UNKNOWN, 4ae REJECTED and f400 "
                "SUPPORTED. No prefix/admission/reference-producer/full-vocabulary replay, "
                "production repair, precision/scale expansion, hardware/GPU/RTL/ACE2, "
                "strict-FP16-state W4A16, new-token or full-model admission claim.",
}
AUDIT_KEYS = (
    "writes", "artifact_overwrites", "forbidden_calls", "external_invocations",
    "prefix_replays", "admission_replays", "reference_producer_replays",
    "full_vocabulary_replays", "closed_diagnostic_replays", "prior_17_512_replays",
    "new_dose_executions", "new_operand_constructions", "operand_rne_scalars",
    "final_rmsnorm_invocations", "selected_row_head_invocations",
    "exact_selected_row_accumulations", "GPU_dispatch", "RTL_dispatch",
)
PRESERVED = (
    "assets", "authenticated_files", "protected_input_identity", "selected_head_row_identity",
    "final_reference_authority", "retained_controls_and_failure_gates", "retained_thresholds",
    "retained_dose_summaries", "closed_crossing_classification", "closed_bracket_classification",
    "closed_localizer_classification", "retained_common_component", "counterfactual_common_component",
)
EXPECTED_TESTS = 14


class IntegrityError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def document(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON binding: " + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise IntegrityError("nonfinite JSON binding: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


class BoundBytes:
    """Read each pinned artifact once; all analysis uses that authenticated snapshot."""

    def __init__(self, loader=Path.read_bytes):
        self.loader = loader
        self.data = {}
        self.pins = {}

    def read(self, pin):
        path = pin["path"]
        require(Path(path).is_absolute(), "nonabsolute binding: " + path)
        if path not in self.data:
            self.data[path] = self.loader(Path(path))
        raw = self.data[path]
        require(digest(raw) == pin["sha256"], "SHA-256 mismatch: " + path)
        for key in ("bytes", "size_bytes"):
            if key in pin:
                require(type(pin[key]) is int and len(raw) == pin[key], "size mismatch: " + path)
        self.pins[path] = {"path": path, "sha256": digest(raw), "bytes": len(raw)}
        return raw


def review_gate(review, mission, round_number):
    require((review["kind"], review["producer_role"], review["mission_id"], review["round"],
             review["review"]["status"]) ==
            ("round_reviewed_handoff", "reviewer", mission, round_number, "done"),
            "missing independent reviewed authority: " + mission)


def capture_bytes(seal, bank):
    require(seal["capture_complete"] is True and seal["timed_out"] is False
            and seal["returncode"] == 0, "incomplete b2 command capture")
    streams = {}
    for name in ("whole_capture", "stdout", "stderr"):
        binding = seal["artifacts"][name]
        source, destination = binding["source"], binding["destination"]
        require((source["sha256"], source["size_bytes"]) ==
                (destination["sha256"], destination["size_bytes"]),
                "source/destination capture binding mismatch: " + name)
        require(Path(source["path"]).parent == Path(seal["original_temporary_directory"])
                and Path(destination["path"]).parent == Path(seal["sealed_build_directory"]),
                "capture original/destination path mismatch: " + name)
        streams[name] = bank.read(destination)
    require(seal["stdout_sha256"] == digest(streams["stdout"])
            and seal["whole_capture_sha256"] == digest(streams["whole_capture"]),
            "stdout/whole-capture hash conflation")
    require(streams["whole_capture"] in (
        streams["stdout"] + streams["stderr"], streams["stderr"] + streams["stdout"]),
        "sealed whole capture does not close over the retained b2 streams")
    return streams["stdout"]


def tests_gate(result, count):
    tests = result["tests"]
    require([tests[k] for k in ("runner", "executed", "errors", "failures", "skipped")] ==
            ["pytest", count, 0, 0, 0], "reviewed focused test census changed")
    require(len(tests["compiled"]) == 2, "missing reviewed source/test pins")


def authenticate(bank):
    for pin, mission, number in zip(REVIEWS, ("b2e10d85b210", "e762f68f110e"), (2, 1)):
        review_gate(document(bank.read(pin)), mission, number)
    seal = document(bank.read(SEAL))
    require((seal["task_id"], seal["attempt_id"], seal["account_uid"], seal["working_directory"]) ==
            ("b2e10d85b210", "b2e10d85b210-round2-attempt001", 1000, str(ROOT)),
            "b2 task/attempt/account/worktree binding changed")
    q = document(capture_bytes(seal, bank))
    qname = NAME.replace("retained_byte_family_contrast", "reverse_quartile_suffix_threshold")
    pname = NAME.replace("retained_byte_family_contrast", "dyadic_straddle_suffix_contrast")
    qcommand = COMMAND.replace(NAME, qname)
    require(q["diagnostic_id"] == qname and q["version"] == 1 and q["command"] == qcommand
            and seal["command"] == qcommand
            and seal["argv"] == [PYTHON, "-B", "-m", "ace3.model.candidates." + qname, "--check"],
            "b2 source/command identity mismatch")
    require(seal["source_test_pins_before"] == seal["source_test_pins_after"],
            "b2 source/test changed during capture")
    tests_gate(q, 15)
    for captured, compiled in zip(seal["source_test_pins_before"], q["tests"]["compiled"]):
        require((captured["path"], captured["sha256"], captured["size_bytes"]) ==
                (compiled["path"], compiled["sha256"], compiled["bytes"]),
                "b2 compiled source/test binding mismatch")
        bank.read(captured)
    raw = bank.read(q["reviewed_dyadic_receipt"])
    # The reviewed historical tool receipt has a summary before its full JSON.
    lines = [line for line in raw.splitlines() if line.startswith(b'{"validation":')]
    require(len(lines) == 1, "missing/ambiguous full reviewed dyadic receipt; summary is not evidence")
    receipt = document(lines[0])
    p, validation = receipt["native_result"], receipt["validation"]
    require(p["diagnostic_id"] == pname and p["version"] == 1
            and p["command"] == COMMAND.replace(NAME, pname),
            "dyadic source/command identity mismatch")
    require([validation[k] for k in ("native_command", "native_exit", "stdout_json_documents")] ==
            [p["command"], 0, 1] and validation["tests"] == p["tests"],
            "dyadic execution/test receipt mismatch")
    tests_gate(p, 14)
    for result in (q, p):
        for pin in (*result["authenticated_pins"], *result["authenticated_files"], *result["tests"]["compiled"]):
            bank.read(pin)
        require(result["status"] == "rejected", "historical rejection changed")
        require(result["retained_common_component"] == result["counterfactual_common_component"] == "UNKNOWN",
                "historical UNKNOWN changed")
        for key, value in result["dispatch_and_write_audit"].items():
            if key.endswith("replays") or key in ("writes", "artifact_overwrites", "forbidden_calls"):
                require(value == 0, "historical forbidden operation: " + key)
    require(q["classification"] == "mixed by control family"
            and p["classification"] == "control-specific suffix/rounding successor",
            "reviewed branch classification changed")
    require(q["dose_search_closed"] is True and q["global_threshold_model_closed"] is True
            and q["report"]["dose_search_closed"] is True
            and q["report"]["global_threshold_model_closed"] is True, "closed branch reopened")
    for key in PRESERVED:
        require(q[key] == p[key], "dyadic/b2 retained binding mismatch: " + key)
    require(q["preserved_dyadic_rejection_counts"] == p["report"]["prediction_closure_counts"]
            == validation["prediction_closure_counts"], "dyadic rejection counts changed")
    for key in ("frozen_original_terminal_vector", "frozen_original_terminal_vector_identity"):
        require(q["report"][key] == p["report"][key], "terminal operand binding changed: " + key)
    return q, p, seal


def words(value, length):
    require(isinstance(value, list) and len(value) == length
            and all(type(w) is int and 0 <= w <= 65535 and w & 0x7c00 != 0x7c00 for w in value),
            "missing/invalid finite FP16 words")
    return value


def decode(word):
    words([word], 1)
    sign = -1 if word & 0x8000 else 1
    exponent, mantissa = (word >> 10) & 31, word & 1023
    return sign * Fraction(mantissa if exponent == 0 else 1024 + mantissa) * Fraction(2) ** (
        -24 if exponent == 0 else exponent - 25)


def array(bank, pin, dtype, shape, field=None):
    raw = bank.read(pin)
    if field is None:
        value = np.load(io.BytesIO(raw), allow_pickle=False)
    else:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            value = archive[field].copy()
    require(value.dtype.str == dtype and value.shape == shape, "retained array shape/dtype: " + pin["path"])
    return value


def partitions(profiles):
    groups = {}
    for dose in DOSES:
        groups[dose] = {}
        require(set(profiles[dose]) == set(CONTROLS), "missing control in retained-byte profiles: " + dose)
        for field in FIELDS:
            buckets = {}
            for control in CONTROLS:
                value = profiles[dose][control][field]
                require(value is not None, "missing retained-byte discriminator field: " + field)
                buckets.setdefault(encoded(value), []).append(control)
            groups[dose][field] = list(buckets.values())
    unique_fields = [
        field for field in FIELDS
        if all(["mapped_all"] in groups[dose][field] for dose in DOSES)
    ]
    if unique_fields:
        classification = UNIQUE
    elif any(len(groups[dose][field]) > 1 for dose in DOSES for field in FIELDS):
        classification = FAMILY
    else:
        classification = UNKNOWN
    return {"classification": classification, "successor": SUCCESSORS[classification],
            "stable_unique_fields": unique_fields, "field_partitions": groups,
            "integrity_error": ("no valid retained-byte discriminator; integrity/evidence prerequisite "
                                "must be supplied and independently reviewed" if classification == UNKNOWN else None)}


def compare(q, p, bank):
    refs = q["final_reference_authority"]["reference"]
    require(refs["reference_only"] is True
            and refs["reference_policy"] == "legacy-binary64-AWQ-fully-independent-propagation",
            "original-input independent global reference authority changed")
    ref_fp16 = array(bank, refs["logits_fp16"], "<u2", (151936,))
    ref_binary64 = array(bank, refs["logits_binary64"], "<f8", (151936,))
    selected_refs = {"fp16_words": [int(ref_fp16[i]) for i in ROWS],
                     "binary64_hex": [float(ref_binary64[i]).hex() for i in ROWS]}
    fixed = {"fp16": [decode(int(ref_fp16[i])) for i in ROWS],
             "binary64": [Fraction(float(ref_binary64[i])) for i in ROWS]}
    controls = q["retained_controls_and_failure_gates"]
    require([c["control"] for c in controls] == list(CONTROLS), "retained control census changed")
    baseline = {}
    for c in controls:
        control, parent = c["control"], c["parent"]
        require(parent["retained_L23"]["source_operand_state_KV_RTZ_checks"],
                "missing source/operand/state/KV/lineage gate: " + control)
        operand = array(bank, parent["terminal_archive"], "<u2", (896,), "stage18").tolist()
        normalized = array(bank, c["arrays"]["rmsnorm"], "<u2", (896,)).tolist()
        logits = array(bank, c["arrays"]["logits"], "<u2", (151936,))
        baseline[control] = {"working_fp16_words": words(operand, 896),
                             "rmsnorm_words": words(normalized, 896),
                             "rmsnorm_scalars": c["rmsnorm_integer_details"],
                             "selected_logit_words": words([int(logits[i]) for i in ROWS], 2)}
    profiles, retained, comparisons, activity = {}, {}, {}, {}
    for dose in DOSES:
        result = p if dose == "17/512" else q
        outputs = [o for o in result["report"]["suffix_outputs"][dose] if o["polarity"] == "reverse"]
        require([o["control"] for o in outputs] == list(CONTROLS), "suffix output census changed: " + dose)
        tables = [r for r in result["report"]["dose_tables"][dose] if r["polarity"] == "reverse"]
        require([(r["control"], r["left_id"], r["right_id"], r["branch"]) for r in tables] ==
                [(c, l, r, b) for c in CONTROLS for l, r in (ROWS, ROWS[::-1])
                 for b in ("fp16", "binary64")], "selected-row/reference census changed: " + dose)
        profiles[dose], retained[dose], activity[dose] = {}, outputs, []
        for output in outputs:
            c, operand = output["control"], output["operand"]
            require(operand["dose"] == operand["signed_dose"] == dose
                    and operand["coordinate_order"] == "all_896_input_order_including_zeros"
                    and operand["frozen_terminal_vector_identity"] ==
                    q["report"]["frozen_original_terminal_vector_identity"], "operand lineage mismatch")
            working = words(operand["working_fp16_words"], 896)
            changed = [i for i, (a, b) in enumerate(zip(baseline[c]["working_fp16_words"], working)) if a != b]
            require(operand["changed_coordinates"] == changed, "retained changed-coordinate closure: " + c)
            normalized = words(output["rmsnorm_words"], 896)
            scalars = output["rmsnorm_scalars"]
            require(set(scalars) == {"mean_q48", "root_q24"}
                    and all(type(v) is int and v > 0 for v in scalars.values()), "missing RMSNorm scalar binding")
            measured = words(output["selected_logit_words"], 2)
            old = baseline[c]["selected_logit_words"]
            active = [a != b for a, b in zip(old, measured)]
            require(active[0] == (dose == "17/512" or c == "mapped_all"),
                    "reviewed mapped_all-only early activation binding changed")
            activity[dose].append({"control": c, "baseline_words": old, "measured_words": measured,
                                   "row319_active": active[0], "row34319_active": active[1]})
            if dose == "17/512":
                endpoint = q["report"]["reviewed_upper_endpoints"][c]
                require(endpoint == {"baseline_words": old, "dose": dose, "measured_words": measured,
                                     "row319_active": active[0]}, "reviewed upper endpoint binding changed")
            profiles[dose][c] = {"working_fp16_words": working, "changed_coordinates": changed,
                                 "rmsnorm_words": normalized, "rmsnorm_scalars": scalars}
            for row in (r for r in tables if r["control"] == c):
                require(row["dose"] == dose, "table dose mismatch")
                indices = [ROWS.index(row[k]) for k in ("left_id", "right_id")]
                a, b = indices
                new_margin, old_margin = decode(measured[a]) - decode(measured[b]), decode(old[a]) - decode(old[b])
                ref = fixed[row["branch"]][a] - fixed[row["branch"]][b]
                require(row["baseline_words"] == [old[a], old[b]]
                        and row["measured_words"] == [measured[a], measured[b]], "table/logit word mismatch")
                for key, expected in (
                    ("baseline_margin", old_margin), ("measured_margin", new_margin),
                    ("observed_delta", new_margin - old_margin), ("fixed_reference_margin", ref),
                    ("retained_margin_change", old_margin - ref), ("intervened_margin_change", new_margin - ref),
                ):
                    require(Fraction(row[key]) == expected, "exact fixed-reference closure: " + key)
        mapped = profiles[dose]["mapped_all"]
        comparisons[dose] = []
        for c in CONTROLS:
            if c == "mapped_all":
                continue
            other = profiles[dose][c]
            comparisons[dose].append({
                "control": c,
                "operand_word_difference_coordinates": [i for i in range(896)
                                                       if mapped["working_fp16_words"][i] != other["working_fp16_words"][i]],
                "changed_coordinate_symmetric_difference": sorted(
                    set(mapped["changed_coordinates"]) ^ set(other["changed_coordinates"])),
                "rmsnorm_word_difference_coordinates": [i for i in range(896)
                                                       if mapped["rmsnorm_words"][i] != other["rmsnorm_words"][i]],
                "rmsnorm_scalar_differences": {k: mapped["rmsnorm_scalars"][k] - other["rmsnorm_scalars"][k]
                                               for k in ("mean_q48", "root_q24")},
                "selected_logit_words_equal": next(o for o in outputs if o["control"] == c)["selected_logit_words"]
                                              == next(o for o in outputs if o["control"] == "mapped_all")["selected_logit_words"],
            })
    return {**partitions(profiles), "retained_observations": retained,
            "mapped_all_comparisons": comparisons, "activity": activity,
            "retained_baselines": baseline, "fixed_selected_references": selected_refs,
            "retained_reference_closures": {dose: [r for r in (p if dose == "17/512" else q)["report"]["dose_tables"][dose]
                                                  if r["polarity"] == "reverse"] for dose in DOSES},
            "observations": 36, "mapped_all_control_comparisons": 32, "exact_reference_closures": 144}


@contextmanager
def read_only(audit):
    import builtins
    import socket

    old_open, old_io_open, old_os_open = builtins.open, io.open, os.open

    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise IntegrityError("filesystem mutation or external dispatch forbidden")

    def opening(original):
        def guarded(file, mode="r", *args, **kwargs):
            if any(c in mode for c in "wax+"):
                audit["writes"] += 1
                refuse()
            return original(file, mode, *args, **kwargs)
        return guarded

    def os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            audit["writes"] += 1
            refuse()
        return old_os_open(path, flags, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(builtins, "open", opening(old_open)))
        stack.enter_context(patch.object(io, "open", opening(old_io_open)))
        stack.enter_context(patch.object(os, "open", os_open))
        for name in ("mkdir", "remove", "unlink", "rename", "replace", "rmdir", "chmod",
                     "link", "symlink", "truncate", "system", "fork", "posix_spawn", "posix_spawnp"):
            stack.enter_context(patch.object(os, name, refuse))
        stack.enter_context(patch.object(subprocess, "Popen", refuse))
        stack.enter_context(patch.object(socket, "socket", refuse))
        yield


def analyze(bank):
    result = {"diagnostic_id": NAME, "version": 1, "command": COMMAND,
              "dose_search_closed": True, "global_threshold_model_closed": True,
              "preregistration": PREREGISTRATION, "normal_host_review": "REQUIRED",
              "claim_boundary": PREREGISTRATION["boundary"]}
    try:
        q, p, seal = authenticate(bank)
        result.update({key: q[key] for key in PRESERVED})
        result.update(b2_capture_bindings=seal, reviewed_dyadic_receipt=q["reviewed_dyadic_receipt"],
                      preserved_dyadic_rejection_counts=q["preserved_dyadic_rejection_counts"],
                      historical_flags={"b2": q["flags"], "dyadic": p["flags"]},
                      prior_attempts=seal["prior_attempts"])
        report = compare(q, p, bank)
        result.update(report=report, classification=report["classification"],
                      successor=report["successor"], integrity_error=report["integrity_error"],
                      status="UNKNOWN" if report["classification"] == UNKNOWN else "measured")
    except (IntegrityError, OSError, LookupError, TypeError, ValueError, ArithmeticError) as error:
        result.update(status="UNKNOWN", classification=UNKNOWN, successor=None,
                      integrity_error=f"{type(error).__name__}: {error}")
    result["authenticated_pins"] = list(bank.pins.values())
    return result


def run_tests(result):
    import pytest

    class Results:
        collected = executed = errors = failures = skipped = 0

        def pytest_configure(self, config):
            config._ace3_retained_byte_result = result

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
        pins.append({"path": str(path), "bytes": len(raw), "sha256": digest(raw)})
    with read_only(audit):
        result = analyze(BoundBytes())
        frozen = encoded(result)
        tests = run_tests(result)
        require(encoded(result) == frozen, "focused tests mutated retained evidence")
        for pin in pins:
            require(digest(Path(pin["path"]).read_bytes()) == pin["sha256"], "current source/test pin changed")
    require(audit == dict.fromkeys(AUDIT_KEYS, 0), "forbidden dispatch/write census changed")
    tests["compiled"] = pins
    result.update(tests=tests, source_test_pins=pins, dispatch_and_write_audit=audit,
                  flags={"stdout_only": True, "retained_bytes_only": True, "candidate_admitted": False,
                         "rounding_causal_attribution": False, "scalar_threshold_transfer_claim": False,
                         "dose_search_closed": True, "global_threshold_model_closed": True})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", required=True, action="store_true")
    parser.parse_args(argv)
    audit = dict.fromkeys(AUDIT_KEYS, 0)
    try:
        result = check(audit)
    except (IntegrityError, OSError, ValueError, ImportError) as error:
        print(json.dumps({"diagnostic_id": NAME, "status": "UNKNOWN", "classification": UNKNOWN,
                          "integrity_error": f"{type(error).__name__}: {error}", "successor": None,
                          "dose_search_closed": True, "global_threshold_model_closed": True,
                          "dispatch_and_write_audit": audit, "normal_host_review": "REQUIRED"}))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.modules[MODULE] = sys.modules[__name__]
    sys.exit(main())
