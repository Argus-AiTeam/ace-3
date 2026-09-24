"""Stdout-only exact cancellation topology of reviewed modal source mismatches."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_source_cancellation_topology_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_component_source_equivalence_classifier_v1"
CAPTURE_ROOT = ROOT / "build/selected-direct-hidden-modal-sources-86faa4fc8c09-attempt002"


def pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


PINS = {
    "stdout": pin(CAPTURE_ROOT / "check.stdout", 1083632,
                  "3060c9577effd0ed78848cfa22eaf5d7028c43c411bd543ce5b74c5fd420772a"),
    "capture": pin(CAPTURE_ROOT / "capture.json", 33689,
                   "3b66b9e76d0cc484b6aa7fd53fa8654a86aae532e888dc2d2e0dfad09c0ecbc1"),
    "review": pin(Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
                  / "86faa4fc8c09/round-0001.json", 689,
                  "a438ad5923bb9e5b5f27a641ab556720ef51d52b79aa1a75fd57d88bf975472c"),
}
SOURCE_PINS = [
    pin(ROOT / "ace3/model/candidates" / (PARENT + ".py"), 20108,
        "b59b67fc15e3a65fb040dc4ea816d5ecc39d481d4dd1f249f09880b1db9a981e"),
    pin(ROOT / "tests" / ("test_" + PARENT + ".py"), 16627,
        "6d5aadebc1cf7eb69d9af35bbcd7e6b4da733339bbfd4249ebd2ef740a765e94"),
]
SNAPSHOT_PINS = [
    {**binding, "path": str(CAPTURE_ROOT / name)}
    for binding, name in zip(SOURCE_PINS, ("source.snapshot.py", "test.snapshot.py"), strict=True)
]
LABELS = ("branch", "ignored-build", "compile", "pytest", "check")
SUFFIXES = (".command.txt", ".argv.json", ".environment.json", ".stdout",
            ".stderr", ".whole-command.log")
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62", "inherited_native",
)
CANONICAL = CONTROLS[0]
EQUAL_CONTROLS = (CANONICAL, "frozen_inherited_down", "inherited_native")
GROUP_CONTROLS = (
    ("frozen_inherited_o", "frozen_inherited_o_down"),
    ("scratch", "scratch_down"), ("mapped62",),
)
COMPONENTS = (
    "input_hidden", "attention_stage11", "mlp_stage17", "actual_residual_boundary",
    "negative_reference_residual_boundary", "q24_to_fp16_conversion",
    "fp16_to_branch_terminal_remainder",
)
SOURCES = ("actual_residual_boundary", "q24_to_fp16_conversion")
SCOPE_FIELDS = ("table", "left_id", "right_id", "branch")
SCOPES = (
    ("coordinate", 34319, 319, "binary64"),
    ("coordinate", 319, 34319, "binary64"),
    ("coordinate_component", 34319, 13, "binary64"),
)
FACTOR = "weight_times_reference_anchor_times_row_difference"
ANCHOR = "reference_inverse_norm_anchor"
MASS_FIELDS = ("signed", "absolute", "cancellation_absolute_mass")
MOVING_FIELDS = tuple(
    family + "." + field
    for family in ("hidden", "weighted")
    for field in (*SOURCES, "mass.absolute", "mass.cancellation_absolute_mass")
)
COORDINATE_FIELDS = {
    FACTOR, ANCHOR, "direct_hidden_weighted_term",
    *(family + "." + field for family in ("hidden", "weighted")
      for field in (*COMPONENTS, "sum", *("mass." + m for m in MASS_FIELDS))),
}
BOUNDARY = (
    "Descriptive retained-rational cancellation topology only, not a causal, performance, "
    "repair, intervention or admission claim. No producer/operator/native decoder/tensor/"
    "RMSNorm/head/row-dot replay, prefix/admission/original-reference replay, row319 "
    "availability/census/recheck work or closed-branch reopening. Independently propagated "
    "original-input global references, exact thresholds and source/operand/state/KV/lineage "
    "gates remain unchanged; accepted evidence and historical failures are preserved. "
    "The missing row319 896-element pre-round producer and NOT_RETAINED_NO_RECONSTRUCTION "
    "binary64 internal stages remain limitations. Native S16 RTZ and Q24-wide residual "
    "state are not strict-FP16-state W4A16. G128 asymmetric packed INT4 native GEMM nibble "
    "ordering, no qzero plus-one, FP16 scales/operator boundaries/KV remain unchanged. "
    "No new-token/full-model admission, precision/scale/hardware/GPU/RTL/FPGA/ACE2 expansion. "
    "Independent Host Reviewer completion is required, not asserted here."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(actual == expected, message)


def bound_bytes(binding):
    path = Path(binding["path"])
    require(not path.is_symlink(), "symlinked retained artifact: " + str(path))
    data = path.read_bytes()
    same(len(data), binding["bytes"], "retained byte count changed: " + str(path))
    same(hashlib.sha256(data).hexdigest(), binding["sha256"], "retained hash changed: " + str(path))
    return data


def decode(data, *, metadata=False):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(value):
        raise ValueError("non-rational JSON number: " + value)

    options = {} if metadata else {"parse_float": reject}
    return json.loads(data, object_pairs_hook=unique, parse_constant=reject, **options)


def rational(text):
    require(type(text) is str, "exact rational string required")
    value = Fraction(text)
    same(str(value), text, "noncanonical rational")
    return value


def zero_counters(values):
    require(bool(values) and all(v is False or type(v) is int and v == 0
                                for v in values.values()), "forbidden dispatch/write/operator/claim")


@contextmanager
def read_only(audit):
    active = True
    allowed = {
        str(SOURCE), str(TEST),
        *(p["path"] for p in (*PINS.values(), *SOURCE_PINS, *SNAPSHOT_PINS)),
        *(str(CAPTURE_ROOT / (label + suffix)) for label in LABELS for suffix in SUFFIXES),
    }

    def guard(event, args):
        if not active:
            return
        forbidden = False
        if event == "open":
            path, mode, flags = args
            forbidden = (not isinstance(path, (str, bytes, os.PathLike))
                         or os.fsdecode(path) not in allowed
                         or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT |
                                          os.O_TRUNC | os.O_APPEND)))
        elif event in ("import", "exec"):
            forbidden = True
        elif event.startswith(("subprocess.", "socket.", "ctypes.", "shutil.")):
            forbidden = True
        elif event in (
            "os.system", "os.fork", "os.forkpty", "os.exec", "os.posix_spawn",
            "os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.link", "os.symlink",
            "os.truncate", "os.chmod", "os.chown", "os.utime",
        ):
            forbidden = True
        if forbidden:
            audit["forbidden_calls"] += 1
            raise RuntimeError("retained-only cancellation classifier forbids " + event)

    sys.addaudithook(guard)
    try:
        yield
    finally:
        active = False


def authenticate():
    review = decode(bound_bytes(PINS["review"]), metadata=True)
    same((review["kind"], review["mission_id"], review["producer_role"], review["review"]["status"]),
         ("round_reviewed_handoff", "86faa4fc8c09", "reviewer", "done"),
         "independent terminal 86faa review required")
    capture = decode(bound_bytes(PINS["capture"]), metadata=True)
    require(capture["success"] is True and bool(capture["checks"])
            and all(v is True for v in capture["checks"].values()), "retained capture failed")
    same(capture["capture_directory"], str(CAPTURE_ROOT), "capture directory splice")
    preflight = capture["preflight"]
    same((preflight["cwd"], preflight["uid"], preflight["python"]),
         (str(ROOT), 1000, PYTHON), "retained workdir/account/interpreter gate")
    same((preflight["environment"]["PYTHONPATH"], preflight["environment"]["PYTHONDONTWRITEBYTECODE"]),
         (str(ROOT), "1"), "retained Python environment gate")
    same(capture["sources_after"], preflight["sources"], "retained source drift")
    same(capture["sources_after"], SOURCE_PINS, "retained source/test splice")
    same(capture["accepted_artifacts_after"], preflight["accepted_artifacts"],
         "retained accepted-artifact overwrite")
    for live, snapshot in zip(SOURCE_PINS, SNAPSHOT_PINS, strict=True):
        same(bound_bytes(live), bound_bytes(snapshot), "retained source snapshot splice")
    same([r["label"] for r in capture["results"]], list(LABELS), "retained command census")
    stdout = None
    for result in capture["results"]:
        require(type(result["exit_status"]) is int and result["exit_status"] == 0
                and result["timed_out"] is False, "retained command failure")
        label = result["label"]
        same([p["path"] for p in result["files"]],
             [str(CAPTURE_ROOT / (label + s)) for s in SUFFIXES], "capture member splice")
        command, argv, environment, output, error, whole = [bound_bytes(p) for p in result["files"]]
        same(command, (result["command"] + "\n").encode(), "command binding")
        same(decode(argv), result["argv"], "argv binding")
        same(decode(environment), {k: preflight[k] for k in
                                  ("cwd", "uid", "python", "python_version", "environment")},
             "environment binding")
        same(error, b"", "retained stderr")
        same(whole, b"COMMAND\n" + command + b"ENVIRONMENT\n" + environment
             + b"\nSTDOUT\n" + output + b"\nSTDERR\n" + error
             + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "whole-command binding")
        if label == "branch":
            same(output, b"argus/full-projection\n", "retained branch")
        if label == "check":
            module = "ace3.model.candidates." + PARENT
            same(result["command"],
                 f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {module} --check",
                 "retained check command")
            same(decode(argv), [PYTHON, "-B", "-m", module, "--check"], "retained check argv")
            same(result["files"][3], PINS["stdout"], "capture/stdout splice")
            stdout = output
    require(stdout is not None, "missing retained stdout")
    retained = decode(stdout, metadata=True)
    same(retained["command"], capture["results"][-1]["command"], "stdout command splice")
    same(retained["compiled_sources"], SOURCE_PINS, "compiled source/test splice")
    return retained, capture, review


def spread(values):
    same(set(values), set(CONTROLS), "exact eight-control scalar census required")
    parsed = {c: rational(values[c]) for c in CONTROLS}
    low, high = min(parsed.values()), max(parsed.values())
    deltas = {c: str(v - parsed[CANONICAL]) for c, v in parsed.items()}
    mismatches = [
        {"control": c, "canonical_control": CANONICAL, "canonical": values[CANONICAL],
         "retained": values[c], "delta": deltas[c]}
        for c, v in parsed.items() if v != parsed[CANONICAL]
    ]
    return {
        "values": values, "deltas_from_canonical": deltas,
        "minimum": str(low), "maximum": str(high), "spread": str(high - low),
        "minimum_controls": [c for c, v in parsed.items() if v == low],
        "maximum_controls": [c for c, v in parsed.items() if v == high],
        "extrema_cover_all_controls": True, "mismatch_rows": mismatches,
        "unknown_controls": [], "integrity_status": "REJECTED" if mismatches else "SUPPORTED",
    }


def row_order(rows):
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, allow_nan=False))


def negated_field(name):
    return (name in (FACTOR, "direct_hidden_weighted_term")
            or name.startswith("weighted.") and name not in
            ("weighted.mass.absolute", "weighted.mass.cancellation_absolute_mass"))


def no_float(value):
    require(not isinstance(value, float), "floating-point report scalar forbidden")
    if isinstance(value, dict):
        for child in value.values():
            no_float(child)
    elif isinstance(value, list):
        for child in value:
            no_float(child)


def report(retained):
    same((retained["diagnostic_id"], retained["status"]),
         (PARENT, "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_COMPONENT_SOURCE_EQUIVALENCE_CLASSIFIER"),
         "wrong retained classifier/boundary")
    require(type(retained["version"]) is int and retained["version"] == 1, "version changed")
    zero_counters(retained["flags"])
    zero_counters(retained["dispatch_and_write_audit"])
    parent = retained["report"]
    no_float(parent)
    same((parent["integrity_status"], len(parent["mismatch_rows"]), parent["unknown_rows"]),
         ("REJECTED", 80, []), "expected input equivalence REJECTED/80/0")
    same(tuple(parent[k] for k in ("unstable_partition_count", "modal_control_count",
                                  "coordinate_account_count", "control_coordinate_account_count",
                                  "control_component_account_count")), (3, 8, 5, 40, 280),
         "retained domain census")
    same(parent["canonical_control"], CANONICAL, "canonical control changed")
    require(parent["descriptive_accounting_only"] is True, "descriptive boundary changed")
    same(parent["separate_weight_and_row_difference_equality"], "UNKNOWN",
         "unretained separate factors must remain UNKNOWN")
    same([tuple(p[k] for k in SCOPE_FIELDS) for p in parent["partitions"]],
         list(SCOPES), "exact three retained partitions required")
    failures, accounts, computed_mismatches = [], [], []
    check_count = 0

    def verify(label, actual, expected, **identity):
        nonlocal check_count
        check_count += 1
        if actual != expected:
            failures.append({**identity, "field": label, "actual": actual, "expected": expected})

    for index, partition in enumerate(parent["partitions"]):
        location = {k: partition[k] for k in SCOPE_FIELDS}
        same(partition["modal_controls"], list(CONTROLS), "modal membership changed")
        same(partition["canonical_control"], CANONICAL, "partition canonical changed")
        same(partition["unknown_rows"], [], "unknown retained partition")
        same([c["coordinate"] for c in partition["coordinates"]],
             [62, 241] if index < 2 else [241], "coordinate census changed")
        fields = {}
        for coordinate in partition["coordinates"]:
            same(set(coordinate["field_spreads"]), COORDINATE_FIELDS, "coordinate field census")
            fields.update({f"coordinate.{coordinate['coordinate']}.{k}": v
                           for k, v in coordinate["field_spreads"].items()})
        for family, prefix in (("candidate_spreads", "candidate."), ("parent_comparator_spreads", "parent.")):
            require(bool(partition[family]), "missing retained comparator fields")
            fields.update({prefix + k: v for k, v in partition[family].items()})
        mismatches = []
        for name, field in fields.items():
            fresh = spread(field["values"])
            verify("retained_spread." + name, field, fresh, **location)
            mismatches.extend({"field": name, **r} for r in fresh["mismatch_rows"])
        verify("partition_mismatch_rows", row_order(partition["mismatch_rows"]),
               row_order(mismatches), **location)
        verify("partition_equivalence", partition["integrity_status"],
               "REJECTED" if mismatches else "SUPPORTED", **location)
        verify("partition_mismatch_count", len(mismatches), 40 if index < 2 else 0, **location)
        computed_mismatches.extend({**location, **r} for r in mismatches)
        same(len(partition["closure_checks"]), 408 if index < 2 else 280, "retained closure census")
        for closure in partition["closure_checks"]:
            verify("retained_closure", closure["actual"], closure["expected"],
                   **location, retained_field=closure["field"])
            verify("retained_closure_status", closure["integrity_status"], "SUPPORTED", **location)
        for coordinate in partition["coordinates"]:
            coord = coordinate["coordinate"]
            field_spreads = coordinate["field_spreads"]
            profiles = []
            for control in CONTROLS:
                identity = {**location, "coordinate": coord, "control": control}
                values = {k: rational(a["values"][control]) for k, a in field_spreads.items()}
                deltas = {k: str(v - rational(field_spreads[k]["values"][CANONICAL]))
                          for k, v in values.items()}
                for component in COMPONENTS:
                    verify("component_product." + component, str(values["weighted." + component]),
                           str(values[FACTOR] * values["hidden." + component]), **identity)
                for family in ("hidden", "weighted"):
                    signed = sum((values[family + "." + c] for c in COMPONENTS), Fraction(0))
                    absolute = sum((abs(values[family + "." + c]) for c in COMPONENTS), Fraction(0))
                    for field, expected in (("sum", signed), ("mass.signed", signed),
                                            ("mass.absolute", absolute),
                                            ("mass.cancellation_absolute_mass", absolute - abs(signed))):
                        verify(family + "." + field, str(values[family + "." + field]),
                               str(expected), **identity)
                    pair_sum = sum((rational(deltas[family + "." + s]) for s in SOURCES), Fraction(0))
                    verify(family + ".source_delta_pair_sum", str(pair_sum), "0", **identity)
                verify("direct_hidden_closure", str(values["weighted.sum"]),
                       str(values["direct_hidden_weighted_term"]), **identity)
                verify("combined_factor_closure", str(values[FACTOR] * values["hidden.sum"]),
                       str(values["weighted.sum"]), **identity)
                moving = coord == 62 and index < 2 and control not in EQUAL_CONTROLS
                verify("moving_field_census", sorted(k for k, v in deltas.items() if v != "0"),
                       sorted(MOVING_FIELDS) if moving else [], **identity)
                if coord == 62:
                    h = rational(deltas["hidden." + SOURCES[0]])
                    verify("actual_source_positive", values["hidden." + SOURCES[0]] > 0, True, **identity)
                    verify("conversion_source_negative", values["hidden." + SOURCES[1]] < 0, True, **identity)
                    verify("hidden_absolute_movement", deltas["hidden.mass.absolute"], str(2 * h), **identity)
                    for family in ("hidden", "weighted"):
                        verify(family + ".cancellation_movement",
                               deltas[family + ".mass.cancellation_absolute_mass"],
                               deltas[family + ".mass.absolute"], **identity)
                    verify("weighted_absolute_movement", deltas["weighted.mass.absolute"],
                           str(abs(values[FACTOR]) * rational(deltas["hidden.mass.absolute"])), **identity)
                profiles.append({"control": control, "field_deltas": deltas,
                                 "hidden_source_delta_sum": str(sum(
                                     (rational(deltas["hidden." + s]) for s in SOURCES), Fraction(0))),
                                 "weighted_source_delta_sum": str(sum(
                                     (rational(deltas["weighted." + s]) for s in SOURCES), Fraction(0)))})
            groups = []
            if coord == 62:
                by_signature = {}
                for profile in profiles:
                    if profile["control"] in EQUAL_CONTROLS:
                        continue
                    signature = tuple(profile["field_deltas"][k] for k in MOVING_FIELDS)
                    by_signature.setdefault(signature, []).append(profile["control"])
                groups = [{"controls": members, "signature": dict(zip(MOVING_FIELDS, signature, strict=True))}
                          for signature, members in by_signature.items()]
                verify("exact_signature_membership", [g["controls"] for g in groups],
                       [list(g) for g in GROUP_CONTROLS], **location)
            accounts.append({**location, "coordinate": coord, "controls": profiles,
                             "signature_groups": groups,
                             "canonical_values": {k: a["values"][CANONICAL] for k, a in field_spreads.items()}})
    verify("global_mismatch_rows", row_order(parent["mismatch_rows"]), row_order(computed_mismatches))
    reverse = parent["reversed_pair"]
    same(len(reverse["checks"]), 616, "retained reversed-pair census")
    verify("retained_reverse_status", reverse["integrity_status"], "SUPPORTED")
    verify("retained_reverse_mismatches", reverse["mismatch_rows"], [])
    verify("retained_reverse_unknowns", reverse["unknown_rows"], [])
    for closure in reverse["checks"]:
        verify("retained_reverse_closure", closure["actual"], closure["expected"])
        verify("retained_reverse_closure_status", closure["integrity_status"], "SUPPORTED")
    for left, right in zip(accounts[:2], accounts[2:4], strict=True):
        same(left["coordinate"], right["coordinate"], "reversed coordinate identity")
        for name, value in left["canonical_values"].items():
            verify("reversed_canonical." + name, right["canonical_values"][name],
                   str(-rational(value)) if negated_field(name) else value,
                   coordinate=left["coordinate"])
        for forward, backward in zip(left["controls"], right["controls"], strict=True):
            same(forward["control"], backward["control"], "reversed control identity")
            for name, delta in forward["field_deltas"].items():
                verify("reversed_delta." + name, backward["field_deltas"][name],
                       str(-rational(delta)) if negated_field(name) else delta,
                       control=forward["control"], coordinate=left["coordinate"])
    accounted, unaccounted = [], []
    for row in computed_mismatches:
        scope = tuple(row[k] for k in SCOPE_FIELDS)
        name = row["field"].removeprefix("coordinate.62.")
        if (scope in SCOPES[:2] and row["field"].startswith("coordinate.62.")
                and name in MOVING_FIELDS and row["control"] not in EQUAL_CONTROLS):
            accounted.append({**row, "classification": "absolute_cancellation_mass_redistribution"
                              if ".mass." in name else "equal_opposite_source_redistribution"})
        else:
            unaccounted.append(row)
    verify("accounted_mismatch_count", len(accounted), 80)
    verify("unaccounted_mismatches", unaccounted, [])
    return {
        "input_equivalence_status": parent["integrity_status"],
        "input_mismatch_count": len(parent["mismatch_rows"]), "input_unknown_count": 0,
        "cancellation_topology_status": "REJECTED" if failures else "SUPPORTED",
        "accounted_mismatch_count": len(accounted), "accounted_mismatch_rows": accounted,
        "unaccounted_mismatch_rows": unaccounted, "failure_count": len(failures), "failures": failures,
        "exact_check_count": check_count, "coordinate_accounts": accounts,
        "equal_controls": list(EQUAL_CONTROLS), "mismatching_control_count": 5,
        "signature_group_count_per_coordinate62_partition": [len(a["signature_groups"])
                                                             for a in accounts if a["coordinate"] == 62],
        "arithmetic": "canonical integer-rational strings; no float",
        "descriptive_accounting_only": True,
        "separate_weight_and_row_difference_equality": parent["separate_weight_and_row_difference_equality"],
        "lineage_separation": parent["lineage_separation"], "reference_scope": parent["reference_scope"],
        "claim_boundary": BOUNDARY,
    }


def check():
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        retained, capture, review = authenticate()
        result = report(retained)
        compiled = []
        for path in (SOURCE, TEST):
            data = path.read_bytes()
            compile(data, str(path), "exec")
            compiled.append(pin(path, len(data), hashlib.sha256(data).hexdigest()))
        flags = dict(retained["flags"])
        audit = {**retained["dispatch_and_write_audit"], **audit}
        zero_counters(audit)
        zero_counters(flags)
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_SOURCE_CANCELLATION_TOPOLOGY_CLASSIFIER",
        "command": COMMAND, "compiled_sources": compiled,
        "input_pins": PINS, "retained_source_pins": SOURCE_PINS, "retained_snapshot_pins": SNAPSHOT_PINS,
        "retained_review": review, "retained_capture_checks": capture["checks"],
        "report": result, "flags": flags, "dispatch_and_write_audit": audit,
        "normal_host_review": "Independent Reviewer completion required; not asserted by Engineer.",
        "claim_boundary": BOUNDARY,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
