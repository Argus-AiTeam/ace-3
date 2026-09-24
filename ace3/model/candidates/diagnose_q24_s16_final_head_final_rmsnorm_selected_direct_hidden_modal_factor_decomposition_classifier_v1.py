"""Stdout-only factor decomposition of authenticated retained modal accounts."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_factor_decomposition_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
MODAL_NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_source_cancellation_topology_classifier_v1"
RAW_NAME = "diagnose_q24_s16_final_head_final_rmsnorm_direct_hidden_residual_margin_bridge_v1"
MODAL_ROOT = ROOT / "build/selected-direct-hidden-modal-source-cancellation-184a3f76ac61-attempt001"
RAW_ROOT = ROOT / "build/direct-hidden-residual-50f5bb2062c9-bc457u29"
REVIEWS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")


def pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


PINS = {
    "modal_stdout": pin(MODAL_ROOT / "check.stdout", 99306,
                        "40ca35bcd97cba89677aafa3d03352b845b0660e2cd34adc1dd1cb1c712f878a"),
    "modal_capture": pin(MODAL_ROOT / "capture.json", 33082,
                         "1082fdf16a0b38b25da89be536da5691db1e5abbca48825609330a79950c8534"),
    "modal_source": pin(ROOT / "ace3/model/candidates" / (MODAL_NAME + ".py"), 26031,
                        "be31bb41982c2ffde99dc1f5177cafebab0919c9c6358b111e00363139f3ee98"),
    "modal_test": pin(ROOT / "tests" / ("test_" + MODAL_NAME + ".py"), 18743,
                      "a13b67f0243b1646ae04b9fa6f2038a85a248d42d3fd4be4897d47ebc8ad5495"),
    "modal_review": pin(REVIEWS / "184a3f76ac61/round-0001.json", 690,
                        "f60cf8e3819f23dce7bde187072d1752da719eb522e42d129807cb7baaf9d264"),
    "raw_stdout": pin(RAW_ROOT / "stdout.json", 6737063,
                      "30790957680088caecfa2539e10e672f0d5ff373573d4c80a71cb3be2d65cdeb"),
    "raw_whole": pin(RAW_ROOT / "whole-command.log", 6740726,
                     "4d6c225bfde2442bb91aa2ecbaab947b361d26a3262524449c558a424a8f85d8"),
    "raw_capture": pin(RAW_ROOT / "capture.json", 7652,
                       "f29d226f2bc4e99d967b89b9f8baeeba7c71c555be742f58367eefcda3dad464"),
    "raw_review": pin(REVIEWS / "50f5bb2062c9/round-0001.json", 689,
                      "feb58e79b284e652e9bb028a2ea79b3f4b48a6b9fd18a58f0465367a6a204bb8"),
}
LABELS = ("branch", "ignored-build", "compile", "pytest", "check")
SUFFIXES = (".command.txt", ".argv.json", ".environment.json", ".stdout",
            ".stderr", ".whole-command.log")
RAW_MEMBERS = ("command.txt", "preflight.json", "runner-command.txt", "stderr.log",
               "stdout.json", "whole-command.log")
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62", "inherited_native",
)
CANONICAL = CONTROLS[0]
SCOPE_FIELDS = ("table", "left_id", "right_id", "branch", "coordinate")
SCOPES = (
    ("coordinate", 34319, 319, "binary64", 62),
    ("coordinate", 34319, 319, "binary64", 241),
    ("coordinate", 319, 34319, "binary64", 62),
    ("coordinate", 319, 34319, "binary64", 241),
    ("coordinate_component", 34319, 13, "binary64", 241),
)
FACTOR = "weight_times_reference_anchor_times_row_difference"
ANCHOR = "reference_inverse_norm_anchor"
FACTORS = ("norm_weight", ANCHOR, "row_difference")
COMPONENTS = (
    "input_hidden", "attention_stage11", "mlp_stage17", "actual_residual_boundary",
    "negative_reference_residual_boundary", "q24_to_fp16_conversion",
    "fp16_to_branch_terminal_remainder",
)
BOUNDARY = (
    "Retained exact-rational factor accounting only, not a causal, performance, repair, "
    "intervention or admission claim. No prefix/admission/original-reference or accepted "
    "producer replay; no native decoder/operator/RMSNorm/head/row-dot execution. No row319 "
    "availability/census/recheck or reconstruction of its missing 896-element pre-round "
    "producer; no closed branch reopened. Independently propagated original-input global "
    "references, exact thresholds, source/operand/state/KV/lineage gates and historical "
    "failures remain unchanged. Binary64 internal stages remain "
    "NOT_RETAINED_NO_RECONSTRUCTION. Native S16 RTZ/Q24-wide residual state does not establish "
    "strict-FP16-state W4A16. G128 asymmetric packed INT4 native GEMM nibble ordering, no "
    "qzero plus-one, FP16 scales/operator boundaries/KV remain unchanged. No precision/scale/"
    "hardware/GPU/RTL/FPGA/ACE2 expansion or new-token/full-model admission. Independent "
    "Host Reviewer completion is required, not asserted here."
)
COUNTERS = (
    "forbidden_calls", "evidence_writes", "prefix_dispatch", "admission_dispatch",
    "reference_producer_dispatch", "accepted_producer_replay", "native_dispatch",
    "decoder_dispatch", "final_rmsnorm_invocations", "lm_head_invocations",
    "local_exact_row_dot_computations", "local_operator_replay", "external_invocations",
    "GPU_dispatch", "RTL_dispatch", "hardware_dispatch", "row319_availability_rechecks",
    "precision_or_scale_expansion", "new_token_claim", "full_model_claim",
)
INPUT_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(actual == expected, message)


def decode(data, *, metadata=False):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(value):
        raise ValueError("non-rational JSON number: " + value)

    return json.loads(data, object_pairs_hook=unique, parse_constant=reject,
                      **({} if metadata else {"parse_float": reject}))


def rational(text):
    require(type(text) is str, "canonical rational string required")
    value = Fraction(text)
    same(str(value), text, "noncanonical rational: " + text)
    return value


def bound_bytes(binding):
    path = Path(binding["path"])
    require(not path.is_symlink(), "symlinked retained artifact: " + str(path))
    data = path.read_bytes()
    same(len(data), binding["bytes"], "retained byte count changed: " + str(path))
    same(hashlib.sha256(data).hexdigest(), binding["sha256"],
         "retained hash changed: " + str(path))
    return data


def zero_counters(values):
    require(bool(values) and all(v is False or type(v) is int and v == 0
                                for v in values.values()), "forbidden counter/claim")


def terminal_review(review, mission):
    same((review["kind"], review["mission_id"], review["producer_role"], review["review"]["status"]),
         ("round_reviewed_handoff", mission, "reviewer", "done"),
         "independent terminal review identity/status")


def allowed_paths():
    return {
        str(SOURCE), str(TEST), str(Path(sys.executable).resolve()),
        *(p["path"] for p in PINS.values()),
        *(str(MODAL_ROOT / (label + suffix)) for label in LABELS for suffix in SUFFIXES),
        *(str(MODAL_ROOT / name) for name in ("source.snapshot.py", "test.snapshot.py")),
        *(str(RAW_ROOT / name) for name in RAW_MEMBERS),
    }


class ForbiddenOperation(RuntimeError):
    pass


@contextmanager
def read_only(audit):
    active = True
    allowed = allowed_paths()

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
            "os.truncate", "os.chmod", "os.chown", "os.utime", "os.listdir", "os.scandir",
        ):
            forbidden = True
        if forbidden:
            audit["forbidden_calls"] += 1
            raise ForbiddenOperation("retained-only factor classifier forbids " + event)

    sys.addaudithook(guard)
    try:
        yield
    finally:
        active = False


def authenticate():
    data = {key: bound_bytes(binding) for key, binding in PINS.items()}
    reviews = {key: decode(data[key], metadata=True) for key in ("modal_review", "raw_review")}
    terminal_review(reviews["modal_review"], "184a3f76ac61")
    terminal_review(reviews["raw_review"], "50f5bb2062c9")
    modal = decode(data["modal_stdout"], metadata=True)
    raw = decode(data["raw_stdout"], metadata=True)
    capture = decode(data["modal_capture"], metadata=True)
    require(capture["success"] is True and bool(capture["checks"])
            and all(v is True for v in capture["checks"].values()), "modal capture failed")
    same(capture["capture_directory"], str(MODAL_ROOT), "modal capture directory")
    preflight = capture["preflight"]
    same((preflight["cwd"], preflight["uid"], preflight["python"]),
         (str(ROOT), 1000, PYTHON), "modal workdir/account/interpreter")
    same((preflight["environment"]["PYTHONPATH"], preflight["environment"]["PYTHONDONTWRITEBYTECODE"]),
         (str(ROOT), "1"), "modal command environment")
    sources = [PINS["modal_source"], PINS["modal_test"]]
    same(capture["sources_after"], sources, "modal source/test pins")
    same(preflight["sources"], sources, "modal source/test drift")
    same(modal["compiled_sources"], sources, "modal compiled sources")
    same(capture["accepted_artifacts_after"], preflight["accepted_artifacts"],
         "modal accepted-artifact drift")
    for key, name in (("modal_source", "source.snapshot.py"), ("modal_test", "test.snapshot.py")):
        same(bound_bytes({**PINS[key], "path": str(MODAL_ROOT / name)}), data[key],
             "modal retained snapshot splice")
    same([r["label"] for r in capture["results"]], list(LABELS), "modal command census")
    for result in capture["results"]:
        require(type(result["exit_status"]) is int and result["exit_status"] == 0
                and result["timed_out"] is False, "modal command failure")
        label = result["label"]
        same([p["path"] for p in result["files"]],
             [str(MODAL_ROOT / (label + s)) for s in SUFFIXES], "modal capture member splice")
        command, argv, environment, output, error, whole = [bound_bytes(p) for p in result["files"]]
        same(command, (result["command"] + "\n").encode(), "modal command bytes")
        same(decode(argv), result["argv"], "modal argv")
        same(decode(environment), {k: preflight[k] for k in
                                  ("cwd", "uid", "python", "python_version", "environment")},
             "modal environment binding")
        same(error, b"", "modal stderr")
        same(whole, b"COMMAND\n" + command + b"ENVIRONMENT\n" + environment
             + b"\nSTDOUT\n" + output + b"\nSTDERR\n" + error
             + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "modal whole command bytes")
        if label == "branch":
            same(output, b"argus/full-projection\n", "modal branch")
        if label == "check":
            expected = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m ace3.model.candidates.{MODAL_NAME} --check"
            same(result["command"], expected, "modal disclosed command")
            same(result["argv"], [PYTHON, "-B", "-m", "ace3.model.candidates." + MODAL_NAME, "--check"],
                 "modal check argv")
            same(modal["command"], expected, "modal stdout command")
            same(output, data["modal_stdout"], "modal stdout splice")
    raw_capture = decode(data["raw_capture"], metadata=True)
    require(type(raw_capture["returncode"]) is int and raw_capture["returncode"] == 0,
            "50f exit failure")
    require(bool(raw_capture["checks"]) and all(v is True for v in raw_capture["checks"].values()),
            "50f capture failure")
    same(raw_capture["workspace"], str(RAW_ROOT), "50f workspace")
    same(set(raw_capture["files"]), set(RAW_MEMBERS), "50f capture census")
    members = {
        name: bound_bytes(pin(RAW_ROOT / name, p["size_bytes"], p["sha256"]))
        for name, p in raw_capture["files"].items()
    }
    same(members["stdout.json"], data["raw_stdout"], "50f stdout splice")
    same(members["whole-command.log"], data["raw_whole"], "50f whole-command splice")
    same(members["whole-command.log"], members["stderr.log"] + members["stdout.json"],
         "50f historical stderr/stdout framing")
    expected = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m ace3.model.candidates.{RAW_NAME} --check"
    same(members["command.txt"], (expected + "\n").encode(), "50f command bytes")
    same(raw["command"], expected, "50f stdout command")
    same((raw["tests"]["executed"], raw["tests"]["failures"], raw["tests"]["errors"],
          raw["tests"]["skipped"]), (22, 0, 0, 0), "50f historical tests")
    same(raw_capture["source_test_pins_after"],
         {p["path"]: {"size_bytes": p["bytes"], "sha256": p["sha256"]}
          for p in raw["tests"]["compiled"]}, "50f source/test capture splice")
    for payload, name, status in (
        (modal, MODAL_NAME, "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_SOURCE_CANCELLATION_TOPOLOGY_CLASSIFIER"),
        (raw, RAW_NAME, "READ_ONLY_FINAL_RMSNORM_DIRECT_HIDDEN_RESIDUAL_MARGIN_BRIDGE"),
    ):
        same((payload["diagnostic_id"], payload["status"], payload["version"]),
             (name, status, 1), "retained diagnostic identity")
        zero_counters(payload["flags"])
        zero_counters(payload["dispatch_and_write_audit"])
    return modal, raw, {"status": "AUTHENTICATED", "input_pins": PINS, "reviews": reviews,
                        "modal_capture_checks": capture["checks"],
                        "raw_capture_checks": raw_capture["checks"],
                        "historical_50f_stderr_preserved_bytes": len(members["stderr.log"])}


def unique_row(rows, field, value, label):
    matches = [(i, r) for i, r in enumerate(rows) if r[field] == value]
    same(len(matches), 1, "missing/ambiguous retained " + label)
    return matches[0]


def selected_row(report, control, left, right, coordinate, prefix):
    ci, c = unique_row(report["controls"], "control", control, "control")
    pairs = [(i, p) for i, p in enumerate(c["pairs"])
             if (p["left_id"], p["right_id"]) == (left, right)]
    same(len(pairs), 1, "missing/ambiguous retained ordered pair")
    pi, pair = pairs[0]
    branch = pair["branches"]["binary64"]
    same(branch["hidden_reference"], "original_input_L23_binary64", "original hidden reference")
    ri, row = unique_row(branch["selected_coordinates"], "coordinate", coordinate, "coordinate")
    pointer = f"{prefix}/controls/{ci}/pairs/{pi}/branches/binary64/selected_coordinates/{ri}"
    return row, pointer


def modal_fields(row):
    fields = {key: row[key] for key in (FACTOR, ANCHOR, "direct_hidden_weighted_term")}
    for family in ("hidden", "weighted"):
        same(set(row[family + "_components"]), set(COMPONENTS), "component census")
        fields.update({family + "." + k: v for k, v in row[family + "_components"].items()})
        fields.update({family + ".mass." + k: v for k, v in row[family + "_component_mass"].items()})
        fields[family + ".sum"] = row[family + "_component_mass"]["signed"]
    return fields


def _classify(modal, raw):
    parent = modal["report"]
    same((parent["cancellation_topology_status"], parent["failure_count"], parent["input_unknown_count"]),
         ("SUPPORTED", 0, 0), "parent cancellation closure failure")
    same((parent["input_equivalence_status"], parent["input_mismatch_count"],
          parent["accounted_mismatch_count"]), ("REJECTED", 80, 80), "parent historical result")
    same(parent["unaccounted_mismatch_rows"], [], "unaccounted parent mismatches")
    same([tuple(a[k] for k in SCOPE_FIELDS) for a in parent["coordinate_accounts"]],
         list(SCOPES), "modal coordinate/partition census")
    report = raw["report"]
    bridge = report["retained_final_rmsnorm_logit_margin_bridge"]
    accounts, mismatches = [], []
    for ai, account in enumerate(parent["coordinate_accounts"]):
        same([c["control"] for c in account["controls"]], list(CONTROLS), "modal control domain")
        identity = dict(zip(SCOPE_FIELDS, SCOPES[ai], strict=True))
        canonical = None
        rows = []
        for mi, control in enumerate(account["controls"]):
            label = control["control"]
            row, pointer = selected_row(report, label, identity["left_id"], identity["right_id"],
                                        identity["coordinate"], "/report")
            detail, detail_pointer = selected_row(
                bridge, label, identity["left_id"], identity["right_id"], identity["coordinate"],
                "/report/retained_final_rmsnorm_logit_margin_bridge")
            fields = modal_fields(row)
            same(set(account["canonical_values"]), set(fields), "modal canonical field census")
            same(set(control["field_deltas"]), set(fields), "modal delta field census")
            retained = {
                k: rational(account["canonical_values"][k]) + rational(control["field_deltas"][k])
                for k in fields
            }
            same({k: rational(v) for k, v in fields.items()}, retained,
                 "184/50f coordinate account binding closure")
            same(control["field_deltas"][FACTOR], "0", "parent combined factor changed")
            require(row["exact_direct_hidden_identity"] is True
                    and detail["exact_weighted_identity"] is True, "retained selected identity")
            same(detail["weighted_terms"]["direct_hidden"], row["direct_hidden_weighted_term"],
                 "50f nested direct-hidden binding")
            weight = rational(detail["hidden_bridge"]["weight"])
            anchor = rational(row[ANCHOR])
            difference = rational(detail["row_difference"])
            left, right = rational(detail["left_weight"]), rational(detail["right_weight"])
            same(left - right, difference, "tied-row subtraction closure")
            values = (weight, anchor, difference)
            product = weight * anchor * difference
            same(product, retained[FACTOR], "factor product closure to 184/50f")
            if canonical is None:
                same(label, CANONICAL, "canonical comparison control")
                canonical = values
            deltas = tuple(v - c for v, c in zip(values, canonical, strict=True))
            # Ordered telescoping identity, valid even when a factor is zero.
            terms = (deltas[0] * canonical[1] * canonical[2],
                     weight * deltas[1] * canonical[2], weight * anchor * deltas[2])
            same(sum(terms), product - canonical[0] * canonical[1] * canonical[2],
                 "factor-delta product closure")
            same(sum(terms), 0, "unchanged modal combined product")
            changed = [name for name, delta in zip(FACTORS, deltas, strict=True) if delta]
            if changed:
                mismatches.append({**identity, "control": label, "changed_factors": changed})
            rows.append({
                "control": label, "factors": dict(zip(FACTORS, map(str, values), strict=True)),
                "left_weight": str(left), "right_weight": str(right),
                FACTOR: str(product), "retained_184_factor": str(retained[FACTOR]),
                "retained_50f_factor": row[FACTOR], "product_closure": True,
                "factor_deltas_from_frozen_inherited": dict(zip(FACTORS, map(str, deltas), strict=True)),
                "ordered_telescoping_product_delta_terms": dict(zip(FACTORS, map(str, terms), strict=True)),
                "product_delta": str(sum(terms)), "changed_factors": changed,
                "decision": "REJECTED" if changed else "SUPPORTED",
                "retained_bindings": {
                    "modal_stdout_pin": "modal_stdout",
                    "modal_account_pointer": f"/report/coordinate_accounts/{ai}",
                    "modal_control_pointer": f"/report/coordinate_accounts/{ai}/controls/{mi}",
                    "raw_stdout_pin": "raw_stdout", "raw_selected_row_pointer": pointer,
                    "raw_factor_row_pointer": detail_pointer,
                },
            })
        accounts.append({**identity, "canonical_control": CANONICAL, "controls": rows})
    for forward, reverse in ((accounts[0], accounts[2]), (accounts[1], accounts[3])):
        for f, r in zip(forward["controls"], reverse["controls"], strict=True):
            same(f["control"], r["control"], "reverse control")
            for factor in FACTORS[:2]:
                same(f["factors"][factor], r["factors"][factor], "reverse weight/anchor")
            same(rational(f["factors"]["row_difference"]),
                 -rational(r["factors"]["row_difference"]), "reverse row orientation")
            same(f["left_weight"], r["right_weight"], "reverse left/right binding")
            same(f["right_weight"], r["left_weight"], "reverse right/left binding")
            same(rational(f[FACTOR]), -rational(r[FACTOR]), "reverse product orientation")
    return {
        "decision": "REJECTED" if mismatches else "SUPPORTED", "unknown_reasons": [],
        "canonical_control": CANONICAL, "modal_controls": list(CONTROLS),
        "coordinate_account_count": len(accounts), "bound_control_coordinate_count": 40,
        "individual_factor_field_count": 120, "product_closure_count": 40,
        "coordinate62_bound_row_count": 16, "reversed_pair_row_checks": 16,
        "individual_factor_equality": not mismatches, "compensating_rows": mismatches,
        "coordinate_accounts": accounts, "arithmetic": "canonical integer-rational strings; no float",
        "factor_delta_identity": "(w-w0)*a0*d0 + w*(a-a0)*d0 + w*a*(d-d0) = w*a*d-w0*a0*d0",
        "reference_scope": report["reference_scope"], "lineage_separation": report["lineage_separation"],
        "parent_source_equivalence_status_preserved": parent["input_equivalence_status"],
        "claim_boundary": BOUNDARY,
    }


def unknown(error, phase):
    return {"decision": "UNKNOWN", "unknown_reasons": [
        {"phase": phase, "error_type": type(error).__name__, "message": str(error)}
    ], "claim_boundary": BOUNDARY}


def classify(modal, raw):
    try:
        return _classify(modal, raw)
    except INPUT_ERRORS as error:
        return unknown(error, "retained_field_or_closure")


def check():
    audit = dict.fromkeys(COUNTERS, 0)
    authentication = {"status": "UNKNOWN", "input_pins": PINS}
    source_pins = []
    runtime = {"python": sys.executable, "version": sys.version, "implementation": sys.implementation.name,
               "cwd": str(Path.cwd()), "uid": os.getuid(),
               "environment": {k: os.environ.get(k) for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")}}
    try:
        with read_only(audit):
            same((runtime["cwd"], runtime["uid"], sys.executable), (str(ROOT), 1000, PYTHON),
                 "current workdir/account/interpreter gate")
            same(runtime["environment"], {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
                 "current command-local environment gate")
            for path in (SOURCE, TEST, Path(sys.executable).resolve()):
                data = path.read_bytes()
                binding = pin(path, len(data), hashlib.sha256(data).hexdigest())
                if path in (SOURCE, TEST):
                    source_pins.append(binding)
                else:
                    runtime["executable_pin"] = binding
            modal, raw, authentication = authenticate()
            result = classify(modal, raw)
            zero_counters(audit)
    except (*INPUT_ERRORS, ForbiddenOperation) as error:
        result = unknown(error, "authentication_or_runtime")
        authentication["error"] = result["unknown_reasons"][0]
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_FACTOR_DECOMPOSITION_CLASSIFIER",
        "decision": result["decision"], "command": COMMAND, "source_test_pins": source_pins,
        "runtime": runtime, "artifact_authentication": authentication, "report": result,
        "dispatch_and_write_audit": audit, "claim_boundary": BOUNDARY,
        "normal_host_review": "Independent Reviewer required; Engineer does not assert completion.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    result = check()
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 2 if result["decision"] == "UNKNOWN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
