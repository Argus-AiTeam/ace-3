"""Stdout-only exact modal/exception contrasts from reviewed 6184 retained rows."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_hotspot_modal_exception_contrast_profiler_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT_NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_hotspot_partition_localizer_v1"
CAPTURE_ROOT = ROOT / "build/selected-direct-hidden-partition-6184a5063c9d-completion-b6volua0"
PRIOR_ROOT = ROOT / "build/selected-direct-hidden-partition-6184a5063c9d-lh_6ajii"
PINS = {
    "stdout": {
        "path": str(CAPTURE_ROOT / "check.stdout"), "bytes": 19526954,
        "sha256": "75c4cd644ad2c264801de7a364b0dc4af9ff1d2c1960a95951e4752a0835a204",
    },
    "capture": {
        "path": str(CAPTURE_ROOT / "capture.json"), "bytes": 27328,
        "sha256": "913dcb43aabbe46dd4124f88d416dce3e193d31a0ef46b1f3b727ca8e0dedbae",
    },
    "review": {
        "path": "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/6184a5063c9d/round-0001.json",
        "bytes": 690,
        "sha256": "2deb0d33524599c6082d77ecc35847a5a85855a119704b21cdee77a2f1dd8b6e",
    },
}
SOURCE_PINS = [
    {
        "path": str(ROOT / "ace3/model/candidates" / (PARENT_NAME + ".py")),
        "bytes": 24308,
        "sha256": "99feb3c5b0089133722790b8aa42f9912072f1ece022cfaa9f0970829ac01d82",
    },
    {
        "path": str(ROOT / "tests" / ("test_" + PARENT_NAME + ".py")),
        "bytes": 18345,
        "sha256": "b3f7c7b7252a4b0f10312b39a9733cb482437f230fe864fe13e468f80524adc4",
    },
]
RETAINED_E795_PINS = {
    "stdout": {
        "path": str(ROOT / "build/selected-direct-hidden-stability-e795493d08bc-fumvbe57/check.stdout"),
        "bytes": 18840034,
        "sha256": "831c13630634ca0b8f6521aea271667851a8311f455e008af813a66c07ff7bb6",
    },
    "capture": {
        "path": str(ROOT / "build/selected-direct-hidden-stability-e795493d08bc-fumvbe57/capture.json"),
        "bytes": 17021,
        "sha256": "247e09a954f046b2b2854522260e2cb1ff054a7e3e5171648b447a906f9385cf",
    },
    "review": {
        "path": "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/e795493d08bc/round-0001.json",
        "bytes": 767,
        "sha256": "2cd5adea4c912fd5b648782dbcd605d6b9a59eaa9c1c5b3b14886c2438d9ca8e",
    },
}
RETAINED_449B_PINS = {
    "stdout": {
        "path": str(ROOT / "build/selected-direct-hidden-hotspots-449b1c24ea06-apfh_msz/check.stdout"),
        "bytes": 14452633,
        "sha256": "67318f659a54c5a8e05fbe8acd0f7e209b7fcdbf670183d1e863b13e1a67f9a2",
    },
    "capture": {
        "path": str(ROOT / "build/selected-direct-hidden-hotspots-449b1c24ea06-apfh_msz/capture.json"),
        "bytes": 5327,
        "sha256": "21dc61bea5786d8674434780599cd12d565ab2cf6cadd9ba465c3b596c73ffc3",
    },
    "review": {
        "path": "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/449b1c24ea06/round-0001.json",
        "bytes": 688,
        "sha256": "566ae7c16df4d327bd028394d112e365fd7822540aed0b4e15bc0a556f0ebc41",
    },
}
RETAINED_50F_PINS = {
    "stdout": {
        "path": str(ROOT / "build/direct-hidden-residual-50f5bb2062c9-bc457u29/stdout.json"),
        "bytes": 6737063,
        "sha256": "30790957680088caecfa2539e10e672f0d5ff373573d4c80a71cb3be2d65cdeb",
    },
    "capture": {
        "path": str(ROOT / "build/direct-hidden-residual-50f5bb2062c9-bc457u29/stdout.json").replace("stdout.json", "capture.json"),
        "bytes": 7652,
        "sha256": "f29d226f2bc4e99d967b89b9f8baeeba7c71c555be742f58367eefcda3dad464",
    },
    "review": {
        "path": "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/50f5bb2062c9/round-0001.json",
        "bytes": 689,
        "sha256": "feb58e79b284e652e9bb028a2ea79b3f4b48a6b9fd18a58f0465367a6a204bb8",
    },
}
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
    "mapped_all", "inherited_native",
)
MODAL_CONTROLS = tuple(c for c in CONTROLS if c != "mapped_all")
PAIRS = ((34319, 319), (34319, 13), (319, 34319))
COMPONENTS = (
    "input_hidden", "attention_stage11", "mlp_stage17",
    "actual_residual_boundary", "negative_reference_residual_boundary",
    "q24_to_fp16_conversion", "fp16_to_branch_terminal_remainder",
)
ACCOUNT_FIELDS = ("control", "left_id", "right_id", "branch")
SCOPES = (
    ("coordinate", 34319, 319, "binary64"),
    ("coordinate", 319, 34319, "binary64"),
    ("coordinate_component", 34319, 13, "binary64"),
)
BOUNDARY = (
    "Descriptive exact retained-rational contrasts only for the three reviewed "
    "pair/branch minority exceptions. Modal means frequency-modal, not causal "
    "baseline. The 27 comparator rows and 24 overlapping mapped_all-minus-modal "
    "control contrasts are dependent accounting, not independent observations, "
    "causes, performance attribution, interventions, model repair or admission. "
    "Complete retained_6184/e795/449b/50f evidence, capture history, original-input "
    "independently propagated global references, exact thresholds, historical "
    "failures and source/operand/state/KV/lineage gates remain unchanged. "
    "Only 6184 capture/source/test/review bytes are reopened; no producer imports "
    "or replay, prefix/admission/reference replay, tensor/native decoding, "
    "RMSNorm, head/row-dot, closed 480/50f/f0/row319 execution or reconstruction. "
    "Binary64 internal stages remain NOT_RETAINED_NO_RECONSTRUCTION; the independent "
    "terminal remainder is separate. The missing row-319 896-element pre-round "
    "producer remains a closed limitation. Q24 residual state is wider than FP16. "
    "Native S16 RTZ, official G128 asymmetric packed INT4, native GEMM nibble "
    "ordering, no qzero plus-one, FP16 scales/operator boundaries/KV are unchanged. "
    "No strict-FP16-state W4A16, new-token/full-model admission, GPU/RTL/FPGA/hardware, "
    "precision/scale expansion or ACE2 changes. Normal independent Host Reviewer "
    "closure REQUIRED."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(actual == expected, message)


def bound_bytes(pin):
    data = Path(pin["path"]).read_bytes()
    same(len(data), pin["bytes"], "retained byte count changed: " + pin["path"])
    same(hashlib.sha256(data).hexdigest(), pin["sha256"],
         "retained hash changed: " + pin["path"])
    return data


def decode(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


@contextmanager
def read_only(audit):
    active = True
    allowed = {
        str(SOURCE), str(TEST), *(p["path"] for p in PINS.values()),
        *(p["path"] for p in SOURCE_PINS),
        *(str((CAPTURE_ROOT if label == "check" else PRIOR_ROOT) / (label + suffix))
          for label in ("branch", "ignored-build", "compile", "pytest", "check")
          for suffix in (".command.txt", ".argv.json", ".environment.json",
                         ".stdout", ".stderr", ".whole-command.log")),
    }

    def guard(event, args):
        if not active:
            return
        forbidden = False
        if event == "open":
            path, mode, flags = args
            forbidden = (
                not isinstance(path, (str, bytes, os.PathLike))
                or os.fsdecode(path) not in allowed
                or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT |
                                 os.O_TRUNC | os.O_APPEND))
            )
        elif event == "import":
            forbidden = args[0].startswith(
                ("ace3.", "numpy", "torch", "safetensors", "ctypes"))
        elif event.startswith(("subprocess.", "socket.", "ctypes.", "shutil.")):
            forbidden = True
        elif event in (
            "os.system", "os.fork", "os.forkpty", "os.exec", "os.posix_spawn",
            "os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.link",
            "os.symlink", "os.truncate", "os.chmod", "os.chown", "os.utime",
        ):
            forbidden = True
        if forbidden:
            audit["forbidden_calls"] += 1
            raise RuntimeError("retained-only profiler forbids " + event)

    sys.addaudithook(guard)
    try:
        yield
    finally:
        active = False


def zero_counters(values):
    require(bool(values) and all(v is False or type(v) is int and v == 0
                                for v in values.values()),
            "retained forbidden dispatch, write or claim")


def validate_review(review):
    same((review["kind"], review["mission_id"], review["producer_role"],
          review["review"]["status"]),
         ("round_reviewed_handoff", "6184a5063c9d", "reviewer", "done"),
         "independent terminal 6184 review required")


def validate_retained(retained):
    same(retained["diagnostic_id"], PARENT_NAME, "wrong retained producer")
    same(retained["status"], "READ_ONLY_SELECTED_DIRECT_HIDDEN_HOTSPOT_PARTITION_LOCALIZER",
         "wrong retained boundary")
    require(type(retained["version"]) is int and retained["version"] == 1,
            "retained version changed")
    same(retained["compiled_sources"], SOURCE_PINS, "retained source/test splice")
    node = retained
    for pins, child in (
        (RETAINED_E795_PINS, "retained_e795"),
        (RETAINED_449B_PINS, "retained_449b"),
        (RETAINED_50F_PINS, "retained_50f"),
    ):
        same(node["input_pins"], pins, "nested pins changed: " + child)
        zero_counters(node["dispatch_and_write_audit"])
        zero_counters(node["flags"])
        node = node[child]
    zero_counters(node["dispatch_and_write_audit"])
    zero_counters(node["flags"])
    require(retained["report"]["descriptive_accounting_only"] is True
            and retained["report"]["exact_gap_closure"] is True,
            "retained accounting boundary changed")


def authenticate():
    review = decode(bound_bytes(PINS["review"]))
    validate_review(review)
    capture = decode(bound_bytes(PINS["capture"]))
    require(capture["success"] is True and bool(capture["checks"])
            and all(v is True for v in capture["checks"].values()),
            "retained capture failed")
    same(capture["capture_directory"], str(CAPTURE_ROOT), "capture directory splice")
    preflight = capture["preflight"]
    same((preflight["cwd"], preflight["uid"], preflight["python"]),
         (str(ROOT), 1000, PYTHON), "retained source/account/interpreter gate")
    same(capture["sources_after"], preflight["sources"], "retained source drift")
    same(capture["sources_after"][:2], SOURCE_PINS, "retained source/test census")
    same(capture["accepted_artifacts_after"], preflight["accepted_artifacts"],
         "retained accepted-artifact overwrite")
    for pin in SOURCE_PINS:
        bound_bytes(pin)
    same([r["label"] for r in capture["results"]],
         ["branch", "ignored-build", "compile", "pytest", "check"],
         "retained command census")
    stdout = None
    for result in capture["results"]:
        require(type(result["exit_status"]) is int and result["exit_status"] == 0,
                "retained command failure")
        label = result["label"]
        directory = CAPTURE_ROOT if label == "check" else PRIOR_ROOT
        same([p["path"] for p in result["files"]], [
            str(directory / (label + suffix)) for suffix in (
                ".command.txt", ".argv.json", ".environment.json", ".stdout",
                ".stderr", ".whole-command.log")], "capture member splice")
        command, argv, environment, output, error, whole = [
            bound_bytes(p) for p in result["files"]]
        same(command, (result["command"] + "\n").encode(), "command binding")
        same(decode(environment), {k: preflight[k] for k in (
            "cwd", "uid", "python", "python_version", "environment")},
             "environment binding")
        same(error, b"", "retained stderr not empty")
        require(whole.startswith(b"COMMAND\n" + command)
                and all(part in whole for part in (environment, output))
                and whole.endswith(b"\nSTDERR\n\nEXIT_STATUS=0\nTIMED_OUT=False\n"),
                "whole-command binding")
        require(isinstance(decode(argv), list) and bool(decode(argv)), "argv binding")
        if label == "branch":
            same(output, b"argus/full-projection\n", "retained branch changed")
        if label == "check":
            expected = f"ace3.model.candidates.{PARENT_NAME}"
            same(result["command"],
                 f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m "
                 f"{expected} --check", "retained command changed")
            same(decode(argv), [PYTHON, "-B", "-m", expected, "--check"],
                 "retained argv changed")
            same(result["files"][3], PINS["stdout"], "capture/stdout splice")
            stdout = output
    require(stdout is not None, "missing retained stdout")
    retained = decode(stdout)
    same(retained["command"], capture["results"][-1]["command"], "stdout command splice")
    validate_retained(retained)
    return retained, capture, review


def rational(text):
    require(type(text) is str, "rational must be an exact string")
    value = Fraction(text)
    same(str(value), text, "noncanonical rational")
    return value


def sign(value):
    return (value > 0) - (value < 0)


def identity(row, kind):
    keys = ("coordinate",) if kind == "coordinate" else ("coordinate", "component")
    same({k for k in ("coordinate", "component") if k in row}, set(keys),
         "ranked identity shape changed")
    require(type(row["coordinate"]) is int and 0 <= row["coordinate"] < 896,
            "invalid coordinate")
    if kind == "coordinate_component":
        require(row["component"] in COMPONENTS, "invalid component")
    return {k: row[k] for k in keys}


def identity_key(row):
    return (row["coordinate"],
            COMPONENTS.index(row["component"]) if "component" in row else -1)


def ranked_rows(table, kind):
    rows = table["rows"]
    require(len(rows) >= 2, "retained ranking needs a runner-up")
    ids = [identity(row, kind) for row in rows]
    keys = [identity_key(row) for row in ids]
    require(len(set(keys)) == len(keys), "duplicate ranked identity")
    values = [rational(row["absolute"]) for row in rows]
    require(all(v >= 0 for v in values), "negative magnitude")
    same(list(zip(values, keys)),
         sorted(zip(values, keys), key=lambda item: (-item[0], item[1])),
         "retained ranking order changed")
    top, second = values[:2]
    require(top > second and top > 0, "unique nonzero winner required")
    for row, value in zip(rows, values, strict=True):
        same(abs(rational(row["signed"])), value, "signed/absolute closure")
        require(type(row["rank"]) is int, "noninteger rank")
        same(row["rank"], 1 + sum(v > value for v in values), "rank closure")
        same(rational(row["gap_from_maximum"]), top - value, "row gap closure")
    for key, value in (
        ("row_count", len(rows)), ("zero_row_count", values.count(Fraction())),
        ("maximum_tie_count_including_zeros", 1),
    ):
        require(type(table[key]) is int and table[key] == value, "ranked count closure")
    for key, value in (
        ("maximum_absolute", top), ("runner_up_absolute", second),
        ("dominance_gap", top - second), ("next_distinct_absolute", second),
        ("gap_to_next_distinct", top - second),
    ):
        same(rational(table[key]), value, "ranked gap closure: " + key)
    same(table["winner_ids"], ids[:1], "ranked winner splice")
    same(table["maximizer_ids_including_zeros"], ids[:1], "ranked maximizer splice")
    require(table["unique_nonzero_hotspot"] is True and table["all_zero"] is False
            and table["exact_gap_and_tie_closure"] is True, "ranked uniqueness changed")
    return dict(zip(keys, rows, strict=True)), {
        "winner_id": ids[0],
        "runner_up_ids": [i for i, v in zip(ids, values, strict=True) if v == second],
        "maximum_absolute": str(top), "runner_up_absolute": str(second),
        "dominance_gap": str(top - second),
    }


def comparator(table, kind, member, exception, modal):
    rows, summary = ranked_rows(table, kind)
    for key, value in summary.items():
        same(member[key], value, "localizer/ranked-row splice: " + key)
    require(identity_key(exception) in rows and identity_key(modal) in rows,
            "missing comparator identity")
    e, m = rows[identity_key(exception)], rows[identity_key(modal)]
    absolute = rational(e["absolute"]) - rational(m["absolute"])
    signed = rational(e["signed"]) - rational(m["signed"])
    expected_winner = exception if member["control"] == "mapped_all" else modal
    same(summary["winner_id"], expected_winner, "control/winner membership changed")
    same(sign(absolute), 1 if member["control"] == "mapped_all" else -1,
         "exception/modal comparator sign changed")
    return {
        **{k: member[k] for k in ACCOUNT_FIELDS}, "roles": member["roles"],
        "membership": "exception" if member["control"] == "mapped_all" else "modal",
        "observed_winner_id": summary["winner_id"],
        "exception_winner_row": dict(e), "modal_winner_row": dict(m),
        "absolute_comparator": str(absolute), "absolute_comparator_sign": sign(absolute),
        "signed_comparator": str(signed), "signed_comparator_sign": sign(signed),
        "rank_difference": e["rank"] - m["rank"],
        "dominance_gap": summary["dominance_gap"],
        "runner_up_ids": summary["runner_up_ids"],
    }


def contrast(exception, modal):
    deltas = {}
    for candidate in ("exception", "modal"):
        field = candidate + "_winner_row"
        for value in ("signed", "absolute"):
            deltas[candidate + "_" + value + "_delta"] = str(
                rational(exception[field][value]) - rational(modal[field][value]))
    for value in ("absolute_comparator", "signed_comparator", "dominance_gap"):
        deltas[value + "_delta"] = str(rational(exception[value]) - rational(modal[value]))
    e_rank, m_rank = exception["rank_difference"], modal["rank_difference"]
    return {
        "exception_control": exception["control"], "modal_control": modal["control"],
        "absolute_comparator_signs": {
            "mapped_all": exception["absolute_comparator_sign"],
            "modal_control": modal["absolute_comparator_sign"],
        },
        "rank_order_signs": {"mapped_all": sign(e_rank), "modal_control": sign(m_rank)},
        "rank_flip": sign(e_rank) * sign(m_rank) == -1,
        "winner_changed": exception["observed_winner_id"] != modal["observed_winner_id"],
        "exception_rank_delta": (exception["exception_winner_row"]["rank"]
                                 - modal["exception_winner_row"]["rank"]),
        "modal_rank_delta": (exception["modal_winner_row"]["rank"]
                             - modal["modal_winner_row"]["rank"]),
        **deltas, "delta_signs": {k: sign(rational(v)) for k, v in deltas.items()},
    }


def report(retained):
    validate_retained(retained)
    parent = retained["report"]
    same(set(parent["tables"]), {"coordinate", "coordinate_component"},
         "retained table census changed")
    unstable = []
    for kind, table in parent["tables"].items():
        partitions = table["partitions"]["by_pair_branch"]
        same([(p["left_id"], p["right_id"], p["branch"]) for p in partitions],
             [(l, r, b) for l, r in PAIRS for b in ("binary64", "fp16")],
             "retained partition census changed")
        require(all(type(p["stable"]) is bool for p in partitions), "invalid stable flag")
        unstable.extend((kind, p["left_id"], p["right_id"], p["branch"])
                        for p in partitions if not p["stable"])
    same(set(unstable), set(SCOPES), "exact three unstable partitions required")
    same(len(unstable), 3, "duplicate unstable partition")
    accounts = retained["retained_e795"]["report"]["accounts"]
    same([tuple(a[k] for k in ACCOUNT_FIELDS) for a in accounts],
         [(c, l, r, b) for c in CONTROLS for l, r in PAIRS for b in ("binary64", "fp16")],
         "nested account census changed")
    account_map = {tuple(a[k] for k in ACCOUNT_FIELDS): a for a in accounts}
    profiles = []
    for kind, left, right, branch in SCOPES:
        table = parent["tables"][kind]
        p = next(p for p in table["partitions"]["by_pair_branch"]
                 if (p["left_id"], p["right_id"], p["branch"]) == (left, right, branch))
        modal = {"coordinate": 241}
        exception = {"coordinate": 62}
        if kind == "coordinate_component":
            modal["component"] = "mlp_stage17"
            exception = {"coordinate": 241, "component": COMPONENTS[-1]}
        require(type(p["account_count"]) is int and p["account_count"] == 9,
                "nine-control partition required")
        same(p["classification"], "VARIABLE_HOTSPOTS", "partition classification changed")
        same(p["unique_modal_winner_id"], modal, "modal identity changed")
        same(p["modal_winner_ids"], [modal], "modal frequency tie or splice")
        classes = p["winner_classes"]
        same({identity_key(c["winner_id"]) for c in classes},
             {identity_key(exception), identity_key(modal)}, "winner class census changed")
        same(len(classes), 2, "duplicate winner class")
        members = {}
        for cls in classes:
            expected = ["mapped_all"] if cls["winner_id"] == exception else list(MODAL_CONTROLS)
            require(type(cls["count"]) is int and cls["count"] == len(expected),
                    "eight/one winner frequency changed")
            same([m["control"] for m in cls["members"]], expected, "control class splice")
            for member in cls["members"]:
                same((member["left_id"], member["right_id"], member["branch"]),
                     (left, right, branch), "cross-partition member")
                same(member["winner_id"], cls["winner_id"], "member winner splice")
                require(member["control"] not in members, "duplicate control membership")
                members[member["control"]] = member
        same(set(members), set(CONTROLS), "control census changed")
        same(p["winner_counts"],
             [{**c["winner_id"], "count": c["count"]} for c in classes],
             "winner count/class splice")
        same(p["exceptions"], [members["mapped_all"]], "exception membership changed")
        localized = [m for m in table["accounts"]
                     if (m["left_id"], m["right_id"], m["branch"]) == (left, right, branch)]
        same(localized, [members[c] for c in CONTROLS], "localizer account splice")
        comparisons = []
        for control in CONTROLS:
            member = members[control]
            a = account_map[(control, left, right, branch)]
            same(a["roles"], member["roles"], "retained role splice")
            ranked = a["tables"][kind]
            same(len(ranked["rows"]),
                 a["selected_coordinate_count"] * (7 if kind == "coordinate_component" else 1),
                 "selected ranked-row census changed")
            comparisons.append(comparator(ranked, kind, member, exception, modal))
        mapped = next(c for c in comparisons if c["control"] == "mapped_all")
        contrasts = [contrast(mapped, c) for c in comparisons if c["control"] != "mapped_all"]
        profiles.append({
            "table": kind, "left_id": left, "right_id": right, "branch": branch,
            "exception_winner_id": exception, "modal_winner_id": modal,
            "control_membership": {"exception": ["mapped_all"], "modal": list(MODAL_CONTROLS)},
            "same_partition_comparator_rows": comparisons,
            "mapped_all_vs_modal_contrast_rows": contrasts,
        })
    result = {
        "unstable_partition_count": len(profiles),
        "same_partition_comparator_row_count": sum(
            len(p["same_partition_comparator_rows"]) for p in profiles),
        "mapped_all_vs_modal_contrast_row_count": sum(
            len(p["mapped_all_vs_modal_contrast_rows"]) for p in profiles),
        "partitions": profiles,
        "comparator_rule": "Exception-winner absolute/signed value minus modal-winner value within one control.",
        "contrast_rule": "Each exact delta is mapped_all minus the named modal control in the same partition.",
        "rank_rule": "Competition ranks by absolute magnitude; rank_difference is exception rank minus modal rank.",
        "dominance_gap_rule": "Maximum minus runner-up absolute value, distinct from the two-candidate comparator.",
        "reference_scope": parent["reference_scope"],
        "lineage_separation": parent["lineage_separation"],
        "exact_rational_closure": True, "descriptive_accounting_only": True,
        "conclusion": (
            "All three retained minority exceptions reverse the winner/modal absolute "
            "ordering against each of their eight modal controls. These 24 rank flips "
            "describe retained accounting only; they do not identify a cause or an intervention."
        ),
    }
    require_expected(result)
    return result


def require_expected(result):
    same((result["unstable_partition_count"], result["same_partition_comparator_row_count"],
          result["mapped_all_vs_modal_contrast_row_count"]), (3, 27, 24),
         "required three/27/24 output census")
    same([(p["table"], p["left_id"], p["right_id"], p["branch"])
          for p in result["partitions"]], list(SCOPES), "output partition splice")
    for p in result["partitions"]:
        same([r["control"] for r in p["same_partition_comparator_rows"]],
             list(CONTROLS), "output comparator membership")
        rows = p["mapped_all_vs_modal_contrast_rows"]
        same([r["modal_control"] for r in rows], list(MODAL_CONTROLS),
             "output contrast membership")
        require(all(r["exception_control"] == "mapped_all" and r["rank_flip"] is True
                    and r["winner_changed"] is True
                    and r["absolute_comparator_signs"] == {"mapped_all": 1, "modal_control": -1}
                    and r["rank_order_signs"] == {"mapped_all": -1, "modal_control": 1}
                    and rational(r["absolute_comparator_delta"]) > 0 for r in rows),
                "required exact comparator/rank flips")


def check():
    require(ROOT == Path("/home/argustest/ace3-argus") and Path.cwd() == ROOT
            and Path(__file__).resolve() == SOURCE and sys.executable == PYTHON
            and os.getuid() == 1000 and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        compiled = []
        for path in (SOURCE, TEST):
            data = path.read_bytes()
            compile(data, str(path), "exec", dont_inherit=True)
            compiled.append({"path": str(path), "bytes": len(data),
                             "sha256": hashlib.sha256(data).hexdigest()})
        retained, capture, review = authenticate()
        result = report(retained)
    same(audit, {"forbidden_calls": 0}, "forbidden dispatch or write")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_HOTSPOT_MODAL_EXCEPTION_CONTRAST_PROFILER",
        "command": COMMAND, "compiled_sources": compiled, "input_pins": PINS,
        "retained_source_pins": SOURCE_PINS,
        "dispatch_and_write_audit": {**retained["dispatch_and_write_audit"], **audit},
        "flags": retained["flags"], "claim_boundary": BOUNDARY,
        "normal_host_review": "REQUIRED", "retained_6184": retained,
        "retained_6184_capture": capture, "retained_6184_review": review, "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
