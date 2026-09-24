"""Stdout-only exact modal-control equivalence over reviewed 0146 retained rows."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_control_equivalence_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT_NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_hotspot_modal_exception_contrast_profiler_v1"
CAPTURE_ROOT = ROOT / "build/selected-direct-hidden-modal-contrast-0146c951608d-pt2zlbcz"
REVIEW_ROOT = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
PINS = {
    "stdout": {
        "path": str(CAPTURE_ROOT / "check.stdout"), "bytes": 19618193,
        "sha256": "233e274d21ddd53f80912ff83e666cc4d3e3db5a8388cde18dd8eb8603b1294f",
    },
    "capture": {
        "path": str(CAPTURE_ROOT / "capture.json"), "bytes": 32049,
        "sha256": "9ce9e56b089f5d0351f1e40dd40820e894615b86325d408241b258cc2a97c2d9",
    },
    "review": {
        "path": str(REVIEW_ROOT / "0146c951608d/round-0001.json"), "bytes": 690,
        "sha256": "b8cd60e1bf22c51be87175bbe88912e3b67bc90568c66e2916592384aa665ba8",
    },
}
SOURCE_PINS = [
    {
        "path": str(ROOT / "ace3/model/candidates" / (PARENT_NAME + ".py")),
        "bytes": 29026,
        "sha256": "ac4e0b564bf0fca1be334edcf23a7bbd49be29ba5f512017e489f31ae7a30712",
    },
    {
        "path": str(ROOT / "tests" / ("test_" + PARENT_NAME + ".py")),
        "bytes": 17875,
        "sha256": "69f1f634c83e3591062312b1f7161234c87134cb07d8d1b5c3cb264f371a18c3",
    },
]
NESTED_PINS = (
    ("retained_6184", {
        "stdout": {
            "path": str(ROOT / "build/selected-direct-hidden-partition-6184a5063c9d-completion-b6volua0/check.stdout"),
            "bytes": 19526954,
            "sha256": "75c4cd644ad2c264801de7a364b0dc4af9ff1d2c1960a95951e4752a0835a204",
        },
        "capture": {
            "path": str(ROOT / "build/selected-direct-hidden-partition-6184a5063c9d-completion-b6volua0/capture.json"),
            "bytes": 27328,
            "sha256": "913dcb43aabbe46dd4124f88d416dce3e193d31a0ef46b1f3b727ca8e0dedbae",
        },
        "review": {
            "path": str(REVIEW_ROOT / "6184a5063c9d/round-0001.json"), "bytes": 690,
            "sha256": "2deb0d33524599c6082d77ecc35847a5a85855a119704b21cdee77a2f1dd8b6e",
        },
    }),
    ("retained_e795", {
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
            "path": str(REVIEW_ROOT / "e795493d08bc/round-0001.json"), "bytes": 767,
            "sha256": "2cd5adea4c912fd5b648782dbcd605d6b9a59eaa9c1c5b3b14886c2438d9ca8e",
        },
    }),
    ("retained_449b", {
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
            "path": str(REVIEW_ROOT / "449b1c24ea06/round-0001.json"), "bytes": 688,
            "sha256": "566ae7c16df4d327bd028394d112e365fd7822540aed0b4e15bc0a556f0ebc41",
        },
    }),
    ("retained_50f", {
        "stdout": {
            "path": str(ROOT / "build/direct-hidden-residual-50f5bb2062c9-bc457u29/stdout.json"),
            "bytes": 6737063,
            "sha256": "30790957680088caecfa2539e10e672f0d5ff373573d4c80a71cb3be2d65cdeb",
        },
        "capture": {
            "path": str(ROOT / "build/direct-hidden-residual-50f5bb2062c9-bc457u29/capture.json"),
            "bytes": 7652,
            "sha256": "f29d226f2bc4e99d967b89b9f8baeeba7c71c555be742f58367eefcda3dad464",
        },
        "review": {
            "path": str(REVIEW_ROOT / "50f5bb2062c9/round-0001.json"), "bytes": 689,
            "sha256": "feb58e79b284e652e9bb028a2ea79b3f4b48a6b9fd18a58f0465367a6a204bb8",
        },
    }),
)
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
    "mapped_all", "inherited_native",
)
MODAL_CONTROLS = tuple(c for c in CONTROLS if c != "mapped_all")
SCOPES = (
    ("coordinate", 34319, 319, "binary64"),
    ("coordinate", 319, 34319, "binary64"),
    ("coordinate_component", 34319, 13, "binary64"),
)
COMPONENTS = (
    "input_hidden", "attention_stage11", "mlp_stage17",
    "actual_residual_boundary", "negative_reference_residual_boundary",
    "q24_to_fp16_conversion", "fp16_to_branch_terminal_remainder",
)
LABELS = ("branch", "ignored-build", "compile", "pytest", "check")
SUFFIXES = (
    ".command.txt", ".argv.json", ".environment.json", ".stdout",
    ".stderr", ".whole-command.log",
)
BOUNDARY = (
    "Descriptive exact retained-rational modal-control equivalence only. "
    "Exactly three retained unstable partitions, eight frequency-modal controls "
    "each, and 84 dependent pair comparisons; no independent-sample, causal, "
    "intervention, performance, model-repair or admission claim. Equality is "
    "restricted to the retained comparator and mapped_all-minus-modal vectors, "
    "not complete control trajectories. Complete retained_0146/6184/e795/449b/50f "
    "evidence, historical failures, original-input independently propagated "
    "global references, exact thresholds and source/operand/state/KV/lineage "
    "gates remain unchanged. Only 0146 stdout/capture/source/test/review and "
    "capture sidecars are read; ancestors are authenticated nested pins, not "
    "reopened, replayed or re-reviewed. No producer imports, tensor/native "
    "decoding, prefix/admission/reference replay, RMSNorm, head/row-dot, closed "
    "480/50f/f0/6184/e795/0146 computation or row319 availability work. Missing "
    "row-319 896-element pre-round producer and NOT_RETAINED_NO_RECONSTRUCTION "
    "binary64 internal stages remain limitations. Q24 residual is wider than "
    "FP16; native S16 RTZ, G128 asymmetric packed INT4, native GEMM nibble order, "
    "no qzero plus-one, FP16 scales/operator boundaries/KV remain unchanged. "
    "No strict-FP16-state W4A16, new-token/full-model admission, GPU/RTL/FPGA/"
    "hardware, precision/scale expansion or ACE2 changes. Normal independent "
    "Host Reviewer closure REQUIRED."
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
        *(str(CAPTURE_ROOT / (label + suffix)) for label in LABELS for suffix in SUFFIXES),
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
            raise RuntimeError("retained-only classifier forbids " + event)

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
         ("round_reviewed_handoff", "0146c951608d", "reviewer", "done"),
         "independent terminal 0146 review required")


def validate_retained(retained):
    same(retained["diagnostic_id"], PARENT_NAME, "wrong retained producer")
    same(retained["status"],
         "READ_ONLY_SELECTED_DIRECT_HIDDEN_HOTSPOT_MODAL_EXCEPTION_CONTRAST_PROFILER",
         "wrong retained boundary")
    require(type(retained["version"]) is int and retained["version"] == 1,
            "retained version changed")
    same(retained["compiled_sources"], SOURCE_PINS, "retained source/test splice")
    node = retained
    for child, pins in NESTED_PINS:
        same(node["input_pins"], pins, "nested pins changed: " + child)
        zero_counters(node["dispatch_and_write_audit"])
        zero_counters(node["flags"])
        node = node[child]
    zero_counters(node["dispatch_and_write_audit"])
    zero_counters(node["flags"])
    require(retained["report"]["descriptive_accounting_only"] is True
            and retained["report"]["exact_rational_closure"] is True,
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
    same([r["label"] for r in capture["results"]], list(LABELS),
         "retained command census")
    stdout = None
    for result in capture["results"]:
        require(type(result["exit_status"]) is int and result["exit_status"] == 0
                and result["timed_out"] is False, "retained command failure")
        label = result["label"]
        same([p["path"] for p in result["files"]],
             [str(CAPTURE_ROOT / (label + suffix)) for suffix in SUFFIXES],
             "capture member splice")
        command, argv, environment, output, error, whole = [
            bound_bytes(p) for p in result["files"]]
        same(command, (result["command"] + "\n").encode(), "command binding")
        same(decode(argv), result["argv"], "argv binding")
        same(decode(environment), {k: preflight[k] for k in (
            "cwd", "uid", "python", "python_version", "environment")},
             "environment binding")
        same(error, b"", "retained stderr not empty")
        same(whole, b"COMMAND\n" + command + b"ENVIRONMENT\n" + environment
             + b"\nSTDOUT\n" + output + b"\nSTDERR\n" + error
             + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "whole-command binding")
        if label == "branch":
            same(output, b"argus/full-projection\n", "retained branch changed")
        if label == "check":
            module = "ace3.model.candidates." + PARENT_NAME
            same(result["command"],
                 f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m "
                 f"{module} --check", "retained command changed")
            same(decode(argv), [PYTHON, "-B", "-m", module, "--check"],
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
         "identity shape changed")
    require(type(row["coordinate"]) is int and 0 <= row["coordinate"] < 896,
            "invalid coordinate")
    if kind == "coordinate_component":
        require(row["component"] in COMPONENTS, "invalid component")
    return {k: row[k] for k in keys}


def comparator_fields(row, partition, control):
    same((row["control"], row["left_id"], row["right_id"], row["branch"]),
         (control, partition["left_id"], partition["right_id"], partition["branch"]),
         "cross-partition comparator")
    is_modal = control != "mapped_all"
    same(row["membership"], "modal" if is_modal else "exception", "control membership")
    values = {}
    for candidate in ("exception", "modal"):
        selected = row[candidate + "_winner_row"]
        same(identity(selected, partition["table"]), partition[candidate + "_winner_id"],
             "candidate identity changed")
        for field in ("signed", "absolute", "gap_from_maximum"):
            values[candidate + "." + field] = rational(selected[field])
        require(type(selected["rank"]) is int and selected["rank"] >= 1, "invalid rank")
        values[candidate + ".rank"] = Fraction(selected["rank"])
        same(abs(values[candidate + ".signed"]), values[candidate + ".absolute"],
             "signed/absolute closure")
    winner = "modal" if is_modal else "exception"
    same(row["observed_winner_id"], partition[winner + "_winner_id"], "winner membership")
    same(values[winner + ".rank"], 1, "winner rank changed")
    maximum = values[winner + ".absolute"]
    for candidate in ("exception", "modal"):
        same(values[candidate + ".gap_from_maximum"],
             maximum - values[candidate + ".absolute"], "candidate gap closure")
    for field in ("absolute_comparator", "signed_comparator", "dominance_gap"):
        values[field] = rational(row[field])
    require(values["dominance_gap"] > 0, "retained unique winner required")
    for field in ("absolute", "signed"):
        same(values[field + "_comparator"],
             values["exception." + field] - values["modal." + field],
             "comparator closure")
        require(type(row[field + "_comparator_sign"]) is int, "invalid sign")
        same(row[field + "_comparator_sign"], sign(values[field + "_comparator"]),
             "comparator sign closure")
    require(type(row["rank_difference"]) is int, "invalid rank difference")
    values["rank_difference"] = Fraction(row["rank_difference"])
    same(values["rank_difference"], values["exception.rank"] - values["modal.rank"],
         "rank difference closure")
    require(bool(row["runner_up_ids"]), "missing runner-up identity")
    ids = [identity(i, partition["table"]) for i in row["runner_up_ids"]]
    same(ids, row["runner_up_ids"], "runner-up identity shape")
    require(len({json.dumps(i, sort_keys=True) for i in ids}) == len(ids),
            "duplicate runner-up identity")
    return values


def snapshot(row, contrast, partition, mapped_values):
    control = row["control"]
    require(control in MODAL_CONTROLS, "non-modal comparison forbidden")
    values = comparator_fields(row, partition, control)
    same((contrast["exception_control"], contrast["modal_control"]),
         ("mapped_all", control), "contrast control splice")
    deltas = {}
    ranks = {}
    for candidate in ("exception", "modal"):
        for field in ("signed", "absolute"):
            key = candidate + "_" + field + "_delta"
            source = candidate + "." + field
            deltas[key] = rational(contrast[key])
            same(deltas[key], mapped_values[source] - values[source], "contrast closure")
        key = candidate + "_rank_delta"
        require(type(contrast[key]) is int, "invalid contrast rank")
        ranks[key] = contrast[key]
        same(Fraction(ranks[key]), mapped_values[candidate + ".rank"]
             - values[candidate + ".rank"], "contrast rank closure")
    for field in ("absolute_comparator", "signed_comparator", "dominance_gap"):
        key = field + "_delta"
        deltas[key] = rational(contrast[key])
        same(deltas[key], mapped_values[field] - values[field], "contrast closure")
    same(contrast["delta_signs"], {k: sign(v) for k, v in deltas.items()},
         "contrast sign closure")
    same(contrast["absolute_comparator_signs"], {
        "mapped_all": sign(mapped_values["absolute_comparator"]),
        "modal_control": sign(values["absolute_comparator"]),
    }, "contrast comparator signs")
    same(contrast["rank_order_signs"], {
        "mapped_all": sign(mapped_values["rank_difference"]),
        "modal_control": sign(values["rank_difference"]),
    }, "contrast rank signs")
    require(contrast["rank_flip"] is True and contrast["winner_changed"] is True,
            "retained contrast membership changed")
    same(sign(values["rank_difference"]), 1, "modal rank order changed")
    same(sign(mapped_values["rank_difference"]), -1, "exception rank order changed")
    fields = {**{"comparator." + k: str(v) for k, v in values.items()},
              **{"contrast." + k: str(v) for k, v in {**deltas, **ranks}.items()}}
    return {
        "control": control, "roles": row["roles"], "fields": fields,
        "signs": {k: sign(rational(v)) for k, v in fields.items()},
        "ranks": {**{k: int(values[k]) for k in (
            "exception.rank", "modal.rank", "rank_difference")}, **ranks},
        "identities": {
            "exception": partition["exception_winner_id"],
            "modal": partition["modal_winner_id"],
            "observed_winner": row["observed_winner_id"],
            "runner_up_ids": row["runner_up_ids"],
        },
        "ties": {
            "candidate_absolute_tie": values["exception.absolute"] == values["modal.absolute"],
            "candidate_signed_tie": values["exception.signed"] == values["modal.signed"],
            "candidate_rank_tie": values["exception.rank"] == values["modal.rank"],
            "zero_candidate_ids": [
                partition[c + "_winner_id"] for c in ("exception", "modal")
                if values[c + ".absolute"] == 0],
            "maximum_candidate_ids": [
                partition[c + "_winner_id"] for c in ("exception", "modal")
                if values[c + ".gap_from_maximum"] == 0],
            "runner_up_tied": len(row["runner_up_ids"]) > 1,
        },
        "contrast_membership": {k: contrast[k] for k in (
            "absolute_comparator_signs", "rank_order_signs", "rank_flip", "winner_changed")},
    }


def compare_controls(left, right):
    same(set(left["fields"]), set(right["fields"]), "pair field census changed")
    deltas = {k: str(rational(v) - rational(right["fields"][k]))
              for k, v in left["fields"].items()}
    agreement = {
        "comparator_exact": all(v == "0" for k, v in deltas.items()
                                if k.startswith("comparator.")),
        "contrast_vector_exact": all(v == "0" for k, v in deltas.items()
                                     if k.startswith("contrast.")),
        "sign": left["signs"] == right["signs"],
        "rank": left["ranks"] == right["ranks"],
        "dominance_gap": all(deltas[k] == "0" for k in (
            "comparator.dominance_gap", "contrast.dominance_gap_delta")),
        "identity_membership": left["identities"] == right["identities"],
        "tie_membership": left["ties"] == right["ties"],
        "contrast_membership": left["contrast_membership"] == right["contrast_membership"],
    }
    return {
        "left_control": left["control"], "right_control": right["control"],
        "field_deltas": deltas,
        "delta_signs": {k: sign(rational(v)) for k, v in deltas.items()},
        "zero_fields": [k for k, v in deltas.items() if v == "0"],
        "nonzero_fields": [k for k, v in deltas.items() if v != "0"],
        "agreement": agreement, "equivalent": all(agreement.values()),
    }


def classify_partition(partition):
    same(partition["control_membership"],
         {"exception": ["mapped_all"], "modal": list(MODAL_CONTROLS)},
         "exact eight-modal/one-exception membership required")
    modal = {"coordinate": 241}
    exception = {"coordinate": 62}
    if partition["table"] == "coordinate_component":
        modal["component"] = "mlp_stage17"
        exception = {"coordinate": 241, "component": COMPONENTS[-1]}
    same(partition["modal_winner_id"], modal, "modal identity splice")
    same(partition["exception_winner_id"], exception, "exception identity splice")
    rows = partition["same_partition_comparator_rows"]
    contrasts = partition["mapped_all_vs_modal_contrast_rows"]
    same([r["control"] for r in rows], list(CONTROLS), "comparator control census")
    same([r["modal_control"] for r in contrasts], list(MODAL_CONTROLS),
         "contrast control census")
    by_control = {r["control"]: r for r in rows}
    mapped_values = comparator_fields(by_control["mapped_all"], partition, "mapped_all")
    controls = [snapshot(by_control[c], contrast, partition, mapped_values)
                for c, contrast in zip(MODAL_CONTROLS, contrasts, strict=True)]
    comparisons = [compare_controls(a, b) for a, b in combinations(controls, 2)]
    spreads = {}
    for field in controls[0]["fields"]:
        values = [rational(c["fields"][field]) for c in controls]
        low, high = min(values), max(values)
        spreads[field] = {
            "minimum": str(low), "maximum": str(high), "spread": str(high - low),
            "zero": high == low,
            "minimum_controls": [c["control"] for c, v in zip(controls, values, strict=True)
                                 if v == low],
            "maximum_controls": [c["control"] for c, v in zip(controls, values, strict=True)
                                 if v == high],
        }
    classes = []
    for control in controls:
        signature = {k: v for k, v in control.items() if k not in ("control", "roles")}
        existing = next((c for c in classes if c["signature"] == signature), None)
        if existing is None:
            classes.append({"signature": signature, "controls": [control["control"]]})
        else:
            existing["controls"].append(control["control"])
    return {
        **{k: partition[k] for k in ("table", "left_id", "right_id", "branch")},
        "modal_controls": list(MODAL_CONTROLS), "modal_control_count": len(controls),
        "control_profiles": controls, "pair_comparisons": comparisons,
        "pair_comparison_count": len(comparisons), "field_spreads": spreads,
        "equivalence_classes": [c["controls"] for c in classes],
        "equivalent_pair_count": sum(p["equivalent"] for p in comparisons),
        "agreement_counts": {k: sum(p["agreement"][k] for p in comparisons)
                             for k in comparisons[0]["agreement"]},
        "classification": "SUPPORTED" if all(p["equivalent"] for p in comparisons)
                          else "REJECTED",
    }


def report(retained):
    validate_retained(retained)
    parent = retained["report"]
    same((parent["unstable_partition_count"], parent["same_partition_comparator_row_count"],
          parent["mapped_all_vs_modal_contrast_row_count"]), (3, 27, 24),
         "retained three/27/24 census changed")
    same([(p["table"], p["left_id"], p["right_id"], p["branch"])
          for p in parent["partitions"]], list(SCOPES), "exact three partition scope required")
    partitions = [classify_partition(p) for p in parent["partitions"]]
    result = {
        "unstable_partition_count": len(partitions),
        "modal_control_pair_comparison_count": sum(p["pair_comparison_count"] for p in partitions),
        "partitions": partitions,
        "supported_partition_count": sum(p["classification"] == "SUPPORTED" for p in partitions),
        "rejected_partition_count": sum(p["classification"] == "REJECTED" for p in partitions),
        "pair_delta_rule": "Left modal control minus right modal control, in retained control order.",
        "spread_rule": "Exact maximum minus minimum over the eight modal controls, with all extrema members.",
        "equivalence_rule": (
            "SUPPORTED iff all 28 pairs have exact equality of every retained comparator "
            "and mapped_all-minus-modal contrast field and matching signs, ranks, "
            "dominance gaps, tie and identity membership; otherwise REJECTED. "
            "Matching winner identities alone is insufficient."
        ),
        "reference_scope": parent["reference_scope"],
        "lineage_separation": parent["lineage_separation"],
        "exact_rational_closure": True, "descriptive_accounting_only": True,
    }
    same(result["modal_control_pair_comparison_count"], 84, "required 84 modal pairs")
    require(all(p["modal_control_count"] == 8 and p["pair_comparison_count"] == 28
                for p in partitions), "required eight controls and 28 pairs per partition")
    return result


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
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_CONTROL_EQUIVALENCE_CLASSIFIER",
        "command": COMMAND, "compiled_sources": compiled, "input_pins": PINS,
        "retained_source_pins": SOURCE_PINS,
        "dispatch_and_write_audit": {**retained["dispatch_and_write_audit"], **audit},
        "flags": retained["flags"], "claim_boundary": BOUNDARY,
        "normal_host_review": "REQUIRED", "retained_0146": retained,
        "retained_0146_capture": capture, "retained_0146_review": review, "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
