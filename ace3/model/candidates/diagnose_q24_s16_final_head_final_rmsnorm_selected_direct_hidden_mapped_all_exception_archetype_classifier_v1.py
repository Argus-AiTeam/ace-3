"""Stdout-only descriptive mapped_all archetypes over reviewed retained rationals."""

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
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mapped_all_exception_archetype_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT_NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_control_equivalence_classifier_v1"
CAPTURE_ROOT = ROOT / "build/selected-direct-hidden-modal-equivalence-3fdafbcebc0f-uyn6nk5a"
REVIEW_ROOT = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")


def pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


PINS = {
    "stdout": pin(CAPTURE_ROOT / "check.stdout", 19969930,
                  "16d13d84b23b43670fbd3222216b285b2672e3140d49b0a489890dccd3890200"),
    "capture": pin(CAPTURE_ROOT / "capture.json", 32594,
                   "6f6edd6d456eff74190539bce7c4f090a71f39f7e2e297f76e7ec7ab63f044d5"),
    "review": pin(REVIEW_ROOT / "3fdafbcebc0f/round-0001.json", 690,
                  "f68bdd6b6ee5f2e40632fadce463264e3e39b937b70e67aaeb8cf8925278e67c"),
}
SOURCE_PINS = [
    pin(ROOT / "ace3/model/candidates" / (PARENT_NAME + ".py"), 29410,
        "4678a6aeca7954dccbe1974331895352be2d229895b8fd2b54c75f3ce37713db"),
    pin(ROOT / "tests" / ("test_" + PARENT_NAME + ".py"), 22014,
        "ae49da3f6f73c2a371662c6afa6990a2c797756dd5bde07963d118c555e46416"),
]
NESTED_PINS = (
    ("retained_0146", {
        "stdout": pin(ROOT / "build/selected-direct-hidden-modal-contrast-0146c951608d-pt2zlbcz/check.stdout", 19618193,
                      "233e274d21ddd53f80912ff83e666cc4d3e3db5a8388cde18dd8eb8603b1294f"),
        "capture": pin(ROOT / "build/selected-direct-hidden-modal-contrast-0146c951608d-pt2zlbcz/capture.json", 32049,
                       "9ce9e56b089f5d0351f1e40dd40820e894615b86325d408241b258cc2a97c2d9"),
        "review": pin(REVIEW_ROOT / "0146c951608d/round-0001.json", 690,
                      "b8cd60e1bf22c51be87175bbe88912e3b67bc90568c66e2916592384aa665ba8"),
    }),
    ("retained_6184", {
        "stdout": pin(ROOT / "build/selected-direct-hidden-partition-6184a5063c9d-completion-b6volua0/check.stdout", 19526954,
                      "75c4cd644ad2c264801de7a364b0dc4af9ff1d2c1960a95951e4752a0835a204"),
        "capture": pin(ROOT / "build/selected-direct-hidden-partition-6184a5063c9d-completion-b6volua0/capture.json", 27328,
                       "913dcb43aabbe46dd4124f88d416dce3e193d31a0ef46b1f3b727ca8e0dedbae"),
        "review": pin(REVIEW_ROOT / "6184a5063c9d/round-0001.json", 690,
                      "2deb0d33524599c6082d77ecc35847a5a85855a119704b21cdee77a2f1dd8b6e"),
    }),
    ("retained_e795", {
        "stdout": pin(ROOT / "build/selected-direct-hidden-stability-e795493d08bc-fumvbe57/check.stdout", 18840034,
                      "831c13630634ca0b8f6521aea271667851a8311f455e008af813a66c07ff7bb6"),
        "capture": pin(ROOT / "build/selected-direct-hidden-stability-e795493d08bc-fumvbe57/capture.json", 17021,
                       "247e09a954f046b2b2854522260e2cb1ff054a7e3e5171648b447a906f9385cf"),
        "review": pin(REVIEW_ROOT / "e795493d08bc/round-0001.json", 767,
                      "2cd5adea4c912fd5b648782dbcd605d6b9a59eaa9c1c5b3b14886c2438d9ca8e"),
    }),
    ("retained_449b", {
        "stdout": pin(ROOT / "build/selected-direct-hidden-hotspots-449b1c24ea06-apfh_msz/check.stdout", 14452633,
                      "67318f659a54c5a8e05fbe8acd0f7e209b7fcdbf670183d1e863b13e1a67f9a2"),
        "capture": pin(ROOT / "build/selected-direct-hidden-hotspots-449b1c24ea06-apfh_msz/capture.json", 5327,
                       "21dc61bea5786d8674434780599cd12d565ab2cf6cadd9ba465c3b596c73ffc3"),
        "review": pin(REVIEW_ROOT / "449b1c24ea06/round-0001.json", 688,
                      "566ae7c16df4d327bd028394d112e365fd7822540aed0b4e15bc0a556f0ebc41"),
    }),
    ("retained_50f", {
        "stdout": pin(ROOT / "build/direct-hidden-residual-50f5bb2062c9-bc457u29/stdout.json", 6737063,
                      "30790957680088caecfa2539e10e672f0d5ff373573d4c80a71cb3be2d65cdeb"),
        "capture": pin(ROOT / "build/direct-hidden-residual-50f5bb2062c9-bc457u29/capture.json", 7652,
                       "f29d226f2bc4e99d967b89b9f8baeeba7c71c555be742f58367eefcda3dad464"),
        "review": pin(REVIEW_ROOT / "50f5bb2062c9/round-0001.json", 689,
                      "feb58e79b284e652e9bb028a2ea79b3f4b48a6b9fd18a58f0465367a6a204bb8"),
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
SCOPE_FIELDS = ("table", "left_id", "right_id", "branch")
COMPONENTS = (
    "input_hidden", "attention_stage11", "mlp_stage17",
    "actual_residual_boundary", "negative_reference_residual_boundary",
    "q24_to_fp16_conversion", "fp16_to_branch_terminal_remainder",
)
FIELDS = tuple(c + "." + f for c in ("exception", "modal")
               for f in ("signed", "absolute", "gap_from_maximum", "rank")) + (
    "absolute_comparator", "signed_comparator", "dominance_gap", "rank_difference",
)
CONTRAST_FIELDS = {
    **{c + "_" + f + "_delta": c + "." + f
       for c in ("exception", "modal") for f in ("signed", "absolute", "rank")},
    **{f + "_delta": f for f in ("absolute_comparator", "signed_comparator", "dominance_gap")},
}
LABELS = ("branch", "ignored-build", "compile", "pytest", "check")
SUFFIXES = (".command.txt", ".argv.json", ".environment.json", ".stdout",
            ".stderr", ".whole-command.log")
BOUNDARY = (
    "Descriptive exact retained-rational mapped_all exception archetypes only, "
    "over three reviewed unstable partitions and dependent eight-control modal "
    "classes. No causal, intervention, performance, repair or admission claim. "
    "Only 3fd stdout/capture/source/test/review and capture sidecars are opened; "
    "0146/6184/e795/449b/50f are authenticated nested pins, not reopened, replayed "
    "or re-reviewed. Complete accepted evidence and historical failures are "
    "preserved. Original-input independently propagated global references, exact "
    "thresholds and source/operand/state/KV/lineage gates remain unchanged. "
    "No producer imports, prefix/admission/reference replay, tensor/native decoding, "
    "RMSNorm, head/row-dot or other model operators, closed 480/50f/f0/6184/e795/"
    "0146/3fd computation, or row319 availability work. The missing row-319 "
    "896-element pre-round producer and NOT_RETAINED_NO_RECONSTRUCTION binary64 "
    "internal stages remain limitations. Q24 residual state is wider than FP16; "
    "native S16 RTZ, G128 asymmetric packed INT4, native GEMM nibble ordering, "
    "no qzero plus-one, FP16 scales/operator boundaries/KV remain unchanged. "
    "No strict-FP16-state W4A16, new-token/full-model admission, GPU/RTL/FPGA/"
    "hardware, precision/scale expansion or ACE2 changes. Host Reviewer REQUIRED."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(actual == expected, message)


def bound_bytes(binding):
    data = Path(binding["path"]).read_bytes()
    same(len(data), binding["bytes"], "retained byte count changed: " + binding["path"])
    same(hashlib.sha256(data).hexdigest(), binding["sha256"],
         "retained hash changed: " + binding["path"])
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
            forbidden = (not isinstance(path, (str, bytes, os.PathLike))
                         or os.fsdecode(path) not in allowed
                         or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT |
                                          os.O_TRUNC | os.O_APPEND)))
        elif event == "import":
            forbidden = args[0].startswith(("ace3.", "numpy", "torch", "safetensors", "ctypes"))
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
            raise RuntimeError("retained-only archetype classifier forbids " + event)

    sys.addaudithook(guard)
    try:
        yield
    finally:
        active = False


def zero_counters(values):
    require(bool(values) and all(v is False or type(v) is int and v == 0
                                for v in values.values()), "forbidden retained dispatch/write/claim")


def validate_retained(retained):
    same((retained["diagnostic_id"], retained["status"]),
         (PARENT_NAME, "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_CONTROL_EQUIVALENCE_CLASSIFIER"),
         "wrong retained producer/boundary")
    require(type(retained["version"]) is int and retained["version"] == 1, "version changed")
    same(retained["compiled_sources"], SOURCE_PINS, "source/test splice")
    node = retained
    for child, pins in NESTED_PINS:
        same(node["input_pins"], pins, "nested pins changed: " + child)
        zero_counters(node["dispatch_and_write_audit"])
        zero_counters(node["flags"])
        node = node[child]
    zero_counters(node["dispatch_and_write_audit"])
    zero_counters(node["flags"])
    report = retained["report"]
    require(report["descriptive_accounting_only"] is True
            and report["exact_rational_closure"] is True, "accounting boundary changed")
    same((report["unstable_partition_count"], report["supported_partition_count"],
          report["rejected_partition_count"], report["modal_control_pair_comparison_count"]),
         (3, 3, 0, 84), "reviewed equivalence census changed")


def authenticate():
    review = decode(bound_bytes(PINS["review"]))
    same((review["kind"], review["mission_id"], review["producer_role"], review["review"]["status"]),
         ("round_reviewed_handoff", "3fdafbcebc0f", "reviewer", "done"),
         "independent terminal 3fd review required")
    capture = decode(bound_bytes(PINS["capture"]))
    require(capture["success"] is True and bool(capture["checks"])
            and all(v is True for v in capture["checks"].values()), "retained capture failed")
    same(capture["capture_directory"], str(CAPTURE_ROOT), "capture directory splice")
    preflight = capture["preflight"]
    same((preflight["cwd"], preflight["uid"], preflight["python"]),
         (str(ROOT), 1000, PYTHON), "retained workdir/account/interpreter gate")
    same(capture["sources_after"], preflight["sources"], "retained source drift")
    same(capture["sources_after"][:2], SOURCE_PINS, "retained source/test census")
    same(capture["accepted_artifacts_after"], preflight["accepted_artifacts"],
         "retained accepted-artifact overwrite")
    for binding in SOURCE_PINS:
        bound_bytes(binding)
    same([r["label"] for r in capture["results"]], list(LABELS), "retained command census")
    stdout = None
    for result in capture["results"]:
        require(type(result["exit_status"]) is int and result["exit_status"] == 0
                and result["timed_out"] is False, "retained command failure")
        label = result["label"]
        same([p["path"] for p in result["files"]],
             [str(CAPTURE_ROOT / (label + s)) for s in SUFFIXES], "capture member splice")
        command, argv, environment, output, error, whole = [
            bound_bytes(p) for p in result["files"]]
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
            module = "ace3.model.candidates." + PARENT_NAME
            same(result["command"],
                 f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {module} --check",
                 "retained check command")
            same(decode(argv), [PYTHON, "-B", "-m", module, "--check"], "retained check argv")
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


def field_comparison(mapped, modal):
    left, right = rational(mapped), rational(modal)
    return {
        "modal_class": modal, "mapped_all": mapped, "delta": str(left - right),
        "signs": {"modal_class": sign(right), "mapped_all": sign(left), "delta": sign(left - right)},
        "ratio": {"status": "EXACT", "value": str(left / right)} if right else
                 {"status": "UNDEFINED_ZERO_MODAL_DENOMINATOR", "value": None},
    }


def identity(item, table):
    keys = {"coordinate"} if table == "coordinate" else {"coordinate", "component"}
    same(set(item), keys, "identity shape")
    require(type(item["coordinate"]) is int and 0 <= item["coordinate"] < 896, "coordinate")
    if table == "coordinate_component":
        require(item["component"] in COMPONENTS, "component")
    return item


def comparator(row, partition, control):
    same((row["control"], row["left_id"], row["right_id"], row["branch"]),
         (control, partition["left_id"], partition["right_id"], partition["branch"]),
         "cross-partition comparator")
    winner = "exception" if control == "mapped_all" else "modal"
    same(row["membership"], winner, "control membership")
    values = {}
    for candidate in ("exception", "modal"):
        selected = row[candidate + "_winner_row"]
        keys = ("coordinate",) if partition["table"] == "coordinate" else ("coordinate", "component")
        same(identity({k: selected[k] for k in keys}, partition["table"]),
             partition[candidate + "_winner_id"], "candidate identity")
        for field in ("signed", "absolute", "gap_from_maximum"):
            values[candidate + "." + field] = rational(selected[field])
        require(type(selected["rank"]) is int and selected["rank"] >= 1, "candidate rank")
        values[candidate + ".rank"] = Fraction(selected["rank"])
        same(abs(values[candidate + ".signed"]), values[candidate + ".absolute"], "magnitude closure")
    for field in ("absolute_comparator", "signed_comparator", "dominance_gap"):
        values[field] = rational(row[field])
    require(type(row["rank_difference"]) is int, "rank difference")
    values["rank_difference"] = Fraction(row["rank_difference"])
    for field in ("absolute", "signed"):
        same(values[field + "_comparator"], values["exception." + field] - values["modal." + field],
             "comparator closure")
        require(type(row[field + "_comparator_sign"]) is int, "comparator sign")
        same(row[field + "_comparator_sign"], sign(values[field + "_comparator"]), "sign closure")
    same(values["rank_difference"], values["exception.rank"] - values["modal.rank"], "rank closure")
    same(values[winner + ".rank"], 1, "winner rank")
    same(row["observed_winner_id"], partition[winner + "_winner_id"], "observed winner")
    require(values["dominance_gap"] > 0, "unique retained winner required")
    for candidate in ("exception", "modal"):
        gap = values[winner + ".absolute"] - values[candidate + ".absolute"]
        require(gap >= 0, "negative candidate gap")
        same(values[candidate + ".gap_from_maximum"], gap, "candidate gap closure")
    runners = [identity(i, partition["table"]) for i in row["runner_up_ids"]]
    require(bool(runners) and len({json.dumps(i, sort_keys=True) for i in runners}) == len(runners),
            "runner-up census")
    return {k: str(values[k]) for k in FIELDS}, {
        "exception": partition["exception_winner_id"], "modal": partition["modal_winner_id"],
        "observed_winner": row["observed_winner_id"], "runner_up_ids": runners,
    }


def memberships(fields, identities):
    return {
        "candidate_absolute_tie": fields["exception.absolute"] == fields["modal.absolute"],
        "candidate_signed_tie": fields["exception.signed"] == fields["modal.signed"],
        "candidate_rank_tie": fields["exception.rank"] == fields["modal.rank"],
        "zero_candidate_ids": [identities[c] for c in ("exception", "modal") if fields[c + ".absolute"] == "0"],
        "maximum_candidate_ids": [identities[c] for c in ("exception", "modal") if fields[c + ".gap_from_maximum"] == "0"],
        "runner_up_tied": len(identities["runner_up_ids"]) > 1,
    }


def classify_partition(equivalence, partition):
    same(tuple(partition[k] for k in SCOPE_FIELDS), tuple(equivalence[k] for k in SCOPE_FIELDS),
         "partition splice")
    same(partition["control_membership"],
         {"exception": ["mapped_all"], "modal": list(MODAL_CONTROLS)}, "eight plus one membership")
    same((equivalence["classification"], equivalence["modal_control_count"],
          equivalence["equivalent_pair_count"], equivalence["pair_comparison_count"]),
         ("SUPPORTED", 8, 28, 28), "reviewed modal class required")
    same(equivalence["equivalence_classes"], [list(MODAL_CONTROLS)], "one modal class required")
    same(equivalence["modal_controls"], list(MODAL_CONTROLS), "modal class order")
    profiles = equivalence["control_profiles"]
    same([p["control"] for p in profiles], list(MODAL_CONTROLS), "modal profile census")
    canonical = profiles[0]
    signature = {k: v for k, v in canonical.items() if k not in ("control", "roles")}
    for profile in profiles:
        same({k: v for k, v in profile.items() if k not in ("control", "roles")},
             signature, "non-equivalent modal member")
    expected_ids = ({"coordinate": 62}, {"coordinate": 241})
    if partition["table"] == "coordinate_component":
        expected_ids = ({"coordinate": 241, "component": COMPONENTS[-1]},
                        {"coordinate": 241, "component": "mlp_stage17"})
    same((partition["exception_winner_id"], partition["modal_winner_id"]),
         expected_ids, "reviewed candidate identities")
    rows = partition["same_partition_comparator_rows"]
    same([r["control"] for r in rows], list(CONTROLS), "comparator census")
    by_control = {r["control"]: r for r in rows}
    modal, modal_ids = comparator(by_control[MODAL_CONTROLS[0]], partition, MODAL_CONTROLS[0])
    mapped, mapped_ids = comparator(by_control["mapped_all"], partition, "mapped_all")
    same({k: canonical["fields"]["comparator." + k] for k in FIELDS}, modal, "canonical field binding")
    same(canonical["identities"], modal_ids, "canonical identity binding")
    modal_ties, mapped_ties = memberships(modal, modal_ids), memberships(mapped, mapped_ids)
    same(canonical["ties"], modal_ties, "canonical tie binding")
    fields = {k: field_comparison(mapped[k], modal[k]) for k in FIELDS}
    contrast = {k: fields[f]["delta"] for k, f in CONTRAST_FIELDS.items()}
    same(set(canonical["fields"]), {"comparator." + k for k in FIELDS}
         | {"contrast." + k for k in CONTRAST_FIELDS}, "retained field census")
    same({k: canonical["fields"]["contrast." + k] for k in CONTRAST_FIELDS},
         contrast, "retained contrast binding")
    same(canonical["signs"], {k: sign(rational(v)) for k, v in canonical["fields"].items()},
         "canonical signs")
    contrasts = partition["mapped_all_vs_modal_contrast_rows"]
    same([c["modal_control"] for c in contrasts], list(MODAL_CONTROLS), "contrast census")
    for row in contrasts:
        same(row["exception_control"], "mapped_all", "contrast exception")
        same({k: str(row[k]) for k in CONTRAST_FIELDS}, contrast, "class contrast binding")
    delta_pattern = [fields[c + ".absolute"]["signs"]["delta"] for c in ("exception", "modal")]
    archetype = {
        (-1, -1): "BOTH_CANDIDATES_CONTRACT",
        (0, -1): "EXCEPTION_FIXED_MODAL_CONTRACTS",
    }.get(tuple(delta_pattern), "OTHER_EXACT_MAGNITUDE_PATTERN")
    categories = {
        "magnitude_archetype": archetype,
        "candidate_absolute_delta_signs": delta_pattern,
        "candidate_signed_signs": {side: [fields[c + ".signed"]["signs"][side] for c in ("exception", "modal")]
                                   for side in ("modal_class", "mapped_all")},
        "comparator_signs": {f: fields[f]["signs"] for f in ("absolute_comparator", "signed_comparator")},
        "rank_transition": {k: [modal[k], mapped[k]] for k in ("exception.rank", "modal.rank", "rank_difference")},
        "dominance_gap_delta_sign": fields["dominance_gap"]["signs"]["delta"],
        "winner_changed": modal_ids["observed_winner"] != mapped_ids["observed_winner"],
        "winner_runner_up_exchange": (modal_ids["runner_up_ids"] == [mapped_ids["observed_winner"]]
                                     and mapped_ids["runner_up_ids"] == [modal_ids["observed_winner"]]),
        "ties": {side: {k: v for k, v in ties.items() if not k.endswith("_ids")}
                 for side, ties in (("modal_class", modal_ties), ("mapped_all", mapped_ties))},
        "candidate_membership": {side: {
            "maximum": [c for c in ("exception", "modal") if ids[c] in ties["maximum_candidate_ids"]],
            "zero": [c for c in ("exception", "modal") if ids[c] in ties["zero_candidate_ids"]],
            "runner_up": [c for c in ("exception", "modal") if ids[c] in ids["runner_up_ids"]],
        } for side, ids, ties in (("modal_class", modal_ids, modal_ties), ("mapped_all", mapped_ids, mapped_ties))},
    }
    return {
        **{k: partition[k] for k in SCOPE_FIELDS},
        "modal_class": {"canonical_control": MODAL_CONTROLS[0], "controls": list(MODAL_CONTROLS), "count": 8},
        "exception_class": {"controls": ["mapped_all"], "count": 1},
        "field_comparisons": fields, "contrast_vector": contrast,
        "identity_membership": {"modal_class": modal_ids, "mapped_all": mapped_ids},
        "tie_membership": {"modal_class": modal_ties, "mapped_all": mapped_ties},
        "categorical_signature": categories,
    }


def cross_partition(partitions):
    classes = []
    for p in partitions:
        signature = p["categorical_signature"]
        group = next((g for g in classes if g["signature"] == signature), None)
        if group is None:
            group = {"signature": signature, "partitions": []}
            classes.append(group)
        group["partitions"].append({k: p[k] for k in SCOPE_FIELDS})
    pairs = []
    for left, right in combinations(partitions, 2):
        a, b = left["categorical_signature"], right["categorical_signature"]
        pairs.append({
            "left": {k: left[k] for k in SCOPE_FIELDS}, "right": {k: right[k] for k in SCOPE_FIELDS},
            "equal_categories": [k for k in a if a[k] == b[k]],
            "different_categories": [k for k in a if a[k] != b[k]],
            "absolute_rank_gap_fields_equal": all(
                left["field_comparisons"][k] == right["field_comparisons"][k]
                for k in FIELDS if "signed" not in k),
            "signed_fields_are_exact_negatives": all(
                rational(left["field_comparisons"][k][side]) == -rational(right["field_comparisons"][k][side])
                for k in FIELDS if "signed" in k for side in ("modal_class", "mapped_all", "delta")),
            "identity_membership_equal": left["identity_membership"] == right["identity_membership"],
        })
    signatures = [p["categorical_signature"] for p in partitions]
    return {
        "categorical_classes": classes, "pair_comparisons": pairs,
        "shared_categories": {k: v for k, v in signatures[0].items() if all(s[k] == v for s in signatures[1:])},
        "magnitude_archetype_groups": [
            {"archetype": name, "partitions": [{k: p[k] for k in SCOPE_FIELDS} for p in partitions
                                             if p["categorical_signature"]["magnitude_archetype"] == name]}
            for name in dict.fromkeys(s["magnitude_archetype"] for s in signatures)],
    }


def report(retained):
    validate_retained(retained)
    equivalences = retained["report"]["partitions"]
    raw = retained["retained_0146"]["report"]["partitions"]
    for partitions in (equivalences, raw):
        same([tuple(p[k] for k in SCOPE_FIELDS) for p in partitions], list(SCOPES), "exact three scopes")
    partitions = [classify_partition(e, p) for e, p in zip(equivalences, raw, strict=True)]
    return {
        "unstable_partition_count": 3, "modal_class_count": 3, "singleton_exception_count": 3,
        "partitions": partitions, "cross_partition": cross_partition(partitions),
        "canonical_rule": "First control in authenticated retained modal order; equivalence excludes only control and roles.",
        "delta_rule": "mapped_all minus canonical modal class for every comparator field and contrast vector.",
        "ratio_rule": "mapped_all divided by canonical modal field; zero modal denominator is explicitly undefined, never zero or infinity.",
        "classification_rule": "Categorical exact magnitude, sign, rank, winner, tie, identity and dominance-gap signatures; descriptive, not an intervention or cause.",
        "reference_scope": retained["report"]["reference_scope"],
        "lineage_separation": retained["report"]["lineage_separation"],
        "exact_rational_closure": True, "descriptive_accounting_only": True,
    }


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
            compiled.append(pin(path, len(data), hashlib.sha256(data).hexdigest()))
        retained, capture, review = authenticate()
        result = report(retained)
    same(audit, {"forbidden_calls": 0}, "forbidden dispatch/write")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_MAPPED_ALL_EXCEPTION_ARCHETYPE_CLASSIFIER",
        "command": COMMAND, "compiled_sources": compiled, "input_pins": PINS,
        "retained_source_pins": SOURCE_PINS, "normal_host_review": "REQUIRED",
        "dispatch_and_write_audit": {**retained["dispatch_and_write_audit"], **audit},
        "flags": retained["flags"], "claim_boundary": BOUNDARY,
        "retained_3fd": retained, "retained_3fd_capture": capture,
        "retained_3fd_review": review, "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
