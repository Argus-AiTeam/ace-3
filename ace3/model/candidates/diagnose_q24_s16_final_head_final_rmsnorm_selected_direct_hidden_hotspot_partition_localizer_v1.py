"""Stdout-only exact partition/exception localization over reviewed e795 accounts."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_hotspot_partition_localizer_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT_NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_hotspot_stability_classifier_v1"
CAPTURE_ROOT = ROOT / "build/selected-direct-hidden-stability-e795493d08bc-fumvbe57"
PINS = {
    "stdout": {
        "path": str(CAPTURE_ROOT / "check.stdout"), "bytes": 18840034,
        "sha256": "831c13630634ca0b8f6521aea271667851a8311f455e008af813a66c07ff7bb6",
    },
    "capture": {
        "path": str(CAPTURE_ROOT / "capture.json"), "bytes": 17021,
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
        "path": str(ROOT / "build/direct-hidden-residual-50f5bb2062c9-bc457u29/capture.json"),
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
PAIRS = ((34319, 319), (34319, 13), (319, 34319))
BRANCHES = ("binary64", "fp16")
COMPONENTS = (
    "input_hidden", "attention_stage11", "mlp_stage17",
    "actual_residual_boundary", "negative_reference_residual_boundary",
    "q24_to_fp16_conversion", "fp16_to_branch_terminal_remainder",
)
TABLES = ("coordinate", "coordinate_component")
ACCOUNT_FIELDS = ("control", "left_id", "right_id", "branch")
PARTITIONS = {
    "by_branch": ("branch",),
    "by_control": ("control",),
    "by_ordered_pair": ("left_id", "right_id"),
    "by_control_branch": ("control", "branch"),
    "by_pair_branch": ("left_id", "right_id", "branch"),
    "by_control_pair": ("control", "left_id", "right_id"),
}
EXPECTED_COUNTS = {
    "coordinate": [{"coordinate": 62, "count": 20}, {"coordinate": 241, "count": 34}],
    "coordinate_component": [
        {"coordinate": 62, "component": "input_hidden", "count": 36},
        {"coordinate": 241, "component": "mlp_stage17", "count": 17},
        {"coordinate": 241, "component": COMPONENTS[-1], "count": 1},
    ],
}
EXCEPTION_GAPS = {
    "coordinate": "1827610068439962304726431454702255/1329227995784915872903807060280344576",
    "coordinate_component": "3832757189268268603981093761768175/1329227995784915872903807060280344576",
}
BOUNDARY = (
    "Descriptive partition/exception accounting only over the same 54 dependent "
    "retained e795 control/ordered-pair/branch accounts. Modal winners are "
    "frequency labels, not causal baselines or independent observations. No "
    "performance attribution, intervention, model repair or successor result. "
    "The complete retained_e795, retained_449b and retained_50f evidence preserves "
    "exact thresholds, independently propagated original-input global references, "
    "source/operand/state/KV/lineage gates and historical failures unchanged. "
    "Only e795 capture/source/test/review bytes are reopened, not ancestor producers "
    "or tensors. Binary64 internal stages remain NOT_RETAINED_NO_RECONSTRUCTION; "
    "the independent terminal remainder is separate. Q24 residual state is wider "
    "than FP16. Native S16 RTZ, official G128 asymmetric packed INT4 with native "
    "GEMM nibble ordering and no qzero plus-one, FP16 scales/operator boundaries/KV "
    "remain unchanged. No prefix/admission/reference or accepted-producer replay, "
    "tensor/native decoding, RMSNorm, head/row-dot or closed 480/50f/f0/row319 "
    "execution. The row-319 missing 896-element pre-round producer remains a "
    "closed limitation. No strict-FP16-state W4A16, new-token/full-model admission, "
    "GPU/RTL/FPGA/hardware, precision/scale expansion or ACE2 changes. "
    "Normal independent Host Reviewer closure REQUIRED."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(actual == expected, message)


def record(path):
    data = Path(path).read_bytes()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


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
        str(SOURCE), str(TEST), *(pin["path"] for pin in PINS.values()),
        str(ROOT / "ace3/model/candidates" / (PARENT_NAME + ".py")),
        str(ROOT / "tests" / ("test_" + PARENT_NAME + ".py")),
        *(str(CAPTURE_ROOT / (label + suffix))
          for label in ("branch", "ignored-build", "pytest", "check")
          for suffix in (".command.txt", ".environment.json", ".stdout",
                         ".stderr", ".whole-command.log")),
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
            raise RuntimeError("retained-only localizer forbids " + event)

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
         ("round_reviewed_handoff", "e795493d08bc", "reviewer", "done"),
         "independent terminal e795 review required")


def validate_retained(retained):
    same(retained["diagnostic_id"], PARENT_NAME, "wrong retained producer")
    same(retained["status"], "READ_ONLY_SELECTED_DIRECT_HIDDEN_HOTSPOT_STABILITY_CLASSIFIER",
         "wrong retained boundary")
    require(type(retained["version"]) is int and retained["version"] == 1,
            "retained version changed")
    same(retained["input_pins"], RETAINED_449B_PINS, "nested 449b pins changed")
    parent = retained["retained_449b"]
    same(parent["input_pins"], RETAINED_50F_PINS, "nested 50f pins changed")
    for node in (retained, parent, parent["retained_50f"]):
        zero_counters(node["dispatch_and_write_audit"])
        zero_counters(node["flags"])
    require(retained["report"]["descriptive_accounting_only"] is True and
            retained["report"]["exact_gap_and_tie_closure"] is True,
            "retained accounting boundary changed")


def authenticate():
    validate_review(decode(bound_bytes(PINS["review"])))
    capture = decode(bound_bytes(PINS["capture"]))
    require(capture["success"] is True and bool(capture["checks"]) and
            all(v is True for v in capture["checks"].values()), "retained capture failed")
    same(capture["capture_directory"], str(CAPTURE_ROOT), "capture directory splice")
    preflight = capture["preflight"]
    same((preflight["cwd"], preflight["uid"], preflight["python"]),
         (str(ROOT), 1000, PYTHON), "retained source/account/interpreter gate")
    same(capture["sources_after"], preflight["sources"], "retained source drift")
    same(capture["accepted_artifacts_after"], preflight["accepted_artifacts"],
         "retained accepted-artifact overwrite")
    sources = capture["sources_after"][:2]
    same([p["path"] for p in sources], [
        str(ROOT / "ace3/model/candidates" / (PARENT_NAME + ".py")),
        str(ROOT / "tests" / ("test_" + PARENT_NAME + ".py")),
    ], "retained source/test census")
    for pin in sources:
        bound_bytes(pin)
    same([r["label"] for r in capture["results"]],
         ["branch", "ignored-build", "pytest", "check"], "retained command census")
    stdout = None
    for result in capture["results"]:
        require(type(result["exit_status"]) is int and result["exit_status"] == 0,
                "retained command failure")
        label = result["label"]
        same([p["path"] for p in result["files"]], [
            str(CAPTURE_ROOT / (label + suffix)) for suffix in (
                ".command.txt", ".environment.json", ".stdout", ".stderr",
                ".whole-command.log")], "capture member splice")
        data = [bound_bytes(pin) for pin in result["files"]]
        same(data[0], (result["command"] + "\n").encode(), "command binding")
        same(decode(data[1]), {k: preflight[k] for k in (
            "cwd", "uid", "python", "python_version", "environment")}, "environment binding")
        same(data[3], b"", "retained stderr not empty")
        require(all(part in data[4] for part in data[:3]), "whole-command binding")
        if label == "branch":
            same(data[2], b"argus/full-projection\n", "retained branch changed")
        if label == "check":
            same(result["command"],
                 f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m "
                 f"ace3.model.candidates.{PARENT_NAME} --check", "retained command changed")
            same(result["files"][2], PINS["stdout"], "capture/stdout splice")
            stdout = data[2]
    require(stdout is not None, "missing retained stdout")
    retained = decode(stdout)
    same(retained["compiled_sources"], sources, "retained compiled source/test splice")
    validate_retained(retained)
    return retained


def rational(text):
    require(type(text) is str, "rational must be an exact string")
    value = Fraction(text)
    same(str(value), text, "noncanonical rational")
    return value


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


def localize_table(table, kind):
    same(kind in TABLES, True, "unsupported hotspot table")
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
    require(top > second and top > 0, "retained domain requires unique nonzero winners")
    for row, value in zip(rows, values, strict=True):
        same(abs(rational(row["signed"])), value, "signed/absolute closure")
        require(type(row["rank"]) is int, "noninteger rank")
        same(row["rank"], 1 + sum(v > value for v in values), "rank closure")
        same(rational(row["gap_from_maximum"]), top - value, "row gap closure")
    for field, expected in (
        ("row_count", len(rows)), ("zero_row_count", values.count(Fraction())),
        ("maximum_tie_count_including_zeros", 1),
    ):
        require(type(table[field]) is int and table[field] == expected,
                "retained count closure: " + field)
    for field, expected in (
        ("maximum_absolute", top), ("runner_up_absolute", second),
        ("dominance_gap", top - second), ("next_distinct_absolute", second),
        ("gap_to_next_distinct", top - second),
    ):
        same(rational(table[field]), expected, "retained gap closure: " + field)
    same(table["winner_ids"], ids[:1], "retained winner splice")
    same(table["maximizer_ids_including_zeros"], ids[:1], "retained maximizer splice")
    require(table["unique_nonzero_hotspot"] is True and table["all_zero"] is False
            and table["exact_gap_and_tie_closure"] is True, "retained uniqueness changed")
    return {
        "winner_id": ids[0],
        "runner_up_ids": [ident for ident, v in zip(ids, values, strict=True) if v == second],
        "maximum_absolute": str(top), "runner_up_absolute": str(second),
        "dominance_gap": str(top - second),
    }


def partition(members):
    require(bool(members), "empty partition")
    groups = {}
    for member in members:
        groups.setdefault(identity_key(member["winner_id"]), []).append(member)
    classes = []
    for key in sorted(groups):
        group = groups[key]
        gaps = [rational(m["dominance_gap"]) for m in group]
        classes.append({
            "winner_id": group[0]["winner_id"], "count": len(group), "members": group,
            "minimum_dominance_gap": str(min(gaps)),
            "maximum_dominance_gap": str(max(gaps)),
        })
    maximum_count = max(c["count"] for c in classes)
    modes = [c["winner_id"] for c in classes if c["count"] == maximum_count]
    unique_mode = modes[0] if len(modes) == 1 else None
    gaps = [rational(m["dominance_gap"]) for m in members]
    return {
        "account_count": len(members), "stable": len(classes) == 1,
        "classification": "STABLE_UNIQUE_HOTSPOT" if len(classes) == 1 else "VARIABLE_HOTSPOTS",
        "winner_counts": [{**c["winner_id"], "count": c["count"]} for c in classes],
        "winner_classes": classes, "modal_winner_ids": modes,
        "unique_modal_winner_id": unique_mode,
        "exceptions": [] if unique_mode is None else
        [m for m in members if m["winner_id"] != unique_mode],
        "minimum_dominance_gap": str(min(gaps)),
        "maximum_dominance_gap": str(max(gaps)),
    }


def report(retained):
    validate_retained(retained)
    parent = retained["report"]
    accounts = parent["accounts"]
    same([tuple(a[k] for k in ACCOUNT_FIELDS) for a in accounts],
         [(c, l, r, b) for c in CONTROLS for l, r in PAIRS for b in BRANCHES],
         "exact 54-account identity/order census changed")
    for key, value in (
        ("control_count", 9), ("diagnostic_pair_count", 27), ("pair_branch_count", 54),
        ("coordinate62_accounts", 54), ("selected_coordinate_accounts", 1035),
        ("weighted_component_count", 7245),
    ):
        require(type(parent[key]) is int and parent[key] == value,
                "retained census changed: " + key)
    same(sum(a["selected_coordinate_count"] for a in accounts), 1035,
         "selected-coordinate census changed")
    same(sum(a["weighted_component_count"] for a in accounts), 7245,
         "weighted-component census changed")
    tables = {}
    for kind in TABLES:
        members = []
        for a in accounts:
            table = a["tables"][kind]
            expected_count = a["selected_coordinate_count"] * (7 if kind == TABLES[1] else 1)
            same(len(table["rows"]), expected_count, "retained row census changed")
            members.append({
                **{k: a[k] for k in ACCOUNT_FIELDS}, "roles": a["roles"],
                **localize_table(table, kind),
            })
        global_partition = partition(members)
        same(global_partition["winner_counts"],
             parent["global"]["tables"][kind]["unique_winner_counts"],
             "retained global winner counts changed")
        partitions, maps = {}, {}
        for name, fields in PARTITIONS.items():
            groups = {}
            for member in members:
                key = tuple(member[f] for f in fields)
                groups.setdefault(key, []).append(member)
            partitions[name] = [
                {**dict(zip(fields, key, strict=True)), **partition(group)}
                for key, group in groups.items()
            ]
            maps[name] = {
                state: [{k: p[k] for k in fields} for p in partitions[name]
                        if p["stable"] is stable]
                for state, stable in (("stable", True), ("unstable", False))
            }
        tables[kind] = {
            "accounts": members, "global": global_partition, "partitions": partitions,
            "partition_map": maps,
            "pair_branch_exceptions": [
                m for p in partitions["by_pair_branch"] for m in p["exceptions"]],
        }
    return {
        "pair_branch_count": len(accounts), "localized_table_count": len(TABLES),
        "selected_coordinate_accounts": 1035, "weighted_component_count": 7245,
        "tables": tables,
        **{k: parent[k] for k in ("lineage_separation", "reference_scope", "ranking_rule")},
        "exception_rule": (
            "An exception differs from its partition's unique frequency-modal winner. "
            "Frequency ties have no distinguished baseline and no labeled exceptions; "
            "all modal identities and complete winner-class membership remain explicit. "
            "Partitions overlap; their counts must not be pooled as independent samples."
        ),
        "gap_rule": "Exact maximum absolute mass minus second-row absolute mass; no float conversion.",
        "conclusion": (
            "Within the retained domain, all pair/branch minority-winner exceptions "
            "are mapped_all: the two reversed outer binary64 coordinate partitions "
            "and the middle binary64 coordinate-component partition. This localizes "
            "winner variability descriptively, not its cause or an intervention."
        ),
        "exact_gap_closure": True, "descriptive_accounting_only": True,
    }


def require_expected(result):
    expected_unstable = {
        "coordinate": [(34319, 319, "binary64"), (319, 34319, "binary64")],
        "coordinate_component": [(34319, 13, "binary64")],
    }
    for kind, table in result["tables"].items():
        same(table["global"]["winner_counts"], EXPECTED_COUNTS[kind],
             "required retained global winners changed")
        unstable = [p for p in table["partitions"]["by_pair_branch"] if not p["stable"]]
        same([(p["left_id"], p["right_id"], p["branch"]) for p in unstable],
             expected_unstable[kind], "required pair/branch instability changed")
        modal = {"coordinate": 241}
        exception = {"coordinate": 62}
        if kind == "coordinate_component":
            modal["component"] = "mlp_stage17"
            exception = {"coordinate": 241, "component": COMPONENTS[-1]}
        for p in unstable:
            same(p["account_count"], 9, "pair/branch account count changed")
            same(p["unique_modal_winner_id"], modal, "modal identity changed")
            same(sorted(c["count"] for c in p["winner_classes"]), [1, 8],
                 "required eight/one split changed")
            same(len(p["exceptions"]), 1, "exception count changed")
            member = p["exceptions"][0]
            same((member["control"], member["winner_id"], member["dominance_gap"]),
                 ("mapped_all", exception, EXCEPTION_GAPS[kind]),
                 "required exception control/identity/gap changed")


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
        retained = authenticate()
        result = report(retained)
        require_expected(result)
    same(audit, {"forbidden_calls": 0}, "forbidden dispatch or write")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_HOTSPOT_PARTITION_LOCALIZER",
        "command": COMMAND, "compiled_sources": compiled, "input_pins": PINS,
        "dispatch_and_write_audit": {**retained["dispatch_and_write_audit"], **audit},
        "flags": retained["flags"], "claim_boundary": BOUNDARY,
        "normal_host_review": "REQUIRED", "retained_e795": retained, "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
