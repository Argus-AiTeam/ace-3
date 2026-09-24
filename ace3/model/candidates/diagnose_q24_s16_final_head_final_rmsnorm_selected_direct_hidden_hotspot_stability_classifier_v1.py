"""Stdout-only exact stability classification of the reviewed 449b rankings."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_hotspot_stability_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT_NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_component_hotspot_audit_v1"
CAPTURE_ROOT = ROOT / "build/selected-direct-hidden-hotspots-449b1c24ea06-apfh_msz"
PINS = {
    "stdout": {
        "path": str(CAPTURE_ROOT / "check.stdout"), "bytes": 14452633,
        "sha256": "67318f659a54c5a8e05fbe8acd0f7e209b7fcdbf670183d1e863b13e1a67f9a2",
    },
    "capture": {
        "path": str(CAPTURE_ROOT / "capture.json"), "bytes": 5327,
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
COMPONENTS = (
    "input_hidden", "attention_stage11", "mlp_stage17",
    "actual_residual_boundary", "negative_reference_residual_boundary",
    "q24_to_fp16_conversion", "fp16_to_branch_terminal_remainder",
)
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
    "mapped_all", "inherited_native",
)
PAIRS = ((34319, 319), (34319, 13), (319, 34319))
BRANCHES = ("binary64", "fp16")
TABLES = ("component", "coordinate", "coordinate_component", *COMPONENTS)
BOUNDARY = (
    "Descriptive retained-ranking stability only, not causal allocation, a "
    "performance attribution, intervention or successor. Partitions reuse "
    "the same 54 dependent control/ordered-pair/branch accounts; counts are "
    "not pooled independent observations or extrapolations beyond selected "
    "unions. Component magnitude is sum of coordinate absolutes, not absolute "
    "net signed mass. Zero-only tables have no nonzero hotspot. All retained "
    "449b/50f pins, exact thresholds, independently propagated original-input "
    "global references, source/operand/state/KV/lineage gates and historical "
    "failures remain unchanged in retained_449b, including retained_50f. "
    "Only 449b capture/source/test/review bytes are reopened, never tensors "
    "or historical producers. Binary64 internal stages remain "
    "NOT_RETAINED_NO_RECONSTRUCTION; the independent terminal remainder stays "
    "separate. Q24 residual state is wider than FP16. Native S16 RTZ, official "
    "G128 asymmetric packed INT4 native GEMM nibble ordering without qzero "
    "plus-one, FP16 scales/operator boundaries/KV are unchanged. No producer, "
    "tensor decoder, RMSNorm, head/row-dot, native decoder, prefix/admission/"
    "reference, closed 480/50f/f0/row319 replay; no strict-FP16-state W4A16, "
    "new-token/full-model admission, hardware/GPU/RTL/FPGA or ACE2 changes. "
    "The row-319 missing 896-element pre-round producer remains a closed "
    "limitation. Normal independent Host Reviewer closure REQUIRED."
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
        str(SOURCE), str(TEST), *(p["path"] for p in PINS.values()),
        str(ROOT / "ace3/model/candidates" / (PARENT_NAME + ".py")),
        str(ROOT / "tests" / ("test_" + PARENT_NAME + ".py")),
        *(str(CAPTURE_ROOT / (label + suffix))
          for label in ("pytest", "check")
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


def authenticate():
    review = decode(bound_bytes(PINS["review"]))
    same((review["mission_id"], review["producer_role"], review["review"]["status"]),
         ("449b1c24ea06", "reviewer", "done"), "independent reviewed input required")
    capture = decode(bound_bytes(PINS["capture"]))
    require(capture["success"] is True and capture["sources_unchanged"] is True,
            "retained capture failed")
    same(capture["capture_directory"], str(CAPTURE_ROOT), "capture directory splice")
    same((capture["preflight"]["cwd"], capture["preflight"]["uid"]),
         (str(ROOT), 1000), "retained source/account gate")
    same(capture["sources_after"], capture["preflight"]["sources"], "source drift")
    expected_sources = [
        str(ROOT / "ace3/model/candidates" / (PARENT_NAME + ".py")),
        str(ROOT / "tests" / ("test_" + PARENT_NAME + ".py")),
    ]
    same([p["path"] for p in capture["sources_after"]], expected_sources,
         "producer source/test census")
    for pin in capture["sources_after"]:
        bound_bytes(pin)
    same([r["label"] for r in capture["results"]], ["pytest", "check"],
         "retained validation census")
    stdout = None
    for result in capture["results"]:
        same(result["exit_status"], 0, "retained command failure")
        label = result["label"]
        expected_paths = [str(CAPTURE_ROOT / (label + suffix)) for suffix in (
            ".command.txt", ".environment.json", ".stdout", ".stderr",
            ".whole-command.log")]
        same([p["path"] for p in result["files"]], expected_paths,
             "capture member splice")
        data = [bound_bytes(pin) for pin in result["files"]]
        same(data[0], (result["command"] + "\n").encode(), "command binding")
        same(data[3], b"", "retained stderr not empty")
        if label == "check":
            expected_command = (
                f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m "
                f"ace3.model.candidates.{PARENT_NAME} --check")
            same(result["command"], expected_command, "retained command changed")
            require(bool(result["checks"]) and
                    all(v is True for v in result["checks"].values()),
                    "retained semantic check failed")
            same(result["files"][2], PINS["stdout"], "capture/stdout splice")
            stdout = data[2]
    require(stdout is not None, "missing retained stdout")
    retained = decode(stdout)
    same(retained["diagnostic_id"], PARENT_NAME, "wrong retained producer")
    same(retained["status"], "READ_ONLY_SELECTED_DIRECT_HIDDEN_COMPONENT_HOTSPOT_AUDIT",
         "wrong retained boundary")
    same(retained["version"], 1, "retained version changed")
    same(retained["compiled_sources"], capture["sources_after"], "compiled source splice")
    same(retained["input_pins"], RETAINED_50F_PINS, "retained 50f pins changed")
    zero_counters(retained["dispatch_and_write_audit"])
    zero_counters(retained["flags"])
    return retained


def rational(text):
    require(type(text) is str, "rational must be an exact string")
    value = Fraction(text)
    same(str(value), text, "noncanonical rational")
    return value


def identity(row):
    result = {k: row[k] for k in ("coordinate", "component") if k in row}
    require(bool(result), "missing ranked identity")
    if "coordinate" in result:
        require(type(result["coordinate"]) is int and 0 <= result["coordinate"] < 896,
                "invalid coordinate identity")
    if "component" in result:
        require(result["component"] in COMPONENTS, "invalid component identity")
    return result


def identity_key(row):
    return (row.get("coordinate", -1),
            COMPONENTS.index(row["component"]) if "component" in row else -1)


def classify_table(table, expected=None):
    rows = table["ranking"]
    require(bool(rows), "empty ranking")
    ids = [identity(row) for row in rows]
    keys = [identity_key(row) for row in ids]
    require(len(set(keys)) == len(keys), "duplicate ranked identity")
    values = [rational(row["absolute"]) for row in rows]
    require(all(v >= 0 for v in values), "negative absolute mass")
    require(list(zip(values, keys)) ==
            sorted(zip(values, keys), key=lambda item: (-item[0], item[1])),
            "ranking order changed")
    if expected is not None:
        same({key: (rational(row["signed"]), value)
              for key, row, value in zip(keys, rows, values, strict=True)},
             expected, "ranked signed/absolute account splice")
    for row, value in zip(rows, values, strict=True):
        require(type(row["rank"]) is int, "noninteger competition rank")
        same(row["rank"], 1 + sum(v > value for v in values), "rank closure")
    maximum = values[0]
    maximizers = [row for row, v in zip(rows, values, strict=True) if v == maximum]
    winners = maximizers if maximum else []
    same(table["maximum_ties"], winners, "maximum tie closure")
    require(table["nonzero_hotspot"] is bool(maximum), "nonzero hotspot closure")
    runner = values[1] if len(values) > 1 else Fraction()
    distinct = next((v for v in values if v < maximum), None)
    return {
        "row_count": len(rows), "zero_row_count": values.count(Fraction()),
        "all_zero": not bool(maximum),
        "maximum_absolute": str(maximum),
        "runner_up_absolute": str(runner),
        "dominance_gap": str(maximum - runner),
        "next_distinct_absolute": None if distinct is None else str(distinct),
        "gap_to_next_distinct": None if distinct is None else str(maximum - distinct),
        "maximizer_ids_including_zeros": [identity(row) for row in maximizers],
        "maximum_tie_count_including_zeros": len(maximizers),
        "winner_ids": [identity(row) for row in winners],
        "unique_nonzero_hotspot": len(winners) == 1,
        "rows": [
            {**ident, "signed": row["signed"], "absolute": str(value),
             "rank": row["rank"], "gap_from_maximum": str(maximum - value)}
            for ident, row, value in zip(ids, rows, values, strict=True)
        ],
        "exact_gap_and_tie_closure": True,
    }


def classify_account(account, branch):
    same(account["hidden_reference"], "original_input_L23_" + branch,
         "original-input reference splice")
    same(account["residual_internal_reference"], "original_input_L23_fp16",
         "internal reference splice")
    same(account["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION",
         "binary64 internal reconstruction")
    require(account["exact_final_margin_identity"] is True, "retained margin failure")
    rows = account["selected_coordinates"]
    coordinates = [row["coordinate"] for row in rows]
    require(all(type(i) is int and 0 <= i < 896 for i in coordinates) and
            coordinates == sorted(set(coordinates)) and 62 in coordinates and
            8 <= len(coordinates) <= 25, "selected union changed")
    cells, direct = {}, {}
    component_signed = [Fraction() for _ in COMPONENTS]
    component_absolute = [Fraction() for _ in COMPONENTS]
    for row in rows:
        same(set(row["weighted_components"]), set(COMPONENTS), "component census changed")
        values = [rational(row["weighted_components"][k]) for k in COMPONENTS]
        total = rational(row["direct_hidden_weighted_term"])
        same(sum(values, Fraction()), total, "selected direct-hidden closure")
        direct[(row["coordinate"], -1)] = (total, abs(total))
        for i, value in enumerate(values):
            cells[(row["coordinate"], i)] = (value, abs(value))
            component_signed[i] += value
            component_absolute[i] += abs(value)
        if branch == "fp16":
            same(values[-1], Fraction(), "FP16 terminal remainder changed")
    component = {(-1, i): (component_signed[i], component_absolute[i])
                 for i in range(len(COMPONENTS))}
    same(set(account["per_component_coordinate_hotspots"]), set(COMPONENTS),
         "per-component census changed")
    tables = {
        "component": classify_table(
            account["component_hotspots_by_coordinate_absolute_mass"], component),
        "coordinate": classify_table(account["coordinate_hotspots"], direct),
        "coordinate_component": classify_table(account["coordinate_component_hotspots"], cells),
    }
    for i, key in enumerate(COMPONENTS):
        tables[key] = classify_table(account["per_component_coordinate_hotspots"][key],
                                    {k: v for k, v in cells.items() if k[1] == i})
    return {"selected_coordinate_count": len(rows),
            "weighted_component_count": len(cells), "tables": tables}


def summarize(tables):
    require(bool(tables), "empty stability partition")
    counts, unique_counts, identities = {}, {}, {}
    winner_sets = []
    for table in tables:
        keys = tuple(identity_key(row) for row in table["winner_ids"])
        winner_sets.append(keys)
        for row, key in zip(table["winner_ids"], keys, strict=True):
            identities[key] = row
            counts[key] = counts.get(key, 0) + 1
            if table["unique_nonzero_hotspot"]:
                unique_counts[key] = unique_counts.get(key, 0) + 1
    zeros = sum(t["all_zero"] for t in tables)
    stable_set = not zeros and all(s == winner_sets[0] for s in winner_sets)
    all_unique = all(t["unique_nonzero_hotspot"] for t in tables)
    if zeros == len(tables):
        classification = "ZERO_ONLY_NO_NONZERO_HOTSPOT"
    elif zeros:
        classification = "MIXED_ZERO_AND_NONZERO_HOTSPOTS"
    elif stable_set:
        classification = "STABLE_UNIQUE_HOTSPOT" if all_unique else "STABLE_TIED_HOTSPOTS"
    else:
        classification = "VARIABLE_HOTSPOTS"
    common = set(winner_sets[0]).intersection(*(set(s) for s in winner_sets[1:]))
    gaps = [rational(t["dominance_gap"]) for t in tables]
    return {
        "account_count": len(tables), "classification": classification,
        "stable_nonzero_winner_set": bool(stable_set),
        "stable_unique_nonzero_hotspot": bool(stable_set and all_unique),
        "all_accounts_unique_nonzero": all_unique,
        "unique_nonzero_accounts": sum(t["unique_nonzero_hotspot"] for t in tables),
        "tied_nonzero_accounts": sum(len(t["winner_ids"]) > 1 for t in tables),
        "zero_only_accounts": zeros,
        "zero_rows": sum(t["zero_row_count"] for t in tables),
        "common_nonzero_maximizer_ids": [identities[k] for k in sorted(common)],
        "winner_membership_counts": [
            {**identities[k], "count": counts[k]} for k in sorted(counts)],
        "unique_winner_counts": [
            {**identities[k], "count": unique_counts[k]} for k in sorted(unique_counts)],
        "minimum_dominance_gap": str(min(gaps)),
        "maximum_dominance_gap": str(max(gaps)),
    }


def partition(accounts):
    return {"account_count": len(accounts),
            "tables": {key: summarize([a["tables"][key] for a in accounts])
                       for key in TABLES}}


def report(retained):
    parent = retained["report"]
    same([c["control"] for c in parent["controls"]], list(CONTROLS), "control census changed")
    same(parent["component_order"], list(COMPONENTS), "component order changed")
    old_controls = retained["retained_50f"]["report"]["controls"]
    same([c["control"] for c in old_controls], list(CONTROLS), "50f control splice")
    accounts = []
    for control, original in zip(parent["controls"], old_controls, strict=True):
        same([(p["left_id"], p["right_id"]) for p in control["pairs"]],
             list(PAIRS), "ordered pair census changed")
        same(len(original["pairs"]), len(PAIRS), "50f pair census changed")
        for pair, old in zip(control["pairs"], original["pairs"], strict=True):
            same({k: pair[k] for k in ("left_id", "right_id", "roles")},
                 {k: old[k] for k in ("left_id", "right_id", "roles")}, "pair/role splice")
            same(sorted(pair["branches"]), list(BRANCHES), "branch census changed")
            for branch in BRANCHES:
                source = pair["branches"][branch]
                for key in ("selected_coordinates", "direct_hidden_accounting",
                            "unchanged_margin_accounting"):
                    same(source[key], old["branches"][branch][key], "50f retained splice: " + key)
                accounts.append({
                    "control": control["control"], "branch": branch,
                    **{k: pair[k] for k in ("left_id", "right_id", "roles")},
                    **classify_account(source, branch),
                })
    census = {
        "control_count": len(parent["controls"]), "diagnostic_pair_count": len(accounts) // 2,
        "pair_branch_count": len(accounts), "coordinate62_accounts": len(accounts),
        "selected_coordinate_accounts": sum(a["selected_coordinate_count"] for a in accounts),
        "weighted_component_count": sum(a["weighted_component_count"] for a in accounts),
    }
    same(list(census.values()), [9, 27, 54, 54, 1035, 7245], "exact consumption census changed")
    for key, value in census.items():
        same(parent[key], value, "retained census splice: " + key)
    global_partition = partition(accounts)
    partitions = {}
    for name, fields in (
        ("by_branch", ("branch",)), ("by_control", ("control",)),
        ("by_ordered_pair", ("left_id", "right_id")),
        ("by_control_branch", ("control", "branch")),
        ("by_pair_branch", ("left_id", "right_id", "branch")),
        ("by_control_pair", ("control", "left_id", "right_id")),
    ):
        groups = {}
        for account in accounts:
            key = tuple(account[f] for f in fields)
            groups.setdefault(key, []).append(account)
        partitions[name] = [
            {**dict(zip(fields, key, strict=True)), **partition(group)}
            for key, group in groups.items()]
    return {
        **census, "accounts": accounts, "global": global_partition,
        "partitions": partitions,
        "lineage_separation": parent["lineage_separation"],
        "reference_scope": parent["reference_scope"],
        "ranking_rule": parent["ranking_rule"],
        "gap_rule": (
            "Exact maximum minus second row magnitude, including ties; a singleton "
            "uses zero as runner-up. Distinct-level gap is separately nullable. "
            "Zero-only maximizers are disclosed but are not nonzero winners. "
            "Stability requires the same nonempty winner set in every account; "
            "stable uniqueness additionally requires singleton winners throughout."
        ),
        "exact_gap_and_tie_closure": True,
        "descriptive_accounting_only": True,
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
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
            compiled.append(record(path))
        retained = authenticate()
        result = report(retained)
        tables = result["global"]["tables"]
        same(tables["component"]["unique_winner_counts"],
             [{"component": "mlp_stage17", "count": 54}], "measured component winner changed")
        same(tables["coordinate"]["unique_winner_counts"],
             [{"coordinate": 62, "count": 20}, {"coordinate": 241, "count": 34}],
             "measured coordinate winners changed")
        same(tables["coordinate_component"]["unique_winner_counts"], [
            {"coordinate": 62, "component": "input_hidden", "count": 36},
            {"coordinate": 241, "component": "mlp_stage17", "count": 17},
            {"coordinate": 241, "component": COMPONENTS[-1], "count": 1},
        ], "measured coordinate-component winners changed")
    same(audit, {"forbidden_calls": 0}, "forbidden dispatch or write")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_HOTSPOT_STABILITY_CLASSIFIER",
        "command": COMMAND, "compiled_sources": compiled, "input_pins": PINS,
        "dispatch_and_write_audit": {
            **retained["dispatch_and_write_audit"], **audit,
            "tensor_decode_invocations": 0, "retained_producer_invocations": 0,
        },
        "flags": retained["flags"], "claim_boundary": BOUNDARY,
        "normal_host_review": "REQUIRED", "retained_449b": retained, "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
