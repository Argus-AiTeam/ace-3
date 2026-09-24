"""Stdout-only exact hotspot accounting from the reviewed 50f retained JSON."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_component_hotspot_audit_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT_NAME = "diagnose_q24_s16_final_head_final_rmsnorm_direct_hidden_residual_margin_bridge_v1"
CAPTURE_ROOT = ROOT / "build/direct-hidden-residual-50f5bb2062c9-bc457u29"
PINS = {
    "stdout": {
        "path": str(CAPTURE_ROOT / "stdout.json"), "bytes": 6737063,
        "sha256": "30790957680088caecfa2539e10e672f0d5ff373573d4c80a71cb3be2d65cdeb",
    },
    "capture": {
        "path": str(CAPTURE_ROOT / "capture.json"), "bytes": 7652,
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
BOUNDARY = (
    "Selected-coordinate/component retained accounting only, not a successor, "
    "intervention, admission, causal allocation or performance attribution. "
    "Rankings are separate for every control/ordered pair/reference branch; "
    "they are not pooled independent observations. Coordinate 62 is included "
    "once in each selected union, never an extra contribution. All zeros and "
    "maximum ties are retained. Binary64 internal stages remain "
    "NOT_RETAINED_NO_RECONSTRUCTION; FP16 terminal minus independently propagated "
    "original-input binary64 terminal is a separate remainder. Original-input "
    "global references, exact thresholds, source/operand/state/KV/lineage gates "
    "and every historical failure are preserved in retained_50f. Original "
    "source/input pins are authenticated transitively by the exact reviewed "
    "stdout; only its capture files and producer source/test are reopened, "
    "not tensor inputs or historical producers. Q24 residual state is wider "
    "than FP16. Native S16 RTZ, official G128 asymmetric packed INT4 GEMM nibble "
    "ordering without qzero plus-one, FP16 scales/operator boundaries/KV are "
    "unchanged. No prefix/admission/reference/native decoder/RMSNorm/head/"
    "selected-row dot/480/50f/f0/row319 producer replay, new-token or full-model "
    "admission, strict-FP16-state W4A16, hardware/GPU/RTL/FPGA or ACE2 change. "
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
        str(SOURCE), str(TEST), *(pin["path"] for pin in PINS.values()),
        str(ROOT / "ace3/model/candidates" / (PARENT_NAME + ".py")),
        str(ROOT / "tests" / ("test_" + PARENT_NAME + ".py")),
        *(str(CAPTURE_ROOT / name) for name in (
            "command.txt", "preflight.json", "runner-command.txt",
            "stderr.log", "whole-command.log")),
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
                or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            )
        elif event == "import":
            name = args[0]
            forbidden = name.startswith(("ace3.", "numpy", "torch", "safetensors", "ctypes"))
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
            raise RuntimeError("retained-only audit forbids " + event)

    sys.addaudithook(guard)
    try:
        yield
    finally:
        active = False


def authenticate():
    review = decode(bound_bytes(PINS["review"]))
    same((review["mission_id"], review["producer_role"], review["review"]["status"]),
         ("50f5bb2062c9", "reviewer", "done"), "independent reviewed input required")
    capture = decode(bound_bytes(PINS["capture"]))
    require(all(value is True for value in capture["checks"].values()),
            "retained capture has a failed check")
    for name, pin in capture["files"].items():
        require(name in {
            "command.txt", "preflight.json", "runner-command.txt", "stderr.log",
            "stdout.json", "whole-command.log"}, "unexpected capture member")
        if name != "stdout.json":
            bound_bytes({"path": str(CAPTURE_ROOT / name),
                         "bytes": pin["size_bytes"], "sha256": pin["sha256"]})
    same(capture["files"]["stdout.json"],
         {"size_bytes": PINS["stdout"]["bytes"], "sha256": PINS["stdout"]["sha256"]},
         "capture/stdout splice")
    for path, pin in capture["source_test_pins_after"].items():
        bound_bytes({"path": path, "bytes": pin["size_bytes"], "sha256": pin["sha256"]})
    retained = decode(bound_bytes(PINS["stdout"]))
    same(retained["diagnostic_id"], PARENT_NAME, "wrong retained producer")
    same(retained["status"], "READ_ONLY_FINAL_RMSNORM_DIRECT_HIDDEN_RESIDUAL_MARGIN_BRIDGE",
         "wrong retained result boundary")
    require(all(value is False or type(value) is int and value == 0
                for value in retained["dispatch_and_write_audit"].values()),
            "retained forbidden work or claim")
    same((retained["tests"]["executed"], retained["tests"]["failures"],
          retained["tests"]["errors"], retained["tests"]["skipped"]),
         (22, 0, 0, 0), "retained focused tests changed")
    return retained


def mass(values):
    values = list(values)
    signed = sum(values, Fraction())
    absolute = sum(map(abs, values), Fraction())
    return {"signed": str(signed), "absolute": str(absolute),
            "cancellation_absolute_mass": str(absolute - abs(signed))}


def signed_masses(values):
    values = list(values)
    return {
        **mass(values),
        "positive_mass": str(sum((v for v in values if v > 0), Fraction())),
        "negative_absolute_mass": str(-sum((v for v in values if v < 0), Fraction())),
        "zero_count": sum(v == 0 for v in values),
    }


def ranked(entries):
    ordered = sorted(entries, key=lambda row: (
        -Fraction(row["absolute"]), row.get("coordinate", -1),
        COMPONENTS.index(row["component"]) if "component" in row else -1))
    previous, rank = None, 0
    result = []
    for position, row in enumerate(ordered, 1):
        value = Fraction(row["absolute"])
        if value != previous:
            rank = position
        result.append({**row, "rank": rank})
        previous = value
    return result


def hotspot(entries):
    ordered = ranked(entries)
    nonzero = any(Fraction(row["absolute"]) != 0 for row in ordered)
    return {
        "ranking": ordered, "nonzero_hotspot": nonzero,
        "maximum_ties": [row for row in ordered if row["rank"] == 1] if nonzero else [],
    }


def branch_account(old, upstream, branch):
    rows = old["selected_coordinates"]
    indices = [row["coordinate"] for row in rows]
    require(all(type(i) is int and 0 <= i < 896 for i in indices)
            and indices == sorted(set(indices)) and 62 in indices
            and 8 <= len(indices) <= 25, "selected coordinate census changed")
    same(indices, [row["coordinate"] for row in upstream["selected_coordinates"]],
         "retained selected union splice")
    same(old["hidden_reference"], "original_input_L23_" + branch, "reference branch splice")
    same(old["residual_internal_reference"], "original_input_L23_fp16",
         "internal reference splice")
    same(old["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION",
         "binary64 stage reconstruction")
    component_values = {key: [] for key in COMPONENTS}
    cells, coordinates, direct_values = [], [], []
    for row, parent_row in zip(rows, upstream["selected_coordinates"], strict=True):
        same(set(row["weighted_components"]), set(COMPONENTS), "component census changed")
        same(set(row["hidden_components"]), set(COMPONENTS), "hidden component census changed")
        factor = Fraction(row["weight_times_reference_anchor_times_row_difference"])
        require(Fraction(row["reference_inverse_norm_anchor"]) > 0, "invalid retained anchor")
        values = []
        for key in COMPONENTS:
            value = Fraction(row["weighted_components"][key])
            same(value, factor * Fraction(row["hidden_components"][key]),
                 "retained weighted component splice")
            if branch == "fp16" and key == COMPONENTS[-1]:
                same(Fraction(row["hidden_components"][key]), Fraction(),
                     "FP16 terminal remainder must be zero")
            values.append(value)
            component_values[key].append(value)
            cells.append({"coordinate": row["coordinate"], "component": key,
                          "signed": str(value), "absolute": str(abs(value))})
        direct = Fraction(row["direct_hidden_weighted_term"])
        same(sum(values, Fraction()), direct, "coordinate direct-hidden closure")
        same(direct, Fraction(parent_row["weighted_terms"]["direct_hidden"]),
             "upstream direct-hidden splice")
        same(mass(values), row["weighted_component_mass"], "coordinate mass splice")
        same(mass(Fraction(v) for v in row["hidden_components"].values()),
             row["hidden_component_mass"], "hidden mass splice")
        for field, component in (
            ("weighted_actual_residual_boundary_parts", "actual_residual_boundary"),
            ("weighted_negative_reference_residual_boundary_parts",
             "negative_reference_residual_boundary"),
        ):
            same(sum((Fraction(v) for v in row[field].values()), Fraction()),
                 Fraction(row["weighted_components"][component]), "boundary subpart splice")
        direct_values.append(direct)
        coordinates.append({
            "coordinate": row["coordinate"], "signed": str(direct),
            "absolute": str(abs(direct)), "component_mass": signed_masses(values),
        })
    totals = {key: mass(values) for key, values in component_values.items()}
    direct_mass = mass(direct_values)
    absolute = sum((Fraction(v["absolute"]) for v in totals.values()), Fraction())
    closure = {
        "component_totals": totals, "selected_direct_hidden": direct_mass,
        "component_absolute_sum": str(absolute),
        "within_coordinate_cancellation_mass": str(absolute - Fraction(direct_mass["absolute"])),
        "across_coordinate_cancellation_mass": direct_mass["cancellation_absolute_mass"],
        "total_component_cancellation_mass": str(absolute - abs(Fraction(direct_mass["signed"]))),
        "exact_selected_identity": True,
    }
    same(closure, old["direct_hidden_accounting"], "selected component closure changed")
    margin = old["unchanged_margin_accounting"]
    same(margin, upstream["accounting"], "unchanged final-margin accounting splice")
    same({key: direct_mass[key] for key in ("signed", "absolute")},
         margin["selected_term_totals"]["direct_hidden"], "margin direct-hidden closure")
    same(sum((Fraction(v["signed"]) for v in margin["selected_term_totals"].values()), Fraction())
         + Fraction(margin["unselected_coordinate_signed_remainder"])
         + Fraction(margin["head_boundary_remainder_change"]),
         Fraction(margin["retained_margin_change"]), "final-margin closure changed")
    coordinate62 = next(row for row in coordinates if row["coordinate"] == 62)
    others = [row for row in coordinates if row["coordinate"] != 62]
    component_rankings = {}
    for key, values in component_values.items():
        entries = [row for row in cells if row["component"] == key]
        component_rankings[key] = {
            **hotspot(entries), "mass": signed_masses(values),
            "coordinate62": next(row for row in entries if row["coordinate"] == 62),
            "other_coordinates_mass": signed_masses(
                Fraction(row["signed"]) for row in entries if row["coordinate"] != 62),
        }
    return {
        "hidden_reference": old["hidden_reference"],
        "residual_internal_reference": old["residual_internal_reference"],
        "binary64_internal_stages": old["binary64_internal_stages"],
        "selected_coordinates": rows,
        "coordinate_hotspots": hotspot(coordinates),
        "coordinate_component_hotspots": hotspot(cells),
        "component_hotspots_by_coordinate_absolute_mass": hotspot([
            {"component": key, **signed_masses(values)}
            for key, values in component_values.items()]),
        "per_component_coordinate_hotspots": component_rankings,
        "coordinate62": {
            "included_once": True, "selected": coordinate62,
            "other_coordinates_mass": signed_masses(Fraction(row["signed"]) for row in others),
        },
        "direct_hidden_accounting": closure,
        "unchanged_margin_accounting": margin,
        "exact_final_margin_identity": True,
    }


def report(retained):
    parent = retained["report"]
    upstream = parent["retained_final_rmsnorm_logit_margin_bridge"]
    same([c["control"] for c in parent["controls"]], list(CONTROLS), "control census changed")
    same([c["control"] for c in upstream["controls"]], list(CONTROLS), "upstream control splice")
    controls, count = [], 0
    for control, original in zip(parent["controls"], upstream["controls"], strict=True):
        same([(p["left_id"], p["right_id"]) for p in control["pairs"]],
             list(PAIRS), "ordered pair census changed")
        same(len(original["pairs"]), len(PAIRS), "upstream pair census changed")
        pairs = []
        for pair, prior in zip(control["pairs"], original["pairs"], strict=True):
            for key in ("left_id", "right_id", "roles"):
                same(pair[key], prior[key], "pair identity splice: " + key)
            same(sorted(pair["branches"]), list(BRANCHES), "branch census changed")
            same(sorted(prior["branches"]), list(BRANCHES), "upstream branch splice")
            branches = {branch: branch_account(pair["branches"][branch],
                                              prior["branches"][branch], branch)
                        for branch in BRANCHES}
            count += sum(len(b["selected_coordinates"]) for b in branches.values())
            pairs.append({**{k: pair[k] for k in ("left_id", "right_id", "roles")},
                          "branches": branches})
        controls.append({"control": control["control"], "pairs": pairs})
    same(count, 1035, "selected account count changed")
    census = {"control_count": 9, "diagnostic_pair_count": 27, "pair_branch_count": 54,
              "coordinate62_accounts": 54, "selected_coordinate_accounts": count,
              "weighted_component_count": count * len(COMPONENTS)}
    for key, value in census.items():
        same(parent[key], value, "retained census splice: " + key)
    return {
        **census, "controls": controls, "lineage_separation": parent["lineage_separation"],
        "reference_scope": parent["reference_scope"],
        "ranking_rule": (
            "Descending exact absolute magnitude; equal magnitudes share competition "
            "rank, then ascending coordinate and declared component order. Component "
            "rank uses sum of coordinate absolutes, not absolute net signed total. "
            "Zero-only tables retain every row but have no nonzero hotspot or maximum ties."
        ),
        "component_order": list(COMPONENTS),
        "cancellation_convention": "absolute_mass - abs(signed_mass), twice the cancelled one-sided mass",
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
    same(audit, {"forbidden_calls": 0}, "forbidden dispatch or write")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_COMPONENT_HOTSPOT_AUDIT",
        "command": COMMAND, "compiled_sources": compiled, "input_pins": PINS,
        "dispatch_and_write_audit": {**retained["dispatch_and_write_audit"], **audit},
        "flags": retained["flags"], "claim_boundary": BOUNDARY,
        "normal_host_review": "REQUIRED", "retained_50f": retained, "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
