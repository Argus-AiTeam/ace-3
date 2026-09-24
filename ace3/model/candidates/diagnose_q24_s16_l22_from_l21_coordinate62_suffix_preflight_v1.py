"""Read-only preparation of nine failed-parent L22/P0 counterfactual consumers."""

import argparse
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l21_sparse_cut_localization_matrix_v1 as matrix
from ace3.model.candidates import residual_exact_grid_q24_reference_v1 as rational


census = matrix.census
require = census.require
ROOT = census.ROOT
NAME = "q24_s16_l22_from_l21_coordinate62_suffix_preflight_v1"
ID = "ace3-q24-s16-l22-from-l21-coordinate62-suffix-preflight-v1"
SOURCE = ROOT / f"ace3/model/candidates/diagnose_{NAME}.py"
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
REFERENCE_MANIFEST = ROOT / (
    "build/q24_software_l9_l23_4d1cfdfc15c0_attempt002/references/freeze.json")
EXPECTED_TESTS = 28
PINS = {
    "localization_source": {
        "path": str(matrix.SOURCE),
        "sha256": "8b9823ca0453b793d26e0641636241eaebeb88e09b148baa82f658a0f76d0eb3",
    },
    "localization_review": {
        "path": str(matrix.HANDOFFS / "6c38406b5e07/round-0001.json"),
        "sha256": "9fdfbb799579f987d0c9648ed926b8f33fa483ac95fba427488a6ea66b66b8a1",
    },
    "census_source": matrix.PINS["census_source"],
    "census_review": matrix.PINS["census_review"],
    "state_oracle": {
        "path": str(ROOT / "ace3/model/candidates/residual_exact_grid_q24_reference_v1.py"),
        "sha256": "dbbe2ac57bf8f6dd0cdc6cece202eeb52f0643bfa07ed542dca0d8c4b58a81e5",
    },
}
FLAGS = {
    **matrix.FLAGS, "native_L22_P0_invocations": 0, "gpu_invocations": 0,
    "dispatch_guard_calls": 0, "evidence_writes": 0,
    "execution_authorized": False, "strict_FP16_state_claim": False,
    "new_token_claim": False, "full_model_claim": False,
}
STATE = {
    "shape": [896], "fields": {"i": "output_i", "z": "output_z", "h": "stage18"},
    "dtypes": {"i": "<i8", "z": "|u1", "h": "<u2"},
    "transformation": "none; entire raw L21 output I/Z/H, not a coordinate cut",
}
KV = {
    "kind": "own empty P0", "keys": ["k", "v"], "shape": [0, 128],
    "dtype": "<u2", "storage": "FP16 words",
    "shared_between_consumers": False, "prior_layer_kv_consumed": False,
}
FUTURE_OUTPUT = {
    "default": str(OUTPUT), "parent": str(ROOT / "build"),
    "basename_pattern": NAME + r"_attempt[0-9]{3,}",
    "fresh": True, "ignored": True, "direct_child": True,
    "create_during_check": False, "overwrite": False,
    "future_creation": "recheck freshness and gates, then exclusive mkdir; separate reviewed executor",
    "artifacts": ["result.json", "validation.json", "interventions.json"],
    "per_control_artifacts": [
        "{control}_L22.npz", "{control}_L22_parent.npz",
        "{control}_L22_local_references.npz", "{control}_L22_gates.json",
    ],
}
COMMAND = (
    "PYTHONPATH=/home/argustest/ace3-argus PYTHONDONTWRITEBYTECODE=1 "
    "/home/argustest/miniconda3/bin/python -B -m "
    "ace3.model.candidates.diagnose_q24_s16_l22_from_l21_coordinate62_suffix_preflight_v1 --check"
)
BOUNDARY = (
    "Preparation only; no L22 numerical result. Nine raw failed L21 parents "
    "are counterfactual inputs, never admission parents. The old sparse-cut "
    "closure cannot certify, substitute for or propagate these nine states. "
    "No L0-L21, accepted-prefix or original-prefix replay; no native execution, "
    "RTL, hardware, GPU or simulation; no evidence overwrite. Q24 residual "
    "state is wider than FP16; native G128 asymmetric packed INT4, GEMM nibble "
    "ordering, no qzero plus-one, FP16 scales/operator boundaries/KV and S16 "
    "RTZ remain unchanged. Thresholds and source/operand/state/KV/lineage "
    "gates remain mandatory. No strict-FP16-state W4A16, new-token or "
    "full-model admission claim. A future executor requires separate review "
    "and current source/identity/model/account/budget/access/concurrency preflights."
)
EXPECTED_CONTRACT = {
    "diagnostic_id": ID, "version": 1, "schema_version": 1, "cwd": str(ROOT),
    "evidence": PINS, "controls": list(census.CONTROLS), "parent_node": [21, 0, 18],
    "consumer_node": [22, 0], "history": [9707], "coordinate": 62,
    "state": STATE, "consumer_kv": KV, "future_output": FUTURE_OUTPUT,
    "reference_manifest": str(REFERENCE_MANIFEST),
    "reference_policy": census.REFERENCE_POLICY, "policy_id": census.POLICY,
    "profile_id": census.PROFILE, "excess_budget": "1/8",
    "local_threshold": census.EXPECTED_CONTRACT["local_threshold"],
    "reference_missing_status": "BLOCKED_MISSING_L22_REFERENCE",
    "interface": ["--check", "--out"], "output": "stdout JSON only; no file writes",
    "flags": FLAGS, "expected_tests": EXPECTED_TESTS, "command_sidecar": COMMAND,
    "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
}


def check_contract(document):
    require(json.dumps(document, sort_keys=True) ==
            json.dumps(EXPECTED_CONTRACT, sort_keys=True), "L22 preflight contract mismatch")


def output_path(value):
    path = Path(value)
    out = path if path.is_absolute() else ROOT / path
    require(out.resolve() == out and out.parent == ROOT / "build"
            and re.fullmatch(FUTURE_OUTPUT["basename_pattern"], out.name) is not None
            and int(out.name.rsplit("attempt", 1)[1]) > 0,
            "output must be a canonical versioned direct child of build")
    require(not out.exists() and not out.is_symlink(), "output already exists or is a symlink")
    result = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--", str(out)],
        capture_output=True, check=False)
    require(result.returncode == 0, "output is not confirmed ignored: " +
            result.stderr.decode(errors="replace"))
    return out


def check_state(arrays):
    for key, field in STATE["fields"].items():
        require(arrays[field].shape == (896,)
                and arrays[field].dtype.str == STATE["dtypes"][key],
                f"raw L21 state schema mismatch: {field}")
    for integer, tag, word in zip(arrays["output_i"], arrays["output_z"], arrays["stage18"]):
        require(rational.project(int(integer), int(tag)) == int(word),
                "raw L21 I/Z/H projection mismatch")


def empty_kv():
    return {key: np.empty((0, 128), dtype="<u2") for key in KV["keys"]}


def check_reference_manifest(extension, previous):
    require(extension["schema"] == "ace3-v3-original-reference-suffix-p0-v1"
            and extension["policy_id"] == census.POLICY
            and extension["binary64_profile"] == census.PROFILE
            and extension["global_reference_policy"] == census.REFERENCE_POLICY
            and extension["reference_only"] is True
            and extension["accepted_ancestor_oracle_replays"] == 0
            and extension["scope"] == {
                "history": [9707], "layers": list(range(9, 24)), "position": 0}
            and extension["layers"]["21"] == previous,
            "original-input reference lineage mismatch")
    current = extension["layers"].get("22")
    if current is None:
        return None
    require(current["input_binary64"] == previous["binary64"]
            and current["input_fp16"] == previous["fp16"]
            and current["prior_kv"] == "own empty P0", "L22 reference re-anchored or KV changed")
    for key, suffix in (("binary64", "binary64.npy"), ("fp16", "fp16.npz")):
        require(current[key]["path"] == str(REFERENCE_MANIFEST.parent / f"layer22_{suffix}"),
                "L22 reference archive substituted")
    return current


def bind_reference(bindings, previous):
    matches = [r for r in bindings if r["path"] == str(REFERENCE_MANIFEST)]
    require(len(matches) <= 1, "duplicate original reference manifest")
    if not matches:
        return {"status": "BLOCKED_MISSING_L22_REFERENCE", "reference": None,
                "manifest": None, "missing": [str(REFERENCE_MANIFEST)]}
    manifest = matches[0]
    try:
        extension = json.loads(census.read_bound(manifest))
    except FileNotFoundError:
        return {"status": "BLOCKED_MISSING_L22_REFERENCE", "reference": None,
                "manifest": manifest, "missing": [str(REFERENCE_MANIFEST)]}
    current = check_reference_manifest(extension, previous)
    if current is None:
        return {"status": "BLOCKED_MISSING_L22_REFERENCE", "reference": None,
                "manifest": manifest, "missing": [str(REFERENCE_MANIFEST) + "#layers/22"]}
    missing = []
    for key in ("binary64", "fp16", "input_binary64", "input_fp16"):
        try:
            census.read_bound(current[key])
        except FileNotFoundError:
            missing.append(current[key]["path"])
    return {
        "status": "BLOCKED_MISSING_L22_REFERENCE" if missing else "BOUND_ORIGINAL_INPUT_L22",
        "reference": current, "manifest": manifest, "missing": missing,
    }


def parent_plans(retained):
    controls = census.summarize(retained["rows"], retained["gates"])
    manifest = census.artifact_manifest(retained["result"]["artifacts"])
    plans, states = [], {}
    for row in controls:
        label = row["control"]
        record = manifest[str(census.BASE / f"{label}_L21.npz")]
        with np.load(io.BytesIO(census.read_bound(record)), allow_pickle=False) as archive:
            state = {field: archive[field] for field in STATE["fields"].values()}
        check_state(state)
        require(int(state["stage18"][62]) ==
                int(row["failures"][0]["actual_fp16_bits"], 16), "L21 state/gate splice")
        states[label] = state
        plans.append({
            "control": label, "parent_node": [21, 0, 18], "consumer_node": [22, 0],
            "parent_archive": record, "parent_state": STATE, "consumer_kv": KV,
            "retained_L21": row, "L22_status": "NOT_EXECUTED",
            "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        })
    return plans, states


def authenticate(out=OUTPUT):
    require(Path(matrix.__file__).resolve() == matrix.SOURCE
            and Path(census.__file__).resolve() == census.SOURCE
            and Path(rational.__file__).resolve() == Path(PINS["state_oracle"]["path"]),
            "read-only helper origin mismatch")
    check_contract(json.loads(CONTRACT.read_bytes()))
    out = output_path(out)
    for pin in PINS.values():
        census.read_bound(pin)
    matrix.check_review(json.loads(census.read_bound(PINS["localization_review"])),
                        "6c38406b5e07", 1)
    localized = matrix.authenticate()
    retained = localized["census"]
    plans, states = parent_plans(retained)
    reference = bind_reference(retained["result"]["input_bindings"],
                               retained["summary"]["original_reference"])
    summary = {
        "diagnostic_id": ID, "schema_version": 1,
        "status": ("PREFLIGHT_READY" if not reference["missing"]
                   else "BLOCKED_MISSING_L22_REFERENCE"),
        "L22_status": "NOT_EXECUTED", "control_count": 9, "controls": plans,
        "future_output": {**FUTURE_OUTPUT, "selected": str(out)},
        "L22_original_reference": reference, "evidence": PINS,
        "contract": matrix.record(CONTRACT), "policy_id": census.POLICY,
        "profile_id": census.PROFILE, "thresholds": retained["summary"]["thresholds"],
        "original_global_reference_unchanged": True,
        "historical_failures_preserved": True,
        "retained_L20_failing_controls": retained["result"]["retained_L20_failing_controls"],
        "certified_census": retained["summary"],
        "localization_comparison": localized["summary"],
        "old_sparse_cut_used_as_parent": False,
        "gate_evidence_scope": "AUTHENTICATED_RETAINED_ONLY; fresh raw parent I/Z/H check",
        "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED", **FLAGS,
    }
    check_summary(summary)
    return {"summary": summary, "states": states, "retained": retained}


def check_summary(summary):
    require(summary["diagnostic_id"] == ID and type(summary["schema_version"]) is int
            and summary["schema_version"] == 1
            and summary["contract"] == matrix.record(CONTRACT), "result schema/contract mismatch")
    for key, value in FLAGS.items():
        require(type(summary[key]) is type(value) and summary[key] == value,
                f"check boundary changed: {key}")
    require(summary["L22_status"] == "NOT_EXECUTED"
            and summary["control_count"] == len(summary["controls"]) == 9
            and [row["control"] for row in summary["controls"]] == list(census.CONTROLS)
            and summary["old_sparse_cut_used_as_parent"] is False,
            "consumer schedule or parent lineage changed")
    for row in summary["controls"]:
        require(row["parent_archive"]["path"] ==
                str(census.BASE / f"{row['control']}_L21.npz")
                and row["parent_state"] == STATE and row["consumer_kv"] == KV
                and row["parent_node"] == [21, 0, 18] and row["consumer_node"] == [22, 0]
                and row["retained_L21"]["control"] == row["control"]
                and row["retained_L21"]["S18"] == "FAIL"
                and row["retained_L21"]["failing_coordinates"] == [62]
                and row["retained_L21"]["failures"][0]["threshold_margin"] == "-3/4"
                and row["L22_status"] == "NOT_EXECUTED", "raw failed parent plan changed")
        matrix.check_false_flags(row)
    reference = summary["L22_original_reference"]
    require((summary["status"] == "PREFLIGHT_READY"
             and reference["status"] == "BOUND_ORIGINAL_INPUT_L22"
             and reference["reference"] is not None and reference["missing"] == [])
            or (summary["status"] == reference["status"] == "BLOCKED_MISSING_L22_REFERENCE"
                and bool(reference["missing"])), "reference readiness mismatch")


def validate(out=OUTPUT):
    require(Path(__file__).resolve() == SOURCE and Path.cwd().resolve() == ROOT
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "repository-bound unoptimized -B command required")
    compiled = []
    for path in (SOURCE, TEST):
        data = path.read_bytes()
        compile(data, str(path), "exec", dont_inherit=True)
        compiled.append(matrix.record(path))
    evidence = authenticate(out)
    spec = importlib.util.spec_from_file_location(f"test_{NAME}", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader missing")
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    tests.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    require(suite.countTestCases() == EXPECTED_TESTS, "focused test count mismatch")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.testsRun == EXPECTED_TESTS and result.wasSuccessful() and not result.skipped,
            "focused tests failed, errored or skipped")
    return {
        **evidence["summary"], "command": COMMAND, "compiled": compiled,
        "collected": suite.countTestCases(), "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--out", default=str(OUTPUT),
                        help="prospective output only; never created by --check")
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.out), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
