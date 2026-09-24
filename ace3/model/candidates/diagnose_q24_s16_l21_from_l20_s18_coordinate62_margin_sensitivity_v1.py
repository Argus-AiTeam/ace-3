"""Retained-only, nine-control L21 coordinate-62 Q24/RNE sensitivity."""

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
import unittest

import numpy as np

from ace3.model.candidates import binary64_fp16_excess_v1 as gate
from ace3.model.candidates import diagnose_q24_s16_l21_from_l20_s18_failure_census_v1 as census
from ace3.model.candidates import residual_exact_grid_q24_reference_v1 as rational


ROOT = Path("/home/argustest/ace3-argus")
PYTHON = Path("/home/argustest/miniconda3/bin/python")
NAME = "q24_s16_l21_from_l20_s18_coordinate62_margin_sensitivity_v1"
ID = "ace3-q24-s16-l21-from-l20-s18-coordinate62-margin-sensitivity-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
SOURCE = ROOT / "ace3/model/candidates" / f"diagnose_{NAME}.py"
TEST = ROOT / "tests" / f"test_{NAME}.py"
CONTRACT = ROOT / "ace3/contracts/candidates" / f"{NAME}.json"
Q = 1 << 24
EXPECTED_TESTS = 24
CONTROLS = census.CONTROLS
COMMAND = (
    f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} "
    f"--out build/{NAME}_attempt001"
)
PINS = {
    "census_source": {
        "path": str(census.SOURCE),
        "sha256": "a8faef5d1deb15803357bc952b0bc4c177b23e044a6a30299d5f29d79599c2f7",
    },
    "census_contract": {
        "path": str(census.CONTRACT),
        "sha256": "21e7d24a96065c1f01efa750e8f368e22e0d9fa5430ed65c7374c44467300138",
    },
    "census_test": {
        "path": str(census.TEST),
        "sha256": "1e2fb7d4f6ac3cfb4bfdb88408468056cca3d9989c2bc5e38b7e3e184421c34f",
    },
    "census_review": {
        "path": "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/ecf478e49bef/round-0001.json",
        "sha256": "4ccd955f1460ad654f3ce190ec4e9ac28605923f8591e6dfa01ec2ad7dd7defc",
    },
    "q24_oracle": {
        "path": str(ROOT / "ace3/model/candidates/residual_exact_grid_q24_reference_v1.py"),
        "sha256": "dbbe2ac57bf8f6dd0cdc6cece202eeb52f0643bfa07ed542dca0d8c4b58a81e5",
    },
}
FLAGS = {
    "native_layer_invocations": 0, "decoder_invocations": 0,
    "prefix_invocations": 0, "admission_invocations": 0,
    "reference_invocations": 0, "rtl_invocations": 0,
    "gpu_invocations": 0, "hardware_invocations": 0,
    "simulation_invocations": 0, "suffix_invocations": 0,
    "reference_recomputation": False, "reference_reanchoring": False,
    "original_prefix_replay": False, "accepted_prefix_replay": False,
    "candidate_admitted": False, "policy_adopted": False,
    "successor_published": False, "state_archive_published": False,
}
BOUNDARY = (
    "Narrow L21 coordinate-62 scalar boundary sensitivity only, not a root cause "
    "or native repair. Q24 residual state is wider than FP16. Native G128 "
    "asymmetric packed INT4, GEMM nibble ordering, no qzero plus-one, FP16 "
    "scales/operator boundaries/KV and S16 RTZ remain unchanged. No strict-FP16-state "
    "W4A16, new-token or full-model admission. No producer, prefix, admission, "
    "reference or successor execution. Independent Host Reviewer reproduction required."
)
AUTHENTICATION = (
    "Pin the reviewed census source/contract/tests and terminal independent review; "
    "reuse only its read-only authenticator for its terminal L21 result/validation/review, "
    "37 artifacts, source/history, nine controls and original L20/L21 references. "
    "The census is stdout-only and has no standalone result.json/validation.json; "
    "do not replay its command or reconstruct a lost stdout capture. The 2593 "
    "ancestral binding records remain pinned, not exhaustively reread or replayed. "
    "Source/operand/state/KV/RTZ gates are authenticated retained facts."
)
EXPECTED_CONTRACT = {
    "diagnostic_id": ID, "version": 1, "cwd": str(ROOT),
    "controls": list(CONTROLS), "node": [21, 0, 18],
    "coordinate": 62, "coordinates_per_control": 896,
    "policy_id": census.POLICY, "profile_id": census.PROFILE,
    "reference_policy": census.REFERENCE_POLICY, "excess_budget": "1/8",
    "local_threshold": census.EXPECTED_CONTRACT["local_threshold"],
    "evidence": PINS, "census_terminal_inputs": census.ANCHORS,
    "interface": ["--out"], "command": COMMAND, "expected_tests": EXPECTED_TESTS,
    "output_scope": f"build/{NAME}_attempt* (absent direct child only)",
    "flags": FLAGS, "authentication_scope": AUTHENTICATION,
    "classification": {
        "SUPPORTED": "All nine minimal cuts pass, one-unit-less fails, only coordinate 62 changes, and common reference/cell/target/margin signatures agree.",
        "REJECTED": "Authenticated arithmetic, minimality, full-vector or common-mechanism checks fail; stop at the first inconsistency.",
        "UNKNOWN": "A required identity, retained binding or schema is unavailable/mismatched; no scalar diagnosis is admitted.",
    },
    "control_agreement": "Different retained Q24 integers and cut sizes are allowed; reference, baseline word/excess, nearest word, passing cells and adjusted integer must agree.",
    "termination": "One diagnostic run then normal independent Reviewer; no successor.",
    "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
}
require = census.require


def record(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def check_contract(document):
    require(document == EXPECTED_CONTRACT, "sensitivity contract mismatch")


def check_review(document):
    require(document["kind"] == "round_reviewed_handoff"
            and document["producer_role"] == "reviewer"
            and document["mission_id"] == "ecf478e49bef"
            and type(document["round"]) is int and document["round"] == 1
            and document["review"]["status"] == "done",
            "census review is not independent terminal DONE")


def cell(word):
    require(type(word) is int and 1 <= word < 0x7BFF,
            "positive interior finite FP16 cell required")
    value = rational.fp16_value(word)
    lower = (rational.fp16_value(word - 1) + value) / 2
    upper = (value + rational.fp16_value(word + 1)) / 2
    minimum = math.ceil(lower * Q)
    maximum = math.floor(upper * Q)
    even = word & 1 == 0
    if not even:
        minimum += Fraction(minimum, Q) == lower
        maximum -= Fraction(maximum, Q) == upper
    require(rational.project(minimum, 0) == rational.project(maximum, 0) == word,
            "RNE cell endpoint projection mismatch")
    return {
        "word": f"{word:04x}", "value": str(value),
        "lower": str(lower), "upper": str(upper),
        "lower_inclusive": even, "upper_inclusive": even,
        "lower_tie_owner": f"{word if even else word - 1:04x}",
        "upper_tie_owner": f"{word if even else word + 1:04x}",
        "minimum_Q24_integer": minimum, "maximum_Q24_integer": maximum,
    }


def scalar(word, reference):
    row = gate.evaluate_layer_final_output(
        actual_fp16_bits=word, reference_binary64_hex=reference)
    row["threshold_margin"] = str(Fraction(1, 8) - Fraction(row["excess_error"]))
    return row


def sensitivity(integer, negzero, reference):
    rational.validate_state(integer, negzero)
    baseline_word = rational.project(integer, negzero)
    baseline = scalar(baseline_word, reference)
    exact = Fraction.from_float(float.fromhex(reference))
    nearest = int(baseline["nearest_fp16_bits"], 16)
    require(0 < exact < 65504 and 1 < nearest < 0x7BFE
            and baseline["accepted"] is False, "failing positive interior scalar required")
    first = last = nearest
    while first > 1 and scalar(first - 1, reference)["accepted"]:
        first -= 1
    while last < 0x7BFE and scalar(last + 1, reference)["accepted"]:
        last += 1
    lower, upper = cell(first), cell(last)
    radius = Fraction(baseline["q"]) + Fraction(1, 8)
    require(rational.fp16_value(first - 1) < exact - radius
            <= rational.fp16_value(first) <= rational.fp16_value(last)
            <= exact + radius < rational.fp16_value(last + 1),
            "passing interval is not exhaustive")
    minimum, maximum = lower["minimum_Q24_integer"], upper["maximum_Q24_integer"]
    require(integer < minimum or integer > maximum, "baseline inside passing Q24 interval")
    adjusted = minimum if integer < minimum else maximum
    delta = adjusted - integer
    one_less = adjusted - (1 if delta > 0 else -1)
    passing = scalar(rational.project(adjusted, 0), reference)
    negative = scalar(rational.project(one_less, 0), reference)
    require(passing["accepted"] is True and negative["accepted"] is False,
            "minimal cut/one-unit-less contradiction")
    return {
        "reference_binary64_hex": reference, "reference_exact": str(exact),
        "baseline_Q24_integer": integer, "baseline_signed_zero": negzero,
        "baseline_Q24_value": str(Fraction(integer, Q)), "baseline": baseline,
        "actual_RNE_cell": cell(baseline_word), "nearest_finite_RNE_cell": cell(nearest),
        "passing_interval": {
            "gate_lower": str(exact - radius), "gate_upper": str(exact + radius),
            "first_cell": lower, "last_cell": upper,
            "minimum_Q24_integer": minimum, "maximum_Q24_integer": maximum,
        },
        "minimal_cut": {
            "adjusted_Q24_integer": adjusted, "adjusted_signed_zero": 0,
            "delta_Q24_units": delta, "delta_exact": str(Fraction(delta, Q)),
            "gate": passing, "one_unit_less_Q24_integer": one_less,
            "one_unit_less_delta_Q24_units": one_less - integer,
            "one_unit_less_gate": negative,
        },
    }


def check_state(arrays):
    for key, dtype in (("output_i", "<i8"), ("output_z", "u1"), ("stage18", "<u2")):
        require(arrays[key].shape == (896,) and arrays[key].dtype == np.dtype(dtype),
                f"invalid retained state shape/dtype: {key}")
    projected = np.array([
        rational.project(int(integer), int(zero))
        for integer, zero in zip(arrays["output_i"], arrays["output_z"])
    ], dtype="<u2")
    require(np.array_equal(projected, arrays["stage18"]),
            "Q24 state/FP16 projection mismatch")


def vector_gate(words, references):
    require(words.shape == references.shape == (896,)
            and words.dtype == np.dtype("<u2") and references.dtype == np.dtype("<f8")
            and np.isfinite(references).all(), "invalid full-vector inputs")
    rows = [scalar(int(word), float(reference).hex())
            for word, reference in zip(words, references)]
    failures = [i for i, row in enumerate(rows) if not row["accepted"]]
    return rows, {
        "status": "FAIL" if failures else "PASS", "coordinates": len(rows),
        "failure_count": len(failures), "failure_indices": failures,
        "minimum_threshold_margin": str(min(Fraction(r["threshold_margin"]) for r in rows)),
        "fp16_payload_sha256": hashlib.sha256(words.tobytes()).hexdigest(),
        "reference_payload_sha256": hashlib.sha256(references.tobytes()).hexdigest(),
    }


def analyze_control(label, arrays, references, reports):
    check_state(arrays)
    words = arrays["stage18"]
    rows, baseline_vector = vector_gate(words, references)
    require(baseline_vector["failure_indices"] == [62], "non-62 or missing baseline failure")
    require(len(reports) == 896, "incomplete retained S18 vector")
    for index, (row, stored) in enumerate(zip(rows, reports)):
        require(stored["index"] == index
                and all(stored[key] == value for key, value in row.items()
                        if key != "threshold_margin"),
                f"retained metric/reference mismatch at coordinate {index}")
    analysis = sensitivity(int(arrays["output_i"][62]), int(arrays["output_z"][62]),
                           float(references[62]).hex())
    vectors = {}
    cut = analysis["minimal_cut"]
    for name, integer, expected in (
        ("minimum", cut["adjusted_Q24_integer"], []),
        ("one_unit_less", cut["one_unit_less_Q24_integer"], [62]),
    ):
        projected = words.copy()
        projected[62] = rational.project(integer, 0)
        changed = np.flatnonzero(projected != words).tolist()
        require(changed == [62] if name == "minimum" else set(changed) <= {62},
                "counterfactual changed coordinates other than 62")
        new_rows, summary = vector_gate(projected, references)
        require(summary["failure_indices"] == expected, f"{name} full-vector gate contradiction")
        require(all(new_rows[i] == rows[i] for i in range(896) if i != 62),
                "counterfactual altered a non-62 metric or reference")
        vectors[name] = {**summary, "changed_coordinates": changed}
    return {
        "control": label, "status": "ANALYZED", "node": [21, 0, 18],
        "retained_S0_S17": "PASS", "retained_source_operand_state_KV_RTZ": "PASS",
        "gate_evidence_scope": "AUTHENTICATED_RETAINED_ONLY",
        "analysis": analysis, "baseline_vector": baseline_vector,
        "counterfactual_vectors": vectors, **FLAGS,
    }


def check_agreement(accounts):
    require([a["control"] for a in accounts] == list(CONTROLS)
            and all(a["status"] == "ANALYZED" for a in accounts),
            "incomplete nine-control analysis")
    def signature(account):
        analysis = account["analysis"]
        return {key: analysis[key] for key in (
            "reference_binary64_hex", "baseline", "nearest_finite_RNE_cell",
            "passing_interval")} | {
                "target": analysis["minimal_cut"]["adjusted_Q24_integer"]}
    require(all(signature(a) == signature(accounts[0]) for a in accounts),
            "controls disagree on common scalar mechanism")


def authenticate(records):
    check_contract(json.loads(CONTRACT.read_bytes()))
    for name, pin in PINS.items():
        data = census.read_bound(pin)
        records.append({**pin, "bytes": len(data)})
        if name == "census_review":
            check_review(json.loads(data))
    retained = census.authenticate()
    records.extend(retained["summary"]["authenticated_files"])
    manifest = census.artifact_manifest(retained["result"]["artifacts"])
    reference_record = retained["summary"]["original_reference"]["binary64"]
    references = np.load(io.BytesIO(census.read_bound(reference_record)), allow_pickle=False)
    states = {}
    for label in CONTROLS:
        pin = manifest[str(census.BASE / f"{label}_L21.npz")]
        with np.load(io.BytesIO(census.read_bound(pin)), allow_pickle=False) as archive:
            states[label] = {key: archive[key] for key in ("output_i", "output_z", "stage18")}
            for value in states[label].values():
                value.flags.writeable = False
    references.flags.writeable = False
    return retained, states, references


def diagnose():
    records = []
    result = {
        "diagnostic_id": ID, "status": "DIAGNOSED", "classification": "UNKNOWN",
        "control_count": 9, "analyzed_control_count": 0,
        "controls": [{"control": label, "status": "NOT_ANALYZED"} for label in CONTROLS],
        "authenticated_files": records, "authentication_scope": AUTHENTICATION,
        "original_global_reference_unchanged": None, "historical_failures_preserved": True,
        "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED", **FLAGS,
    }
    start = time.monotonic()
    try:
        retained, states, references = authenticate(records)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["error"] = {"phase": "authentication", "type": type(error).__name__,
                           "message": str(error)}
        result["seconds"] = {"authentication": time.monotonic() - start}
        return result
    result.update({
        "original_global_reference_unchanged": True,
        "original_reference": retained["summary"]["original_reference"],
        "policy_id": census.POLICY, "profile_id": census.PROFILE,
        "thresholds": retained["summary"]["thresholds"],
        "census_terminal_inputs": census.ANCHORS,
        "retained_L20_failing_controls": retained["summary"]["retained_L20_failing_controls"],
        "seconds": {"authentication": time.monotonic() - start},
    })
    start = time.monotonic()
    label = None
    manifest = census.artifact_manifest(retained["result"]["artifacts"])
    try:
        for index, label in enumerate(CONTROLS):
            result["controls"][index] = analyze_control(
                label, states[label], references,
                retained["gates"][label][18]["binary64_v1"]["rows"])
            result["controls"][index]["state_evidence"] = manifest[
                str(census.BASE / f"{label}_L21.npz")]
            result["controls"][index]["gates_evidence"] = manifest[
                str(census.BASE / f"{label}_L21_gates.json")]
            result["analyzed_control_count"] += 1
        label = None
        check_agreement(result["controls"])
    except (ValueError, KeyError, TypeError, gate.EvaluatorDefect) as error:
        result["classification"] = "REJECTED"
        result["error"] = {"phase": "arithmetic_localization", "control": label,
                           "type": type(error).__name__, "message": str(error)}
    else:
        result["classification"] = "SUPPORTED"
    result["seconds"]["scalar_and_vector_analysis"] = time.monotonic() - start
    return result


def output_path(text):
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    prefix = NAME + "_attempt"
    require(path == path.resolve() and path.parent == ROOT / "build"
            and path.name.startswith(prefix) and len(path.name) > len(prefix),
            "output must be an absent versioned direct build child")
    require(not path.exists(), "output already exists; evidence overwrite forbidden")
    return path


def write_json(path, document):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    require(Path.cwd().resolve() == ROOT and Path(__file__).resolve() == SOURCE
            and Path(sys.executable) == PYTHON and sys.dont_write_bytecode
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
            "repository/interpreter/bytecode identity mismatch")
    out = output_path(args.out)
    out.mkdir()
    origins = {key: record(path) for key, path in (
        ("source", SOURCE), ("contract", CONTRACT), ("test", TEST))}
    write_json(out / "command.json", {
        "cwd": str(ROOT), "executable": sys.executable, "argv": sys.argv,
        "command": COMMAND.replace(f"build/{NAME}_attempt001", shlex.quote(args.out)),
        "dont_write_bytecode": sys.dont_write_bytecode, "origins": origins,
        "environment": {key: os.environ.get(key) for key in (
            "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "python_version": sys.version, "numpy_version": np.__version__,
        "output": str(out), **FLAGS,
    })
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(record(path))
    spec = importlib.util.spec_from_file_location(f"test_{NAME}", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader missing")
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    collected = suite.countTestCases()
    with (out / "tests.log").open("x", encoding="utf-8") as stream:
        checked = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    passed = (collected == checked.testsRun == EXPECTED_TESTS
              and checked.wasSuccessful() and not checked.skipped)
    validation = {
        "diagnostic_id": ID, "status": "VALIDATED_SOFTWARE_SURFACE" if passed else "FAIL",
        "compiled": compiled, "origins": origins, "collected": collected,
        "executed": checked.testsRun, "failures": len(checked.failures),
        "errors": len(checked.errors), "skipped": len(checked.skipped),
        "tests_log": record(out / "tests.log"), "command": record(out / "command.json"),
        "ancestor_tests_executed": False, "diagnostic_executed": False, **FLAGS,
    }
    write_json(out / "validation.json", validation)
    require(passed, f"focused validation failed; see {out / 'tests.log'}")
    result = diagnose()
    require(origins == {key: record(Path(value["path"])) for key, value in origins.items()},
            "source/contract/test changed during diagnostic")
    result.update({"validation": record(out / "validation.json"),
                   "command": record(out / "command.json"), "origins": origins})
    write_json(out / "result.json", result)
    print(json.dumps({"status": result["status"], "classification": result["classification"],
                      "analyzed_control_count": result["analyzed_control_count"],
                      "tests_executed": checked.testsRun, "result": record(out / "result.json"),
                      "error": result.get("error")}, sort_keys=True))
    return 0 if result["classification"] == "SUPPORTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
