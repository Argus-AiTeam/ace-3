"""Independent scalar oracle, falsification tests, and the reviewed shared sealer."""

import copy
from fractions import Fraction
import json
from pathlib import Path
import struct
import sys
from unittest.mock import patch

import numpy as np
import pytest

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mlp_stage17_full_vector_s18_boundary_diagnostic_v1 as diagnostic


def value(word):
    return Fraction.from_float(struct.unpack("<e", struct.pack("<H", word))[0])


def rne(total, negative_zero=False):
    if not total:
        return 0x8000 if negative_zero else 0
    sign = 0x8000 if total < 0 else 0
    magnitude = abs(total)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < Fraction(2) ** exponent:
        exponent -= 1
    shift = max(-24, exponent - 10)
    units = round(magnitude / Fraction(2) ** shift)
    rounded = units * Fraction(2) ** shift
    return struct.unpack("<H", struct.pack("<e", float(rounded)))[0] | sign


def verify_coordinate(row):
    left = Fraction(row["scratch_i"], 1 << 24)
    right = value(int(row["stage17_fp16_word"], 16))
    total = left + right
    zero = row["scratch_i"] == 0 and row["scratch_z"] == 1 and row["stage17_fp16_word"] == "8000"
    reference = float.fromhex(row["original_input_binary64_hex"])
    exact_ref = Fraction.from_float(reference)
    nearest = struct.unpack("<H", struct.pack("<e", reference))[0]
    floor = abs(value(nearest) - exact_ref)
    actual = value(int(row["actual_fp16_word"], 16))
    radius = floor + Fraction(1, 8)
    expected = {
        "unrounded_fixed_operand_sum": str(total),
        "expected_q24_i": int(total * (1 << 24)), "expected_q24_z": int(zero),
        "expected_rne_word": f"{rne(total, zero):04x}",
        "nearest_fp16_word": f"{nearest:04x}", "nearest_fp16_floor": str(floor),
        "actual_error": str(abs(actual - exact_ref)),
        "excess": str(abs(actual - exact_ref) - floor),
        "threshold_margin": str(radius - abs(actual - exact_ref)),
        "binary64_authority_radius": str(radius),
        "unrounded_outside_radius": abs(total - exact_ref) > radius,
        "unrounded_threshold_margin": str(radius - abs(total - exact_ref)),
        "fixed_operand_rne_passable": abs(value(rne(total, zero)) - exact_ref) <= radius,
    }
    return row["state_error"] is None and all(row[k] == v for k, v in expected.items())


def sample(total=Fraction(1), reference=1.25):
    word = rne(total)
    row = diagnostic.coordinate(0, int(total * (1 << 24)), 0, word, 0,
                                int(total * (1 << 24)), 0, word, rne(Fraction(reference)),
                                reference)
    row["oracle_matches"] = verify_coordinate(row)
    return row


@pytest.mark.parametrize("total,reference", [
    (Fraction(1), 1.25), (Fraction(-1), -1.25),
    (Fraction(1, 1 << 24), 0.0), (Fraction(2049, 2048), 1.0),
    (Fraction(2051, 2048), 1.0), (Fraction(65504), 65504.0),
])
def test_independent_exact_accounts(total, reference):
    row = sample(total, reference)
    assert row["oracle_matches"] and row["state_matches"] and row["rne_matches"]
    assert Fraction(row["unrounded_fixed_operand_sum"]) == total


@pytest.mark.parametrize("left_z,right_word,expected_z", [(1, 0x8000, 1), (0, 0x8000, 0), (1, 0, 0)])
def test_signed_zero(left_z, right_word, expected_z):
    word = expected_z << 15
    row = diagnostic.coordinate(0, 0, left_z, left_z << 15, right_word, 0, expected_z,
                                word, 0, 0.0)
    assert row["expected_q24_z"] == expected_z and verify_coordinate(row)
    assert row["state_matches"] and row["rne_matches"]


def test_cancellation_is_positive_zero():
    row = diagnostic.coordinate(0, 1 << 24, 0, 0x3c00, 0xbc00, 0, 0, 0, 0, 0.0)
    assert row["expected_q24_z"] == 0 and verify_coordinate(row)


def test_wide_state_is_not_fp16_view_operand():
    row = sample(Fraction(1) + Fraction(1, 1 << 24), 1.25)
    assert row["unrounded_fixed_operand_sum"] != row["fp16_view_operand_sum_not_state_operand"]
    assert row["scratch_minus_fp16_view"] == "1/16777216"


def test_exact_threshold_equality_and_adjacent_binary64():
    boundary = sample(Fraction(1), 1.125)
    assert boundary["threshold_margin"] == "0" and boundary["accepted"]
    outside = sample(Fraction(1), 1.126953125)
    assert Fraction(outside["threshold_margin"]) < 0 and not outside["accepted"]


def test_boundary_supported_and_rounding_only_rejected():
    outside = sample()
    assert diagnostic.classify([outside] * 83, [])[0] == "SUPPORTED"
    rounded_failure = sample(Fraction(1) + Fraction(1, 4096), 1.1256103515625)
    assert not rounded_failure["accepted"] and not rounded_failure["unrounded_outside_radius"]
    assert diagnostic.classify([rounded_failure] * 83, [])[0] == "REJECTED"


@pytest.mark.parametrize("defect", [
    "state_matches", "rne_matches", "scratch_view_matches", "oracle_matches",
    "accepted", "fixed_operand_rne_passable", "unrounded_outside_radius",
])
def test_negative_interpretations_are_rejected(defect):
    row = sample()
    row[defect] = not row[defect]
    assert diagnostic.classify([row] * 83, [])[0] == "REJECTED"


@pytest.mark.parametrize("issue", ["kv_k", "kv_v", "residual_state_lineage", "original_stage17_operand"])
def test_integrity_mismatch_is_rejected_not_unknown(issue):
    assert diagnostic.classify([sample()] * 83, [issue])[0] == "REJECTED"


def test_failure_census_and_oracle_mutation():
    row = sample()
    assert diagnostic.classify([row] * 82, [])[0] == "REJECTED"
    row["nearest_fp16_floor"] = "1"
    assert not verify_coordinate(row)


@pytest.mark.parametrize("defect", [None, "kv", "state", "operand", "gate"])
def test_complete_control_integrity(defect):
    actual = {
        "input_i": np.full(896, 1 << 24, dtype="<i8"),
        "input_z": np.zeros(896, dtype="u1"),
        "input_hidden": np.full(896, 0x3c00, dtype="<u2"),
        "scratch_i": np.full(896, 1 << 24, dtype="<i8"),
        "scratch_z": np.zeros(896, dtype="u1"),
        "stage12": np.full(896, 0x3c00, dtype="<u2"),
        "stage03": np.zeros(128, dtype="<u2"), "stage05": np.zeros(128, dtype="<u2"),
        "output_cache_k": np.zeros((1, 128), dtype="<u2"),
        "output_cache_v": np.zeros((1, 128), dtype="<u2"),
    }
    output = {"stage17": [0] * 896, "stage18": [0x3c00] * 896,
              "output_i": [1 << 24] * 896, "output_z": [0] * 896}
    reference = {"stage17": np.zeros(896, dtype="<u2"),
                 "stage18": np.full(896, 0x3d00, dtype="<u2")}
    binary64 = np.ones(896, dtype="<f8")
    binary64[:83] = 1.25
    failed, passed = sample()["binary64_gate"], sample(reference=1.0)["binary64_gate"]
    gates = [{**(failed if k < 83 else passed), "index": k} for k in range(896)]
    report = {"stage": 18, "status": "FAIL", "kv_lineage": "PASS",
              "residual_state_lineage": "PASS", "binary64_v1": {
                  "rows": gates, "failures": copy.deepcopy(gates[:83]),
                  "failure_count": 83, "passed": False}}
    if defect == "kv":
        actual["output_cache_k"][0, 0] = 1
    elif defect == "state":
        output["output_i"][895] += 1
    elif defect == "operand":
        reference["stage17"][895] = 1
    elif defect == "gate":
        gates[895]["excess_error"] = "1"
    audit = dict.fromkeys(diagnostic.COUNTS, 0)
    result = diagnostic.diagnose_control(
        "synthetic", actual, output, reference, binary64, report, sys.modules[__name__], audit)
    assert result["status"] == ("SUPPORTED" if defect is None else "REJECTED")
    assert result["failure_count"] == result["unrounded_outside_radius_count"] == 83
    assert audit["exact_coordinate_accounts"] == audit["independent_scalar_oracles"] == 896


@pytest.mark.parametrize("integer,zero,word", [(1 << 24, 1, 0x3c00), (1, 0, 0x3c00)])
def test_state_transition_mismatch(integer, zero, word):
    row = diagnostic.coordinate(0, 1 << 24, 0, 0x3c00, 0, integer, zero, word, word, 1.25)
    row["oracle_matches"] = verify_coordinate(row)
    assert not row["state_matches"]
    assert diagnostic.classify([row] * 83, [])[0] == "REJECTED"


def test_forbidden_dispatch_and_write_guards(tmp_path):
    audit = dict.fromkeys(diagnostic.COUNTS, 0)
    with diagnostic.read_only(audit):
        for call in (
            lambda: diagnostic.previous.check(),
            lambda: diagnostic.previous.native.state.add({}, None),
            lambda: diagnostic.previous.native.projection(None, None, None),
            lambda: diagnostic.previous.parent.rmsnorm(None, None),
            lambda: diagnostic.previous.parent.logits(None, None),
            lambda: diagnostic.previous.layer.expected_stage(18, None, None, None),
            lambda: (tmp_path / "forbidden").write_text("no"),
        ):
            with pytest.raises(RuntimeError):
                call()
    assert audit["forbidden_calls"] == 7 and not (tmp_path / "forbidden").exists()


@pytest.mark.parametrize("member", ["stdout", "capture", "outer", "review"])
def test_authenticated_member_tampering_unknown(member, monkeypatch):
    pins = copy.deepcopy(diagnostic.PINS)
    pins[member]["sha256"] = "0" * 64
    monkeypatch.setattr(diagnostic, "PINS", pins)
    with pytest.raises(ValueError):
        diagnostic.authenticate()


def test_authenticate_without_closed_execution():
    audit = dict.fromkeys(diagnostic.COUNTS, 0)
    with diagnostic.read_only(audit):
        result = diagnostic.authenticate()
    assert result["status"] == "REJECTED"
    assert all(report["binary64_v1"]["failure_count"] == 83
               for report in result["stage_reports"].values())
    assert not any(audit.values())


@pytest.mark.parametrize("path", [diagnostic.previous.SOURCE, diagnostic.previous.TEST])
def test_parent_source_test_drift_rejected(path, monkeypatch):
    binding = diagnostic.capture.binding

    def changed(candidate):
        pin = binding(candidate)
        return {**pin, "sha256": "0" * 64} if Path(candidate) == path else pin

    monkeypatch.setattr(diagnostic.capture, "binding", changed)
    with pytest.raises(ValueError, match="source/test"):
        diagnostic.authenticate()


def test_unknown_is_one_json_and_nonzero(monkeypatch, capsys):
    def fail():
        raise ValueError("capture authentication defect")
    monkeypatch.setattr(diagnostic, "check", fail)
    assert diagnostic.main(["--check"]) == 1
    output = capsys.readouterr()
    assert not output.err and len(output.out.splitlines()) == 1
    assert json.loads(output.out)["status"] == "UNKNOWN"


def test_uncaptured_execution_refused(monkeypatch):
    monkeypatch.setattr(diagnostic.previous.os, "readlink", lambda path: "/dev/null")
    with pytest.raises(ValueError, match="capture"):
        diagnostic.capture_preflight()


def transport(mode, directory):
    directory = Path(directory).absolute()
    attempt = directory if mode == "--launch" else directory.parent
    if (attempt.parent != diagnostic.ROOT / "build"
            or not attempt.name.startswith("mlp17-full-vector-s18-boundary-")
            or attempt.resolve() != attempt):
        raise RuntimeError("mission-scoped fresh capture path required")
    harness = diagnostic.previous.load_module(diagnostic.previous.TEST, diagnostic.NAME + "_transport")
    original_scope = diagnostic.capture.same_scope
    lock_root = diagnostic.ROOT / "build/mlp17-full-vector-s18-boundary-locks"
    (lock_root / "build").mkdir(parents=True, exist_ok=True)

    def scoped_lock(root, scope):
        diagnostic.same((root, scope), (diagnostic.ROOT, diagnostic.MODULE), "scope identity")
        return original_scope(lock_root, scope)

    # The shared lock keeps its stable inode, inside this mission's writable prefix.
    with patch.object(harness, "diagnostic", diagnostic), patch.object(
            diagnostic.capture, "same_scope", scoped_lock):
        return harness.launch(directory) if mode == "--launch" else harness.capture_run(directory)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--launch", "--run"):
        raise SystemExit("usage: test module --launch BUILD_ATTEMPT | --run BUILD_ATTEMPT/run")
    raise SystemExit(transport(sys.argv[1], sys.argv[2]))
