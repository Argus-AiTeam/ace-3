"""Independent host binary16 oracle, falsification tests and sealed one-shot runner."""

import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import sys
from unittest.mock import patch

import pytest

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mlp_stage17_s18_coordinate761_rne_cell_anatomy_v1 as diagnostic


def value(word):
    return Fraction.from_float(struct.unpack("<e", struct.pack("<H", word))[0])


def word(number):
    try:
        return struct.unpack("<H", struct.pack("<e", number))[0]
    except OverflowError:
        return 0xfc00 if number < 0 else 0x7c00


def verify_account(row, cell):
    left = Fraction(row["scratch_i"], 1 << 24)
    total = left + value(int(row["stage17_fp16_word"], 16))
    # A finite Q24-to-half input has at most 41 significant bits: binary64 is exact here.
    if abs(total) >= 65520 or Fraction.from_float(float(total)) != total:
        return False
    zero = int(total == 0 and row["scratch_i"] == 0 and row["scratch_z"] == 1
               and row["stage17_fp16_word"] == "8000")
    expected_word = word(-0.0 if zero else float(total))
    actual_word = int(row["actual_fp16_word"], 16)
    ref = float.fromhex(row["original_input_binary64_hex"])
    exact_ref = Fraction.from_float(ref)
    nearest = word(ref)
    floor = abs(value(nearest) - exact_ref)
    radius = floor + Fraction(1, 8)
    rounded = value(actual_word)
    expected = {
        "unrounded_fixed_operand_sum": str(total),
        "expected_q24_i": int(total * (1 << 24)), "actual_q24_i": int(total * (1 << 24)),
        "expected_q24_z": zero, "actual_q24_z": zero,
        "expected_rne_word": f"{expected_word:04x}", "actual_fp16_word": f"{expected_word:04x}",
        "nearest_fp16_word": f"{nearest:04x}", "nearest_fp16_floor": str(floor),
        "original_input_binary64_value": str(exact_ref), "binary64_authority_radius": str(radius),
        "excess_budget": "1/8", "threshold_margin": str(radius - abs(rounded - exact_ref)),
        "unrounded_threshold_margin": str(radius - abs(total - exact_ref)),
        "actual_error": str(abs(rounded - exact_ref)),
        "excess": str(abs(rounded - exact_ref) - floor),
        "accepted": abs(rounded - exact_ref) <= radius,
    }
    normalize = (lambda w: w & 0x7fff) if actual_word & 0x7fff == 0 else (lambda w: w)
    for key, direction in (("lower", math.inf), ("upper", -math.inf)):
        endpoint = float(Fraction(cell[key]))
        inside = normalize(word(math.nextafter(endpoint, direction)))
        outside = normalize(word(math.nextafter(endpoint, -direction)))
        tie_owned = normalize(word(endpoint)) == normalize(actual_word)
        if (inside != normalize(actual_word) or outside == normalize(actual_word)
                or tie_owned != cell[key + "_tie_owned"]):
            return False
    left_view = word(-0.0 if left == 0 and row["scratch_z"] == 1 else float(left))
    return (row["state_error"] is None and left_view == int(row["stage12_fp16_word"], 16)
            and all(row[k] == v for k, v in expected.items()))


def sample(total=Fraction(-1936905, 524288), reference=float.fromhex("-0x1.c8d0e46e19dcep+1")):
    integer = int(total * (1 << 24))
    return diagnostic.previous.coordinate(
        761, integer, 0, word(float(total)), 0, integer, 0, word(float(total)),
        word(reference), reference)


def analyzed(row=None):
    return diagnostic.anatomy(sample() if row is None else row, sys.modules[__name__])


@pytest.mark.parametrize("bits", [0x3c00, 0x3c01, 0xbc00, 0xbc01, 0x400, 0x3ff, 1,
                                  0x8001, 0, 0x8000, 0x7bff, 0xfbff, 0xc364])
def test_exact_cells_and_tie_ownership(bits):
    cell = diagnostic.rne_cell(bits)
    lower, upper = Fraction(cell["lower"]), Fraction(cell["upper"])
    assert lower < value(bits) < upper
    assert cell["lower_tie_owned"] == cell["upper_tie_owned"] == (bits & 1 == 0)
    normalize = (lambda w: w & 0x7fff) if bits & 0x7fff == 0 else (lambda w: w)
    for endpoint, direction in ((lower, math.inf), (upper, -math.inf)):
        assert normalize(word(math.nextafter(float(endpoint), direction))) == normalize(bits)
        assert normalize(word(math.nextafter(float(endpoint), -direction))) != normalize(bits)
        assert (normalize(word(float(endpoint))) == normalize(bits)) == (bits & 1 == 0)


def test_eight_case_exact_crossing_and_independent_oracle():
    result = analyzed()
    assert result["outward_radius_crossing"] and not any(result["anomaly_flags"].values())
    assert result["rne_cell"]["lower"] == "-3785/1024"
    assert result["rne_cell"]["upper"] == "-3783/1024"
    assert result["before_rounding_radius_margin"] == "20995690215/562949953421312"
    assert result["after_rounding_radius_margin"] == "-519096447257/562949953421312"
    assert Fraction(result["outward_error_movement"]) == Fraction(503, 524288)


@pytest.mark.parametrize("total,reference", [
    (Fraction(1), 1.125), (Fraction(2049, 2048), 1.0),
    (Fraction(2051, 2048), 1.0), (Fraction(1, 1 << 24), 0.0), (Fraction(65504), 65504.0),
])
def test_threshold_ties_subnormal_and_finite_limit(total, reference):
    row = sample(total, reference)
    result = analyzed(row)
    assert verify_account(row, result["rne_cell"])
    assert not result["anomaly_flags"]["oracle_mismatch"]
    if total == 1:
        assert result["after_rounding_radius_margin"] == "0" and row["accepted"]
    if total == Fraction(1, 1 << 24):
        assert result["anomaly_flags"]["subnormal"]
    if total in (Fraction(2049, 2048), Fraction(2051, 2048)):
        assert result["anomaly_flags"]["unexpected_tie"]


def test_negative_zero_and_cancellation_are_not_hidden():
    row = diagnostic.previous.coordinate(761, 0, 1, 0x8000, 0x8000, 0, 1, 0x8000, 0, 0.0)
    result = analyzed(row)
    assert result["anomaly_flags"]["signed_zero"]
    assert not result["anomaly_flags"]["oracle_mismatch"]
    assert row["expected_q24_z"] == 1
    row = diagnostic.previous.coordinate(761, 1 << 24, 0, 0x3c00, 0xbc00, 0, 0, 0, 0, 0.0)
    assert analyzed(row)["anomaly_flags"]["signed_zero"]
    assert row["expected_q24_z"] == 0


def test_overflow_is_an_explicit_anomaly():
    integer = 65520 * (1 << 24)
    row = diagnostic.previous.coordinate(
        761, integer, 0, 0x7bff, 0, integer, 0, 0x7bff, 0x7bff, 65504.0)
    result = analyzed(row)
    assert result["anomaly_flags"]["overflow"] and result["anomaly_flags"]["state_mismatch"]
    assert result["anomaly_flags"]["oracle_mismatch"]


@pytest.mark.parametrize("key,replacement,flag", [
    ("scratch_exact_value", "1", "scratch_view_substitution"),
    ("unrounded_fixed_operand_sum", "1", "scratch_view_substitution"),
    ("actual_q24_i", 1, "oracle_mismatch"),
    ("nearest_fp16_floor", "1", "oracle_mismatch"),
    ("state_matches", False, "state_mismatch"),
    ("threshold_margin", "0", "cell_geometry_mismatch"),
    ("actual_fp16_word", "7c00", "malformed_operands"),
    ("unrounded_fixed_operand_sum", "not-rational", "malformed_operands"),
])
def test_localized_arithmetic_anomalies(key, replacement, flag):
    row = sample()
    row[key] = replacement
    assert analyzed(row)["anomaly_flags"][flag]


def controls():
    rows = []
    for name in diagnostic.CONTROLS:
        result = analyzed(sample(Fraction(-61905946, 1 << 24)) if name == "mapped_all" else None)
        rows.append({**result, "control": name, "integrity_issues": [], "failure_count": 83,
                     "unrounded_outside_radius_count": 83 if name == "mapped_all" else 82})
    return rows


def test_supported_joint_mechanism_preserves_mapped_all_distinction():
    rows = controls()
    assert diagnostic.classify(rows) == ("SUPPORTED", [])
    mapped = next(r for r in rows if r["control"] == "mapped_all")
    assert mapped["account"]["accepted"] and not mapped["outward_radius_crossing"]


@pytest.mark.parametrize("defect", ["count", "crossing", "identity", "state", "oracle", "kv"])
def test_joint_negative_interpretations(defect):
    rows = controls()
    if defect == "count":
        rows[0]["unrounded_outside_radius_count"] = 83
    elif defect == "crossing":
        rows[0]["outward_radius_crossing"] = False
    elif defect == "identity":
        rows.reverse()
    elif defect in ("state", "oracle"):
        rows[0]["anomaly_flags"][defect + "_mismatch"] = True
    else:
        rows[0]["integrity_issues"].append("kv_k")
    assert diagnostic.classify(rows)[0] == "REJECTED"


@pytest.mark.parametrize("member", ["stdout", "capture", "outer", "review"])
def test_f3_authentication_defect(member, monkeypatch):
    pins = copy.deepcopy(diagnostic.PINS)
    pins[member]["sha256"] = "0" * 64
    monkeypatch.setattr(diagnostic, "PINS", pins)
    with pytest.raises(ValueError):
        diagnostic.authenticate()


@pytest.mark.parametrize("member", ["stdout", "capture", "outer", "review"])
def test_f96_authentication_splice_defect(member, monkeypatch):
    pins = copy.deepcopy(diagnostic.previous.PINS)
    pins[member]["sha256"] = "0" * 64
    monkeypatch.setattr(diagnostic.previous, "PINS", pins)
    with pytest.raises(ValueError, match="lineage pins"):
        diagnostic.authenticate()


def test_both_reviewed_captures_authenticate_without_producer_execution():
    audit = dict.fromkeys(diagnostic.COUNTS, 0)
    with diagnostic.read_only(audit):
        retained, parent = diagnostic.authenticate()
    assert retained["status"] == parent["status"] == "REJECTED"
    assert not any(audit.values())


@pytest.mark.parametrize("source", [diagnostic.previous.SOURCE, diagnostic.previous.TEST,
                                    diagnostic.previous.previous.SOURCE,
                                    diagnostic.previous.previous.TEST])
def test_f3_and_f96_source_test_drift(source, monkeypatch):
    binding = diagnostic.capture.binding

    def changed(path):
        pin = binding(path)
        return {**pin, "sha256": "0" * 64} if Path(path) == source else pin

    monkeypatch.setattr(diagnostic.capture, "binding", changed)
    with pytest.raises(ValueError, match="source/test"):
        diagnostic.authenticate()


def test_closed_dispatch_and_writes_blocked(tmp_path):
    audit = dict.fromkeys(diagnostic.COUNTS, 0)
    with diagnostic.read_only(audit):
        for call in (
            diagnostic.previous.check, diagnostic.previous.previous.check,
            lambda: diagnostic.previous.diagnose_control(None),
            lambda: diagnostic.previous.previous.native.state.add({}, None),
            lambda: diagnostic.previous.previous.parent.rmsnorm(None, None),
            lambda: diagnostic.previous.previous.parent.logits(None, None),
            lambda: (tmp_path / "forbidden").write_text("no"),
        ):
            with pytest.raises(RuntimeError):
                call()
    assert audit["forbidden_calls"] == 7 and not (tmp_path / "forbidden").exists()


def test_unknown_is_one_json_and_nonzero(monkeypatch, capsys):
    def failure():
        raise ValueError("capture authentication defect")
    monkeypatch.setattr(diagnostic, "check", failure)
    assert diagnostic.main(["--check"]) == 1
    output = capsys.readouterr()
    assert output.err == "" and len(output.out.splitlines()) == 1
    result = json.loads(output.out)
    assert result["status"] == "UNKNOWN" and result["flags"] == diagnostic.FLAGS


def test_uncaptured_execution_refused(monkeypatch):
    monkeypatch.setattr(diagnostic.previous.previous.os, "readlink", lambda path: "/dev/null")
    with pytest.raises(ValueError, match="capture"):
        diagnostic.capture_preflight()


def transport(mode, directory):
    directory = Path(directory).absolute()
    attempt = directory if mode == "--launch" else directory.parent
    if (attempt.parent != diagnostic.ROOT / "build" or attempt.resolve() != attempt
            or not attempt.name.startswith("s18-coordinate761-rne-cell-456967d2260e-attempt")):
        raise RuntimeError("fresh mission-scoped build capture required")
    harness = diagnostic.previous.previous.load_module(
        diagnostic.previous.previous.TEST, diagnostic.NAME + "_transport")
    with patch.object(harness, "diagnostic", diagnostic):
        return harness.launch(directory) if mode == "--launch" else harness.capture_run(directory)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--launch", "--run"):
        raise SystemExit("usage: test module --launch BUILD_ATTEMPT | --run BUILD_ATTEMPT/run")
    raise SystemExit(transport(sys.argv[1], sys.argv[2]))
