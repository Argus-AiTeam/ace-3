from copy import deepcopy
from fractions import Fraction
import io
import json
from pathlib import Path
import struct
import subprocess
from unittest.mock import patch

import pytest

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_row319_retained_operand_rounding_cell_audit_v1 as d


def oracle(word):
    return Fraction(struct.unpack("<e", struct.pack("<H", word))[0])


@pytest.fixture(scope="module")
def native(request):
    result = getattr(request.config, "_ace3_operand_cell_result", None)
    return d.analyze(d.retained.BoundBytes()) if result is None else result


def groups():
    return {dose: [list(d.CONTROLS[:7]) + [d.CONTROLS[8]], ["mapped_all"]] for dose in d.DOSES}


def test_all_finite_word_geometry_independent_oracle():
    for word in range(65536):
        if word & 0x7c00 == 0x7c00:
            continue
        cell = d.cell(word)
        assert Fraction(cell["value"]) == oracle(word)
        assert cell["sign_bit"] == word >> 15
        assert cell["significand_parity"] == word % 2
        if word & 0x7fff not in (0, 0x7bff):
            lower = oracle(cell["lower_neighbor_word"])
            upper = oracle(cell["upper_neighbor_word"])
            assert lower < oracle(word) < upper
            assert Fraction(cell["lower_midpoint"]) == (lower + oracle(word)) / 2
            assert Fraction(cell["upper_midpoint"]) == (upper + oracle(word)) / 2
        assert cell["lower_inclusive"] == cell["upper_inclusive"] == (word % 2 == 0)


def test_even_ties_odd_rejection_and_signed_cells():
    for word in (0x3c00, 0xbc00, 0x400, 0x8400):
        c = d.cell(word)
        assert d.bind(word, c["lower_midpoint"])["tie"] == "lower"
        assert d.bind(word, c["upper_midpoint"])["tie"] == "upper"
    for word in (0x3c01, 0xbc01, 1, 0x8001):
        for key in ("lower_midpoint", "upper_midpoint"):
            with pytest.raises(d.IntegrityError, match="outside"):
                d.bind(word, d.cell(word)[key])


def test_zero_subnormal_and_overflow_boundaries():
    assert d.cell(0)["upper_midpoint"] == "1/33554432"
    assert d.cell(0x8000)["lower_midpoint"] == "-1/33554432"
    assert d.bind(0, "1/33554432")["tie"] == "upper"
    assert d.bind(0x8000, "-1/33554432")["tie"] == "lower"
    for word in (0, 0x8000):
        with pytest.raises(d.IntegrityError, match="signed-zero"):
            d.bind(word, "0")
    assert d.cell(0x7bff)["upper_midpoint"] == "65520"
    assert d.cell(0xfbff)["lower_midpoint"] == "-65520"
    for word, value in ((0x7bff, "65520"), (0xfbff, "-65520")):
        with pytest.raises(d.IntegrityError, match="outside"):
            d.bind(word, value)
    assert d.ordinal(0) == d.ordinal(0x8000)
    assert d.ordinal(1) - d.ordinal(0x8001) == 2


def test_missing_partial_and_malformed_prerequisites():
    assert d.prerequisites({"working_fp16_words": []}) is None
    for record in ({"unrounded_operand_values": None}, {"unrounded_operand_values": ["1"]},
                   {"unrounded_cell": []}, {"tie_binding": "summary"}):
        with pytest.raises(d.IntegrityError):
            d.prerequisites(record)
    for value in (1.0, "1.0", "nan", "2/2"):
        with pytest.raises(ValueError):
            d.bind(0x3c00, value)


def test_four_materially_distinct_classifications():
    unique = groups()
    uniform = {dose: [list(d.CONTROLS)] for dose in d.DOSES}
    assert d.classify(unique, unique, [True] * 72) == d.SUPPORTED
    assert d.classify(unique, {dose: [] for dose in d.DOSES}, [False] * 72) == d.WORDS_ONLY
    assert d.classify(uniform, unique, [True] * 72) == d.MIXED
    assert d.classify(unique, uniform, [True] * 72) == d.MIXED
    assert len(set(d.SUCCESSORS.values())) == 4 and d.SUCCESSORS[d.UNKNOWN] is None


def test_partial_rotating_and_shared_controls():
    unique = groups()
    for dose in d.DOSES:
        changed = deepcopy(unique)
        changed[dose] = [list(d.CONTROLS)]
        assert d.classify(changed, unique, [False] * 72) == d.MIXED
        assert d.classify(unique, changed, [True] * 72) == d.MIXED
    with pytest.raises(d.IntegrityError, match="partial"):
        d.classify(unique, unique, [True] + [False] * 71)


def test_review_and_bound_bytes_fail_closed():
    review = {"kind": "round_reviewed_handoff", "producer_role": "engineer",
              "mission_id": "4b02378f0989", "round": 1, "review": {"status": "done"}}
    with pytest.raises(d.IntegrityError):
        d.retained.review_gate(review, "4b02378f0989", 1)
    bank = d.retained.BoundBytes(lambda path: b"changed")
    with pytest.raises(d.IntegrityError, match="SHA-256"):
        bank.read(d.SEAL)


def test_missing_authentication_is_unknown():
    def missing(path):
        raise FileNotFoundError(str(path))
    result = d.analyze(d.retained.BoundBytes(missing))
    assert result["classification"] == d.UNKNOWN
    assert "FileNotFoundError" in result["integrity_error"]
    assert result["availability"]["actual_cell_binding"] == "unconfirmed"
    assert result["successor"] is None and not any(result["successor_flags"].values())
    assert result["dose_search_closed"] and result["global_threshold_model_closed"]


def test_write_and_dispatch_refusal():
    audit = dict.fromkeys(d.AUDIT_KEYS, 0)
    with d.retained.read_only(audit):
        for call in (lambda: Path("/forbidden").write_bytes(b"x"),
                     lambda: subprocess.run(["true"], check=True)):
            with pytest.raises(d.IntegrityError):
                call()
    assert audit["writes"] == 1 and audit["forbidden_calls"] == 2


def test_strict_json_and_one_stdout_document(native):
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{}\n{}'):
        with pytest.raises(ValueError):
            d.retained.document(raw)
    stdout = io.StringIO()
    with patch.object(d, "check", return_value=native), patch("sys.stdout", stdout):
        assert d.main(["--check"]) == 0
    assert json.loads(stdout.getvalue()) == native


def test_independent_live_coordinate_and_partition_oracle(native):
    assert "report" in native, native.get("integrity_error")
    source, report = native["retained_family_report"], native["report"]
    assert len(report["observations"]) == 36
    profiles = {dose: {} for dose in d.DOSES}
    for observation in report["observations"]:
        dose, control = observation["dose"], observation["control"]
        baseline = source["retained_baselines"][control]["working_fp16_words"]
        output = source["retained_observations"][dose][d.CONTROLS.index(control)]["operand"]
        assert len(observation["coordinates"]) == 896
        expected = []
        for i, move in enumerate(observation["coordinates"]):
            a, b = baseline[i], output["working_fp16_words"][i]
            assert (move["coordinate"], move["baseline_word"], move["dose_word"]) == (i, a, b)
            delta = str(oracle(b) - oracle(a))
            assert move["numeric_delta"] == delta
            assert move["word_changed"] == (a != b)
            rank_a = -(a & 32767) if a & 32768 else a
            rank_b = -(b & 32767) if b & 32768 else b
            assert move["neighbor_steps"] == rank_b - rank_a
            assert move["baseline_binding"] is None and move["dose_binding"] is None
            expected.append(delta)
        assert observation["changed_word_count"] == len(output["changed_coordinates"])
        profiles[dose][control] = expected
    for dose, values in profiles.items():
        assert [c for c in d.CONTROLS if values[c] == values["mapped_all"]] == ["mapped_all"]
        for group in report["word_delta_partitions"][dose]:
            assert group == [c for c in d.CONTROLS if values[c] == values[group[0]]]


def test_live_exact_availability_and_missing_cell_boundary(native):
    assert native["classification"] == d.WORDS_ONLY, native.get("integrity_error")
    availability = native["availability"]
    assert availability == {
        "retained_baseline_vectors": 9, "retained_dose_vectors": 36,
        "retained_baseline_fp16_bytes": 16128, "retained_dose_fp16_bytes": 64512,
        "coordinate_pairs": 32256, "word_cell_geometry": True,
        "unrounded_baseline_dose_record_occurrences": 0,
        "actual_cell_binding": False, "actual_tie_binding": False,
    }
    assert len(native["report"]["missing_bindings"]) == 72
    assert all(not v for v in native["report"]["bound_cell_signature_partitions"].values())
    assert native["successor"] == d.SUCCESSORS[d.WORDS_ONLY]


def test_original_reference_and_historical_failure_closure(native):
    count = 0
    for dose in d.DOSES:
        for row in native["retained_family_report"]["retained_reference_closures"][dose]:
            old = oracle(row["baseline_words"][0]) - oracle(row["baseline_words"][1])
            new = oracle(row["measured_words"][0]) - oracle(row["measured_words"][1])
            fixed = Fraction(row["fixed_reference_margin"])
            assert Fraction(row["observed_delta"]) == new - old
            assert Fraction(row["retained_margin_change"]) == old - fixed
            assert Fraction(row["intervened_margin_change"]) == new - fixed
            count += 1
    assert count == 144
    assert native["retained_common_component"] == native["counterfactual_common_component"] == "UNKNOWN"
    assert native["prior_attempts"]


def test_live_corruption_cannot_be_word_only_success(native):
    parent = {key: native[key] for key in d.retained.PRESERVED}
    for key in ("historical_flags", "prior_attempts", "preserved_dyadic_rejection_counts"):
        parent[key] = native[key]
    parent["classification"] = d.boundary.OPERAND
    parent["retained_family_report"] = deepcopy(native["retained_family_report"])
    parent["retained_family_report"]["retained_observations"]["65/2048"][0]["operand"]["changed_coordinates"] = []
    parent["report"] = {}
    with patch.object(d, "authenticate", return_value=(parent, native["parent_capture_bindings"])):
        result = d.analyze(d.retained.BoundBytes())
    assert result["classification"] == d.UNKNOWN
    assert "changed-coordinate" in result["integrity_error"]
    assert not any(result["successor_flags"].values())


def test_live_authentication_pins_and_capture_identity(native):
    paths = {pin["path"] for pin in native["authenticated_pins"]}
    assert d.REVIEW["path"] in paths and d.SEAL["path"] in paths
    assert d.boundary.REVIEW["path"] in paths and d.boundary.SEAL["path"] in paths
    assert all(pin["path"] in paths for pin in d.retained.REVIEWS)
    seal = native["parent_capture_bindings"]
    assert seal["stdout_sha256"] != seal["whole_capture_sha256"]
    assert seal["source_test_pins_before"] == seal["source_test_pins_after"]


def test_non_admission_and_closed_branch_flags(native):
    assert native["normal_host_review"] == "REQUIRED"
    assert native["flags"] == d.boundary.base_result()["flags"]
    assert native["dose_search_closed"] and native["global_threshold_model_closed"]
    assert "wider Q24" in native["claim_boundary"]
    assert "independently propagated global references" in native["claim_boundary"]
    assert sum(native["successor_flags"].values()) == 1
