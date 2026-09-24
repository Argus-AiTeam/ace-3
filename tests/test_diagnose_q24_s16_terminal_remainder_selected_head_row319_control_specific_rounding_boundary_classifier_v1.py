"""Independent exact-bit response and integrity tests; no suffix or parent check."""

from copy import deepcopy
from fractions import Fraction
import io
import json
from pathlib import Path
import struct
import subprocess
from unittest.mock import patch

import pytest

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_row319_control_specific_rounding_boundary_classifier_v1 as d


def profiles():
    return {dose: {c: {f: [0] for f in d.FIELDS} for c in d.CONTROLS} for dose in d.DOSES}


def split(p, field):
    for dose in d.DOSES:
        p[dose]["mapped_all"][field] = [1]
    return p


def fp16(word):
    return Fraction(struct.unpack("<e", struct.pack("<H", word))[0])


@pytest.fixture(scope="module")
def native(request):
    result = getattr(request.config, "_ace3_rounding_boundary_result", None)
    return d.analyze(d.retained.BoundBytes()) if result is None else result


def test_operand_is_earliest_even_when_downstream_also_splits():
    p = split(split(profiles(), "operand_delta"), "rmsnorm_output_delta")
    assert d.classify(p)["classification"] == d.OPERAND
    assert d.classify(p)["localizing_fields"] == ["operand_delta"]


def test_rmsnorm_requires_equal_upstream_response():
    for field in ("rmsnorm_scalar_delta", "rmsnorm_output_delta"):
        result = d.classify(split(profiles(), field))
        assert result["classification"] == d.RMSNORM
        assert sum(result["successor_flags"].values()) == 1


def test_head_requires_complete_equal_upstream_response():
    assert d.classify(split(profiles(), "head_rne_delta"))["classification"] == d.HEAD
    p = profiles()
    for dose in d.DOSES:
        for control in d.CONTROLS:
            p[dose][control]["head_rne_delta"] = None
    result = d.classify(p)
    assert result["classification"] == d.UNKNOWN and result["successor"] is None
    assert not any(result["successor_flags"].values())
    assert not result["field_availability"]["head_rne_delta"]


def test_shared_partial_and_rotating_splits_are_mixed():
    p = split(profiles(), "operand_delta")
    p["17/512"]["mapped_all"]["operand_delta"] = [0]
    assert d.classify(p)["classification"] == d.MIXED
    p = split(profiles(), "operand_delta")
    for dose in d.DOSES:
        p[dose]["mapped62"]["operand_delta"] = [1]
    assert d.classify(p)["classification"] == d.MIXED
    p = profiles()
    for i, dose in enumerate(d.DOSES):
        p[dose]["mapped_all"][d.FIELDS[1 + i % 2]] = [1]
    assert d.classify(p)["classification"] == d.MIXED


def test_uniform_and_missing_profiles_never_mean_head_support():
    assert d.classify(profiles())["classification"] == d.UNKNOWN
    for mutate in (lambda p: p.pop("17/512"),
                   lambda p: p.update({"1/16": {}}),
                   lambda p: p["65/2048"].pop("scratch"),
                   lambda p: p["33/1024"]["mapped_all"].update(head_rne_delta=None)):
        p = profiles()
        mutate(p)
        with pytest.raises(d.IntegrityError):
            d.classify(p)


def test_exact_word_delta_oracle_and_baseline_offset_removal():
    before = [0, 32768, 1, 1023, 15360, 18462, 48128]
    after = [1, 0, 2, 1024, 15361, 18463, 48129]
    assert d.word_delta(before, after, len(before)) == [
        str(fp16(b) - fp16(a)) for a, b in zip(before, after)]
    assert d.word_delta([15360], [15361], 1) == d.word_delta([15362], [15363], 1)
    for words in ([True], [31744], [65536], []):
        with pytest.raises(d.IntegrityError):
            d.word_delta(words, [0], 1)


def test_root_cells_use_retained_scalar_inequalities_only():
    assert d.root_cell({"mean_q48": 99, "root_q24": 9}) == {
        "lower_squared_slack_q48": 18, "upper_squared_slack_q48": 1}
    assert d.root_cell({"mean_q48": 100, "root_q24": 10})["lower_squared_slack_q48"] == 0
    for scalars in ({"mean_q48": 100, "root_q24": 9},
                    {"mean_q48": 99, "root_q24": 10},
                    {"mean_q48": True, "root_q24": 1}, {}):
        with pytest.raises(d.IntegrityError):
            d.root_cell(scalars)


def test_selected_head_rne_cells_and_even_ties():
    for word in (15360, 15361, 18462, 48128):
        assert d.rne_residual(word, str(fp16(word))) == 0
    midpoint = (fp16(18462) + fp16(18463)) / 2
    assert d.rne_residual(18462, str(midpoint)) == fp16(18462) - midpoint
    with pytest.raises(d.IntegrityError):
        d.rne_residual(18463, str(midpoint))
    with pytest.raises(d.IntegrityError):
        d.rne_residual(18462, "0")


def test_head_scalars_cannot_be_reconstructed_from_logits():
    old = {"selected_logit_words": [18462, 18490]}
    new = {"selected_logit_words": [18463, 18490]}
    assert d.head_response(old, new) is None
    record = {"exact_accumulator": [str(fp16(w)) for w in old["selected_logit_words"]],
              "post_hidden_rne_accumulator": [str(fp16(w)) for w in old["selected_logit_words"]]}
    old["selected_head_accumulators"] = record
    with pytest.raises(d.IntegrityError):
        d.head_response(old, new)
    new = deepcopy(old)
    new["selected_head_accumulators"]["exact_accumulator"][0] = str(fp16(18462) - Fraction(1, 1024))
    response = d.head_response(old, new)
    assert response[0] == {"row": 319, "hidden_rne_effect_delta": "1/1024",
                           "head_rne_residual_delta": "0"}
    assert response[1]["row"] == 34319


def test_review_authority_and_bound_bytes_refuse_mutation():
    review = {"kind": "round_reviewed_handoff", "producer_role": "reviewer",
              "mission_id": "412ec79f743b", "round": 1, "review": {"status": "done"}}
    d.retained.review_gate(review, "412ec79f743b", 1)
    with pytest.raises(d.IntegrityError):
        d.retained.review_gate({**review, "producer_role": "engineer"}, "412ec79f743b", 1)
    bank = d.retained.BoundBytes(lambda path: b"sealed")
    for pin in ({"path": "/sealed", "sha256": "0" * 64},
                {"path": "/sealed", "sha256": d.retained.digest(b"sealed"), "bytes": 7}):
        with pytest.raises(d.IntegrityError):
            bank.read(pin)


def test_capture_segments_and_separate_stream_hashes():
    streams = {"stdout": b"{}\n", "stderr": b"ok\n", "whole_capture": b"ok\n{}\n"}
    seal = {"capture_complete": True, "timed_out": False, "returncode": 0,
            "original_temporary_directory": "/original", "sealed_build_directory": "/sealed",
            "stdout_sha256": d.retained.digest(streams["stdout"]),
            "whole_capture_sha256": d.retained.digest(streams["whole_capture"]), "artifacts": {},
            "stream_segments": [
                {"stream": "stderr", "size_bytes": 3, "stream_offset": 0, "whole_offset": 0},
                {"stream": "stdout", "size_bytes": 3, "stream_offset": 0, "whole_offset": 3}]}
    files = {}
    for name, raw in streams.items():
        pin = {"path": "/sealed/" + name, "sha256": d.retained.digest(raw), "size_bytes": len(raw)}
        seal["artifacts"][name] = {"source": {**pin, "path": "/original/" + name}, "destination": pin}
        files[pin["path"]] = raw
    bank = d.retained.BoundBytes(lambda path: files[str(path)])
    assert d.sealed_stdout(seal, bank) == b"{}\n"
    for mutate in (lambda s: s["stream_segments"][1].update(whole_offset=0),
                   lambda s: s["stream_segments"].pop(),
                   lambda s: s.update(stdout_sha256=s["whole_capture_sha256"]),
                   lambda s: s["artifacts"]["stderr"]["source"].update(size_bytes=0)):
        changed = deepcopy(seal)
        mutate(changed)
        with pytest.raises(d.IntegrityError):
            d.sealed_stdout(changed, bank)


def test_missing_authentication_is_explicit_unknown():
    def missing(path):
        raise FileNotFoundError(str(path))
    result = d.analyze(d.retained.BoundBytes(missing))
    assert result["classification"] == d.UNKNOWN and result["successor"] is None
    assert "FileNotFoundError" in result["integrity_error"]
    assert result["dose_search_closed"] and result["global_threshold_model_closed"]
    assert not any(result["successor_flags"].values())


def test_no_write_or_external_dispatch():
    audit = dict.fromkeys(d.AUDIT_KEYS, 0)
    with d.retained.read_only(audit):
        for call in (lambda: Path("/forbidden").write_bytes(b"x"),
                     lambda: subprocess.run(["true"], check=True)):
            with pytest.raises(d.IntegrityError):
                call()
    assert audit["writes"] == 1 and audit["forbidden_calls"] == 2


def test_strict_json_and_single_stdout_document(native):
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{}\n{}'):
        with pytest.raises(ValueError):
            d.retained.document(raw)
    stdout = io.StringIO()
    with patch.object(d, "check", return_value=native), patch("sys.stdout", stdout):
        assert d.main(["--check"]) == 0
    assert json.loads(stdout.getvalue()) == native


def test_independent_live_response_and_classification_oracle(native):
    assert "report" in native, native.get("integrity_error")
    source, result = native["retained_family_report"], native["report"]
    assert (result["observation_count"], result["selected_row_observation_count"]) == (36, 72)
    unique = {f: True for f in d.FIELDS[:3]}
    split_fields = set()
    for dose in d.DOSES:
        for output in source["retained_observations"][dose]:
            control = output["control"]
            baseline = source["retained_baselines"][control]
            profile = result["response_profiles"][dose][control]
            for field, old, new in (
                ("operand_delta", baseline["working_fp16_words"], output["operand"]["working_fp16_words"]),
                ("rmsnorm_output_delta", baseline["rmsnorm_words"], output["rmsnorm_words"]),
            ):
                assert profile[field] == [str(fp16(b) - fp16(a)) for a, b in zip(old, new)]
            assert profile["rmsnorm_scalar_delta"] == {
                k: output["rmsnorm_scalars"][k] - baseline["rmsnorm_scalars"][k]
                for k in ("mean_q48", "root_q24")}
        for field in d.FIELDS[:3]:
            values = [result["response_profiles"][dose][c][field] for c in d.CONTROLS]
            unique[field] &= sum(v == values[7] for v in values) == 1
            if any(v != values[0] for v in values):
                split_fields.add(field)
            for group in result["field_partitions"][field][dose]:
                assert group == [c for c, v in zip(d.CONTROLS, values)
                                 if v == values[d.CONTROLS.index(group[0])]]
    expected = d.UNKNOWN
    for label, fields in ((d.OPERAND, d.FIELDS[:1]), (d.RMSNORM, d.FIELDS[1:3])):
        if split_fields.intersection(fields):
            expected = label if any(unique[f] for f in fields) else d.MIXED
            break
    assert result["classification"] == expected
    assert len(result["missing_bindings"]) == 36
    assert all(p["head_rne_delta"] is None for controls in result["response_profiles"].values()
               for p in controls.values())


def test_independent_selected_rows_and_reference_closure(native):
    assert "report" in native, native.get("integrity_error")
    report = native["retained_family_report"]
    count = 0
    for dose in d.DOSES:
        for row in report["retained_reference_closures"][dose]:
            old = fp16(row["baseline_words"][0]) - fp16(row["baseline_words"][1])
            new = fp16(row["measured_words"][0]) - fp16(row["measured_words"][1])
            fixed = Fraction(row["fixed_reference_margin"])
            assert Fraction(row["observed_delta"]) == new - old
            assert Fraction(row["retained_margin_change"]) == old - fixed
            assert Fraction(row["intervened_margin_change"]) == new - fixed
            count += 1
    assert count == 144
    for observation in native["report"]["observations"]:
        assert observation["selected_row_activity"][0] == (
            observation["control"] == "mapped_all" or observation["dose"] == "17/512")


def test_live_missing_field_and_integrity_are_not_scientific_success(native):
    assert "report" in native, native.get("integrity_error")
    family = {"report": deepcopy(native["retained_family_report"])}
    family.update({key: native[key] for key in d.retained.PRESERVED})
    for key in ("historical_flags", "prior_attempts", "preserved_dyadic_rejection_counts"):
        family[key] = native[key]
    family["classification"] = native["parent_classification"]
    del family["report"]["retained_observations"]["65/2048"][0]["rmsnorm_scalars"]
    with patch.object(d, "authenticate", return_value=(family, native["family_capture_bindings"])):
        result = d.analyze(d.retained.BoundBytes())
    assert result["classification"] == d.UNKNOWN and result["successor"] is None
    assert "rmsnorm_scalars" in result["integrity_error"]


def test_preserved_boundary_flags_and_five_distinct_outcomes(native):
    assert native["normal_host_review"] == "REQUIRED"
    assert native["dose_search_closed"] and native["global_threshold_model_closed"]
    assert native["flags"]["candidate_admitted"] is False
    assert native["flags"]["rounding_causal_attribution"] is False
    assert native["retained_common_component"] == native["counterfactual_common_component"] == "UNKNOWN"
    assert len(d.SUCCESSORS) == 5 and len(set(d.SUCCESSORS.values())) == 5
    assert "wider Q24" in native["claim_boundary"]
    assert "independently propagated global references" in native["claim_boundary"]
