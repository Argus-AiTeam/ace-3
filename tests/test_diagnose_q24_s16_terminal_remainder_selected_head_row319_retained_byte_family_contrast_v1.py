"""Independent retained-byte and integrity oracles; no parent diagnostic execution."""

from copy import deepcopy
from fractions import Fraction
import io
import json
from pathlib import Path
import struct
import subprocess
from unittest.mock import patch

import pytest

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_row319_retained_byte_family_contrast_v1 as d


def profiles():
    return {dose: {c: {f: [0] for f in d.FIELDS} for c in d.CONTROLS} for dose in d.DOSES}


@pytest.fixture(scope="module")
def native(request):
    retained = getattr(request.config, "_ace3_retained_byte_result", None)
    if retained is not None:
        return retained
    return d.analyze(d.BoundBytes())


def test_independent_fp16_word_oracle():
    for word in (0, 32768, 1, 1023, 1024, 15360, 18461, 18462, 18463, 18490, 31743, 64511):
        assert d.decode(word) == Fraction(struct.unpack("<e", struct.pack("<H", word))[0])
    for word in (True, -1, 65536, 31744, 32256):
        with pytest.raises(d.IntegrityError):
            d.decode(word)


def test_unique_requires_same_upstream_field_at_all_four_doses():
    p = profiles()
    for dose in d.DOSES:
        p[dose]["mapped_all"]["working_fp16_words"] = [1]
    result = d.partitions(p)
    assert result["classification"] == d.UNIQUE
    assert result["stable_unique_fields"] == ["working_fp16_words"]
    p["17/512"]["mapped_all"]["working_fp16_words"] = [0]
    assert d.partitions(p)["classification"] == d.FAMILY


def test_family_and_inconsistent_partitions():
    p = profiles()
    for dose in d.DOSES:
        for c in ("mapped_all", "mapped62"):
            p[dose][c]["rmsnorm_words"] = [3]
    result = d.partitions(p)
    assert result["classification"] == d.FAMILY
    assert ["mapped62", "mapped_all"] in result["field_partitions"]["65/2048"]["rmsnorm_words"]
    for n, dose in enumerate(d.DOSES):
        p[dose]["mapped_all"][d.FIELDS[n]] = [n + 7]
    assert d.partitions(p)["classification"] == d.FAMILY


def test_no_discriminator_selects_no_scientific_successor():
    result = d.partitions(profiles())
    assert result["classification"] == d.UNKNOWN and result["successor"] is None
    assert result["integrity_error"]
    assert d.SUCCESSORS[d.UNIQUE] != d.SUCCESSORS[d.FAMILY]


def test_missing_profile_bindings_are_not_empty_families():
    for mutate in (lambda p: p["65/2048"].pop("mapped62"),
                   lambda p: p["17/512"]["mapped_all"].pop("rmsnorm_scalars"),
                   lambda p: p["33/1024"]["mapped_all"].update(rmsnorm_words=None)):
        p = profiles()
        mutate(p)
        with pytest.raises((d.IntegrityError, KeyError)):
            d.partitions(p)


def test_selected_logits_cannot_create_discriminator():
    p = profiles()
    for dose in d.DOSES:
        p[dose]["mapped_all"]["selected_logit_words"] = [18463, 18490]
    assert d.partitions(p)["classification"] == d.UNKNOWN
    assert "selected_logit_words" not in d.FIELDS


def test_pin_hash_size_and_missing_bytes():
    raw = b"retained bytes\n"
    good = {"path": "/retained", "sha256": d.digest(raw), "bytes": len(raw)}
    bank = d.BoundBytes(lambda path: raw)
    assert bank.read(good) == raw
    for pin in ({**good, "sha256": "0" * 64}, {**good, "bytes": len(raw) + 1}):
        with pytest.raises(d.IntegrityError):
            bank.read(pin)
    def missing(path):
        raise FileNotFoundError(str(path))
    result = d.analyze(d.BoundBytes(missing))
    assert result["classification"] == d.UNKNOWN and result["successor"] is None
    assert "FileNotFoundError" in result["integrity_error"]
    assert result["dose_search_closed"] and result["global_threshold_model_closed"]


def test_separate_stdout_stderr_whole_capture_bindings():
    raw = {"stdout": b'{"ok":true}\n', "stderr": b"tests passed\n"}
    raw["whole_capture"] = raw["stderr"] + raw["stdout"]
    seal = {"capture_complete": True, "timed_out": False, "returncode": 0,
            "original_temporary_directory": "/original", "sealed_build_directory": "/sealed",
            "stdout_sha256": d.digest(raw["stdout"]),
            "whole_capture_sha256": d.digest(raw["whole_capture"]), "artifacts": {}}
    files = {}
    for name, data in raw.items():
        source = {"path": "/original/" + name, "sha256": d.digest(data), "size_bytes": len(data)}
        destination = {**source, "path": "/sealed/" + name}
        seal["artifacts"][name] = {"source": source, "destination": destination}
        files[destination["path"]] = data
    bank = d.BoundBytes(lambda path: files[str(path)])
    assert d.capture_bytes(seal, bank) == raw["stdout"]
    for mutate in (
        lambda s: s.update(stdout_sha256=s["whole_capture_sha256"]),
        lambda s: s["artifacts"]["stderr"]["source"].update(size_bytes=0),
        lambda s: s["artifacts"]["stdout"]["source"].update(path="/other/stdout"),
        lambda s: s.update(capture_complete=False),
    ):
        changed = deepcopy(seal)
        mutate(changed)
        with pytest.raises(d.IntegrityError):
            d.capture_bytes(changed, bank)


def test_json_census_duplicates_and_nonfinite():
    assert d.document(b'{"binding":1}\n') == {"binding": 1}
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{}\n{}'):
        with pytest.raises(ValueError):
            d.document(raw)


def test_review_authority_not_self_report():
    review = {"kind": "round_reviewed_handoff", "producer_role": "reviewer",
              "mission_id": "b2e10d85b210", "round": 2, "review": {"status": "done"}}
    d.review_gate(review, "b2e10d85b210", 2)
    for changed in ({**review, "producer_role": "engineer"},
                    {**review, "review": {"status": "blocked"}},
                    {**review, "mission_id": "other"}):
        with pytest.raises(d.IntegrityError):
            d.review_gate(changed, "b2e10d85b210", 2)


def test_write_and_external_dispatch_guards():
    audit = dict.fromkeys(d.AUDIT_KEYS, 0)
    with d.read_only(audit):
        for operation in (
            lambda: Path(d.ROOT / "build/diagnostics/row319-retained-byte-family-contrast/forbidden").write_bytes(b"x"),
            lambda: subprocess.run(["true"], check=True),
        ):
            with pytest.raises(d.IntegrityError):
                operation()
    assert audit["writes"] == 1 and audit["forbidden_calls"] == 2


def test_missing_required_report_binding_unknown(native):
    if "report" in native:
        bank = d.BoundBytes()
        q, p, _ = d.authenticate(bank)
        q = deepcopy(q)
        del q["report"]["suffix_outputs"]["65/2048"][0]["rmsnorm_scalars"]
        with patch.object(d, "authenticate", return_value=(q, p, native["b2_capture_bindings"])):
            result = d.analyze(bank)
        assert result["classification"] == d.UNKNOWN and result["successor"] is None
        assert "rmsnorm_scalars" in result["integrity_error"]
    else:
        assert native["classification"] == d.UNKNOWN and native["integrity_error"]


def test_independent_retained_group_and_reference_oracles(native):
    if "report" not in native:
        assert native["status"] == "UNKNOWN" and native["successor"] is None
        return
    report = native["report"]
    assert (report["observations"], report["mapped_all_control_comparisons"],
            report["exact_reference_closures"]) == (36, 32, 144)
    unique = set(d.FIELDS)
    for dose in d.DOSES:
        outputs = report["retained_observations"][dose]
        assert [o["control"] for o in outputs] == list(d.CONTROLS)
        for field in d.FIELDS:
            values = [(o["operand"][field] if field in ("working_fp16_words", "changed_coordinates") else o[field])
                      for o in outputs]
            mapped = values[d.CONTROLS.index("mapped_all")]
            if sum(v == mapped for v in values) != 1:
                unique.discard(field)
            for group in report["field_partitions"][dose][field]:
                first = values[d.CONTROLS.index(group[0])]
                assert group == [c for c, value in zip(d.CONTROLS, values) if value == first]
        for row in report["retained_reference_closures"][dose]:
            def margin(ws):
                return Fraction(struct.unpack("<e", struct.pack("<H", ws[0]))[0]) - Fraction(
                    struct.unpack("<e", struct.pack("<H", ws[1]))[0])
            assert margin(row["measured_words"]) - margin(row["baseline_words"]) == Fraction(row["observed_delta"])
            assert Fraction(row["measured_margin"]) - Fraction(row["fixed_reference_margin"]) == Fraction(row["intervened_margin_change"])
    assert set(report["stable_unique_fields"]) == unique
    if unique:
        assert report["classification"] == d.UNIQUE


def test_single_stdout_document_and_closed_non_admission(native):
    output = io.StringIO()
    with patch.object(d, "check", return_value=native), patch("sys.stdout", output):
        assert d.main(["--check"]) == 0
    result = json.loads(output.getvalue())
    assert result["dose_search_closed"] is True and result["global_threshold_model_closed"] is True
    assert result["normal_host_review"] == "REQUIRED"
    assert d.PREREGISTRATION["successors"][d.UNKNOWN] is None
    assert "wider Q24" in result["claim_boundary"]
    assert "independently propagated global references" in result["claim_boundary"]
