from copy import deepcopy
from fractions import Fraction
import io
import json
from pathlib import Path
import re
import struct
import subprocess
from unittest.mock import patch

import pytest

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_row319_unrounded_operand_provenance_availability_v1 as d


@pytest.fixture(scope="module")
def native(request):
    result = getattr(request.config, "_ace3_operand_availability_result", None)
    return d.analyze(d.SearchBytes()) if result is None else result


def fixture_scan(document, extra=None):
    path = str(d.ROOT / "build/fixture-only-provenance.json")
    raw = d.retained.encoded(document)
    data = {path: raw, **(extra or {})}
    bank = d.SearchBytes(lambda p: data[str(p)])
    bank.read({"path": path, "sha256": d.retained.digest(raw), "bytes": len(raw)})
    expected = {(c, dose): [0x3c00] * 896 for c in d.CONTROLS for dose in ("baseline", *d.DOSES)}
    result = d.base_result()
    states = d.inspect_artifacts(bank, expected, result)
    return states, result, bank


def operand(control="mapped_all", dose="65/2048"):
    return {"control": control, "dose": dose, "polarity": "reverse", "row": 319,
            "working_fp16_words": [0x3c00] * 896, "unrounded_operand_values": ["1"] * 896}


def test_four_distinct_outcome_gates():
    keys = [(c, dose) for c in d.CONTROLS for dose in ("baseline", *d.DOSES)]
    assert d.classify(dict.fromkeys(keys, []), []) == d.ABSENT
    assert d.classify(dict.fromkeys(keys, [{}]), []) == d.COMPLETE
    partial = dict.fromkeys(keys, [])
    partial[keys[0]] = [{}]
    assert d.classify(partial, []) == d.PARTIAL
    assert d.classify(dict.fromkeys(keys, []), [{"error": "bad"}]) == d.PARTIAL
    assert len(set(d.SUCCESSORS.values())) == 4 and d.SUCCESSORS[d.UNKNOWN] is None


def test_complete_inline_fixed_matrix():
    records = [operand(c, dose) for c in d.CONTROLS for dose in ("baseline", *d.DOSES)]
    states, result, _ = fixture_scan(records)
    assert not result["provenance_issues"]
    assert d.classify(states, []) == d.COMPLETE
    table = d.matrix(states)
    assert len(table) == 36
    assert all(row["baseline"]["availability"] == row["operand"]["availability"] == "available"
               for row in table)


def test_partial_payload_not_absence():
    record = operand()
    record["unrounded_operand_values"] = ["1"]
    states, result, _ = fixture_scan(record)
    assert d.classify(states, result["provenance_issues"]) == d.PARTIAL
    assert "incomplete" in result["provenance_issues"][0]["error"]
    row = next(row for row in d.matrix(states, issues=result["provenance_issues"])
               if row["control"] == "mapped_all" and row["dose"] == "65/2048")
    assert row["operand"]["availability"] == "partial/malformed"


def test_word_control_and_dose_identity():
    for field, value in (("working_fp16_words", [0x3c01] * 896),
                         ("control", "invented"), ("dose", "1/1024")):
        record = operand()
        record[field] = value
        states, result, _ = fixture_scan(record)
        assert not any(states.values()) and result["provenance_issues"]


def test_exact_cells_independent_binary16_oracle():
    word = 0x3c00
    oracle = lambda w: Fraction(struct.unpack("<e", struct.pack("<H", w))[0])
    low = (oracle(word - 1) + oracle(word)) / 2
    high = (oracle(word + 1) + oracle(word)) / 2
    binding = {"word": word, "sign_bit": 0, "lower_slack": str(1 - low),
               "upper_slack": str(high - 1), "tie": "interior"}
    assert d.payload_signature("exact_cell_bindings", [binding] * 896, [word] * 896) == (
        d.payload_signature("unrounded_operand_values", ["1"] * 896, [word] * 896))
    for field, value in (("sign_bit", 1), ("lower_slack", "-1"), ("upper_slack", "1"),
                         ("tie", "upper"), ("word", word + 1)):
        bad = {**binding, field: value}
        with pytest.raises(d.IntegrityError):
            d.payload_signature("exact_cell_bindings", [bad] * 896, [word] * 896)


def test_unrounded_ties_noncanonical_and_zero_authority():
    for values, word in ((["1.0"] * 896, 0x3c00), (["nan"] * 896, 0x3c00),
                         (["0"] * 896, 0), (["0"] * 896, 0x8000),
                         (["1"] * 896, 0x3c01)):
        with pytest.raises(ValueError):
            d.payload_signature("unrounded_operand_values", values, [word] * 896)


def test_conflicting_bindings_are_partial():
    a, b = operand(), operand()
    b["unrounded_operand_values"] = ["4097/4096"] * 896
    states, result, _ = fixture_scan([a, b])
    assert states["mapped_all", "65/2048"]
    assert "conflicting" in result["provenance_issues"][0]["error"]


def test_geometry_and_frozen_state_are_not_bindings():
    states, result, _ = fixture_scan({"word_cells": [d.cells.cell(0x3c00)],
                                    "frozen_terminal_vector": ["1"] * 896,
                                    "hidden_i": [16777216] * 896,
                                    "actual_cell_binding": False,
                                    "unrounded_baseline_dose_record_occurrences": 0})
    assert d.classify(states, result["provenance_issues"]) == d.ABSENT


def test_unknown_schema_is_malformed():
    record = operand()
    record["unrounded_other_operand"] = record.pop("unrounded_operand_values")
    states, result, _ = fixture_scan(record)
    assert d.classify(states, result["provenance_issues"]) == d.PARTIAL
    raw = io.BytesIO()
    d.retained.np.savez(raw, s16_unrounded_binary64=d.retained.np.ones(896, dtype="<f8"))
    data = raw.getvalue()
    path = str(d.ROOT / "build/fixture-only-malformed-s16.npz")
    bank = d.SearchBytes(lambda p: data)
    bank.read({"path": path, "sha256": d.retained.digest(data), "bytes": len(data)})
    result = d.base_result()
    d.inspect_artifacts(bank, {}, result)
    assert result["provenance_issues"]
    assert not result["searched_artifacts"][0]["non_target_unrounded_fields"]


def test_linked_existing_manifest_and_missing_pin():
    linked_path = str(d.ROOT / "build/fixture-only-linked.json")
    raw = d.retained.encoded(operand())
    pin = {"path": linked_path, "sha256": d.retained.digest(raw), "bytes": len(raw)}
    states, result, bank = fixture_scan({"operand_provenance_manifest": pin}, {linked_path: raw})
    assert states["mapped_all", "65/2048"] and not result["provenance_issues"]
    assert linked_path in bank.pins
    assert any(f["path"] == linked_path and f["referenced_by"] for f in result["searched_artifacts"])
    with pytest.raises(d.IntegrityError, match="unpinned"):
        fixture_scan({"operand_provenance_manifest": {"path": linked_path}})
    values = d.retained.encoded(["1"] * 896)
    pin = {"path": linked_path, "sha256": d.retained.digest(values), "bytes": len(values)}
    record = operand()
    record["unrounded_operand_values"] = pin
    states, result, bank = fixture_scan(record, {linked_path: values})
    assert states["mapped_all", "65/2048"][0]["payload_pin"] == pin
    assert {f["path"] for f in result["searched_artifacts"]} == set(bank.pins)


def test_hash_size_and_missing_bytes_fail_closed():
    bank = d.SearchBytes(lambda p: b"changed")
    with pytest.raises(d.IntegrityError, match="SHA-256"):
        bank.read(d.SEAL)
    assert bank.attempts[0]["actual"]["bytes"] == 7
    with pytest.raises(d.IntegrityError, match="size"):
        bank.read({"path": str(d.ROOT / "build/size"), "sha256": d.retained.digest(b"changed"), "bytes": 8})
    def missing(path):
        raise FileNotFoundError(str(path))
    result = d.analyze(d.SearchBytes(missing))
    assert result["classification"] == d.UNKNOWN and result["successor"] is None
    assert result["artifact_read_facts"][0]["expected"] == d.REVIEW
    assert all(row["operand"]["availability"] == "unconfirmed" for row in result["availability_matrix"])


def test_independent_review_gate_and_strict_documents():
    review = {"kind": "round_reviewed_handoff", "producer_role": "engineer",
              "mission_id": "c150a7e4c29e", "round": 1, "review": {"status": "done"}}
    with pytest.raises(d.IntegrityError):
        d.retained.review_gate(review, "c150a7e4c29e", 1)
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{}\n{}'):
        with pytest.raises(ValueError):
            d.retained.document(raw)


def test_write_dispatch_and_socket_guards():
    audit = dict.fromkeys(d.AUDIT_KEYS, 0)
    with d.retained.read_only(audit):
        for call in (lambda: Path("/forbidden").write_bytes(b"x"),
                     lambda: subprocess.run(["true"], check=True)):
            with pytest.raises(d.IntegrityError):
                call()
    assert audit["writes"] == 1 and audit["forbidden_calls"] == 2


def test_one_stdout_json_document(native):
    stdout = io.StringIO()
    with patch.object(d, "check", return_value=native), patch("sys.stdout", stdout):
        assert d.main(["--check"]) == 0
    assert json.loads(stdout.getvalue()) == native
    bad = d.unknown(d.base_result(), d.IntegrityError("missing bytes"))
    stdout = io.StringIO()
    with patch.object(d, "check", return_value=bad), patch("sys.stdout", stdout):
        assert d.main(["--check"]) == 1
    assert json.loads(stdout.getvalue())["successor"] is None


def test_native_review_and_fixed_matrix(native):
    assert native["status"] == "measured", native.get("integrity_error")
    assert native["authenticated_review_missions"] == [
        "c150a7e4c29e", "4b02378f0989", "412ec79f743b", "b2e10d85b210", "e762f68f110e"]
    assert native["classification"] == d.ABSENT
    assert native["distinct_operand_records"] == 45 and native["available_operand_records"] == 0
    expected = {(c, dose, 319, "reverse") for c in d.CONTROLS for dose in d.DOSES}
    actual = {(r["control"], r["dose"], r["row"], r["polarity"]) for r in native["availability_matrix"]}
    assert actual == expected and len(native["availability_matrix"]) == 36
    assert not native["provenance_issues"]


def test_native_exact_search_census_and_source_facts(native):
    pins = {p["path"]: p for p in native["authenticated_pins"]}
    searched = {p["path"]: p for p in native["searched_artifacts"]}
    assert pins.keys() == searched.keys()
    assert len(searched) == len(native["searched_artifacts"])
    for path, fact in searched.items():
        assert {k: fact[k] for k in ("path", "sha256", "bytes")} == pins[path]
        if fact["operand_search"]:
            assert fact["referenced_by"], path
            if fact["format"] in ("json", "json-line receipt"):
                raw = Path(path).read_bytes()
                assert not re.search(rb'"(?:unrounded_operand_values|exact_cell_bindings)"\s*:', raw)
                assert not re.search(rb'"[^"]*(?:unrounded|cell_binding)[^"]*"\s*:\s*[\[{]', raw)
        assert not fact["candidate_fields"], (path, fact["candidate_fields"])
    assert any(f["format"] == "npz" and f["array_members"] for f in searched.values())
    assert any(f["format"] == "json-line receipt" for f in searched.values())
    non_target = [(fact, item) for fact in searched.values() for item in fact["non_target_unrounded_fields"]]
    assert len(non_target) == 36
    for fact, item in non_target:
        assert item["name"] == "s16_unrounded_binary64"
        assert item["shape"] == [4864] and item["dtype"] == "<f8"
        assert item["source"] == pins[item["source"]["path"]]
        with d.retained.np.load(io.BytesIO(Path(fact["path"]).read_bytes()), allow_pickle=False) as archive:
            assert archive["s16_unrounded_binary64"].shape == archive["stage16"].shape == (4864,)
            assert archive["s16_unrounded_binary64"].dtype.str == "<f8"
            assert archive["stage16"].dtype.str == "<u2"


def test_native_independent_capture_stream_hashes(native):
    import hashlib
    seal = native["parent_capture_bindings"]
    streams = {}
    for name, binding in seal["artifacts"].items():
        pin = binding["destination"]
        raw = Path(pin["path"]).read_bytes()
        assert len(raw) == pin["size_bytes"]
        assert hashlib.sha256(raw).hexdigest() == pin["sha256"]
        assert binding["source"]["sha256"] == pin["sha256"]
        streams[name] = raw
    assert streams["whole_capture"] == streams["stderr"] + streams["stdout"]
    assert seal["stdout_sha256"] != seal["whole_capture_sha256"]
    parent = json.loads(streams["stdout"])
    for key in (*d.retained.PRESERVED, "historical_flags", "prior_attempts",
                "preserved_dyadic_rejection_counts"):
        assert native[key] == parent[key]


def test_native_boundaries_and_historical_failures(native):
    assert native["dose_search_closed"] and native["global_threshold_model_closed"]
    for name in ("candidate_admitted", "full_model_admitted", "new_token_admitted",
                 "rounding_causal_attribution", "scalar_threshold_transfer_claim",
                 "actual_suffix_threshold_proof", "strict_fp16_state_w4a16", "production_repair"):
        assert native["flags"][name] is False
    assert native["retained_common_component"] == native["counterfactual_common_component"] == "UNKNOWN"
    assert len(native["prior_attempts"]) == 3
    assert all("unavailable" in a["raw_capture"] for a in native["prior_attempts"])


def test_walk_context_and_closed_execution_not_called():
    doc = {"retained_baselines": {"mapped_all": {"working_fp16_words": [0x3c00] * 896,
                                               "unrounded_operand_values": ["1"] * 896}}}
    targets = [(d.cells, "compare"), (d.cells, "analyze"), (d.cells, "check"),
               (d.boundary, "compare"), (d.retained, "compare")]
    from contextlib import ExitStack
    with ExitStack() as stack:
        for owner, name in targets:
            stack.enter_context(patch.object(owner, name, side_effect=AssertionError("closed execution")))
        states, result, _ = fixture_scan(deepcopy(doc))
    assert states["mapped_all", "baseline"] and not result["provenance_issues"]
