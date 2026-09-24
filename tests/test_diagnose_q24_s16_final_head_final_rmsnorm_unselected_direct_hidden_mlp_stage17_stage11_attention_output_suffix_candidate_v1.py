"""Synthetic arrays only; never enter --check or load the real native runtime."""

import builtins
import copy
import hashlib
import importlib
import io
import json
import os
import socket
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage11_attention_output_suffix_candidate_v1 as d
from tests.test_stage11_current_runtime_release import case, clone, production, SEMANTIC_DEFECTS


@pytest.fixture(autouse=True)
def zero_invocations(monkeypatch, record_property):
    counts = {"scientific": 0, "model": 0, "producer": 0, "service": 0}

    def denied(kind):
        def call(*args, **kwargs):
            counts[kind] += 1
            raise AssertionError("forbidden test invocation: " + kind)
        return call

    raw_import = builtins.__import__

    def importing(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "transformers") or (
                name.startswith("ace3.model.") and name not in (
                    d.__name__, "ace3.model.candidates.diagnostic_capture_v1",
                    "ace3.model.stage11_current_runtime_release")):
            return denied("model")()
        return raw_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    monkeypatch.setattr(subprocess, "Popen", denied("producer"))
    monkeypatch.setattr(os, "system", denied("producer"))
    monkeypatch.setattr(socket, "socket", denied("service"))
    monkeypatch.setattr(d, "_load_runtime", denied("model"))
    monkeypatch.setattr(d, "check", denied("producer"))
    monkeypatch.setattr(np, "load", denied("model"))
    prior_profile = sys.getprofile()

    def profile(frame, event, arg):
        if (event == "call" and frame.f_globals.get("__name__") == d.__name__
                and frame.f_code.co_name in ("check", "execution_authorization", "_load_runtime")):
            counts["scientific"] += 1
            raise AssertionError("real execution/authority/runtime path entered")

    sys.setprofile(profile)
    try:
        yield counts
    finally:
        sys.setprofile(prior_profile)
        for kind, count in counts.items():
            record_property("forbidden_" + kind, count)
    assert counts == {"scientific": 0, "model": 0, "producer": 0, "service": 0}


def synthetic_pin(role, data=None):
    if data is None:
        data = ("synthetic-not-authority:" + role).encode()
    return {"path": str(d.ROOT / "build/synthetic_stage11_pin" / role), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def fixture():
    actual = {
        "input_hidden": np.full(896, 0x3c00, dtype="<u2"),
        "input_i": np.full(896, (1 << 24) + 1, dtype="<i8"),
        "input_z": np.zeros(896, dtype="u1"),
        "stage03": np.arange(128, dtype="<u2"),
        "stage05": np.arange(128, dtype="<u2"),
    }
    for stage in range(6, 19):
        actual[f"stage{stage:02d}"] = np.full(896, 0x3c00, dtype="<u2")
    actual["scratch_i"] = np.full(896, (2 << 24) + 1, dtype="<i8")
    actual["scratch_z"] = np.zeros(896, dtype="u1")
    for kind, stage in (("k", "stage05"), ("v", "stage03")):
        actual["input_cache_" + kind] = np.zeros((0, 128), dtype="<u2")
        actual["output_cache_" + kind] = actual[stage][None, :].copy()
    reference = {"stage11": np.arange(896, dtype="<u2")}
    reference["stage11"][0] = 0x8000
    reference["stage13"] = np.full(896, 0x3c00, dtype="<u2")
    reference["stage17"] = np.full(896, 0x4000, dtype="<u2")

    controls = ["synthetic_" + str(i) for i in range(9)]
    contract = {
        "inputs": {**{control: synthetic_pin(control) for control in controls},
                   "replacement_stage11": synthetic_pin("replacement_stage11",
                                                        reference["stage11"].tobytes())},
        "references": {role: synthetic_pin(role) for role in (
            "original_input_L23_fp16", "original_input_L23_binary64",
            "original_input_final_fp16", "original_input_final_binary64")},
        "sources": {role: synthetic_pin(role) for role in (
            "historical_generation", "current_diagnostic", "current_tests")},
        "thresholds": {"synthetic_exact_threshold": "1/1099511627776"},
        "controls": controls,
        "profile": copy.deepcopy(d.PROFILE),
        "gates": dict.fromkeys(d.GATES, "PASS"),
        "source_identity": {"layer": 23, "position": 0, "token_id": 9707},
        "lineage": {"original_input_global": "retained-only-not-reanchored"},
        "historical_failures": [
            {"control": control, "gates": {
                "synthetic_gate": {"status": "FAIL", "failed_elements": 2,
                                   "failed_coordinates": [0, 1]}}}
            for control in controls],
        "launch_constraints": {"account": "external", "role_models": "external",
                               "budget": "external", "access": "external", "concurrency": "external"},
    }
    return actual, reference, contract


@pytest.fixture
def retained_members(monkeypatch):
    """Real byte authentication/NPZ parsing, restricted to synthetic in-memory files."""
    actual, reference, contract = fixture()
    stream = io.BytesIO()
    np.savez(stream, **reference)
    role = "original_input_L23_fp16"
    contract["references"][role] = synthetic_pin(role, stream.getvalue())
    files = {contract["references"][role]["path"]: stream.getvalue(),
             contract["inputs"]["replacement_stage11"]["path"]: reference["stage11"].tobytes()}

    def read_bytes(path):
        if str(path) not in files:
            raise FileNotFoundError("missing synthetic member: " + str(path))
        return files[str(path)]

    def load(stream, *, allow_pickle):
        assert isinstance(stream, io.BytesIO) and allow_pickle is False
        assert stream.getvalue() == files[contract["references"][role]["path"]]
        return np.lib.npyio.NpzFile(stream, allow_pickle=False)

    monkeypatch.setattr(d.Path, "read_bytes", read_bytes)
    monkeypatch.setattr(np, "load", load)
    return actual, reference, contract, files


def test_canonical_member_encoding_preserves_content_not_names():
    _, reference, _ = fixture()
    expected = b"".join(int(word).to_bytes(2, "little") for word in reference["stage11"])
    assert len(expected) == 1792 and expected[:2] == b"\x00\x80"
    assert d.canonical_fp16_member(reference, "stage11") == expected
    strided = np.zeros(1792, dtype="<u2")
    strided[::2] = reference["stage11"]
    assert d.canonical_fp16_member({"stage13": strided[::2]}, "stage13") == expected
    assert d.canonical_fp16_member(reference, "stage13") != expected


def test_bound_replacement_is_consumed_with_nine_unchanged_control_pins(retained_members):
    actual, reference, contract, files = retained_members
    original = copy.deepcopy(contract)
    replacement = d._bound_replacement_stage11(contract)
    assert replacement.tobytes() == files[contract["inputs"]["replacement_stage11"]["path"]]
    reference["stage11"] = replacement
    candidate = d.Candidate(actual, reference, contract)
    assert candidate.prepare()["stage11"].tobytes() == replacement.tobytes()
    identity = candidate.metadata()["diagnostic_identity"]
    assert identity == d._identity_for_contract(contract)
    assert contract == original and identity["inputs"] == original["inputs"]
    assert set(identity["inputs"]) == set(contract["controls"]) | {"replacement_stage11"}
    assert len(contract["controls"]) == 9
    for control in contract["controls"]:
        assert identity["inputs"][control] == synthetic_pin(control)
    account = identity["output_accounts"]["replacement"]
    assert account["input"] == "replacement_stage11" and account["member"] == "stage11"
    assert account["encoding"] == "raw-C-order-little-endian-uint16" and account["bytes"] == 1792


@pytest.mark.parametrize("defect", [
    "missing_pin", "missing_file", "wrong_member", "reencoded", "mutated",
    "stale_pin", "missing_archive_member", "ambiguous_archive_member", "mutated_archive",
])
def test_replacement_member_binding_defects_terminate_without_science(defect, retained_members):
    actual, reference, contract, files = retained_members
    role = "replacement_stage11"
    pin = contract["inputs"][role]
    archive_role = "original_input_L23_fp16"
    archive_pin = contract["references"][archive_role]
    if defect == "missing_pin":
        del contract["inputs"][role]
    elif defect == "missing_file":
        del files[pin["path"]]
    elif defect in ("wrong_member", "reencoded", "mutated", "stale_pin"):
        data = files[pin["path"]]
        if defect == "wrong_member":
            data = reference["stage13"].tobytes()
        elif defect == "reencoded":
            stream = io.BytesIO()
            np.lib.format.write_array(stream, reference["stage11"], version=(1, 0), allow_pickle=False)
            data = stream.getvalue()
        else:
            data = data[:-1] + bytes([data[-1] ^ 1])
        files[pin["path"]] = data
        if defect != "stale_pin":
            contract["inputs"][role] = synthetic_pin(role, data)
        with pytest.raises(ValueError, match="replacement_stage11"):
            if defect == "stale_pin":
                changed_reference = copy.deepcopy(reference)
                changed_reference["stage11"] = np.frombuffer(data, dtype="<u2").copy()
                d.Candidate(actual, changed_reference, contract)
            else:
                d.Candidate(actual, reference, contract)
    elif defect == "mutated_archive":
        data = files[archive_pin["path"]]
        files[archive_pin["path"]] = data[:-1] + bytes([data[-1] ^ 1])
    else:
        stream = io.BytesIO()
        if defect == "missing_archive_member":
            np.savez(stream, stage13=reference["stage13"])
        else:
            stream.write(files[archive_pin["path"]])
            member = io.BytesIO()
            np.lib.format.write_array(member, reference["stage11"], allow_pickle=False)
            with zipfile.ZipFile(stream, "a") as archive:
                archive.writestr("stage11", member.getvalue())
        files[archive_pin["path"]] = stream.getvalue()
        contract["references"][archive_role] = synthetic_pin(archive_role, stream.getvalue())
    with pytest.raises((ValueError, d.UnavailableBinding),
                       match="replacement_stage11|capture member|reference member"):
        d._bound_replacement_stage11(contract)


@pytest.mark.parametrize("defect", ["missing_pin", "missing_control", "extra_input", "control_alias"])
def test_exact_identity_input_census_rejects_missing_or_ambiguous_roles(defect):
    actual, reference, contract = fixture()
    if defect == "missing_pin":
        del contract["inputs"]["replacement_stage11"]
    elif defect == "missing_control":
        del contract["inputs"][contract["controls"][0]]
    elif defect == "extra_input":
        contract["inputs"]["replacement_stage13"] = synthetic_pin("replacement_stage13")
    else:
        contract["controls"][0] = "replacement_stage11"
    for operation in (lambda: d.Candidate(actual, reference, contract),
                      lambda: d._identity_for_contract(contract)):
        with pytest.raises((ValueError, d.UnavailableBinding), match="replacement_stage11"):
            operation()


def test_import_and_exact_vector_copy_are_non_generative(zero_invocations):
    importlib.reload(d)
    actual, reference, contract = fixture()
    original = {k: v.tobytes() for k, v in actual.items()}
    candidate = d.Candidate(actual, reference, contract)
    working = candidate.prepare()
    assert working["stage11"].tobytes() == reference["stage11"].tobytes()
    assert np.count_nonzero(working["stage11"] != actual["stage11"]) == 896
    assert int(working["stage11"][0]) == 0x8000
    assert all(not v.flags.writeable for v in working.values())
    assert {k: v.tobytes() for k, v in actual.items()} == original
    assert working["input_i"][0] == (1 << 24) + 1
    for key in actual:
        assert not np.shares_memory(actual[key], working[key])
        if key != "stage11":
            assert working[key].tobytes() == original[key]
    reference["stage11"][:] = 0
    actual["input_i"][:] = 0
    contract["thresholds"].clear()
    assert candidate.prepare()["stage11"].tobytes() == working["stage11"].tobytes()
    assert candidate.prepare()["input_i"][0] == (1 << 24) + 1
    assert zero_invocations == {"scientific": 0, "model": 0, "producer": 0, "service": 0}


@pytest.mark.parametrize("key", list(fixture()[0]))
def test_every_non_replaced_byte_and_the_full_replacement_are_protected(key):
    candidate = d.Candidate(*fixture())
    working = {k: v.copy() for k, v in candidate.prepare().items()}
    if working[key].size:
        working[key].flat[-1] ^= 1
    else:
        working[key] = np.zeros((1, 128), dtype="<u2")
    with pytest.raises(ValueError, match="protected"):
        candidate.verify(working)


@pytest.mark.parametrize("defect", ["coordinate_only", "extra_output", "missing", "dtype", "shape"])
def test_incomplete_replacement_and_unrequested_outputs_fail(defect):
    actual, reference, contract = fixture()
    candidate = d.Candidate(actual, reference, contract)
    working = candidate.prepare()
    if defect == "coordinate_only":
        working["stage11"] = actual["stage11"].copy()
        working["stage11"][241] = reference["stage11"][241]
    elif defect == "extra_output":
        working["pair_logits"] = np.zeros(2, dtype="<u2")
    elif defect == "missing":
        del working["stage11"]
    elif defect == "dtype":
        working["stage11"] = working["stage11"].astype("<u4")
    else:
        working["stage11"] = working["stage11"][None, :]
    with pytest.raises(ValueError):
        candidate.verify(working)


@pytest.mark.parametrize("defect", ["missing", "dtype", "shape", "infinity", "nan", "kv", "q24"])
def test_input_and_lineage_defects_fail(defect):
    actual, reference, contract = fixture()
    if defect == "missing":
        del reference["stage11"]
    elif defect == "dtype":
        reference["stage11"] = reference["stage11"].astype("<f2")
    elif defect == "shape":
        reference["stage11"] = reference["stage11"][:-1]
    elif defect in ("infinity", "nan"):
        reference["stage11"][0] = 0x7c00 if defect == "infinity" else 0x7e00
    elif defect == "kv":
        actual["output_cache_v"][0, 0] ^= 1
    else:
        actual["input_i"] = actual["input_i"].astype("<f2")
    with pytest.raises((ValueError, d.UnavailableBinding)):
        d.Candidate(actual, reference, contract)


@pytest.mark.parametrize("field", list(fixture()[2]))
def test_every_frozen_binding_is_required_and_cannot_change(field):
    actual, reference, contract = fixture()
    candidate = d.Candidate(actual, reference, contract)
    del contract[field]
    with pytest.raises(d.UnavailableBinding, match=field):
        d.Candidate(actual, reference, contract)
    with pytest.raises(ValueError, match="frozen"):
        candidate.verify(candidate.prepare(), contract=contract)


def test_historical_failures_accept_exact_retained_list_without_changing_it():
    actual, reference, contract = fixture()
    failures = copy.deepcopy(contract["historical_failures"])
    original = {"retained_controls_and_failure_gates": copy.deepcopy(failures),
                "report": {"lineage_separation": copy.deepcopy(contract["lineage"])}}
    before = copy.deepcopy((contract, original))
    d._contract(contract)
    d._retained_failure_lineage(contract, original)
    candidate = d.Candidate(actual, reference, contract)
    candidate.verify(candidate.prepare(), contract=contract)
    assert candidate.metadata()["frozen_contract"]["historical_failures"] == failures
    assert [record["control"] for record in failures] == contract["controls"]
    assert (contract, original) == before
    contract["historical_failures"][0]["gates"]["synthetic_gate"]["status"] = "PASS"
    with pytest.raises(ValueError, match="frozen"):
        candidate.verify(candidate.prepare(), contract=contract)


@pytest.mark.parametrize("failures", [
    {}, {"synthetic_retained_parent": "FAIL"}, None, False, 0, "", "FAIL",
    tuple(fixture()[2]["historical_failures"]),
])
def test_historical_failures_reject_non_list(failures):
    actual, reference, contract = fixture()
    contract["historical_failures"] = failures
    with pytest.raises(ValueError, match="must be a list: historical_failures"):
        d._contract(contract)
    with pytest.raises(ValueError, match="must be a list: historical_failures"):
        d.Candidate(actual, reference, contract)
    original = {"retained_controls_and_failure_gates": fixture()[2]["historical_failures"],
                "report": {"lineage_separation": copy.deepcopy(contract["lineage"])}}
    with pytest.raises(ValueError, match="retained failure/lineage binding changed"):
        d._retained_failure_lineage(contract, original)


@pytest.mark.parametrize("defect", [
    "empty", "missing", "extra", "reordered", "duplicate", "control_name",
    "missing_control", "non_record",
])
def test_historical_failure_census_defects_reject_before_suffix(defect):
    actual, reference, contract = fixture()
    failures = contract["historical_failures"]
    original = {"retained_controls_and_failure_gates": copy.deepcopy(failures),
                "report": {"lineage_separation": copy.deepcopy(contract["lineage"])}}
    if defect == "empty":
        failures.clear()
    elif defect == "missing":
        failures.pop()
    elif defect == "extra":
        failures.append(copy.deepcopy(failures[0]))
    elif defect == "reordered":
        failures[0], failures[1] = failures[1], failures[0]
    elif defect == "duplicate":
        failures[1] = copy.deepcopy(failures[0])
    elif defect == "control_name":
        failures[0]["control"] = "unretained"
    elif defect == "missing_control":
        del failures[0]["control"]
    else:
        failures[0] = failures[0]["control"]
    before = copy.deepcopy((contract, original))
    with pytest.raises(ValueError, match="nine records in retained control order"):
        d._contract(contract)
    with pytest.raises(ValueError, match="nine records in retained control order"):
        d.Candidate(actual, reference, contract)
    with pytest.raises(ValueError, match="retained failure/lineage binding changed"):
        d._retained_failure_lineage(contract, original)
    assert (contract, original) == before


@pytest.mark.parametrize("changed_side", ["declared", "retained"])
@pytest.mark.parametrize("defect", [
    "status", "nested_count", "numeric_type", "nested_boolean", "nested_type",
    "nested_order", "missing_field", "extra_field",
])
def test_changed_or_unretained_failure_content_rejects_before_suffix(defect, changed_side):
    _, _, contract = fixture()
    original = {"retained_controls_and_failure_gates": copy.deepcopy(contract["historical_failures"]),
                "report": {"lineage_separation": copy.deepcopy(contract["lineage"])}}
    failures = (contract["historical_failures"] if changed_side == "declared"
                else original["retained_controls_and_failure_gates"])
    gate = failures[0]["gates"]["synthetic_gate"]
    if defect == "status":
        gate["status"] = "PASS"
    elif defect == "nested_count":
        gate["failed_elements"] = 1
    elif defect == "numeric_type":
        gate["failed_elements"] = 2.0
    elif defect == "nested_boolean":
        gate["failed_coordinates"][0] = False
    elif defect == "nested_type":
        gate["failed_coordinates"] = tuple(gate["failed_coordinates"])
    elif defect == "nested_order":
        gate["failed_coordinates"].reverse()
    elif defect == "missing_field":
        del failures[0]["gates"]
    else:
        failures[0]["unretained"] = "FAIL"
    before = copy.deepcopy((contract, original))
    d._contract(contract)
    with pytest.raises(ValueError, match="retained failure/lineage binding changed"):
        d._retained_failure_lineage(contract, original)
    assert (contract, original) == before


def test_failures_do_not_default_missing_retained_evidence_or_skip_lineage():
    _, _, contract = fixture()
    original = {"report": {"lineage_separation": copy.deepcopy(contract["lineage"])}}
    with pytest.raises(KeyError, match="retained_controls_and_failure_gates"):
        d._retained_failure_lineage(contract, original)
    original["retained_controls_and_failure_gates"] = copy.deepcopy(contract["historical_failures"])
    original["report"]["lineage_separation"] = {"original_input_global": "changed"}
    with pytest.raises(ValueError, match="retained failure/lineage binding changed"):
        d._retained_failure_lineage(contract, original)


@pytest.mark.parametrize("gate", d.GATES)
def test_failed_source_operand_state_kv_lineage_gates_reject(gate):
    actual, reference, contract = fixture()
    contract["gates"][gate] = "FAIL"
    with pytest.raises(ValueError, match="integrity"):
        d.Candidate(actual, reference, contract)


def test_global_reference_threshold_and_source_roles_are_not_reanchored():
    actual, reference, contract = fixture()
    candidate = d.Candidate(actual, reference, contract)
    reference["stage17"][0] ^= 1
    with pytest.raises(ValueError, match="original-input"):
        candidate.verify(candidate.prepare(), original_fp16=reference)
    for field in ("references", "sources", "thresholds", "launch_constraints"):
        changed = copy.deepcopy(contract)
        first = next(iter(changed[field]))
        changed[field][first] = "mutated"
        with pytest.raises(ValueError, match="frozen"):
            candidate.verify(candidate.prepare(), contract=changed)
    for role in contract["sources"]:
        changed = copy.deepcopy(contract)
        del changed["sources"][role]
        with pytest.raises(d.UnavailableBinding, match=role):
            d.Candidate(actual, reference, changed)
    report = candidate.metadata()
    assert report["frozen_contract"] == contract
    assert report["diagnostic_identity"]["sources"] == contract["sources"]


@pytest.mark.parametrize("index", range(9))
def test_each_recipe_step_is_exact_and_non_executable(index):
    candidate = d.Candidate(*fixture())
    recipe = d.suffix_recipe()
    candidate.verify(candidate.prepare(), recipe=recipe)
    recipe[index]["inputs"] = ["local_reference"]
    with pytest.raises(ValueError, match="recipe"):
        candidate.verify(candidate.prepare(), recipe=recipe)


@pytest.mark.parametrize("operation", [*d.FORBIDDEN, *range(12, 19),
                                       "final_rmsnorm", "selected_tied_head"])
def test_all_invocations_block_before_any_callback(operation):
    candidate = d.Candidate(*fixture())
    called = []
    with pytest.raises(RuntimeError, match="forbids invocation"):
        candidate.invoke(operation, lambda: called.append(True))
    assert called == []
    recipe = d.suffix_recipe()
    recipe.append({"operator": operation})
    with pytest.raises(ValueError, match="recipe"):
        candidate.verify(candidate.prepare(), recipe=recipe)
    audit = candidate.metadata()["dispatch_and_write_audit"]
    assert audit["blocked_invocation_attempts"] == 1
    assert all(audit[key] == 0 for key in d.COUNTERS)


def test_complete_identity_accounts_and_termination_without_authority():
    candidate = d.Candidate(*fixture())
    report = candidate.metadata()
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    identity = report["diagnostic_identity"]
    assert set(identity) == {"diagnostic", "observation", "inputs", "references", "sources",
                             "command_semantics", "permitted_invocations", "output_accounts",
                             "termination", "boundary"}
    recipe = identity["command_semantics"]["future_recipe"]
    assert [r["stage"] for r in recipe] == [*range(12, 19), "final_rmsnorm", "selected_tied_head"]
    assert recipe[0]["inputs"] == ["input_i", "input_z", "input_hidden", "stage11"]
    assert recipe[6]["inputs"] == ["scratch_i", "scratch_z", "stage12", "stage17"]
    assert recipe[-1]["rows"] == [34319, 13]
    accounts = identity["output_accounts"]
    assert accounts["replacement"]["coordinates"] == list(range(896))
    assert accounts["directional_rows"] == 18 and accounts["stage_reports"]["count"] == 63
    assert accounts["branches"] == ["fp16", "binary64"]
    assert len(accounts["outputs_per_control"]) == len(set(accounts["outputs_per_control"])) == 14
    assert all(identity["permitted_invocations"][name] == 0 for name in d.FORBIDDEN)
    assert identity["permitted_invocations"]["selected_tied_head_rows"] == 18
    assert report["dispatch_authorized"] is False and report["proposal_issued"] is False
    assert report["scientific_classification"] is None
    assert report["binding_status"] == "REQUIRES_EXTERNAL_AUTHENTICATION_NOT_PERFORMED"
    assert all(value == 0 for value in report["dispatch_and_write_audit"].values())
    assert report["normal_host_review"] == "REQUIRED"
    assert report["lane_terminated"] and identity["termination"]["lane_terminated"]
    assert set(report["engineering_outcomes"]) == {"PASS", "FAIL", "UNKNOWN"}
    assert callable(d.check) and callable(d.main)
    assert not any(hasattr(d, name) for name in ("execute", "classify"))
    report["frozen_contract"]["thresholds"].clear()
    assert candidate.metadata()["frozen_contract"]["thresholds"]


def test_wrong_accounting_and_pair_or_precision_changes_fail():
    actual, reference, contract = fixture()
    candidate = d.Candidate(actual, reference, contract)
    recipe = d.suffix_recipe()
    recipe[-1]["rows"].reverse()
    with pytest.raises(ValueError, match="recipe"):
        candidate.verify(candidate.prepare(), recipe=recipe)
    contract["profile"]["residual_state"] = "FP16"
    with pytest.raises(ValueError, match="profile"):
        d.Candidate(actual, reference, contract)
    candidate._counts["scientific_invocations"] = 1
    with pytest.raises(ValueError, match="count"):
        candidate.metadata()


class SyntheticTensor(np.ndarray):
    def square(self):
        return self * self

    def numpy(self):
        return np.asarray(self)


def synthetic_runtime():
    """Array arithmetic doubles, with the reviewed guard interface and no imports/I/O."""
    calls = []
    decoded = lambda words: words.view("<f2").astype("<f8").view(SyntheticTensor)
    rne = lambda value: np.asarray(value, dtype="<f2").view("<u2")

    def add(state, words):
        total = state["i"] + (decoded(words) * (1 << 24)).astype("<i8")
        zero = ((total == 0) & (state["i"] == 0) & (state["z"] == 1) & (words == 32768)).astype("u1")
        h = rne(total / (1 << 24))
        h[(total == 0) & (zero == 1)] = 32768
        return {"i": total, "z": zero, "h": h}

    def rtz(value):
        rounded = np.asarray(value, dtype="<f2")
        outside = np.abs(rounded.astype("<f8")) > np.abs(value)
        rounded[outside] = np.nextafter(rounded[outside], np.float16(0))
        return rounded.view("<u2")

    native = SimpleNamespace(state=SimpleNamespace(add=add), decoded=decoded, rne=rne, toward_zero=rtz,
        torch=SimpleNamespace(from_numpy=lambda a: a.view(SyntheticTensor), rsqrt=lambda a: 1 / np.sqrt(a)),
        functional=SimpleNamespace(silu=lambda a: a / (1 + np.exp(-a))))
    native.projection = lambda tensors, name, words: rne(decoded(words) * tensors[name][0])
    parent = SimpleNamespace(
        rmsnorm=lambda words, weights: (rne(decoded(words) * weights), {"synthetic": True}),
        logits=lambda words, weights: rne(weights.astype("<f8") @ decoded(words)))
    layer = SimpleNamespace(
        expected_stage=lambda stage, arrays, state, tensors: arrays[f"stage{stage:02d}"].copy(),
        stage_report=lambda stage, arrays, trajectory, binary64, expected: {
            "stage": stage, "status": "PASS", "residual_state_lineage": "PASS", "kv_lineage": "PASS"})

    @contextmanager
    def guards(audit, active, tensors, operands):
        raw_projection, raw_expected = native.projection, layer.expected_stage
        raw_norm, raw_head = parent.rmsnorm, parent.logits

        def projection(actual_tensors, name, words):
            stage = active["stage"]
            kind, source = {14: ("gate", 13), 15: ("up", 13), 17: ("down", 16)}[stage]
            assert actual_tensors is tensors and name == "model.layers.23.mlp." + kind + "_proj"
            assert words is active["arrays"][f"stage{source:02d}"]
            audit["projections"] += 1
            return raw_projection(actual_tensors, name, words)

        def expected(stage, arrays, state, actual_tensors):
            assert stage == active["stage"] and arrays is active["arrays"]
            assert state is active["state"] and actual_tensors is tensors
            calls.append(stage)
            audit["local_oracles"] += 1
            return raw_expected(stage, arrays, state, actual_tensors)

        def norm(words, weights):
            assert active["stage"] == 19 and weights is operands[0]
            assert words is active["arrays"]["stage18"]
            calls.append(19)
            audit["final_norms"] += 1
            return raw_norm(words, weights)

        def head(words, weights):
            assert active["stage"] == 20 and weights is operands[1] and weights.shape == (2, 896)
            assert words is active["arrays"]["final_rmsnorm"]
            calls.append(20)
            audit["pair_heads"] += 1
            return raw_head(words, weights)

        with d.patch.object(native, "projection", projection), d.patch.object(layer, "expected_stage", expected), \
                d.patch.object(parent, "rmsnorm", norm), d.patch.object(parent, "logits", head):
            yield

    return SimpleNamespace(native=native, layer=layer, parent=parent,
                           guards=SimpleNamespace(suffix_only=guards), calls=calls)


def executor_fixture():
    actual, reference, contract = fixture()
    reference["stage11"] = np.full(896, 0x3800, dtype="<u2")
    contract["inputs"]["replacement_stage11"] = synthetic_pin(
        "replacement_stage11", reference["stage11"].tobytes())
    runtime = synthetic_runtime()
    tensors = {"model.layers.23.post_attention_layernorm.weight": np.ones(896, dtype="<f2"),
               **{"model.layers.23.mlp." + name + "_proj": np.array([0.5], dtype="<f2")
                  for name in ("gate", "up", "down")}}
    operands = (np.ones(896, dtype="<f2"),
                np.stack([np.full(896, 0.25, dtype="<f2"), np.full(896, 0.125, dtype="<f2")]))
    return actual, reference, contract, runtime, tensors, operands


def synthetic_final_oracle(scratch, arrays, operands, account):
    expected_i = scratch["scratch_i"] + (
        arrays["stage17"].view("<f2").astype("<f8") * (1 << 24)).astype("<i8")
    assert np.array_equal(arrays["output_i"], expected_i)
    assert np.array_equal(arrays["stage18"], (expected_i / (1 << 24)).astype("<f2").view("<u2"))
    assert account == {"synthetic": True}
    expected = [np.float16(sum(float(a) * float(b) for a, b in
                              zip(arrays["final_rmsnorm"].view("<f2"), row, strict=True)))
                for row in operands[1]]
    assert np.array_equal(arrays["pair_logits"], np.asarray(expected, dtype="<f2").view("<u2"))


def run_synthetic(parts, oracle=synthetic_final_oracle):
    actual, reference, contract, runtime, tensors, operands = parts
    candidate = d.Candidate(actual, reference, contract)
    result = candidate.run_suffix(runtime, tensors, operands, reference,
                                  np.zeros(896, dtype="<f8"), contract, oracle)
    return candidate, result


def test_dormant_executor_arithmetic_order_and_output_census(zero_invocations):
    parts = executor_fixture()
    actual, reference, contract, runtime, tensors, operands = parts
    before = d._snapshot(actual)
    candidate, (outputs, reports, audit, _) = run_synthetic(parts)
    assert runtime.calls == list(range(12, 21))
    assert audit == d.SUFFIX_COUNTS
    assert np.all(outputs["scratch_i"] == (3 << 23) + 1)
    assert np.all(outputs["stage12"] == 0x3e00)
    v = np.full(896, 1.5, dtype="<f8")
    s13 = (v / np.sqrt(np.mean(v * v) + 1e-6)).astype("<f2")
    assert np.array_equal(outputs["stage13"], s13.view("<u2"))
    gate = (s13.astype("<f8") * 0.5).astype("<f2")
    product = (gate.astype("<f8") / (1 + np.exp(-gate.astype("<f8")))) * gate.astype("<f8")
    assert np.array_equal(outputs["s16_unrounded_binary64"], product)
    assert np.all(np.abs(outputs["stage16"].view("<f2").astype("<f8")) <= np.abs(product))
    assert d._snapshot(actual) == before and np.all(reference["stage11"] == 0x3800)
    assert all(candidate.metadata()["dispatch_and_write_audit"][key] == 0 for key in d.COUNTERS)
    all_outputs = {c: outputs for c in contract["controls"]}
    all_reports = {c: reports for c in contract["controls"]}
    rows = [{"control": c, "branch": b} for c in contract["controls"] for b in d.BRANCHES]
    d.output_census(contract["controls"], all_outputs, all_reports, rows)
    with pytest.raises(ValueError, match="consumed"):
        candidate.run_suffix(runtime, tensors, operands, reference, np.zeros(896, dtype="<f8"),
                             contract, synthetic_final_oracle)
    assert zero_invocations == {"scientific": 0, "model": 0, "producer": 0, "service": 0}


@pytest.mark.parametrize("stage", [0, 11, 19, 20])
def test_new_shared_guard_restrictions_fail_before_arithmetic(stage):
    parts = executor_fixture()
    actual, reference, contract, runtime, tensors, operands = parts
    arrays = d.Candidate(actual, reference, contract).prepare()
    active = {"stage": stage, "arrays": arrays,
              "state": {"i": arrays["input_i"], "z": arrays["input_z"], "h": arrays["input_hidden"]}}
    audit = dict.fromkeys(d.SUFFIX_COUNTS, 0)
    with d.suffix_only(runtime, audit, active, tensors, operands):
        with pytest.raises(ValueError, match="outside"):
            runtime.native.projection(tensors, "model.layers.23.self_attn.o_proj", arrays["stage10"])
        with pytest.raises(ValueError, match="outside"):
            runtime.native.state.add(active["state"], arrays["stage11"])
        with pytest.raises(ValueError, match="outside"):
            runtime.layer.expected_stage(stage, arrays, active["state"], tensors)
    assert audit == dict.fromkeys(d.SUFFIX_COUNTS, 0)


def test_q24_guard_rejects_substituted_state_or_operand():
    actual, reference, contract, runtime, tensors, operands = executor_fixture()
    arrays = d.Candidate(actual, reference, contract).prepare()
    state = {"i": arrays["input_i"], "z": arrays["input_z"], "h": arrays["input_hidden"]}
    active = {"stage": 12, "arrays": arrays, "state": state}
    audit = dict.fromkeys(d.SUFFIX_COUNTS, 0)
    with d.suffix_only(runtime, audit, active, tensors, operands):
        with pytest.raises(ValueError, match="scope"):
            runtime.native.state.add(dict(state), arrays["stage11"])
        with pytest.raises(ValueError, match="scope"):
            runtime.native.state.add(state, arrays["stage11"].copy())
    assert audit == dict.fromkeys(d.SUFFIX_COUNTS, 0)


@pytest.mark.parametrize("defect", ["source", "operand", "state", "kv", "lineage", "reference", "threshold"])
def test_executor_detects_mid_suffix_mutations(defect):
    parts = executor_fixture()
    actual, reference, contract, runtime, tensors, operands = parts
    raw = runtime.layer.expected_stage

    def corrupt(stage, arrays, state, supplied):
        if defect == "source":
            tensors["model.layers.23.mlp.up_proj"][0] = 0.25
        elif defect == "operand":
            operands[1][0, 0] = 0
        elif defect == "state":
            arrays["input_i"] = arrays["input_i"].copy() + 1
        elif defect == "kv":
            arrays["output_cache_v"] = arrays["output_cache_v"].copy() + 1
        elif defect == "reference":
            reference["stage11"][0] ^= 1
        else:
            contract["lineage" if defect == "lineage" else "thresholds"]["changed"] = True
        return raw(stage, arrays, state, supplied)

    runtime.layer.expected_stage = corrupt
    with pytest.raises(ValueError):
        run_synthetic(parts)
    assert runtime.calls == [12]


@pytest.mark.parametrize("defect", ["rows", "norm", "dtype", "nan"])
def test_executor_rejects_final_operand_expansion_before_dispatch(defect):
    parts = list(executor_fixture())
    norm, head = parts[-1]
    if defect == "rows":
        head = np.zeros((3, 896), dtype="<f2")
    elif defect == "norm":
        norm = norm[:-1]
    elif defect == "dtype":
        head = head.astype("<f4")
    else:
        head[0, 0] = np.nan
    parts[-1] = norm, head
    with pytest.raises(ValueError, match="restriction"):
        run_synthetic(parts)
    assert parts[3].calls == []


@pytest.mark.parametrize("operation", [0, 11, 21, *d.FORBIDDEN])
def test_compute_rejects_forbidden_stage_without_runtime_dispatch(operation):
    with pytest.raises(RuntimeError, match="forbidden suffix stage"):
        d.compute_step(SimpleNamespace(native=None, parent=None), {}, (),
                       {"stage": operation, "arrays": {}})


@pytest.mark.parametrize("defect", ["control", "output", "stage", "branch"])
def test_executor_output_census_rejects_omissions(defect):
    parts = executor_fixture()
    _, (outputs, reports, _, _) = run_synthetic(parts)
    controls = parts[2]["controls"]
    all_outputs = {c: dict(outputs) for c in controls}
    all_reports = {c: list(reports) for c in controls}
    rows = [{"control": c, "branch": b} for c in controls for b in d.BRANCHES]
    if defect == "control":
        all_outputs.pop(controls[-1])
    elif defect == "output":
        all_outputs[controls[0]].pop("stage12")
    elif defect == "stage":
        all_reports[controls[0]].reverse()
    else:
        rows.pop()
    with pytest.raises(ValueError, match="census"):
        d.output_census(controls, all_outputs, all_reports, rows)


@pytest.fixture
def authorization_fixture(case, monkeypatch):
    """Exact emitter serialization; only the future claim/review are virtual."""
    release = d.release
    output, documents, unchanged = release.validate_after_claim(
        case["put"](case["request_path"], case["request"]))
    for name, raw in zip(release.OUTPUTS, documents):
        case["virtual"][str(output / name)] = raw
    unchanged()
    declaration = json.loads(documents[1])
    issuance = json.loads(documents[2])
    envelope = json.loads(documents[3])
    launch = declaration["launch"]
    monkeypatch.setattr(d, "os", SimpleNamespace(
        environ=launch["environment"], getuid=lambda: launch["uid"]))
    monkeypatch.setattr(d, "sys", SimpleNamespace(executable=launch["argv"][0]))
    monkeypatch.setattr(d.capture, "retained_document", release.document)
    native = d._native_running_claim
    backlog_path = case["native"] / "backlog.jsonl"
    monkeypatch.setattr(d, "_native_running_claim",
                        lambda path, mission: native(backlog_path, mission))

    def rewrite(pin, value):
        raw = value if isinstance(value, bytes) else release.json_bytes(value)
        if pin["path"] == str(backlog_path):
            backlog_path.write_bytes(raw)
        else:
            case["virtual"][pin["path"]] = raw
        return release.pin_bytes(pin["path"], raw)

    preflight = {"execution_authorization": envelope, "mission_id": launch["mission_id"],
                 "sources": [release.pin(d.SOURCE), release.pin(d.TEST)],
                 "environment": launch["environment"],
                 "capture_implementation": d.capture.implementation_pins(),
                 "launch_constraints": launch["constraints"], "role": "engineer",
                 "independent_host_review": "REQUIRED", "model_or_service_calls_authorized": 0,
                 "command_budget": {"compile": 1, "pytest": 1, "check": 1},
                 "normal_running_claim": case["request"]["native_record"],
                 "native_backlog": str(release.LIFE / "backlog.jsonl")}

    def rebind():
        envelope["declaration"] = rewrite(envelope["declaration"], declaration)
        issuance["declaration"] = envelope["declaration"]
        envelope["issuance"] = rewrite(envelope["issuance"], issuance)

    return SimpleNamespace(preflight=preflight, declaration=declaration, issuance=issuance,
                           envelope=envelope, backlog=release.pin(backlog_path),
                           rewrite=rewrite, rebind=rebind, case=case)


def test_running_claim_accepts_actual_emitter_declaration_and_retained_lineage(
        authorization_fixture, zero_invocations):
    f = authorization_fixture
    assert d.validate_launch_claim(f.preflight) == f.envelope
    assert d.validate_execution_authorization(f.preflight) == f.declaration
    assert f.envelope["review"]["mission_id"] != f.preflight["mission_id"]
    assert all(value == 0 for value in zero_invocations.values())


@pytest.mark.parametrize("defect", [
    "same_mission_review", "stale_ed8_only", "wrong_mission", "missing_claim",
    "missing_live_claim", "retried_claim", "terminal_claim", "wrong_claim_mission",
    "claim_before_start", "authority_before_claim", "claim_selector", "claim_kind",
    "changed_source", "changed_contract_pin", "changed_launch_pin", "changed_claim_pin",
    "changed_issuance_pin", "changed_snapshot", "changed_authority_bundle",
    "changed_argv", "changed_environment", "changed_constraints",
    "stale_review_pin", "stale_latest_round", "missing_review_links",
    "engineer_declaration", "engineer_issuance", "manager_review", "engineer_review",
    "nonterminal_review", "wrong_evidence_mission", "wrong_origin_review_mission",
    "issuance_wrong_mission", "issuance_wrong_claim",
])
def test_launch_authorization_rejects_invalid_bindings(authorization_fixture, defect):
    f = authorization_fixture
    declaration, envelope = f.declaration, f.envelope
    if defect == "same_mission_review":
        envelope["review"]["mission_id"] = f.preflight["mission_id"]
        declaration["accepted_preflight_mission"] = f.preflight["mission_id"]
    elif defect == "stale_ed8_only":
        del envelope["issuance"]
    elif defect == "wrong_mission":
        f.preflight["mission_id"] = "wrong-mission"
    elif defect == "missing_claim":
        del f.case["virtual"][declaration["claim"]["pin"]["path"]]
    elif defect in ("missing_live_claim", "retried_claim", "terminal_claim"):
        row = {"id": f.preflight["mission_id"], "status": "running", "started_ts": 1700000000.0}
        if defect == "missing_live_claim":
            row["id"] = "another-mission"
        elif defect == "retried_claim":
            row["started_ts"] += 100
        else:
            row["status"] = "done"
        f.rewrite(f.backlog, (json.dumps(row) + "\n").encode())
    elif defect in ("wrong_claim_mission", "claim_before_start", "claim_kind"):
        claim = d.capture.retained_document(declaration["claim"]["pin"])
        claim.update({"wrong_claim_mission": {"mission_id": "wrong-mission"},
                      "claim_before_start": {"started_ts": 1800000000.0},
                      "claim_kind": {"kind": "engineer_claim"}}[defect])
        declaration["claim"]["pin"] = f.rewrite(declaration["claim"]["pin"], claim)
        f.issuance["claim"] = declaration["claim"]["pin"]
    elif defect == "authority_before_claim":
        declaration["issued_at_utc"] = "2023-11-14T22:13:20+00:00"
    elif defect == "claim_selector":
        declaration["claim"]["mission_path"] = ["decoy", "mission_id"]
    elif defect == "changed_source":
        f.rewrite(declaration["contract"]["sources"]["current_diagnostic"], b"changed source\n")
    elif defect in ("changed_contract_pin", "changed_launch_pin"):
        changed = copy.deepcopy(declaration)
        if defect == "changed_contract_pin":
            changed["contract"]["thresholds"] = {"changed": "1"}
        else:
            changed["launch"]["argv"].append("--changed")
        f.rewrite(envelope["declaration"], changed)
    elif defect in ("changed_claim_pin", "changed_issuance_pin"):
        pin = declaration["claim"]["pin"] if defect == "changed_claim_pin" else envelope["issuance"]
        f.rewrite(pin, b"changed pinned bytes\n")
    elif defect == "changed_snapshot":
        declaration["source_snapshots"]["historical_generation"] = declaration["source_snapshots"]["current_tests"]
    elif defect == "changed_authority_bundle":
        declaration["accepted_identity"]["authority_bundle"] = declaration["accepted_proposal"]
    elif defect == "changed_argv":
        declaration["launch"]["argv"].append("--changed")
    elif defect == "changed_environment":
        declaration["launch"]["environment"] = {"CHANGED": "1"}
    elif defect == "changed_constraints":
        declaration["launch"]["constraints"] = {"account": "changed"}
    elif defect == "stale_review_pin":
        f.rewrite(envelope["review"]["review"], b"{}\n")
    elif defect == "stale_latest_round":
        envelope["review"]["latest"] = f.rewrite(envelope["review"]["latest"], {
            "kind": "handoff_ref", "handoff": {"path": "different-round.json"}})
    elif defect == "missing_review_links":
        envelope["review"]["checkpoint"] = f.rewrite(envelope["review"]["checkpoint"], b"unlinked\n")
    elif defect in ("engineer_declaration", "engineer_issuance"):
        (declaration if defect == "engineer_declaration" else f.issuance)["issuer"] = "engineer"
    elif defect in ("manager_review", "engineer_review", "nonterminal_review"):
        review = d.capture.retained_document(envelope["review"]["review"])
        if defect == "nonterminal_review":
            review["review"]["status"] = "continue"
        else:
            review["producer_role"] = defect.removesuffix("_review")
        envelope["review"]["review"] = f.rewrite(envelope["review"]["review"], review)
    elif defect == "wrong_evidence_mission":
        declaration["accepted_preflight_mission"] = "wrong-evidence"
    elif defect == "wrong_origin_review_mission":
        pin = declaration["accepted_preflight_terminal_receipt"]
        terminal = d.capture.retained_document(pin)
        terminal["manager_authority"]["review"]["mission_id"] = f.preflight["mission_id"]
        f.rewrite(pin, terminal)
    elif defect == "issuance_wrong_mission":
        f.issuance["mission_id"] = "wrong-mission"
    elif defect == "issuance_wrong_claim":
        f.issuance["claim"] = declaration["accepted_proposal"]
    else:
        raise AssertionError(defect)
    if defect not in ("stale_ed8_only", "changed_contract_pin", "changed_launch_pin",
                      "changed_issuance_pin"):
        f.rebind()
    with pytest.raises((RuntimeError, ValueError, KeyError, FileNotFoundError)):
        d.validate_execution_authorization(f.preflight)


@pytest.mark.parametrize("owner", ["declaration", "issuance"])
@pytest.mark.parametrize("budget", [None, 0, 2, True, "1", 1.0])
def test_launch_requires_exact_one_shot_budget(authorization_fixture, owner, budget):
    f = authorization_fixture
    document = getattr(f, owner)
    if budget is None:
        del document["scientific_run_budget"]
    else:
        document["scientific_run_budget"] = budget
    f.rebind()
    with pytest.raises((ValueError, KeyError)):
        d.validate_execution_authorization(f.preflight)


@pytest.mark.parametrize("member", [
    "authority_bundle", "launcher_capture", "launcher_identity",
    "launcher_expected_identity", "terminal_receipt", "proposal", "receipt_index",
])
@pytest.mark.parametrize("defect", ["missing", "altered", "cross_spliced"])
def test_candidate_lineage_member_rejections(authorization_fixture, member, defect):
    f = authorization_fixture
    run = f.case["request"]["accepted_identity"]
    if defect == "missing":
        del run[member]
    elif defect == "altered":
        f.rewrite(run[member], b"{}\n")
    else:
        run[member] = run["proposal" if member != "proposal" else "terminal_receipt"]
    f.issuance["manager_request"] = f.case["put"](f.case["request_path"], f.case["request"])
    f.rebind()
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        d.validate_execution_authorization(f.preflight)


def test_candidate_obsolete_authorization_schema_rejected(authorization_fixture):
    f = authorization_fixture
    f.declaration["accepted_identity"] = {
        key: {} for key in ("directory", "receipt", "preflight", "commands",
                           "label", "decision_path", "proposal_path", "payload_source")
    }
    f.rebind()
    with pytest.raises((ValueError, KeyError)):
        d.validate_execution_authorization(f.preflight)


@pytest.mark.parametrize("relative,keys,value", SEMANTIC_DEFECTS)
def test_candidate_shared_lineage_rejects_repinned_semantics(
        authorization_fixture, clone, relative, keys, value):
    f = authorization_fixture

    def edit(documents):
        node = documents[relative]
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = (copy.deepcopy(documents["launcher.capture.json"]["files"][0])
                          if value == "splice" else value)

    run, contract = clone(edit)
    for key in ("mission_id", "created_at", "runtime", "implementation_sources"):
        contract[key] = f.case["contract"][key]
    f.case["rebind_contract"](contract)
    f.issuance["manager_request"] = f.case["put"](f.case["request_path"], f.case["request"])
    f.rebind()
    with pytest.raises(ValueError):
        d.validate_execution_authorization(f.preflight)
