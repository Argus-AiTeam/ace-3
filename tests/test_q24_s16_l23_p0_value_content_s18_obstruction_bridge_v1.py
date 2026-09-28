"""No-science regression coverage using the actual reviewed retained envelope."""

import copy
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ace3.model.candidates import q24_s16_l23_p0_value_content_s18_obstruction_bridge_v1 as bridge


@pytest.fixture(scope="module")
def retained():
    with bridge.readonly_guard() as audit:
        data = bridge.load_inputs()
    assert audit["forbidden_events"] == []
    return data


def parse(retained, envelope=None, failure=None, compiled=None, command=None):
    _, old, cmd, tests, stdout, _ = retained
    return bridge.parse_envelope(
        stdout if envelope is None else json.dumps(envelope),
        old if failure is None else failure,
        tests if compiled is None else compiled,
        cmd if command is None else command,
    )


def test_actual_retained_envelope(retained):
    envelope = parse(retained)
    assert envelope["status"] == "UNKNOWN"
    assert envelope["error"] == "source changed during execution"
    assert [r["control"] for r in envelope["events"][1:]] == list(bridge.CONTROLS)
    assert envelope["events"][0]["phase"] == "compile_tests"
    assert all(not value for key, value in envelope["flags"].items()
               if key.endswith("_claim") or key == "candidate_admitted")


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "reordered", "unknown"])
def test_control_identity_negatives(retained, mutation):
    envelope = json.loads(retained[4])
    rows = envelope["events"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(rows[-1])
    elif mutation == "duplicate":
        rows[-1] = rows[1]
    elif mutation == "reordered":
        rows[1], rows[2] = rows[2], rows[1]
    else:
        rows[1]["control"] = "unissued"
    with pytest.raises(bridge.BindingError):
        parse(retained, envelope)


@pytest.mark.parametrize("key", ["paired_final_margins", "status", "substitution", "timing_seconds"])
def test_control_shared_fields(retained, key):
    envelope = json.loads(retained[4])
    envelope["events"][1][key] = None
    with pytest.raises(bridge.BindingError, match="shared fields"):
        parse(retained, envelope)


@pytest.mark.parametrize("key", ["status", "error", "error_type", "automatic_replay_authorized", "flags"])
def test_envelope_shared_fields(retained, key):
    envelope = json.loads(retained[4])
    envelope[key] = None
    with pytest.raises(bridge.BindingError, match="envelope mismatch"):
        parse(retained, envelope)


@pytest.mark.parametrize("key", ["phase", "executed", "compiled"])
def test_compile_envelope_mismatch(retained, key):
    envelope = json.loads(retained[4])
    envelope["events"][0][key] = None
    with pytest.raises(bridge.BindingError, match="compile event"):
        parse(retained, envelope)


def test_compile_origin_authentication(retained):
    compiled = copy.deepcopy(retained[3])
    compiled["compiled"][0]["bytes"] += 1
    with pytest.raises(bridge.BindingError, match="origin pins"):
        parse(retained, compiled=compiled)


def test_jsonl_and_nonobject_rejected(retained):
    _, failure, command, compiled, stdout, _ = retained
    rows = json.loads(stdout)["events"]
    for text in ("\n".join(json.dumps(r) for r in rows), "[]", stdout.decode() + "{}"):
        with pytest.raises((bridge.BindingError, json.JSONDecodeError)):
            bridge.parse_envelope(text, failure, compiled, command)


def test_failure_order_cannot_redefine_controls(retained):
    failure = copy.deepcopy(retained[1])
    failure["controls"][0], failure["controls"][1] = failure["controls"][1], failure["controls"][0]
    with pytest.raises(bridge.BindingError, match="control order"):
        parse(retained, failure=failure)


@pytest.mark.parametrize("field", ["digest", "size", "missing", "outside"])
def test_pin_failures(field):
    path, digest, size = bridge.CLOSURE, bridge.CLOSURE_SHA256, bridge.CLOSURE_BYTES
    if field == "digest":
        digest = "0" * 64
    elif field == "size":
        size += 1
    elif field == "missing":
        path = path.parent / "obstruction-bridge-nonexistent-input.json"
    else:
        path = Path("/outside-isolated-worktree")
    with pytest.raises((bridge.BindingError, FileNotFoundError)):
        bridge.authenticate(path, digest, size)


def test_closure_corruption_is_unknown(monkeypatch):
    monkeypatch.setattr(bridge, "CLOSURE_SHA256", "0" * 64)
    report = bridge.check()
    assert report["status"] == "UNKNOWN"
    assert not report["original_status_authenticated"]
    assert "sha256 mismatch" in report["error"]


def test_transitive_pin_mismatch_is_unknown(retained, monkeypatch):
    closure = copy.deepcopy(retained[0])
    closure["pins"][0]["declared"]["bytes"] += 1
    authenticate = bridge.authenticate

    def changed(path, digest, size=None):
        if path == bridge.CLOSURE:
            return json.dumps(closure).encode()
        return authenticate(path, digest, size)

    monkeypatch.setattr(bridge, "authenticate", changed)
    report = bridge.check()
    assert report["status"] == "UNKNOWN"
    assert "closure pin declaration" in report["error"]


def test_retained_bytes_drift_is_unknown(monkeypatch):
    read_bytes = Path.read_bytes

    def changed(path):
        data = read_bytes(path)
        return data + b"\n" if path.name == "check.stdout" else data

    monkeypatch.setattr(Path, "read_bytes", changed)
    report = bridge.check()
    assert report["status"] == "UNKNOWN"
    assert "sha256 mismatch" in report["error"]


def test_exact_historical_authority_pins_authenticated(retained):
    inventory = retained[0]["preservation"]["sha256_before"]
    assert {Path(p) for p in inventory if not Path(p).is_relative_to(bridge.ROOT)} == bridge.HISTORICAL_AUTHORITY


@pytest.mark.parametrize("field", ["failure_count", "accepted", "excess_over_budget", "reference"])
def test_s18_family_and_arithmetic_mismatch(retained, field):
    closure = copy.deepcopy(retained[0])
    row = closure["controls"][0]
    if field == "failure_count":
        row["failure_count"] += 1
    elif field == "reference":
        row["S18"][53]["reference_binary64_hex"] = "0x1p+0"
    else:
        row["S18"][53][field] = True if field == "accepted" else "0"
    with pytest.raises(bridge.BindingError, match="S18"):
        bridge.build_report(closure, retained[1])


def test_archived_failure_family_mismatch(retained):
    failure = copy.deepcopy(retained[1])
    failure["controls"][0]["reports"][-1]["binary64_v1"]["failures"].pop()
    with pytest.raises(bridge.BindingError, match="failure family census"):
        bridge.build_report(retained[0], failure)


@pytest.mark.parametrize("word,excess,accepted", [
    ("3c80", "1/8", True), ("3c81", "129/1024", False), ("3c00", "0", True),
])
def test_exact_threshold(word, excess, accepted):
    row = bridge.s18_quantities(word, "0x1p+0")
    assert row["excess_error"] == excess
    assert row["accepted"] is accepted
    assert F(row["excess_over_budget"]) == F(excess) - F(1, 8)
    assert F(row["threshold_margin"]) == -F(row["excess_over_budget"])


@pytest.mark.parametrize("word,ref", [("7c00", "0x1p+0"), ("0000", "nan"), ("fc00", "inf")])
def test_nonfinite_operands(word, ref):
    with pytest.raises(bridge.BindingError, match="nonfinite"):
        bridge.s18_quantities(word, ref)


def test_signed_zero_and_ties_even():
    assert bridge.s18_quantities("8000", "-0x0p+0")["nearest_fp16_bits"] == "8000"
    assert bridge.s18_quantities("3c01", float(F(2049, 2048)).hex())["nearest_fp16_bits"] == "3c00"
    assert bridge.s18_quantities("3c01", float(F(2051, 2048)).hex())["nearest_fp16_bits"] == "3c02"


@pytest.mark.parametrize("key", [
    "retained_margin_change", "substituted_margin_change", "paired_intervention_delta",
    "strict_direction_product", "strict_direction_pass", "pair", "reference_branch",
])
def test_margin_arithmetic_negatives(retained, key):
    row = copy.deepcopy(retained[0]["controls"][0]["paired_margins"][0])
    row[key] = True if key == "strict_direction_pass" else "0"
    with pytest.raises(bridge.BindingError):
        bridge.margin_bridge(row)


def test_equal_and_zero_movement_do_not_pass():
    row = {
        "pair": bridge.PAIR, "reference_branch": "fp16",
        "retained_actual_margin": "1", "retained_reference_margin": "1",
        "substituted_actual_margin": "1", "retained_margin_change": "0",
        "substituted_margin_change": "0", "paired_intervention_delta": "0",
        "strict_direction_product": "0", "strict_direction_pass": False,
    }
    assert not bridge.margin_bridge(row)["strict_direction_pass"]
    row["strict_direction_pass"] = True
    with pytest.raises(bridge.BindingError, match="strict direction"):
        bridge.margin_bridge(row)


@pytest.mark.parametrize("operation", ["write", "os_open", "dispatch", "remove", "producer_import"])
def test_forbidden_dispatch_write_guards(operation):
    target = bridge.OUT / "obstruction-bridge-forbidden-write"
    with bridge.readonly_guard() as audit:
        with pytest.raises(bridge.BindingError, match="forbidden"):
            if operation == "write":
                target.open("w")
            elif operation == "os_open":
                os.open(target, os.O_WRONLY | os.O_CREAT)
            elif operation == "dispatch":
                subprocess.run([sys.executable, "-c", "raise AssertionError('must not run')"], check=True)
            elif operation == "remove":
                os.remove(target)
            else:
                sys.audit("import", "ace3.model.candidates.forbidden_producer", None, [], [], [])
    assert len(audit["forbidden_events"]) == 1
    assert not target.exists()


def test_coherent_counterexample_rejects_structure(retained):
    closure, failure = copy.deepcopy(retained[:2])
    # Same valid archived margin at mapped_all changes the claimed two-cohort relation.
    replacement = closure["controls"][0]["paired_margins"]
    closure["controls"][7]["paired_margins"] = copy.deepcopy(replacement)
    failure["controls"][7]["paired_final_margins"] = [
        {k: v for k, v in m.items() if k not in {"strict_direction_pass", "strict_direction_product"}}
        for m in replacement
    ]
    assert bridge.build_report(closure, failure)["status"] == "REJECTED"


def test_real_main_single_json_output_and_preservation(capsys, monkeypatch):
    before = hashlib.sha256(bridge.CLOSURE.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", [str(Path(bridge.__file__))])
    bridge.main()
    output = capsys.readouterr()
    assert output.err == ""
    assert len(output.out.splitlines()) == 1
    report = json.loads(output.out)
    assert report["status"] == "SUPPORTED", report.get("error")
    assert report["original_science_status"] == "UNKNOWN"
    assert report["original_status_authenticated"]
    assert report["common_core"] == [53, 56, 62, 190, 208, 241, 289, 490, 493, 494, 502, 648]
    assert report["ordinary_only"] == [783, 827]
    assert report["mapped_all_only"] == []
    assert report["ordinary_exact_family_uniform"]
    assert report["margin_relation"]["ordinary_delta"] == "-3/32"
    assert report["margin_relation"]["mapped_all_delta"] == "-11/128"
    assert report["margin_relation"]["mapped_minus_ordinary"] == "1/128"
    assert report["margin_relation"]["strict_direction_passes"] == 0
    assert report["historical_failure"]["error"] == "source changed during execution"
    assert len(report["historical_checker_failures"]) == 4
    assert report["audit"]["forbidden_events"] == []
    assert report["science_invocations_this_round"] == 0
    assert report["authenticated_retained_files"] == 156
    assert hashlib.sha256(bridge.CLOSURE.read_bytes()).hexdigest() == before
    # The runner opts into one new result file; normal tests never write evidence.
    output_path = os.environ.get("ACE3_OBSTRUCTION_BRIDGE_RESULT")
    if output_path:
        path = Path(output_path)
        assert path.parent == bridge.OUT and path.name.startswith("obstruction-bridge-")
        with path.open("x") as stream:
            stream.write(output.out)


def test_dispatch_flags_not_accepted(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--execute"])
    with pytest.raises(bridge.BindingError, match="no dispatch"):
        bridge.main()
