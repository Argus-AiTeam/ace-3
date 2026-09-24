import copy
from fractions import Fraction
import json
import os
import socket

import pytest

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_suffix_boundary_asymmetry_cancellation_localizer_v1 as localizer


def summaries():
    result = []
    for spec in localizer.SPECS:
        fields = ["control", "polarity", "left_id", "right_id", "branch",
                  "predicted_delta", "observed_delta",
                  *("same_polarity_" + d + "_dose_delta" for d in spec["comparators"]),
                  "retained_margin_change", "intervened_margin_change"]
        rows = []
        for control in localizer.CONTROLS:
            for polarity in localizer.POLARITIES:
                for left, right in localizer.PAIRS:
                    sign = 1 if left == 319 else -1
                    movement = Fraction(sign, 128) * (1 if polarity == "reverse" else -1)
                    row = [control, polarity, left, right, "binary64",
                           str(movement), str(movement)]
                    row += [str(movement)] * len(spec["comparators"])
                    row += [str(Fraction(sign, 2)), str(Fraction(sign, 2) + movement)]
                    rows.append(row)
        result.append({
            "status": spec["outcome"][0], "directional_agreements": spec["outcome"][1],
            "strict_dose_agreements": spec["outcome"][2],
            "protected_input_identity": localizer.INPUT_IDENTITY,
            "native_exit": 0, "stdout_json_documents": 1, "contrast_count": 72,
            "retained_common_component": "UNKNOWN", "normal_host_review": "REQUIRED",
            "tests": {"executed": 14, "errors": 0, "failures": 0, "skipped": 0},
            "dispatch_and_write_audit": {"forbidden_calls": 0,
                                        "final_rmsnorm_invocations": 27,
                                        "selected_row_head_invocations": 27},
            "changed_coordinate_counts": {
                p: {c: spec["changed_words"] for c in localizer.CONTROLS}
                for p in localizer.POLARITIES},
            "contrast_fields": fields, "contrast_values": rows,
            "reverse_zero_obstructions": 32, "recovered_reverse_zero_contrasts": 32,
            "flags": {"historical_failures_preserved": True,
                      "original_global_reference_unchanged": True, "candidate_admitted": False},
        })
    return result


def test_lossy_captures_never_localize_from_counts_or_margins():
    inputs = summaries()
    before = copy.deepcopy(inputs)
    result = localizer.localize(inputs)
    assert inputs == before
    assert result["classification"] == "UNKNOWN"
    assert result["retained_binary64_margin_closures"] == 108
    assert result["retained_ordered_pair_reversals"] == 54
    assert result["exact_decomposition_closures"] == 0
    assert len(result["accounts"]) == 216
    for account in result["accounts"]:
        assert all(c["status"] == "UNKNOWN" for c in account["components"].values())
        if account["branch"] == "fp16":
            assert account["margin"]["status"] == "UNKNOWN"
            assert "observed_delta" not in account["margin"]
        else:
            margin = account["margin"]
            assert Fraction(margin["intervened_margin_change"]) - Fraction(
                margin["retained_margin_change"]) == Fraction(margin["observed_delta"])
    assert len({tuple(a[k] for k in ("dose", "control", "polarity", "left_id",
                                    "right_id", "branch")) for a in result["accounts"]}) == 216


@pytest.mark.parametrize("mutation", [
    "missing", "duplicate", "order", "closure", "reversal", "comparator",
    "identity", "count", "outcome", "test_skip", "dispatch", "flag", "fraction",
])
def test_malformed_retained_bindings_fail_explicitly(mutation):
    values = summaries()
    item = values[2]
    if mutation == "missing":
        del item["contrast_fields"]
    elif mutation == "duplicate":
        item["contrast_values"][1] = item["contrast_values"][0]
    elif mutation == "order":
        item["contrast_values"].reverse()
    elif mutation == "closure":
        item["contrast_values"][0][-1] = "99"
    elif mutation == "reversal":
        item["contrast_values"][1][5] = "99"
    elif mutation == "comparator":
        item["contrast_values"][0][7] = "99"
        item["contrast_values"][1][7] = "-99"
    elif mutation == "identity":
        item["protected_input_identity"] = "bad"
    elif mutation == "count":
        item["changed_coordinate_counts"]["forward"]["mapped_all"] = 742
    elif mutation == "outcome":
        item["status"] = "supported"
    elif mutation == "test_skip":
        item["tests"]["skipped"] = 1
    elif mutation == "dispatch":
        item["dispatch_and_write_audit"]["forbidden_calls"] = 1
    elif mutation == "flag":
        item["flags"]["candidate_admitted"] = True
    elif mutation == "fraction":
        item["contrast_values"][0][6] = 0.5
    with pytest.raises((localizer.IntegrityError, KeyError)):
        localizer.localize(values)


def test_receipt_command_and_completion_are_bound():
    spec = {**localizer.SPECS[0], "command": localizer.digest(b"synthetic command")}
    summary = {"changed_coordinate_counts": {}}
    events = [
        {"type": "assistant.message", "data": {"toolRequests": [{
            "toolCallId": spec["call"], "name": "bash",
            "arguments": {"command": "synthetic command"}}]}},
        {"type": "tool.execution_complete", "data": {
            "toolCallId": spec["call"], "success": True,
            "result": {"content": json.dumps(summary, separators=(",", ":"))}}},
    ]
    raw = lambda: "\n".join(map(json.dumps, events)).encode()
    retained, capture = localizer.extract_receipt(raw(), spec)
    assert retained == summary
    assert capture["command_sha256"] == spec["command"]
    assert capture["producer_environment"]["status"] == "UNKNOWN"
    events[0]["data"]["toolRequests"][0]["arguments"]["command"] += " changed"
    with pytest.raises(localizer.IntegrityError, match="command mismatch"):
        localizer.extract_receipt(raw(), spec)
    events[0]["data"]["toolRequests"][0]["arguments"]["command"] = "synthetic command"
    events.append(events[-1])
    with pytest.raises(localizer.IntegrityError, match="duplicate"):
        localizer.extract_receipt(raw(), spec)


def test_duplicate_json_members_and_hash_tampering_fail(tmp_path):
    with pytest.raises(localizer.IntegrityError, match="duplicate"):
        localizer.unique_json('{"a":1,"a":2}')
    path = tmp_path / "receipt"
    path.write_bytes(b"original")
    pin = localizer.binding(path)
    assert localizer.read_bound(pin) == b"original"
    path.write_bytes(b"changed")
    with pytest.raises(localizer.IntegrityError, match="SHA256"):
        localizer.read_bound(pin)


@pytest.mark.parametrize("operation", ["write", "mkdir", "system", "socket", "producer"])
def test_guard_blocks_actual_side_effects_before_dispatch(operation, tmp_path):
    audit = dict.fromkeys(localizer.AUDIT_KEYS, 0)
    target = tmp_path / "forbidden"
    if operation == "producer":
        namespace = {"__name__": "ace3.model.candidates.closed_producer"}
        exec("def closed():\n    raise AssertionError('body must never execute')", namespace)
    with localizer.retained_only(audit), pytest.raises(localizer.IntegrityError):
        if operation == "write":
            target.write_bytes(b"forbidden")
        elif operation == "mkdir":
            target.mkdir()
        elif operation == "system":
            os.system("exit 97")
        elif operation == "socket":
            socket.socket()
        else:
            namespace["closed"]()
    assert not target.exists()
    assert audit["blocked_attempts"] == 1
    assert all(value == 0 for key, value in audit.items() if key != "blocked_attempts")


def test_failure_is_unknown_and_terminates_without_live_receipt_reads():
    result = localizer.check(localizer.ROOT / "tests")
    assert result["classification"] == "UNKNOWN"
    assert result["lane_terminated"] is True
    assert result["normal_host_review"] == "REQUIRED"
    assert "canonical ignored-build attempt" in result["integrity_failure"]["message"]
    assert "accounts" not in result
    assert result["flags"]["candidate_admitted"] is False


@pytest.mark.parametrize("mutation", ["argv", "account", "environment", "source", "validation"])
def test_execution_preflight_tampering_is_unknown(tmp_path, monkeypatch, mutation):
    attempt = tmp_path
    argv = [os.sys.executable, "-B", "-m", localizer.MODULE, "--check", "--attempt", str(attempt)]
    environment = dict(os.environ)
    command = " ".join(f"{key}={localizer.shlex.quote(value)}"
                       for key, value in sorted(environment.items()))
    command += " " + localizer.shlex.join(argv)
    preflight = {
        "argv": argv, "environment": environment, "command": command,
        "cwd": str(localizer.ROOT), "uid": os.getuid(),
        "source": localizer.binding(localizer.SOURCE), "test": localizer.binding(localizer.TEST),
        "python": localizer.binding(os.sys.executable),
        "validation": {"tests": 1, "errors": 0, "failures": 0, "skipped": 0},
    }
    monkeypatch.setattr(os.sys, "argv", [str(localizer.SOURCE), *argv[4:]])
    if mutation == "argv":
        preflight["argv"] = ["wrong"]
    elif mutation == "account":
        preflight["uid"] = -1
    elif mutation == "environment":
        preflight["environment"] = {}
    elif mutation == "source":
        preflight["source"]["sha256"] = "changed"
    else:
        preflight["validation"]["skipped"] = 1
    (attempt / "preflight.json").write_text(json.dumps(preflight))
    (attempt / "check.command.txt").write_text(command + "\n")
    with pytest.raises(localizer.IntegrityError):
        localizer.execution_identity(attempt)


def test_command_binding_preserves_bytes_without_assuming_environment_order():
    environment = {"PATH": "/usr/bin:/bin", "LC_CTYPE": "C.UTF-8", "SPACES": "a b"}
    argv = ["python", "-B", "-m", "diagnostic", "--check"]
    first = b"PATH=/usr/bin:/bin LC_CTYPE=C.UTF-8 SPACES='a b' python -B -m diagnostic --check\n"
    second = b"LC_CTYPE=C.UTF-8 SPACES='a b' PATH=/usr/bin:/bin python -B -m diagnostic --check\n"
    for raw in (first, second):
        assert localizer.bind_command_sidecar(raw, environment, argv) == localizer.digest(raw)
    assert localizer.digest(first) != localizer.digest(second)


@pytest.mark.parametrize("raw", [
    b"A=1 A=1 python --check",
    b"A=1 B=changed python --check",
    b"A=1 python --check",
    b"A=1 B=2 python --different",
    b"unexpected B=2 python --check",
])
def test_command_binding_rejects_real_identity_changes(raw):
    with pytest.raises(localizer.IntegrityError):
        localizer.bind_command_sidecar(raw, {"A": "1", "B": "2"}, ["python", "--check"])
