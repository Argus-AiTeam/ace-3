"""Independent integer-pair dose oracle and hostile retained-account checks."""

import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

from ace3.model.candidates import (
    diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_source_only_dose_rank_classifier_v1 as diagnostic,
)


def pair(text):
    parts = text.split("/")
    return int(parts[0]), int(parts[1]) if len(parts) == 2 else 1


def exact(n, d):
    divisor = math.gcd(n, d)
    n, d = n // divisor, d // divisor
    if d < 0:
        n, d = -n, -d
    return str(n) if d == 1 else f"{n}/{d}"


def subtract(left, right):
    a, b = pair(left)
    c, d = pair(right)
    return exact(a * d - c * b, b * d)


def multiply(left, right):
    a, b = pair(left)
    c, d = pair(right)
    return exact(a * c, b * d)


def negative(text):
    n, d = pair(text)
    return exact(-n, d)


@pytest.fixture(scope="module")
def retained():
    parent, helper = diagnostic.load_helpers()

    def forbidden(*args, **kwargs):
        raise AssertionError("ancestor semantic execution is forbidden")

    parent.check = parent.classify = parent.report = parent.main = forbidden
    helper.check = helper.classify = helper._classify = helper.main = forbidden
    collapse, factor, modal, raw, authentication = diagnostic.authenticate(parent, helper)
    assert authentication["status"] == "AUTHENTICATED"
    return collapse, factor, modal, raw, parent, helper


def classify(retained, collapse=None):
    original, factor, modal, raw, parent, _ = retained
    return diagnostic.classify(original if collapse is None else collapse, factor, modal, raw, parent)


def test_independent_integer_pair_oracle(retained):
    collapse, _, _, raw, parent, _ = retained
    result = classify(retained)
    assert result["decision"] == "SUPPORTED", result
    assert result["failure_count"] == 0
    assert result["counts"] == {
        "component_accounts": 280, "control_coordinate_accounts": 40,
        "nonzero_source_only_movements": 20, "zero_source_accounts": 260,
        "factor_proportionality_checks": 280, "reverse_component_checks": 112,
    }
    observed, zero_count = {}, 0
    for account in collapse["report"]["coordinate_accounts"]:
        base = parent.resolve(raw, account["controls"][0]["retained_bindings"]["raw_selected_row_pointer"])
        for control in account["controls"]:
            row = parent.resolve(raw, control["retained_bindings"]["raw_selected_row_pointer"])
            for component, h in row["hidden_components"].items():
                dh = subtract(h, base["hidden_components"][component])
                dw = subtract(row["weighted_components"][component], base["weighted_components"][component])
                assert dw == multiply(dh, row[parent.FACTOR])
                key = (account["left_id"], account["right_id"], account["coordinate"], control["control"], component)
                if dh != "0":
                    observed[key] = (dh, dw)
                else:
                    zero_count += 1
                    assert dw == "0"
    assert zero_count == 260
    assert len(observed) == 20
    actual = {(r["left_id"], r["right_id"], r["coordinate"], r["control"], r["component"]):
              (r["source_delta"], r["weighted_movement"]) for r in result["nonzero_source_only_movements"]}
    assert actual == observed
    for (left, right, coordinate, control, component), (dh, dw) in observed.items():
        assert observed[(right, left, coordinate, control, component)] == (dh, negative(dw))
    assert [(g["group"], g["absolute_dose"], g["controls"]) for g in result["dose_groups"]] == [
        ("frozen_inherited_o", "3/32768", ["frozen_inherited_o", "frozen_inherited_o_down"]),
        ("scratch", "321/4194304", ["scratch", "scratch_down"]),
        ("mapped62", "933/16777216", ["mapped62"]),
    ]
    assert result["absolute_dose_rank_descending"] == ["frozen_inherited_o", "scratch", "mapped62"]
    values = {g["group"]: g["absolute_dose"] for g in result["dose_groups"]}
    assert [g["gap"] for g in result["pairwise_rational_gaps"]] == [
        "63/4194304", "603/16777216", "351/16777216",
    ]
    for gap in result["pairwise_rational_gaps"]:
        assert gap["gap"] == subtract(values[gap["larger"]], values[gap["smaller"]])
        assert pair(gap["gap"])[0] > 0 and gap["positive"]
    assert all(r["hidden_and_dose_equal"] and r["weighted_and_factor_opposite"]
               for r in result["reverse_pair_checks"])


@pytest.mark.parametrize("field,value", [
    ("source_delta", "1"), ("weighted_movement", "1"), ("source_only_term", "1"),
    ("source_only_residual", "1"), ("canonical_hidden", "0"), ("canonical_weighted", "0"),
])
def test_arithmetic_mismatch_rejected(retained, field, value):
    changed = copy.deepcopy(retained[0])
    changed["report"]["coordinate_accounts"][0]["controls"][1]["components"]["actual_residual_boundary"][field] = value
    result = classify(retained, changed)
    assert result["decision"] == "REJECTED"
    assert result["failure_count"] > 0


def test_reverse_source_orientation_rejected(retained):
    changed = copy.deepcopy(retained[0])
    term = changed["report"]["coordinate_accounts"][2]["controls"][1]["components"]["actual_residual_boundary"]
    term["source_delta"] = negative(term["source_delta"])
    result = classify(retained, changed)
    assert result["decision"] == "REJECTED"
    assert any(f["field"] == "reverse_hidden_signature" for f in result["failures"])


def test_reverse_weighted_orientation_rejected(retained):
    changed = copy.deepcopy(retained[0])
    term = changed["report"]["coordinate_accounts"][2]["controls"][1]["components"]["actual_residual_boundary"]
    term["weighted_movement"] = negative(term["weighted_movement"])
    result = classify(retained, changed)
    assert result["decision"] == "REJECTED"
    assert any(f["field"] == "reverse_weighted_orientation" for f in result["failures"])


def test_zero_source_movement_rejected(retained):
    changed = copy.deepcopy(retained[0])
    changed["report"]["coordinate_accounts"][0]["controls"][2]["components"]["input_hidden"]["weighted_movement"] = "1"
    assert classify(retained, changed)["decision"] == "REJECTED"


@pytest.mark.parametrize("kind", ["count", "missing_control", "rank", "gap"])
def test_counts_rank_and_gaps_rejected(retained, monkeypatch, kind):
    changed = copy.deepcopy(retained[0])
    if kind == "count":
        changed["report"]["counts"]["zero_source_count"] = 259
    elif kind == "missing_control":
        changed["report"]["coordinate_accounts"][0]["controls"].pop()
    elif kind == "rank":
        monkeypatch.setattr(diagnostic, "GROUPS", dict(reversed(list(diagnostic.GROUPS.items()))))
    else:
        monkeypatch.setattr(diagnostic, "GAPS", {key: "0" for key in diagnostic.GAPS})
    assert classify(retained, changed)["decision"] == "REJECTED"


@pytest.mark.parametrize("kind", ["missing", "duplicate", "pointer", "factor_binding", "unsupported"])
def test_ambiguous_or_incompatible_retained_input_unknown(retained, kind):
    changed = copy.deepcopy(retained[0])
    rows = changed["report"]["coordinate_accounts"][0]["controls"]
    if kind == "missing":
        del rows[1]["components"]["actual_residual_boundary"]["source_delta"]
    elif kind == "duplicate":
        rows.append(copy.deepcopy(rows[1]))
    elif kind == "pointer":
        rows[1]["retained_bindings"]["raw_selected_row_pointer"] = "/report/missing"
    elif kind == "factor_binding":
        rows[1]["weight_times_reference_anchor_times_row_difference"] = "0"
    else:
        changed["decision"] = "UNKNOWN"
    assert classify(retained, changed)["decision"] == "UNKNOWN"


@pytest.mark.parametrize("target", ["stdout", "capture", "review", "source", "test", "outer_capture", "launcher"])
def test_all_root_pins_enforced(retained, monkeypatch, target):
    pins = copy.deepcopy(diagnostic.PINS)
    pins[target]["sha256"] = "0" * 64
    monkeypatch.setattr(diagnostic, "PINS", pins)
    with pytest.raises(ValueError, match="artifact hash"):
        diagnostic.authenticate(retained[-2], retained[-1])


@pytest.mark.parametrize("kind", ["splice", "exit", "review", "counter", "outer"])
def test_capture_and_review_defects_unknown_contract(retained, monkeypatch, kind):
    read = diagnostic.bound_bytes

    def corrupt(binding):
        data = read(binding)
        if binding == diagnostic.PINS["capture"] and kind in ("splice", "exit"):
            value = json.loads(data)
            if kind == "splice":
                value["results"][-1]["argv"][-1] = "--other"
            else:
                value["results"][-1]["exit_status"] = 1
            return json.dumps(value).encode()
        if binding == diagnostic.PINS["review"] and kind == "review":
            value = json.loads(data)
            value["producer_role"] = "engineer"
            return json.dumps(value).encode()
        if binding == diagnostic.PINS["stdout"] and kind == "counter":
            value = json.loads(data)
            value["dispatch_and_write_audit"]["accepted_producer_replay"] = 1
            return json.dumps(value).encode()
        if binding == diagnostic.PINS["outer_capture"] and kind == "outer":
            value = json.loads(data)
            value["exit_status"] = 1
            return json.dumps(value).encode()
        return data

    monkeypatch.setattr(diagnostic, "bound_bytes", corrupt)
    with pytest.raises(ValueError):
        diagnostic.authenticate(retained[-2], retained[-1])


def test_no_dispatch_or_writes(retained, monkeypatch, tmp_path):
    parent, helper = retained[-2:]
    paths = diagnostic.allowed_paths(parent, helper)
    monkeypatch.setattr(helper, "allowed_paths", lambda: paths)
    audit = dict.fromkeys(helper.COUNTERS, 0)
    with helper.read_only(audit):
        assert classify(retained)["decision"] == "SUPPORTED"
    assert all(v == 0 for v in audit.values())
    with pytest.raises(helper.ForbiddenOperation), helper.read_only(audit):
        (tmp_path / "forbidden").write_text("no")
    assert audit["forbidden_calls"] == 1


def test_stdlib_only_and_cli_shape(monkeypatch, capsys):
    import ast

    tree = ast.parse(diagnostic.SOURCE.read_bytes())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    modules = [name.name.split(".")[0] for node in imports if isinstance(node, ast.Import) for name in node.names]
    modules += [node.module.split(".")[0] for node in imports if isinstance(node, ast.ImportFrom)]
    assert set(modules) <= sys.stdlib_module_names
    for decision, status in (("SUPPORTED", 0), ("REJECTED", 0), ("UNKNOWN", 2)):
        monkeypatch.setattr(diagnostic, "check", lambda: {"decision": decision})
        assert diagnostic.main(["--check"]) == status
        assert json.loads(capsys.readouterr().out) == {"decision": decision}


def test_bound_bytes_and_duplicate_json(retained, tmp_path):
    path = tmp_path / "retained"
    data = b'{"a":1,"a":2}'
    path.write_bytes(data)
    binding = diagnostic.pin(path, len(data), hashlib.sha256(data).hexdigest())
    assert diagnostic.bound_bytes(binding) == data
    with pytest.raises(ValueError, match="duplicate JSON"):
        retained[-1].decode(data)
    path.write_bytes(data + b"\n")
    with pytest.raises(ValueError, match="byte count"):
        diagnostic.bound_bytes(binding)
