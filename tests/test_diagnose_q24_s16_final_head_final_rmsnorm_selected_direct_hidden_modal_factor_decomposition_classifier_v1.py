"""Independent integer-pair oracle and hostile retained-input controls."""

import copy
import json
import math
from pathlib import Path
import sys

import pytest

from ace3.model.candidates import (
    diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_factor_decomposition_classifier_v1 as diagnostic,
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


def product(*texts):
    n, d = 1, 1
    for text in texts:
        a, b = pair(text)
        n, d = n * a, d * b
    return exact(n, d)


def subtract(left, right):
    a, b = pair(left)
    c, d = pair(right)
    return exact(a * d - c * b, b * d)


@pytest.fixture(scope="module")
def retained():
    # Read each retained byte source once; mutations below use private memory only.
    data = {p["path"]: Path(p["path"]).read_bytes() for p in diagnostic.PINS.values()}
    modal_capture = json.loads(data[diagnostic.PINS["modal_capture"]["path"]])
    bindings = [p for r in modal_capture["results"] for p in r["files"]]
    for name, key in (("source.snapshot.py", "modal_source"), ("test.snapshot.py", "modal_test")):
        bindings.append({**diagnostic.PINS[key], "path": str(diagnostic.MODAL_ROOT / name)})
    raw_capture = json.loads(data[diagnostic.PINS["raw_capture"]["path"]])
    bindings.extend(diagnostic.pin(diagnostic.RAW_ROOT / name, p["size_bytes"], p["sha256"])
                    for name, p in raw_capture["files"].items())
    for binding in bindings:
        if binding["path"] not in data:
            data[binding["path"]] = Path(binding["path"]).read_bytes()
    modal = json.loads(data[diagnostic.PINS["modal_stdout"]["path"]])
    raw = json.loads(data[diagnostic.PINS["raw_stdout"]["path"]])
    return modal, raw, data


def resolve(document, pointer):
    value = document
    for part in pointer.split("/")[1:]:
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def test_independent_retained_factor_oracle(retained):
    modal, raw, _ = retained
    result = diagnostic.classify(modal, raw)
    assert result["decision"] == "SUPPORTED", result
    assert result["bound_control_coordinate_count"] == result["product_closure_count"] == 40
    assert result["coordinate62_bound_row_count"] == 16
    assert result["individual_factor_field_count"] == 120
    assert result["compensating_rows"] == []
    for account in result["coordinate_accounts"]:
        for row in account["controls"]:
            binding = row["retained_bindings"]
            source = resolve(raw, binding["raw_selected_row_pointer"])
            detail = resolve(raw, binding["raw_factor_row_pointer"])
            parent = resolve(modal, binding["modal_account_pointer"])
            factors = row["factors"]
            assert factors["norm_weight"] == detail["hidden_bridge"]["weight"]
            assert factors[diagnostic.ANCHOR] == source[diagnostic.ANCHOR]
            assert factors["row_difference"] == subtract(detail["left_weight"], detail["right_weight"])
            expected = product(detail["hidden_bridge"]["weight"], source[diagnostic.ANCHOR],
                               detail["row_difference"])
            assert expected == row[diagnostic.FACTOR] == source[diagnostic.FACTOR]
            assert expected == parent["canonical_values"][diagnostic.FACTOR]
            assert set(row["factor_deltas_from_frozen_inherited"].values()) == {"0"}
    forward, _, reverse, _, context = result["coordinate_accounts"]
    assert forward["controls"][0]["factors"]["norm_weight"] == "99/16"
    assert forward["controls"][0]["factors"]["row_difference"] == "-135/8192"
    assert reverse["controls"][0]["factors"]["row_difference"] == "135/8192"
    assert context["controls"][0]["factors"]["norm_weight"] == "25/2"


def alter_detail(raw, control="scratch", scale="2"):
    bridge = raw["report"]["retained_final_rmsnorm_logit_margin_bridge"]
    for c in bridge["controls"]:
        if c["control"] == control:
            for p in c["pairs"]:
                if {p["left_id"], p["right_id"]} == {34319, 319}:
                    for row in p["branches"]["binary64"]["selected_coordinates"]:
                        if row["coordinate"] == 62:
                            row["hidden_bridge"]["weight"] = product(row["hidden_bridge"]["weight"], scale)
                            for field in ("left_weight", "right_weight", "row_difference"):
                                row[field] = product(row[field], "1/2")


def test_exact_compensating_factors_rejected(retained):
    modal, source, _ = retained
    raw = copy.deepcopy(source)
    alter_detail(raw)
    result = diagnostic.classify(modal, raw)
    assert result["decision"] == "REJECTED"
    assert len(result["compensating_rows"]) == 2
    assert result["product_closure_count"] == 40
    for row in result["compensating_rows"]:
        assert row["changed_factors"] == ["norm_weight", "row_difference"]
    moved = result["coordinate_accounts"][0]["controls"][4]
    terms = list(moved["ordered_telescoping_product_delta_terms"].values())
    assert terms[1] == "0"
    assert terms[0] == product(terms[2], "-1")
    assert moved["product_delta"] == "0"


def test_noncompensating_product_is_unknown(retained):
    modal, source, _ = retained
    raw = copy.deepcopy(source)
    alter_detail(raw, scale="3")
    result = diagnostic.classify(modal, raw)
    assert result["decision"] == "UNKNOWN"
    assert "factor product closure" in result["unknown_reasons"][0]["message"]


def test_anchor_compensation_is_not_individual_equality(retained):
    parent, source, _ = retained
    modal, raw = copy.deepcopy(parent), copy.deepcopy(source)
    bridge = raw["report"]["retained_final_rmsnorm_logit_margin_bridge"]
    for account in modal["report"]["coordinate_accounts"]:
        if account["coordinate"] != 62:
            continue
        control = next(c for c in account["controls"] if c["control"] == "scratch")
        anchor = account["canonical_values"][diagnostic.ANCHOR]
        control["field_deltas"][diagnostic.ANCHOR] = anchor
        row, _ = diagnostic.selected_row(raw["report"], "scratch", account["left_id"],
                                         account["right_id"], 62, "/report")
        row[diagnostic.ANCHOR] = product(anchor, "2")
        detail, _ = diagnostic.selected_row(bridge, "scratch", account["left_id"],
                                            account["right_id"], 62, "/report/bridge")
        detail["hidden_bridge"]["weight"] = product(detail["hidden_bridge"]["weight"], "1/2")
    result = diagnostic.classify(modal, raw)
    assert result["decision"] == "REJECTED"
    assert len(result["compensating_rows"]) == 2
    for row in result["compensating_rows"]:
        assert row["changed_factors"] == ["norm_weight", diagnostic.ANCHOR]


@pytest.mark.parametrize("mutation", [
    "missing_weight", "ambiguous_control", "ambiguous_pair", "ambiguous_coordinate",
    "wrong_ordered_pair", "source_delta", "anchor", "row_subtraction", "missing_context",
    "noncanonical", "float", "wrong_reference", "nested_direct_hidden",
])
def test_missing_spliced_ambiguous_fields_unknown(retained, mutation):
    parent, source, _ = retained
    modal, raw = copy.deepcopy(parent), copy.deepcopy(source)
    bridge = raw["report"]["retained_final_rmsnorm_logit_margin_bridge"]
    control = bridge["controls"][0]
    p = control["pairs"][0]
    row = next(r for r in p["branches"]["binary64"]["selected_coordinates"] if r["coordinate"] == 62)
    if mutation == "missing_weight":
        del row["hidden_bridge"]["weight"]
    elif mutation == "ambiguous_control":
        bridge["controls"].append(copy.deepcopy(control))
    elif mutation == "ambiguous_pair":
        control["pairs"].append(copy.deepcopy(p))
    elif mutation == "ambiguous_coordinate":
        p["branches"]["binary64"]["selected_coordinates"].append(copy.deepcopy(row))
    elif mutation == "wrong_ordered_pair":
        p["left_id"], p["right_id"] = p["right_id"], p["left_id"]
    elif mutation == "source_delta":
        modal["report"]["coordinate_accounts"][0]["controls"][1]["field_deltas"]["hidden.input_hidden"] = "1"
    elif mutation == "anchor":
        modal["report"]["coordinate_accounts"][0]["canonical_values"][diagnostic.ANCHOR] = "1"
    elif mutation == "row_subtraction":
        row["left_weight"] = "0"
    elif mutation == "missing_context":
        modal["report"]["coordinate_accounts"].pop()
    elif mutation == "noncanonical":
        row["hidden_bridge"]["weight"] = "198/32"
    elif mutation == "float":
        row["hidden_bridge"]["weight"] = 6.1875
    elif mutation == "wrong_reference":
        p["branches"]["binary64"]["hidden_reference"] = "local_reconstruction"
    elif mutation == "nested_direct_hidden":
        row["weighted_terms"]["direct_hidden"] = "0"
    assert diagnostic.classify(modal, raw)["decision"] == "UNKNOWN"


def test_authentication_from_cached_exact_bytes(retained, monkeypatch):
    _, _, data = retained
    monkeypatch.setattr(Path, "read_bytes", lambda path: data[str(path)])
    _, _, authentication = diagnostic.authenticate()
    assert authentication["status"] == "AUTHENTICATED"
    assert authentication["historical_50f_stderr_preserved_bytes"] == 3663


@pytest.mark.parametrize("name", list(diagnostic.PINS))
def test_every_required_pin_tamper_rejected(retained, monkeypatch, name):
    _, _, original = retained
    data = dict(original)
    path = diagnostic.PINS[name]["path"]
    value = data[path]
    data[path] = bytes([value[0] ^ 1]) + value[1:]
    monkeypatch.setattr(Path, "read_bytes", lambda path: data[str(path)])
    with pytest.raises(ValueError, match="retained hash changed"):
        diagnostic.authenticate()


@pytest.mark.parametrize("mutation", ["mission", "role", "kind", "status"])
def test_review_identity_not_decoy_fields(retained, mutation):
    review = json.loads(retained[2][diagnostic.PINS["modal_review"]["path"]])
    if mutation == "status":
        review["review"]["status"] = "continue"
    else:
        key = {"mission": "mission_id", "role": "producer_role", "kind": "kind"}[mutation]
        review[key] = "wrong"
    review["decoy"] = {"producer_role": "reviewer", "status": "done", "mission_id": "184a3f76ac61"}
    with pytest.raises(ValueError, match="independent terminal review"):
        diagnostic.terminal_review(review, "184a3f76ac61")


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":0.5}', b'{} {}'])
def test_strict_json(payload):
    with pytest.raises(ValueError):
        diagnostic.decode(payload)


@pytest.mark.parametrize("event,args", [
    ("open", (str(diagnostic.SOURCE), "w", 1)),
    ("open", ("/not-a-retained-input", "r", 0)),
    ("subprocess.Popen", ()), ("socket.connect", ()), ("import", ()),
    ("exec", ()), ("os.scandir", ()), ("os.remove", ()), ("os.mkdir", ()),
])
def test_read_only_guard(event, args):
    audit = {"forbidden_calls": 0}
    with diagnostic.read_only(audit):
        with pytest.raises(diagnostic.ForbiddenOperation):
            sys.audit(event, *args)
    assert audit["forbidden_calls"] == 1


def test_unknown_is_explicit_stdout_and_nonzero_exit(monkeypatch, capsys):
    monkeypatch.setattr(diagnostic, "check", lambda: {"decision": "UNKNOWN", "reason": "missing"})
    assert diagnostic.main(["--check"]) == 2
    assert json.loads(capsys.readouterr().out)["decision"] == "UNKNOWN"


@pytest.mark.parametrize("argv", [[], ["--che"], ["--check", "--output", "x"], ["--check", "--replay"]])
def test_no_output_or_replay_cli(argv):
    with pytest.raises(SystemExit) as error:
        diagnostic.main(argv)
    assert error.value.code == 2


def test_zero_counter_rejects_success_shaped_values():
    for value in (1, True, "0", None, 0.0):
        with pytest.raises(ValueError):
            diagnostic.zero_counters({"forbidden": value})
    diagnostic.zero_counters({"a": 0, "b": False})
