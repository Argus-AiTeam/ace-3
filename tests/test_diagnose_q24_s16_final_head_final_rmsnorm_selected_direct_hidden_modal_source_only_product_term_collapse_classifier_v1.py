"""Independent integer-pair oracle and hostile retained-input tests."""

import copy
import json
import math
from pathlib import Path

import pytest

from ace3.model.candidates import (
    diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_source_only_product_term_collapse_classifier_v1 as diagnostic,
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


def add(left, right):
    a, b = pair(left)
    c, d = pair(right)
    return exact(a * d + c * b, b * d)


def negative(text):
    n, d = pair(text)
    return exact(-n, d)


def subtract(left, right):
    return add(left, negative(right))


def multiply(*texts):
    n, d = 1, 1
    for text in texts:
        a, b = pair(text)
        n, d = n * a, d * b
    return exact(n, d)


def absolute(text):
    n, d = pair(text)
    return exact(abs(n), d)


def total(texts):
    result = "0"
    for text in texts:
        result = add(result, text)
    return result


def resolve(document, pointer):
    value = document
    for part in pointer.split("/")[1:]:
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


@pytest.fixture(scope="module")
def retained():
    helper = diagnostic.load_helper()
    def forbidden(*args, **kwargs):
        raise AssertionError("ancestor semantic execution forbidden")
    helper.check = helper.classify = helper._classify = helper.main = forbidden
    factor, modal, raw, authentication = diagnostic.authenticate(helper)
    assert authentication["status"] == "AUTHENTICATED"
    return factor, modal, raw


def test_independent_raw_integer_pair_oracle(retained):
    factor, modal, raw = retained
    result = diagnostic.classify(*retained)
    assert result["decision"] == "SUPPORTED", result.get("unknown_reasons", result.get("failures"))
    assert result["failure_count"] == 0
    assert result["coordinate_account_count"] == 5
    assert result["control_coordinate_account_count"] == 40
    assert result["counts"] == {
        "component_account_count": 280, "nonzero_weighted_movement_count": 20,
        "source_only_movement_count": 20, "zero_source_count": 260,
        "zero_source_movement_count": 0, "zero_factor_delta_term_count": 1400,
        "nonzero_factor_delta_term_count": 0, "nonzero_individual_factor_delta_count": 0,
        "nonzero_combined_factor_delta_count": 0, "retained_field_movement_count": 80,
        "reversed_component_checks": 112,
    }
    for account in result["coordinate_accounts"]:
        base = resolve(raw, account["controls"][0]["retained_bindings"]["raw_selected_row_pointer"])
        for control in account["controls"]:
            binding = control["retained_bindings"]
            row = resolve(raw, binding["raw_selected_row_pointer"])
            detail = resolve(raw, binding["raw_factor_row_pointer"])
            retained_factor = resolve(factor, binding["factor_control_pointer"])
            assert retained_factor["control"] == control["control"]
            assert resolve(modal, binding["modal_control_pointer"])["control"] == control["control"]
            f = multiply(detail["hidden_bridge"]["weight"], row[diagnostic.FACTORS[1]],
                         subtract(detail["left_weight"], detail["right_weight"]))
            assert f == control[diagnostic.FACTOR] == base[diagnostic.FACTOR]
            for name, term in control["components"].items():
                h, h0 = row["hidden_components"][name], base["hidden_components"][name]
                p, p0 = row["weighted_components"][name], base["weighted_components"][name]
                assert multiply(h, f) == p
                assert multiply(h0, f) == p0
                assert term["source_delta"] == subtract(h, h0)
                assert term["weighted_movement"] == subtract(p, p0)
                assert term["source_only_term"] == multiply(subtract(h, h0), f) == subtract(p, p0)
                assert set(term["individual_factor_delta_terms"].values()) == {"0"}
            for family in ("hidden", "weighted"):
                values = list(row[family + "_components"].values())
                signed, mass = total(values), total(map(absolute, values))
                for field, value in zip(diagnostic.MASSES, (signed, mass, subtract(mass, absolute(signed)))):
                    assert control["mass"][family][field]["value"] == value
                    assert control["mass"][family][field]["movement"] == subtract(
                        value, base[family + "_component_mass"][field])
    for fi, ri in ((0, 2), (1, 3)):
        for f, r in zip(result["coordinate_accounts"][fi]["controls"],
                        result["coordinate_accounts"][ri]["controls"]):
            for name in diagnostic.COMPONENTS:
                assert f["components"][name]["source_delta"] == r["components"][name]["source_delta"]
                assert f["components"][name]["weighted_movement"] == negative(
                    r["components"][name]["weighted_movement"])


@pytest.mark.parametrize("factors,base", [
    (("2", "3", "5"), ("1", "3", "5")),
    (("2", "4", "5"), ("2", "3", "5")),
    (("2", "3", "7"), ("2", "3", "5")),
    (("2", "3", "5"), ("1", "6", "5")),
    (("0", "3", "5"), ("0", "4", "5")),
])
def test_general_product_identity_including_compensation_and_zero(factors, base):
    f, f0 = tuple(map(diagnostic.rational, factors)), tuple(map(diagnostic.rational, base))
    p, p0 = multiply("7", *factors), multiply("11", *base)
    result = diagnostic.decomposition(diagnostic.rational("7"), f, diagnostic.rational(p),
                                      diagnostic.rational("11"), f0, diagnostic.rational(p0))
    assert result["four_term_identity_residual"] == result["combined_identity_residual"] == "0"
    expected = multiply("-4", *base)
    assert result["source_only_term"] == expected
    assert add(expected, total(result["individual_factor_delta_terms"].values())) == subtract(p, p0)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "pointer", "control", "noncanonical", "splice"])
def test_missing_ambiguous_or_incompatible_bindings_unknown(retained, mutation):
    factor, modal, raw = copy.deepcopy(retained)
    row = factor["report"]["coordinate_accounts"][0]["controls"][1]
    if mutation == "missing":
        del row["factors"]["norm_weight"]
    elif mutation == "duplicate":
        factor["report"]["coordinate_accounts"][0]["controls"].append(copy.deepcopy(row))
    elif mutation == "pointer":
        row["retained_bindings"]["raw_selected_row_pointer"] = "/report/controls/0"
    elif mutation == "control":
        row["control"] = "scratch"
    elif mutation == "noncanonical":
        row["factors"]["norm_weight"] = "198/32"
    else:
        row["factors"]["norm_weight"] = "1"
    result = diagnostic.classify(factor, modal, raw)
    assert result["decision"] == "UNKNOWN"
    assert result["unknown_reasons"]


@pytest.mark.parametrize("mutation", ["product", "mass", "zero_source", "individual_factor", "compensation"])
def test_arithmetic_and_source_only_failures_rejected(retained, mutation):
    factor, modal, raw = copy.deepcopy(retained)
    fr = factor["report"]["coordinate_accounts"][0]["controls"][1]
    mr = modal["report"]["coordinate_accounts"][0]["controls"][1]
    rr = resolve(raw, fr["retained_bindings"]["raw_selected_row_pointer"])
    detail = resolve(raw, fr["retained_bindings"]["raw_factor_row_pointer"])
    if mutation in ("individual_factor", "compensation"):
        weight = multiply(fr["factors"]["norm_weight"], "2")
        fr["factors"]["norm_weight"] = detail["hidden_bridge"]["weight"] = weight
        if mutation == "compensation":
            difference = multiply(fr["factors"]["row_difference"], "1/2")
            fr["factors"]["row_difference"] = detail["row_difference"] = difference
            left = add(difference, fr["right_weight"])
            fr["left_weight"] = detail["left_weight"] = left
    else:
        if mutation == "mass":
            key = "weighted.mass.absolute"
            rr["weighted_component_mass"]["absolute"] = add(rr["weighted_component_mass"]["absolute"], "1")
        else:
            name = "input_hidden" if mutation == "zero_source" else "actual_residual_boundary"
            key = "weighted." + name
            rr["weighted_components"][name] = add(rr["weighted_components"][name], "1")
        mr["field_deltas"][key] = add(mr["field_deltas"][key], "1")
    result = diagnostic.classify(factor, modal, raw)
    assert result["decision"] == "REJECTED", result
    assert result["failure_count"] > 0
    if mutation in ("individual_factor", "compensation"):
        assert result["counts"]["nonzero_individual_factor_delta_count"] > 0
        assert result["counts"]["nonzero_factor_delta_term_count"] > 0
    if mutation == "zero_source":
        assert result["counts"]["zero_source_movement_count"] > 0


@pytest.mark.parametrize("name", list(diagnostic.PINS))
def test_each_required_pin_is_authenticated(name, monkeypatch):
    original = Path.read_bytes
    binding = diagnostic.PINS[name]

    def changed(path):
        data = original(path)
        return data + b" " if str(path) == binding["path"] else data

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="retained byte count"):
        diagnostic.bound_bytes(binding)


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":0.5}'])
def test_strict_scientific_json(payload):
    with pytest.raises(ValueError):
        diagnostic.load_helper().decode(payload)


@pytest.mark.parametrize("event,args", [
    ("open", ("/tmp/forbidden-modal-product-write", "w", 65)),
    ("subprocess.Popen", ()), ("import", ()), ("os.listdir", (".",)),
    ("exec", ()), ("socket.connect", ()),
])
def test_no_write_replay_import_or_dispatch(event, args):
    import sys
    helper = diagnostic.load_helper()
    audit = {"forbidden_calls": 0}
    with pytest.raises(helper.ForbiddenOperation):
        with helper.read_only(audit):
            sys.audit(event, *args)
    assert audit["forbidden_calls"] == 1


def test_authentication_failure_is_unknown_stdout(monkeypatch, capsys):
    def fail():
        raise ValueError("authentication defect")
    monkeypatch.setattr(diagnostic, "load_helper", fail)
    assert diagnostic.main(["--check"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == result["artifact_authentication"]["status"] == "UNKNOWN"
    assert result["report"]["unknown_reasons"][0]["message"] == "authentication defect"


@pytest.mark.parametrize("argv", [[], ["--output", "/tmp/not-authorized"], ["--replay"], ["--che"]])
def test_stdout_only_cli(argv):
    with pytest.raises(SystemExit) as error:
        diagnostic.main(argv)
    assert error.value.code == 2
