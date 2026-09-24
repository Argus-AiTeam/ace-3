"""Independent bit-decoding/RNE and row-ablation checks, without operator replay."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import struct
import subprocess
from unittest.mock import patch

import pytest

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_asymmetric_rne_threshold_localizer_v1 as d


def decode(word):
    return Fraction(struct.unpack("<e", struct.pack("<H", word))[0])


def nearest(value):
    lo, hi = 0, 0x7bff
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if decode(mid) <= abs(value):
            lo = mid
        else:
            hi = mid - 1
    word = min((lo, min(lo + 1, 0x7bff)), key=lambda w: (abs(decode(w) - abs(value)), w & 1))
    return word | (0x8000 if value < 0 else 0)


@pytest.fixture(scope="module")
def diagnostic_context(request):
    supplied = getattr(request.config, "_ace3_asymmetric_context", None)
    if supplied is not None:
        return supplied
    audit = dict.fromkeys(d.AUDIT_KEYS, 0)
    with d.read_only(audit):
        context = d.authenticate()
        report = d.localize(context)
    assert not any(audit.values())
    return context, report


def test_independent_row_partitions(diagnostic_context):
    _, report = diagnostic_context
    counts = dict.fromkeys(("early_active", "late_threshold", "inactive"), 0)
    for row in report["rows"]:
        b, s = Fraction(row["baseline_exact"]), Fraction(row["signed_row_dose_slope"])
        lo, hi = nearest(b + d.LOW * s), nearest(b + d.HIGH * s)
        assert (lo, hi) == (row["lower_word"], row["upper_word"])
        state = "early_active" if lo != row["baseline_word"] else "late_threshold" if hi != lo else "inactive"
        assert row["state"] == state
        counts[state] += 1
        assert Fraction(row["bracket_increment"]) == decode(hi) - decode(lo)
    assert report["row_state_counts"] == counts
    assert sum(counts.values()) == 36
    assert sum(map(len, report["row_state_partitions"].values())) == 36


def test_independent_exact_midpoints_and_slacks(diagnostic_context):
    for row in diagnostic_context[1]["rows"]:
        b, s, word = Fraction(row["baseline_exact"]), Fraction(row["signed_row_dose_slope"]), row["baseline_word"]
        direction = 1 if s > 0 else -1
        neighbor = word + (direction if word < 0x8000 else -direction)
        midpoint = (decode(word) + decode(neighbor)) / 2
        threshold = (midpoint - b) / s
        t = row["threshold"]
        assert Fraction(t["midpoint"]) == midpoint
        assert Fraction(t["dose_to_midpoint"]) == threshold
        assert t["tie_winner_word"] == nearest(midpoint)
        assert t["crossing_at_equality"] == bool(word & 1)
        for dose in (d.LOW, d.HIGH):
            assert Fraction(t["bracket"][str(dose)]["dose_slack"]) == threshold - dose
            assert Fraction(t["bracket"][str(dose)]["accumulator_slack"]) == direction * (midpoint - b - dose * s)
        assert row["threshold_equation_residual"] == "0"


def test_independent_reference_and_margin_increments(diagnostic_context):
    context, report = diagnostic_context
    lookup = {(r["control"], r["polarity"], r["row_id"]): r for r in report["rows"]}
    for item in report["contrasts"]:
        c, p, l, r, branch = (item[k] for k in ("control", "polarity", "left_id", "right_id", "branch"))
        left, right = lookup[c, p, l], lookup[c, p, r]
        refs = context["evidence"]["references"]["logits_" + branch]
        ref = decode(int(refs[l])) - decode(int(refs[r])) if branch == "fp16" else Fraction(float(refs[l])) - Fraction(float(refs[r]))
        margins = {}
        for name, dose in (("lower", d.LOW), ("upper", d.HIGH)):
            words = [nearest(Fraction(x["baseline_exact"]) + dose * Fraction(x["signed_row_dose_slope"]))
                     for x in (left, right)]
            margins[name] = decode(words[0]) - decode(words[1])
            assert Fraction(item[name + "_margin_change"]) == margins[name] - ref
        assert Fraction(item["closed_bracket_increment"]) == margins["upper"] - margins["lower"]
        assert sum(map(Fraction, item["row_increment_contributions"].values())) == margins["upper"] - margins["lower"]


def test_selected_necessary_sufficient_not_total_dominance(diagnostic_context):
    _, report = diagnostic_context
    assert report["successor"] == d.LATE
    assert [r["state"] for r in report["selected_rows"]] == ["late_threshold", "early_active"]
    assert len(report["coupled_contrasts"]) == 4
    for item in report["coupled_contrasts"]:
        sign = 1 if item["left_id"] == 319 else -1
        unit = Fraction(sign, 128)
        assert Fraction(item["lower_closed_delta"]) == unit
        assert Fraction(item["upper_closed_delta"]) == 2 * unit
        assert Fraction(item["closed_bracket_increment"]) == unit
        assert item["row_increment_contributions"] == {"319": str(unit), "34319": "0"}
        assert item["lower_row_contributions"]["319"] == "0"
        assert item["lower_row_contributions"]["34319"] == str(unit)
        assert set(item["upper_row_contributions"].values()) == {str(unit)}
        assert item["row319_held_at_lower_increment"] == "0"
        assert item["row34319_held_at_lower_increment"] == str(unit)


def test_independent_affine_and_working_closure(diagnostic_context):
    for row in diagnostic_context[1]["rows"]:
        b, t, w = (Fraction(row[n + "_exact"]) for n in ("baseline", "target", "working"))
        assert t - b == d.HIGH * Fraction(row["signed_row_dose_slope"])
        assert w - t == Fraction(row["hidden_rne_dot_remainder"])
        for name in ("baseline", "target", "working"):
            assert nearest(Fraction(row[name + "_exact"])) == row[name + "_word"]
        assert row["working_dose_threshold"] is None


def test_materially_different_successors_and_closure_failure(diagnostic_context):
    report = diagnostic_context[1]
    coupled, rows = deepcopy(report["coupled_contrasts"]), deepcopy(report["selected_rows"])
    assert d.decide(coupled, rows)["successor"] == d.LATE
    rows[0]["state"] = "early_active"
    for item in coupled:
        item["row_increment_contributions"] = {"319": "0", "34319": item["closed_bracket_increment"]}
    assert d.decide(coupled, rows)["successor"] == d.EARLY
    for item in coupled:
        half = str(Fraction(item["closed_bracket_increment"]) / 2)
        item["row_increment_contributions"] = {"319": half, "34319": half}
    assert d.decide(coupled, rows)["successor"] == d.MIXED
    coupled[0]["row_increment_contributions"]["319"] = "0"
    with pytest.raises(ValueError, match="closure"):
        d.decide(coupled, rows)
    with pytest.raises(ValueError, match="census"):
        d.decide(coupled[:-1], rows)


def test_endpoint_ties_nondyadic_and_signed_fp16_boundaries():
    for word in (1, 2, 1023, 1024, 0x3bff, 0x3c00, 0x3c01, 0xbc00, 0xbc01, 0x8001):
        for slope in (Fraction(3), Fraction(-3)):
            account = d.bracket.threshold(decode(word), word, slope)
            t = Fraction(account["dose_to_midpoint"])
            midpoint = decode(word) + t * slope
            assert account["tie_winner_word"] == nearest(midpoint)
            assert d.bracket.reached(account, t) == bool(word & 1)
            assert not d.bracket.reached(account, t - Fraction(1, 1 << 80))
            assert d.bracket.reached(account, t + Fraction(1, 1 << 80))
    assert d.bracket.threshold(Fraction(1), 0x3c00, Fraction(3))["dose_to_midpoint"] == "1/6144"
    for dose in (d.LOW, d.HIGH):
        word = 0x3c01
        slope = ((decode(word) + decode(word + 1)) / 2 - decode(word)) / dose
        assert d.bracket.reached(d.bracket.threshold(decode(word), word, slope), dose)
        assert not d.bracket.reached(d.bracket.threshold(decode(word + 1), word + 1, -slope), dose)


def test_scalar_and_partition_mutations(diagnostic_context):
    context = diagnostic_context[0]
    for key, value in (("baseline_exact", "0"), ("lower_word", 0), ("upper_word", 0),
                       ("signed_row_dose_slope", "1"), ("hidden_rne_dot_remainder", "0")):
        changed = {**context, "bracket": deepcopy(context["bracket"])}
        changed["bracket"]["rows"][0][key] = value
        with pytest.raises(ValueError):
            d.localize(changed)
    changed = {**context, "bracket": deepcopy(context["bracket"])}
    changed["bracket"]["rows"][0]["threshold"]["crossing_at_equality"] = True
    changed["bracket"]["rows"][0]["threshold"]["midpoint"] = "0"
    with pytest.raises(ValueError):
        d.localize(changed)
    changed["bracket"]["rows"] = changed["bracket"]["rows"][::-1]
    with pytest.raises(ValueError, match="census"):
        d.localize(changed)


def test_contrast_reference_and_dose_mutations(diagnostic_context):
    context = diagnostic_context[0]
    for key in ("fixed_reference_margin", "lower_closed_delta", "upper_margin_change"):
        changed = {**context, "bracket": deepcopy(context["bracket"])}
        changed["bracket"]["contrasts"][0][key] = "0"
        with pytest.raises(ValueError):
            d.localize(changed)
    changed = {**context, "bracket": deepcopy(context["bracket"])}
    changed["bracket"]["closed_dose_consistency"][0]["closed_delta"] = "0"
    with pytest.raises(ValueError):
        d.localize(changed)


def test_reviewed_receipt_gates(diagnostic_context):
    context = diagnostic_context[0]
    for key, value in (("status", "supported"), ("flags", {}), ("retained_thresholds", {}),
                       ("protected_input_identity", "foreign"),
                       ("retained_common_component", "PASS")):
        receipt = deepcopy(context["receipt"])
        receipt["native_result"][key] = value
        with pytest.raises((ValueError, KeyError)):
            d.bind_bracket(receipt, context["evidence"], context["summary"], context["summaries"])
    receipt = deepcopy(context["receipt"])
    receipt["validation"]["native_exit"] = 1
    with pytest.raises(ValueError):
        d.bind_bracket(receipt, context["evidence"], context["summary"], context["summaries"])


def test_source_pins_and_selection_mutations():
    for pin in (*d.PINS, d.RECEIPT):
        with pytest.raises(ValueError):
            d.base.read_bound({**pin, "sha256": "0" * 64})
    for key, value in (("IDS", (319, 13)), ("BRANCHES", ("fp16",)),
                       ("LOW", 0.03125), ("HIGH", Fraction(1, 16))):
        with patch.object(d, key, value), pytest.raises(ValueError):
            d.selection_gate()


def test_forbidden_dispatch_and_overwrites():
    audit = {"forbidden_calls": 0}
    operations = [
        lambda: d.bracket.check(), lambda: d.bracket.solve(None, None, None),
        lambda: d.bracket.solve_rows(None, None), lambda: d.crossing.attribute(None, None, None, None),
        lambda: d.scalar.measure(None, None, None), lambda: d.parent.execute("forbidden"),
        lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
        lambda: subprocess.Popen(["false"]), lambda: io.open(d.SOURCE, "w"),
        lambda: os.open(d.TEST, os.O_WRONLY | os.O_TRUNC), lambda: d.SOURCE.unlink(),
    ]
    with d.read_only(audit):
        for operation in operations:
            with pytest.raises(RuntimeError):
                operation()
    assert audit["forbidden_calls"] == len(operations)


def test_counts_immutable_evidence_and_historical_boundaries(diagnostic_context):
    context, report = diagnostic_context
    for key in ("lower_exact_margin_agreements", "upper_exact_margin_agreements",
                "target_contrast_closures", "working_contrast_closures", "increment_closures"):
        assert report[key] == 72
    assert report["closed_dose_consistency_count"] == report["closed_dose_consistency_agreements"] == 504
    assert report["other_single_row_contrasts"] == 68
    assert report["retained_scalar_cell_closures"] == 108
    assert context["bracket"]["classification"] == "rejected"
    assert context["crossing"]["classification"] == "supported"
    identity = d.closed.digest(context["evidence"])
    with pytest.raises(ValueError):
        d.closed.protect({**context["evidence"], "rows": {}}, identity)
    assert d.closed.digest(context["evidence"]) == identity
    assert all(v is False or v == 0 for k, v in d.FLAGS.items()
               if k not in ("historical_failures_preserved", "original_global_reference_unchanged"))


def test_single_json_unknown_and_fixed_cli():
    for error in (ValueError("arithmetic integrity"), RuntimeError("forbidden replay")):
        stream = io.StringIO()
        with patch.object(d, "check", side_effect=error), redirect_stdout(stream):
            code = d.main(["--check"])
        result = json.loads(stream.getvalue())
        assert (code, result["classification"], result["successor"]) == (1, "UNKNOWN/integrity", None)
        assert str(error) in result["integrity_error"]
    stream = io.StringIO()
    with patch.object(d, "check", return_value={"classification": d.LATE}), redirect_stdout(stream):
        assert d.main(["--check"]) == 0
    assert json.loads(stream.getvalue()) == {"classification": d.LATE}
    with redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        d.main(["--check", "--dose", "1/32"])
