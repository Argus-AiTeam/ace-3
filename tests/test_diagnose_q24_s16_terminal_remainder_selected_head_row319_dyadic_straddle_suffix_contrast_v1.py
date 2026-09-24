"""Independent bit/rational oracle and fresh suffix checks, with no closed replay."""

from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import struct
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_row319_dyadic_straddle_suffix_contrast_v1 as d


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
    supplied = getattr(request.config, "_ace3_dyadic_context", None)
    if supplied is not None:
        return supplied
    audit = dict.fromkeys(d.AUDIT_KEYS, 0)
    context, report = d.experiment(audit)
    assert audit["final_rmsnorm_invocations"] == audit["selected_row_head_invocations"] == 36
    assert audit["forbidden_calls"] == audit["writes"] == audit["artifact_overwrites"] == 0
    return context, report


def test_independent_dyadic_threshold_derivation(diagnostic_context):
    report = diagnostic_context[1]["dyadic_threshold_derivation"]
    for row in report["rows"]:
        b, s = Fraction(row["baseline_exact"]), Fraction(row["signed_row_dose_slope"])
        word = row["baseline_word"]
        direction = 1 if s > 0 else -1
        neighbor = word + (direction if word < 0x8000 else -direction)
        midpoint = (decode(word) + decode(neighbor)) / 2
        threshold = (midpoint - b) / s
        assert Fraction(row["threshold"]["dose_to_midpoint"]) == threshold
        assert row["threshold"]["tie_winner_word"] == nearest(midpoint)
        for dose in d.DOSES:
            prediction = row["dyadic_predictions"][str(dose)]
            assert prediction["fp16_word"] == nearest(b + dose * s)
            assert Fraction(prediction["exact"]) == b + dose * s
            assert prediction["active"] == (prediction["fp16_word"] != word)
    assert d.DOSES[0] < Fraction(report["row319_threshold"]) < d.DOSES[1]
    assert 0 < Fraction(report["row34319_threshold"]) < Fraction(1, 32)
    for row in report["strict_inequalities"]:
        left, right = Fraction(row["left"]), Fraction(row["right"])
        assert row["positive_difference"] == right.numerator * left.denominator - left.numerator * right.denominator > 0


def test_independent_all_terminal_operand_roundings(diagnostic_context):
    context, report = diagnostic_context
    for dose, outputs in report["suffix_outputs"].items():
        for output in outputs:
            c, p = output["control"], output["polarity"]
            baseline = context["evidence"]["archives"][c]["stage18"]
            signed = Fraction(dose) * (-1 if p == "forward" else 1)
            words = [nearest(decode(int(w)) + signed * delta)
                     for w, delta in zip(baseline, context["delta"], strict=True)]
            assert words == output["operand"]["working_fp16_words"]
            assert output["operand"]["changed_coordinates"] == [
                i for i, (a, b) in enumerate(zip(baseline, words, strict=True)) if a != b]
            assert output["operand"]["frozen_terminal_vector_identity"] == d.closed.digest(context["delta"])


def test_independent_selected_head_dot_oracle(diagnostic_context):
    context, report = diagnostic_context
    for outputs in report["suffix_outputs"].values():
        for output in outputs:
            hidden = list(map(decode, output["rmsnorm_words"]))
            expected = []
            for row_id in d.IDS:
                weights = context["evidence"]["rows"][row_id].view("<u2")
                dot = sum((a * decode(int(b)) for a, b in zip(hidden, weights, strict=True)), Fraction())
                expected.append(nearest(dot))
            assert output["selected_logit_words"] == expected


def test_independent_reference_margin_closure(diagnostic_context):
    context, report = diagnostic_context
    for dose, table in report["dose_tables"].items():
        outputs = {(o["control"], o["polarity"]): o for o in report["suffix_outputs"][dose]}
        for row in table:
            c, p, l, r, branch = (row[k] for k in ("control", "polarity", "left_id", "right_id", "branch"))
            refs = context["evidence"]["references"]["logits_" + branch]
            ref = decode(int(refs[l])) - decode(int(refs[r])) if branch == "fp16" else Fraction(float(refs[l])) - Fraction(float(refs[r]))
            old = context["evidence"]["arrays"][c]["logits"]
            baseline = decode(int(old[l])) - decode(int(old[r]))
            words = outputs[c, p]["selected_logit_words"]
            new = decode(words[d.IDS.index(l)]) - decode(words[d.IDS.index(r)])
            assert Fraction(row["observed_delta"]) == new - baseline
            assert Fraction(row["fixed_reference_margin"]) == ref
            assert Fraction(row["retained_margin_change"]) == baseline - ref
            assert Fraction(row["intervened_margin_change"]) == new - ref


def test_prediction_census_and_selected_outcome(diagnostic_context):
    report = diagnostic_context[1]
    assert len(report["dyadic_threshold_derivation"]["rows"]) == 36
    assert all(len(rows) == 72 for rows in report["dose_tables"].values())
    assert d.decide(report["dose_tables"]) == {
        k: report[k] for k in ("status", "classification", "successor", "integrity_error", "prediction_closure_counts")}
    if report["classification"] == d.SUPPORT:
        for dose, table in report["dose_tables"].items():
            assert report["prediction_closure_counts"][dose]["scalar_threshold_agreements"] == 72
            assert report["prediction_closure_counts"][dose]["mapped_all_reverse_agreements"] == 4
            for row in table:
                if (row["control"], row["polarity"]) == ("mapped_all", "reverse"):
                    i = (row["left_id"], row["right_id"]).index(319)
                    assert (row["measured_words"][i] != row["baseline_words"][i]) == (Fraction(dose) == d.DOSES[1])
    else:
        assert report["classification"] in (d.LOW_MOVES, d.HIGH_STATIC, d.CONTROL, d.ROUNDING)


def close_measurement(row):
    new = decode(row["measured_words"][0]) - decode(row["measured_words"][1])
    row["measured_margin"] = str(new)
    row["observed_delta"] = str(new - Fraction(row["baseline_margin"]))
    row["intervened_margin_change"] = str(new - Fraction(row["fixed_reference_margin"]))
    row["prediction_agrees"] = row["observed_delta"] == row["predicted_delta"]
    row["individual_words_agree"] = row["measured_words"] == row["predicted_words"]


def supported_tables(report):
    tables = deepcopy(report["dose_tables"])
    for table in tables.values():
        for row in table:
            row["measured_words"] = row["predicted_words"].copy()
            close_measurement(row)
    assert d.decide(tables)["classification"] == d.SUPPORT
    return tables


def test_materially_distinct_scientific_successors(diagnostic_context):
    report = diagnostic_context[1]
    for outcome, dose, row_id in (
        (d.LOW_MOVES, d.DOSES[0], 319), (d.HIGH_STATIC, d.DOSES[1], 319),
        (d.CONTROL, d.DOSES[0], 34319), (d.ROUNDING, d.DOSES[1], 319),
    ):
        tables = supported_tables(report)
        for row in tables[str(dose)]:
            if (row["control"], row["polarity"]) != ("mapped_all", "reverse"):
                continue
            i = (row["left_id"], row["right_id"]).index(row_id)
            row["measured_words"][i] = (row["baseline_words"][i] if outcome in (d.HIGH_STATIC, d.CONTROL)
                                        else row["predicted_words"][i] + 1)
            close_measurement(row)
        assert d.decide(tables)["classification"] == outcome
    tables = supported_tables(report)
    row = tables[str(d.DOSES[0])][0]
    row["measured_words"][0] += 1
    close_measurement(row)
    assert d.decide(tables)["classification"] == d.CONTROL


def test_algebraic_closure_and_census_mutations(diagnostic_context):
    for field in ("observed_delta", "measured_margin", "predicted_delta", "fixed_reference_margin"):
        tables = deepcopy(diagnostic_context[1]["dose_tables"])
        tables[str(d.DOSES[0])][0][field] = "123"
        with pytest.raises(ValueError, match="closure"):
            d.decide(tables)
    tables = deepcopy(diagnostic_context[1]["dose_tables"])
    tables[str(d.DOSES[0])].pop()
    with pytest.raises(ValueError, match="census"):
        d.decide(tables)


def test_retained_scalar_and_receipt_mutations(diagnostic_context):
    context = diagnostic_context[0]
    for field in ("baseline_exact", "signed_row_dose_slope"):
        changed = {**context, "localizer": deepcopy(context["localizer"])}
        changed["localizer"]["rows"][0][field] = "0"
        with pytest.raises(ValueError):
            d.derive(changed)
    for field in ("protected_input_identity", "classification", "retained_common_component"):
        receipt = deepcopy(context["localizer_receipt"])
        receipt["native_result"][field] = "changed"
        with pytest.raises(ValueError):
            d.bind_localizer(receipt, context)


def test_closed_doses_wrong_shapes_and_precision_are_refused(diagnostic_context):
    context = diagnostic_context[0]
    words = context["evidence"]["archives"][d.parent.CONTROLS[0]]["stage18"]
    for dose in (Fraction(1, 32), Fraction(3, 64), Fraction(1, 16), 17 / 512):
        with pytest.raises(ValueError, match="dose"):
            d.prepare(words, context["delta"], "forward", dose)
    for bad in (words[:-1], words.astype(np.uint32)):
        with pytest.raises(ValueError, match="operand"):
            d.prepare(bad, context["delta"], "forward", d.DOSES[0])
    with pytest.raises(ValueError, match="dyadic"):
        d.prepare(words, (Fraction(1, 3),) * 896, "forward", d.DOSES[0])


def test_independent_rne_ties_signs_and_subnormals():
    for word in (1, 2, 1023, 1024, 0x3bff, 0x3c00, 0x3c01, 0xbc00, 0xbc01, 0x8001):
        for other in (word - 1, word + 1):
            midpoint = (decode(word) + decode(other)) / 2
            for value in (midpoint, midpoint - Fraction(1, 1 << 80), midpoint + Fraction(1, 1 << 80)):
                assert d.round_scalar(value)["fp16_word"] == nearest(value)


def test_replay_and_write_guards_without_side_effects():
    attempts = (
        lambda: d.localizer.localize(None), lambda: d.bracket.solve(None, None, None),
        lambda: d.crossing.attribute(None, None, None, None), lambda: d.scalar.measure(None, None, None),
        lambda: d.stage.prepare(None, None, None), lambda: d.parent.execute(),
        lambda: subprocess.Popen(["forbidden"]), lambda: d.SOURCE.write_text("forbidden"),
        lambda: os.open(d.SOURCE, os.O_WRONLY | os.O_TRUNC), lambda: os.replace(d.SOURCE, d.TEST),
    )
    for attempt in attempts:
        audit = dict.fromkeys(d.AUDIT_KEYS, 0)
        with d.stage.hotspot.read_only(audit), d.no_replay(audit):
            with pytest.raises(RuntimeError):
                attempt()
        assert audit["forbidden_calls"] == 1
    assert audit["writes"] == 1


def test_suffix_dispatch_rejects_unregistered_operand(diagnostic_context):
    evidence = diagnostic_context[0]["evidence"]
    words = evidence["archives"][d.parent.CONTROLS[0]]["stage18"]
    rows = np.stack([evidence["rows"][i] for i in d.IDS])
    audit = dict.fromkeys(d.AUDIT_KEYS, 0)
    changed = words.copy()
    changed[0] ^= 1
    with pytest.raises(ValueError, match="unregistered"):
        with d.closed.suffix_only(audit, [words], evidence["weight_array"], rows):
            d.parent.rmsnorm(changed, evidence["weight_array"])
    assert audit["final_rmsnorm_invocations"] == audit["selected_row_head_invocations"] == 0


def test_single_json_cli_and_unknown_integrity():
    for failure in (None, ValueError("authentication failure"), RuntimeError("dispatch failure")):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"status": "supported"}, side_effect=failure), redirect_stdout(stream):
            code = d.main(["--check"])
        result, end = json.JSONDecoder().raw_decode(stream.getvalue())
        assert not stream.getvalue()[end:].strip()
        assert code == (1 if failure else 0)
        if failure:
            assert result["classification"] == "UNKNOWN/integrity"
            assert result["successor"] is None and str(failure) in result["integrity_error"]


def test_frozen_input_and_non_admission_boundaries(diagnostic_context):
    context, report = diagnostic_context
    assert report["frozen_original_terminal_vector"] == list(map(str, context["delta"]))
    assert context["bracket"]["classification"] == "rejected"
    assert context["localizer"]["classification"] == d.localizer.LATE
    assert not d.FLAGS["candidate_admitted"] and not d.FLAGS["full_model_claim"]
    assert not d.FLAGS["strict_FP16_state_claim"] and not d.FLAGS["precision_or_scale_expansion"]
    assert not d.FLAGS["baseline_suffix_replay"] and not d.FLAGS["working_remainder_interpolation"]
    assert d.FLAGS["stage18_operand_construction"] and d.FLAGS["new_dose_operand_execution"]
