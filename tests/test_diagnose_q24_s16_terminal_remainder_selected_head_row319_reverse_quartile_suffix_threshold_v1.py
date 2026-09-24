"""Independent rational/bit oracles for the three new reverse suffix doses."""

from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
import io
import json
import math
import os
import struct
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_row319_reverse_quartile_suffix_threshold_v1 as d


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
    supplied = getattr(request.config, "_ace3_reverse_quartile_context", None)
    if supplied is not None:
        return supplied
    audit = dict.fromkeys(d.AUDIT_KEYS, 0)
    context, report = d.experiment(audit)
    assert audit == d.expected_audit()
    return context, report


def test_exact_quartile_order(diagnostic_context):
    derivation = diagnostic_context[1]["scalar_prediction_derivation"]
    expected = [Fraction(i, 2048) for i in range(64, 69)]
    assert list(map(Fraction, derivation["dose_order"])) == expected
    assert list(d.DOSES) == expected[1:-1]
    for row, left, right in zip(derivation["strict_inequalities"], expected, expected[1:], strict=False):
        assert (Fraction(row["left"]), Fraction(row["right"])) == (left, right)
        assert row["positive_difference"] == right.numerator * left.denominator - left.numerator * right.denominator > 0


def test_independent_scalar_inactivity_prediction(diagnostic_context):
    rows = diagnostic_context[1]["scalar_prediction_derivation"]["rows"]
    assert len(rows) == 18
    for row in rows:
        b, s = Fraction(row["baseline_exact"]), Fraction(row["signed_row_dose_slope"])
        word = row["baseline_word"]
        neighbor = word + ((1 if s > 0 else -1) if word < 0x8000 else (-1 if s > 0 else 1))
        midpoint = (decode(word) + decode(neighbor)) / 2
        assert Fraction(row["threshold"]["dose_to_midpoint"]) == (midpoint - b) / s
        assert row["threshold"]["tie_winner_word"] == nearest(midpoint)
        for dose in d.DOSES:
            p = row["quartile_predictions"][str(dose)]
            assert Fraction(p["exact"]) == b + dose * s
            assert p["fp16_word"] == nearest(b + dose * s)
            assert p["active"] == (p["fp16_word"] != word)
            if row["row_id"] == 319:
                assert not p["active"] and Fraction(p["dose_slack"]) > 0


def test_independent_all_operand_roundings(diagnostic_context):
    context, report = diagnostic_context
    count = 0
    for dose, outputs in report["suffix_outputs"].items():
        assert len(outputs) == 9
        for output in outputs:
            assert output["polarity"] == "reverse"
            words = context["evidence"]["archives"][output["control"]]["stage18"]
            expected = [nearest(decode(int(w)) + Fraction(dose) * v)
                        for w, v in zip(words, context["delta"], strict=True)]
            assert output["operand"]["working_fp16_words"] == expected
            assert output["operand"]["changed_coordinates"] == [
                i for i, (a, b) in enumerate(zip(words, expected, strict=True)) if a != b]
            assert output["operand"]["frozen_terminal_vector_identity"] == d.closed.digest(context["delta"])
            count += len(expected)
    assert count == 24192


def test_independent_integer_rmsnorm_oracle(diagnostic_context):
    context, report = diagnostic_context
    weights = context["evidence"]["weight_array"].view("<u2")
    wq = [int(decode(int(w)) * 2**24) for w in weights]
    for outputs in report["suffix_outputs"].values():
        for output in outputs:
            words = output["operand"]["working_fp16_words"]
            aq = [int(decode(w) * 2**24) for w in words]
            mean = round(Fraction(sum(a * a for a in aq), 896) + d.parent.norm.EPSILON_Q48)
            root = math.isqrt(mean)
            expected = []
            for a, w, aw, ww in zip(aq, wq, words, weights, strict=True):
                q = round(Fraction(abs(a * w), root)) * (-1 if a * w < 0 else 1)
                word = nearest(Fraction(q, 2**24))
                if q == 0:
                    word = (aw ^ int(ww)) & 0x8000
                expected.append(word)
            assert output["rmsnorm_scalars"] == {"mean_q48": mean, "root_q24": root}
            assert output["rmsnorm_words"] == expected


def test_independent_selected_dot_oracle(diagnostic_context):
    context, report = diagnostic_context
    for outputs in report["suffix_outputs"].values():
        for output in outputs:
            hidden = list(map(decode, output["rmsnorm_words"]))
            expected = []
            for row_id in (319, 34319):
                weights = context["evidence"]["rows"][row_id].view("<u2")
                dot = sum((a * decode(int(w)) for a, w in zip(hidden, weights, strict=True)), Fraction())
                expected.append(nearest(dot))
            assert expected == output["selected_logit_words"]


def test_independent_original_reference_closures(diagnostic_context):
    context, report = diagnostic_context
    for dose, table in report["dose_tables"].items():
        outputs = {o["control"]: o for o in report["suffix_outputs"][dose]}
        for row in table:
            c, l, r, b = (row[k] for k in ("control", "left_id", "right_id", "branch"))
            refs = context["evidence"]["references"]["logits_" + b]
            ref = decode(int(refs[l])) - decode(int(refs[r])) if b == "fp16" else Fraction(float(refs[l])) - Fraction(float(refs[r]))
            words = outputs[c]["selected_logit_words"]
            new = decode(words[d.IDS.index(l)]) - decode(words[d.IDS.index(r)])
            old = context["evidence"]["arrays"][c]["logits"]
            baseline = decode(int(old[l])) - decode(int(old[r]))
            assert Fraction(row["fixed_reference_margin"]) == ref
            assert Fraction(row["observed_delta"]) == new - baseline
            assert Fraction(row["retained_margin_change"]) == baseline - ref
            assert Fraction(row["intervened_margin_change"]) == new - ref


def close(row):
    new = decode(row["measured_words"][0]) - decode(row["measured_words"][1])
    row["measured_margin"] = str(new)
    row["observed_delta"] = str(new - Fraction(row["baseline_margin"]))
    row["intervened_margin_change"] = str(new - Fraction(row["fixed_reference_margin"]))
    row["prediction_agrees"] = row["observed_delta"] == row["predicted_delta"]
    row["individual_words_agree"] = row["measured_words"] == row["predicted_words"]


def synthetic(report, first_by_control):
    tables = deepcopy(report["dose_tables"])
    for index, table in enumerate(tables.values()):
        for row in table:
            i = (row["left_id"], row["right_id"]).index(319)
            row["measured_words"][i] = row["baseline_words"][i] + (index >= first_by_control[row["control"]])
            close(row)
    return tables


def test_terminal_branch_closure_and_sampled_not_continuous_onset(diagnostic_context):
    report = diagnostic_context[1]
    uniform_successor = (
        "dose search closed; may motivate a distinct control-specific rounding-mechanism "
        "question using retained bytes, subject to new preregistration and independent review")
    for first, outcome in enumerate((d.NEAR, d.MIDDLE, d.MIDDLE, d.UPPER_ONLY)):
        tables = synthetic(report, dict.fromkeys(d.parent.CONTROLS, first))
        result = d.decide(tables)
        assert result["classification"] == outcome and result["successor"] == d.SUCCESSORS[outcome]
        assert result["successor"] == uniform_successor
        assert result["dose_search_closed"] and not result["global_threshold_model_closed"]
        assert result["status"] == ("supported" if first == 3 else "rejected")
        assert all(r["first_activation_quartile"] == first + 1 for r in result["per_control_first_activation"])
    firsts = dict.fromkeys(d.parent.CONTROLS, 3)
    firsts["mapped_all"] = 0
    mixed = d.decide(synthetic(report, firsts))
    assert mixed["classification"] == d.MIXED
    assert mixed["dose_search_closed"] and mixed["global_threshold_model_closed"]
    assert mixed["successor"] == (
        "dose search and global-threshold model closed; may motivate at most one "
        "preregistered family-contrast question using retained bytes, subject to independent review")
    tables = synthetic(report, dict.fromkeys(d.parent.CONTROLS, 0))
    for row in tables[str(d.DOSES[1])]:
        i = (row["left_id"], row["right_id"]).index(319)
        row["measured_words"][i] = row["baseline_words"][i]
        close(row)
    result = d.decide(tables)
    assert result["classification"] == d.NEAR
    assert all(r["activity_reversion_observed"] for r in result["per_control_first_activation"])
    assert not result["continuous_onset_in_sampling_interval_claimed"]


def test_measured_first_activation_census(diagnostic_context):
    context, report = diagnostic_context
    assert len(report["per_control_first_activation"]) == 9
    assert all(len(t) == 36 for t in report["dose_tables"].values())
    for account in report["per_control_first_activation"]:
        active = [next(o for o in report["suffix_outputs"][str(dose)] if o["control"] == account["control"])
                  ["selected_logit_words"][0] != context["prior_endpoints"][account["control"]]["baseline_words"][0]
                  for dose in d.DOSES]
        first = next((i for i, value in enumerate(active) if value), 3)
        assert account["first_activation_quartile"] == first + 1
        assert account["first_measured_active_dose"] == str((*d.DOSES, d.UPPER)[first])
    decision = d.decide(report["dose_tables"])
    assert decision == {k: report[k] for k in decision}


def test_closure_and_census_mutations(diagnostic_context):
    for field in ("observed_delta", "measured_margin", "predicted_delta", "fixed_reference_margin"):
        tables = deepcopy(diagnostic_context[1]["dose_tables"])
        tables[str(d.DOSES[0])][0][field] = "123"
        with pytest.raises(ValueError):
            d.decide(tables)
    tables = deepcopy(diagnostic_context[1]["dose_tables"])
    tables[str(d.DOSES[0])].pop()
    with pytest.raises(ValueError, match="census"):
        d.decide(tables)
    tables = deepcopy(diagnostic_context[1]["dose_tables"])
    row = tables[str(d.DOSES[0])][1]
    row["measured_words"] = row["measured_words"].copy()
    row["measured_words"][0] += 1
    close(row)
    with pytest.raises(ValueError, match="consistency"):
        d.decide(tables)


def test_reviewed_prior_and_scalar_mutations(diagnostic_context):
    context = diagnostic_context[0]
    for key in ("protected_input_identity", "classification", "retained_thresholds", "retained_common_component"):
        receipt = deepcopy(context["prior_receipt"])
        receipt["native_result"][key] = "changed"
        with pytest.raises(ValueError):
            d.bind_prior(receipt, context)
    for key in ("baseline_exact", "signed_row_dose_slope"):
        changed = {**context, "localizer": deepcopy(context["localizer"])}
        row = next(r for r in changed["localizer"]["rows"] if r["polarity"] == "reverse")
        row[key] = "0"
        with pytest.raises(ValueError):
            d.derive(changed)


def test_closed_doses_polarities_shapes_and_precision_refused(diagnostic_context):
    context = diagnostic_context[0]
    words = context["evidence"]["archives"][d.parent.CONTROLS[0]]["stage18"]
    for dose in (Fraction(1, 32), Fraction(17, 512), Fraction(9, 256), Fraction(3, 64), 65 / 2048):
        with pytest.raises(ValueError, match="dose"):
            d.prepare(words, context["delta"], "reverse", dose)
    with pytest.raises(ValueError, match="reverse"):
        d.prepare(words, context["delta"], "forward", d.DOSES[0])
    for bad in (words[:-1], words.astype(np.uint32)):
        with pytest.raises(ValueError, match="operand"):
            d.prepare(bad, context["delta"], "reverse", d.DOSES[0])
    with pytest.raises(ValueError, match="dyadic"):
        d.prepare(words, (Fraction(1, 3),) * 896, "reverse", d.DOSES[0])


def test_independent_rne_ties_signs_and_subnormals():
    for word in (1, 2, 1023, 1024, 0x3bff, 0x3c00, 0x3c01, 0xbc00, 0xbc01, 0x8001):
        for other in (word - 1, word + 1):
            midpoint = (decode(word) + decode(other)) / 2
            for value in (midpoint, midpoint - Fraction(1, 1 << 80), midpoint + Fraction(1, 1 << 80)):
                assert d.prior.round_scalar(value)["fp16_word"] == nearest(value)


def test_replay_write_and_operand_dispatch_guards(diagnostic_context):
    attempts = (
        lambda: d.prior.measure(None, None, None),
        lambda: d.prior.prepare(None, None, "reverse", d.UPPER),
        lambda: d.prior.localizer.localize(None),
        lambda: d.parent.execute(),
        lambda: subprocess.Popen(["forbidden"]),
        lambda: d.SOURCE.write_text("forbidden"),
        lambda: os.open(d.SOURCE, os.O_WRONLY | os.O_TRUNC),
        lambda: os.replace(d.SOURCE, d.TEST),
    )
    for attempt in attempts:
        audit = dict.fromkeys(d.AUDIT_KEYS, 0)
        with d.stage.hotspot.read_only(audit), d.no_replay(audit):
            with pytest.raises(RuntimeError):
                attempt()
        assert audit["forbidden_calls"] == 1
    evidence = diagnostic_context[0]["evidence"]
    words = evidence["archives"][d.parent.CONTROLS[0]]["stage18"]
    changed = words.copy()
    changed[0] ^= 1
    rows = np.stack([evidence["rows"][i] for i in d.IDS])
    audit = dict.fromkeys(d.AUDIT_KEYS, 0)
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
            assert result["dose_search_closed"]
            assert not result["flags"]["further_dose_search_permitted"]
            assert "exact missing integrity" in result["preregistration"]["integrity"]
            assert "supplied and independently reviewed" in result["preregistration"]["integrity"]


def test_preserved_failure_and_non_admission_boundaries(diagnostic_context):
    context, report = diagnostic_context
    assert context["prior_receipt"]["native_result"]["status"] == "rejected"
    assert context["bracket"]["classification"] == "rejected"
    assert report["reviewed_upper_endpoints"] == context["prior_endpoints"]
    assert all(e["row319_active"] for e in context["prior_endpoints"].values())
    assert report["frozen_original_terminal_vector"] == list(map(str, context["delta"]))
    for flag in ("candidate_admitted", "full_model_claim", "strict_FP16_state_claim",
                 "precision_or_scale_expansion", "baseline_suffix_replay", "prior_17_512_replay",
                 "working_remainder_interpolation", "new_forward_dose_execution", "continuous_threshold_claim",
                 "further_dose_search_permitted", "scalar_threshold_transfer_claim"):
        assert not d.FLAGS[flag]
    assert report["dose_search_closed"]
    for boundary in ("every outcome closes dose search", "including rejection and UNKNOWN",
                     "No further dose subdivision", "mini-bisection", "coordinate choice",
                     "reference mutation", "scalar-threshold transfer claim"):
        assert boundary in d.PREREGISTRATION["branch_closure"]
    assert d.FLAGS["historical_failures_preserved"] and d.FLAGS["original_global_reference_unchanged"]
