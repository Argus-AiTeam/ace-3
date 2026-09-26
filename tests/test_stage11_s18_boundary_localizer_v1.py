"""Independent rational suffix oracle and retained-boundary regressions."""

from fractions import Fraction
import math

import numpy as np
import pytest

from ace3.model.candidates import stage11_s18_boundary_localizer_v1 as d


def scalar(word):
    word = int(word)
    sign, exponent, mantissa = word >> 15, (word >> 10) & 31, word & 1023
    assert exponent != 31
    value = Fraction(mantissa if exponent == 0 else 1024 + mantissa)
    value *= Fraction(2) ** (-24 if exponent == 0 else exponent - 25)
    return -value if sign else value


def round_word(value, negative_zero=False):
    if not value:
        return 32768 if negative_zero else 0
    negative, value = value < 0, abs(value)
    lo, hi = 0, 31743
    while lo < hi:
        mid = (lo + hi) // 2
        if scalar(mid) < value:
            lo = mid + 1
        else:
            hi = mid
    word = min({lo, max(0, lo - 1)}, key=lambda w: (abs(scalar(w) - value), w & 1, w))
    return word | (32768 if negative else 0)


def verify_suffix(words, operands, norm, account, logits):
    decoded = [scalar(w) * (1 << 24) for w in words]
    mean = round(sum((v * v for v in decoded), Fraction()) / 896 + 281474977)
    root = math.isqrt(mean)
    assert account == {"mean_q48": mean, "root_q24": root}
    for i, weight in enumerate(operands[0].view("<u2")):
        q24 = round(decoded[i] * scalar(weight) * (1 << 24) / root)
        assert int(norm[i]) == round_word(Fraction(q24, 1 << 24),
                                         bool((int(words[i]) ^ int(weight)) & 32768))
    for row, weights in enumerate(operands[1].view("<u2")):
        value = sum((scalar(a) * scalar(b) for a, b in zip(norm, weights, strict=True)), Fraction())
        assert int(logits[row]) == round_word(value)


@pytest.fixture(scope="module")
def table():
    return d.finite_table()


@pytest.mark.parametrize("word,reference", [(0x4000, Fraction(1)), (0xc000, Fraction(-1)),
                                         (0x3800, Fraction(1, 3)), (0x8000, Fraction(0)),
                                         (0x3c80, Fraction(1)), (0x3c81, Fraction(1))])
def test_nearest_passable_exhaustive(word, reference, table):
    selected, floor = d.nearest(word, reference, table)
    values, words = table
    independent_floor = min(abs(scalar(w) - reference) for w in words)
    assert floor == independent_floor
    passing = [w for w, v in zip(words, values) if abs(v - reference) <= floor + Fraction(1, 8)]
    expected = word if word in passing else min(
        passing, key=lambda w: (abs(scalar(w) - scalar(word)), scalar(w), w))
    assert selected == expected


@pytest.mark.parametrize("word", [-1, 65536, 0x7c00, 0xfc00, 0x7e00])
def test_invalid_words(word):
    with pytest.raises(ValueError, match="finite"):
        d.scalar(word)


def test_adjust_preserves_all_other_words(table):
    from ace3.model.candidates import binary64_fp16_excess_v1 as policy
    words = [0x3c00] * 896
    words[53] = 0x4000
    words[783] = 0x8000
    references = np.ones(896, dtype="<f8")
    references[783] = 0
    failures = [{"index": 53, **policy.evaluate_layer_final_output(
        actual_fp16_bits=0x4000, reference_binary64_hex=float(1).hex())}]
    adjusted, reports = d.adjust(words, references, failures, table, policy.evaluate_layer_final_output)
    assert [i for i, (a, b) in enumerate(zip(words, adjusted)) if a != b] == [53]
    assert adjusted[783] == 0x8000 and reports[0]["adjusted_gate"]["accepted"]
    assert words[53] == 0x4000
    with pytest.raises(ValueError, match="scalar gate"):
        d.adjust(words, references, [], table, policy.evaluate_layer_final_output)
    failures[0]["q"] = "1"
    with pytest.raises(ValueError, match="scalar gate"):
        d.adjust(words, references, failures, table, policy.evaluate_layer_final_output)


@pytest.mark.parametrize("delta,status", [("1/64", "SUPPORTED"), ("0", "REJECTED"),
                                         ("-3/32", "REJECTED")])
def test_directional_classification(delta, status):
    rows = [{"control": c, "branch": b, "predicted_delta": "1/5", "margin_delta": delta}
            for c in d.CONTROLS for b in ("fp16", "binary64")]
    assert d.classify(rows) == status
    with pytest.raises(ValueError, match="census"):
        d.classify(rows[:-1])


def test_hash_and_source_identity_drift(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"original")
    record = d.pin(path)
    assert d.bound(record) == b"original"
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash/size"):
        d.bound(record)


def test_failure_union():
    assert len(d.COMMON) == 12 and len(d.UNION) == 14
    assert set(d.UNION) - set(d.COMMON) == {783, 827}


def test_independent_suffix_and_corruption():
    from ace3.model.candidates import q24_s16_final_head_from_l23_coordinate62_suffix_execution_v1 as final
    words = np.full(896, 0x3c00, dtype="<u2")
    words[0] = 0x8000
    operands = (np.ones(896, dtype="<f2"), np.full((2, 896), 1 / 1024, dtype="<f2"))
    norm, account = final.rmsnorm(words, operands[0])
    logits = final.logits(norm, operands[1])
    verify_suffix(words, operands, norm, account, logits)
    logits[0] += 1
    with pytest.raises(AssertionError):
        verify_suffix(words, operands, norm, account, logits)


def test_missing_binding_fails_before_suffix(monkeypatch):
    monkeypatch.setattr(d, "authenticate", lambda: (_ for _ in ()).throw(ValueError("missing pin")))
    audit = {"final_rmsnorm_invocations": 0, "tied_pair_head_invocations": 0}
    with pytest.raises(ValueError, match="missing pin"):
        d.check(audit)
    assert not any(audit.values())
