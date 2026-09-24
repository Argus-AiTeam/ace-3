"""Independent integer-grid and nearest-even oracles; no parent execution."""

from copy import deepcopy
from fractions import Fraction
import hashlib
from pathlib import Path
import struct
import sys
import unittest

from ace3.model.candidates import diagnose_q24_s16_final_head_accumulation_boundary_cutoff_classifier_v1 as d


EVIDENCE = None


def q24(word):
    exponent, mantissa = (int(word) >> 10) & 31, int(word) & 1023
    if exponent == 31:
        raise ValueError("nonfinite oracle word")
    integer = mantissa if exponent == 0 else (1024+mantissa) << (exponent-1)
    return -integer if int(word) & 32768 else integer


def binary64(value):
    word = struct.unpack("<Q", struct.pack("<d", value))[0]
    exponent, mantissa = (word >> 52) & 2047, word & ((1 << 52)-1)
    if exponent == 2047:
        raise ValueError("nonfinite oracle double")
    significand = mantissa if exponent == 0 else mantissa+(1 << 52)
    value = Fraction(significand) * Fraction(2) ** ((1 if exponent == 0 else exponent)-1075)
    return -value if word >> 63 else value


def fixture():
    values = ["1/8", "1/16", "1/16", "1/8", "1/32", "3/32",
              "0", "-1/32", "-1/32"]
    return dict(zip(d.COLUMNS, ["synthetic", 34319, 319, "fp16", *values], strict=True))


class ClassifierTests(unittest.TestCase):
    def test_runtime_pins_match_executing_dependencies(self):
        executable = Path(sys.executable).read_bytes()
        pins = d.runtime_pins()
        self.assertEqual(Path(pins["python"]["path"]).resolve(),
                         Path(sys.executable).resolve())
        expected = {
            "python_version": sys.version,
            "python": {
                "path": pins["python"]["path"], "bytes": len(executable),
                "sha256": hashlib.sha256(executable).hexdigest(),
            },
            "numpy_version": d.bridge.mixed.np.__version__,
            "torch_version": str(d.bridge.mixed.torch.__version__),
        }
        self.assertEqual(pins, expected)
        if EVIDENCE is not None:
            self.assertEqual(EVIDENCE["runtime_pins"], expected)

    def test_runtime_pins_required_in_result_and_capture(self):
        pins = d.runtime_pins()
        result = {"runtime_pins": deepcopy(pins),
                  "preregistration": {"runtime_pins": deepcopy(pins)}}
        command, metadata = {"runtime_pins": deepcopy(pins)}, {"runtime_pins": deepcopy(pins)}
        self.assertEqual(d.validate_runtime_pins(result, command, metadata), pins)
        for index in range(4):
            for field in (None, *pins):
                for missing in (True, False):
                    with self.subTest(document=index, field=field, missing=missing):
                        r, c, m = deepcopy((result, command, metadata))
                        document = (r, r["preregistration"], c, m)[index]
                        target = document if field is None else document["runtime_pins"]
                        key = "runtime_pins" if field is None else field
                        if missing:
                            del target[key]
                        else:
                            target[key] = "changed"
                        with self.assertRaises(ValueError):
                            d.validate_runtime_pins(r, c, m)

    def test_half_integer_grid(self):
        for word in (0, 1, 1023, 1024, 15360, 0x7bff, 0x8000, 0x8001, 0xfbff):
            self.assertEqual(d.half_value(word), Fraction(q24(word), 1 << 24))

    def test_half_ties_even_and_sign(self):
        for lower in (0, 1, 1022, 1023, 1024, 0x3bff, 0x3c00, 0x3c01, 0x7bfe):
            midpoint = Fraction(q24(lower)+q24(lower+1), 1 << 25)
            expected = lower+(lower & 1)
            self.assertEqual(d.round_half(midpoint), expected)
            self.assertEqual(d.round_half(-midpoint), expected | 32768)
            epsilon = Fraction(1, 1 << 50)
            self.assertEqual(d.round_half(midpoint-epsilon), lower)
            self.assertEqual(d.round_half(midpoint+epsilon), lower+1)

    def test_nonfinite_and_overflow_refused(self):
        for value in (65520, -65520, 65536):
            with self.assertRaises(ValueError):
                d.round_half(Fraction(value))
        with self.assertRaises(ValueError):
            d.half_value(0x7c00)

    def test_preregistered_sign_and_crossing(self):
        p = d.prediction(fixture())
        self.assertEqual(p["predicted_relative_margin_movement"], "1/32")
        self.assertEqual(p["predicted_movement_sign"], 1)
        self.assertEqual(p["predicted_sign_pattern"], [1, 1, 1])

    def test_exact_counterfactual_closure(self):
        row = fixture()
        result = d.classify_pair(row, (Fraction(1, 8), Fraction()),
                                 (Fraction(1, 32), Fraction()), (0x3000, 0), (0x2800, 0))
        self.assertEqual(result["relative_margin_movement"], "1/32")
        self.assertTrue(result["direction_agrees"])
        self.assertTrue(result["crossing_pattern_agrees"])
        self.assertEqual(set(result["exact_closure_residuals"].values()), {"0"})

    def test_parent_splice_refused(self):
        for column in d.COLUMNS[4:]:
            row = fixture()
            row[column] = str(Fraction(row[column])+1)
            with self.assertRaises(ValueError):
                d.account_closure(row)

    def test_retained_capture_tamper_refused(self):
        with self.assertRaises(ValueError):
            d.decode_capture(b'{"branch_totals": {}}')

    def test_decisions_are_not_tautological(self):
        rows = [{"direction_agrees": True, "crossing_pattern_agrees": True} for _ in range(27)]
        self.assertEqual(d.decision(rows), "SUPPORTED")
        for key in ("direction_agrees", "crossing_pattern_agrees"):
            changed = deepcopy(rows)
            changed[7][key] = False
            self.assertEqual(d.decision(changed), "REJECTED")
        with self.assertRaises(ValueError):
            d.decision(rows[:26])
        self.assertEqual(len(set(d.SUCCESSORS.values())), 3)
        row = fixture()
        values = ("1", "1", "0", "1", "8191/8192", "1/8192", "0", "-1/8192", "-1/8192")
        row.update(zip(d.COLUMNS[4:], values, strict=True))
        rounded = d.classify_pair(row, (Fraction(1), Fraction()),
                                  (Fraction(8191, 8192), Fraction()),
                                  (0x3c00, 0), (0x3c00, 0))
        self.assertFalse(rounded["direction_agrees"])
        self.assertFalse(rounded["crossing_pattern_agrees"])
        self.assertEqual(rounded["relative_margin_movement"], "0")

    def test_live_dots_independent_bit_oracle(self):
        if EVIDENCE is None:
            self.assertEqual(binary64(0.5), Fraction(1, 2))
            return
        for key, (vector, weights) in EVIDENCE["operands"].items():
            w = [q24(word) for word in weights.view("<u2")]
            if vector.dtype.str == "<f2":
                x = [q24(word) for word in vector.view("<u2")]
                oracle = Fraction(sum(a*b for a, b in zip(x, w, strict=True)), 1 << 48)
            else:
                oracle = sum((binary64(float(a))*b for a, b in zip(vector, w, strict=True)),
                             Fraction()) / (1 << 24)
            self.assertEqual(EVIDENCE["dots"][key], oracle)

    def test_live_rne_independent_neighbor_oracle(self):
        if EVIDENCE is None:
            self.assertEqual(d.round_half(Fraction(1, 2)), 0x3800)
            return
        for key, word in EVIDENCE["rounded"].items():
            value = EVIDENCE["dots"][key]
            magnitude, target = abs(value), word & 0x7fff
            candidates = range(max(0, target-1), min(0x7bff, target+1)+1)
            best = min(candidates, key=lambda v: (abs(Fraction(q24(v), 1 << 24)-magnitude),
                                                  v & 1))
            self.assertEqual(target, best)
            self.assertEqual(bool(word & 32768), value < 0)

    def test_live_matrix_and_no_reference_mutation(self):
        if EVIDENCE is None:
            self.assertEqual(d.IDS, (13, 319, 34319))
            return
        self.assertEqual(len(EVIDENCE["accounts"]), 27)
        self.assertEqual(len(EVIDENCE["dots"]), 33)
        self.assertEqual(len(EVIDENCE["rounded"]), 30)
        for row in EVIDENCE["accounts"]:
            self.assertEqual(row["actual_margin_movement"], "0")
            self.assertEqual(row["parent_account"]["branch"], "fp16")
        self.assertFalse(any(key[:2] == ("reference", "binary64") for key in EVIDENCE["rounded"]))

    def test_replay_and_write_guards(self):
        audit = {"forbidden_calls": 0}
        with d.selected_only(audit):
            for operation in (d.bridge.measure, d.bridge.mixed.exact_head,
                              d.parent.rmsnorm, d.parent.logits):
                with self.assertRaises(RuntimeError):
                    operation(None)
            with self.assertRaises(RuntimeError):
                d.SOURCE.open("wb")
        self.assertEqual(audit["forbidden_calls"], 5)


if __name__ == "__main__":
    unittest.main()
