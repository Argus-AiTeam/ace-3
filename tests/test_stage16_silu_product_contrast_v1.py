"""Independent integer/rational oracle and real synthetic retained-entry checks."""

import copy
from fractions import Fraction as F
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

from ace3.model.candidates import stage16_silu_product_contrast_v1 as d


previous = d.retained.module_from(d.parent.TEST, "stage16_contrast_synthetic_fixtures")
suffix_oracle = previous.suffix_oracle
verify_projection = previous.verify_projection


def verify_stage16(gate, up, actual, unrounded):
    for index, (g, u) in enumerate(zip(gate, up, strict=True)):
        gi = int(suffix_oracle.scalar(g) * (1 << 24))
        ui = int(suffix_oracle.scalar(u) * (1 << 24))
        exponent, remainder = divmod(abs(gi), 11629080)
        polynomial = -3329
        for coefficient in (23302, -139810, 699051, -2796203,
                            8388608, -16777216, 16777216):
            polynomial = coefficient + round(F(remainder * polynomial, 1 << 24))
        exponential = 0 if exponent >= 63 else round(F(polynomial, 1 << exponent))
        negative = round(F(exponential << 24, (1 << 24) + exponential))
        sigmoid = negative if gi < 0 else (1 << 24) - negative
        integer = round(F(gi * ui * sigmoid, 1 << 48))
        value = F(integer, 1 << 24)
        sign = 0x8000 if value < 0 or (not value and (int(g) ^ int(u)) & 0x8000) else 0
        magnitude = abs(integer)
        if magnitude < 1024:
            word = magnitude
        else:
            shift = magnitude.bit_length() - 11
            word = ((shift + 1) << 10) | ((magnitude >> shift) - 1024)
        assert F(float(unrounded[index])) == value, f"product coordinate {index}"
        assert bool(np.signbit(unrounded[index])) == bool(sign), f"zero sign {index}"
        assert int(actual[index]) == sign | word, f"RTZ coordinate {index}"


class ArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d.native.torch.set_num_threads(1)

    def test_native_accurate_product_and_rtz_independent_oracle(self):
        gate = np.resize(np.array([0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00,
                                   0x4900, 0xc900, 0x7bff, 0xfbff], dtype="<u2"), 4864)
        up = np.resize(np.array([0xbc00, 0x3c00, 0x3800, 0x3800, 0x3c00], dtype="<u2"), 4864)
        words, values = d.stage16(gate, up)
        verify_stage16(gate, up, words, values)
        words[4] ^= 1
        with self.assertRaisesRegex(AssertionError, "RTZ coordinate 4"):
            verify_stage16(gate, up, words, values)

    def test_invalid_inputs_and_overflow_rejected(self):
        good = np.full(4864, 0x3c00, dtype="<u2")
        for bad in (good[:-1], good.astype("<i4"), np.full(4864, 0x7c00, dtype="<u2")):
            for args in ((bad, good), (good, bad)):
                with self.assertRaises(ValueError):
                    d.stage16(*args)
        with self.assertRaisesRegex(ValueError, "RTZ operand"):
            d.stage16(np.full(4864, 0x7bff, dtype="<u2"), good + 0x400)

    def test_state_and_kv_rejected_before_stage16(self):
        for key in ("stage12", "output_i", "output_cache_v", "stage03", "stage16"):
            raw, tensors, operands, _, _ = previous.fixture()
            raw[key].flat[0] ^= (1 << 14) if key == "output_i" else 1
            with self.subTest(key=key), mock.patch.object(d, "stage16") as arithmetic:
                with self.assertRaises(ValueError):
                    d.compute_suffix(raw, tensors, operands)
                arithmetic.assert_not_called()

    def test_reference_free_mechanism_and_unchanged_thresholds(self):
        self.assertEqual(list(inspect.signature(d.stage16).parameters), ["gate", "up"])
        for function in (d.stage16, d.compute_suffix):
            for forbidden in ("reference", "nearest", "compare(", "margin", "classified"):
                self.assertNotIn(forbidden, inspect.getsource(function))
        existing = previous.fixtures.IntegerContrastTests()
        existing.test_exact_s18_threshold()
        existing.test_zero_opposite_and_correct_original_reference_movement()

    def test_closed_producers_and_writes_denied(self):
        for function, args in (
                (d.parent.compute_suffix, (None,) * 3), (d.parent.evaluate, (None,)),
                (d.base.execute, ()), (d.prior.execute, ()),
                (d.retained.selection.nearest, (None,) * 3),
                (d.native.projection, ({}, d.PREFIX + "mlp.down_proj", None)),
                (d.native.state.project, ({}, d.PREFIX + "mlp.gate_proj", None)),
                (d.prior.audit_event, ("subprocess.Popen", ())),
                (d.prior.audit_event, ("open", (str(d.SOURCE), "w", 577)))):
            audit = d.base.new_audit()
            with d.guarded(audit), self.assertRaises(ValueError):
                function(*args)
            self.assertTrue(audit["forbidden_events"])
            self.assertEqual(audit["writes"], [])


class AuthenticatedEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d.native.torch.set_num_threads(1)

    def setUp(self):
        self.fixture = previous.AuthenticatedEntryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root / "stage17"
        self.root.mkdir()
        self.document = copy.deepcopy(self.fixture.documents[0])
        self.document["mission_id"] = d.STAGE17_MISSION
        for row in self.document["controls"]:
            raw = d.parent.archive(row["payload"])
            row["protected_arrays"] = {k: d.retained.digest(v) for k, v in raw.items()}
            path = self.root / (row["control"] + ".npz")
            with path.open("xb") as stream:
                np.savez(stream, **raw)
            row["payload"] = d.pin(path)
        result = self.root / "result.json"
        result.write_text(json.dumps(self.document))
        review = self.root / "review.json"
        review.write_text(json.dumps({"mission_id": d.STAGE17_MISSION, "producer_role": "reviewer",
                                      "review": {"status": "done"}}))
        self.result_pin, self.review_pin = d.pin(result), d.pin(review)
        for patch in (mock.patch.object(d, "RESULT_PIN", self.result_pin),
                      mock.patch.object(d, "REVIEW_PIN", self.review_pin)):
            patch.start()
            self.addCleanup(patch.stop)

    def test_real_nine_control_entry_authentication_arithmetic_and_gates(self):
        before = {str(p): d.pin(p) for p in self.fixture.root.rglob("*") if p.is_file()}
        with mock.patch.object(d.parent.parent, "check", side_effect=AssertionError("science")) as science:
            result, payloads = d.evaluate(sys.modules[__name__])
        science.assert_not_called()
        self.assertEqual(result["status"], "SUPPORTED")
        self.assertEqual(list(payloads), list(d.CONTROLS))
        self.assertEqual(result["audit"]["forbidden_events"], [])
        self.assertEqual(result["audit"]["writes"], [])
        self.assertEqual(result["audit"]["stage_dispatches"],
                         [[c, 23, 0, s] for c in d.CONTROLS for s in (16, 17, 18)])
        self.assertTrue(all(r["component_oracles"]["status"] == "PASS" for r in result["controls"]))
        for arrays in payloads.values():
            self.assertTrue(np.all(arrays["stage16"] == 0))
            self.assertTrue(np.all(arrays["q24_state_contrast_stage16"] == 1))
        self.assertEqual(before, {str(p): d.pin(p) for p in self.fixture.root.rglob("*") if p.is_file()})

    def test_payload_tamper_rejected_before_arithmetic(self):
        path = Path(self.document["controls"][0]["payload"]["path"])
        with path.open("ab") as stream:
            stream.write(b"tamper")
        with mock.patch.object(d, "stage16") as arithmetic:
            with self.assertRaises(ValueError):
                d.inputs()
            arithmetic.assert_not_called()

    def test_invalid_review_identity_order_and_lineage(self):
        pristine = copy.deepcopy(self.document)
        for defect in ("identity", "budget", "threshold", "operand", "order", "lineage"):
            document = copy.deepcopy(pristine)
            if defect == "identity":
                document["mission_id"] = d.parent.parent.MISSION
            elif defect == "budget":
                document["science_invocations"] = True
            elif defect == "threshold":
                document["thresholds"] = {}
            elif defect == "operand":
                document["operand_pins"] = []
            elif defect == "order":
                document["controls"].reverse()
            else:
                document["controls"][0]["protected_arrays"] = {}
            path = Path(self.result_pin["path"])
            path.write_text(json.dumps(document))
            self.result_pin.update(d.pin(path))
            with self.subTest(defect=defect), mock.patch.object(d, "stage16") as arithmetic:
                with self.assertRaises(ValueError):
                    d.inputs()
                arithmetic.assert_not_called()
        path.write_text(json.dumps(pristine))
        self.result_pin.update(d.pin(path))
        path = Path(self.review_pin["path"])
        path.write_text(json.dumps({"mission_id": d.STAGE17_MISSION, "producer_role": "engineer",
                                   "review": {"status": "done"}}))
        self.review_pin.update(d.pin(path))
        with self.assertRaisesRegex(ValueError, "independent Stage17 review"):
            d.inputs()


if __name__ == "__main__":
    unittest.main()
