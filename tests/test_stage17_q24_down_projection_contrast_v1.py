"""Synthetic authenticated emitter-to-entry tests; never a production dispatch."""

import copy
from fractions import Fraction as F
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from ace3.model.candidates import stage17_q24_down_projection_contrast_v1 as d


fixtures = d.retained.module_from(d.base.TEST, "down_synthetic_fixtures")
suffix_oracle = fixtures.suffix_oracle


def verify_projection(tensors, words, actual):
    """Independent exact group dots and rational cross-group FP16 rounding."""
    name = d.PREFIX + "mlp.down_proj"
    lanes = np.arange(8)
    shifts = 4 * ((lanes % 2) * 4 + lanes // 2)
    q = ((tensors[name + ".qweight"].astype("<i8")[..., None] >> shifts) & 15).reshape(4864, 896)
    z = ((tensors[name + ".qzeros"].astype("<i8")[..., None] >> shifts) & 15).reshape(38, 896)
    x = words.view("<f2").astype("<f8") * (1 << 24)
    # Each 128-term integer group dot is below 2**51, hence binary64-exact.
    dots = np.stack([x[g * 128:(g + 1) * 128] @
                     (q[g * 128:(g + 1) * 128] - z[g]).astype("<f8") for g in range(38)])
    scales = tensors[name + ".scales"]
    for column in range(896):
        value = sum((F(float(dots[g, column])) * F(float(scales[g, column]))
                     for g in range(38)), F()) / (1 << 24)
        assert int(actual[column]) == suffix_oracle.round_word(value, value < 0), (
            f"projection coordinate {column}")


def fixture():
    raw, tensors, operands, refs, classified = fixtures.fixture()
    raw["s16_unrounded_binary64"] = np.zeros(4864, dtype="<f8")
    return raw, tensors, operands, refs, classified


class DownProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d.native.torch.set_num_threads(1)

    def test_native_nibbles_zero_groups_scales_and_signed_underflow(self):
        raw, tensors, _, _, _ = fixture()
        name, words = d.PREFIX + "mlp.down_proj", raw["stage16"]
        words[0], words[128], words[-128] = 0x3c00, 0x0001, 0x8001
        tensors[name + ".qweight"][0] = 0x76543210
        tensors[name + ".qzeros"][0] = 0x11111111
        tensors[name + ".scales"][0] = 0.5
        tensors[name + ".qweight"][[128, 4736]] = 0x11111111
        tensors[name + ".scales"][[1, 37]] = 0.5
        actual = d.projection(tensors, words)
        verify_projection(tensors, words, actual)
        np.testing.assert_array_equal(actual[:8].view("<f2"), [-0.5, 1.5, 0, 2, 0.5, 2.5, 1, 3])
        tensors[name + ".qweight"][:] = 0
        tensors[name + ".qzeros"][:] = 0
        tensors[name + ".qweight"][4736] = 0x11111111
        actual = d.projection(tensors, words)
        self.assertTrue(np.all(actual == 0x8000))
        verify_projection(tensors, words, actual)
        actual[0] ^= 1
        with self.assertRaisesRegex(AssertionError, "coordinate 0"):
            verify_projection(tensors, words, actual)

    def test_cross_group_cancellation_and_ties(self):
        raw, tensors, _, _, _ = fixture()
        name, words = d.PREFIX + "mlp.down_proj", raw["stage16"]
        words[[0, 128, 256, 4736]] = [0x7bff, 0xfbff, 0x3c00, 0x1400]
        tensors[name + ".qweight"][[0, 128, 256, 4736]] = 0x11111111
        tensors[name + ".scales"][37] = 0.5
        actual = d.projection(tensors, words)
        verify_projection(tensors, words, actual)
        self.assertTrue(np.all(actual == 0x3c00))

    def test_invalid_projection_inputs(self):
        for defect in ("shape", "dtype", "scale_nan", "scale_inf", "bias", "words", "word_dtype", "word_shape"):
            raw, tensors, _, _, _ = fixture()
            name = d.PREFIX + "mlp.down_proj"
            if defect == "shape":
                tensors[name + ".qweight"] = tensors[name + ".qweight"][:-1]
            elif defect == "dtype":
                tensors[name + ".qzeros"] = tensors[name + ".qzeros"].astype("<i8")
            elif defect.startswith("scale_"):
                tensors[name + ".scales"][0, 0] = np.nan if defect == "scale_nan" else np.inf
            elif defect == "bias":
                tensors[name + ".bias"] = np.zeros(896, dtype="<f2")
            elif defect == "word_dtype":
                raw["stage16"] = raw["stage16"].astype("<i4")
            elif defect == "word_shape":
                raw["stage16"] = raw["stage16"][:-1]
            else:
                raw["stage16"][0] = 0x7c00
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                d.projection(tensors, raw["stage16"])

    def test_overflow_is_rejected_not_clamped(self):
        raw, tensors, _, _, _ = fixture()
        raw["stage16"][:] = 0x7bff
        tensors[d.PREFIX + "mlp.down_proj.qweight"][:] = 0x11111111
        with self.assertRaisesRegex(ValueError, "overflow"):
            d.projection(tensors, raw["stage16"])

    def test_state_kv_and_rtz_rejected_before_projection(self):
        for key in ("stage12", "output_i", "output_cache_v", "stage03", "stage16"):
            raw, tensors, operands, _, _ = fixture()
            raw[key].flat[0] ^= (1 << 14) if key == "output_i" else 1
            with self.subTest(key=key), mock.patch.object(d, "projection") as project:
                with self.assertRaises(ValueError):
                    d.compute_suffix(raw, tensors, operands)
                project.assert_not_called()

    def test_forbidden_producers_and_all_writes(self):
        for function, args in (
                (d.parent.compute_suffix, (None,) * 3),
                (d.parent.parent.stage13, (None,) * 3),
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

    def test_reference_free_construction_and_original_thresholds(self):
        self.assertEqual(list(inspect.signature(d.projection).parameters), ["tensors", "words"])
        self.assertEqual(list(inspect.signature(d.compute_suffix).parameters), ["raw", "tensors", "operands"])
        for function in (d.projection, d.compute_suffix):
            for forbidden in ("reference", "nearest", "compare(", "margin", "classified"):
                self.assertNotIn(forbidden, inspect.getsource(function))
        previous = fixtures.IntegerContrastTests()
        previous.test_exact_s18_threshold()
        previous.test_zero_opposite_and_correct_original_reference_movement()


class AuthenticatedEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d.native.torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix="stage17_q24_down_projection_contrast_v1-test-", dir=d.ROOT / "build")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw, self.tensors, self.operands, self.refs, classified = fixture()
        self.reference = np.ones(896, dtype="<f8")
        self.trajectory = dict(raw)
        self.bridge = {"controls": [{**copy.deepcopy(classified), "control": c} for c in d.CONTROLS]}
        self.closure = {"thresholds": {"S18": "exact excess <= 1/8"}, "historical_failure": "retained"}
        self.documents, self.result_pins, self.review_pins = [], [], []
        for index, mission in enumerate((d.parent.MISSION, d.parent.parent.MISSION)):
            root = self.root / str(index)
            root.mkdir()
            document = {
                "mission_id": mission, "status": "REJECTED", "science_budget": 1,
                "science_invocations": 1, "non_admission": True,
                "audit": {"forbidden_events": []}, "protected_input_files_unchanged": True,
                "thresholds": self.closure["thresholds"], "source_pins": [d.pin(d.parent.SOURCE)],
                "input_pins": [], "operand_pins": [d.retained.digest(v) for v in self.operands],
                "tensor_pins": {k: d.retained.digest(v) for k, v in self.tensors.items()}, "controls": [],
            }
            for control in d.CONTROLS:
                payload = dict(raw)
                payload["q24_state_contrast_stage13"] = raw["stage13"].copy()
                if index == 0:
                    # Genuine Stage14/15 emitter shape: top-level Stage16 is current;
                    # an old prefixed Stage16 must not override it in the consumer.
                    payload["q24_state_contrast_stage16"] = np.ones(4864, dtype="<u2")
                    payload["final_logits"] = np.full(2, 0x3c00, dtype="<u2")
                path = root / (control + ".npz")
                with path.open("xb") as stream:
                    np.savez(stream, **payload)
                protected = dict(raw)
                protected["q24_state_contrast_stage13"] = raw["stage13"].copy()
                document["controls"].append({
                    "control": control, "payload": d.pin(path),
                    "stages": [{"stage": s, "status": "PASS"} for s in range(10, 19)],
                    "S18": {"status": "FAIL"}, "component_oracles": {"status": "PASS"},
                    "paired_margins": [{"reference_branch": b, "direction_pass": False}
                                       for b in ("fp16", "binary64")],
                    "protected_bytes_unchanged": True,
                    "protected_arrays": {k: d.retained.digest(v) for k, v in protected.items()},
                })
            review = root / "review.json"
            with review.open("x") as stream:
                json.dump({"mission_id": mission, "producer_role": "reviewer",
                           "review": {"status": "done"}}, stream)
            self.review_pins.append(d.pin(review))
            self.documents.append(document)
        self.publish_documents()
        self.base_values = (self.closure, {}, self.bridge, self.reference, self.refs,
                            self.operands, self.tensors, self.trajectory, [])
        self.patches = [
            mock.patch.object(d, "RESULT_PINS", self.result_pins),
            mock.patch.object(d, "REVIEW_PINS", self.review_pins),
            mock.patch.object(d.base, "inputs", side_effect=lambda: (*self.base_values[:-1], [])),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def publish_documents(self):
        self.result_pins[:] = []
        for index in (1, 0):
            if index == 0:
                self.documents[0]["retained_stage13"] = d.pin(self.root / "1/result.json")
            path = self.root / str(index) / "result.json"
            path.write_text(json.dumps(self.documents[index]))
        self.result_pins.extend(d.pin(self.root / str(i) / "result.json") for i in (0, 1))

    def test_full_authenticated_entry_nine_controls_no_science_or_writes(self):
        before = {str(p): d.pin(p) for p in self.root.rglob("*") if p.is_file()}
        oracle = __import__(__name__, fromlist=[""])
        with mock.patch.object(d.parent, "check", side_effect=AssertionError("science")) as science:
            result, payloads = d.evaluate(oracle)
        science.assert_not_called()
        self.assertEqual(result["status"], "SUPPORTED")
        self.assertEqual(list(payloads), list(d.CONTROLS))
        self.assertEqual(result["audit"]["forbidden_events"], [])
        self.assertEqual(result["audit"]["writes"], [])
        self.assertEqual(result["audit"]["stage_dispatches"],
                         [[c, 23, 0, s] for c in d.CONTROLS for s in (17, 18)])
        self.assertTrue(all(r["component_oracles"]["status"] == "PASS" for r in result["controls"]))
        self.assertEqual(before, {str(p): d.pin(p) for p in self.root.rglob("*") if p.is_file()})
        for arrays in payloads.values():
            self.assertTrue(np.all(arrays["stage16"] == 0))
            self.assertTrue(np.all(arrays["q24_state_contrast_stage16"] == 1))
        rows = copy.deepcopy(result["controls"])
        rows[0]["paired_margins"][0]["direction_pass"] = False
        self.assertEqual(d.parent.scientific_status(rows), "REJECTED")
        rows[0]["stages"].pop()
        with self.assertRaises(ValueError):
            d.parent.scientific_status(rows)

    def test_authentication_tamper_before_arithmetic(self):
        path = Path(self.documents[0]["controls"][0]["payload"]["path"])
        with path.open("ab") as stream:
            stream.write(b"tamper")
        with mock.patch.object(d, "projection") as project, self.assertRaises(ValueError):
            d.inputs()
        project.assert_not_called()

    def test_invalid_identity_review_threshold_tensor_and_lineage(self):
        pristine = copy.deepcopy(self.documents)
        for defect in ("identity", "threshold", "tensor", "operand", "order", "state", "payload"):
            self.documents = copy.deepcopy(pristine)
            doc = self.documents[0]
            if defect == "identity":
                doc["science_invocations"] = True
            elif defect == "threshold":
                doc["thresholds"] = {}
            elif defect == "tensor":
                doc["tensor_pins"] = {}
            elif defect == "operand":
                doc["operand_pins"] = []
            elif defect == "order":
                doc["controls"].reverse()
            elif defect == "state":
                doc["controls"][0]["protected_arrays"] = {}
            else:
                doc["controls"][0]["payload"] = doc["controls"][1]["payload"]
            self.publish_documents()
            with self.subTest(defect=defect), mock.patch.object(d, "projection") as project:
                with self.assertRaises(ValueError):
                    d.inputs()
                project.assert_not_called()
        review_path = Path(self.review_pins[0]["path"])
        review_path.write_text(json.dumps({
            "mission_id": d.parent.MISSION, "producer_role": "engineer", "review": {"status": "done"}}))
        self.review_pins[0] = d.pin(review_path)
        with self.assertRaisesRegex(ValueError, "independent retained review"):
            d.inputs()


if __name__ == "__main__":
    unittest.main()
