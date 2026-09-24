"""Task-native guards and small exact arithmetic checks; no decoder replay."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_s16_final_head_from_l23_coordinate62_suffix_execution_v1 as d


EVIDENCE = None


class FinalHeadExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = EVIDENCE if EVIDENCE is not None else d.authenticate()

    def test_reviewed_nine_parents(self):
        d.check_summary(self.evidence["summary"])
        self.assertEqual(list(self.evidence["states"]), list(d.CONTROLS))

    def test_final_reference_identity(self):
        final = self.evidence["summary"]["final_reference"]
        self.assertEqual(final["output"], str(d.preflight.FINAL_REFERENCE_OUTPUT))
        self.assertEqual(final["terminal_review"]["mission_id"], "eff468ba33e5")
        self.assertEqual(final["terminal_review"]["review_status"], "done")

    def test_revision_refusal(self):
        summary = deepcopy(self.evidence["summary"])
        summary["asset_binding_revision"] = 2
        with self.assertRaises(ValueError):
            d.check_summary(summary)

    def test_reference_substitution_refusal(self):
        summary = deepcopy(self.evidence["summary"])
        summary["final_reference"]["authority"]["manifest"]["path"] += ".other"
        with self.assertRaises(ValueError):
            d.check_summary(summary)

    def test_reordered_parent_refusal(self):
        summary = deepcopy(self.evidence["summary"])
        summary["controls"].reverse()
        with self.assertRaises(ValueError):
            d.check_summary(summary)

    def test_erased_failure_refusal(self):
        summary = deepcopy(self.evidence["summary"])
        summary["controls"][0]["retained_L23"]["retained_L21"]["S18"] = "PASS"
        with self.assertRaises(ValueError):
            d.check_summary(summary)

    def test_noncanonical_outputs(self):
        for path in ("/tmp/output", "build/nested/" + d.OUTPUT.name,
                     "build/" + d.NAME + "_attempt000", "build/../build/" + d.OUTPUT.name):
            with self.subTest(path=path), self.assertRaises(ValueError):
                d.output_path(path, fresh=True)

    def test_occupied_output_refusal(self):
        with patch.object(d.Path, "exists", return_value=True), self.assertRaises(ValueError):
            d.output_path(d.OUTPUT, fresh=True)

    def test_nonignored_output_refusal(self):
        with patch.object(d.Path, "exists", return_value=False), \
                patch.object(d.subprocess, "run") as run, self.assertRaises(ValueError):
            run.return_value.returncode = 1
            d.output_path(d.OUTPUT, fresh=True)

    def test_rmsnorm_known_values(self):
        words = np.array([1, -1], dtype="<f2").view("<u2")
        actual, details = d.rmsnorm(words, np.ones(2, dtype="<f2"))
        np.testing.assert_array_equal(actual, words)
        self.assertGreater(details["root_q24"], 0)

    def test_rmsnorm_nonfinite_refusal(self):
        with self.assertRaises(ValueError):
            d.rmsnorm(np.array([0x7c00], dtype="<u2"), np.ones(1, dtype="<f2"))

    def test_head_exact_fraction_oracle(self):
        hidden = np.array([1, -0.5, 2**-24, 0.25], dtype="<f2")
        weights = np.array([[1, 2, 1, 0], [1, 0, 0, 2**-9],
                            [-1, 0, 0, -2**-9], [0.25, 0.5, 0, 0]], dtype="<f2")
        actual = d.logits(hidden.view("<u2"), weights)
        expected = []
        for row in weights:
            value = sum(Fraction.from_float(float(a)) * Fraction.from_float(float(b))
                        for a, b in zip(row, hidden, strict=True))
            expected.append(float(value))
        np.testing.assert_array_equal(actual, np.array(expected, dtype="<f2").view("<u2"))

    def test_head_overflow_refusal(self):
        with self.assertRaises(ValueError):
            d.logits(np.array([65504], dtype="<f2").view("<u2"),
                     np.array([[65504]], dtype="<f2"))

    def test_topk_ties_signed_zero(self):
        words = np.array([0, -0.0, 2, 2, -1], dtype="<f2").view("<u2")
        self.assertEqual([row["token_id"] for row in d.top_k(words, words=True)],
                         [2, 3, 0, 1, 4])

    def test_comparison_exact_error(self):
        words = np.array([1, 2], dtype="<f2").view("<u2")
        metrics = d.comparison(words, np.array([1.125, 2.0]), words)
        self.assertEqual(metrics["binary64_max_absolute_error"], "1/8")
        self.assertEqual(metrics["reviewed_fp16_mismatch_count"], 0)

    def test_array_corruption_refusal(self):
        stream = io.BytesIO()
        np.save(stream, np.ones(2, dtype="<u2"), allow_pickle=False)
        for payload, shape in ((stream.getvalue() + b"x", (2,)),
                               (stream.getvalue(), (3,))):
            with self.assertRaises(ValueError):
                d.checked_array(payload, shape, "<u2")

    def test_forbidden_dispatch(self):
        audit = d.new_audit()
        with d.no_dispatch(audit):
            with self.assertRaises(RuntimeError):
                d.preflight.parent.execute(None)
            with self.assertRaises(RuntimeError):
                d.subprocess.run(["false"])
        self.assertEqual(audit["forbidden_calls"], 2)

    def test_extra_or_reordered_control_refusal(self):
        for label, controls in ((d.CONTROLS[1], []), (d.CONTROLS[0], list(d.CONTROLS))):
            audit = d.new_audit()
            audit["controls"] = controls
            with self.assertRaises(ValueError):
                d.run_control(label, {}, (), audit)

    def _validate_stored(self):
        result = json.loads(d.read_bound(d.RETAINED_RESULT))
        with patch.object(d, "authenticate", return_value=self.evidence), \
                patch.object(d, "load_references", return_value={}), \
                patch.object(d, "comparisons", side_effect=[
                    row["comparisons"] for row in result["controls"]]), \
                patch.object(d, "run_control", side_effect=AssertionError("no dispatch")), \
                patch.object(d, "execute", side_effect=AssertionError("no execution")), \
                patch.object(d, "load_operands", side_effect=AssertionError("no operands")), \
                patch.object(d, "write", side_effect=AssertionError("no evidence writes")):
            return d.validate(d.OUTPUT)

    def test_stored_validation(self):
        paths = sorted(d.OUTPUT.iterdir(), reverse=True)
        with patch.object(d.Path, "iterdir", return_value=iter(paths)):
            result = self._validate_stored()
        self.assertEqual(result["status"], "VALIDATED_FINAL_HEAD_EVIDENCE")
        self.assertEqual(result["controls_verified"], 9)
        self.assertEqual(result["arrays_verified"], 18)
        self.assertEqual(result["retained_L23_failures"], 9)
        self.assertEqual(result["validation_dispatch"], d.new_audit())
        self.assertEqual(result["normal_host_review"], "REQUIRED")
        self.assertEqual(result["flags"], d.FLAGS)
        for name, pin in d.RETAINED_SOURCES.items():
            self.assertEqual(result["execution_origins"][name], pin)
            self.assertEqual(result["validation_origins"][name], d.record(d.Path(pin["path"])))
            self.assertNotEqual(result["validation_origins"][name], pin)

    def test_stored_validation_file_census_refusal(self):
        paths = sorted(d.OUTPUT.iterdir())
        for label, census in (("missing", paths[1:]),
                              ("extra", paths + [d.OUTPUT / "unexpected.json"])):
            with self.subTest(label=label), \
                    patch.object(d.Path, "iterdir", return_value=iter(census)), \
                    self.assertRaisesRegex(ValueError, "unexpected/missing evidence files"):
                self._validate_stored()

    def test_stored_validation_result_pin_refusal(self):
        record = d.record

        def changed_record(path):
            pin = record(path)
            if str(path) == d.RETAINED_RESULT["path"]:
                pin["sha256"] = "0" * 64
            return pin

        with patch.object(d, "record", side_effect=changed_record), \
                self.assertRaisesRegex(ValueError, "retained execution result changed"):
            self._validate_stored()

    def test_stored_validation_dependency_source_refusal(self):
        origins = d.source_context()
        origins[d.preflight.__name__]["sha256"] = "0" * 64
        with patch.object(d, "source_context", return_value=origins), \
                self.assertRaisesRegex(ValueError, "execution source bindings changed"):
            self._validate_stored()


if __name__ == "__main__":
    unittest.main()
