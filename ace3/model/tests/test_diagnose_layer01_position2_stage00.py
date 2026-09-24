from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import diagnose_layer01_position2_stage00 as diagnostic  # noqa: E402


class Layer01Position2Stage00DiagnosticTests(unittest.TestCase):
    def test_controlled_rmsnorm_comparison_localizes_first_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = diagnostic.generate(
                diagnostic.DEFAULT_ATTEMPT,
                diagnostic.DEFAULT_ATTEMPT004,
                diagnostic.DEFAULT_ATTEMPT005,
                diagnostic.DEFAULT_REFERENCE,
                Path(temporary) / "diagnosis",
            )

        layer00 = result["layer00_stage00_control"]
        self.assertTrue(layer00["rtl_reproduced_by_integer_rmsnorm"])
        self.assertTrue(layer00["reference_reproduced_by_accepted_rmsnorm"])
        self.assertEqual(layer00["rtl_vs_accepted"]["different_count"], 1)
        self.assertEqual(
            layer00["earliest_exact_divergence"]["stage_name"],
            "input_rmsnorm",
        )
        self.assertFalse(layer00["nearest_root_changes_floor_root"])
        self.assertEqual(
            layer00["accepted_stage00_injected_replay"][
                "final_vs_preserved_rtl"
            ]["different_count"],
            0,
        )
        self.assertGreater(
            layer00["accepted_q_injected_replay"][
                "final_vs_preserved_rtl"
            ]["different_count"],
            0,
        )
        handoff = result["layer00_to_layer01_handoff"]
        self.assertTrue(handoff["rtl_final_equals_layer01_input"])
        self.assertNotEqual(
            handoff["rtl_semantic_sha256"],
            handoff["accepted_semantic_sha256"],
        )
        layer01 = result["layer01_stage00_control"]
        self.assertTrue(layer01["rtl_reproduced_by_integer_rmsnorm"])
        self.assertTrue(layer01["reference_reproduced_by_accepted_rmsnorm"])
        self.assertEqual(
            layer01["crossed_comparisons"][
                "rtl_vs_accepted_arithmetic_same_rtl_handoff"
            ]["different_count"],
            0,
        )
        self.assertEqual(
            layer01["crossed_comparisons"][
                "accepted_arithmetic_reference_handoff_vs_accepted"
            ]["different_count"],
            0,
        )
        self.assertEqual(result["status"], "ROOT_CAUSE_LOCALIZED")


if __name__ == "__main__":
    unittest.main()
