from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import diagnose_layer01_position2_stage08 as diagnostic  # noqa: E402


class Layer01Position2Stage08DiagnosticTests(unittest.TestCase):
    def test_controlled_projection_localizes_upstream_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = diagnostic.generate(
                diagnostic.DEFAULT_ATTEMPT,
                diagnostic.DEFAULT_REFERENCE,
                Path(temporary) / "diagnosis",
            )

        combinations = result["score_combinations"]
        self.assertEqual(
            combinations["rtl_q_rtl_k"]["failure_count"],
            10,
        )
        self.assertEqual(
            combinations["reference_q_rtl_k"]["failure_count"],
            0,
        )
        self.assertEqual(
            combinations["rtl_q_reference_k"]["failure_count"],
            10,
        )
        self.assertEqual(
            combinations["reference_q_reference_k"]["exact_match_count"],
            42,
        )
        self.assertEqual(
            result["first_reported_failure_values"]["preserved_rtl_stage08"],
            {"bits": "5021", "value": 33.03125},
        )
        self.assertEqual(
            result["first_reported_failure_values"][
                "accepted_reference_stage08"
            ],
            {"bits": "5027", "value": 33.21875},
        )
        self.assertEqual(
            result["repair_required"],
            "localize and repair the upstream layer00-to-layer01 hidden-state "
            "trajectory; changing only layer01 Q projection arithmetic cannot "
            "remove the material stage08 mismatch",
        )
        q_decomposition = result["q_projection_rope_decomposition"]
        self.assertEqual(
            q_decomposition["reference_projection_rtl_rope"]["failure_count"],
            0,
        )
        self.assertEqual(
            q_decomposition["rtl_projection_reference_rope"]["failure_count"],
            10,
        )
        controlled = result["controlled_projection_from_actual_norm"]
        self.assertEqual(
            controlled["rtl_norm_to_binary64_q_scores"]["failure_count"],
            10,
        )
        self.assertTrue(
            controlled["reference_norm_reproduces_reference_q_projection"]
        )
        self.assertEqual(result["status"], "ROOT_CAUSE_REFINED_UPSTREAM")


if __name__ == "__main__":
    unittest.main()
