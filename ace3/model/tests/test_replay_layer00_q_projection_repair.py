from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import replay_layer00_q_projection_repair as replay  # noqa: E402


class Layer00QProjectionRepairReplayTests(unittest.TestCase):
    def test_repaired_dependency_cone_passes_accepted_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = replay.generate(
                replay.DEFAULT_ATTEMPT,
                replay.DEFAULT_REFERENCE,
                Path(temporary) / "attempt",
            )

        self.assertEqual(result["status"], "PASS")
        repaired = result["repaired_comparisons"]
        self.assertTrue(repaired["layer00_stage01"]["within_tolerance"])
        self.assertTrue(repaired["layer00_stage18"]["within_tolerance"])
        self.assertTrue(repaired["layer01_stage00"]["within_tolerance"])
        self.assertTrue(repaired["layer01_stage01"]["within_tolerance"])
        self.assertEqual(repaired["layer01_stage08"]["failure_count"], 0)
        reduction = result["reduction"]
        self.assertLess(
            reduction["layer00_stage18"]["max_absolute_error_after"],
            reduction["layer00_stage18"]["max_absolute_error_before"],
        )
        self.assertLess(
            reduction["layer01_stage00"]["max_absolute_error_after"],
            reduction["layer01_stage00"]["max_absolute_error_before"],
        )
        self.assertLess(
            reduction["layer01_stage08"]["failure_count_after"],
            reduction["layer01_stage08"]["failure_count_before"],
        )


if __name__ == "__main__":
    unittest.main()
