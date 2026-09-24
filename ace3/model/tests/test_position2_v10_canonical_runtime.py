"""Focused tests for the source-bound position-2 v10 runtime harness."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ace3/model"
sys.path.insert(0, str(MODEL))

import position2_v10_canonical_runtime as runtime  # noqa: E402


class Position2V10CanonicalRuntimeTests(unittest.TestCase):
    def test_static_inputs_bind_the_accepted_repaired_source_path(self) -> None:
        bindings = runtime.verify_static_inputs()
        self.assertEqual(bindings["accepted_package_id"], runtime.PACKAGE_ID)
        self.assertEqual(
            bindings["accepted_source_archive"]["sha256"],
            "b0aeb44ef1cb33d250864016c88297297cf0630187440d617cc1afee554f79f0",
        )
        self.assertEqual(bindings["accepted_source_set"]["file_count"], 30)
        validator = next(
            item
            for item in bindings["accepted_source_set"]["files"]
            if item["path"] == runtime.VALIDATOR_RELATIVE.as_posix()
        )
        self.assertEqual(
            validator["sha256"],
            "6b9fe66a6ba3fd0249ce7e618a6fd395907707a39225d8d56565e4ba8f6ec00c",
        )
        self.assertEqual(
            bindings["checkpoint"]["sha256"],
            "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b",
        )

    def test_preflight_names_the_canonical_validator_without_execution(
        self,
    ) -> None:
        run_id = "ace3-position2-fresh-v10-unit-test-0001"
        preflight = runtime.verify_preflight(run_id)
        self.assertEqual(
            preflight["status"],
            "PASS_READY_FOR_ONE_DURABLE_SUBMISSION",
        )
        self.assertEqual(
            preflight["canonical_validator"]["path"],
            runtime.VALIDATOR_RELATIVE.as_posix(),
        )
        self.assertEqual(
            preflight["runtime_action_counts"],
            {
                "durable_runner_submissions": 0,
                "canonical_validator_invocations": 0,
            },
        )

    def test_traversal_acceptance_requires_all_24_causal_layers(self) -> None:
        layers = [
            {
                "layer_index": index,
                "position": 2,
                "independent_oracle_comparison": {
                    "rtl_matches_exact_integer_oracle": True,
                    "within_tolerance": True,
                },
            }
            for index in range(24)
        ]
        document = {
            "schema_version": 3,
            "kind": "ace3_selected_token_position2_fresh_traversal_evidence",
            "status": "COMPLETE",
            "current_continuation_attempt": {
                "position": 2,
                "selected_token_id": 271,
                "layer_order": list(range(24)),
                "natural_terminal_layers": 24,
                "layers": layers,
                "post_layer23": {
                    "hidden_sha256": "0" * 64,
                    "natural_terminal": True,
                    "independent_oracle_within_tolerance": True,
                },
            },
        }
        accepted = runtime.accepted_traversal(document)
        self.assertEqual(accepted["layer_count"], 24)
        document["current_continuation_attempt"]["layers"] = layers[:-1]
        with self.assertRaisesRegex(
            runtime.RuntimeErrorV10,
            "24-layer traversal was not accepted",
        ):
            runtime.accepted_traversal(document)


if __name__ == "__main__":
    unittest.main()
