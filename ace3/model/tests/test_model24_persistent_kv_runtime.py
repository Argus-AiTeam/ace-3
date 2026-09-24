from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import model24_persistent_kv_runtime as runtime  # noqa: E402


def valid_document() -> dict:
    lineage = [
        {
            "layer_index": layer,
            "source_position_count": 2,
            "target_position_count": 3,
            "rtl_position_index": 2,
            "natural_terminal": True,
            "exact_integer_oracle_match": True,
        }
        for layer in range(24)
    ]
    return {
        "schema_version": 1,
        "kind": "ace3_model24_persistent_kv_second_token",
        "status": "NUMERIC_MISMATCH_REVIEW_REQUIRED",
        "input": {
            "token_history": [9707, 1879, 0],
            "injected_token_id": 0,
            "position_ordinal": 3,
        },
        "cache_lineage": lineage,
        "tied_lm_head_topk": {
            "status": "PASS",
            "comparison": {
                "failure_count": 0,
                "top_k_mismatch_count": 0,
            },
            "selected_token_id": 11,
            "top_k": [{} for _ in range(10)],
        },
        "host_continuation": {"selected_token_id": 11},
        "independent_policy_assessment": {
            "status": "NUMERIC_MISMATCH",
            "reference_seeds_from_rtl": False,
            "overall_within_tolerance": False,
            "first_rtl_mismatch": {"failure_count": 10},
        },
    }


class PersistentKvContractTests(unittest.TestCase):
    def test_exact_history_and_complete_cache_lineage_are_accepted(self) -> None:
        runtime._validate_core_contract(valid_document())

    def test_token_history_cache_and_selection_drift_are_rejected(self) -> None:
        document = valid_document()
        checks = runtime.focused_drift_checks(document)
        self.assertEqual(checks["status"], "PASS")
        self.assertEqual(checks["rejected_count"], 3)
        self.assertTrue(all(row["rejected"] for row in checks["checks"]))

    def test_missing_layer_and_oracle_mismatch_are_rejected(self) -> None:
        for mutate in (
            lambda item: item["cache_lineage"].pop(),
            lambda item: item["cache_lineage"][7].__setitem__(
                "exact_integer_oracle_match", False
            ),
        ):
            with self.subTest(mutate=mutate):
                document = copy.deepcopy(valid_document())
                mutate(document)
                with self.assertRaises(runtime.PersistentKvError):
                    runtime._validate_core_contract(document)

    def test_transaction_engine_configuration_is_restored(self) -> None:
        engine = runtime.transaction_engine
        before = (
            engine.POSITION0_TOKEN_ID,
            engine.POSITION1_TOKEN_ID,
            engine.SELECTED_TOKEN_ID,
            engine.POSITION,
            engine.compare_live_layers,
        )
        with runtime.configured_transaction_engine():
            self.assertEqual(
                (
                    engine.POSITION0_TOKEN_ID,
                    engine.POSITION1_TOKEN_ID,
                    engine.SELECTED_TOKEN_ID,
                    engine.POSITION,
                ),
                (9707, 1879, 0, 2),
            )
        after = (
            engine.POSITION0_TOKEN_ID,
            engine.POSITION1_TOKEN_ID,
            engine.SELECTED_TOKEN_ID,
            engine.POSITION,
            engine.compare_live_layers,
        )
        self.assertEqual(after, before)

    def test_final_rmsnorm_compiler_owns_compiled_directory(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            output = runtime._prepare_final_rmsnorm_output(Path(temporary))
            self.assertFalse((output / "compiled").exists())
            self.assertTrue((output / "tmp").is_dir())

    def test_independent_final_norm_uses_fixed_material_policy(self) -> None:
        expected = np.asarray([1000.0, 1.0], dtype="<f2").view("<u2")
        actual = np.asarray([1000.5, 1.25], dtype="<f2").view("<u2")

        comparison = runtime.compare_fp16_policy(actual, expected)

        self.assertEqual(comparison["absolute_tolerance"], 0.125)
        self.assertEqual(comparison["relative_tolerance"], 0.001)
        self.assertEqual(comparison["max_ulp_distance_allowed"], 1)
        self.assertEqual(comparison["failure_count"], 1)
        self.assertEqual(
            comparison["first_material_mismatch"]["element_index"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
