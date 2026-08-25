#!/usr/bin/env python3
"""Focused contract and mutation tests for official Model24 evidence."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPOSITORY_ROOT / "ace3" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from official_model24_next_token import (  # noqa: E402
    ARTIFACT_NAME,
    LAYER_COUNT,
    LAYER_STAGE_COUNT,
    LAYER_TENSOR_COUNT,
    Model24ExecutionError,
    _layer_tensor_names,
    validate_document,
)


class OfficialModel24NextTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifact = REPOSITORY_ROOT / "build" / "official_model24_next_token" / ARTIFACT_NAME
        cls.document = json.loads(artifact.read_text(encoding="ascii"))

    def test_complete_official_execution_evidence(self) -> None:
        summary = validate_document(self.document)
        self.assertEqual(summary["layers"], LAYER_COUNT)
        self.assertEqual(summary["layer_tensors"], LAYER_COUNT * LAYER_TENSOR_COUNT)
        self.assertEqual(
            summary["intermediate_hashes"],
            LAYER_COUNT * LAYER_STAGE_COUNT,
        )
        self.assertTrue(
            self.document["token_decision"]["argmax_matches_independent_reference"]
        )

    def test_layer_tensor_names_cover_distinct_official_namespaces(self) -> None:
        inventories = [_layer_tensor_names(layer_id) for layer_id in range(LAYER_COUNT)]
        self.assertTrue(all(len(names) == LAYER_TENSOR_COUNT for names in inventories))
        self.assertEqual(
            len({name for names in inventories for name in names}),
            LAYER_COUNT * LAYER_TENSOR_COUNT,
        )

    def test_missing_layer_and_broken_lineage_are_rejected(self) -> None:
        missing = copy.deepcopy(self.document)
        missing["layers"].pop()
        with self.assertRaisesRegex(Model24ExecutionError, "24 layers"):
            validate_document(missing)

        broken = copy.deepcopy(self.document)
        broken["layers"][1]["input_hidden_sha256"] = "0" * 64
        with self.assertRaisesRegex(Model24ExecutionError, "lineage"):
            validate_document(broken)

        detached_terminal = copy.deepcopy(self.document)
        detached_terminal["terminal_hidden_state"]["source_layer_output_sha256"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(Model24ExecutionError, "descend"):
            validate_document(detached_terminal)

    def test_missing_tensor_and_false_argmax_are_rejected(self) -> None:
        missing = copy.deepcopy(self.document)
        missing["layers"][7]["consumed_tensors"].pop()
        with self.assertRaisesRegex(Model24ExecutionError, "tensor count"):
            validate_document(missing)

        false_argmax = copy.deepcopy(self.document)
        false_argmax["token_decision"]["argmax_matches_independent_reference"] = False
        with self.assertRaisesRegex(Model24ExecutionError, "argmax"):
            validate_document(false_argmax)


if __name__ == "__main__":
    unittest.main()
