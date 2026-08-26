#!/usr/bin/env python3
"""Focused tests for the layer-3 Token-0 diagnostic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

from layer3_token0_diagnostic import (  # noqa: E402
    DiagnosticError,
    decode_stage_records,
    distribution,
)


class TraceDecodeTests(unittest.TestCase):
    def test_pair_interleaved_vector_is_restored_by_index(self) -> None:
        records = [(index, index + 1) for index in range(896)]
        records[0], records[32] = records[32], records[0]
        decoded = decode_stage_records(records, stage=4, token=0)
        self.assertEqual(decoded[0], 1)
        self.assertEqual(decoded[32], 33)

    def test_attention_duplicate_indices_preserve_head_order(self) -> None:
        records = [
            (key_position, head * 2 + key_position)
            for head in range(14)
            for key_position in range(2)
        ]
        decoded = decode_stage_records(records, stage=9, token=1)
        self.assertEqual(decoded.tolist(), list(range(28)))

    def test_attention_order_mismatch_fails_closed(self) -> None:
        records = [(index & 1, index) for index in range(28)]
        records[0], records[1] = records[1], records[0]
        with self.assertRaisesRegex(DiagnosticError, "ordering"):
            decode_stage_records(records, stage=8, token=1)


class DistributionTests(unittest.TestCase):
    def test_reports_sparse_affected_dimension_and_ulp(self) -> None:
        bits = np.asarray([0x3C00, 0x6400], dtype="<u2")
        result = distribution(
            bits,
            np.asarray([1.0, 1023.6], dtype=np.float64),
            stage=18,
            token=0,
        )
        self.assertEqual(result["count_abs_gt_0_1"], 1)
        self.assertEqual(
            result["affected_coordinates_abs_gt_0_1"],
            [{"dimension": 1}],
        )
        self.assertLessEqual(result["fp16_ulps_at_worst"], 1.0)


if __name__ == "__main__":
    unittest.main()
