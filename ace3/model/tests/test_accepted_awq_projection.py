from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import decoder_layer0_oracle  # noqa: E402
from accepted_awq_projection import project_fp16_stage  # noqa: E402


class AcceptedAwqProjectionTests(unittest.TestCase):
    def test_native_asymmetric_packing_and_fp16_boundary(self) -> None:
        activations = np.full(128, 0x3C00, dtype="<u2")
        qweight = np.zeros((128, 1), dtype="<u4")
        qzeros = np.zeros((1, 1), dtype="<u4")
        for lane, physical in enumerate((0, 4, 1, 5, 2, 6, 3, 7)):
            qweight[:, 0] |= np.uint32((lane + 2) << (4 * physical))
            qzeros[:, 0] |= np.uint32(2 << (4 * physical))
        scales = np.full(8, 0x3000, dtype="<u2")
        bias = np.full(8, 0x3C00, dtype="<u2")

        result = project_fp16_stage(
            activations,
            qweight.reshape(-1),
            qzeros.reshape(-1),
            scales,
            8,
            bias,
        )

        expected = np.asarray(
            [np.float16(1.0 + lane * 128 * 0.125) for lane in range(8)],
            dtype="<f2",
        ).view("<u2")
        np.testing.assert_array_equal(result, expected)

    def test_shared_q_projection_path_uses_accepted_boundary(self) -> None:
        activations = [0x3C00] * 128
        qweight = np.zeros((128, 1), dtype="<u4")
        qzeros = np.zeros((1, 1), dtype="<u4")
        for lane, physical in enumerate((0, 4, 1, 5, 2, 6, 3, 7)):
            qweight[:, 0] |= np.uint32((lane + 2) << (4 * physical))
            qzeros[:, 0] |= np.uint32(2 << (4 * physical))
        values = {
            "model.layers.0.self_attn.q_proj.qweight:": (
                qweight.reshape(-1).tolist()
            ),
            "model.layers.0.self_attn.q_proj.qzeros:": (
                qzeros.reshape(-1).tolist()
            ),
            "model.layers.0.self_attn.q_proj.scales:": [0x3000] * 8,
        }

        result = decoder_layer0_oracle._module(
            values,
            "self_attn.q_proj",
            activations,
            8,
            [0x3C00] * 8,
        )

        expected = [
            int(value)
            for value in np.asarray(
                [np.float16(1.0 + lane * 128 * 0.125) for lane in range(8)],
                dtype="<f2",
            ).view("<u2")
        ]
        self.assertEqual(result, expected)

    def test_rejects_projection_geometry_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "qweight geometry mismatch"):
            project_fp16_stage(
                [0x3C00] * 128,
                [0] * 127,
                [0],
                [0x3C00] * 8,
                8,
            )


if __name__ == "__main__":
    unittest.main()
