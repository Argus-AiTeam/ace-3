#!/usr/bin/env python3
"""Focused checks for the Q26 RMSNorm divisor correction."""

from __future__ import annotations

import unittest

from ace3.model.fp16_adaptation_oracle import rmsnorm


class RMSNormQ26Tests(unittest.TestCase):
    def test_q26_distinguishing_vector(self) -> None:
        activations = [
            0xBC00,
            0x1000,
            0x2800,
            0xB800,
            0xC000,
            0x3A00,
            0x3800,
            0xC000,
        ]
        weights = [
            0x3A00,
            0xBC00,
            0x3800,
            0x3E00,
            0x3C00,
            0x3A00,
            0x3800,
            0x3800,
        ]
        outputs, mean_q48, rms_q24 = rmsnorm(activations, weights)
        self.assertEqual(mean_q48, 354077393745825)
        self.assertEqual(rms_q24, 18816944)
        self.assertEqual(
            [item[0] for item in outputs],
            [0xB959, 0x8F22, 0x2322, 0xB959, 0xBF22, 0x3803, 0x3322, 0xBB22],
        )
        self.assertNotEqual(
            [item[0] for item in outputs],
            [0xB95A, 0x8F22, 0x2322, 0xB95A, 0xBF22, 0x3803, 0x3322, 0xBB22],
        )


if __name__ == "__main__":
    unittest.main()
