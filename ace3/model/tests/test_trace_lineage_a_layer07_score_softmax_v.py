import unittest

from trace_lineage_a_layer07_score_softmax_v import (
    PROBABILITY_PARTS, score_bits, softmax_components, v_projection,
)


class ScoreSoftmaxVTests(unittest.TestCase):
    def test_equal_and_shifted_scores(self):
        rows = softmax_components([0x4000]*4, [0]*4, [0x3400]*4, [0x3400]*4)
        self.assertTrue(all(not any(r["components_q24"].values()) for r in rows))

    def test_local_error_not_absorbed(self):
        row = softmax_components([0, 0], [0, 0], [0x3801, 0x3800], [0x3800]*2)[0]
        self.assertNotEqual(row["components_q24"]["local_softmax"], 0)
        self.assertTrue(all(row["components_q24"][key] == 0
                            for key in PROBABILITY_PARTS if key != "local_softmax"))

    def test_causal_row_extreme_delta_and_negative_scores(self):
        rows = softmax_components([0, 0xdc00], [0, 0xdc00],
                                  [0x3c00, 0], [0x3c00, 0])
        self.assertTrue(all(not any(r["components_q24"].values()) for r in rows))
        negative = softmax_components([0xbc00]*2, [0xbc00]*2, [0x3800]*2, [0x3800]*2)
        self.assertTrue(all(not any(r["components_q24"].values()) for r in negative))

    def test_score_cancellation_and_scale(self):
        self.assertEqual(score_bits([0x3c00]*64, [0x3c00, 0xbc00]*32), (0, 0))
        self.assertEqual(score_bits([0x3c00]*64, [0x3c00]*64), (64 << 48, 0x4800))

    def test_native_lane_order_no_zero_plus_one_and_bias(self):
        tensors = {"qweight": [0x76543210]*128, "qzeros": [0x76543210],
                   "scales": [0x3c00]*8, "bias": [0xbc00]*8}
        for lane in range(8):
            two, single, measured = v_projection([0x3c00]*128, tensors, lane)
            self.assertEqual((two, single, measured["actual_q48"]), (0xbc00, 0xbc00, 0))
        tensors["qweight"] = [0x87654321]*128
        for lane in range(8):
            two, single, _ = v_projection([0x3c00]*128, tensors, lane)
            self.assertEqual((two, single), (0x57f0, 0x57f0))

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            softmax_components([0x7c00], [0], [0x3c00], [0x3c00])
        with self.assertRaises(ValueError):
            score_bits([0]*63, [0]*64)
