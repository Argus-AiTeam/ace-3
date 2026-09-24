import unittest
from pathlib import Path

import diagnose_lineage_a_position3_boundary as boundary
from retained_causal_repair_campaign import (
    compare, paired_cone, propagate_post_attention, quotient_only_rmsnorm, records,
    retained_layer_package,
    projection_contributions, root_only_rmsnorm, round_sqrt_ratio,
)
from fp16_adaptation_oracle import EPSILON_Q48, rmsnorm


class RetainedCampaignTests(unittest.TestCase):
    def test_gate_preserves_strict_relative_and_ulp(self):
        self.assertEqual(compare([0xdc24], [0xdc25], boundary)["material_failures"], 0)
        failure = compare([0xdc23], [0xdc25], boundary)
        self.assertEqual(failure["material_failures"], 1)
        self.assertEqual(failure["failures"][0]["absolute_error_exact"], "1/2")
        self.assertEqual(failure["failures"][0]["ordered_FP16_ULP"], 2)

    def test_failed_stage_is_not_consumed(self):
        calls = []

        def projection(stage, values):
            calls.append(stage)
            return [0x4000]

        output, comparisons, stopped = propagate_post_attention(
            [0x3c00], [0x3c00], projection,
            lambda a, b: self.fail("failed projection reached SiLU"),
            lambda a, b: self.fail("failed projection reached residual"),
            {str(s): [0x3c00] for s in range(13, 19)}, boundary)
        self.assertEqual(stopped, 14)
        self.assertEqual(calls, [14])
        self.assertEqual(set(output), {12, 13, 14})
        self.assertEqual(len(comparisons), 2)

    def test_passing_cone_reaches_endpoint(self):
        calls = []

        def projection(stage, values):
            calls.append(stage)
            return list(values)

        output, comparisons, stopped = propagate_post_attention(
            [0x3c00], [0x3c00], projection, lambda a, b: a, lambda a, b: b,
            {str(s): [0x3c00] for s in range(13, 19)}, boundary)
        self.assertIsNone(stopped)
        self.assertEqual(calls, [14, 15, 17])
        self.assertEqual(output[18], [0x3c00])
        self.assertEqual(len(comparisons), 6)

    def test_bad_geometry_and_nonfinite_fail_explicitly(self):
        for actual, reference in (([], []), ([0], []), ([0x7c00], [0])):
            with self.assertRaises(ValueError):
                compare(actual, reference, boundary)

    def test_nested_binding_records(self):
        rec = dict(path="/not/read", bytes=3, sha256="digest")
        self.assertEqual(list(records({"roots": [rec]})), [rec])

    def test_quotient_keeps_native_reduction_and_signed_zero(self):
        for values in ([0, 0x8000], [0x3c00, 0xbc00], [1, 0x8001], [0x0400, 0x8400]):
            weights = [0x3c00, 0x3c00]
            outputs, reduction = quotient_only_rmsnorm(values, weights, EPSILON_Q48, boundary)
            _, mean, root = rmsnorm(values, weights)
            self.assertEqual(reduction["mean_q48"], mean)
            self.assertEqual(reduction["rms_q24"], root)
            self.assertEqual([v & 0x8000 for v in outputs], [v & 0x8000 for v in values])
        self.assertEqual(quotient_only_rmsnorm([0, 0x8000], [0x3c00]*2,
                                              EPSILON_Q48, boundary)[0], [0, 0x8000])

    def test_quotient_nonfinite_and_geometry_rejected(self):
        for a, w in (([], []), ([0], []), ([0x7c00], [0x3c00]), ([0], [0xfc00])):
            with self.assertRaises(ValueError):
                quotient_only_rmsnorm(a, w, EPSILON_Q48, boundary)

    def test_root_only_exact_midpoints(self):
        for product, expected in ((0, 0), (1, 0), (3, 2), (5, 2), (7, 4), (-7, 4)):
            self.assertEqual(round_sqrt_ratio(product, 4), expected)
        self.assertEqual(round_sqrt_ratio(1, 3), 1)
        self.assertEqual(round_sqrt_ratio(1, 5), 0)
        with self.assertRaises(ValueError):
            round_sqrt_ratio(1, 0)

    def test_root_only_keeps_native_mean_and_q24_rounding(self):
        from decimal import Decimal, localcontext
        for values in ([0, 0x8000], [0x3c00, 0xbc00], [1, 0x8001],
                       [0x0400, 0x8400], [0x3bff, 0x3c01]):
            weights = [0x3c00, 0xbc00]
            outputs, detail = root_only_rmsnorm(values, weights, EPSILON_Q48, boundary)
            _, mean, root = rmsnorm(values, weights)
            self.assertEqual((detail["mean_q48"], detail["rms_q24"]), (mean, root))
            with localcontext() as context:
                context.prec = 120
                expected = [round(Decimal(boundary.units(a))*Decimal(boundary.units(w))
                                  / Decimal(mean).sqrt())
                            for a, w in zip(values, weights, strict=True)]
            self.assertEqual(detail["quotient_RNE_q24"], expected)
            self.assertEqual([v & 0x8000 for v in outputs],
                             [(a ^ w) & 0x8000 for a, w in zip(values, weights, strict=True)])
        for a, w in (([], []), ([0], []), ([0x7c00], [0x3c00]),
                     ([0], [0xfc00]), ([0x3c00, 0], [0x7bff, 0x7bff])):
            with self.assertRaises(ValueError):
                root_only_rmsnorm(a, w, EPSILON_Q48, boundary)

    def test_native_gate_up_contributions(self):
        shifts = (0, 16, 4, 20, 8, 24, 12, 28)
        packed = sum(i << shifts[i] for i in range(8))
        tensors = dict(qweight=[packed]*256, qzeros=[0x22222222, 0x33333333],
                       scales=[0x3c00]*8 + [0x3800]*8)
        for index in range(8):
            rows = projection_contributions([0x3c00]*128 + [0xbc00]*128,
                                            tensors, index, boundary)
            self.assertEqual(sum(r["contribution_q48"] for r in rows),
                             128*(index-2)*2**48 - 128*(index-3)*2**47)
            self.assertEqual((rows[0]["qweight"], rows[0]["qzero"]), (index, 2))
            self.assertEqual(rows[128]["qzero"], 3)

    def test_root_only_separate_history_and_stop(self):
        actual = {p: {s: [0x3c00] for s in range(19)} for p in range(2)}
        saved, seen = [], []

        def operator(stage, stages, incoming, cache, p):
            if stage in (4, 5):
                return stages[1 if stage == 4 else 2]
            if stage in (6, 7):
                return stages[5 if stage == 6 else 3]
            if stage == 8:
                seen.append((p, tuple(tuple(v) for v in cache["k"])))
                return [0x4000] if p == 1 and cache["k"][0] == [0x3c01] else [0x3c00]
            return [0x3c00]

        endpoints, comparisons, failure = paired_cone(
            actual, actual, {p: [0x3c00] for p in actual}, {p: [0x3c00] for p in actual},
            operator, lambda stage, a, b: (list(a), list(b)),
            lambda *args: [0x3c01], boundary, lambda p, values: saved.append(values),
            candidate_name="root_only")
        self.assertEqual(set(endpoints), {0})
        self.assertEqual((failure["position"], failure["stage"]), (1, 8))
        self.assertEqual(failure["changed_history"], [0])
        self.assertIn((1, ((0x3c01,), (0x3c01,))), seen)
        self.assertEqual(set(saved[-1]["root_only"]), set(range(9)))
        self.assertEqual(saved[-1]["frozen_Q48"][8], [0x3c00])

    def test_paired_cone_stops_before_any_failed_dependency(self):
        actual = {0: {s: [0x3c00] for s in range(19)}}
        saved = []
        endpoints, comparisons, failure = paired_cone(
            actual, actual, {0: [0x3c00]}, {0: [0x3c00]},
            lambda *args: [0x3c00],
            lambda *args: self.fail("failed RMSNorm reached projection"),
            lambda *args: [0x4000], boundary, lambda p, values: saved.append(values))
        self.assertEqual(endpoints, {})
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(failure["stage"], 0)
        self.assertEqual(set(saved[0]["quotient_only"]), {0})

    def test_retained_launch_schemas_all_layers(self):
        build = Path("/retained")
        for layer in range(1, 9):
            with self.subTest(layer=layer):
                directory = build / (
                    "token358_position3_full_continuation_attempt"
                    + ("001" if layer == 2 else "003")) / f"layer{layer:02d}"
                package_path = (
                    build / "token358_position3_layer1_preflight_attempt002/package/launch_package.json"
                    if layer == 1 else directory / "prepared.json")
                transaction = (
                    build / "token358_position3_layer1_attempt002/evidence/transaction.json"
                    if layer == 1 else directory / "transaction.json")
                stages = {str(s): {"stage": s} for s in range(19)}
                package = dict(layer_index=layer, position=3, kv_parent=dict(
                    layer_index=layer, valid_positions=[0, 1, 2],
                    state={"path": "/retained/model24_persistent_kv_selected_token_corrected_q_attempt005/state"}))
                package["independent_expected_stages" if layer == 1 else "independent_stages"] = stages
                files = {package_path: package}
                if layer >= 3:
                    files[directory / "result.json"] = dict(
                        layer=layer, position=3, transaction={"path": str(transaction)})
                normalized, got_transaction, got_path = retained_layer_package(layer, build, files.__getitem__)
                self.assertEqual((got_transaction, got_path), (transaction, package_path))
                self.assertEqual(normalized["independent_stages"], stages)
                package["kv_parent"]["layer_index"] = layer + 1
                with self.assertRaisesRegex(ValueError, "historical KV mismatch"):
                    retained_layer_package(layer, build, files.__getitem__)

    def test_control_reconstruction_failure_is_recorded_without_advancing(self):
        actual = {0: {s: [0x3c00] for s in range(19)}}
        saved = []
        endpoints, comparisons, failure = paired_cone(
            actual, actual, {0: [0x3c00]}, {0: [0x3c00]},
            lambda *args: [0x3c01],
            lambda *args: self.fail("mismatched control reached projection"),
            None, boundary, lambda p, values: saved.append(values))
        self.assertEqual(endpoints, {})
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(failure["failure_taxonomy"], "frozen_runtime_reconstruction_mismatch")
        self.assertEqual(failure["retained_runtime"], [0x3c00])
        self.assertEqual(set(saved[0]["frozen_Q48"]), {0})


if __name__ == "__main__":
    unittest.main()
