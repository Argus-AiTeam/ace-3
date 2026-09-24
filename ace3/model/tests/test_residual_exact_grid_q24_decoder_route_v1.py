"""Synthetic state/route regressions; no model or decoder simulation claims."""

from dataclasses import replace
import hashlib
import json
import unittest

from ace3.model.candidates import residual_exact_grid_q24_reference_v1 as oracle
from ace3.model.candidates import residual_exact_grid_q24_runtime_v1 as codec
from ace3.model.candidates import residual_exact_grid_q24_decoder_route_v1 as route


def metadata(layer):
    return dict(codec.DECODER_CONSTANTS, slot=0, position=0, next_layer=layer,
                model_id="Qwen/Qwen2.5-0.5B-Instruct-AWQ",
                root_id="synthetic-unit-root", token_history_id="synthetic-unit-history",
                producer_id=f"synthetic-producer-{layer}", kv_lineage_id=f"synthetic-kv-{layer}")


def root_parent():
    return route.paired_parent([(1 << 24, 0)] * 896, metadata(0), [0x3C00] * 896)


class DecoderCodecTests(unittest.TestCase):
    def test_decoder_profile_is_explicit_and_lossless(self):
        parent = root_parent()
        self.assertEqual(len(bytes.fromhex(json.loads(parent.text)["payload_hex"])), 8064)
        self.assertEqual(parent.records(), ((1 << 24, 0),) * 896)
        with self.assertRaises(ValueError):
            codec.decode(parent.text, trusted_metadata=parent.metadata)
        with self.assertRaises(ValueError):
            codec.encode(parent.records(), parent.metadata)

    def test_views_match_independent_rational_oracle(self):
        states = [0, 1, 1023, 1024, (1 << 24) + 8192, 65520 * (1 << 24) - 1]
        for integer in states + [-i for i in states]:
            self.assertEqual(codec.fp16_view(integer, 0), oracle.project(integer, 0))
        self.assertEqual(codec.fp16_view(0, 1), 0x8000)
        for integer, tag in ((1, 1), (0, 2), (65520 << 24, 0)):
            with self.assertRaises(ValueError):
                codec.fp16_view(integer, tag)

    def test_no_invented_root_even_when_fp16_view_matches(self):
        parent = route.paired_parent([((1 << 24) + 1, 0)] * 896,
                                     metadata(0), [0x3C00] * 896)
        with self.assertRaisesRegex(ValueError, "invented model-root"):
            parent.records()

    def test_digest_owner_and_fp16_only_rejection(self):
        parent = root_parent()
        bad = replace(parent, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            bad.records()
        bad = replace(parent, metadata=dict(parent.metadata, root_id="other"))
        with self.assertRaisesRegex(ValueError, "producer/owner"):
            bad.records()
        text = json.dumps(dict(parent.metadata, hidden=[0x3C00] * 896))
        bad = replace(parent, text=text, sha256=hashlib.sha256(text.encode()).hexdigest())
        with self.assertRaisesRegex(ValueError, "FP16-only"):
            bad.records()
        with self.assertRaisesRegex(ValueError, "H/state disagreement"):
            replace(parent, hidden=(0x3C01,) * 896).records()


class OrderedRouteTests(unittest.TestCase):
    def setUp(self):
        self.dispatched = []
        self.persisted = []

    def execute(self, layer, parent):
        self.dispatched.append((layer, parent.sha256))
        updates = [oracle.add(i, z, 0x1000) for i, z in parent.records()]
        produced = route.paired_parent(
            [(i, z) for i, z, _ in updates], metadata(layer + 1),
            [h for _, _, h in updates])
        # A mock of the capture interface, explicitly not simulator evidence.
        return route.LayerOutput(produced, layer, parent.sha256, 0, True, True, 0,
                                 "actual_decoder_rtl")

    def admit(self, parent, output):
        return dict.fromkeys(("local_operator_fp16", "binary64_v1", "residual_state_lineage",
                              "kv_lineage", "source_runtime"), "PASS") | {
            "input_sha256": parent.sha256, "output_sha256": output.parent.sha256}

    def persist(self, output, report):
        self.persisted.append(output.parent.sha256)

    def test_nine_ordered_actual_parent_handoffs_in_mock_route(self):
        root = root_parent()
        last = route.run_root_through_l8(root, self.execute, self.admit, self.persist)
        self.assertEqual([i for i, _ in self.dispatched], list(range(9)))
        self.assertEqual([p for _, p in self.dispatched],
                         [root.sha256] + self.persisted[:-1])
        self.assertEqual(last.metadata["next_layer"], 9)
        self.assertEqual(last.records(), (((1 << 24) + 9 * 8192, 0),) * 896)
        self.assertEqual(len(self.persisted), 9)

    def test_failed_gate_prevents_next_dispatch_and_commit(self):
        def fail(parent, output):
            report = self.admit(parent, output)
            if output.layer == 2:
                report["binary64_v1"] = "FAIL"
            return report
        with self.assertRaisesRegex(ValueError, "L2:.*binary64_v1"):
            route.run_root_through_l8(root_parent(), self.execute, fail, self.persist)
        self.assertEqual(len(self.dispatched), 3)
        self.assertEqual(len(self.persisted), 2)

    def test_source_state_and_runtime_faults_fail_closed(self):
        for change in (
            {"input_sha256": "0" * 64}, {"layer": 1}, {"prior_kv_count": 1},
            {"completed": False}, {"idle": False}, {"fault_code": 3},
            {"evidence_kind": "software_screen"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                route.run_root_through_l8(
                    root_parent(),
                    lambda layer, parent: replace(self.execute(layer, parent), **change),
                    self.admit, self.persist)
        self.assertEqual(self.persisted, [])

    def test_missing_and_unbound_admission_cannot_forward(self):
        for report in ({}, {"local_operator_fp16": "PASS"},
                       dict.fromkeys(("local_operator_fp16", "binary64_v1",
                                      "residual_state_lineage", "kv_lineage",
                                      "source_runtime"), "PASS")):
            with self.subTest(report=report), self.assertRaises(ValueError):
                route.run_root_through_l8(root_parent(), self.execute,
                                         lambda parent, output: report, self.persist)
        self.assertEqual(self.persisted, [])


if __name__ == "__main__":
    unittest.main()
