"""Synthetic entry, pinned-review fixtures and rational oracle, never retained science."""

from fractions import Fraction as F
import hashlib
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_value_content_stage13_rmsnorm_integer_contrast_v1 as d


suffix_oracle = d.retained.module_from(d.retained.ORACLE, "stage13_independent_suffix_oracle")


def verify_norm(words, weights, actual, details):
    decoded = [suffix_oracle.scalar(w) * (1 << 24) for w in words]
    mean = round(sum((x * x for x in decoded), F()) / 896 + 281474977)
    root = math.isqrt(mean)
    assert details == {"mean_q48": mean, "root_q24": root}, "integer RMSNorm mean/root"
    for i, weight in enumerate(weights.view("<u2")):
        integer = round(decoded[i] * suffix_oracle.scalar(weight) * (1 << 24) / root)
        expected = suffix_oracle.round_word(
            F(integer, 1 << 24), bool((int(words[i]) ^ int(weight)) & 0x8000))
        assert int(actual[i]) == expected, f"integer RMSNorm coordinate {i}"


def fixture():
    ones = np.full(896, 0x3c00, dtype="<u2")
    zeros = np.zeros(896, dtype="<u2")
    raw = {"input_hidden": ones.copy(), "input_i": np.full(896, 1 << 24, dtype="<i8"),
           "input_z": np.zeros(896, dtype="u1"), "scratch_i": np.full(896, 1 << 24, dtype="<i8"),
           "scratch_z": np.zeros(896, dtype="u1"), "output_i": np.full(896, 1 << 24, dtype="<i8"),
           "output_z": np.zeros(896, dtype="u1"), "retained_stage07": np.zeros(128, dtype="<u2"),
           "final_logits": np.zeros(2, dtype="<u2")}
    for s, size in d.layer.prior.local.SIZES.items():
        raw[f"stage{s:02d}"] = np.zeros(size, dtype="<u2")
    raw.update(stage09=np.full(14, 0x3c00, dtype="<u2"), stage12=ones.copy(),
               stage13=ones.copy(), stage18=ones.copy())
    for kind in ("k", "v"):
        raw["input_cache_" + kind] = np.zeros((0, 128), dtype="<u2")
        raw["output_cache_" + kind] = np.zeros((1, 128), dtype="<u2")
    tensors = {d.PREFIX + "post_attention_layernorm.weight": np.ones(896, dtype="<f2")}
    for name, inputs, outputs in (("self_attn.o_proj", 896, 896), ("mlp.gate_proj", 896, 4864),
                                  ("mlp.up_proj", 896, 4864), ("mlp.down_proj", 4864, 896)):
        name = d.PREFIX + name
        tensors[name + ".qweight"] = np.zeros((inputs, outputs // 8), dtype="<i4")
        tensors[name + ".qzeros"] = np.zeros((inputs // 128, outputs // 8), dtype="<i4")
        tensors[name + ".scales"] = np.ones((inputs // 128, outputs), dtype="<f2")
    operands = (np.ones(896, dtype="<f2"), np.zeros((2, 896), dtype="<f2"))
    operands[1][0] = 1 / 1024
    refs = {"fp16": np.zeros(151936, dtype="<u2"), "binary64": np.zeros(151936, dtype="<f8")}
    refs["fp16"][34319], refs["binary64"][34319] = 0x4000, 2
    classified = {"control": d.CONTROLS[0], "paired_margins": [
        {"pair": d.PAIR, "reference_branch": b, "retained_actual_margin": "0",
         "substituted_actual_margin": "0", "retained_reference_margin": "2",
         "required_direction": "2"} for b in ("fp16", "binary64")]}
    return raw, tensors, operands, refs, classified


class IntegerContrastTests(unittest.TestCase):
    def test_actual_retained_stage13_entry_through_native_suffix_and_gates(self):
        raw, tensors, operands, refs, classified = fixture()
        protected = {k: d.retained.digest(v) for k, v in raw.items()}
        for array in (*raw.values(), *tensors.values(), *operands):
            array.flags.writeable = False
        oracle = type("Oracle", (), {"verify_norm": staticmethod(verify_norm), "suffix_oracle": suffix_oracle})
        audit = d.new_audit()
        with d.guarded(d.OUTPUT, audit):
            payload, row = d.run_control(
                d.CONTROLS[0], raw, {"control": d.CONTROLS[0], "failure_count": 0},
                classified, tensors, raw, np.ones(896, dtype="<f8"), refs, operands, oracle, audit)
        self.assertEqual(row["component_oracles"]["status"], "PASS")
        self.assertEqual([r["stage"] for r in row["stages"]], list(range(10, 19)))
        self.assertTrue(all(r["status"] == "PASS" for r in row["stages"]))
        self.assertTrue(all(m["direction_pass"] for m in row["paired_margins"]))
        self.assertEqual(audit["stage_dispatches"], [[d.CONTROLS[0], 23, 0, s] for s in range(13, 19)])
        self.assertEqual(audit["forbidden_events"], [])
        self.assertEqual(protected, {k: d.retained.digest(payload[k]) for k in raw})
        self.assertIn("integer_contrast_stage13", payload)
        self.assertIn("integer_contrast_output_i", payload)

    def test_integer_semantics_signed_zero_subnormal_and_dynamic_range(self):
        words = np.resize(np.array([0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff],
                                   dtype="<u2"), 896)
        weights = np.resize(np.array([1, -1, 0.5, -0.5], dtype="<f2"), 896)
        actual, details = d.stage13(words, weights)
        verify_norm(words, weights, actual, details)
        actual[62] ^= 1
        with self.assertRaisesRegex(AssertionError, "coordinate 62"):
            verify_norm(words, weights, actual, details)

    def test_invalid_actual_entry_schema_and_nonfinite(self):
        for words, weights in (
                (np.zeros(895, dtype="<u2"), np.ones(896, dtype="<f2")),
                (np.zeros(896, dtype="<f2"), np.ones(896, dtype="<f2")),
                (np.full(896, 0x7c00, dtype="<u2"), np.ones(896, dtype="<f2")),
                (np.zeros(896, dtype="<u2"), np.full(896, np.nan, dtype="<f2"))):
            with self.assertRaises(ValueError):
                d.stage13(words, weights)

    def test_reference_free_entry_signatures(self):
        self.assertEqual(list(inspect.signature(d.stage13).parameters), ["words", "gamma"])
        self.assertEqual(list(inspect.signature(d.compute_suffix).parameters), ["raw", "tensors", "operands"])
        text = inspect.getsource(d.compute_suffix)
        for forbidden in ("reference", "nearest", "compare(", "margin", "classified"):
            self.assertNotIn(forbidden, text.split('"""')[-1])

    def test_protected_state_and_kv_defects(self):
        for key in ("stage12", "output_i", "output_cache_v", "stage03"):
            raw, _, _, _, _ = fixture()
            raw[key].flat[0] ^= (1 << 14) if key == "output_i" else 1
            with self.assertRaises(ValueError):
                d.prior.check_state(raw)
        raw, _, _, _, _ = fixture()
        raw["output_i"][0] += 1
        d.prior.check_state(raw)

    def test_exact_s18_threshold(self):
        words = np.full(896, 0x3c00, dtype="<u2")
        self.assertEqual(d.prior.compare(words, np.full(896, 1.125, dtype="<f8"))["status"], "PASS")
        self.assertEqual(d.prior.compare(words, np.full(896, 1.126953125, dtype="<f8"))["status"], "FAIL")

    def test_zero_opposite_and_correct_original_reference_movement(self):
        raw, _, _, refs, classified = fixture()
        for word, expected in ((0, False), (0xbc00, False), (0x3c00, True)):
            rows = d.retained.paired_margins(np.array([word, 0], dtype="<u2"), raw["final_logits"],
                                           classified["paired_margins"], refs)
            self.assertTrue(all(m["direction_pass"] == expected for m in rows))

    def test_numerical_oracle_or_direction_failure_is_rejected(self):
        def rows():
            return [{"control": c, "stages": [{"stage": s, "status": "PASS"} for s in range(10, 19)],
                     "S18": {"status": "PASS"}, "component_oracles": {"status": "PASS"},
                     "paired_margins": [{"direction_pass": True}, {"direction_pass": True}]}
                    for c in d.CONTROLS]
        self.assertEqual(d.scientific_status(rows()), "SUPPORTED")
        for defect in ("gate", "oracle", "direction"):
            result = rows()
            if defect == "gate":
                result[0]["stages"][3]["status"] = "FAIL"
            elif defect == "oracle":
                result[0]["component_oracles"]["status"] = "FAIL"
            else:
                result[0]["paired_margins"][1]["direction_pass"] = False
            self.assertEqual(d.scientific_status(result), "REJECTED")

    def test_no_producer_reference_or_output_selection_dispatch(self):
        for function, args in (
                (d.prior.execute, ()), (d.retained.execute, ()),
                (d.retained.selection.nearest, (None,) * 3), (d.retained.bridge.check, ()),
                (d.native.projection, ({}, d.PREFIX + "self_attn.o_proj", None))):
            audit = d.new_audit()
            with d.guarded(d.OUTPUT, audit):
                with self.assertRaises(ValueError):
                    function(*args)
            self.assertTrue(audit["forbidden_events"])

    def test_exclusive_write_and_process_guards(self):
        for event, args in (("subprocess.Popen", ()), ("socket.connect", ()),
                            ("open", (str(d.SOURCE), "w", 577)),
                            ("open", (str(d.OUTPUT / "occupied.npz"), "w", 577))):
            audit = d.new_audit()
            with d.guarded(d.OUTPUT, audit):
                with self.assertRaises(ValueError):
                    d.prior.audit_event(event, args)
            self.assertTrue(audit["forbidden_events"])


class PinnedReviewTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        handoffs = Path(directory.name).resolve()
        self.review_path = handoffs / "9602adc7f96e/round-0001.json"
        self.review_path.parent.mkdir()
        self.latest_path = self.review_path.with_name("latest.json")
        self.latest_path.write_text(json.dumps({"handoff": {"path": str(self.review_path)}}))
        self.review = {"mission_id": "9602adc7f96e", "producer_role": "reviewer",
                       "review": {"status": "done"}}
        self.review_path.write_text(json.dumps(self.review))
        digest = hashlib.sha256(self.review_path.read_bytes()).hexdigest()
        for target, name, value in ((d.retained, "HANDOFFS", handoffs),
                                    (d, "RTZ_REVIEW_HASH", digest)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def read_review(self):
        audit = d.new_audit()
        with d.guarded(d.OUTPUT, audit):
            records = d.rtz_review_pins()
        self.assertEqual(audit["forbidden_events"], [])
        self.assertEqual(audit["writes"], [])
        self.assertEqual(audit["stage_dispatches"], [])
        return records

    def test_exact_external_review_uses_real_pinned_reader(self):
        self.assertFalse(self.review_path.is_relative_to(d.ROOT))
        records = self.read_review()
        self.assertEqual(records[0], {"path": str(self.review_path), "sha256": d.RTZ_REVIEW_HASH})
        self.assertEqual(records[1], d.pin(self.latest_path))

    def test_wrong_review_hash_is_rejected(self):
        with mock.patch.object(d, "RTZ_REVIEW_HASH", "0" * 64):
            with self.assertRaisesRegex(ValueError, "retained hash/size mismatch"):
                self.read_review()

    def test_hash_matching_wrong_review_identity_is_rejected(self):
        for field, value in (("mission_id", "69c9d9ec6bbe"), ("producer_role", "engineer"),
                             ("review", {"status": "continue"})):
            with self.subTest(field=field):
                self.review_path.write_text(json.dumps({**self.review, field: value}))
                digest = hashlib.sha256(self.review_path.read_bytes()).hexdigest()
                with mock.patch.object(d, "RTZ_REVIEW_HASH", digest):
                    with self.assertRaisesRegex(ValueError, "independent RTZ review missing"):
                        self.read_review()

    def test_latest_cannot_select_another_hash_matching_path(self):
        other = self.review_path.with_name("round-0002.json")
        other.write_bytes(self.review_path.read_bytes())
        self.latest_path.write_text(json.dumps({"handoff": {"path": str(other)}}))
        with self.assertRaisesRegex(ValueError, "independent RTZ review missing"):
            self.read_review()

    def test_symlinked_review_is_rejected(self):
        other = self.review_path.with_name("other.json")
        self.review_path.rename(other)
        self.review_path.symlink_to(other)
        with self.assertRaisesRegex(ValueError, "noncanonical retained pin"):
            self.read_review()

    def test_relative_review_path_is_rejected(self):
        with mock.patch.object(d.retained, "HANDOFFS", Path("relative-handoffs")):
            with self.assertRaisesRegex(ValueError, "noncanonical retained pin"):
                self.read_review()


if __name__ == "__main__":
    unittest.main()
