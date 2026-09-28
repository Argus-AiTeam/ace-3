"""Synthetic tests and independent exact RTZ oracle; never replay retained science."""

from bisect import bisect_right
from fractions import Fraction as F
import inspect
import struct
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_value_content_s18_rtz_rounding_contrast_v1 as d


suffix_oracle = d.retained.module_from(d.retained.ORACLE, "rtz_independent_suffix_oracle")
LATTICE = tuple(F(struct.unpack("<e", struct.pack("<H", w))[0]) for w in range(0x7c00))


def oracle_word(integer, tag):
    magnitude = F(abs(integer), 1 << 24)
    word = bisect_right(LATTICE, magnitude) - 1
    return word | (0x8000 if integer < 0 or (integer == 0 and tag) else 0)


def verify_rtz(integers, tags, words):
    assert len(integers) == len(tags) == len(words) == 896
    for integer, tag, word in zip(integers, tags, words, strict=True):
        assert int(word) == oracle_word(int(integer), int(tag))


class RtzTests(unittest.TestCase):
    def test_every_finite_cell_both_signs_and_q24_endpoints(self):
        for word, value in enumerate(LATTICE):
            low = int(value * (1 << 24))
            high = (int(LATTICE[word + 1] * (1 << 24)) - 1
                    if word + 1 < len(LATTICE) else 65520 * (1 << 24) - 1)
            for integer in {low, high, (low + high) // 2}:
                for signed in {integer, -integer}:
                    self.assertEqual(d.rtz_word(signed, 0), oracle_word(signed, 0))
        self.assertEqual(d.rtz_word(0, 1), 0x8000)

    def test_invalid_native_state_and_vector_schema(self):
        for integer, tag in ((0, 2), (1, 1), (-1, 1), (65520 << 24, 0),
                             (-(65520 << 24), 0), (1 << 63, 0), (1.0, 0)):
            with self.assertRaises(ValueError):
                d.rtz_word(integer, tag)
        with self.assertRaises(d.retained.bridge.BindingError):
            d.rtz_view(np.zeros(895, dtype="<i8"), np.zeros(895, dtype="u1"))

    def test_reference_independent_function_signature(self):
        self.assertEqual(list(inspect.signature(d.rtz_view).parameters), ["integers", "tags"])
        i = np.full(896, (1 << 24) + 16000, dtype="<i8")
        z = np.zeros(896, dtype="u1")
        words = d.rtz_view(i, z)
        d.compare(words, np.ones(896, dtype="<f8"))
        d.compare(words, np.full(896, 2, dtype="<f8"))
        np.testing.assert_array_equal(words, d.rtz_view(i, z))

    def fixture(self, control):
        integers = np.full(896, (1 << 24) + 16000, dtype="<i8")
        tags = np.zeros(896, dtype="u1")
        h = np.array([d.state.fp16_view(int(i), int(z)) for i, z in zip(integers, tags)],
                     dtype="<u2")
        raw = {"stage18": h, "stage12": h.copy(), "input_hidden": h.copy(),
               "input_i": integers.copy(), "input_z": tags.copy(),
               "scratch_i": integers.copy(), "scratch_z": tags.copy(),
               "output_i": integers, "output_z": tags, "final_logits": np.zeros(2, dtype="<u2"),
               "stage03": np.zeros(128, dtype="<u2"), "stage06": np.zeros(128, dtype="<u2"),
               "retained_stage07": np.zeros(128, dtype="<u2"),
               "stage09": np.full(14, 0x3c00, dtype="<u2")}
        for kind in ("k", "v"):
            raw["input_cache_" + kind] = np.zeros((0, 128), dtype="<u2")
            raw["output_cache_" + kind] = np.zeros((1, 128), dtype="<u2")
        old = {"control": control, "failure_count": 0}
        bridge = {"control": control, "paired_margins": [
            {"pair": d.PAIR, "reference_branch": b, "retained_actual_margin": "0",
             "substituted_actual_margin": "0", "retained_reference_margin": "2",
             "required_direction": "2"} for b in ("fp16", "binary64")]}
        return raw, old, bridge

    def test_nine_synthetic_real_guarded_conversion_norm_and_pair(self):
        operands = (np.ones(896, dtype="<f2"), np.zeros((2, 896), dtype="<f2"))
        operands[1][0] = 1 / 1024
        refs = {"fp16": np.zeros(151936, dtype="<u2"),
                "binary64": np.zeros(151936, dtype="<f8")}
        refs["fp16"][34319], refs["binary64"][34319] = 0x4000, 2
        audit = d.new_audit()
        oracle = type("Oracle", (), {"verify_rtz": staticmethod(verify_rtz),
                                     "suffix_oracle": suffix_oracle})
        for control in d.CONTROLS:
            raw, old, bridge = self.fixture(control)
            with d.guarded(d.OUTPUT, audit):
                payload, row = d.run_control(
                    control, raw, old, bridge, np.ones(896, dtype="<f8"),
                    refs, operands, oracle, audit)
            self.assertEqual(row["S18"]["status"], "PASS")
            self.assertTrue(all(m["direction_pass"] for m in row["paired_margins"]))
            self.assertEqual(row["changed_coordinate_count"], 896)
            for key, value in raw.items():
                self.assertEqual(d.retained.digest(payload[key]), d.retained.digest(value))
        self.assertEqual(audit["forbidden_events"], [])
        self.assertEqual(audit["controls"], list(d.CONTROLS))
        self.assertEqual(audit["rtz_coordinates"], 8064)
        self.assertEqual(audit["final_rmsnorm_invocations"], 9)
        self.assertEqual(audit["pair_head_invocations"], 9)

    def test_state_kv_and_oracle_corruption(self):
        for field in ("stage18", "output_cache_v", "stage03"):
            raw, _, _ = self.fixture("scratch")
            raw[field].flat[0] ^= 1
            with self.assertRaises(d.retained.bridge.BindingError):
                d.check_state(raw)
        i, z = np.ones(896, dtype="<i8"), np.zeros(896, dtype="u1")
        words = d.rtz_view(i, z)
        words[4] ^= 1
        with self.assertRaises(AssertionError):
            verify_rtz(i, z, words)

    def test_unchanged_exact_threshold_and_strict_directions(self):
        words = np.full(896, 0x3c00, dtype="<u2")
        self.assertEqual(d.compare(words, np.full(896, 1.125, dtype="<f8"))["status"], "PASS")
        self.assertEqual(d.compare(words, np.full(896, 1.126953125, dtype="<f8"))["status"], "FAIL")
        _, _, bridge = self.fixture("scratch")
        refs = {"fp16": np.zeros(151936, dtype="<u2"),
                "binary64": np.zeros(151936, dtype="<f8")}
        refs["fp16"][34319], refs["binary64"][34319] = 0x4000, 2
        for word, expected in ((0, False), (0xbc00, False), (0x3c00, True)):
            rows = d.retained.paired_margins(
                np.array([word, 0], dtype="<u2"), np.zeros(2, dtype="<u2"),
                bridge["paired_margins"], refs)
            self.assertTrue(all(m["direction_pass"] == expected for m in rows))
        bridge["paired_margins"][0]["required_direction"] = "-2"
        with self.assertRaises(d.retained.bridge.BindingError):
            d.retained.paired_margins(words[:2], np.zeros(2, dtype="<u2"),
                                     bridge["paired_margins"], refs)

    def test_dispatch_and_exclusive_write_guards(self):
        for function, args in (
                (d.retained.execute, ()), (d.retained.intervene, (None,) * 5),
                (d.retained.selection.nearest, (None,) * 3), (d.retained.suffix.check, ({},)),
                (d.retained.bridge.check, ())):
            audit = d.new_audit()
            with d.guarded(d.OUTPUT, audit):
                with self.assertRaises(d.retained.bridge.BindingError):
                    function(*args)
            self.assertTrue(audit["forbidden_events"])
        for event, args in (
                ("subprocess.Popen", ()), ("socket.connect", ()), ("os.mkdir", ()),
                ("import", ("torch",)), ("open", (str(d.SOURCE), "w", 577)),
                ("open", (str(d.OUTPUT / "occupied"), "w", 577))):
            audit = d.new_audit()
            with d.guarded(d.OUTPUT, audit):
                with self.assertRaises(d.retained.bridge.BindingError):
                    d.audit_event(event, args)
            self.assertTrue(audit["forbidden_events"])

    def test_occupied_attempt_refused_without_science(self):
        with patch.object(PathType, "iterdir", return_value=iter([d.OUTPUT / "command.json"])), \
                patch.object(d.subprocess, "check_output", return_value="argus/full-projection\n"), \
                patch.object(d.subprocess, "run") as git:
            git.return_value.returncode = 0
            with self.assertRaisesRegex(d.retained.bridge.BindingError, "occupied"):
                d.execute()


PathType = type(d.OUTPUT)
