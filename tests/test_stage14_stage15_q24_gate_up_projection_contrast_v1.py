"""Repeatable synthetic entry checks; no retained science dispatch or claim."""

import copy
from fractions import Fraction as F
import hashlib
import inspect
import io
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from ace3.model.candidates import stage14_stage15_q24_gate_up_projection_contrast_v1 as d


fixtures = d.retained.module_from(d.base.TEST, "gate_up_synthetic_fixtures")
suffix_oracle = fixtures.suffix_oracle


class HistoricalRecordTests(unittest.TestCase):
    OLD = {
        "path": "/home/argustest/ace3-argus/ace3/model/stage11_current_runtime_release.py",
        "bytes": 130008,
        "sha256": "aa8a02de4d174a9c8101881e81a026341ee50cb461beac499944d9fbc1d0d115"}
    SNAPSHOT = str(d.ROOT / "build" /
                   "argus-stage11-value-content-current-census-release-f0385ca1ad3a-attempt001" /
                   "source.py")

    def test_exact_historical_mapping_does_not_mutate_record(self):
        record = dict(self.OLD)
        with mock.patch.object(d.retained.selection, "bound", return_value=b"authenticated") as bound:
            self.assertEqual(d.historical_record(record), b"authenticated")
        bound.assert_called_once_with({**self.OLD, "path": self.SNAPSHOT})
        self.assertEqual(record, self.OLD)

    def test_changed_record_never_uses_snapshot(self):
        for change in (
                {"path": self.OLD["path"] + ".other"},
                {"path": self.OLD["path"].replace("/model/", "/model/../model/")},
                {"bytes": 130009}, {"sha256": "0" * 64}, {"mapping": self.SNAPSHOT}):
            record = {**self.OLD, **change}
            with self.subTest(change=change), mock.patch.object(
                    d.retained.selection, "bound", wraps=d.retained.selection.bound) as bound:
                with mock.patch.object(Path, "read_bytes", return_value=b"changed"):
                    with self.assertRaises(ValueError):
                        d.historical_record(record)
                bound.assert_called_once_with(record)
        with self.assertRaises(TypeError):
            d.historical_record(self.OLD, {self.OLD["path"]: self.SNAPSHOT})

    def test_snapshot_tamper_and_live_bytes_cannot_satisfy_old_pin(self):
        for data in (b"changed", b"x" * 130008, b"x" * 130703):
            with self.subTest(size=len(data)), mock.patch.object(
                    Path, "read_bytes", autospec=True, return_value=data) as read:
                with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                    d.historical_record(self.OLD)
                read.assert_called_once_with(Path(self.SNAPSHOT))

    def test_other_records_and_current_pins_use_exact_path(self):
        data = b"isolated current source"
        current = {**self.OLD, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for record in (current, {**current, "path": str(d.SOURCE)}):
            with self.subTest(record=record), mock.patch.object(
                    Path, "read_bytes", autospec=True, return_value=data) as read:
                self.assertEqual(d.historical_record(record), data)
                read.assert_called_once_with(Path(record["path"]))
                with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                    d.historical_record({**record, "bytes": record["bytes"] + 1})
        with mock.patch.object(Path, "read_bytes", return_value=data):
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                d.retained.selection.bound(self.OLD)


def verify_projection(tensors, name, words, actual):
    """Independent binary64-exact group dots, rational cross-group sum/round."""
    lanes = np.arange(8)
    shifts = 4 * ((lanes % 2) * 4 + lanes // 2)
    q = ((tensors[name + ".qweight"].astype("<i8")[..., None] >> shifts) & 15).reshape(896, 4864)
    z = ((tensors[name + ".qzeros"].astype("<i8")[..., None] >> shifts) & 15).reshape(7, 4864)
    # Every group dot is exactly representable: 128 * 15 * 65504 * 2**24 < 2**51.
    x = words.view("<f2").astype("<f8") * (1 << 24)
    dots = np.stack([x[g * 128:(g + 1) * 128] @
                     (q[g * 128:(g + 1) * 128] - z[g]).astype("<f8") for g in range(7)])
    scales = tensors[name + ".scales"]
    for column in range(4864):
        value = sum((F(float(dots[g, column])) * F(float(scales[g, column]))
                     for g in range(7)), F()) / (1 << 24)
        expected = suffix_oracle.round_word(value, value < 0)
        assert int(actual[column]) == expected, f"projection coordinate {column}"


class GateUpContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d.native.torch.set_num_threads(1)

    def test_actual_suffix_entry_schema_protection_and_oracles(self):
        raw, tensors, operands, refs, classified = fixtures.fixture()
        protected = {k: d.retained.digest(v) for k, v in raw.items()}
        for value in (*raw.values(), *tensors.values(), *operands):
            value.flags.writeable = False
        audit = d.base.new_audit()
        with d.guarded(d.OUTPUT, audit):
            arrays, row = d.evaluate_control(
                d.CONTROLS[0], raw, raw, classified, tensors, raw,
                np.ones(896, dtype="<f8"), refs, operands,
                __import__(__name__, fromlist=[""]), audit)
        self.assertEqual(row["component_oracles"]["status"], "PASS")
        self.assertTrue(all(s["status"] == "PASS" for s in row["stages"]))
        self.assertTrue(all(m["direction_pass"] for m in row["paired_margins"]))
        self.assertEqual(audit["stage_dispatches"],
                         [[d.CONTROLS[0], 23, 0, s] for s in range(14, 19)])
        self.assertEqual(audit["forbidden_events"], [])
        self.assertEqual(audit["writes"], [])
        self.assertEqual(protected, {k: d.retained.digest(v) for k, v in raw.items()})
        np.testing.assert_array_equal(arrays["stage13"], raw["stage13"])
        rows = [{**copy.deepcopy(row), "control": c} for c in d.CONTROLS]
        self.assertEqual(d.scientific_status(rows), "SUPPORTED")
        rows[0]["paired_margins"][0]["direction_pass"] = False
        self.assertEqual(d.scientific_status(rows), "REJECTED")
        rows[0]["stages"].pop()
        with self.assertRaisesRegex(ValueError, "schema"):
            d.scientific_status(rows)

    def test_native_nibble_zero_scale_group_and_signed_underflow(self):
        _, tensors, _, _, _ = fixtures.fixture()
        name = d.PREFIX + "mlp.gate_proj"
        words = np.zeros(896, dtype="<u2")
        words[0], words[128], words[256] = 0x3c00, 0x0001, 0x8001
        tensors[name + ".qweight"][0] = 0x76543210
        tensors[name + ".qzeros"][0] = 0x11111111
        tensors[name + ".scales"][0] = 0.5
        tensors[name + ".qweight"][128:257:128] = 0x11111111
        tensors[name + ".scales"][1:3] = 0.5
        actual = d.projection(tensors, name, words)
        verify_projection(tensors, name, words, actual)
        np.testing.assert_array_equal(actual[:8].view("<f2"), [-0.5, 1.5, 0, 2, 0.5, 2.5, 1, 3])
        tensors[name + ".qweight"][:] = 0
        tensors[name + ".qzeros"][:] = 0
        tensors[name + ".qweight"][256] = 0x11111111
        actual = d.projection(tensors, name, words)
        self.assertTrue(np.all(actual == 0x8000))
        verify_projection(tensors, name, words, actual)
        actual[0] ^= 1
        with self.assertRaisesRegex(AssertionError, "coordinate 0"):
            verify_projection(tensors, name, words, actual)

    def test_invalid_operands_and_tensor_contracts(self):
        for defect in ("shape", "dtype", "nan", "bias", "name", "words"):
            raw, tensors, _, _, _ = fixtures.fixture()
            name = d.PREFIX + "mlp.gate_proj"
            if defect == "shape":
                tensors[name + ".qweight"] = tensors[name + ".qweight"][:-1]
            elif defect == "dtype":
                tensors[name + ".qzeros"] = tensors[name + ".qzeros"].astype("<i8")
            elif defect == "nan":
                tensors[name + ".scales"][0, 0] = np.nan
            elif defect == "bias":
                tensors[name + ".bias"] = np.zeros(4864, dtype="<f2")
            elif defect == "name":
                name = d.PREFIX + "mlp.down_proj"
            else:
                raw["stage13"][0] = 0x7c00
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                d.projection(tensors, name, raw["stage13"])

    def test_state_and_kv_rejection_before_projection(self):
        for key in ("stage12", "output_i", "output_cache_v", "stage03"):
            raw, tensors, operands, _, _ = fixtures.fixture()
            raw[key].flat[0] ^= (1 << 14) if key == "output_i" else 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.compute_suffix(raw, tensors, operands)

    def test_forbidden_producers_reference_selection_and_writes(self):
        for function, args in (
                (d.parent.compute_suffix, (None,) * 3),
                (d.parent.stage13, (None,) * 3),
                (d.base.execute, ()), (d.prior.execute, ()),
                (d.retained.selection.nearest, (None,) * 3),
                (d.native.projection, ({}, d.PREFIX + "mlp.gate_proj", None)),
                (d.native.state.project, ({}, d.PREFIX + "self_attn.o_proj", None)),
                (d.prior.audit_event, ("subprocess.Popen", ())),
                (d.prior.audit_event, ("open", (str(d.SOURCE), "w", 577)))):
            audit = d.base.new_audit()
            with d.guarded(d.OUTPUT, audit):
                with self.assertRaises(ValueError):
                    function(*args)
            self.assertTrue(audit["forbidden_events"])

    def test_reference_free_bit_interfaces(self):
        self.assertEqual(list(inspect.signature(d.projection).parameters), ["tensors", "name", "words"])
        self.assertEqual(list(inspect.signature(d.compute_suffix).parameters), ["raw", "tensors", "operands"])
        for function in (d.projection, d.compute_suffix):
            text = inspect.getsource(function)
            for forbidden in ("reference", "nearest", "compare(", "margin", "classified"):
                self.assertNotIn(forbidden, text)

    def test_retained_archive_emitter_consumer_boundary(self):
        raw, _, _, _, _ = fixtures.fixture()
        raw["q24_state_contrast_stage13"] = raw["stage13"].copy()
        stream = io.BytesIO()
        np.savez(stream, **raw)
        document = {"mission_id": d.parent.MISSION, "status": "REJECTED",
                    "science_invocations": 1, "non_admission": True,
                    "audit": {"forbidden_events": []},
                    "controls": [{"control": c} for c in d.CONTROLS]}
        with mock.patch.object(d.retained.selection, "bound", return_value=stream.getvalue()):
            original, prepared = d.retained_input(document, {"payload": {}})
        np.testing.assert_array_equal(prepared["stage13"], original["q24_state_contrast_stage13"])
        self.assertFalse(prepared["stage13"].flags.writeable)
        document["controls"].reverse()
        with self.assertRaisesRegex(ValueError, "identity"):
            d.retained_input(document, {"payload": {}})

    def test_source_pins_and_missing_handoff_are_fail_closed(self):
        record = d.pin(d.SOURCE)
        self.assertEqual(d.retained.selection.bound(record), d.SOURCE.read_bytes())
        with self.assertRaises(ValueError):
            d.retained.selection.bound({**record, "sha256": "0" * 64})
        with mock.patch.object(d, "OUTPUT", d.ROOT / "build" / (d.NAME + "-absent-synthetic")):
            with self.assertRaisesRegex(ValueError, "Manager-provisioned"):
                d.check()

    def test_original_reference_direction_and_threshold(self):
        previous = fixtures.IntegerContrastTests()
        previous.test_exact_s18_threshold()
        previous.test_zero_opposite_and_correct_original_reference_movement()


if __name__ == "__main__":
    unittest.main()
