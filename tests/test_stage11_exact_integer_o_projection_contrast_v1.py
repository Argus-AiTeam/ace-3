"""Independent rational arithmetic and isolated nine-control entry coverage."""

import copy
from fractions import Fraction as F
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

from ace3.model.candidates import stage11_exact_integer_o_projection_contrast_v1 as d


silu_tests = d.retained.module_from(d.silu.TEST, "stage11_silu_oracle")
down_tests = silu_tests.previous
norm_tests = d.retained.module_from(d.norm.TEST, "stage11_norm_oracle")
gate_tests = d.retained.module_from(d.gate_up.TEST, "stage11_gate_oracle")
suffix_oracle = silu_tests.suffix_oracle
verify_norm = norm_tests.verify_norm
verify_stage16 = silu_tests.verify_stage16


def verify_projection(tensors, name, words, actual):
    if name == d.PREFIX + "mlp.down_proj":
        return down_tests.verify_projection(tensors, words, actual)
    if name in {d.PREFIX + "mlp.gate_proj", d.PREFIX + "mlp.up_proj"}:
        return gate_tests.verify_projection(tensors, name, words, actual)
    assert name == d.PREFIX + "self_attn.o_proj"
    lanes = np.arange(8)
    shifts = 4 * ((lanes % 2) * 4 + lanes // 2)
    q = ((tensors[name + ".qweight"].astype("<i8")[..., None] >> shifts) & 15).reshape(896, 896)
    z = ((tensors[name + ".qzeros"].astype("<i8")[..., None] >> shifts) & 15).reshape(7, 896)
    x = words.view("<f2").astype("<f8") * (1 << 24)
    # Each group dot is an integer below 2**51; only the cross-group sum needs rationals.
    dots = np.stack([x[g * 128:(g + 1) * 128] @
                     (q[g * 128:(g + 1) * 128] - z[g]).astype("<f8") for g in range(7)])
    for column in range(896):
        value = sum((F(float(dots[g, column])) * F(float(tensors[name + ".scales"][g, column]))
                     for g in range(7)), F()) / (1 << 24)
        assert int(actual[column]) == suffix_oracle.round_word(value, value < 0), (
            f"o_proj coordinate {column}")


def verify_add(integers, zeros, words, actual_i, actual_z, actual_h):
    for index, (integer, zero, word) in enumerate(zip(integers, zeros, words, strict=True)):
        total = int(integer) + int(suffix_oracle.scalar(word) * (1 << 24))
        tag = int(total == 0 and int(integer) == 0 and int(zero) == 1 and int(word) == 0x8000)
        assert int(actual_i[index]) == total, f"residual integer {index}"
        assert int(actual_z[index]) == tag, f"residual zero {index}"
        assert int(actual_h[index]) == suffix_oracle.round_word(F(total, 1 << 24), bool(tag)), (
            f"residual FP16 {index}")


class ArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d.native.torch.set_num_threads(1)

    def test_native_nibbles_no_plus_one_and_single_round(self):
        raw, tensors, _, _, _ = down_tests.fixture()
        name, words = d.PREFIX + "self_attn.o_proj", raw["stage10"]
        words[[0, 128, 768]] = [0x3c00, 0x0001, 0x8001]
        tensors[name + ".qweight"][0] = 0x76543210
        tensors[name + ".qzeros"][0] = 0x11111111
        tensors[name + ".scales"][0] = 0.5
        tensors[name + ".qweight"][[128, 768]] = 0x11111111
        tensors[name + ".scales"][[1, 6]] = 0.5
        actual = d.projection(tensors, words)
        verify_projection(tensors, name, words, actual)
        np.testing.assert_array_equal(actual[:8].view("<f2"), [-0.5, 1.5, 0, 2, 0.5, 2.5, 1, 3])
        actual[0] ^= 1
        with self.assertRaisesRegex(AssertionError, "o_proj coordinate 0"):
            verify_projection(tensors, name, words, actual)

    def test_cancellation_tie_and_negative_underflow(self):
        raw, tensors, _, _, _ = down_tests.fixture()
        name, words = d.PREFIX + "self_attn.o_proj", raw["stage10"]
        words[[0, 128, 256, 768]] = [0x7bff, 0xfbff, 0x3c00, 0x1400]
        tensors[name + ".qweight"][[0, 128, 256, 768]] = 0x11111111
        tensors[name + ".scales"][6] = 0.5
        actual = d.projection(tensors, words)
        self.assertTrue(np.all(actual == 0x3c00))
        verify_projection(tensors, name, words, actual)
        words[:] = 0
        words[768] = 0x8001
        actual = d.projection(tensors, words)
        self.assertTrue(np.all(actual == 0x8000))
        verify_projection(tensors, name, words, actual)

    def test_invalid_operands_and_overflow_are_not_clamped(self):
        for defect in ("shape", "dtype", "nan", "inf", "bias", "words", "word_dtype", "word_shape", "overflow"):
            raw, tensors, _, _, _ = down_tests.fixture()
            name, words = d.PREFIX + "self_attn.o_proj", raw["stage10"]
            if defect == "shape":
                tensors[name + ".qweight"] = tensors[name + ".qweight"][:-1]
            elif defect == "dtype":
                tensors[name + ".qzeros"] = tensors[name + ".qzeros"].astype("<i8")
            elif defect in ("nan", "inf"):
                tensors[name + ".scales"][0, 0] = float(defect)
            elif defect == "bias":
                tensors[name + ".bias"] = np.zeros(896, dtype="<f2")
            elif defect == "words":
                words[0] = 0x7c00
            elif defect == "word_dtype":
                words = words.astype("<i4")
            elif defect == "word_shape":
                words = words[:-1]
            else:
                words[:] = 0x7bff
                tensors[name + ".qweight"][:] = 0x11111111
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                d.projection(tensors, words)

    def test_nonzero_full_suffix_uses_new_scratch_and_independent_oracles(self):
        raw, tensors, operands, _, _ = down_tests.fixture()
        raw["stage10"][0] = 0x3c00
        for name in ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            tensors[d.PREFIX + name + ".qweight"][0] = 0x76543210
            tensors[d.PREFIX + name + ".qzeros"][0] = 0x11111111
            tensors[d.PREFIX + name + ".scales"][:] = 0.125
        before = {k: d.retained.digest(v) for k, v in raw.items()}
        for value in (*raw.values(), *tensors.values(), *operands):
            value.flags.writeable = False
        arrays, detail, final_detail, timings = d.compute_suffix(raw, tensors, operands)
        verify_projection(tensors, d.PREFIX + "self_attn.o_proj", raw["stage10"], arrays["stage11"])
        verify_add(raw["input_i"], raw["input_z"], arrays["stage11"],
                   arrays["scratch_i"], arrays["scratch_z"], arrays["stage12"])
        verify_norm(arrays["scratch_i"], arrays["scratch_z"],
                    tensors[d.PREFIX + "post_attention_layernorm.weight"], arrays["stage13"], detail)
        for stage, name in ((14, "gate_proj"), (15, "up_proj"), (17, "down_proj")):
            verify_projection(tensors, d.PREFIX + "mlp." + name,
                              arrays["stage16" if stage == 17 else "stage13"], arrays[f"stage{stage}"])
        verify_stage16(arrays["stage14"], arrays["stage15"], arrays["stage16"],
                       arrays["s16_unrounded_binary64"])
        verify_add(arrays["scratch_i"], arrays["scratch_z"], arrays["stage17"],
                   arrays["output_i"], arrays["output_z"], arrays["stage18"])
        suffix_oracle.verify_suffix(
            arrays["stage18"], operands, arrays["final_rmsnorm"], final_detail, arrays["final_logits"])
        self.assertTrue(np.any(arrays["scratch_i"] != raw["scratch_i"]))
        self.assertTrue(np.any(arrays["stage16"] & 0x7fff))
        self.assertEqual(set(timings), {*(f"stage{s}" for s in range(11, 19)), "final_rmsnorm", "pair_head"})
        self.assertEqual(before, {k: d.retained.digest(v) for k, v in raw.items()})
        for key in d.PROTECTED:
            self.assertEqual(d.retained.digest(arrays[key]), before[key])

    def test_reference_free_interface_thresholds_and_zero_change_falsification(self):
        for function, parameters in ((d.projection, ["tensors", "words"]),
                                     (d.compute_suffix, ["raw", "tensors", "operands"])):
            self.assertEqual(list(inspect.signature(function).parameters), parameters)
            for forbidden in ("reference", "nearest", "compare(", "margin", "classified"):
                self.assertNotIn(forbidden, inspect.getsource(function))
        checks = down_tests.fixtures.IntegerContrastTests()
        checks.test_exact_s18_threshold()
        checks.test_zero_opposite_and_correct_original_reference_movement()
        self.assertEqual(d.scientific_status([{"changed_words": {"11": 0}}]), "REJECTED")

    def test_state_kv_and_rtz_failure_precedes_stage11(self):
        for key in ("stage12", "output_i", "output_cache_v", "stage03", "stage16"):
            raw, tensors, operands, _, _ = down_tests.fixture()
            raw[key].flat[0] ^= (1 << 14) if key == "output_i" else 1
            with self.subTest(key=key), mock.patch.object(d, "projection") as arithmetic:
                with self.assertRaises(ValueError):
                    d.compute_suffix(raw, tensors, operands)
                arithmetic.assert_not_called()

    def test_closed_producers_prefix_other_projections_and_writes_denied(self):
        for function, args in (
                (d.silu.evaluate, (None,)), (d.down.compute_suffix, (None,) * 3),
                (d.base.stage13, (None,) * 2), (d.base.execute, ()), (d.prior.execute, ()),
                (d.retained.selection.nearest, (None,) * 3),
                (d.native.projection, ({}, d.PREFIX + "self_attn.o_proj", None)),
                (d.native.state.project, ({}, d.PREFIX + "self_attn.v_proj", None)),
                (d.prior.audit_event, ("subprocess.Popen", ())),
                (d.prior.audit_event, ("open", (str(d.SOURCE), "w", 577)))):
            audit = d.base.new_audit()
            with d.guarded(audit), self.assertRaises(ValueError):
                function(*args)
            self.assertTrue(audit["forbidden_events"])
            self.assertEqual(audit["writes"], [])
            self.assertIsNone(d.prior._active)
            self.assertIsNone(sys.getprofile())


class AuthenticatedFixture(unittest.TestCase):
    """Build genuine predecessor/92e schemas with isolated synthetic arrays."""

    def setUp(self):
        self.fixture = silu_tests.AuthenticatedEntryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root / "stage16"
        self.root.mkdir()
        self.document = copy.deepcopy(self.fixture.document)
        self.document["mission_id"] = d.BASELINE_MISSION
        self.document["retained_stage17"] = d.pin(Path(d.silu.RESULT_PIN["path"]))
        for row in self.document["controls"]:
            raw = d.down.archive(row["payload"])
            row["protected_arrays"] = {k: d.retained.digest(v) for k, v in raw.items()}
            path = self.root / (row["control"] + ".npz")
            with path.open("xb") as stream:
                np.savez(stream, **raw)
            row["payload"] = d.pin(path)
        result = self.root / "result.json"
        result.write_text(json.dumps(self.document))
        review = self.root / "review.json"
        review.write_text(json.dumps({"mission_id": d.BASELINE_MISSION, "producer_role": "reviewer",
                                      "review": {"status": "done"}}))
        self.result_pin, self.review_pin = d.pin(result), d.pin(review)
        for patch in (mock.patch.object(d, "RESULT_PIN", self.result_pin),
                      mock.patch.object(d, "REVIEW_PIN", self.review_pin)):
            patch.start()
            self.addCleanup(patch.stop)


class AuthenticatedEntryTests(AuthenticatedFixture):
    @classmethod
    def setUpClass(cls):
        d.native.torch.set_num_threads(1)

    def test_nine_control_entry_and_real_emitter_consumer(self):
        from ace3.model.candidates import stage11_exact_integer_o_projection_execution_v1 as wrapper
        before = {str(p): d.pin(p) for p in self.fixture.fixture.root.rglob("*") if p.is_file()}
        result, payloads = d.evaluate(sys.modules[__name__])
        self.assertEqual(result["status"], "REJECTED")  # Zero S11 changes is not support.
        self.assertEqual(list(payloads), list(d.CONTROLS))
        self.assertEqual(result["audit"]["forbidden_events"], [])
        self.assertEqual(result["audit"]["writes"], [])
        self.assertEqual(result["audit"]["stage_dispatches"],
                         [[c, 23, 0, s] for c in d.CONTROLS for s in range(11, 19)])
        self.assertEqual({p["path"] for p in result["source_pins"]}, d.project_sources())
        for row in result["controls"]:
            self.assertEqual(row["component_oracles"]["status"], "PASS")
            self.assertTrue(all(stage["status"] == "PASS" for stage in row["stages"]))
            self.assertTrue(all(margin["direction_pass"] for margin in row["paired_margins"]))
        self.assertEqual(before, {str(p): d.pin(p) for p in self.fixture.fixture.root.rglob("*") if p.is_file()})
        out = self.root / "synthetic-emitter"
        out.mkdir()
        audit = d.base.new_audit()
        with wrapper.writing(out, audit):
            wrapper.emit_payloads(out, result, payloads)
        self.assertEqual(audit["forbidden_events"], [])
        self.assertEqual(audit["writes"], [str(out / (c + ".npz")) for c in d.CONTROLS])
        for row in result["controls"]:
            arrays = d.down.archive(row["payload"])
            d.down.validate_state(arrays)
            for key in d.PROTECTED:
                self.assertEqual(d.retained.digest(arrays[key]), row["protected_arrays"][key])
        self.assertFalse((out / "command.json").exists())

    def test_latest_identity_budget_threshold_operand_order_and_lineage_rejection(self):
        pristine = copy.deepcopy(self.document)
        for defect in ("identity", "budget", "threshold", "operand", "order", "lineage", "parent"):
            document = copy.deepcopy(pristine)
            if defect == "identity":
                document["mission_id"] = d.silu.STAGE17_MISSION
            elif defect == "budget":
                document["science_invocations"] = True
            elif defect == "threshold":
                document["thresholds"] = {}
            elif defect == "operand":
                document["operand_pins"] = []
            elif defect == "order":
                document["controls"].reverse()
            elif defect == "parent":
                document["retained_stage17"] = {}
            else:
                document["controls"][0]["protected_arrays"] = {}
            path = Path(self.result_pin["path"])
            path.write_text(json.dumps(document))
            self.result_pin.update(d.pin(path))
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                d.inputs()

    def test_payload_tamper_rejected_without_science(self):
        path = Path(self.document["controls"][0]["payload"]["path"])
        with path.open("ab") as stream:
            stream.write(b"tamper")
        audit = d.base.new_audit()
        with mock.patch.object(d, "compute_suffix") as arithmetic, self.assertRaises(ValueError):
            d.evaluate(sys.modules[__name__], audit=audit)
        arithmetic.assert_not_called()
        self.assertEqual(audit["science_invocations"], 0)


class RetainedInputTests(unittest.TestCase):
    def test_latest_retained_inputs_read_only_without_arithmetic(self):
        audit = d.base.new_audit()
        with mock.patch.object(d, "compute_suffix", side_effect=AssertionError("science")) as arithmetic:
            with d.guarded(audit):
                values = d.inputs()
        arithmetic.assert_not_called()
        prepared = values[-1]
        self.assertEqual(len(prepared), 9)
        for _, raw in prepared:
            self.assertEqual(raw["stage10"].shape, (896,))
            self.assertFalse(raw["stage10"].flags.writeable)
        self.assertEqual(audit["writes"], [])
        self.assertEqual(audit["forbidden_events"], [])
        self.assertEqual(audit["stage_dispatches"], [])
        self.assertEqual(audit.get("science_invocations", 0), 0)


if __name__ == "__main__":
    unittest.main()
