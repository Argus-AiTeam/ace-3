"""L23 execution guards use certified parents, never real native dispatch."""

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ace3.model.candidates import q24_s16_l23_from_l22_coordinate62_suffix_execution_v1 as d


EVIDENCE = None


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = EVIDENCE if EVIDENCE is not None else d.authenticate()

    def result(self):
        return {
            "diagnostic_id": d.ID, "status": "DIAGNOSED",
            "output": str(d.OUTPUT), "output_created_exclusively": True,
            "preflight": deepcopy(self.evidence["summary"]), "preflight_pins": d.PINS,
            "flags": d.FLAGS, **d.FLAGS, "native_layer_invocations": 9,
            "native_L23_P0_invocations": 9,
            "audit": {"native_controls": list(d.CONTROLS), "forbidden_calls": 0},
            "claim_boundary": d.BOUNDARY, "normal_host_review": "REQUIRED",
            "controls": [{
                **{key: deepcopy(row[key]) for key in (
                    "control", "retained_L21", "retained_L22", "L22_failure", "parent_archive")},
                "mandatory_statuses": ["PASS"] * 19, "L23_status": "PASS",
                "candidate_admitted": False, "policy_adopted": False,
                "successor_published": False, "source_operand_state_KV_RTZ_checks": "PASS",
            } for row in self.evidence["summary"]["controls"]],
        }

    def test_exact_nine_order(self):
        self.assertEqual(list(d.parents_from(self.evidence)), [
            "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
            "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
            "mapped_all", "inherited_native"])

    def test_l21_l22_failure_history(self):
        for row in self.evidence["summary"]["controls"]:
            self.assertEqual(row["retained_L21"]["failing_coordinates"], [62])
            self.assertEqual(row["retained_L21"]["failures"][0]["threshold_margin"], "-3/4")
            self.assertEqual(row["retained_L22"]["mandatory_statuses"], ["PASS"] * 18 + ["FAIL"])
            self.assertEqual(row["retained_L22"]["S18_failure_indices"], [62])
            self.assertEqual(Fraction(row["L22_failure"]["excess_budget"]), Fraction(1, 8))
            self.assertLess(Fraction(row["L22_failure"]["threshold_margin"]), 0)

    def test_parent_exact_copies(self):
        for label, state in d.parents_from(self.evidence).items():
            raw = self.evidence["states"][label]
            d.preflight.parent.check_state(raw)
            d.prior.retained.verify_parent(state, state)
            for key, field in d.preflight.STATE["fields"].items():
                self.assertTrue(d.np.array_equal(state[key], raw[field]))
                self.assertIsNot(state[key], raw[field])

    def test_original_reference(self):
        binding = self.evidence["summary"]["L23_original_reference"]
        previous = self.evidence["result"]["preflight"]["L22_original_reference"]["reference"]
        self.assertEqual(binding["status"], "BOUND_ORIGINAL_INPUT_L23")
        self.assertEqual(binding["reference"]["input_binary64"], previous["binary64"])
        self.assertEqual(binding["reference"]["input_fp16"], previous["fp16"])
        self.assertEqual(binding["reference"]["prior_kv"], "own empty P0")

    def test_review_and_source_pins(self):
        for pin in d.PINS.values():
            d.preflight.read_bound(pin)
        review = json.loads(d.preflight.read_bound(d.PINS["preflight_review"]))
        d.preflight.parent.matrix.check_review(review, d.REVIEW_MISSION, 1)
        for key, value in (("producer_role", "engineer"), ("mission_id", "wrong")):
            changed = deepcopy(review)
            changed[key] = value
            with self.assertRaises(ValueError):
                d.preflight.parent.matrix.check_review(changed, d.REVIEW_MISSION, 1)

    def test_result_history_mutations(self):
        d.check_result(self.result(), self.evidence, d.OUTPUT)
        for key in ("retained_L21", "retained_L22", "L22_failure", "parent_archive"):
            result = self.result()
            result["controls"][0][key] = {}
            with self.assertRaisesRegex(ValueError, "history/lineage"):
                d.check_result(result, self.evidence, d.OUTPUT)

    def test_result_reference_mutation(self):
        result = self.result()
        result["preflight"]["L23_original_reference"]["reference"]["input_binary64"] = {}
        with self.assertRaisesRegex(ValueError, "preflight lineage"):
            d.check_result(result, self.evidence, d.OUTPUT)

    def test_result_schedule_and_count(self):
        result = self.result()
        result["controls"].reverse()
        with self.assertRaisesRegex(ValueError, "reordered"):
            d.check_result(result, self.evidence, d.OUTPUT)
        for key in ("native_layer_invocations", "native_L23_P0_invocations"):
            result = self.result()
            result[key] = 10
            with self.assertRaisesRegex(ValueError, "count"):
                d.check_result(result, self.evidence, d.OUTPUT)

    def test_result_gates_and_admission(self):
        for key, value in (("mandatory_statuses", ["PASS"] * 18), ("L23_status", "FAIL"),
                           ("candidate_admitted", True), ("policy_adopted", True),
                           ("successor_published", True)):
            result = self.result()
            result["controls"][0][key] = value
            with self.assertRaisesRegex(ValueError, "result boundary"):
                d.check_result(result, self.evidence, d.OUTPUT)
        result = self.result()
        result["candidate_admitted"] = True
        with self.assertRaisesRegex(ValueError, "boundary"):
            d.check_result(result, self.evidence, d.OUTPUT)

    def test_fresh_output(self):
        with patch.object(Path, "exists", return_value=False):
            self.assertEqual(d.output_path(d.OUTPUT), d.OUTPUT)

    def test_occupied_output(self):
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(d.OUTPUT)

    def test_unsafe_paths(self):
        for path in (d.OUTPUT / "nested", d.ROOT / d.OUTPUT.name, Path("/tmp") / d.OUTPUT.name,
                     d.OUTPUT.with_name(d.NAME + "_attempt000"),
                     d.OUTPUT.with_name("unversioned"), d.OUTPUT / ".." / d.OUTPUT.name):
            with self.subTest(path=path), patch.object(d.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    d.output_path(path)
                run.assert_not_called()

    def test_symlink_output(self):
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "symlink"):
                d.output_path(d.OUTPUT)

    def test_nonignored_output(self):
        with patch.object(Path, "exists", return_value=False):
            for code in (1, 128):
                with patch.object(d.subprocess, "run") as run:
                    run.return_value.returncode, run.return_value.stderr = code, b"refused"
                    with self.assertRaisesRegex(ValueError, "ignored"):
                        d.output_path(d.OUTPUT)

    def test_nine_dispatch_bound(self):
        parents = d.parents_from(self.evidence)
        audit = {"native_controls": [], "forbidden_calls": 0}
        with patch.object(d.producer.native.candidate, "_stages",
                          side_effect=lambda *args: iter(range(19))) as raw:
            with d.suffix_only(parents, audit):
                for label in d.CONTROLS:
                    self.assertEqual(list(d.producer.native.candidate._stages(
                        {}, 23, parents[label], {})), list(range(19)))
                with self.assertRaisesRegex(ValueError, "outside nine"):
                    list(d.producer.native.candidate._stages({}, 23, parents[d.CONTROLS[0]], {}))
            self.assertEqual(raw.call_count, 9)
        self.assertEqual(audit, {"native_controls": list(d.CONTROLS), "forbidden_calls": 0})

    def test_wrong_layer_and_parent(self):
        parents = d.parents_from(self.evidence)
        audit = {"native_controls": [], "forbidden_calls": 0}
        with patch.object(d.producer.native.candidate, "_stages") as raw:
            with d.suffix_only(parents, audit):
                for layer in (0, 21, 22, 24, True):
                    with self.assertRaises(ValueError):
                        list(d.producer.native.candidate._stages(
                            {}, layer, parents[d.CONTROLS[0]], {}))
                state = deepcopy(parents[d.CONTROLS[0]])
                state["i"][62] += 1
                with self.assertRaises(ValueError):
                    list(d.producer.native.candidate._stages({}, 23, state, {}))
                with self.assertRaises(ValueError):
                    list(d.producer.native.candidate._stages(
                        {}, 23, parents[d.CONTROLS[0]], {"input_cache_k": []}))
            raw.assert_not_called()

    def test_reordered_schedule(self):
        with self.assertRaisesRegex(ValueError, "reordered"):
            with d.suffix_only(dict(reversed(list(d.parents_from(self.evidence).items()))),
                               {"native_controls": [], "forbidden_calls": 0}):
                self.fail("guard admitted reordered schedule")

    def test_external_and_prefix_guards(self):
        audit = {"native_controls": [], "forbidden_calls": 0}
        with d.suffix_only(d.parents_from(self.evidence), audit):
            for call in (lambda: d.subprocess.Popen(["false"]),
                         lambda: d.os.system("false"), lambda: d.previous.execute(d.OUTPUT)):
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit, {"native_controls": [], "forbidden_calls": 3})

    def test_empty_independent_fp16_kv(self):
        kvs = list(self.evidence["kvs"].values())
        for index, kv in enumerate(kvs):
            for value in kv.values():
                self.assertEqual(value.shape, (0, 128))
                self.assertEqual(value.dtype, d.np.dtype("<u2"))
                for other in kvs[index + 1:]:
                    self.assertTrue(all(value is not array for array in other.values()))

    def test_nonadmission_boundary(self):
        self.assertTrue(all(value == 0 for value in d.FLAGS.values()))
        self.assertIn("wider than FP16", d.BOUNDARY)
        self.assertIn("Independent Host Reviewer", d.BOUNDARY)
        self.assertIn("--execute --out build/" + d.OUTPUT.name, d.command_for(d.OUTPUT))


if __name__ == "__main__":
    unittest.main()
