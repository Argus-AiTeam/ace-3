"""Task-native execution guards and raw failed-parent invariants; no native replay."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l22_from_l21_coordinate62_suffix_execution_v1 as d


EVIDENCE = None


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = EVIDENCE if EVIDENCE is not None else d.authenticate()

    def parents(self):
        return d.parents_from(self.evidence)

    def audit(self):
        return {"native_controls": [], "forbidden_calls": 0}

    def result(self):
        return {
            "diagnostic_id": d.ID, "status": "DIAGNOSED",
            "output": str(d.OUTPUT), "output_created_exclusively": True,
            "preflight": deepcopy(self.evidence["summary"]), "preflight_pins": d.PINS,
            "flags": d.FLAGS, **d.FLAGS, "native_layer_invocations": 9,
            "native_L22_P0_invocations": 9,
            "audit": {"native_controls": list(d.CONTROLS), "forbidden_calls": 0},
            "claim_boundary": d.BOUNDARY, "normal_host_review": "REQUIRED",
            "controls": [{
                "control": row["control"], "retained_L21": deepcopy(row["retained_L21"]),
                "parent_archive": row["parent_archive"], "mandatory_statuses": ["PASS"] * 19,
                "L22_status": "PASS", "candidate_admitted": False,
                "policy_adopted": False, "successor_published": False,
                "source_operand_state_KV_RTZ_checks": "PASS",
            } for row in self.evidence["summary"]["controls"]],
        }

    def test_result_schedule_mutation(self):
        result = self.result()
        d.check_result(result, self.evidence, d.OUTPUT)
        result["controls"].reverse()
        with self.assertRaisesRegex(ValueError, "reordered"):
            d.check_result(result, self.evidence, d.OUTPUT)

    def test_result_failure_relabel_rejected(self):
        result = self.result()
        result["controls"][0]["retained_L21"]["S18"] = "PASS"
        with self.assertRaisesRegex(ValueError, "raw L21 failure"):
            d.check_result(result, self.evidence, d.OUTPUT)

    def test_result_reference_reanchor_rejected(self):
        result = self.result()
        result["preflight"]["L22_original_reference"]["reference"]["input_binary64"] = {}
        with self.assertRaisesRegex(ValueError, "preflight lineage"):
            d.check_result(result, self.evidence, d.OUTPUT)

    def test_result_admission_rejected(self):
        result = self.result()
        result["candidate_admitted"] = True
        with self.assertRaisesRegex(ValueError, "boundary"):
            d.check_result(result, self.evidence, d.OUTPUT)

    def test_result_invocation_count_rejected(self):
        for key in ("native_layer_invocations", "native_L22_P0_invocations"):
            result = self.result()
            result[key] = 10
            with self.assertRaisesRegex(ValueError, "count"):
                d.check_result(result, self.evidence, d.OUTPUT)

    def test_result_incomplete_gates_rejected(self):
        result = self.result()
        result["controls"][0]["mandatory_statuses"].pop()
        with self.assertRaisesRegex(ValueError, "result boundary"):
            d.check_result(result, self.evidence, d.OUTPUT)

    def test_contract_exact(self):
        d.same(json.loads(d.CONTRACT.read_bytes()), d.EXPECTED_CONTRACT, "contract")

    def test_exact_nine_order(self):
        self.assertEqual(list(self.parents()), [
            "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
            "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
            "mapped_all", "inherited_native"])

    def test_raw_l21_failures_preserved(self):
        for row in self.evidence["summary"]["controls"]:
            self.assertEqual(row["retained_L21"]["mandatory_statuses"], ["PASS"] * 18 + ["FAIL"])
            self.assertEqual(row["retained_L21"]["failing_coordinates"], [62])
            self.assertEqual(row["retained_L21"]["failures"][0]["threshold_margin"], "-3/4")

    def test_parent_exact_copies(self):
        parents = self.parents()
        for label, state in parents.items():
            raw = self.evidence["states"][label]
            d.preflight.check_state(raw)
            for key, field in d.preflight.STATE["fields"].items():
                self.assertTrue(d.np.array_equal(state[key], raw[field]))
                self.assertIsNot(state[key], raw[field])

    def test_original_reference_binding(self):
        binding = self.evidence["summary"]["L22_original_reference"]
        previous = self.evidence["retained"]["summary"]["original_reference"]
        self.assertEqual(binding["status"], "BOUND_ORIGINAL_INPUT_L22")
        self.assertEqual(binding["reference"]["input_binary64"], previous["binary64"])
        self.assertEqual(binding["reference"]["input_fp16"], previous["fp16"])

    def test_review_terminal_and_identity(self):
        review = json.loads(d.preflight.census.read_bound(d.PINS["preflight_review"]))
        d.preflight.matrix.check_review(review, "b9de1724bcd7", 1)
        for key, value in (("producer_role", "engineer"), ("mission_id", "wrong")):
            changed = deepcopy(review)
            changed[key] = value
            with self.assertRaises(ValueError):
                d.preflight.matrix.check_review(changed, "b9de1724bcd7", 1)

    def test_fresh_output(self):
        self.assertEqual(d.output_path(d.OUTPUT), d.OUTPUT)

    def test_occupied_refusal(self):
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(d.OUTPUT)

    def test_unsafe_output_refusal(self):
        for value in (d.OUTPUT / "nested", d.ROOT / d.OUTPUT.name,
                      d.OUTPUT.with_name(d.NAME + "_attempt000"),
                      d.OUTPUT.with_name("unversioned")):
            with self.assertRaises(ValueError):
                d.output_path(value)

    def test_symlink_refusal(self):
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "symlink"):
                d.output_path(d.OUTPUT)

    def test_nonignored_refusal(self):
        with patch.object(d.subprocess, "run") as run:
            run.return_value.returncode, run.return_value.stderr = 1, b"not ignored"
            with self.assertRaisesRegex(ValueError, "ignored"):
                d.output_path(d.OUTPUT)

    def test_guard_nine_only(self):
        parents, audit = self.parents(), self.audit()
        with patch.object(d.producer.native.candidate, "_stages", return_value=iter(())):
            with d.suffix_only(parents, audit):
                for state in parents.values():
                    list(d.producer.native.candidate._stages({}, 22, state, {}))
                with self.assertRaises(ValueError):
                    list(d.producer.native.candidate._stages({}, 22, parents[d.CONTROLS[0]], {}))
        self.assertEqual(audit["native_controls"], list(d.CONTROLS))

    def test_guard_wrong_layer(self):
        parents, audit = self.parents(), self.audit()
        with patch.object(d.producer.native.candidate, "_stages", return_value=iter(())):
            with d.suffix_only(parents, audit):
                for layer in (0, 21, 23, True):
                    with self.assertRaises(ValueError):
                        list(d.producer.native.candidate._stages(
                            {}, layer, parents[d.CONTROLS[0]], {}))
        self.assertEqual(audit["native_controls"], [])

    def test_guard_mutated_parent(self):
        parents, audit = self.parents(), self.audit()
        changed = deepcopy(parents[d.CONTROLS[0]])
        changed["i"][62] += 1
        with patch.object(d.producer.native.candidate, "_stages", return_value=iter(())):
            with d.suffix_only(parents, audit):
                with self.assertRaises(ValueError):
                    list(d.producer.native.candidate._stages({}, 22, changed, {}))
        self.assertEqual(audit["native_controls"], [])

    def test_guard_reordered_schedule(self):
        with self.assertRaises(ValueError):
            with d.suffix_only(dict(reversed(list(self.parents().items()))), self.audit()):
                self.fail("reordered controls accepted")

    def test_external_and_prefix_guards(self):
        audit = self.audit()
        with d.suffix_only(self.parents(), audit):
            for call in (lambda: d.subprocess.run(["false"]),
                         lambda: d.os.system("false"),
                         lambda: d.producer.native.execute_layers()):
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit, {"native_controls": [], "forbidden_calls": 3})

    def test_empty_independent_fp16_kv(self):
        caches = [d.preflight.empty_kv() for _ in d.CONTROLS]
        self.assertEqual(len({id(c[k]) for c in caches for k in ("k", "v")}), 18)
        for cache in caches:
            for value in cache.values():
                self.assertEqual(value.shape, (0, 128))
                self.assertEqual(value.dtype.str, "<u2")

    def test_nonadmission_flags(self):
        for key in ("candidate_admitted", "policy_adopted", "successor_published",
                    "strict_FP16_state_claim", "new_token_claim", "full_model_claim",
                    "reference_recomputation", "reference_reanchoring", "admission_replay"):
            self.assertIs(d.FLAGS[key], False)
        self.assertEqual(d.FLAGS["native_L0_L21_invocations"], 0)


if __name__ == "__main__":
    unittest.main()
