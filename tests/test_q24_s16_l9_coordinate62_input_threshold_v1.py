"""Frozen binding checks, synthetic arithmetic and mocked dispatch; no native execution."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_input_threshold_v1 as d


EXPECTED_TESTS = 32
HISTORICAL_L9 = (
    "q24_s16_l9_coordinate62_entry_producer_cone_d0755047099e_attempt001",
    "569320a0f55ffdd30647decb676c06d5a83610163761ab5a1807ef648fb97d54",
    d.entry.ID,
)


def parent(value):
    return d.entry.paired.mapped_parent(np.full(896, value, dtype="<f8"))


@contextmanager
def policy_evidence():
    directory, digest, diagnostic_id = d.REVIEWED[0]
    root = d.ROOT / "build" / directory
    bindings = {}
    documents = {}

    def document(path, value, sha):
        item = {"path": str(path), "bytes": len(json.dumps(value)), "sha256": sha}
        bindings[str(path)] = item
        documents[str(path)] = value
        return item

    gate = {"policy_id": d.entry.prior.gates.POLICY_ID}
    gate_binding = document(d.entry.prior.gates.CONTRACT, gate, "synthetic-gate")
    contract = {"diagnostic_id": diagnostic_id, "policy_id": gate["policy_id"]}
    contract_binding = document(root / "contract.json", contract, "synthetic-contract")
    validation = {"contract": contract_binding}
    validation_binding = document(root / "validation.json", validation, "synthetic-validation")
    reviewed = {
        "diagnostic_id": diagnostic_id, "status": "DIAGNOSED",
        "policy_id": gate["policy_id"], "native_retained_bitwise_reproduction": True,
        "unchanged_baseline_gates": True, "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS", "rtl_invocations": 0,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "input_bindings": [gate_binding], "artifacts": [], "validation": validation_binding,
        "origins_after_execution": {},
    }
    document(root / "result.json", reviewed, digest)

    def bind(item, bound):
        d.require(item == bindings[item["path"]], "binding mismatch")
        bound[item["path"]] = item.copy()
        return SimpleNamespace(read_text=lambda: json.dumps(documents[item["path"]]))

    with patch.object(d, "REVIEWED", d.REVIEWED[:1]), \
            patch.object(d.entry, "authenticate", return_value={"bound": {}}), \
            patch.object(d.entry, "record", side_effect=lambda path: bindings[str(path)]), \
            patch.object(d.entry.upstream, "bind_input", side_effect=bind), \
            patch.object(d.entry.upstream, "execute_layer") as execute:
        yield reviewed, validation, contract, gate
        execute.assert_not_called()


class InputThresholdTests(unittest.TestCase):
    def test_selected_frozen_parent_and_all_recorded_bindings_authenticate(self):
        selected = d.REVIEWED[0]
        self.assertEqual(selected[:2], (
            "q24_s16_l9_coordinate62_producer_cone_a38731cbbd8f_attempt001",
            "c36ed0cb236b06b2fd72e2f73502f38f022e55c7fb67bf0f0b2fc705915539d4"))
        with patch.object(d, "REVIEWED", (selected,)), \
                patch.object(d.entry, "authenticate", return_value={"bound": {}}), \
                patch.object(d.entry.upstream, "execute_layer") as execute:
            data = d.authenticate()
            execute.assert_not_called()
        evidence = data["input_threshold_evidence"][0]
        self.assertEqual(evidence["result"]["sha256"], selected[1])
        self.assertTrue(evidence["read_only"])
        self.assertFalse(evidence["policy_binding"]["top_level_present"])
        reviewed = json.loads(Path(evidence["result"]["path"]).read_text())
        bindings = reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]
        bindings += [item for item in reviewed["origins_after_execution"].values() if "path" in item]
        for item in bindings:
            self.assertEqual(data["bound"][item["path"]],
                             {key: item[key] for key in ("path", "bytes", "sha256")})

    def test_historical_parent_cannot_replace_selected_pin(self):
        historical = d.entry.record(d.ROOT / "build" / HISTORICAL_L9[0] / "result.json")
        self.assertEqual(historical["sha256"], HISTORICAL_L9[1])
        with patch.object(d.entry, "authenticate", return_value={"bound": {}}), \
                patch.object(d.entry, "record", return_value=historical), \
                patch.object(d.entry.upstream, "execute_layer") as execute:
            with self.assertRaisesRegex(ValueError, "wrong reviewed input-threshold evidence"):
                d.authenticate()
            execute.assert_not_called()

    def test_historical_parent_source_mismatch_rejected_even_with_historical_pin(self):
        with patch.object(d, "REVIEWED", (HISTORICAL_L9,)), \
                patch.object(d.entry, "authenticate", return_value={"bound": {}}), \
                patch.object(d.entry.upstream, "execute_layer") as execute:
            with self.assertRaisesRegex(
                    ValueError, "binding mismatch: .*diagnose_q24_s16_l9_coordinate62_entry"):
                d.authenticate()
            execute.assert_not_called()

    def test_unchanged_l10_source_mismatch_remains_a_sweep_blocker(self):
        with patch.object(d, "REVIEWED", d.REVIEWED[1:]), \
                patch.object(d.entry, "authenticate", return_value={"bound": {}}), \
                patch.object(d.entry.upstream, "execute_layer") as execute:
            with self.assertRaisesRegex(
                    ValueError, "binding mismatch: .*diagnose_q24_s16_l9_coordinate62_entry"):
                d.authenticate()
            execute.assert_not_called()

    def test_output_accepts_legacy_and_mission_prefixes(self):
        for prefix in (
                "q24_s16_l9_coordinate62_input_threshold_",
                "q24_s16_l9_coordinate62_wire_selected_parent_"):
            with self.subTest(prefix=prefix):
                path = Path("build") / (prefix + "example")
                self.assertEqual(d.output_directory(path), d.ROOT / path)

    def test_output_rejects_outside_nested_and_unrelated_paths(self):
        prefix = d.OUTPUT_PREFIXES[1]
        for path in (Path("/tmp") / (prefix + "example"),
                     d.ROOT / "build/nested" / (prefix + "example"),
                     d.ROOT / "build" / (prefix + "example") / ".." / "unrelated",
                     d.ROOT / "build/unrelated"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "bounded build"):
                d.output_directory(path)

    def test_existing_output_is_not_replaced_or_executed(self):
        with TemporaryDirectory(dir=d.ROOT / "build", prefix=d.OUTPUT_PREFIXES[1]) as directory:
            out = Path(directory)
            sentinel = out / "retained.txt"
            sentinel.write_text("preserve")
            with patch.object(d.sys, "argv", [d.MODULE, "--out", str(out)]), \
                    patch.object(d, "validate") as validate, \
                    patch.object(d, "execute_sweep") as sweep:
                with self.assertRaises(FileExistsError):
                    d.main()
                validate.assert_not_called()
                sweep.assert_not_called()
            self.assertEqual(sentinel.read_text(), "preserve")
            self.assertEqual(list(out.iterdir()), [sentinel])

    def test_check_cli_never_creates_output_or_runs_sweep(self):
        with patch.object(d.sys, "argv", [d.MODULE, "--check"]), \
                patch.object(d, "validate", return_value={"status": "synthetic"}) as validate, \
                patch.object(d, "execute_sweep") as sweep, \
                patch.object(Path, "mkdir") as mkdir, patch("builtins.print") as output:
            self.assertEqual(d.main(), 0)
            validate.assert_called_once_with()
            sweep.assert_not_called()
            mkdir.assert_not_called()
            output.assert_called_once_with(json.dumps({"status": "synthetic"}))

    def test_explicit_matching_policy_authenticates_without_contract(self):
        with policy_evidence() as (_, validation, _, _):
            del validation["contract"]
            evidence = d.authenticate()["input_threshold_evidence"][0]
            self.assertTrue(evidence["policy_binding"]["top_level_present"])
            self.assertIsNone(evidence["policy_binding"]["contract"])

    def test_missing_top_level_policy_uses_matching_frozen_contract_and_gate(self):
        with policy_evidence() as (reviewed, validation, _, _):
            del reviewed["policy_id"]
            data = d.authenticate()
            evidence = data["input_threshold_evidence"][0]["policy_binding"]
            self.assertFalse(evidence["top_level_present"])
            self.assertEqual(evidence["contract"], validation["contract"])
            self.assertIn(validation["contract"]["path"], data["bound"])
            self.assertEqual(evidence["policy_id"], d.entry.prior.gates.POLICY_ID)

    def test_mismatched_or_null_top_level_policy_cannot_use_matching_contract(self):
        for value in ("different-policy", None):
            with self.subTest(value=value), policy_evidence() as (reviewed, _, _, _):
                reviewed["policy_id"] = value
                with self.assertRaisesRegex(ValueError, "reviewed policy mismatch"):
                    d.authenticate()

    def test_mismatched_contract_policy_rejected_with_or_without_top_level(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit), policy_evidence() as (reviewed, _, contract, _):
                if not explicit:
                    del reviewed["policy_id"]
                contract["policy_id"] = "different-policy"
                with self.assertRaisesRegex(ValueError, "contract policy mismatch"):
                    d.authenticate()

    def test_mismatched_gate_policy_rejected_with_or_without_top_level(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit), policy_evidence() as (reviewed, _, _, gate):
                if not explicit:
                    del reviewed["policy_id"]
                gate["policy_id"] = "different-policy"
                with self.assertRaisesRegex(ValueError, "gate policy mismatch"):
                    d.authenticate()

    def test_missing_policy_requires_validation_bound_contract(self):
        with policy_evidence() as (reviewed, validation, _, _):
            del reviewed["policy_id"]
            del validation["contract"]
            with self.assertRaisesRegex(ValueError, "missing reviewed policy contract"):
                d.authenticate()

    def test_wrong_contract_identity_or_binding_rejected(self):
        for changed_binding in (False, True):
            with self.subTest(changed_binding=changed_binding), policy_evidence() as (
                    reviewed, validation, contract, _):
                del reviewed["policy_id"]
                if changed_binding:
                    validation["contract"] = {**validation["contract"], "sha256": "changed"}
                else:
                    contract["diagnostic_id"] = "another-diagnostic"
                with self.assertRaisesRegex(ValueError, "binding mismatch|contract policy mismatch"):
                    d.authenticate()

    def test_missing_or_duplicate_gate_binding_rejected(self):
        for count in (0, 2):
            with self.subTest(count=count), policy_evidence() as (reviewed, _, _, _):
                del reviewed["policy_id"]
                reviewed["input_bindings"] *= count
                with self.assertRaisesRegex(ValueError, "missing or duplicate reviewed gate"):
                    d.authenticate()

    def test_plan_is_fixed_bounded_and_independent(self):
        controls = d.plan()
        self.assertEqual(len(controls), 13)
        self.assertEqual(len({row["label"] for row in controls}), 13)
        self.assertEqual([row["step"] for row in controls[:9]], list(range(9)))
        self.assertEqual([row["offset_q24"] for row in controls[9:]], [-1, 1, -1, 1])
        controls[0]["step"] = 99
        self.assertEqual(d.plan()[0]["step"], 0)

    def test_endpoints_and_unselected_coordinates_are_exact(self):
        actual, mapped = parent(1), parent(2)
        saved = {k: v.copy() for k, v in actual.items()}
        for index in (0, 8):
            result = d.input_parent(actual, mapped, d.plan()[index])
            endpoint = actual if index == 0 else mapped
            for key in actual:
                self.assertEqual(result[key][62], endpoint[key][62])
                np.testing.assert_array_equal(result[key][np.arange(896) != 62],
                                              actual[key][np.arange(896) != 62])
        d.entry.paired.same_arrays(actual, saved)

    def test_interpolation_uses_exact_ties_even_q24(self):
        actual, mapped = parent(0), parent(4 * 2**-24)
        values = [int(d.input_parent(actual, mapped, control)["i"][62])
                  for control in d.plan()[:9]]
        self.assertEqual(values, [0, 0, 1, 2, 2, 2, 3, 4, 4])

    def test_one_unit_controls_do_not_relift_fp16(self):
        actual = parent(1)
        lower = d.input_parent(actual, actual, d.plan()[9])
        upper = d.input_parent(actual, actual, d.plan()[10])
        self.assertEqual(lower["i"][62], (1 << 24) - 1)
        self.assertEqual(upper["i"][62], (1 << 24) + 1)
        self.assertEqual(lower["h"][62], actual["h"][62])
        self.assertEqual(upper["h"][62], actual["h"][62])

    def test_signed_zero_is_preserved(self):
        negative, positive = parent(-0.0), parent(0.0)
        for index, expected in ((0, 1), (4, 0), (8, 0)):
            result = d.input_parent(negative, positive, d.plan()[index])
            self.assertEqual(result["z"][62], expected)
            self.assertEqual(result["h"][62], expected << 15)
        result = d.input_parent(negative, negative, d.plan()[4])
        self.assertEqual(result["h"][62], 0x8000)

    def test_unknown_control_and_inconsistent_parent_rejected(self):
        actual = parent(1)
        with self.assertRaisesRegex(ValueError, "unknown"):
            d.input_parent(actual, actual, {"step": 9})
        actual["h"][62] = 0
        with self.assertRaisesRegex(ValueError, "parent/view mismatch"):
            d.input_parent(actual, actual, d.plan()[0])

    def test_dispatch_rejects_all_forbidden_layers_before_delegate(self):
        with patch.object(d.entry.upstream, "execute_layer") as execute:
            for layer in list(range(-2, 9)) + list(range(14, 25)) + [True, 9.0, "9", None]:
                with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "restricted"):
                    d.execute_layer(layer, 0, None, None)
            execute.assert_not_called()

    def test_dispatch_rejects_non_p0_before_delegate(self):
        with patch.object(d.entry.upstream, "execute_layer") as execute:
            for position in (-1, 1, True, 0.0, "0", None):
                with self.subTest(position=position), self.assertRaisesRegex(ValueError, "restricted"):
                    d.execute_layer(9, position, None, None)
            execute.assert_not_called()

    def test_underlying_dispatch_guard_also_rejects_forbidden_layers(self):
        with patch.object(d.entry.native, "stages") as stages:
            for layer in list(range(-2, 9)) + list(range(14, 25)) + [True, 9.0, "9"]:
                with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "outside"):
                    d.entry.upstream.execute_layer(None, layer, None, None, None)
            stages.assert_not_called()

    def test_allowed_dispatch_preserves_reference_and_tensor_identity(self):
        item = {key: object() for key in ("tensors", "trajectory", "reference")}
        actual = object()
        with patch.object(d.entry.upstream, "execute_layer", return_value="observed") as execute:
            for layer in d.LAYERS:
                self.assertEqual(d.execute_layer(layer, 0, actual, {"layers": {layer: item}}),
                                 "observed")
                execute.assert_called_with(item["tensors"], layer, actual,
                                           item["trajectory"], item["reference"])

    def test_control_threads_only_l9_l13_and_retains_failures(self):
        actual, mapped = parent(1), parent(2)
        states = [parent(value) for value in range(3, 8)]
        data = {"layers": {layer: {"reference": np.ones(896)} for layer in d.LAYERS}}
        calls = []

        def execute(layer, position, incoming, supplied):
            self.assertEqual(position, 0)
            self.assertIs(supplied, data)
            if calls:
                self.assertIs(incoming, states[len(calls) - 1])
            else:
                self.assertEqual(incoming["i"][62], 2 << 24)
            calls.append(layer)
            return {"output": states[len(calls) - 1]}, {}, [{"status": "FAIL"}]

        with patch.object(d, "execute_layer", side_effect=execute), \
                patch.object(d.entry.prior.retained, "state_from",
                             side_effect=lambda arrays, *_: arrays["output"]):
            row = d.execute_control(d.plan()[8], actual, mapped, data)
        self.assertEqual(calls, list(d.LAYERS))
        self.assertFalse(row["candidate_admitted"])
        self.assertEqual([r["reports"] for r in row["layers"]], [[{"status": "FAIL"}]] * 5)

    def test_baseline_requires_retained_reproduction(self):
        actual = parent(1)
        item = {"arrays": {"saved": np.zeros(1)}, "locals": {}, "reports": []}
        with patch.object(d, "execute_layer", return_value=({}, {}, [])) as execute:
            with self.assertRaisesRegex(ValueError, "archive key mismatch"):
                d.execute_control(d.plan()[0], actual, actual, {"layers": {9: item}})
        self.assertEqual(execute.call_count, 1)

    def test_threshold_intervals_do_not_assume_monotonicity(self):
        rows = [{"control": control, "layers": [
            {"layer": layer, "index62": {"accepted": control["step"] in (2, 3, 6)}}
            for layer in d.LAYERS]} for control in d.plan()]
        result = d.threshold_intervals(rows, 9)
        self.assertEqual([r["left_step"] for r in result], [1, 3, 5, 6])
        self.assertFalse(result[1]["right_accepted"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            d.threshold_intervals(rows[:-1], 9)
        with self.assertRaisesRegex(ValueError, "outside"):
            d.threshold_intervals(rows, 8)

    def test_numerical_threshold_is_unchanged(self):
        measure = d.entry.prior.measure
        self.assertTrue(measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        self.assertFalse(measure(0x6630, reference)["accepted"])
        self.assertTrue(measure(0x662f, reference)["accepted"])

    def test_authentication_rejects_changed_evidence_without_execution(self):
        with patch.object(d.entry, "authenticate", return_value={"bound": {}}), \
                patch.object(d.entry, "record", return_value={"sha256": "changed"}), \
                patch.object(d.entry.upstream, "execute_layer") as execute:
            with self.assertRaisesRegex(ValueError, "wrong reviewed"):
                d.authenticate()
            execute.assert_not_called()
        with self.assertRaisesRegex(ValueError, "outside isolated repository"):
            d.entry.upstream.bind_input({"path": "/tmp/not-repository", "bytes": 0,
                                        "sha256": "changed"}, {})

    def test_contract_and_module_origins_keep_non_admission_boundary(self):
        contract = json.loads(d.CONTRACT.read_text())
        self.assertEqual(contract["controls"], d.plan())
        self.assertEqual(contract["native_layers"], list(d.LAYERS))
        self.assertEqual(contract["max_native_layer_evaluations"], 65)
        self.assertEqual(contract["normal_host_review"], "REQUIRED")
        for key in ("candidate_admitted", "policy_adopted", "successor_published",
                    "scientific_result_claim", "accepted_L0_L8_execution"):
            self.assertIs(contract[key], False)
        origins = d.source_context()
        for name in (d.MODULE, d.TEST_MODULE, d.runtime.ORIGIN_ONLY_TEST):
            self.assertTrue(Path(origins[name]["path"]).is_relative_to(d.ROOT))
        with patch.dict(d.sys.modules, {
                "ace3.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "origin mismatch"):
                d.source_context()
