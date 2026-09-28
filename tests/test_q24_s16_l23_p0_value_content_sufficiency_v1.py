import io
import json
from contextlib import ExitStack
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_value_content_sufficiency_v1 as d


class ValueSufficiencyTests(unittest.TestCase):
    def fixture(self):
        actual = {key: np.zeros(896, dtype="<u2") for key in d.PROTECTED}
        for stage in (3, 6, 7):
            actual[f"stage{stage:02d}"] = np.zeros(128, dtype="<u2")
        actual["stage09"] = np.full(14, 0x3c00, dtype="<u2")
        for kind in ("k", "v"):
            actual["input_cache_" + kind] = np.empty((0, 128), dtype="<u2")
            actual["output_cache_" + kind] = np.zeros((1, 128), dtype="<u2")
        reference = {key: value.copy() for key, value in actual.items()}
        reference["stage03"][:] = reference["stage07"][:] = 0x3c00
        return actual, reference

    def rows(self):
        old = np.zeros(34320, dtype="<f2")
        ref = old.copy()
        ref[d.PAIR[0]] = 2
        margins = d.margin_rows(old.view("<u2"), np.array([1, 0], dtype="<f2").view("<u2"),
                                {"logits_fp16": ref.view("<u2"),
                                 "logits_binary64": ref.astype("<f8")})
        return [{"control": label, "reports": [{"stage": s, "status": "PASS"}
                                              for s in range(10, 19)],
                 "paired_final_margins": [dict(m) for m in margins]} for label in d.CONTROLS]

    def test_all_nine_pass(self):
        self.assertEqual(d.scientific_status(self.rows()), "SUPPORTED")

    def test_one_failure_rejects_rescue(self):
        rows = self.rows()
        rows[-1]["reports"][-1]["status"] = "FAIL"
        self.assertEqual(d.scientific_status(rows), "REJECTED")

    def test_missing_control_is_defect(self):
        with self.assertRaises((RuntimeError, ValueError)):
            d.scientific_status(self.rows()[:-1])

    def test_reordered_controls_is_defect(self):
        with self.assertRaises((RuntimeError, ValueError)):
            d.scientific_status(self.rows()[::-1])

    def test_missing_stage_is_defect(self):
        rows = self.rows()
        rows[0]["reports"].pop()
        with self.assertRaises((RuntimeError, ValueError)):
            d.scientific_status(rows)

    def test_unknown_is_not_scientific_failure(self):
        rows = self.rows()
        rows[0]["reports"][0]["status"] = "UNKNOWN"
        with self.assertRaises((RuntimeError, ValueError)):
            d.scientific_status(rows)

    def test_only_stage07_substituted(self):
        actual, reference = self.fixture()
        with patch.object(d.layer.preflight.parent, "check_state"):
            _, arrays = d.prepare(actual, reference)
        d.protected(actual, arrays, reference)
        self.assertTrue(np.all(arrays["stage07"] == 0x3c00))
        self.assertTrue(np.all(arrays["output_cache_v"] == 0))
        self.assertFalse(arrays["stage07"].flags.writeable)
        self.assertNotIn("stage10", arrays)

    def test_nonunit_probability_rejected(self):
        actual, reference = self.fixture()
        actual["stage09"][0] = 0
        with self.assertRaises((RuntimeError, ValueError)):
            d.prepare(actual, reference)

    def test_cache_splice_rejected(self):
        actual, reference = self.fixture()
        actual["output_cache_v"][0, 0] = 1
        with patch.object(d.layer.preflight.parent, "check_state"):
            with self.assertRaises((RuntimeError, ValueError)):
                d.prepare(actual, reference)

    def test_gqa_value_mapping(self):
        arrays = {"stage07": np.arange(128, dtype="<f2").view("<u2"),
                  "stage09": np.full(14, 0x3c00, dtype="<u2")}
        stages = d.suffix_stages({}, {}, arrays)
        self.assertEqual(next(stages), 10)
        expected = np.repeat(arrays["stage07"].reshape(2, 64), 7, axis=0).reshape(-1)
        np.testing.assert_array_equal(arrays["stage10"], expected)

    def test_signed_paired_margins(self):
        old = np.zeros(34320, dtype="<f2")
        new = np.zeros(2, dtype="<f2")
        ref = old.copy()
        old[34319], new[0], ref[34319] = 2, 3, 1
        rows = d.margin_rows(old.view("<u2"), new.view("<u2"),
                             {"logits_fp16": ref.view("<u2"), "logits_binary64": ref.astype("<f8")})
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["pair"] for r in rows], [list(d.PAIR)] * 2)
        self.assertEqual(rows[0]["retained_margin_change"], "1")
        self.assertEqual(rows[0]["substituted_margin_change"], "2")
        self.assertEqual(rows[0]["paired_intervention_delta"], "1")

    def test_forbidden_write_and_native_prefix(self):
        audit = {"file_write_opens": 0, "forbidden_calls": 0}
        with d.write_scope(d.OUTPUT, audit):
            with self.assertRaises((RuntimeError, ValueError)):
                io.open(d.ROOT / "forbidden.tmp", "w")
        with d.final.no_dispatch(audit):
            with self.assertRaises(RuntimeError):
                d.native._stages({}, 23, {}, {})
        self.assertEqual(audit["forbidden_calls"], 1)

    def test_zero_movement_rejects(self):
        rows = self.rows()
        margin = rows[0]["paired_final_margins"][0]
        margin.update(substituted_actual_margin="0", substituted_margin_change="-2",
                      paired_intervention_delta="0")
        self.assertEqual(d.scientific_status(rows), "REJECTED")

    def test_opposite_movement_rejects(self):
        rows = self.rows()
        margin = rows[-1]["paired_final_margins"][-1]
        margin.update(substituted_actual_margin="-1", substituted_margin_change="-3",
                      paired_intervention_delta="-1")
        self.assertEqual(d.scientific_status(rows), "REJECTED")

    def test_already_at_reference_rejects_direction(self):
        rows = self.rows()
        margin = rows[0]["paired_final_margins"][0]
        margin.update(retained_reference_margin="0", retained_margin_change="0",
                      substituted_margin_change="1")
        self.assertEqual(d.scientific_status(rows), "REJECTED")

    def test_missing_reference_is_defect(self):
        rows = self.rows()
        rows[0]["paired_final_margins"].pop()
        with self.assertRaises((RuntimeError, ValueError)):
            d.scientific_status(rows)

    def test_duplicate_reference_is_defect(self):
        rows = self.rows()
        rows[0]["paired_final_margins"][1]["reference_branch"] = "fp16"
        with self.assertRaises((RuntimeError, ValueError)):
            d.scientific_status(rows)

    def test_wrong_pair_is_defect(self):
        rows = self.rows()
        rows[0]["paired_final_margins"][0]["pair"] = [319, 34319]
        with self.assertRaises((RuntimeError, ValueError)):
            d.scientific_status(rows)

    def test_inconsistent_margin_is_defect(self):
        rows = self.rows()
        rows[0]["paired_final_margins"][0]["paired_intervention_delta"] = "2"
        with self.assertRaises((RuntimeError, ValueError)):
            d.scientific_status(rows)

    def test_full_vocabulary_head_rejected_before_dispatch(self):
        with patch.object(d.final, "logits") as logits:
            with self.assertRaises((RuntimeError, ValueError)):
                d.pair_logits(np.zeros(896, dtype="<u2"), np.zeros((3, 896), dtype="<f2"))
            logits.assert_not_called()

    def test_pair_head_uses_native_exact_arithmetic(self):
        words = np.full(896, 0x3c00, dtype="<u2")
        weights = np.zeros((2, 896), dtype="<f2")
        weights[0, 0], weights[1, 1] = 2, -1
        np.testing.assert_array_equal(d.pair_logits(words, weights),
                                      np.array([2, -1], dtype="<f2").view("<u2"))

    def test_full_vocabulary_candidate_logits_rejected(self):
        values = np.zeros(34320, dtype="<u2")
        with self.assertRaises((RuntimeError, ValueError)):
            d.margin_rows(values, values,
                          {"logits_fp16": values, "logits_binary64": values.astype("<f8")})

    def test_occupied_output_prevents_any_dispatch(self):
        with patch.object(d.Path, "exists", return_value=True), \
                patch.object(d, "manager_binding", return_value={"output": d.OUTPUT}), \
                patch.object(d, "suffix_stages") as stages, \
                patch.object(d.retained, "authenticate") as authenticate:
            with self.assertRaises((RuntimeError, ValueError)):
                d.execute()
            stages.assert_not_called()
            authenticate.assert_not_called()

    def binding_fixture(self):
        """Synthetic transport records only; never external execution authority."""
        pins = {}
        documents = {}

        def pin(path):
            path = str(path)
            if path not in pins:
                pins[path] = {"path": path, "bytes": 0, "sha256": "synthetic"}
            return pins[path]

        authority_path = d.AUTHORITY / "build-authorization.json"
        authority = {
            "mission_id": d.MISSION, "authority_id": "synthetic",
            "life_dir": str(d.release.LIFE), "issuer": "manager", "expected_attempt": 1,
            "authorized_science": True, "scientific_run_budget": 1,
            "authorized_root": "build/synthetic-capture",
            "manager_inputs": {
                "issuer": pin(d.ROOT / ".argus/live/manager-tools/publish-stage11-science-issuance-v1.py"),
                "science_input": pin(d.AUTHORITY / "synthetic-input.json"),
            },
        }
        inputs = {
            "mission_id": d.MISSION, "authority_id": "synthetic", "attempt": 1,
            "scientific_run_budget": 1, "capture_root": authority["authorized_root"],
            "candidate": pin(d.SOURCE), "candidate_tests": pin(d.TEST),
            "release_source": pin(d.Path(d.release.__file__)),
            "runner": pin(d.ROOT / ".argus/live/manager-tools/run-stage11-science-once-v1.py"),
            "capture_implementation": [], "contract": pin(d.AUTHORITY / "synthetic-contract.json"),
            "contract_review": {k: "synthetic" for k in
                                ("mission_id", "latest", "review", "checkpoint", "mission")},
            "accepted_review": {"mission_id": "synthetic"},
            "operational_release_selection": pin(d.AUTHORITY / "synthetic-selection.json"),
        }
        output = d.ROOT / "build/synthetic-fresh-value-content"
        declaration = {
            "launch": {"argv": [d.sys.executable, "-B", "-m", "ace3.model.candidates." + d.NAME,
                                "--check", "--output", str(output)]},
            "contract": {"sources": {"current_diagnostic": inputs["candidate"],
                                     "current_tests": inputs["candidate_tests"]}},
        }
        envelope = {
            "mission_id": d.MISSION, "attempt": 1, "scientific_run_budget": 1,
            "declaration": pin(d.AUTHORITY / "synthetic-declaration.json"),
            "issuance": pin(d.AUTHORITY / "synthetic-issuance.json"),
            "current_release_contract": inputs["contract"],
            "current_release_review": {k: inputs["contract_review"][k] for k in
                                       ("mission_id", "latest", "review", "checkpoint")},
            "operational_release_selection": inputs["operational_release_selection"],
            "review": inputs["accepted_review"],
        }
        documents.update({
            str(authority_path): authority,
            str(d.AUTHORITY / "manager-owner-prepared.json"): {
                "mission_id": d.MISSION, "authority_id": "synthetic",
                "state": "PREPARED_BEFORE_NATIVE_HANDOFF", "build_authorization": pin(authority_path)},
            inputs["candidate"]["path"]: {},
            authority["manager_inputs"]["science_input"]["path"]: inputs,
            str(d.AUTHORITY / "issued/envelope.json"): envelope,
            envelope["declaration"]["path"]: declaration,
            envelope["issuance"]["path"]: {"declaration": envelope["declaration"]},
        })
        self.enterContext(patch.object(d, "record", side_effect=pin))
        self.enterContext(patch.object(d, "read_bound",
                                       side_effect=lambda p: json.dumps(documents[p["path"]]).encode()))
        self.enterContext(patch.object(d.capture, "implementation_pins", return_value=[]))
        validator = self.enterContext(patch.object(d.release, "validate_execution_envelope",
                                                   return_value=declaration))
        return authority, inputs, declaration, output, validator

    def test_real_binding_interface_selects_only_issued_output(self):
        _, _, _, output, validator = self.binding_fixture()
        binding = d.manager_binding()
        self.assertEqual(binding["output"], output)
        validator.assert_called_once_with(binding["envelope"])

    def test_preallocated_science_identity(self):
        self.assertEqual(d.MISSION, "8ee0b3aaeecb")
        self.assertEqual(d.AUTHORITY.name, "stage11-science-8ee0b3aaeecb-attempt001")

    def test_extra_launch_argument_rejected(self):
        _, _, declaration, _, _ = self.binding_fixture()
        declaration["launch"]["argv"].append("--extra")
        with self.assertRaises((RuntimeError, ValueError)):
            d.manager_binding()

    def test_occupied_issued_output_rejected_before_dispatch(self):
        self.binding_fixture()
        with patch.object(d.Path, "exists", return_value=True), \
                patch.object(d, "captured_launch") as dispatch:
            with self.assertRaisesRegex((RuntimeError, ValueError), "occupied output"):
                d.execute()
            dispatch.assert_not_called()

    def test_manager_boolean_budget_rejected(self):
        authority, _, _, _, validator = self.binding_fixture()
        authority["scientific_run_budget"] = True
        with self.assertRaises((RuntimeError, ValueError)):
            d.manager_binding()
        validator.assert_not_called()

    def test_stale_manager_candidate_rejected(self):
        _, inputs, _, _, validator = self.binding_fixture()
        inputs["candidate"] = dict(inputs["candidate"], sha256="stale")
        with self.assertRaises((RuntimeError, ValueError)):
            d.manager_binding()
        validator.assert_not_called()

    def test_historical_manager_output_rejected(self):
        _, _, declaration, _, _ = self.binding_fixture()
        declaration["launch"]["argv"][-1] = str(d.OUTPUT)
        with self.assertRaises((RuntimeError, ValueError)):
            d.manager_binding()

    def test_output_override_rejected(self):
        self.binding_fixture()
        with self.assertRaises((RuntimeError, ValueError)):
            d.manager_binding(d.ROOT / "build/not-issued")

    def test_missing_output_argument_rejected(self):
        _, _, declaration, _, _ = self.binding_fixture()
        del declaration["launch"]["argv"][-2:]
        with self.assertRaises((RuntimeError, ValueError)):
            d.manager_binding()

    def test_manager_rejection_prevents_dispatch_and_writes(self):
        _, _, _, _, validator = self.binding_fixture()
        validator.side_effect = RuntimeError("independent review/runtime gate")
        with patch.object(d.Path, "mkdir") as mkdir, \
                patch.object(d, "captured_launch") as launch, \
                patch.object(d.retained, "authenticate") as authenticate:
            with self.assertRaisesRegex(RuntimeError, "independent review/runtime gate"):
                d.execute()
            mkdir.assert_not_called()
            launch.assert_not_called()
            authenticate.assert_not_called()

    def test_validation_does_not_consume_or_reissue_live_authority(self):
        _, _, _, output, validator = self.binding_fixture()
        self.assertEqual(d.manager_binding(live=False)["output"], output)
        validator.assert_not_called()

    def test_missing_binding_cli_is_one_unknown_document(self):
        for mode in ("--execute", "--validate"):
            with self.subTest(mode=mode), \
                    patch.object(d.sys, "argv", [str(d.SOURCE), mode]), \
                    patch.object(d, "record", side_effect=FileNotFoundError(
                        str(d.AUTHORITY / "build-authorization.json"))), \
                    patch.object(d.sys, "stdout", new_callable=io.StringIO) as stdout, \
                    patch.object(d.retained, "authenticate") as authenticate:
                with self.assertRaises(SystemExit) as exit_status:
                    d.main()
                self.assertEqual(exit_status.exception.code, 1)
                result = json.loads(stdout.getvalue())
                self.assertEqual(result["status"], "UNKNOWN")
                self.assertEqual(result["expected_binding"]["authority_path"],
                                 str(d.AUTHORITY / "build-authorization.json"))
                self.assertEqual(result["expected_binding"]["authority_schema"]
                                 ["scientific_run_budget"], 1)
                self.assertFalse(result["automatic_replay_authorized"])
                authenticate.assert_not_called()


def _source_stability_entry(entry, defect):
    """Real entry/census/write guards; synthetic authority, inputs and suffix only."""
    role = "ace3.model._value_source_stability_fixture"
    extra_role = role + "_added"
    module = ModuleType(role)
    module.__file__ = str(d.SOURCE)
    with ExitStack() as stack:
        directory = stack.enter_context(tempfile.TemporaryDirectory(
            prefix="test-value-source-stability-", dir=d.ROOT / "build"))
        output = d.Path(directory)
        output.rmdir()  # Only the empty directory just allocated by this fixture.
        stack.enter_context(patch.dict(d.sys.modules, {role: module}))
        stack.enter_context(patch.dict(d.os.environ, {"PYTHONPATH": str(d.ROOT)}))
        stack.enter_context(patch.object(d.sys, "dont_write_bytecode", True))
        stack.enter_context(patch.object(d.final, "PYTHON", d.sys.executable))
        stack.enter_context(patch.object(d.native.torch, "set_num_threads"))
        contexts = []
        source_context = d.final.source_context

        def observed_context():
            context = source_context()
            contexts.append(dict(context))
            return context

        stack.enter_context(patch.object(d.final, "source_context", side_effect=observed_context))
        origins = {
            **source_context(),
            "value_sufficiency_source": d.record(d.SOURCE),
            "value_sufficiency_test": d.record(d.TEST),
            "stage13_suffix": d.record(d.layer.SOURCE),
            "stage13_tests": d.record(d.layer.TEST),
        }
        fp16, binary64 = d.record(d.SOURCE), d.record(d.TEST)
        summary = {
            "thresholds": {"synthetic_source_stability_fixture": True},
            "assets": {"synthetic": True},
            "L23_original_reference": {"reference": {
                "fp16": fp16, "binary64": binary64, "prior_kv": "own empty P0"}},
            "final_reference": {"reference": {"input_fp16": fp16}},
        }
        baseline = {
            "preflight": summary,
            "controls": [{"control": label, "parent": {"terminal_archive": fp16}}
                         for label in d.CONTROLS],
        }
        binding = {
            "output": output, "pins": [fp16, binary64],
            "declaration": {
                "runtime_sources": origins,
                "contract": {
                    "controls": list(d.CONTROLS), "thresholds": summary["thresholds"],
                    "inputs": {label: fp16 for label in d.CONTROLS},
                    "references": {
                        "original_input_L23_fp16": fp16,
                        "original_input_L23_binary64": binary64,
                        "original_input_final_fp16": fp16,
                        "original_input_final_binary64": binary64,
                    },
                },
            },
        }
        actual, trajectory = ValueSufficiencyTests().fixture()
        old_logits = np.zeros(34320, dtype="<u2")
        references = {"logits_fp16": old_logits,
                      "logits_binary64": np.zeros(34320, dtype="<f8")}
        prepared = []
        stages = []
        focused_reports = []

        def mutate():
            if defect.endswith("added"):
                d.sys.modules[extra_role] = module
            elif defect == "setup-removed":
                del d.sys.modules[role]
            elif defect == "setup-rebound":
                module.__file__ = str(d.TEST)

        def authenticate():
            prepared.append("authenticate")
            if defect.startswith("setup-"):
                mutate()
            return (baseline, {label: {"logits": old_logits} for label in d.CONTROLS},
                    references, [fp16, binary64])

        def synthetic_suffix(tensors, parent, arrays):
            for stage in range(10, 19):
                stages.append(stage)
                arrays[f"stage{stage:02d}"] = np.zeros(896, dtype="<u2")
                if stage == 16:
                    arrays["s16_unrounded_binary64"] = np.zeros(896, dtype="<f8")
                yield stage

        focused_tests = d.focused_tests
        manager_binding = d.manager_binding

        def focused():
            before = source_context()
            with patch.object(d, "manager_binding", manager_binding):
                report = focused_tests()
            assert source_context() == before
            assert report["loaded"] == report["executed"] > 0
            assert not any(report[key] for key in (
                "failures", "errors", "skipped", "unexpected_successes"))
            focused_reports.append(report)
            if defect == "focused-added":
                mutate()
            # Install numerical doubles only after the real focused unit suite.
            stack.enter_context(patch.object(d, "predecessor_review", return_value=fp16))
            stack.enter_context(patch.object(d.retained, "authenticate", side_effect=authenticate))
            stack.enter_context(patch.object(d.layer, "load_inputs",
                                            return_value=({}, trajectory, {}, {})))
            stack.enter_context(patch.object(d.final, "load_operands", side_effect=lambda _: [
                np.zeros(896, dtype="<u2"), np.zeros((34320, 896), dtype="<f2")]))
            stack.enter_context(patch.object(d.final.preflight, "bind_assets",
                                            return_value=summary["assets"]))
            stack.enter_context(patch.object(d, "archive", return_value=actual))
            stack.enter_context(patch.object(d.layer.preflight.parent, "check_state"))
            stack.enter_context(patch.object(d, "suffix_stages", side_effect=synthetic_suffix))
            stack.enter_context(patch.object(d.layer, "expected_stage", return_value=None))
            stack.enter_context(patch.object(d.layer, "stage_report", side_effect=lambda stage, *args: {
                "stage": stage, "status": "FAIL"}))
            stack.enter_context(patch.object(d.final, "rmsnorm",
                                            return_value=(np.zeros(896, dtype="<u2"), {})))
            stack.enter_context(patch.object(d, "pair_logits",
                                            return_value=np.zeros(2, dtype="<u2")))
            return report

        stack.enter_context(patch.object(d, "focused_tests", side_effect=focused))
        stack.enter_context(patch.object(d, "manager_binding", return_value=binding))
        launch = stack.enter_context(patch.object(d, "captured_launch"))
        stack.enter_context(patch.object(
            d.sys, "argv", [str(d.SOURCE), "--" + entry, "--output", str(output)]))
        stdout = stack.enter_context(patch.object(d.sys, "stdout", new_callable=io.StringIO))
        if defect == "stable":
            d.main()
        else:
            with unittest.TestCase().assertRaises(SystemExit) as failure:
                d.main()
            assert failure.exception.code == 1
        document = json.loads(stdout.getvalue())
        launch.assert_called_once_with(binding)
        assert len(focused_reports) == 1
        assert prepared == ["authenticate"]
        assert stages == list(range(10, 19)) * 9
        assert len(contexts) == 2
        aliases = {"value_sufficiency_source", "value_sufficiency_test",
                   "stage13_suffix", "stage13_tests"}
        assert origins == {**contexts[0], **{key: origins[key] for key in aliases}}
        assert set(origins) - contexts[0].keys() == aliases
        compile_report = json.loads((output / "compile_tests.json").read_text())
        assert compile_report["loaded"] == compile_report["executed"] == focused_reports[0]["loaded"]
        assert {pin["path"] for pin in compile_report["compiled"]} == {
            pin["path"] for pin in origins.values()}
        if defect == "stable":
            assert contexts[0] == contexts[1]
            assert document["status"] == "REJECTED"  # Deliberately failing synthetic gates.
            result = json.loads((output / "result.json").read_text())
            assert result["origins"] == origins
            assert result["audit"]["file_write_opens"] == 12
            assert not (output / "failure.json").exists()
        else:
            assert document["status"] == "UNKNOWN"
            assert document["automatic_replay_authorized"] is False
            result = json.loads((output / "failure.json").read_text())
            assert result["error"] == document["error"]
            prefix = "source changed during execution: "
            assert result["error"].startswith(prefix)
            changed_role = extra_role if defect.endswith("added") else role
            assert json.loads(result["error"][len(prefix):]) == {
                changed_role: {
                    "expected": contexts[0].get(changed_role),
                    "actual": contexts[1].get(changed_role),
                },
            }
            assert result["automatic_replay_authorized"] is False
            assert not (output / "result.json").exists()
        assert result["audit"]["forbidden_calls"] == 0
        assert result["audit"]["retained_evidence_writes"] == 0
        assert result["audit"]["native_L0_L22_invocations"] == 0
        assert result["audit"]["native_S0_S9_invocations"] == 0
        assert len(result["controls"]) == 9
        assert result["flags"] == d.FLAGS


def test_source_context_stable_actual_entry():
    for entry in ("check", "execute"):
        _source_stability_entry(entry, "stable")


def test_source_context_drift_actual_entry():
    for entry, defect in (
        ("execute", "focused-added"),
        ("check", "setup-added"),
        ("check", "setup-removed"),
        ("check", "setup-rebound"),
    ):
        _source_stability_entry(entry, defect)
