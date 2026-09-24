import ast
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ace3.model.candidates import analyze_q24_s16_l9_coordinate62_producer_cone_v1 as a


def reports(layer):
    rows = [{
        "index": i, "profile_id": a.PROFILE, "reference_policy": a.REFERENCE,
        "excess_budget": "1/8", "actual_error": "0", "q": "0",
        "excess_error": "0", "accepted": True,
    } for i in range(896)]
    summary = {"comparisons": 896, "failure_count": 0, "failures": [], "passed": True}
    result = []
    for stage in range(19):
        gate = {
            "node": [layer, 0, stage], "stage": stage, "policy_id": a.POLICY,
            "kv_lineage": "PASS", "residual_state_lineage": "PASS",
            "fp16_role": "independent-whole-FP16-trajectory-diagnostic",
            "fp16": copy.deepcopy(summary), "status": "PASS",
        }
        if stage == 18:
            gate["binary64_v1"] = {
                "coordinates": 896, "failure_count": 0, "failures": [],
                "passed": True, "profile_id": a.PROFILE, "role": "mandatory", "rows": rows,
            }
        else:
            gate["local_operator_fp16"] = {**copy.deepcopy(summary), "role": "mandatory"}
        result.append(gate)
    return result


class AnalyzerTests(unittest.TestCase):
    def test_live_repository_module_origins_without_native_import(self):
        before = set(a.sys.modules)
        for name in (a.MODULE, a.TEST_MODULE, a.SOURCE_MODULE, a.SOURCE_TEST):
            self.assertTrue(Path(a.module_origin(name)["path"]).is_relative_to(a.ROOT))
        self.assertEqual(before, set(a.sys.modules))

    def test_foreign_loaded_origin_rejected(self):
        with patch.object(a.sys.modules[a.MODULE], "__file__", "/tmp/foreign.py"):
            with self.assertRaisesRegex(ValueError, "loaded origin"):
                a.module_origin(a.MODULE)

    def test_binding_hash_and_source_escape_rejected(self):
        with tempfile.TemporaryDirectory(prefix=a.PREFIX + "test_", dir=a.ROOT / "build") as tmp:
            path = Path(tmp) / "source.py"
            path.write_text("pass\n")
            record = {"path": str(path), "bytes": 5,
                      "sha256": hashlib.sha256(b"pass\n").hexdigest()}
            self.assertEqual(a.Bindings().bind(record), path)
            path.write_text("fail\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                a.Bindings().bind(record)
            link = Path(tmp) / "foreign"
            link.symlink_to("/tmp")
            with self.assertRaisesRegex(ValueError, "non-repository"):
                a.repository_path(str(link / "source.py"))

    def test_conflicting_duplicate_binding_rejected(self):
        with tempfile.TemporaryDirectory(prefix=a.PREFIX + "test_", dir=a.ROOT / "build") as tmp:
            path = Path(tmp) / "source.py"
            path.write_bytes(b"")
            record = {"path": str(path), "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
            bindings = a.Bindings()
            bindings.bind(record)
            with self.assertRaisesRegex(ValueError, "conflicting"):
                bindings.bind({**record, "sha256": "0" * 64})

    def test_prefix_requires_corrected_existing_build_root(self):
        with tempfile.TemporaryDirectory(prefix=a.PREFIX + "test_", dir=a.ROOT / "build") as tmp:
            self.assertEqual(a.check_root(Path(tmp)), Path(tmp))
        for root in (a.ROOT / "build/q24_s16_l9_coordinate62_entry_producer_cone_bad",
                     a.ROOT / "q24_s16_l9_coordinate62_producer_cone_bad",
                     a.ROOT / "build/q24_s16_l9_coordinate62_producer_cone_missing"):
            with self.assertRaises(ValueError):
                a.check_root(root)

    def test_unreviewed_result_fails_before_source_loading(self):
        with tempfile.TemporaryDirectory(prefix=a.PREFIX + "test_", dir=a.ROOT / "build") as tmp:
            (Path(tmp) / "result.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "pinned reviewed"):
                a.analyze(Path(tmp))

    def test_complete_l9_l13_saved_reports(self):
        for layer in range(9, 14):
            self.assertEqual(len(a.check_reports(reports(layer), layer)), 19)

    def test_layer_stage_order_and_coverage_rejected(self):
        for layer in (0, 8, 14):
            with self.assertRaises(ValueError):
                a.check_reports(reports(layer), layer)
        gates = reports(9)
        for changed in (gates[:-1], [gates[1], gates[0], *gates[2:]]):
            with self.assertRaises(ValueError):
                a.check_reports(changed, 9)

    def test_policy_state_kv_and_mandatory_role_rejected(self):
        for key, value in (("policy_id", "different"), ("kv_lineage", "FAIL"),
                           ("residual_state_lineage", "FAIL"), ("status", "FAIL"),
                           ("fp16_role", "mandatory")):
            gates = reports(9)
            gates[0][key] = value
            with self.assertRaises(ValueError):
                a.check_reports(gates, 9)
        gates = reports(9)
        gates[0]["local_operator_fp16"]["role"] = "diagnostic"
        with self.assertRaises(ValueError):
            a.check_reports(gates, 9)

    def test_reference_threshold_and_coordinate_coverage_rejected(self):
        for key, value in (("reference_policy", "actual-seeded"), ("excess_budget", "1/4"),
                           ("profile_id", "other"), ("index", 61)):
            gates = reports(9)
            gates[18]["binary64_v1"]["rows"][62][key] = value
            with self.assertRaises(ValueError):
                a.check_reports(gates, 9)

    def test_exact_excess_threshold_and_preserved_failure(self):
        gates = reports(13)
        global_gate = gates[18]["binary64_v1"]
        row = global_gate["rows"][62]
        row.update(actual_error="5/8", q="1/2", excess_error="1/8")
        a.check_reports(gates, 13)
        row.update(actual_error="41/64", excess_error="9/64", accepted=False)
        global_gate.update(failures=[row], failure_count=1, passed=False)
        gates[18]["status"] = "FAIL"
        a.check_reports(gates, 13)
        row["accepted"] = True
        with self.assertRaises(ValueError):
            a.check_reports(gates, 13)

    def test_first_diagnostic_and_mandatory_divergence_are_distinct(self):
        actual = {layer: reports(layer) for layer in range(9, 14)}
        actual[9][12]["fp16"].update(failures=[{"index": 62}], failure_count=1, passed=False)
        actual[13][18]["binary64_v1"]["failures"] = [{"index": 62}]
        found = a.first_divergences(actual)
        self.assertEqual(found["first_material_divergence"]["node"], [9, 0, 12])
        self.assertEqual(found["first_mandatory_failure"]["node"], [13, 0, 18])
        self.assertEqual(found["first_material_divergence"]["role"], "diagnostic_only")

    def test_no_coordinate_failure_does_not_invent_divergence(self):
        actual = {layer: reports(layer) for layer in range(9, 14)}
        actual[9][0]["fp16"]["failures"] = [{"index": 61}]
        self.assertEqual(a.first_divergences(actual),
                         {"first_material_divergence": None, "first_mandatory_failure": None})

    def test_ranking_uses_exact_magnitudes_not_rescue_or_gate_failure(self):
        decomposition = {"signed_parts": {
            "incoming_Q24_minus_original": "1/4", "S11_minus_original": "-1/16",
            "S17_minus_original": "1/8", "final_FP16_projection": "-1/2",
            "negative_original_addition_roundoff": "0",
        }, "signed_error": "-3/16"}
        controls = {f"frozen_{name}": {
            "index62": {"accepted": name == "inherited"},
            "all_L10_L13_gates_pass": name == "inherited",
        } for name in ("inherited", "o", "down")}
        rows = a.rank_producers(decomposition, controls, {9: reports(9)})
        self.assertEqual([r["producer"] for r in rows], ["inherited", "down", "o"])
        self.assertEqual([r["rank"] for r in rows], [1, 2, 3])
        self.assertEqual(rows[0]["evidence_kind"], "frozen_input_only")
        self.assertIsNone(rows[0]["local_gate_status"])
        self.assertEqual(rows[2]["local_gate_status"], "PASS")
        self.assertEqual(rows[2]["signed_Q24_error_contribution"], "-1/16")
        self.assertEqual(rows, a.rank_producers(decomposition, controls, {9: reports(9)}))
        decomposition["signed_error"] = "0"
        with self.assertRaisesRegex(ValueError, "does not close"):
            a.rank_producers(decomposition, controls, {9: reports(9)})

    def test_analyzer_has_no_native_import_execution_or_file_writes(self):
        tree = ast.parse(Path(a.__file__).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, (
                    "write_text", "write_bytes", "mkdir", "unlink", "execute_layer",
                    "diagnose", "run", "system", "Popen", "import_module", "exec_module"))
                if node.func.attr == "open":
                    self.assertEqual(ast.literal_eval(node.args[0]), "rb")
        self.assertTrue(all(name.split(".")[0] in {
            "argparse", "fractions", "hashlib", "importlib", "json", "pathlib", "sys",
        } for name in imports))


if __name__ == "__main__":
    unittest.main()
