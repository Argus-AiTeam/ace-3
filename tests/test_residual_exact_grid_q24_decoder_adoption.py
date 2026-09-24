"""Declarative checks only: do not import Q24 arithmetic, codec or decoder code."""

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "ace3/model/candidates/residual_exact_grid_q24_decoder_adoption_static_v1.py"
SPEC = importlib.util.spec_from_file_location("q24_adoption_static", CHECKER)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load the Q24 static specification checker")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class Q24DecoderAdoptionStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = checker.load_json(checker.PACKET.read_bytes())

    def test_schema_authority_and_frozen_bindings(self):
        checker.validate_schema(self.packet)
        self.assertEqual(checker.validate_bindings(self.packet), 13)
        self.assertEqual(sys.version.split()[0], self.packet["validation"]["python_version"])
        self.assertTrue((ROOT / self.packet["specification"]).is_file())

    def test_compile_and_import_boundary(self):
        source = CHECKER.read_text()
        compile(source, str(CHECKER), "exec")
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertEqual(imported, {"hashlib", "json", "pathlib"})
        self.assertFalse(hasattr(checker, "main"))

    def test_reject_duplicate_nonfinite_and_unknown_schema(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '[]'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                checker.load_json(text)
        for key in ("schema_version", "unexpected"):
            packet = deepcopy(self.packet)
            packet[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                checker.validate_schema(packet)
        packet = deepcopy(self.packet)
        packet["state"]["implicit_carry"] = 0
        with self.assertRaisesRegex(ValueError, "unknown or missing fields: state"):
            checker.validate_schema(packet)

    def test_reject_invented_parents_and_scope_promotion(self):
        changes = [
            ("state", "fp16_only_nonroot_seed", True),
            ("state", "reference_derived_carry", True),
            ("state", "primitive_fixture_as_model_parent", True),
            ("state", "admitted_q24_parent", "accepted-fp16-L8"),
            ("state", "root_artifact", "invented-Q24-root"),
            ("state", "rounding", "truncate"),
            ("state", "finite_view_integer_limit_exclusive", 1099511627776),
            ("scope", "specified_layers", list(range(24))),
            ("scope", "positions", [0, 1]),
            ("scope", "execution_cone", ["L0"]),
            ("scope", "excluded", []),
            ("authority", "runtime_launch_authorized", True),
            ("authority", "production_policy_adopted", True),
            ("authority", "independent_review", "PASS"),
            ("status", "L0_L9_numerical_correctness", "PASS"),
            ("precision", "unchanged_strict_w4a16", True),
            ("identities", "decoder_policy_id",
             self.packet["identities"]["primitive_policy_id"]),
            ("references", "global_candidate_reseeding", True),
            ("references", "s13_input", "unrounded_T"),
            ("references", "local_gate", "abs_error <= 1"),
            ("references", "s18", "native_FP16_only"),
        ]
        for section, key, value in changes:
            packet = deepcopy(self.packet)
            packet[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                checker.validate_schema(packet)

    def test_reject_changed_binding_without_rewriting_history(self):
        packet = deepcopy(self.packet)
        packet["bindings"]["mission"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source identity mismatch: mission"):
            checker.validate_bindings(packet)
