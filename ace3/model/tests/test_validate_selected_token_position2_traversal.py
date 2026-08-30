#!/usr/bin/env python3
"""Focused tests for durable selected-token continuation validation."""

from __future__ import annotations

import inspect
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys
from typing import Any

import numpy as np

MODEL = Path(__file__).resolve().parents[1]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from validate_selected_token_position2_traversal import (  # noqa: E402
    CONSUMED_SOURCE_PATHS,
    ContinuationError,
    ROOT,
    assemble_evidence,
    consumed_source_records,
    execute_exact_transaction,
    execute_live_traversal,
    exact_hex_comparison,
    generate,
    main,
    parse_natural_terminal,
    require_consumed_source_bindings,
    require_ordered_layers,
    selected_accurate_silu_gate,
    semantic_hidden_sha256,
    sha256_bytes,
    validate,
)

EXPECTED_DECODER_RTL_PATHS = {
    "ace3/rtl/ace3_fp16_fixed.sv",
    "ace3/rtl/ace3_q47_48_to_f16_rne.sv",
    "ace3/rtl/ace3_awq_w4a16_g128_dot_lane.sv",
    "ace3/rtl/ace3_awq_w4a16_projection_engine.sv",
    "ace3/rtl/ace3_fp16_rmsnorm_core.sv",
    "ace3/rtl/ace3_fp16_residual_add_core.sv",
    "ace3/rtl/ace3_fp16_silu_gate_core.sv",
    "ace3/rtl/ace3_qwen2_rope_pair.sv",
    "ace3/rtl/ace3_fp16_kv_cache.sv",
    "ace3/rtl/ace3_attention_score_core.sv",
    "ace3/rtl/ace3_attention_softmax_core.sv",
    "ace3/rtl/ace3_attention_value_core.sv",
    "ace3/rtl/ace3_decoder_qzeros_address.sv",
    "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
}


class SelectedTokenPosition2ValidationTests(unittest.TestCase):
    def fake_source_closure(self, root: Path) -> dict[str, dict[str, Any]]:
        for index, relative_path in enumerate(CONSUMED_SOURCE_PATHS.values()):
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"source-{index}\n", encoding="ascii")
        return consumed_source_records(root)

    def test_consumed_source_closure_covers_validation_dependencies(self) -> None:
        records = consumed_source_records()
        self.assertEqual(set(records), set(CONSUMED_SOURCE_PATHS))
        self.assertEqual(
            {
                label: Path(record["path"]).relative_to(ROOT).as_posix()
                for label, record in records.items()
            },
            CONSUMED_SOURCE_PATHS,
        )
        self.assertFalse(
            any(Path(path).parts[0] == "build" for path in CONSUMED_SOURCE_PATHS.values())
        )
        self.assertEqual(
            CONSUMED_SOURCE_PATHS["projection_oracle"],
            "ace3/model/projection_oracle.py",
        )
        self.assertEqual(
            {
                path
                for label, path in CONSUMED_SOURCE_PATHS.items()
                if label.endswith("_rtl")
            },
            EXPECTED_DECODER_RTL_PATHS,
        )

    def test_tampered_live_dependency_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stored = self.fake_source_closure(root)
            dependency = root / CONSUMED_SOURCE_PATHS["official_model24_next_token"]
            dependency.write_text("tampered-source\n", encoding="ascii")
            with self.assertRaisesRegex(
                ContinuationError,
                "official_model24_next_token source binding mismatch",
            ):
                require_consumed_source_bindings(stored, root)

    def test_stale_dependency_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stored = self.fake_source_closure(root)
            stored["model24_oracle"] = {
                **stored["model24_oracle"],
                "sha256": "0" * 64,
            }
            with self.assertRaisesRegex(
                ContinuationError,
                "model24_oracle source binding mismatch",
            ):
                require_consumed_source_bindings(stored, root)

    def test_semantic_hidden_digest_uses_ordered_little_endian_fp16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.hex"
            path.write_text("0000003c00\n000001bc00\n", encoding="ascii")
            self.assertEqual(
                semantic_hidden_sha256(path, expected_rows=2),
                sha256_bytes(b"\x00<\x00\xbc"),
            )

    def test_semantic_hidden_rejects_reordered_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.hex"
            path.write_text("0000013c00\n000000bc00\n", encoding="ascii")
            with self.assertRaisesRegex(ContinuationError, "ordering mismatch"):
                semantic_hidden_sha256(path, expected_rows=2)

    def test_exact_hex_comparison_records_first_trace_difference(self) -> None:
        comparison = exact_hex_comparison(
            b"0000020a00071af8\n",
            b"0000020a00071af9\n",
            "trace",
        )
        self.assertEqual(comparison["mismatch_count"], 1)
        self.assertFalse(comparison["exact_match"])
        self.assertEqual(
            comparison["first_difference"],
            {
                "row": 0,
                "actual": "0000020a00071af8",
                "expected": "0000020a00071af9",
                "signal": "trace[position=2,stage=10,index=7]",
                "actual_bits": "0x1af8",
                "expected_bits": "0x1af9",
            },
        )

    def test_exact_hex_comparison_records_no_difference(self) -> None:
        payload = b"0000003c00\n000001bc00\n"
        comparison = exact_hex_comparison(payload, payload, "final_hidden")
        self.assertTrue(comparison["exact_match"])
        self.assertEqual(comparison["mismatch_count"], 0)
        self.assertIsNone(comparison["first_difference"])

    def test_selected_q24_exp_silu_preserves_binary64_reference_distinction(
        self,
    ) -> None:
        gate_bits = 0xBB29
        up_bits = 0x2D20
        q24_bits, invalid, saturated = selected_accurate_silu_gate(
            gate_bits,
            up_bits,
        )
        gate = float(
            np.asarray(gate_bits, dtype="<u2").view("<f2").astype(np.float64)
        )
        up = float(
            np.asarray(up_bits, dtype="<u2").view("<f2").astype(np.float64)
        )
        binary64_bits = int(
            np.asarray(
                (gate / (1.0 + math.exp(-gate))) * up,
                dtype="<f2",
            ).view("<u2")
        )

        self.assertFalse(invalid)
        self.assertFalse(saturated)
        self.assertEqual(q24_bits, 0xA552)
        self.assertEqual(binary64_bits, 0xA553)
        self.assertNotEqual(q24_bits, binary64_bits)

    def test_canonical_route_has_only_fresh_current_execution(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(execute_live_traversal).parameters),
            ("live_root", "embedding_bits"),
        )
        source = inspect.getsource(execute_live_traversal)
        self.assertIn(
            'require(not live_root.exists(), "live traversal output already exists")',
            source,
        )
        self.assertIn("position0_bits = token_embedding_bits(", source)
        self.assertIn("position1_bits = token_embedding_bits(", source)
        self.assertEqual(source.count("execute_exact_transaction("), 3)

        generation_source = inspect.getsource(generate)
        authenticated = generation_source.index("official_embedding_binding()")
        selected_bits = generation_source.index("selected_embedding_bits()")
        execution = generation_source.index("execute_live_traversal(")
        self.assertLess(authenticated, selected_bits)
        self.assertLess(selected_bits, execution)

    def test_fresh_evidence_has_no_historical_package_dependency(self) -> None:
        embedding = {"token_id": 271, "sha256": "a" * 64}
        document = assemble_evidence(embedding, {"status": "COMPLETE"})
        self.assertEqual(document["schema_version"], 3)
        self.assertEqual(
            document["kind"],
            "ace3_selected_token_position2_fresh_traversal_evidence",
        )
        self.assertEqual(
            document["selected_token"],
            {
                "token_id": 271,
                "source": "authenticated original checkpoint embedding",
                "embedding_sha256": "a" * 64,
            },
        )
        source = (MODEL / "validate_selected_token_position2_traversal.py").read_text(
            encoding="utf-8"
        )
        stale_bindings = (
            "".join(("accepted", "_tied_head_binding")),
            "_".join(("V1", "PACKAGE")),
            "_".join(("V4", "PACKAGE")),
            "_".join(("V4", "REVIEW")),
            "".join(("repair", "8_lm_head_position2")),
        )
        for binding in stale_bindings:
            self.assertNotIn(binding, source)

    def test_old_evidence_identity_is_rejected_before_artifact_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence.json"
            evidence.write_text(
                '{"schema_version": 2, "kind": "obsolete", "status": "COMPLETE"}',
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                ContinuationError,
                "fresh traversal evidence identity mismatch",
            ):
                validate(evidence)

    def test_exact_trace_and_final_hidden_are_hard_gates(self) -> None:
        source = inspect.getsource(execute_exact_transaction)
        self.assertIn('trace_comparison["exact_match"]', source)
        self.assertIn('final_comparison["exact_match"]', source)
        self.assertIn("require(", source)

    def test_stale_ancestry_markers_are_absent_from_canonical_surfaces(self) -> None:
        upper = (
            "_".join(("PRIOR", "ROOT")),
            "_".join(("PRIOR", "RESULT")),
            "_".join(("REPAIR8", "ROOT")),
        )
        call = "".join(("predecessor", "_state_records()"))
        operations = (
            "-".join(("generate", "reduced", "layer0")),
            "-".join(("validate", "reduced", "layer0")),
            "".join(("re", "sume")),
            "-".join(("reduced", "predecessor")),
            "-".join(("scratch", "successor")),
        )
        markers = (*upper, call, *operations)
        validator = MODEL / "validate_selected_token_position2_traversal.py"
        makefile = ROOT / "Makefile"
        make_source = makefile.read_text(encoding="utf-8")
        target_source = make_source.split(
            "model24-selected-token-position2-tests:",
            1,
        )[1].split("model24-selected-token-position2-lm-head-tests:", 1)[0]
        surfaces = (
            validator.read_text(encoding="utf-8"),
            Path(__file__).read_text(encoding="utf-8"),
            target_source,
        )
        for marker in markers:
            for surface in surfaces:
                self.assertNotIn(marker, surface)

    def test_cli_exposes_only_canonical_operations(self) -> None:
        source = inspect.getsource(main)
        self.assertIn('choices=("generate", "validate")', source)
        unavailable = "-".join(("generate", "reduced", "layer0"))
        completed = subprocess.run(
            [
                sys.executable,
                str(MODEL / "validate_selected_token_position2_traversal.py"),
                unavailable,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("invalid choice", completed.stderr)

    def test_layer_order_rejects_duplicate_or_missing_layer(self) -> None:
        ordered = [{"layer_index": index} for index in range(24)]
        require_ordered_layers(ordered)
        ordered[8] = {"layer_index": 7}
        with self.assertRaisesRegex(ContinuationError, "ordered 0 through 23"):
            require_ordered_layers(ordered)

    def test_natural_terminal_rejects_duplicate_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terminal.txt"
            path.write_text(
                "schema=ace3_decoder_token_transaction_v1 "
                "layer_index=0 position=2 natural_terminal=1 "
                "natural_terminal=1 exit_code=0 trace_count=1 "
                "final_count=896 done_count=1\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ContinuationError, "duplicated"):
                parse_natural_terminal(path, 0)

    def test_natural_terminal_rejects_failed_simulator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terminal.txt"
            path.write_text(
                "schema=ace3_decoder_token_transaction_v1 "
                "layer_index=0 position=2 natural_terminal=0 "
                "exit_code=1 trace_count=1 final_count=0 done_count=0\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ContinuationError, "natural terminal"):
                parse_natural_terminal(path, 0)

    def test_natural_terminal_accepts_explicit_replay_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terminal.txt"
            path.write_text(
                "schema=ace3_decoder_token_transaction_v1 "
                "layer_index=3 position=1 natural_terminal=1 "
                "exit_code=0 trace_count=1 final_count=896 done_count=1\n",
                encoding="ascii",
            )
            self.assertEqual(
                parse_natural_terminal(path, 3, position=1),
                {"trace_count": 1, "final_count": 896, "done_count": 1},
            )


if __name__ == "__main__":
    unittest.main()
