#!/usr/bin/env python3
"""Focused tests for repair8 lm_head/position-2 preparation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ace3/model"
ONE_SHOT_LAUNCHER = MODEL / "model24_one_shot_python_launcher.py"
RECEIPTFIX = MODEL / "prepare_model24_receiptfix_package.py"
LAUNCH_CONTRACT = ROOT / "ace3/contracts/model24_r10_durable_launch_contract.json"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from prepare_repair8_lm_head_position2 import (  # noqa: E402
    PreparationError,
    load_terminal_bits,
    terminal_payload,
    topk_from_bits,
)
from streaming_lm_head_reference import decode_f16_q24  # noqa: E402

RECEIPTFIX_SPEC = importlib.util.spec_from_file_location("model24_receiptfix", RECEIPTFIX)
RECEIPTFIX_MODULE = importlib.util.module_from_spec(RECEIPTFIX_SPEC)
assert RECEIPTFIX_SPEC.loader is not None
RECEIPTFIX_SPEC.loader.exec_module(RECEIPTFIX_MODULE)


class Repair8LmHeadPosition2Tests(unittest.TestCase):
    @staticmethod
    def _snapshot(root: Path) -> dict[str, tuple[int, int, int, str]]:
        snapshot = {}
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            payload = path.read_bytes()
            status = path.stat()
            snapshot[str(path.relative_to(root))] = (
                status.st_mode,
                status.st_mtime_ns,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
        return snapshot

    def test_terminal_parser_preserves_little_endian_fp16_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.hex"
            path.write_text("0000003c00\n000001bc00\n", encoding="ascii")
            bits = load_terminal_bits(path, hidden_size=2)
            self.assertEqual(bits, [0x3C00, 0xBC00])
            self.assertEqual(terminal_payload(bits), b"\x00<\x00\xbc")

    def test_terminal_parser_rejects_reordered_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.hex"
            path.write_text("0000013c00\n000000bc00\n", encoding="ascii")
            with self.assertRaisesRegex(PreparationError, "feature index"):
                load_terminal_bits(path, hidden_size=2)

    def test_topk_uses_ascending_token_id_for_equal_fp16_logits(self) -> None:
        class Reference:
            decode_f16_q24 = staticmethod(decode_f16_q24)

        bits = np.asarray([0x3C00, 0x4000, 0x4000, 0xBC00], dtype="<u2")
        winners = topk_from_bits(bits, Reference, top_k=3)
        self.assertEqual([item[0] for item in winners], [1, 2, 0])

    def test_topk_rejects_nonfinite_logits(self) -> None:
        class Reference:
            decode_f16_q24 = staticmethod(decode_f16_q24)

        with self.assertRaisesRegex(PreparationError, "nonfinite"):
            topk_from_bits(np.asarray([0x3C00, 0x7C00], dtype="<u2"), Reference, top_k=1)

    def test_one_shot_launcher_binds_torch_capable_python_before_consumption(self) -> None:
        system_python = Path("/usr/bin/python3")
        self.assertTrue(system_python.is_file())
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            consumed = scratch / "position2_model24_exactly_once_v1"
            state = consumed / "sealed_state"
            state.mkdir(parents=True)
            (consumed / "authority.json").write_text('{"consumed":true}\n', encoding="utf-8")
            (state / "terminal.json").write_text('{"terminal":true}\n', encoding="utf-8")
            entrypoint = scratch / "consume_authority.py"
            marker = scratch / "entrypoint-invoked"
            entrypoint.write_text(
                "from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text('invoked\\n')\n",
                encoding="utf-8",
            )
            before = self._snapshot(consumed)

            rejected_receipt = scratch / "rejected-python.json"
            rejected = subprocess.run(
                [
                    str(system_python),
                    str(ONE_SHOT_LAUNCHER),
                    "--project-python",
                    str(system_python),
                    "--entrypoint",
                    str(entrypoint),
                    "--receipt",
                    str(rejected_receipt),
                    "--dry-run",
                    "--",
                    str(marker),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("dependency probe failed", rejected.stderr)
            self.assertFalse(rejected_receipt.exists())
            self.assertFalse(marker.exists())
            self.assertEqual(self._snapshot(consumed), before)

            accepted_receipt = scratch / "accepted-python.json"
            accepted = subprocess.run(
                [
                    str(system_python),
                    str(ONE_SHOT_LAUNCHER),
                    "--project-python",
                    sys.executable,
                    "--entrypoint",
                    str(entrypoint),
                    "--receipt",
                    str(accepted_receipt),
                    "--dry-run",
                    "--",
                    str(marker),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            receipt = json.loads(accepted_receipt.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["project_python"]["path"],
                str(Path(sys.executable).resolve()),
            )
            self.assertTrue(receipt["project_python"]["required_imports"]["torch"])
            self.assertEqual(receipt["mode"], "dry-run")
            self.assertFalse(receipt["authority_consumed_by_launcher"])
            self.assertFalse(receipt["entrypoint_invoked_at_receipt_creation"])
            self.assertFalse(marker.exists())
            self.assertEqual(self._snapshot(consumed), before)

    def test_receiptfix_contract_requires_review_and_package_hash_chain(self) -> None:
        contract = json.loads(LAUNCH_CONTRACT.read_text(encoding="utf-8"))
        required = set(contract["execution_authority_schema"]["required_fields"])
        self.assertIn("package_acceptance_sha256", required)
        self.assertIn("package_manifest_sha256", required)
        self.assertIn("package_seal_sha256", required)
        self.assertFalse(
            contract["accepted_review"]["package_acceptance_is_execution_authority"]
        )
        self.assertEqual(
            set(contract["validation_modes"]), {"package", "review", "launch"}
        )

    def test_receiptfix_validator_matches_pre_child_durable_namespace(self) -> None:
        namespace = {"__name__": "receiptfix_validator_test"}
        exec(
            compile(
                RECEIPTFIX_MODULE.VALIDATOR_SOURCE,
                "validate-package.py",
                "exec",
            ),
            namespace,
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = base / "package"
            package.mkdir()
            task = "ace3-model24-r10-test-1234"
            manifest = {
                "identity": {"task_id": task},
                "execution": {
                    "cwd": str(base),
                    "launcher_argv": [str(package / "launch-once.sh")],
                },
                "namespaces": {
                    "receipt_namespace": str(base / ".argus_subagents")
                },
            }
            self.assertIsNone(
                namespace["validate_receipt_namespace"]("package", manifest)
            )
            self.assertIsNone(
                namespace["validate_receipt_namespace"]("review", manifest)
            )
            root = base / ".argus_subagents"
            logs = root / f"{task}_logs"
            logs.mkdir(parents=True)
            for name in ("stdout.log", "stderr.log"):
                (logs / name).write_text("", encoding="ascii")
            (root / f"{task}.json").write_text(
                json.dumps(
                    {
                        "state": "starting",
                        "task_id": task,
                        "run_id": f"{task}-123456789",
                        "mode": "direct",
                        "cwd": str(base),
                        "command": str(package / "launch-once.sh"),
                        "worker_pid": os.getpid(),
                        "submitted_at": time.time(),
                    }
                ),
                encoding="ascii",
            )
            receipt = namespace["validate_receipt_namespace"]("launch", manifest)
            self.assertEqual(receipt["task_id"], task)
            (root / "foreign.json").write_text("{}\n", encoding="ascii")
            with self.assertRaisesRegex(SystemExit, "foreign or extra"):
                namespace["validate_receipt_namespace"]("launch", manifest)

    def test_receiptfix_is_static_until_explicit_prepare_and_binds_runner_order(
        self,
    ) -> None:
        source = RECEIPTFIX.read_text(encoding="utf-8")
        prepare_start = source.index("def prepare(")
        prepare_source = source[
            prepare_start : source.index("def main()", prepare_start)
        ]
        self.assertNotIn("argus_skill.tools.subagent", prepare_source)
        self.assertIn('"execution_invocations": 0', source)
        launcher = RECEIPTFIX_MODULE.render_launcher(
            Path("/tmp/prep"), Path("/tmp/output"), "0123456789abcdef"
        ).decode("utf-8")
        self.assertIn("--mode launch", launcher)

        cli = (RECEIPTFIX_MODULE.RUNNER_ROOT / "_cli.py").read_text(
            encoding="utf-8"
        )
        direct = (RECEIPTFIX_MODULE.RUNNER_ROOT / "_direct_run.py").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            cli.index("_write_task(task_id, initial_task)"),
            cli.index("pid = os.fork()"),
        )
        self.assertLess(
            direct.index("log_dir.mkdir(parents=True, exist_ok=True)"),
            direct.index("_launch_durable_command("),
        )
        self.assertLess(
            direct.index('stdout_path = log_dir / "stdout.log"'),
            direct.index("_launch_durable_command("),
        )


if __name__ == "__main__":
    unittest.main()
