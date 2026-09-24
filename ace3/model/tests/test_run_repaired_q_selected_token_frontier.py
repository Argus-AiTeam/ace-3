from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import run_repaired_q_selected_token_frontier as frontier  # noqa: E402


class RepairedQSelectedTokenFrontierTests(unittest.TestCase):
    def test_trace_stages_normalize_indexed_records(self) -> None:
        trace = []
        for stage, count in frontier.STAGE_COUNTS.items():
            index_period = frontier.PERIODIC_STAGE_INDICES.get(stage)
            if index_period is None:
                trace.extend(
                    (stage, index, index, frontier.POSITION)
                    for index in reversed(range(count))
                )
            else:
                trace.extend(
                    (
                        stage,
                        ordinal % index_period,
                        ordinal,
                        frontier.POSITION,
                    )
                    for ordinal in range(count)
                )

        stages = frontier.trace_stages(trace)

        self.assertEqual(set(stages), set(range(19)))
        np.testing.assert_array_equal(
            stages[4],
            np.arange(frontier.STAGE_COUNTS[4], dtype="<u2"),
        )
        np.testing.assert_array_equal(
            stages[8],
            np.arange(frontier.STAGE_COUNTS[8], dtype="<u2"),
        )

    def test_runtime_and_accepted_q_modes_are_explicit(self) -> None:
        self.assertIsNot(frontier.exact_runtime_oracle, frontier.accepted_oracle)
        self.assertEqual(frontier.MAX_LAYER, 21)
        self.assertEqual(frontier.POSITION, 2)

    def test_snapshot_records_paths_relative_to_attempt_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "layer04/trace.hex"
            artifact.parent.mkdir()
            artifact.write_text("0000\n", encoding="ascii")

            records = frontier.snapshot(root)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["relative_path"], "layer04/trace.hex")
            self.assertEqual(records[0]["path"], str(artifact.resolve()))

    def test_producer_trace_binding_authenticates_exact_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            position_dir = root / "layer04/position000"
            for name in ("raw", "exact_oracle", "vectors"):
                (position_dir / name).mkdir(parents=True)
            trace = b"0000000000000000\n" * 23324
            for path in (
                position_dir / "raw/trace.hex",
                position_dir / "exact_oracle/trace.hex",
                position_dir / "vectors/trace.hex",
            ):
                path.write_bytes(trace)
            trace_record = {
                "bytes": len(trace),
                "path": str(position_dir / "vectors/trace.hex"),
                "sha256": hashlib.sha256(trace).hexdigest(),
            }
            (position_dir / "vectors/boundary_manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "ace3_decoder_token_runtime_vector_contract",
                        "layer_index": 4,
                        "position": 0,
                        "trace": trace_record,
                        "trace_records": 23324,
                    }
                ),
                encoding="ascii",
            )
            (position_dir / "raw/terminal.txt").write_text(
                "schema=ace3_decoder_token_transaction_v1 layer_index=4 "
                "position=0 natural_terminal=1 exit_code=0 "
                "trace_count=23324 final_count=896 done_count=1\n",
                encoding="ascii",
            )

            with mock.patch.object(frontier, "SOURCE_ATTEMPT", root):
                payload, binding = frontier.producer_authenticated_trace(4, 0)

            self.assertEqual(payload, trace)
            self.assertEqual(
                binding["kind"],
                "original_producer_exact_oracle_trace_binding",
            )
            self.assertTrue(binding["natural_terminal"])


if __name__ == "__main__":
    unittest.main()
