#!/usr/bin/env python3
"""Focused tests for the read-only Model24 runtime envelope."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

from model24_runtime_envelope import (  # noqa: E402
    RuntimeEnvelopeError,
    build_runtime_envelope,
    validate_runtime_envelope,
)


def layer_pass(layer: int) -> str:
    phases = ",".join(["0", *("1" for _ in range(36))])
    return (
        f"DECODER_LAYER_TOKEN_ENGINE_VERILATOR_PASS layer={layer} "
        "trace_count=46676 final_count=1792 cycles=30789799 stalls=17 "
        "token0_cycles=15387575 token1_cycles=15391256 "
        "phase_p_run=30738645 phase_final=2 "
        f"phase_cycles={phases} reset=pass clear=pass slot_isolation=pass\n"
    )


def write_timed(path: Path, payload: str, timestamp: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="ascii")
    os.utime(path, (timestamp, timestamp))


class Model24RuntimeEnvelopeTests(unittest.TestCase):
    def test_partial_completed_layers_produce_conservative_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_timed(root / "initial_embeddings.hex", "0000000000\n", 1000.0)
            ends = (1061.0, 1131.0, 1211.0)
            for layer, end in enumerate(ends):
                start = 1000.0 if layer == 0 else ends[layer - 1]
                layer_dir = root / "layers" / f"layer{layer:02d}"
                write_timed(layer_dir / "compile.log", "compile\n", start + 10.0)
                write_timed(
                    layer_dir / "simulation.log",
                    layer_pass(layer),
                    end - 1.0,
                )
                write_timed(
                    layer_dir / "record.json",
                    json.dumps(
                        {
                            "layer_index": layer,
                            "comparison": {
                                "within_tolerance": layer != 1,
                            },
                        }
                    ),
                    end,
                )
            document = build_runtime_envelope(
                root,
                snapshot={"load_average": [72.0, 200.0, 400.0]},
                captured_at_utc="2026-08-28T13:44:29Z",
            )
            validate_runtime_envelope(document)
            summary = document["summary"]
            self.assertEqual(summary["completed_layers"], 3)
            self.assertEqual(summary["median_layer_seconds"], 75.0)
            self.assertEqual(summary["maximum_layer_seconds"], 80.0)
            self.assertGreater(summary["aggregate_simulation_share"], 0.8)
            self.assertFalse(
                document["completed_layer_timings"][1][
                    "legacy_absolute_comparison_within_tolerance"
                ]
            )
            self.assertEqual(
                document["claim_boundary"]["full_24_layer_run"],
                "not launched by this capture",
            )

    def test_fewer_than_three_layers_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_timed(root / "initial_embeddings.hex", "0000000000\n", 1000.0)
            with self.assertRaisesRegex(
                RuntimeEnvelopeError,
                "at least three contiguous",
            ):
                build_runtime_envelope(root, snapshot={})


if __name__ == "__main__":
    unittest.main()
