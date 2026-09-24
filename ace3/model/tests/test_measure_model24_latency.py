#!/usr/bin/env python3
"""Focused tests for Model24 diagnostic latency evidence."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

from measure_model24_latency import (  # noqa: E402
    CLAIM_BOUNDARY,
    MeasurementError,
    parse_controller_cycles,
    parse_layer_cycles,
    validate_measurement_document,
)


def layer_pass(layer: int, cycles: int) -> str:
    phases = ",".join(str(value) for value in range(1, 38))
    return (
        f"DECODER_LAYER_TOKEN_ENGINE_VERILATOR_PASS layer={layer} "
        f"trace_count=46676 final_count=1792 cycles={cycles} stalls=17 "
        "token0_cycles=400 token1_cycles=500 phase_p_run=100 phase_final=10 "
        f"phase_cycles={phases} reset=pass clear=pass slot_isolation=pass\n"
    )


def measurement_document() -> dict:
    layers = [
        parse_layer_cycles(layer_pass(layer, 1000 + layer), layer)
        for layer in range(24)
    ]
    return {
        "schema_version": 1,
        "kind": "ace3_model24_latency_diagnostics",
        "measurements": {
            "software_dialogue_continuation": {
                "wall_seconds_diagnostic": 1.25,
            },
            "rtl_controller_cascade": {
                "wall_seconds_diagnostic": 3.75,
                "controller_wall_seconds_diagnostic": 0.25,
                "decoder_wall_seconds_diagnostic": 3.5,
                "controller_scheduler_cycles": 123,
                "aggregate_decoder_harness_cycles": sum(
                    record["cycles"] for record in layers
                ),
                "aggregate_token0_cycles": sum(
                    record["token0_cycles"] for record in layers
                ),
                "aggregate_token1_cycles": sum(
                    record["token1_cycles"] for record in layers
                ),
                "aggregate_stall_cycles": sum(
                    record["stall_cycles"] for record in layers
                ),
                "layers": layers,
            },
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


class MeasureModel24LatencyTests(unittest.TestCase):
    def test_cycle_logs_are_unambiguous_and_non_vacuous(self) -> None:
        controller = (
            "MODEL24_LAYER_CONTROLLER_VERILATOR_PASS layers=24 checkpoints=24 "
            "retained_backpressure=pass terminal=layer23 fail_closed=pass "
            "clear_recovery=pass cycles=321 numerical_rtl=not_claimed\n"
        )
        self.assertEqual(parse_controller_cycles(controller), 321)
        record = parse_layer_cycles(layer_pass(7, 2048), 7)
        self.assertEqual(record["cycles"], 2048)
        self.assertEqual(len(record["phase_cycles"]), 37)
        with self.assertRaisesRegex(MeasurementError, "ambiguous controller PASS"):
            parse_controller_cycles(controller + controller)
        with self.assertRaisesRegex(MeasurementError, "index mismatch"):
            parse_layer_cycles(layer_pass(7, 2048), 8)

    def test_measurement_aggregates_and_claim_boundary_are_enforced(self) -> None:
        document = measurement_document()
        validate_measurement_document(document)

        broken_sum = copy.deepcopy(document)
        broken_sum["measurements"]["rtl_controller_cascade"][
            "aggregate_decoder_harness_cycles"
        ] += 1
        with self.assertRaisesRegex(MeasurementError, "aggregate mismatch"):
            validate_measurement_document(broken_sum)

        hardware_claims = copy.deepcopy(document)
        hardware_claims["claim_boundary"]["hardware_latency"] = "measured"
        with self.assertRaisesRegex(MeasurementError, "claim boundary"):
            validate_measurement_document(hardware_claims)


if __name__ == "__main__":
    unittest.main()
