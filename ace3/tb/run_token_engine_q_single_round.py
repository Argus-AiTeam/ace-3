#!/usr/bin/env python3
"""One immutable, input-boundary replay through the real token controller."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ace3/model"))

import numpy as np
import torch

from run_q_projection_single_round import (
    AWQ_REVERSE_ORDER, RTL as LEGACY_RTL, authenticate,
    complete_projection_output, project_fp16_stage, record, require, run,
    vectors, write_json,
)

TOP = "ace3_token_engine_q_single_round_tb"
RTL = [
    "ace3_fp16_fixed.sv", "ace3_awq_w4a16_g128_dot_lane.sv",
    "ace3_q47_48_to_f16_rne.sv", "ace3_awq_w4a16_projection_engine.sv",
    "ace3_fp16_rmsnorm_core.sv", "ace3_fp16_residual_add_core.sv",
    "ace3_fp16_silu_gate_core.sv", "ace3_qwen2_rope_pair.sv",
    "ace3_fp16_kv_cache.sv", "ace3_attention_score_core.sv",
    "ace3_attention_softmax_core.sv", "ace3_attention_value_core.sv",
    "ace3_decoder_qzeros_address.sv", "ace3_decoder_layer0_token_engine.sv",
]


def token_vectors(output: Path, cases: list[dict]) -> tuple[list[int], list[dict]]:
    official = {c["channel"]: c for c in cases if c["name"].startswith("official-")}
    activation = next(iter(official.values()))["args"][0]
    jobs, rows = [], []
    for kind, count in ((0, 896), (1, 128), (2, 128)):
        jobs.append((kind << 13) | count)
        for channel in range(count):
            case = official.get(channel)
            args = (activation, [0]*896, [0]*7, [0x3C00]*7, channel % 8, 0)
            if case is not None:
                args = case["args"]
            expected = complete_projection_output(*args, single_round_bias=kind == 0)[:4]
            rows.append(dict(kind=kind, channel=channel, args=args, expected=expected,
                             name=case["name"] if case else "zero-delta-filler"))
    for case in cases:
        if case["name"].startswith("official-"):
            continue
        a, w, z, s, lane, b = case["args"]
        shift = 4 * AWQ_REVERSE_ORDER[lane]
        # Directed operands are relocated to channel zero; official words are not changed.
        args = (a, [(v >> shift) & 15 for v in w],
                [(v >> shift) & 15 for v in z], s, 0, b)
        for kind in (0, 1, 2):
            expected = complete_projection_output(*args, single_round_bias=kind == 0)[:4]
            require(expected == tuple(case["accepted" if kind == 0 else "legacy"]),
                    f"directed lane relocation changed arithmetic: {case['name']}")
            if kind == 0 and not expected[2]:
                with np.errstate(over="ignore", invalid="ignore"):
                    independent = project_fp16_stage(
                        a, args[1], args[2], np.repeat(np.asarray(s, dtype="<u2"), 8),
                        8, [b]*8)
                require(int(independent[0]) == expected[1], "independent directed oracle")
            jobs.append((kind << 13) | 1)
            rows.append(dict(kind=kind, channel=0, args=args, expected=expected,
                             name=case["name"]))
    with (output / "jobs.hex").open("x") as stream:
        stream.write("".join(f"{job:04x}\n" for job in jobs))
    with (output / "biases.hex").open("x") as bias, (
        output / "token_meta.hex").open("x") as meta, (
        output / "token_pairs.hex").open("x") as pairs:
        for row in rows:
            a, w, z, s, _, b = row["args"]
            bias.write(f"{b:04x}\n")
            meta.writelines(f"{scale:04x}{zero:08x}\n" for scale, zero in zip(s, z, strict=True))
            pairs.writelines(f"{act:04x}{weight:08x}\n" for act, weight in zip(a, w, strict=True))
    return jobs, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.parent == ROOT / "build" and
            output.name.startswith("token_engine_q_single_round_"), "attempt path outside scope")
    output.mkdir(exist_ok=False)
    source = output / "source"
    source.mkdir()
    source_paths = [ROOT / "ace3/rtl" / name for name in dict.fromkeys(RTL + LEGACY_RTL)]
    source_paths += [
        ROOT / "ace3/tb" / f"{TOP}.sv",
        ROOT / "ace3/tb/ace3_q_projection_single_round_tb.sv",
        Path(__file__).resolve(), ROOT / "ace3/contracts/decoder_layer0_token_engine.json",
    ]
    source_paths += [ROOT / "ace3/model" / name for name in (
        "run_q_projection_single_round.py", "accepted_awq_projection.py",
        "projection_oracle.py", "awq_bit_oracle.py", "fp16_adaptation_oracle.py",
        "generate_projection_vectors.py",
    )]
    original_sources = [record(path) for path in source_paths]
    for path in source_paths:
        shutil.copyfile(path, source / path.name)
    cases, bindings = vectors(output)
    jobs, rows = token_vectors(output, cases)
    for binding in original_sources + bindings:
        authenticate(binding)
    vector_bindings = [record(path) for path in sorted(output.glob("*.hex"))]
    write_json(output / "frozen.json", dict(
        scope="Real token-engine projection boundary only; no RMSNorm/full-token/full-model claim",
        input_boundary="Only norm1_mem, psel_q, token_position_q and S_P_START are seeded. "
                       "No child state, accumulator, result, expected value or output is injected.",
        official="Authenticated layer0 position2 Q operands, unmodified packed words at original "
                 "channels; unsampled channels use finite zero-delta fillers. K/V replay Q "
                 "operands to check legacy arithmetic, not official K/V tensor accuracy.",
        public_contract=(source / "ace3_decoder_layer0_token_engine.sv").read_text().split(");", 1)[0] + ");",
        score_policy="Zero exact bit/invalid/overflow/pre-bias-accumulator mismatches; "
                     "all protocol assertions must complete naturally",
        oracle="Independent binary64 Torch accepted-policy cross-check and exact Python integer "
               "dot/flags oracle; expected values are not read by either DUT",
        sources=original_sources, snapshot=[record(path) for path in sorted(source.iterdir())],
        inputs=bindings, vectors=vector_bindings, jobs=jobs,
        rows=[{k: v for k, v in row.items() if k != "args"} for row in rows],
        numpy=np.__version__, torch=torch.__version__, python=sys.version,
        environment={key: os.environ.get(key) for key in
                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    ))
    for tool in ("iverilog", "vvp"):
        run(output, f"{tool}-version", [tool, "-V"])
    comparisons = {}
    for layer in (0, 21):
        label = f"token-layer{layer}"
        binary = output / f"{label}.vvp"
        run(output, label + "-compile", [
            "iverilog", "-g2012", "-Wall", "-s", TOP,
            f"-P{TOP}.JOBS={len(jobs)}", f"-P{TOP}.OUTPUTS={len(rows)}",
            f"-P{TOP}.LAYER_INDEX={layer}", "-o", str(binary),
            *[str(source / name) for name in RTL], str(source / f"{TOP}.sv"),
        ])
        write_json(output / f"{label}.binary.json", record(binary))
        for binding in vector_bindings:
            authenticate(binding)
        run(output, label + "-simulate", [
            "vvp", str(binary), f"+VECTOR_DIR={output}",
            f"+OUTPUT={output / (label + '.actual.hex')}",
            f"+WAVE={output / (label + '.vcd')}",
        ])
        actual = [line.split() for line in (output / f"{label}.actual.hex").read_text().splitlines()]
        require(len(actual) == len(rows), f"{label}: missing outputs")
        mismatches = []
        for index, (actual_row, row) in enumerate(zip(actual, rows, strict=True)):
            acc, value, invalid, overflow = row["expected"]
            expected = [index, row["kind"], row["channel"], value,
                        acc & ((1 << 102)-1), int(invalid), int(overflow)]
            if [int(value, 16) for value in actual_row] != expected:
                mismatches.append(dict(index=index, name=row["name"],
                                       actual=actual_row, expected=expected))
        write_json(output / f"{label}.comparison.json", dict(
            outputs=len(rows), mismatches=mismatches,
            q_outputs=sum(row["kind"] == 0 for row in rows),
            official_q_outputs=sum(row["kind"] == 0 and row["name"].startswith("official-")
                                   for row in rows)))
        require(not mismatches, f"{label}: numerical mismatch")
        comparisons[label] = dict(outputs=len(rows), mismatches=0, binary=record(binary),
                                  actual=record(output / f"{label}.actual.hex"))
    legacy_top = "ace3_q_projection_single_round_tb"
    binary = output / "legacy.vvp"
    run(output, "legacy-compile", [
        "iverilog", "-g2012", "-Wall", "-s", legacy_top,
        f"-P{legacy_top}.CASES={len(cases)}", f"-P{legacy_top}.SELECT_Q=0",
        "-o", str(binary), *[str(source / name) for name in LEGACY_RTL],
        str(source / f"{legacy_top}.sv"),
    ])
    write_json(output / "legacy.binary.json", record(binary))
    run(output, "legacy-simulate", [
        "vvp", str(binary), f"+VECTOR_DIR={output}",
        f"+OUTPUT={output / 'legacy.actual.hex'}", f"+WAVE={output / 'legacy.vcd'}",
    ])
    actual = [line.split() for line in (output / "legacy.actual.hex").read_text().splitlines()]
    require(len(actual) == len(cases), "legacy output count")
    for line, case in zip(actual, cases, strict=True):
        acc, value, invalid, overflow = case["legacy"]
        require([int(v, 16) for v in line] ==
                [case["channel"], value, acc & ((1 << 102)-1), int(invalid), int(overflow)],
                f"legacy mismatch: {case['name']}")
    for binding in original_sources + vector_bindings:
        authenticate(binding)
    write_json(output / "result.json", dict(
        status="TARGETED_TOKEN_ENGINE_Q_PASS", comparisons=comparisons,
        legacy=dict(outputs=len(cases), mismatches=0, binary=record(binary),
                    actual=record(output / "legacy.actual.hex")),
        reviewer="Pending independent Host L2 Reviewer; not an L2 acceptance",
        boundary="Projection-input seeded token-engine RTL simulation only; "
                 "O/FFN/down policy parameters checked, not revalidated numerically; "
                 "no full-model, synthesis, PPA, FPGA or hardware claims",
    ))
    print(f"TARGETED_TOKEN_ENGINE_Q_PASS outputs={len(rows)} per-layer output={output}")


if __name__ == "__main__":
    main()
