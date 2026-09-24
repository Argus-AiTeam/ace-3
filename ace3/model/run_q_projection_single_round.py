#!/usr/bin/env python3
"""Bounded, immutable Q-only simulation; expected values never enter the DUT."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import struct
import subprocess
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import numpy as np
import torch

from accepted_awq_projection import project_fp16_stage
from awq_bit_oracle import AWQ_REVERSE_ORDER
from generate_projection_vectors import SOURCES, checked_bytes
from projection_oracle import complete_projection_output

ROOT = Path(__file__).resolve().parents[2]
REPAIR = ROOT / "build/model24_persistent_kv_second_token_layer00_q_repair_attempt004/result.json"
FRONTIER = ROOT / "build/model24_persistent_kv_selected_token_repaired_q_attempt004/result.json"
RTL = [
    "ace3_fp16_fixed.sv",
    "ace3_awq_w4a16_g128_dot_lane.sv",
    "ace3_q47_48_to_f16_rne.sv",
    "ace3_awq_w4a16_projection_engine.sv",
    "ace3_qkv_projection_cluster.sv",
]
TOP = "ace3_q_projection_single_round_tb"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def record(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def authenticate(binding: dict) -> bytes:
    path = Path(binding["path"])
    actual = record(path)
    require(actual["sha256"] == binding["sha256"], f"hash mismatch: {path}")
    require(actual["bytes"] == binding["bytes"], f"size mismatch: {path}")
    return path.read_bytes()


def write_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def run(output: Path, name: str, args: list[str]) -> None:
    with (output / f"{name}.command.txt").open("x") as stream:
        stream.write(shlex.join(args) + "\n")
    with (output / f"{name}.stdout.txt").open("x") as stdout, (
        output / f"{name}.stderr.txt"
    ).open("x") as stderr:
        result = subprocess.run(args, cwd=ROOT, stdout=stdout, stderr=stderr)
    (output / f"{name}.status.txt").write_text(f"{result.returncode}\n")
    require(result.returncode == 0, f"{name} failed; see {output}")


def vectors(output: Path) -> tuple[list[dict], list[dict]]:
    repair = json.loads(REPAIR.read_text())
    bindings = [record(REPAIR), record(FRONTIER)]
    for binding in repair["authenticated_inputs"]["sources"]:
        if Path(binding["path"]).name in (
            "accepted_awq_projection.py", "replay_layer00_q_projection_repair.py"
        ):
            authenticate(binding)
            bindings.append(binding)
    manifest_binding = repair["authenticated_inputs"]["layer00_input"]["manifest"]
    manifest = json.loads(authenticate(manifest_binding))
    bindings.append(manifest_binding)
    trace_binding = repair["authenticated_inputs"]["traces"][2]
    rows = authenticate(trace_binding).decode("ascii").splitlines()
    bindings.append(trace_binding)
    # Only stage00 is a DUT input; no software-repaired Q or hidden output is used.
    stage0 = [(int(row[8:12], 16), int(row[12:16], 16)) for row in rows
              if len(row) == 16 and row[:8] == "00000200"]
    require([index for index, _ in sorted(stage0)] == list(range(896)),
            "layer00 position2 stage00 trace geometry mismatch")
    activations = [bits for _, bits in sorted(stage0)]
    fixture = ROOT / "ace3/fixtures/qwen2.5-0.5b-instruct-awq/layer0-q-proj"
    raw = {name: checked_bytes(name, fixture) for name in SOURCES}
    bindings.extend(record(fixture / filename) for filename, _ in SOURCES.values())
    qw = np.frombuffer(raw["qweight"], dtype="<u4").reshape(896, 112)
    qz = np.frombuffer(raw["qzeros"], dtype="<u4").reshape(7, 112)
    scales = np.frombuffer(raw["scales"], dtype="<u2").reshape(7, 896)
    tensors = [tensor for tensor in manifest["tensors"]
               if tensor["checkpoint_tensor"]["name"] == "model.layers.0.self_attn.q_proj.bias"]
    require(len(tensors) == 1, "missing official Q bias")
    bias_binding = tensors[0]["serialized"]
    bias = [int(row, 16) for row in authenticate(bias_binding).decode("ascii").splitlines()]
    require(len(bias) == 896, "Q bias geometry mismatch")
    require(hashlib.sha256(struct.pack("<896H", *bias)).hexdigest() ==
            "e9612c72520a62dd903796d9535de2081e1cd15a724b2972513599ff276b9e72",
            "official Q bias semantic hash mismatch")
    bindings.append(bias_binding)
    independent = project_fp16_stage(
        activations, qw.reshape(-1), qz.reshape(-1), scales.reshape(-1), 896, bias)
    candidates = []
    for channel in range(896):
        packed = channel // 8
        args = (activations, qw[:, packed].tolist(), qz[:, packed].tolist(),
                scales[:, channel].tolist(), channel % 8, bias[channel])
        legacy = complete_projection_output(*args)
        accepted = complete_projection_output(*args, single_round_bias=True)
        require(accepted[1] == independent[channel],
                f"independent binary64 oracle disagreement channel={channel}")
        candidates.append(dict(name=f"official-position2-channel{channel}", channel=channel,
                               args=args, legacy=legacy[:4], accepted=accepted[:4]))
    f16 = lambda bits: float(np.asarray([bits], dtype="<u2").view("<f2")[0])
    ranked = sorted(range(896), key=lambda i: abs(
        f16(candidates[i]["legacy"][1]) - f16(candidates[i]["accepted"][1])), reverse=True)
    require(any(c["legacy"][1] != c["accepted"][1] for c in candidates),
            "official pre-round bias discrepancy not reproduced")
    cases = [candidates[i] for i in sorted(set(range(8)) | set(ranked[:8]))]
    # Sparse triples are (input index, activation bits, signed native INT4 delta).
    directed = [
        ("bias-before-round", 0x0001, [(0, 0x8001, 1)], {0: 0x3800}),
        ("normal-even-tie", 0x1000, [(0, 0x3C00, 1)], {}),
        ("normal-odd-tie", 0x1000, [(0, 0x3C01, 1)], {}),
        ("cross-group-cancellation", 0x0001,
         [(0, 0x7BFF, 7), (128, 0x7BFF, -7), (256, 0x8001, 1)], {2: 0x3800}),
        ("negative-underflow", 0, [(0, 0x8001, 1)], {0: 0x3800}),
        ("subnormal-normal-tie", 0x0001, [(0, 0x03FF, 1), (128, 0x8001, 1)],
         {1: 0x3800}),
        ("bias-rescues-intermediate-overflow", 0xFBFF, [(0, 0x7BFF, 2)], {}),
        ("positive-overflow", 0x7BFF, [(0, 0x7BFF, 1)], {}),
        ("negative-overflow", 0xFBFF, [(0, 0xFBFF, 1)], {}),
        ("exact-zero", 0x8000, [], {}),
        ("nonfinite-bias", 0x7E00, [], {}),
        ("nonfinite-activation", 0, [(0, 0x7C00, 1)], {}),
        ("nonfinite-scale", 0, [(0, 0x3C00, 1)], {0: 0x7C00}),
    ]
    for number, (name, b, active, group_scales) in enumerate(directed):
        lane = number % 8
        shift = 4 * AWQ_REVERSE_ORDER[lane]
        a, w, z, s = [0] * 896, [8 << shift] * 896, [8 << shift] * 7, [0x3C00] * 7
        for group, scale in group_scales.items():
            s[group] = scale
        for index, activation, delta in active:
            a[index], w[index] = activation, (8 + delta) << shift
        args = (a, w, z, s, lane, b)
        legacy = complete_projection_output(*args)
        accepted = complete_projection_output(*args, single_round_bias=True)
        if not accepted[2]:
            with np.errstate(over="ignore", invalid="ignore"):
                independent_case = project_fp16_stage(
                    a, w, z, np.repeat(np.asarray(s, dtype="<u2"), 8),
                    8, [b] * 8)
            require(int(independent_case[lane]) == accepted[1], f"directed oracle: {name}")
        cases.append(dict(name=name, channel=lane, args=args,
                          legacy=legacy[:4], accepted=accepted[:4]))
    headers, metadata, pairs = [], [], []
    for case in cases:
        a, w, z, s, _, b = case["args"]
        headers.append(f"{case['channel']:04x}{b:04x}")
        metadata.extend(f"{scale:04x}{zero:08x}" for scale, zero in zip(s, z, strict=True))
        pairs.extend(f"{activation:04x}{weight:08x}" for activation, weight in zip(a, w, strict=True))
    for name, lines in (("cases.hex", headers), ("meta.hex", metadata), ("pairs.hex", pairs)):
        (output / name).write_text("\n".join(lines) + "\n")
    (output / "expected.hex").write_text("".join(f"{c['accepted'][1]:04x}\n" for c in cases))
    return cases, bindings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    source_paths = [ROOT / "ace3/rtl" / name for name in RTL] + [
        ROOT / "ace3/tb" / f"{TOP}.sv", Path(__file__).resolve(),
        ROOT / "ace3/model/projection_oracle.py", ROOT / "ace3/model/accepted_awq_projection.py",
        ROOT / "ace3/model/awq_bit_oracle.py", ROOT / "ace3/model/fp16_adaptation_oracle.py",
        ROOT / "ace3/model/generate_projection_vectors.py",
    ]
    for path in source_paths:
        shutil.copyfile(path, source / path.name)
    cases, bindings = vectors(output)
    summaries = [{k: v for k, v in case.items() if k != "args"} for case in cases]
    write_json(output / "frozen.json", dict(
        scope="bounded Q only; no continuation seed or full-model acceptance",
        baseline=args.baseline, policy="exact products and cross-group sum plus bias; one FP16 RNE",
        source=[record(path) for path in source.iterdir()], bindings=bindings,
        vectors=[record(output / name) for name in ("cases.hex", "meta.hex", "pairs.hex", "expected.hex")],
        cases=summaries, numpy=np.__version__, torch=torch.__version__,
        public_contract="Copied exact module/port/parameter declarations in source/*.sv; no aliases",
        comparison="exact bits/flags/pre-bias accumulator; binary64 independent oracle for finite operands"))
    run(output, "iverilog-version", ["iverilog", "-V"])
    run(output, "vvp-version", ["vvp", "-V"])
    comparisons = {}
    for selected, label in ((1, "q"), (0, "legacy")):
        binary = output / f"{label}.vvp"
        run(output, f"{label}-compile", [
            "iverilog", "-g2012", "-Wall", "-s", TOP, f"-P{TOP}.CASES={len(cases)}",
            f"-P{TOP}.SELECT_Q={selected}", "-o", str(binary),
            *[str(source / name) for name in RTL], str(source / f"{TOP}.sv")])
        run(output, f"{label}-simulate", [
            "vvp", str(binary), f"+VECTOR_DIR={output}", f"+OUTPUT={output / (label + '.actual.hex')}",
            f"+WAVE={output / (label + '.vcd')}"])
        actual = [line.split() for line in (output / f"{label}.actual.hex").read_text().splitlines()]
        require(len(actual) == len(cases), f"{label}: missing outputs")
        mismatches = []
        for i, (row, case) in enumerate(zip(actual, cases, strict=True)):
            expected = case["legacy" if args.baseline or not selected else "accepted"]
            acc, value, invalid, overflow = expected
            require([int(x, 16) for x in row] ==
                    [case["channel"], value, acc & ((1 << 102) - 1), int(invalid), int(overflow)],
                    f"{label} numeric mismatch: {case['name']}: {row} expected={expected}")
            if int(row[1], 16) != case["accepted"][1]:
                mismatches.append(case["name"])
        comparisons[label] = dict(outputs=len(actual), accepted_bit_mismatches=mismatches,
                                  binary=record(binary), actual=record(output / f"{label}.actual.hex"))
    require(bool(comparisons["legacy"]["accepted_bit_mismatches"]), "lost legacy discrepancy")
    write_json(output / "result.json", dict(
        status="BASELINE_DISCREPANCY_REPRODUCED" if args.baseline else "BOUNDED_RTL_Q_PASS",
        comparisons=comparisons, independent_oracle="binary64 torch dot+bias vs integer exact accumulator",
        failure_taxonomy="numerical_boundary_double_rounding",
        root_cause="legacy rounds the dot before bias, losing terms/cancellation at the FP16 boundary",
        regression="official-position2 Q channels and directed bias-before-round/cancellation/tie cases",
        reviewer="pending Host independent Reviewer; no full-model acceptance"))
    print(f"{'BASELINE' if args.baseline else 'BOUNDED_RTL_Q'}_PASS cases={len(cases)} output={output}")


if __name__ == "__main__":
    main()
