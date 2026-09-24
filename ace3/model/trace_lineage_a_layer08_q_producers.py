#!/usr/bin/env python3
"""Trace retained A-only Q producers without changing or executing RTL."""

from __future__ import annotations

import argparse
import ast
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from diagnose_lineage_a_position3_boundary import (
    BUILD, HISTORY, POLICY, PREFIX, Inputs, accepts, decode_trace, rounded_dot,
    units, write,
)
from fp16_adaptation_oracle import rmsnorm
from qwen2_rope_oracle import rotate_pair

ROOT = BUILD.parent
PRIOR = BUILD / "lineage_a_position3_boundary_e4fa3091ffbd_attempt001"
CHANNELS = (222, 254, 223, 255)
PAIRS = ((222, 254), (223, 255))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def values(bits):
    return np.asarray(bits, dtype="<u2").view("<f2").astype(np.float64)


def bits(value):
    return np.asarray(value, dtype="<f2").view("<u2").reshape(-1).tolist()


def scalar(bit):
    return {"bits": f"{bit:04x}", "value": units(bit) / 2**24}


def rational(value, denominator=1 << 48):
    return str(Fraction(int(value), denominator))


def exact_projection(activation, tensors, channel):
    packed, lane = divmod(channel, 8)
    shift = 4 * (0, 4, 1, 5, 2, 6, 3, 7)[lane]
    weight = []
    for index in range(len(activation)):
        group = index // 128
        q = (int(tensors["qweight"][index, packed]) >> shift) & 15
        z = (int(tensors["qzeros"][group, packed]) >> shift) & 15
        weight.append((q - z) * units(int(tensors["scales"][group, channel])))
    total = sum(units(x) * w for x, w in zip(activation, weight, strict=True))
    total += units(int(tensors["bias"][channel])) << 24
    return rounded_dot(total << 3), total, weight


def run(out):
    started = time.monotonic()
    torch.set_num_threads(1)
    evidence = Inputs()
    prior = evidence.load(PRIOR / "frozen.json", root=True)
    require(prior["token_history"] == HISTORY and prior["comparison_policy"] == POLICY,
            "prior lineage or policy mismatch")
    for path in (PREFIX / "seal.json", PREFIX / "frozen.json",
                 PREFIX / "layer07/seal.json", PREFIX / "layer08/seal.json"):
        evidence.load(path)
    frozen = evidence.load(PREFIX / "frozen.json")
    original = evidence.load(frozen["original_frozen"]["path"])
    p7 = evidence.load(PREFIX / "layer07/prepared.json")
    p8 = evidence.load(PREFIX / "layer08/prepared.json")
    r7 = evidence.load(PREFIX / "layer07/result.json")
    r8 = evidence.load(PREFIX / "layer08/result.json")
    review = evidence.load(prior["accepted_tail_review"]["path"])
    selected = evidence.load(prior["selected_token"]["path"])
    require(review["kind"] == "round_reviewed_handoff"
            and review["producer_role"] == "reviewer"
            and review["review"]["status"] == "done", "tail review identity/status")
    require(selected["actual_token_id"] == selected["expected_token_id"] == 358
            and selected["matches"] is True, "tail selected-token mismatch")
    require(p8["input_hidden"] == r7["output_hidden"], "layer07 hidden parent mismatch")
    require((r7["layer"], r7["position"], r8["layer"], r8["position"]) == (7, 3, 8, 3),
            "transaction identity mismatch")
    require(frozen["comparison_policy"] == original["comparison_policy"] == POLICY,
            "numerical policy mismatch")
    require(p8["binary"] == p8["kv_parent"]["live_binary"]
            and p8["kv_parent"]["valid_positions"] == [0, 1, 2]
            and p8["kv_parent"]["layer_index"] == 8
            and "/model24_persistent_kv_selected_token_corrected_q_attempt005/"
            in p8["kv_parent"]["state"]["path"], "A-only state/binary ABI mismatch")
    evidence.read(p8["binary"]["path"])
    evidence.read(p8["kv_parent"]["state"]["path"])
    hidden_data = evidence.read(p8["vectors"]["input"]["path"])
    require(hidden_data == evidence.read(r7["output_hidden"]["path"]),
            "serialized incoming hidden is not layer07 actual output")
    rows = hidden_data.decode("ascii").splitlines()
    require(len(rows) == 896 and all(len(row) == 10 and int(row[:6], 16) == i
                                   for i, row in enumerate(rows)), "hidden input framing")
    incoming = [int(row[-4:], 16) for row in rows]

    def hex_input(path):
        rows = evidence.read(path).decode("ascii").splitlines()
        require(all(len(row) == 4 for row in rows), "FP16 reference framing")
        return [int(row, 16) for row in rows]

    actual = decode_trace(evidence.read(PREFIX / "layer08/position003/raw/trace.hex"), 3)
    actual7 = decode_trace(evidence.read(PREFIX / "layer07/position003/raw/trace.hex"), 3)
    require(actual7[18] == incoming, "layer07 final trace/hidden mismatch")
    reference = {s: hex_input(p8["independent_stages"][str(s)]["path"])
                 for s in (0, 1, 4, 8)}
    ref7 = {s: hex_input(p7["independent_stages"][str(s)]["path"])
            for s in (12, 17, 18)}
    independent_parent = evidence.load(p8["independent_parent"]["path"])
    require(independent_parent["history"] == HISTORY[:3]
            and independent_parent["layer"] == 8, "independent reference lineage")
    tensor_hashes = {x["name"]: x["sha256"]
                     for x in independent_parent["checkpoint_tensor_hashes"]}
    tensors, norm_weight = {}, None
    for item in p8["vectors"]["tensors"]:
        meta = item["checkpoint_tensor"]
        name = meta["name"]
        if "input_layernorm" not in name and "self_attn.q_proj." not in name:
            continue
        dtype = "<u2" if meta["dtype"] == "F16" else "<u4"
        data = np.asarray([int(x, 16) for x in
                           evidence.read(item["serialized"]["path"]).splitlines()], dtype=dtype)
        require(data.nbytes == meta["bytes"] and hashlib.sha256(data.tobytes()).hexdigest()
                == meta["sha256"] == tensor_hashes[name], f"official tensor mismatch: {name}")
        data = data.reshape(meta["shape"])
        if "input_layernorm" in name:
            norm_weight = data.reshape(-1).tolist()
        else:
            tensors[name.rsplit(".", 1)[1]] = data
    require(norm_weight is not None and set(tensors) == {"bias", "scales", "qweight", "qzeros"},
            "missing Q producer tensor")
    source = BUILD / "independent_fp16_trajectory_20260906_1133/source/official_single_decoder_layer.py"
    nodes = [node for node in ast.parse(evidence.read(source)).body
             if isinstance(node, ast.FunctionDef)
             and node.name in {"_torch_unpack", "_torch_linear", "_torch_rmsnorm"}]
    require(len(nodes) == 3, "independent numerical source incomplete")
    scope = {"np": np, "torch": torch, "GROUP_SIZE": 128, "_require": require,
             "AWQ_REVERSE_ORDER": (0, 4, 1, 5, 2, 6, 3, 7)}
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
                       names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    rope_rows = evidence.read(p8["vectors"]["rope_coefficients"]["path"]).decode("ascii").splitlines()
    require(len(rope_rows) == 128 and all(
        len(row) == 14 and int(row[:4], 16) == i // 32
        and int(row[4:6], 16) == i % 32 for i, row in enumerate(rope_rows)),
        "RoPE coefficient position/pair framing")
    coefficients = {p: tuple(int(rope_rows[96 + p][i:i + 4], 16) for i in (6, 10))
                    for p in range(32)}
    sources = [record(Path(__file__).parent / name) for name in (
        Path(__file__).name, "diagnose_lineage_a_position3_boundary.py",
        "fp16_adaptation_oracle.py", "qwen2_rope_oracle.py", "awq_bit_oracle.py")]
    write(out / "frozen.json", {
        "attempt_id": out.name, "kind": "A_only_retained_Q_producer_diagnostic_v1",
        "token_history": HISTORY, "channels": CHANNELS, "pairs": PAIRS,
        "sources": sources, "command": record(out / "command.sh"),
        "authenticated_inputs": list(evidence.records.values()),
        "public_interfaces": frozen["public_interfaces"], "profile": frozen["profile"],
        "comparison_policy": POLICY, "python": sys.version, "numpy": np.__version__,
        "torch": torch.__version__, "threads": 1, "device": "cpu",
        "method": [
            "Authenticate retained A-only layer07 hidden, layer08 trace, native AWQ tensors.",
            "Reconstruct stage00 from actual hidden with the integer model and independent Torch.",
            "Exact integer Q48 projection plus bias, one RNE; compare with independent Torch.",
            "Telescoping projection/norm-operator/hidden-drift decomposition on four channels.",
            "Separate retained FP16 multiply/add RoPE from single-round mathematical RoPE.",
            "Rank all 896 signed input contributions, not selected reference-Q substitutions.",
        ],
        "regressions": ["original stage08 fails only at 13,15",
                        "reference stage00/01/04 reconstruct without re-anchoring",
                        "projection contributions telescope exactly",
                        "RMSNorm Decimal80 cross-check of independent binary64"],
        "RTL_execution": "NOT_RUN: retained trace diagnosis, not a new RTL attempt",
        "lineage_B_imports": 0, "production_edits": 0, "binary64_admission": False,
    })
    for item in sources:
        data = Path(item["path"]).read_bytes()
        with (out / Path(item["path"]).name).open("xb") as stream:
            stream.write(data)
    independent_norm = {}
    raw_norm = {}
    decimal_max_error = {}
    for label, hidden in (("actual_hidden", incoming), ("reference_hidden", ref7[18])):
        raw = scope["_torch_rmsnorm"](torch.from_numpy(values(hidden)[None, :]),
                                     np.asarray(norm_weight, dtype="<u2").view("<f2"))
        raw_norm[label] = raw.numpy().reshape(-1)
        independent_norm[label] = raw.to(torch.float16).numpy().view("<u2").reshape(-1).tolist()
        with localcontext() as context:
            context.prec = 80
            h = [Decimal(units(x)) / (1 << 24) for x in hidden]
            g = [Decimal(units(x)) / (1 << 24) for x in norm_weight]
            denominator = (sum(x*x for x in h) / len(h) + Decimal("0.000001")).sqrt()
            oracle = [float(x*w / denominator) for x, w in zip(h, g, strict=True)]
        decimal_max_error[label] = float(np.max(np.abs(raw_norm[label] - oracle)))
        require(np.allclose(raw_norm[label], oracle, rtol=2e-15, atol=1e-15),
                "independent norm/Decimal80 discrepancy")
    require(independent_norm["reference_hidden"] == reference[0], "reference norm reconstruction")
    modeled_norm, mean_q48, rms_q24 = rmsnorm(incoming, norm_weight)
    require(not any(invalid or saturated for _, invalid, saturated in modeled_norm),
            "RMSNorm model invalid/saturated")
    norm_model_exact = [x[0] for x in modeled_norm] == actual[0]
    operands = {"actual_stage00": actual[0],
                "independent_norm_actual_hidden": independent_norm["actual_hidden"],
                "independent_norm_reference_hidden": independent_norm["reference_hidden"]}
    projected, totals, weights = {}, {}, {}
    for label, operand in operands.items():
        projected[label], totals[label] = {}, {}
        for channel in CHANNELS:
            output, total, weight = exact_projection(operand, tensors, channel)
            projected[label][channel], totals[label][channel] = output, total
            weights[channel] = weight
    torch_tensors = {"q." + name: data.view("<f2" if data.dtype.itemsize == 2 else "<i4")
                     for name, data in tensors.items()}
    for label, operand in operands.items():
        raw = scope["_torch_linear"](torch.from_numpy(values(operand)[None, :]),
                                     torch_tensors, "q", "q.bias")
        expected = bits(raw.numpy())
        require(all(expected[c] == projected[label][c] for c in CHANNELS),
                f"exact projection/independent Torch disagreement: {label}")
    require(all(projected["independent_norm_reference_hidden"][c] == reference[1][c]
                for c in CHANNELS), "reference projection reconstruction")
    channels, contribution_rows = [], []
    for c in CHANNELS:
        row = {"channel": c, "stage01_actual": scalar(actual[1][c]),
               "stage01_reference": scalar(reference[1][c]),
               "controls": {label: scalar(outputs[c]) for label, outputs in projected.items()},
               "exact_pre_round_projection": {label: rational(v[c]) for label, v in totals.items()}}
        chain = [actual[1][c], *(projected[label][c] for label in operands), reference[1][c]]
        names = ("projection_operator", "norm_operator", "incoming_hidden", "reference_reconstruction")
        row["signed_stage01_decomposition"] = {
            name: rational(units(a) - units(b), 1 << 24)
            for name, a, b in zip(names, chain, chain[1:], strict=False)}
        raw_contributions = []
        for i, w in enumerate(weights[c]):
            norm_delta = (units(actual[0][i]) - units(independent_norm["actual_hidden"][i])) * w
            hidden_delta = (units(independent_norm["actual_hidden"][i]) - units(reference[0][i])) * w
            raw_contributions.append((i, norm_delta, hidden_delta))
            contribution_rows.append({
                "q_channel": c, "input_index": i, "group": i // 128,
                "weight_q24": w, "hidden_actual_bits": f"{incoming[i]:04x}",
                "hidden_reference_bits": f"{ref7[18][i]:04x}",
                "stage00_actual_bits": f"{actual[0][i]:04x}",
                "stage00_same_hidden_independent_bits": f"{independent_norm['actual_hidden'][i]:04x}",
                "stage00_reference_bits": f"{reference[0][i]:04x}",
                "norm_operator_contribution": rational(norm_delta),
                "hidden_drift_contribution": rational(hidden_delta),
            })
        require(sum(n+h for _, n, h in raw_contributions)
                == totals["actual_stage00"][c] - totals["independent_norm_reference_hidden"][c],
                "exact projection decomposition does not telescope")
        row["raw_norm_operator_contribution"] = rational(sum(n for _, n, _ in raw_contributions))
        row["raw_hidden_drift_contribution"] = rational(sum(h for _, _, h in raw_contributions))
        row["largest_hidden_contributors"] = [
            {"input_index": i, "contribution": rational(h),
             "layer07_stage12_actual": scalar(actual7[12][i]),
             "layer07_stage12_reference": scalar(ref7[12][i]),
             "layer07_stage17_actual": scalar(actual7[17][i]),
             "layer07_stage17_reference": scalar(ref7[17][i]),
             "layer07_stage18_actual": scalar(incoming[i]),
             "layer07_stage18_reference": scalar(ref7[18][i])}
            for i, _, h in sorted(raw_contributions, key=lambda x: (-abs(x[2]), x[0]))[:8]]
        channels.append(row)
    pair_rows = []
    for low, high in PAIRS:
        pair = low % 64
        cos, sin = coefficients[pair]
        angle = 3.0 / (1_000_000.0 ** (2 * pair / 64))
        trig = torch.tensor(angle, dtype=torch.float64)
        c, s = float(trig.cos()), float(trig.sin())

        def rotate_math(q, quantized_coefficients=False):
            x, y = (units(q[i]) / 2**24 for i in (low, high))
            co, si = (units(cos) / 2**24, units(sin) / 2**24) if quantized_coefficients else (c, s)
            return bits([x*co-y*si, y*co+x*si])

        reconstructed = rotate_pair(actual[1][low], actual[1][high], cos, sin)
        require(not reconstructed[2] and not reconstructed[3], "RoPE model flags")
        require(list(reconstructed[:2]) == [actual[4][low], actual[4][high]],
                "retained same-input RoPE reconstruction")
        require(rotate_math(reference[1]) == [reference[4][low], reference[4][high]],
                "reference same-input RoPE reconstruction")
        controls = {
            "actual": [actual[4][low], actual[4][high]],
            "same_projection_single_round_fp16_coefficients": rotate_math(actual[1], True),
            "same_projection_mathematical_rope": rotate_math(actual[1]),
            **{label + "_mathematical_rope": rotate_math(q) for label, q in projected.items()},
            "reference": [reference[4][low], reference[4][high]],
        }
        pair_rows.append({"channels": [low, high], "cos": scalar(cos), "sin": scalar(sin),
                          "controls": {label: [scalar(b) for b in v] for label, v in controls.items()}})
    failing = [i for i, (a, r) in enumerate(zip(actual[8], reference[8], strict=True))
               if not accepts(a, r)]
    require(failing == [13, 15], "original failure regression")
    with (out / "projection_contributions.csv").open("x", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contribution_rows[0]))
        writer.writeheader()
        writer.writerows(contribution_rows)
    write(out / "stage00.json", {
        "integer_model_matches_all_actual": norm_model_exact,
        "mean_q48": mean_q48, "rms_q24": rms_q24,
        "independent_decimal80_max_absolute_difference": decimal_max_error,
        "channels": [{"index": i, "hidden_actual": scalar(incoming[i]),
                      "hidden_reference": scalar(ref7[18][i]),
                      "actual": scalar(actual[0][i]),
                      "integer_model": scalar(modeled_norm[i][0]),
                      "same_hidden_independent": scalar(independent_norm["actual_hidden"][i]),
                      "reference": scalar(reference[0][i])} for i in range(896)],
    })
    projection_exact = all(actual[1][c] == projected["actual_stage00"][c] for c in CHANNELS)
    write(out / "result.json", {
        "status": "PRODUCER_DIAGNOSIS_COMPLETE_NOT_REPAIRED",
        "attempt_id": out.name, "token_history": HISTORY, "channels": channels, "pairs": pair_rows,
        "stage01_exact_same_input_on_four_channels": projection_exact,
        "stage00_integer_model_exact_all_channels": norm_model_exact,
        "original_stage08_failure_indices": failing,
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "root_cause_hypothesis": (
            "Use the signed decomposition to distinguish incoming-hidden drift from local "
            "RMSNorm/projection/RoPE rounding; weighted contributors are not unique causal producers."),
        "next_discriminating_boundary": {
            "producer": "layer07 position3 stage18 residual: stage12 plus stage17",
            "consumer": "layer08 position3 stage00 RMSNorm -> Q stage01 -> stage04",
            "coordinates": sorted({x["input_index"] for row in channels
                                   for x in row["largest_hidden_contributors"]}),
            "experiment": (
                "Hold the authenticated layer07 incoming hidden and historical K/V fixed. "
                "At the listed coordinates independently recompute stage17 from actual stage16, "
                "then stage18 from actual stage12/17. Separate local rounding from inherited "
                "stage12/stage16 drift. Do not replace reference or execution inputs."),
            "condition": "Only after reviewing the measured local reconstruction and decomposition.",
        },
        "regression": "Preserve original stage08 failures13/15 and exact reference reconstructions.",
        "repair": None, "RTL_invocations": 0, "lineage_B_state_imports": 0,
        "production_edits": 0, "third_token_selected": False,
        "binary64_evaluated": False, "independent_review": "pending normal Host Reviewer",
        "elapsed_seconds": time.monotonic() - started,
    })
    write(out / "seal.json", {"artifacts": [record(p) for p in sorted(out.iterdir())
                                            if p.is_file() and p.suffix != ".log"]})
    for p in out.iterdir():
        if p.is_file() and p.suffix != ".log":
            p.chmod(0o444)
    print(json.dumps({"status": "PRODUCER_DIAGNOSIS_COMPLETE_NOT_REPAIRED",
                      "projection_exact": projection_exact, "norm_model_exact": norm_model_exact,
                      "original_failures": failing, "elapsed_seconds": time.monotonic() - started}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    out = parser.parse_args().attempt_dir.resolve()
    require(out.parent == BUILD / "model24_selected_token_position3_continuations"
            and out.name.startswith("lineage_a_q_producer_"), "output outside bounded scope")
    require(out.is_dir() and {p.name for p in out.iterdir()} <= {"command.sh", "run.log"},
            "attempt directory not fresh")
    try:
        run(out)
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as error:
        write(out / "failure.json", {
            "status": "DIAGNOSTIC_NO_ACCEPTANCE", "failure_taxonomy": "evidence_or_evaluator_failure",
            "root_cause_hypothesis": str(error),
            "regression": "Correct the rejected binding/calculation in a fresh attempt, preserving this one.",
            "RTL_correctness_conclusion": None, "RTL_invocations": 0,
        })
        raise


if __name__ == "__main__":
    main()
