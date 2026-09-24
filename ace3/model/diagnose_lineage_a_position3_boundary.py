#!/usr/bin/env python3
"""Re-measure lineage A's retained boundary; never launch or alter RTL."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
from pathlib import Path
import sys
from fractions import Fraction

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
PREFIX = BUILD / "token358_position3_full_continuation_attempt003"
LAYER0 = BUILD / "token358_position3_layer0_attempt001"
LAYER1 = BUILD / "token358_position3_layer1_attempt002/evidence"
LAYER2 = BUILD / "token358_position3_full_continuation_attempt002"
TAIL = BUILD / "position2_tail_runtime_attempt001"
HISTORY = [9707, 1879, 0, 358]
POLICY = {
    "absolute_tolerance": 0.125,
    "accept": "finite AND (abs_error<=0.125 OR (relative_error<0.001 AND orderedFP16_ULP<=1))",
    "max_ulp_distance": 1,
    "relative_denominator": "max(abs(reference),2^-14)",
    "relative_tolerance_strict": 0.001,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def units(bits):
    """Represent finite binary16 exactly in units of 2**-24."""
    require(isinstance(bits, int) and 0 <= bits <= 0xffff, "invalid FP16 encoding")
    exponent, fraction = (bits >> 10) & 31, bits & 1023
    require(exponent != 31, "nonfinite FP16 operand")
    magnitude = fraction if exponent == 0 else (1024 + fraction) << (exponent - 1)
    return -magnitude if bits & 0x8000 else magnitude


def ordered(bits):
    return 0x8000 - (bits & 0x7fff) if bits & 0x8000 else 0x8000 + bits


def accepts(actual, reference):
    a, r = units(actual), units(reference)
    error = abs(a - r)
    return error <= (1 << 21) or (
        1000 * error < max(abs(r), 1024)
        and abs(ordered(actual) - ordered(reference)) <= 1
    )


POSITIVE = [units(bits) for bits in range(0x7c00)]
SCORE_GRID = [value << 27 for value in POSITIVE]


def rounded_dot(total):
    """RNE of sum(Q24*Q24)/8, without floating-point intermediate rounding."""
    magnitude = abs(total)
    require(magnitude <= SCORE_GRID[-1], "score outside finite FP16 range")
    right = bisect.bisect_left(SCORE_GRID, magnitude)
    candidates = {right, max(0, right - 1)}
    bits = min(candidates, key=lambda b: (abs(SCORE_GRID[b] - magnitude), b & 1))
    return bits | (0x8000 if total < 0 else 0)


def exact(value, denominator):
    fraction = Fraction(value, denominator)
    return {"rational": str(fraction), "decimal": float(fraction)}


def score_measurement(total, reference):
    bits = rounded_dot(total)
    return {
        "bits": f"{bits:04x}", "value": units(bits) / 2**24,
        "exact_score": exact(total, 1 << 51),
        "within_unchanged_gate": accepts(bits, reference),
    }


def rounding_margin(total, actual, reference):
    require(not accepts(actual, reference), "margin requires a failing scalar")
    candidates = (
        bits for bits in range(0x10000)
        if (bits & 0x7c00) != 0x7c00 and accepts(bits, reference)
    )
    target = min(candidates, key=lambda b: (
        abs((units(b) << 27) - total), b & 1, b
    ))
    direction = 1 if units(target) > units(actual) else -1
    neighbor_order = ordered(target) - direction
    neighbor = (neighbor_order - 0x8000 if neighbor_order >= 0x8000
                else 0x8000 | (0x8000 - neighbor_order))
    boundary = (units(target) + units(neighbor)) << 26
    return {
        "nearest_passing_bits": f"{target:04x}",
        "rounding_boundary": exact(boundary, 1 << 51),
        "required_signed_dot_change": exact(boundary - total, 1 << 51),
        "boundary_inclusive": not bool(target & 1),
        "meaning": "Necessary score displacement, not an authorized arithmetic correction.",
    }


def write(path, value):
    with path.open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)


class Inputs:
    def __init__(self):
        self.bindings = {}
        self.data = {}
        self.records = {}

    def register(self, value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= value.keys():
                path = str(Path(value["path"]).resolve())
                if path.startswith(str(BUILD) + "/") or "/handoffs/" in path:
                    previous = self.bindings.get(path)
                    require(previous is None or previous["sha256"] == value["sha256"],
                            f"conflicting historical binding: {path}")
                    self.bindings[path] = value
            for child in value.values():
                self.register(child)
        elif isinstance(value, list):
            for child in value:
                self.register(child)

    def read(self, path, *, root=False):
        path = str(Path(path).resolve())
        if path not in self.data:
            data = Path(path).read_bytes()
            record = {"path": path, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()}
            if not root:
                expected = self.bindings[path]
                require(record["sha256"] == expected["sha256"]
                        and record["bytes"] == expected.get("bytes", record["bytes"]),
                        f"input binding mismatch: {path}")
            self.records[path], self.data[path] = record, data
        elif not root:
            require(self.records[path]["sha256"] == self.bindings[path]["sha256"],
                    f"root binding mismatch: {path}")
        return self.data[path]

    def load(self, path, *, root=False):
        value = json.loads(self.read(path, root=root))
        self.register(value)
        return value


def decode_trace(data, position):
    stages = {}
    for line in data.decode("ascii").splitlines():
        require(len(line) == 16 and int(line[:2], 16) == 0
                and int(line[2:6], 16) == position, "trace framing/position mismatch")
        stage, index, bits = int(line[6:8], 16), int(line[8:12], 16), int(line[12:], 16)
        stages.setdefault(stage, []).append((index, bits))
    decoded = {}
    for stage, rows in stages.items():
        if stage in (8, 9):
            require([i for i, _ in rows] == list(range(position + 1)) * 14,
                    "attention head/key ordering mismatch")
            decoded[stage] = [b for _, b in rows]
        else:
            require(sorted(i for i, _ in rows) == list(range(len(rows))),
                    "duplicate/missing trace index")
            decoded[stage] = [b for _, b in sorted(rows)]
    return decoded


def run(out):
    inputs = Inputs()
    roots = [TAIL, LAYER0, LAYER1, LAYER2, PREFIX,
             BUILD / "token358_position3_full_continuation_attempt001"]
    for root in roots:
        inputs.load(root / "seal.json", root=True)
    zero = inputs.load(LAYER0 / "frozen.json")
    package = inputs.load(zero["package"]["path"])
    selected = inputs.load(package["parents"]["selected_token"]["path"])
    tail_result = inputs.load(package["parents"]["tail_result"]["path"])
    review = inputs.load(package["parents"]["tail_review"]["path"])
    inputs.read(package["parents"]["tail_seal"]["path"])
    inputs.read(selected["parent_layer23"]["path"])
    inputs.read(selected["tail_input"]["path"])
    require(review["kind"] == "round_reviewed_handoff"
            and review["producer_role"] == "reviewer"
            and review["review"]["status"] == "done", "tail review not accepted")
    require(package["token_history"] == HISTORY and package["position"] == 3
            and selected["actual_token_id"] == selected["expected_token_id"] == 358
            and selected["matches"] is True
            and tail_result["selected_token_id"] == 358, "accepted seed mismatch")
    require(package["parents"]["selected_token"]["path"] == str(TAIL / "selected_token.json"),
            "foreign tail lineage")
    embedding = inputs.read(zero["embedding"]["file"]["path"])
    prepared0 = inputs.load(LAYER0 / "prepared.json")
    require(inputs.read(prepared0["vectors"]["input"]["path"]) == embedding,
            "actual layer0 input is not the accepted token embedding")
    one = inputs.load(LAYER1 / "frozen.json")
    frozen = inputs.load(PREFIX / "frozen.json")
    original = inputs.load(frozen["original_frozen"]["path"])
    require(frozen["comparison_policy"] == POLICY, "FP16 interstage policy changed")
    require(original["comparison_policy"] == POLICY, "original comparison policy changed")
    results = [inputs.load(path) for path in (
        LAYER0 / "result.json", LAYER1 / "result.json", LAYER2 / "result.json",
        *(PREFIX / f"layer{layer:02d}/result.json" for layer in range(3, 9)),
    )]
    prepared = {layer: inputs.load(PREFIX / f"layer{layer:02d}/prepared.json")
                for layer in range(3, 9)}
    transactions = []
    for layer, result in enumerate(results):
        require((result["layer"], result["position"]) == (layer, 3),
                "nonsequential layer/position")
        if layer:
            parent = (one["input_hidden"] if layer == 1 else
                      result["input_hidden"] if layer == 2 else prepared[layer]["input_hidden"])
            require(parent["sha256"] == results[layer - 1]["output_hidden"]["sha256"]
                    and parent["path"] == results[layer - 1]["output_hidden"]["path"],
                    "actual hidden dependency break")
            inputs.read(parent["path"])
        kv = result["kv_parent"]
        require(kv["layer_index"] == layer and kv["valid_positions"] == [0, 1, 2],
                "historical KV identity mismatch")
        require("/model24_persistent_kv_selected_token_corrected_q_attempt005/"
                in kv["state"]["path"], "foreign historical KV lineage")
        inputs.read(kv["state"]["path"])
        rows = result.get("independent_comparisons", result.get("comparisons"))
        require([row["stage"] for row in rows] == list(range(19)),
                "missing or reordered stage gates")
        trace_path = Path(result["output_hidden"]["path"]).with_name("trace.hex")
        trace = inputs.read(trace_path)
        expected = [inputs.read(row["independent_reference"]["path"]) for row in rows]
        transactions.append((trace, expected))
    p8 = prepared[8]
    require(p8["binary"] == p8["kv_parent"]["live_binary"],
            "failing-layer serialized state/binary ABI mismatch")
    inputs.read(p8["binary"]["path"])
    independent_parent = inputs.load(p8["independent_parent"]["path"])
    require(independent_parent["history"] == HISTORY[:3]
            and independent_parent["layer"] == 8, "independent history mismatch")
    historical = [inputs.read(item["path"]) for item in p8["kv_parent"]["actual_kv_traces"]]
    independent_k_path = PREFIX / "layer08/independent/position003_own_k.npy"
    independent_k_data = inputs.read(independent_k_path)
    independent_parent_k = inputs.read(independent_parent["own_cache"]["k"]["path"])
    source = Path(__file__).read_bytes()
    with (out / "diagnostic.py").open("xb") as stream:
        stream.write(source)
    (out / "diagnostic.py").chmod(0o444)
    write(out / "frozen.json", {
        "kind": "lineage_A_position3_retained_output_diagnostic_v1",
        "attempt_id": out.name, "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "python": sys.version, "numpy": np.__version__, "argv": sys.argv,
        "token_history": HISTORY, "selected_token": package["parents"]["selected_token"],
        "accepted_tail_review": package["parents"]["tail_review"],
        "profile": frozen["profile"], "public_interfaces": frozen["public_interfaces"],
        "bound_layer08_binary": p8["binary"], "comparison_policy": POLICY,
        "authenticated_inputs": list(inputs.records.values()),
        "measurement_order": "layers0..8, stages0..18; stop at first failing stage",
        "scalar_diagnostic": (
            "Exact rational Q/K decomposition, full-Q/full-K substitutions, nearest "
            "passing RNE cell, and all 32 paired-Q substitutions per failing head. "
            "Pairs are (d,d+32); only retained independent post-RoPE Q is substituted "
            "in software, with every actual K fixed. Recheck all 56 stage08 gates. "
            "These are conditional sensitivity measurements, never execution inputs."
        ),
        "RTL_invocations": 0, "lineage_B_state_imports": 0,
        "binary64_gate": "not evaluated; no continuous-driver admission",
    })
    gates, failure, failure_trace, failure_reference = [], None, None, None
    for layer, (trace_bytes, reference_bytes) in enumerate(transactions):
        trace = decode_trace(trace_bytes, 3)
        for stage, data in enumerate(reference_bytes):
            expected = [int(row, 16) for row in data.splitlines()]
            actual = trace[stage]
            require(len(actual) == len(expected), "actual/reference stage width mismatch")
            failed = [i for i, (a, r) in enumerate(zip(actual, expected)) if not accepts(a, r)]
            gates.append({"layer": layer, "position": 3, "stage": stage,
                          "elements": len(actual), "failure_indices": failed})
            if failed:
                failure = gates[-1]
                failure_trace, failure_reference = trace, expected
                break
        if failure:
            break
    require(failure is not None and (failure["layer"], failure["stage"]) == (8, 8),
            "retained earliest-failure identity changed")
    k = np.load(io.BytesIO(independent_k_data), allow_pickle=False)
    parent_k = np.load(io.BytesIO(independent_parent_k), allow_pickle=False)
    require(k.dtype == np.dtype("<u2") and k.shape == (4, 2, 64),
            "independent K dtype/shape mismatch")
    require(np.array_equal(k[:3], parent_k), "independent K parentage mismatch")
    actual_k = [decode_trace(data, position)[6] for position, data in enumerate(historical)]
    actual_k.append(failure_trace[6])
    q_reference = [int(row, 16) for row in transactions[8][1][4].splitlines()]
    require(len(q_reference) == len(failure_trace[4]) == 896, "Q shape mismatch")
    attribution = []
    for index, (a, r) in enumerate(zip(failure_trace[8], failure_reference)):
        head, position = divmod(index, 4)
        qa = [units(b) for b in failure_trace[4][head * 64:(head + 1) * 64]]
        qr = [units(b) for b in q_reference[head * 64:(head + 1) * 64]]
        ka = [units(b) for b in actual_k[position][(head // 7) * 64:(head // 7 + 1) * 64]]
        kr = [units(int(b)) for b in k[position, head // 7]]
        actual_dot = sum(q * key for q, key in zip(qa, ka))
        reference_dot = sum(q * key for q, key in zip(qr, kr))
        require(rounded_dot(actual_dot) == a and rounded_dot(reference_dot) == r,
                f"exact own-input score reconstruction failed at {index}")
        q_term = sum((a - r) * key for a, r, key in zip(qa, qr, kr))
        k_term = sum(q * (a - r) for q, a, r in zip(qr, ka, kr))
        cross = sum((a - r) * (k - s) for a, r, k, s in zip(qa, qr, ka, kr))
        require(q_term + k_term + cross == actual_dot - reference_dot,
                "exact score decomposition identity failed")
        if index in failure["failure_indices"]:
            attribution.append({
                "index": index, "head": head, "key_position": position,
                "actual_bits": f"{a:04x}", "reference_bits": f"{r:04x}",
                "actual": units(a) / 2**24, "reference": units(r) / 2**24,
                "absolute_error": abs(units(a) - units(r)) / 2**24,
                "relative_error": abs(units(a) - units(r)) / max(abs(units(r)), 1024),
                "ulp": abs(ordered(a) - ordered(r)),
                "actual_exact_score": exact(actual_dot, 1 << 51),
                "reference_exact_score": exact(reference_dot, 1 << 51),
                "Q_contribution": exact(q_term, 1 << 51),
                "K_contribution": exact(k_term, 1 << 51),
                "interaction": exact(cross, 1 << 51),
                "reference_Q_actual_K": score_measurement(
                    sum(q * key for q, key in zip(qr, ka)), r),
                "actual_Q_reference_K": score_measurement(
                    sum(q * key for q, key in zip(qa, kr)), r),
                "margin": rounding_margin(actual_dot, a, r),
            })
    pair_cases = []
    raw_q_reference = [int(row, 16) for row in transactions[8][1][1].splitlines()]
    for head in sorted({index // 4 for index in failure["failure_indices"]}):
        for dimension in range(32):
            channels = [head * 64 + dimension, head * 64 + dimension + 32]
            candidate_q = failure_trace[4][head * 64:(head + 1) * 64].copy()
            for channel in channels:
                candidate_q[channel % 64] = q_reference[channel]
            scores = failure_trace[8].copy()
            for position in range(4):
                keys = actual_k[position][(head // 7) * 64:(head // 7 + 1) * 64]
                scores[head * 4 + position] = rounded_dot(sum(
                    units(q) * units(key) for q, key in zip(candidate_q, keys)))
            pair_cases.append({
                "head": head, "Q_channels": channels,
                "stage01_actual_bits": [f"{failure_trace[1][c]:04x}" for c in channels],
                "stage01_reference_bits": [f"{raw_q_reference[c]:04x}" for c in channels],
                "stage04_actual_bits": [f"{failure_trace[4][c]:04x}" for c in channels],
                "stage04_reference_bits": [f"{q_reference[c]:04x}" for c in channels],
                "head_scores_bits": [f"{b:04x}" for b in scores[head * 4:(head + 1) * 4]],
                "stage08_failure_indices": [
                    i for i, (a, r) in enumerate(zip(scores, failure_reference))
                    if not accepts(a, r)
                ],
            })
    clearing_pairs = [row["Q_channels"] for row in pair_cases
                      if not row["stage08_failure_indices"]]
    result = {
        "status": "EARLIEST_RETAINED_FAILURE_CONFIRMED_NOT_REPAIRED",
        "attempt_id": out.name, "token_history": HISTORY,
        "gate_count": len(gates), "gates": gates, "earliest_failure": failure,
        "failing_scalars": attribution, "exact_own_input_scores_reconstructed": 56,
        "conditional_paired_Q_sensitivity": pair_cases,
        "conditional_stage08_clearing_Q_pairs": clearing_pairs,
        "next_discriminating_point": {
            "layer": 8, "position": 3, "candidate_Q_channel_pairs": clearing_pairs,
            "experiment": (
                "Trace these paired post-RoPE Q changes back through stage01 Q projection "
                "and stage00 RMSNorm with authenticated layer07 actual hidden held fixed, "
                "then distinguish local operator rounding from incoming hidden drift. "
                "Stage01/04 actual/reference bits are retained per pair. If no individual "
                "pair clears, use the measured full-Q substitution to discriminate the "
                "distributed Q cone. Do not inject any substituted Q into RTL."
            ),
            "scope": "A-only retained-output software sensitivity; not a causal RTL repair",
        },
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "root_cause_hypothesis": (
            "Inherited Q/K operand differences cross the FP16 score rounding boundary. "
            "The score operator is exact on each trajectory's own operands; the "
            "upstream producer is not uniquely established by this measurement."
        ),
        "repair_decision": (
            "No score-core repair is justified. Exact contributions and necessary "
            "rounding displacements are not an implementable general upstream repair. "
            "A changed softmax/Q-RoPE profile needs its affected hidden/KV cone and "
            "separate numerical admission; B's layers0-8 state is not A's accepted tail."
        ),
        "regression": "Preserve all prior passing stages and both failing scalar identities; "
                      "require exact own-input rational score reconstruction for all 56 scores.",
        "RTL_execution": "NOT_RUN: unchanged replay already exists as attempt004",
        "RTL_invocations": 0, "production_edits": 0, "lineage_B_state_imports": 0,
        "third_token_selected": False, "binary64_evaluated": False,
        "independent_review": "pending normal Host Reviewer",
    }
    write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in (
        "status", "attempt_id", "gate_count", "earliest_failure", "failing_scalars",
        "RTL_invocations", "third_token_selected",
    )}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.attempt_dir.resolve()
    require(out.parent == BUILD and out.name.startswith("lineage_a_position3_boundary_"),
            "diagnostic output must be a fresh scoped build directory")
    out.mkdir()
    try:
        run(out)
    except (OSError, ValueError, KeyError, TypeError) as error:
        write(out / "failure.json", {
            "status": "DIAGNOSTIC_INPUT_OR_EVALUATOR_REJECT",
            "failure_taxonomy": "evidence_binding_or_evaluator_failure",
            "root_cause_hypothesis": str(error),
            "regression": "Resolve the exact rejected binding/schema before a fresh attempt.",
            "RTL_invocations": 0, "RTL_correctness_conclusion": None,
        })
        raise


if __name__ == "__main__":
    main()
