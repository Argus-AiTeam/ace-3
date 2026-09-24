"""Exact retained-operand feasibility diagnostic; never executes decoder RTL."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys

import numpy as np

from ace3.model.candidates import binary64_fp16_excess_v1 as binary64
from ace3.model.candidates import decoder_gate_policy as policy
from ace3.model.fp16_adaptation_oracle import decode_f16_q24, residual_add

ROOT = Path(__file__).resolve().parents[3]
RETAINED_RESULT_SHA256 = "523e4e88dc827a11c4e568c1b1249f59c53ea5fafb6b3d67e5b7e60e267827ac"
INDEX = 62


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def half(bits):
    require(type(bits) is int and 0 <= bits <= 0xffff and bits & 0x7c00 != 0x7c00,
            "invalid finite FP16 encoding")
    return Fraction.from_float(struct.unpack("<e", struct.pack("<H", bits))[0])


def gate(actual, reference):
    av, rv = half(actual), half(reference)
    absolute = abs(av - rv)
    relative = absolute / max(abs(rv), Fraction(1, 1 << 14))
    def ordered(bits):
        return -int(bits & 0x7fff) if bits & 0x8000 else bits
    ulp = abs(ordered(actual) - ordered(reference))
    return {"absolute_error": str(absolute), "relative_error": str(relative), "ulp": ulp,
            "passed": absolute <= Fraction(1, 8) or (relative < Fraction(1, 1000) and ulp <= 1)}


def independent_add(a, b):
    total = half(a) + half(b)
    if abs(total) >= 65520:
        return (0xfbff if total < 0 else 0x7bff), False, True
    # Every sum of two finite binary16 values is exactly representable in binary64.
    rounded = struct.unpack("<H", struct.pack("<e", float(total)))[0]
    if total == 0 and a == b == 0x8000:
        rounded = 0x8000
    return rounded, False, False


def admitted_fp16(domain, reference):
    exact = [bits for bits in domain if gate(bits, reference)["passed"]]
    comparison = policy.compare_fp16(
        np.asarray(domain, dtype="<u2"), np.full(len(domain), reference, dtype="<u2"))
    rejected = {row["index"] for row in comparison["failures"]}
    require(exact == [bits for i, bits in enumerate(domain) if i not in rejected],
            "independent rational predicate disagrees with unchanged FP16 evaluator")
    return exact


def scalar(bits):
    return {"bits": f"{bits:04x}", "value": str(half(bits))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    attempt, out = args.attempt.resolve(), args.out.resolve()
    require(out.parent == ROOT / "build" and out.name.startswith("l8_s12_cross_boundary_feasibility_"),
            "output outside bounded ignored build namespace")
    out.mkdir(exist_ok=False)
    bindings = {}

    def bind(rec):
        current = record(rec["path"])
        require(all(current[k] == rec[k] for k in ("path", "bytes", "sha256")),
                f"input authentication failure: {rec['path']}")
        bindings[current["path"]] = current
        return Path(current["path"])

    def load(rec):
        return json.loads(bind(rec).read_text())

    result_rec = record(attempt / "result.json")
    require(result_rec["sha256"] == RETAINED_RESULT_SHA256, "unreviewed retained result")
    retained = load(result_rec)
    frozen = load(retained["freeze"])
    manifest = load(record(attempt / "output_manifest.json"))
    members = {r["path"]: r for r in manifest["artifacts"]}
    require(len(members) == len(manifest["artifacts"]), "duplicate retained manifest entry")
    bind(members[str(attempt / "result.json")])
    bind(members[str(attempt / "freeze.json")])
    review = load(record(args.review))
    require(review["kind"] == "round_reviewed_handoff" and review["producer_role"] == "reviewer"
            and review["mission_id"] == "2f35dc72423f" and review["review"]["status"] == "done",
            "missing genuine policy-bound predecessor review")
    require(frozen["policy_id"] == policy.POLICY_ID
            and frozen["binary64_profile"] == binary64.PROFILE_ID
            and retained["first_failure"]["node"] == [8, 0, 12]
            and retained["first_failure"]["fp16"]["failures"][0]["index"] == INDEX
            and retained["decoder_rtl_invocations"] == 0, "unexpected reviewed frontier")
    inputs = {r["path"]: r for r in frozen["input_bindings"]}
    for rec in frozen["sources"]:
        bind(rec)

    def input_ending(suffix):
        matches = [rec for path, rec in inputs.items() if path.endswith(suffix)]
        require(len(matches) == 1, f"missing/ambiguous retained input: {suffix}")
        return matches[0]

    for suffix in ("/fp16_adaptation_oracle.py", "/binary64_fp16_excess_v1.py"):
        bind(input_ending(suffix))
    old_freeze = load(input_ending(
        "/single_round_residual_rtl_execution_a06f66e432b0_attempt001/freeze.json"))
    for rec in old_freeze["sources"]:
        bind(rec)
    stages = {}
    for layer in (7, 8):
        provenance = [p for p in retained["software_provenance"] if p["layer"] == layer]
        require(len(provenance) == 1 and provenance[0]["input_state"] is None
                and provenance[0]["kv_policy"] == "own empty prior P0 KV", "P0 provenance mismatch")
        rec = provenance[0]["stages"]
        require(rec == (members if layer == 8 else inputs)[rec["path"]], "unbound stage archive")
        with np.load(bind(rec), allow_pickle=False) as archive:
            stages[layer] = {k: archive[k].copy() for k in archive.files}
        arrays = stages[layer]
        require(all(a.dtype == np.dtype("<u2") for a in arrays.values())
                and arrays["input_hidden"].shape == (896,)
                and all(arrays[f"stage{s:02d}"].shape == (896,) for s in (11, 12, 18))
                and all(arrays[f"input_cache_{k}"].shape == (0, 128) for k in ("k", "v"))
                and all(arrays[f"output_cache_{k}"].shape == (1, 128)
                        and np.array_equal(arrays[f"output_cache_{k}"][0], arrays[f"stage{s:02d}"])
                        for k, s in (("k", 6), ("v", 7))), "hidden/KV shape or own-P0-state mismatch")
        require(all(not np.any((a & 0x7c00) == 0x7c00) for a in arrays.values()),
                "nonfinite retained stage")
    require(np.array_equal(stages[7]["stage18"], stages[8]["input_hidden"]),
            "L7 actual hidden is not L8 actual input")
    with np.load(bind(input_ending("/retained_graph.npz")), allow_pickle=False) as graph:
        h_reference = graph["l7_p0_s18_reference"].copy()
    reference_freeze = load(input_ending(
        "/layer08_position2_q_only_rope_rtl_86938809f0a2_attempt010/frozen.json"))
    transactions = [r for r in reference_freeze["reference_transactions"]
                    if r["layer"] == 8 and r["position"] == 0]
    require(len(transactions) == 1, "missing/ambiguous independent L8 reference")
    refs = {}
    for stage in ("11", "12"):
        rec = transactions[0]["stages"][stage]
        words = bind(rec).read_text().split()
        require(len(words) == 896 and all(len(w) == 4 for w in words), "FP16 hex shape")
        array = np.asarray([int(w, 16) for w in words], dtype="<u2")
        require(hashlib.sha256(array.tobytes()).hexdigest() == rec["semantic_sha256"],
                "FP16 reference semantic drift")
        refs[stage] = array
    reference64 = np.load(bind(input_ending("/layer07_position000_reference.npy")), allow_pickle=False)
    require(reference64.dtype == np.dtype("<f8") and reference64.shape == (896,)
            and h_reference.dtype == np.dtype("<u2") and h_reference.shape == (896,),
            "original reference shape/dtype mismatch")
    r_hex = float(reference64[INDEX]).hex()
    r = Fraction.from_float(binary64._reference(r_hex))
    a_h, a_o, a_s12 = (int(stages[8][k][INDEX]) for k in ("input_hidden", "stage11", "stage12"))
    r_h, r_o, r_s12 = int(h_reference[INDEX]), int(refs["11"][INDEX]), int(refs["12"][INDEX])
    sources = [record(ROOT / path) for path in (
        "ace3/model/candidates/diagnose_l8_s12_cross_boundary.py",
        "tests/test_l8_s12_cross_boundary_feasibility.py",
        "ace3/model/fp16_adaptation_oracle.py",
        "ace3/model/awq_bit_oracle.py",
        "ace3/model/candidates/binary64_fp16_excess_v1.py",
        "ace3/model/candidates/decoder_gate_policy.py",
        "ace3/contracts/candidates/binary64_fp16_excess_v1.json",
        "ace3/contracts/candidates/decoder_gate_policy_v2.json")]
    freeze = {
        "scope": "L8/P0/index62 retained software scalar diagnostic, not RTL or trajectory admission",
        "policy_id": policy.POLICY_ID, "binary64_profile": binary64.PROFILE_ID,
        "bindings": list(bindings.values()), "sources": sources,
        "tools": {"python": sys.version, "executable": sys.executable,
                  "numpy": np.__version__, "platform": platform.platform()},
        "domain": "all 63488 finite binary16 encodings, including both signed zeros",
        "arithmetic": "exact rational enumeration; native residual_add; independent struct RNE oracle",
        "H": "abs(h-original_binary64_r)-min_finite_FP16(abs(h-r)) <= 1/8",
        "O_and_S12": "unchanged finite FP16 gate, original independently propagated references",
        "controls": ["actual_H_actual_O", "actual_H_reference_O",
                     "reference_H_actual_O", "reference_H_reference_O"],
        "no_reference_injection": True, "decoder_rtl_invocations": 0,
        "state_imports": 0, "public_RTL_contract_changed": False,
        "prior_public_contract": frozen["public_contract"],
        "prior_parameters": frozen["parameters"],
        "original_failure": retained["first_failure"],
        "validation_command": "PYTHONPATH=. python3 -m unittest discover -s tests "
                              "-p test_l8_s12_cross_boundary_feasibility.py",
        "decision_rule": "Empty H x O -> fixed-policy native-boundary blocker; nonempty -> scalar possibility only",
    }
    write_json(out / "freeze.json", freeze)
    domain = [b for b in range(65536) if b & 0x7c00 != 0x7c00]
    values = {b: half(b) for b in domain}
    require(len(domain) == 63488 and all(values[b] == Fraction(decode_f16_q24(b)[0], 1 << 24)
                                       for b in domain), "independent finite domain decoding mismatch")
    q = min(abs(v - r) for v in values.values())
    nearest, native_q = binary64._nearest(float(reference64[INDEX]))
    require(q == native_q, "independent representability lower bound mismatch")
    h_set = [b for b in domain if abs(values[b] - r) - q <= Fraction(1, 8)]
    o_set, s12_set = admitted_fp16(domain, r_o), admitted_fp16(domain, r_s12)
    h_rows = []
    for bits in h_set:
        row = binary64.evaluate_layer_final_output(actual_fp16_bits=bits, reference_binary64_hex=r_hex)
        require(row["accepted"] and Fraction(row["q"]) == q, "binary64-v1 evaluator mismatch")
        h_rows.append({**scalar(bits), "actual_error": str(abs(values[bits] - r)),
                       "excess_error": str(abs(values[bits] - r) - q)})
    require(a_h in h_set and a_o in o_set, "retained witness outside admitted operand sets")
    write_json(out / "hidden_set.json", {"reference_binary64_hex": r_hex, "q": str(q),
                                       "nearest": scalar(nearest), "admitted": h_rows})
    with (out / "o_set.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["bits", "value", "absolute_error",
                                                    "relative_error", "ulp", "passed"])
        writer.writeheader()
        writer.writerows({**scalar(b), **gate(b, r_o)} for b in o_set)
    controls = []
    for hn, h in (("actual", a_h), ("reference", r_h)):
        for on, o in (("actual", a_o), ("reference", r_o)):
            native = residual_add(h, o)
            require(native == independent_add(h, o), "same-input residual oracle mismatch")
            output, invalid, saturated = native
            controls.append({"name": f"{hn}_H_{on}_O", "H": scalar(h), "O": scalar(o),
                             "exact_sum": str(half(h) + half(o)), "output": scalar(output),
                             "invalid": invalid, "saturated": saturated, "S12": gate(output, r_s12),
                             "diagnostic_only_not_a_spliced_trajectory": True})
    require(int(controls[0]["output"]["bits"], 16) == a_s12
            and int(controls[3]["output"]["bits"], 16) == r_s12,
            "retained actual/reference S12 not reproduced on its own inputs")
    outputs, admitted_pairs, pair_count = set(), 0, 0
    with (out / "pairs.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["H_bits", "O_bits", "exact_sum", "output_bits",
                                                    "output_value", "invalid", "saturated",
                                                    "absolute_error", "relative_error", "ulp", "passed"])
        writer.writeheader()
        for h in h_set:
            for o in o_set:
                native = residual_add(h, o)
                require(native == independent_add(h, o), "Cartesian same-input residual oracle mismatch")
                output, invalid, saturated = native
                comparison = gate(output, r_s12)
                require(not invalid and not saturated, "unexpected finite-domain residual exception")
                outputs.add(output)
                admitted_pairs += comparison["passed"]
                pair_count += 1
                writer.writerow({"H_bits": f"{h:04x}", "O_bits": f"{o:04x}",
                                 "exact_sum": str(half(h) + half(o)), "output_bits": f"{output:04x}",
                                 "output_value": str(half(output)), "invalid": invalid,
                                 "saturated": saturated, **comparison})
    require(pair_count == len(h_set) * len(o_set), "incomplete Cartesian enumeration")
    for rec in list(bindings.values()) + sources:
        require(record(rec["path"]) == rec, "source/input changed during diagnostic")
    blocked = admitted_pairs == 0
    result = {
        "status": "AUTHENTICATED_SCALAR_BOUNDARY_BLOCKER" if blocked else "SCALAR_COMBINATIONS_EXIST",
        "freeze": record(out / "freeze.json"), "policy_id": policy.POLICY_ID,
        "reference_binary64_hex": r_hex, "q": str(q), "H": h_rows,
        "actual": {"H": scalar(a_h), "O": scalar(a_o), "S12": scalar(a_s12)},
        "reference_FP16": {"H": scalar(r_h), "O": scalar(r_o), "S12": scalar(r_s12)},
        "signed_drifts": {"H": str(half(a_h) - half(r_h)), "O": str(half(a_o) - half(r_o)),
                          "S12": str(half(a_s12) - half(r_s12))},
        "finite_encoding_count": len(domain), "H_count": len(h_set), "O_count": len(o_set),
        "O_min": scalar(min(o_set, key=half)), "O_max": scalar(max(o_set, key=half)),
        "pair_count": pair_count, "S12_admitted_pairs": admitted_pairs,
        "native_outputs": [scalar(b) for b in sorted(outputs)],
        "S12_admitted_outputs": [scalar(b) for b in s12_set],
        "controls": controls,
        "independent_oracle": {"finite_decodings": len(domain), "FP16_gate_domains": 2,
                               "residual_pairs": pair_count, "controls": len(controls),
                               "all_agreed": True},
        "failure_taxonomy": "cross_boundary_admissible_set_empty" if blocked else None,
        "root_cause_hypothesis": (
            "At this scalar, the inherited H displacement, not the observed O displacement, "
            "determines the S12 failure; controls do not identify a unique upstream producer."),
        "regression": "Retain exact H/O domain enumeration and all four own/cross-input residual controls.",
        "next_decision": (
            "No H-only or O-only repair, nor any joint admitted H/O repair, can clear this scalar "
            "under native residual_add and the unchanged references/gates. Route this authenticated "
            "cross-boundary incompatibility through Reviewer/Planner for a separately authorized "
            "boundary/contract decision; do not replay RTL or infer an O-only repair. "
            "Same-input agreement establishes residual-add correctness here, not O-projection correctness."
            if blocked else
            "Existence is necessary, not a coherent trajectory repair; evaluate a general candidate "
            "only after independently justified upstream and state-lineage analysis."),
        "limits": "Cartesian relaxation, not reachability proof. No O-projection replay or unique producer "
                  "localization. Original S18 diagnostic and S12 FAIL remain unchanged. No RTL, model, "
                  "token, synthesis, or hardware admission.",
        "normal_host_review": "REQUIRED", "decoder_rtl_invocations": 0,
        "references_injected_into_execution": False, "gates_or_policy_changed": False,
    }
    write_json(out / "result.json", result)
    write_json(out / "output_manifest.json", {
        "artifacts": [record(p) for p in sorted(out.iterdir()) if p.is_file()],
        "normal_host_review": "REQUIRED"})
    print(json.dumps({k: result[k] for k in (
        "status", "H_count", "O_count", "pair_count", "S12_admitted_pairs",
        "O_min", "O_max", "native_outputs", "signed_drifts")}, sort_keys=True))


if __name__ == "__main__":
    main()
