#!/usr/bin/env python3
"""Account for retained Q drift; never execute or alter an RTL trajectory."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
BASE = BUILD / "model24_selected_token_position3_continuations"
Q = BASE / "lineage_a_q_producer_968f6ec61614_attempt002"
DOWN = BASE / "lineage_a_layer07_downproj_afcbb5dd88d5_attempt002"
SILU = BASE / "lineage_a_layer07_silu_24788b40db9f_attempt002"
EARLY = BUILD / "retained_l2p0s16_90efa10b0762_attempt002"
CHANNELS = (222, 223, 253, 254)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    consumed = {}
    cached = {}

    def authenticated(record):
        path = Path(record["path"])
        if path not in cached:
            data = path.read_bytes()
            if len(data) != record["bytes"]:
                raise ValueError(f"retained size mismatch: {path}")
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"retained digest mismatch: {path}")
            consumed[str(path)] = record
            cached[path] = data
        elif consumed[str(path)] != record:
            raise ValueError(f"conflicting retained binding: {path}")
        return cached[path]

    def package(path, seal_name="seal.json"):
        seal = json.loads((path / seal_name).read_text())
        records = seal if isinstance(seal, list) else seal["artifacts"]
        by_path = {Path(r["path"]): r for r in records}

        def load(name):
            return json.loads(authenticated(by_path[path / name]))

        return load, by_path

    load_q, records = package(Q)
    frozen = load_q("frozen.json")
    previous = load_q("result.json")
    norm = load_q("stage00.json")
    rows = list(csv.DictReader(
        authenticated(records[Q / "projection_contributions.csv"]).decode("ascii").splitlines()))
    input_records = {Path(r["path"]): r for r in frozen["authenticated_inputs"]}

    def input_json(path):
        return json.loads(authenticated(input_records[path]))

    # Import only the exact previously frozen arithmetic helpers, not a new model.
    for name in ("trace_lineage_a_layer08_q_producers.py",
                 "diagnose_lineage_a_position3_boundary.py",
                 "fp16_adaptation_oracle.py", "qwen2_rope_oracle.py",
                 "awq_bit_oracle.py"):
        record = next(r for r in frozen["sources"] if Path(r["path"]).name == name)
        archived = records[Q / name]
        if (archived["bytes"], archived["sha256"]) != (record["bytes"], record["sha256"]):
            raise ValueError(f"archived source binding mismatch: {name}")
        authenticated(archived)
    sys.path.insert(0, str(Q))
    from trace_lineage_a_layer08_q_producers import exact_projection
    from diagnose_lineage_a_position3_boundary import accepts, decode_trace, units
    import numpy as np

    prefix = BUILD / "token358_position3_full_continuation_attempt003"
    prepared = input_json(prefix / "layer08/prepared.json")
    tensors = {}
    for item in prepared["vectors"]["tensors"]:
        meta = item["checkpoint_tensor"]
        if "self_attn.q_proj." not in meta["name"]:
            continue
        data = np.asarray(
            [int(v, 16) for v in authenticated(
                input_records[Path(item["serialized"]["path"])]).splitlines()],
            dtype="<u2" if meta["dtype"] == "F16" else "<u4")
        if data.nbytes != meta["bytes"] or hashlib.sha256(data.tobytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"native tensor mismatch: {meta['name']}")
        tensors[meta["name"].rsplit(".", 1)[1]] = data.reshape(meta["shape"])

    actual = decode_trace(authenticated(input_records[
        prefix / "layer08/position003/raw/trace.hex"]), 3)
    reference = {}
    for stage in (1, 4, 8):
        path = Path(prepared["independent_stages"][str(stage)]["path"])
        reference[stage] = [int(v, 16) for v in authenticated(input_records[path]).splitlines()]
    operands = {
        label: [int(c[label]["bits"], 16) for c in norm["channels"]]
        for label in ("actual", "same_hidden_independent", "reference")
    }
    if operands["actual"] != actual[0]:
        raise ValueError("retained normalization table does not match trace")
    if len(norm["channels"]) != 896 or [c["index"] for c in norm["channels"]] != list(range(896)):
        raise ValueError("retained normalization coordinate coverage")

    report = []
    all_terms = []
    csv_by_key = {(int(r["q_channel"]), int(r["input_index"])): r for r in rows}
    for channel in CHANNELS:
        projected = {label: exact_projection(v, tensors, channel)
                     for label, v in operands.items()}
        actual_exact = projected["actual"][0] == actual[1][channel]
        reference_exact = projected["reference"][0] == reference[1][channel]
        if channel in frozen["channels"] and not actual_exact:
            raise ValueError(f"same-input projection discrepancy at Q{channel}")
        if channel in frozen["channels"] and not reference_exact:
            raise ValueError(f"independent projection reconstruction discrepancy at Q{channel}")
        weighted = []
        operator_total = Fraction(0)
        for index, weight in enumerate(projected["actual"][2]):
            inherited = Fraction((units(operands["same_hidden_independent"][index])
                                  - units(operands["reference"][index])) * weight, 1 << 48)
            local = Fraction((units(operands["actual"][index])
                              - units(operands["same_hidden_independent"][index])) * weight, 1 << 48)
            if channel in frozen["channels"]:
                retained = csv_by_key[channel, index]
                if (inherited != Fraction(retained["hidden_drift_contribution"])
                        or local != Fraction(retained["norm_operator_contribution"])):
                    raise ValueError(f"retained contribution discrepancy: {channel}/{index}")
            weighted.append((index, inherited))
            operator_total += local
            all_terms.append({"channel": channel, "normalized_input_index": index,
                              "inherited_q48_term": str(inherited),
                              "local_norm_q48_term": str(local), "weight_q24": weight})
        inherited_total = sum((v for _, v in weighted), Fraction(0))
        if inherited_total + operator_total != Fraction(
                projected["actual"][1] - projected["reference"][1], 1 << 48):
            raise ValueError(f"full projection decomposition does not close: {channel}")
        ranked = sorted(weighted, key=lambda v: (-abs(v[1]), v[0]))
        l1 = sum((abs(v) for _, v in weighted), Fraction(0))
        channel5 = dict(weighted)[5]
        report.append({
            "channel": channel, "rope_partner": channel - 32 if channel % 64 >= 32 else channel + 32,
            "previous_selected_channel": channel in frozen["channels"],
            "same_input_projection_exact": actual_exact,
            "independent_projection_reconstruction_exact": reference_exact,
            "modeled_actual_projection_bits": f"{projected['actual'][0]:04x}",
            "modeled_reference_projection_bits": f"{projected['reference'][0]:04x}",
            "stage01_actual": f"{actual[1][channel]:04x}",
            "stage01_reference": f"{reference[1][channel]:04x}",
            "stage04_actual": f"{actual[4][channel]:04x}",
            "stage04_reference": f"{reference[4][channel]:04x}",
            "exact_inherited_sum": str(inherited_total),
            "exact_local_norm_sum": str(operator_total),
            "inherited_l1": str(l1), "normalized_index5_term": str(channel5),
            "normalized_index5_rank": next(i + 1 for i, (j, _) in enumerate(ranked) if j == 5),
            "normalized_index5_l1_share": str(abs(channel5) / l1) if l1 else None,
            "top8": [{"index": i, "term": str(v)} for i, v in ranked[:8]],
        })

    failures = [i for i, (a, r) in enumerate(zip(actual[8], reference[8], strict=True))
                if not accepts(a, r)]
    if failures != previous["original_stage08_failure_indices"] or failures != [13, 15]:
        raise ValueError(f"original score failure set changed: {failures}")
    witnesses = []
    for index in failures:
        a, r = actual[8][index], reference[8][index]
        witnesses.append({
            "index": index, "head": index // 4, "key_position": index % 4,
            "actual_bits": f"{a:04x}", "reference_bits": f"{r:04x}",
            "signed_error": str(Fraction(units(a) - units(r), 1 << 24)),
            "within_unchanged_gate": False,
            "Q_channels_used": [192, 255],
        })

    load_down, _ = package(DOWN)
    load_silu, _ = package(SILU)
    down, silu = load_down("result.json"), load_silu("result.json")
    load_early, _ = package(EARLY, "artifacts.json")
    early = load_early("combined_mathematical_earliest_producer.json")
    early_result = load_early("result.json")
    boundary = next(row for row in early["reviewed_decomposition"] if row["index"] == 261)
    residual = down["stage18"]
    if (Fraction(residual["stage12_signed_delta"]) + Fraction(residual["stage17_signed_delta"])
            + Fraction(residual["rounding_delta"]) != Fraction(residual["stage18_signed_delta"])):
        raise ValueError("retained residual branch decomposition does not close")

    result = {
        "status": "RETAINED_ACCOUNTING_COMPLETE_NO_CANDIDATE",
        "requested_channels": list(CHANNELS), "previous_selected_channels": frozen["channels"],
        "scope": "Read-only arithmetic on attempt002 outputs and their explicitly bound retained inputs.",
        "channel_coverage": "253 is not 255: reconstruct 253 from the same frozen full native tensors and norm/trace arrays; no alias or reference intervention.",
        "comparison_policy": frozen["comparison_policy"],
        "channels": report, "score_failures": witnesses,
        "exact_projection_terms": len(all_terms),
        "dominance_definition": "Descending absolute exact pre-round Q contribution, with complete 896-term L1 denominator; not score-error causal attribution.",
        "nearest_shared_cone": "L7/P3/S12 and S17 -> rounded S18 -> full-vector L8/P3/S0 RMSNorm -> native Q S1 -> RoPE S4 -> head3 scores S8.",
        "residual_index5": residual, "stage17_index5": down["stage17"],
        "selected_silu_weighted_decomposition": silu["weighted_decomposition_sum"],
        "retained_silu_boundary": silu["root_cause_hypothesis"],
        "early_boundary": boundary,
        "early_boundary_candidate_disposition": early_result["candidate_decision"],
        "early_boundary_link": "Common RMSNorm/trajectory dependency type only. These retained packages do not contain a candidate/control sensitivity joining L0/P0/S0/261 to all four L8/P3 Q channels or scores13/15. Different positions and candidate histories cannot be equated.",
        "scientific_blocker": "An additive contribution rank is not a repair intervention. RMSNorm couples all hidden coordinates through its denominator; residual branches oppose one another; downstream nonlinear/rounding and historical K/V paths are not controlled by this accounting. No complete four-channel score-weighted causal intervention or candidate-native affected-history regression is established. No unique dominant upstream implementation defect or general repair follows.",
        "candidate": None,
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "root_cause_hypothesis": "Inherited multi-branch hidden drift reaches the Q projection; the responsible general upstream arithmetic change remains unidentified.",
        "regression": "Any genuinely different general candidate must retain full independent FP16 gates (including L8/P3/S8/13,15 and L2/P0/S16/813), binary64-v1 and compatible affected hidden/K/V histories. Do not repeat the rejected quotient/root RMSNorm variants.",
        "next_discriminating_question": "Can a general, separately justified change upstream of the shared residual/norm cone reduce the full signed Q/score drift on compatible histories without the known L2/P0/S16/813 regression? The present retained tables cannot answer this intervention question.",
        "RTL_invocations": 0, "reference_injections": 0, "token358_reruns": 0,
        "numerical_policy_changes": 0, "binary64_evaluated": False,
        "third_token_claim": False, "independent_review": "Requires normal Host Reviewer; no review identity minted.",
        "authentication_boundary": "Consumed bytes match existing attempt002 seals/frozen bindings. This is not a new external trust root or independent Reviewer verdict.",
        "consumed_existing_bindings": list(consumed.values()),
    }
    with (out / "contributions.csv").open("x", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_terms[0]))
        writer.writeheader()
        writer.writerows(all_terms)
    with (out / "result.json").open("x", encoding="ascii") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(result["status"])
    for row in report:
        print(f"Q{row['channel']}: index5 rank={row['normalized_index5_rank']}, "
              f"L1 share={row['normalized_index5_l1_share']}, inherited={row['exact_inherited_sum']}")
    print("score failure indices:", failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)
