"""Bounded canonical-local diagnostic, not a candidate or reference-policy adoption.

Recompute all captured L0-L2/P0 stages from the official token9707 embedding,
using the frozen independent v3 operators and exact rational Q24 state.
Same-input retained comparisons and the independently propagated control are
separate. Global expectations remain the original binary64 arrays. An earliest
bit divergence is not, by itself, a causal localization of the global failure.

--decompose reuses the reviewed control without executing any model operator.
Residual deltas are exact signed fractions, not counterfactual causal effects.
Original-input binary64 endpoints support a layer-wide branch-sum split only;
they do not supply a global S12 reference or identify S11 versus S17.

--original-global captures the frozen original binary64 recurrence from the
official embedding, never from either Q24 lineage. Every regenerated endpoint
must match the retained original bits before its operands enter the diagnostic.
This adds evidence, not a new reference policy or candidate admission.

--producer-cones reuses reviewed original-global captures, without layer calls.
For both Q24 lineages it evaluates independent frozen binary64 operators on
actual operands and telescopes every relevant P0 producer coordinate. Local
FP16 oracle differences, output rounding and operand drift stay separate.
The S16 two-input split replaces gate before up; it is an ordered diagnostic,
not a unique causal attribution or a new mandatory reference boundary.

--weighted-backward consumes that reviewed telescope without model calls.
Binary64 adjoints transport finite differences to the official root; exact
rational ledgers expose coefficient/reduction remainders rather than hiding
them in upstream effects. S16 uses the retained gate-first secant. RMSNorm
splits the actual-radius direct term from the shared-radius coupling term.
These conventions define an accounting path, not a unique causal intervention.

--repair-sufficiency tests three uniform S16 rounding counterfactuals after the
reviewed canonical L0/S15 prefix. A frozen original binary64 SiLU/product
operator consumes each control's own FP16 operands; its output is rounded to
FP16 (RNE, toward zero, or away from zero). All other operators are canonical
local software controls, not independent validation of that implementation.
Each control owns its Q24 residual and FP16 KV history and stops at its first
mandatory v3 failure. No unrounded S16 value reaches S17. These controls neither
adopt directed rounding nor establish a native implementation or RTL repair.
"""

import argparse
from collections import Counter
import csv
from fractions import Fraction
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
from safetensors import safe_open

from ace3.model.candidates import decoder_gate_policy_v3 as gates
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import q24_software_root_v3 as retained
from ace3.model.candidates import residual_exact_grid_q24_reference_v1 as rational


ROOT = retained.ROOT
INPUT = ROOT / "build/q24_software_root_995d0ba22311_attempt002"
VALIDATION = ROOT / "build/q24_software_root_995d0ba22311_validation002"
REVIEW = retained.HANDOFFS / "995d0ba22311/round-0001.json"
WITNESS = {
    "index": 62, "actual_fp16_bits": "616e", "nearest_fp16_bits": "616f",
    "reference_binary64_hex": "0x1.5bc1b05c9106ap+9",
    "excess_error": "1/2", "excess_budget": "1/8",
}
CONTROL_INPUT = ROOT / "build/q24_software_root_1783b114f9d3_attempt001"
CONTROL_REVIEW = retained.HANDOFFS / "1783b114f9d3/round-0001.json"
CAPTURE_INPUT = ROOT / "build/q24_software_root_e6b4f7dd0144_attempt002"
CAPTURE_REVIEW = retained.HANDOFFS / "e6b4f7dd0144/round-0001.json"
PRODUCER_INPUT = ROOT / "build/q24_software_root_a33b39ff3899_attempt002"
PRODUCER_REVIEW = retained.HANDOFFS / "a33b39ff3899/round-0001.json"
WEIGHTED_INPUT = ROOT / "build/q24_software_root_e0701737bb56_attempt001"
WEIGHTED_REVIEW = retained.HANDOFFS / "e0701737bb56/round-0001.json"
ROUNDING_CONTROLS = ("rne", "toward_zero", "away_zero")
CONE_STAGES = (0, 3, 10, 11, 13, 14, 15, 16, 17)
CONE_TERMS = ("local_fp16_delta", "oracle_rounding_difference",
              "binary64_to_fp16_rounding", "operand_trajectory_delta")
require = local.require


def cone_values(value, count):
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype("<f8")
            and value.shape == (count,) and np.all(np.isfinite(value)),
            "invalid binary64 cone operand")
    return value


def cone_original(captured, previous):
    """Authenticate captured edges; P0 has exactly one visible cache entry."""
    original = dict(captured)
    require(np.array_equal(cone_values(original["input_hidden"], 896).view("<u8"),
                           cone_values(previous, 896).view("<u8")),
            "original hidden trajectory splice")
    for kind, stage in (("k", 2), ("v", 3)):
        require(np.array_equal(cone_values(original["output_cache_" + kind], 128).view("<u8"),
                               cone_values(original[f"stage{stage:02d}"], 128).view("<u8")),
                "original P0 cache identity")
    original["stage07"] = original["output_cache_v"]
    original["stage09"] = np.ones(14, dtype="<f8")
    for stage in (1, 2, 3, 11, 14, 15, 17):
        producer = local.OPERANDS[stage][0]
        size = local.SIZES[int(producer[5:])]
        require(np.array_equal(cone_values(original[f"stage{stage:02d}_input"], size).view("<u8"),
                               cone_values(original[producer], size).view("<u8")),
                "original projection producer identity")
    return original


def cone_evaluator(namespace, tensors, layer):
    """Use original Torch operators, independent of both Q24 implementations."""
    import torch
    import torch.nn.functional as functional

    require(type(layer) is int and layer in range(3), "cone scope is L0-L2")
    projections = {}
    for stage in (3, 11, 14, 15, 17):
        prefix = f"model.layers.{layer}.{local.PROJECTIONS[stage]}"
        q = namespace["_unpack_words"](tensors[prefix + ".qweight"])
        z = namespace["_unpack_words"](tensors[prefix + ".qzeros"])
        weight = (q - np.repeat(z, 128, axis=0)).astype("<f8")
        weight *= np.repeat(tensors[prefix + ".scales"].astype("<f8"), 128, axis=0)
        projections[stage] = SimpleNamespace(
            reference_weight=torch.from_numpy(weight),
            reference_bias=torch.from_numpy(tensors[prefix + ".bias"].astype("<f8"))
            if stage == 3 else None)

    def evaluate(stage, operands):
        require(stage in CONE_STAGES and set(operands) == set(local.OPERANDS[stage]),
                "wrong cone operator/operand identity")
        x = {name: torch.from_numpy(cone_values(
            value, 896 if name == "input_hidden" else local.SIZES[int(name[5:])])).reshape(1, -1)
             for name, value in operands.items()}
        first = x[local.OPERANDS[stage][0]]
        with torch.no_grad():
            if stage in (0, 13):
                norm = "input_layernorm" if stage == 0 else "post_attention_layernorm"
                output = namespace["_torch_rmsnorm"](
                    first, tensors[f"model.layers.{layer}.{norm}.weight"])
            elif stage in projections:
                output = namespace["_reference_projection"](first, projections[stage])
            elif stage == 10:
                values = x["stage07"].reshape(1, 2, 64)[:, torch.arange(14) // 7, :]
                output = torch.einsum("bhk,khd->bhd", first.reshape(1, 14, 1), values)
            else:
                output = functional.silu(first) * x["stage15"]
        return cone_values(output.numpy().reshape(-1).copy(), local.SIZES[stage])

    return evaluate, projections


def cone_rows(words, local_words, same_input, original, layer, stage, lineage):
    require(type(layer) is int and layer in range(3) and stage in CONE_STAGES,
            "cone row scope is L0-L2/P0")
    count = local.SIZES[stage]
    actual = local.finite_words(words, (count,))
    canonical = local.finite_words(local_words, (count,))
    cone_values(same_input, count)
    cone_values(original, count)
    rounded_words = local.rounded(same_input)
    rounded = local.finite_words(rounded_words, (count,))
    rows = []
    for index in range(count):
        a, c, h, f, g = (Fraction(float(v[index]))
                         for v in (actual, canonical, rounded, same_input, original))
        terms = (a - c, c - h, h - f, f - g)
        require(sum(terms) == a - g, "operator telescope does not close")
        rows.append({
            "layer": layer, "position": 0, "stage": stage, "lineage": lineage, "index": index,
            "actual_fp16_bits": f"{int(words[index]):04x}",
            "local_fp16_bits": f"{int(local_words[index]):04x}",
            "binary64_rne_fp16_bits": f"{int(rounded_words[index]):04x}",
            "same_input_binary64_hex": float(same_input[index]).hex(),
            "original_binary64_hex": float(original[index]).hex(),
            **dict(zip(CONE_TERMS, map(str, terms), strict=True)),
            "same_input_operator_delta": str(a - f), "global_operator_delta": str(a - g),
            "classification": drift_class(f - g, a - f),
            "gate_operand_delta": "", "up_operand_delta": "", "closure_error": "0"})
    return rows


def down_weighted_rows(stage16_rows, stage17_row, weight):
    """Exact real-product accounting; binary64 dot rounding remains explicit."""
    require(len(stage16_rows) == 4864 and stage17_row["stage"] == 17
            and stage17_row["layer"] == 2 and stage17_row["index"] == 62,
            "down attribution scope/coverage")
    cone_values(weight, 4864)
    terms = (*CONE_TERMS[:3], "gate_operand_delta", "up_operand_delta")
    totals = {term: Fraction(0) for term in terms}
    rows = []
    for index, (source, scale) in enumerate(zip(stage16_rows, weight, strict=True)):
        require(source["stage"] == 16 and source["layer"] == 2 and source["index"] == index
                and source["lineage"] == stage17_row["lineage"], "down producer splice")
        w = Fraction(float(scale))
        contributions = {term: w * Fraction(source[term]) for term in terms}
        delta = w * Fraction(source["global_operator_delta"])
        require(sum(contributions.values()) == delta, "weighted producer telescope does not close")
        for term in terms:
            totals[term] += contributions[term]
        rows.append({"layer": 2, "position": 0, "stage": 17, "output_index": 62,
                     "lineage": source["lineage"], "input_index": index,
                     "weight_binary64_hex": float(scale).hex(),
                     **{term: str(value) for term, value in contributions.items()},
                     "weighted_input_delta": str(delta), "closure_error": "0"})
    weighted = sum(totals.values())
    reduction = Fraction(stage17_row["operand_trajectory_delta"]) - weighted
    own = Fraction(stage17_row["same_input_operator_delta"])
    require(own + weighted + reduction == Fraction(stage17_row["global_operator_delta"]),
            "down output telescope does not close")
    return rows, {
        "layer": 2, "position": 0, "stage": 17, "index": 62, "lineage": stage17_row["lineage"],
        "weighted_stage16_terms": {term: str(value) for term, value in totals.items()},
        "weighted_input_delta": str(weighted), "binary64_reduction_rounding_delta": str(reduction),
        "same_input_operator_delta": str(own),
        "global_operator_delta": stage17_row["global_operator_delta"], "closure_error": "0"}


def producer_cones(out, namespace, tensors, data, captured, references, trajectories,
                   root, embedding, prior, diagnosis, progress):
    started = time.monotonic()
    progress.update(original_global_layer_invocations=0, local_oracle_invocations=0,
                    model_operator_invocation_unit="frozen binary64 single-operator calls, including S16 ordered hybrids")
    parents = {"actual": root, "canonical": root}
    previous = embedding.view("<f2").astype("<f8")
    retained.verify_parent(root, root, embedding=embedding)
    summaries, focus, weighted, phases, artifacts, global_gates = [], [], [], [], [], []
    earliest, coordinates = None, 0
    path = out / "producer_cones.csv"
    with path.open("x", newline="", encoding="ascii") as stream:
        writer = None
        for layer in range(3):
            begin = time.monotonic()
            original = cone_original(captured[layer], previous)
            verify_global_endpoint(original, references[layer])
            operator, projections = cone_evaluator(namespace, tensors[layer], layer)

            def evaluate(stage, operands):
                progress["model_operator_invocations"] += 1
                return operator(stage, operands)

            for name in parents:
                arrays = data[layer][name]
                retained.verify_parent(retained.state_from(arrays, "input", "input_hidden"), parents[name])
                for stage, count in local.SIZES.items():
                    local.finite_words(arrays[f"stage{stage:02d}"], (count,))
                for stage in (6, 7, 12, 18):
                    retained.check_stage_state(stage, arrays, parents[name])
                require(np.array_equal(arrays["stage09"], np.full(14, 0x3c00, dtype="<u2")),
                        "P0 singleton softmax identity")
                parents[name] = retained.state_from(arrays, "output", "stage18")
                for stage in (12, 18):
                    original_residual_rows(arrays, original, layer, stage, name)
                evaluated = gates.evaluate_decoder_stage(
                    stage=18, actual=arrays["stage18"], reference=trajectories[layer],
                    policy=gates.POLICY_ID, reference_binary64=references[layer])
                expected = (data[layer]["reports"][18]["binary64_v1"] if name == "actual"
                            else data[layer]["control_reports"][18]["global_binary64_v1"])
                require(evaluated["binary64_v1"] == expected, "retained global gate changed")
                global_gates.append({"layer": layer, "lineage": name,
                                     "binary64_v1": evaluated["binary64_v1"]})
            stage16, saved = {}, {name: {} for name in parents}
            for stage in CONE_STAGES:
                expected = cone_values(original[f"stage{stage:02d}"], local.SIZES[stage])
                replay = evaluate(stage, {k: original[k] for k in local.OPERANDS[stage]})
                require(np.array_equal(replay.view("<u8"), expected.view("<u8")),
                        f"original operator reproduction drift L{layer}/S{stage}")
                for name in parents:
                    arrays = data[layer][name]
                    operands = {k: arrays[k] for k in local.OPERANDS[stage]}
                    values = {k: local.finite_words(v, (896,) if k == "input_hidden"
                              else (local.SIZES[int(k[5:])],)) for k, v in operands.items()}
                    same = evaluate(stage, values)
                    progress["local_oracle_invocations"] += 1
                    local_words = local.local_reference(stage, operands, tensors[layer], layer)
                    retained_local = (data[layer]["local"][f"stage{stage:02d}"] if name == "actual"
                                      else arrays[f"stage{stage:02d}"])
                    require(np.array_equal(local_words, retained_local), "retained local oracle drift")
                    rows = cone_rows(arrays[f"stage{stage:02d}"], local_words, same, expected,
                                     layer, stage, name)
                    saved[name][f"stage{stage:02d}"] = same
                    saved[name][f"local_stage{stage:02d}"] = local_words
                    if stage == 16:
                        hybrid = evaluate(stage, {"stage14": values["stage14"],
                                                  "stage15": original["stage15"]})
                        saved[name]["stage16_actual_gate_original_up"] = hybrid
                        for index, row in enumerate(rows):
                            row["gate_operand_delta"] = str(Fraction(float(hybrid[index]))
                                                            - Fraction(float(expected[index])))
                            row["up_operand_delta"] = str(Fraction(float(same[index]))
                                                          - Fraction(float(hybrid[index])))
                            require(Fraction(row["gate_operand_delta"]) + Fraction(row["up_operand_delta"])
                                    == Fraction(row["operand_trajectory_delta"]), "SiLU telescope closure")
                        stage16[name] = rows
                    if stage in (11, 17):
                        focus.append(rows[62])
                    if layer == 2 and stage == 17:
                        contributions, total = down_weighted_rows(
                            stage16[name], rows[62], projections[17].reference_weight.numpy()[:, 62])
                        expected_delta = next(r["operator_delta"] for r in prior["original_global_index62"]
                                              if r["layer"] == 2 and r["stage"] == 18 and r["lineage"] == name)
                        require(total["global_operator_delta"] == expected_delta,
                                "reviewed down-projection witness changed")
                        contribution_path = out / f"layer02_{name}_down_index62.csv"
                        with contribution_path.open("x", newline="", encoding="ascii") as destination:
                            contribution_writer = csv.DictWriter(destination, fieldnames=list(contributions[0]))
                            contribution_writer.writeheader()
                            contribution_writer.writerows(contributions)
                        artifacts.append(retained.record(contribution_path))
                        weighted.append(total)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                    writer.writerows(rows)
                    coordinates += len(rows)
                    if earliest is None:
                        earliest = next((r for r in rows if Fraction(r["global_operator_delta"])), None)
                    summaries.append(dict(decomposition_totals(rows, CONE_TERMS), layer=layer,
                                          stage=stage, lineage=name,
                                          retained_local_fp16_diagnostic=compare(
                                              arrays[f"stage{stage:02d}"], local_words)))
            for name, arrays in saved.items():
                artifacts.append(retained.save(out / f"layer{layer:02d}_{name}_same_input.npz", arrays))
            previous = original["stage18"]
            phases.append({"phase": "producer_cone_operators_and_exact_accounting", "layer": layer,
                           "seconds": time.monotonic() - begin})
    require(coordinates == 6 * sum(local.SIZES[s] for s in CONE_STAGES),
            "incomplete producer coordinate coverage")
    witness = preserve_witness(data[2]["reports"][18], diagnosis)
    require(witness == prior["preserved_rejecting_witness"], "reviewed FAIL witness changed")
    retained.write(out / "global_gates.json", global_gates)
    artifacts.extend(retained.record(p) for p in (path, out / "global_gates.json"))
    return {
        "status": "DIAGNOSTIC_COMPLETE", "numerical_status": "FAIL",
        "preserved_rejecting_witness": witness, "producer_coordinates": coordinates,
        "original_operator_reproduced_coordinates": coordinates // 2,
        "original_operator_bit_mismatches": 0, "operator_telescope_closure_errors": 0,
        "weighted_down_input_coordinates": 9728, "producer_totals": summaries,
        "index62_producer_rows": focus, "layer02_down_index62": weighted,
        "earliest_observed_operator_drift": earliest,
        "earliest_responsible_operator": "UNRESOLVED: first observed rounding is not unique causal responsibility",
        "remaining_missing_boundary": "All immediate operator/operand terms are measured. The ordered S16 gate/up "
        "effects and S13 input-trajectory error are not yet propagated backward as weighted effects through the "
        "coupled RMSNorm and prior residual/layer recurrences; unique earliest responsibility for L2/index62 "
        "cannot be inferred from unweighted upstream coordinate deltas.",
        "failure_taxonomy": "global_numerical",
        "root_cause_hypothesis": "Measured down-projection same-input and weighted S16 operand terms separate "
        "local rounding from inherited drift; no faulty operator or causal repair is established.",
        "regression": "All producer and weighted-input telescopes close exactly; original L2/P0/S18/index62 "
        "616e with excess 1/2 remains FAIL against 1/8.",
        "state_and_kv_identity": "PASS", "artifacts": artifacts, "phases": phases,
        "producer_seconds": time.monotonic() - started,
        "original_global_layer_invocations": 0, "model_operator_invocations": progress["model_operator_invocations"],
        "model_operator_invocation_unit": "frozen binary64 single-operator calls, including S16 ordered hybrids",
        "local_oracle_invocations": progress["local_oracle_invocations"]}


def weighted_sum(weights, values):
    cone_values(weights, len(values))
    return sum((Fraction(float(w)) * Fraction(v) for w, v in zip(weights, values, strict=True)),
               Fraction(0))


def exact_delta(actual, original):
    cone_values(actual, len(original))
    cone_values(original, len(actual))
    return [Fraction(float(a)) - Fraction(float(g)) for a, g in zip(actual, original, strict=True)]


def finite_secant(effects, changes):
    require(len(effects) == len(changes), "secant coverage mismatch")
    slopes = []
    for effect, change in zip(effects, changes, strict=True):
        require(change != 0 or effect == 0, "nonzero secant effect with unchanged operand")
        slopes.append(float(effect / change) if change else 0.0)
    return cone_values(np.asarray(slopes, dtype="<f8"), len(slopes))


def rms_backward(weights, actual, original, gamma):
    """Finite-difference identity, including the all-coordinate radius coupling."""
    count = len(weights)
    for value in (weights, actual, original, gamma):
        cone_values(value, count)
    ra = math.sqrt(float(np.mean(actual * actual)) + 1e-6)
    rg = math.sqrt(float(np.mean(original * original)) + 1e-6)
    direct = weights * gamma / ra
    coupling = (-float(np.sum(weights * gamma * original)) / (ra * rg * (ra + rg))
                * (actual + original) / count)
    return cone_values(direct, count), cone_values(coupling, count)


def weighted_backward(out, namespace, tensors, data, captured, references, root, embedding,
                      prior, producer_rows, saved, diagnosis):
    """Transport the reviewed S17 scalar, not a re-anchored global reference."""
    rows, artifacts, summaries, phases, rms_terms = [], [], [], [], []
    parents = {"actual": root, "canonical": root}
    originals = {}
    previous = embedding.view("<f2").astype("<f8")
    retained.verify_parent(root, root, embedding=embedding)
    for layer in range(3):
        originals[layer] = cone_original(captured[layer], previous)
        verify_global_endpoint(originals[layer], references[layer])
        previous = originals[layer]["stage18"]
        for name in parents:
            arrays = data[layer][name]
            retained.verify_parent(retained.state_from(arrays, "input", "input_hidden"), parents[name])
            for stage, count in local.SIZES.items():
                local.finite_words(arrays[f"stage{stage:02d}"], (count,))
            for stage in (6, 7, 12, 18):
                retained.check_stage_state(stage, arrays, parents[name])
            require(np.array_equal(arrays["stage09"], np.full(14, 0x3c00, dtype="<u2")),
                    "weighted P0 singleton attention identity")
            for stage in (12, 18):
                original_residual_rows(arrays, originals[layer], layer, stage, name)
            parents[name] = retained.state_from(arrays, "output", "stage18")

    for name in parents:
        start = len(rows)
        pending = np.zeros(896, dtype="<f8")
        weights_archive = {}

        def leaf(layer, stage, term, weights, values):
            contributions = [Fraction(float(w)) * Fraction(v)
                             for w, v in zip(weights, values, strict=True)]
            total = weighted_sum(weights, values)
            ranked = sorted(enumerate(contributions), key=lambda x: abs(x[1]), reverse=True)[:8]
            rows.append({
                "lineage": name, "layer": layer, "position": 0, "stage": stage, "term": term,
                "coordinates": len(values), "weighted_effect": str(total),
                "absolute_coordinate_sum": str(sum(map(abs, contributions), Fraction(0))),
                "nonzero_coordinates": sum(v != 0 for v in contributions),
                "largest_coordinates": [{"index": i, "effect": str(v)} for i, v in ranked if v],
            })
            return total

        def scalar(layer, stage, term, value):
            return leaf(layer, stage, term, np.ones(1, dtype="<f8"), [value])

        for layer in reversed(range(3)):
            begin, layer_start = time.monotonic(), len(rows)
            arrays, original = data[layer][name], originals[layer]
            same = saved[(layer, name)]
            _, projections = cone_evaluator(namespace, tensors[layer], layer)
            values = {key: local.finite_words(arrays[key], (len(original[key]),))
                      for key in ("input_hidden", *(f"stage{s:02d}" for s in CONE_STAGES), "stage12", "stage18")}
            deltas = {key: exact_delta(value, original[key]) for key, value in values.items()}
            state_deltas = {
                prefix: [Fraction(int(i), 1 << 24) - Fraction(float(g))
                         for i, g in zip(arrays[prefix + "_i"], original[key], strict=True)]
                for prefix, key in (("input", "input_hidden"), ("scratch", "stage12"), ("output", "stage18"))}
            seed = np.zeros(896, dtype="<f8")
            if layer == 2:
                seed[62] = 1.0
            target = weighted_sum(pending, state_deltas["output"]) + weighted_sum(seed, deltas["stage17"])

            def merge(stage, left, right, delta):
                merged = left + right
                scalar(layer, stage, "adjoint_merge_rounding",
                       weighted_sum(left, delta) + weighted_sum(right, delta) - weighted_sum(merged, delta))
                return merged

            def operator(stage, weights, incoming):
                key = f"stage{stage:02d}"
                source = producer_rows[(layer, name, stage)]
                require(len(source) == len(weights), "weighted producer coverage")
                for index, row in enumerate(source):
                    require(int(row["index"]) == index and row["position"] == "0"
                            and row["actual_fp16_bits"] == f"{int(arrays[key][index]):04x}"
                            and row["original_binary64_hex"] == float(original[key][index]).hex()
                            and row["same_input_binary64_hex"] == float(same[key][index]).hex()
                            and sum(Fraction(row[t]) for t in CONE_TERMS) == deltas[key][index],
                            "weighted producer data/telescope drift")
                weights_archive[f"layer{layer:02d}_{key}"] = weights.copy()
                own = sum((leaf(layer, stage, term, weights, [Fraction(r[term]) for r in source])
                           for term in CONE_TERMS[:3]), Fraction(0))
                transported = sum((weighted_sum(w, deltas[k]) for k, w in incoming.items()), Fraction(0))
                scalar(layer, stage, "binary64_operator_and_transport_remainder",
                       weighted_sum(weights, deltas[key]) - own - transported)

            def projection(stage, weights):
                incoming = projections[stage].reference_weight.numpy() @ weights
                operator(stage, weights, {local.OPERANDS[stage][0]: incoming})
                return incoming

            def norm(stage, weights):
                key = local.OPERANDS[stage][0]
                norm_name = "input_layernorm" if stage == 0 else "post_attention_layernorm"
                gamma = tensors[layer][f"model.layers.{layer}.{norm_name}.weight"].astype("<f8")
                direct, coupled = rms_backward(weights, values[key], original[key], gamma)
                incoming = direct + coupled
                weights_archive[f"layer{layer:02d}_stage{stage:02d}_direct"] = direct
                weights_archive[f"layer{layer:02d}_stage{stage:02d}_coupled"] = coupled
                rms_terms.append({
                    "lineage": name, "layer": layer, "stage": stage,
                    "direct_weighted_input_effect": str(weighted_sum(direct, deltas[key])),
                    "coupled_weighted_input_effect": str(weighted_sum(coupled, deltas[key])),
                    "merged_weighted_input_effect": str(weighted_sum(incoming, deltas[key])),
                })
                operator(stage, weights, {key: incoming})
                return incoming

            def residual(stage, weights):
                before = "input_hidden" if stage == 12 else "stage12"
                rounding = [Fraction(float(g)) + Fraction(float(b)) - Fraction(float(y))
                            for g, b, y in zip(original[before], original[f"stage{stage-1:02d}"],
                                               original[f"stage{stage:02d}"], strict=True)]
                leaf(layer, stage, "negative_original_binary64_residual_rounding", weights, rounding)

            def view(key, prefix, weights):
                projection_error = [Fraction(float(h)) - Fraction(int(i), 1 << 24)
                                    for h, i in zip(values[key], arrays[prefix + "_i"], strict=True)]
                weights_archive[f"layer{layer:02d}_{prefix}_view"] = weights.copy()
                leaf(layer, key, "Q24_to_FP16_view_projection", weights, projection_error)

            residual(18, pending)
            down = merge(17, pending, seed, deltas["stage17"])
            activation = projection(17, down)
            hybrid = same["stage16_actual_gate_original_up"]
            gate_effect = exact_delta(hybrid, original["stage16"])
            up_effect = exact_delta(same["stage16"], hybrid)
            gate = activation * finite_secant(gate_effect, deltas["stage14"])
            up = activation * finite_secant(up_effect, deltas["stage15"])
            operator(16, activation, {"stage14": gate, "stage15": up})
            gate_input, up_input = projection(14, gate), projection(15, up)
            norm_input = merge(13, gate_input, up_input, deltas["stage13"])
            scratch_view = norm(13, norm_input)
            view("stage12", "scratch", scratch_view)
            scratch = merge("scratch_i", pending, scratch_view, state_deltas["scratch"])
            residual(12, scratch)
            attention = projection(11, scratch)
            value_weights = np.zeros((2, 64), dtype="<f8")
            np.add.at(value_weights, np.arange(14) // 7, attention.reshape(14, 64))
            value_weights = value_weights.reshape(-1)
            # P0's authenticated S7 is S3; probability one eliminates Q/K dependence.
            operator(10, attention, {"stage03": value_weights})
            input_view = norm(0, projection(3, value_weights))
            view("input_hidden", "input", input_view)
            pending = merge("input_i", scratch, input_view, state_deltas["input"])
            layer_effect = sum((Fraction(r["weighted_effect"]) for r in rows[layer_start:]), Fraction(0))
            require(layer_effect + weighted_sum(pending, state_deltas["input"]) == target,
                    "weighted layer recurrence does not close")
            weights_archive[f"layer{layer:02d}_input_i"] = pending.copy()
            phases.append({"lineage": name, "layer": layer, "phase": "weighted_backward",
                           "seconds": time.monotonic() - begin, "closure_error": "0"})
        require(all(v == 0 for v in state_deltas["input"]), "official root has nonzero state drift")
        expected = next(r for r in prior["layer02_down_index62"] if r["lineage"] == name)
        total = sum((Fraction(r["weighted_effect"]) for r in rows[start:]), Fraction(0))
        require(total == Fraction(expected["global_operator_delta"]), "weighted root telescope does not close")
        own_rows = rows[start:]
        earliest = next((r for r in sorted(own_rows, key=lambda r: (
            r["layer"], r["stage"] if isinstance(r["stage"], int) else 19))
                         if r["term"] in CONE_TERMS[:3] and Fraction(r["absolute_coordinate_sum"])), None)
        remainders = [r for r in own_rows if r["term"] in (
            "adjoint_merge_rounding", "binary64_operator_and_transport_remainder")]
        summaries.append({
            "lineage": name, "layer": 2, "position": 0, "stage": 17, "index": 62,
            "global_operator_delta": str(total), "root_state_delta": "0", "closure_error": "0",
            "earliest_weighted_operator_contribution": earliest,
            "transport_remainder_absolute_sum": str(sum(
                (abs(Fraction(r["weighted_effect"])) for r in remainders), Fraction(0))),
            "largest_signed_contributions": sorted(
                own_rows, key=lambda r: abs(Fraction(r["weighted_effect"])), reverse=True)[:12],
        })
        artifacts.append(retained.save(out / f"{name}_backward_weights.npz", weights_archive))
    witness = preserve_witness(data[2]["reports"][18], diagnosis)
    require(witness == prior["preserved_rejecting_witness"], "reviewed backward witness changed")
    retained.write(out / "weighted_terms.json", rows)
    artifacts.append(retained.record(out / "weighted_terms.json"))
    return {
        "status": "DIAGNOSTIC_COMPLETE", "numerical_status": "FAIL", "preserved_rejecting_witness": witness,
        "weighted_backward": summaries, "rmsnorm_coupling": rms_terms, "phases": phases,
        "weighted_term_rows": len(rows), "weighted_operator_coordinates": sum(
            r["coordinates"] for r in rows if r["term"] == CONE_TERMS[0]),
        "state_and_kv_identity": "PASS", "operator_telescope_closure_errors": 0,
        "original_global_layer_invocations": 0, "model_operator_invocations": 0,
        "local_oracle_invocations": 0, "artifacts": artifacts,
        "earliest_responsible_operator": "No unique faulty operator established; earliest weighted contributions "
        "are reported separately for each lineage, under the frozen finite-difference path.",
        "remaining_missing_boundary": "The numerical telescope reaches the official root with zero state drift. "
        "Finite binary64 operator/coefficient/reduction remainders remain explicitly quantified, not assigned to "
        "upstream operators. Unique causal responsibility or a repair requires distinguishing these "
        "path-dependent rounding contributions with an independently justified intervention; this is not a "
        "missing S13/RMSNorm/residual capture and not a candidate-admission result.",
        "failure_taxonomy": "global_numerical",
        "root_cause_hypothesis": "Weighted local rounding and inherited trajectory effects coexist; "
        "the decomposition alone does not establish a defective operator or a sufficient repair.",
        "regression": "Exact weighted closure through both L0-L2 residual histories to the official root; "
        "unchanged L2/P0/S18/index62 616e excess 1/2 remains FAIL against 1/8.",
    }


def capture_global_layer(namespace, tensors, layer, hidden):
    """Observe unchanged frozen functions; never substitute their return values."""
    import torch

    require(type(layer) is int and layer in range(3), "global capture scope is L0-L2")
    require(hidden.dtype == torch.float64 and tuple(hidden.shape) == (1, 896)
            and bool(torch.isfinite(hidden).all()), "invalid original hidden")
    projections = {}
    for stage, suffix in local.PROJECTIONS.items():
        prefix = f"model.layers.{layer}.{suffix}"
        q = namespace["_unpack_words"](tensors[prefix + ".qweight"])
        z = namespace["_unpack_words"](tensors[prefix + ".qzeros"])
        scales = tensors[prefix + ".scales"].astype("<f8")
        weight = (q - np.repeat(z, 128, axis=0)).astype("<f8")
        weight *= np.repeat(scales, 128, axis=0)
        name = {1: "q", 2: "k", 3: "v", 11: "o", 14: "gate", 15: "up", 17: "down"}[stage]
        projections[name] = SimpleNamespace(
            reference_weight=torch.from_numpy(weight),
            reference_bias=torch.from_numpy(tensors[prefix + ".bias"].astype("<f8"))
            if stage in (1, 2, 3) else None)
    state = SimpleNamespace(
        input_norm=tensors[f"model.layers.{layer}.input_layernorm.weight"],
        post_attention_norm=tensors[f"model.layers.{layer}.post_attention_layernorm.weight"],
        projections=projections,
        reference_k=torch.empty((0, 2, 64), dtype=torch.float64),
        reference_v=torch.empty((0, 2, 64), dtype=torch.float64))
    captured = {"input_hidden": hidden.numpy().reshape(-1).copy()}
    projection, norm = namespace["_reference_projection"], namespace["_torch_rmsnorm"]
    calls = []

    def array(value):
        require(value.dtype == torch.float64 and bool(torch.isfinite(value).all()),
                "nonfinite/nonbinary64 original operand")
        return value.detach().numpy().reshape(-1).copy()

    def capture_projection(activation, operator):
        name = next(k for k, v in projections.items() if v is operator)
        stage = {"q": 1, "k": 2, "v": 3, "o": 11, "gate": 14, "up": 15, "down": 17}[name]
        calls.append(stage)
        value = projection(activation, operator)
        captured[f"stage{stage:02d}_input"] = array(activation)
        captured[f"stage{stage:02d}"] = array(value)
        return value

    def capture_norm(activation, weight):
        require(weight is state.input_norm or weight is state.post_attention_norm,
                "wrong original norm selection")
        stage = 0 if weight is state.input_norm else 13
        calls.append(stage)
        value = norm(activation, weight)
        captured[f"stage{stage:02d}"] = array(value)
        if stage == 13:
            captured["stage12"] = array(activation)
        return value

    namespace["_reference_projection"], namespace["_torch_rmsnorm"] = capture_projection, capture_norm
    try:
        with torch.no_grad():
            output = namespace["_reference_layer_step"](state, hidden, 0)
    finally:
        namespace["_reference_projection"], namespace["_torch_rmsnorm"] = projection, norm
    require(calls == [0, 1, 2, 3, 11, 13, 14, 15, 17], "original operator order changed")
    captured["stage18"] = array(output)
    captured["stage10"], captured["stage16"] = captured["stage11_input"], captured["stage17_input"]
    for name, source in (("k", "stage02"), ("v", "stage03")):
        cache = getattr(state, "reference_" + name)
        require(tuple(cache.shape) == (1, 2, 64), "original P0 cache geometry")
        captured["output_cache_" + name] = array(cache)
        require(np.array_equal(captured["output_cache_" + name].view("<u8"),
                               captured[source].view("<u8")), "original P0 cache identity")
    return output, captured


def verify_global_endpoint(captured, expected):
    for value in (captured["stage18"], expected):
        require(value.dtype == np.dtype("<f8") and value.shape == (896,)
                and np.all(np.isfinite(value)), "invalid original endpoint")
    require(np.array_equal(captured["stage18"].view("<u8"), expected.view("<u8")),
            "original binary64 endpoint drift; captured operands are not admissible evidence")


def original_residual_rows(arrays, original, layer, stage, lineage):
    require(type(layer) is int and layer in range(3) and stage in (12, 18),
            "original decomposition scope is L0-L2 S12/S18")
    before, after = ("input", "scratch") if stage == 12 else ("scratch", "output")
    incoming_key = "input_hidden" if stage == 12 else "stage12"
    branch_key, output_key = f"stage{stage - 1:02d}", f"stage{stage:02d}"
    parent = retained.state_from(arrays, "input", "input_hidden")
    retained.verify_parent(parent, parent)
    retained.check_stage_state(stage, arrays, parent)
    for key in (incoming_key, branch_key, output_key):
        value = original[key]
        require(value.dtype == np.dtype("<f8") and value.shape == (896,)
                and np.all(np.isfinite(value)), f"invalid original operand {key}")
    require(np.array_equal((original[incoming_key] + original[branch_key]).view("<u8"),
                           original[output_key].view("<u8")), "original residual identity mismatch")
    rows = []
    for index in range(896):
        prior, branch, final = [Fraction.from_float(float(original[k][index]))
                                for k in (incoming_key, branch_key, output_key)]
        incoming = Fraction(int(arrays[before + "_i"][index]), 1 << 24) - prior
        operator = rational.fp16_value(int(arrays[branch_key][index])) - branch
        outgoing = Fraction(int(arrays[after + "_i"][index]), 1 << 24)
        view = rational.fp16_value(int(arrays[output_key][index]))
        rounding = final - prior - branch
        projection = view - outgoing
        require(incoming + operator - rounding == outgoing - final
                and incoming + operator - rounding + projection == view - final,
                "original residual telescope does not close")
        rows.append({
            "layer": layer, "position": 0, "stage": stage, "index": index, "lineage": lineage,
            "comparison": "q24_minus_original_global_binary64",
            "original_incoming_binary64_hex": float(original[incoming_key][index]).hex(),
            "original_operator_binary64_hex": float(original[branch_key][index]).hex(),
            "original_outgoing_binary64_hex": float(original[output_key][index]).hex(),
            "incoming_i": int(arrays[before + "_i"][index]),
            "incoming_z": int(arrays[before + "_z"][index]),
            "operator_fp16_bits": f"{int(arrays[branch_key][index]):04x}",
            "outgoing_i": int(arrays[after + "_i"][index]),
            "outgoing_z": int(arrays[after + "_z"][index]),
            "outgoing_fp16_bits": f"{int(arrays[output_key][index]):04x}",
            "incoming_delta": str(incoming), "operator_delta": str(operator),
            "original_residual_rounding": str(rounding),
            "state_delta": str(outgoing - final), "projection_error": str(projection),
            "global_error": str(view - final), "classification": drift_class(incoming, operator),
            "closure_error": "0",
        })
    return rows


def root_state(embedding):
    local.finite_words(embedding, (896,))
    rows = [rational.root(int(word)) for word in embedding]
    return {"i": np.asarray([r[0] for r in rows], dtype="<i8"),
            "z": np.asarray([r[1] for r in rows], dtype="u1"),
            "h": embedding.copy()}


def canonical_layer(tensors, layer, parent):
    require(type(layer) is int and 0 <= layer <= 2, "control scope is L0-L2 only")
    retained.verify_parent(parent, parent)
    arrays = {"input_hidden": parent["h"].copy(), "input_i": parent["i"].copy(),
              "input_z": parent["z"].copy(),
              "input_cache_k": np.empty((0, 128), dtype="<u2"),
              "input_cache_v": np.empty((0, 128), dtype="<u2")}
    for stage in range(19):
        if stage in (12, 18):
            state = retained.transition_reference(
                parent if stage == 12 else scratch,
                arrays["stage11" if stage == 12 else "stage17"])
            prefix = "scratch" if stage == 12 else "output"
            arrays[prefix + "_i"], arrays[prefix + "_z"] = state["i"], state["z"]
            word = state["h"]
            if stage == 12:
                scratch = state
        else:
            word = local.local_reference(
                stage, {k: arrays[k] for k in local.OPERANDS[stage]}, tensors, layer)
        arrays[f"stage{stage:02d}"] = word
        if stage in (6, 7):
            arrays["output_cache_" + ("k" if stage == 6 else "v")] = word.reshape(1, 128).copy()
        retained.check_stage_state(stage, arrays, parent)
    return arrays


def rounding_control(values, mode):
    """General FP16 rounding controls, independent of any witness or weights."""
    require(mode in ROUNDING_CONTROLS, "unknown rounding control")
    require(isinstance(values, np.ndarray) and values.dtype == np.dtype("<f8")
            and np.all(np.isfinite(values)) and np.all(np.abs(values) <= 65504),
            "invalid finite FP16-range control operand")
    with np.errstate(over="raise", invalid="raise"):
        rounded = values.astype("<f2")
        if mode != "rne":
            adjust = (np.abs(rounded.astype("<f8")) > np.abs(values) if mode == "toward_zero"
                      else np.abs(rounded.astype("<f8")) < np.abs(values))
            target = (np.copysign(np.zeros_like(rounded), values) if mode == "toward_zero"
                      else np.copysign(np.full_like(rounded, np.inf), values))
            rounded[adjust] = np.nextafter(rounded[adjust], target[adjust], dtype=np.float16)
    words = rounded.view("<u2")
    local.finite_words(words, values.shape)
    return words


def repair_sufficiency(out, namespace, tensors, data, references, trajectories, root,
                       embedding, diagnosis, result):
    witness = preserve_witness(data[2]["reports"][18], diagnosis)
    retained.verify_parent(root, root, embedding=embedding)
    for lineage in ("actual", "canonical"):
        previous = root
        for layer in range(3):
            arrays = data[layer][lineage]
            retained.verify_parent(retained.state_from(arrays, "input", "input_hidden"), previous)
            for stage in range(19):
                retained.check_stage_state(stage, arrays, previous)
            previous = retained.state_from(arrays, "output", "stage18")

    arms, phases = [], []
    result.update(model_operator_invocations=0, local_oracle_invocations=0,
                  original_global_layer_invocations=0, counterfactual_layer_invocations=0)
    for mode in ROUNDING_CONTROLS:
        parent, layers = root, []
        arm = {"rounding": mode, "status": "PASS", "layers": layers,
               "witness": None, "witness_disposition": "NOT_REACHED",
               "failure_taxonomy": None, "candidate_admitted": False}
        for layer in range(3):
            begin = time.monotonic()
            directory = out / mode / f"layer{layer:02d}"
            directory.mkdir(parents=True)
            arrays = {"input_hidden": parent["h"].copy(), "input_i": parent["i"].copy(),
                      "input_z": parent["z"].copy(),
                      "input_cache_k": np.empty((0, 128), dtype="<u2"),
                      "input_cache_v": np.empty((0, 128), dtype="<u2")}
            evaluate, _ = cone_evaluator(namespace, tensors[layer], layer)
            result["counterfactual_layer_invocations"] += 1
            reports, s16_operands = [], {}
            for stage in range(19):
                key = f"stage{stage:02d}"
                reused = layer == 0 and stage < 16
                if reused:
                    word = data[0]["canonical"][key].copy()
                    if stage == 12:
                        for suffix in ("i", "z"):
                            arrays["scratch_" + suffix] = data[0]["canonical"]["scratch_" + suffix].copy()
                elif stage in (12, 18):
                    before = parent if stage == 12 else retained.state_from(arrays, "scratch", "stage12")
                    state = retained.transition_reference(before, arrays[f"stage{stage - 1:02d}"])
                    prefix = "scratch" if stage == 12 else "output"
                    arrays[prefix + "_i"], arrays[prefix + "_z"] = state["i"], state["z"]
                    word = state["h"]
                elif stage == 16:
                    operands = {k: arrays[k].view("<f2").astype("<f8") for k in local.OPERANDS[16]}
                    unrounded = evaluate(16, operands)
                    result["model_operator_invocations"] += 1
                    word = rounding_control(unrounded, mode)
                    s16_operands = {**operands, "unrounded_binary64": unrounded, "output_fp16": word}
                else:
                    word = local.local_reference(stage, {k: arrays[k] for k in local.OPERANDS[stage]},
                                                 tensors[layer], layer)
                    result["model_operator_invocations"] += 1
                arrays[key] = word
                if stage in (6, 7):
                    arrays["output_cache_" + ("k" if stage == 6 else "v")] = word.reshape(1, 128).copy()
                state_expected = retained.check_stage_state(stage, arrays, parent)
                expected = None
                if stage < 18:
                    if stage == 16:
                        expected = local.local_reference(
                            stage, {k: arrays[k] for k in local.OPERANDS[stage]}, tensors[layer], layer)
                        result["local_oracle_invocations"] += 1
                    else:
                        expected = state_expected if stage == 12 else word
                report = gates.evaluate_decoder_stage(
                    stage=stage, actual=word, reference=trajectories[layer][stage],
                    policy=gates.POLICY_ID, local_reference=expected,
                    reference_binary64=references[layer] if stage == 18 else None)
                report.update(
                    node=[layer, 0, stage], reused_reviewed_prefix=reused,
                    local_comparison_boundary="independent_numpy_vs_frozen_torch_s16" if stage == 16
                    else "original_input_global_binary64" if stage == 18
                    else "rational_state_identity" if stage == 12
                    else "canonical_reference_identity_not_independent_validation")
                reports.append(report)
                if layer == 2 and stage == 18:
                    arm["witness"] = report["binary64_v1"]["rows"][62]
                    arm["witness_disposition"] = "PASS" if arm["witness"]["accepted"] else "FAIL"
                if report["status"] != "PASS":
                    gate = report.get("binary64_v1", report.get("local_operator_fp16", {}))
                    arm.update(
                        status=report["status"], first_failure=report["node"],
                        failure_count=gate.get("failure_count"),
                        failure_taxonomy="global_numerical" if stage == 18 else "local_operator_numerical",
                        root_cause_hypothesis="Uniform S16 rounding and its coherently propagated canonical "
                        "suffix do not satisfy this first unchanged mandatory gate; weighted contributions "
                        "alone do not predict the nonlinear, quantized intervention.",
                        regression=f"{mode} first failure L{layer}/P0/S{stage}; preserve full gate report")
                    break
            archive = retained.save(directory / "stages.npz", arrays)
            operand_record = retained.save(directory / "s16_operands.npz", s16_operands) if s16_operands else None
            retained.write(directory / "comparisons.json", reports)
            layers.append({"layer": layer, "stages": archive, "s16_operands": operand_record,
                           "comparisons": retained.record(directory / "comparisons.json"),
                           "checked_stage_vectors": len(reports),
                           "checked_coordinates": sum(local.SIZES[r["stage"]] for r in reports),
                           "state_and_kv_identity": "PASS"})
            phases.append({"phase": "rounding_control", "rounding": mode, "layer": layer,
                           "seconds": time.monotonic() - begin})
            print(json.dumps({"rounding": mode, "layer": layer, "status": arm["status"],
                              "last_stage": reports[-1]["stage"]}), flush=True)
            if arm["status"] != "PASS":
                break
            parent = retained.state_from(arrays, "output", "stage18")
        arms.append(arm)
    survivors = [arm["rounding"] for arm in arms if arm["status"] == "PASS"]
    return {
        "status": "DIAGNOSTIC_COMPLETE", "numerical_status": "COUNTERFACTUAL_PASS" if survivors else "FAIL",
        "evidence_kind": "bounded_cpu_s16_rounding_repair_sufficiency_screen",
        "arms": arms, "surviving_counterfactuals": survivors, "phases": phases,
        "preserved_rejecting_witness": witness,
        "remaining_missing_boundary": (
            "A passing canonical-software counterfactual needs a separately implemented source-compatible "
            "candidate and independent all-operator verification; identity comparisons are not admission."
            if survivors else
            "None of the three uniform S16 FP16-rounding controls clears the unchanged gates. "
            "The reviewed S16 weighted rounding contribution is not a sufficient repair prescription; "
            "the retained inherited gate/up and RMSNorm drift remains uncorrected. This does not "
            "exclude other coherent general interventions in L0-L2."),
        "failure_taxonomy": None if survivors else "repair_sufficiency_not_established",
        "root_cause_hypothesis": "Weighted rounding attribution does not specify a realizable gate-preserving repair.",
        "regression": "all coordinates before each first failure, separate control states, immutable original witness",
        "claim_boundary": "Isolated Q24 software counterfactuals only; canonical identity checks are not "
        "independent candidate verification. No strict-FP16-state W4A16, RTL, admission, tokens or full model.",
    }


def compare(left, right):
    require(left.shape == right.shape and left.dtype == right.dtype == np.dtype("<u2"),
            "comparison shape/dtype mismatch")
    numerical = gates.legacy.compare_fp16(left, right)
    indices = np.flatnonzero(left != right)
    return {"coordinates": left.size, "bit_differences": int(indices.size),
            "first_bit_difference": int(indices[0]) if indices.size else None,
            "fp16_gate": numerical}


def preserve_witness(report, diagnosis):
    rows = report["binary64_v1"]["rows"]
    row = rows[62]
    require(all(row[key] == value for key, value in WITNESS.items()),
            "retained rejecting witness changed")
    require(not row["accepted"] and report["binary64_v1"]["failure_count"] == 1,
            "retained rejection/coverage changed")
    require(all(diagnosis["first_failure"][key] == value for key, value in WITNESS.items()),
            "validation002 witness differs")
    return row


def drift_class(incoming, operator):
    if incoming and operator:
        return "both"
    if incoming:
        return "incoming_only"
    return "operator_only" if operator else "neither"


def residual_rows(actual, canonical, same_input, layer, stage):
    require(type(layer) is int and layer in range(3) and stage in (12, 18),
            "decomposition scope is L0-L2 S12/S18 only")
    before, after = ("input", "scratch") if stage == 12 else ("scratch", "output")
    hidden = "input_hidden" if stage == 12 else "stage12"
    branch = f"stage{stage - 1:02d}"
    local.finite_words(same_input[branch], (896,))
    for arrays in (actual, canonical):
        parent = retained.state_from(arrays, "input", "input_hidden")
        retained.verify_parent(parent, parent)
        retained.check_stage_state(stage, arrays, parent)
        local.finite_words(arrays[branch], (896,))
    rows = []
    for index in range(896):
        values = {}
        for name, arrays in (("actual", actual), ("canonical", canonical)):
            incoming = Fraction(int(arrays[before + "_i"][index]), 1 << 24)
            outgoing = Fraction(int(arrays[after + "_i"][index]), 1 << 24)
            word = int(arrays[branch][index])
            h = int(arrays[f"stage{stage:02d}"][index])
            values[name] = (incoming, rational.fp16_value(word), outgoing,
                            rational.fp16_value(h) - outgoing)
        a, c = values["actual"], values["canonical"]
        incoming, operator = a[0] - c[0], a[1] - c[1]
        local_value = rational.fp16_value(int(same_input[branch][index]))
        local_error, operand_drift = a[1] - local_value, local_value - c[1]
        state_delta, projection_delta = a[2] - c[2], a[3] - c[3]
        view_delta = rational.fp16_value(int(actual[f"stage{stage:02d}"][index])) - \
            rational.fp16_value(int(canonical[f"stage{stage:02d}"][index]))
        require(incoming + operator == state_delta
                and local_error + operand_drift == operator
                and state_delta + projection_delta == view_delta,
                "exact residual decomposition does not close")
        row = {
            "layer": layer, "position": 0, "stage": stage, "index": index,
            "comparison": "retained_minus_reviewed_canonical_q24",
            "incoming_delta": str(incoming), "operator_delta": str(operator),
            "operator_same_input_arithmetic_delta": str(local_error),
            "operator_operand_trajectory_delta": str(operand_drift),
            "state_delta": str(state_delta), "projection_delta": str(projection_delta),
            "view_delta": str(view_delta), "classification": drift_class(incoming, operator),
            "state_cancellation": bool(incoming and operator and not state_delta),
            "closure_error": "0",
            "same_input_operator_fp16_bits": f"{int(same_input[branch][index]):04x}",
        }
        for name, arrays in (("actual", actual), ("canonical", canonical)):
            for prefix, stored in (("incoming", before), ("outgoing", after)):
                row[f"{name}_{prefix}_i"] = int(arrays[stored + "_i"][index])
                row[f"{name}_{prefix}_z"] = int(arrays[stored + "_z"][index])
            row[f"{name}_incoming_fp16_bits"] = f"{int(arrays[hidden][index]):04x}"
            row[f"{name}_operator_fp16_bits"] = f"{int(arrays[branch][index]):04x}"
            row[f"{name}_outgoing_fp16_bits"] = f"{int(arrays[f'stage{stage:02d}'][index]):04x}"
        rows.append(row)
    return rows


def global_rows(arrays, previous, reference, layer, lineage):
    require(type(layer) is int and layer in range(3), "global split scope is L0-L2")
    for values in (previous, reference):
        require(values.dtype == np.dtype("<f8") and values.shape == (896,)
                and np.all(np.isfinite(values)) and np.all(np.abs(values) <= 65504),
                "invalid original binary64 reference")
    rows = []
    for index in range(896):
        prior, target = (Fraction.from_float(float(x[index])) for x in (previous, reference))
        incoming = Fraction(int(arrays["input_i"][index]), 1 << 24) - prior
        increment = sum((rational.fp16_value(int(arrays[f"stage{s:02d}"][index]))
                         for s in (11, 17)), Fraction())
        output = Fraction(int(arrays["output_i"][index]), 1 << 24)
        view = rational.fp16_value(int(arrays["stage18"][index]))
        operator = increment - (target - prior)
        projection = view - output
        require(incoming + operator + projection == view - target,
                "exact global decomposition does not close")
        rows.append({
            "layer": layer, "position": 0, "stage": 18, "index": index,
            "lineage": lineage, "comparison": "original_input_global_binary64",
            "original_parent_binary64_hex": float(previous[index]).hex(),
            "original_output_binary64_hex": float(reference[index]).hex(),
            "incoming_delta": str(incoming), "net_operator_delta": str(operator),
            "projection_error": str(projection), "global_error": str(view - target),
            "classification": drift_class(incoming, operator), "closure_error": "0",
            "net_operator_scope": "S11+S17 minus original endpoint increment; includes binary64 residual rounding",
        })
    return rows


def decomposition_totals(rows, terms):
    return {
        "coordinates": len(rows), "classifications": dict(Counter(r["classification"] for r in rows)),
        "closure_failures": sum(r["closure_error"] != "0" for r in rows),
        "terms": {
            term: {"signed_sum": str(sum((Fraction(r[term]) for r in rows), Fraction())),
                   "absolute_sum": str(sum((abs(Fraction(r[term])) for r in rows), Fraction())),
                   "maximum_absolute": str(max(abs(Fraction(r[term])) for r in rows)),
                   "nonzero_coordinates": sum(Fraction(r[term]) != 0 for r in rows)}
            for term in terms},
    }


def decompose(out, command, *, original_global=False, producer_cones_only=False, weighted_backward_only=False,
              repair_sufficiency_only=False):
    require(sum((original_global, producer_cones_only, weighted_backward_only, repair_sufficiency_only)) <= 1,
            "conflicting diagnostic modes")
    require(out.parent == ROOT / "build" and out.name.startswith("q24_software_root_"),
            "decomposition output must be an isolated software-root build path")
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    result = {
        "status": "BLOCKED", "evidence_kind": "retained_cpu_exact_residual_decomposition",
        "rtl_invocations": 0, "l9_invocations": 0, "candidate_invocations": 0,
        "model_operator_invocations": 0, "candidate_admitted": False, "policy_adopted": False,
        "normal_host_review": "REQUIRED", "policy_id": gates.POLICY_ID,
        "claim_boundary": "L0-L2/P0 token9707 diagnostic; no candidate, RTL, tokens or full-model admission",
    }
    bindings = {}

    def bind(rec):
        signature = {k: rec[k] for k in ("path", "bytes", "sha256")}
        if signature["path"] not in bindings:
            require(retained.record(signature["path"]) == signature,
                    f"binding drift: {signature['path']}")
            bindings[signature["path"]] = signature
        require(bindings[signature["path"]] == signature, "conflicting retained binding")
        return Path(signature["path"])

    def read(rec):
        return json.loads(bind(rec).read_text())

    def load(rec):
        with np.load(bind(rec), allow_pickle=False) as archive:
            return {k: archive[k].copy() for k in archive.files}

    phase = "authentication"
    try:
        review = read(retained.record(CONTROL_REVIEW))
        require(review["kind"] == "round_reviewed_handoff"
                and review["producer_role"] == "reviewer"
                and review["mission_id"] == "1783b114f9d3"
                and review["review"]["status"] == "done", "missing genuine canonical control review")
        control = read(retained.record(CONTROL_INPUT / "result.json"))
        freeze = read(control["bindings"])
        require(control["status"] == "FAIL" and freeze["layers"] == [0, 1, 2]
                and freeze["history"] == [9707] and freeze["position"] == 0
                and control["policy_id"] == gates.POLICY_ID
                and not control["candidate_admitted"], "reviewed control scope/policy changed")
        changed = {str(Path(__file__).resolve()),
                   str(Path(__file__).parents[1] / "tests/test_q24_software_canonical_control_v1.py")}
        snapshots = {s["original"]["path"]: s for s in freeze["sources"]}
        for rec in freeze["input_bindings"]:
            if rec["path"] in changed:
                saved = snapshots[rec["path"]]
                require(saved["original"] == rec
                        and saved["snapshot"]["sha256"] == rec["sha256"],
                        "missing immutable prior source")
                bind(saved["snapshot"])
            else:
                bind(rec)
        for saved in snapshots.values():
            require(saved["snapshot"]["sha256"] == saved["original"]["sha256"],
                    "prior source snapshot identity mismatch")
            bind(saved["snapshot"])
        original = read(bindings[str(INPUT / "result.json")])
        original_freeze = read(bindings[str(INPUT / "freeze.json")])
        diagnosis = read(bindings[str(VALIDATION / "failure_diagnosis.json")])
        require(original["status"] == diagnosis["status"] == "FAIL"
                and original["first_failure"]["node"] == [2, 0, 18]
                and original_freeze["contract"]["state_id"] == retained.STATE_ID
                and original_freeze["contract"]["policy_id"] == gates.POLICY_ID,
                "retained rejecting lineage changed")
        require([x["layer"] for x in original["layers"]] ==
                [x["layer"] for x in control["layers"]] == [0, 1, 2], "layer order changed")
        data, references, trajectories = {}, {}, {}
        for layer, (entry, canonical) in enumerate(zip(original["layers"], control["layers"], strict=True)):
            require(entry["position"] == 0, "retained position mismatch")
            reports = read(entry["reports"])
            require([r["node"] for r in reports] == [[layer, 0, s] for s in range(19)],
                    "retained report stage coverage mismatch")
            data[layer] = {
                "actual": load(entry["actual_stages"]), "canonical": load(canonical["actual_stages"]),
                "local": load(bindings[str(INPUT / f"layer{layer:02d}/local_references.npz")]),
                "reports": reports, "control_reports": read(canonical["comparisons"]),
            }
        global_control = read(freeze["global_reference_lineage"]["control"])
        require(freeze["global_reference_lineage"]["control"]["sha256"] == retained.CONTROL_SHA,
                "original global trajectory identity changed")
        for case in global_control["cases"]:
            if case["position"] == 0 and case["layer"] in range(3):
                require(case["layer"] not in references, "duplicate original global layer")
                references[case["layer"]] = np.load(bind(case["binary64_reference_array"]), allow_pickle=False)
        require(set(references) == set(range(3)), "missing original global reference")
        model = read(original_freeze["global_reference_lineage"]["model_freeze"])
        for transaction in model["reference_transactions"]:
            layer = transaction["layer"]
            if transaction["position"] == 0 and layer in range(3):
                require(layer not in trajectories, "duplicate original FP16 layer")
                trajectories[layer] = ({
                    int(s): retained.read_words(bind(rec), local.SIZES[int(s)])
                    for s, rec in transaction["stages"].items()} if repair_sufficiency_only
                    else retained.read_words(bind(transaction["stages"]["18"]), 896))
                if repair_sufficiency_only:
                    require(set(trajectories[layer]) == set(range(19)), "missing original trajectory stage")
        require(set(trajectories) == set(range(3)), "missing original FP16 trajectory")
        root = load(retained.record(CONTROL_INPUT / "root_state.npz"))
        embedding = retained.read_words(bind(original_freeze["state_lineage"]["root"]["input"]), 896, indexed=True)
        global_namespace, global_metadata = None, None
        if original_global or producer_cones_only or weighted_backward_only or repair_sufficiency_only:
            import torch
            import torch.nn.functional as functional
            from ace3.model.candidates import remaining_layers_v3 as reference_loader

            prior_dir = ROOT / "build/q24_software_root_b98eee0a2e34_attempt003"
            prior_review = read(retained.record(retained.HANDOFFS / "b98eee0a2e34/round-0001.json"))
            require(prior_review["producer_role"] == "reviewer"
                    and prior_review["mission_id"] == "b98eee0a2e34"
                    and prior_review["review"]["status"] == "done", "missing prior decomposition review")
            prior_result = read(retained.record(prior_dir / "result.json"))
            require(prior_result["status"] == "DIAGNOSTIC_COMPLETE"
                    and prior_result["numerical_status"] == "FAIL", "prior diagnostic disposition changed")
            if producer_cones_only or weighted_backward_only or repair_sufficiency_only:
                capture_review = read(retained.record(CAPTURE_REVIEW))
                require(capture_review["kind"] == "round_reviewed_handoff"
                        and capture_review["producer_role"] == "reviewer"
                        and capture_review["mission_id"] == "e6b4f7dd0144"
                        and capture_review["review"]["status"] == "done", "missing genuine capture review")
                prior_result = read(retained.record(CAPTURE_INPUT / "result.json"))
                prior_freeze = read(prior_result["bindings"])
                require(prior_result["status"] == "DIAGNOSTIC_COMPLETE"
                        and prior_result["numerical_status"] == "FAIL"
                        and prior_result["original_endpoint_bit_mismatches"] == 0
                        and prior_result["original_endpoint_coordinates"] == 2688
                        and prior_result["policy_id"] == gates.POLICY_ID
                        and not prior_result["candidate_admitted"] and not prior_result["policy_adopted"],
                        "reviewed capture disposition changed")
                prior_sources = {s["original"]["path"]: s for s in prior_freeze["sources"]}
                for rec in prior_freeze["input_bindings"]:
                    if rec["path"] in changed:
                        require(prior_sources[rec["path"]]["original"] == rec,
                                "wrong prior changed-source binding")
                        bind(prior_sources[rec["path"]]["snapshot"])
                    else:
                        bind(rec)
                for saved in prior_sources.values():
                    require(saved["snapshot"]["sha256"] == saved["original"]["sha256"],
                            "prior capture source snapshot drift")
                    bind(saved["snapshot"])
                for rec in prior_freeze["original_global_capture"]["helper_sources"]:
                    if rec["path"] in changed:
                        require(prior_sources[rec["path"]]["original"] == rec,
                                "wrong changed-source snapshot")
                        bind(prior_sources[rec["path"]]["snapshot"])
                    else:
                        bind(rec)
                require(len(prior_result["original_global_operands"]) == 3, "capture layer coverage")
                original_captures = {}
                for layer, rec in enumerate(prior_result["original_global_operands"]):
                    require(Path(rec["path"]).name == f"original_global_layer{layer:02d}.npz",
                            "capture layer selection")
                    original_captures[layer] = load(rec)
            if weighted_backward_only or repair_sufficiency_only:
                producer_review = read(retained.record(PRODUCER_REVIEW))
                require(producer_review["kind"] == "round_reviewed_handoff"
                        and producer_review["producer_role"] == "reviewer"
                        and producer_review["mission_id"] == "a33b39ff3899"
                        and producer_review["review"]["status"] == "done", "missing genuine producer review")
                producer_result = read(retained.record(PRODUCER_INPUT / "result.json"))
                producer_freeze = read(producer_result["bindings"])
                require(producer_result["status"] == "DIAGNOSTIC_COMPLETE"
                        and producer_result["numerical_status"] == "FAIL"
                        and producer_result["policy_id"] == gates.POLICY_ID
                        and not producer_result["candidate_admitted"] and not producer_result["policy_adopted"]
                        and producer_result["producer_coordinates"] == 115200
                        and producer_result["original_operator_bit_mismatches"] == 0,
                        "reviewed producer disposition changed")
                old_sources = {s["original"]["path"]: s for s in producer_freeze["sources"]}
                for rec in producer_freeze["input_bindings"]:
                    if rec["path"] in changed:
                        require(old_sources[rec["path"]]["original"] == rec
                                and old_sources[rec["path"]]["snapshot"]["sha256"] == rec["sha256"],
                                "producer changed-source snapshot drift")
                        bind(old_sources[rec["path"]]["snapshot"])
                    else:
                        bind(rec)
                for source in old_sources.values():
                    require(source["snapshot"]["sha256"] == source["original"]["sha256"],
                            "producer source snapshot drift")
                    bind(source["snapshot"])
                producer_artifacts = {Path(rec["path"]).name: rec for rec in producer_result["artifacts"]}
                saved = {(layer, name): load(producer_artifacts[f"layer{layer:02d}_{name}_same_input.npz"])
                         for layer in range(3) for name in ("actual", "canonical")}
                producer_rows = {(layer, name, stage): [] for layer in range(3)
                                 for name in ("actual", "canonical") for stage in CONE_STAGES}
                with bind(producer_artifacts["producer_cones.csv"]).open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        key = (int(row["layer"]), row["lineage"], int(row["stage"]))
                        require(key in producer_rows and int(row["index"]) == len(producer_rows[key]),
                                "producer coordinate order/identity")
                        producer_rows[key].append(row)
                require(all(len(rows) == local.SIZES[key[2]] for key, rows in producer_rows.items()),
                        "producer all-coordinate coverage")
            if repair_sufficiency_only:
                reviewed = read(retained.record(WEIGHTED_REVIEW))
                require(reviewed["kind"] == "round_reviewed_handoff"
                        and reviewed["producer_role"] == "reviewer"
                        and reviewed["mission_id"] == "e0701737bb56"
                        and reviewed["review"]["status"] == "done", "missing genuine weighted review")
                weighted = read(retained.record(WEIGHTED_INPUT / "result.json"))
                weighted_freeze = read(weighted["bindings"])
                require(weighted["status"] == "DIAGNOSTIC_COMPLETE" and weighted["numerical_status"] == "FAIL"
                        and weighted["policy_id"] == gates.POLICY_ID
                        and weighted["weighted_operator_coordinates"] == 115200
                        and not weighted["candidate_admitted"] and not weighted["policy_adopted"],
                        "weighted attribution scope/disposition changed")
                snapshots = {s["original"]["path"]: s for s in weighted_freeze["sources"]}
                for rec in weighted_freeze["input_bindings"]:
                    if rec["path"] in changed:
                        saved_source = snapshots[rec["path"]]
                        require(saved_source["original"] == rec
                                and saved_source["snapshot"]["sha256"] == rec["sha256"],
                                "weighted source snapshot mismatch")
                        bind(saved_source["snapshot"])
                    else:
                        bind(rec)
                for source in snapshots.values():
                    require(source["snapshot"]["sha256"] == source["original"]["sha256"],
                            "weighted source snapshot drift")
                    bind(source["snapshot"])
                for rec in weighted["artifacts"]:
                    bind(rec)
                terms = read(next(rec for rec in weighted["artifacts"]
                                  if Path(rec["path"]).name == "weighted_terms.json"))
                require({r["lineage"] for r in weighted["weighted_backward"]} == {"actual", "canonical"},
                        "weighted lineage coverage changed")
                for summary in weighted["weighted_backward"]:
                    require(sum((Fraction(row["weighted_effect"]) for row in terms
                                 if row["lineage"] == summary["lineage"]), Fraction())
                            == Fraction(summary["global_operator_delta"])
                            and summary["root_state_delta"] == summary["closure_error"] == "0",
                            "reviewed attribution closure changed")
                result["selection_evidence"] = [
                    row for row in terms if row["layer"] == 2 and row["stage"] == 16
                    and row["term"] == "binary64_to_fp16_rounding"]
            original_source = read(global_control["binary64_reference_freeze"])
            require(original_source["policy"] == "legacy-binary64-AWQ-fully-independent-propagation",
                    "original global policy changed")
            packages = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "safetensors")}
            require(packages == original_source["packages"], "original global package drift")
            global_namespace = {
                "np": np, "torch": torch, "torch_functional": functional, "math": math,
                "AWQ_REVERSE_ORDER": (0, 4, 1, 5, 2, 6, 3, 7),
                "HEAD_DIM": 64, "HIDDEN_SIZE": 896, "QUERY_HEADS": 14, "KEY_VALUE_HEADS": 2,
            }
            original_functions = {}
            for filename, names in (
                    ("official_single_decoder_layer.py", ("_unpack_words", "_torch_rmsnorm")),
                    ("official_model24_dialogue.py", ("_reference_projection", "_reference_layer_step"))):
                matches = [r for r in original_source["inputs_and_sources"]
                           if "/ace3-continuous-generation-20260906/" in r["path"]
                           and r["path"].endswith("/" + filename)]
                require(len(matches) == 1, "missing unique original binary64 source")
                bind(matches[0])
                reference_loader.definitions(matches[0], names, global_namespace)
                original_functions[filename] = {"source": matches[0], "functions": list(names)}
            tensors = {}
            tensor_records = {r["name"]: r for r in model["checkpoint_tensors"]}
            with safe_open(str(bind(model["checkpoint"])), framework="numpy") as checkpoint:
                official = checkpoint.get_slice("model.embed_tokens.weight")[9707:9708]
                require(np.array_equal(official.view("<u2").reshape(-1), embedding),
                        "original official embedding identity")
                for layer in range(3):
                    tensors[layer] = {name: checkpoint.get_tensor(name) for name in local.tensor_shapes(layer)}
                    local.authenticate_tensors(tensors[layer], tensor_records, layer)
            helper_sources = []
            for module in tuple(sys.modules.values()):
                filename = getattr(module, "__file__", None)
                if filename and Path(filename).resolve().is_relative_to(ROOT / "ace3/model"):
                    helper_sources.append(retained.record(bind(retained.record(filename))))
            global_metadata = {
                "original_functions": original_functions, "helper_sources": helper_sources,
                "packages": packages, "torch_threads": torch.get_num_threads(),
                "source_reference_freeze": global_control["binary64_reference_freeze"],
                "root": original_freeze["state_lineage"]["root"], "candidate_data_used": False,
                "public_contract": "capture_global_layer(namespace,tensors,layer,hidden); L0-L2/P0 float64[1,896]",
                "endpoint_rule": "all 2688 original S18 binary64 bit patterns must reproduce exactly",
                "residual_equation": "H-R = incoming_delta + operator_delta - original_residual_rounding + projection_error",
                "scope": "independently propagated original binary64 hidden and own-layer empty P0 binary64 KV",
            }
        sources = []
        (out / "source").mkdir()
        for path in sorted(changed):
            rec = retained.record(path)
            copy = out / "source" / Path(path).name
            shutil.copyfile(path, copy)
            require(retained.record(copy)["sha256"] == rec["sha256"], "current source copy drift")
            sources.append({"original": rec, "snapshot": retained.record(copy)})
        retained.write(out / "freeze.json", {
            "input_bindings": list(bindings.values()), "sources": sources,
            "command": retained.record(command), "review": retained.record(CONTROL_REVIEW),
            "tools": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
            "public_contract": "--repair-sufficiency --out BUILD_PATH --command COMMAND_FILE; L0-L2/P0 only"
            if repair_sufficiency_only else "--weighted-backward --out BUILD_PATH --command COMMAND_FILE; L0-L2/P0 only"
            if weighted_backward_only else "--producer-cones --out BUILD_PATH --command COMMAND_FILE; L0-L2/P0 only"
            if producer_cones_only else "--original-global --out BUILD_PATH --command COMMAND_FILE; L0-L2/P0 only"
            if original_global else "--decompose --out BUILD_PATH --command COMMAND_FILE; L0-L2/P0 only",
            "scope": {"layers": [0, 1, 2],
                      "stages": list(range(19)) if repair_sufficiency_only
                      else sorted(set(CONE_STAGES) | {12, 18}) if weighted_backward_only
                      else list(CONE_STAGES) if producer_cones_only else [12, 18],
                      "position": 0, "history": [9707]},
            "policy_id": gates.POLICY_ID, "state_id": retained.STATE_ID,
            "equations": {
                "paired": "delta_I = incoming_delta + operator_delta; delta_H = delta_I + projection_delta",
                "operator": "operator_delta = same_input_arithmetic_delta + operand_trajectory_delta",
                "global": "H-R = (I_in-R_prev) + (O11+O17-(R-R_prev)) + (H-I_out)"},
            "coverage": {"weighted_operator_coordinates": 115200, "weighted_lineages": 2,
                         "root": "official token9707 embedding", "target": "L2/P0/S17/index62"}
            if weighted_backward_only else {"producer_coordinates": 6 * sum(local.SIZES[s] for s in CONE_STAGES),
                         "weighted_down_index62_coordinates": 9728, "global_coordinates_per_lineage": 2688}
            if producer_cones_only else {"paired_residual_coordinates": 5376, "global_coordinates_per_lineage": 2688,
                         "original_global_residual_coordinates": 10752 if original_global else 0},
            "score_policy": "diagnostic completion only; preserve v3/local and original binary64-v1 FAIL",
            "unresolved_boundary": "Finite-difference path dependence and explicitly measured binary64 transport remainders"
            if weighted_backward_only else "Unique earliest attribution requires weighted propagation through prior nonlinear recurrences"
            if producer_cones_only else "Original operands require fresh endpoint-authenticated capture"
            if original_global else "No authenticated global S12/branch arrays in these inputs; no unique operator attribution",
            "original_global_capture": global_metadata,
            "arithmetic_lineages": ["retained attempt002", "reviewed canonical-local control"],
            "state_lineages": "separate root-origin I/Z/H recurrences with each layer's own empty P0 KV",
            "rtl_invocations": 0, "l9_invocations": 0,
            "planned_original_reference_layer_calls": 3 if original_global else 0})
        authentication_seconds = time.monotonic() - started
        if repair_sufficiency_only:
            retained.write(out / "counterfactual_contract.json", {
                "rounding_controls": list(ROUNDING_CONTROLS), "operator": "S16 SiLU(gate)*up",
                "intervention_scope": "uniform full S16 vectors at L0-L2/P0; never coordinate-specific",
                "arithmetic": "frozen original Torch binary64 operator followed by explicit FP16 rounding",
                "unmodified_operators": "canonical local controls; identity agreement is not independent validation",
                "state": "unchanged authorized Q24 I/Z/H; FP16 operator boundaries and own-layer FP16 KV",
                "root": "official token9707; reuse reviewed canonical L0/S0-S15 only",
                "suffix": "actual control operands and residual feed their own next operator/layer; no splices",
                "reference": "unchanged original-input global binary64 and FP16 diagnostic trajectories",
                "gates": "v3 exact mandatory local FP16 and S18 binary64-v1; stop each arm at first failure",
                "prior_reviews": [retained.record(path) for path in
                                  (WEIGHTED_REVIEW, PRODUCER_REVIEW, CAPTURE_REVIEW)],
                "plan_basis": result["selection_evidence"],
                "precision_change": False, "policy_adopted": False, "candidate_admitted": False,
                "rounding_policy_change": "explicit counterfactual only, not adopted",
                "score_policy": "a survivor is software counterfactual evidence, not independent native admission",
            })
            phase = "repair_sufficiency"
            result.update(bindings=retained.record(out / "freeze.json"),
                          counterfactual_contract=retained.record(out / "counterfactual_contract.json"))
            result.update(repair_sufficiency(out, global_namespace, tensors, data, references,
                                             trajectories, root, embedding, diagnosis, result))
            result["phases"].insert(0, {"phase": "authentication", "seconds": authentication_seconds})
            result["seconds"] = time.monotonic() - started
            retained.write(out / "result.json", result)
            print(f"REPAIR_SUFFICIENCY_{result['numerical_status']} result={out / 'result.json'}", flush=True)
            return 0
        if weighted_backward_only:
            retained.write(out / "weighted_contract.json", {
                "public_contract": "--weighted-backward --out BUILD_PATH --command COMMAND_FILE; L0-L2/P0",
                "target": "retained minus original binary64 L2/P0/S17/index62, separately for both lineages",
                "prior_review": retained.record(PRODUCER_REVIEW),
                "upstream_review": retained.record(CAPTURE_REVIEW),
                "prior_result": retained.record(PRODUCER_INPUT / "result.json"),
                "projection_rule": "input adjoint = binary64 W @ output adjoint; exact reduction remainder retained",
                "s16_rule": "gate-first retained binary64 finite secants; zero change/effect has zero multiplier",
                "rms_rule": "direct_i=v_i*gamma_i/ra; coupled_i=-sum(v*gamma*g)*(a_i+g_i)"
                "/(N*ra*rg*(ra+rg)); r(x)=sqrt(mean(x*x)+1e-6)",
                "residual_rule": "Iout-Gout=(Iin-Gin)+(branch-Gbranch)-original_add_rounding; H=I+view_error",
                "precision": "binary64 transport weights; exact rational signed effects and explicit transport remainders",
                "independent_oracle": "reuse authenticated reviewed original and same-input binary64 captures; "
                "no candidate arithmetic or fresh layer/operator invocation",
                "score_policy": "exact accounting only; original global FAIL and v3 unchanged",
                "candidate_admitted": False, "policy_adopted": False, "rtl_invocations": 0, "l9_invocations": 0,
            })
            phase = "weighted_backward"
            result.update(evidence_kind="reviewed_software_weighted_backward_attribution",
                          bindings=retained.record(out / "freeze.json"),
                          weighted_contract=retained.record(out / "weighted_contract.json"))
            result.update(weighted_backward(out, global_namespace, tensors, data, original_captures,
                                            references, root, embedding, producer_result, producer_rows,
                                            saved, diagnosis))
            result["phases"].insert(0, {"phase": "authentication", "seconds": authentication_seconds})
            result["seconds"] = time.monotonic() - started
            retained.write(out / "result.json", result)
            print(f"WEIGHTED_BACKWARD_{result['status']} result={out / 'result.json'}", flush=True)
            return 0
        if producer_cones_only:
            retained.write(out / "producer_contract.json", {
                "public_contract": "--producer-cones --out BUILD_PATH --command COMMAND_FILE; L0-L2/P0",
                "layers": [0, 1, 2], "position": 0, "history": [9707], "stages": list(CONE_STAGES),
                "prior_review": retained.record(CAPTURE_REVIEW),
                "prior_result": retained.record(CAPTURE_INPUT / "result.json"),
                "equation": "A-G = (A-L) + (L-RNE16(F(A))) + (RNE16(F(A))-F(A)) + (F(A)-G)",
                "symbols": {"A": "retained FP16 output", "L": "existing local FP16 oracle",
                            "F": "independent frozen original binary64 operator on named operands",
                            "G": "reviewed original-global capture, bit-reproduced by F(original operands)"},
                "independence": "F imports no candidate arithmetic. L is independent of the retained candidate "
                "but is the canonical lineage implementation, so A-L there is identity accounting, not independent validation.",
                "s16_operand_order": "original gate/up -> actual gate/original up -> actual gate/up",
                "p0_pruning": "One causal key implies probability one; verify all 14 retained S9 values and own-layer "
                "cache identity. Q/K/score values cannot affect finite P0 attention output.",
                "coverage": {"producer_coordinates": 6 * sum(local.SIZES[s] for s in CONE_STAGES),
                             "weighted_L2_down_index62_inputs": 2 * 4864},
                "score_policy": "diagnostic exact telescope closure only; no numerical admission",
                "rtl_invocations": 0, "l9_invocations": 0,
                "candidate_admitted": False, "policy_adopted": False})
            phase = "producer_cones"
            result.update(evidence_kind="reviewed_original_global_same_input_producer_telescope",
                          bindings=retained.record(out / "freeze.json"),
                          producer_contract=retained.record(out / "producer_contract.json"))
            result.update(producer_cones(out, global_namespace, tensors, data, original_captures,
                                         references, trajectories, root, embedding, prior_result, diagnosis, result))
            result["phases"].insert(0, {"phase": "authentication", "seconds": authentication_seconds})
            result["seconds"] = time.monotonic() - started
            retained.write(out / "result.json", result)
            print(f"PRODUCER_CONES_{result['status']} result={out / 'result.json'}", flush=True)
            return 0
        captured_global, original_rows, original_totals, capture_records, capture_phases = {}, [], [], [], []
        if original_global:
            phase = "original_global_capture"
            result["original_global_layer_invocations"] = 0
            result["model_operator_invocation_unit"] = "complete original reference layer function calls"
            global_hidden = torch.from_numpy(embedding.view("<f2").astype("<f8").reshape(1, 896))
            for layer in range(3):
                begin = time.monotonic()
                result["original_global_layer_invocations"] += 1
                result["model_operator_invocations"] += 1
                global_hidden, captured = capture_global_layer(global_namespace, tensors[layer], layer, global_hidden)
                # Preserve the raw attempted capture even if endpoint authentication rejects it.
                capture_records.append(retained.save(out / f"original_global_layer{layer:02d}.npz", captured))
                verify_global_endpoint(captured, references[layer])
                captured_global[layer] = captured
                capture_phases.append({"phase": "original_global_capture", "layer": layer,
                                       "seconds": time.monotonic() - begin})
            result.update(original_endpoint_coordinates=2688, original_endpoint_bit_mismatches=0)
        phase = "exact_decomposition"
        retained.verify_parent(root, root, embedding=embedding)
        parents = {"actual": root, "canonical": root}
        previous = embedding.view("<f2").astype("<f8")
        paired, global_split, totals, global_totals, global_gates = [], [], [], [], []
        for layer in range(3):
            d = data[layer]
            for name in parents:
                arrays = d[name]
                retained.verify_parent(retained.state_from(arrays, "input", "input_hidden"), parents[name])
                for stage in (6, 7, 12, 18):
                    retained.check_stage_state(stage, arrays, parents[name])
                parents[name] = retained.state_from(arrays, "output", "stage18")
                if original_global:
                    for stage in (12, 18):
                        split = original_residual_rows(arrays, captured_global[layer], layer, stage, name)
                        original_rows.extend(split)
                        original_totals.append(dict(decomposition_totals(split, (
                            "incoming_delta", "operator_delta", "original_residual_rounding",
                            "state_delta", "projection_error", "global_error")),
                            layer=layer, stage=stage, lineage=name))
                rows = global_rows(arrays, previous, references[layer], layer, name)
                global_split.extend(rows)
                summary = decomposition_totals(rows, (
                    "incoming_delta", "net_operator_delta", "projection_error", "global_error"))
                global_totals.append(dict(summary, layer=layer, stage=18, lineage=name))
                expected = (d["reports"][18]["binary64_v1"] if name == "actual"
                            else d["control_reports"][18]["global_binary64_v1"])
                evaluated = gates.evaluate_decoder_stage(
                    stage=18, actual=arrays["stage18"], reference=trajectories[layer],
                    policy=gates.POLICY_ID, reference_binary64=references[layer])["binary64_v1"]
                require(evaluated == expected, "original global all-coordinate gate changed")
                global_gates.append({"layer": layer, "lineage": name, "binary64_v1": evaluated})
            for stage in (12, 18):
                rows = residual_rows(d["actual"], d["canonical"], d["local"], layer, stage)
                paired.extend(rows)
                summary = decomposition_totals(rows, (
                    "incoming_delta", "operator_delta", "operator_same_input_arithmetic_delta",
                    "operator_operand_trajectory_delta", "state_delta", "projection_delta", "view_delta"))
                totals.append(dict(summary, layer=layer, stage=stage,
                                   state_cancellations=sum(r["state_cancellation"] for r in rows)))
            previous = references[layer]
        require(len(paired) == len(global_split) == 5376, "incomplete all-coordinate decomposition")
        witness = preserve_witness(data[2]["reports"][18], diagnosis)
        require(control["control_index62"] == control["preserved_rejecting_witness"] == witness,
                "canonical witness changed")
        for filename, rows in (("residual_decomposition.csv", paired), ("global_decomposition.csv", global_split)):
            with (out / filename).open("x", newline="", encoding="ascii") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if original_global:
            require(len(original_rows) == 10752, "incomplete original-global residual coverage")
            with (out / "original_global_residual_decomposition.csv").open("x", newline="", encoding="ascii") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(original_rows[0]))
                writer.writeheader()
                writer.writerows(original_rows)
            result.update(
                evidence_kind="original_global_cpu_operand_capture_and_exact_residual_decomposition",
                original_global_coordinates=len(original_rows), original_global_totals=original_totals,
                original_global_operands=capture_records,
                original_global_decomposition=retained.record(out / "original_global_residual_decomposition.csv"),
                original_global_index62=[r for r in original_rows if r["index"] == 62],
                earliest_original_branch_drift=next((r for r in original_rows if Fraction(r["operator_delta"])), None))
        retained.write(out / "global_gates.json", global_gates)
        first = next((r for r in paired if Fraction(r["operator_delta"])), None)
        result.update(
            status="DIAGNOSTIC_COMPLETE", numerical_status="FAIL", preserved_rejecting_witness=witness,
            residual_totals=totals, global_totals=global_totals,
            paired_coordinates=len(paired), global_coordinates=len(global_split),
            index62_lineage=[r for r in paired if r["index"] == 62],
            index62_global_lineage=[r for r in global_split if r["index"] == 62],
            earliest_paired_residual_operator_drift=first,
            earliest_same_input_bit_divergence=control["earliest_same_input_bit_divergence"],
            earliest_responsible_operator="UNRESOLVED: paired differences are not the cause of the shared global failure",
            global_operator_boundary="Only combined S11+S17 endpoint increment is available, not global S12/S11/S17 operands",
            state_and_kv_identity="PASS",
            failure_taxonomy="global_numerical",
            root_cause_hypothesis="Shared FP16 operator-boundary/Q24 trajectory drift remains; no unique responsible operator established",
            regression="All 5376 residual coordinates close exactly; both lineages retain L2/P0/S18/index62 616e excess 1/2",
            artifacts=[retained.record(out / name) for name in (
                "residual_decomposition.csv", "global_decomposition.csv", "global_gates.json")],
            phases=[{"phase": "authentication", "seconds": authentication_seconds},
                    *capture_phases,
                    {"phase": phase, "seconds": time.monotonic() - started - authentication_seconds
                     - sum(p["seconds"] for p in capture_phases)}])
        if original_global:
            result.update(
                global_operator_boundary="Original-global S11/S12/S17 and projection inputs captured for every L0-L2/P0 coordinate",
                earliest_responsible_operator="UNRESOLVED: earliest nonzero branch drift is not unique causal responsibility",
                remaining_missing_boundary="No same-input error telescope through the original-vs-Q24 S11/S17 producer cones; "
                "captured projection/RMSNorm operands alone do not separate accumulated upstream drift from each operator's rounding",
                bindings=retained.record(out / "freeze.json"))
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError) as exc:
        result.update(status="BLOCKED", failure_taxonomy=phase, reason=f"{type(exc).__name__}: {exc}",
                      root_cause_hypothesis="required retained evidence or exact identity is unverifiable",
                      regression="preserve this failed attempt; no numerical attribution")
        print(result["reason"], file=sys.stderr, flush=True)
    result["seconds"] = time.monotonic() - started
    retained.write(out / "result.json", result)
    print(f"RESIDUAL_DECOMPOSITION_{result['status']} result={out / 'result.json'}", flush=True)
    return 0 if result["status"] == "DIAGNOSTIC_COMPLETE" else 1


def run(out, command):
    require(out.parent == ROOT / "build" and out.name.startswith("q24_software_root_"),
            "control output must be an isolated software-root build path")
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    bindings, phases = {}, []
    result = {"status": "BLOCKED", "rtl_invocations": 0, "l9_invocations": 0,
              "candidate_invocations": 0, "normal_host_review": "REQUIRED",
              "policy_id": gates.POLICY_ID, "candidate_admitted": False,
              "evidence_kind": "canonical_local_cpu_q24_diagnostic",
              "claim_boundary": "L0-L2/P0 fixed token9707 only; no RTL, tokens or full model",
              "failure_taxonomy": None}

    def bind(rec):
        signature = {k: rec[k] for k in ("path", "bytes", "sha256")}
        if signature["path"] not in bindings:
            require(retained.record(signature["path"]) == signature,
                    f"binding drift: {signature['path']}")
            bindings[signature["path"]] = signature
        require(bindings[signature["path"]] == signature, "conflicting binding")
        return Path(signature["path"])

    def read(path_or_record):
        rec = retained.record(path_or_record) if not isinstance(path_or_record, dict) else path_or_record
        return json.loads(bind(rec).read_text())

    def load(rec):
        with np.load(bind(rec), allow_pickle=False) as archive:
            return {k: archive[k].copy() for k in archive.files}

    phase = "authentication"
    try:
        review = read(REVIEW)
        require(review["kind"] == "round_reviewed_handoff"
                and review["producer_role"] == "reviewer"
                and review["mission_id"] == "995d0ba22311"
                and review["review"]["status"] == "done", "missing genuine input review")
        freeze, old_result = read(INPUT / "freeze.json"), read(INPUT / "result.json")
        diagnosis = read(VALIDATION / "failure_diagnosis.json")
        require(old_result["status"] == diagnosis["status"] == "FAIL"
                and old_result["first_failure"]["node"] == [2, 0, 18]
                and [r["layer"] for r in old_result["layers"]] == [0, 1, 2],
                "unexpected retained frontier")
        require(freeze["contract"]["policy_id"] == gates.POLICY_ID
                and freeze["contract"]["state_id"] == retained.STATE_ID
                and freeze["scope"]["history"] == [9707] and freeze["scope"]["position"] == 0,
                "retained policy/state/history mismatch")
        for item in freeze["sources"]:
            bind(item["original"])
            snapshot = bind(item["snapshot"])
            require(retained.record(snapshot)["sha256"] == item["original"]["sha256"],
                    "retained source snapshot mismatch")
        for name in ("command.sh", "launch.command.sh", "stdout.log", "stderr.log",
                     "exit_code.txt", "inspect_failure.py", "inspect_failure.command.sh",
                     "failure_diagnosis.stdout.log", "failure_diagnosis.stderr.log"):
            bind(retained.record(VALIDATION / name))
        require((VALIDATION / "exit_code.txt").read_text().strip() == "1",
                "retained process rejection missing")
        control_rec = freeze["global_reference_lineage"]["control"]
        model_rec = freeze["global_reference_lineage"]["model_freeze"]
        require(control_rec["sha256"] == retained.CONTROL_SHA,
                "original global control identity mismatch")
        control, model = read(control_rec), read(model_rec)
        policy = read(gates.CONTRACT)
        require(model_rec["sha256"] == policy["trusted_independent_freeze_sha256"],
                "original independent model identity mismatch")
        specification = read(model["reference_source"])
        require(model["checkpoint"] == control["checkpoint"] == specification["checkpoint"]
                and model["embeddings"] == control["embeddings"]
                and model["reference_source"] == control["fp16_reference_root"]
                and model["checkpoint_tensors"] == specification["checkpoint_tensors"],
                "independent original-input lineage mismatch")
        read(control["binary64_reference_freeze"])
        read(control["binary64_array_binding_freeze"])
        with bind(control["binary64_csv"]).open(newline="") as stream:
            rows = [r for r in csv.DictReader(stream)
                    if int(r["position"]) == 0 and 0 <= int(r["layer"]) <= 2]
        original = {(int(r["layer"]), int(r["index"])): r["reference_binary64_hex"] for r in rows}
        require(len(original) == len(rows) == 3 * 896, "global coordinate coverage mismatch")
        global_refs, trajectories = {}, {}
        for key, case in zip(control["ordered_cases"], control["cases"], strict=True):
            layer, position = key
            if position != 0 or layer not in range(3):
                continue
            require(layer not in global_refs, "duplicate global reference")
            values = np.load(bind(case["binary64_reference_array"]), allow_pickle=False)
            require(values.dtype == np.dtype("<f8") and values.shape == (896,)
                    and np.all(np.isfinite(values)) and np.all(np.abs(values) <= 65504)
                    and all(float(v).hex() == original[layer, i] for i, v in enumerate(values)),
                    "original binary64 array/CSV mismatch")
            global_refs[layer] = values
        for transaction in model["reference_transactions"]:
            layer = transaction["layer"]
            if transaction["position"] != 0 or layer not in range(3):
                continue
            require(layer not in trajectories, "duplicate FP16 trajectory")
            trajectories[layer] = {
                int(s): retained.read_words(bind(rec), local.SIZES[int(s)])
                for s, rec in transaction["stages"].items()}
            require(set(trajectories[layer]) == set(range(19)), "missing trajectory stage")
        require(set(global_refs) == set(trajectories) == set(range(3)), "missing reference layer")
        embedding_rec = model["embeddings"][0]
        require(embedding_rec == freeze["state_lineage"]["root"]
                and embedding_rec["token"] == 9707, "embedding lineage mismatch")
        embedding = retained.read_words(bind(embedding_rec["input"]), 896, indexed=True)
        canonical_records = {r["name"]: r for r in model["checkpoint_tensors"]}
        require(len(canonical_records) == len(model["checkpoint_tensors"]), "duplicate tensor")
        tensors = {}
        with safe_open(str(bind(model["checkpoint"])), framework="numpy") as checkpoint:
            official = checkpoint.get_slice("model.embed_tokens.weight")[9707:9708]
            require(official.dtype == np.dtype("<f2") and official.shape == (1, 896)
                    and np.array_equal(official.reshape(-1).view("<u2"), embedding),
                    "official embedding mismatch")
            for layer in range(3):
                tensors[layer] = {name: checkpoint.get_tensor(name) for name in local.tensor_shapes(layer)}
                local.authenticate_tensors(tensors[layer], canonical_records, layer)
        actual, old_local, old_reports = {}, {}, {}
        for layer, entry in enumerate(old_result["layers"]):
            require(entry["position"] == 0, "retained position mismatch")
            actual[layer] = load(entry["actual_stages"])
            old_reports[layer] = read(entry["reports"])
            old_local[layer] = load(retained.record(INPUT / f"layer{layer:02d}/local_references.npz"))
            require([r["node"] for r in old_reports[layer]] == [[layer, 0, s] for s in range(19)],
                    "retained stage/coordinate coverage mismatch")
            require(set(old_local[layer]) == {f"stage{s:02d}" for s in range(18)},
                    "retained local reference coverage mismatch")
            parent = entry["input_parent"]
            require(parent["next_layer"] == layer and parent["state_id"] == retained.STATE_ID
                    and parent["history"] == [9707] and parent["position"] == 0
                    and parent["policy_id"] == gates.POLICY_ID
                    and parent["evidence_kind"] == "cpu_software_q24"
                    and parent["rtl_admissible"] is False, "retained parent owner mismatch")
            expected_parent = root_state(embedding) if layer == 0 else retained.state_from(
                actual[layer - 1], "output", "stage18")
            retained.verify_parent(load(parent["state"]), expected_parent,
                                   embedding=embedding if layer == 0 else None)
        witness = preserve_witness(old_reports[2][-1], diagnosis)
        sources = []
        source_dir = out / "source"
        source_dir.mkdir()
        paths = {Path(__file__), Path(__file__).parents[1] / "tests/test_q24_software_canonical_control_v1.py"}
        paths.update(Path(item["original"]["path"]) for item in freeze["sources"])
        for index, path in enumerate(sorted(paths)):
            rec = retained.record(path)
            bind(rec)
            copy = source_dir / f"{index:02d}_{path.name}"
            shutil.copyfile(path, copy)
            require(retained.record(copy)["sha256"] == rec["sha256"], "source copy drift")
            sources.append({"original": rec, "snapshot": retained.record(copy)})
        retained.write(out / "freeze.json", {
            "sources": sources, "input_bindings": list(bindings.values()),
            "command": retained.record(command), "review": retained.record(REVIEW),
            "tools": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
            "policy_id": gates.POLICY_ID, "layers": [0, 1, 2], "position": 0, "history": [9707],
            "arithmetic_lineage": "independent canonical v3 operators; no candidate arithmetic calls",
            "state_lineage": "official root; rational Q24 I/Z/H; own empty P0 KV for each layer",
            "global_reference_lineage": {"control": control_rec, "candidate_data_used": False},
            "comparison_plan": "all 57 captured stage vectors, all 54 same-input local vectors, all 2688 global coordinates",
            "failure_rule": "FAIL if any canonical-control S18 fails original binary64-v1; no admission",
            "localization_rule": "earliest same-input bit divergence is descriptive, not unique causality",
            "rtl_invocations": 0, "l9_invocations": 0})
        phases.append({"phase": phase, "seconds": time.monotonic() - started})
        phase = "all_coordinate_control"
        parent, actual_parent = root_state(embedding), root_state(embedding)
        retained.save(out / "root_state.npz", parent)
        summaries, first_divergence, global_failures = [], None, []
        for layer in range(3):
            begin = time.monotonic()
            arrays = canonical_layer(tensors[layer], layer, parent)
            directory = out / f"layer{layer:02d}"
            directory.mkdir()
            reports, summaries_local = [], []
            for stage in range(19):
                key = f"stage{stage:02d}"
                expected = retained.check_stage_state(stage, actual[layer], actual_parent)
                if stage < 18:
                    if stage != 12:
                        expected = local.local_reference(stage, {
                            k: actual[layer][k] for k in local.OPERANDS[stage]}, tensors[layer], layer)
                    require(np.array_equal(expected, old_local[layer][key]),
                            f"retained independent local reference drift L{layer} S{stage}")
                    comparison = compare(actual[layer][key], expected)
                    comparison.update(layer=layer, stage=stage)
                    summaries_local.append(comparison)
                    if first_divergence is None and comparison["bit_differences"]:
                        index = comparison["first_bit_difference"]
                        first_divergence = {
                            "node": [layer, 0, stage], "index": index,
                            "actual": f"{int(actual[layer][key][index]):04x}",
                            "canonical_same_input": f"{int(expected[index]):04x}",
                            "operator": "RMSNorm" if stage in (0, 13) else str(stage),
                            "causal_attribution": "not established by bit divergence"}
                if stage == 18:
                    reproduced = gates.evaluate_decoder_stage(
                        stage=18, actual=actual[layer][key], reference=trajectories[layer][stage],
                        policy=gates.POLICY_ID, reference_binary64=global_refs[layer])
                    require(reproduced["binary64_v1"] == old_reports[layer][stage]["binary64_v1"],
                            "retained global all-coordinate comparison drift")
                report = {
                    "stage": stage, "control_vs_retained": compare(arrays[key], actual[layer][key]),
                    "control_vs_original_fp16_diagnostic": compare(arrays[key], trajectories[layer][stage])}
                if stage == 18:
                    evaluated = gates.evaluate_decoder_stage(
                        stage=18, actual=arrays[key], reference=trajectories[layer][stage],
                        policy=gates.POLICY_ID, reference_binary64=global_refs[layer])
                    report["global_binary64_v1"] = evaluated["binary64_v1"]
                    global_failures.extend(dict(row, layer=layer, position=0)
                                           for row in evaluated["binary64_v1"]["failures"])
                    if layer == 2:
                        result["control_index62"] = evaluated["binary64_v1"]["rows"][62]
                reports.append(report)
            archive = retained.save(directory / "canonical_stages.npz", arrays)
            retained.write(directory / "comparisons.json", reports)
            retained.write(directory / "retained_same_input.json", summaries_local)
            summaries.append({
                "layer": layer, "stages": len(reports),
                "stage_coordinates": sum(r["control_vs_retained"]["coordinates"] for r in reports),
                "control_vs_retained_bit_differences": sum(r["control_vs_retained"]["bit_differences"] for r in reports),
                "control_vs_retained_gate_failures": sum(r["control_vs_retained"]["fp16_gate"]["failure_count"] for r in reports),
                "legacy_trajectory_gate_failures": sum(r["control_vs_original_fp16_diagnostic"]["fp16_gate"]["failure_count"] for r in reports),
                "same_input_coordinates": sum(r["coordinates"] for r in summaries_local),
                "same_input_bit_differences": sum(r["bit_differences"] for r in summaries_local),
                "same_input_gate_failures": sum(r["fp16_gate"]["failure_count"] for r in summaries_local),
                "global_coordinates": reports[-1]["global_binary64_v1"]["coordinates"],
                "global_failures": reports[-1]["global_binary64_v1"]["failure_count"],
                "state_and_kv_identity": "PASS", "actual_stages": archive,
                "comparisons": retained.record(directory / "comparisons.json"),
                "same_input_comparisons": retained.record(directory / "retained_same_input.json"),
                "seconds": time.monotonic() - begin})
            parent = retained.state_from(arrays, "output", "stage18")
            actual_parent = retained.state_from(actual[layer], "output", "stage18")
            print(json.dumps(summaries[-1]), flush=True)
        result.update(
            status="FAIL" if global_failures else "PASS", layers=summaries,
            preserved_rejecting_witness=witness, global_failures=global_failures,
            witness_disposition="eliminated" if result["control_index62"]["accepted"] else "retained",
            earliest_same_input_bit_divergence=first_divergence,
            earliest_responsible_operator="not uniquely localized",
            failure_taxonomy="global_numerical" if global_failures else None,
            root_cause_hypothesis=(
                "canonical FP16 operator-boundary/Q24 trajectory still misses original global accuracy"
                if global_failures else
                "canonical recomputation collectively removes the observed global miss; individual causal operator unproven"),
            regression="all captured coordinates and immutable L2/P0/S18/index62 616e excess 1/2")
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError) as exc:
        result.update(status="BLOCKED", failure_taxonomy=phase,
                      reason=f"{type(exc).__name__}: {exc}",
                      root_cause_hypothesis="a required control dependency is unverifiable",
                      regression="frozen command and failing dependency; no numerical conclusion")
        print(result["reason"], file=sys.stderr, flush=True)
    result.update(seconds=time.monotonic() - started, phases=phases,
                  bindings=retained.record(out / "freeze.json") if (out / "freeze.json").exists() else None)
    retained.write(out / "result.json", result)
    print(f"CANONICAL_LOCAL_CONTROL_{result['status']} result={out / 'result.json'}", flush=True)
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--command", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--decompose", action="store_true",
                        help="decompose authenticated retained results; do not recompute model operators")
    mode.add_argument("--original-global", action="store_true",
                      help="capture original-global L0-L2/P0 operands and decompose both retained Q24 lineages")
    mode.add_argument("--producer-cones", action="store_true",
                      help="telescope same-input producer errors using reviewed original-global captures")
    mode.add_argument("--weighted-backward", action="store_true",
                      help="propagate reviewed S17/index62 effects through RMSNorm and prior Q24 recurrences")
    mode.add_argument("--repair-sufficiency", action="store_true",
                      help="screen uniform S16 FP16 rounding counterfactuals in the reviewed software cone")
    args = parser.parse_args()
    if args.decompose or args.original_global or args.producer_cones or args.weighted_backward or args.repair_sufficiency:
        return decompose(args.out.resolve(), args.command.resolve(), original_global=args.original_global,
                         producer_cones_only=args.producer_cones, weighted_backward_only=args.weighted_backward,
                         repair_sufficiency_only=args.repair_sufficiency)
    return run(args.out.resolve(), args.command.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
