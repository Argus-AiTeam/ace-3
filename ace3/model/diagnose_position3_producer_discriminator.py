#!/usr/bin/env python3
"""Regenerate authenticated historical same-input producers, not repaired RTL."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import diagnose_position3_early_hidden_cone as cone
import diagnose_position3_parentage_trace as parentage
from diagnose_position3_upstream_hidden import Evidence, record, require, write_json


CANDIDATES = ((2, "k"), (4, "v"), (5, "k"), (7, "k"), (7, "v"))
CHAINS = {"k": (0, 2, 5, 6), "v": (0, 3, 7)}


class HistoricalOperators(cone.Operators):
    """Only RoPE position differs from the retained position-3 implementation."""

    def __init__(self, layer, tensors, scope, position):
        super().__init__(layer, tensors, scope)
        self.position = position

    def independent_raw(self, stage, stages, incoming, caches):
        if stage in (4, 5):
            x = self.tensor(stages[1 if stage == 4 else 2]).reshape(-1, 64)
            frequency = 1.0 / (
                1_000_000.0 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
            angle = torch.outer(
                torch.arange(self.position, self.position + 1, dtype=torch.float64),
                frequency)
            low, high = x[:, :32], x[:, 32:]
            return torch.cat((low * angle.cos() - high * angle.sin(),
                              high * angle.cos() + low * angle.sin()), dim=1).reshape(-1)
        return super().independent_raw(stage, stages, incoming, caches)

    def rtl_model(self, stage, stages, incoming, caches):
        if stage in (4, 5):
            operand = stages[1 if stage == 4 else 2]
            return np.asarray(cone._rope(
                operand.tolist(), operand.size // 64, self.position), dtype="<u2")
        return super().rtl_model(stage, stages, incoming, caches)


def exponential_candidate(scores):
    probabilities = []
    with localcontext() as context:
        context.prec = 80
        for row in cone.values(scores).reshape(14, 4):
            shifted = [Decimal(float(x)) - Decimal(float(max(row))) for x in row]
            exps = [int((x.exp() * (1 << 24)).to_integral_value(
                rounding=ROUND_HALF_EVEN)) for x in shifted]
            probabilities.extend(
                round(Fraction(x << 24, sum(exps))) / (1 << 24) for x in exps)
    return np.asarray(probabilities, dtype="<f2").view("<u2")


def run(out):
    started = time.monotonic()
    require(out.parent == cone.ROOT / "build"
            and out.name.startswith("token358_position3_producer_discriminator_"),
            "output outside writable scope")
    measurement = out / "measurement001"
    measurement.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    sources = [record(module.__file__) for module in tuple(sys.modules.values())
               if getattr(module, "__file__", None)
               and Path(module.__file__).resolve().is_relative_to(cone.ROOT / "ace3/model")
               and Path(module.__file__).suffix == ".py"]
    write_json(measurement / "frozen.json", {
        "objective": (
            "Regenerate historical same-input RMSNorm/Q/K/V/RoPE/cache producers, "
            "distinguish arithmetic from inherited trajectory and FP16 cast policy, "
            "and compare layer0 stage09 against the retained exponential candidate."),
        "source": record(__file__), "sources": sources,
        "command": record(out / "command.sh"),
        "roots": [record(folder / "seal.json")
                  for folder in (parentage.SOFTMAX, parentage.FACTORIAL)],
        "comparison_policy": parentage.POLICY,
        "historical_layers": list(range(9)), "historical_positions": [0, 1, 2],
        "historical_stages": list(range(8)), "candidate_chains": CHAINS,
        "historical_candidates": CANDIDATES,
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "device": "cpu", "threads": 1,
        "public_RTL_contract": (
            "Unchanged modules, ports, parameters, FP16 activation/KV and G128 asymmetric "
            "native AWQ INT4 weights/zeros, no zero plus-one. No RTL compile or execution."),
        "intervention_boundary": (
            "Historical cuts change only the named cache producer contribution with "
            "its actual hidden input held fixed. They are not a changed-prefix rollout. "
            "Direct binary64-to-FP16 and retained Torch casts are reported separately."),
        "regressions": [
            "Fresh parentage traversal and all 52 endpoint/affected-stage verdicts.",
            "Original layer08 stage08 failures remain indices13,15.",
            "Authenticate actual inputs, traces, independent caches and official tensors.",
            "Source-model and independent same-input outputs retained as arrays.",
            "Recompute the retained Decimal80 exponential seed without endpoint tuning.",
        ],
        "production_changes": 0, "RTL_invocations": 0, "official_repair_attempts": 0,
    })
    (measurement / "source.py").write_bytes(Path(__file__).read_bytes())

    parentage.run(out / "parentage")
    evidence = Evidence()
    parent_root = record(out / "parentage/seal.json")
    evidence.register(parent_root)
    evidence.load(parent_root["path"])
    baseline = evidence.load(out / "parentage/result.json")
    evidence.register(baseline["authenticated_inputs"])
    catalog = evidence.load(out / "parentage/operators.json")
    factorial = evidence.load(out / "parentage/factorial_remeasurement.json")
    require(baseline["factorial_cases_remeasured"] == len(factorial["cases"]) == 52,
            "factorial regression count")
    require(factorial["cases"]["exp0_later_k0_v0"]["layer08_stage08"][
        "failure_indices"] == [13, 15], "original failure regression")

    frozen = parentage.json_input(evidence, parentage.SOFTMAX / "frozen.json")
    for key in ("helpers", "production_sources"):
        evidence.register(frozen[key])
        for item in frozen[key]:
            evidence.read(item["path"])
    require(record(cone.__file__)["sha256"] == frozen["source"]["sha256"],
            "retained operator implementation changed")
    plans, _, _, _ = cone.plans(evidence)
    scope = cone.namespace(evidence)
    summaries, cuts, gaps = [], [], []
    softmax_result = None
    for layer, _, prepared in plans:
        tick = time.monotonic()
        ref = evidence.load(prepared["independent_parent"]["path"])
        tensors = cone.load_tensors(evidence, prepared, ref)
        layer_catalog = catalog["layers"][layer]
        history = evidence.load(layer_catalog["historical_result"]["path"])
        factorial_report = evidence.load(parentage.FACTORIAL / f"layer{layer:02d}.json")
        factorial_arrays = parentage.arrays(evidence, factorial_report["arrays"]["path"])
        old_reference = {kind: evidence.array(ref["own_cache"][kind]["path"])
                         for kind in ("k", "v")}
        layer_arrays, rows = {}, []
        layer_cuts = {}
        for transaction in history["positions"]:
            position = transaction["position"]
            require(position in (0, 1, 2) and transaction["layer_index"] == layer,
                    "historical transaction identity")
            actual = evidence.trace(transaction["raw"]["trace"]["path"], position)
            incoming = evidence.bits(transaction["vectors"]["input"]["path"], True)
            ops = HistoricalOperators(layer, tensors, scope, position)
            chain, local_direct, local_torch, reconstructed = {}, {}, {}, {}
            layer_arrays[f"p{position}_input"] = incoming
            for stage in range(8):
                raw = ops.independent_raw(stage, actual, incoming, {})
                local_torch[stage] = raw.to(torch.float16).numpy().view("<u2").reshape(-1)
                local_direct[stage] = cone.half_bits(raw.numpy())
                reconstructed[stage] = ops.rtl_model(stage, actual, incoming, {})
                chain[stage] = ops.independent(stage, chain, incoming, {})
                for name, value in (
                    ("actual", actual[stage]), ("local_torch", local_torch[stage]),
                    ("local_direct", local_direct[stage]), ("rtl_model", reconstructed[stage]),
                    ("same_hidden_independent_chain", chain[stage]),
                    ("independent_raw_f64", raw.numpy().reshape(-1))):
                    layer_arrays[f"p{position}_s{stage:02d}_{name}"] = value
                exact = np.array_equal(actual[stage], reconstructed[stage])
                row = {
                    "layer": layer, "position": position, "stage": stage,
                    "operator": parentage.OPERATORS[stage],
                    "actual_trace": transaction["raw"]["trace"],
                    "actual_input": transaction["vectors"]["input"],
                    "array_prefix": f"p{position}_s{stage:02d}_",
                    "rtl_model_on_actual": parentage.measurement(
                        actual[stage], reconstructed[stage]),
                    "source_model_retained_exact": bool(exact),
                    "independent_on_actual_torch": parentage.measurement(
                        actual[stage], local_torch[stage]),
                    "independent_on_actual_direct": parentage.measurement(
                        actual[stage], local_direct[stage]),
                    "cast_discrimination": parentage.measurement(
                        local_direct[stage], local_torch[stage]),
                    "same_hidden_independent_chain": parentage.measurement(
                        actual[stage], chain[stage]),
                }
                if not exact:
                    gaps.append({
                        "taxonomy": "source_model_reconstruction_mismatch",
                        "layer": layer, "position": position, "stage": stage,
                        "hypothesis": "Retained producer arithmetic is not captured by this model.",
                        "regression": row["rtl_model_on_actual"],
                        "missing_observable": (
                            "Source-bound exact reconstruction of this retained operator "
                            "before any counterfactual using it can be causal evidence."),
                    })
                old_stage = next((entry for entry in catalog["stages"]
                                  if entry["position"] == position
                                  and entry["layer"] == layer
                                  and entry["stage"] == stage), None)
                if old_stage is not None:
                    expected = evidence.array(old_stage["independent_stage"]["path"])
                    layer_arrays[f"p{position}_s{stage:02d}_retained_trajectory"] = expected
                    row["retained_independent_stage"] = old_stage["independent_stage"]
                    row["trajectory"] = parentage.measurement(actual[stage], expected)
                rows.append(row)
            for kind, endpoint in (("k", 6), ("v", 7)):
                layer_arrays[f"p{position}_independent_cache_{kind}"] = (
                    old_reference[kind][position])
                require(np.array_equal(
                    actual[endpoint].reshape(2, 64),
                    factorial_arrays[f"actual_cache_{kind}"][position])
                        and np.array_equal(old_reference[kind],
                                           factorial_arrays[f"independent_old_{kind}"])
                        and np.array_equal(actual[endpoint], actual[5 if kind == "k" else 3]),
                        "historical cache edge")
                if (layer, kind) not in CANDIDATES:
                    continue
                production_chain = CHAINS[kind]
                for seed in production_chain[:-1]:
                    name = f"layer{layer:02d}_{kind}_stage{seed:02d}"
                    state = {s: actual[s].copy() for s in range(8)}
                    state[seed] = local_direct[seed]
                    suffix = production_chain[production_chain.index(seed) + 1:]
                    usable = all(np.array_equal(actual[s], reconstructed[s])
                                 for s in (seed, *suffix))
                    if not usable:
                        continue
                    for stage in suffix:
                        state[stage] = ops.rtl_model(stage, state, incoming, {})
                    layer_cuts.setdefault(name, []).append(
                        (position, state[endpoint].reshape(2, 64)))
        for name, positions in layer_cuts.items():
            if [position for position, _ in positions] != [0, 1, 2]:
                gaps.append({
                    "taxonomy": "incomplete_source_local_cut",
                    "name": name, "reconstructed_positions": [p for p, _ in positions],
                    "missing_observable": (
                        "Exact production reconstruction at all three historical positions; "
                        "partial cuts are not substituted or treated as complete."),
                })
                continue
            kind = name.split("_")[1]
            value = np.stack([bits for _, bits in positions])
            layer_arrays[name] = value
            cuts.append({
                "name": name, "layer": layer, "kind": kind,
                "seed_stage": int(name[-2:]), "positions": [0, 1, 2],
                "array_key": name,
                "vs_independent_historical_cache": parentage.measurement(
                    value, old_reference[kind]),
                "independent_cache": ref["own_cache"][kind],
                "endpoint_replay_performed": False,
                "boundary": (
                    "Source-local direct-FP16 operator cut on retained actual operands, "
                    "then reconstructed production cache suffix; other contributions fixed."),
            })

        if layer == 0:
            report = evidence.load(parentage.SOFTMAX / "layer00.json")
            saved = parentage.arrays(evidence, report["arrays"]["path"])
            actual = {stage: saved[f"actual_{stage:02d}"] for stage in range(19)}
            ops = HistoricalOperators(layer, tensors, scope, 3)
            reconstructed = ops.rtl_model(9, actual, saved["actual_incoming"], {})
            independent = ops.independent(9, actual, saved["actual_incoming"], {})
            candidate = exponential_candidate(actual[8])
            require(np.array_equal(reconstructed, actual[9]),
                    "layer0 stage09 source-model reconstruction")
            require(np.array_equal(candidate, saved["exp1_k0_v0_09"]),
                    "retained exponential seed regression")
            proof = evidence.load(report["operator_proof"]["path"])
            stage_proof = next(row for row in proof["stages"] if row["stage"] == 9)
            local = parentage.measurement(actual[9], independent)
            require(parentage.core(local) == parentage.core(
                stage_proof["independent_on_actual_operands"]),
                "layer0 stage09 same-input regression")
            layer_arrays.update({
                "position3_stage08_actual": actual[8],
                "position3_stage09_actual": actual[9],
                "position3_stage09_rtl_model": reconstructed,
                "position3_stage09_independent_same_input": independent,
                "position3_stage09_exponential_candidate": candidate,
            })
            softmax_result = {
                "layer": 0, "position": 3, "stage": 9,
                "source_model_retained_exact": True,
                "independent_on_actual": local, "retained_candidate_exact": True,
                "candidate_definition": frozen["candidate"],
                "original_endpoint": factorial["cases"]["exp0_later_k0_v0"]["layer08_stage08"],
                "candidate_endpoint": factorial["cases"]["exp1_later_k0_v0"]["layer08_stage08"],
                "bounded_candidate_affected_gates_clear": factorial["cases"][
                    "exp1_later_k0_v0"]["endpoint_and_affected_gates_clear"],
                "endpoint_evidence": record(out / "parentage/factorial_remeasurement.json"),
            }
        archive = measurement / f"layer{layer:02d}.npz"
        np.savez_compressed(archive, **layer_arrays)
        archive_record = record(archive)
        for cut in cuts:
            if cut["layer"] == layer:
                cut["arrays"] = archive_record
        for row in rows:
            row["arrays"] = archive_record
        write_json(measurement / f"layer{layer:02d}.json", {
            "layer": layer, "arrays": archive_record, "stages": rows,
            "historical_result": layer_catalog["historical_result"],
            "tensors": prepared["vectors"]["tensors"], "seconds": time.monotonic() - tick,
        })
        summaries.extend(rows)
        print(json.dumps({"layer": layer, "new_same_input_comparisons": len(rows),
                          "seconds": time.monotonic() - tick}), flush=True)

    for cut in cuts:
        gaps.append({
            "taxonomy": "operator_specific_downstream_counterfactual_unmeasured",
            "name": cut["name"], "arrays": cut["arrays"], "array_key": cut["array_key"],
            "missing_observable": (
                "Layer08 position3 stage08 and every changed-stage gate after replacing "
                "only this bound historical cache contribution and propagating the "
                "production-model position3 affected cone. The retained 52 cases replace "
                "whole independently propagated caches, not this operator-local cut."),
            "known_difference_from_retained_substitution": cut["vs_independent_historical_cache"],
        })
    require(len(summaries) == 216 and softmax_result is not None, "diagnostic coverage")
    for item in sources:
        require(record(item["path"]) == item, "source changed during diagnostic")
    result = {
        "status": "BOUNDED_SAME_INPUT_PRODUCERS_REGENERATED_DISCRIMINATOR_GAP_SEALED",
        "new_historical_same_input_comparisons": len(summaries),
        "local_direct_arithmetic_differences": [
            {"layer": r["layer"], "position": r["position"], "stage": r["stage"],
             "comparison": r["independent_on_actual_direct"], "arrays": r["arrays"],
             "array_prefix": r["array_prefix"]}
            for r in summaries if r["independent_on_actual_direct"]["different_count"]],
        "historical_source_model_reconstruction_mismatches": sum(
            not row["source_model_retained_exact"] for row in summaries),
        "historical_source_local_cuts": cuts, "layer0_stage09": softmax_result,
        "factorial_cases_remeasured": 52, "retained_passing_cases": factorial["passing_cases"],
        "parentage_seal": parent_root, "missing_observables": gaps,
        "unique_earliest_production_cause_identified": False,
        "production_intervention_authorized": False,
        "fresh_official_repaired_attempt_authorized": False,
        "isolated_candidate_selection": {
            "target": "General layer0 stage09 exponential operator under existing interfaces.",
            "basis": "Freshly reproduced same-input difference and bounded factorial sufficiency.",
            "authority": (
                "Planner may select a separate isolated RTL candidate after independent "
                "Reviewer validation. This diagnostic grants no production repair authority."),
            "historical_rollout": (
                "Affected prefix/KV invalidation remains required before official traversal; "
                "fixed historical caches do not establish changed-operator rollout correctness."),
        },
        "production_changes": 0, "RTL_invocations": 0, "official_repair_attempts": 0,
        "prefix_RTL_replays": 0, "acceptance_policy_changes": 0, "third_token_claim": False,
        "review": "Host independent Reviewer required; no review verdict inferred.",
        "authenticated_inputs": sorted(evidence.consumed.values(), key=lambda item: item["path"]),
        "seconds": time.monotonic() - started,
    }
    write_json(measurement / "result.json", result)
    write_json(measurement / "seal.json", {
        "files": [record(path) for path in sorted(measurement.iterdir())],
        "parentage_seal": parent_root,
    })
    print(json.dumps({key: result[key] for key in (
        "status", "new_historical_same_input_comparisons", "factorial_cases_remeasured",
        "historical_source_model_reconstruction_mismatches", "official_repair_attempts",
        "seconds")}), flush=True)
    print(json.dumps({
        "cuts_vs_whole_independent_cache": {
            cut["name"]: cut["vs_independent_historical_cache"]["different_count"]
            for cut in cuts},
        "local_direct_differences": [
            [r["position"], r["layer"], r["stage"],
             r["independent_on_actual_direct"]["different_count"]]
            for r in summaries if r["independent_on_actual_direct"]["different_count"]],
        "missing_observables": len(gaps),
        "seal": str(measurement / "seal.json"),
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    try:
        run(output)
    except (AssertionError, KeyError, ValueError, OSError) as error:
        failure = output / "measurement001/failure.json"
        if failure.parent.is_dir() and not failure.exists():
            write_json(failure, {
                "status": "DIAGNOSTIC_FAILED_NO_RTL_CONCLUSION",
                "taxonomy": "diagnostic_authentication_or_reconstruction_failure",
                "hypothesis": "A frozen artifact edge or numerical reconstruction is incompatible.",
                "error": str(error), "traceback": traceback.format_exc(),
                "regression": "Fresh same-input diagnostic; original failures and gates unchanged.",
                "official_repair_attempts": 0,
            })
        raise
