#!/usr/bin/env python3
"""Trace sealed historical producers without changing or replaying production."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from diagnose_position3_upstream_hidden import Evidence, compared, record, require, write_json


ROOT = Path(__file__).resolve().parents[2]
SOFTMAX = ROOT / "build/token358_position3_softmax_kv_discrimination_attempt001/measurement002"
FACTORIAL = ROOT / "build/token358_position3_historical_producer_factorization_attempt001/measurement001"
LAYER0 = ROOT / "build/token358_position3_layer0_attempt001"
POLICY = {
    "absolute_tolerance": 0.125,
    "accept": "finite AND (abs_error<=0.125 OR (relative_error<0.001 AND orderedFP16_ULP<=1))",
    "max_ulp_distance": 1,
    "relative_denominator": "max(abs(reference),2^-14)",
    "relative_tolerance_strict": 0.001,
}
OPERATORS = [
    "input RMSNorm", "Q projection", "K projection", "V projection",
    "Q RoPE", "K RoPE", "K cache write", "V cache write",
    "attention score", "softmax", "attention value", "O projection",
    "attention residual", "post-attention RMSNorm", "gate projection",
    "up projection", "SiLU multiply", "down projection", "MLP residual",
]


def records(value):
    if isinstance(value, dict):
        if "path" in value and ("sha256" in value or "file_sha256" in value):
            yield value
        else:
            for child in value.values():
                yield from records(child)
    elif isinstance(value, list):
        for child in value:
            yield from records(child)


def arrays(evidence, path):
    with np.load(io.BytesIO(evidence.read(path)), allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    require(all(a.dtype == np.dtype("<u2") for a in result.values()), "archive precision")
    return result


def measurement(actual, independent):
    require(actual.size == independent.size, "comparison geometry")
    require(np.isfinite(actual.view("<f2")).all()
            and np.isfinite(independent.view("<f2")).all(), "nonfinite diagnostic input")
    return compared(actual.reshape(-1), independent.reshape(-1))


def core(comparison):
    return {key: comparison[key] for key in
            ("different_count", "failure_indices", "max_absolute_error")}


def json_input(evidence, path):
    # Historical manifests can name different revisions of the same live source.
    # Register only the selected artifact edges, not those historical source aliases.
    return json.loads(evidence.read(path))


def run(out):
    started = time.monotonic()
    out = out.resolve()
    standalone = out.parent == ROOT / "build" and out.name.startswith("token358_position3_")
    discriminator_child = (
        out.name == "parentage" and out.parent.parent == ROOT / "build"
        and out.parent.name.startswith("token358_position3_producer_discriminator_"))
    require(standalone or discriminator_child, "output outside diagnostic scope")
    out.mkdir(exist_ok=False)
    roots = [record(folder / "seal.json") for folder in (SOFTMAX, FACTORIAL)]
    helpers = [record(module.__file__) for module in tuple(sys.modules.values())
               if getattr(module, "__file__", None)
               and Path(module.__file__).resolve().is_relative_to(ROOT / "ace3/model")
               and Path(module.__file__).suffix == ".py"]
    write_json(out / "frozen.json", {
        "question": "Which historical/current operator divergences can the sealed evidence distinguish?",
        "roots": roots, "source": record(__file__), "helpers": helpers,
        "argv": sys.argv, "python": sys.version, "numpy": np.__version__,
        "comparison_policy": POLICY,
        "ordering": "position then layer then stage; bit divergence is not causal attribution",
        "public_RTL_contract": "Unchanged; no compilation, RTL execution or candidate implementation.",
        "scope": "Position2 parentage; layers0-8 historical K/V and position3 operators through layer08 stage08.",
        "regressions": [
            "Compare actual traces and independent stage files to both retained archives.",
            "Recompute all 52 endpoint and affected-stage verdicts from sealed FP16 arrays.",
            "Check cache writes, historical state lineage, and sequential hidden edges.",
            "Keep original layer08 stage08 failure indices13,15 and frozen gates.",
        ],
        "official_RTL_repair_attempts": 0,
    })
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    evidence = Evidence()
    evidence.register(roots)
    retained = []
    for folder in (SOFTMAX, FACTORIAL):
        seal = evidence.load(folder / "seal.json")
        for item in seal["files"]:
            evidence.read(item["path"])
        frozen = json_input(evidence, folder / "frozen.json")
        require(frozen["comparison_policy"] == POLICY, "retained gate drift")
        require(hashlib.sha256(evidence.read(folder / "source.py")).hexdigest()
                == frozen["source"]["sha256"], "retained source snapshot mismatch")
        result = evidence.load(folder / "result.json")
        require(result["RTL_invocations"] == result["official_attempts"] == 0,
                "retained software evidence boundary")
        retained.append(result)
    soft_result, factor_result = retained
    require(len(factor_result["cases"]) == 52, "factorial case count")

    f0 = json_input(evidence, LAYER0 / "frozen.json")
    evidence.register([f0["package"], f0["review"], f0["parent_frozen"]])
    package = json_input(evidence, f0["package"]["path"])
    evidence.register([package["parents"], package["prefix"], package["kv_parents"],
                       package["embedding"]])
    require(package["selected_token_id"] == 358 and package["position"] == 3
            and package["token_history"] == [9707, 1879, 0, 358], "accepted token lineage")
    parent_reviews = []
    review_records = [f0["review"]]
    for parent in package["parents"].values():
        for item in records(parent):
            evidence.read(item["path"])
        if "review" in parent:
            review_records.append(parent["review"])
    for item in review_records:
        review = json_input(evidence, item["path"])
        require(review["producer_role"] == "reviewer"
                and review["review"]["status"] == "done", "parent review not accepted")
        parent_reviews.append({"record": item, "review": review["review"]})
    parent_frozen = json_input(evidence, f0["parent_frozen"]["path"])
    reference_roots = parent_frozen["independent_references"]
    evidence.register(reference_roots)
    reference_indices = []
    for item in records(reference_roots):
        if item["path"].endswith(".json"):
            index = evidence.load(item["path"])
            reference_indices.append({"record": item, "keys": sorted(index)})

    parentage = {
        "package": f0["package"], "position2_producer_manifest": f0["parent_frozen"],
        "selected_token_id": 358, "history": package["token_history"],
        "profile": package["profile"], "parents": package["parents"],
        "reviews": parent_reviews, "reference_indices": reference_indices,
        "boundary": "Reuses authenticated accepted parent records; does not re-adjudicate or replay them.",
    }
    write_json(out / "parentage.json", parentage)

    stage_rows, history_rows, hidden_edges, coverage = [], [], [], []
    branch_results = {name: {"affected_stage_failures": []}
                      for name in factor_result["cases"]}
    previous = {}
    layer_records = []
    for layer in range(9):
        soft_report = evidence.load(SOFTMAX / f"layer{layer:02d}.json")
        factor_report = evidence.load(FACTORIAL / f"layer{layer:02d}.json")
        proof = evidence.load(soft_report["operator_proof"]["path"])
        saved = arrays(evidence, soft_report["arrays"]["path"])
        factorial = arrays(evidence, factor_report["arrays"]["path"])
        parent = next(p for p in package["kv_parents"] if p["layer_index"] == layer)
        require(parent["actual_kv_traces"] == proof["actual_kv_parents"]
                and parent["valid_positions"] == [0, 1, 2], f"historical parent {layer}")
        historical_result = evidence.load(parent["layer_result"]["path"])
        require(historical_result["layer_index"] == layer, f"parent layer {layer}")
        for item in (parent["state"], parent["live_binary"]):
            evidence.read(item["path"])
        ref_paths = [p for p in evidence.bindings
                     if p.endswith(f"/layer{layer:02d}_generation1.json")
                     and "/independent_fp16_trajectory_" in p]
        require(len(ref_paths) == 1, f"independent index identity {layer}")
        ref = evidence.load(ref_paths[0])
        require(ref["layer"] == layer and ref["history"] == [9707, 1879, 0]
                and ref["own_cache"] == proof["independent_kv_parents"],
                f"independent historical identity {layer}")
        independent_old = {kind: evidence.array(ref["own_cache"][kind]["path"])
                           for kind in ("k", "v")}
        actual = evidence.trace(soft_report["actual_trace"]["path"], 3)
        incoming = evidence.bits(proof["actual_input"]["path"], True)
        require(np.array_equal(incoming, saved["actual_incoming"]), f"position3 input {layer}")
        if layer:
            require(np.array_equal(incoming, previous[3]), f"position3 hidden edge {layer}")
            hidden_edges.append({"position": 3, "from_layer": layer - 1, "to_layer": layer,
                                 "input": proof["actual_input"], "stage18_exact": True})
        else:
            require(proof["actual_input"] == package["embedding"]["file"],
                    "selected token embedding binding")
        last = 8 if layer == 8 else 18
        proof_stages = {entry["stage"]: entry for entry in proof["stages"]}
        require(len(proof_stages) == len(proof["stages"]), "duplicate proof stage")
        for stage in range(last + 1):
            a, independent = actual[stage], saved[f"independent_{stage:02d}"]
            require(np.array_equal(a, saved[f"actual_{stage:02d}"])
                    and np.array_equal(a, factorial[f"actual_{stage:02d}"])
                    and np.array_equal(independent, factorial[f"independent_{stage:02d}"]),
                    f"position3 archive trace binding {layer}/{stage}")
            comparison = measurement(a, independent)
            entry = proof_stages.get(stage)
            independent_source = {
                "archive": soft_report["arrays"], "array_key": f"independent_{stage:02d}",
            }
            if entry is not None:
                expected = evidence.bits(entry["independent_stage"])
                require(np.array_equal(independent, expected), "independent stage file binding")
                require(core(comparison) == core(entry["trajectory"]), "retained trajectory verdict")
                independent_source["stage_file"] = evidence.consumed[
                    str(Path(entry["independent_stage"]).resolve())]
            stage_rows.append({
                "position": 3, "layer": layer, "stage": stage, "operator": OPERATORS[stage],
                "comparison": comparison, "actual_trace": soft_report["actual_trace"],
                "independent_stage": independent_source,
                "same_input_retained_comparison": (
                    entry["independent_on_actual_operands"] if entry is not None else None),
                "same_input_evidence": soft_report["operator_proof"],
                "same_input_boundary": (
                    "Retained source-bound diagnostic, not recomputed arithmetic."
                    if entry is not None else
                    "No stage-local comparison in this proof; trajectory arrays only, not local attribution."),
            })
            for name, branch in branch_results.items():
                state = factorial[f"{name}_{stage:02d}"]
                delta = measurement(state, independent)
                if not np.array_equal(a, state) and delta["failure_indices"]:
                    branch["affected_stage_failures"].append(
                        {"layer": layer, "stage": stage, "comparison": delta})
                if layer == 8 and stage == 8:
                    branch["layer08_stage08"] = delta
                    branch["array"] = factor_report["arrays"]
                    branch["array_key"] = f"{name}_{stage:02d}"
        if last == 18:
            previous[3] = actual[18]

        for position, trace_record in enumerate(parent["actual_kv_traces"]):
            old = evidence.trace(trace_record["path"], position)
            transaction = next(p for p in historical_result["positions"]
                               if p["position"] == position)
            require(transaction["raw"]["trace"] == trace_record, "parent trace receipt")
            old_input = evidence.bits(transaction["vectors"]["input"]["path"], True)
            require(hashlib.sha256(old_input.tobytes()).hexdigest()
                    == transaction["input"]["sha256"], "historical input semantic binding")
            require(np.array_equal(evidence.bits(transaction["output"]["path"], True), old[18]),
                    "historical final hidden trace")
            if layer:
                require(np.array_equal(old_input, previous[position]),
                        f"historical hidden lineage {position}/{layer}")
                hidden_edges.append({"position": position, "from_layer": layer - 1,
                                     "to_layer": layer, "input": transaction["vectors"]["input"],
                                     "stage18_exact": True})
            previous[position] = old[18]
            require(np.array_equal(old[5], old[6]) and np.array_equal(old[3], old[7]),
                    "historical cache write identity")
            evidence.read(transaction["output_state"]["path"])
            require(transaction["output_state"] == parent["position_states"][position],
                    "historical state output binding")
            if position == 2:
                require(transaction["output_state"] == parent["state"], "position2 consumed state")
            for kind, stage in (("k", 6), ("v", 7)):
                require(independent_old[kind].shape == (3, 2, 64), "independent cache geometry")
                require(np.array_equal(old[stage].reshape(2, 64),
                                       factorial[f"actual_cache_{kind}"][position])
                        and np.array_equal(independent_old[kind],
                                           factorial[f"independent_old_{kind}"]),
                        "historical factorial operand")
                history_rows.append({
                    "position": position, "layer": layer, "stage": stage, "kind": kind,
                    "comparison": measurement(old[stage], independent_old[kind][position]),
                    "actual_trace": trace_record, "independent_cache": ref["own_cache"][kind],
                    "independent_slice": [position, ":", ":"],
                    "producer_chain": ([0, 2, 5, 6] if kind == "k" else [0, 3, 7]),
                })
            available = {}
            pattern = re.compile(
                rf"/trajectory/layer{layer:02d}/position{position:03d}/stage(\d\d)\.npy$")
            for path in evidence.bindings:
                match = pattern.search(path)
                if match:
                    stage = int(match[1])
                    require(stage not in available, "ambiguous independent stage source")
                    available[stage] = path
            coverage.append({"position": position, "layer": layer,
                             "sealed_independent_operator_stages": sorted(available),
                             "cache_slices_available": ["k", "v"],
                             "boundary": "Only referenced sealed indices searched, not unbound neighboring files."})
            for stage, path in sorted(available.items()):
                independent = evidence.array(path)
                if stage in (6, 7):
                    kind = "k" if stage == 6 else "v"
                    require(np.array_equal(independent.reshape(2, 64),
                                           independent_old[kind][position]), "oracle cache stage edge")
                stage_rows.append({
                    "position": position, "layer": layer, "stage": stage,
                    "operator": OPERATORS[stage], "comparison": measurement(old[stage], independent),
                    "actual_trace": trace_record, "independent_stage": evidence.consumed[path],
                    "same_input_boundary": "Independent trajectory; no historical same-input intervention measured.",
                })
        layer_records.append({"layer": layer, "parent": parent,
                              "historical_result": parent["layer_result"],
                              "softmax_report": record(SOFTMAX / f"layer{layer:02d}.json"),
                              "factorial_report": record(FACTORIAL / f"layer{layer:02d}.json")})

    passing = []
    for name, branch in branch_results.items():
        clear = not branch["layer08_stage08"]["failure_indices"] and not branch["affected_stage_failures"]
        branch["endpoint_and_affected_gates_clear"] = clear
        require(core(branch["layer08_stage08"]) == core(factor_result["cases"][name]["layer08_stage08"])
                and clear == factor_result["cases"][name]["endpoint_and_affected_gates_clear"],
                f"factorial semantic verdict {name}")
        if clear:
            passing.append(name)
    require(sorted(passing) == sorted(factor_result["passing_conditional_cases"]), "passing case set")
    require(branch_results["exp0_later_k0_v0"]["layer08_stage08"]["failure_indices"] == [13, 15],
            "original failure changed")
    stage_rows.sort(key=lambda row: (row["position"], row["layer"], row["stage"]))
    first_by_position = {}
    for row in stage_rows:
        if row["comparison"]["different_count"]:
            first_by_position.setdefault(str(row["position"]), row)
    witnesses = [row for row in stage_rows
                 if row["position"] == 3 and row["layer"] == 8 and row["stage"] == 8]
    production_differences = [row for row in stage_rows
                              if row["position"] == 3
                              and row["same_input_retained_comparison"] is not None
                              and row["same_input_retained_comparison"]["different_count"]]
    unique_history_alternatives = [name for name in passing if name.startswith("exp0_")]
    require(len(unique_history_alternatives) > 1, "nonunique history evidence no longer holds")
    write_json(out / "operators.json", {"stages": stage_rows, "historical_kv": history_rows,
                                      "hidden_edges": hidden_edges, "coverage": coverage,
                                      "layers": layer_records})
    write_json(out / "factorial_remeasurement.json", {"cases": branch_results,
                                                     "passing_cases": passing,
                                                     "retained_result": record(FACTORIAL / "result.json")})
    write_json(out / "result.json", {
        "status": "BOUNDED_PARENTAGE_OPERATOR_TRACE_COMPLETE_NONUNIQUE_PRODUCERS",
        "first_observed_bit_divergence_by_position": first_by_position,
        "first_position3_same_input_operator_divergence": production_differences[0],
        "layer08_stage08": witnesses,
        "historical_substitution_alternatives_without_exponential_change": unique_history_alternatives,
        "operator_comparisons": len(stage_rows), "historical_cache_comparisons": len(history_rows),
        "hidden_edges": len(hidden_edges), "factorial_cases_remeasured": len(branch_results),
        "missing_operator_coverage": [row for row in coverage
                                      if len(row["sealed_independent_operator_stages"]) != 19],
        "missing_stage_local_proof": [
            {"position": row["position"], "layer": row["layer"], "stage": row["stage"],
             "proof": row["same_input_evidence"]}
            for row in stage_rows if row["position"] == 3
            and row["same_input_retained_comparison"] is None
        ],
        "causal_boundary": (
            "Earliest recorded bit divergence is not the earliest cause of the endpoint. "
            "Historical K/V interventions replace whole independently propagated tensors; "
            "they do not isolate hidden input, RMSNorm, K/V projection or K RoPE arithmetic. "
            "Multiple distinct history substitutions clear the same endpoint and affected gates. "
            "Where full stage trajectories exist they reveal trajectory differences, not "
            "source-specific historical counterfactual sufficiency. Missing sealed intermediate "
            "stages are listed explicitly. The unique earliest production cause remains unresolved."
        ),
        "hypothesis_regression": {
            "taxonomy": "historical_state_and_operator_policy_confounding",
            "hypothesis": "More than one upstream producer can explain the retained endpoint.",
            "regression": "All 52 sealed branches and their affected-stage gates recomputed without replay.",
        },
        "candidate_selection": {
            "candidate": json_input(evidence, SOFTMAX / "frozen.json")["candidate"],
            "evidence_supports_planner_selection_after_independent_review": True,
            "scope": "Isolated general layer0 stage09 exponential RTL candidate; not uniquely localized repair.",
            "review_status": "Host independent Reviewer pending; no verdict inferred from sealed bytes.",
        },
        "production_intervention_authorized": False,
        "fresh_official_repaired_attempt_authorized": False,
        "required_before_official_traversal": (
            "Independent candidate RTL/source-bound operator review and affected-prefix/KV "
            "invalidation/regeneration requirements; retained mixed histories are not rollout evidence."
        ),
        "RTL_invocations": 0, "official_RTL_repair_attempts": 0,
        "production_changes": 0, "prefix_replays": 0, "acceptance_policy_changes": 0,
        "third_token_claim": False, "review": "required; next_owner=reviewer",
        "authenticated_inputs": sorted(evidence.consumed.values(), key=lambda item: item["path"]),
        "seconds": time.monotonic() - started,
    })
    write_json(out / "seal.json", {"files": [record(path) for path in sorted(out.iterdir())]})
    print(json.dumps({"status": "BOUNDED_PARENTAGE_OPERATOR_TRACE_COMPLETE_NONUNIQUE_PRODUCERS",
                      "output": str(out), "operator_comparisons": len(stage_rows),
                      "historical_cache_comparisons": len(history_rows),
                      "factorial_cases_remeasured": len(branch_results),
                      "first_by_position": {p: [r["layer"], r["stage"]]
                                            for p, r in first_by_position.items()},
                      "official_RTL_repair_attempts": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
