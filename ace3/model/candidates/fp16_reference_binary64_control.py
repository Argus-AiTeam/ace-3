"""Retained, fixed-history reference control; never executes RTL or a model."""

from __future__ import annotations

import argparse
import ast
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import time

import numpy as np

from ace3.model.candidates import binary64_fp16_excess_v1 as profile

ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "build/layer08_position2_q_only_rope_rtl_86938809f0a2_attempt010"
OLD = ROOT / "build/layer08_position3_softmax_exp_replay_attempt005"
LEGACY = OLD / "binary64_evaluation_attempt001"
PROFILE = ROOT / "build/binary64_fp16_excess_v1/attempt001"
RETRO = ROOT / "build/binary64_fp16_excess_v1/retrospective_attempt002"
HANDOFFS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
CASES = [(layer, position) for layer in range(9) for position in range(3)]
CSV_SHA = "1ba6a66a74a34730da6e5667c1ca68779fa86723e0f6d43eab19fc3b58822e2e"
B_FREEZE_SHA = "03ab838054fe8318d369e8f260801c1e0193ad1c43720133afb96fbb5b90f785"
PROFILE_FREEZE_SHA = "dc13832db133d9c701d69fb5a95c90b89f37ced56785f6c5a8df20fb3481206b"
LOGICAL_SOURCE = ROOT / "ace3/model/run_final_rmsnorm_from_layer23.py"
ARCHIVED_SOURCE = Path(
    "/home/argustest/argustest2/ace3-continuous-generation-20260906/"
    "ace3/model/run_final_rmsnorm_from_layer23.py"
)


class InputBlocker(ValueError):
    """An authenticated input or semantic prerequisite is unsatisfied."""


def require(condition, reason):
    if not condition:
        raise InputBlocker(reason)


def load(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def write(path, value):
    with Path(path).open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def records(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= value.keys():
            yield value
        for item in value.values():
            yield from records(item)
    elif isinstance(value, list):
        for item in value:
            yield from records(item)


def half(bits):
    return Fraction.from_float(struct.unpack(">e", int(bits).to_bytes(2, "big"))[0])


def helper(path, name, namespace):
    # Isolate the existing pure helper, without importing its model/RTL runtime.
    source = Path(path).read_text(encoding="ascii")
    selected = [node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(selected) == 1, f"helper definition missing or ambiguous: {path}:{name}")
    exec(ast.get_source_segment(source, selected[0]), namespace)
    return namespace[name]


def run(out):
    started = time.monotonic()
    out.mkdir()
    bindings = {}
    observations = {}
    stamps = {}
    source_equivalences = []

    def authenticate(item):
        path = item["path"]
        if path not in observations:
            observations[path] = record(path)
            stat = Path(path).stat()
            stamps[path] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        observed = observations[path]
        require(observed["bytes"] == item["bytes"]
                and observed["sha256"] == item["sha256"],
                f"binding mismatch: expected={item!r}; observed={observed!r}")
        bindings[path] = observed
        return observed

    def read_bound(item):
        authenticate(item)
        return load(item["path"])

    def bind_path(path):
        return authenticate(record(path))

    command = ["env", f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}",
               sys.executable, "-B", "-m",
               "ace3.model.candidates.fp16_reference_binary64_control",
               "--out", str(out)]
    source = Path(__file__).resolve()
    with (out / "source.py").open("xb") as stream:
        stream.write(source.read_bytes())
    registration = {
        "command": command, "cwd": str(Path.cwd()), "source": record(source),
        "source_snapshot": record(out / "source.py"),
        "python": record(sys.executable), "python_version": sys.version,
        "numpy_version": np.__version__,
        "ordered_cases": CASES, "coordinate_order": "layer, position, index",
        "expected_outputs": 27, "expected_coordinates": 24192,
        "domain": "candidate B layers0-8/positions0-2; fixed tokens [9707,1879,0]",
        "intended_analysis": [
            "Actual B vs retained independent FP16 stage18, original finite stage gate.",
            "Independent FP16 stage18 vs original binary64, separately labeled v1 control.",
            "Actual B vs original binary64, unchanged v1.",
            "All coordinates; exact errors, q, excess, bits, first/full failures.",
            "Separate actual/reference bit identities from implementation deviations.",
            "Cross-check v1 with independent struct binary16 RNE and exact fractions.",
        ],
        "stage_gate": "finite AND (abs<=0.125 OR (relative<0.001 AND ordered_ULP<=1))",
        "relative_denominator": "max(abs(reference),2^-14)",
        "profile_id": profile.PROFILE_ID, "excess_budget": "1/8",
        "source_equivalence_plan": {
            "original_path": str(LOGICAL_SOURCE),
            "archive_path": str(ARCHIVED_SOURCE),
            "requirement": (
                "Both source identities must already be bound in the original binary64 "
                "freeze, with identical bytes and SHA256 to B's frozen source. Authenticate "
                "the archived bytes, never import them or change the live source/prior seals."
            ),
        },
        "no_replay_no_adoption": True,
        "review": "Normal independent Host Reviewer after Engineer handoff; not fabricated here.",
    }
    write(out / "preregistration.json", registration)
    measured = 0
    try:
        for item in records(registration):
            authenticate(item)
        b_result = load(B / "result.json")
        bind_path(B / "result.json")
        require(b_result["freeze"]["sha256"] == B_FREEZE_SHA, "wrong B freeze selector")
        b_freeze = read_bound(b_result["freeze"])
        require(b_freeze["selectors"] == {
            "layers": list(range(9)), "positions": list(range(3)), "stages": list(range(19))
        }, "B selectors differ from the fixed diagnostic domain")
        require(b_freeze["token_history"] == [9707, 1879, 0]
                and b_result["status"] == "PASS_BOUNDED_RTL_CANDIDATE"
                and b_result["independent_stage_comparisons"] == 513
                and not b_result["binary64_profile_adoption"],
                "B scope/status/policy binding mismatch")
        require(b_freeze["comparison_policy"] == {
            "absolute_tolerance": 0.125,
            "accept": "finite AND (abs_error<=0.125 OR (relative_error<0.001 AND orderedFP16_ULP<=1))",
            "max_ulp_distance": 1,
            "relative_denominator": "max(abs(reference),2^-14)",
            "relative_tolerance_strict": 0.001,
        }, "original FP16 gate changed")
        reviews = []
        for mission, round_ in [("77a1655e196f", 1), ("86938809f0a2", 9),
                                ("83fae4b2cbc6", 1)]:
            path = HANDOFFS / mission / f"round-{round_:04d}.json"
            receipt = load(path)
            require(receipt["kind"] == "round_reviewed_handoff"
                    and receipt["producer_role"] == "reviewer"
                    and receipt["mission_id"] == mission
                    and receipt["review"]["status"] == "done",
                    f"genuine existing review missing: {path}")
            reviews.append(bind_path(path))
        pf_binding = bind_path(PROFILE / "freeze.json")
        require(pf_binding["sha256"] == PROFILE_FREEZE_SHA, "reviewed profile freeze drift")
        pf = load(PROFILE / "freeze.json")
        for item in pf["sources"] + pf["source_snapshots"]:
            authenticate(item)
        profile_result = load(PROFILE / "result.json")
        bind_path(PROFILE / "result.json")
        require(profile_result["freeze_sha256"] == PROFILE_FREEZE_SHA
                and profile_result["status"] == "PASS_GENERAL_SCALAR_CONTRACT_TESTS",
                "profile result does not bind reviewed source")
        retro_result = load(RETRO / "result.json")
        bind_path(RETRO / "result.json")
        retro = read_bound(retro_result["freeze"])
        legacy_result = load(LEGACY / "result.json")
        bind_path(LEGACY / "result.json")
        legacy = load(LEGACY / "frozen.json")
        bind_path(LEGACY / "frozen.json")
        require(legacy["policy"] == profile.REFERENCE_POLICY
                and legacy["absolute_tolerance"] == 0.125
                and legacy_result["binary64_available_scope_status"] == "FAIL",
                "original binary64 policy/status mismatch")
        csv_binding = legacy_result["full_binary64_deltas"]
        require(csv_binding["sha256"] == CSV_SHA, "wrong original binary64 CSV")
        authenticate(csv_binding)
        reference_root = read_bound(b_freeze["reference_source"])
        require(reference_root["checkpoint"] == b_freeze["checkpoint"]
                and reference_root["token_history"][:3] == b_freeze["token_history"]
                and reference_root["embeddings"][:3] == b_freeze["embeddings"],
                "FP16 reference and B checkpoint/token/embedding roots incompatible")
        require(reference_root["trajectory_oracle"] ==
                "Unmodified independently propagated FP16-interstage reference; own hidden/KV"
                and reference_root["profile"] == {
                    "activations": "FP16", "kv": "FP16", "scales": "FP16",
                    "weights": "G128 asymmetric packed INT4; native GEMM nibble order; no qzero +1",
                }, "declared FP16 reference arithmetic/profile mismatch")
        legacy_bindings = {item["path"]: item for item in legacy["inputs_and_sources"]}
        for item in [b_freeze["reference_source"], b_freeze["checkpoint"]]:
            require(legacy_bindings.get(item["path"]) == item,
                    f"binary64 and FP16 root mismatch: {item['path']}")
            authenticate(item)
        for group in ("source_closure", "retained_source_closure"):
            for item in b_freeze[group]:
                if item["path"] == str(LOGICAL_SOURCE):
                    archived = legacy_bindings.get(str(ARCHIVED_SOURCE))
                    require(legacy_bindings.get(item["path"]) == item
                            and archived is not None
                            and all(archived[name] == item[name] for name in ("bytes", "sha256")),
                            "historical source equivalent lacks matching original frozen bindings")
                    source_equivalences.append({
                        "original_frozen_binding": item,
                        "authenticated_archive": authenticate(archived),
                        "provenance": bindings[str(LEGACY / "frozen.json")],
                        "scope": "Exact historical source bytes only; no source execution or adoption.",
                    })
                else:
                    authenticate(item)
        require(len(source_equivalences) == 1, "missing or duplicate source equivalence")
        authenticate(b_freeze["script"])
        authenticate(b_freeze["delta_exporter"])
        authenticate(b_freeze["retained_result"])
        authenticate(legacy["script"])
        archived_sources = [item for item in legacy["inputs_and_sources"]
                            if "ace3-continuous-generation-20260906/" in item["path"]
                            and item["path"].endswith(".py")]
        require(archived_sources, "binary64 source archive absent")
        for item in archived_sources:
            authenticate(item)

        parser_path = ROOT / "ace3/model/validate_selected_token_position2_traversal.py"
        order_path = ROOT / "ace3/model/replay_layer00_q_projection_repair.py"
        require(str(parser_path) in bindings and str(order_path) in bindings,
                "existing parser/comparison helper not source-bound")
        namespace = {"np": np, "Path": Path, "HIDDEN_SIZE": 896, "require": require}
        load_bits = helper(parser_path, "load_hidden_bits", namespace)
        ordered = helper(order_path, "ordered_f16", namespace)

        def words(item, raw):
            authenticate(item)
            lines = Path(item["path"]).read_text("ascii").splitlines()
            width = 10 if raw else 4
            require(len(lines) == 896 and all(
                re.fullmatch(f"[0-9a-fA-F]{{{width}}}", line) for line in lines
            ), f"invalid FP16 record encoding: {item['path']}")
            value = (load_bits(Path(item["path"])) if raw else
                     np.array([int(line, 16) for line in lines], dtype="<u2"))
            require(np.isfinite(value.view("<f2")).all(), f"nonfinite FP16: {item['path']}")
            if "semantic_sha256" in item:
                require(hashlib.sha256(value.tobytes()).hexdigest() == item["semantic_sha256"],
                        f"FP16 semantic hash mismatch: {item['path']}")
            return value

        embeddings = []
        for item in b_freeze["embeddings"]:
            require(legacy_bindings.get(item["input"]["path"]) == item["input"],
                    "binary64 embedding root mismatch")
            value = words(item["input"], True)
            require(hashlib.sha256(value.tobytes()).hexdigest() == item["semantic_sha256"],
                    "embedding semantic hash mismatch")
            embeddings.append(value)
        with Path(b_freeze["checkpoint"]["path"]).open("rb") as stream:
            header_size = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(header_size))
            tensor = header["model.embed_tokens.weight"]
            require(tensor["dtype"] == "F16" and tensor["shape"][1] == 896,
                    "checkpoint embedding geometry/dtype mismatch")
            for token, value in zip(b_freeze["token_history"], embeddings):
                stream.seek(8 + header_size + tensor["data_offsets"][0] + token * 896 * 2)
                require(stream.read(896 * 2) == value.tobytes(),
                        f"token {token} embedding differs from checkpoint")

        ref_transactions = {(x["layer"], x["position"]): x
                            for x in b_freeze["reference_transactions"]}
        transactions = {(x["layer"], x["position"]): x for x in b_result["transactions"]}
        require(len(b_result["transactions"]) == len(b_freeze["reference_transactions"]) == 27
                and set(transactions) == set(ref_transactions) == set(CASES),
                "duplicate or missing transaction selectors")
        array_bindings = {item["path"]: item for item in records(retro)
                          if item["path"].endswith("_reference.npy")}
        data, frozen_cases = {}, []
        with Path(csv_binding["path"]).open(newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        identities = [(int(x["layer"]), int(x["position"]), int(x["index"])) for x in csv_rows]
        full_domain = {(l, p, i) for l in range(9)
                       for p in range(3 if l == 8 else 4) for i in range(896)}
        require(len(identities) == len(set(identities)) == 35 * 896
                and set(identities) == full_domain, "binary64 CSV coverage/uniqueness mismatch")
        csv_values = {key: row["reference_binary64_hex"]
                      for key, row in zip(identities, csv_rows)}
        for layer, position in CASES:
            key = layer, position
            result = read_bound(transactions[key]["result"])
            prepared = read_bound(result["prepared"])
            ref_tx = ref_transactions[key]
            require(prepared["reference_parent"] == ref_tx["parent"]
                    and prepared["independent_stages"] == ref_tx["stages"]
                    and result["independent_stages"] == ref_tx["stages"],
                    f"FP16 selector/parent binding mismatch: {key}")
            parent = read_bound(ref_tx["parent"])
            require(parent["independent_stages"] == ref_tx["stages"],
                    f"FP16 reference source receipt disagreement: {key}")
            reference = ref_tx["stages"]["18"]
            actual = result["transaction"]["output"]
            require(result["transaction"]["layer_index"] == layer
                    and result["transaction"]["position"] == position,
                    f"actual transaction coordinates mismatch: {key}")
            reference_input = (b_freeze["embeddings"][position]["input"] if layer == 0
                               else ref_transactions[layer - 1, position]["stages"]["18"])
            require(prepared["independent_input"] == reference_input,
                    f"FP16 independent hidden parent mismatch: {key}")
            authenticate(reference_input)
            require(len(prepared["independent_history"]) == position,
                    f"FP16 history length mismatch: {key}")
            for previous, history in enumerate(prepared["independent_history"]):
                expected = ref_transactions[layer, previous]
                require(history["parent"] == expected["parent"]
                        and history["stages"] == expected["stages"],
                        f"FP16 causal K/V history mismatch: {key}/{previous}")
                for stage in ("6", "7"):
                    authenticate(history["stages"][stage])
            actual_input = (b_freeze["embeddings"][position]["input"] if layer == 0
                            else data[layer - 1, position]["actual_binding"])
            require(all(prepared["input_hidden"][name] == actual_input[name]
                        for name in ("path", "bytes", "sha256")),
                    f"B actual hidden parent mismatch: {key}")
            authenticate(prepared["input_hidden"])
            expected_state = None if position == 0 else data[layer, position - 1]["state"]
            require(prepared["input_state"] == expected_state,
                    f"B state lineage mismatch: {key}")
            authenticate(prepared["binary"])
            authenticate(result["transaction"]["output_state"])
            a, f = words(actual, True), words(reference, False)
            array_path = str(LEGACY / f"layer{layer:02d}_position{position:03d}_reference.npy")
            require(array_path in array_bindings, f"unbound binary64 array: {array_path}")
            authenticate(array_bindings[array_path])
            r = np.load(array_path, allow_pickle=False)
            require(r.dtype == np.dtype("float64") and r.shape == (896,)
                    and np.isfinite(r).all() and (np.abs(r) <= 65504).all(),
                    f"binary64 dtype/geometry/range mismatch: {key}")
            hexes = [csv_values[layer, position, i] for i in range(896)]
            require(all(float(value).hex() == text for value, text in zip(r, hexes)),
                    f"original CSV/array binary64 bit disagreement: {key}")
            data[key] = {
                "actual": a, "fp16": f, "binary64": hexes, "actual_binding": actual,
                "state": result["transaction"]["output_state"],
                "stage18": next(x for x in result["comparisons"] if x["stage"] == 18),
            }
            frozen_cases.append({
                "layer": layer, "position": position, "indices": list(range(896)),
                "actual": actual, "fp16_reference": reference,
                "binary64_reference_array": array_bindings[array_path],
                "result": transactions[key]["result"], "prepared": result["prepared"],
                "fp16_reference_parent": ref_tx["parent"],
            })
        require(ref_transactions[3, 0]["stages"]["18"]["sha256"] ==
                "2e6119b09493cb118e6d64aa985fad2c09d0fe6e9d89d4af81afd2744eafb062",
                "operator-selected FP16 reference witness file mismatch")
        freeze = {
            **registration, "preregistration": record(out / "preregistration.json"),
            "bindings": list(bindings.values()), "cases": frozen_cases,
            "source_equivalences": source_equivalences,
            "existing_reviews": reviews,
            "checkpoint": b_freeze["checkpoint"], "embeddings": b_freeze["embeddings"],
            "fp16_reference_root": b_freeze["reference_source"],
            "binary64_reference_freeze": record(LEGACY / "frozen.json"),
            "binary64_array_binding_freeze": retro_result["freeze"],
            "binary64_csv": csv_binding,
            "comparability": (
                "Same authenticated checkpoint and original embeddings/tokens; causal prefix "
                "of the retained independent histories. Different declared precision trajectories, "
                "not identical arithmetic or official CUDA/PyTorch kernel execution. No reference "
                "regeneration, actual-operand re-anchoring, or candidate-native greedy claim."
            ),
        }
        write(out / "freeze.json", freeze)
        columns = ["actual_vs_fp16_stage_gate", "fp16_reference_vs_binary64_v1",
                   "actual_vs_binary64_v1"]
        failures = {name: [] for name in columns}
        identical, deviations, oracle_count = 0, 0, 0
        case_summaries = []
        with (out / "all_coordinates.jsonl").open("x", encoding="ascii") as stream:
            for layer, position in CASES:
                item = data[layer, position]
                order_a, order_f = ordered(item["actual"]), ordered(item["fp16"])
                case_failures = {name: 0 for name in columns}
                case_identical = 0
                for index, (a_bits, f_bits, r_hex) in enumerate(zip(
                    item["actual"], item["fp16"], item["binary64"]
                )):
                    a_bits, f_bits = int(a_bits), int(f_bits)
                    a, f, r = half(a_bits), half(f_bits), Fraction.from_float(float.fromhex(r_hex))
                    difference = abs(a - f)
                    relative = difference / max(abs(f), Fraction(1, 16384))
                    ulp = abs(int(order_a[index]) - int(order_f[index]))
                    stage_pass = (float(difference) <= 0.125 or
                                  (float(relative) < 0.001 and ulp <= 1))
                    metrics = {}
                    nearest = int.from_bytes(struct.pack(">e", float.fromhex(r_hex)), "big")
                    oracle_q = abs(half(nearest) - r)
                    for name, bits, value in [("fp16_reference_vs_binary64_v1", f_bits, f),
                                              ("actual_vs_binary64_v1", a_bits, a)]:
                        metric = profile.evaluate_layer_final_output(
                            actual_fp16_bits=bits, reference_binary64_hex=r_hex)
                        oracle_error = abs(value - r)
                        require(Fraction(metric["actual_error"]) == oracle_error
                                and Fraction(metric["q"]) == oracle_q
                                and Fraction(metric["excess_error"]) == oracle_error - oracle_q
                                and metric["accepted"] == (oracle_error - oracle_q <= Fraction(1, 8))
                                and metric["nearest_fp16_bits"] == f"{nearest:04x}",
                                f"independent v1 arithmetic disagreement: {layer}/{position}/{index}/{name}")
                        oracle_count += 1
                        metrics[name] = {k: metric[k] for k in
                                         ("actual_error", "q", "excess_error", "accepted")}
                    same = a_bits == f_bits
                    identical += same
                    deviations += not same
                    case_identical += same
                    row = {
                        "layer": layer, "position": position, "stage": 18, "index": index,
                        "actual_bits": f"{a_bits:04x}", "fp16_reference_bits": f"{f_bits:04x}",
                        "actual_fraction": str(a), "fp16_reference_fraction": str(f),
                        "reference_binary64_hex": r_hex, "reference_binary64_fraction": str(r),
                        "nearest_fp16_bits": f"{nearest:04x}", "actual_equals_fp16_bits": same,
                        "actual_vs_fp16_stage_gate": {
                            "actual_error": str(difference), "relative_error": str(relative),
                            "relative_error_binary64_hex": float(relative).hex(),
                            "ordered_fp16_ulp": ulp, "finite": True, "accepted": stage_pass,
                        }, **metrics,
                    }
                    for name in columns:
                        if not row[name]["accepted"]:
                            failures[name].append(row)
                            case_failures[name] += 1
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    measured += 1
                require(case_identical == item["stage18"]["exact_match_count"]
                        and case_failures[columns[0]] == item["stage18"]["failure_count"],
                        f"original stage18 receipt disagreement: {layer}/{position}")
                case_summaries.append({
                    "layer": layer, "position": position, "coordinates": 896,
                    "bit_identical": case_identical, "failure_counts": case_failures,
                })
        require(measured == 24192 and identical + deviations == measured
                and oracle_count == 2 * measured, "incomplete control census")
        changed = [path for path, stamp in stamps.items()
                   if (Path(path).stat().st_size, Path(path).stat().st_mtime_ns,
                       Path(path).stat().st_ino) != stamp]
        require(not changed, f"bound input changed during read-only control: {changed}")
        summaries = {
            name: {
                "coordinates": measured, "failure_count": len(rows),
                "failing_outputs": len({(r["layer"], r["position"]) for r in rows}),
                "first_failure": rows[0] if rows else None,
                "failures_with_identical_actual_and_fp16": sum(r["actual_equals_fp16_bits"] for r in rows),
                "status": "FAIL" if rows else "PASS",
            } for name, rows in failures.items()
        }
        write(out / "failures.json", failures)
        result = {
            "status": "COMPLETE_FIXED_HISTORY_REFERENCE_CONTROL",
            "review": "PENDING_NORMAL_INDEPENDENT_HOST_REVIEWER",
            "freeze": record(out / "freeze.json"),
            "all_coordinates": record(out / "all_coordinates.jsonl"),
            "full_failures": record(out / "failures.json"),
            "coverage": {"outputs": 27, "coordinates": measured, "per_output": 896},
            "actual_fp16_bit_identical": identical, "actual_fp16_bit_deviations": deviations,
            "comparisons": summaries, "per_output": case_summaries,
            "independent_struct_fraction_comparisons": oracle_count,
            "authenticated_files": len(bindings), "bound_input_metadata_unchanged": True,
            "numerical_admission": False, "policy_adoption": False,
            "rtl_or_model_replays": 0, "elapsed_seconds": time.monotonic() - started,
            "interpretation": (
                "A rejecting declared FP16 reference establishes that v1 imposes additional "
                "binary64 fidelity at those coordinates, not an RTL deviation at bit-identical "
                "coordinates. Neither a defective oracle/policy nor a fully correct RTL graph "
                "follows. The independent-reference label is numerical, not official kernel execution."
            ),
            "recommendation_to_parent": {
                "current_disposition": "Keep all existing gates/verdicts; no adoption in this task.",
                "retain_v1": (
                    "Keep v1 mandatory for any claim of stronger binary64 fidelity. Passing it "
                    "requires a justified general implementation/declared-arithmetic improvement "
                    "and all unchanged stage gates, with affected-history validation. Scientific "
                    "cost: native-FP16 conformance alone is insufficient and may be rejected; "
                    "correct local RNE must not be replaced by arbitrary witness-specific values."
                ),
                "future_versioned_separation": (
                    "If the parent instead intends native-FP16 conformance as its admission claim, "
                    "explicitly version a policy separating that claim from binary64 fidelity, "
                    "retain both measurements and legacy failures, and obtain independent review. "
                    "Scientific cost: this is a weaker admission claim, not improved arithmetic "
                    "or evidence of binary64 fidelity, model fidelity, or useful dialogue."
                ),
                "unproven": [
                    "Universal satisfiability or impossibility of both gates.",
                    "Complete RTL graph correctness or a specific implementation/oracle defect.",
                    "Candidate-native greedy history, layer9-23/tail compatibility, third token.",
                    "Whole-model fidelity and generalization beyond this fixed-history domain.",
                ],
            },
        }
        write(out / "result.json", result)
        print(json.dumps({k: result[k] for k in (
            "status", "coverage", "actual_fp16_bit_identical", "actual_fp16_bit_deviations",
            "comparisons", "authenticated_files", "elapsed_seconds")}, sort_keys=True))
        return 0
    except (InputBlocker, FileNotFoundError, PermissionError) as error:
        result = {
            "status": "AUTHENTICATED_INPUT_OR_SEMANTIC_BLOCKER",
            "reason": str(error), "error_type": type(error).__name__,
            "coordinates_measured": measured,
            "preregistration": record(out / "preregistration.json"),
            "authenticated_bindings": list(bindings.values()),
            "observed_bindings": list(observations.values()),
            "source_equivalences": source_equivalences,
            "review": "PENDING_NORMAL_INDEPENDENT_HOST_REVIEWER",
            "numerical_admission": False, "policy_adoption": False,
        }
        write(out / "result.json", result)
        print(json.dumps({k: result[k] for k in
                          ("status", "reason", "error_type", "coordinates_measured")}))
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.parent == ROOT / "build"
            and re.fullmatch(r"fp16_reference_binary64_control_attempt[0-9]+", out.name),
            "output must be a fresh isolated reference-control attempt under build/")
    require(not out.exists(), "refusing to replace an existing attempt")
    return run(out)


if __name__ == "__main__":
    raise SystemExit(main())
