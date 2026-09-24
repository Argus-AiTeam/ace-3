#!/usr/bin/env python3
"""Bounded retained-data numerical interventions, never simulator state."""

import argparse
from bisect import bisect_left
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib
import json
from math import isqrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from retained_layer00_stage18_producer_split_diagnostic import record, require


def records(value):
    if isinstance(value, dict):
        if all(key in value for key in ("path", "bytes", "sha256")):
            yield value
        else:
            for item in value.values():
                yield from records(item)
    elif isinstance(value, list):
        for item in value:
            yield from records(item)


def compare(actual, reference, boundary):
    require(len(actual) == len(reference) and actual, "comparison geometry")
    failures = []
    for index, (a, r) in enumerate(zip(actual, reference, strict=True)):
        error = abs(boundary.units(a) - boundary.units(r))
        if not boundary.accepts(a, r):
            failures.append(dict(
                index=index, actual=f"{a:04x}", reference=f"{r:04x}",
                absolute_error_exact=str(Fraction(error, 2**24)),
                relative_error_exact=str(Fraction(error, max(abs(boundary.units(r)), 1024))),
                ordered_FP16_ULP=abs(boundary.ordered(a) - boundary.ordered(r))))
    return dict(coordinates=len(actual), material_failures=len(failures), failures=failures,
                bit_differences=sum(a != r for a, r in zip(actual, reference, strict=True)))


def propagate_post_attention(root, norm, project, activate, residual, reference, boundary):
    """Stop a changed trajectory before consuming any failed stage."""
    stages = {12: root, 13: norm}
    comparisons = []
    for stage in range(13, 19):
        if stage in (14, 15, 17):
            stages[stage] = project(stage, stages[16 if stage == 17 else 13])
        elif stage == 16:
            stages[stage] = activate(stages[14], stages[15])
        elif stage == 18:
            stages[stage] = residual(root, stages[17])
        comparison = compare(stages[stage], reference[str(stage)], boundary)
        comparisons.append(dict(stage=stage, **comparison))
        if comparison["material_failures"]:
            return stages, comparisons, stage
    return stages, comparisons, None


def quotient_only_rmsnorm(activation, weights, epsilon_q48, boundary):
    """Keep the native mean/root; round the exact quotient directly to FP16."""
    require(len(activation) == len(weights) and activation, "RMSNorm geometry")
    xs = [boundary.units(x) for x in activation]
    ws = [boundary.units(w) for w in weights]
    sumsq = sum(x*x for x in xs)
    mean = round(Fraction(sumsq, len(xs))) + epsilon_q48
    root = isqrt(mean)
    require(root > 0, "RMSNorm zero divisor")
    result = []
    for a, w, x, g in zip(activation, weights, xs, ws, strict=True):
        quotient = Fraction(abs(x*g), root)
        require(quotient <= boundary.POSITIVE[-1], "RMSNorm finite range exceeded")
        upper = bisect_left(boundary.POSITIVE, quotient)
        bits = min({upper, max(0, upper-1)},
                   key=lambda b: (abs(boundary.POSITIVE[b]-quotient), b & 1))
        result.append(bits | ((a ^ w) & 0x8000))
    return result, dict(sumsq_q48=sumsq, mean_q48=mean, rms_q24=root)


def round_sqrt_ratio(product, mean):
    """Round abs(product)/sqrt(mean) to an integer using exact midpoint tests."""
    require(mean > 0, "RMSNorm zero divisor")
    square = product * product
    lower = isqrt(square // mean)
    midpoint = 4 * square - (2 * lower + 1)**2 * mean
    return lower + int(midpoint > 0 or (midpoint == 0 and lower % 2 == 1))


def root_only_rmsnorm(activation, weights, epsilon_q48, boundary):
    """Remove only the floor sqrt, retaining native mean and Q24 quotient RNE."""
    require(len(activation) == len(weights) and activation, "RMSNorm geometry")
    xs = [boundary.units(a) for a in activation]
    ws = [boundary.units(w) for w in weights]
    sumsq = sum(x*x for x in xs)
    mean = round(Fraction(sumsq, len(xs))) + epsilon_q48
    result, quotients = [], []
    for a, w, x, g in zip(activation, weights, xs, ws, strict=True):
        quotient = round_sqrt_ratio(x*g, mean)
        require(quotient <= boundary.POSITIVE[-1], "RMSNorm finite range exceeded")
        upper = bisect_left(boundary.POSITIVE, quotient)
        bits = min({upper, max(0, upper-1)},
                   key=lambda b: (abs(boundary.POSITIVE[b]-quotient), b & 1))
        result.append(bits | ((a ^ w) & 0x8000))
        quotients.append(-quotient if (a ^ w) & 0x8000 else quotient)
    return result, dict(sumsq_q48=sumsq, mean_q48=mean, rms_q24=isqrt(mean),
                        quotient_RNE_q24=quotients)


def projection_contributions(activation, tensors, index, boundary):
    """Exact native G128 GEMM contributions to one unbiased output."""
    require(activation and len(activation) % 128 == 0, "projection input geometry")
    groups = len(activation) // 128
    require(len(tensors["scales"]) % groups == 0, "projection scale geometry")
    count = len(tensors["scales"]) // groups
    require(count > 0 and count % 8 == 0 and 0 <= index < count, "projection output geometry")
    words = count // 8
    require(len(tensors["qweight"]) == len(activation)*words
            and len(tensors["qzeros"]) == groups*words, "packed projection geometry")
    shift = (0, 16, 4, 20, 8, 24, 12, 28)[index % 8]
    rows = []
    for i, bits in enumerate(activation):
        group = i // 128
        packed_w = tensors["qweight"][i*words + index//8]
        packed_z = tensors["qzeros"][group*words + index//8]
        require(0 <= packed_w < 2**32 and 0 <= packed_z < 2**32, "packed word range")
        w, z = (packed_w >> shift) & 15, (packed_z >> shift) & 15
        scale = tensors["scales"][group*count + index]
        coefficient = (w-z)*boundary.units(scale)
        rows.append(dict(index=i, group=group, activation=f"{bits:04x}",
                         qweight=w, qzero=z, scale=f"{scale:04x}",
                         coefficient_q24=coefficient,
                         contribution_q48=boundary.units(bits)*coefficient))
    return rows


DEPENDENCIES = {0: [], 1: [0], 2: [0], 3: [0], 4: [1], 5: [2], 6: [5], 7: [3],
                8: [4, 6], 9: [8], 10: [9, 7], 11: [10], 12: [11], 13: [12],
                14: [13], 15: [13], 16: [14, 15], 17: [16], 18: [12, 17]}
PROJECTIONS = {1: (0, "self_attn.q_proj"), 2: (0, "self_attn.k_proj"),
               3: (0, "self_attn.v_proj"), 11: (10, "self_attn.o_proj"),
               14: (13, "mlp.gate_proj"), 15: (13, "mlp.up_proj"),
               17: (16, "mlp.down_proj")}


def retained_layer_package(layer, build, load):
    """Normalize the two retained public launch-package schemas, not their RTL ABI."""
    require(1 <= layer <= 8, "retained package layer outside the bounded cone")
    if layer == 1:
        path = build / "token358_position3_layer1_preflight_attempt002/package/launch_package.json"
        package = load(path)
        require(package["layer_index"] == layer and package["position"] == 3,
                "layer01 launch-package coordinate mismatch")
        prepared = dict(package, independent_stages=package["independent_expected_stages"])
        transaction = build / "token358_position3_layer1_attempt002/evidence/transaction.json"
    else:
        attempt = "001" if layer == 2 else "003"
        directory = build / f"token358_position3_full_continuation_attempt{attempt}/layer{layer:02d}"
        path = directory / "prepared.json"
        prepared = load(path)
        transaction = directory / "transaction.json"
        if layer >= 3:
            result = load(directory / "result.json")
            require(result["layer"] == layer and result["position"] == 3
                    and result["transaction"]["path"] == str(transaction),
                    "retained transaction identity mismatch")
    require(prepared["kv_parent"]["layer_index"] == layer
            and prepared["kv_parent"]["valid_positions"] == [0, 1, 2]
            and "/model24_persistent_kv_selected_token_corrected_q_attempt005/"
            in prepared["kv_parent"]["state"]["path"],
            "retained A-lineage historical KV mismatch")
    require(set(prepared["independent_stages"]) == {str(s) for s in range(19)},
            "incomplete retained independent stages")
    return prepared, transaction, path


def paired_cone(actual, reference, incoming, candidate_incoming, operator, project,
                intervention, boundary, save, candidate_name="quotient_only"):
    """Reuse identical operands, retaining separate causal K/V for both arms."""
    require(candidate_name != "frozen_Q48", "candidate/control names must be distinct")
    caches = {arm: {"k": [], "v": []} for arm in ("frozen_Q48", candidate_name)}
    endpoints, comparisons = {}, []
    for p in sorted(actual):
        arms = {arm: {} for arm in caches}
        changed = {}
        for stage in range(19):
            control, candidate = arms["frozen_Q48"], arms[candidate_name]
            history = []
            if stage in (8, 10):
                key = "k" if stage == 8 else "v"
                history = [i for i in range(p)
                           if caches["frozen_Q48"][key][i] != caches[candidate_name][key][i]]
            parents = [s for s in DEPENDENCIES[stage] if changed[s]]
            root_changed = stage in (0, 12) and incoming[p] != candidate_incoming[p]
            intervene = stage == 0 and intervention is not None
            affected = bool(intervene or parents or history or root_changed)
            if stage in PROJECTIONS:
                operand = PROJECTIONS[stage][0]
                control[stage], candidate[stage] = project(
                    stage, control[operand], candidate[operand] if affected else control[operand])
            else:
                control[stage] = operator(stage, control, incoming[p], caches["frozen_Q48"], p)
                candidate[stage] = (
                    intervention(candidate_incoming[p], p) if intervene else
                    operator(stage, candidate, candidate_incoming[p], caches[candidate_name], p)
                    if affected else list(control[stage]))
            reconstruction = compare(control[stage], actual[p][stage], boundary)
            changed[stage] = control[stage] != candidate[stage]
            comparison = dict(
                position=p, stage=stage, recomputed=affected, changed_parents=parents,
                changed_history=history, changed_incoming=root_changed,
                reconstructed_vs_runtime=reconstruction,
                candidate_vs_reference=compare(candidate[stage], reference[p][stage], boundary),
                frozen_vs_reference=compare(control[stage], reference[p][stage], boundary),
                candidate_vs_frozen=compare(candidate[stage], control[stage], boundary))
            comparisons.append(comparison)
            if reconstruction["bit_differences"] or comparison["candidate_vs_reference"]["material_failures"]:
                failure = dict(
                    **comparison, failure_taxonomy="independent_FP16_trajectory_mismatch",
                    root_cause_hypothesis=f"The {candidate_name} perturbation propagates through "
                    "arm-local hidden/K/V and exceeds the unchanged independent stage gate.",
                    regression="Reproduce the first failing stage from these arm-local operands; "
                    "do not consume its failed output.",
                    operands={arm: {"incoming": incoming[p] if arm == "frozen_Q48"
                                    else candidate_incoming[p],
                                    "stages": values, "kv": caches[arm]}
                              for arm, values in arms.items()},
                    independent_reference=reference[p][stage])
                if reconstruction["bit_differences"]:
                    failure.update(
                        failure_taxonomy="frozen_runtime_reconstruction_mismatch",
                        root_cause_hypothesis="The software control does not reproduce the "
                        "authenticated runtime arithmetic on its own retained operands.",
                        regression="Reconstruct this exact control stage before any continuation.",
                        retained_runtime=actual[p][stage])
                save(p, dict(**arms, independent_retained=reference[p]))
                return endpoints, comparisons, failure
            if stage in (6, 7):
                key = "k" if stage == 6 else "v"
                for arm in arms:
                    caches[arm][key].append(list(arms[arm][stage]))
        endpoints[p] = {arm: values[18] for arm, values in arms.items()}
        save(p, dict(**arms, independent_retained=reference[p]))
    return endpoints, comparisons, None


def run_quotient_only(root, out, root_only=False, combined=False):
    require(not (root_only and combined), "select only one RMSNorm candidate")
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        with (out / name).open("x", encoding="ascii") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")

    bindings, consumed, conflicts = {}, {}, {}

    def register(value):
        for rec in records(value):
            if rec["path"] in bindings:
                if any(bindings[rec["path"]][k] != rec[k] for k in ("bytes", "sha256")):
                    conflicts.setdefault(rec["path"], [bindings[rec["path"]]]).append(rec)
                    continue
            bindings[rec["path"]] = rec

    def checked(path):
        path = Path(path)
        require(str(path) in bindings, "missing authenticated dependency: " + str(path))
        require(str(path) not in conflicts, "conflicting consumed binding: " + str(path))
        got = record(path)
        expected = bindings[str(path)]
        require(all(got[k] == expected[k] for k in ("path", "bytes", "sha256")),
                "changed retained dependency: " + str(path))
        consumed[str(path)] = got
        return path

    def load(path):
        obj = json.loads(checked(path).read_text())
        register(obj)
        return obj

    register(record(root / "artifacts.json"))
    load(root / "artifacts.json")
    freeze = load(root / "frozen.json")
    auth = load(root / "parent_authentication.json")
    review = load(auth["review"]["path"])
    require(review["kind"] == "round_reviewed_handoff"
            and review["producer_role"] == "reviewer"
            and review["mission_id"] == "d004a401c237"
            and review["review"]["status"] == "done", "reviewed discriminator required")
    coordinates = load(auth["reviewed_parent"]["coordinates.json"]["path"])
    candidate_name = "combined_mathematical" if combined else (
        "root_only" if root_only else "quotient_only")
    predecessor_roots, predecessor_reviews, predecessor_stages = {}, {}, {}
    predecessor_results, predecessor_accounting = {}, {}
    if root_only or combined:
        predecessors = ({
            "0fe3b4cb1657": ("retained_quotient_only_0fe3b4cb1657_attempt003", 2),
            "371d2a404cad": ("retained_root_only_371d2a404cad_attempt002", 1),
        } if combined else {
            "6eb2e0362261": ("retained_layer02_stage16_index813_localization_6eb2e0362261_attempt001", 1),
            "0fe3b4cb1657": ("retained_quotient_only_0fe3b4cb1657_attempt003", 2),
        })
        for mission, (directory, round_number) in predecessors.items():
            archive = root.parent / directory
            register(record(archive / "artifacts.json"))
            load(archive / "artifacts.json")
            predecessor_roots[mission] = consumed[str(archive / "artifacts.json")]
            receipt = Path(auth["review"]["path"]).parent.parent / mission / (
                f"round-{round_number:04d}.json")
            register(record(receipt))
            verdict = load(receipt)
            require(verdict["kind"] == "round_reviewed_handoff"
                    and verdict["producer_role"] == "reviewer"
                    and verdict["mission_id"] == mission
                    and verdict["review"]["status"] == "done", "predecessor review required")
            predecessor_reviews[mission] = consumed[str(receipt)]
            result = load(archive / "result.json")
            if combined:
                first = result["first_material_failure"]
                require(result["status"] == "CANDIDATE_REJECTED"
                        and (first["layer"], first["position"], first["stage"]) == (2, 0, 16)
                        and [r["index"] for r in first["comparison"]["failures"]] == [813]
                        and result["frozen_control_bit_differences"] == 0,
                        "reviewed single-factor rejection changed")
                predecessor_results[mission] = dict(
                    identity=consumed[str(archive / "result.json")], result=result)
                continue
            if mission == "6eb2e0362261":
                load(archive / "parent_authentication.json")
                stage_path = root.parent / (
                    "retained_layer02_rmsnorm_candidate_propagation_2cbdd5f727db_attempt002"
                    "/position0_stages.json")
            else:
                stage_path = archive / "layer02_position0_stages.json"
            predecessor_stages[mission] = (load(stage_path), consumed[str(stage_path)])
        if combined:
            archive = root.parent / "retained_root_only_371d2a404cad_attempt002"
            for label in ("0fe3b4cb1657", "root_only"):
                path = archive / f"layer02_position0_projection_{label}.json"
                load(path)
                predecessor_accounting[label] = consumed[str(path)]
    sources = out / "sources"
    sources.mkdir()
    for source in sorted((root / "sources").iterdir()):
        if source.suffix in (".py", ".sv"):
            with (sources / source.name).open("xb") as stream:
                stream.write(checked(source).read_bytes())
    with (sources / Path(__file__).name).open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(sources))
    boundary = importlib.import_module("diagnose_lineage_a_position3_boundary")
    adaptation = importlib.import_module("fp16_adaptation_oracle")
    exact = importlib.import_module("exact_projection")
    projection = importlib.import_module("projection_oracle")
    rope = importlib.import_module("qwen2_rope_oracle")
    attention = importlib.import_module("attention_oracle")
    rounding = importlib.import_module("trace_lineage_a_layer07_downproj")
    silu = importlib.import_module("trace_lineage_a_layer07_silu")
    mathematical = (importlib.import_module("trace_lineage_a_layer07_rmsnorm")
                    if combined else None)
    numpy = importlib.import_module("numpy")
    require(freeze["comparison_policy"] == boundary.POLICY
            and freeze["token_history"] == boundary.HISTORY, "frozen policy/history mismatch")
    # Historical freezes describe multiple live-source revisions. Only immutable
    # retained dependencies are consumed; conflicting paths remain inadmissible.
    expanded = set()
    while True:
        pending = sorted(p for p in bindings if p not in expanded
                         and Path(p).is_relative_to(root.parent)
                         and Path(p).name in ("frozen.json", "artifacts.json",
                                              "parent_authentication.json"))
        if not pending:
            break
        for path in pending:
            load(path)
            expanded.add(path)
    packages = {layer: retained_layer_package(layer, root.parent, load)
                for layer in range(1, 9)}
    supplemental_references = []
    for layer, (prepared, _, _) in packages.items():
        for generation in (0, 1):
            path = Path(prepared["independent_parent"]["path"]).with_name(
                f"layer{layer:02d}_generation{generation}.json")
            if str(path) not in bindings:
                rec = record(path)
                register(rec)
                supplemental_references.append(rec)
            history = load(path)
            require(history["layer"] == layer
                    and history["positions"] == ([0, 1] if generation == 0 else [2])
                    and history["history"] == boundary.HISTORY[:generation+2],
                    "retained reference history mismatch")
    tool = shutil.which("iverilog")
    require(tool is not None, "host iverilog unavailable")
    require(freeze["public_port_contract"] in
            (sources / "ace3_decoder_layer0_token_engine.sv").read_text(),
            "public port contract changed")
    preregistration = dict(
        candidate="layer00-stage00-no-Q24-quotient-v1",
        intervention="Q48 sumsq, native RNE mean and epsilon, floor Q24 sqrt unchanged; "
        "exact activation*weight/root quotient rounded once directly to binary16 RNE. "
        "No change to any other operator or reference.",
        hypothesis="Removing quotient double rounding alone can avoid the rejected "
        "fully single-round candidate's regression while improving the original L8/P3 boundary.",
        stop_rule="First candidate material stage failure or frozen-runtime reconstruction error. "
        "Never feed failed outputs forward. Layers ordered, positions ordered within layer.",
        original_endpoint=dict(layer=8, position=3, stage=8, indices=[13, 15]),
        comparison_policy=freeze["comparison_policy"], token_history=freeze["token_history"],
        public_port_contract=freeze["public_port_contract"], public_tops=freeze["public_tops"],
        public_parameters=freeze["public_parameters"], evaluator=record(sources / Path(__file__).name),
        source_manifest=[record(p) for p in sorted(sources.iterdir())],
        root_manifest=record(root / "artifacts.json"), predecessor_review=auth["review"],
        layer_packages={str(layer): consumed[str(package[2])]
                        for layer, package in packages.items()},
        supplemental_reference_content_roots=supplemental_references,
        unconsumed_historical_binding_conflicts=conflicts,
        tools=[record(tool), record(sys.executable)],
        provenance=freeze["provenance"],
        boundary="Retained authenticated A-lineage RTL paired with software candidate. "
        "Fixed tokens, not candidate-native greedy generation. Original positions0/1 "
        "independent reference input-consumption provenance remains unavailable. "
        "No new RTL candidate execution/admission or binary64-v1 evaluation.",
        numerical_policy_changed=False)
    if root_only:
        preregistration.update(
            candidate="layer00-stage00-no-floor-sqrt-v1",
            intervention="Keep native Q48 sumsq, RNE mean and epsilon. Replace floor Q24 sqrt "
            "only by the exact positive sqrt of that same mean; round activation*weight/sqrt(mean) "
            "to Q24 RNE using squared midpoint comparisons, then retain binary16 RNE and signed zero. "
            "Only layer00 stage00 changes; all other operators and references remain frozen.",
            hypothesis="The reviewed root-only factor may avoid the L2/P0/S16/index813 regression "
            "of both full single-round and quotient-only variants while improving L8/P3.",
            candidate_set=["root_only"], predecessor_roots=predecessor_roots,
            predecessor_reviews=predecessor_reviews,
            projection_diagnostic=dict(layer=2, position=0, input_stage=13,
                                       projection_stages=[14, 15], output_index=813,
                                       order="Measure both rejected predecessors before paired propagation; "
                                       "measure this candidate if its L2 operands are reached.",
                                       reference_substitution="Diagnostic only; never fed into execution."))
    if combined:
        preregistration.update(
            candidate="layer00-stage00-combined-mathematical-RNE-v1",
            candidate_set=["combined_mathematical"],
            intervention="Only layer00 stage00: round x_i*w_i/sqrt(sum(x_j*x_j)/N + "
            "EPSILON_Q48/2^48) once directly to binary16 RNE, preserving signed zero. "
            "Use exact integer squared midpoint comparisons; remove mean RNE, floor sqrt "
            "and Q24 quotient RNE together. Retain native epsilon, FP16 interstage/KV, "
            "all other operators, independent histories and gates. This is the reviewed "
            "d004a401c237 no_mean_RNE mathematical decomposition, not a reference injection.",
            hypothesis="The combined mathematical intervention may behave differently from "
            "the reviewed quotient-only and root-only rejections; neither isolated rejection "
            "nor stage00 agreement establishes downstream benefit.",
            predecessor_roots=predecessor_roots, predecessor_reviews=predecessor_reviews,
            consumed_rejections=predecessor_results,
            retained_predecessor_projection_accounting=predecessor_accounting,
            projection_diagnostic=dict(
                layer=2, position=0, input_stage=13, projection_stages=[14, 15],
                output_index=813, order="Measure candidate arm-local gate/up operands if reached; "
                "consume reviewed predecessor accounting without repeating it.",
                reference_substitution="Diagnostic only; never fed into execution."),
            independent_operator_oracle="Decimal at 80 and 120 digits, all 3584 stage00 "
            "coordinates joined to the reviewed exact decomposition.",
            validation_command="PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 "
            "OMP_NUM_THREADS=1 PYTHONPATH=ace3/model:ace3/model/tests python3 -m unittest "
            "test_retained_causal_repair_campaign test_trace_lineage_a_layer07_rmsnorm",
            launch_command=sys.argv)
    write("preregistration.json", preregistration)
    with (out / "tool_versions.log").open("x") as stream:
        stream.write(f"Python {sys.version}\nNumPy {numpy.__version__}\n")
        subprocess.run([tool, "-V"], stdout=stream, stderr=subprocess.STDOUT, check=True)
    argv = [tool, "-g2012"]
    for top in freeze["public_tops"]:
        argv.extend(("-s", top))
    for name, value in freeze["public_parameters"].items():
        argv.extend(("-P", f"{name}={value}"))
    argv += ["-o", str(out / "public_contracts.vvp")]
    argv += [str(p) for p in sorted(sources.glob("*.sv"))]
    with (out / "compile.log").open("x") as stream:
        compiled = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=False)
    write("compile.json", dict(argv=argv, returncode=compiled.returncode))
    require(compiled.returncode == 0, "public contract compilation failed")

    def finite(values):
        require(all(not bad and not sat for _, bad, sat in values), "invalid/saturated operator")
        return [int(bits) for bits, _, _ in values]

    def hex_bits(rec, framed=False):
        register(rec)
        data = checked(rec["path"]).read_bytes()
        values = rounding.framed_hidden(data) if framed else rounding.hex_rows(data)
        if "semantic_sha256" in rec:
            require(hashlib.sha256(b"".join(x.to_bytes(2, "little") for x in values)).hexdigest()
                    == rec["semantic_sha256"], "semantic FP16 input mismatch")
        return values

    def tensors_for(layer, prepared):
        history = load(prepared["independent_parent"]["path"])
        identity = {r["name"]: r for r in history["checkpoint_tensor_hashes"]}
        tensors = {}
        for rec in prepared["vectors"]["tensors"]:
            meta = rec["checkpoint_tensor"]
            name = meta["name"]
            require(name.startswith(f"model.layers.{layer}.")
                    and all(meta[k] == identity[name][k]
                            for k in ("name", "dtype", "shape", "sha256")),
                    "checkpoint/reference tensor mismatch")
            data = hex_bits(rec["serialized"])
            payload = b"".join(x.to_bytes(2 if meta["dtype"] == "F16" else 4, "little")
                               for x in data)
            require(len(payload) == meta["bytes"]
                    and hashlib.sha256(payload).hexdigest() == meta["sha256"],
                    "native AWQ tensor bytes mismatch")
            tensors[name.removeprefix(f"model.layers.{layer}.")] = data
        require(len(tensors) == 26, "incomplete decoder tensor cone")
        return tensors

    def array_bits(rec):
        path = Path(rec["path"])
        got = record(path)
        require(got["sha256"] == rec["file_sha256"], "independent array file mismatch")
        register(got)
        values = numpy.load(checked(path), allow_pickle=False)
        require(values.dtype in (numpy.dtype("<f2"), numpy.dtype("<u2"))
                and list(values.shape) == rec["shape"], "independent FP16 array format changed")
        bits = values.view("<u2").reshape(-1)
        require(hashlib.sha256(bits.tobytes()).hexdigest() == rec["semantic_sha256"],
                "independent array semantic mismatch")
        return bits.tolist()

    def project(stage, a, b):
        suffix = PROJECTIONS[stage][1]
        ts = {k: tensors[suffix+"."+k] for k in ("qweight", "qzeros", "scales")}
        outputs, totals = exact.project_vectors(a, b, ts)
        bias = tensors.get(suffix+".bias")
        for label in ("A", "R"):
            if bias is not None:
                if stage in (1, 2):
                    biased = [v + (boundary.units(w) << 24)
                              for v, w in zip(totals[label], bias, strict=True)]
                    outputs[label] = [rounding.round_q48(v) for v in biased]
                    outputs[label] = [bits | 0x8000 if bits == 0 and v < 0 else bits
                                      for bits, v in zip(outputs[label], biased, strict=True)]
                else:
                    outputs[label] = finite([adaptation.residual_add(v, w)
                                             for v, w in zip(outputs[label], bias, strict=True)])
        count, words = len(outputs["A"]), len(outputs["A"]) // 8
        selected = {0, count-1, max(range(count),
                    key=lambda i: abs(boundary.units(outputs["A"][i])
                                      - boundary.units(outputs["R"][i])))}
        for values, label in ((a, "A"), (b, "R")):
            for i in sorted(selected):
                oracle = projection.complete_projection_output(
                    values, ts["qweight"][i//8::words], ts["qzeros"][i//8::words],
                    ts["scales"][i::count], i % 8,
                    None if bias is None else bias[i], single_round_bias=stage in (1, 2))
                require(not (oracle[2] or oracle[3]) and oracle[0] == totals[label][i]
                        and oracle[1] == outputs[label][i], "scalar projection oracle disagreement")
        return outputs["A"], outputs["R"]

    def operator(stage, stages, incoming, cache, p):
        if stage in (0, 13):
            suffix = "input_layernorm.weight" if stage == 0 else "post_attention_layernorm.weight"
            return finite(adaptation.rmsnorm(incoming if stage == 0 else stages[12],
                                             tensors[suffix])[0])
        if stage in (4, 5):
            source = stages[1 if stage == 4 else 2]
            result = list(source)
            for base in range(0, len(source), 64):
                for pair in range(32):
                    lo, hi, bad, sat = rope.rotate_pair(source[base+pair], source[base+pair+32],
                                                        *rope.qwen2_coefficient(p, pair))
                    require(not (bad or sat), "RoPE invalid/saturated")
                    result[base+pair], result[base+pair+32] = lo, hi
            return result
        if stage in (6, 7):
            return list(stages[5 if stage == 6 else 3])
        if stage == 8:
            result = []
            for h in range(14):
                for pos in range(p+1):
                    row = attention.attention_score(stages[4][h*64:(h+1)*64],
                        cache["k"][pos][(h//7)*64:(h//7+1)*64], [True]*64, p, pos)
                    require(not (row.invalid or row.cache_miss or row.saturation), "score invalid")
                    result.append(row.score_f16)
            return result
        if stage == 9:
            result = []
            for h in range(14):
                row = attention.attention_softmax(stages[8][h*(p+1):(h+1)*(p+1)],
                    list(range(p+1)), [True]*(p+1), [False]*(p+1), [False]*(p+1), p)
                require(not (row.invalid or row.cache_miss or row.row_error), "softmax invalid")
                result.extend(row.probabilities_f16)
            return result
        if stage == 10:
            result = []
            for h in range(14):
                for dim in range(64):
                    row = attention.attention_value(stages[9][h*(p+1):(h+1)*(p+1)],
                        [cache["v"][pos][(h//7)*64+dim] for pos in range(p+1)],
                        [True]*(p+1), [False]*(p+1))
                    require(not (row.invalid or row.cache_miss or row.row_error or row.saturation),
                            "AV invalid")
                    result.append(row.value_f16)
            return result
        if stage in (12, 18):
            return finite([adaptation.residual_add(a, b) for a, b in zip(
                incoming if stage == 12 else stages[12], stages[11 if stage == 12 else 17],
                strict=True)])
        require(stage == 16, "unsupported operator")
        return finite([adaptation.silu_gate_exp(a, b)
                       for a, b in zip(stages[14], stages[15], strict=True)])

    def intervention(values, p):
        weights = tensors["input_layernorm.weight"]
        native, mean, rms = adaptation.rmsnorm(values, weights)
        if combined:
            outputs = mathematical.mathematical_outputs(values, weights)
            sumsq = sum(boundary.units(x)**2 for x in values)
            detail = dict(sumsq_q48=sumsq, mean_q48=mean, rms_q24=rms,
                          mathematical_mean_q48_exact=str(
                              Fraction(sumsq, len(values)) + adaptation.EPSILON_Q48))
        else:
            rms_function = root_only_rmsnorm if root_only else quotient_only_rmsnorm
            outputs, detail = rms_function(values, weights, adaptation.EPSILON_Q48, boundary)
        require(mean == detail["mean_q48"] and rms == detail["rms_q24"],
                "candidate changed the native mean/floor-root control")
        for precision in (80, 120):
            with localcontext() as context:
                context.prec = precision
                if combined:
                    divisor = (Decimal(detail["sumsq_q48"])/len(values)
                               + adaptation.EPSILON_Q48).sqrt()
                    independent = [silu.nearest_decimal(
                        Decimal(boundary.units(a))*Decimal(boundary.units(w))
                        / divisor / 2**24, (a ^ w) & 0x8000)
                        for a, w in zip(values, weights, strict=True)]
                elif root_only:
                    divisor = Decimal(mean).sqrt()
                    quotients = [round(Decimal(boundary.units(a))*Decimal(boundary.units(w))
                                       / divisor) for a, w in zip(values, weights, strict=True)]
                    require(quotients == detail["quotient_RNE_q24"],
                            "independent root-only Q24 oracle disagreement")
                    independent = [silu.nearest_decimal(Decimal(q)/2**24, (a ^ w) & 0x8000)
                                   for q, a, w in zip(quotients, values, weights, strict=True)]
                else:
                    independent = [silu.nearest_decimal(
                        Decimal(boundary.units(a))*Decimal(boundary.units(w))
                        / Decimal(rms) / 2**24, (a ^ w) & 0x8000)
                        for a, w in zip(values, weights, strict=True)]
            require(outputs == independent, "independent RMSNorm oracle disagreement")
        retained = [r for r in coordinates if r["position"] == p]
        require(len(retained) == len(values) and all(
            r["index"] == i and int(r["activation"], 16) == values[i]
            and int(r["weight"], 16) == weights[i]
            and int(r["outputs"]["frozen_Q48"], 16) == native[i][0]
            and int(r["outputs"]["no_mean_RNE"] if combined else
                    r["remove_floor_sqrt_only"] if root_only else
                    r["outputs"]["no_Q24_quotient"], 16) == outputs[i]
            for i, r in enumerate(retained)), "reviewed quotient discriminator join failed")
        artifact_name = candidate_name if (root_only or combined) else "quotient"
        write(f"layer00_position{p}_{artifact_name}.json", dict(
            **detail, actual_input=values, weights=weights, outputs=outputs,
            frozen=finite(native), reference_identity=auth["reviewed_parent"]["coordinates.json"],
            independent_decimal_precisions=[80, 120]))
        return outputs

    def measure_projection_inputs(label, arms, identity):
        measured = {}
        for stage in (14, 15):
            suffix = PROJECTIONS[stage][1]
            ts = {k: tensors[suffix+"."+k] for k in ("qweight", "qzeros", "scales")}
            count = len(ts["scales"]) // (len(arms["frozen_Q48"]["13"]) // 128)
            words = count // 8
            stage_rows = {}
            for arm, stages in arms.items():
                rows = projection_contributions(stages["13"], ts, 813, boundary)
                total = sum(row["contribution_q48"] for row in rows)
                local = rounding.round_q48(total)
                oracle = projection.complete_projection_output(
                    stages["13"], ts["qweight"][813//8::words],
                    ts["qzeros"][813//8::words], ts["scales"][813::count], 813 % 8)
                require(not (oracle[2] or oracle[3]) and oracle[:2] == (total, local),
                        "independent index813 projection oracle disagreement")
                if arm != "independent_retained":
                    require(local == stages[str(stage)][813], "own-input index813 projection mismatch")
                stage_rows[arm] = dict(
                    total_q48=total, local=f"{local:04x}",
                    retained=f"{stages[str(stage)][813]:04x}", contributions=rows,
                    own_input_vs_retained=compare([local], [stages[str(stage)][813]], boundary))
            for arm, value in stage_rows.items():
                deltas = {}
                for baseline in ("frozen_Q48", "independent_retained"):
                    base = stage_rows[baseline]
                    rows = [dict(index=r["index"], delta_q48=r["contribution_q48"]-b["contribution_q48"])
                            for r, b in zip(value["contributions"], base["contributions"], strict=True)]
                    total = sum(row["delta_q48"] for row in rows)
                    require(total == value["total_q48"]-base["total_q48"], "projection accounting gap")
                    deltas[baseline] = dict(
                        total_q48=total, exact=str(Fraction(total, 2**48)),
                        positive_q48=sum(r["delta_q48"] for r in rows if r["delta_q48"] > 0),
                        negative_q48=sum(r["delta_q48"] for r in rows if r["delta_q48"] < 0),
                        largest=sorted(rows, key=lambda r: abs(r["delta_q48"]), reverse=True)[:12],
                        per_input=rows)
                value["deltas"] = deltas
            measured[str(stage)] = stage_rows
        write(f"layer02_position0_projection_{label}.json", dict(
            layer=2, position=0, input_stage=13, output_index=813, stages=measured,
            source_identity=identity, tensor_package=consumed[str(packages[2][2])],
            semantics="G128 native GEMM nibble ordering, qweight-qzero without plus one, FP16 scales; "
            "exact Q48 terms independently checked against scalar projection oracle. "
            "Independent operands are diagnostic only; no reference injection into propagation."))

    def evaluate_layer(layer, endpoints):
        nonlocal tensors
        if layer == 0:
            execution = load(root.parent / "token358_position3_layer0_attempt001/frozen.json")
            later = load(root.parent / "token358_position3_layer0_attempt001/result.json")
            transaction = load(later["transaction"]["path"])
            prepared = dict(independent_parent=execution["independent_parent"],
                            vectors=transaction["vectors"])
        else:
            prepared, transaction_path, _ = packages[layer]
            transaction = load(transaction_path)
            require(transaction["layer_index"] == layer and transaction["position"] == 3,
                    "retained transaction coordinate mismatch")
            old = load(prepared["kv_parent"]["layer_result"]["path"])
        tensors = tensors_for(layer, prepared)
        actual, reference, incoming, candidate_incoming = {}, {}, {}, {}
        reference_identities = {}
        for p in range(4):
            if layer == 0:
                retained_path = root / f"position{p}_stages.json"
                retained = load(retained_path)
                actual[p] = {int(k): v for k, v in retained["frozen_Q48"].items()}
                reference[p] = {int(k): v for k, v in retained["independent_retained"].items()}
                reference_identities[p] = consumed[str(retained_path)]
            else:
                rec = old["positions"][p] if p < 3 else transaction
                trace = rec["raw"]["trace"]["path"] if p < 3 else str(
                    Path(rec["output"]["path"]).with_name("trace.hex"))
                actual[p] = boundary.decode_trace(checked(trace).read_bytes(), p)
                if p == 3:
                    reference[p] = {s: hex_bits(prepared["independent_stages"][str(s)])
                                    for s in range(19)}
                    reference_identities[p] = prepared["independent_stages"]
                else:
                    history_path = Path(prepared["independent_parent"]["path"])
                    if p < 2:
                        history_path = history_path.with_name(f"layer{layer:02d}_generation0.json")
                    history = load(history_path)
                    reference_identities[p] = consumed[str(history_path)]
                    reference[p] = {int(Path(rec["path"]).stem[5:]): array_bits(rec)
                                    for rec in history["stages"]
                                    if int(Path(rec["path"]).parent.name[8:]) == p}
            if layer == 0:
                incoming[p] = [int(r["activation"], 16) for r in coordinates if r["position"] == p]
                candidate_incoming[p] = incoming[p]
            else:
                incoming[p] = endpoints[p]["frozen_Q48"]
                candidate_incoming[p] = endpoints[p][candidate_name]
                require(incoming[p] == hex_bits(old["positions"][p]["vectors"]["input"]
                                               if p < 3 else prepared["input_hidden"], True),
                        "retained actual hidden parent mismatch")
            require(set(actual[p]) == set(reference[p]) == set(range(19)),
                    "incomplete retained stage geometry")
        write(f"layer{layer:02d}_inputs.json", dict(
            incoming=incoming, candidate_incoming=candidate_incoming,
            independent_reference_identities=reference_identities, consumed=list(consumed.values())))
        def save_stages(p, values):
            write(f"layer{layer:02d}_position{p}_stages.json", values)
            if (root_only or combined) and layer == 2 and p == 0 and all(
                    s in values[candidate_name] for s in (13, 14, 15)):
                normalized = {arm: {str(s): bits for s, bits in stages.items()}
                              for arm, stages in values.items()}
                measure_projection_inputs(candidate_name, normalized,
                                          record(out / f"layer{layer:02d}_position{p}_stages.json"))

        endpoints, comparisons, failure = paired_cone(
            actual, reference, incoming, candidate_incoming, operator, project,
            intervention if layer == 0 else None, boundary,
            save_stages, candidate_name=candidate_name)
        if failure:
            failure["independent_reference_identity"] = reference_identities[failure["position"]]
        return endpoints, comparisons, failure

    phase_start = time.monotonic()
    all_comparisons, layer_timings = [], {}
    tensors = {}
    if root_only:
        tensors = tensors_for(2, packages[2][0])
        for label, (stages, identity) in predecessor_stages.items():
            measure_projection_inputs(label, stages, identity)
    projection_seconds = time.monotonic() - phase_start
    endpoints = failure = None
    for layer in range(9):
        layer_started = time.monotonic()
        try:
            endpoints, comparisons, failure = evaluate_layer(layer, endpoints)
        except (ValueError, KeyError, OSError) as error:
            failure = dict(
                layer=layer, position=None, stage=None,
                failure_taxonomy="retained_evidence_execution_blocker",
                exception_type=type(error).__name__, message=str(error),
                root_cause_hypothesis="A retained dependency or schema is incompatible "
                "with the authenticated layer adapter; this is not a numerical rejection.",
                regression="Resolve the named dependency against its frozen package in a fresh attempt.")
            comparisons = []
        all_comparisons.extend(dict(layer=layer, **row) for row in comparisons)
        layer_timings[str(layer)] = time.monotonic()-layer_started
        write(f"layer{layer:02d}_comparisons.json", comparisons)
        if failure:
            failure["layer"] = layer
            failure["retained_archive"] = record(root / "artifacts.json")
            write("failure.json", failure)
            break
    numerical_failure = failure is not None and failure["failure_taxonomy"] == (
        "independent_FP16_trajectory_mismatch")
    control_differences = sum(row["reconstructed_vs_runtime"]["bit_differences"]
                              for row in all_comparisons)
    result = dict(
        status=("CANDIDATE_REJECTED" if numerical_failure else
                "EVIDENCE_BLOCKED" if failure else "SOFTWARE_ENDPOINT_REQUIRES_REVIEW"),
        candidate=preregistration["candidate"],
        first_material_failure=None if not numerical_failure else dict(
            layer=failure["layer"], position=failure["position"], stage=failure["stage"],
            comparison=failure["candidate_vs_reference"]),
        failure_taxonomy=None if failure is None else failure["failure_taxonomy"],
        stage_comparisons=len(all_comparisons), frozen_control_bit_differences=control_differences,
        original_endpoint_reached=any(r["layer"] == 8 and r["position"] == 3 and r["stage"] == 8
                                     for r in all_comparisons),
        phases_seconds=dict(authentication_compile=phase_start-started,
                            predecessor_projection=projection_seconds, layers=layer_timings),
        elapsed_seconds=time.monotonic()-started, RTL_simulations=0,
        production_adoption=False, binary64_admission_evaluated=False, third_token=False,
        boundary=preregistration["boundary"], review_status="PENDING_NORMAL_HOST_REVIEWER")
    write("result.json", result)
    write("consumed_artifacts.json", list(consumed.values()))
    write("artifacts.json", [record(p) for p in sorted(out.rglob("*"))
                             if p.is_file() and "__pycache__" not in p.parts])
    for path in out.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps(result, sort_keys=True))


def run(parent, predecessor, review, out):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        with (out / name).open("x", encoding="ascii") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")

    authenticated = {}

    def checked(rec):
        path = Path(rec["path"])
        got = record(path)
        require(all(got[k] == rec[k] for k in ("path", "bytes", "sha256")),
                "changed consumed artifact: " + str(path))
        authenticated[str(path)] = got
        return path

    def load(rec):
        return json.loads(checked(rec).read_text())

    receipt = load(record(review))
    require(receipt["kind"] == "round_reviewed_handoff"
            and receipt["producer_role"] == "reviewer"
            and receipt["mission_id"] == "6eb2e0362261"
            and receipt["review"]["status"] == "done", "predecessor not independently complete")
    manifest = load(record(predecessor / "artifacts.json"))
    previous_index = {r["path"]: r for r in manifest}
    previous_freeze = load(previous_index[str(predecessor / "frozen.json")])
    previous = load(previous_index[str(predecessor / "result.json")])
    require(previous["classification"] == "inherited_gate_up_operand_drift"
            and not previous["supports_repair"], "predecessor outcome changed")
    auth = load(previous_freeze["parent_authentication"])
    upstream_index = {r["path"]: r for r in records(auth)}
    parent_manifest = load(upstream_index[str(parent / "artifacts.json")])
    parent_index = {r["path"]: r for r in parent_manifest}
    freeze = load(upstream_index[str(parent / "frozen.json")])
    failed = load(upstream_index[str(parent / "result.json")])
    require(freeze["layer"] == 2 and failed["candidate_material_failures"] == 1
            and failed["frozen_material_failures"] == 0
            and failed["frozen_control_bit_differences"] == 0, "parent failure scope changed")
    stages = load(parent_index[str(parent / "position0_stages.json")])
    require(set(stages) == {"frozen_Q48", "single_round", "independent_retained"},
            "foreign arm or reference lineage")
    inputs_index = {r["path"]: r for r in freeze["authenticated_inputs"]}
    prepared_path = next(p for p in inputs_index
                         if p.endswith("token358_position3_full_continuation_attempt001/"
                                       "layer02/prepared.json"))
    prepared = load(inputs_index[prepared_path])
    history = load(prepared["independent_parent"])
    tensor_hashes = {r["name"]: r for r in history["checkpoint_tensor_hashes"]}
    tensors = {}
    tensor_bindings = []
    suffixes = ("post_attention_layernorm.weight", "mlp.gate_proj.",
                "mlp.up_proj.", "mlp.down_proj.")
    for rec in prepared["vectors"]["tensors"]:
        meta = rec["checkpoint_tensor"]
        suffix = meta["name"].removeprefix("model.layers.2.")
        if not suffix.startswith(suffixes):
            continue
        require(meta["name"].startswith("model.layers.2.")
                and all(meta[k] == tensor_hashes[meta["name"]][k]
                        for k in ("name", "dtype", "shape", "sha256")),
                "independent/checkpoint tensor identity mismatch")
        data = [int(row, 16) for row in checked(rec["serialized"]).read_text().split()]
        payload = b"".join(x.to_bytes(2 if meta["dtype"] == "F16" else 4, "little")
                           for x in data)
        require(len(payload) == meta["bytes"]
                and hashlib.sha256(payload).hexdigest() == meta["sha256"],
                "serialized/checkpoint tensor mismatch")
        tensors[suffix] = data
        tensor_bindings.append(rec)
    require(len(tensors) == 10, "incomplete post-attention tensor cone")

    sources = out / "sources"
    sources.mkdir()
    source_bindings = []
    for path in sorted((parent / "sources").iterdir()):
        if path.suffix not in (".py", ".sv"):
            continue
        checked(parent_index[str(path)])
        target = sources / path.name
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        source_bindings.append(dict(origin=record(path), snapshot=record(target)))
    evaluator = sources / Path(__file__).name
    with evaluator.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())

    preregistration = dict(
        hypotheses=[
            dict(id="projection_arithmetic",
                 hypothesis="The inherited regression contains a gate/up projection implementation error.",
                 support="An exact native AWQ own-input reduction disagrees with retained stage14/15.",
                 reject="All frozen/candidate gate/up coordinates agree, with scalar oracle cross-checks."),
            dict(id="post_attention_rms_rounding",
                 hypothesis="Removing intermediate stage13 RMS rounding clears the candidate-only regression.",
                 support="Exact mathematical stage13, independently Decimal-cross-checked, clears every "
                 "stage13-18 material gate on the inherited candidate operands.",
                 reject="First material failure anywhere in the affected measured cone."),
            dict(id="inherited_hidden_drift",
                 hypothesis="Even single-round post-attention normalization cannot repair the inherited candidate.",
                 support="The mathematical candidate fails a downstream gate with exact projections.",
                 reject="The changed cone passes; broader rollout still needs independently reviewed semantics.")],
        arms=["frozen_Q48", "single_round"],
        intervention="Same general mathematical RMSNorm formula as the reviewed stage00 candidate, "
        "applied conditionally at stage13 to each arm's own retained stage12. This is a "
        "numerical candidate experiment, not adoption of a new arithmetic boundary.",
        scope="Layer02 position0 post-attention cone. Earlier hidden/K/V unchanged within each arm. "
        "No reference operand substitution and no replay of predecessor stage16 factorial.",
        comparison_policy=freeze["comparison_policy"], token_history=freeze["token_history"],
        stop_rule="Halt each changed arm at its first material stage failure; no failed output is consumed.",
        original_blocker="A layer08 position3 stage08 indices13/15 remains unexecuted by this candidate.",
        binary64="No source-bound independent binary64 layer-final reference is supplied by this "
        "paired FP16 archive. No binary64 admission; prior failures remain unchanged. If stage18 "
        "is reached, bind the original independent binary64 trajectory before RTL/full traversal.",
        provenance=freeze["provenance"],
        evaluator=record(evaluator), predecessor_review=record(review),
        consumed=list(authenticated.values()), tensors=tensor_bindings, sources=source_bindings)
    write("preregistration.json", preregistration)

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(sources))
    boundary = importlib.import_module("diagnose_lineage_a_position3_boundary")
    adaptation = importlib.import_module("fp16_adaptation_oracle")
    exact = importlib.import_module("exact_projection")
    projection = importlib.import_module("projection_oracle")
    norm_model = importlib.import_module("trace_lineage_a_layer07_rmsnorm")
    silu = importlib.import_module("trace_lineage_a_layer07_silu")
    require(boundary.POLICY == freeze["comparison_policy"], "material policy mismatch")

    tool = shutil.which("iverilog")
    require(tool is not None, "iverilog unavailable")
    argv = [tool, "-g2012"]
    for top in freeze["public_tops"]:
        argv.extend(("-s", top))
    for name, value in freeze["public_parameters"].items():
        argv.extend(("-P", f"{name}={value}"))
    argv += ["-o", str(out / "public_contracts.vvp")]
    argv += [str(p) for p in sorted(sources.glob("*.sv"))]
    with (out / "compile.log").open("x") as log:
        compiled = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
    write("compile.json", dict(argv=argv, returncode=compiled.returncode, tool=record(tool)))
    require(compiled.returncode == 0, "source-bound public contract compilation failed")
    phase_started = time.monotonic()
    reference = stages["independent_retained"]
    projection_checks = []
    scalar_checks = []
    prefixes = {14: "mlp.gate_proj", 15: "mlp.up_proj", 17: "mlp.down_proj"}

    def project(stage, values):
        prefix = prefixes[stage]
        ts = {k: tensors[prefix + "." + k] for k in ("qweight", "qzeros", "scales")}
        result, totals = exact.project_vectors(values, values, ts)
        count = len(result["A"])
        words = count // 8
        selected = {0, count - 1, max(range(count), key=lambda i: abs(totals["A"][i]))}
        for index in sorted(selected):
            oracle = projection.complete_projection_output(
                values, ts["qweight"][index // 8::words], ts["qzeros"][index // 8::words],
                ts["scales"][index::count], index % 8)
            require(not (oracle[2] or oracle[3]) and oracle[0] == totals["A"][index]
                    and oracle[1] == result["A"][index], "scalar projection disagreement")
            scalar_checks.append(dict(stage=stage, index=index))
        return result["A"]

    def finite(values):
        require(all(not invalid and not saturated for _, invalid, saturated in values),
                "invalid or saturated operator result")
        return [bits for bits, _, _ in values]

    for arm in ("frozen_Q48", "single_round"):
        for stage in (14, 15):
            result = project(stage, stages[arm]["13"])
            comparison = compare(result, stages[arm][str(stage)], boundary)
            projection_checks.append(dict(arm=arm, stage=stage, **comparison))
    write("projection_checks.json", dict(full_coordinates=projection_checks,
                                       scalar_checks=scalar_checks))
    defect = any(row["bit_differences"] for row in projection_checks)
    if defect:
        write("result.json", dict(status="BLOCKED_PROJECTION_RECONSTRUCTION",
                                 projection_checks=projection_checks,
                                 elapsed_seconds=time.monotonic() - started))
        return

    results = []
    for arm in ("frozen_Q48", "single_round"):
        root = stages[arm]["12"]
        weights = tensors["post_attention_layernorm.weight"]
        native = finite(adaptation.rmsnorm(root, weights)[0])
        require(native == stages[arm]["13"], "retained native stage13 reconstruction mismatch")
        mathematical = norm_model.mathematical_outputs(root, weights)
        for precision in (80, 120):
            with localcontext() as context:
                context.prec = precision
                xs = [Decimal(boundary.units(x)) / 2**24 for x in root]
                rms = (sum(x*x for x in xs) / len(xs)
                       + Decimal(adaptation.EPSILON_Q48) / 2**48).sqrt()
                independent = [silu.nearest_decimal(
                    x / rms * Decimal(boundary.units(w)) / 2**24, (a ^ w) & 0x8000)
                    for x, a, w in zip(xs, root, weights, strict=True)]
            require(mathematical == independent, "exact/Decimal RMSNorm mismatch")
        outputs, comparisons, stop = propagate_post_attention(
            root, mathematical, project,
            lambda g, u: finite([adaptation.silu_gate_exp(a, b)
                                for a, b in zip(g, u, strict=True)]),
            lambda a, b: finite([adaptation.residual_add(x, y)
                                for x, y in zip(a, b, strict=True)]),
            reference, boundary)
        write(arm + "_stages.json", outputs)
        results.append(dict(arm=arm, stopped_stage=stop, comparisons=comparisons,
                            norm_bit_changes=sum(a != b for a, b in
                                                 zip(native, mathematical, strict=True))))
    changed = next(r for r in results if r["arm"] == "single_round")
    supported = changed["stopped_stage"] is None and all(
        r["stopped_stage"] is None for r in results)
    result = dict(
        status="CONDITIONAL_CANDIDATE_REQUIRES_REVIEW" if supported else "CANDIDATE_REJECTED",
        projection_arithmetic_hypothesis="rejected",
        post_attention_rms_rounding_hypothesis="conditionally_supported" if supported else "rejected",
        inherited_hidden_drift_hypothesis="unresolved" if supported else "supported_in_measured_scope",
        original_stage00_only_candidate="rejected: retained L2/P0/stage16 regression remains",
        candidate=results, scalar_projection_checks=len(scalar_checks),
        candidate_scope="Conditional stage13 mathematical intervention on retained arm-local operands; "
        "not a rollout of mathematical RMSNorm at all earlier stages/layers.",
        trajectory_repaired=False, RTL_simulations=0, third_token=False, production_adoption=False,
        binary64_admission_evaluated=False, binary64_status=preregistration["binary64"],
        original_blocker=preregistration["original_blocker"], provenance=freeze["provenance"],
        next_action=("Independent review of general semantics and conditional scope before a "
                     "candidate rollout that regenerates all affected earlier hidden/K/V."
                     if supported else
                     "Reject both the stage00-only candidate and this conditional stage13 extension. "
                     "The retained frozen control avoids the new regression but does not repair L8/P3. "
                     "No unchanged continuation or RTL experiment is justified by these results."),
        review_status="PENDING_NORMAL_HOST_REVIEWER",
        phases_seconds=dict(authentication_freeze_compile=phase_started-started,
                            measured_cone=time.monotonic()-phase_started),
        elapsed_seconds=time.monotonic()-started)
    write("result.json", result)
    write("artifacts.json", [record(p) for p in sorted(out.rglob("*"))
                             if p.is_file() and "__pycache__" not in p.parts])
    for path in out.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps(result, sort_keys=True))


def localize_l2p0s16(root, out):
    """Account for retained producer deltas without replaying a rejected trajectory."""
    started = time.monotonic()
    require(out.parent == root.parent and out.name.startswith("retained_")
            and "l2p0s16" in out.name, "localization output outside the bounded namespace")
    out.mkdir(exist_ok=False)
    bindings, consumed = {}, {}

    def register(value):
        for rec in records(value):
            bindings.setdefault(rec["path"], {})[(rec["bytes"], rec["sha256"])] = rec

    def checked(path):
        path = Path(path)
        variants = bindings.get(str(path), {})
        require(len(variants) == 1, "missing/conflicting consumed binding: " + str(path))
        got = record(path)
        require((got["bytes"], got["sha256"]) in variants,
                "changed consumed artifact: " + str(path))
        consumed[str(path)] = got
        return path

    def load(path):
        obj = json.loads(checked(path).read_text())
        register(obj)
        return obj

    def write(name, value):
        with (out / name).open("x", encoding="ascii") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")

    directories = {
        "quotient_only": root.parent / "retained_quotient_only_0fe3b4cb1657_attempt003",
        "root_only": root.parent / "retained_root_only_371d2a404cad_attempt002",
        "combined_mathematical": root,
    }
    retained, inputs, rejections = {}, {}, {}
    for arm, directory in directories.items():
        register(record(directory / "artifacts.json"))
        load(directory / "artifacts.json")
        rejections[arm] = load(directory / "result.json")
        first = rejections[arm]["first_material_failure"]
        require(rejections[arm]["status"] == "CANDIDATE_REJECTED"
                and rejections[arm]["frozen_control_bit_differences"] == 0
                and (first["layer"], first["position"], first["stage"]) == (2, 0, 16)
                and [v["index"] for v in first["comparison"]["failures"]] == [813],
                "retained rejection changed")
        retained[arm] = {layer: load(directory / f"layer{layer:02d}_position0_stages.json")
                         for layer in range(3)}
        inputs[arm] = {layer: load(directory / f"layer{layer:02d}_inputs.json")
                       for layer in range(3)}
    prereg = load(root / "preregistration.json")
    parent = Path(prereg["root_manifest"]["path"]).parent
    load(parent / "artifacts.json")
    auth = load(parent / "parent_authentication.json")
    coordinates = load(auth["reviewed_parent"]["coordinates.json"]["path"])
    review_records = dict(prereg["predecessor_reviews"])
    review_records["d004a401c237"] = prereg["predecessor_review"]
    review_directory = Path(prereg["predecessor_review"]["path"]).parent.parent
    receipts = sorted((review_directory / "08e70e80a3ea").glob("round-*.json"))
    require(receipts, "combined candidate has no external review")
    review_records["08e70e80a3ea"] = record(receipts[-1])
    old = root.parent / "retained_layer02_stage16_index813_localization_6eb2e0362261_attempt001"
    register(record(old / "artifacts.json"))
    load(old / "artifacts.json")
    localization = load(old / "result.json")
    root_prereg = load(directories["root_only"] / "preregistration.json")
    review_records["6eb2e0362261"] = root_prereg["predecessor_reviews"]["6eb2e0362261"]
    for mission, rec in review_records.items():
        register(rec)
        review = load(rec["path"])
        require(review["kind"] == "round_reviewed_handoff"
                and review["producer_role"] == "reviewer"
                and review["mission_id"] == mission and review["review"]["status"] == "done",
                "required genuine predecessor review missing: " + mission)

    sources = root / "sources"
    for path in sorted(sources.iterdir()):
        if path.suffix in (".py", ".sv"):
            checked(path)
    sys.path.insert(0, str(sources))
    boundary = importlib.import_module("diagnose_lineage_a_position3_boundary")
    adaptation = importlib.import_module("fp16_adaptation_oracle")
    projection = importlib.import_module("projection_oracle")
    rounding = importlib.import_module("trace_lineage_a_layer07_downproj")
    require(prereg["comparison_policy"] == boundary.POLICY,
            "retained comparison policy differs from the source-bound evaluator")
    require(prereg["public_port_contract"] in
            checked(sources / "ace3_decoder_layer0_token_engine.sv").read_text(),
            "retained public port contract changed")

    tensors, tensor_identities = {}, {}
    for layer in range(3):
        if layer == 0:
            result = load(root.parent / "token358_position3_layer0_attempt001/result.json")
            package = load(result["transaction"]["path"])
        else:
            package = load(prereg["layer_packages"][str(layer)]["path"])
        tensors[layer] = {}
        for tensor in package["vectors"]["tensors"]:
            meta, serialized = tensor["checkpoint_tensor"], tensor["serialized"]
            prefix = f"model.layers.{layer}."
            require(meta["name"].startswith(prefix), "cross-layer tensor binding")
            name = meta["name"].removeprefix(prefix)
            if not (name.endswith("layernorm.weight") or any(
                    name.startswith(PROJECTIONS[s][1] + ".") for s in (3, 11, 14, 15, 17))):
                continue
            values = rounding.hex_rows(checked(serialized["path"]).read_bytes())
            payload = b"".join(v.to_bytes(2 if meta["dtype"] == "F16" else 4, "little")
                               for v in values)
            require(len(payload) == meta["bytes"]
                    and hashlib.sha256(payload).hexdigest() == meta["sha256"],
                    "native checkpoint tensor serialization mismatch")
            tensors[layer][name] = values
            tensor_identities[meta["name"]] = dict(checkpoint=meta, serialized=serialized)
    tool = shutil.which("iverilog")
    require(tool is not None, "public-contract compiler unavailable on host")
    compile_argv = [tool, "-g2012"]
    for top in prereg["public_tops"]:
        compile_argv += ["-s", top]
    for name, value in prereg["public_parameters"].items():
        compile_argv += ["-P", f"{name}={value}"]
    compile_argv += ["-o", str(out / "public_contracts.vvp")]
    compile_argv += [str(p) for p in sorted(sources.glob("*.sv"))]
    write("frozen.json", dict(
        objective="L2/P0/S16/index813 retained producer localization",
        evaluator=record(Path(__file__)), reviews=review_records,
        consumed=list(consumed.values()), tensors=tensor_identities,
        public_port_contract=prereg["public_port_contract"], public_tops=prereg["public_tops"],
        public_parameters=prereg["public_parameters"], compile_argv=compile_argv,
        tools=[record(tool), record(sys.executable)], python_version=sys.version,
        comparison_policy=prereg["comparison_policy"], token_history=prereg["token_history"],
        provenance=prereg["provenance"], candidate_set=list(directories),
        scope="Retained position0 only. Complete RMSNorm/residual operand tables and exact "
              "projection contributions along deterministic dominant-delta spines. Spines are "
              "not an exhaustive scalar causal graph or a unique-cause proof.",
        selection="At projections select largest absolute exact contribution delta; at residuals "
                  "select largest exact operand delta; ties use lowest index/left operand. "
                  "Trace both gate/up parents and both candidate/control and control/reference spines. "
                  "RMSNorm denominator contributions are reported in full, not treated as constant.",
        stop_rule="Fail on any consumed binding, control identity, own-input, or oracle disagreement. "
                  "No failed trajectory is advanced; no new candidate is evaluated.",
        numerical_policy_changed=False, RTL_simulations=0))
    with (out / "compile.log").open("x") as stream:
        compiled = subprocess.run(compile_argv, stdout=stream, stderr=subprocess.STDOUT, check=False)
    write("compile.json", dict(argv=compile_argv, returncode=compiled.returncode))
    require(compiled.returncode == 0, "public contract compilation failed; no numerical conclusion")
    measured_at = time.monotonic()

    def nearest(value, zero_sign=0):
        sign = 0x8000 if value < 0 else zero_sign if value == 0 else 0
        magnitude = abs(value)
        require(magnitude <= boundary.POSITIVE[-1], "finite projection/RMSNorm range exceeded")
        upper = bisect_left(boundary.POSITIVE, magnitude)
        bits = min({upper, max(0, upper-1)},
                   key=lambda b: (abs(boundary.POSITIVE[b]-magnitude), b & 1))
        return bits | sign

    def delta(a, b):
        return boundary.units(a) - boundary.units(b)

    summaries, first_producers = {}, {}
    projection_count = 0
    for candidate in directories:
        arms = ("frozen_Q48", candidate, "independent_retained")
        stages = retained[candidate]
        for layer in range(3):
            for arm in ("frozen_Q48", "independent_retained"):
                require(stages[layer][arm] == retained["combined_mathematical"][layer][arm],
                        "candidate archives do not share the same frozen/control/reference")
            for arm, key in (("frozen_Q48", "incoming"), (candidate, "candidate_incoming")):
                if layer:
                    require(inputs[candidate][layer][key]["0"] == stages[layer-1][arm]["18"],
                            "hidden handoff does not join its own arm")
        normalized, residuals = {}, {}
        for layer in range(3):
            incoming = {
                "frozen_Q48": inputs[candidate][layer]["incoming"]["0"],
                candidate: inputs[candidate][layer]["candidate_incoming"]["0"],
                "independent_retained": (
                    stages[layer-1]["independent_retained"]["18"] if layer else
                    inputs[candidate][layer]["incoming"]["0"]),
            }
            for stage in (0, 13):
                weights = tensors[layer]["input_layernorm.weight" if stage == 0
                                         else "post_attention_layernorm.weight"]
                table = {}
                for arm in arms:
                    values = incoming[arm] if stage == 0 else stages[layer][arm]["12"]
                    xs = [boundary.units(v) for v in values]
                    sumsq = sum(x*x for x in xs)
                    mean_exact = Fraction(sumsq, len(xs)) + adaptation.EPSILON_Q48
                    mean, rows = round(mean_exact), []
                    root_q24 = isqrt(mean)
                    native, oracle_mean, oracle_root = adaptation.rmsnorm(values, weights)
                    require((mean, root_q24) == (oracle_mean, oracle_root),
                            "independent native RMSNorm mean/root disagreement")
                    for i, (a, w, x) in enumerate(zip(values, weights, xs, strict=True)):
                        product = x * boundary.units(w)
                        quotient = Fraction(product, root_q24)
                        q24 = round(quotient)
                        bits = nearest(Fraction(q24), (a ^ w) & 0x8000)
                        require(not native[i][1] and not native[i][2] and bits == native[i][0],
                                "independent native RMSNorm scalar disagreement")
                        actual = stages[layer][arm][str(stage)][i]
                        if arm != "independent_retained" and not (layer == stage == 0
                                                                  and arm == candidate):
                            require(bits == actual, "retained own-input RMSNorm disagreement")
                        rows.append(dict(index=i, activation=f"{a:04x}", weight=f"{w:04x}",
                                         square_q48=x*x, numerator_q48=product,
                                         quotient_q24_exact=str(quotient), quotient_q24_RNE=q24,
                                         native_output=f"{bits:04x}", retained_output=f"{actual:04x}",
                                         fp16_rounding_error_q24_exact=str(
                                             Fraction(boundary.units(bits))-q24)))
                    table[arm] = dict(sumsq_q48=sumsq, mean_q48_exact=str(mean_exact),
                                      mean_q48_RNE=mean, floor_root_q24=root_q24, rows=rows)
                normalized[f"{layer}:{stage}"] = table
            for stage in (12, 18):
                if str(stage) not in stages[layer][candidate]:
                    continue
                table = {}
                for arm in arms:
                    left = incoming[arm] if stage == 12 else stages[layer][arm]["12"]
                    right = stages[layer][arm]["11" if stage == 12 else "17"]
                    rows = []
                    for i, (a, b) in enumerate(zip(left, right, strict=True)):
                        total = boundary.units(a) + boundary.units(b)
                        bits = nearest(Fraction(total), (a & b) & 0x8000)
                        actual = stages[layer][arm][str(stage)][i]
                        if arm != "independent_retained":
                            require(bits == actual, "retained own-input residual disagreement")
                        rows.append(dict(index=i, left=f"{a:04x}", right=f"{b:04x}",
                                         sum_q24=total, local=f"{bits:04x}",
                                         retained=f"{actual:04x}"))
                    table[arm] = rows
                residuals[f"{layer}:{stage}"] = table
        write(f"{candidate}_rmsnorm.json", normalized)
        write(f"{candidate}_residuals.json", residuals)
        changed = [i for i, (a, b) in enumerate(zip(stages[0][candidate]["0"],
                                                   stages[0]["frozen_Q48"]["0"], strict=True))
                   if a != b]
        root_rows = [r for r in coordinates if r["position"] == 0]
        require(len(root_rows) == len(stages[0][candidate]["0"]), "stage00 decomposition geometry")
        output_key = "no_mean_RNE" if candidate == "combined_mathematical" else "no_Q24_quotient"
        for i, row in enumerate(root_rows):
            require(row["index"] == i
                    and int(row["activation"], 16) == inputs[candidate][0]["incoming"]["0"][i]
                    and int(row["outputs"]["frozen_Q48"], 16) == stages[0]["frozen_Q48"]["0"][i]
                    and int(row["remove_floor_sqrt_only"] if candidate == "root_only"
                            else row["outputs"][output_key], 16) == stages[0][candidate]["0"][i],
                    "reviewed stage00 decomposition does not join this candidate")
        write(f"{candidate}_earliest_producer.json", dict(
            layer=0, position=0, stage=0, changed_indices=changed,
            reviewed_decomposition=root_rows,
            claim="Earliest changed producer on this retained candidate/control trajectory; "
                  "not a unique scalar cause and not an established RTL rounding defect."))
        first_producers[candidate] = changed
        nodes = {}

        def trace(layer, stage, index, baseline):
            nonlocal projection_count
            key = f"{baseline}:{layer}:{stage}:{index}"
            if key in nodes:
                return key
            lo, hi = (candidate, "frozen_Q48") if baseline == "candidate_control" else (
                "frozen_Q48", "independent_retained")
            node = dict(layer=layer, position=0, stage=stage, index=index,
                        operands_source=consumed[str(directories[candidate] /
                                                     f"layer{layer:02d}_position0_stages.json")],
                        retained={a: f"{stages[layer][a][str(stage)][index]:04x}" for a in arms},
                        selected_parents=[])
            nodes[key] = node
            if stage in PROJECTIONS:
                source, suffix = PROJECTIONS[stage]
                ts = {k: tensors[layer][suffix+"."+k] for k in ("qweight", "qzeros", "scales")}
                count = len(stages[layer][candidate][str(stage)])
                words, terms = count // 8, {}
                for arm in arms:
                    values = stages[layer][arm][str(source)]
                    rows = projection_contributions(values, ts, index, boundary)
                    total = sum(r["contribution_q48"] for r in rows)
                    bias = tensors[layer].get(suffix+".bias")
                    oracle = projection.complete_projection_output(
                        values, ts["qweight"][index//8::words],
                        ts["qzeros"][index//8::words], ts["scales"][index::count],
                        index % 8, None if bias is None else bias[index])
                    projected = nearest(Fraction(total, 2**24))
                    local = projected
                    if bias is not None:
                        local = nearest(Fraction(boundary.units(projected)
                                                 + boundary.units(bias[index])))
                    require(not (oracle[2] or oracle[3]) and oracle[:2] == (total, local),
                            "independent scalar projection contribution disagreement")
                    if arm != "independent_retained":
                        require(local == stages[layer][arm][str(stage)][index],
                                "retained own-input projection disagreement")
                    terms[arm] = dict(total_q48=total, unbiased_exact=str(Fraction(total, 2**48)),
                                      projected=f"{projected:04x}", local=f"{local:04x}",
                                      bias=None if bias is None else f"{bias[index]:04x}",
                                      rounding_error_exact=str(
                                          Fraction(boundary.units(projected), 2**24)
                                          - Fraction(total, 2**48)), contributions=rows)
                differences = [a["contribution_q48"]-b["contribution_q48"] for a, b in zip(
                    terms[lo]["contributions"], terms[hi]["contributions"], strict=True)]
                require(sum(differences) == terms[lo]["total_q48"]-terms[hi]["total_q48"],
                        "projection contribution accounting gap")
                selected = max(range(len(differences)), key=lambda i: abs(differences[i]))
                node.update(projections=terms, contribution_delta_q48=differences,
                            delta_q48=sum(differences),
                            positive_delta_q48=sum(v for v in differences if v > 0),
                            negative_delta_q48=sum(v for v in differences if v < 0))
                projection_count += 1
                if any(differences):
                    node["selected_parents"].append(trace(layer, source, selected, baseline))
                else:
                    node["spine_stop"] = "All exact projection contribution deltas are zero."
            elif stage in (0, 13):
                node["full_numerator_and_denominator_table"] = f"{candidate}_rmsnorm.json"
                if layer == stage == 0:
                    node["earliest_producer"] = root_rows[index]
                else:
                    node["selected_parents"].append(trace(
                        layer-1 if stage == 0 else layer, 18 if stage == 0 else 12, index, baseline))
            elif stage in (12, 18):
                rows = residuals[f"{layer}:{stage}"]
                differences = [delta(int(rows[lo][index][operand], 16),
                                     int(rows[hi][index][operand], 16))
                               for operand in ("left", "right")]
                node["operand_delta_q24"] = differences
                node["full_residual_table"] = f"{candidate}_residuals.json"
                if abs(differences[0]) >= abs(differences[1]):
                    if stage == 12:
                        parent_node = (layer-1, 18) if layer else None
                        if layer == 0:
                            require(differences[0] == 0, "token root changed")
                            node["spine_stop"] = "Authenticated unchanged layer00 token root."
                    else:
                        parent_node = (layer, 12)
                else:
                    parent_node = (layer, 11 if stage == 12 else 17)
                if parent_node is not None:
                    node["selected_parents"].append(trace(*parent_node, index, baseline))
            elif stage == 16:
                node["gate_up"] = {arm: {
                    "gate": f"{stages[layer][arm]['14'][index]:04x}",
                    "up": f"{stages[layer][arm]['15'][index]:04x}",
                    "product_q48": boundary.units(stages[layer][arm]["14"][index])
                                   * boundary.units(stages[layer][arm]["15"][index])}
                    for arm in arms}
                for parent_stage in (14, 15):
                    node["selected_parents"].append(trace(layer, parent_stage, index, baseline))
            elif stage == 10:
                require(all(all(v == 0x3c00 for v in stages[layer][a]["9"]) for a in arms),
                        "position0 singleton attention is not exact one")
                v_index = (index // 64 // 7) * 64 + index % 64
                require(all(stages[layer][a]["10"][index] == stages[layer][a]["7"][v_index]
                            for a in arms), "singleton attention/GQA value join changed")
                node["selected_parents"].append(trace(layer, 7, v_index, baseline))
            else:
                require(stage == 7, "unsupported producer in position0 spine")
                require(all(stages[layer][a]["7"][index] == stages[layer][a]["3"][index]
                            for a in arms), "V cache copy mismatch")
                node["selected_parents"].append(trace(layer, 3, index, baseline))
            return key

        spine_roots = [trace(2, 16, 813, baseline)
                       for baseline in ("candidate_control", "control_reference")]
        write(f"{candidate}_contribution_spines.json", dict(roots=spine_roots, nodes=nodes))
        final = {a: stages[2][a]["16"][813] for a in arms}
        require(final[candidate] == 0xdc23 and final["frozen_Q48"] == 0xdc24
                and final["independent_retained"] == 0xdc25, "retained witness identity changed")
        summaries[candidate] = dict(
            target={a: f"{v:04x}" for a, v in final.items()},
            comparison=compare([final[candidate]], [final["independent_retained"]], boundary),
            stage00_changed_coordinates=len(changed), spine_nodes=len(nodes),
            layer01_final_candidate_control=compare(stages[1][candidate]["18"],
                                                    stages[1]["frozen_Q48"]["18"], boundary),
            stage13_candidate_control=compare(stages[2][candidate]["13"],
                                              stages[2]["frozen_Q48"]["13"], boundary),
            l2_gate_up={str(s): {a: f"{stages[2][a][str(s)][813]:04x}" for a in arms}
                        for s in (14, 15)})
    result = dict(
        status="DIAGNOSTIC_COMPLETE_NO_REPAIR_PREREGISTERED", summaries=summaries,
        earliest_changed_producer=dict(layer=0, position=0, stage=0, indices=first_producers),
        exact_projection_nodes=projection_count,
        retained_local_stage16_disposition=localization,
        failure_taxonomy="independent_FP16_trajectory_mismatch",
        root_cause_hypothesis="The stage00 rounding interventions alter the retained hidden trajectory. "
        "At L2/P0 the inherited shared gate offset 4c84 versus 4c85 combines with the candidate "
        "up transition cb56 to cb55, producing dc23 rather than reference dc25. Complete norm "
        "denominator and residual tables plus exact contribution spines connect this to layer00 "
        "stage00. Correct own-input RNE is not proof of independent trajectory agreement.",
        regression="Preserve L2/P0/S16/813 dc23 versus dc25 and all three reviewed rejections; "
        "any future general candidate must clear this affected-prefix gate before advancing.",
        candidate_decision="No single new repair candidate is justified by this accounting. "
        "The three reviewed mathematical interventions all retain the material regression. "
        "A largest contribution is not a unique cause or authority to change a correct local RNE. "
        "No additional per-edge successor task is created; connected follow-up remains with "
        "the operator-queued batched repair.",
        proof_boundary="Authenticated retained software candidate/control/reference diagnosis, "
        "not fresh RTL evidence. Full projection terms are exact on the selected spines; other "
        "historical scalar paths are not uniquely localized. Original positions0/1 independent "
        "reference input-consumption provenance remains unavailable. No binary64 admission, "
        "candidate-native token generation, production adoption, or numerical policy change.",
        timings_seconds=dict(authentication_and_compile=measured_at-started,
                             numerical_accounting=time.monotonic()-measured_at,
                             total=time.monotonic()-started),
        numerical_policy_changed=False, production_adoption=False, RTL_simulations=0,
        review_status="PENDING_NORMAL_HOST_REVIEWER")
    write("consumed_artifacts.json", list(consumed.values()))
    write("result.json", result)
    write("artifacts.json", [record(p) for p in sorted(out.iterdir()) if p.is_file()])
    for path in out.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps({k: result[k] for k in (
        "status", "earliest_changed_producer", "exact_projection_nodes", "timings_seconds")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    rms_candidate = parser.add_mutually_exclusive_group()
    rms_candidate.add_argument("--quotient-only-root", type=Path)
    rms_candidate.add_argument("--root-only-root", type=Path)
    rms_candidate.add_argument("--combined-mathematical-root", type=Path)
    rms_candidate.add_argument("--localize-l2p0s16-root", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.localize_l2p0s16_root:
        localize_l2p0s16(args.localize_l2p0s16_root.resolve(), args.out.resolve())
    elif args.combined_mathematical_root:
        run_quotient_only(args.combined_mathematical_root.resolve(), args.out.resolve(),
                          combined=True)
    elif args.root_only_root:
        run_quotient_only(args.root_only_root.resolve(), args.out.resolve(), root_only=True)
    elif args.quotient_only_root:
        run_quotient_only(args.quotient_only_root.resolve(), args.out.resolve())
    else:
        if args.parent is None or args.predecessor is None or args.review is None:
            parser.error("--parent, --predecessor and --review are required")
        run(args.parent.resolve(), args.predecessor.resolve(),
            args.review.resolve(), args.out.resolve())
