#!/usr/bin/env python3
"""Isolated Q-only RoPE candidate; retained references are never execution inputs."""

import bisect
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
CLOSURE = ROOT / "build/layer08_position3_softmax_exp_replay_attempt003/dependency_closure"
RETAINED = ROOT / "build/layer08_position3_softmax_exp_replay_attempt005"
SOFTWARE = ROOT / "build/layer08_position2_rope_split_86938809f0a2_attempt007"
OUT = ROOT / "build/layer08_position2_q_only_rope_rtl_86938809f0a2_attempt010"
RTL = ROOT / "ace3/rtl/candidates/q_only_fused_rope_v1/ace3_qwen2_rope_pair.sv"
TB = ROOT / "ace3/tb/ace3_q_only_fused_rope_candidate_tb.sv"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/86938809f0a2/round-0005.json")
ASSESSMENT = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/83fae4b2cbc6/round-0001.json")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


d = module("q_rope_retained_dependency", CLOSURE / "replay.py")


def integer_q24(bits):
    exponent, mantissa = (bits >> 10) & 31, bits & 1023
    if exponent == 31:
        raise ValueError("nonfinite operand")
    magnitude = mantissa if exponent == 0 else (1024 + mantissa) << (exponent - 1)
    return -magnitude if bits & 32768 else magnitude


def make_cases(path):
    # Independent nearest-lattice oracle: no production rounding helper is used.
    lattice = [integer_q24(bits) << 24 for bits in range(0x7C00)]

    def nearest(total):
        magnitude = abs(total)
        if magnitude >= 65520 << 48:
            return (0xFBFF if total < 0 else 0x7BFF), 1
        upper = bisect.bisect_left(lattice, magnitude)
        options = [i for i in (upper - 1, upper) if 0 <= i < len(lattice)]
        bits = min(options, key=lambda i: (abs(lattice[i] - magnitude), i & 1))
        return bits | (0x8000 if total < 0 else 0), 0

    edges = [0, 0x8000, 1, 3, 0x8001, 0x03FF, 0x0400, 0x3800,
             0x3BFF, 0x3C00, 0x3C01, 0xBC00, 0x7BFF, 0xFBFF, 0x7C00, 0x7E00]
    inputs = [(a, b, c, s) for a in edges for b in edges
              for c, s in ((0x3C00, 0), (0x3800, 0x3800), (0x3C00, 0xBC00))]
    rng = random.Random(20260907)
    inputs += [tuple(rng.randrange(65536) for _ in range(4)) for _ in range(2048)]
    inputs += [(0x7BFF, 0xD000, 0x3C00, 0x3800),
               (1, 0, 0x3800, 0), (3, 0, 0x3800, 0),
               (0x8001, 0, 0x3800, 0)]
    with path.open("x", encoding="ascii") as stream:
        for operands in inputs:
            invalid = any((x & 0x7C00) == 0x7C00 for x in operands)
            if invalid:
                low, high, saturated = 0, 0, 0
            else:
                a, b, c, s = map(integer_q24, operands)
                low, low_sat = nearest(a * c - b * s)
                high, high_sat = nearest(b * c + a * s)
                saturated = low_sat | high_sat
            for key in (0, 1):
                stream.write(" ".join(f"{x:x}" for x in
                             (key, *operands, low, high, int(invalid), saturated)) + "\n")
    return len(inputs) * 2


def fused_q_rope(baseline):
    from awq_bit_oracle import q47_48_to_f16
    from fp16_adaptation_oracle import decode_f16_q24
    from qwen2_rope_oracle import qwen2_coefficient

    def rotate(vector, heads, position):
        d.require(heads in (14, 2), "unexpected Q/K head contract")
        if heads == 2:
            return baseline(vector, heads, position)
        result = vector.copy()
        for head in range(heads):
            for pair in range(32):
                low, high = head * 64 + pair, head * 64 + pair + 32
                cosine, sine = qwen2_coefficient(position, pair)
                a, b, c, s = [decode_f16_q24(int(x))[0] for x in
                              (vector[low], vector[high], cosine, sine)]
                for index, total in ((low, a * c - b * s), (high, b * c + a * s)):
                    bits, saturated = q47_48_to_f16(total)
                    d.require(not saturated, "candidate-local rotary overflow")
                    result[index] = 0x8000 if bits == 0 and total < 0 else bits
        return result
    return rotate


def main():
    OUT.mkdir()
    started = time.monotonic()
    phase, active, transactions = "binding", None, []
    primitive_pass = False
    try:
        frozen = d.load(CLOSURE / "frozen.json")
        software_freeze = d.load(SOFTWARE / "freeze.json")
        expected_bindings = {r["path"]: r for r in software_freeze["inputs_and_sources"]}
        retained_record = expected_bindings[str(RETAINED / "result.json")]
        d.authenticate(retained_record)
        retained = d.load(RETAINED / "result.json")
        software = d.load(SOFTWARE / "result.json")
        d.require(software["status"] == "SOFTWARE_CANDIDATE_FOUND", "software selection changed")
        for path, mission, status in ((REVIEW, "86938809f0a2", "continue"),
                                      (ASSESSMENT, "83fae4b2cbc6", "done")):
            review = json.loads(path.read_text(encoding="utf-8"))
            d.require(review["kind"] == "round_reviewed_handoff" and
                      review["producer_role"] == "reviewer" and
                      review["mission_id"] == mission and
                      review["review"]["status"] == status, "review scope/status mismatch")
        for item in frozen["sources"] + [frozen["checkpoint"], frozen["python"]]:
            d.authenticate(item)
        for name, version in frozen["python_packages"].items():
            d.require(importlib.metadata.version(name) == version, f"{name} drift")
        d.require(Path(sys.executable).resolve() == Path(frozen["python"]["path"]),
                  "interpreter differs from retained runtime")
        reference_records = {}
        for item in retained["transactions"]:
            key = item["layer"], item["position"]
            if key[0] > 8 or key[1] > 2:
                continue
            d.authenticate(item["result"])
            result = d.load(item["result"]["path"])
            refs = result["independent_stages"]
            d.require(set(refs) == {str(i) for i in range(19)}, "incomplete independent stages")
            for rec in refs.values():
                d.authenticate(rec)
            reference_records[key] = {"parent": item["result"], "stages": refs}
        d.require(len(reference_records) == 27, "missing retained independent trajectory")
        for item in frozen["embeddings"][:3]:
            d.authenticate(item["input"])
        source = OUT / "source"
        source.mkdir()
        for rec in frozen["sources"]:
            path = Path(rec["path"])
            if path.parent == CLOSURE / "source":
                shutil.copyfile(RTL if path.name == RTL.name else path, source / path.name)
        shutil.copyfile(TB, source / TB.name)
        old_rope = (CLOSURE / "source" / RTL.name).read_text("ascii")
        d.require(d.header(RTL.read_text("ascii")) == d.header(old_rope),
                  "public RoPE interface changed")
        baseline = old_rope.replace("module ace3_qwen2_rope_pair (",
                                    "module ace3_qwen2_rope_pair_baseline (", 1)
        with (source / "baseline.sv").open("x", encoding="ascii") as stream:
            stream.write(baseline)
        cases = OUT / "operator_cases.hex"
        case_count = make_cases(cases)
        tools = {name: d.tool(name, flag) for name, flag in
                 (("verilator", "--version"), ("iverilog", "-V"), ("vvp", "-V"),
                  ("make", "--version"), ("g++", "--version"))}
        prior, oracle = d.imports()
        traversal, np = prior.traversal, prior.np
        namespace = oracle.run_token.__globals__
        namespace["_rope"] = fused_q_rope(namespace["_rope"])
        d.SOURCE = source
        commands = [d.decoder_command(tools, layer, OUT / f"layer{layer:02d}/obj", True)
                    for layer in range(9)]
        delta_path = CLOSURE.parent / "delta_helpers.py"
        delta_record = {item["path"]: item for item in frozen["sources"]}[str(delta_path)]
        d.authenticate(delta_record)
        delta = module("q_rope_retained_delta", delta_path)
        d.write(OUT / "frozen.json", {
            "attempt": OUT.name, "official_attempt": False,
            "change": "Q-only exact Q48 RoPE sum then one FP16 RNE; K/V/SiLU unchanged",
            "script": d.record(Path(__file__)), "candidate": d.record(RTL),
            "source_closure": [d.record(p) for p in sorted(source.iterdir())],
            "retained_source_closure": frozen["sources"],
            "reference_source": d.record(CLOSURE / "frozen.json"),
            "retained_result": retained_record,
            "software_freeze": d.record(SOFTWARE / "freeze.json"),
            "software_result": d.record(SOFTWARE / "result.json"),
            "review": d.record(REVIEW), "assessment_review": d.record(ASSESSMENT),
            "reference_transactions": [
                {"layer": key[0], "position": key[1], **rec}
                for key, rec in sorted(reference_records.items())],
            "delta_exporter": delta_record,
            "public_interfaces": {**frozen["public_interfaces"], "rope": d.header(old_rope)},
            "parameters": frozen["parameters"], "prompt": frozen["prompt"],
            "token_history": frozen["token_history"][:3],
            "embeddings": frozen["embeddings"][:3],
            "checkpoint": frozen["checkpoint"],
            "checkpoint_tensors": frozen["checkpoint_tensors"],
            "tools": tools, "python": frozen["python"],
            "python_packages": frozen["python_packages"],
            "comparison_policy": frozen["comparison_policy"],
            "score_policy": frozen["score_policy"],
            "compile_commands": commands,
            "operator_cases": d.record(cases), "operator_case_count": case_count,
            "operator_seed": 20260907,
            "operator_oracle": "Exact integer Q48 and independent nearest finite FP16 lattice, ties even; signed nonzero underflow, exact cancellation +0; legacy saturation/invalid protocol.",
            "selectors": {"layers": list(range(9)), "positions": [0, 1, 2],
                          "stages": list(range(19))},
            "state_policy": "No old binary/state imports. Rebuild all layers0-8; each position0 starts empty; positions1/2 restore only this binary's preceding actual state. Actual hidden feeds each next layer.",
            "independent_policy": "Authenticated retained independently propagated stage files, never injected into execution.",
            "binary64_profile": "Not evaluated or adopted; legacy/v1 FAIL artifacts unchanged.",
            "position3_executed": False,
        })
        phase = "primitive_compile"
        primitive = OUT / "operator.vvp"
        d.run_command([tools["iverilog"]["executable"]["path"], "-g2012",
                       "-s", "ace3_q_only_fused_rope_candidate_tb", "-o", str(primitive),
                       *[str(source / name) for name in
                         ("ace3_fp16_fixed.sv", "ace3_q47_48_to_f16_rne.sv",
                          RTL.name, "baseline.sv", TB.name)]],
                      OUT, "operator_compile", 120)
        phase = "primitive_simulation"
        d.run_command([tools["vvp"]["executable"]["path"], str(primitive),
                       f"+CASES={cases}"], OUT, "operator_simulation", 120)
        primitive_pass = True
        d.write(OUT / "operator_result.json", {
            "status": "PASS", "cases": case_count, "binary": d.record(primitive),
            "execution": d.record(OUT / "operator_simulation.execution.json"),
            "freeze": d.record(OUT / "frozen.json")})
        actual, hidden_records = {}, {}
        for position, item in enumerate(frozen["embeddings"][:3]):
            actual[-1, position] = traversal.load_hidden_bits(Path(item["input"]["path"]))
            hidden_records[-1, position] = item["input"]

        def logged(argv, log_path):
            d.run_command(list(argv), log_path.parent, log_path.stem, 7200)

        traversal.run_logged = logged
        with prior.safe_open(frozen["checkpoint"]["path"], framework="np") as checkpoint:
            for layer in range(9):
                directory = OUT / f"layer{layer:02d}"
                directory.mkdir()
                active, phase = [layer, None], "decoder_compile"
                d.run_command(commands[layer], directory, "compile", 1800)
                binary = directory / f"obj/V{d.TOP}"
                binary_record = d.record(binary)
                d.write(directory / "binary.json", {
                    "binary": binary_record, "freeze": d.record(OUT / "frozen.json"),
                    "imported_old_states": 0,
                    "generated_abi": [d.record(p) for p in sorted(binary.parent.iterdir())
                                      if p.suffix in (".h", ".cpp")]})
                cache_k, cache_v, state = [], [], None
                for position in range(3):
                    active, phase = [layer, position], "preparation"
                    td = directory / f"position{position:03d}"
                    td.mkdir()
                    refs = reference_records[layer, position]["stages"]
                    expected = {int(stage): np.asarray(
                        [int(row, 16) for row in Path(rec["path"]).read_text("ascii").split()],
                        dtype="<u2") for stage, rec in refs.items()}
                    hidden = actual[layer - 1, position]
                    for item in [hidden_records[layer - 1, position], binary_record,
                                 *([state] if state else [])]:
                        d.authenticate(item)
                    vectors = traversal.materialize_transaction_vectors(
                        checkpoint, layer, position, hidden, td / "vectors")
                    originals = {item["name"]: item for item in frozen["checkpoint_tensors"]}
                    for tensor in vectors["tensors"]:
                        meta = tensor["checkpoint_tensor"]
                        d.require(meta["sha256"] == originals[meta["name"]]["sha256"],
                                  "checkpoint tensor binding changed")
                    values = traversal.layer_oracle_values(checkpoint, layer, vectors)
                    final, trace = oracle.run_token(
                        values, hidden.tolist(), position,
                        [row[:] for row in cache_k], [row[:] for row in cache_v],
                        accurate_silu=True)
                    vectors.update(traversal.materialize_runtime_vector_contract(
                        layer, position, final, trace, td / "vectors"))
                    d.write(td / "prepared.json", {
                        "input_hidden": hidden_records[layer - 1, position],
                        "input_state": state, "binary": binary_record, "vectors": vectors,
                        "independent_stages": refs,
                        "independent_input": (
                            reference_records[layer - 1, position]["stages"]["18"]
                            if layer else frozen["embeddings"][position]["input"]),
                        "independent_history": [
                            reference_records[layer, p] for p in range(position)],
                        "reference_parent": reference_records[layer, position]["parent"]})
                    for item in [vectors["input"], vectors["rope_coefficients"],
                                 vectors["trace"], vectors["final_hidden"],
                                 *[t["serialized"] for t in vectors["tensors"]]]:
                        d.authenticate(item)
                    phase = "simulation"
                    transaction, output = traversal.execute_transaction(
                        binary, layer, position, hidden, vectors, td / "vectors",
                        td / "runtime", td / "candidate.state",
                        Path(state["path"]) if state else None)
                    phase = "comparison"
                    payload = Path(transaction["raw"]["trace"]["path"]).read_bytes()
                    comparisons = [{
                        "stage": stage,
                        **prior.comparison.comparison(
                            prior.frontier.trace_stage(
                                payload, position, stage, expected[stage].size,
                                position + 1 if stage in (8, 9) else None),
                            expected[stage])} for stage in range(19)]
                    exact = {
                        "trace": traversal.exact_hex_comparison(
                            payload, Path(vectors["trace"]["path"]).read_bytes(), "trace"),
                        "final_hidden": traversal.exact_hex_comparison(
                            Path(transaction["output"]["path"]).read_bytes(),
                            Path(vectors["final_hidden"]["path"]).read_bytes(), "final_hidden")}
                    witnesses = delta.deltas(
                        prior, payload, {int(k): v for k, v in refs.items()},
                        position, td / "full_deltas.csv")
                    d.write(td / "result.json", {
                        "transaction": transaction, "comparisons": comparisons,
                        "exact": exact, "stage08_witnesses": witnesses,
                        "independent_stages": refs, "candidate_state_parent": state,
                        "full_deltas": d.record(td / "full_deltas.csv"),
                        "prepared": d.record(td / "prepared.json")})
                    transactions.append({
                        "layer": layer, "position": position,
                        "result": d.record(td / "result.json"),
                        "failure_count": sum(c["failure_count"] for c in comparisons)})
                    d.require(all(c["failure_count"] == 0 for c in comparisons),
                              "independent FP16 trajectory mismatch; stop affected cone")
                    d.require(all(c["exact_match"] for c in exact.values()),
                              "candidate-local exact mismatch; stop affected cone")
                    cache_k.append(prior.frontier.trace_stage(payload, position, 6, 128).tolist())
                    cache_v.append(prior.frontier.trace_stage(payload, position, 7, 128).tolist())
                    actual[layer, position] = output
                    hidden_records[layer, position] = transaction["output"]
                    state = transaction["output_state"]
                    d.write(td / "completed.json", {
                        "result": d.record(td / "result.json"), "output_state": state,
                        "output_hidden": transaction["output"], "binary": binary_record,
                        "kv_valid_positions": list(range(position + 1))})
                    print(f"completed layer={layer} position={position} stages=19", flush=True)
        d.write(OUT / "result.json", {
            "status": "PASS_BOUNDED_RTL_CANDIDATE", "transactions": transactions,
            "independent_stage_comparisons": len(transactions) * 19,
            "operator_result": d.record(OUT / "operator_result.json"),
            "freeze": d.record(OUT / "frozen.json"),
            "elapsed_seconds": time.monotonic() - started,
            "position3_executed": False, "normal_host_review": "PENDING",
            "binary64_profile_adoption": False})
    except (RuntimeError, OSError, ValueError, KeyError, ArithmeticError,
            subprocess.SubprocessError) as error:
        d.write(OUT / "failure.json", {
            "active": active, "phase": phase, "error": str(error),
            "failure_taxonomy": (
                "trajectory_comparison" if phase == "comparison" else
                "operator_arithmetic_or_protocol" if phase == "primitive_simulation" else
                "runtime_failure" if phase == "simulation" else
                "evaluator_no_execution_dependency"),
            "root_cause_hypothesis": (
                "Candidate RTL or its local oracle differs at the preserved boundary; inspect exact and independent deltas."
                if phase in ("comparison", "primitive_simulation") else
                "The named source, tool or execution binding prevented completion; no RTL correctness conclusion."),
            "regression": "Preserve this attempt; correct the named failing boundary before a fresh attempt with the same cases, selectors and gates.",
            "operator_pass": primitive_pass, "transactions": transactions,
            "position3_executed": False, "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
