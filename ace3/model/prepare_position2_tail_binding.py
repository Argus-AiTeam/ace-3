#!/usr/bin/env python3
"""Bind admitted position-2 layer23 to a compile-only, immutable tail package."""

from __future__ import annotations

import ast
import copy
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ace3/model"))

import numpy as np

import fp16_adaptation_oracle as fixed
import prepare_repair8_lm_head_position2 as head
import run_final_rmsnorm_from_layer23 as evidence
import streaming_lm_head_reference as reference

CONTRACT = ROOT / "ace3/contracts/position2_tail_binding.json"
PARENT = ROOT / "build/layer23_runtime_launch_attempt002"
LAYER = ROOT / "build/model24_corrected_q_layers22_23_attempt001/layer23"
CHECKPOINT = ROOT / "build/model24_rtl_cascade/checkpoint/model.safetensors"
REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
    "0ffba022ad92/round-0006.json"
)
RTL = (
    "ace3/rtl/ace3_fp16_fixed.sv",
    "ace3/rtl/ace3_fp16_rmsnorm_core.sv",
    "ace3/rtl/ace3_q47_48_to_f16_rne.sv",
    "ace3/rtl/ace3_streaming_tied_lm_head_topk.sv",
)
require = evidence.require
load = evidence.load_json
write = evidence.write_json
record = head.file_record
BUILD_ENVIRONMENT = ("PATH", "PYTHONPATH", "PYTHONSAFEPATH", "LD_LIBRARY_PATH",
                     "LD_PRELOAD", "IVERILOG_ICONFIG", "IVERILOG_VPI_MODULE_PATH")


def authenticate(row: dict) -> None:
    path = Path(row["path"])
    require(path.is_file() and not path.is_symlink(), f"missing or symlink artifact: {path}")
    actual = record(path)
    require(all(actual[key] == row[key] for key in ("path", "bytes", "sha256")),
            f"artifact drift: {path}")


def authenticate_tree(value: object) -> None:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            authenticate(value)
        for item in value.values():
            authenticate_tree(item)
    elif isinstance(value, list):
        for item in value:
            authenticate_tree(item)


def admitted_parent(contract: dict) -> tuple[dict, list[int]]:
    expected = contract["parent"]
    launch = record(PARENT / "result.json")
    seal = record(PARENT / "seal.json")
    result = record(LAYER / "result.json")
    for row, key in ((launch, "launch_result_sha256"), (seal, "seal_sha256"),
                     (result, "layer_result_sha256")):
        require(row["sha256"] == expected[key], f"foreign parent: {key}")
    authenticate_tree(load(PARENT / "seal.json"))
    review = load(REVIEW)
    require(review["producer_role"] == "reviewer"
            and review["mission_id"] == expected["review_mission"]
            and review["round"] == expected["review_round"]
            and review["review"]["status"] == "done"
            and review["review"]["reason"].startswith(
                "Layer23 attempt002 exited naturally; all three transactions consumed admitted layer22 RTL outputs byte-for-byte"),
            "missing independent layer23 admission")
    launch_data = load(PARENT / "result.json")
    require(launch_data["attempt_id"] == expected["attempt_id"]
            and launch_data["layer_result"] == result
            and launch_data["natural_terminal_transactions"] == 3
            and launch_data["material_mismatches"] == 0,
            "launch/result identity mismatch")
    layer = load(LAYER / "result.json")
    require(layer["layer_index"] == 23 and layer["actual_output_fed_rtl_chain"]
            and layer["earliest_material_mismatch"] is None
            and [row["stage"] for row in layer["independent_comparisons"]] == list(range(19))
            and all(row["failure_count"] == 0 for row in layer["independent_comparisons"])
            and [row["position"] for row in layer["positions"]] == [0, 1, 2],
            "incomplete admitted layer23 chain")
    terminal = layer["positions"][2]["output"]
    require(terminal["path"] == str(LAYER / "position002/raw/final.hex")
            and terminal["sha256"] == expected["terminal_file_sha256"]
            and terminal["semantic_sha256"] == expected["terminal_semantic_sha256"],
            "foreign terminal hidden state")
    authenticate(terminal)
    bits = head.load_terminal_bits(Path(terminal["path"]))
    require(head.sha256_bytes(head.terminal_payload(bits)) ==
            expected["terminal_semantic_sha256"], "terminal semantic mismatch")
    require(np.isfinite(np.asarray(bits, dtype="<u2").view("<f2")).all(),
            "nonfinite terminal hidden state")
    return {"attempt_id": expected["attempt_id"], "launch": launch, "seal": seal,
            "result": result, "review": record(REVIEW), "terminal": terminal}, bits


def source_paths() -> list[Path]:
    paths = {ROOT / name for name in RTL}
    paths.update((CONTRACT, ROOT / "ace3/contracts/streaming_tied_lm_head_topk.json"))
    paths.update((ROOT / "ace3/model/run_position2_tail.py",
                  ROOT / "ace3/model/tests/test_audit_position2_tail_lineage.py"))
    pending = [Path(__file__).resolve()]
    while pending:
        path = pending.pop()
        if path in paths:
            continue
        paths.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module
                     else [])
            for name in names:
                local = ROOT / "ace3/model" / (name.split(".")[0] + ".py")
                if local.is_file() and local not in paths:
                    pending.append(local)
    return sorted(paths)


def log_command(output: Path, name: str, argv: list[str]) -> dict:
    write(output / f"{name}.command.json",
          {"argv": argv, "cwd": str(ROOT),
           "environment": {key: os.environ.get(key) for key in BUILD_ENVIRONMENT}})
    with (output / f"{name}.log").open("xb") as log:
        child = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               timeout=40, check=False)
    receipt = {"exit_code": child.returncode, "command": record(output / f"{name}.command.json"),
               "log": record(output / f"{name}.log")}
    write(output / f"{name}.exit.json", receipt)
    require(child.returncode == 0, f"command failed: {name}")
    return receipt


def independent_rmsnorm(bits: list[int], weights: list[int]) -> list[int]:
    # Decimal never calls the production Q24 decode, division, sqrt or rounding helpers.
    def decode(raw: int) -> Decimal:
        exponent, fraction = (raw >> 10) & 31, raw & 1023
        require(exponent != 31, "nonfinite RMSNorm operand")
        magnitude = (Decimal(fraction) * Decimal(2) ** -24 if exponent == 0
                     else Decimal(1024 + fraction) * Decimal(2) ** (exponent - 25))
        return -magnitude if raw & 0x8000 else magnitude

    with localcontext() as context:
        context.prec = 100
        x = [decode(raw) * (1 << 24) for raw in bits]
        w = [decode(raw) * (1 << 24) for raw in weights]
        mean = ((sum(value * value for value in x) + 281474977 * len(x))
                / len(x)).to_integral_value(rounding=ROUND_HALF_EVEN)
        divisor = int(mean.sqrt())
        outputs = []
        for raw_x, raw_w, value, weight in zip(bits, weights, x, w, strict=True):
            quotient = (value * weight / divisor).to_integral_value(rounding=ROUND_HALF_EVEN)
            negative = quotient < 0 or (quotient == 0 and bool((raw_x ^ raw_w) & 0x8000))
            magnitude = abs(quotient)
            exponent = 0
            while magnitude >= Decimal(2048) * (Decimal(2) ** exponent):
                exponent += 1
            significand = int((magnitude / (Decimal(2) ** max(exponent, 0)))
                              .to_integral_value(rounding=ROUND_HALF_EVEN))
            if significand == 2048:
                significand //= 2
                exponent += 1
            payload = significand if significand < 1024 else ((exponent + 1) << 10) | (significand - 1024)
            require(payload < 0x7c00, "independent RMSNorm overflow")
            outputs.append(payload | (0x8000 if negative else 0))
    return outputs


def fp16_interstage_comparison(actual: np.ndarray, expected: np.ndarray) -> dict:
    require(actual.dtype == expected.dtype == np.dtype("<u2")
            and actual.shape == expected.shape and actual.size > 0,
            "invalid FP16 comparison shape or encoding")
    actual_values = actual.view("<f2").astype(np.float64)
    expected_values = expected.view("<f2").astype(np.float64)
    require(np.isfinite(actual_values).all() and np.isfinite(expected_values).all(),
            "nonfinite FP16 comparison operand")
    absolute = np.abs(actual_values - expected_values)
    relative = absolute / np.maximum(np.abs(expected_values), 2.0**-14)
    ulp = np.asarray([abs(evidence.ordered_f16(int(a)) - evidence.ordered_f16(int(b)))
                      for a, b in zip(actual.flat, expected.flat, strict=True)]).reshape(actual.shape)
    failures = np.flatnonzero((absolute > .125) & ((relative >= .001) | (ulp > 1)))
    return {"reference": "qwen2_fp16_interstage_final_rmsnorm",
            "material_failure_rule": "absolute > 0.125 AND (relative >= 0.001 OR ULP > 1)",
            "finite_required": True, "relative_denominator_floor": 2.0**-14,
            "failure_count": len(failures),
            "first_material_mismatch_index": int(failures[0]) if len(failures) else None,
            "max_absolute_error": float(absolute.max()),
            "max_relative_error": float(relative.max()), "max_ulp_distance": int(ulp.max())}


def rms_expectation(output: Path, bits: list[int], tensors: dict) -> tuple[list[int], dict]:
    norm = tensors["model.norm.weight"]
    weights = np.memmap(CHECKPOINT, dtype="<u2", mode="r", offset=norm["offset"],
                        shape=(896,)).tolist()
    expected = independent_rmsnorm(bits, weights)
    production, mean, divisor = fixed.rmsnorm(bits, weights)
    actual = [row[0] for row in production]
    require(not any(row[1] or row[2] for row in production), "RMSNorm invalid/saturated")
    integer_mismatches = sum(a != b for a, b in zip(actual, expected, strict=True))
    require(integer_mismatches == 0, "independent RMSNorm integer mismatch")
    x = np.asarray(bits, dtype="<u2").view("<f2").astype(np.float64)
    w = np.asarray(weights, dtype="<u2").view("<f2").astype(np.float64)
    mathematical = (x / np.sqrt(np.mean(x * x) + 1e-6) * w).astype("<f2")
    accepted = np.asarray(expected, dtype="<u2")
    absolute = np.abs(accepted.view("<f2").astype(np.float64) - mathematical.astype(np.float64))
    relative = absolute / np.maximum(np.abs(mathematical.astype(np.float64)), 1e-30)
    def ordered(raw: int) -> int:
        return 0x8000 - (raw & 0x7fff) if raw & 0x8000 else 0x8000 + raw
    ulp = np.asarray([abs(ordered(int(a)) - ordered(int(b)))
                      for a, b in zip(accepted, mathematical.view("<u2"), strict=True)])
    failures = int(np.count_nonzero((absolute > .125) & (relative > .001) & (ulp > 1)))
    require(np.isfinite(mathematical).all() and failures == 0, "mathematical RMSNorm material mismatch")
    interstage = evidence.fp16_interstage_expected(
        np.asarray([bits], dtype="<u2"), np.asarray(weights, dtype="<u2"))[0]
    current = fp16_interstage_comparison(accepted, interstage)
    require(current["failure_count"] == 0, "FP16-interstage RMSNorm material mismatch")
    # Keep the legacy runtime's reference file distinct; never substitute its recurrence.
    for name, values in (("tail_input.hex", bits), ("norm_weight.hex", weights),
                         ("final_rmsnorm_expected.hex", expected),
                         ("mathematical_rmsnorm_expected.hex", mathematical.view("<u2")),
                         ("fp16_interstage_rmsnorm_expected.hex", interstage)):
        with (output / name).open("x", encoding="ascii") as stream:
            stream.writelines(f"{int(raw):04x}\n" for raw in values)
    return expected, {"integer_mismatches": integer_mismatches,
                     "fp16_interstage_comparison": current,
                     "mathematical_comparison_role": "retained legacy diagnostic; not the active gate",
                     "mathematical_material_mismatches": failures,
                     "mathematical_bit_differences": int(np.count_nonzero(ulp)),
                     "mathematical_max_absolute_error": float(absolute.max()),
                     "mean_q48": mean, "rms_q24": divisor,
                     "input_semantic_sha256": head.sha256_bytes(head.terminal_payload(bits)),
                     "output_semantic_sha256": head.sha256_bytes(head.terminal_payload(expected))}


def admit(package: dict, output: Path, contract: dict) -> None:
    require(package["parent"]["attempt_id"] == contract["parent"]["attempt_id"]
            and package["parent"]["result"]["sha256"] == contract["parent"]["layer_result_sha256"],
            "foreign_parent")
    for key, path, digest in (
        ("launch", PARENT / "result.json", contract["parent"]["launch_result_sha256"]),
        ("seal", PARENT / "seal.json", contract["parent"]["seal_sha256"]),
        ("result", LAYER / "result.json", contract["parent"]["layer_result_sha256"]),
        ("terminal", LAYER / "position002/raw/final.hex", contract["parent"]["terminal_file_sha256"]),
    ):
        require(package["parent"][key]["path"] == str(path)
                and package["parent"][key]["sha256"] == digest, "foreign_parent")
    require(package["parent"]["review"]["path"] == str(REVIEW), "foreign_parent_review")
    require(not any((output / name).exists()
                    for name in contract["admission"]["runtime_namespaces_must_be_absent"]),
            "consumed_attempt")
    require(package["runtime_tail_invocations"] == 0
            and package["execution_authority"] is False, "unexpected runtime authority")
    for item in package["sources"]:
        authenticate(item["current"])
        authenticate(item["frozen"])
    for item in package["inputs"]:
        authenticate(item)
    bits = [int(line, 16) for line in (output / "tail_input.hex").read_text().splitlines()]
    require(len(bits) == 896 and head.sha256_bytes(head.terminal_payload(bits)) ==
            contract["parent"]["terminal_semantic_sha256"], "mismatched_tail_input")
    authenticate_tree(package["parent"])
    authenticate(package["checkpoint"])
    authenticate(package["head_oracle"])
    authenticate_tree(package["builds"])


def negative_controls(package: dict, output: Path, contract: dict) -> list[dict]:
    controls = []
    root = output / "negative_controls"
    root.mkdir()
    for name in contract["admission"]["reject"]:
        candidate = copy.deepcopy(package)
        case = root / name
        case.mkdir()
        shutil.copyfile(output / "tail_input.hex", case / "tail_input.hex")
        if name == "foreign_parent":
            candidate["parent"]["result"]["sha256"] = "0" * 64
        elif name == "consumed_attempt":
            write(case / "consumed.json", {"fixture": True})
        elif name == "stale_source":
            candidate["sources"][0]["current"]["sha256"] = "0" * 64
        else:
            with (case / "tail_input.hex").open("w", encoding="ascii") as stream:
                stream.write("0000\n" * 896)
        try:
            admit(candidate, case, contract)
        except evidence.AttemptError as error:
            expected_reason = "artifact drift" if name == "stale_source" else name
            require(expected_reason in str(error), f"wrong negative-control rejection: {name}: {error}")
            controls.append({"case": name, "rejected": True, "reason": str(error)})
        else:
            raise evidence.AttemptError(f"negative control accepted: {name}")
    return controls


def comparison_mapping() -> dict:
    path = ROOT / "ace3/model/run_position2_tail.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "execute")
    expressions = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ("mathematical", "relative", "material"):
                text = ast.unparse(node.value)
                if target.id != "relative" or "mathematical" in text:
                    expressions[target.id] = {"expression": text, "line": node.lineno,
                                              "end_line": node.end_lineno}
    require(set(expressions) == {"mathematical", "relative", "material"},
            "unrecognized tail runtime reference mapping")
    require(expressions["relative"]["expression"].endswith(", 1e-30)")
            and expressions["material"]["expression"] ==
            "np.flatnonzero((absolute > 0.125) & (relative > 0.001) & (ulp > 1))"
            and "mathematical_rmsnorm_expected.hex" in expressions["mathematical"]["expression"],
            "tail runtime mapping changed; reassess rather than infer compatibility")
    return {"runtime": record(path), "expressions": expressions,
            "current_execution_compatible": False,
            "runtime_reference": "float64 RMSNorm with one final FP16 rounding",
            "active_reference": "FP32 mean-square/rsqrt; FP16 normalized activation; FP16 weight product",
            "active_failure_rule": "nonfinite OR (absolute > 0.125 AND (relative >= 0.001 OR ULP > 1))",
            "active_relative_denominator": "max(abs(reference), 2^-14)",
            "preparation_reference_function": "run_final_rmsnorm_from_layer23.fp16_interstage_expected",
            "preparation_comparison_function": "prepare_position2_tail_binding.fp16_interstage_comparison",
            "head_reference": "Unchanged exact Q24-product accumulation, FP16 rounding and deterministic Top-K; not the RMSNorm trajectory gate."}


def admission_diagnosis(output: Path, contract: dict, parent: dict, tensors: dict,
                        sources: list, builds: list, mapping: dict) -> dict:
    import audit_position2_tail_lineage as audit

    prior_path = ROOT / "build/position2_tail_admission_542e8365de0e_attempt003/result.json"
    prior = load(prior_path)
    historical = prior["historical_runtime"]
    authenticate_tree(historical)
    history_dir = Path(historical["result"]["path"]).parent
    sealed = load(Path(historical["seal"]["path"]))["files"]
    freeze_row = next(row for row in sealed if row["path"] == str(history_dir / "frozen.json"))
    require(historical["result"] in sealed, "historical result not sealed")
    authenticate(freeze_row)
    freeze = load(Path(freeze_row["path"]))
    require(freeze["parent"] == parent == prior["parent"], "foreign_parent")
    authenticate(freeze["package"])
    legacy = load(Path(freeze["package"]["path"]))
    require(tensors == legacy["tensors"] == prior["official_tensors"], "official tensor identity drift")
    require(contract["final_rmsnorm"]["mathematical_material_failure"] ==
            "absolute error exceeds 0.125 AND relative error exceeds 0.001 AND ULP distance exceeds 1",
            "tail contract mapping changed; reassess explicitly")
    runner = ROOT / "ace3/model/run_final_rmsnorm_from_layer23.py"
    retained = next(row["frozen"] for row in legacy["sources"]
                    if row["current"]["path"] == str(runner))
    authenticate(retained)
    policies = {"retained": audit.probe_rmsnorm_policy(Path(retained["path"]).read_text()),
                "current": audit.probe_rmsnorm_policy(runner.read_text())}
    require(policies["current"]["required_policy_disagreements"] == 0
            and policies["retained"]["required_policy_disagreements"] == 3,
            "RMSNorm policy controls changed")
    foreign = copy.deepcopy(legacy)
    foreign["parent"]["attempt_id"] = "model24_layer23_attempt001"
    try:
        admit(foreign, output, contract)
    except evidence.AttemptError as error:
        require(str(error) == "foreign_parent", f"unexpected foreign-parent rejection: {error}")
        rejection = {"attempt_id": foreign["parent"]["attempt_id"],
                     "rejected": True, "reason": str(error)}
    else:
        raise evidence.AttemptError("foreign parent accepted")
    retained_result = load(Path(historical["result"]["path"]))
    require(retained_result["runtime_tail_invocations"] == 1
            and retained_result["natural_exit"] is True, "historical consumption changed")
    reviews = []
    for mission, round_number, status in (("1693a83a95c6", 3, "done"),
                                          ("1f9852a2d4f5", 1, "continue")):
        path = REVIEW.parent.parent / mission / f"round-{round_number:04d}.json"
        review = load(path)
        require(review["producer_role"] == "reviewer" and review["mission_id"] == mission
                and review["round"] == round_number and review["review"]["status"] == status,
                "historical independent review identity changed")
        reviews.append(record(path))
    for item in sources:
        authenticate_tree(item)
    require(not any((output / name).exists()
                    for name in contract["admission"]["runtime_namespaces_must_be_absent"]),
            "unexpected runtime namespace")
    result = {
        "schema": "ace3_position2_tail_current_policy_admission_v1",
        "attempt_id": output.name, "status": "BLOCKED_NO_EXECUTION_AUTHORITY",
        "classification": "retained_contract_runtime_reference_mapping",
        "parent": parent, "official_tensors": tensors, "checkpoint": prior["checkpoint"],
        "source_bindings": sources, "compile_results": builds,
        "frozen": record(output / "frozen.json"), "prior_diagnosis": record(prior_path),
        "policy_controls": policies, "retained_comparator_source": retained,
        "comparison_mapping": mapping, "foreign_parent_control": rejection,
        "contract": record(CONTRACT),
        "retained_contract_predicate": contract["final_rmsnorm"]["mathematical_material_failure"],
        "blocker": "The immutable legacy contract and out-of-scope run_position2_tail.execute still "
                   "consume the float64 mathematical reference with AND/> and floor 1e-30. "
                   "Preparation now separately enforces the active FP16-interstage gate; "
                   "that does not make the legacy runtime enforce it. No new execution is needed "
                   "to answer this mapping question or to reconstruct the already accepted tail.",
        "historical_runtime": historical, "historical_reviews": reviews,
        "historical_authority_state": "consumed; not reset or transferred",
        "execution_authority": False, "authority_created": False,
        "runtime_tail_invocations": 0, "unconsumed_authority_count": 0,
        "failure_taxonomy": "contract_reference_mapping_incompatibility",
        "root_cause_hypothesis": "Distinct reference recurrences and acceptance predicates, not producer arithmetic drift.",
        "regression": "Exact Fraction controls, current preparation gate/reference, foreign parent and nonfinite rejection.",
        "independent_review": "pending normal Host Reviewer",
        "claim_boundary": "Source admission diagnosis and compile-only interfaces, not a numerical RTL verdict, new token or hardware result.",
    }
    write(output / "result.json", result)
    return result


def prepare(output: Path, *, admission_only: bool = False) -> dict:
    require(output.parent == ROOT / "build", "attempt must be directly under ignored build/")
    require(not output.exists(), "consumed attempt identity; choose a fresh output")
    output.mkdir()
    write(output / "invocation.command.json",
          {"argv": [sys.executable, "-B", *sys.argv], "cwd": str(ROOT)})
    try:
        contract = load(CONTRACT)
        sources = []
        for path in source_paths():
            destination = output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(path.read_bytes())
            sources.append({"current": record(path), "frozen": record(destination)})
        compiler = shutil.which("iverilog")
        require(compiler is not None, "host Icarus unavailable; local container inspection required")
        public = []
        for key, source in (("final_rmsnorm", RTL[1]), ("tied_head", RTL[3])):
            text = (output / "sources" / source).read_text()
            declaration = re.search(r"module\s+" + contract[key]["top"] + r"\s*#\(.*?\n\);", text, re.S)
            require(declaration is not None, f"public declaration missing: {key}")
            public.append({"top": contract[key]["top"], "parameters": contract[key]["parameters"],
                           "declaration": declaration.group(0)})
        write(output / "frozen.json", {"contract": contract, "sources": sources,
                                     "compiler": record(Path(compiler)), "public_interfaces": public,
                                     "admission_only": admission_only,
                                     "environment": {key: os.environ.get(key) for key in BUILD_ENVIRONMENT},
                                     "tools": {"python": sys.version, "numpy": np.__version__,
                                               "torch": head.torch.__version__},
                                     "score_policy": "binding preparation only; never RTL runtime PASS"})
        builds = [log_command(output, "compiler-version", [compiler, "-V"])]
        for item in public:
            binary = output / (item["top"] + ".vvp")
            argv = [compiler, "-g2012", "-s", item["top"], "-o", str(binary)]
            argv += [f"-P{item['top']}.{key}={value}" for key, value in item["parameters"].items()]
            argv += [str(output / "sources" / name) for name in RTL]
            receipt = log_command(output, item["top"], argv)
            receipt["compiled_artifact"] = record(binary)
            builds.append(receipt)
        parent, bits = admitted_parent(contract)
        tensors = reference.tensor_records(CHECKPOINT)
        require(reference.CHECKPOINT_SHA256 == contract["model"]["checkpoint_sha256"],
                "checkpoint contract drift")
        for name, tensor in tensors.items():
            tensor["sha256"] = reference.sha256_range(CHECKPOINT, tensor["offset"], tensor["bytes"])
        mapping = comparison_mapping()
        if admission_only:
            builds.append(log_command(output, "policy-regressions", [
                sys.executable, "-B", "-c",
                "import sys, unittest; sys.path.insert(0, sys.argv.pop(1)); unittest.main(module=None)",
                str(ROOT),
                "ace3.model.tests.test_audit_position2_tail_lineage"]))
            return admission_diagnosis(output, contract, parent, tensors, sources, builds, mapping)
        require(mapping["current_execution_compatible"],
                "legacy contract/runtime mapping is not current-policy admission; use --admission-only")
        expected, rms = rms_expectation(output, bits, tensors)
        # Bound the existing int64 vector oracle before allowing its exact-product sum.
        hidden_q24 = reference.decode_array_q24(np.asarray(expected, dtype="<u2"))
        weights = np.memmap(CHECKPOINT, dtype="<u2", mode="r",
                            offset=tensors["model.embed_tokens.weight"]["offset"], shape=(151936, 896))
        bound = 0
        for begin in range(0, 151936, 512):
            decoded = reference.decode_array_q24(np.asarray(weights[begin:begin + 512]))
            bound = max(bound, int(np.abs(decoded).max()) * int(np.abs(hidden_q24).max()) * 896)
        require(bound < (1 << 63), "exact head oracle int64 accumulation bound exceeded")
        oracle, checks, _ = head.full_vocabulary_oracle(CHECKPOINT, tensors, expected, reference)
        oracle["kind"] = "ace3_position2_tail_head_prediction"
        oracle["runtime_receipt"] = False
        oracle["int64_accumulation_absolute_bound"] = bound
        write(output / "head_oracle.json", oracle)
        write(output / "selected_logit_checks.json", checks)
        package = {"schema": contract["schema"], "attempt_id": output.name,
                   "status": "PREPARATION_PASS_REVIEW_REQUIRED", "parent": parent,
                   "checkpoint": record(CHECKPOINT), "tensors": tensors, "sources": sources,
                   "builds": builds, "final_rmsnorm": rms,
                   "head_oracle": record(output / "head_oracle.json"),
                   "inputs": [record(output / name) for name in
                              ("tail_input.hex", "norm_weight.hex", "final_rmsnorm_expected.hex",
                               "mathematical_rmsnorm_expected.hex", "fp16_interstage_rmsnorm_expected.hex")],
                   "selection_policy": contract["tied_head"]["selection"],
                   "runtime_tail_invocations": 0, "execution_authority": False,
                   "claim_boundary": contract["claim_boundary"]}
        admit(package, output, contract)
        package["negative_controls"] = negative_controls(package, output, contract)
        write(output / "runtime_pass_package.json", package)
        write(output / "result.json", {"status": "PREPARATION_PASS_REVIEW_REQUIRED",
                                       "package": record(output / "runtime_pass_package.json"),
                                       "runtime_tail_invocations": 0, "independent_review": "pending Host Reviewer"})
    except (evidence.AttemptError, head.PreparationError,
            OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        write(output / "result.json", {"status": "FAIL", "error": str(error),
                                       "failure_taxonomy": "binding_or_preparation_failure",
                                       "root_cause_hypothesis": str(error),
                                       "regression": "fresh binding admission and named negative controls",
                                       "runtime_tail_invocations": 0})
        raise
    finally:
        write(output / "seal.json", {"files": [record(path) for path in sorted(output.rglob("*"))
                                              if path.is_file()]})
        for path in output.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
    return load(output / "result.json")


if __name__ == "__main__":
    require(len(sys.argv) in (2, 3) and (len(sys.argv) == 2 or sys.argv[2] == "--admission-only"),
            "usage: prepare_position2_tail_binding.py build/FRESH_ATTEMPT [--admission-only]")
    result = prepare(Path(sys.argv[1]).resolve(), admission_only=len(sys.argv) == 3)
    print(json.dumps({key: result[key] for key in
                      ("status", "runtime_tail_invocations")}, sort_keys=True))
