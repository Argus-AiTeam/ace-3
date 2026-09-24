"""Reviewed-anchor final-RMSNorm mixed-factor cutoff classifier."""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
from safetensors import safe_open
import torch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_vector_cutoff_classifier_v1 as vector_cutoff


ROOT = Path("/home/argustest/ace3-argus")
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_mixed_factor_cutoff_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
TASK = "480141a28a63"
ATTEMPT = 3
STATE = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05")
FAILED_OUTPUTS = tuple(
    ROOT / f"build/q24_s16_final_head_mixed_factor_{TASK}_anchors_attempt{attempt:03d}"
    for attempt in (1, 2)
)
OUTPUT = ROOT / f"build/q24_s16_final_head_mixed_factor_{TASK}_anchors_attempt003"
VECTOR = ROOT / "build/q24_s16_final_head_final_rmsnorm_vector_cutoff_classifier_v1_875bdd31dc4d_attempt002"
CONTROL = ROOT / "build/argus-paused-owner-normalization-480-20260915T172823Z-attempt004"
PYTHON = "/home/argustest/miniconda3/bin/python"
ENVIRONMENT = {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
ARGV = [PYTHON, "-B", "-m", MODULE, "--anchor-worker"]
CLASSIFIER_ARGV = [PYTHON, "-B", "-m", MODULE, "--classifier-worker"]
CLASSIFIER_COMMAND = (
    " ".join(f"{k}={shlex.quote(v)}" for k, v in ENVIRONMENT.items())
    + " " + shlex.join(CLASSIFIER_ARGV)
)
CLASSIFIER_PREFIX = f"q24_s16_final_head_mixed_factor_{TASK}_classifier_"
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
    "mapped_all", "inherited_native",
)
LABELS = (*CONTROLS, "reference_fp16")
WIDTH, EPSILON = 896, Fraction(1, 1000000)
RESULT_PIN = {
    "path": str(VECTOR / "result.json"), "bytes": 286542,
    "sha256": "e410090c9d4f899587b58522656e6fe1d7ade0e941d85d89ea91e391e9d8b17f",
}
RAW_PIN = {
    "path": str(VECTOR / "raw.capture"), "bytes": 296356,
    "sha256": "60f4841392bab46e74176fa455e814776588e9d01d13432df5377a953790c4a3",
}
RESUME_PIN = {
    "path": str(CONTROL / "resume-live-use-contract.json"),
    "sha256": "6396aff30fd43c566fb67d614c4bda5849da1f57ec366683e2cdc6fadc51abe7",
}
REVIEW_PIN = {
    "path": str(CONTROL / "native-review/context/handoffs/480141a28a63/round-0001.json"),
    "sha256": "e8898fa5b9ce571394d44b9f53165c9af08ce15244de7683f3bbbb37992ef9da",
}
VECTOR_REVIEW_PIN = {
    "path": str(STATE / "handoffs/875bdd31dc4d/round-0001.json"),
    "sha256": "76b47fb6249de3ddc705af2cb676f11951d960aba22dd549be68afc153760f0f",
}
ANCHOR_RESULT_PIN = {
    "path": str(OUTPUT / "result.json"), "bytes": 359573,
    "sha256": "43645074f392e21868635259efe9bb0fa370bc9f87a92d56f12a3ea0aa7dd986",
}
ANCHOR_RAW_PIN = {
    "path": str(OUTPUT / "raw.capture"), "bytes": 377455,
    "sha256": "31c8102c944a018224c427ef560fc3a9a9e327242b21333c0cef32cf7d012949",
}
ANCHOR_CAPTURE_PIN = {
    "path": str(OUTPUT / "capture.json"), "bytes": 2736,
    "sha256": "3aceb324fb784808b0bb7f952a8274b7f2c577feaa38fecd5f25ffbe9c26a825",
}
ANCHOR_REVIEW_PIN = {
    "path": str(STATE / "handoffs/480141a28a63/round-0001.json"), "bytes": 1055,
    "sha256": "aefb984d66f713222d64a1d0a34c17dd10b72332c0fe2b8bff3c96e721ec520a",
}
ROWS = (13, 319, 34319)
PAIRS = ((34319, 319), (34319, 13), (319, 34319))
FACTORS = ("direct_hidden", "scalar_scale", "interaction")
FACTOR_SUCCESSORS = {
    "direct_hidden": "nested final-RMSNorm direct-hidden residual successor",
    "scalar_scale": "nested final-RMSNorm scalar-scale residual successor",
    "interaction": "nested final-RMSNorm interaction residual successor",
}
CLASSIFIER_SUCCESSORS = {
    "SUPPORTED": FACTOR_SUCCESSORS,
    "REJECTED": "two-factor/cancellation localization with authenticated full closure",
    "UNKNOWN": "authentication, pin, closure or capture-integrity repair only",
}
EXPECTED_TESTS = 20
PREREGISTRATION = {
    "phase": "fresh_generation_time_ten_anchor_prerequisite",
    "labels": list(LABELS), "width": WIDTH, "epsilon": str(EPSILON),
    "scientific_case_limit": 10,
    "arithmetic": "exact sum(h_i*h_i)/896 + epsilon; one binary64 conversion, sqrt and reciprocal",
    "norm_weight_role": "authenticated downstream binding only; no multiplication",
    "classifier_consumption": "FORBIDDEN_UNTIL_NORMAL_INDEPENDENT_HOST_REVIEW",
    "later_classifier": {
        "rows": [13, 319, 34319],
        "ordered_pairs": [[34319, 319], [34319, 13], [319, 34319]],
        "cases": 27,
        "direct_hidden": "(h_actual-h_reference)*weight*s_reference",
        "scalar_scale": "h_reference*weight*(s_actual-s_reference)",
        "interaction": "(h_actual-h_reference)*weight*(s_actual-s_reference)",
        "substitutions": "y_actual minus exactly one named factor; no scalar RNE",
        "closure": "direct+scale+interaction = weight*(h_actual*s_actual-h_reference*s_reference); retain the separate actual-minus-reference boundary remainder",
        "prediction": "each factor must reproduce all 27 vector-cutoff movement signs and endpoint crossing patterns; zero remains distinct",
        "SUPPORTED": "select each reproducing factor's nested successor; no dominance claim",
        "REJECTED": "authenticated full closure but no reproducing single factor: two-factor/cancellation localization",
        "UNKNOWN": "authentication, pin, closure or capture-integrity failure only",
    },
    "claim_boundary": (
        "New evidence, not recovery of ff18166de237. No model operator, RMSNorm "
        "vector output, selected-head arithmetic, prefix/admission/reference "
        "producer or decoder replay. No closed branch reopening, GPU/RTL/FPGA/"
        "ACE2 work, precision/scale expansion or production action. Preserve "
        "historical failures, thresholds, independent original-input references "
        "and source/operand/state/KV/lineage gates. Native-S16-RTZ/Q24 residual "
        "state is wider than FP16; native INT4 weights and FP16 boundaries/KV "
        "are unchanged. No strict-FP16-state W4A16, new-token or full-model admission."
    ),
}
CLASSIFIER_PREREGISTRATION = {
    "phase": "reviewed_ten_anchor_mixed_factor_classifier",
    "task": TASK, "rows": list(ROWS), "ordered_pairs": [list(pair) for pair in PAIRS],
    "controls": list(CONTROLS), "cases": len(CONTROLS) * len(PAIRS),
    "factors": list(FACTORS), "anchor_source": ANCHOR_RESULT_PIN,
    "anchor_review": ANCHOR_REVIEW_PIN, "vector_cutoff_source": RESULT_PIN,
    "arithmetic": (
        "exact rational selected-row dots of y_actual minus one factor, where "
        "direct_hidden=(h_actual-h_reference)*weight*s_reference, "
        "scalar_scale=h_reference*weight*(s_actual-s_reference), and "
        "interaction=(h_actual-h_reference)*weight*(s_actual-s_reference)"
    ),
    "decision_rule": (
        "SUPPORTED iff at least one single factor reproduces all 27 reviewed "
        "vector-cutoff movement signs and endpoint crossing patterns; otherwise "
        "REJECTED when direct+scale+interaction plus explicit retained boundary "
        "closes every case"
    ),
    "unknown_rule": "authentication, pin, closure or capture-integrity failure only",
    "successors": CLASSIFIER_SUCCESSORS,
    "not_independent_samples": True,
}
CLASSIFIER_BOUNDARY = (
    "CPU-only selected-row exact arithmetic over rows 13, 319 and 34319. "
    "Consumes the reviewed ten-anchor scalar capture, reviewed vector-cutoff "
    "accounts, retained stage18 FP16 operands, final norm weights and tied-head "
    "row weights. It does not regenerate anchors, execute RMSNorm/vector/model/"
    "prefix/admission/reference producers, rerun decoder/native suffixes, mutate "
    "references, run a full-vocabulary head or reopen row319/closed branches. "
    "Q24 residual state remains wider than FP16; INT4 weights, native-S16-RTZ, "
    "FP16 operator boundaries/KV, thresholds, lineage gates and historical "
    "failures remain unchanged. The result is component localization only, not "
    "causal dominance, repair, strict-FP16-state W4A16, new-token or full-model admission."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def record(path):
    path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def check_pin(pin):
    path = Path(pin["path"])
    require(path.is_absolute(), "non-absolute authenticated input")
    actual = record(path)
    require(actual["sha256"] == pin["sha256"]
            and ("bytes" not in pin or actual["bytes"] == pin["bytes"]),
            "authenticated input changed: " + str(path))
    return actual


def bound_json(pin):
    check_pin(pin)
    data = Path(pin["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == pin["sha256"],
            "input changed while being read")
    return json.loads(data)


def write_new(path, payload):
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def identity():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == PYTHON and os.getuid() == 1000
            and sys.dont_write_bytecode and not sys.flags.optimize
            and all(os.environ.get(k) == v for k, v in ENVIRONMENT.items()),
            "isolated source/workdir/account/interpreter/environment gate failed")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    require(branch == "argus/full-projection", "isolated branch changed")


def verify_review(review, artifact, mission):
    require(review["kind"] == "round_reviewed_handoff"
            and review["schema_version"] == 3
            and review["producer_role"] == "reviewer"
            and review["mission_id"] == mission
            and review["review"]["status"] == "done", "independent review gate failed")
    if artifact is not None:
        require(any(row["ref"] == artifact["path"]
                    and row["sha256"] == artifact["sha256"]
                    for row in review["review"]["artifact_bindings"]),
                "review does not bind the exact resume contract")


def authenticate_resolution():
    contract = bound_json(RESUME_PIN)
    review = bound_json(REVIEW_PIN)
    verify_review(review, RESUME_PIN, TASK)
    index_path = Path(REVIEW_PIN["path"]).with_name("latest.json")
    index = json.loads(index_path.read_bytes())
    require(index["kind"] == "handoff_ref"
            and index["handoff"]["path"] == REVIEW_PIN["path"],
            "request-specific control review superseded")
    require(contract["kind"] == "paused_operator_resolve_resume",
            "wrong native resume contract")
    binding = contract["binding"]
    require(binding["task_id"] == TASK
            and binding["project_state_root"] == str(STATE)
            and binding["project_workdir"] == str(ROOT)
            and binding["directive_id"] == "3d1f9b17003f4f1e9e7fbc236b72be92",
            "request-specific directive/task identity changed")
    source_pins = [check_pin({"path": p, "sha256": h})
                   for p, h in contract["sources"].items()]
    rows = [json.loads(line) for line in (STATE / "backlog.jsonl").read_bytes().splitlines()
            if line.strip()]
    matches = [row for row in rows if row["id"] == TASK]
    require(len(matches) == 1, "same-ID native task is not unique")
    item = matches[0]
    card = item["operator_decision"]
    tx = card["paused_operator_resolution"]
    require(tx["phase"] == "committed" and tx["binding"] == binding
            and tx["contract"] == RESUME_PIN
            and tx["review"]["path"] == REVIEW_PIN["path"]
            and tx["review"]["sha256"] == REVIEW_PIN["sha256"]
            and card["status"] == "resolved" and item["status"] == "running"
            and item["owns_paths"] == binding["writable_paths"],
            "task lacks its committed, reviewed native resolution")
    before = tx["before"]
    pending = before["operator_decision"]
    require(before["status"] == "paused_operator"
            and not before["running_owner"]
            and pending["status"] == "pending"
            and pending["id"] == binding["decision_id"]
            and pending["revision"] == binding["decision_revision"]
            and pending["question"] == before["pending_question"] == binding["question"]
            and before["objective"] == item["objective"] == binding["objective"],
            "resolution does not bind the original pending decision")
    require(not any(row["id"] != TASK and row["status"] == "running"
                    and set(row["owns_paths"]) & set(item["owns_paths"]) for row in rows),
            "concurrent writer owns this task's paths")
    audit_path = STATE / "operator-authorizations.jsonl"
    audit = [json.loads(line) for line in audit_path.read_bytes().splitlines() if line.strip()]
    events = [row for row in audit if row.get("authorization_id") == tx["authorization_id"]]
    require(len(events) == 2 and [e["event"] for e in events] == ["issued", "consumed"],
            "native resume authorization audit is incomplete")
    issued, consumed = events
    require(all(consumed.get(k) == v for k, v in issued.items() if k != "event")
            and issued["metadata"]["paused_operator"] == binding
            and issued["source_message_id"] == binding["directive_id"]
            and issued["source_channel"] == binding["actor"]
            and issued["scope"] == binding["scope"]
            and issued["allowed_write_paths"] == binding["writable_paths"]
            and "resume_blocked_work" in issued["allowed_actions"]
            and consumed["consumed_action"] == "resume_blocked_work"
            and consumed["paused_operator_resolution"] == {**tx, "phase": "prepared"},
            "native authorization/request/consumption binding changed")
    control_root = STATE / "campaign-control"
    head = json.loads((control_root / "HEAD.json").read_bytes())
    snapshot_path = control_root / head["snapshot"]
    snapshot = json.loads(snapshot_path.read_bytes())
    require(all(head[k] == issued[k]
                for k in ("campaign_id", "objective_sha256", "campaign_epoch"))
            and snapshot["stage_projection"]["paused_operator_resolution"][
                tx["authorization_id"]] == {**tx, "phase": "prepared"},
            "native control index does not close the same-ID transaction")
    return {
        "authorization_id": tx["authorization_id"], "phase": tx["phase"],
        "directive_id": binding["directive_id"], "decision_id": binding["decision_id"],
        "decision_revision": binding["decision_revision"],
        "contract": check_pin(RESUME_PIN), "review": check_pin(REVIEW_PIN),
        "review_index": record(index_path), "source_pins": source_pins,
        "control_snapshot": record(snapshot_path),
        "consumed_record_sha256": hashlib.sha256(json_bytes(consumed)).hexdigest(),
        "authorization_audit_source": str(audit_path),
    }


def nested_pins(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            path = Path(value["path"])
            yield {**value, "path": str(path if path.is_absolute() else ROOT / path)}
        for child in value.values():
            yield from nested_pins(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_pins(child)


def validate_words(words):
    require(isinstance(words, np.ndarray) and words.shape == (WIDTH,)
            and words.dtype.str == "<u2"
            and np.all((words & 0x7c00) != 0x7c00),
            "stage18 must contain exactly 896 finite retained FP16 words")


def load_words(pin):
    check_pin(pin)
    payload = Path(pin["path"]).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == pin["sha256"],
            "stage18 archive changed during read")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate archive field")
        words = archive["stage18"].copy()
    validate_words(words)
    words.flags.writeable = False
    return words


def authenticate_operands():
    saved = bound_json(RESULT_PIN)
    check_pin(RAW_PIN)
    require(Path(RAW_PIN["path"]).read_bytes().endswith(Path(RESULT_PIN["path"]).read_bytes()),
            "reviewed vector result is not the retained raw-capture suffix")
    verify_review(bound_json(VECTOR_REVIEW_PIN), None, "875bdd31dc4d")
    require(saved["decision"] == "SUPPORTED" and saved["direction_agreements"] == 27
            and saved["crossing_pattern_agreements"] == 27,
            "reviewed vector cutoff result changed")
    unique = {}
    # Historical producer identities are not current executable-source pins.
    current = {k: v for k, v in saved.items() if k != "original_execution_sources"}
    for pin in nested_pins(current):
        if pin["path"] in unique:
            require(unique[pin["path"]]["sha256"] == pin["sha256"],
                    "conflicting retained dependency pins")
        else:
            unique[pin["path"]] = check_pin(pin)
    runtime = saved["runtime_pins"]
    require(sys.version == runtime["python_version"]
            and record(Path(sys.executable)) == runtime["python"]
            and np.__version__ == runtime["numpy_version"]
            and str(torch.__version__) == runtime["torch_version"],
            "reviewed Python/NumPy/Torch runtime pins changed")
    controls = saved["retained_controls_and_failure_gates"]
    require(tuple(row["control"] for row in controls) == CONTROLS,
            "fixed nine-control census changed")
    final = saved["final_reference_authority"]
    require(final["status"] == "BOUND_ORIGINAL_INPUT_FINAL" and not final["missing"]
            and final["terminal_review"]["review_status"] == "done"
            and final["reference"]["reference_only"],
            "independent original-input FP16 reference authority changed")
    input_pins = {}
    for row in controls:
        require(row["parent"]["retained_L23"]["source_operand_state_KV_RTZ_checks"] == "PASS",
                "retained source/operand/state/KV/lineage gate changed")
        input_pins[row["control"]] = row["parent"]["terminal_archive"]
    input_pins["reference_fp16"] = final["reference"]["input_fp16"]
    words = {label: load_words(pin) for label, pin in input_pins.items()}
    assets = saved["assets"]
    check_pin(assets["checkpoint"])
    with safe_open(assets["checkpoint"]["path"], framework="numpy") as model:
        weight = model.get_tensor("model.norm.weight")
    weight_pin = assets["tensors"]["model.norm.weight"]
    require(weight.shape == (WIDTH,) and weight.dtype.str == "<f2"
            and np.all(np.isfinite(weight))
            and weight_pin == {"shape": [WIDTH], "dtype": "float16",
                               "sha256": hashlib.sha256(weight.tobytes()).hexdigest()},
            "downstream norm-weight binding changed")
    require(tuple(words) == LABELS, "ten-anchor census changed")
    return {
        "saved": saved, "words": words, "input_pins": input_pins,
        "norm_weight": weight.copy(), "weight_pin": weight_pin,
        "authenticated_pins": list(unique.values()),
        "runtime_pins": runtime,
    }


def scalar_anchor(words):
    validate_words(words)
    values = [Fraction.from_float(float(v)) for v in words.view("<f2")]
    squared_sum = sum((v * v for v in values), Fraction())
    mean = squared_sum / WIDTH
    radicand = mean + EPSILON
    rounded = float(radicand)
    require(math.isfinite(rounded) and rounded > 0, "invalid scalar radicand")
    root = math.sqrt(rounded)
    inverse = 1.0 / root
    require(math.isfinite(inverse) and inverse > 0, "invalid scalar anchor")
    anchor, root_q = Fraction.from_float(inverse), Fraction.from_float(root)
    return {
        "sum_squares": str(squared_sum), "mean_square": str(mean),
        "epsilon": str(EPSILON), "radicand": str(radicand),
        "radicand_binary64_hex": rounded.hex(), "root_binary64_hex": root.hex(),
        "inverse_norm_anchor": str(anchor), "inverse_norm_anchor_hex": inverse.hex(),
        "mean_square_closure": str(mean * WIDTH - squared_sum),
        "radicand_closure": str(radicand - mean - EPSILON),
        "radicand_binary64_conversion_delta": str(Fraction(rounded) - radicand),
        "root_square_defect": str(root_q * root_q - Fraction(rounded)),
        "inverse_root_identity_defect": str(anchor * root_q - 1),
        "inverse_square_identity_defect": str(anchor * anchor * radicand - 1),
    }


def validate_anchor(row):
    require(row["epsilon"] == str(EPSILON), "scalar epsilon changed")
    q = {k: Fraction(row[k]) for k in (
        "sum_squares", "mean_square", "radicand", "inverse_norm_anchor")}
    rounded = float.fromhex(row["radicand_binary64_hex"])
    root = float.fromhex(row["root_binary64_hex"])
    inverse = float.fromhex(row["inverse_norm_anchor_hex"])
    require(all(math.isfinite(v) and v > 0 for v in (rounded, root, inverse))
            and Fraction(inverse) == q["inverse_norm_anchor"], "scalar anchor encoding changed")
    require(q["mean_square"] * WIDTH == q["sum_squares"]
            and q["radicand"] == q["mean_square"] + EPSILON
            and row["mean_square_closure"] == row["radicand_closure"] == "0",
            "exact scalar closure failed")
    expected = {
        "radicand_binary64_conversion_delta": Fraction(rounded) - q["radicand"],
        "root_square_defect": Fraction(root) ** 2 - Fraction(rounded),
        "inverse_root_identity_defect": Fraction(inverse) * Fraction(root) - 1,
        "inverse_square_identity_defect": Fraction(inverse) ** 2 * q["radicand"] - 1,
    }
    require(all(Fraction(row[k]) == v for k, v in expected.items()),
            "scalar approximation-defect accounting changed")


def dependency_pins():
    paths = {Path(sys.executable), SOURCE, TEST}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and Path(name).is_file():
            paths.add(Path(name).absolute())
    return [record(path) for path in sorted(paths)]


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec")
        compiled.append(record(path))
    spec = importlib.util.spec_from_file_location("mixed_factor_anchor_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused tests unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.testsRun == 12 and result.wasSuccessful() and not result.skipped,
            "focused ten-anchor oracle/tests failed")
    return {"executed": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped), "compiled": compiled}


def anchor_worker():
    identity()
    require(OUTPUT.is_dir() and not OUTPUT.is_symlink(), "unique capture root is required")
    command = json.loads((OUTPUT / "command.json").read_bytes())
    require(command["argv"] == ARGV and command["task"] == TASK
            and command["source_test_pins"] == [record(SOURCE), record(TEST)],
            "generation command/source/test binding changed")
    for fd, name in ((1, "stdout.raw"), (2, "stderr.raw")):
        require(Path(f"/proc/self/fd/{fd}").resolve() == OUTPUT / name,
                "generation streams are not the durable command capture")
    write_new(OUTPUT / "attempt.started.json", json_bytes({
        "task": TASK, "attempt": ATTEMPT, "scientific_case_limit": 10,
        "source_test_pins": command["source_test_pins"],
    }))
    start = time.monotonic()
    resolution = authenticate_resolution()
    evidence = authenticate_operands()
    authenticated = time.monotonic()
    print("PREREGISTRATION=" + json.dumps(PREREGISTRATION, sort_keys=True), flush=True)
    os.fsync(1)
    anchors = []
    for label in LABELS:
        words = evidence["words"][label]
        row = {
            "label": label, "archive": evidence["input_pins"][label], "field": "stage18",
            "input": {"shape": [WIDTH], "dtype": "<u2", "bytes": words.nbytes,
                      "sha256": hashlib.sha256(words.tobytes()).hexdigest()},
            **scalar_anchor(words),
        }
        validate_anchor(row)
        anchors.append(row)
        print("ANCHOR=" + json.dumps(row, sort_keys=True), flush=True)
        os.fsync(1)
    computed = time.monotonic()
    result = {
        "task": TASK, "phase": PREREGISTRATION["phase"], "attempt": ATTEMPT,
        "status": "CAPTURED_PENDING_INDEPENDENT_REVIEW",
        "normal_host_review": "REQUIRED_BEFORE_CLASSIFIER_CONSUMPTION",
        "preregistration": PREREGISTRATION, "anchors": anchors,
        "resume_resolution": resolution,
        "reviewed_vector": RESULT_PIN, "reviewed_vector_capture": RAW_PIN,
        "reviewed_vector_review": VECTOR_REVIEW_PIN,
        "norm_weight": evidence["weight_pin"],
        "authenticated_pins": evidence["authenticated_pins"],
        "runtime_pins": evidence["runtime_pins"],
        "source_test_pins": command["source_test_pins"],
        "thresholds": evidence["saved"]["thresholds"],
        "final_reference_authority": evidence["saved"]["final_reference_authority"],
        "retained_controls_and_failure_gates": evidence["saved"]["retained_controls_and_failure_gates"],
        "original_execution_sources": evidence["saved"]["original_execution_sources"],
        "dispatch": {
            "scientific_cases": len(anchors), "exact_square_terms": len(anchors) * WIDTH,
            "binary64_radicand_conversions": len(anchors), "binary64_sqrt_calls": len(anchors),
            "binary64_reciprocals": len(anchors), "norm_weight_multiplications": 0,
            "rmsnorm_vector_outputs": 0, "selected_head_calls": 0, "model_operator_calls": 0,
        },
    }
    evidence["result"] = result
    result["tests"] = focused_tests(evidence)
    require([record(SOURCE), record(TEST)] == command["source_test_pins"],
            "source/test changed during the one permitted generation")
    result["dependency_pins"] = dependency_pins()
    result["phase_seconds"] = {
        "authentication": authenticated - start, "ten_anchors": computed - authenticated,
        "compile_and_oracle_tests": time.monotonic() - computed,
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    os.fsync(1)


def framed_capture(command, stdout, stderr, returncode):
    return (b"ACE3-COMMAND-CAPTURE-V1\n" + json_bytes({
        "command_bytes": len(command), "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr), "returncode": returncode,
    }) + command + stdout + stderr)


def validate_capture(output):
    output = Path(output)
    require(output == OUTPUT and not output.is_symlink(), "unexpected anchor capture root")
    metadata = json.loads((output / "capture.json").read_bytes())
    expected = {"command.json", "source.py", "test.py", "attempt.started.json",
                "stdout.raw", "stderr.raw", "raw.capture", "result.json"}
    require(set(metadata["files"]) == expected and metadata["returncode"] == 0
            and metadata["task"] == TASK, "incomplete or failed anchor capture")
    for name, pin in metadata["files"].items():
        require(pin["path"] == str(output / name) and not (output / name).is_symlink(),
                "capture file escaped its unique root")
        check_pin(pin)
    command_bytes = (output / "command.json").read_bytes()
    command = json.loads(command_bytes)
    stdout, stderr = (output / "stdout.raw").read_bytes(), (output / "stderr.raw").read_bytes()
    require((output / "raw.capture").read_bytes() == framed_capture(command_bytes, stdout, stderr, 0)
            and metadata["whole_capture"] == metadata["files"]["raw.capture"]
            and metadata["stdout"] == metadata["files"]["stdout.raw"]
            and metadata["stderr"] == metadata["files"]["stderr.raw"],
            "whole-command and separate-stream byte bindings do not close")
    require(command["argv"] == ARGV and command["task"] == TASK and command["attempt"] == ATTEMPT
            and command["cwd"] == str(ROOT) and command["uid"] == 1000
            and command["environment"] == ENVIRONMENT
            and command["original_source_path"] == str(output / "stdout.raw"),
            "captured command identity changed")
    result = json.loads((output / "result.json").read_bytes())
    require(stdout.splitlines(keepends=True)[-1] == (output / "result.json").read_bytes()
            and result["preregistration"] == PREREGISTRATION
            and result["task"] == TASK and result["attempt"] == ATTEMPT
            and result["status"] == "CAPTURED_PENDING_INDEPENDENT_REVIEW"
            and result["normal_host_review"] == "REQUIRED_BEFORE_CLASSIFIER_CONSUMPTION",
            "captured result/preregistration/review boundary changed")
    require(tuple(row["label"] for row in result["anchors"]) == LABELS
            and result["source_test_pins"] == command["source_test_pins"],
            "captured case census or compiled identities changed")
    lines = stdout.splitlines()
    require(lines[0] == b"PREREGISTRATION=" + json_bytes(PREREGISTRATION).rstrip(b"\n")
            and len(lines) == 12, "missing or extra generation-time scalar records")
    for line, row in zip(lines[1:-1], result["anchors"], strict=True):
        require(line == b"ANCHOR=" + json_bytes(row).rstrip(b"\n"),
                "generation-time scalar record was changed")
        validate_anchor(row)
    for name, pin in zip(("source.py", "test.py"), command["source_test_pins"], strict=True):
        require(record(output / name)["sha256"] == pin["sha256"],
                "captured source/test snapshot changed")
    return result


def sign(value):
    return (value > 0) - (value < 0)


@contextmanager
def classifier_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("mixed-factor classifier forbids replay or regeneration")

    with vector_cutoff.selected_only(audit), ExitStack() as stack:
        for target in (
            (sys.modules[__name__], ("anchor_worker", "capture_anchors")),
            (vector_cutoff, ("execute_check", "run_vectors", "run_tests")),
            (vector_cutoff.parent, ("rmsnorm", "logits")),
        ):
            module, names = target
            for name in names:
                if callable(getattr(module, name, None)):
                    stack.enter_context(patch.object(module, name, forbidden))
        yield


def exact_half_vector(words):
    validate_words(words)
    return [Fraction.from_float(float(value)) for value in words.view("<f2")]


def exact_weight_vector(values):
    require(isinstance(values, np.ndarray) and values.shape == (WIDTH,)
            and values.dtype.str == "<f2" and np.all(np.isfinite(values)),
            "weight vector binding changed")
    return [Fraction.from_float(float(value)) for value in values]


def anchor_map(anchor_result):
    require(tuple(row["label"] for row in anchor_result["anchors"]) == LABELS,
            "reviewed scalar anchor census changed")
    rows = {}
    for row in anchor_result["anchors"]:
        validate_anchor(row)
        rows[row["label"]] = row
    return rows


def authenticate_classifier_inputs():
    check_pin(ANCHOR_RESULT_PIN)
    check_pin(ANCHOR_RAW_PIN)
    check_pin(ANCHOR_CAPTURE_PIN)
    verify_review(bound_json(ANCHOR_REVIEW_PIN), None, TASK)
    anchor_result = validate_capture(OUTPUT)
    require(Path(ANCHOR_RAW_PIN["path"]).read_bytes()
            == framed_capture((OUTPUT / "command.json").read_bytes(),
                              (OUTPUT / "stdout.raw").read_bytes(),
                              (OUTPUT / "stderr.raw").read_bytes(), 0),
            "reviewed anchor raw capture frame changed")
    require(Path(ANCHOR_RESULT_PIN["path"]).read_bytes()
            == (OUTPUT / "stdout.raw").read_bytes().splitlines(keepends=True)[-1],
            "reviewed anchor result is not stdout suffix")
    operands = authenticate_operands()
    require(anchor_result["reviewed_vector"] == RESULT_PIN
            and anchor_result["reviewed_vector_capture"] == RAW_PIN
            and anchor_result["reviewed_vector_review"] == VECTOR_REVIEW_PIN,
            "reviewed anchor capture no longer binds the vector cutoff source")
    require(anchor_result["runtime_pins"] == operands["runtime_pins"],
            "anchor/vector runtime pins diverged")
    vector_result = operands["saved"]
    require(vector_result["decision"] == "SUPPORTED"
            and vector_result["direction_agreements"] == 27
            and vector_result["crossing_pattern_agreements"] == 27,
            "reviewed vector-cutoff result is not durable SUPPORTED input")
    rows = vector_cutoff.bridge.contributions.load_rows(vector_result["assets"], set(ROWS))
    require(tuple(rows) == ROWS, "selected tied-head row census changed")
    return {
        "anchor_result": anchor_result, "anchors": anchor_map(anchor_result),
        "operands": operands, "vector_result": vector_result, "row_weights": rows,
    }


def factor_vectors(actual_words, reference_words, norm_weight, actual_anchor, reference_anchor):
    actual, reference = exact_half_vector(actual_words), exact_half_vector(reference_words)
    weights = exact_weight_vector(norm_weight)
    sa, sr = Fraction(actual_anchor["inverse_norm_anchor"]), Fraction(reference_anchor["inverse_norm_anchor"])
    require(sa > 0 and sr > 0, "reviewed scalar anchors must be positive")
    vectors = {factor: [] for factor in FACTORS}
    full = []
    for ha, hr, weight in zip(actual, reference, weights, strict=True):
        direct = (ha - hr) * weight * sr
        scale = hr * weight * (sa - sr)
        interaction = (ha - hr) * weight * (sa - sr)
        vectors["direct_hidden"].append(direct)
        vectors["scalar_scale"].append(scale)
        vectors["interaction"].append(interaction)
        full.append(direct + scale + interaction)
        require(direct + scale + interaction == weight * (ha * sa - hr * sr),
                "mixed-factor coordinate closure failed")
    vectors["full"] = full
    return vectors


def exact_factor_dot(vector, row, audit):
    weights = exact_weight_vector(row)
    require(len(vector) == WIDTH, "factor vector width changed")
    audit["selected_row_factor_dot_computations"] += 1
    audit["selected_row_scalar_products"] += WIDTH
    return sum((value * weight for value, weight in zip(vector, weights, strict=True)), Fraction())


def vector_account_key(row):
    return row["control"], row["left_id"], row["right_id"], row["branch"]


def expected_vector_accounts(vector_result):
    accounts = [row for row in vector_result["accounts"] if row["branch"] == "fp16"]
    expected = {(control, left, right, "fp16")
                for control in CONTROLS for left, right in PAIRS}
    keys = [vector_account_key(row) for row in accounts]
    require(len(keys) == len(set(keys)) == 27 and set(keys) == expected,
            "reviewed vector account matrix changed")
    require(all(row["direction_agrees"] and row["crossing_pattern_agrees"]
                for row in accounts), "reviewed vector account is not supported")
    return accounts


def classify_case(row, factor_dots):
    before = Fraction(row["before_exact_margin"])
    reference_after = Fraction(row["after_exact_margin"])
    vector_sum = Fraction(row["parent_account"]["vector_coordinate_sum"])
    require(before - reference_after == vector_sum
            and row["observed_movement_sign"] == sign(reference_after - before)
            and row["observed_crossing_pattern"] == [sign(before), sign(reference_after)],
            "reviewed vector-cutoff movement target changed")
    left, right = row["left_id"], row["right_id"]
    factors = {}
    full_margin = Fraction()
    for factor in FACTORS:
        margin = factor_dots[factor][left] - factor_dots[factor][right]
        after = before - margin
        movement = after - before
        factors[factor] = {
            "factor_margin": str(margin),
            "after_exact_margin": str(after),
            "margin_movement": str(movement),
            "movement_sign": sign(movement),
            "crossing_pattern": [sign(before), sign(after)],
            "direction_agrees": sign(movement) == row["observed_movement_sign"],
            "crossing_pattern_agrees": [sign(before), sign(after)] == row["observed_crossing_pattern"],
        }
        full_margin += margin
    require(full_margin == factor_dots["full"][left] - factor_dots["full"][right],
            "summed factor margin does not equal full mixed factor")
    boundary = vector_sum - full_margin
    full_after = before - full_margin - boundary
    require(full_after == reference_after, "full mixed-factor plus boundary closure failed")
    return {
        "control": row["control"], "left_id": left, "right_id": right, "branch": row["branch"],
        "role": row["role"], "before_exact_margin": str(before),
        "reviewed_vector_after_exact_margin": str(reference_after),
        "reviewed_vector_margin_movement": row["vector_swap_margin_movement"],
        "reviewed_movement_sign": row["observed_movement_sign"],
        "reviewed_crossing_pattern": row["observed_crossing_pattern"],
        "retained_actual_margin": row["retained_actual_margin"],
        "retained_reference_margin": row["immutable_reference_margin"],
        "factor_results": factors,
        "full_factor_margin": str(full_margin),
        "retained_boundary_margin": str(boundary),
        "full_plus_boundary_after_margin": str(full_after),
        "closure_residuals": {
            "factor_sum_minus_full_factor": "0",
            "full_plus_boundary_minus_vector": str(full_margin + boundary - vector_sum),
            "full_plus_boundary_after_minus_reference_after": str(full_after - reference_after),
        },
    }


def classifier_decision(accounts):
    require(len(accounts) == 27, "classifier account census changed")
    for row in accounts:
        require(set(row["factor_results"]) == set(FACTORS), "factor account census changed")
        require(set(row["closure_residuals"].values()) == {"0"}, "full closure residual changed")
    supported = [
        factor for factor in FACTORS
        if all(row["factor_results"][factor]["direction_agrees"]
               and row["factor_results"][factor]["crossing_pattern_agrees"]
               for row in accounts)
    ]
    return ("SUPPORTED" if supported else "REJECTED"), supported


def run_classifier(evidence, audit):
    anchor_rows = evidence["anchors"]
    reference_words = evidence["operands"]["words"]["reference_fp16"]
    reference_anchor = anchor_rows["reference_fp16"]
    norm_weight = evidence["operands"]["norm_weight"]
    row_weights = evidence["row_weights"]
    accounts, dots, factor_closures = [], {}, []
    for control in CONTROLS:
        vectors = factor_vectors(evidence["operands"]["words"][control], reference_words,
                                 norm_weight, anchor_rows[control], reference_anchor)
        dots[control] = {
            factor: {row: exact_factor_dot(values, row_weights[row], audit) for row in ROWS}
            for factor, values in vectors.items()
        }
        for row in ROWS:
            factor_closures.append({
                "control": control, "row": row,
                "direct_plus_scale_plus_interaction_minus_full": str(
                    sum((dots[control][factor][row] for factor in FACTORS), Fraction())
                    - dots[control]["full"][row]),
            })
    require(audit["selected_row_factor_dot_computations"] == len(CONTROLS) * (len(FACTORS) + 1) * len(ROWS),
            "factor dot budget changed")
    require(audit["selected_row_scalar_products"]
            == len(CONTROLS) * (len(FACTORS) + 1) * len(ROWS) * WIDTH,
            "factor scalar-product budget changed")
    require(all(row["direct_plus_scale_plus_interaction_minus_full"] == "0"
                for row in factor_closures), "row-level factor closure failed")
    by_key = {vector_account_key(row): row for row in expected_vector_accounts(evidence["vector_result"])}
    for control in CONTROLS:
        for left, right in PAIRS:
            accounts.append(classify_case(by_key[control, left, right, "fp16"], dots[control]))
    decision, supported = classifier_decision(accounts)
    evidence.update(classifier_accounts=accounts, factor_dots=dots,
                    factor_closures=factor_closures, supported_factors=supported)
    return decision, supported


def classifier_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(record(path))
    spec = importlib.util.spec_from_file_location("mixed_factor_classifier_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused tests unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    require(suite.countTestCases() == EXPECTED_TESTS, "focused test census changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and not result.skipped, "focused classifier tests failed")
    return {"executed": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped), "compiled": compiled}


def execute_classifier():
    identity()
    audit = {"forbidden_calls": 0, "selected_row_factor_dot_computations": 0,
             "selected_row_scalar_products": 0}
    start = time.monotonic()
    with classifier_only(audit):
        evidence = authenticate_classifier_inputs()
        evidence["result"] = evidence["anchor_result"]
        evidence["words"] = evidence["operands"]["words"]
        authenticated = time.monotonic()
        print("CLASSIFIER_PREREGISTRATION=" + json.dumps(
            CLASSIFIER_PREREGISTRATION, sort_keys=True), flush=True)
        os.fsync(1)
        decision, supported = run_classifier(evidence, audit)
        classified = time.monotonic()
        result = {
            "diagnostic_id": NAME, "version": 1, "task": TASK,
            "phase": CLASSIFIER_PREREGISTRATION["phase"],
            "status": "CLASSIFIED_PENDING_INDEPENDENT_REVIEW",
            "execution_valid": True, "command": CLASSIFIER_COMMAND,
            "decision": decision, "supported_factors": supported,
            "successor": ({factor: FACTOR_SUCCESSORS[factor] for factor in supported}
                          if supported else CLASSIFIER_SUCCESSORS[decision]),
            "preregistration": CLASSIFIER_PREREGISTRATION,
            "accounts": evidence["classifier_accounts"],
            "factor_row_closures": evidence["factor_closures"],
            "factor_agreement_counts": {
                factor: {
                    "direction": sum(row["factor_results"][factor]["direction_agrees"]
                                     for row in evidence["classifier_accounts"]),
                    "crossing_pattern": sum(row["factor_results"][factor]["crossing_pattern_agrees"]
                                            for row in evidence["classifier_accounts"]),
                } for factor in FACTORS
            },
            "reviewed_anchor": ANCHOR_RESULT_PIN,
            "reviewed_anchor_capture": ANCHOR_RAW_PIN,
            "reviewed_anchor_review": ANCHOR_REVIEW_PIN,
            "reviewed_vector": RESULT_PIN,
            "reviewed_vector_capture": RAW_PIN,
            "reviewed_vector_review": VECTOR_REVIEW_PIN,
            "runtime_pins": evidence["operands"]["runtime_pins"],
            "source_test_pins": [record(SOURCE), record(TEST)],
            "dependency_pins": dependency_pins(),
            "authenticated_pins": [
                ANCHOR_RESULT_PIN, ANCHOR_RAW_PIN, ANCHOR_CAPTURE_PIN, ANCHOR_REVIEW_PIN,
                *evidence["operands"]["authenticated_pins"],
            ],
            "thresholds": evidence["vector_result"]["thresholds"],
            "final_reference_authority": evidence["vector_result"]["final_reference_authority"],
            "retained_controls_and_failure_gates": evidence["vector_result"][
                "retained_controls_and_failure_gates"],
            "original_execution_sources": evidence["vector_result"]["original_execution_sources"],
            "dispatch": {
                **audit, "cases": 27, "single_factor_substitutions": 81,
                "full_closure_cases": 27, "rmsnorm_invocations": 0,
                "rmsnorm_vector_outputs": 0, "scalar_anchor_regeneration": 0,
                "scalar_rne_computations": 0, "selected_head_rows": list(ROWS),
                "model_operator_calls": 0, "prefix_admission_reference_producer_replay": 0,
                "full_vocabulary_head_calls": 0, "artifact_overwrites": 0,
            },
            "normal_host_review": "REQUIRED",
            "claim_boundary": CLASSIFIER_BOUNDARY,
            "phase_seconds": {"authentication": authenticated - start,
                              "factor_arithmetic": classified - authenticated},
        }
        evidence["classifier_result"] = result
        result["tests"] = classifier_tests(evidence)
        result["phase_seconds"]["compile_and_tests"] = time.monotonic() - classified
        require([record(SOURCE), record(TEST)] == result["source_test_pins"],
                "source/test changed during classifier execution")
    return result


def validate_classifier_result(result):
    require(result["diagnostic_id"] == NAME and result["task"] == TASK
            and result["phase"] == CLASSIFIER_PREREGISTRATION["phase"]
            and result["status"] == "CLASSIFIED_PENDING_INDEPENDENT_REVIEW"
            and result["execution_valid"] and result["command"] == CLASSIFIER_COMMAND
            and result["normal_host_review"] == "REQUIRED",
            "classifier result identity changed")
    require(result["preregistration"] == CLASSIFIER_PREREGISTRATION,
            "classifier preregistration changed")
    for row in result["accounts"]:
        before = Fraction(row["before_exact_margin"])
        reviewed_after = Fraction(row["reviewed_vector_after_exact_margin"])
        require(row["reviewed_movement_sign"] == sign(
                Fraction(row["reviewed_vector_margin_movement"]))
                and row["reviewed_crossing_pattern"] == [sign(before), sign(reviewed_after)],
                "reviewed vector target account changed")
        for factor, account in row["factor_results"].items():
            margin = Fraction(account["factor_margin"])
            after = Fraction(account["after_exact_margin"])
            movement = Fraction(account["margin_movement"])
            require(after == before - margin and movement == after - before
                    and account["movement_sign"] == sign(movement)
                    and account["crossing_pattern"] == [sign(before), sign(after)]
                    and account["direction_agrees"] == (sign(movement) == row["reviewed_movement_sign"])
                    and account["crossing_pattern_agrees"]
                    == ([sign(before), sign(after)] == row["reviewed_crossing_pattern"]),
                    "stored factor account changed: " + factor)
        require(Fraction(row["full_factor_margin"]) == sum(
                Fraction(row["factor_results"][factor]["factor_margin"]) for factor in FACTORS)
                and Fraction(row["full_factor_margin"]) + Fraction(row["retained_boundary_margin"])
                == before - reviewed_after,
                "stored full mixed-factor closure changed")
    decision, supported = classifier_decision(result["accounts"])
    require(result["decision"] == decision and result["supported_factors"] == supported,
            "stored classifier decision changed")
    expected_successor = ({factor: FACTOR_SUCCESSORS[factor] for factor in supported}
                          if supported else CLASSIFIER_SUCCESSORS[decision])
    require(result["successor"] == expected_successor, "stored successor changed")
    for factor in FACTORS:
        counts = result["factor_agreement_counts"][factor]
        require(counts["direction"] == sum(
            row["factor_results"][factor]["direction_agrees"] for row in result["accounts"])
            and counts["crossing_pattern"] == sum(
                row["factor_results"][factor]["crossing_pattern_agrees"]
                for row in result["accounts"]), "factor agreement count changed")
    tests = result["tests"]
    require(tests["executed"] == EXPECTED_TESTS and len(tests["compiled"]) == 2
            and all(tests[k] == 0 for k in ("failures", "errors", "skipped")),
            "stored focused tests failed")
    dispatch = result["dispatch"]
    require(dispatch["cases"] == 27 and dispatch["single_factor_substitutions"] == 81
            and dispatch["selected_row_factor_dot_computations"] == 108
            and dispatch["selected_row_scalar_products"] == 96768
            and all(dispatch[k] == 0 for k in (
                "forbidden_calls", "rmsnorm_invocations", "rmsnorm_vector_outputs",
                "scalar_anchor_regeneration", "scalar_rne_computations",
                "model_operator_calls", "prefix_admission_reference_producer_replay",
                "full_vocabulary_head_calls", "artifact_overwrites")),
            "stored dispatch budget changed")


def validate_classifier_capture(output):
    output = Path(output).resolve(strict=True)
    require(output.parent == ROOT / "build" and output.name.startswith(CLASSIFIER_PREFIX)
            and not output.is_symlink(), "classifier output outside allowed build prefix")
    metadata = json.loads((output / "capture.json").read_bytes())
    expected = {"command.json", "source.py", "test.py", "attempt.started.json",
                "stdout.raw", "stderr.raw", "raw.capture", "result.json"}
    require(set(metadata["files"]) == expected and metadata["returncode"] == 0
            and metadata["task"] == TASK, "incomplete or failed classifier capture")
    for name, pin in metadata["files"].items():
        path = output / name
        require(pin["path"] == str(path) and not path.is_symlink(),
                "classifier capture file escaped output")
        check_pin(pin)
    command_bytes = (output / "command.json").read_bytes()
    command = json.loads(command_bytes)
    stdout, stderr = (output / "stdout.raw").read_bytes(), (output / "stderr.raw").read_bytes()
    require((output / "raw.capture").read_bytes()
            == framed_capture(command_bytes, stdout, stderr, metadata["returncode"])
            and metadata["whole_capture"] == metadata["files"]["raw.capture"]
            and metadata["stdout"] == metadata["files"]["stdout.raw"]
            and metadata["stderr"] == metadata["files"]["stderr.raw"],
            "classifier whole/stdout/stderr byte bindings do not close")
    require(command["argv"] == CLASSIFIER_ARGV and command["task"] == TASK
            and metadata["attempt"] == command["attempt"]
            and command["cwd"] == str(ROOT) and command["uid"] == 1000
            and command["environment"] == ENVIRONMENT
            and command["original_source_path"] == str(output / "stdout.raw")
            and command["classifier_source_path"] == str(SOURCE),
            "classifier command identity changed")
    lines = stdout.splitlines(keepends=True)
    require(len(lines) >= 2 and lines[0].rstrip(b"\n") == b"CLASSIFIER_PREREGISTRATION="
            + json_bytes(CLASSIFIER_PREREGISTRATION).rstrip(b"\n"),
            "classifier preregistration is not first stdout record")
    result_bytes = lines[-1]
    require((output / "result.json").read_bytes() == result_bytes,
            "classifier result is not stdout suffix")
    result = json.loads(result_bytes)
    validate_classifier_result(result)
    require(result["source_test_pins"] == command["source_test_pins"]
            == [record(SOURCE), record(TEST)], "classifier source/test pins changed")
    for name, pin in zip(("source.py", "test.py"), result["source_test_pins"], strict=True):
        require(record(output / name)["sha256"] == pin["sha256"],
                "classifier source/test snapshot changed")
    for pin in result["authenticated_pins"]:
        check_pin(pin)
    return {"status": "VALID_CLASSIFIER_CAPTURE", "decision": result["decision"],
            "supported_factors": result["supported_factors"],
            "successor": result["successor"], "normal_host_review": "REQUIRED",
            "whole_capture": record(output / "raw.capture"),
            "stdout": record(output / "stdout.raw"), "stderr": record(output / "stderr.raw"),
            "result": record(output / "result.json")}


def capture_classifier(attempt):
    identity()
    require(type(attempt) is int and attempt >= 1, "attempt must be positive")
    output = ROOT / "build" / f"{CLASSIFIER_PREFIX}attempt{attempt:03d}"
    require(subprocess.run(["git", "check-ignore", "-q", str(output)], cwd=ROOT,
                           check=False).returncode == 0,
            "classifier output is not ignored")
    output.mkdir(exist_ok=False)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    command = {
        "task": TASK, "attempt": attempt, "argv": CLASSIFIER_ARGV,
        "command": CLASSIFIER_COMMAND, "cwd": str(ROOT), "uid": os.getuid(),
        "branch": branch, "environment": ENVIRONMENT,
        "original_source_path": str(output / "stdout.raw"),
        "classifier_source_path": str(SOURCE),
        "anchor_source_path": str(OUTPUT / "result.json"),
        "source_test_pins": [record(SOURCE), record(TEST)],
        "runtime_pins": vector_cutoff.previous.runtime_pins(),
        "capture_format": "length-framed command/stdout/stderr; no interleaving inferred",
    }
    write_new(output / "command.json", json_bytes(command))
    for name, source in (("source.py", SOURCE), ("test.py", TEST)):
        write_new(output / name, source.read_bytes())
    write_new(output / "attempt.started.json", json_bytes({
        "task": TASK, "attempt": attempt, "rows": list(ROWS),
        "cases": len(CONTROLS) * len(PAIRS), "factors": list(FACTORS),
        "source_test_pins": command["source_test_pins"],
    }))
    with (output / "stdout.raw").open("xb") as stdout, (output / "stderr.raw").open("xb") as stderr:
        run = subprocess.run(CLASSIFIER_ARGV, cwd=ROOT, env={**os.environ, **ENVIRONMENT},
                             stdout=stdout, stderr=stderr, check=False)
        for stream in (stdout, stderr):
            stream.flush()
            os.fsync(stream.fileno())
    stdout, stderr = (output / "stdout.raw").read_bytes(), (output / "stderr.raw").read_bytes()
    write_new(output / "raw.capture", framed_capture(
        (output / "command.json").read_bytes(), stdout, stderr, run.returncode))
    if run.returncode == 0:
        write_new(output / "result.json", stdout.splitlines(keepends=True)[-1])
    files = {p.name: record(p) for p in output.iterdir() if p.is_file()}
    write_new(output / "capture.json", json_bytes({
        "task": TASK, "attempt": attempt, "returncode": run.returncode,
        "original_source_path": str(output / "stdout.raw"),
        "files": files, "whole_capture": files["raw.capture"],
        "stdout": files["stdout.raw"], "stderr": files["stderr.raw"],
    }))
    if run.returncode != 0:
        sys.stderr.buffer.write(stderr)
        print(json.dumps({"status": "CLASSIFIER_CAPTURE_FAILED", "output": str(output),
                          "returncode": run.returncode, "decision": "UNKNOWN"}))
        return run.returncode
    validation = validate_classifier_capture(output)
    write_new(output / "validation.json", json_bytes(validation))
    print(json.dumps({"status": "CLASSIFIED_PENDING_INDEPENDENT_REVIEW",
                      "output": str(output), **validation}, sort_keys=True))
    return 0


def capture_anchors():
    identity()
    require(subprocess.run(["git", "check-ignore", "-q", str(OUTPUT)], check=False).returncode == 0,
            "anchor output is not ignored")
    failures = []
    for output in FAILED_OUTPUTS:
        failure = json.loads((output / "capture.json").read_bytes())
        require(failure["returncode"] == 1 and failure["task"] == TASK,
                "prior attempt is not a preserved pre-arithmetic authentication failure")
        for name, pin in failure["files"].items():
            require(pin["path"] == str(output / name), "failed capture path changed")
            check_pin(pin)
        require((output / "stdout.raw").read_bytes() == b"",
                "prior attempt reached preregistration/arithmetic; another generation is forbidden")
        failures.append(record(output / "capture.json"))
    OUTPUT.mkdir(exist_ok=False)
    command = {
        "task": TASK, "attempt": ATTEMPT, "argv": ARGV, "command": shlex.join(ARGV),
        "cwd": str(ROOT), "uid": os.getuid(), "environment": ENVIRONMENT,
        "original_source_path": str(OUTPUT / "stdout.raw"), "producer_source_path": str(SOURCE),
        "source_test_pins": [record(SOURCE), record(TEST)],
        "preserved_pre_arithmetic_failures": failures,
        "capture_format": "length-framed command/stdout/stderr; no interleaving inferred",
    }
    write_new(OUTPUT / "command.json", json_bytes(command))
    for name, source in (("source.py", SOURCE), ("test.py", TEST)):
        write_new(OUTPUT / name, source.read_bytes())
    with (OUTPUT / "stdout.raw").open("xb") as stdout, (OUTPUT / "stderr.raw").open("xb") as stderr:
        run = subprocess.run(ARGV, cwd=ROOT, env={**os.environ, **ENVIRONMENT},
                             stdout=stdout, stderr=stderr, check=False)
        for stream in (stdout, stderr):
            stream.flush()
            os.fsync(stream.fileno())
    stdout, stderr = (OUTPUT / "stdout.raw").read_bytes(), (OUTPUT / "stderr.raw").read_bytes()
    write_new(OUTPUT / "raw.capture", framed_capture(
        (OUTPUT / "command.json").read_bytes(), stdout, stderr, run.returncode))
    if run.returncode == 0:
        write_new(OUTPUT / "result.json", stdout.splitlines(keepends=True)[-1])
    files = {p.name: record(p) for p in OUTPUT.iterdir() if p.is_file()}
    write_new(OUTPUT / "capture.json", json_bytes({
        "task": TASK, "attempt": ATTEMPT, "returncode": run.returncode,
        "original_source_path": str(OUTPUT / "stdout.raw"), "files": files,
        "whole_capture": files["raw.capture"], "stdout": files["stdout.raw"],
        "stderr": files["stderr.raw"],
    }))
    if run.returncode != 0:
        sys.stderr.buffer.write(stderr)
        print(json.dumps({"status": "CAPTURE_FAILED", "output": str(OUTPUT),
                          "returncode": run.returncode, "scientific_classification": None}))
        return run.returncode
    result = validate_capture(OUTPUT)
    print(json.dumps({"status": result["status"], "output": str(OUTPUT),
                      "tests": result["tests"], "dispatch": result["dispatch"],
                      "whole_capture": files["raw.capture"],
                      "next_owner": "reviewer"}, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture-anchors", action="store_true")
    mode.add_argument("--anchor-worker", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--validate-anchors", type=Path)
    mode.add_argument("--run-classifier", action="store_true")
    mode.add_argument("--classifier-worker", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--validate-classifier", type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.capture_anchors:
        return capture_anchors()
    if args.anchor_worker:
        anchor_worker()
        return 0
    if args.run_classifier:
        return capture_classifier(args.attempt)
    if args.classifier_worker:
        try:
            result = execute_classifier()
        except (OSError, ValueError, RuntimeError) as error:
            print(str(error), file=sys.stderr, flush=True)
            print(json.dumps({
                "diagnostic_id": NAME, "execution_valid": False, "decision": "UNKNOWN",
                "successor": CLASSIFIER_SUCCESSORS["UNKNOWN"], "error": str(error),
                "runtime_pins": vector_cutoff.previous.runtime_pins(),
                "normal_host_review": "REQUIRED", "claim_boundary": CLASSIFIER_BOUNDARY,
            }, sort_keys=True), flush=True)
            return 1
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    if args.validate_classifier:
        print(json.dumps(validate_classifier_capture(args.validate_classifier), sort_keys=True))
        return 0
    result = validate_capture(args.validate_anchors)
    print(json.dumps({"status": result["status"], "scientific_recomputation": 0,
                      "normal_host_review": result["normal_host_review"]}))
    return 0


if __name__ == "__main__":
    sys.modules[MODULE] = sys.modules[__name__]
    raise SystemExit(main())
