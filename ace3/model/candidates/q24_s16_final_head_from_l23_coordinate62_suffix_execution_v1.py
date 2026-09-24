"""Nine isolated CPU final-head suffixes; --validate never reruns the operators.

Run the disclosed --execute --out build/<name>_attemptNNN command once, then
--validate with the same output. Validation emits JSON only to stdout.
Neither mode publishes a token, adopts a policy, or admits a candidate.
"""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model import fp16_adaptation_oracle as norm
from ace3.model import streaming_lm_head_reference as head
from ace3.model.candidates import q24_s16_final_head_from_l23_coordinate62_suffix_preflight_v1 as preflight


ROOT = preflight.ROOT
NAME = "q24_s16_final_head_from_l23_coordinate62_suffix_execution_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
PYTHON = preflight.parent.PYTHON
CONTROLS = preflight.CONTROLS
REVIEW_MISSION = "df999d90b523"
require, same, record, read_bound = (
    preflight.require, preflight.same, preflight.record, preflight.read_bound)
write = preflight.parent.write
PINS = {
    "source": {"path": str(preflight.SOURCE),
               "sha256": "61a26c08f7bece930618e9e53cda4035444b0f840b9d6297a93dd0a5cbb2a7b7"},
    "test": {"path": str(ROOT / f"tests/test_{preflight.NAME}.py"),
             "sha256": "57073b6aea56df2f23ab6bc51325847435ad89770b1715529a562243731eb36a"},
    "review": {"path": str(preflight.preflight.HANDOFFS / REVIEW_MISSION / "round-0001.json"),
               "sha256": "a97ca91bc6cec5202179d0a1090c6e6a365472fc23624b8eacc575f0cba37b4c"},
    "mission": {"path": str(preflight.preflight.HANDOFFS / REVIEW_MISSION / "mission.json"),
                "sha256": "c0bd7bef447e6bb3be89e0477f4676cbb5d0e6b5c2ce29f6cc231dccd3252a1a"},
}
FLAGS = {
    **preflight.parent.FLAGS, "native_L0_L23_invocations": 0,
    "native_layer_invocations": 0, "external_invocations": 0,
    "tokenizer_decode_invocations": 0, "token_published": False,
    "reference_rmsnorm_invocations": 0, "reference_lm_head_invocations": 0,
    "retained_evidence_writes": 0,
}
BOUNDARY = (
    "Nine isolated final RMSNorm/tied lm_head/top-k CPU-software suffixes of raw failed "
    "L23/P0 states only. Preserve all L21/L22/L23 failures, thresholds, original-input "
    "global references and source/operand/state/KV/lineage gates. Q24 residual state "
    "is wider than FP16; INT4 weights and FP16 operators/KV are unchanged. No prefix "
    "or admission replay, reference recomputation/reanchoring, decoder, RTL, hardware, "
    "GPU, simulation, external service, token/successor publication, strict-FP16-state "
    "W4A16, new-token or full-model admission. Normal independent Host review REQUIRED."
)
ARITHMETIC = {
    "rmsnorm": "existing integer RMSNorm; epsilon Q48=281474977; integer sqrt; Q24 RNE",
    "lm_head": "exact FP16 products, Q47.48 sum, one FP16 RNE; checked int64 subrange",
    "top_k": preflight.FINAL_CONTRACT["head"]["order"],
    "comparison": "diagnostic comparisons only; no new final-head acceptance threshold",
}
EXPECTED_TESTS = 22
RETAINED_RESULT = {
    "path": str(OUTPUT / "result.json"),
    "sha256": "83fdad3480a01361f80362096f4842cca5b33b66fdf6138b187ed29a384bcb1f",
    "bytes": 317209,
}
RETAINED_SOURCES = {
    MODULE: {
        "path": str(SOURCE),
        "sha256": "5a3a3e3700afef82b149c8cff60e69b0bf0c9ba0784ad35501596fa1f44adb2a",
        "bytes": 22121,
    },
    "focused_test": {
        "path": str(TEST),
        "sha256": "e978c668567c0ce82b3ff6f8af247a8d97e729efd7d1b3f78b112234f28efbc5",
        "bytes": 5723,
    },
}


def output_path(value, *, fresh):
    value = Path(value)
    out = value if value.is_absolute() else ROOT / value
    require(out.resolve() == out and out.parent == ROOT / "build"
            and re.fullmatch(NAME + r"_attempt[0-9]{3,}", out.name) is not None
            and int(out.name.rsplit("attempt", 1)[1]) > 0, "noncanonical output")
    require(not out.is_symlink() and (not out.exists() if fresh else out.is_dir()),
            "output occupied, symlinked or missing")
    checked = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--", str(out)],
        capture_output=True, check=False)
    require(checked.returncode == 0, "output not confirmed ignored")
    return out


def command_for(out):
    return (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE}"
            f" --execute --out build/{out.name}")


def source_context():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == PYTHON and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize and os.getuid() == 1000,
            "source/workdir/interpreter/account/bytecode gate failed")
    branch = subprocess.run(["git", "-C", str(ROOT), "branch", "--show-current"],
                            capture_output=True, text=True, check=True).stdout.strip()
    require(branch == "argus/full-projection", "isolated branch changed")
    sources = {}
    for name, module in sorted(sys.modules.items()):
        if name == "ace3" or name.startswith("ace3."):
            path = getattr(module, "__file__", None)
            if path:
                path = Path(path)
                require(path.is_relative_to(ROOT) and path.resolve() == path,
                        "source origin escaped isolated worktree")
                sources[name] = record(path)
    sources[MODULE], sources["focused_test"] = record(SOURCE), record(TEST)
    return sources


def check_summary(summary):
    require(summary["status"] == "PREFLIGHT_READY"
            and summary["asset_binding_revision"] == 3
            and summary["control_count"] == 9, "reviewed preflight not ready")
    same([row["control"] for row in summary["controls"]], list(CONTROLS),
         "raw failed-parent schedule changed")
    same(summary["final_contract"], preflight.FINAL_CONTRACT, "operator contract changed")
    same(summary["final_reference"]["authority"], preflight.FINAL_REFERENCE_PINS,
         "final-reference authority changed")
    same(summary["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9},
         "L23 failure history changed")
    for row in summary["controls"]:
        require(row["retained_L23"]["L23_status"] == "FAIL"
                and row["retained_L23"]["retained_L22"]["L22_status"] == "FAIL"
                and row["retained_L23"]["retained_L21"]["S18"] == "FAIL",
                "failed-parent history removed")


def authenticate():
    for pin in PINS.values():
        read_bound(pin)
    review, mission = (json.loads(read_bound(PINS[key])) for key in ("review", "mission"))
    latest = json.loads((preflight.preflight.HANDOFFS / REVIEW_MISSION / "latest.json").read_bytes())
    backlog = [json.loads(line) for line in preflight.preflight.BACKLOG.read_text().splitlines()
               if line.strip()]
    preflight.check_review(review, latest, backlog, mission=REVIEW_MISSION,
                           round_number=1, pin=PINS["review"])
    require(mission["mission_id"] == REVIEW_MISSION
            and mission["node_key"] == "bind-final-head-preflight-reviewed-reference"
            and mission["execution_workdir"] == str(ROOT), "preflight review scope changed")
    evidence = preflight.authenticate()
    check_summary(evidence["summary"])
    return evidence


@contextmanager
def no_dispatch(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("decoder/prefix/admission/reference/external dispatch forbidden")

    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates.") and name != MODULE:
                for attribute in ("stages", "_stages", "continuation_stages", "native_layer",
                                  "execute_layer", "execute_layers", "execute", "run",
                                  "diagnose", "original_branches", "reference_arrays"):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        stack.enter_context(patch.object(subprocess, "Popen", forbidden))
        stack.enter_context(patch.object(os, "system", forbidden))
        yield


def checked_array(payload, shape, dtype):
    require(payload.startswith(b"\x93NUMPY"), "expected NPY")
    stream = io.BytesIO(payload)
    array = np.load(stream, allow_pickle=False)
    require(array.shape == shape and array.dtype.str == dtype
            and stream.tell() == len(payload)
            and np.all(np.isfinite(array.view("<f2") if dtype == "<u2" else array)),
            "invalid final array")
    canonical = io.BytesIO()
    np.save(canonical, array, allow_pickle=False)
    require(canonical.getvalue() == payload, "noncanonical NPY")
    return array


def load_references(summary):
    final = summary["final_reference"]["reference"]
    return {key: checked_array(read_bound(final[key]), shape, dtype)
            for key, shape, dtype in preflight.FINAL_ARRAYS}


def load_operands(summary):
    assets = summary["assets"]
    with preflight.parent.producer.legacy.safe_open(
            assets["checkpoint"]["path"], framework="numpy") as model:
        operands = [model.get_tensor(name) for name in ("model.norm.weight", "lm_head.weight")]
    for name, array in zip(("model.norm.weight", "lm_head.weight"), operands, strict=True):
        pin = assets["tensors"][name]
        require(array.dtype.str == "<f2" and list(array.shape) == pin["shape"]
                and np.all(np.isfinite(array))
                and hashlib.sha256(array.tobytes()).hexdigest() == pin["sha256"],
                "final operand changed after authentication")
        array.flags.writeable = False
    return operands


def rmsnorm(words, weights):
    outputs, mean, root = norm.rmsnorm(words.tolist(), weights.view("<u2").tolist())
    require(all(not invalid and not saturated for _, invalid, saturated in outputs),
            "invalid or saturated final RMSNorm")
    return np.asarray([bits for bits, _, _ in outputs], dtype="<u2"), {
        "mean_q48": mean, "root_q24": root}


def logits(words, weights):
    hidden = head.decode_array_q24(words)
    output = np.empty(len(weights), dtype="<u2")
    hidden_bound = sum(abs(int(value)) for value in hidden)
    for start in range(0, len(weights), 512):
        chunk = head.decode_array_q24(weights[start:start + 512].view("<u2"))
        # This proves every product and partial sum fits the exact int64 subrange
        # of Q47.48; no overflow, truncation, floating matmul or saturation fallback.
        require(int(np.max(np.abs(chunk))) * hidden_bound <= (1 << 63) - 1,
                "exact Q48 int64 accumulation bound exceeded")
        accumulators = np.sum(chunk * hidden, axis=1, dtype=np.int64)
        for index, accumulator in enumerate(accumulators, start):
            bits, saturated = head.fixed_to_f16(int(accumulator), 48)
            require(not saturated, "final logit saturated")
            output[index] = bits
    return output


def top_k(array, *, words):
    values = array.view("<f2").astype("<f8") if words else array
    require(values.ndim == 1 and np.all(np.isfinite(values)), "nonfinite top-k")
    ids = np.lexsort((np.arange(len(values)), -values))[:10]
    return [{"token_id": int(index), "value_hex": float(values[index]).hex()}
            for index in ids]


def comparison(actual, binary64, fp16):
    values = actual.view("<f2").astype("<f8")
    errors = np.abs(values - binary64)
    largest = np.flatnonzero(errors == np.max(errors))
    exact = lambda index: abs(Fraction.from_float(float(values[index]))
                              - Fraction.from_float(float(binary64[index])))
    worst = max(map(int, largest), key=exact)
    mismatches = np.flatnonzero(actual != fp16)
    return {
        "elements": len(actual), "binary64_max_absolute_error": str(exact(worst)),
        "binary64_worst_index": worst,
        "binary64_reference_hex": float(binary64[worst]).hex(),
        "actual_fp16_bits": f"{int(actual[worst]):04x}",
        "reviewed_fp16_mismatch_count": int(len(mismatches)),
        "reviewed_fp16_first_mismatch": int(mismatches[0]) if len(mismatches) else None,
        "comparison_scope": "diagnostic; no final-head admission threshold",
    }


def comparisons(arrays, references):
    result = {key: comparison(arrays[key], references[key + "_binary64"],
                              references[key + "_fp16"]) for key in ("rmsnorm", "logits")}
    actual = top_k(arrays["logits"], words=True)
    result["top_k"] = {"actual_diagnostic_only": actual}
    ids = [row["token_id"] for row in actual]
    for key, words in (("binary64", False), ("fp16", True)):
        expected = top_k(references["logits_" + key], words=words)
        expected_ids = [row["token_id"] for row in expected]
        result["top_k"][key] = {
            "reference": expected, "same_order": ids == expected_ids,
            "overlap_count": len(set(ids) & set(expected_ids)),
        }
    return result


def run_control(label, state, operands, audit):
    index = len(audit["controls"])
    require(index < 9 and label == CONTROLS[index], "extra or reordered final-head control")
    preflight.preflight.parent.check_state(state)
    audit["controls"].append(label)
    started = time.monotonic()
    audit["final_rmsnorm_invocations"] += 1
    normalized, norm_details = rmsnorm(state["stage18"], operands[0])
    norm_done = time.monotonic()
    audit["lm_head_invocations"] += 1
    scores = logits(normalized, operands[1])
    return {"rmsnorm": normalized, "logits": scores}, norm_details, {
        "rmsnorm": norm_done - started, "lm_head": time.monotonic() - norm_done}


def new_audit():
    return {"controls": [], "final_rmsnorm_invocations": 0, "lm_head_invocations": 0,
            "top_k_invocations": 0, "forbidden_calls": 0}


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(record(path))
    spec = importlib.util.spec_from_file_location("final_head_execution_tests", TEST)
    require(spec is not None and spec.loader is not None, "test loader missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    require(suite.countTestCases() == EXPECTED_TESTS, "focused test census changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "focused tests failed, errored or skipped")
    return {"compiled": compiled, "executed": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped)}


def check_result(result, evidence, out):
    require(result["diagnostic_id"] == NAME and result["status"] == "FINAL_HEAD_EVIDENCE"
            and result["output"] == str(out) and result["output_created_exclusively"] is True
            and result["normal_host_review"] == "REQUIRED", "execution identity changed")
    same(result["flags"], FLAGS, "non-admission boundary changed")
    same(result["claim_boundary"], BOUNDARY, "claim boundary changed")
    same(result["arithmetic"], ARITHMETIC, "arithmetic contract changed")
    same(result["preflight_pins"], PINS, "preflight pins changed")
    same(result["preflight"], evidence["summary"], "retained history/reference/threshold splice")
    same(result["audit"], {
        "controls": list(CONTROLS), "final_rmsnorm_invocations": 9,
        "lm_head_invocations": 9, "top_k_invocations": 9, "forbidden_calls": 0,
    }, "final-head execution census changed")
    same([row["control"] for row in result["controls"]], list(CONTROLS), "control census changed")
    for row, plan in zip(result["controls"], evidence["summary"]["controls"], strict=True):
        same(row["parent"], plan, "raw failed L23 parent/history changed")
    tests = result["tests"]
    require(tests["executed"] == (18 if out == OUTPUT else EXPECTED_TESTS)
            and all(tests[key] == 0 for key in ("failures", "errors", "skipped")),
            "compile/test evidence changed")
    same([pin["path"] for pin in tests["compiled"]], [str(SOURCE), str(TEST)],
         "compiled source census changed")
    if out == OUTPUT:
        same(tests["compiled"], list(RETAINED_SOURCES.values()),
             "retained execution compiled sources changed")
    else:
        for pin in tests["compiled"]:
            read_bound(pin)


def execute(value):
    out = output_path(value, fresh=True)
    origins = source_context()
    out.mkdir(exist_ok=False)
    audit, rows = new_audit(), []
    write(out / "command.json", {
        "command": command_for(out), "cwd": str(ROOT), "uid": os.getuid(),
        "executable": sys.executable, "argv": sys.argv,
        "host_managed_controls": "unchanged role/model/account/budget/access/locks; no service calls",
    })
    try:
        started = time.monotonic()
        evidence = authenticate()
        tests = focused_tests(evidence)
        references = load_references(evidence["summary"])
        operands = load_operands(evidence["summary"])
        authentication_seconds = time.monotonic() - started
        artifacts = [record(out / "command.json")]
        with no_dispatch(audit):
            for plan in evidence["summary"]["controls"]:
                label = plan["control"]
                arrays, details, timings = run_control(label, evidence["states"][label],
                                                       operands, audit)
                start = time.monotonic()
                audit["top_k_invocations"] += 1
                metrics = comparisons(arrays, references)
                timings["comparison_and_top_k"] = time.monotonic() - start
                pins = {}
                for key, array in arrays.items():
                    path = out / f"{label}_{key}.npy"
                    with path.open("xb") as stream:
                        np.save(stream, array, allow_pickle=False)
                    pins[key] = record(path)
                    artifacts.append(pins[key])
                rows.append({"control": label, "parent": plan, "arrays": pins,
                             "comparisons": metrics, "rmsnorm_integer_details": details,
                             "timing_seconds": timings})
                print(json.dumps({"control": label, "status": "FINAL_HEAD_RECORDED",
                                  "comparisons": metrics}), flush=True)
        same(source_context(), origins, "source changed during execution")
        write(out / "result.json", {
            "diagnostic_id": NAME, "status": "FINAL_HEAD_EVIDENCE", "output": str(out),
            "output_created_exclusively": True, "preflight_pins": PINS,
            "preflight": evidence["summary"], "origins": origins, "tests": tests,
            "audit": audit, "controls": rows, "artifacts": artifacts, "flags": FLAGS,
            "claim_boundary": BOUNDARY, "arithmetic": ARITHMETIC,
            "authentication_and_tests_seconds": authentication_seconds,
            "normal_host_review": "REQUIRED",
        })
    except Exception as error:
        write(out / "failure.json", {
            "status": "FAILED", "error_type": type(error).__name__, "error": str(error),
            "audit": audit, "completed_controls": [row["control"] for row in rows],
            "flags": FLAGS,
        })
        raise


def validate(value):
    out = output_path(value, fresh=False)
    origins = source_context()
    evidence = authenticate()
    result_pin = record(out / "result.json")
    execution_origins = origins
    if out == OUTPUT:
        # Bind the immutable execution, not the later validator/test repair.
        same(result_pin, RETAINED_RESULT, "retained execution result changed")
        execution_origins = {**origins, **RETAINED_SOURCES}
    result = json.loads(read_bound(result_pin))
    check_result(result, evidence, out)
    same(result["origins"], execution_origins, "execution source bindings changed")
    expected = {"command.json", "result.json"} | {
        f"{label}_{key}.npy" for label in CONTROLS for key in ("rmsnorm", "logits")}
    require({path.name for path in out.iterdir()} == expected,
            "unexpected/missing evidence files")
    pins = {pin["path"]: pin for pin in result["artifacts"]}
    require(len(pins) == len(result["artifacts"]) == 19
            and set(pins) == {str(out / name) for name in expected - {"result.json"}},
            "artifact census changed")
    for pin in pins.values():
        read_bound(pin)
    command = json.loads(read_bound(pins[str(out / "command.json")]))
    require(command["command"] == command_for(out) and command["cwd"] == str(ROOT)
            and command["uid"] == 1000 and command["executable"] == PYTHON,
            "disclosed command changed")
    references, audit = load_references(evidence["summary"]), new_audit()
    with no_dispatch(audit), patch(MODULE + ".rmsnorm", side_effect=RuntimeError("no recomputation")), \
            patch(MODULE + ".logits", side_effect=RuntimeError("no recomputation")):
        for row in result["controls"]:
            arrays = {}
            for key, shape in (("rmsnorm", (896,)), ("logits", (151936,))):
                pin = pins[str(out / f"{row['control']}_{key}.npy")]
                same(row["arrays"][key], pin, "control array splice")
                arrays[key] = checked_array(read_bound(pin), shape, "<u2")
            same(row["comparisons"], comparisons(arrays, references),
                 "stored RMSNorm/logit/top-k comparison changed")
    same(audit, new_audit(), "validation dispatched computation")
    return {"diagnostic_id": NAME, "status": "VALIDATED_FINAL_HEAD_EVIDENCE",
            "result": result_pin, "controls_verified": 9, "arrays_verified": 18,
            "validation_origins": origins, "execution_origins": execution_origins,
            "validation_dispatch": audit, "retained_L23_failures": 9,
            "normal_host_review": "REQUIRED", "flags": FLAGS, "claim_boundary": BOUNDARY}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args(argv)
    if args.execute:
        execute(args.out)
    else:
        print(json.dumps(validate(args.out), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
