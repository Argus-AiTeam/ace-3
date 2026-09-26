"""Retained b497 S18 localizer with a producer-free final arithmetic import path."""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import stage11_s18_boundary_localizer_v1 as retained


ROOT = retained.ROOT
SOURCE = Path(__file__).resolve()
TEST = ROOT / "tests/test_stage11_s18_boundary_localizer_v2.py"
SUMMARY = {
    "path": str(ROOT / "build/q24_s16_l23_from_l22_coordinate62_suffix_execution_v1_attempt001/result.json"),
    "sha256": "e0f23a1d333ce4389b936a900e2e2881def2ddea07a9a10fa50a554f5d70aac9",
}
require = retained.require
BOUNDARY = retained.BOUNDARY
FORBIDDEN = ("stages", "_stages", "continuation_stages", "native_layer",
             "execute_layer", "execute_layers", "execute", "run", "diagnose",
             "original_branches", "reference_arrays")


def new_audit():
    return dict.fromkeys(("forbidden_calls", "admission_invocations", "prefix_invocations",
                          "reference_invocations", "producer_invocations", "service_invocations",
                          "final_rmsnorm_invocations", "tied_pair_head_invocations"), 0)


class ReadOnly:
    def __init__(self, audit):
        self.audit = audit

    def __call__(self, event, args):
        if ((event == "open" and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
                or event in ("subprocess.Popen", "os.system", "os.remove", "os.rename",
                             "os.mkdir", "socket.connect", "socket.bind")):
            self.audit["forbidden_calls"] += 1
            raise RuntimeError(f"retained-only diagnostic forbids {event}: {args!r}")


@contextmanager
def no_dispatch(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("decoder/prefix/admission/reference/external dispatch forbidden")

    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates.") and module is not sys.modules[__name__]:
                for attribute in FORBIDDEN:
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        stack.enter_context(patch.object(retained, "check", forbidden))
        stack.enter_context(patch.object(retained, "main", forbidden))
        stack.enter_context(patch.object(subprocess, "Popen", forbidden))
        stack.enter_context(patch.object(os, "system", forbidden))
        yield


def load_operands(assets):
    from safetensors import safe_open

    retained.bound(assets["checkpoint"])
    with safe_open(assets["checkpoint"]["path"], framework="numpy") as model:
        operands = [model.get_tensor(name) for name in ("model.norm.weight", "lm_head.weight")]
    for name, array in zip(("model.norm.weight", "lm_head.weight"), operands, strict=True):
        binding = assets["tensors"][name]
        require(array.dtype.str == "<f2" and list(array.shape) == binding["shape"]
                and np.all(np.isfinite(array))
                and hashlib.sha256(array.tobytes()).hexdigest() == binding["sha256"],
                "final operand changed after authentication")
    require(operands[0].shape == (896,) and operands[1].shape == (151936, 896),
            "final operand width/vocabulary changed")
    selected = (operands[0], operands[1][list(retained.PAIR)].copy())
    for array in selected:
        array.flags.writeable = False
    return selected


def operand_assets(summary):
    from ace3.model import streaming_lm_head_reference as head

    checkpoint = summary["checkpoint"]
    require(checkpoint["path"] == str(ROOT / "build/model24_rtl_cascade/checkpoint/model.safetensors")
            and checkpoint["sha256"] == head.CHECKPOINT_SHA256
            and checkpoint["bytes"] == head.CHECKPOINT_BYTES, "retained checkpoint identity changed")
    return {"checkpoint": checkpoint, "tensors": {
        "model.norm.weight": {"shape": [896], "sha256": head.FINAL_NORM_SHA256},
        "lm_head.weight": {"shape": [151936, 896], "sha256": head.TIED_WEIGHT_SHA256}}}


def rmsnorm(words, weights):
    from ace3.model import fp16_adaptation_oracle as norm

    outputs, mean, root = norm.rmsnorm(words.tolist(), weights.view("<u2").tolist())
    require(all(not invalid and not saturated for _, invalid, saturated in outputs),
            "invalid or saturated final RMSNorm")
    return np.asarray([bits for bits, _, _ in outputs], dtype="<u2"), {
        "mean_q48": mean, "root_q24": root}


def logits(words, weights):
    from ace3.model import streaming_lm_head_reference as head

    hidden = head.decode_array_q24(words)
    chunk = head.decode_array_q24(weights.view("<u2"))
    require(int(np.max(np.abs(chunk))) * sum(abs(int(v)) for v in hidden) <= (1 << 63) - 1,
            "exact Q48 int64 accumulation bound exceeded")
    output = []
    for accumulator in np.sum(chunk * hidden, axis=1, dtype=np.int64):
        bits, saturated = head.fixed_to_f16(int(accumulator), 48)
        require(not saturated, "final logit saturated")
        output.append(bits)
    return np.asarray(output, dtype="<u2")


def check(audit):
    result, records = retained.authenticate()
    from ace3.model.candidates import binary64_fp16_excess_v1 as policy

    require(policy.EXCESS_BUDGET == Fraction(1, 8), "S18 excess threshold changed")
    summary = json.loads(retained.bound(SUMMARY))
    assets = operand_assets(summary)
    operands = load_operands(assets)
    reference_pins = result["frozen_contract"]["references"]
    reference64 = np.load(io.BytesIO(retained.bound(reference_pins["original_input_L23_binary64"])),
                          allow_pickle=False)
    require(reference64.shape == (896,) and reference64.dtype.str == "<f8",
            "S18 reference type changed")
    references = {b: np.load(io.BytesIO(retained.bound(reference_pins["original_input_final_" + b])),
                             allow_pickle=False) for b in ("fp16", "binary64")}
    own_sources = {"candidate": retained.pin(SOURCE), "test": retained.pin(TEST),
                   "retained_helpers": retained.pin(retained.SOURCE)}
    spec = importlib.util.spec_from_file_location("stage11_s18_boundary_localizer_v2_oracle", TEST)
    require(spec is not None and spec.loader is not None, "missing suffix oracle")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    table = retained.finite_table()
    original = retained.encoded(result)
    rows, controls = [], []
    with no_dispatch(audit):
        for control in retained.CONTROLS:
            output = result["outputs"][control]
            words, reports = retained.adjust(
                output["stage18"], reference64,
                result["stage_reports"][control][-1]["binary64_v1"]["failures"],
                table, policy.evaluate_layer_final_output)
            start = time.monotonic()
            audit["final_rmsnorm_invocations"] += 1
            norm, account = rmsnorm(words, operands[0])
            norm_end = time.monotonic()
            audit["tied_pair_head_invocations"] += 1
            scores = logits(norm, operands[1])
            head_end = time.monotonic()
            oracle.verify_suffix(words, operands, norm, account, scores)
            new = retained.scalar(int(scores[0])) - retained.scalar(int(scores[1]))
            old = retained.scalar(output["pair_logits"][0]) - retained.scalar(output["pair_logits"][1])
            for branch in ("fp16", "binary64"):
                parent = next(r for r in result["rows"]
                              if r["control"] == control and r["branch"] == branch)
                ref = references[branch]
                margin = (retained.scalar(int(ref[retained.PAIR[0]]))
                          - retained.scalar(int(ref[retained.PAIR[1]])) if branch == "fp16" else (
                              Fraction.from_float(float(ref[retained.PAIR[0]]))
                              - Fraction.from_float(float(ref[retained.PAIR[1]]))))
                require(str(margin) == parent["independent_original_reference_margin"]
                        and str(old) == parent["intervened_margin"], "terminal reference/operand splice")
                delta = new - Fraction(parent["retained_margin"])
                rows.append({**parent, "b497_margin": str(old), "adjusted_margin": str(new),
                             "adjusted_margin_error": str(new - margin), "margin_delta": str(delta),
                             "boundary_adjustment_delta": str(new - old),
                             "direction_observed": Fraction(parent["predicted_delta"]) * delta > 0})
            controls.append({"control": control, "changed_coordinates": [r["index"] for r in reports],
                             "substitutions": reports, "adjusted_stage18": words.tolist(),
                             "final_rmsnorm": norm.tolist(), "pair_logits": scores.tolist(),
                             "S18_binary64_status": "PASS", "independent_suffix_oracle": "PASS",
                             "timing_seconds": {"rmsnorm": norm_end - start,
                                                "pair_head": head_end - norm_end}})
    require(retained.encoded(result) == original, "retained bytes changed in memory")
    for record in [*records, *retained.PINS.values(), SUMMARY, assets["checkpoint"],
                   *own_sources.values()]:
        retained.bound(record)
    require(audit == {**new_audit(), "final_rmsnorm_invocations": 9,
                     "tied_pair_head_invocations": 9}, "suffix/forbidden dispatch census changed")
    status = retained.classify(rows)
    return {"status": status, "hypothesis": "S18 boundary substitutions remove opposite terminal response",
            "classification": ("direction_restored" if status == "SUPPORTED"
                               else "boundary_only_explanation_rejected"),
            "controls": controls, "rows": rows, "failure_union": list(retained.UNION),
            "common_failures": list(retained.COMMON), "sources": own_sources,
            "retained": retained.PINS, "retained_bytes_preserved": True,
            "frozen_contract": result["frozen_contract"], "operand_assets": assets,
            "suffix_oracle": "PASS"}


def guarded_check():
    audit = new_audit()
    sys.dont_write_bytecode = True
    sys.addaudithook(ReadOnly(audit))
    try:
        result = check(audit)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, AssertionError) as error:
        result = {"status": "UNKNOWN", "error": type(error).__name__ + ": " + str(error)}
    result.update({"audit": audit, "boundary": BOUNDARY, "non_admission": True,
                   "normal_independent_review": "REQUIRED"})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    result = guarded_check()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] != "UNKNOWN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
