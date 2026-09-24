"""One non-admitting L23/P0 value substitution; only S10-S18 and final head run."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_impact_census_v1 as retained
from ace3.model.candidates import q24_s16_final_head_from_l23_coordinate62_suffix_execution_v1 as final
from ace3.model.candidates import q24_s16_l23_from_l22_coordinate62_suffix_execution_v1 as layer


ROOT = final.ROOT
NAME = "q24_s16_l23_p0_value_content_sufficiency_v1"
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
CONTROLS = final.CONTROLS
native = layer.producer.native.candidate
require, same, record, read_bound = final.require, final.same, final.record, final.read_bound
PROTECTED = (
    "input_hidden", "input_i", "input_z", "input_cache_k", "input_cache_v",
    "output_cache_k", "output_cache_v",
) + tuple(f"stage{s:02d}" for s in range(10) if s != 7)
PAIRS = ((319, 34319), (34319, 319), (34319, 13))
FLAGS = {
    "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
    "strict_FP16_state_claim": False, "new_token_claim": False, "full_model_claim": False,
    "historical_failures_preserved": True, "original_global_reference_unchanged": True,
    "reference_recomputation": False, "stage07_native_operator_validated": False,
}
BOUNDARY = (
    "Conditional rescue of nine fresh substituted S10-S18 outputs only; final-head "
    "comparisons retain their existing diagnostic-only scope and have no admission "
    "threshold. Historical L0-L22 and unmodified L23 FAIL remain unchanged. Q24 "
    "residual state is wider than FP16; native INT4/FP16 operators and retained FP16 "
    "KV remain fixed. Stage07 is an explicit non-cache-writing intervention, not "
    "validation of an unchanged native V operator. No unique-root-cause or "
    "failure-creation sufficiency claim. No prefix/admission/reference replay, "
    "hardware, GPU, RTL, token publication or full-model admission."
)


def archive(pin):
    with np.load(io.BytesIO(read_bound(pin)), allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)), "duplicate archive fields")
        return {key: saved[key] for key in saved.files}


def prepare(actual, reference):
    for arrays in (actual, reference):
        for stage, size in ((3, 128), (6, 128), (7, 128), (9, 14)):
            layer.prior.local.finite_words(arrays[f"stage{stage:02d}"], (size,))
        require(np.array_equal(arrays["stage03"], arrays["stage07"]),
                "retained/reference V source lineage changed")
        require(np.array_equal(arrays["stage09"], np.full(14, 0x3c00, dtype="<u2")),
                "P0 probabilities are not all exactly one")
    layer.preflight.parent.check_state(actual)
    for kind, stage in (("k", 6), ("v", 7)):
        require(actual["input_cache_" + kind].shape == (0, 128)
                and actual["output_cache_" + kind].shape == (1, 128)
                and np.array_equal(actual["output_cache_" + kind][0],
                                   actual[f"stage{stage:02d}"]),
                "selected P0 KV/source identity changed")
    parent = {key: actual[field].copy() for key, field in
              (("h", "input_hidden"), ("i", "input_i"), ("z", "input_z"))}
    arrays = {key: actual[key].copy() for key in PROTECTED}
    arrays["stage07"] = reference["stage07"].copy()
    for value in arrays.values():
        value.flags.writeable = False
    return parent, arrays


def protected(actual, arrays, reference):
    for key in PROTECTED:
        require(actual[key].dtype == arrays[key].dtype
                and np.array_equal(actual[key], arrays[key]), "protected field changed: " + key)
    require(np.array_equal(arrays["stage07"], reference["stage07"]),
            "substituted value changed")


def suffix_stages(tensors, parent, arrays):
    """Keep the pinned native producer untouched; reuse its arithmetic primitives."""
    prefix = "model.layers.23."
    scratch = None
    for stage in range(10, 19):
        if stage == 10:
            value = native.decoded(arrays["stage07"]).reshape(2, 64).repeat_interleave(7, dim=0)
            word = native.rne((native.decoded(arrays["stage09"])[:, None] * value).reshape(-1))
        elif stage in (11, 14, 15, 17):
            name, source = {
                11: ("self_attn.o_proj", 10), 14: ("mlp.gate_proj", 13),
                15: ("mlp.up_proj", 13), 17: ("mlp.down_proj", 16),
            }[stage]
            word = native.projection(tensors, prefix + name, arrays[f"stage{source:02d}"])
        elif stage == 12:
            scratch = native.state.add(parent, arrays["stage11"])
            arrays["scratch_i"], arrays["scratch_z"] = scratch["i"], scratch["z"]
            word = scratch["h"]
        elif stage == 13:
            value = native.decoded(arrays["stage12"])
            gamma = native.torch.from_numpy(
                tensors[prefix + "post_attention_layernorm.weight"].astype("<f8"))
            word = native.rne(
                (value * native.torch.rsqrt(value.square().mean() + 1e-6)) * gamma)
        elif stage == 16:
            value = (native.functional.silu(native.decoded(arrays["stage14"]))
                     * native.decoded(arrays["stage15"]))
            arrays["s16_unrounded_binary64"] = value.numpy().copy()
            word = native.toward_zero(arrays["s16_unrounded_binary64"])
        else:
            require(scratch is not None, "missing intervened residual scratch")
            successor = native.state.add(scratch, arrays["stage17"])
            arrays["output_i"], arrays["output_z"] = successor["i"], successor["z"]
            word = successor["h"]
        arrays[f"stage{stage:02d}"] = word
        yield stage


def margin_rows(baseline, substituted, references):
    actual = baseline.view("<f2").astype("<f8")
    changed = substituted.view("<f2").astype("<f8")
    rows = []
    for branch in ("fp16", "binary64"):
        reference = references["logits_" + branch]
        if branch == "fp16":
            reference = reference.view("<f2").astype("<f8")
        exact = lambda x: Fraction.from_float(float(x))
        for left, right in PAIRS:
            a = exact(actual[left]) - exact(actual[right])
            s = exact(changed[left]) - exact(changed[right])
            r = exact(reference[left]) - exact(reference[right])
            rows.append({
                "pair": [left, right], "reference_branch": branch,
                "retained_actual_margin": str(a), "retained_reference_margin": str(r),
                "retained_margin_change": str(a - r), "substituted_actual_margin": str(s),
                "substituted_margin_change": str(s - r), "paired_intervention_delta": str(s - a),
            })
    return rows


def scientific_status(rows):
    same([row["control"] for row in rows], list(CONTROLS), "incomplete/reordered controls")
    for row in rows:
        same([r["stage"] for r in row["reports"]], list(range(10, 19)),
             "incomplete affected cone")
        require(all(r["status"] in ("PASS", "FAIL") for r in row["reports"]),
                "invalid acceptance result")
    return "PASS" if all(r["status"] == "PASS" for row in rows for r in row["reports"]) else "FAIL"


@contextmanager
def write_scope(out, audit):
    import builtins

    def permitted(path, writing):
        if writing:
            require(not isinstance(path, int), "descriptor write not authorized")
            path = Path(path).absolute()
            require(path.parent == out and path.resolve() == path,
                    "write outside fresh output root")
            audit["file_write_opens"] += 1

    def wrap(original):
        def opening(path, mode="r", *args, **kwargs):
            permitted(path, isinstance(mode, str) and any(c in mode for c in "wax+"))
            return original(path, mode, *args, **kwargs)
        return opening

    original_os_open = os.open

    def os_open(path, flags, *args, **kwargs):
        permitted(path, flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        return original_os_open(path, flags, *args, **kwargs)

    with patch.object(builtins, "open", wrap(builtins.open)), \
            patch.object(io, "open", wrap(io.open)), patch.object(os, "open", os_open):
        yield


def predecessor_review():
    directory = layer.preflight.HANDOFFS / "9b2a69ef5a4e"
    latest = json.loads((directory / "latest.json").read_bytes())
    require(latest["kind"] == "handoff_ref", "predecessor is not reviewed")
    path = Path(latest["handoff"]["path"])
    require(path.parent == directory and path.resolve() == path, "review path escaped")
    review = json.loads(path.read_bytes())
    layer.preflight.parent.matrix.check_review(review, "9b2a69ef5a4e", review["round"])
    backlog = [json.loads(line) for line in layer.preflight.BACKLOG.read_text().splitlines()
               if line.strip()]
    rows = [row for row in backlog if row["id"] == "9b2a69ef5a4e"]
    require(len(rows) == 1 and rows[0]["status"] == "done"
            and rows[0]["outcome"]["review_status"] == "done"
            and rows[0]["finished_ts"] >= review["created_at"],
            "predecessor review lacks terminal corroboration")
    return record(path)


def focused_tests():
    import importlib.util

    spec = importlib.util.spec_from_file_location(NAME + "_tests", TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == 12 and not result.skipped,
            "focused tests failed:\n" + output.getvalue())
    return {"executed": result.testsRun, "failures": 0, "errors": 0,
            "skipped": 0, "synthetic_stage10_invocations": 1,
            "retained_control_dispatches": 0, "raw_output": output.getvalue()}


def execute():
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "occupied output; zero dispatch")
    require(OUTPUT.resolve() == OUTPUT and OUTPUT.parent == ROOT / "build"
            and Path(__file__).resolve() == SOURCE, "noncanonical source/output")
    ignored = final.subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--", str(OUTPUT)],
        capture_output=True, check=False)
    require(ignored.returncode == 0, "output not confirmed ignored")
    native.torch.set_num_threads(1)
    require(str(native.torch.tensor(0).device) == "cpu", "non-CPU default device")
    origins = final.source_context()
    origins["value_sufficiency_source"], origins["value_sufficiency_test"] = record(SOURCE), record(TEST)
    OUTPUT.mkdir(exist_ok=False)
    audit = {"controls": [], "stage_dispatches": [], "final_rmsnorm_invocations": 0,
             "lm_head_invocations": 0, "top_k_invocations": 0, "forbidden_calls": 0,
             "file_write_opens": 0, "native_L0_L22_invocations": 0,
             "native_S0_S9_invocations": 0, "retained_evidence_writes": 0,
             "reference_producer_invocations": 0, "external_invocations": 0}
    rows, artifacts = [], []

    def save_json(name, payload):
        with (OUTPUT / name).open("x") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        return record(OUTPUT / name)

    try:
        with write_scope(OUTPUT, audit):
            artifacts.append(save_json("command.json", {
                "argv": sys.argv, "cwd": str(ROOT), "uid": os.getuid(),
                "executable": sys.executable, "origins": origins,
                "host_managed_controls": "unchanged role/model/account/budget/access/concurrency locks",
            }))
            compiled = []
            for pin in {p["path"]: p for p in origins.values()}.values():
                compile(read_bound(pin), pin["path"], "exec")
                compiled.append(pin)
            tests = focused_tests()
            tests["compiled"] = compiled
            artifacts.append(save_json("compile_tests.json", tests))
            predecessor = predecessor_review()
            started = time.monotonic()
            baseline, baseline_arrays, references, pins = retained.authenticate()
            summary = baseline["preflight"]
            tensors, trajectory, reference, checkpoint = layer.load_inputs({"summary": summary})
            operands = final.load_operands(summary)
            # Rebind checkpoint/tokenizer/tied-tensor identities without running their producers.
            same(final.preflight.bind_assets(checkpoint), summary["assets"], "asset identity changed")
            require(summary["L23_original_reference"]["reference"]["prior_kv"] == "own empty P0",
                    "reference KV/source scope changed")
            same(summary["L23_original_reference"]["reference"]["fp16"],
                 summary["final_reference"]["reference"]["input_fp16"],
                 "reference input lineage splice")
            for tensor in tensors.values():
                tensor.flags.writeable = False
            prepared = []
            for row in baseline["controls"]:
                actual = archive(row["parent"]["terminal_archive"])
                parent, arrays = prepare(actual, trajectory)
                prepared.append((row, actual, parent, arrays))
            same([r["control"] for r, _, _, _ in prepared], list(CONTROLS),
                 "nine-control source order changed")
            authentication_seconds = time.monotonic() - started
            with final.no_dispatch(audit):
                for row, actual, parent, arrays in prepared:
                    label = row["control"]
                    audit["controls"].append(label)
                    reports, timings = [], {"native_seconds": 0.0, "oracle_seconds": 0.0}
                    stages = suffix_stages(tensors, parent, arrays)
                    for expected_stage in range(10, 19):
                        began = time.monotonic()
                        audit["stage_dispatches"].append([label, 23, 0, expected_stage])
                        stage = next(stages)
                        require(stage == expected_stage, "out-of-cone dispatch")
                        timings["native_seconds"] += time.monotonic() - began
                        began = time.monotonic()
                        expected = layer.expected_stage(stage, arrays, parent, tensors)
                        reports.append(layer.stage_report(stage, arrays, trajectory, reference, expected))
                        if expected is not None:
                            arrays[f"local_reference_stage{stage:02d}"] = expected
                        timings["oracle_seconds"] += time.monotonic() - began
                    require(next(stages, None) is None, "extra suffix stage")
                    protected(actual, arrays, trajectory)
                    require(np.array_equal(layer.prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                                           arrays["stage16"]), "S16 RTZ boundary changed")
                    began = time.monotonic()
                    audit["final_rmsnorm_invocations"] += 1
                    arrays["final_rmsnorm"], details = final.rmsnorm(arrays["stage18"], operands[0])
                    timings["final_rmsnorm_seconds"] = time.monotonic() - began
                    began = time.monotonic()
                    audit["lm_head_invocations"] += 1
                    arrays["final_logits"] = final.logits(arrays["final_rmsnorm"], operands[1])
                    timings["lm_head_seconds"] = time.monotonic() - began
                    metrics = final.comparisons(
                        {"rmsnorm": arrays["final_rmsnorm"], "logits": arrays["final_logits"]}, references)
                    audit["top_k_invocations"] += 3
                    margins = margin_rows(baseline_arrays[label]["logits"], arrays["final_logits"], references)
                    payload = OUTPUT / f"{label}.npz"
                    with payload.open("xb") as stream:
                        np.savez(stream, **arrays, retained_stage07=actual["stage07"])
                    pin = record(payload)
                    artifacts.append(pin)
                    entry = {
                        "control": label, "reports": reports, "final_comparisons": metrics,
                        "paired_final_margins": margins, "parent": row["parent"], "raw_payload": pin,
                        "substitution": {
                            "source": summary["L23_original_reference"]["reference"]["fp16"],
                            "field": "stage07", "shape": [128], "dtype": "<u2",
                            "changed_components": int(np.count_nonzero(actual["stage07"] != arrays["stage07"])),
                            "retained_KV_unchanged": True, "retained_state_unchanged": True,
                            "stage03_and_cache_not_substituted": True,
                        },
                        "rmsnorm_integer_details": details, "timing_seconds": timings,
                        "status": "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL",
                    }
                    rows.append(entry)
                    print(json.dumps({k: entry[k] for k in
                                      ("control", "status", "final_comparisons", "paired_final_margins",
                                       "substitution", "timing_seconds")}), flush=True)
            same(final.source_context(), {k: v for k, v in origins.items()
                                          if k not in ("value_sufficiency_source", "value_sufficiency_test")},
                 "source changed during execution")
            for pin in pins:
                read_bound(pin)
            for pin in (record(SOURCE), record(TEST)):
                same(pin, origins["value_sufficiency_source" if pin["path"] == str(SOURCE)
                                  else "value_sufficiency_test"], "new source changed")
            status = scientific_status(rows)
            require(len(audit["stage_dispatches"]) == 81
                    and audit["final_rmsnorm_invocations"] == audit["lm_head_invocations"] == 9
                    and audit["forbidden_calls"] == 0 and audit["file_write_opens"] == 11,
                    "dispatch/write census mismatch")
            result = {
                "diagnostic_id": NAME, "status": status, "controls": rows, "origins": origins,
                "predecessor_review": predecessor, "input_pins": pins, "preflight": summary,
                "thresholds": summary["thresholds"], "final_reference": summary["final_reference"],
                "audit": audit, "artifacts": artifacts, "output": str(OUTPUT),
                "expected_final_file_count": 12, "authentication_seconds": authentication_seconds,
                "flags": FLAGS, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
                "decision": ("conditional all-nine rescue; not unique cause"
                             if status == "PASS" else
                             "reject all-nine rescue; close attention analysis; next Planner "
                             "contrast must use non-attention retained evidence"),
            }
            save_json("result.json", result)
            require(len(list(OUTPUT.iterdir())) == 12 and audit["file_write_opens"] == 12,
                    "final write census mismatch")
            print(json.dumps({"status": status, "output": str(OUTPUT), "audit": audit,
                              "decision": result["decision"], "flags": FLAGS,
                              "claim_boundary": BOUNDARY}), flush=True)
    except Exception as error:
        with write_scope(OUTPUT, audit):
            save_json("failure.json", {"status": "UNKNOWN", "error_type": type(error).__name__,
                                      "error": str(error), "audit": audit, "controls": rows,
                                      "flags": FLAGS, "automatic_replay_authorized": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    execute()


if __name__ == "__main__":
    from importlib import import_module

    import_module("ace3.model.candidates." + NAME).main()
