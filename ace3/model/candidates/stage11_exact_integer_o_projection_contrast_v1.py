"""Reference-independent exact Stage11 G128 reduction; CPU non-admission only.

The latest reviewed 92e suffix is the baseline, not a license to replay it.
Only Stage11 onward executes, using its retained Stage10 and original inputs.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

import numpy as np

from ace3.model.candidates import stage16_silu_product_contrast_v1 as silu


down, gate_up = silu.parent, silu.parent.parent
norm = gate_up.parent
base, retained, prior = silu.base, silu.retained, silu.prior
native, layer, require, pin = silu.native, silu.layer, silu.require, silu.pin
ROOT, PREFIX, CONTROLS, PAIR = silu.ROOT, silu.PREFIX, silu.CONTROLS, silu.PAIR
SOURCE = Path(__file__).resolve()
TEST = ROOT / f"tests/test_{SOURCE.stem}.py"
SOURCE_PATHS = frozenset(silu.project_sources() | {str(SOURCE), str(TEST)})
BASELINE_MISSION = "92e4223ef310"
RESULT_PIN = {
    "path": str(ROOT / "build/stage16_silu_product_execution_v1-92e4223ef310-attempt001/result.json"),
    "bytes": 35791001,
    "sha256": "e17c787ca2d15de6bdd674822966eab5a3bd916dda36c64bf345521b3d5cbebf",
}
REVIEW_PIN = {
    "path": str(retained.HANDOFFS / BASELINE_MISSION / "round-0004.json"),
    "bytes": 741,
    "sha256": "9ec233d5e215387ccb5df80719d4c3061c050471e58cf7c63b3a90d1df7ef2b7",
}
PROTECTED = tuple(f"stage{s:02d}" for s in range(11)) + (
    "input_hidden", "input_i", "input_z", "input_cache_k", "input_cache_v",
    "output_cache_k", "output_cache_v", "retained_stage07",
)
BOUNDARY = (
    "CPU-only width896/L23/P0/pair[34319,13], nine controls, non-admission. "
    "Exact integer G128 Stage11 o_proj with one Q47.48-to-FP16 conversion; "
    "native GEMM nibble order, no qzero plus-one, FP16 scales/operators/KV. "
    "S12 wider Q24 add, accepted Q24-state S13 RMSNorm, exact S14/S15, "
    "accurate integer S16 SiLU/product with native RTZ, exact S17, wider Q24 "
    "S18, final integer RMSNorm and tied pair. References are comparators only. "
    "No S0-S10/prefix/admission/reference/closed-producer replay, output patching, "
    "precision/scale expansion, GPU/RTL/FPGA/hardware or full-model admission. "
    "Retained value-substitution ancestry is not native-V repair."
)


def project_sources():
    return set(SOURCE_PATHS)


def validate_projection(tensors, words):
    layer.prior.local.finite_words(words, (896,))
    name = PREFIX + "self_attn.o_proj"
    for suffix, shape, dtype in (
            ("qweight", (896, 112), "<i4"), ("qzeros", (7, 112), "<i4"),
            ("scales", (7, 896), "<f2")):
        value = tensors.get(name + "." + suffix)
        require(isinstance(value, np.ndarray) and value.shape == shape
                and value.dtype.str == dtype, "canonical o_proj tensor " + suffix)
        if suffix == "scales":
            require(np.all(np.isfinite(value)), "nonfinite o_proj scales")
    require(name + ".bias" not in tensors, "canonical o_proj has no bias")


def projection(tensors, words):
    validate_projection(tensors, words)
    # As in accepted gate/up/down, preserve the sign of negative underflow.
    return native.state.project(tensors, PREFIX + "self_attn.o_proj", words, single_bias=True)


def inputs():
    values = silu.inputs()
    closure, bridge, reference, refs, operands, tensors, trajectory, records, prepared = values
    document = json.loads(gate_up.historical_record(RESULT_PIN))
    review = json.loads(gate_up.historical_record(REVIEW_PIN))
    require(review["mission_id"] == BASELINE_MISSION and review["producer_role"] == "reviewer"
            and review["review"]["status"] == "done", "independent 92e review")
    require(document["mission_id"] == BASELINE_MISSION
            and document["status"] == gate_up.scientific_status(document["controls"]) == "REJECTED"
            and type(document["science_invocations"]) is int
            and type(document["science_budget"]) is int
            and document["science_invocations"] == document["science_budget"] == 1
            and document["non_admission"] is True
            and document["audit"]["forbidden_events"] == []
            and document["protected_input_files_unchanged"] is True
            and document["thresholds"] == closure["thresholds"], "92e identity/budget/thresholds")
    require(document["operand_pins"] == [retained.digest(v) for v in operands]
            and document["tensor_pins"] == {k: retained.digest(v) for k, v in tensors.items()},
            "92e tensor/operand lineage")
    require([row["control"] for row in document["controls"]] == list(CONTROLS),
            "92e ordered controls")
    require(document["retained_stage17"] == pin(Path(silu.RESULT_PIN["path"])),
            "92e Stage17 parent binding")
    current = []
    for control, row, (original, raw) in zip(CONTROLS, document["controls"], prepared, strict=True):
        require(row["payload"]["path"] == str(Path(RESULT_PIN["path"]).parent / (control + ".npz")),
                "92e payload splice")
        require(row["protected_arrays"] == {k: retained.digest(v) for k, v in raw.items()},
                "92e protected input lineage")
        actual = down.archive(row["payload"])
        down.validate_state(actual)
        validate_projection(tensors, actual["stage10"])
        for key in silu.PROTECTED:
            require(retained.digest(actual[key]) == retained.digest(raw[key]),
                    "92e protected source/state/KV: " + key)
        current.append((original, actual))
    records.extend([RESULT_PIN, REVIEW_PIN, *document["source_pins"], *document["input_pins"],
                    *(row["payload"] for row in document["controls"])])
    for record in records:
        gate_up.historical_record(record)
    return closure, bridge, reference, refs, operands, tensors, trajectory, records, current


def compute_suffix(raw, tensors, operands):
    down.validate_state(raw)
    arrays, timings = dict(raw), {}
    began = time.monotonic()
    arrays["stage11"] = projection(tensors, raw["stage10"])
    timings["stage11"] = time.monotonic() - began
    began = time.monotonic()
    scratch = native.state.add(
        {"i": raw["input_i"], "z": raw["input_z"], "h": raw["input_hidden"]}, arrays["stage11"])
    arrays["scratch_i"], arrays["scratch_z"], arrays["stage12"] = scratch["i"], scratch["z"], scratch["h"]
    timings["stage12"] = time.monotonic() - began
    began = time.monotonic()
    arrays["stage13"], detail = norm.stage13(
        scratch["i"], scratch["z"], tensors[PREFIX + "post_attention_layernorm.weight"])
    timings["stage13"] = time.monotonic() - began
    for stage, name in ((14, "gate_proj"), (15, "up_proj")):
        began = time.monotonic()
        arrays[f"stage{stage}"] = gate_up.projection(
            tensors, PREFIX + "mlp." + name, arrays["stage13"])
        timings[f"stage{stage}"] = time.monotonic() - began
    began = time.monotonic()
    arrays["stage16"], arrays["s16_unrounded_binary64"] = silu.stage16(
        arrays["stage14"], arrays["stage15"])
    timings["stage16"] = time.monotonic() - began
    began = time.monotonic()
    arrays["stage17"] = down.projection(tensors, arrays["stage16"])
    timings["stage17"] = time.monotonic() - began
    began = time.monotonic()
    state = native.state.add(scratch, arrays["stage17"])
    arrays["output_i"], arrays["output_z"], arrays["stage18"] = state["i"], state["z"], state["h"]
    timings["stage18"] = time.monotonic() - began
    began = time.monotonic()
    arrays["final_rmsnorm"], final_detail = retained.suffix.rmsnorm(arrays["stage18"], operands[0])
    timings["final_rmsnorm"] = time.monotonic() - began
    began = time.monotonic()
    arrays["final_logits"] = retained.suffix.logits(arrays["final_rmsnorm"], operands[1])
    timings["pair_head"] = time.monotonic() - began
    down.validate_state(arrays)
    return arrays, detail, final_detail, timings


@contextmanager
def guarded(audit):
    require(prior._active is None and sys.getprofile() is None, "nested execution guard")
    if not prior._hook_installed:
        sys.addaudithook(prior.audit_event)
        prior._hook_installed = True
    forbidden = {
        "check", "main", "execute", "run", "diagnose", "stages", "_stages",
        "continuation_stages", "native_layer", "execute_layer", "execute_layers",
        "original_branches", "reference_arrays", "build_report", "intervene",
        "adjust", "nearest", "finite_table", "run_control", "guarded", "suffix_stages",
        "compute_suffix", "evaluate_control", "evaluate",
        "evaluate_actual_rtl_result", "evaluate_p0_transaction",
    }
    native_source = str(Path(native.__file__).resolve())
    state_source = str(Path(native.state.__file__).resolve())

    def profile(frame, event, arg):
        if event != "call":
            return
        file, name = frame.f_code.co_filename, frame.f_code.co_name
        if not file.startswith(str(ROOT) + "/"):
            return
        key = file.removeprefix(str(ROOT) + "/") + ":" + name
        audit["project_calls"][key] = audit["project_calls"].get(key, 0) + 1
        denied = file not in SOURCE_PATHS or (file != str(SOURCE) and name in forbidden)
        if name == "stage13":
            denied |= file != str(norm.SOURCE)
        if file == native_source:
            denied |= name not in {"decoded", "rne", "toward_zero", "<genexpr>"}
        if file == state_source and name == "project":
            denied |= frame.f_locals["prefix"] not in {
                PREFIX + p for p in ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
        if name in {"expected_stage", "stage_report", "local_reference", "check_stage_state",
                    "evaluate_decoder_stage"}:
            denied |= frame.f_locals.get("stage") not in range(10, 19)
        if denied:
            audit["forbidden_events"].append(key)
            raise retained.bridge.BindingError("unapproved project dispatch: " + key)

    prior._active = (None, audit)
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(None)
        prior._active = None


def evaluate_control(control, raw, original, classified, tensors, trajectory,
                     reference, refs, operands, oracle, audit):
    require(control == classified["control"], "retained control lineage splice")
    protected = {k: retained.digest(v) for k, v in raw.items()}
    arrays, detail, final_detail, timings = compute_suffix(raw, tensors, operands)
    audit["controls"].append(control)
    audit["stage_dispatches"].extend([[control, 23, 0, s] for s in range(11, 19)])
    audit["final_rmsnorm_invocations"] += 1
    audit["pair_head_invocations"] += 1
    began = time.monotonic()
    oracle.verify_projection(tensors, PREFIX + "self_attn.o_proj", raw["stage10"], arrays["stage11"])
    oracle.verify_add(raw["input_i"], raw["input_z"], arrays["stage11"],
                      arrays["scratch_i"], arrays["scratch_z"], arrays["stage12"])
    oracle.verify_norm(arrays["scratch_i"], arrays["scratch_z"],
                       tensors[PREFIX + "post_attention_layernorm.weight"], arrays["stage13"], detail)
    for stage, name in ((14, "gate_proj"), (15, "up_proj"), (17, "down_proj")):
        oracle.verify_projection(tensors, PREFIX + "mlp." + name,
                                 arrays["stage16" if stage == 17 else "stage13"], arrays[f"stage{stage}"])
    oracle.verify_stage16(arrays["stage14"], arrays["stage15"], arrays["stage16"],
                          arrays["s16_unrounded_binary64"])
    oracle.verify_add(arrays["scratch_i"], arrays["scratch_z"], arrays["stage17"],
                      arrays["output_i"], arrays["output_z"], arrays["stage18"])
    oracle.suffix_oracle.verify_suffix(
        arrays["stage18"], operands, arrays["final_rmsnorm"], final_detail, arrays["final_logits"])
    reports = []
    state = {"h": raw["input_hidden"], "i": raw["input_i"], "z": raw["input_z"]}
    for stage in range(10, 19):
        expected = layer.expected_stage(stage, arrays, state, tensors)
        reports.append(layer.stage_report(stage, arrays, trajectory, reference, expected))
        audit["stage_gate_checks"].append([control, 23, 0, stage])
    after = prior.compare(arrays["stage18"], reference)
    require(after["status"] == reports[-1]["status"], "S18 comparator disagreement")
    margins = retained.paired_margins(
        arrays["final_logits"], original["final_logits"], classified["paired_margins"], refs)
    require(protected == {k: retained.digest(v) for k, v in raw.items()}, "retained bytes changed")
    for key in PROTECTED:
        require(retained.digest(arrays[key]) == protected[key], "protected array: " + key)
    timings["oracles_and_gates"] = time.monotonic() - began
    return arrays, {
        "control": control, "stages": reports, "S18": after, "paired_margins": margins,
        "component_oracles": {"status": "PASS", "errors": []},
        "protected_arrays": protected, "protected_bytes_unchanged": True,
        "changed_words": {str(s): int(np.count_nonzero(arrays[f"stage{s}"] != raw[f"stage{s}"]))
                          for s in range(11, 19)},
        "baseline_pair_words": raw["final_logits"].tolist(),
        "candidate_pair_words": arrays["final_logits"].tolist(),
        "stage13_integer_details": detail, "final_rmsnorm_integer_details": final_detail,
        "timing_seconds": timings,
    }


def dispatch_census(audit):
    def count(module, name):
        key = str(Path(module.__file__).resolve().relative_to(ROOT)) + ":" + name
        return audit["project_calls"].get(key, 0)
    require(audit["forbidden_events"] == audit["writes"] == []
            and audit["controls"] == list(CONTROLS)
            and audit["stage_dispatches"] == [[c, 23, 0, s] for c in CONTROLS for s in range(11, 19)]
            and audit["stage_gate_checks"] == [[c, 23, 0, s] for c in CONTROLS for s in range(10, 19)]
            and audit["final_rmsnorm_invocations"] == audit["pair_head_invocations"] == 9,
            "ordered dispatch/gate/write census")
    require(count(norm, "stage13") == count(silu, "stage16") == 9
            and count(native.state, "project") == 36 and count(native.state, "add") == 18
            and count(native, "toward_zero") == 9 and count(native, "projection") == 0
            and count(retained.fp16_adaptation_oracle, "rmsnorm") == 9,
            "observed arithmetic dispatch census")


def scientific_status(rows):
    if not any(row["changed_words"]["11"] for row in rows):
        return "REJECTED"
    return gate_up.scientific_status(rows)


def evaluate(oracle, *, audit=None):
    """In-memory entry for a separately authorized one-use science owner."""
    audit = base.new_audit() if audit is None else audit
    require(audit == base.new_audit(), "fresh evaluation audit required")
    audit["science_invocations"] = 0
    rows, payloads = [], {}
    sources = [pin(Path(p)) for p in sorted(SOURCE_PATHS)]
    with guarded(audit):
        closure, bridge, reference, refs, operands, tensors, trajectory, records, prepared = inputs()
        operand_pins = [retained.digest(v) for v in operands]
        tensor_pins = {k: retained.digest(v) for k, v in tensors.items()}
        reference_pins = [retained.digest(reference), *[retained.digest(refs[b])
                                                     for b in ("fp16", "binary64")]]
        trajectory_pins = {k: retained.digest(v) for k, v in trajectory.items()}
        audit["science_invocations"] = 1
        for control, (original, raw), classified in zip(
                CONTROLS, prepared, bridge["controls"], strict=True):
            payloads[control], row = evaluate_control(
                control, raw, original, classified, tensors, trajectory, reference,
                refs, operands, oracle, audit)
            rows.append(row)
        for record in records:
            gate_up.historical_record(record)
        for record in sources:
            retained.selection.bound(record)
        require(operand_pins == [retained.digest(v) for v in operands]
                and tensor_pins == {k: retained.digest(v) for k, v in tensors.items()}
                and reference_pins == [retained.digest(reference), *[retained.digest(refs[b])
                                          for b in ("fp16", "binary64")]]
                and trajectory_pins == {k: retained.digest(v) for k, v in trajectory.items()},
                "authenticated operand/reference mutation")
        dispatch_census(audit)
        status = scientific_status(rows)
    return {
        "schema_version": 1, "status": status, "controls": rows, "audit": audit,
        "non_admission": True, "normal_independent_review": "REQUIRED",
        "automatic_replay_authorized": False, "boundary": BOUNDARY,
        "source_pins": sources, "input_pins": records, "operand_pins": operand_pins,
        "tensor_pins": tensor_pins, "reference_pins": reference_pins,
        "thresholds": closure["thresholds"], "historical_failure": closure["historical_failure"],
        "baseline": RESULT_PIN, "protected_input_files_unchanged": True,
        "flags": base.closure_source.FLAGS,
    }, payloads
