"""Reference-independent native Stage17 contrast, CPU-only and non-admission.

This is a library, not a science owner or claim publisher. A separately issued
Manager-owned runner must supply execution authority before using retained data.
Repeatable software tests use isolated synthetic inputs through the same entry.
Only S17/S18, final integer RMSNorm and the two-row head execute; S10-S16
are gated without replay. Native S16 RTZ, FP16 scales/operators/KV, wider Q24
residual state, original references and exact thresholds remain unchanged.
"""

from contextlib import contextmanager
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

from ace3.model.candidates import stage14_stage15_q24_gate_up_projection_contrast_v1 as parent


base, retained, prior = parent.base, parent.retained, parent.prior
native, layer, require, pin = parent.native, parent.layer, parent.require, parent.pin
ROOT, PREFIX, CONTROLS, PAIR = parent.ROOT, parent.PREFIX, parent.CONTROLS, parent.PAIR
SOURCE = Path(__file__).resolve()
TEST = ROOT / f"tests/test_{SOURCE.stem}.py"
SOURCE_PATHS = frozenset(
    (parent.project_sources() | {str(SOURCE), str(TEST)}) - {
        str(SOURCE.with_name("stage17_q24_down_projection_execution_v1.py")),
        str(ROOT / "tests/test_stage17_q24_down_projection_execution_v1.py"),
    })
RESULT_PINS = (
    {"path": str(parent.OUTPUT / "result.json"),
     "sha256": "0f5bade5a1bf007fe19830ca9dae4b73cbeb05e2ac946d7d0fd7fb52a50c69a4"},
    {"path": str(parent.RETAINED_RESULT), "sha256": parent.RETAINED_HASH},
)
REVIEW_PINS = (
    {"path": str(retained.HANDOFFS / parent.MISSION / "round-0003.json"),
     "sha256": "93e56d7e1380ad09e498701a31503bce8d6aa4ac2a029edbaa0204a054481072"},
    {"path": str(retained.HANDOFFS / parent.parent.MISSION / "round-0003.json"),
     "sha256": "9a65c2b1ffb692d045bc812c5e3a097762a3f0dcc0b44ab577a57b9bce201466"},
)
PROTECTED = tuple(f"stage{s:02d}" for s in range(17)) + (
    "input_hidden", "input_i", "input_z", "scratch_i", "scratch_z",
    "input_cache_k", "input_cache_v", "output_cache_k", "output_cache_v",
    "retained_stage07", "s16_unrounded_binary64",
)
BOUNDARY = (
    "CPU-only width896/L23/P0/pair[34319,13], nine controls, non-admission. "
    "Exact integer native G128 INT4 down projection from retained Stage16 FP16 "
    "words, GEMM nibble order, no qzero plus-one, FP16 scales and boundaries/KV, "
    "unchanged S16 RTZ and wider Q24 residuals. Original references are read-only "
    "comparators. No prefix/admission/Stage13/closed-producer replay, output "
    "patching, strict-FP16-state W4A16, new-token/full-model or hardware claim. "
    "Retained value-substitution ancestry is not native-V repair."
)


def project_sources():
    return set(SOURCE_PATHS)


def projection(tensors, words):
    name = PREFIX + "mlp.down_proj"
    require(isinstance(words, np.ndarray) and words.shape == (4864,)
            and words.dtype.str == "<u2"
            and not np.any((words & 0x7c00) == 0x7c00), "stage16 finite FP16 words")
    for suffix, shape, dtype in (
            ("qweight", (4864, 112), "<i4"), ("qzeros", (38, 112), "<i4"),
            ("scales", (38, 896), "<f2")):
        value = tensors.get(name + "." + suffix)
        require(isinstance(value, np.ndarray) and value.shape == shape
                and value.dtype.str == dtype, "canonical down tensor " + suffix)
        if suffix == "scales":
            require(np.all(np.isfinite(value)), "nonfinite down scales")
    require(name + ".bias" not in tensors, "canonical down has no bias")
    return native.state.project(tensors, name, words, single_bias=True)


def validate_state(raw):
    prior.check_state(raw)
    layer.prior.local.finite_words(raw["stage16"], (4864,))
    value = raw["s16_unrounded_binary64"]
    require(value.shape == (4864,) and value.dtype.str == "<f8"
            and np.all(np.isfinite(value)), "retained S16 unrounded schema")
    require(np.array_equal(layer.prior.rtz_reference(value), raw["stage16"]),
            "retained S16 RTZ mismatch")
    for kind in ("k", "v"):
        require(raw["input_cache_" + kind].dtype.str == "<u2", "input FP16 KV dtype")


def compute_suffix(raw, tensors, operands):
    validate_state(raw)
    arrays, timings = dict(raw), {}
    began = time.monotonic()
    arrays["stage17"] = projection(tensors, raw["stage16"])
    timings["stage17"] = time.monotonic() - began
    began = time.monotonic()
    state = native.state.add(
        {"i": raw["scratch_i"], "z": raw["scratch_z"], "h": raw["stage12"]},
        arrays["stage17"])
    arrays["output_i"], arrays["output_z"], arrays["stage18"] = state["i"], state["z"], state["h"]
    timings["stage18"] = time.monotonic() - began
    began = time.monotonic()
    arrays["final_rmsnorm"], detail = retained.suffix.rmsnorm(arrays["stage18"], operands[0])
    timings["final_rmsnorm"] = time.monotonic() - began
    began = time.monotonic()
    arrays["final_logits"] = retained.suffix.logits(arrays["final_rmsnorm"], operands[1])
    timings["pair_head"] = time.monotonic() - began
    prior.check_state(arrays)
    return arrays, detail, timings


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
        "compute_suffix", "stage13", "evaluate_control",
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
        if file == native_source:
            denied |= name not in {"decoded", "rne", "<genexpr>"}
        if file == state_source and name == "project":
            denied |= frame.f_locals["prefix"] != PREFIX + "mlp.down_proj"
        if name in {"expected_stage", "stage_report", "local_reference", "check_stage_state",
                    "evaluate_decoder_stage"}:
            denied |= frame.f_locals.get("stage") not in range(10, 19)
        if denied:
            audit["forbidden_events"].append(key)
            raise retained.bridge.BindingError("unapproved project dispatch: " + key)

    # No output directory: the inherited audit hook rejects every write.
    prior._active = (None, audit)
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(None)
        prior._active = None


def archive(record):
    with np.load(io.BytesIO(parent.historical_record(record)), allow_pickle=False) as data:
        require(len(data.files) == len(set(data.files)), "duplicate archive keys")
        result = {key: data[key] for key in data.files}
    for value in result.values():
        value.flags.writeable = False
    return result


def inputs():
    closure, _, bridge, reference, refs, operands, tensors, trajectory, records = base.inputs()
    documents = [json.loads(parent.historical_record(p)) for p in RESULT_PINS]
    for document, mission, review_pin, result_pin in zip(
            documents, (parent.MISSION, parent.parent.MISSION),
            REVIEW_PINS, RESULT_PINS, strict=True):
        review = json.loads(parent.historical_record(review_pin))
        require(review["mission_id"] == mission and review["producer_role"] == "reviewer"
                and review["review"]["status"] == "done", "independent retained review")
        require(document["mission_id"] == mission and document["status"] == "REJECTED"
                and type(document["science_invocations"]) is int
                and document["science_invocations"] == document["science_budget"] == 1
                and document["non_admission"] is True
                and document["audit"]["forbidden_events"] == []
                and document["protected_input_files_unchanged"] is True
                and document["thresholds"] == closure["thresholds"],
                "retained execution identity/thresholds")
        require(parent.scientific_status(document["controls"]) == "REJECTED",
                "retained classification")
        require(document["operand_pins"] == [retained.digest(v) for v in operands]
                and document["tensor_pins"] == {k: retained.digest(v) for k, v in tensors.items()},
                "retained tensor/operand lineage")
        records.extend([result_pin, review_pin, *document["source_pins"], *document["input_pins"],
                        *(row["payload"] for row in document["controls"])])
    recent, stage13 = documents
    require(recent["retained_stage13"] == pin(Path(RESULT_PINS[1]["path"])),
            "Stage14/15 to Stage13 lineage")
    require([r["control"] for r in bridge["controls"]] == list(CONTROLS),
            "original classifier control census")
    prepared = []
    for row, old in zip(recent["controls"], stage13["controls"], strict=True):
        for item, root in ((row, RESULT_PINS[0]), (old, RESULT_PINS[1])):
            require(item["payload"]["path"] == str(
                Path(root["path"]).parent / (item["control"] + ".npz")), "payload control splice")
        raw = archive(row["payload"])
        original, previous = parent.retained_input(stage13, old)
        require(row["protected_arrays"] == {k: retained.digest(v) for k, v in previous.items()},
                "Stage13 protected array lineage")
        for key in tuple(f"stage{s:02d}" for s in range(14)) + PROTECTED[17:-1]:
            require(retained.digest(raw[key]) == retained.digest(previous[key]),
                    "Stage14/15 protected prefix/state/KV: " + key)
        validate_state(raw)
        prepared.append((original, raw))
    for record in records:
        parent.historical_record(record)
    return closure, bridge, reference, refs, operands, tensors, trajectory, records, prepared


def evaluate_control(control, raw, original, classified, tensors, trajectory,
                     reference, refs, operands, oracle, audit):
    require(control == classified["control"], "retained control lineage splice")
    protected = {k: retained.digest(v) for k, v in raw.items()}
    arrays, detail, timings = compute_suffix(raw, tensors, operands)
    audit["controls"].append(control)
    audit["stage_dispatches"].extend([[control, 23, 0, s] for s in (17, 18)])
    audit["final_rmsnorm_invocations"] += 1
    audit["pair_head_invocations"] += 1
    began = time.monotonic()
    oracle.verify_projection(tensors, raw["stage16"], arrays["stage17"])
    oracle.suffix_oracle.verify_suffix(
        arrays["stage18"], operands, arrays["final_rmsnorm"], detail, arrays["final_logits"])
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
        "changed_words": {"17": int(np.count_nonzero(arrays["stage17"] != raw["stage17"]))},
        "final_rmsnorm_integer_details": detail, "timing_seconds": timings,
    }


def dispatch_census(audit):
    def count(module, name):
        key = str(Path(module.__file__).resolve().relative_to(ROOT)) + ":" + name
        return audit["project_calls"].get(key, 0)
    require(audit["forbidden_events"] == audit["writes"] == []
            and audit["controls"] == list(CONTROLS)
            and audit["stage_dispatches"] == [[c, 23, 0, s] for c in CONTROLS for s in (17, 18)]
            and audit["stage_gate_checks"] == [[c, 23, 0, s] for c in CONTROLS for s in range(10, 19)]
            and audit["final_rmsnorm_invocations"] == audit["pair_head_invocations"] == 9,
            "ordered dispatch/gate/write census")
    require(count(native.state, "project") == count(native.state, "add") == 9
            and count(native, "projection") == count(native, "toward_zero") == 0
            and count(retained.fp16_adaptation_oracle, "rmsnorm") == 9,
            "observed arithmetic dispatch census")


def evaluate(oracle, *, audit=None):
    """In-memory suffix entry; does not issue authority, claim or write evidence."""
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
        audit["science_invocations"] = 1
        for control, (original, raw), classified in zip(
                CONTROLS, prepared, bridge["controls"], strict=True):
            payloads[control], row = evaluate_control(
                control, raw, original, classified, tensors, trajectory, reference,
                refs, operands, oracle, audit)
            rows.append(row)
        for record in records:
            parent.historical_record(record)
        for record in sources:
            retained.selection.bound(record)
        require(operand_pins == [retained.digest(v) for v in operands]
                and tensor_pins == {k: retained.digest(v) for k, v in tensors.items()}
                and reference_pins == [retained.digest(reference), *[retained.digest(refs[b])
                                          for b in ("fp16", "binary64")]],
                "authenticated operand/reference mutation")
        dispatch_census(audit)
        status = parent.scientific_status(rows)
    return {
        "schema_version": 1, "status": status, "controls": rows, "audit": audit,
        "non_admission": True, "normal_independent_review": "REQUIRED",
        "automatic_replay_authorized": False, "boundary": BOUNDARY,
        "source_pins": sources, "input_pins": records, "operand_pins": operand_pins,
        "tensor_pins": tensor_pins, "thresholds": closure["thresholds"],
        "historical_failure": closure["historical_failure"],
        "protected_input_files_unchanged": True, "flags": base.closure_source.FLAGS,
    }, payloads
