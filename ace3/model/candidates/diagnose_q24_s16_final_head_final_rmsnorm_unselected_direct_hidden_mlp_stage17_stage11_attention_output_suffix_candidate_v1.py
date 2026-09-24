"""Dormant S11 replacement executor, not launch authority.

Import, construction and preparation remain non-generative. --check requires a
byte-exact launcher capture containing an externally pinned Manager execution
declaration bound to its live claim and a separately reviewed ACCEPTED_NOVEL
decision. Terminal review of the unfinished execution mission is not launch
authority; the completed evidence missions retain their own Reviewer bindings.
It never issues declarations, custody or identity preflights. Missing authority
fails before importing the native arithmetic or loading any model/retained array.
Only a separately claimed future task may enter that path. Synthetic tests inject
array-only arithmetic; they must not call --check or import the real runtime.

inputs requires the nine unchanged control archive pins plus replacement_stage11.
Its canonical member encoding is exactly 896 little-endian uint16 FP16 words in
C/input order (1792 bytes), without a header, container or numeric conversion.
The same encoding applies to a stage13 member; content, not member/path spelling,
distinguishes operands. Execution authenticates both the original-input L23 FP16
archive and replacement bytes, requires exact member equality, and consumes those
words without creating a member file or issuing any authority.

historical_failures must be an ordered list of exactly nine records, with control
names matching the frozen controls in order. Before suffix work the complete list
must equal the authenticated retained_controls_and_failure_gates value exactly;
structural acceptance alone does not authenticate that evidence.
"""

import argparse
import copy
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

import ace3.model.candidates.diagnostic_capture_v1 as capture
from ace3.model.candidates.diagnostic_capture_v1 import UnavailableBinding
from ace3.model import stage11_current_runtime_release as release


NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage11_attention_output_suffix_candidate_v1"
PAIR = (34319, 13)
WIDTH = 896
REPLACEMENT = "replacement_stage11"
MODULE = "ace3.model.candidates." + NAME
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
BRANCHES = ("fp16", "binary64")
SUFFIX_COUNTS = {"forbidden_calls": 0, "projections": 3, "local_oracles": 7,
                 "s12_adds": 1, "s18_adds": 1, "final_norms": 1, "pair_heads": 1}
COUNTERS = (
    "scientific_invocations", "model_invocations", "producer_invocations",
    "service_invocations", "prefix_invocations", "reference_invocations",
    "admission_invocations", "full_vocabulary_invocations",
    "closed_producer_invocations", "identity_preflight_invocations",
    "custody_emissions", "scientific_classifications",
)
GATES = ("source", "operand", "state", "kv", "lineage")
PROFILE = {
    "residual_state": "Q24-wide-not-FP16",
    "stage16_rounding": "native-S16-RTZ",
    "weights": "G128-asymmetric-packed-INT4",
    "nibble_order": "native-GEMM",
    "qzero_plus_one": False,
    "scales": "FP16",
    "operator_boundaries": "FP16",
    "kv": "FP16",
}
FORBIDDEN = (
    "prefix", "original_reference", "admission", "full_vocabulary",
    "closed_producer", "native_decoder", "service", "check", "execute",
    "classify", "preflight_identity", "custody", "token_selection",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _words(arrays, key, shape, dtype="<u2"):
    if key not in arrays:
        raise UnavailableBinding("missing retained array: " + key)
    value = arrays[key]
    _require(isinstance(value, np.ndarray) and value.shape == shape
             and value.dtype.str == dtype, "retained shape/dtype changed: " + key)
    if dtype == "<u2":
        _require(np.isfinite(value.view("<f2")).all(), "nonfinite FP16 words: " + key)
    return value


def canonical_fp16_member(arrays, member):
    """Encode one full-width FP16-word member without changing any word bits."""
    return _words(arrays, member, (WIDTH,)).tobytes(order="C")


def _snapshot(arrays):
    _require(all(isinstance(k, str) and isinstance(v, np.ndarray) and not v.dtype.hasobject
                 for k, v in arrays.items()), "retained arrays must be named numeric arrays")
    return {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}


def suffix_recipe():
    """The sole S12-S18/final/two-row cone; description itself never dispatches."""
    return [
        {"stage": 12, "operator": "q24_add", "inputs": ["input_i", "input_z", "input_hidden", "stage11"],
         "outputs": ["scratch_i", "scratch_z", "stage12"]},
        {"stage": 13, "operator": "post_attention_rmsnorm", "inputs": ["stage12"],
         "outputs": ["stage13"]},
        {"stage": 14, "operator": "mlp_gate_projection", "inputs": ["stage13"],
         "outputs": ["stage14"]},
        {"stage": 15, "operator": "mlp_up_projection", "inputs": ["stage13"],
         "outputs": ["stage15"]},
        {"stage": 16, "operator": "native_silu_product_rtz", "inputs": ["stage14", "stage15"],
         "outputs": ["s16_unrounded_binary64", "stage16"]},
        {"stage": 17, "operator": "mlp_down_projection", "inputs": ["stage16"],
         "outputs": ["stage17"]},
        {"stage": 18, "operator": "q24_add", "inputs": ["scratch_i", "scratch_z", "stage12", "stage17"],
         "outputs": ["output_i", "output_z", "stage18"]},
        {"stage": "final_rmsnorm", "operator": "final_rmsnorm", "inputs": ["stage18"],
         "outputs": ["final_rmsnorm"]},
        {"stage": "selected_tied_head", "operator": "selected_tied_head",
         "inputs": ["final_rmsnorm"], "rows": list(PAIR), "outputs": ["pair_logits"]},
    ]


def _contract(contract):
    required = ("inputs", "references", "sources", "thresholds", "controls",
                "profile", "gates", "source_identity", "lineage",
                "historical_failures", "launch_constraints")
    for field in required:
        if field not in contract:
            raise UnavailableBinding("missing retained contract binding: " + field)
    _require(isinstance(contract["historical_failures"], list),
             "retained contract binding must be a list: historical_failures")
    for field in ("inputs", "references", "sources", "thresholds", "lineage",
                  "launch_constraints"):
        _require(isinstance(contract[field], dict) and bool(contract[field]),
                 "empty retained contract binding: " + field)
    for field in ("inputs", "references", "sources"):
        for role, pin in contract[field].items():
            _require(isinstance(role, str) and bool(role) and isinstance(pin, dict),
                     "invalid named byte binding: " + field)
            _require(set(pin) == {"path", "bytes", "sha256"}
                     and isinstance(pin["path"], str) and bool(pin["path"])
                     and type(pin["bytes"]) is int and pin["bytes"] > 0
                     and isinstance(pin["sha256"], str) and len(pin["sha256"]) == 64
                     and all(c in "0123456789abcdef" for c in pin["sha256"]),
                     "invalid exact byte pin: " + role)
    for role in ("original_input_L23_fp16", "original_input_L23_binary64",
                 "original_input_final_fp16", "original_input_final_binary64"):
        if role not in contract["references"]:
            raise UnavailableBinding("missing original-input reference binding: " + role)
    for role in ("historical_generation", "current_diagnostic", "current_tests"):
        if role not in contract["sources"]:
            raise UnavailableBinding("missing source-role binding: " + role)
    controls = contract["controls"]
    _require(isinstance(controls, list) and len(controls) == 9
             and all(isinstance(c, str) and c for c in controls)
             and len(set(controls)) == 9, "nine retained controls required")
    if REPLACEMENT not in contract["inputs"]:
        raise UnavailableBinding("missing retained input binding: " + REPLACEMENT)
    _require(REPLACEMENT not in controls
             and set(contract["inputs"]) == set(controls) | {REPLACEMENT},
             "nine control archives plus replacement_stage11 pin census required")
    failures = contract["historical_failures"]
    _require(len(failures) == 9
             and all(isinstance(record, dict) for record in failures)
             and [record.get("control") for record in failures] == controls,
             "historical_failures must contain nine records in retained control order")
    _require(_encoded(contract["profile"]) == _encoded(PROFILE), "precision/profile changed")
    _require(contract["gates"] == dict.fromkeys(GATES, "PASS"), "retained integrity gate failed")
    _require(contract["source_identity"] == {"layer": 23, "position": 0, "token_id": 9707},
             "L23/P0 source identity changed")
    for field in ("account", "role_models", "budget", "access", "concurrency"):
        if field not in contract["launch_constraints"]:
            raise UnavailableBinding("missing external launch constraint: " + field)


class Candidate:
    """Immutable byte baseline with a mutable *working copy*, never a producer.

    The constructor's contract must later originate in independently authenticated
    retained evidence. Here it is only structurally checked and frozen verbatim;
    no claim is made that its source pins or retained PASS assertions are trusted.
    Thresholds are carried exactly, not parsed, relaxed, or replaced by defaults.
    """

    def __init__(self, actual, original_fp16, contract):
        _contract(contract)
        for key in ("input_hidden", "stage11"):
            _words(actual, key, (WIDTH,))
        _words(actual, "input_i", (WIDTH,), "<i8")
        zeros = _words(actual, "input_z", (WIDTH,), "|u1")
        _require(np.isin(zeros, (0, 1)).all(), "invalid Q24 signed-zero state")
        for kind, stage in (("k", "stage05"), ("v", "stage03")):
            _words(actual, "input_cache_" + kind, (0, 128))
            cached = _words(actual, "output_cache_" + kind, (1, 128))
            producer = _words(actual, stage, (128,))
            _require(cached[0].tobytes() == producer.tobytes(), "retained P0 KV lineage changed")
        replacement = canonical_fp16_member(original_fp16, "stage11")
        pin = contract["inputs"][REPLACEMENT]
        _require((pin["bytes"], pin["sha256"]) ==
                 (len(replacement), hashlib.sha256(replacement).hexdigest()),
                 "replacement_stage11 canonical member pin mismatch")
        self._actual = _snapshot(actual)
        self._reference = _snapshot(original_fp16)
        self._contract = _encoded(contract)
        self._recipe = _encoded(suffix_recipe())
        self._counts = dict.fromkeys(COUNTERS, 0)
        self._blocked_attempts = 0
        self._dispatch = None
        self._consumed = False

    def prepare(self):
        arrays = {}
        for key, (dtype, shape, data) in self._actual.items():
            if key == "stage11":
                dtype, shape, data = self._reference[key]
            arrays[key] = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            arrays[key].flags.writeable = False
        self.verify(arrays)
        return arrays

    def verify(self, arrays, *, original_fp16=None, contract=None, recipe=None):
        """Reject any non-S11 mutation; this phase permits no suffix execution."""
        expected = {**self._actual, "stage11": self._reference["stage11"]}
        _require(_snapshot(arrays) == expected,
                 "exact S11 replacement or protected source/operand/state/KV bytes changed")
        if original_fp16 is not None:
            _require(_snapshot(original_fp16) == self._reference, "original-input reference changed")
        if contract is not None:
            _require(_encoded(contract) == self._contract,
                     "frozen thresholds/reference/source/lineage/launch binding changed")
        if recipe is not None:
            _require(_encoded(recipe) == self._recipe, "forbidden or changed suffix recipe")
        _require(self._counts == dict.fromkeys(COUNTERS, 0), "non-generative invocation count changed")

    def invoke(self, operation, *args, **kwargs):
        if (self._dispatch is None or operation != self._dispatch[0] or args or kwargs):
            self._blocked_attempts += 1
            raise RuntimeError("candidate forbids invocation: " + str(operation))
        _, callback = self._dispatch
        self._dispatch = None
        return callback()

    def protected(self, arrays, completed):
        allowed = {key for step in suffix_recipe()[:completed] for key in step["outputs"]}
        expected = {**self._actual, "stage11": self._reference["stage11"]}
        snapshot = _snapshot(arrays)
        _require(set(snapshot) == set(expected) | allowed, "unexpected suffix output census")
        for key, value in expected.items():
            if key not in allowed:
                _require(snapshot[key] == value, "protected source/operand/state/KV changed: " + key)
        for key in allowed:
            if key.endswith(("_i", "_z")):
                value = _words(arrays, key, (WIDTH,), "<i8" if key.endswith("_i") else "|u1")
                if key.endswith("_z"):
                    _require(np.isin(value, (0, 1)).all(), "invalid residual zero state")
            elif key == "s16_unrounded_binary64":
                value = _words(arrays, key, self._actual["stage16"][1], "<f8")
                _require(np.isfinite(value).all(), "nonfinite native S16 operand")
            else:
                shape = (2,) if key == "pair_logits" else (
                    self._actual[key][1] if key in self._actual else (WIDTH,))
                _words(arrays, key, shape)

    def run_suffix(self, runtime, tensors, operands, trajectory, binary64, contract, final_oracle,
                   *, audit=None):
        """Run once on already authenticated inputs (or explicit synthetic doubles).

        This in-process arithmetic seam grants no launch authority. The CLI alone
        loads real inputs, after consuming external review/claim/capture bindings.
        """
        arrays = self.prepare()
        self.verify(arrays, original_fp16=trajectory, contract=contract, recipe=suffix_recipe())
        _require(not self._consumed, "suffix attempt already consumed")
        _require(len(operands) == 2 and tuple(v.shape for v in operands) == ((WIDTH,), (2, WIDTH))
                 and all(v.dtype.str == "<f2" and np.isfinite(v).all() for v in operands),
                 "final norm/two-row operand restriction")
        _require(binary64.dtype.str == "<f8" and binary64.shape == (WIDTH,)
                 and np.isfinite(binary64).all(), "original-input binary64 reference changed")
        protected_inputs = {"tensors": _snapshot(tensors), "trajectory": _snapshot(trajectory),
                            "binary64": _snapshot({"value": binary64}),
                            "operands": _snapshot({str(i): v for i, v in enumerate(operands)})}

        def unchanged(completed):
            self.protected(arrays, completed)
            _require(_encoded(contract) == self._contract, "frozen contract changed during suffix")
            _require(_encoded(suffix_recipe()) == self._recipe, "frozen suffix recipe changed")
            current = {"tensors": _snapshot(tensors), "trajectory": _snapshot(trajectory),
                       "binary64": _snapshot({"value": binary64}),
                       "operands": _snapshot({str(i): v for i, v in enumerate(operands)})}
            _require(current == protected_inputs, "source/operand/reference changed during suffix")

        if audit is None:
            audit = dict.fromkeys(SUFFIX_COUNTS, 0)
        before = {key: audit[key] for key in SUFFIX_COUNTS}
        state = {"i": arrays["input_i"], "z": arrays["input_z"], "h": arrays["input_hidden"]}
        active = {"arrays": arrays, "state": state}
        reports = []
        self._consumed = True
        with suffix_only(runtime, audit, active, tensors, operands):
            for index, operation in enumerate([*range(12, 19), "final_rmsnorm", "selected_tied_head"]):
                unchanged(index)
                active["stage"] = operation if isinstance(operation, int) else (
                    19 if operation == "final_rmsnorm" else 20)
                self._dispatch = (operation, lambda: compute_step(runtime, tensors, operands, active))
                try:
                    self.invoke(operation)
                finally:
                    self._dispatch = None
                unchanged(index + 1)
                if isinstance(operation, int):
                    expected = runtime.layer.expected_stage(operation, arrays, state, tensors)
                    reports.append(runtime.layer.stage_report(
                        operation, arrays, trajectory, binary64, expected))
                    unchanged(index + 1)
            # The reviewed final oracle must use the *new* S12 Q24 scratch, not
            # the retained scratch from before the S11 intervention.
            final_oracle({"scratch_i": arrays["scratch_i"], "scratch_z": arrays["scratch_z"]},
                         arrays, operands, active["norm_account"])
            unchanged(9)
        _require({key: audit[key] - before[key] for key in SUFFIX_COUNTS} == SUFFIX_COUNTS,
                 "suffix dispatch census changed")
        _require([r["stage"] for r in reports] == list(range(12, 19))
                 and all(r["status"] in ("PASS", "FAIL")
                         and r["residual_state_lineage"] == r["kv_lineage"] == "PASS"
                         for r in reports), "suffix gate census/integrity changed")
        outputs = {key: arrays[key].copy() for step in suffix_recipe() for key in step["outputs"]}
        return outputs, reports, audit, active["norm_account"]

    def metadata(self):
        contract = json.loads(self._contract)
        counts = dict(self._counts)
        _require(counts == dict.fromkeys(COUNTERS, 0), "non-generative invocation count changed")
        accounts = {
            "controls": contract["controls"], "branches": list(BRANCHES),
            "pair": list(PAIR), "directional_rows": 18,
            "replacement": {"stage": 11, "coordinates": list(range(WIDTH)),
                            "input": REPLACEMENT, "reference": "original_input_L23_fp16",
                            "member": "stage11", "word_dtype": "<u2",
                            "encoding": "raw-C-order-little-endian-uint16",
                            "bytes": 2 * WIDTH},
            "stage_reports": {"stages": list(range(12, 19)), "count": 63,
                              "gates": ["unchanged_local", "original_input_fp16",
                                        "original_input_binary64", *GATES]},
            "outputs_per_control": [key for op in suffix_recipe() for key in op["outputs"]],
            "contrasts_per_row": [
                "predicted_delta", "mlp_vector_delta", "retained_margin", "intervened_margin",
                "independent_original_reference_margin", "retained_margin_error",
                "intervened_margin_error", "margin_delta", "direction_observed",
            ],
            "prediction": "Freeze the negative retained full-stage17 weighted error before any "
                          "future dispatch; both stage17 and pair-margin movements must have "
                          "its strict nonzero sign. No nonlinear magnitude equality.",
            "final_checks": "Independent final RMSNorm and selected-row arithmetic oracle; "
                            "no new final-head admission threshold.",
        }
        permissions = {**dict.fromkeys(FORBIDDEN, 0),
                       **{f"stage{s:02d}": 9 for s in range(12, 19)},
                       "local_stage_oracles": 63, "final_rmsnorm": 9,
                       "selected_tied_head_rows": 18}
        termination = {
            "lane_terminated": True,
            "condition": "This implementation branch ends after compile, synthetic non-generative "
                         "tests and normal independent Host review; stop automatic scheduling.",
            "future_lane_condition": "One separately authorized observation ends on supported, "
                                     "rejected or unavailable/integrity failure; no replay or retuning.",
        }
        boundary = {
            "profile": copy.deepcopy(PROFILE), "cpu_only": True, "non_admission": True,
            "original_input_global_references_unchanged": True,
            "historical_failures_preserved": True, "closed_lanes_remain_closed": True,
            "strict_fp16_state_w4a16": False, "new_token_admission": False,
            "full_model_admission": False, "hardware_or_gpu": False, "ace2_action": False,
        }
        return {
            "schema_version": 1, "status": "IMPLEMENTED_NOT_EXECUTED",
            "diagnostic_identity": {
                "diagnostic": NAME,
                "observation": {"layer": 23, "position": 0, "pair": list(PAIR),
                                "replacement_stage": 11, "replacement_width": WIDTH,
                                "hypothesis": "Full original-input FP16 attention-output replacement "
                                              "may drive the retained stage17/final-margin response "
                                              "missed by the prior coordinate intervention."},
                "inputs": contract["inputs"], "references": contract["references"],
                "sources": contract["sources"],
                "command_semantics": {"mode": "dormant-executor-requires-external-authority",
                                      "entry": ["-B", "-m", MODULE, "--check"],
                                      "future_recipe": json.loads(self._recipe)},
                "permitted_invocations": permissions, "output_accounts": accounts,
                "termination": termination, "boundary": boundary,
            },
            "frozen_contract": contract,
            "binding_status": "REQUIRES_EXTERNAL_AUTHENTICATION_NOT_PERFORMED",
            "dispatch_authorized": False, "normal_host_review": "REQUIRED",
            "proposal_issued": False, "scientific_classification": None,
            "dispatch_and_write_audit": {**counts, "evidence_writes": 0,
                                         "blocked_invocation_attempts": self._blocked_attempts},
            "engineering_outcomes": {
                "PASS": "Synthetic checks establish only that this surface carries the frozen "
                        "intervention for independent proposal/review gating.",
                "FAIL": "A forbidden mutation/invocation or incorrect account was accepted.",
                "UNKNOWN": "One concrete required binding/interface fact is unavailable.",
            },
            "future_outcomes": {
                "supported": "Only bounded local sufficiency if every unchanged gate and both "
                             "directional contrasts pass; never admission.",
                "rejected": "A failed numerical gate or zero/opposite contrast falsifies sufficiency.",
                "unknown": "Missing binding or integrity defect terminates without a scientific claim.",
            },
            "lane_terminated": True,
        }


def compute_step(runtime, tensors, operands, active):
    """Use the reviewed native primitives, never a decoder or prefix producer."""
    native, parent = runtime.native, runtime.parent
    arrays, stage = active["arrays"], active["stage"]
    prefix = "model.layers.23."
    if stage in (12, 18):
        source, operand, output = (("state", "stage11", "scratch") if stage == 12 else
                                   ("scratch", "stage17", "output"))
        state = native.state.add(active[source], arrays[operand])
        arrays[output + "_i"], arrays[output + "_z"] = state["i"], state["z"]
        if stage == 12:
            active["scratch"] = state
        word = state["h"]
    elif stage == 13:
        value = native.decoded(arrays["stage12"])
        gamma = native.torch.from_numpy(tensors[prefix + "post_attention_layernorm.weight"].astype("<f8"))
        word = native.rne((value * native.torch.rsqrt(value.square().mean() + 1e-6)) * gamma)
    elif stage in (14, 15, 17):
        name, source = {14: ("gate", 13), 15: ("up", 13), 17: ("down", 16)}[stage]
        word = native.projection(tensors, prefix + "mlp." + name + "_proj", arrays[f"stage{source:02d}"])
    elif stage == 16:
        value = native.functional.silu(native.decoded(arrays["stage14"])) * native.decoded(arrays["stage15"])
        arrays["s16_unrounded_binary64"] = value.numpy().copy()
        word = native.toward_zero(arrays["s16_unrounded_binary64"])
    elif stage == 19:
        arrays["final_rmsnorm"], active["norm_account"] = parent.rmsnorm(arrays["stage18"], operands[0])
        return
    elif stage == 20:
        arrays["pair_logits"] = parent.logits(arrays["final_rmsnorm"], operands[1])
        return
    else:
        raise RuntimeError("forbidden suffix stage: " + str(stage))
    arrays[f"stage{stage:02d}"] = word


@contextmanager
def suffix_only(runtime, audit, active, tensors, operands):
    """Narrow the reviewed attention suffix guards to the frozen S11 replacement."""
    native, layer = runtime.native, runtime.layer
    raw_add = native.state.add

    def add(state, words):
        stage = active.get("stage")
        _require(stage in (12, 18), "Q24 add outside S12/S18")
        source, operand = ("state", "stage11") if stage == 12 else ("scratch", "stage17")
        _require(state is active[source] and words is active["arrays"][operand],
                 "Q24 state/operand scope changed")
        audit[f"s{stage}_adds"] += 1
        return raw_add(state, words)

    with runtime.guards.suffix_only(audit, active, tensors, operands), ExitStack() as stack:
        projection, expected = native.projection, layer.expected_stage

        def projected(*args):
            _require(active.get("stage") in (14, 15, 17), "projection outside S12-S18 suffix")
            return projection(*args)

        def oracle(stage, *args):
            _require(12 <= stage <= 18, "oracle outside S12-S18 suffix")
            return expected(stage, *args)

        def forbidden(*args, **kwargs):
            audit["forbidden_calls"] += 1
            raise RuntimeError("closed producer/reference/admission dispatch forbidden")

        for module in tuple(sys.modules.values()):
            if module and getattr(module, "__name__", "").startswith("ace3.model.candidates."):
                for name in ("execute", "classify", "_stages", "preflight_identity", "emit_custody"):
                    if callable(getattr(module, name, None)):
                        stack.enter_context(patch.object(module, name, forbidden))
        stack.enter_context(patch.object(native.state, "add", add))
        stack.enter_context(patch.object(native, "projection", projected))
        stack.enter_context(patch.object(layer, "expected_stage", oracle))
        yield


def output_census(controls, outputs, reports, rows):
    _require(len(controls) == 9 and len(set(controls)) == 9, "nine-control census changed")
    _require(list(outputs) == list(reports) == list(controls), "output control census changed")
    keys = {key for step in suffix_recipe() for key in step["outputs"]}
    _require(all(set(value) == keys for value in outputs.values()), "output-account census changed")
    _require(all([r["stage"] for r in report] == list(range(12, 19))
                 for report in reports.values()), "S12-S18 report census changed")
    _require([(r["control"], r["branch"]) for r in rows] ==
             [(c, b) for c in controls for b in BRANCHES], "directional output census changed")


def _load_runtime():
    from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_full_vector_suffix_intervention_v1
    return diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_full_vector_suffix_intervention_v1


class LaunchClaimError(ValueError, RuntimeError):
    """Claim rejection compatible with both candidate and captured-launch callers."""

    def __init__(self, message, differences=None):
        self.claim_differences = {} if differences is None else differences
        super().__init__(json.dumps({
            "message": message, "claim_differences": self.claim_differences,
        }, sort_keys=True, allow_nan=False))


def _claim_identity(record):
    # Only textual Host diagnostics are volatile; unknown and budget fields bind.
    volatile = ("notes", "last_error")
    _require(isinstance(record, dict)
             and all(key not in record or isinstance(record[key], str) for key in volatile),
             "native claim bookkeeping must be text")
    return {key: value for key, value in record.items() if key not in volatile}


def _bind_claim(expected, live, message):
    if _encoded(_claim_identity(expected)) != _encoded(_claim_identity(live)):
        differences = {
            key: {"authorized_present": key in expected, "live_present": key in live,
                  "authorized": expected.get(key), "live": live.get(key)}
            for key in sorted(expected.keys() | live.keys())
            if (key in expected) != (key in live)
            or _encoded(expected.get(key)) != _encoded(live.get(key))
        }
        raise LaunchClaimError(message, differences)


def _native_running_claim(backlog, mission_id):
    live = None
    with Path(backlog).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("id") == mission_id:
                live = row
    _require(live is not None and live["status"] == "running"
             and type(live["started_ts"]) in (int, float)
             and 0 < live["started_ts"] <= datetime.now(timezone.utc).timestamp(),
             "running claim is missing, terminal or superseded")
    return live


def _validate_claim_snapshot(declaration):
    launch = declaration["launch"]
    claim = capture.retained_document(declaration["claim"]["pin"])
    _require(claim["schema_version"] == 1 and claim["kind"] == "native_running_claim_snapshot"
             and declaration["claim"]["mission_path"] == ["mission_id"]
             and declaration["claim"]["status_path"] == ["status"]
             and declaration["claim"]["constraints_path"] == ["constraints"]
             and claim["mission_id"] == launch["mission_id"] and claim["status"] == "running"
             and claim["constraints"] == launch["constraints"],
             "current normal claim required")
    observed = datetime.fromisoformat(claim["observed_at_utc"])
    issued = datetime.fromisoformat(declaration["issued_at_utc"])
    _require(observed.tzinfo is not None and issued.tzinfo is not None
             and type(claim["started_ts"]) in (int, float)
             and 0 < claim["started_ts"] <= observed.timestamp() <= issued.timestamp()
             <= datetime.now(timezone.utc).timestamp(),
             "Manager authorization must follow the running claim")
    live = _native_running_claim(claim["source"]["backlog"], launch["mission_id"])
    _require(live["started_ts"] == claim["started_ts"],
             "running claim is missing, terminal or superseded")
    if "native_record" in claim:
        _bind_claim(claim["native_record"], live, "Manager snapshot native claim changed")
    return claim


def select_execution_authorization(authorizations, *, current_claim, backlog):
    """Select existing Manager pins by the runtime claim, never by the last issuance.

    Manager callers supply the full current native backlog row and its independently
    known backlog path after normal claim acquisition. A missing mission entry or a
    changed stable claim fails closed. Only textual Host notes/last_error may
    drift; this does not issue authority, refresh a snapshot, consume/reset a
    budget, or replace the remaining source/review launch gates.
    """
    mission = current_claim["id"]
    _bind_claim(current_claim, _native_running_claim(backlog, mission),
                "exact current native running claim required")
    _require(mission in authorizations, "Manager authorization unavailable for current mission")
    envelope = authorizations[mission]
    declaration = _execution_declaration(envelope)
    _require(declaration["launch"]["mission_id"] == mission,
             "Manager authorization mission differs from current native claim")
    claim = _validate_claim_snapshot(declaration)
    _require(claim["source"]["backlog"] == str(backlog)
             and claim["started_ts"] == current_claim["started_ts"],
             "Manager authorization snapshot differs from current native claim")
    _bind_claim(current_claim, _native_running_claim(backlog, mission),
                "native running claim changed during authorization selection")
    return envelope


def validate_launch_claim(preflight):
    """Mandatory captured --check gate; runtime context cannot come from authority."""
    try:
        current = preflight["normal_running_claim"]
        _require(current["id"] == preflight["mission_id"], "capture/current mission binding changed")
        return select_execution_authorization(
            {preflight["mission_id"]: preflight["execution_authorization"]},
            current_claim=current, backlog=preflight["native_backlog"])
    except LaunchClaimError:
        raise
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise LaunchClaimError(f"{type(error).__name__}: {error}") from error


def execution_authorization():
    """Consume, never emit, external authority and the shared retained capture."""
    _require(Path.cwd() == ROOT and sys.dont_write_bytecode and not sys.flags.optimize,
             "source/interpreter gate")
    out = Path(os.readlink("/proc/self/fd/1"))
    _require(out.name == "check.stdout" and out.parent.name == "run"
             and out.parent.parent.parent == ROOT / "build" and out.resolve() == out,
             "byte-exact shared capture required")
    _require(os.readlink("/proc/self/fd/2") == str(out.with_name("check.stderr")), "stderr capture")
    preflight = json.loads(out.with_name("check.environment.json").read_bytes())
    validate_launch_claim(preflight)
    declaration = validate_execution_authorization(preflight)
    launch = declaration["launch"]
    capture.verify_command(out.with_name("check.command.txt").read_text(),
                           json.loads(out.with_name("check.argv.json").read_bytes()),
                           preflight["environment"], launch["argv"], launch["environment"])
    return declaration, preflight


def _execution_declaration(envelope):
    declaration = capture.retained_document(envelope["declaration"])
    _require(declaration["schema_version"] == 1
             and declaration["kind"] == "stage11_suffix_execution_authorization"
             and declaration["issuer"] == "manager" and declaration["authorized"] is True
             and type(declaration["scientific_run_budget"]) is int
             and declaration["scientific_run_budget"] == 1,
             "independent Manager one-shot execution authority required")
    issuance = capture.retained_document(envelope["issuance"])
    _require(issuance["schema_version"] == 1
             and issuance["kind"] == "manager_stage11_execution_issuance"
             and issuance["issuer"] == "manager"
             and issuance["authorized_science"] is True
             and issuance["authorized_before_running_claim"] is False
             and type(issuance["scientific_run_budget"]) is int
             and issuance["scientific_run_budget"] == 1
             and issuance["declaration"] == envelope["declaration"]
             and issuance["claim"] == declaration["claim"]["pin"]
             and issuance["mission_id"] == declaration["launch"]["mission_id"]
             and issuance["accepted_identity_receipt"] == declaration["accepted_identity"]["launcher_capture"]
             and issuance["accepted_proposal"] == declaration["accepted_proposal"],
             "current Manager issuance/declaration/claim binding required")
    release.validate_execution_envelope(envelope)
    return declaration


def validate_execution_authorization(preflight):
    """Read-only evidence gates; captured launch additionally requires runtime context."""
    envelope = preflight["execution_authorization"]
    declaration = _execution_declaration(envelope)
    contract = declaration["contract"]
    _contract(contract)
    # The reviewed scientific identity keeps its historical source pins. The
    # shared release validator separately binds today's repaired implementation.
    identity_contract = dict(contract, sources=declaration["identity"]["sources"])
    _require(declaration["identity"] == _identity_for_contract(identity_contract),
             "reviewed diagnostic identity changed")
    _require(preflight["sources"] == [capture.binding(SOURCE), capture.binding(TEST)]
             and contract["sources"]["current_diagnostic"] == preflight["sources"][0]
             and contract["sources"]["current_tests"] == preflight["sources"][1],
             "current candidate/test bytes changed")
    capture.verify_source_roles(declaration["source_roles"])
    _require(declaration["source_roles"]["current_diagnostic"]["source"] ==
             contract["sources"]["current_diagnostic"]
             and declaration["source_roles"]["historical_generation"]["source"] ==
             contract["sources"]["historical_generation"], "source role splice")
    _require(set(declaration["source_snapshots"]) == set(contract["sources"]), "source snapshot census")
    for name, pin in contract["sources"].items():
        capture.verify_source_snapshot(pin, declaration["source_snapshots"][name])
    launch = declaration["launch"]
    argv = [sys.executable, "-B", "-m", MODULE, "--check"]
    _require(launch["cwd"] == str(ROOT) and launch["uid"] == os.getuid() == 1000
             and launch["environment"] == dict(os.environ)
             and launch["constraints"] == preflight["launch_constraints"] == contract["launch_constraints"]
             and launch["mission_id"] == preflight["mission_id"]
             and launch["mission_id"] not in ("51830fd5b6a1", "11859e7a4a7f"),
             "current account/claim/environment/role/model/budget/access/concurrency gate")
    capture.verify_command(launch["command"], argv, dict(os.environ), launch["argv"], launch["environment"])
    _require(preflight["environment"] == launch["environment"], "capture environment changed")
    _require(preflight["capture_implementation"] == capture.implementation_pins()
             and preflight["role"] == "engineer" and preflight["independent_host_review"] == "REQUIRED"
             and preflight["model_or_service_calls_authorized"] == 0
             and preflight["command_budget"] == {"compile": 1, "pytest": 1, "check": 1},
             "capture/role/review/command budget changed")
    _validate_claim_snapshot(declaration)
    review = envelope["review"]
    _require(review["mission_id"] == declaration["accepted_preflight_mission"]
             and review["mission_id"] != launch["mission_id"],
             "terminal evidence review must be distinct from the running execution mission")
    terminal_pin = declaration["accepted_preflight_terminal_receipt"]
    terminal = capture.retained_document(terminal_pin)
    _require(terminal["mission_id"] == review["mission_id"]
             and terminal["status"] == "ACCEPTED_NOVEL"
             and terminal["dispatch_authorized"] is False,
             "accepted terminal evidence splice")
    authority = terminal["manager_authority"]
    origin_review = authority["review"]
    _require(origin_review["mission_id"] not in (launch["mission_id"], review["mission_id"]),
             "independent Manager-origin evidence review required")
    return declaration


def _identity_for_contract(contract):
    # Metadata does not depend on retained arrays and confers no authority.
    _contract(contract)
    candidate = Candidate.__new__(Candidate)
    candidate._contract = _encoded(contract)
    candidate._recipe = _encoded(suffix_recipe())
    candidate._counts = dict.fromkeys(COUNTERS, 0)
    candidate._blocked_attempts = 0
    return candidate.metadata()["diagnostic_identity"]


def _bound_reference(pin, member):
    value = np.load(io.BytesIO(capture.verify_source_snapshot(pin, pin)), allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        with value:
            if value.files.count(member) != 1:
                raise UnavailableBinding("missing/ambiguous retained reference member: " + str(member))
            return value[member].copy()
    _require(member is None, "reference member does not match retained array")
    return value


def _bound_replacement_stage11(contract):
    """Authenticate the operand against the exact archived member, before runtime load."""
    _contract(contract)
    pin = contract["inputs"][REPLACEMENT]
    data = capture.verify_source_snapshot(pin, pin)
    reference = _bound_reference(contract["references"]["original_input_L23_fp16"], "stage11")
    _require(data == canonical_fp16_member({"stage11": reference}, "stage11"),
             "replacement_stage11 bytes differ from canonical original-input L23 FP16 stage11 member")
    return np.frombuffer(data, dtype="<u2").copy()


def _retained_failure_lineage(contract, original):
    """Require full, type-sensitive equality to the authenticated retained report."""
    _require(contract["historical_failures"] == original["retained_controls_and_failure_gates"]
             and _encoded(contract["historical_failures"]) ==
             _encoded(original["retained_controls_and_failure_gates"])
             and contract["lineage"] == original["report"]["lineage_separation"],
             "retained failure/lineage binding changed")


def check(audit=None):
    if audit is None:
        audit = dict.fromkeys((*COUNTERS, *SUFFIX_COUNTS), 0)
    declaration, preflight = execution_authorization()
    contract = declaration["contract"]
    replacement = _bound_replacement_stage11(contract)
    runtime = _load_runtime()
    runtime.native.torch.set_num_threads(1)
    _require(str(runtime.native.torch.tensor(0).device) == "cpu", "CPU-only execution")
    _require(contract["controls"] == list(runtime.CONTROLS), "retained controls changed")
    oracle = runtime.shared.load_module(runtime.TEST, NAME + "_reviewed_final_oracle")
    with runtime.guards.no_writes(audit), runtime.parent.no_dispatch(audit):
        sources = {**runtime.parent.source_context(), "stage13_suffix": capture.binding(runtime.SOURCE),
                   "stage13_tests": capture.binding(runtime.TEST)}
        _require(sources == declaration["runtime_sources"], "reviewed runtime source bytes changed")
        original = runtime.direct.authenticate()
        result, baseline, references, files = runtime.retained.authenticate()
        summary = result["preflight"]
        _require(contract["thresholds"] == summary["thresholds"], "original thresholds changed")
        _retained_failure_lineage(contract, original)
        _require(contract["references"]["original_input_L23_fp16"] ==
                 summary["L23_original_reference"]["reference"]["fp16"]
                 and contract["references"]["original_input_L23_binary64"] ==
                 summary["L23_original_reference"]["reference"]["binary64"],
                 "original-input L23 references reanchored")
        archives = {}
        _require(set(contract["inputs"]) == set(runtime.CONTROLS) | {REPLACEMENT},
                 "working archive/replacement pin census")
        for row in result["controls"]:
            pin = row["parent"]["terminal_archive"]
            _require(pin == contract["inputs"][row["control"]], "working input pin changed")
            archives[row["control"]] = runtime.layer.archive(pin)
        tensors, trajectory, binary64, checkpoint = runtime.layer.load_inputs({"summary": summary})
        _require(canonical_fp16_member(trajectory, "stage11") == replacement.tobytes(),
                 "loaded original-input stage11 differs from bound replacement_stage11")
        trajectory["stage11"] = replacement
        _require(runtime.parent.preflight.bind_assets(checkpoint) == summary["assets"], "model asset identity")
        full_operands = runtime.parent.load_operands(summary)
        operands = (full_operands[0], full_operands[1][list(PAIR)].copy())
        for branch in BRANCHES:
            role = "original_input_final_" + branch
            _require(contract["references"][role] in files, "unauthenticated final reference pin")
            reference = _bound_reference(contract["references"][role], declaration["reference_members"][role])
            _require(_snapshot({"v": reference}) == _snapshot({"v": references["logits_" + branch]}),
                     "original-input final reference reanchored")
        for value in (*tensors.values(), *trajectory.values(), binary64, *operands,
                      *references.values(), *(v for a in archives.values() for v in a.values())):
            value.flags.writeable = False
        # The reviewed full-vector predictor depends on retained stage17, not on
        # the replacement stage. No stage13 producer or classifier is called.
        predictions = runtime.preregister(original, archives, trajectory, operands)["rows"]
        outputs, reports, norm_accounts = {}, {}, {}
        audit["scientific_invocations"] += 1
        for control in contract["controls"]:
            candidate = Candidate(archives[control], trajectory, contract)
            outputs[control], reports[control], _, norm_accounts[control] = candidate.run_suffix(
                runtime, tensors, operands, trajectory, binary64, contract, oracle.verify_final_suffix,
                audit=audit)
        rows = []
        exact = runtime.exact
        for prediction in predictions:
            control, branch = prediction["control"], prediction["branch"]
            mlp = sum((Fraction(f) * (exact(new) - exact(old)) for f, new, old in
                       zip(prediction["factors"], outputs[control]["stage17"],
                           archives[control]["stage17"], strict=True)), Fraction())
            old = exact(baseline[control]["logits"][PAIR[0]]) - exact(baseline[control]["logits"][PAIR[1]])
            new = exact(outputs[control]["pair_logits"][0]) - exact(outputs[control]["pair_logits"][1])
            ref = references["logits_" + branch]
            ref_margin = (exact(ref[PAIR[0]]) - exact(ref[PAIR[1]]) if branch == "fp16" else
                          Fraction.from_float(float(ref[PAIR[0]])) - Fraction.from_float(float(ref[PAIR[1]])))
            p, delta = Fraction(prediction["predicted_delta"]), new - old
            rows.append({"control": control, "branch": branch, "predicted_delta": str(p),
                         "mlp_vector_delta": str(mlp), "retained_margin": str(old),
                         "intervened_margin": str(new), "independent_original_reference_margin": str(ref_margin),
                         "retained_margin_error": str(old - ref_margin), "intervened_margin_error": str(new - ref_margin),
                         "margin_delta": str(delta), "direction_observed": p * mlp > 0 and p * delta > 0})
        output_census(contract["controls"], outputs, reports, rows)
        _require({key: audit[key] for key in SUFFIX_COUNTS} ==
                 {key: count * 9 for key, count in SUFFIX_COUNTS.items()}, "complete dispatch census")
        for pin in (*sources.values(), *files, *original["hidden_pins"], *runtime.direct.PINS.values(),
                    *contract["inputs"].values(), *contract["references"].values()):
            runtime.bound_bytes(pin)
        _require(execution_authorization() == (declaration, preflight), "execution authority changed")
    supported = all(r["direction_observed"] for r in rows) and all(
        r["status"] == "PASS" for report in reports.values() for r in report)
    return {"diagnostic_identity": _identity_for_contract(contract), "status": "SUPPORTED" if supported else "REJECTED",
            "outputs": {c: {k: v.tolist() for k, v in values.items()} for c, values in outputs.items()},
            "stage_reports": reports, "rows": rows, "norm_accounts": norm_accounts,
            "dispatch_and_write_audit": audit, "capture_preflight": preflight,
            "frozen_contract": contract, "non_admission": True, "lane_terminated": True,
            "normal_host_review": "REQUIRED", "final_arithmetic_oracle": "PASS"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", required=True, action="store_true")
    parser.parse_args(argv)
    audit = dict.fromkeys((*COUNTERS, *SUFFIX_COUNTS), 0)
    try:
        result = check(audit)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, AssertionError, ArithmeticError) as error:
        print(json.dumps({"diagnostic": NAME, "status": "UNKNOWN", "error": str(error),
                          "error_type": type(error).__name__, "lane_terminated": True,
                          **({"claim_differences": error.claim_differences}
                             if isinstance(error, LaunchClaimError) else {}),
                          "dispatch_and_write_audit": audit, "non_admission": True,
                          "normal_host_review": "REQUIRED"}, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    from importlib import import_module

    raise SystemExit(import_module(MODULE).main())
