"""Retained-only diagnostic identity gate; no imports or execution of producers.

Call preflight_identity with independently trusted proposal/catalog byte pins and
the actual proposed launch identity BEFORE reserving a scientific attempt. A
catalog must enumerate the entire scoped closed-lane set supplied by Planner.
Pins supplied by an untrusted proposer are not an authority root. ACCEPTED_NOVEL
does not replace account, role/model, access, concurrency or one-shot budget gates.
This module has no dispatch API.

diagnostic_identity_authority_bundle_v1 can materialize these schema-1 inputs
from explicitly pinned retained fields. Its output pins are not authority roots:
validate the bundle against independently trusted origins before consumption.
diagnostic_identity_custody_v1 preserves creation/review-time declarations as
proposal/catalog records for future lanes; its custody PASS is not novelty.

Schema 1 proposal: identity, source_roles, source_snapshots, launch. The catalog
contains lanes with mission_id, latest/review/checkpoint pins, native (pin and
mission_path/status_path), capture (directory/receipt/preflight/commands/label),
identity_paths and source_snapshots. Paths are nonempty JSON key/index lists into
authenticated native bytes or captured stdout, not expressions to evaluate.
Embedded identities retain their strict captured-field semantics. A separately
Manager-pinned current declaration can describe legacy stage13 bytes through
the explicit provenance adapter below. It never inserts identity into stdout.
Reading a pending declaration for review is not authority to consume it.
Reference provenance uses named precision-specific fields, never a recursive
hash search. Command equivalence ignores only environment-key ordering.

Identity fields: diagnostic, observation, inputs, references, sources,
command_semantics, permitted_invocations, output_accounts, termination, boundary.
Input/reference/source maps contain exact byte pins. Source snapshots separately
bind historical bytes; a changed live source cannot substitute for a snapshot.
Packaging (task ID, wrapper, output directory) is deliberately outside identity.
Novelty requires a different structural observation, different input CONTENT,
and different operator permissions against EVERY closed lane. Reference and
boundary changes are forbidden. All other single-field changes fail closed.
"""

import ast
from datetime import datetime, timezone
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np

from ace3.model.candidates import diagnostic_capture_v1 as capture


FIELDS = frozenset({
    "diagnostic", "observation", "inputs", "references", "sources",
    "command_semantics", "permitted_invocations", "output_accounts",
    "termination", "boundary",
})
CURRENT_DECLARATION = "current-review-declaration-over-legacy-bytes"
MEMBER_ENCODING = "raw-C-order-little-endian-uint16"
STAGE13_BOUNDARY = (
    "One L23/P0 full stage13 FP16-vector replacement, pair (34319,13), nine "
    "retained controls. Only stage14-stage18, final RMSNorm and two head rows "
    "execute. Original-input FP16/binary64 references, exact existing thresholds, "
    "native-S16-RTZ, G128 asymmetric INT4/native GEMM/no qzero plus-one, FP16 "
    "scales/operator/KV boundaries and wider-than-FP16 Q24 residual state are fixed. "
    "No prefix/admission/original-reference/closed-producer/source-bridge replay, "
    "strict-FP16-state W4A16, new-token/full-model admission, precision or scale "
    "expansion, hardware/GPU/RTL or ACE2 action. Terminate this full-vector lane "
    "after its one observation; independent Host review is required."
)
STAGE13_FLAGS = {
    "accepted_prefix_replay": False, "admission_invocations": 0, "admission_replay": False,
    "candidate_admitted": False, "closed_producer_replay": 0, "external_service_invocations": 0,
    "full_model_claim": False, "full_vector_stage13_intervention": True,
    "full_vocabulary_head_recomputation": False, "gpu_invocations": 0, "hardware_invocations": 0,
    "historical_failures_preserved": True, "native_L0_L22_invocations": 0, "new_token_claim": False,
    "original_global_reference_unchanged": True, "original_prefix_replay": False,
    "original_reference_replay": 0, "policy_adopted": False, "prefix_invocations": 0,
    "prefix_replay": 0, "reference_producer_invocations": 0, "reference_reanchoring": False,
    "reference_recomputation": False, "retained_evidence_writes": 0, "rtl_invocations": 0,
    "score_replay": 0, "simulation_invocations": 0, "softmax_replay": 0,
    "stage13_rmsnorm_replay": 0, "strict_FP16_state_claim": False,
    "successor_published": False, "token_selection": False, "v_projection_replay": 0,
}


def _at(document, path, binding_name):
    if not isinstance(path, list) or not path:
        raise RuntimeError("nonempty retained JSON path required")
    for key in path:
        if not isinstance(key, str) and type(key) is not int:
            raise RuntimeError("invalid retained JSON path component")
        try:
            if isinstance(document, list) and (type(key) is not int or key < 0):
                raise KeyError(key)
            document = document[key]
        except (KeyError, IndexError, TypeError) as error:
            raise capture.UnavailableBinding(
                f"missing retained field {path!r} in {binding_name}") from error
    return document


def _content(pins):
    return {name: {"bytes": pin["bytes"], "sha256": pin["sha256"]}
            for name, pin in pins.items()}


def _input_content(identity):
    return sorted((pin["bytes"], pin["sha256"]) for pin in identity["inputs"].values())


def _identity(identity, snapshots):
    if not isinstance(identity, dict) or set(identity) != FIELDS:
        raise RuntimeError("complete diagnostic identity required")
    if not isinstance(identity["diagnostic"], str) or not identity["diagnostic"]:
        raise RuntimeError("diagnostic identity required")
    for field in ("observation", "command_semantics", "output_accounts", "boundary"):
        if not isinstance(identity[field], dict) or not identity[field]:
            raise RuntimeError(f"nonempty structural {field} required")
    permissions = identity["permitted_invocations"]
    if (not isinstance(permissions, dict) or not permissions
            or not all(isinstance(k, str) and k and type(v) is int and v >= 0
                       for k, v in permissions.items())):
        raise RuntimeError("exact permitted operator invocation counts required")
    termination = identity["termination"]
    if (not isinstance(termination, dict) or termination.get("lane_terminated") is not True
            or not isinstance(termination.get("condition"), str) or not termination["condition"]):
        raise RuntimeError("explicit closed-lane termination claim required")
    if not isinstance(snapshots, dict) or set(snapshots) != set(identity["sources"]):
        raise RuntimeError("complete historical source snapshots required")
    normalized = dict(identity)
    for field in ("inputs", "references", "sources"):
        pins = identity[field]
        if not isinstance(pins, dict) or not pins:
            raise RuntimeError(f"fixed {field} pins required")
        for name, pin in pins.items():
            if not isinstance(name, str) or not name:
                raise RuntimeError("named fixed pin required")
            capture.verify_source_snapshot(pin, snapshots[name] if field == "sources" else pin)
        normalized[field] = _content(pins)
    return normalized


def verify_generation_origin(source_roles, runs):
    """Require the original path as well as bytes; a snapshot is not a new origin."""
    capture.verify_source_roles(source_roles)
    sources = []
    for run in runs:
        sources.extend(run["preflight"]["sources"])
        sources.extend(run["preflight"].get("capture_implementation", []))
    source = source_roles["historical_generation"]["source"]
    if source not in sources:
        raise RuntimeError(
            "historical source role is not bound to a retained generation: "
            f"{source['path']} ({source['sha256']}); "
            "matching snapshot bytes do not authenticate a different original source path")


def _same(actual, expected, name):
    if capture.encoded(actual) != capture.encoded(expected):
        raise RuntimeError(f"current declaration {name} mismatch")


def _retained_run(run):
    """JSON command maps are unordered; the pinned receipt owns execution order."""
    receipt = capture.retained_document(run["receipt"])
    labels = [row["label"] for row in receipt["results"]]
    if len(set(labels)) != len(labels) or set(labels) != set(run["commands"]):
        raise RuntimeError("capture command census mismatch")
    return capture.verify_retained_run(
        Path(run["directory"]), run["receipt"], run["preflight"],
        {label: run["commands"][label] for label in labels}, require_success=False)


def verify_stage_member(spec, member):
    """Compare a canonical member to authenticated NPZ bytes, without arithmetic."""
    if (set(spec) != {"source_archive", "source_member", "encoding", "derived_pin"}
            or spec["source_member"] != member or spec["encoding"] != MEMBER_ENCODING):
        raise RuntimeError("canonical stage-member provenance required")
    raw = capture._retained_bytes(spec["source_archive"], capture._validate_pin(spec["source_archive"]))
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        if archive.files.count(member) != 1:
            raise RuntimeError("unique canonical stage member required")
        words = archive[member]
        if words.dtype.str != "<u2" or words.shape != (896,):
            raise RuntimeError("canonical stage member shape/dtype mismatch")
        encoded = words.tobytes(order="C")
    declared = capture._retained_bytes(spec["derived_pin"], capture._validate_pin(spec["derived_pin"]))
    if declared != encoded:
        raise RuntimeError("canonical stage-member bytes mismatch")
    return words


def _native_event(lane, issued):
    spec = lane["native"]
    native = capture.retained_document(spec["pin"])
    fields = ("source_path", "byte_offset", "byte_length", "line_sha256")
    _same({key: native[key] for key in fields}, {key: spec[key] for key in fields},
          "native excerpt metadata")
    offset, length = spec["byte_offset"], spec["byte_length"]
    if type(offset) is not int or offset < 0 or type(length) is not int or length <= 0:
        raise RuntimeError("native event byte range required")
    path = capture._canonical_path(spec["source_path"])
    try:
        with path.open("rb") as stream:
            if offset:
                stream.seek(offset - 1)
                if stream.read(1) != b"\n":
                    raise RuntimeError("native event offset is not a line boundary")
            stream.seek(offset)
            raw = stream.read(length)
    except FileNotFoundError as error:
        raise capture.UnavailableBinding(f"missing native event original: {path}") from error
    if (len(raw) != length or not raw.endswith(b"\n") or raw.count(b"\n") != 1
            or hashlib.sha256(raw).hexdigest() != spec["line_sha256"]):
        raise RuntimeError("native event excerpt/range mismatch")
    event = capture.decode_retained(raw)
    _same(native["event"], event, "native event bytes")
    _same(spec["mission_path"], ["event", "item_id"], "native mission selector")
    _same(spec["status_path"], ["event", "status"], "native status selector")
    review = capture.retained_document(lane["review"])
    if (event.get("type") != "life.mission.completed"
            or event.get("item_id") != lane["mission_id"] or event.get("status") != "done"
            or event.get("success") is not True or event.get("final_review_source") != "reviewer"
            or event.get("final_review_status") != "done"
            or event.get("independent_review_required") is not True
            or event.get("context_packet") != lane["latest"]["path"]
            or event.get("rounds") != review.get("round")
            or type(event.get("ts")) not in (int, float)
            or not review["created_at"] <= event["ts"] <= issued):
        raise RuntimeError("native completion/latest Reviewer/time mismatch")
    return native


def _current_review(lane, authority, graph, review_pending):
    latest, review = (capture.retained_document(lane[key]) for key in ("latest", "review"))
    checkpoint = capture._retained_bytes(lane["checkpoint"], capture._validate_pin(lane["checkpoint"]))
    if (latest.get("kind") != "handoff_ref"
            or latest.get("handoff") != {"path": lane["review"]["path"]}
            or review.get("kind") != "round_reviewed_handoff"
            or review.get("producer_role") != "reviewer"
            or review.get("mission_id") != lane["mission_id"]
            or review.get("review", {}).get("status") != "done"
            or review.get("checkpoint") != {"path": lane["checkpoint"]["path"]}):
        raise RuntimeError("latest terminal Reviewer identity mismatch")
    current = authority.get("review")
    if current is None:
        if not review_pending or not checkpoint.strip():
            raise capture.UnavailableBinding("separate current reviewed declaration receipts required")
    else:
        current_review = capture.verify_terminal_review(
            current["latest"], current["review"], current["checkpoint"], current["mission_id"])
        if (current["review"]["path"] == lane["review"]["path"]
                or current["mission_id"] == lane["mission_id"]
                or type(current_review.get("created_at")) not in (int, float)
                or not graph["issued_timestamp"] <= current_review["created_at"]
                <= datetime.now(timezone.utc).timestamp()):
            raise RuntimeError("separate present-time independent declaration review required")
        pins = [authority["issuance"], *graph["origins"].values(), lane["latest"],
                lane["review"], lane["native"]["pin"], lane["capture"]["receipt"],
                *lane["source_snapshots"].values()]
        capture.verify_review_links(current["review"], current["checkpoint"], current["mission_id"], pins)
        current_text = capture._retained_bytes(
            current["checkpoint"], capture._validate_pin(current["checkpoint"]))
        if (lane["checkpoint"]["sha256"].encode() not in current_text
                or lane["checkpoint"]["path"].encode() not in current_text):
            raise RuntimeError("current review lacks exact legacy checkpoint binding")
    # The current native excerpt did not exist at legacy review time.
    if checkpoint.strip():
        capture.verify_review_links(lane["review"], lane["checkpoint"], lane["mission_id"],
                                    [lane["capture"]["receipt"]])


def _legacy_stage13_identity(lane, receipt, row, payload):
    identity, provenance = lane["diagnostic_identity"], lane["identity_provenance"]
    if "diagnostic_identity" in payload or lane["legacy_diagnostic_identity_embedded"] is not False:
        raise RuntimeError("current legacy declaration cannot override embedded identity")
    if not isinstance(provenance, dict) or set(provenance) != FIELDS:
        raise RuntimeError("complete field-by-field provenance required")
    stdout = row["files"][3]
    selectors = {
        "diagnostic": {"path": ["diagnostic_id"]},
        "observation": {"paths": [["preregistration"], ["stage13_bindings"]]},
        "references": {"paths": [["L23_reference_authority"], ["final_reference_authority"]]},
        "output_accounts": {"paths": [["outputs"], ["stage_reports"], ["rows"]]},
        "permitted_invocations": {"path": ["dispatch_and_write_audit"]},
        "termination": {"paths": [["lane_terminated"], ["status"]]},
        "boundary": {"paths": [["claim_boundary"], ["flags"], ["original_thresholds"]]},
    }
    for field, paths in selectors.items():
        _same(provenance[field], {"pin": stdout, **paths}, f"{field} provenance")
        for path in paths.get("paths", [paths.get("path")]):
            _at(payload, path, stdout["path"])
    _same(provenance["sources"], {"pin": row["files"][2], "path": ["sources"]},
          "source provenance")
    _same(provenance["command_semantics"],
          {"receipt": lane["capture"]["receipt"], "label": row["label"]}, "command provenance")
    inputs = provenance["inputs"]
    _same(set(inputs) == {"working_archives", "replacement_stage13"}, True, "input provenance census")
    _same(inputs["working_archives"], {"pin": stdout, "path": ["outputs"]},
          "working-archive provenance")
    replacement = inputs["replacement_stage13"]
    words = verify_stage_member(replacement, "stage13")
    bindings, outputs, plan = payload["stage13_bindings"], payload["outputs"], payload["preregistration"]
    controls = list(outputs)
    if len(controls) != 9 or set(bindings) != {*controls, "original_input_fp16"}:
        raise RuntimeError("legacy control/archive census mismatch")
    for control in [*controls, "original_input_fp16"]:
        binding = bindings[control]
        _same({key: binding[key] for key in ("member", "shape", "dtype")},
              {"member": "stage13", "shape": [896], "dtype": "<u2"}, "legacy stage binding")
    _same(bindings["original_input_fp16"]["archive"], replacement["source_archive"],
          "replacement original-input archive")
    expected_inputs = {control: bindings[control]["archive"] for control in controls}
    expected_inputs["replacement_stage13"] = replacement["derived_pin"]
    _same(identity["inputs"], expected_inputs, "input pins")
    _same(plan["pair"], [34319, 13], "frozen pair")
    _same(plan["source_position"], 0, "position")
    _same(plan["coordinates"], list(range(896)), "full-vector coordinates")
    for control in controls:
        _same(outputs[control]["stage13"], words.tolist(), "retained replacement words")
    _same(identity["diagnostic"], payload["diagnostic_id"], "diagnostic")
    _same(identity["observation"], {
        "hypothesis": "Full original-input FP16 stage13 replacement sufficiency was tested and terminated.",
        "layer": 23, "pair": plan["pair"], "position": plan["source_position"],
        "replacement_stage": 13, "replacement_width": len(plan["coordinates"])}, "observation")
    command = identity["command_semantics"]
    environment = receipt["preflight"]["environment"]
    declared_command = capture.verify_command(
        command["command"], command["argv"], environment, row["argv"], environment)
    retained_command = capture.verify_command(
        row["command"], row["argv"], environment, row["argv"], environment)
    _same({**command, "command": declared_command},
          {"argv": row["argv"], "command": retained_command,
           "mode": "retained-executed-full-vector-suffix",
           "stages": [14, 15, 16, 17, 18, "final_rmsnorm", "selected_tied_head"]},
          "command semantics")
    _same(identity["permitted_invocations"], payload["dispatch_and_write_audit"], "operator accounts")
    _same(identity["permitted_invocations"], {
        "final_norms": 9, "forbidden_calls": 0, "local_oracles": 45,
        "pair_heads": 9, "projections": 27, "s18_adds": 9}, "bounded invocation census")
    reports, rows = payload["stage_reports"], payload["rows"]
    _same(sorted(reports), sorted(controls), "stage report controls")
    for reports_for_control in reports.values():
        _same([report["stage"] for report in reports_for_control], [14, 15, 16, 17, 18], "suffix stages")
    _same(sorted((item["control"], item["branch"]) for item in rows),
          sorted((control, branch) for control in controls for branch in ("fp16", "binary64")),
          "row census")
    _same(identity["output_accounts"], {
        "controls": len(controls), "pair": plan["pair"], "rows": len(rows),
        "stage_reports": sum(len(value) for value in reports.values()),
        "replacement": {"bytes": replacement["derived_pin"]["bytes"], "encoding": MEMBER_ENCODING,
                        "input": "replacement_stage13", "member": "stage13"}}, "output accounts")
    _same(set(identity["sources"]) == {"historical_diagnostic", "historical_tests"}, True,
          "historical source-role census")
    _same([identity["sources"][name] for name in ("historical_diagnostic", "historical_tests")],
          receipt["preflight"]["sources"], "historical sources")
    _same(payload["lane_terminated"], True, "retained termination")
    if payload["status"] not in {"REJECTED", "SUPPORTED", "UNKNOWN"}:
        raise RuntimeError("retained terminal observation status required")
    _same(identity["termination"], {
        "lane_terminated": True,
        "condition": f"The reviewed {lane['mission_id']} full-stage13-vector suffix lane terminated "
                     f"after its one retained {payload['status']} observation."}, "termination")
    reference_fields = {}
    for kind, dtype in (("fp16", "<u2"), ("binary64", "<f8")):
        reference_fields["original_input_L23_" + kind] = (
            payload["L23_reference_authority"]["reference"][kind])
        logits = payload["final_reference_authority"]["reference"]["logits_" + kind]
        _same({key: logits[key] for key in ("dtype", "shape")},
              {"dtype": dtype, "shape": [151936]}, f"named final {kind} reference metadata")
        reference_fields["original_input_final_" + kind] = {
            key: logits[key] for key in ("path", "bytes", "sha256")}
    _same(identity["references"], reference_fields, "named original reference pins")
    _same(identity["references"]["original_input_L23_fp16"], replacement["source_archive"],
          "original reference member source")
    source = capture.verify_source_snapshot(identity["sources"]["historical_diagnostic"],
                                            lane["source_snapshots"]["historical_diagnostic"])
    literals = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    literals[target.id] = node.value.value
    _same(literals.get("NAME"), identity["diagnostic"], "source diagnostic literal")
    _same(literals.get("BOUNDARY"), payload["claim_boundary"], "source boundary literal")
    _same(payload["claim_boundary"], STAGE13_BOUNDARY, "supported legacy boundary encoding")
    boundary = identity["boundary"]
    _same(boundary, {
        "ace2_action": False, "closed_lanes_remain_closed": True, "cpu_only": True,
        "full_model_admission": False, "hardware_or_gpu": False, "historical_failures_preserved": True,
        "new_token_admission": False, "non_admission": True,
        "original_input_global_references_unchanged": True,
        "profile": {"kv": "FP16", "nibble_order": "native-GEMM", "operator_boundaries": "FP16",
                    "qzero_plus_one": False, "residual_state": "Q24-wide-not-FP16", "scales": "FP16",
                    "stage16_rounding": "native-S16-RTZ", "weights": "G128-asymmetric-packed-INT4"},
        "strict_fp16_state_w4a16": False}, "non-admission boundary")
    _same(payload["flags"], STAGE13_FLAGS, "complete retained boundary flags")
    normalized = _identity(identity, lane["source_snapshots"])
    normalized["command_semantics"] = {**command, "command": declared_command}
    return normalized


def _closed_lane(lane, *, manager_authority=None, review_pending=False):
    if lane.get("declaration_mode") == CURRENT_DECLARATION:
        from ace3.model.candidates import diagnostic_identity_custody_v1 as custody

        authority = manager_authority if manager_authority is not None else lane.get("current_declaration")
        if authority is None:
            raise capture.UnavailableBinding("independent Manager issuance for current declaration required")
        graph = custody.authenticate_manager_origins(authority["issuance"], authority["proposal_id"])
        declared = [item for item in graph["terminals"]["lanes"] if item["mission_id"] == lane["mission_id"]]
        _same(declared, [{key: value for key, value in lane.items() if key != "current_declaration"}],
              "Manager terminal declaration")
        _current_review(lane, authority, graph, review_pending)
        _native_event(lane, graph["issued_timestamp"])
        receipt = _retained_run(lane["capture"])
        rows = [row for row in receipt["results"] if row["label"] == lane["capture"]["label"]]
        if len(rows) != 1:
            raise RuntimeError("unique retained identity payload required")
        return _legacy_stage13_identity(lane, receipt, rows[0], capture.retained_document(rows[0]["files"][3]))
    if "declaration_mode" in lane or "diagnostic_identity" in lane:
        raise RuntimeError("unsupported terminal identity declaration mode")
    mission = lane["mission_id"]
    capture.verify_terminal_review(lane["latest"], lane["review"], lane["checkpoint"], mission)
    native_spec, run = lane["native"], lane["capture"]
    capture.verify_review_links(lane["review"], lane["checkpoint"], mission,
                                [native_spec["pin"], run["receipt"]])
    native = capture.retained_document(native_spec["pin"])
    if (_at(native, native_spec["mission_path"], native_spec["pin"]["path"]) != mission
            or _at(native, native_spec["status_path"], native_spec["pin"]["path"]) != "done"):
        raise RuntimeError("native terminal mission/status mismatch")
    receipt = capture.verify_retained_run(
        Path(run["directory"]), run["receipt"], run["preflight"], run["commands"],
        require_success=False)
    rows = [row for row in receipt["results"] if row["label"] == run["label"]]
    if len(rows) != 1:
        raise capture.UnavailableBinding(f"missing retained payload label: {run['label']}")
    payload_pin = rows[0]["files"][3]
    payload = capture.retained_document(payload_pin)
    paths = lane["identity_paths"]
    if not isinstance(paths, dict) or set(paths) != FIELDS:
        raise capture.UnavailableBinding(f"missing retained scientific identity field binding: {mission}")
    identity = {field: _at(payload, paths[field], payload_pin["path"]) for field in FIELDS}
    if not all(pin in receipt["preflight"]["sources"] for pin in identity["sources"].values()):
        raise RuntimeError("scientific source/retained generation preflight mismatch")
    return _identity(identity, lane["source_snapshots"])


def preflight_identity(proposal_pin, catalog_pin, launch):
    """Return a fail-closed decision only; never run a command or reserve a run."""
    result = {"status": "REJECTED", "dispatch_authorized": False,
              "scientific_invocations": 0, "producer_invocations": 0}
    try:
        proposal, catalog = (capture.retained_document(pin) for pin in (proposal_pin, catalog_pin))
        if (not isinstance(proposal, dict) or not isinstance(catalog, dict)
                or type(proposal.get("schema_version")) is not int
                or type(catalog.get("schema_version")) is not int
                or proposal["schema_version"] != 1 or catalog["schema_version"] != 1):
            raise RuntimeError("unsupported diagnostic identity schema")
        capture.verify_source_roles(proposal["source_roles"])
        expected = proposal["launch"]
        actual_command = capture.verify_command(
            launch["command"], launch["argv"], launch["environment"],
            expected["argv"], expected["environment"])
        expected_command = capture.verify_command(
            expected["command"], expected["argv"], expected["environment"],
            expected["argv"], expected["environment"])
        if capture.encoded({**launch, "command": actual_command}) != capture.encoded(
                {**expected, "command": expected_command}):
            raise RuntimeError("proposed command/account/role/budget identity mismatch")
        candidate = _identity(proposal["identity"], proposal["source_snapshots"])
        lanes = catalog["lanes"]
        if not isinstance(lanes, list) or not lanes:
            raise capture.UnavailableBinding("missing scoped terminal-lane catalog")
        closed = [(lane["mission_id"], _closed_lane(lane)) for lane in lanes]
        comparisons = []
        for mission, identity in closed:
            changed = sorted(field for field in FIELDS
                             if capture.encoded(candidate[field]) != capture.encoded(identity[field]))
            comparisons.append({"mission_id": mission, "changed_fields": changed,
                                "input_content_changed": _input_content(candidate) != _input_content(identity)})
        result["comparisons"] = comparisons
        for comparison in comparisons:
            changed = set(comparison["changed_fields"])
            if not changed or changed <= {"diagnostic"}:
                result.update(reason="closed_equivalent", closed_mission=comparison["mission_id"])
                return result
            if changed & {"references", "boundary"}:
                result.update(reason="fixed_reference_or_boundary_mutation",
                              closed_mission=comparison["mission_id"])
                return result
            if (not {"observation", "inputs", "permitted_invocations"} <= changed
                    or not comparison["input_content_changed"]):
                result.update(reason="no_material_novelty", closed_mission=comparison["mission_id"])
                return result
        result.update(status="ACCEPTED_NOVEL", reason="material_identity_difference_only")
    except capture.UnavailableBinding as error:
        result.update(status="UNKNOWN", unavailable_binding=str(error))
    except (RuntimeError, KeyError, TypeError, ValueError) as error:
        result.update(reason=f"invalid_identity_or_binding: {error}")
    return result
