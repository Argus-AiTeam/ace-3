"""Non-generative schema-1 custody for future diagnostic lanes.

emit_custody(origins, launch, attempt, reservation) accepts independently trusted
byte pins, not task names, live-source discovery, or narrative field selectors.
origins has exactly two pins:

* proposal: schema_version=1, kind="diagnostic_identity_declaration", explicit
  diagnostic_identity, source_roles, source_snapshots and complete launch.
* terminals: schema_version=1, kind="diagnostic_terminal_declarations", ordered
  unique scoped_missions and lanes. Each lane explicitly supplies mission_id,
  latest/review/checkpoint pins, native, capture and source_snapshots, using the
  shared identity gate's shapes. No terminal_catalog field is required here.

Creation-time declarations supply the proposal. Review-time declarations supply
terminal receipt/native/review bindings. Embedded identities remain strict.
Current legacy declarations instead require a separately supplied Manager
issuance and independent current review, with exact field provenance checked by
the shared closed-lane reader. No historical stdout or checkpoint is rewritten.
Historical/current source roles retain the shared snapshot separation rules.

Outputs are diagnostic_proposal.json and terminal_catalog.json, directly usable
by diagnostic_identity_preflight_v1, plus custody-receipt.json. All inputs must
be outside the caller-reserved fresh ignored build attempt. Files are exclusive;
UNKNOWN/REJECTED retains a receipt without publishing either authority input.
validate_custody is read-only and requires independently supplied origins and
the original creation reservation, not trust in the receipt's own pins.
authority_bundle_origins validates that custody before exposing field selectors
for the shared authority-bundle interface. Call it again with the independent
declaration pins when reviewing a downstream bundle; emitted pins alone are
never a replacement for those roots.

PASS establishes custody only, NOT novelty, execution permission, independent
review, native completion, or admission. Caller-established origin trust and
scope completeness remain external requirements; this emitter cannot mint them.
authenticate_manager_origins is a read-only prerequisite check of an independently
pinned issuance graph. validate_manager_declarations additionally reads closed
identities and provenance for independent review, without emitting custody or
making a novelty decision. A mismatched generation origin must
be resolved by the authority owner, never repaired by relocating an original pin.
It neither imports producers nor dispatches commands or services. Existing
threshold/reference/source/operand/state/KV/lineage, account/model/access/budget
gates and Q24-wide native-S16-RTZ/INT4/FP16 boundaries are unchanged.
"""

from datetime import datetime, timezone
from pathlib import Path

from ace3.model.candidates import diagnostic_capture_v1 as capture
from ace3.model.candidates import diagnostic_identity_authority_bundle_v1 as bundle
from ace3.model.candidates import diagnostic_identity_preflight_v1 as gate


MEMBERS = ("diagnostic_proposal.json", "terminal_catalog.json", "custody-receipt.json")
LANE_FIELDS = frozenset({
    "mission_id", "latest", "review", "checkpoint", "native", "capture", "source_snapshots",
})
CURRENT_LANE_FIELDS = LANE_FIELDS | {
    "declaration_mode", "diagnostic_identity", "identity_provenance",
    "legacy_diagnostic_identity_embedded",
}
LAUNCH_FIELDS = frozenset({
    "argv", "command", "environment", "cwd", "uid", "role", "model", "access", "budget",
})


def _declaration(pin, kind, fields):
    document = capture.retained_document(pin)
    if (not isinstance(document, dict) or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1 or document.get("kind") != kind):
        raise RuntimeError(f"explicit schema-1 {kind} origin required; outputs are not origins")
    for field in sorted(fields):
        gate._at(document, [field], pin["path"])
    if set(document) != {"schema_version", "kind", *fields}:
        raise RuntimeError(f"unexpected {kind} declaration fields")
    return document


def authenticate_manager_origins(issuance_pin, proposal_id, *, now=None):
    """Authenticate issuance roots without custody emission or identity preflight.

    The caller supplies the trusted Manager pin independently. This deliberately
    stops at origin authentication: a returned graph is not a validated catalog.
    """
    issuance = capture.retained_document(issuance_pin)
    if (not isinstance(issuance, dict)
            or type(issuance.get("schema_version")) is not int
            or issuance["schema_version"] != 1
            or issuance.get("kind") != "manager_diagnostic_identity_issuance"
            or issuance.get("issuer") != "manager"
            or issuance.get("proposal_id") != proposal_id
            or issuance.get("authority_root") is not True
            or issuance.get("authorized_science") is not False):
        raise RuntimeError("independently pinned non-generative Manager issuance required")
    issued = datetime.fromisoformat(issuance["issued_at_utc"])
    current = datetime.now(timezone.utc) if now is None else now
    if (issued.tzinfo is None or current.tzinfo is None
            or issued.utcoffset().total_seconds() != 0 or issued > current):
        raise RuntimeError("current UTC Manager issuance time required")
    if (not isinstance(issuance.get("zero_actions"), dict) or not issuance["zero_actions"]
            or any(type(count) is not int or count != 0
                   for count in issuance["zero_actions"].values())):
        raise RuntimeError("non-generative issuance action counters required")
    origins = capture.retained_document(issuance["origins_manifest"])
    if origins != {"proposal": issuance["proposal_declaration"],
                   "terminals": issuance["terminal_declarations"]}:
        raise RuntimeError("Manager issuance/origins declaration pins mismatch")
    root_paths = [issuance_pin["path"], issuance["origins_manifest"]["path"],
                  origins["proposal"]["path"], origins["terminals"]["path"]]
    if len(set(root_paths)) != len(root_paths):
        raise RuntimeError("separate Manager issuance/origin/declaration roots required")
    original = _declaration(
        origins["proposal"], "diagnostic_identity_declaration",
        {"diagnostic_identity", "source_roles", "source_snapshots", "launch"})
    terminals = _declaration(
        origins["terminals"], "diagnostic_terminal_declarations", {"scoped_missions", "lanes"})
    scope = issuance["terminal_scope"]
    lanes = terminals["lanes"]
    if (not isinstance(scope, list) or not scope
            or not all(isinstance(mission, str) and mission for mission in scope)
            or len(set(scope)) != len(scope)
            or terminals["scoped_missions"] != scope
            or not isinstance(lanes, list)
            or not all(isinstance(lane, dict) and set(lane) in (LANE_FIELDS, CURRENT_LANE_FIELDS)
                       for lane in lanes)
            or [lane.get("mission_id") for lane in lanes] != scope):
        raise RuntimeError("Manager issuance/terminal scope census mismatch")
    launch = capture.retained_document(issuance["actual_launch"])
    if not isinstance(original["launch"], dict) or not LAUNCH_FIELDS <= set(original["launch"]):
        raise RuntimeError("complete declared launch/account/role/model/access/budget required")
    capture.verify_launch_identity(launch, original["launch"])
    gate._identity(original["diagnostic_identity"], original["source_snapshots"])
    capture.verify_source_roles(original["source_roles"])
    if (original["source_roles"]["current_diagnostic"]["source"]
            not in original["diagnostic_identity"]["sources"].values()):
        raise RuntimeError("current diagnostic source role is not a declared identity source")
    for lane in lanes:
        run = lane["capture"]
        gate._retained_run(run)
        review = capture.retained_document(lane["review"])
        if (type(review.get("created_at")) not in (int, float)
                or not 0 <= review["created_at"] <= issued.timestamp()):
            raise RuntimeError("Manager issuance predates retained review")
    gate.verify_generation_origin(original["source_roles"], [lane["capture"] for lane in lanes])
    return {"issuance": issuance, "origins": origins, "proposal": original,
            "terminals": terminals, "launch": launch, "issued_timestamp": issued.timestamp()}


def validate_manager_declarations(issuance_pin, proposal_id):
    """Read-only prerequisite for normal independent review, never consumption."""
    graph = authenticate_manager_origins(issuance_pin, proposal_id)
    issuance, original = graph["issuance"], graph["proposal"]
    for key in ("candidate_source", "candidate_test", "contract", "runtime_sources"):
        pin = issuance[key]
        capture._retained_bytes(pin, capture._validate_pin(pin))
    contract = capture.retained_document(issuance["contract"])
    for field in ("inputs", "references", "sources"):
        gate._same(contract[field], original["diagnostic_identity"][field], f"proposal/contract {field}")
    gate._same(issuance["candidate_source"], contract["sources"]["current_diagnostic"], "candidate source")
    gate._same(issuance["candidate_test"], contract["sources"]["current_tests"], "candidate tests")
    gate._same(original["source_roles"]["current_diagnostic"]["source"],
               issuance["candidate_source"], "current candidate source role")
    runtime = capture.retained_document(issuance["runtime_sources"])
    if not isinstance(runtime, dict) or not runtime:
        raise RuntimeError("complete runtime source map required")
    for pin in runtime.values():
        capture.verify_source_snapshot(pin, pin)
    members = issuance["replacement_members"]
    for key, member in (("stage11", "stage11"), ("stage13_comparator", "stage13")):
        gate.verify_stage_member({
            "source_archive": members["source_archive"], "source_member": member,
            "derived_pin": members[key], "encoding": members["encoding"]}, member)
    gate._same(contract["inputs"]["replacement_stage11"], members["stage11"], "proposal replacement pin")
    gate._same(contract["references"]["original_input_L23_fp16"], members["source_archive"],
               "proposal original-input archive")
    gate._same(issuance["legacy_disclosure"], {
        "declarations_existed_in_952602cea3cc": False,
        "declarations_existed_in_e72f25703250": False,
        "declarations_existed_in_fddc2074be92": False,
        "legacy_stdout_embedded_schema1_identity": False, "old_artifacts_modified": False},
        "present-time disclosure")
    authority = {"issuance": issuance_pin, "proposal_id": proposal_id}
    identities = {
        lane["mission_id"]: gate._closed_lane(
            lane, manager_authority=authority, review_pending=True)
        for lane in graph["terminals"]["lanes"]}
    for lane in graph["terminals"]["lanes"]:
        gate._same(lane["diagnostic_identity"]["inputs"]["replacement_stage13"],
                   members["stage13_comparator"], "terminal replacement pin")
        payload = capture.retained_document(lane["identity_provenance"]["boundary"]["pin"])
        gate._same(contract["thresholds"], payload["original_thresholds"], "unchanged thresholds")
        gate._same(contract["historical_failures"], payload["retained_controls_and_failure_gates"],
                   "retained ordered failure census")
        gate._same(contract["lineage"], payload["retained_lineage_separation"], "retained lineage")
    return {"issuance": issuance_pin, "origins": graph["origins"],
            "scoped_missions": graph["terminals"]["scoped_missions"],
            "closed_identities": identities, "source_roles": graph["proposal"]["source_roles"],
            "normal_independent_review": "REQUIRED", "dispatch_authorized": False,
            "identity_preflight_invoked": False}


def _documents(origins, launch, attempt, manager_authority=None):
    if not isinstance(origins, dict) or set(origins) != {"proposal", "terminals"}:
        raise RuntimeError("independent proposal/terminal declaration pins required")
    bundle._outside_inputs(attempt, origins, launch)
    original = _declaration(
        origins["proposal"], "diagnostic_identity_declaration",
        {"diagnostic_identity", "source_roles", "source_snapshots", "launch"})
    terminals = _declaration(
        origins["terminals"], "diagnostic_terminal_declarations", {"scoped_missions", "lanes"})
    bundle._outside_inputs(attempt, original, terminals)
    proposal = {"schema_version": 1, "identity": original["diagnostic_identity"],
                **{field: original[field] for field in
                   ("source_roles", "source_snapshots", "launch")}}
    expected = proposal["launch"]
    if not isinstance(expected, dict) or not LAUNCH_FIELDS <= set(expected):
        raise RuntimeError("complete declared launch/account/role/model/access/budget required")
    canonical_launch = capture.verify_launch_identity(launch, expected)
    capture.verify_source_roles(proposal["source_roles"])
    gate._identity(proposal["identity"], proposal["source_snapshots"])
    roles = proposal["source_roles"]
    if roles["current_diagnostic"]["source"] not in proposal["identity"]["sources"].values():
        raise RuntimeError("current diagnostic source role is not a declared identity source")
    scope, declarations = terminals["scoped_missions"], terminals["lanes"]
    if (not isinstance(scope, list) or not scope
            or not all(isinstance(mission, str) and mission for mission in scope)
            or len(set(scope)) != len(scope)):
        raise RuntimeError("unique nonempty explicit terminal scope required")
    if (not isinstance(declarations, list)
            or not all(isinstance(lane, dict) and set(lane) in (LANE_FIELDS, CURRENT_LANE_FIELDS)
                       for lane in declarations)
            or [lane["mission_id"] for lane in declarations] != scope):
        raise RuntimeError("terminal declarations/scope census mismatch")
    lanes = []
    for declaration in declarations:
        run = declaration["capture"]
        directory = capture._canonical_path(run["directory"])
        if directory.is_relative_to(attempt):
            raise RuntimeError("terminal capture conflated with output attempt")
        if declaration.get("declaration_mode") == gate.CURRENT_DECLARATION:
            if manager_authority is None:
                raise capture.UnavailableBinding("independently reviewed Manager authority required")
            graph = authenticate_manager_origins(
                manager_authority["issuance"], manager_authority["proposal_id"])
            gate._same(graph["origins"], origins, "custody Manager origins")
            bundle._outside_inputs(attempt, manager_authority)
            lane = {**declaration, "current_declaration": manager_authority}
        else:
            lane = {**declaration, "identity_paths": {
                field: ["diagnostic_identity", field] for field in sorted(gate.FIELDS)}}
        gate._closed_lane(lane)
        lanes.append(lane)
    gate.verify_generation_origin(roles, [lane["capture"] for lane in declarations])
    return proposal, {"schema_version": 1, "scoped_missions": scope, "lanes": lanes}, canonical_launch


def _result():
    return {"schema_version": 1, "status": "REJECTED", "dispatch_authorized": False,
            "scientific_invocations": 0, "producer_invocations": 0, "service_invocations": 0,
            "authority_root": False, "scientific_or_admission_claim": False,
            "normal_independent_review": "REQUIRED"}


def _prepare(origins, launch, attempt, reservation, manager_authority=None):
    if not isinstance(reservation, dict) or reservation.get("path") != str(attempt):
        raise RuntimeError("custody reservation/attempt mismatch")
    audit = {"attempt": str(attempt), "writes": [str(attempt / name) for name in MEMBERS]}
    confinement = capture.verify_write_audit(audit, reservation)
    result = {**_result(), "origins": origins, "reservation": reservation,
              "write_audit": audit, "write_confinement": confinement}
    if manager_authority is not None:
        result["manager_authority"] = manager_authority
    documents = None
    try:
        proposal, catalog, canonical_launch = _documents(origins, launch, attempt, manager_authority)
        result.update(status="PASS", launch=canonical_launch, source_roles=proposal["source_roles"],
                      scoped_missions=catalog["scoped_missions"])
        documents = (proposal, catalog)
    except capture.UnavailableBinding as error:
        result.update(status="UNKNOWN", unavailable_binding=str(error))
    except (RuntimeError, KeyError, TypeError, ValueError) as error:
        result.update(reason=f"invalid_diagnostic_custody: {error}")
    return result, documents


def emit_custody(origins, launch, attempt, reservation, *, manager_authority=None):
    """Publish only after all declarations, capture members and review links pass."""
    attempt = capture._canonical_path(str(attempt))
    if any((attempt / name).exists() or (attempt / name).is_symlink() for name in MEMBERS):
        raise RuntimeError("custody outputs already exist")
    result, documents = _prepare(origins, launch, attempt, reservation, manager_authority)
    if documents is not None:
        for key, name, document in zip(("proposal", "catalog"), MEMBERS[:2], documents, strict=True):
            result[key] = capture.save(attempt, name, capture.encoded(document))
    result["receipt"] = capture.save(attempt, MEMBERS[2], capture.encoded(result))
    return result


def validate_custody(receipt_pin, trusted_origins, launch, *, reservation=None, manager_authority=None):
    """Re-derive from independent byte roots, not a self-pinned emitted manifest."""
    result = _result()
    try:
        receipt = capture.retained_document(receipt_pin)
        attempt = Path(receipt_pin["path"]).parent
        if (not isinstance(receipt, dict)
                or capture.encoded(receipt.get("origins")) != capture.encoded(trusted_origins)
                or capture.encoded(receipt.get("reservation")) != capture.encoded(reservation)
                or capture.encoded(receipt.get("manager_authority")) != capture.encoded(manager_authority)):
            raise RuntimeError("custody independent origins/reservation mismatch")
        expected, documents = _prepare(trusted_origins, launch, attempt, reservation, manager_authority)
        if documents is not None:
            for key, name, document in zip(("proposal", "catalog"), MEMBERS[:2], documents, strict=True):
                if capture._retained_bytes(receipt[key], attempt / name) != capture.encoded(document):
                    raise RuntimeError(f"custody {key} differs from exact declarations/captures")
                expected[key] = receipt[key]
        elif any((attempt / name).exists() for name in MEMBERS[:2]):
            raise RuntimeError("unavailable/rejected custody published authority inputs")
        if capture.encoded(receipt) != capture.encoded(expected):
            raise RuntimeError("custody receipt/decision/write-audit/counter mismatch")
        result = expected
    except capture.UnavailableBinding as error:
        result.update(status="UNKNOWN", unavailable_binding=str(error))
    except (RuntimeError, KeyError, TypeError, ValueError) as error:
        result.update(status="REJECTED", reason=f"invalid_diagnostic_custody: {error}")
    return result


def authority_bundle_origins(receipt_pin, trusted_origins, launch, *, reservation=None, manager_authority=None):
    """Derive bundle selectors only after authenticating independent custody roots."""
    checked = validate_custody(receipt_pin, trusted_origins, launch, reservation=reservation,
                              manager_authority=manager_authority)
    if checked["status"] == "UNKNOWN":
        raise capture.UnavailableBinding(checked["unavailable_binding"])
    if checked["status"] != "PASS":
        raise RuntimeError(f"authority bundle requires valid custody: {checked.get('reason')}")
    return {
        "proposal": {"pin": checked["proposal"],
                     "paths": {field: [field] for field in bundle.PROPOSAL_FIELDS}},
        "catalog": {"pin": checked["catalog"], "lanes_path": ["lanes"],
                    "scope_path": ["scoped_missions"]},
    }
