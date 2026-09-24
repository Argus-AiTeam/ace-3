"""Materialize schema-1 identity inputs from explicitly trusted retained originals.

``origins`` has two entries. ``proposal`` supplies an exact ``pin`` and ``paths``
mapping identity/source_roles/source_snapshots/launch to nonempty JSON paths.
``catalog`` supplies an exact ``pin``, ``lanes_path`` and ``scope_path``; the
selected values must be the complete lane descriptors and scoped mission IDs.
These fields must already exist in the originals. No narrative parsing, source
execution, historical reconstruction or hash-only substitution is supported.

The caller, not this module, establishes independent trust in these original
pins and the completeness of their scope. A newly written manifest, output pin,
or this module's PASS is NOT an authority root. Reviewers must pass their own
trusted origins and creation reservation to validate_bundle; it re-derives the
documents before calling the shared identity gate. PASS means ACCEPTED_NOVEL only at that gate, never
execution, model, access, account, concurrency or budget authorization.

materialize writes exclusively into a caller-reserved ignored build attempt.
It retains a receipt even for UNKNOWN/REJECTED, without emitting invented
proposal/catalog documents. validate_bundle is read-only.
"""

from pathlib import Path

from ace3.model.candidates import diagnostic_capture_v1 as capture
from ace3.model.candidates import diagnostic_identity_preflight_v1 as gate


PROPOSAL_FIELDS = ("identity", "source_roles", "source_snapshots", "launch")
MEMBERS = ("proposal.json", "terminal-catalog.json", "authority-bundle.json")


def _pins(value):
    if isinstance(value, dict):
        if set(value) == {"path", "bytes", "sha256"}:
            yield value
        else:
            for item in value.values():
                yield from _pins(item)
    elif isinstance(value, list):
        for item in value:
            yield from _pins(item)


def _documents(origins):
    if not isinstance(origins, dict) or set(origins) != {"proposal", "catalog"}:
        raise RuntimeError("explicit trusted proposal/catalog origins required")
    proposal_spec, catalog_spec = origins["proposal"], origins["catalog"]
    if (not isinstance(proposal_spec, dict) or set(proposal_spec) != {"pin", "paths"}
            or not isinstance(catalog_spec, dict)
            or set(catalog_spec) != {"pin", "lanes_path", "scope_path"}):
        raise RuntimeError("exact original pins and retained field paths required")
    catalog_original = capture.retained_document(catalog_spec["pin"])
    lanes = gate._at(catalog_original, catalog_spec["lanes_path"], catalog_spec["pin"]["path"])
    scope = gate._at(catalog_original, catalog_spec["scope_path"], catalog_spec["pin"]["path"])
    if (not isinstance(scope, list) or not scope
            or not all(isinstance(mission, str) and mission for mission in scope)
            or len(set(scope)) != len(scope)):
        raise RuntimeError("complete unique retained terminal scope required")
    if (not isinstance(lanes, list) or not all(isinstance(lane, dict) for lane in lanes)
            or [lane.get("mission_id") for lane in lanes] != scope):
        raise RuntimeError("terminal catalog/scope census mismatch")
    paths = proposal_spec["paths"]
    if not isinstance(paths, dict) or set(paths) != set(PROPOSAL_FIELDS):
        raise RuntimeError("complete proposal field paths required")
    original = capture.retained_document(proposal_spec["pin"])
    proposal = {"schema_version": 1, **{
        field: gate._at(original, paths[field], proposal_spec["pin"]["path"])
        for field in PROPOSAL_FIELDS}}
    capture.verify_source_roles(proposal["source_roles"])
    gate._identity(proposal["identity"], proposal["source_snapshots"])
    roles = proposal["source_roles"]
    if roles["current_diagnostic"]["source"] not in proposal["identity"]["sources"].values():
        raise RuntimeError("current diagnostic role is not a proposed identity source")
    generation_sources = []
    for lane in lanes:
        preflight = lane["capture"]["preflight"]
        generation_sources.extend(preflight["sources"])
        generation_sources.extend(preflight.get("capture_implementation", []))
    if roles["historical_generation"]["source"] not in generation_sources:
        raise RuntimeError("historical source role is not bound to a retained generation")
    return proposal, {"schema_version": 1, "lanes": lanes}


def _outside_inputs(attempt, *values):
    for value in values:
        for pin in _pins(value):
            path = capture._validate_pin(pin)
            if path.is_relative_to(attempt):
                raise RuntimeError(f"authority/input/source conflated with output attempt: {path}")


def _result():
    return {"schema_version": 1, "status": "REJECTED", "dispatch_authorized": False,
            "scientific_invocations": 0, "producer_invocations": 0,
            "normal_independent_review": "REQUIRED"}


def _decision(result, proposal_pin, catalog_pin, launch):
    decision = gate.preflight_identity(proposal_pin, catalog_pin, launch)
    result["identity_preflight"] = decision
    result["status"] = "PASS" if decision["status"] == "ACCEPTED_NOVEL" else decision["status"]
    if decision["status"] == "UNKNOWN":
        result["unavailable_binding"] = decision["unavailable_binding"]
    elif decision["status"] == "REJECTED":
        result["reason"] = decision["reason"]


def materialize(origins, launch, attempt, reservation):
    """Consume externally trusted origins; never infer trust from emitted pins."""
    attempt = capture._canonical_path(str(attempt))
    if reservation.get("path") != str(attempt):
        raise RuntimeError("bundle reservation/attempt mismatch")
    audit = {"attempt": str(attempt), "writes": [str(attempt / name) for name in MEMBERS]}
    confinement = capture.verify_write_audit(audit, reservation)
    if any((attempt / name).exists() or (attempt / name).is_symlink() for name in MEMBERS):
        raise RuntimeError("bundle outputs already exist")
    result = {**_result(), "origins": origins, "reservation": reservation, "write_audit": audit,
              "write_confinement": confinement}
    try:
        _outside_inputs(attempt, origins, launch)
        proposal, catalog = _documents(origins)
        _outside_inputs(attempt, proposal, catalog)
        proposal_pin = capture.save(attempt, MEMBERS[0], capture.encoded(proposal))
        catalog_pin = capture.save(attempt, MEMBERS[1], capture.encoded(catalog))
        result.update(proposal=proposal_pin, catalog=catalog_pin,
                      source_roles=proposal["source_roles"])
        _decision(result, proposal_pin, catalog_pin, launch)
    except capture.UnavailableBinding as error:
        result.update(status="UNKNOWN", unavailable_binding=str(error))
    except (RuntimeError, KeyError, TypeError, ValueError) as error:
        result.update(reason=f"invalid_authority_bundle: {error}")
    result["receipt"] = capture.save(attempt, MEMBERS[2], capture.encoded(result))
    return result


def validate_bundle(bundle_pin, trusted_origins, launch, *, reservation=None):
    """Authenticate outputs against independent origins, never their self-pins."""
    result = _result()
    try:
        bundle = capture.retained_document(bundle_pin)
        if (not isinstance(bundle, dict) or type(bundle.get("schema_version")) is not int
                or bundle["schema_version"] != 1
                or capture.encoded(bundle.get("origins")) != capture.encoded(trusted_origins)):
            raise RuntimeError("bundle/independent origin mismatch")
        attempt = Path(bundle_pin["path"]).parent
        expected_audit = {"attempt": str(attempt),
                          "writes": [str(attempt / name) for name in MEMBERS]}
        if (bundle.get("write_audit") != expected_audit
                or bundle.get("reservation") != reservation
                or capture.verify_write_audit(expected_audit, reservation)
                != bundle.get("write_confinement")):
            raise RuntimeError("bundle write-audit/reservation mismatch")
        _outside_inputs(attempt, trusted_origins, launch)
        proposal, catalog = _documents(trusted_origins)
        _outside_inputs(attempt, proposal, catalog)
        for key, name, document in (("proposal", MEMBERS[0], proposal),
                                    ("catalog", MEMBERS[1], catalog)):
            data = capture._retained_bytes(bundle[key], attempt / name)
            if data != capture.encoded(document):
                raise RuntimeError(f"{key} differs from retained original fields")
        if bundle.get("source_roles") != proposal["source_roles"]:
            raise RuntimeError("bundle source-role receipt mismatch")
        _decision(result, bundle["proposal"], bundle["catalog"], launch)
        if (bundle.get("status") != result["status"]
                or bundle.get("identity_preflight") != result["identity_preflight"]
                or any(capture.encoded(bundle.get(key)) != capture.encoded(result[key]) for key in
                       ("dispatch_authorized", "scientific_invocations", "producer_invocations"))):
            raise RuntimeError("bundle decision/counter receipt mismatch")
    except capture.UnavailableBinding as error:
        result.update(status="UNKNOWN", unavailable_binding=str(error))
    except (RuntimeError, KeyError, TypeError, ValueError) as error:
        result.update(status="REJECTED", reason=f"invalid_authority_bundle: {error}")
    return result
