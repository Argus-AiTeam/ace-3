"""Non-generative bundle tests and a single guarded retained validation.

PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B \
tests/test_diagnostic_identity_authority_bundle_v1.py --validate FRESH_BUILD_ATTEMPT

The shared validation runner compiles all scoped sources, executes both focused
suites once, then authenticates retained bytes without invoking their commands.
Real schema-3 narratives remain UNKNOWN if an original catalog/identity is
absent. Synthetic fixtures establish complete schema-1 behavior, not real-lane
authority or independent review.
"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import pytest

from ace3.model.candidates import diagnostic_capture_v1 as capture
from ace3.model.candidates import diagnostic_identity_authority_bundle_v1 as bundle
from ace3.model.candidates import diagnostic_identity_preflight_v1 as gate


prior = sys.modules.get("test_diagnostic_identity_preflight_v1")
if prior is None:
    spec = importlib.util.spec_from_file_location(
        "test_diagnostic_identity_preflight_v1",
        Path(__file__).with_name("test_diagnostic_identity_preflight_v1.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("shared identity test/validation module is unavailable")
    prior = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = prior
    spec.loader.exec_module(prior)


@pytest.fixture
def case(tmp_path):
    def make(directory):
        fixture = prior.identity_fixture(directory)
        source = fixture[1]["identity"]["sources"]["operator"]
        snapshot = prior.put(directory, "operator-snapshot.py", Path(source["path"]).read_bytes())
        fixture[1]["source_roles"] = {
            "historical_generation": {"source": source, "retained": snapshot},
            "current_diagnostic": {"source": source, "retained": source}}
        prior.novel(fixture)
        return fixture

    if tmp_path.is_relative_to(capture.ROOT / "build"):
        yield make(tmp_path)
    else:
        with tempfile.TemporaryDirectory(dir=capture.ROOT / "build", prefix="bundle-unit-") as path:
            yield make(Path(path))


def origins_for(case):
    directory, proposal, catalog, _ = case
    original = prior.put(directory, "proposal-original.json", {"retained": proposal})
    catalog_original = prior.put(directory, "catalog-original.json", {
        "scoped_missions": [lane["mission_id"] for lane in catalog["lanes"]],
        "terminal_lanes": catalog["lanes"]})
    return {
        "proposal": {"pin": original,
                     "paths": {field: ["retained", field] for field in bundle.PROPOSAL_FIELDS}},
        "catalog": {"pin": catalog_original, "lanes_path": ["terminal_lanes"],
                    "scope_path": ["scoped_missions"]}}


def produce(case, origins=None):
    if origins is None:
        origins = origins_for(case)
    attempt = case[0] / "bundle"
    reservation = capture.reserve_authentication_attempt(attempt)
    return bundle.materialize(origins, case[3], attempt, reservation), origins


def assert_no_dispatch(result):
    assert result["dispatch_authorized"] is False
    assert result["scientific_invocations"] == result["producer_invocations"] == 0


def test_complete_materialization_and_independent_validation(case):
    result, origins = produce(case)
    assert result["status"] == "PASS"
    assert_no_dispatch(result)
    assert capture.retained_document(result["proposal"])["schema_version"] == 1
    assert capture.retained_document(result["catalog"])["schema_version"] == 1
    checked = bundle.validate_bundle(result["receipt"], origins, case[3],
                                     reservation=result["reservation"])
    assert checked["status"] == "PASS"
    assert_no_dispatch(checked)
    assert gate.preflight_identity(result["proposal"], result["catalog"], case[3])[
        "status"] == "ACCEPTED_NOVEL"


@pytest.mark.parametrize("mutation", ["value", "argv", "duplicate", "extra_env", "uid",
                                    "cwd", "role", "budget", "model", "access"])
def test_launch_changes_rejected(case, mutation):
    origins = origins_for(case)
    launch = case[3]
    if mutation == "value":
        launch["environment"]["B"] = "different"
    elif mutation == "argv":
        launch["argv"].append("--different")
    elif mutation == "duplicate":
        launch["command"] = "A=one " + launch["command"]
    elif mutation == "extra_env":
        launch["environment"]["C"] = "extra"
    else:
        launch[mutation] = "changed"
    if mutation in {"value", "argv", "extra_env"}:
        launch["command"] = capture.environment_command(launch["environment"], launch["argv"])
    result, _ = produce(case, origins)
    assert result["status"] == "REJECTED"
    assert_no_dispatch(result)


def test_environment_order_only_is_invariant(case):
    origins = origins_for(case)
    launch = case[3]
    launch["environment"] = dict(reversed(list(launch["environment"].items())))
    launch["command"] = capture.environment_command(launch["environment"], launch["argv"])
    result, _ = produce(case, origins)
    assert result["status"] == "PASS"
    assert bundle.validate_bundle(result["receipt"], origins, launch,
                                  reservation=result["reservation"])["status"] == "PASS"


@pytest.mark.parametrize("field", bundle.PROPOSAL_FIELDS)
def test_missing_original_field_is_one_unknown(case, field):
    del case[1][field]
    result, _ = produce(case)
    assert result["status"] == "UNKNOWN"
    assert f"['retained', '{field}']" in result["unavailable_binding"]
    assert not (case[0] / "bundle/proposal.json").exists()
    assert_no_dispatch(result)


@pytest.mark.parametrize("field", ["path", "bytes", "sha256"])
def test_exact_original_pins(case, field):
    origins = origins_for(case)
    pin = origins["proposal"]["pin"]
    pin[field] = (str(case[0] / "missing-original") if field == "path" else
                  pin[field] + 1 if field == "bytes" else "0" * 64)
    result, _ = produce(case, origins)
    assert result["status"] == "UNKNOWN"
    assert pin["path"] in result["unavailable_binding"]


@pytest.mark.parametrize("mutation", ["missing", "conflated", "unlinked_historical", "unlinked_current"])
def test_source_role_separation(case, mutation):
    roles = case[1]["source_roles"]
    if mutation == "missing":
        del roles["historical_generation"]
    elif mutation == "conflated":
        roles["historical_generation"] = deepcopy(roles["current_diagnostic"])
    else:
        other = prior.put(case[0], "unrelated.py", b"unrelated bytes")
        if mutation == "unlinked_historical":
            snapshot = prior.put(case[0], "unrelated-snapshot.py", b"unrelated bytes")
            roles["historical_generation"] = {"source": other, "retained": snapshot}
        else:
            roles["current_diagnostic"] = {"source": other, "retained": other}
    assert produce(case)[0]["status"] == "REJECTED"


def test_closed_equivalent_never_reopens(case):
    lane = case[2]["lanes"][0]
    receipt = capture.retained_document(lane["capture"]["receipt"])
    original = capture.retained_document(receipt["results"][0]["files"][3])["diagnostic_identity"]
    original["diagnostic"] = "packaging-only-rename"
    case[1]["identity"] = original
    result, _ = produce(case)
    assert result["status"] == "REJECTED"
    assert result["reason"] == "closed_equivalent"


@pytest.mark.parametrize("mutation", ["missing_scope", "duplicate", "omitted_lane", "hash_only",
                                    "narrative", "missing_native", "missing_identity_path"])
def test_catalog_completeness(case, mutation):
    if mutation == "missing_native":
        Path(case[2]["lanes"][0]["native"]["pin"]["path"]).unlink()
    elif mutation == "missing_identity_path":
        del case[2]["lanes"][0]["identity_paths"]["observation"]
    origins = origins_for(case)
    catalog = capture.retained_document(origins["catalog"]["pin"])
    if mutation == "missing_scope":
        del catalog["scoped_missions"]
    elif mutation == "duplicate":
        catalog["scoped_missions"] *= 2
        catalog["terminal_lanes"] *= 2
    elif mutation == "omitted_lane":
        catalog["terminal_lanes"] = []
    elif mutation == "hash_only":
        del catalog["terminal_lanes"][0]["native"]["pin"]["bytes"]
    elif mutation == "narrative":
        catalog["terminal_lanes"] = "this task was reviewed and completed"
    origins["catalog"]["pin"] = prior.put(case[0], "catalog-mutant.json", catalog)
    result, _ = produce(case, origins)
    assert result["status"] in {"UNKNOWN", "REJECTED"}
    assert_no_dispatch(result)


def test_duplicate_original_assignments_rejected(case):
    origins = origins_for(case)
    origins["proposal"]["pin"] = prior.put(
        case[0], "duplicate.json", b'{"retained": {}, "retained": {}}')
    result, _ = produce(case, origins)
    assert result["status"] == "REJECTED" and "duplicate" in result["reason"]


@pytest.mark.parametrize("mutation", ["origin", "proposal", "catalog", "audit", "counter", "role"])
def test_self_pinned_receipt_cannot_replace_authority(case, mutation):
    result, origins = produce(case)
    receipt = capture.retained_document(result["receipt"])
    if mutation == "origin":
        receipt["origins"]["proposal"]["paths"]["identity"] = ["other"]
    elif mutation in {"proposal", "catalog"}:
        document = capture.retained_document(receipt[mutation])
        document["schema_version"] = 3
        receipt[mutation] = prior.put(case[0], f"forged-{mutation}.json", document)
    elif mutation == "audit":
        receipt["write_audit"]["writes"].append(str(case[0] / "outside"))
    elif mutation == "counter":
        receipt["producer_invocations"] = 1
    else:
        receipt["source_roles"] = {}
    pin = prior.put(case[0] / "bundle", "forged-receipt.json", receipt)
    assert bundle.validate_bundle(pin, origins, case[3],
                                  reservation=result["reservation"])["status"] == "REJECTED"


def test_validator_requires_independent_reservation(case):
    result, origins = produce(case)
    assert bundle.validate_bundle(result["receipt"], origins, case[3])["status"] == "REJECTED"


def test_output_cannot_stand_in_for_source_or_original(case):
    origins = origins_for(case)
    origins["proposal"]["pin"]["path"] = str(case[0] / "bundle/proposal.json")
    result, _ = produce(case, origins)
    assert result["status"] == "REJECTED" and "conflated" in result["reason"]


def test_outside_write_and_existing_output_rejected(case):
    attempt = case[0] / "bundle"
    reservation = capture.reserve_authentication_attempt(attempt)
    with pytest.raises(RuntimeError, match="outside-attempt"):
        capture.verify_write_audit(
            {"attempt": str(attempt), "writes": [str(case[0] / "outside")]}, reservation)
    prior.put(attempt, "proposal.json", b"preserve existing bytes")
    with pytest.raises(RuntimeError, match="already exist"):
        bundle.materialize(origins_for(case), case[3], attempt, reservation)
    assert (attempt / "proposal.json").read_bytes() == b"preserve existing bytes"


@pytest.mark.parametrize("schema", [True, 3, "1", None])
def test_preflight_schema_requires_exact_version(case, schema):
    case[1]["schema_version"] = schema
    assert prior.decide(case)["status"] == "REJECTED"


def retained_validation(attempt):
    positive = prior._retained_capture_validation(attempt)
    prior.put(attempt, "source-role.receipt.json", positive["source_roles"])
    # The actual latest Reviewer bytes are evidence, not an invented catalog.
    review_pin = positive["review"]["review"]
    original = capture.retained_document(review_pin)
    mission = Path(original["mission_context"])
    origins = {
        "proposal": {"pin": capture.binding(mission),
                     "paths": {field: ["diagnostic_proposal", field]
                               for field in bundle.PROPOSAL_FIELDS}},
        "catalog": {"pin": review_pin, "lanes_path": ["terminal_catalog", "lanes"],
                    "scope_path": ["terminal_catalog", "scoped_missions"]}}
    prior.put(attempt, "proposal-origin.receipt.json", origins["proposal"])
    prior.put(attempt, "catalog-origin.receipt.json", {
        **origins["catalog"], "latest": positive["latest"],
        "checkpoint": positive["review"]["checkpoint"],
        "trust": "observed retained bindings, not an independent schema-1 authority root"})
    reservation = capture.retained_document(capture.binding(attempt / "write-reservation.json"))
    result = bundle.materialize(origins, positive["identity"], attempt, reservation)
    if result["status"] == "UNKNOWN":
        raise capture.UnavailableBinding(result["unavailable_binding"])
    if result["status"] != "PASS":
        raise RuntimeError(f"retained authority materialization rejected: {result.get('reason')}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--validate":
        raise SystemExit("expected --validate FRESH_IGNORED_ATTEMPT")
    outcome = prior.validate(
        Path(sys.argv[2]).absolute(), retained_validation=retained_validation,
        extra_sources=(Path(bundle.__file__).resolve(), Path(__file__).resolve()),
        test_paths=(Path(prior.__file__).resolve(), Path(__file__).resolve()))
    print(json.dumps({key: value for key, value in outcome.items() if key != "members"}, sort_keys=True))
    raise SystemExit(outcome["exit_status"])
