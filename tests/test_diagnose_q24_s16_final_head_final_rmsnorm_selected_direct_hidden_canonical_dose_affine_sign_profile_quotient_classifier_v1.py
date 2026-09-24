"""Integer-only quotient oracle, hostile retained controls and exactly-once capture."""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

from ace3.model.candidates import (
    diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_canonical_dose_affine_sign_profile_quotient_classifier_v1 as diagnostic,
    diagnostic_capture_v1 as capture_helper,
)


def pair(text):
    parts = text.split("/")
    return int(parts[0]), int(parts[1]) if len(parts) == 2 else 1


def product_equal(left, right, multiplier=1):
    a, b = pair(left)
    c, d = pair(right)
    return a * d == multiplier * b * c


def less(left, right):
    a, b = pair(left)
    c, d = pair(right)
    return a * d < b * c


@pytest.fixture(scope="module")
def retained():
    modules = diagnostic.load_helpers()

    def forbidden(*args, **kwargs):
        raise AssertionError("ancestor semantic execution forbidden")

    for module in modules:
        for name in ("check", "classify", "report", "_classify", "main", "root_margin", "nearest", "numeric_fields"):
            setattr(module, name, forbidden)
    documents, authentication = diagnostic.authenticate(*modules)
    assert authentication["status"] == "AUTHENTICATED"
    assert authentication["review"]["producer_role"] == "reviewer"
    return documents, modules


def test_independent_integer_oracle_complete_quotient(retained):
    documents, modules = retained
    result = diagnostic.classify(documents, modules)
    assert result["decision"] == "SUPPORTED", result.get("unknown_reasons", result.get("failures"))
    assert result["counts"] == diagnostic.COUNTS
    assert result["class_order"] == ["frozen_inherited", "mapped62", "scratch", "frozen_inherited_o"]
    source = documents[0]["report"]["root_margin_matrix"]
    records = result["field_membership_matrix"]
    assert len(source) == len(records) == 735
    expected_groups, root_counts = {}, {}
    for index, (row, member) in enumerate(zip(source, records, strict=True)):
        assert member["retained_root_pointer"] == f"/report/root_margin_matrix/{index}"
        for key in (*modules[-2].SCOPE_FIELDS, "field"):
            assert member[key] == row[key]
        a, b = pair(row["baseline"])
        m, n = pair(row["slope"])
        vector, absolute = [], {}
        for canonical in result["class_order"]:
            v, w = pair(row["classes"][canonical]["value"])
            d, e = pair(row["classes"][canonical]["dose"])
            assert v * b * n * e == w * (a * n * e + m * d * b)
            vector.append((v > 0) - (v < 0))
            absolute[canonical] = (abs(v), w)
            assert member["class_values"][canonical] == row["classes"][canonical]["value"]
        kind = "UNIQUE" if m else "NO_ROOT" if a else "ALL_DOSES"
        if m:
            p, q = pair(row["root"])
            assert a * n * q + m * p * b == 0
            where = ("LOWER_ENDPOINT" if p == 0 else "ABOVE" if less("3/32768", row["root"])
                     else "UNEXPECTED")
        else:
            assert row["root"] is None
            where = kind
        root_counts[where] = root_counts.get(where, 0) + 1
        closest = next(iter(absolute.values()))
        for numerator, denominator in absolute.values():
            if numerator * closest[1] < closest[0] * denominator:
                closest = numerator, denominator
        nearest = [c for c, (v, w) in absolute.items() if v * closest[1] == closest[0] * w]
        stable = len(set(vector)) == 1
        strict = stable and 0 not in vector
        parts = row["field"].split("/")
        swapped = row["field"] in ("/left_weight", "/right_weight")
        preserve = (swapped or
                    parts[1] in ("factors", "factor_deltas") and parts[2] != "row_difference" or
                    parts[1] == "components" and parts[3] in ("hidden", "canonical_hidden", "source_delta") or
                    parts[1] == "mass" and (parts[2] == "hidden" or parts[3] != "signed"))
        paired = row["table"] == "coordinate"
        rule = ("SWAP_WEIGHTS" if swapped else "PRESERVE" if preserve else "NEGATE"
                ) if paired else "UNPAIRED_RETAINED_SCOPE"
        profile = {"sign_vector": vector, "root_kind": kind, "root": row["root"], "root_location": where,
                   "sign_stable": stable, "strict_sign_stable": strict,
                   "no_sign_reversal": not (-1 in vector and 1 in vector),
                   "nearest_classes": nearest, "orientation_rule": rule}
        assert {k: member[k] for k in diagnostic.PROFILE_FIELDS} == profile
        key = (tuple(vector), kind, row["root"], where, stable, strict,
               profile["no_sign_reversal"], tuple(nearest), rule)
        expected_groups.setdefault(key, []).append(index)
        assert member["orientation_multiplier"] == ((1 if preserve else -1) if paired else None)
        if where == "LOWER_ENDPOINT":
            assert vector[0] == 0 and all(v == vector[1] != 0 for v in vector[1:])
            assert nearest == ["frozen_inherited"]
        elif kind in ("ALL_DOSES", "NO_ROOT"):
            assert nearest == result["class_order"]
    actual_groups = {}
    for group_id, group in enumerate(result["quotient"]):
        profile = group["profile"]
        key = (tuple(profile["sign_vector"]), profile["root_kind"], profile["root"], profile["root_location"],
               profile["sign_stable"], profile["strict_sign_stable"], profile["no_sign_reversal"],
               tuple(profile["nearest_classes"]), profile["orientation_rule"])
        assert key not in actual_groups
        actual_groups[key] = group["members"]
        assert group["count"] == len(group["members"]) > 0
        assert all(records[i]["group"] == group_id for i in group["members"])
    assert actual_groups == expected_groups
    assert result["profile_count"] == len(expected_groups)
    assert root_counts == diagnostic.ROOT_COUNTS == result["root_location_counts"]
    assert result["sign_stability_counts"] == diagnostic.STABILITY_COUNTS
    for name, expected in (("lower_endpoint_departures", 20), ("above_interval_roots", 16),
                           ("identically_zero_fields", 485), ("constant_nonzero_fields", 214)):
        assert len(result[name]) == expected and len(set(result[name])) == expected
    assert sorted(sum((result[k] for k in ("lower_endpoint_departures", "above_interval_roots",
                                         "identically_zero_fields", "constant_nonzero_fields")), [])) == list(range(735))
    covered = []
    for item in result["orientation_pairs"]:
        forward, reverse = source[item["forward"]], source[item["reverse"]]
        f, r = records[item["forward"]], records[item["reverse"]]
        assert (forward["left_id"], forward["right_id"]) == (reverse["right_id"], reverse["left_id"])
        assert forward["coordinate"] == reverse["coordinate"]
        expected_target = {"/left_weight": "/right_weight", "/right_weight": "/left_weight"}.get(
            forward["field"], forward["field"])
        assert reverse["field"] == f["reverse_field"] == expected_target
        assert item["multiplier"] == f["orientation_multiplier"]
        for name in ("baseline", "slope"):
            assert product_equal(reverse[name], forward[name], item["multiplier"])
        assert r["sign_vector"] == [item["multiplier"] * s for s in f["sign_vector"]]
        assert r["root"] == f["root"] and r["nearest_classes"] == f["nearest_classes"]
        assert (item["forward_group"], item["reverse_group"]) == (f["group"], r["group"])
        covered.extend((item["forward"], item["reverse"]))
    assert len(covered) == len(set(covered)) == 588
    assert len(result["unpaired_fields"]) == 147
    assert sorted(covered + result["unpaired_fields"]) == list(range(735))
    assert all(records[i]["table"] == "coordinate_component" for i in result["unpaired_fields"])
    assert result["failure_count"] == 0 and result["failures"] == [] and result["unknown_reasons"] == []
    for key in ("reference_scope", "lineage_separation", "claim_boundary", "root_domain", "margin_definition"):
        assert result[key] == documents[0]["report"][key]
    assert result["parent_source_equivalence_status_preserved"] == "REJECTED"


@pytest.mark.parametrize("kind", [
    "count", "histogram", "aggregate_stability", "value", "baseline", "slope", "sign", "root", "root_kind",
    "location", "root_residual", "distance", "distance_residual", "absolute_margin", "minimum", "interval",
    "stability", "strict", "reversal", "tie", "control_residual", "orientation_target", "orientation_sign",
    "orientation_residual", "orientation_pair",
])
def test_numeric_sign_count_and_orientation_mismatches_rejected(retained, kind):
    documents, modules = retained
    changed = list(documents)
    for index in (0, 1):
        changed[index] = {**changed[index], "report": copy.deepcopy(changed[index]["report"])}
    report = changed[0]["report"]
    row = next(r for r in report["root_margin_matrix"] if r["root_location"] == "LOWER_ENDPOINT")
    entry = row["classes"]["mapped62"]
    if kind == "count":
        report["counts"]["coefficient_records"] -= 1
    elif kind == "histogram":
        report["root_location_counts"]["LOWER_ENDPOINT"] -= 1
    elif kind == "aggregate_stability":
        report["sign_stability_counts"]["sign_stable"] -= 1
    elif kind in ("value", "sign", "distance", "distance_residual", "absolute_margin"):
        key = {"distance": "signed_root_distance", "distance_residual": "root_distance_identity_residual"}.get(kind, kind)
        entry[key] = 0 if kind == "sign" else "7"
    elif kind in ("baseline", "slope", "root"):
        row[kind] = "7"
    elif kind == "root_kind":
        row["root_kind"] = "NO_ROOT"
    elif kind == "location":
        row["root_location"] = "ABOVE"
    elif kind == "root_residual":
        row["root_equation_residual"] = "1"
    elif kind in ("minimum", "interval"):
        row["minimum_retained_absolute_margin" if kind == "minimum" else "interval_minimum_absolute_margin"] = "1"
    elif kind in ("stability", "strict", "reversal"):
        key = {"stability": "sign_stable", "strict": "strict_sign_stable", "reversal": "no_sign_reversal"}[kind]
        row[key] = not row[key]
    elif kind == "tie":
        row["nearest_classes"].append("mapped62")
    elif kind == "control_residual":
        row["retained_value_residuals"]["mapped62"] = "1"
    else:
        orientation = changed[1]["report"]["orientation_closure_matrix"][0]
        item = orientation["fields"]["/left_weight"]
        if kind == "orientation_target":
            item["reverse_field"] = "/left_weight"
        elif kind == "orientation_sign":
            item["multiplier"] = -1
        elif kind == "orientation_pair":
            orientation["reverse_pair"] = [34319, 319]
        else:
            item["residuals"]["baseline"] = "1"
    result = diagnostic.classify(changed, modules)
    assert result["decision"] == "REJECTED", (kind, result)
    assert result["failure_count"] > 0 and result["unknown_reasons"] == []


@pytest.mark.parametrize("kind", ["missing", "duplicate", "noncanonical", "dose", "lineage", "binding", "boundary"])
def test_incomplete_or_incompatible_evidence_unknown(retained, kind):
    documents, modules = retained
    changed = list(documents)
    for index in (0, 2):
        changed[index] = {**changed[index], "report": copy.deepcopy(changed[index]["report"])}
    report = changed[0]["report"]
    if kind == "missing":
        report["root_margin_matrix"].pop()
    elif kind == "duplicate":
        report["root_margin_matrix"].append(report["root_margin_matrix"][0])
    elif kind == "noncanonical":
        report["root_margin_matrix"][0]["baseline"] = "0/2"
    elif kind == "dose":
        report["dose_classes"][0]["absolute_dose"] = "1"
    elif kind == "lineage":
        report["lineage_separation"] = "spliced"
    elif kind == "boundary":
        report["claim_boundary"] = "admitted"
    else:
        changed[2]["report"]["reconstruction_residual_matrix"][0]["retained_bindings"] = {}
    result = diagnostic.classify(changed, modules)
    assert result["decision"] == "UNKNOWN", (kind, result)
    assert result["unknown_reasons"]


@pytest.mark.parametrize("target", list(diagnostic.PINS))
def test_all_predecessor_pins_authenticated(retained, monkeypatch, target):
    pins = copy.deepcopy(diagnostic.PINS)
    pins[target]["sha256"] = "0" * 64
    monkeypatch.setattr(diagnostic, "PINS", pins)
    with pytest.raises(ValueError, match="artifact hash"):
        diagnostic.authenticate(*retained[1])


@pytest.mark.parametrize("kind", ["whole", "stderr", "argv", "environment", "exit", "review", "counter", "predecessor"])
def test_complete_capture_runtime_review_authentication(retained, monkeypatch, kind):
    original = diagnostic.bound_bytes
    capture = json.loads(original(diagnostic.PINS["capture"]))
    check = capture["results"][-1]
    target = {
        "whole": check["files"][5]["path"], "stderr": check["files"][4]["path"],
        "argv": check["files"][1]["path"], "environment": check["files"][2]["path"],
        "exit": diagnostic.PINS["capture"]["path"], "review": diagnostic.PINS["review"]["path"],
        "counter": diagnostic.PINS["stdout"]["path"], "predecessor": diagnostic.PINS["stdout"]["path"],
    }[kind]

    def corrupted(binding):
        data = original(binding)
        if binding["path"] != target:
            return data
        if kind in ("whole", "stderr"):
            return data + b"tampered"
        value = json.loads(data)
        if kind == "argv":
            value[-1] = "--other"
        elif kind == "environment":
            value["uid"] = 0
        elif kind == "exit":
            value["results"][-1]["exit_status"] = 1
        elif kind == "review":
            value["producer_role"] = "engineer"
        elif kind == "counter":
            value["dispatch_and_write_audit"]["prefix_dispatch"] = 1
        else:
            value["artifact_authentication"]["retained_predecessor_raw_bindings"] = {}
        return json.dumps(value).encode()

    monkeypatch.setattr(diagnostic, "bound_bytes", corrupted)
    monkeypatch.setattr(retained[1][0], "bound_bytes", corrupted)
    with pytest.raises(ValueError):
        diagnostic.authenticate(*retained[1])


def test_read_only_and_no_ancestor_semantics(retained, tmp_path):
    documents, modules = retained
    helper = modules[-1]
    audit = dict.fromkeys(helper.COUNTERS, 0)
    with helper.read_only(audit):
        assert diagnostic.classify(documents, modules)["decision"] == "SUPPORTED"
        with pytest.raises(helper.ForbiddenOperation):
            (tmp_path / "forbidden").write_text("no")
        with pytest.raises(helper.ForbiddenOperation):
            subprocess.run([diagnostic.PYTHON, "-V"], check=True)
    assert audit["forbidden_calls"] == 2
    assert not (tmp_path / "forbidden").exists()


def test_authentication_failure_and_cli(monkeypatch, capsys):
    def fail():
        raise ValueError("authentication pin mismatch")

    monkeypatch.setattr(diagnostic, "load_helpers", fail)
    assert diagnostic.main(["--check"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "UNKNOWN"
    assert "authentication pin mismatch" in result["report"]["unknown_reasons"][0]
    with pytest.raises(SystemExit) as error:
        diagnostic.main(["--check", "--output", "forbidden"])
    assert error.value.code == 2


def test_uncaptured_stdout_unknown():
    result = diagnostic.check()
    assert result["decision"] == "UNKNOWN"
    assert result["byte_exact_capture"]["status"] == "UNKNOWN"


encoded = capture_helper.encoded
binding = capture_helper.binding
save = capture_helper.save


def capture(directory):
    if (Path.cwd(), os.getuid(), sys.executable) != (diagnostic.ROOT, 1000, diagnostic.PYTHON):
        raise RuntimeError("account/workdir/interpreter gate")
    with capture_helper.same_scope(diagnostic.ROOT, diagnostic.MODULE):
        return _capture(directory)


def _capture(directory):
    """Run compile/pytest/check once; retain failures as well as successful complete bytes."""
    root, python = diagnostic.ROOT, diagnostic.PYTHON
    if (Path.cwd(), os.getuid(), sys.executable) != (root, 1000, python):
        raise RuntimeError("account/workdir/interpreter gate")
    directory = Path(directory).absolute()
    if (directory.parent.parent != root / "build" or directory.name != "run"
            or directory.resolve() != directory):
        raise RuntimeError("fresh build/<attempt>/run required")
    directory.mkdir(exist_ok=False)
    sources = [binding(diagnostic.SOURCE), binding(diagnostic.TEST)]
    preflight = {
        "cwd": str(root), "uid": os.getuid(), "python": python, "python_version": sys.version,
        "environment": diagnostic.ENVIRONMENT, "sources": sources, "executable": binding(Path(python).resolve()),
        "independent_host_review": "REQUIRED", "scope": diagnostic.SCOPE,
        "model_or_service_calls_authorized": 0, "attempt": str(directory.parent),
        "capture_implementation": capture_helper.implementation_pins(),
    }
    results, failure, pytest_xml = [], None, None

    def run(label, argv):
        return capture_helper.run_command(directory, label, argv, preflight, results)

    try:
        if run("branch", ["git", "branch", "--show-current"]) != b"argus/full-projection\n":
            raise RuntimeError("branch gate")
        run("ignored-build", ["git", "check-ignore", str(directory)])
        probe = (
            "import json,os,subprocess; "
            "rows=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines(); "
            f"module={diagnostic.MODULE!r}; test={str(diagnostic.TEST)!r}; "
            "matches=[r for r in rows if int(r.split(None,1)[0]) != os.getpid() "
            "and ((' -m '+module+' ') in r or (' -m pytest ' in r and test in r))]; print(json.dumps(matches))"
        )
        if json.loads(run("concurrency", [python, "-B", "-c", probe])) != []:
            raise RuntimeError("same-scope concurrency")
        code = ("from pathlib import Path; "
                f"paths={[str(diagnostic.SOURCE), str(diagnostic.TEST)]!r}; "
                "[compile(Path(p).read_bytes(),p,'exec') for p in paths]; print('compiled 2 Python files')")
        run("compile", [python, "-B", "-c", code])
        run("pytest", [python, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                       "--basetemp", str(directory / "pytest-tmp"), "--junitxml", str(directory / "pytest.xml"),
                       str(diagnostic.TEST)])
        pytest_xml = binding(directory / "pytest.xml")
        suites = ET.fromstring(diagnostic.bound_bytes(pytest_xml)).findall("testsuite")
        if not suites or sum(int(s.attrib["tests"]) for s in suites) == 0 or any(
            int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped")
        ):
            raise RuntimeError("test failure/error/skip")
        result = json.loads(run("check", [python, "-B", "-m", diagnostic.MODULE, "--check"]))
        if (result["artifact_authentication"]["status"], result["decision"]) != ("AUTHENTICATED", "SUPPORTED"):
            raise RuntimeError("diagnostic not authenticated SUPPORTED")
        if set(result["dispatch_and_write_audit"]) != set(diagnostic.load_helpers()[-1].COUNTERS) or any(
            type(v) is not int or v != 0 for v in result["dispatch_and_write_audit"].values()
        ):
            raise RuntimeError("forbidden counter")
        if [binding(diagnostic.SOURCE), binding(diagnostic.TEST)] != sources:
            raise RuntimeError("source/test drift")
        if capture_helper.implementation_pins() != preflight["capture_implementation"]:
            raise RuntimeError("capture implementation drift")
    except (OSError, ValueError, KeyError, RuntimeError, ET.ParseError) as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    receipt = {"preflight": preflight, "results": results, "failure": failure, "pytest_xml": pytest_xml,
               "success": failure is None, "sources_after": [binding(diagnostic.SOURCE), binding(diagnostic.TEST)],
               "capture_implementation_after": capture_helper.implementation_pins()}
    captured = save(directory, "capture.json", encoded(receipt))
    print(json.dumps({"capture": captured, "success": failure is None, "failure": failure}))
    return 1 if failure else 0


def launch(directory):
    directory = Path(directory).absolute()
    if directory.parent != diagnostic.ROOT / "build" or directory.resolve() != directory:
        raise RuntimeError("fresh direct build child required")
    if (Path.cwd(), os.getuid(), sys.executable) != (diagnostic.ROOT, 1000, diagnostic.PYTHON):
        raise RuntimeError("account/workdir/interpreter gate")
    directory.mkdir(exist_ok=False)
    argv = [diagnostic.PYTHON, "-B", str(diagnostic.TEST), "--run", str(directory / "run")]
    identity = {
        "argv": argv, "command": diagnostic.environment_command(diagnostic.ENVIRONMENT, argv),
        "cwd": str(Path.cwd()), "uid": os.getuid(), "environment": diagnostic.ENVIRONMENT,
        "launcher": binding(diagnostic.TEST), "invocation_argv": sys.orig_argv,
    }
    output, outer = capture_helper.seal_launcher(directory, identity)
    status = outer["exit_status"] or (0 if outer["success"] else 1)
    if outer["success"]:
        helper = diagnostic.load_helpers()[-1]
        receipt = helper.decode(output)
        captured = helper.decode(diagnostic.bound_bytes(receipt["capture"]), metadata=True)
        stdout = (directory / "run/check.stdout").read_bytes()
        preflight, inputs = diagnostic.verify_capture(
            directory, captured, outer, stdout, [binding(diagnostic.SOURCE), binding(diagnostic.TEST)],
            diagnostic.MODULE, helper,
        )
        result = helper.decode(stdout, metadata=True)
        diagnostic.same(result["byte_exact_capture"]["command_inputs"], inputs, "current capture identity")
        diagnostic.same(result["source_test_pins"], preflight["sources"], "current source pins")
        diagnostic.same(result["report"]["counts"], diagnostic.COUNTS, "current census")
        diagnostic.same(result["report"]["failure_count"], 0, "current failures")
    print(output.decode(), end="")
    print(json.dumps(outer))
    return status


def test_shared_capture_check_cli(monkeypatch, capsys):
    """Exercise CLI serialization only; never replay this closed scientific producer."""
    result = {"decision": "UNKNOWN", "reason": "synthetic capture test"}
    monkeypatch.setattr(diagnostic, "check", lambda: result)
    assert diagnostic.main(["--check"]) == 2
    assert capsys.readouterr().out.encode() == (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("--launch", "--run"):
        raise SystemExit({"--launch": launch, "--run": capture}[sys.argv[1]](Path(sys.argv[2])))
    raise SystemExit("use --launch or --run <fresh build attempt>")
