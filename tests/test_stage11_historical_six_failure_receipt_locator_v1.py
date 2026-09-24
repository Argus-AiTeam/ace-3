"""One in-memory compile/pytest round; no cache, evidence or frozen-root writes.

Run:
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B \
tests/test_stage11_historical_six_failure_receipt_locator_v1.py
"""

import builtins
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess

import pytest


SOURCE = Path(__file__).resolve().parents[1] / (
    "ace3/model/candidates/stage11_historical_six_failure_receipt_locator_v1.py"
)
SPEC = importlib.util.spec_from_file_location("stage11_receipt_locator", SOURCE)
locator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(locator)


def encoded(value):
    return value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()


def pin(path, raw):
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def evidence(monkeypatch):
    repository = Path("/synthetic/ace3-argus")
    project = Path("/synthetic/project")
    roots = locator._scopes(repository, project)
    files = {}
    index = roots[0] / "historical-index.json"
    entries = []
    for number in range(6):
        path = roots[4] / f"original-{number}.json"
        raw = encoded({
            "failure_id": f"retained-{number}", "status": "FAIL",
            "gate": ("account", "workdir", "interpreter")[number % 3],
            "reason": f"Retained synthetic gate mismatch {number}; never executed.",
        })
        files[path] = raw
        entries.append({"failure_id": f"retained-{number}", "failure": pin(path, raw)})

    def publish(records=None, path=index):
        files[path] = encoded({"historical_six_gate_failures": {
            "count_required_by_authority": 6,
            "individual_failure_evidence": entries if records is None else records,
            "status": "RETAINED_UNRESOLVED", "reason": "Independent fixture, not production evidence.",
        }})

    def inventory(root):
        for path in sorted(files):
            if path == root or root in path.parents:
                yield path, None, None

    real_read = Path.read_bytes
    real_is_file = Path.is_file
    real_exists = Path.exists
    monkeypatch.setattr(locator, "_inventory", inventory)
    monkeypatch.setattr(Path, "exists", lambda path: path in files if repository in path.parents
                        or project in path.parents else real_exists(path))
    monkeypatch.setattr(Path, "is_file", lambda path: path in files if repository in path.parents
                        or project in path.parents else real_is_file(path))
    monkeypatch.setattr(Path, "read_bytes", lambda path: files[path] if path in files else real_read(path))
    publish()
    return {
        "repository": repository, "project": project, "roots": roots,
        "files": files, "entries": entries, "index": index, "publish": publish,
        "locate": lambda: locator.locate_receipts(repository_root=repository, project_root=project),
    }


def blocked(report, code, reason):
    assert report["status"] == "BLOCKED", report
    assert any(item["code"] == code and reason in item["reason"]
               for item in report["blockers"]), report
    assert not report["closure_proven"]
    assert not report["release_consumption_admissible"]


def test_six_distinct_located_receipts_are_not_closure(evidence):
    report = evidence["locate"]()
    assert report["status"] == "LOCATED", report
    assert report["located_count"] == 6
    assert [entry["failure"] for entry in report["records"]] == [
        entry["failure"] for entry in evidence["entries"]
    ]
    assert {entry["gate"] for entry in report["records"]} == locator.GATES
    assert all(entry["closure_eligibility"]["status"] == "ABSENT" for entry in report["records"])
    assert not report["closure_proven"] and not report["release_consumption_admissible"]
    assert not report["release_consumption_authorized"]


def test_retained_closures_only_qualify_for_independent_review(evidence):
    for number, entry in enumerate(evidence["entries"]):
        path = evidence["roots"][2] / f"closure-{number}.json"
        raw = encoded({
            "failure_id": entry["failure_id"], "status": "PASS",
            "original_failure": entry["failure"], "test_node": f"test_gate[{number}]",
        })
        evidence["files"][path] = raw
        entry["closure"] = pin(path, raw)
    evidence["publish"]()
    report = evidence["locate"]()
    assert report["status"] == "LOCATED", report
    assert all(entry["closure_eligibility"]["eligible_for_independent_closure_review"]
               for entry in report["records"])
    assert not report["closure_proven"] and not report["release_consumption_admissible"]


@pytest.mark.parametrize("count,code", [(0, "ABSENT"), (5, "ABSENT"), (7, "AMBIGUOUS")])
def test_exact_count_not_a_placeholder_census(evidence, count, code):
    records = evidence["entries"][:count]
    if count == 7:
        path = evidence["roots"][4] / "seventh.json"
        raw = encoded({"failure_id": "seventh", "status": "FAIL", "gate": "account"})
        evidence["files"][path] = raw
        records = records + [{"failure_id": "seventh", "failure": pin(path, raw)}]
    evidence["publish"](records)
    report = evidence["locate"]()
    blocked(report, code, f"Located {count}")
    assert report["located_count"] == count


def test_duplicate_id_in_one_index_is_not_six(evidence):
    evidence["entries"][-1] = deepcopy(evidence["entries"][0])
    evidence["publish"]()
    blocked(evidence["locate"](), "AMBIGUOUS", "duplicate failure_id")


def test_repeated_citations_coalesce_deterministically(evidence):
    evidence["publish"](list(reversed(evidence["entries"])), evidence["roots"][2] / "round-0001.json")
    evidence["publish"](path=evidence["roots"][3])
    evidence["publish"](path=evidence["roots"][1] / "authority.json")
    report = evidence["locate"]()
    assert report["status"] == "LOCATED", report
    assert report["located_count"] == 6
    assert report == evidence["locate"]()


def test_standalone_receipts_need_existing_pins_not_a_prebuilt_six_index(evidence):
    del evidence["files"][evidence["index"]]
    pins = []
    for entry in evidence["entries"]:
        path = Path(entry["failure"]["path"])
        payload = json.loads(evidence["files"][path])
        payload["stage"] = "stage11"
        raw = encoded(payload)
        evidence["files"][path] = raw
        pins.append(pin(path, raw))
    evidence["files"][evidence["roots"][2] / "retained-census.json"] = encoded({"files": pins})
    report = evidence["locate"]()
    assert report["status"] == "LOCATED", report
    assert report["located_count"] == 6
    assert all(record["index_locations"] for record in report["records"])


def test_unpinned_standalone_failure_is_not_authenticated(evidence):
    evidence["files"].clear()
    evidence["files"][evidence["roots"][4] / "stage11-old/original.json"] = encoded({
        "failure_id": "unindexed", "status": "FAIL", "gate": "account",
    })
    blocked(evidence["locate"](), "ABSENT", "no independently retained pin")


def test_missing_original_receipt_is_absent(evidence):
    del evidence["files"][Path(evidence["entries"][0]["failure"]["path"])]
    blocked(evidence["locate"](), "ABSENT", "missing evidence file")


def test_escaped_json_field_and_jsonl_discovery(evidence):
    raw = evidence["files"].pop(evidence["index"])
    raw = raw.replace(b"historical_six_gate_failures", b"historical_six_gate_\\u0066ailures")
    evidence["files"][evidence["roots"][3]] = b'{"unrelated":1}\n' + raw + b"\n"
    assert evidence["locate"]()["status"] == "LOCATED"


def test_inventory_does_not_descend_unrelated_builds_or_symlinks(monkeypatch):
    root = Path("/synthetic/build")
    monkeypatch.setattr(Path, "exists", lambda path: True)
    monkeypatch.setattr(Path, "is_file", lambda path: False)
    monkeypatch.setattr(Path, "is_symlink", lambda path: path.name == "stage11-link")

    def walk(path, *, followlinks, onerror):
        assert path == root and followlinks is False
        directories = ["unrelated-model", "stage11-old", "stage11-link"]
        yield str(root), directories, []
        assert directories == ["stage11-old"]
        yield str(root / "stage11-old"), [], ["receipt.json", "model.raw", "candidate.py"]

    monkeypatch.setattr(os, "walk", walk)
    assert list(locator._inventory(root)) == [
        (root / "stage11-link", "AMBIGUOUS", "symlink evidence directory"),
        (root / "stage11-old/receipt.json", None, None),
    ]


def test_conflicting_pins_for_same_id_are_ambiguous(evidence):
    records = deepcopy(evidence["entries"])
    path = evidence["roots"][4] / "conflicting-original.json"
    raw = encoded({"failure_id": records[0]["failure_id"], "status": "FAIL", "gate": "workdir"})
    evidence["files"][path] = raw
    records[0]["failure"] = pin(path, raw)
    evidence["publish"](records, evidence["roots"][2] / "conflict.json")
    blocked(evidence["locate"](), "AMBIGUOUS", "conflicting receipts")


@pytest.mark.parametrize("raw,reason", [
    (b"{", "malformed JSON"),
    (b'{"failure_id":"retained-0","failure_id":"retained-0"}', "duplicate JSON field"),
    (b'{"failure_id":"retained-0","value":NaN}', "nonfinite JSON"),
    (b"[]", "not an object"),
    (encoded({"failure_id": "other", "status": "FAIL", "gate": "account"}), "bind retained-0"),
    (encoded({"failure_id": "retained-0", "status": "PASS", "gate": "account"}), "bind retained-0"),
    (encoded({"failure_id": "retained-0", "status": "FAIL", "gate": "science"}), "structured"),
    (encoded({"failure_id": "retained-0", "status": "FAIL", "reason": "account failed"}), "structured"),
])
def test_malformed_or_unclassified_original_receipts(evidence, raw, reason):
    entry = evidence["entries"][0]
    path = Path(entry["failure"]["path"])
    evidence["files"][path] = raw
    entry["failure"] = pin(path, raw)
    evidence["publish"]()
    blocked(evidence["locate"](), "AMBIGUOUS", reason)


@pytest.mark.parametrize("field,value,reason", [
    ("bytes", True, "byte count"),
    ("bytes", 0, "pin mismatch"),
    ("sha256", "0" * 64, "pin mismatch"),
    ("sha256", "F" * 64, "SHA-256"),
    ("unexpected_authority", True, "exactly"),
])
def test_pin_authentication_is_not_a_self_computed_pass(evidence, field, value, reason):
    evidence["entries"][0]["failure"][field] = value
    evidence["publish"]()
    blocked(evidence["locate"](), "AMBIGUOUS", reason)


@pytest.mark.parametrize("path,reason", [
    ("/home/argustest/ace2/original.json", "nonlocal evidence path"),
    ("/other-project/handoffs/receipt.json", "nonlocal evidence path"),
    ("build/../receipt.json", "parent traversal"),
    ("https://example.invalid/receipt.json", "nonlocal evidence URL"),
    ("/synthetic/project/role-sessions/engineer.json", "nonlocal evidence path"),
    ("/synthetic/project/handoffs/mission/role-sessions/engineer.json", "excluded role-session"),
])
def test_nonlocal_paths_rejected_before_read(evidence, monkeypatch, path, reason):
    evidence["entries"][0]["failure"]["path"] = path
    evidence["publish"]()
    original = locator._read_bytes

    def read(candidate):
        assert str(candidate) != path, "rejected path was read"
        return original(candidate)

    monkeypatch.setattr(locator, "_read_bytes", read)
    blocked(evidence["locate"](), "AMBIGUOUS", reason)


def test_symlink_member_rejected_before_read(evidence, monkeypatch):
    alias = Path(evidence["entries"][0]["failure"]["path"])
    resolve = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda path: Path("/outside/receipt.json")
                        if path == alias else resolve(path))
    blocked(evidence["locate"](), "AMBIGUOUS", "symlink evidence path")


def test_missing_and_malformed_indexes_do_not_invent_ids(evidence):
    evidence["files"][evidence["index"]] = encoded({
        "historical_six_gate_failures": {
            "count_required_by_authority": 6, "individual_failure_evidence": None,
        },
    })
    report = evidence["locate"]()
    blocked(report, "ABSENT", "Located 0")
    assert report["records"] == []
    assert report["unresolved_indexes"]
    evidence["files"][evidence["index"]] = b'{"historical_six_gate_failures":'
    blocked(evidence["locate"](), "AMBIGUOUS", "malformed JSON")


@pytest.mark.parametrize("change", [
    {"status": "FAIL"}, {"failure_id": "other"}, {"test_node": ""},
    {"original_failure": {}}, {"authorization_action": "release"},
])
def test_closure_must_bind_the_original_failure(evidence, change):
    entry = evidence["entries"][0]
    closure = {
        "failure_id": entry["failure_id"], "status": "PASS",
        "original_failure": entry["failure"], "test_node": "test_gate",
    }
    closure.update(change)
    raw = encoded(closure)
    path = evidence["roots"][4] / "closure.json"
    evidence["files"][path] = raw
    entry["closure"] = pin(path, raw)
    evidence["publish"]()
    blocked(evidence["locate"](), "AMBIGUOUS", "closure")


@pytest.mark.parametrize("arguments", [
    ["--check"], ["--validate"], ["--runtime"], ["--output", "existing.json"],
    ["--repository-root", "/other-project"], ["--hardware"],
])
def test_forbidden_cli_dispatches_are_not_options(monkeypatch, arguments):
    def deny():
        pytest.fail("invalid CLI action reached discovery")
    monkeypatch.setattr(locator, "locate_receipts", deny)
    with pytest.raises(SystemExit) as error:
        locator.main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("is_blocked,exit_code", [(False, 0), (True, 2)])
def test_cli_deterministic_json(monkeypatch, capsys, is_blocked, exit_code):
    report = {"status": "BLOCKED" if is_blocked else "LOCATED", "blocked": is_blocked}
    monkeypatch.setattr(locator, "locate_receipts", lambda: report)
    assert locator.main([]) == exit_code
    assert capsys.readouterr().out == json.dumps(report, sort_keys=True, indent=2) + "\n"


def test_current_retained_evidence_once_no_writes_or_dispatch(monkeypatch, capsys):
    def snapshot():
        return {str(path): pin(path, path.read_bytes())
                for path in locator.FROZEN_ROOT.rglob("*") if path.is_file()}

    before = snapshot()
    assert before, "Frozen 016 evidence must exist for the live observation"
    real_import = builtins.__import__
    real_open, real_io_open, real_os_open = builtins.open, io.open, os.open

    def deny(*args, **kwargs):
        pytest.fail("locator attempted a write or forbidden dispatch")

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"ace3", "argus_skill", "numpy", "torch"}
        return real_import(name, *args, **kwargs)

    def read_only_open(file, mode="r", *args, **kwargs):
        assert not set(mode) & set("wax+"), "attempted write"
        return real_open(file, mode, *args, **kwargs)

    def read_only_io_open(file, mode="r", *args, **kwargs):
        assert not set(mode) & set("wax+"), "attempted write"
        return real_io_open(file, mode, *args, **kwargs)

    def read_only_os_open(path, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        return real_os_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", guarded_import)
        guard.setattr(builtins, "open", read_only_open)
        guard.setattr(io, "open", read_only_io_open)
        guard.setattr(os, "open", read_only_os_open)
        for name in ("system", "posix_spawn", "posix_spawnp", "fork", "execv", "execve",
                     "mkdir", "remove", "unlink", "rename", "replace", "chmod", "truncate"):
            guard.setattr(os, name, deny)
        guard.setattr(subprocess, "Popen", deny)
        guard.setattr(socket, "socket", deny)
        report = locator.locate_receipts()
    assert report["blocked"], report
    assert report["blockers"], report
    assert not report["closure_proven"] and not report["release_consumption_admissible"]
    assert any(str(locator.FROZEN_ROOT / "release-inputs.json") in item["path"]
               for item in report["unresolved_indexes"]), report
    assert snapshot() == before, "Frozen 016 evidence changed"
    with capsys.disabled():
        print("Current retained locator observation: " + json.dumps({
            key: report[key] for key in (
                "status", "located_count", "required_count", "unresolved_indexes",
                "scan", "closure_proven", "release_consumption_admissible",
            )
        }, sort_keys=True))
        print("Concrete blockers: " + json.dumps(report["blockers"][:6], sort_keys=True))
        print(f"Blocker count: {len(report['blockers'])}; frozen files unchanged: {len(before)}")


if __name__ == "__main__":
    for source in (SOURCE, Path(__file__).resolve()):
        compile(source.read_bytes(), str(source), "exec")
    print("Targeted compilation: PASS (two files, in memory)")
    raise SystemExit(pytest.main([
        str(Path(__file__).resolve()), "-q", "-p", "no:cacheprovider",
        "--capture=sys", "--assert=plain", "--noconftest",
    ]))
