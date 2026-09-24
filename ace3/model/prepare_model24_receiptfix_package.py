#!/usr/bin/env python3
"""Prepare a fresh r10 package that matches durable-runner launch ordering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import tarfile


REPOSITORY = Path("/home/argustest/ace3-argus")
REVIEWED_NONCE = "5a61f0b24cfa722b"
REVIEWED_PREP = Path(
    f"/home/argustest/ace3-model24-r10-prep-20260828-{REVIEWED_NONCE}"
)
REVIEWED_REVIEW = Path(
    f"/home/argustest/ace3-model24-r10-review-20260828-{REVIEWED_NONCE}/review.json"
)
REVIEWED_SEAL = REVIEWED_PREP / "package/seal.json"
REVIEWED_REVIEW_SHA256 = "812c1bb4947cc4adc87b67c5afc6b0424035d02d295c950fbd497f30677dfad5"
REVIEWED_SEAL_SHA256 = "818e838592d024854b7f71eaff75b1888a4f6ba5d51bb5da2a2ae9988588b741"
COMMIT = "42c895ce1e5fea00525e9f2f7fef66f0fbb8e118"
TREE = "93f57e9e0c87c3f8638282475e12f8f5289b498b"
CHECKPOINT = REPOSITORY / "build/model24_rtl_cascade/checkpoint/model.safetensors"
MODEL_PYTHON = Path("/home/argustest/miniconda3/bin/python3")
ARGUS_PYTHON = Path(
    "/home/argustest/argustest2/argus-skill-latest/.venv/bin/python"
)
RUNNER_ROOT = Path(
    "/home/argustest/argustest2/argus-skill-latest/argus_skill/tools/subagent"
)
RUNNER_SOURCES = {
    "_cli.py": "edb2de6fa5aa297dc387354bd6806b2ba8b424dd5f1b1bebd60fc661bf652af7",
    "_direct_run.py": "b754f94260071ed3e6676cb4276e20bf4746d5c3ff0320e198c0030009580f82",
    "_registry.py": "a44d40f0248b9b31121cbcd67f3082477620d606f56a8b7effa3f9e3e3a688dc",
}
LAUNCH_CONTRACT = REPOSITORY / "ace3/contracts/model24_r10_durable_launch_contract.json"

STATIC_PACKAGE_NAMES = (
    "launch-contract.json",
    "launch-once.sh",
    "prepare-and-seal.py",
    "receiptfix-builder.py",
    "receiptfix-construction.json",
    "seal-terminal.py",
    "test-launch-contract.py",
    "validate-package.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )


def snapshot_tree(root: Path) -> dict[str, object]:
    require(root.exists(), f"protected path missing: {root}")
    if root.is_file():
        return {
            "path": str(root),
            "kind": "file",
            "bytes": root.stat().st_size,
            "sha256": digest(root),
        }
    hasher = hashlib.sha256()
    files = 0
    total = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            kind = b"L"
            payload = os.readlink(path).encode("utf-8")
        elif path.is_file():
            kind = b"F"
            payload = path.read_bytes()
            files += 1
            total += len(payload)
        elif path.is_dir():
            kind = b"D"
            payload = b""
        else:
            raise SystemExit(f"unsupported protected entry: {path}")
        hasher.update(
            kind
            + b"\0"
            + relative
            + b"\0"
            + f"{mode:o}".encode("ascii")
            + b"\0"
        )
        hasher.update(hashlib.sha256(payload).digest())
    return {
        "path": str(root),
        "kind": "directory",
        "files": files,
        "bytes": total,
        "tree_sha256": hasher.hexdigest(),
    }


def write_new(path: Path, payload: bytes, mode: int) -> None:
    require(path.parent.is_dir(), f"write parent is absent: {path.parent}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(mode)


def transformed_template(payload: bytes, base: Path, output: Path, nonce: str) -> bytes:
    text = payload.decode("utf-8")
    task_id = f"ace3-model24-r10-42c895c-{nonce}"
    replacements = (
        (str(REVIEWED_PREP), str(base)),
        (
            f"/home/argustest/ace3-model24-r10-output-20260828-{REVIEWED_NONCE}",
            str(output),
        ),
        (
            f"/home/argustest/ace3-model24-r10-authority-20260828-{REVIEWED_NONCE}.json",
            f"/home/argustest/ace3-model24-r10-authority-20260828-{nonce}.json",
        ),
        (
            f"/home/argustest/ace3-model24-r10-review-20260828-{REVIEWED_NONCE}",
            f"/home/argustest/ace3-model24-r10-review-20260828-{nonce}",
        ),
        (f"ace3-model24-r10-42c895c-{REVIEWED_NONCE}", task_id),
        (REVIEWED_NONCE, nonce),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text.encode("utf-8")


VALIDATOR_SOURCE = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"missing regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="ascii"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _pid_alive(value: object) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def validate_seal(package: Path) -> tuple[dict, dict]:
    seal_path = package / "seal.json"
    seal = load_json(seal_path)
    records = seal.get("package_files")
    require(isinstance(records, list), "package seal file list is missing")
    sealed_names = set()
    for record in records:
        require(isinstance(record, dict), "malformed package seal record")
        relative = Path(str(record.get("path", "")))
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe sealed path")
        name = relative.as_posix()
        require(name not in sealed_names, f"duplicate sealed path: {name}")
        sealed_names.add(name)
        path = package / relative
        require(path.is_file() and not path.is_symlink(), f"missing package file: {name}")
        require(path.stat().st_size == record.get("bytes"), f"size mismatch: {name}")
        require(digest(path) == record.get("sha256"), f"hash mismatch: {name}")
        require(path.stat().st_mode & 0o222 == 0, f"writable package file: {name}")
    actual_names = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name != "seal.json"
    }
    require(actual_names == sealed_names, "package file set differs from seal")
    manifest = load_json(package / "package.json")
    require(
        manifest["execution"]["current_invocation_count"] == 0,
        "invocation count is not zero",
    )
    require(manifest["execution"]["cardinality"] == 1, "cardinality is not one")
    return seal, manifest


def validate_receipt_namespace(mode: str, manifest: dict) -> dict | None:
    receipt_root = Path(manifest["namespaces"]["receipt_namespace"])
    if mode in {"package", "review"}:
        require(
            not receipt_root.exists(),
            f"{mode} mode requires absent receipt namespace",
        )
        return None
    require(
        receipt_root.is_dir() and not receipt_root.is_symlink(),
        "active receipt namespace is missing",
    )
    task_id = manifest["identity"]["task_id"]
    receipt_path = receipt_root / f"{task_id}.json"
    log_dir = receipt_root / f"{task_id}_logs"
    require(
        {path.name for path in receipt_root.iterdir()}
        == {receipt_path.name, log_dir.name},
        "launch receipt namespace contains foreign or extra entries",
    )
    require(
        receipt_path.is_file() and not receipt_path.is_symlink(),
        "active task receipt is missing",
    )
    require(
        log_dir.is_dir() and not log_dir.is_symlink(),
        "active task log directory is missing",
    )
    require(
        {path.name for path in log_dir.iterdir()} == {"stdout.log", "stderr.log"},
        "active task log directory differs from pre-child durable ordering",
    )
    for name in ("stdout.log", "stderr.log"):
        path = log_dir / name
        require(
            path.is_file() and not path.is_symlink(),
            f"active task log is missing: {name}",
        )
    receipt = load_json(receipt_path)
    require(receipt.get("task_id") == task_id, "active receipt task mismatch")
    require(receipt.get("mode") == "direct", "active receipt mode mismatch")
    require(
        receipt.get("cwd") == manifest["execution"]["cwd"],
        "active receipt cwd mismatch",
    )
    require(
        receipt.get("command") == manifest["execution"]["launcher_argv"][0],
        "active receipt command mismatch",
    )
    run_id = receipt.get("run_id")
    require(
        isinstance(run_id, str)
        and run_id.startswith(task_id + "-")
        and run_id[len(task_id) + 1 :].isdigit(),
        "active receipt run id mismatch",
    )
    state = receipt.get("state")
    require(state in {"starting", "running"}, "active receipt is already terminal")
    submitted_at = receipt.get("submitted_at")
    recent_submitter = False
    if state == "starting" and isinstance(submitted_at, (int, float)):
        recent_submitter = (
            _pid_alive(receipt.get("submitter_pid"))
            and 0 <= time.time() - float(submitted_at) <= 120
        )
    require(
        _pid_alive(receipt.get("worker_pid"))
        or _pid_alive(receipt.get("pid"))
        or recent_submitter,
        "active receipt has no live durable-runner process",
    )
    for key, path in (
        ("stdout_log", log_dir / "stdout.log"),
        ("stderr_log", log_dir / "stderr.log"),
    ):
        if key in receipt:
            require(Path(receipt[key]) == path, f"active receipt {key} mismatch")
    return receipt


def validate_review_namespace(
    mode: str, package: Path, manifest: dict
) -> tuple[Path, dict] | None:
    review_root = Path(manifest["namespaces"]["review_namespace"])
    if mode == "package":
        require(
            not review_root.exists(),
            "package mode requires absent review namespace",
        )
        return None
    if not review_root.exists():
        require(mode == "review", "launch mode requires accepted review namespace")
        return None
    require(
        review_root.is_dir() and not review_root.is_symlink(),
        "review namespace is not a regular directory",
    )
    require(
        {path.name for path in review_root.iterdir()} == {"review.json"},
        "review namespace must contain only review.json",
    )
    review_path = review_root / "review.json"
    review = load_json(review_path)
    require(review_path.stat().st_mode & 0o222 == 0, "accepted review is writable")
    expected = {
        "schema_version": 1,
        "kind": "ace3_model24_r10_independent_l2_review",
        "candidate": str(package.parent),
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "execution_invocations": 0,
        "execution_authority_withheld": True,
    }
    for key, value in expected.items():
        require(review.get(key) == value, f"review binding mismatch: {key}")
    verdict = review.get("verdict")
    require(
        verdict in {"ACCEPT", "REJECT"},
        "review verdict must be ACCEPT or REJECT",
    )
    bound = review.get("bound_hashes")
    require(isinstance(bound, dict), "review bound_hashes are missing")
    required_hashes = {
        "package_manifest_sha256": digest(package / "package.json"),
        "package_seal_sha256": digest(package / "seal.json"),
        "review_request_sha256": digest(package / "review-request.json"),
        "launcher_sha256": digest(package / "launch-once.sh"),
        "validator_sha256": digest(package / "validate-package.py"),
        "launch_contract_sha256": digest(package / "launch-contract.json"),
    }
    for key, value in required_hashes.items():
        require(bound.get(key) == value, f"review hash mismatch: {key}")
    require(
        review.get("manager_may_issue_execution_authority") is (verdict == "ACCEPT"),
        "review authority disposition mismatch",
    )
    if mode == "launch":
        require(verdict == "ACCEPT", "launch mode requires an accepted review")
    return review_path, review


def expected_authority(package: Path, manifest: dict, review_path: Path) -> dict:
    return {
        "schema_version": 1,
        "kind": "ace3_model24_r10_execution_authority",
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "package_acceptance_path": str(review_path),
        "package_acceptance_sha256": digest(review_path),
        "package_manifest_sha256": digest(package / "package.json"),
        "package_seal_sha256": digest(package / "seal.json"),
        "launcher_argv": manifest["execution"]["launcher_argv"],
        "cardinality": 1,
        "manager_exactly_once_directive": True,
    }


def validate_authority(
    mode: str,
    package: Path,
    manifest: dict,
    review_state,
    authority_arg: Path | None,
) -> dict | None:
    authority_path = Path(manifest["namespaces"]["authority_namespace"])
    if mode in {"package", "review"}:
        require(authority_arg is None, f"{mode} mode forbids --require-authority")
        require(
            not authority_path.exists(),
            f"{mode} mode requires absent authority namespace",
        )
        return None
    require(authority_arg is not None, "launch mode requires --require-authority")
    require(
        authority_arg.resolve() == authority_path.resolve(),
        "authority namespace mismatch",
    )
    require(review_state is not None, "launch mode lacks accepted review")
    authority = load_json(authority_path)
    require(
        authority_path.stat().st_mode & 0o222 == 0,
        "execution authority is writable",
    )
    contract = load_json(package / "launch-contract.json")
    required_fields = contract["execution_authority_schema"]["required_fields"]
    require(
        set(authority) == set(required_fields),
        "execution authority fields differ from schema",
    )
    require(
        authority == expected_authority(package, manifest, review_state[0]),
        "execution authority does not exactly bind accepted package",
    )
    return authority


def validate_output(
    mode: str, manifest: dict, invocation_output: Path | None
) -> None:
    output = Path(manifest["namespaces"]["output_namespace"])
    if mode in {"package", "review"}:
        require(
            invocation_output is None,
            f"{mode} mode forbids --invocation-output",
        )
        require(
            not output.exists(),
            f"{mode} mode requires absent output namespace",
        )
        return
    require(invocation_output is not None, "launch mode requires --invocation-output")
    require(
        invocation_output.resolve() == output.resolve(),
        "invocation output namespace mismatch",
    )
    require(
        output.is_dir() and not output.is_symlink(),
        "claimed output namespace is missing",
    )
    require(not any(output.iterdir()), "claimed output namespace is not empty")


def validate_package(
    package: Path,
    mode: str,
    authority_arg: Path | None = None,
    invocation_output: Path | None = None,
) -> dict:
    package = package.resolve(strict=True)
    _seal, manifest = validate_seal(package)
    receipt = validate_receipt_namespace(mode, manifest)
    review_state = validate_review_namespace(mode, package, manifest)
    authority = validate_authority(
        mode, package, manifest, review_state, authority_arg
    )
    validate_output(mode, manifest, invocation_output)
    return {
        "mode": mode,
        "receipt": None if receipt is None else receipt["run_id"],
        "review": None if review_state is None else review_state[1]["verdict"],
        "authority": authority is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument(
        "--mode", required=True, choices=("package", "review", "launch")
    )
    parser.add_argument("--require-authority", type=Path)
    parser.add_argument("--invocation-output", type=Path)
    args = parser.parse_args()
    result = validate_package(
        args.package, args.mode, args.require_authority, args.invocation_output
    )
    print(
        "MODEL24_R10_PACKAGE_VALID "
        + " ".join(f"{key}={value}" for key, value in result.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


TERMINAL_SEALER_SOURCE = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def artifact(path: Path) -> dict[str, object] | None:
    if not path.is_file() or path.is_symlink():
        return None
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        return {}
    value = json.loads(path.read_text(encoding="ascii"))
    return value if isinstance(value, dict) else {}


def seal(
    package: Path,
    output: Path,
    task_id: str,
    nonce: str,
    phase: str,
    exit_code: int,
) -> dict:
    manifest = load_json(package / "package.json")
    namespaces = manifest.get("namespaces", {})
    receipt_root = Path(
        namespaces.get("receipt_namespace", package.parent / ".argus_subagents")
    )
    review_root = Path(namespaces.get("review_namespace", ""))
    authority_path = Path(namespaces.get("authority_namespace", ""))
    receipt_path = receipt_root / f"{task_id}.json"
    log_dir = receipt_root / f"{task_id}_logs"
    receipt = load_json(receipt_path)
    output_files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "terminal.json":
            record = artifact(path)
            if record is not None:
                output_files.append(record)
    payload = {
        "schema_version": 1,
        "kind": "ace3_model24_r10_terminal",
        "task_id": task_id,
        "nonce": nonce,
        "run_id": receipt.get("run_id"),
        "durable_receipt_state": receipt.get("state"),
        "status": "COMPLETE" if exit_code == 0 else "FAILED",
        "first_failure_phase": None if exit_code == 0 else phase,
        "exit_code": exit_code,
        "cardinality": 1,
        "invocation_count": 1,
        "retry": False,
        "replay": False,
        "resume": False,
        "package_manifest": artifact(package / "package.json"),
        "package_seal": artifact(package / "seal.json"),
        "package_acceptance": artifact(review_root / "review.json"),
        "execution_authority": artifact(authority_path),
        "durable_receipt": artifact(receipt_path),
        "durable_stdout": artifact(log_dir / "stdout.log"),
        "durable_stderr": artifact(log_dir / "stderr.log"),
        "controller_terminal": artifact(output / "terminal.txt"),
        "controller_stdout": artifact(output / "stdout.log"),
        "controller_stderr": artifact(output / "stderr.log"),
        "output_files_before_terminal": output_files,
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    terminal = output / "terminal.json"
    with terminal.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()
    seal(
        args.package,
        args.output,
        args.task_id,
        args.nonce,
        args.phase,
        args.exit_code,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


FOCUSED_TEST_SOURCE = r'''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "receiptfix_validator", PACKAGE / "validate-package.py"
)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    path.chmod(mode)


class LaunchContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name) / "prep"
        self.package = self.base / "package"
        self.package.mkdir(parents=True)
        self.output = Path(self.temporary.name) / "output"
        self.review = Path(self.temporary.name) / "review"
        self.authority = Path(self.temporary.name) / "authority.json"
        self.task = "ace3-model24-r10-test-1234"
        self.nonce = "0123456789abcdef"
        (self.package / "validate-package.py").write_text(
            "validator\n", encoding="ascii"
        )
        (self.package / "launch-once.sh").write_text(
            "#!/bin/sh\n", encoding="ascii"
        )
        (self.package / "review-request.json").write_text("{}\n", encoding="ascii")
        (self.package / "launch-contract.json").write_bytes(
            (PACKAGE / "launch-contract.json").read_bytes()
        )
        self.manifest = {
            "identity": {"nonce": self.nonce, "task_id": self.task},
            "execution": {
                "cwd": str(self.base),
                "launcher_argv": [str(self.package / "launch-once.sh")],
                "current_invocation_count": 0,
                "cardinality": 1,
            },
            "namespaces": {
                "receipt_namespace": str(self.base / ".argus_subagents"),
                "review_namespace": str(self.review),
                "authority_namespace": str(self.authority),
                "output_namespace": str(self.output),
            },
        }
        write_json(self.package / "package.json", self.manifest)
        for path in self.package.iterdir():
            path.chmod(0o555 if path.suffix in {".py", ".sh"} else 0o444)
        records = []
        for path in sorted(self.package.iterdir()):
            records.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
        write_json(self.package / "seal.json", {"package_files": records})

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def add_review(self, verdict: str = "ACCEPT") -> Path:
        bound = {
            "package_manifest_sha256": digest(self.package / "package.json"),
            "package_seal_sha256": digest(self.package / "seal.json"),
            "review_request_sha256": digest(self.package / "review-request.json"),
            "launcher_sha256": digest(self.package / "launch-once.sh"),
            "validator_sha256": digest(self.package / "validate-package.py"),
            "launch_contract_sha256": digest(self.package / "launch-contract.json"),
        }
        path = self.review / "review.json"
        write_json(
            path,
            {
                "schema_version": 1,
                "kind": "ace3_model24_r10_independent_l2_review",
                "candidate": str(self.base),
                "nonce": self.nonce,
                "task_id": self.task,
                "verdict": verdict,
                "execution_invocations": 0,
                "execution_authority_withheld": True,
                "manager_may_issue_execution_authority": verdict == "ACCEPT",
                "bound_hashes": bound,
            },
        )
        return path

    def add_receipt(self) -> None:
        root = self.base / ".argus_subagents"
        logs = root / f"{self.task}_logs"
        logs.mkdir(parents=True)
        (logs / "stdout.log").write_text("", encoding="ascii")
        (logs / "stderr.log").write_text("", encoding="ascii")
        write_json(
            root / f"{self.task}.json",
            {
                "state": "running",
                "task_id": self.task,
                "run_id": f"{self.task}-123456789",
                "mode": "direct",
                "cwd": str(self.base),
                "command": str(self.package / "launch-once.sh"),
                "worker_pid": os.getpid(),
                "pid": os.getpid(),
                "submitted_at": time.time(),
                "stdout_log": str(logs / "stdout.log"),
                "stderr_log": str(logs / "stderr.log"),
            },
            mode=0o644,
        )

    def test_package_review_and_launch_modes(self) -> None:
        result = VALIDATOR.validate_package(self.package, "package")
        self.assertEqual(result["mode"], "package")
        self.add_review("REJECT")
        result = VALIDATOR.validate_package(self.package, "review")
        self.assertEqual(result["review"], "REJECT")

    def test_launch_accepts_exact_active_runner_namespace_and_bound_authority(self) -> None:
        review_path = self.add_review()
        self.add_receipt()
        self.output.mkdir()
        write_json(
            self.authority,
            VALIDATOR.expected_authority(self.package, self.manifest, review_path),
        )
        result = VALIDATOR.validate_package(
            self.package, "launch", self.authority, self.output
        )
        self.assertTrue(result["authority"])
        self.assertEqual(result["review"], "ACCEPT")

    def test_launch_rejects_extra_receipt_entry_and_unbound_acceptance(self) -> None:
        review_path = self.add_review()
        self.add_receipt()
        self.output.mkdir()
        authority = VALIDATOR.expected_authority(
            self.package, self.manifest, review_path
        )
        authority["package_acceptance_sha256"] = "0" * 64
        write_json(self.authority, authority)
        with self.assertRaisesRegex(SystemExit, "exactly bind"):
            VALIDATOR.validate_package(
                self.package, "launch", self.authority, self.output
            )
        self.authority.unlink()
        write_json(
            self.authority,
            VALIDATOR.expected_authority(self.package, self.manifest, review_path),
        )
        (self.base / ".argus_subagents/foreign.json").write_text(
            "{}\n", encoding="ascii"
        )
        with self.assertRaisesRegex(SystemExit, "foreign or extra"):
            VALIDATOR.validate_package(
                self.package, "launch", self.authority, self.output
            )


if __name__ == "__main__":
    unittest.main()
'''


def render_launcher(base: Path, output: Path, nonce: str) -> bytes:
    task_id = f"ace3-model24-r10-42c895c-{nonce}"
    authority = Path(
        f"/home/argustest/ace3-model24-r10-authority-20260828-{nonce}.json"
    )
    script = f'''#!/usr/bin/env bash
set -euo pipefail

PREP={base}
PACKAGE="$PREP/package"
OUTPUT={output}
AUTHORITY={authority}
TASK_ID={task_id}
NONCE={nonce}
PHASE=invocation_claim

mkdir "$OUTPUT"
seal_terminal() {{
    local code=$?
    trap - EXIT
    if ! {MODEL_PYTHON} -B "$PACKAGE/seal-terminal.py" \\
        --package "$PACKAGE" \\
        --output "$OUTPUT" \\
        --task-id "$TASK_ID" \\
        --nonce "$NONCE" \\
        --phase "$PHASE" \\
        --exit-code "$code"; then
        exit 125
    fi
    exit "$code"
}}
trap seal_terminal EXIT

PHASE=package_validation
{MODEL_PYTHON} -B "$PACKAGE/validate-package.py" \\
    --package "$PACKAGE" \\
    --mode launch \\
    --require-authority "$AUTHORITY" \\
    --invocation-output "$OUTPUT"
PHASE=resource_gates
ulimit -n 4096
ulimit -v 100663296
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PHASE=model24_rtl_cascade
timeout --kill-after=120s 18000s {MODEL_PYTHON} -B \\
    "$PREP/source/ace3/model/controller_model24_rtl_cascade.py" \\
    --repository-root "$PREP/source" \\
    --checkpoint {CHECKPOINT} \\
    --tensor-map "$PREP/source/ace3/contracts/model24_tensor_map.json" \\
    --bindings "$PACKAGE/bindings.json" \\
    --simulation-dir "$PACKAGE/controller-inputs" \\
    --output-dir "$OUTPUT" \\
    --fresh >"$OUTPUT/stdout.log" 2>"$OUTPUT/stderr.log"
PHASE=natural_terminal
test -f "$OUTPUT/terminal.txt"
'''
    return script.encode("utf-8")


def patch_prepare_source(payload: bytes, base: Path, nonce: str) -> bytes:
    output = Path(f"/home/argustest/ace3-model24-r10-output-20260828-{nonce}")
    source = transformed_template(payload, base, output, nonce).decode("utf-8")
    declarations = "".join(
        f'    PACKAGE / "{name}",\n' for name in STATIC_PACKAGE_NAMES
    )
    declarations += (
        '    PREFLIGHT / "run-layer0-preflight.py",\n'
        '    PREFLIGHT / "layer0-materializer.json",\n'
    )
    marker = "GENERATED_WRITE_TARGETS = (\n"
    require(marker in source, "generated target declaration marker is missing")
    source = source.replace(marker, marker + declarations, 1)

    protected_marker = "    paths = historical + r10\n"
    require(protected_marker in source, "protected-path marker is missing")
    source = source.replace(
        protected_marker,
        protected_marker.rstrip("\n")
        + f' + [Path("{REVIEWED_REVIEW.parent}")]\n',
        1,
    )

    run_template_marker = (
        '            "review_namespace": str(REVIEW),\n'
        "        }\n"
    )
    require(run_template_marker in source, "run-template marker is missing")
    source = source.replace(
        run_template_marker,
        '            "review_namespace": str(REVIEW),\n'
        '            "package_acceptance_path": str(REVIEW / "review.json"),\n'
        '            "validation_modes": ["package", "review", "launch"],\n'
        '            "launch_contract": str(PACKAGE / "launch-contract.json"),\n'
        "        }\n",
        1,
    )

    package_marker = '            "protected_history": {\n'
    require(package_marker in source, "package launch-contract marker is missing")
    source = source.replace(
        package_marker,
        '            "launch_contract": {\n'
        '                "path": str(PACKAGE / "launch-contract.json"),\n'
        '                "sha256": hash_file(PACKAGE / "launch-contract.json")["sha256"],\n'
        '                "validator": str(PACKAGE / "validate-package.py"),\n'
        '                "validator_sha256": hash_file(PACKAGE / "validate-package.py")["sha256"],\n'
        '                "modes": ["package", "review", "launch"],\n'
        '                "launch_receipt_state": "exact active durable task receipt plus stdout/stderr log directory",\n'
        '                "accepted_review_path": str(REVIEW / "review.json"),\n'
        '                "authority_binds_package_acceptance_sha256": True,\n'
        '            },\n'
        '            "repair_origin": {\n'
        f'                "reviewed_candidate": "{REVIEWED_PREP}",\n'
        f'                "reviewed_package_seal_sha256": "{REVIEWED_SEAL_SHA256}",\n'
        f'                "rejected_review": "{REVIEWED_REVIEW}",\n'
        f'                "rejected_review_sha256": "{REVIEWED_REVIEW_SHA256}",\n'
        '                "review_verdict": "REJECT",\n'
        '                "repaired_contradiction": "launch_time_namespace_contract",\n'
        '                "prior_evidence_mutated": False,\n'
        '            },\n'
        + package_marker,
        1,
    )

    request_marker = (
        '            "execution_authority_withheld": True,\n'
        '            "stage_transition": False,\n'
        "        })\n"
    )
    require(request_marker in source, "review-request marker is missing")
    source = source.replace(
        request_marker,
        '            "execution_authority_withheld": True,\n'
        '            "validation_modes": ["package", "review", "launch"],\n'
        '            "launch_contract_sha256": hash_file(PACKAGE / "launch-contract.json")["sha256"],\n'
        '            "launcher_sha256": hash_file(PACKAGE / "launch-once.sh")["sha256"],\n'
        '            "validator_sha256": hash_file(PACKAGE / "validate-package.py")["sha256"],\n'
        '            "accepted_review_contract": {\n'
        '                "path": str(REVIEW / "review.json"),\n'
        '                "namespace_entries": ["review.json"],\n'
        '                "required_bound_hashes": [\n'
        '                    "package_manifest_sha256", "package_seal_sha256",\n'
        '                    "review_request_sha256", "launcher_sha256",\n'
        '                    "validator_sha256", "launch_contract_sha256",\n'
        '                ],\n'
        '                "package_acceptance_is_execution_authority": False,\n'
        '            },\n'
        '            "stage_transition": False,\n'
        "        })\n",
        1,
    )
    return source.encode("utf-8")


def runner_contract() -> list[dict[str, object]]:
    records = []
    for name, expected in RUNNER_SOURCES.items():
        path = RUNNER_ROOT / name
        require(
            path.is_file(),
            f"canonical durable-runner source missing: {path}",
        )
        require(
            digest(path) == expected,
            f"canonical durable-runner source changed: {name}",
        )
        records.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": expected}
        )
    return records


def prepare(base: Path, output: Path, nonce: str) -> dict[str, object]:
    authority = Path(
        f"/home/argustest/ace3-model24-r10-authority-20260828-{nonce}.json"
    )
    review = Path(f"/home/argustest/ace3-model24-r10-review-20260828-{nonce}")
    require(not base.exists(), f"destination already exists: {base}")
    for path in (output, authority, review):
        require(not path.exists(), f"fresh dynamic namespace already exists: {path}")
    require(REVIEWED_PREP.is_dir(), "reviewed r10 candidate is unavailable")
    require(
        digest(REVIEWED_REVIEW) == REVIEWED_REVIEW_SHA256,
        "reviewed L2 evidence changed",
    )
    require(
        digest(REVIEWED_SEAL) == REVIEWED_SEAL_SHA256,
        "reviewed package seal changed",
    )
    require(
        MODEL_PYTHON.is_file() and os.access(MODEL_PYTHON, os.X_OK),
        "canonical model Python is unavailable",
    )
    require(
        ARGUS_PYTHON.is_file() and os.access(ARGUS_PYTHON, os.X_OK),
        "canonical durable-runner Python is unavailable",
    )
    require(LAUNCH_CONTRACT.is_file(), "repository launch contract is missing")
    protected_before = {
        "reviewed_candidate": snapshot_tree(REVIEWED_PREP),
        "reviewed_review": snapshot_tree(REVIEWED_REVIEW.parent),
    }
    runner_records = runner_contract()

    source = base / "source"
    package = base / "package"
    preflight = base / "preflight"
    archive = base / "source-export.tar"
    base.mkdir(mode=0o700)
    source.mkdir(mode=0o700)
    package.mkdir(mode=0o700)
    preflight.mkdir(mode=0o700)

    commit = subprocess.run(
        ["git", "rev-parse", f"{COMMIT}^{{commit}}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", f"{COMMIT}^{{tree}}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(commit == COMMIT and tree == TREE, "published source authority changed")
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={archive}", COMMIT],
        cwd=REPOSITORY,
        check=True,
    )
    with tarfile.open(archive, "r") as stream:
        stream.extractall(source, filter="data")

    old_package = REVIEWED_PREP / "package"
    old_preflight = REVIEWED_PREP / "preflight"
    task_id = f"ace3-model24-r10-42c895c-{nonce}"
    static_payloads = {
        package / "launch-contract.json": LAUNCH_CONTRACT.read_bytes(),
        package / "launch-once.sh": render_launcher(base, output, nonce),
        package / "prepare-and-seal.py": patch_prepare_source(
            (old_package / "prepare-and-seal.py").read_bytes(), base, nonce
        ),
        package / "receiptfix-builder.py": Path(__file__).read_bytes(),
        package / "seal-terminal.py": TERMINAL_SEALER_SOURCE.encode("utf-8"),
        package / "test-launch-contract.py": FOCUSED_TEST_SOURCE.encode("utf-8"),
        package / "validate-package.py": VALIDATOR_SOURCE.encode("utf-8"),
        preflight / "run-layer0-preflight.py": transformed_template(
            (old_preflight / "run-layer0-preflight.py").read_bytes(),
            base,
            output,
            nonce,
        ),
    }
    construction_path = package / "receiptfix-construction.json"
    declared_static = sorted(
        str(path) for path in (*static_payloads, construction_path)
    )
    construction = {
        "schema_version": 1,
        "kind": "ace3_model24_r10_receiptfix_construction",
        "nonce": nonce,
        "task_id": task_id,
        "declared_static_write_targets": declared_static,
        "all_static_write_parents_exist": all(
            path.parent.is_dir() for path in static_payloads
        ),
        "reviewed_evidence_before": protected_before,
        "reviewed_review_sha256": REVIEWED_REVIEW_SHA256,
        "reviewed_package_seal_sha256": REVIEWED_SEAL_SHA256,
        "canonical_durable_runner_sources": runner_records,
        "execution_invocations": 0,
        "simulator_invocations": 0,
        "model_invocations": 0,
        "subagent_invocations": 0,
        "package_launch_invocations": 0,
        "allowed_construction_subprocesses": [
            "git rev-parse",
            "git archive",
            "guarded layer-0 preflight",
            "preparation and seal",
        ],
    }
    static_payloads[construction_path] = canonical_json(construction)
    declared_set = frozenset(static_payloads)
    require(len(declared_set) == len(static_payloads), "duplicate static write target")
    for path, payload in static_payloads.items():
        require(path in declared_set, f"undeclared static write target: {path}")
        mode = 0o700 if path.suffix in {".py", ".sh"} else 0o600
        write_new(path, payload, mode)

    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    preflight_result = subprocess.run(
        [str(MODEL_PYTHON), "-B", str(preflight / "run-layer0-preflight.py")],
        cwd=base,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    preparation_result = subprocess.run(
        [str(MODEL_PYTHON), "-B", str(package / "prepare-and-seal.py")],
        cwd=base,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    protected_after = {
        "reviewed_candidate": snapshot_tree(REVIEWED_PREP),
        "reviewed_review": snapshot_tree(REVIEWED_REVIEW.parent),
    }
    require(
        protected_after == protected_before,
        "reviewed r10 evidence changed during repair preparation",
    )
    for path in (base / ".argus_subagents", output, authority, review):
        require(not path.exists(), f"execution/review side effect created: {path}")
    terminal = json.loads(
        (preflight / "preparation-terminal.json").read_text(encoding="ascii")
    )
    require(
        terminal.get("execution_invocations") == 0,
        "preparation terminal reports execution",
    )
    result = json.loads(preparation_result.stdout.strip())
    require(result.get("zero_state") is True, "prepared package is not in zero state")
    return {
        "base": str(base),
        "package": str(package),
        "output": str(output),
        "authority": str(authority),
        "review": str(review),
        "task_id": task_id,
        "nonce": nonce,
        "seal_sha256": digest(package / "seal.json"),
        "preparation_terminal_sha256": digest(
            preflight / "preparation-terminal.json"
        ),
        "preflight_stdout": preflight_result.stdout.strip(),
        "execution_invocations": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", default=secrets.token_hex(8))
    args = parser.parse_args()
    require(
        len(args.nonce) == 16
        and all(character in "0123456789abcdef" for character in args.nonce),
        "nonce must be 16 lowercase hexadecimal characters",
    )
    base = Path(
        f"/home/argustest/ace3-model24-r10-prep-20260828-{args.nonce}"
    )
    output = Path(
        f"/home/argustest/ace3-model24-r10-output-20260828-{args.nonce}"
    )
    print(json.dumps(prepare(base, output, args.nonce), sort_keys=True))


if __name__ == "__main__":
    main()
