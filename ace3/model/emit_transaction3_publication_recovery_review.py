#!/usr/bin/env python3
"""Independently review a sealed transaction-003 publication-only package."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping


PACKAGE_MEMBERS = {
    "adjudication.json",
    "package-manifest.json",
    "publication_recovery.py",
    "review-emitter.py",
    "review-request.json",
}
EXPECTED_TREE = {
    "file_count": 69,
    "total_bytes": 25208783,
    "tree_sha256": "3be5e041f076b16d14553de04865c1beb4a7833b9d4dace01c69bbd4e3f3accb",
}
EXPECTED_HIDDEN_SHA256 = (
    "95e7911ce6afbc8dba048e3910b22489288fedfdeee33b47021622d937b67463"
)
FORBIDDEN_IMPORTS = {
    "importlib",
    "numpy",
    "safetensors",
    "subprocess",
    "torch",
}
FORBIDDEN_CALLS = {
    "compile",
    "eval",
    "exec",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "popen",
    "posix_spawn",
    "posix_spawnp",
    "system",
}
REVIEWER_FORBIDDEN_CALLS = FORBIDDEN_CALLS | {
    "flock",
    "mkdir",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "rmtree",
    "unlink",
}


class ReviewError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def canonical_json(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("ascii")


def load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in document, f"duplicate JSON key {key}: {path}")
            document[key] = value
        return document

    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"regular file required: {path}")
    require(not stat.S_ISLNK(metadata.st_mode), f"symlink rejected: {path}")
    document = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    require(isinstance(document, dict), f"JSON object required: {path}")
    return document


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"regular file required: {path}")
    require(not stat.S_ISLNK(metadata.st_mode), f"symlink rejected: {path}")
    return {
        "path": str(path),
        "bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def authenticate_record(record: Mapping[str, Any], label: str) -> None:
    require(
        set(("path", "bytes", "sha256")).issubset(record),
        f"{label} record is incomplete",
    )
    path = Path(record["path"])
    actual = file_record(path)
    require(
        all(actual[key] == record[key] for key in actual),
        f"{label} authentication differs",
    )
    if "mode" in record:
        require(
            f"{stat.S_IMODE(path.stat().st_mode):04o}" == record["mode"],
            f"{label} mode differs",
        )


def source_boundary(payload: str, reviewer_only: bool) -> None:
    tree = ast.parse(payload)
    forbidden_calls = REVIEWER_FORBIDDEN_CALLS if reviewer_only else FORBIDDEN_CALLS
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            require(
                all(
                    alias.name.split(".", 1)[0] not in FORBIDDEN_IMPORTS
                    for alias in node.names
                ),
                "source imports executable, model, oracle, or RTL support",
            )
        elif isinstance(node, ast.ImportFrom):
            require(
                (node.module or "").split(".", 1)[0] not in FORBIDDEN_IMPORTS,
                "source imports executable, model, oracle, or RTL support",
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = ""
            require(
                name.lower() not in forbidden_calls,
                f"forbidden source call: {name}",
            )


def terminal_fields(payload: bytes) -> dict[str, str]:
    lines = payload.decode("ascii").splitlines()
    require(len(lines) == 1, "terminal evidence must contain exactly one row")
    fields: dict[str, str] = {}
    for item in lines[0].split():
        require(item.count("=") == 1, "terminal field is malformed")
        key, value = item.split("=", 1)
        require(key not in fields, f"duplicate terminal field: {key}")
        fields[key] = value
    require(
        fields.get("natural_terminal") == "1"
        and fields.get("exit_code") == "0"
        and fields.get("done_count") == "1"
        and fields.get("final_count") == "896",
        "natural RTL terminal facts differ",
    )
    return fields


def comparison_facts(document: Mapping[str, Any]) -> None:
    require(
        document.get("integer_mismatches") == 0
        and document.get("rtl_matches_exact_integer_oracle") is True
        and document.get("exact_integer_oracle_output_sha256")
        == EXPECTED_HIDDEN_SHA256,
        "exact-oracle comparison differs",
    )


def one_inventory_record(
    inventory: list[dict[str, Any]], suffix: str
) -> dict[str, Any]:
    matches = [
        record for record in inventory if str(record.get("path", "")).endswith(suffix)
    ]
    require(len(matches) == 1, f"inventory member is not unique: {suffix}")
    return matches[0]


def evidence_records(document: object) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    if isinstance(document, dict):
        if set(("path", "bytes", "sha256")).issubset(document):
            records.append(document)
        else:
            for value in document.values():
                records.extend(evidence_records(value))
    elif isinstance(document, list):
        for value in document:
            records.extend(evidence_records(value))
    return records


def validate_adjudication(
    adjudication: Mapping[str, Any], authenticate: bool = True
) -> None:
    require(
        adjudication.get("schema_version") == 1
        and adjudication.get("kind")
        == "ace3_transaction3_postconsume_publication_recovery_adjudication"
        and adjudication.get("status") == "PASS"
        and adjudication.get("transaction_index") == 3,
        "adjudication identity differs",
    )
    parent = adjudication.get("authoritative_parent", {})
    consumed = adjudication.get("consumed_r5", {})
    finding = adjudication.get("finding", {})
    receipt = adjudication.get("reconstructed_receipt", {})
    preview = adjudication.get("publication_preview", {})
    require(
        parent.get("generation") == 3 and parent.get("cursor") == 3,
        "wrong parent generation or cursor",
    )
    require(consumed.get("replay_authorized") is False, "replay is not rejected")
    require(
        finding
        == {
            "natural_terminal": 1,
            "rtl_exit_code": 0,
            "done_count": 1,
            "final_count": 896,
            "integer_mismatches": 0,
            "exact_oracle_agreement": True,
            "sole_failure": "completion receipt natural_rtl_terminal mapping",
            "generation4_publication_attempted": False,
        },
        "adjudicated terminal, comparison, or sole-failure finding differs",
    )
    require(
        preview.get("generation") == 4
        and preview.get("cursor") == 4
        and preview.get("live_publication_performed") is False
        and adjudication.get("recovery_authority_created") is False,
        "publication or recovery-authority boundary differs",
    )
    result = receipt.get("result", {})
    semantics = receipt.get("output_semantics", {})
    timing = receipt.get("timing", {})
    require(
        receipt.get("status") == "COMPLETE"
        and receipt.get("transaction_index") == 3
        and result.get("natural_rtl_terminal") is True
        and result.get("exact_integer_oracle_match") is True
        and result.get("output_hidden_elements") == 896
        and result.get("output_state_position") == 4,
        "reconstructed receipt result differs",
    )
    require(
        semantics.get("hidden", {}).get("semantic_sha256")
        == EXPECTED_HIDDEN_SHA256
        and semantics.get("state", {}).get("position") == 4,
        "reconstructed output or state semantics differ",
    )
    require(
        timing.get("started_at") is None
        and timing.get("completed_at") is None
        and timing.get("transaction_seconds") is None,
        "reconstructed receipt timing makes an unsupported claim",
    )
    frozen = adjudication.get("frozen_transaction003", {})
    inventory = frozen.get("inventory")
    require(isinstance(inventory, list), "frozen inventory is absent")
    tree = frozen.get("tree", {})
    require(
        all(tree.get(key) == value for key, value in EXPECTED_TREE.items()),
        "frozen tree summary differs",
    )
    require(len(inventory) == EXPECTED_TREE["file_count"], "partial inventory")
    hidden_record = one_inventory_record(inventory, "/position003/raw/final.hex")
    hidden_output = receipt.get("outputs", {}).get("hidden", {})
    state_record = receipt.get("outputs", {}).get("state")
    require(
        isinstance(hidden_output, dict)
        and all(
            hidden_output.get(key) == value
            for key, value in file_record(Path(hidden_record["path"])).items()
        ),
        "receipt hidden output binding differs",
    )
    state_matches = [
        record
        for record in inventory
        if isinstance(state_record, dict)
        and record.get("path") == state_record.get("path")
    ]
    require(
        len(state_matches) == 1
        and state_record == file_record(Path(state_matches[0]["path"])),
        "receipt state output binding differs",
    )
    if authenticate:
        for index, record in enumerate(evidence_records(adjudication)):
            authenticate_record(record, f"adjudication evidence {index}")
        paths = {Path(record["path"]) for record in inventory}
        root = Path(os.path.commonpath([str(path) for path in paths]))
        require(root.name == "transaction-003", "transaction evidence root differs")
        actual_paths = {
            path for path in root.rglob("*") if path.is_file() or path.is_symlink()
        }
        require(actual_paths == paths, "partial or additional transaction artifacts")
        terminal_record = one_inventory_record(
            inventory, "/position003/raw/terminal.txt"
        )
        terminal_fields(Path(terminal_record["path"]).read_bytes())
        comparison_record = one_inventory_record(inventory, "/comparison.json")
        comparison_facts(load_json(Path(comparison_record["path"])))
        raw_trace = one_inventory_record(inventory, "/position003/raw/trace.hex")
        oracle_trace = one_inventory_record(
            inventory, "/position003/exact_oracle/trace.hex"
        )
        raw_final = one_inventory_record(inventory, "/position003/raw/final.hex")
        oracle_final = one_inventory_record(
            inventory, "/position003/exact_oracle/final.hex"
        )
        require(
            Path(raw_trace["path"]).read_bytes()
            == Path(oracle_trace["path"]).read_bytes()
            and Path(raw_final["path"]).read_bytes()
            == Path(oracle_final["path"]).read_bytes(),
            "raw RTL output differs from exact oracle",
        )


def expect_rejection(operation: Callable[[], None], label: str) -> str:
    try:
        operation()
    except ReviewError:
        return label
    raise ReviewError(f"adversarial control was accepted: {label}")


def mutate(
    adjudication: Mapping[str, Any],
    operation: Callable[[dict[str, Any]], None],
    authenticate: bool = False,
) -> None:
    candidate = copy.deepcopy(adjudication)
    operation(candidate)
    validate_adjudication(candidate, authenticate=authenticate)


def adversarial_controls(
    adjudication: Mapping[str, Any], publication_source: str, emitter_source: str
) -> list[str]:
    inventory = adjudication["frozen_transaction003"]["inventory"]
    terminal_record = one_inventory_record(
        inventory, "/position003/raw/terminal.txt"
    )
    terminal_payload = Path(terminal_record["path"]).read_bytes()
    comparison_record = one_inventory_record(inventory, "/comparison.json")
    comparison = load_json(Path(comparison_record["path"]))
    controls = [
        expect_rejection(
            lambda: terminal_fields(
                b"natural_terminal=0".join(
                    terminal_payload.split(b"natural_terminal=1")
                )
            ),
            "altered-terminal",
        ),
        expect_rejection(
            lambda: comparison_facts({**comparison, "integer_mismatches": 1}),
            "altered-comparison",
        ),
        expect_rejection(
            lambda: mutate(
                adjudication,
                lambda item: item["reconstructed_receipt"]["outputs"]["hidden"].update(
                    {"sha256": "0" * 64}
                ),
            ),
            "altered-output",
        ),
        expect_rejection(
            lambda: mutate(
                adjudication,
                lambda item: item["reconstructed_receipt"]["output_semantics"][
                    "state"
                ].update({"position": 5}),
            ),
            "altered-state",
        ),
        expect_rejection(
            lambda: mutate(
                adjudication,
                lambda item: item["frozen_transaction003"]["inventory"][0].update(
                    {"sha256": "0" * 64}
                ),
                authenticate=True,
            ),
            "altered-hash",
        ),
        expect_rejection(
            lambda: mutate(
                adjudication,
                lambda item: item["frozen_transaction003"]["inventory"].pop(),
            ),
            "partial-artifacts",
        ),
        expect_rejection(
            lambda: mutate(
                adjudication,
                lambda item: item["authoritative_parent"].update({"generation": 2}),
            ),
            "wrong-parent-generation",
        ),
        expect_rejection(
            lambda: mutate(
                adjudication,
                lambda item: item["consumed_r5"].update(
                    {"replay_authorized": True}
                ),
            ),
            "replay",
        ),
        expect_rejection(
            lambda: source_boundary(
                publication_source + "\nimport subprocess\n", reviewer_only=False
            ),
            "executable-model-rtl-call",
        ),
        expect_rejection(
            lambda: source_boundary(
                emitter_source + "\nos.replace('a', 'b')\n", reviewer_only=True
            ),
            "reviewer-publication-call",
        ),
    ]
    return controls


def assess_package(
    package: Path, output: Path
) -> tuple[str, list[str], str]:
    try:
        require(package.is_absolute(), "package path must be absolute")
        require(output.is_absolute(), "review output must be absolute")
        metadata = package.lstat()
        require(
            stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
            "sealed package directory is absent",
        )
        require(
            stat.S_IMODE(metadata.st_mode) & 0o222 == 0,
            "sealed package directory is writable",
        )
        seal = load_json(package / "package-seal.json")
        require(
            seal.get("kind")
            == "ace3_transaction3_publication_only_recovery_seal"
            and seal.get("status") == "SEALED_REVIEW_REQUIRED"
            and seal.get("publication_performed") is False
            and seal.get("recovery_authority_created") is False,
            "package seal identity differs",
        )
        members = seal.get("members")
        require(
            isinstance(members, dict) and set(members) == PACKAGE_MEMBERS,
            "sealed package member set differs",
        )
        actual = {path.name for path in package.iterdir() if path.is_file()}
        require(
            actual == PACKAGE_MEMBERS | {"package-seal.json"},
            "package file set differs",
        )
        for name, record in members.items():
            require(
                record.get("path") == str(package / name),
                f"sealed package path differs: {name}",
            )
            authenticate_record(record, f"sealed package member {name}")
            require(
                stat.S_IMODE((package / name).stat().st_mode) & 0o222 == 0,
                f"sealed package member is writable: {name}",
            )
        manifest = load_json(package / "package-manifest.json")
        request = load_json(package / "review-request.json")
        require(
            manifest.get("status") == "SEALED_REVIEW_REQUIRED"
            and manifest.get("review_path") == str(output)
            and manifest.get("reviewer_emitter") == str(package / "review-emitter.py")
            and manifest.get("recovery_authority") is None
            and manifest.get("computation")
            == {"model": 0, "oracle": 0, "rtl": 0, "transaction": 0},
            "package review, computation, or authority boundary differs",
        )
        require(
            request.get("required_role") == "reviewer"
            and request.get("review_output") == str(output)
            and request.get("reviewer_emitter")
            == file_record(package / "review-emitter.py"),
            "review request binding differs",
        )
        generation = Path(manifest["publication"]["generation"])
        staging = generation.with_name(f".{generation.name}.prepared")
        require(
            not generation.exists() and not staging.exists(),
            "generation4 or its staging directory already exists",
        )
        publication_source = (package / "publication_recovery.py").read_text(
            encoding="utf-8"
        )
        emitter_source = (package / "review-emitter.py").read_text(encoding="utf-8")
        source_boundary(publication_source, reviewer_only=False)
        source_boundary(emitter_source, reviewer_only=True)
        adjudication = load_json(package / "adjudication.json")
        validate_adjudication(adjudication)
        controls = adversarial_controls(
            adjudication, publication_source, emitter_source
        )
        return "PASS", controls, "all independent checks passed"
    except ReviewError as error:
        return "REJECT", [], str(error)


def write_review(path: Path, payload: bytes) -> None:
    require(path.is_absolute(), "review output must be absolute")
    parent = path.parent
    metadata = parent.lstat()
    require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        "review output parent must be an existing real directory",
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(
        stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
        "review output is writable",
    )
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    status, controls, reason = assess_package(arguments.package, arguments.output)
    review = {
        "schema_version": 1,
        "kind": "ace3_transaction3_publication_recovery_independent_review",
        "status": status,
        "producer_role": "reviewer",
        "preparation_participation": False,
        "recovery_authority_created": False,
        "publication_performed": False,
        "package_seal": file_record(arguments.package / "package-seal.json"),
        "adjudication": file_record(arguments.package / "adjudication.json"),
        "reviewer_emitter": file_record(arguments.package / "review-emitter.py"),
        "adversarial_controls": controls,
        "reason": reason,
    }
    write_review(arguments.output, canonical_json(review))
    print(
        "TRANSACTION3_PUBLICATION_RECOVERY_REVIEW "
        f"status={status} output={arguments.output} publication=0 "
        "model=0 oracle=0 rtl=0 transaction=0 authority=0"
    )


if __name__ == "__main__":
    main()
