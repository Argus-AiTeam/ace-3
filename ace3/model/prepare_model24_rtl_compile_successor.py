#!/usr/bin/env python3
"""Build and seal a zero-authority successor to failed Model24 r15."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any


REPOSITORY = Path("/home/argustest/ace3-argus")
R15_CANDIDATE = Path(
    "/home/argustest/ace3-model24-r15-prep-20260829-f0742dfe9b4caa45"
)
R15_REVIEW = Path(
    "/home/argustest/ace3-model24-r15-review-20260829-f0742dfe9b4caa45/review.json"
)
R15_OUTPUT = Path(
    "/home/argustest/ace3-model24-r15-output-20260829-f0742dfe9b4caa45"
)
R15_TERMINAL = Path(
    "/home/argustest/ace3-model24-r15-terminal-20260829-f0742dfe9b4caa45"
)
R15_TASK_ID = "ace3-model24-r15-42c895c-f0742dfe9b4caa45"
R15_RUN_ID = f"{R15_TASK_ID}-1787987640572415506"
VALIDATOR = REPOSITORY / "ace3/model/validate_model24_rtl_compile_successor.py"
COMPILE_ARGV = [
    "make",
    "--no-print-directory",
    "model24-rtl-layer-compile",
    "MODEL24_RTL_LAYER_INDEX=0",
    "MODEL24_RTL_ACCURATE_SILU=1",
]
EXPECTED_BINARY_RELATIVE = Path(
    "build/model24_rtl_cascade/compiled/layer0/obj_dir/"
    "Vace3_decoder_layer0_token_engine"
)
SUCCESSOR_OVERLAYS = (
    Path("ace3/rtl/ace3_decoder_layer0_token_engine.sv"),
    Path("ace3/rtl/ace3_fp16_silu_gate_core.sv"),
    Path("ace3/tb/ace3_decoder_layer0_token_engine_main.cpp"),
)
CHANGED_PATHS = ["Makefile", *(path.as_posix() for path in SUCCESSOR_OVERLAYS)]
MAKEFILE_CLOSURE = """

# Model24 per-layer RTL compile closure used by controller_model24_rtl_cascade.py.
MODEL24_RTL_CASCADE_DIR := $(BUILD_DIR)/model24_rtl_cascade
MODEL24_RTL_LAYER_INDEX ?= 0
MODEL24_RTL_ACCURATE_SILU ?= 1
MODEL24_RTL_LAYER_DIR := $(MODEL24_RTL_CASCADE_DIR)/compiled/layer$(MODEL24_RTL_LAYER_INDEX)
MODEL24_RTL_LAYER_OBJ_DIR := $(MODEL24_RTL_LAYER_DIR)/obj_dir
MODEL24_RTL_LAYER_BIN := $(MODEL24_RTL_LAYER_OBJ_DIR)/Vace3_decoder_layer0_token_engine

.PHONY: model24-rtl-layer-compile
model24-rtl-layer-compile:
	@test "$(MODEL24_RTL_LAYER_INDEX)" -ge 0 -a "$(MODEL24_RTL_LAYER_INDEX)" -le 23
	@rm -rf "$(MODEL24_RTL_LAYER_OBJ_DIR)"
	@mkdir -p "$(MODEL24_RTL_LAYER_DIR)"
	@"$(VERILATOR)" --cc --exe --build --savable --Wall -Wno-fatal \\
	    -GLAYER_INDEX="$(MODEL24_RTL_LAYER_INDEX)" \\
	    -GACCURATE_SILU="$(MODEL24_RTL_ACCURATE_SILU)" \\
	    --top-module ace3_decoder_layer0_token_engine \\
	    --Mdir "$(MODEL24_RTL_LAYER_OBJ_DIR)" $(DECODER_RTL) \\
	    "$(DECODER_CPP_TB)"
	@test -x "$(MODEL24_RTL_LAYER_BIN)"
""".lstrip("\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_new(path: Path, payload: bytes, mode: int) -> None:
    require(path.parent.is_dir(), f"write parent missing: {path.parent}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(mode)


def tree_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        require(not path.is_symlink(), f"symlink forbidden in sealed tree: {path}")
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            records.append({"path": relative, "kind": "directory", "mode": mode})
        elif path.is_file():
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": mode,
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    return records


def readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o500)
        else:
            path.chmod(0o500 if path.stat().st_mode & 0o111 else 0o400)
    root.chmod(0o500)


def writable_tree(root: Path) -> None:
    root.chmod(0o700)
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
        else:
            path.chmod(0o700 if path.stat().st_mode & 0o111 else 0o600)


def r15_artifacts() -> dict[str, Path]:
    logs = R15_TERMINAL / ".argus_subagents" / f"{R15_TASK_ID}_logs"
    return {
        "compile.log": R15_OUTPUT / "rtl-cascade/layers/layer00/compile.log",
        "exit-code.txt": logs / f"exit_code.{R15_RUN_ID}",
        "launch-terminal.json": R15_OUTPUT / "launch-terminal.json",
        "package.json": R15_CANDIDATE / "package/package.json",
        "review.json": R15_REVIEW,
        "runner-receipt.json": R15_TERMINAL / ".argus_subagents" / f"{R15_TASK_ID}.json",
        "seal.json": R15_CANDIDATE / "package/seal.json",
        "source-tree.json": R15_CANDIDATE / "package/source-tree.json",
        "stderr.log": logs / "stderr.log",
        "stdout.log": logs / "stdout.log",
        "terminal-manifest.json": R15_TERMINAL / "manifest.json",
    }


def toolchain_record() -> dict[str, Any]:
    record: dict[str, Any] = {}
    for name, command in (
        ("make", ["make", "--version"]),
        ("verilator", ["verilator", "--version"]),
        ("cxx", ["c++", "--version"]),
    ):
        path = shutil.which(command[0])
        result = subprocess.run(command, capture_output=True, check=False)
        record[name] = {
            "path": path,
            "exit_code": result.returncode,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
        }
    return record


def prepare(output: Path) -> dict[str, Any]:
    output = output.absolute()
    require(not output.exists(), f"successor output already exists: {output}")
    require(REPOSITORY.resolve() == Path.cwd().resolve(), "run from the ACE-3 worktree")
    require(VALIDATOR.is_file(), f"successor validator missing: {VALIDATOR}")
    require((R15_CANDIDATE / "source").is_dir(), "r15 sealed source missing")

    original_artifacts = r15_artifacts()
    for path in original_artifacts.values():
        require(path.is_file() and not path.is_symlink(), f"r15 evidence missing: {path}")
    original_hashes = {name: digest(path) for name, path in original_artifacts.items()}

    source = output / "source"
    package = output / "package"
    compile_root = package / "evidence/layer0-compile"
    ancestry_root = package / "provenance/r15-failed-ancestry"
    output.mkdir(mode=0o700)
    shutil.copytree(R15_CANDIDATE / "source", source, copy_function=shutil.copy2)
    writable_tree(source)
    package.mkdir(mode=0o700)
    compile_root.mkdir(parents=True)
    ancestry_root.mkdir(parents=True)

    makefile = source / "Makefile"
    before = makefile.read_text(encoding="utf-8")
    require("model24-rtl-layer-compile:" not in before, "r15 source already has compile target")
    after = before.rstrip() + "\n" + MAKEFILE_CLOSURE
    makefile.write_text(after, encoding="utf-8")
    patch = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="r15-source/Makefile",
            tofile="successor-source/Makefile",
        )
    ).encode("utf-8")
    write_new(package / "makefile-closure.patch", patch, 0o400)
    overlay_records = [
        {
            "path": "Makefile",
            "r15_sha256": digest(R15_CANDIDATE / "source/Makefile"),
            "successor_sha256": digest(makefile),
        }
    ]
    for relative in SUCCESSOR_OVERLAYS:
        r15_path = R15_CANDIDATE / "source" / relative
        worktree_path = REPOSITORY / relative
        require(r15_path.is_file(), f"r15 closure source missing: {relative}")
        require(worktree_path.is_file(), f"worktree closure source missing: {relative}")
        require(digest(r15_path) != digest(worktree_path), f"closure overlay is unchanged: {relative}")
        shutil.copy2(worktree_path, source / relative)
        overlay_records.append(
            {
                "path": relative.as_posix(),
                "r15_sha256": digest(r15_path),
                "successor_sha256": digest(source / relative),
            }
        )
    write_new(
        package / "source-overlays.json",
        canonical_json(
            {
                "schema_version": 1,
                "kind": "ace3_model24_rtl_compile_successor_overlays",
                "records": overlay_records,
            }
        ),
        0o400,
    )

    toolchain_path = compile_root / "toolchain.json"
    write_new(toolchain_path, canonical_json(toolchain_record()), 0o400)
    completed = subprocess.run(
        COMPILE_ARGV,
        cwd=source,
        capture_output=True,
        check=False,
    )
    write_new(compile_root / "stdout.log", completed.stdout, 0o400)
    write_new(compile_root / "stderr.log", completed.stderr, 0o400)
    built_binary = source / EXPECTED_BINARY_RELATIVE
    compile_status = "PASS" if completed.returncode == 0 and built_binary.is_file() else "FAIL"
    hashes = {
        "stdout.log": digest(compile_root / "stdout.log"),
        "stderr.log": digest(compile_root / "stderr.log"),
        "toolchain.json": digest(toolchain_path),
    }
    if compile_status == "PASS":
        binary_copy = compile_root / EXPECTED_BINARY_RELATIVE.name
        shutil.copy2(built_binary, binary_copy)
        binary_copy.chmod(0o500)
        hashes[binary_copy.name] = digest(binary_copy)
        diagnosis = "compile completed and expected binary was produced"
    else:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        detail = stderr or stdout or "command returned no diagnostic text"
        diagnosis = f"compile exited {completed.returncode}: {detail}"

    compile_record = {
        "schema_version": 1,
        "kind": "ace3_model24_rtl_layer0_compile_evidence",
        "status": compile_status,
        "argv": COMPILE_ARGV,
        "cwd": str(source),
        "layer_index": 0,
        "accurate_silu": 1,
        "exit_code": completed.returncode,
        "expected_binary_relative": EXPECTED_BINARY_RELATIVE.as_posix(),
        "hashes": hashes,
        "diagnosis": diagnosis,
    }
    write_new(compile_root / "result.json", canonical_json(compile_record), 0o400)

    build_root = source / "build"
    if build_root.exists():
        shutil.rmtree(build_root)
    for name, path in original_artifacts.items():
        write_new(ancestry_root / name, path.read_bytes(), 0o400)
    ancestry = {
        "schema_version": 1,
        "kind": "ace3_model24_r15_failed_ancestry",
        "status": "FAILED",
        "launch_reused": False,
        "authority_created": False,
        "authority_consumed": False,
        "original_paths": {name: str(path) for name, path in original_artifacts.items()},
        "original_hashes": original_hashes,
        "failure": {
            "exit_code": 1,
            "diagnosis": "make: *** No rule to make target 'model24-rtl-layer-compile'.  Stop.",
        },
    }
    write_new(ancestry_root / "evidence.json", canonical_json(ancestry), 0o400)

    for name, path in original_artifacts.items():
        require(digest(path) == original_hashes[name], f"r15 artifact mutated: {name}")
    readonly_tree(source)
    source_records = tree_records(source)
    source_tree_sha256 = hashlib.sha256(canonical_json(source_records)).hexdigest()
    source_manifest = {
        "schema_version": 1,
        "kind": "ace3_model24_rtl_compile_successor_source_tree",
        "base": str(R15_CANDIDATE / "source"),
        "base_package_seal_sha256": original_hashes["seal.json"],
        "changed_paths": CHANGED_PATHS,
        "records": source_records,
        "tree_sha256": source_tree_sha256,
    }
    write_new(package / "source-tree.json", canonical_json(source_manifest), 0o400)
    write_new(package / "validate-package.py", VALIDATOR.read_bytes(), 0o500)

    manifest = {
        "schema_version": 1,
        "kind": "ace3_model24_rtl_compile_successor",
        "identity": {"candidate": str(output)},
        "source": {
            "repository": str(REPOSITORY),
            "branch": "argus/full-projection",
            "r15_base": str(R15_CANDIDATE),
            "changed_paths": CHANGED_PATHS,
            "source_tree_sha256": source_tree_sha256,
        },
        "compile": compile_record,
        "r15_failed_ancestry": {
            "source": str(R15_CANDIDATE),
            "package_seal_sha256": original_hashes["seal.json"],
            "terminal_manifest_sha256": original_hashes["terminal-manifest.json"],
            "status": "FAILED",
            "launch_reused": False,
        },
        "execution_authority": {
            "created": False,
            "consumed": False,
            "launch_invocations": 0,
        },
    }
    write_new(package / "package.json", canonical_json(manifest), 0o400)
    review_request = {
        "schema_version": 1,
        "kind": "ace3_model24_rtl_compile_successor_review_request",
        "candidate": str(output),
        "scope": "sealed source Makefile compile closure and bounded layer-0 compile",
        "compile_status": compile_status,
        "execution_authority_created": False,
        "r15_is_failed_ancestry_only": True,
        "required_hashes": {
            "package_manifest_sha256": digest(package / "package.json"),
            "source_tree_manifest_sha256": digest(package / "source-tree.json"),
            "source_overlays_sha256": digest(package / "source-overlays.json"),
            "compile_result_sha256": digest(compile_root / "result.json"),
            "r15_failed_ancestry_sha256": digest(ancestry_root / "evidence.json"),
        },
    }
    write_new(package / "review-request.json", canonical_json(review_request), 0o400)

    readonly_tree(package / "evidence")
    readonly_tree(package / "provenance")
    package_records = tree_records(package)
    seal = {
        "schema_version": 1,
        "kind": "ace3_model24_rtl_compile_successor_seal",
        "package_manifest_sha256": digest(package / "package.json"),
        "review_request_sha256": digest(package / "review-request.json"),
        "source_tree_sha256": source_tree_sha256,
        "package_entries": package_records,
    }
    write_new(package / "seal.json", canonical_json(seal), 0o400)
    readonly_tree(package)

    result = subprocess.run(
        [str(package / "validate-package.py"), "--package", str(package)],
        cwd=output,
        capture_output=True,
        check=False,
    )
    require(result.returncode == 0, result.stderr.decode("utf-8", errors="replace"))
    validated = json.loads(result.stdout)
    return {
        "candidate": str(output),
        "package_seal_sha256": digest(package / "seal.json"),
        "source_tree_manifest_sha256": digest(package / "source-tree.json"),
        "compile_result_sha256": digest(compile_root / "result.json"),
        "compile_status": compile_status,
        "expected_binary_sha256": hashes.get(EXPECTED_BINARY_RELATIVE.name),
        "validator": validated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(prepare(args.output)).decode("ascii"), end="")


if __name__ == "__main__":
    main()
