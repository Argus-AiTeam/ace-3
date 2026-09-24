#!/usr/bin/env python3
"""Validate the sealed Model24 r15 RTL-compile successor package."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path
from typing import Any


R15_HASHES = {
    "compile.log": "b231166325bceb88e191fa3aecfe98deb8467d9800a8615cc4682b688dc0f2d4",
    "exit-code.txt": "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
    "launch-terminal.json": "5f3b48f22f57148ff3c2091fb940750cc8f2b3794b19e654df83262f9e7a44fb",
    "package.json": "30fb476cff03078411b7a53ec485842b2c76612b45f192b4d64b97d95a157db4",
    "review.json": "8d6576b5d940fa2974b70b233461b0420eabd1601c80037c85225df793f7f64e",
    "runner-receipt.json": "240191800adcba665d59e9267b15544497afc7485addb75c86f790f2bc75b630",
    "seal.json": "963b815111f95aaf4dd27eedb7102c1d241e339ef56225fd072d07d5980e1042",
    "source-tree.json": "e34da4d1d7a453267f955c869f2eba5caa95a6110cd0c30b7dc2428a8c0971ff",
    "stderr.log": "43ffa80f369f7638a3200698aba8861b236eadb9174821887476244ed5499c6d",
    "stdout.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "terminal-manifest.json": "ceea176a433cf7485dfc506880fae9c3c1eb9ec20f431b3a43598fb582636ae5",
}
COMPILE_ARGV = [
    "make",
    "--no-print-directory",
    "model24-rtl-layer-compile",
    "MODEL24_RTL_LAYER_INDEX=0",
    "MODEL24_RTL_ACCURATE_SILU=1",
]
EXPECTED_BINARY = "Vace3_decoder_layer0_token_engine"
CHANGED_PATHS = [
    "Makefile",
    "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
    "ace3/rtl/ace3_fp16_silu_gate_core.sv",
    "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp",
]
MAKEFILE_CLOSURE_LINES = (
    "MODEL24_RTL_LAYER_INDEX ?= 0",
    "MODEL24_RTL_ACCURATE_SILU ?= 1",
    "model24-rtl-layer-compile:",
    '-GLAYER_INDEX="$(MODEL24_RTL_LAYER_INDEX)"',
    '-GACCURATE_SILU="$(MODEL24_RTL_ACCURATE_SILU)"',
    "--top-module ace3_decoder_layer0_token_engine",
    '$(DECODER_CPP_TB)',
)


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


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"regular JSON file required: {path}")
    value = json.loads(path.read_text(encoding="ascii"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def tree_records(root: Path) -> list[dict[str, Any]]:
    require(root.is_dir() and not root.is_symlink(), f"regular directory required: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        require(not path.is_symlink(), f"symlink forbidden in sealed tree: {relative}")
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
        else:
            raise SystemExit(f"unsupported sealed-tree entry: {relative}")
    return records


def validate(package: Path) -> dict[str, Any]:
    package = package.resolve(strict=True)
    candidate = package.parent
    source = candidate / "source"
    seal = load_json(package / "seal.json")
    manifest = load_json(package / "package.json")
    source_manifest = load_json(package / "source-tree.json")
    compile_record = load_json(package / "evidence/layer0-compile/result.json")
    ancestry = load_json(package / "provenance/r15-failed-ancestry/evidence.json")

    require(
        manifest["kind"] == "ace3_model24_rtl_compile_successor",
        "successor manifest kind mismatch",
    )
    require(manifest["identity"]["candidate"] == str(candidate), "candidate path mismatch")
    require(manifest["execution_authority"] == {
        "created": False,
        "consumed": False,
        "launch_invocations": 0,
    }, "successor authority boundary is open")
    require(seal["kind"] == "ace3_model24_rtl_compile_successor_seal", "seal kind mismatch")
    package_records = [
        record for record in tree_records(package) if record["path"] != "seal.json"
    ]
    require(package_records == seal["package_entries"], "sealed package differs from seal")
    require(digest(package / "package.json") == seal["package_manifest_sha256"], "manifest hash mismatch")
    require(digest(package / "review-request.json") == seal["review_request_sha256"], "review request hash mismatch")
    require(package.stat().st_mode & 0o222 == 0, "package root is writable")
    require(all(record["mode"] & 0o222 == 0 for record in package_records), "package entry is writable")

    source_records = source_manifest.get("records")
    require(isinstance(source_records, list), "source records missing")
    source_tree_sha256 = hashlib.sha256(canonical_json(source_records)).hexdigest()
    require(source_tree_sha256 == source_manifest["tree_sha256"], "source manifest self-hash mismatch")
    require(source_tree_sha256 == seal["source_tree_sha256"], "source tree is not seal-bound")
    require(source_manifest["changed_paths"] == CHANGED_PATHS, "source closure path list mismatch")
    require(manifest["source"]["changed_paths"] == CHANGED_PATHS, "manifest closure path list mismatch")
    require(tree_records(source) == source_records, "sealed source differs from source manifest")
    require(source.stat().st_mode & 0o222 == 0, "source root is writable")
    require(all(record["mode"] & 0o222 == 0 for record in source_records), "source entry is writable")

    makefile = (source / "Makefile").read_text(encoding="utf-8")
    for line in MAKEFILE_CLOSURE_LINES:
        require(line in makefile, f"sealed Makefile compile closure missing: {line}")
    require(compile_record["argv"] == COMPILE_ARGV, "layer-0 compile argv mismatch")
    require(compile_record["layer_index"] == 0, "compile layer index mismatch")
    require(compile_record["accurate_silu"] == 1, "accurate-SiLU compile binding mismatch")
    require(
        compile_record["expected_binary_relative"]
        == "build/model24_rtl_cascade/compiled/layer0/obj_dir/" + EXPECTED_BINARY,
        "expected Verilator binary path mismatch",
    )
    compile_root = package / "evidence/layer0-compile"
    for name in ("stdout.log", "stderr.log", "toolchain.json"):
        require(
            digest(compile_root / name) == compile_record["hashes"][name],
            f"compile evidence hash mismatch: {name}",
        )
    if compile_record["status"] == "PASS":
        require(compile_record["exit_code"] == 0, "passing compile has nonzero exit")
        binary = compile_root / EXPECTED_BINARY
        require(binary.is_file() and not binary.is_symlink(), "expected Verilator binary missing")
        require(binary.stat().st_mode & stat.S_IXUSR, "expected Verilator binary is not executable")
        require(
            digest(binary) == compile_record["hashes"][EXPECTED_BINARY],
            "expected Verilator binary hash mismatch",
        )
        require(compile_record["diagnosis"] == "compile completed and expected binary was produced", "pass diagnosis mismatch")
    else:
        require(compile_record["status"] == "FAIL", "compile status must be PASS or FAIL")
        require(compile_record["exit_code"] != 0, "failed compile has zero exit")
        require(not (compile_root / EXPECTED_BINARY).exists(), "failed compile retained a success binary")
        require(compile_record["diagnosis"], "failed compile diagnosis missing")

    ancestry_root = package / "provenance/r15-failed-ancestry"
    require(ancestry["kind"] == "ace3_model24_r15_failed_ancestry", "r15 ancestry kind mismatch")
    require(ancestry["status"] == "FAILED", "r15 ancestry is not failure-only")
    require(ancestry["launch_reused"] is False, "r15 launch was reused")
    require(ancestry["original_hashes"] == R15_HASHES, "r15 ancestry hash manifest mismatch")
    for name, expected in R15_HASHES.items():
        require(digest(ancestry_root / name) == expected, f"r15 ancestry changed: {name}")
    require(
        (ancestry_root / "compile.log").read_text(encoding="utf-8")
        == "make: *** No rule to make target 'model24-rtl-layer-compile'.  Stop.\n",
        "r15 terminal failure diagnosis changed",
    )

    review_request = load_json(package / "review-request.json")
    overlays = load_json(package / "source-overlays.json")
    require([record["path"] for record in overlays["records"]] == CHANGED_PATHS, "overlay record paths mismatch")
    for record in overlays["records"]:
        require(
            digest(source / record["path"]) == record["successor_sha256"],
            f"overlay hash mismatch: {record['path']}",
        )
    require(review_request["candidate"] == str(candidate), "review candidate mismatch")
    require(review_request["execution_authority_created"] is False, "review requests authority")
    require(
        review_request["required_hashes"] == {
            "package_manifest_sha256": digest(package / "package.json"),
            "source_tree_manifest_sha256": digest(package / "source-tree.json"),
            "source_overlays_sha256": digest(package / "source-overlays.json"),
            "compile_result_sha256": digest(compile_root / "result.json"),
            "r15_failed_ancestry_sha256": digest(ancestry_root / "evidence.json"),
        },
        "review hash bindings mismatch",
    )
    return {
        "schema_version": 1,
        "kind": "ace3_model24_rtl_compile_successor_validation",
        "status": "PASS",
        "candidate": str(candidate),
        "compile_status": compile_record["status"],
        "expected_binary_sha256": compile_record["hashes"].get(EXPECTED_BINARY),
        "r15_ancestry_status": "FAILED",
        "execution_authority_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    print(canonical_json(validate(args.package)).decode("ascii"), end="")


if __name__ == "__main__":
    main()
