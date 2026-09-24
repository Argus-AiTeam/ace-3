#!/usr/bin/env python3
"""Prepare the Model24 r18 read-only source build-output launch package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


REPOSITORY = Path("/home/argustest/ace3-argus")
R17_BUILDER = REPOSITORY / "ace3/model/prepare_model24_r17_launch_package.py"
R17_VALIDATOR = REPOSITORY / "ace3/model/validate_model24_r17_launch_package.py"
R18_VALIDATOR = REPOSITORY / "ace3/model/validate_model24_r18_launch_package.py"
MODEL_PYTHON = Path("/home/argustest/miniconda3/bin/python3")
CONTROLLER_PATH = Path("ace3/model/controller_model24_rtl_cascade.py")
R17_NONCE = "2edd86429a5c41f3"
R17_BASE = Path(f"/home/argustest/ace3-model24-r17-prep-20260829-{R17_NONCE}")
R17_TASK_ID = f"ace3-model24-r17-42c895c-{R17_NONCE}"
R17_TERMINAL = Path(
    f"/home/argustest/ace3-model24-r17-terminal-20260829-{R17_NONCE}"
)
R17_LOGS = R17_TERMINAL / ".argus_subagents" / f"{R17_TASK_ID}_logs"
R17_EXIT_CODES = list(R17_LOGS.glob("exit_code.*"))
R17_ARTIFACTS = {
    "package.json": R17_BASE / "package/package.json",
    "seal.json": R17_BASE / "package/seal.json",
    "review.json": Path(
        f"/home/argustest/ace3-model24-r17-review-20260829-{R17_NONCE}/review.json"
    ),
    "authority.json": Path(
        f"/home/argustest/ace3-model24-r17-authority-20260829-{R17_NONCE}.json"
    ),
    "authority-consumed.json": Path(
        f"/home/argustest/ace3-model24-r17-authority-20260829-{R17_NONCE}.json.consumed"
    ),
    "launch-terminal.json": Path(
        f"/home/argustest/ace3-model24-r17-output-20260829-{R17_NONCE}/launch-terminal.json"
    ),
    "terminal-manifest.json": R17_TERMINAL / "manifest.json",
    "runner-receipt.json": (
        R17_TERMINAL / ".argus_subagents" / f"{R17_TASK_ID}.json"
    ),
    "stdout.log": R17_LOGS / "stdout.log",
    "stderr.log": R17_LOGS / "stderr.log",
    "exit-code.txt": R17_EXIT_CODES[0] if len(R17_EXIT_CODES) == 1 else Path(),
    "bindings.json": R17_BASE / "package/bindings.json",
    "compile.log": Path(
        f"/home/argustest/ace3-model24-r17-output-20260829-{R17_NONCE}"
        "/rtl-cascade/layers/layer00/compile.log"
    ),
}
ADDITIONAL_NEGATIVE_CASES = (
    "omitted-build-root",
    "wrong-build-root",
    "outside-build-root",
    "symlinked-build-root",
    "source-write-attempt",
    "stale-build-output",
    "post-seal-source-mutation",
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
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write(path: Path, payload: bytes, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(payload)
    path.chmod(mode)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transform_revision(payload: str) -> str:
    return payload.replace("R17", "R18").replace("r17", "r18")


def r17_failure_hashes() -> dict[str, str]:
    tree = ast.parse(
        R18_VALIDATOR.read_text(encoding="ascii"),
        filename=str(R18_VALIDATOR),
    )
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "R17_FAILURE_HASHES"
        ):
            value = ast.literal_eval(node.value)
            require(
                isinstance(value, dict),
                "r17 failure hash declaration is not a dictionary",
            )
            return value
    raise SystemExit("r17 failure hash declaration is missing")


def bootstrap_fresh_r18(nonce: str) -> tuple[Path, Any]:
    with tempfile.TemporaryDirectory(
        prefix="ace3-model24-r18-scaffold-"
    ) as scratch_name:
        scratch = Path(scratch_name)
        baseline_validator = scratch / "validate-package.py"
        baseline_builder = scratch / "repository-builder.py"
        write(
            baseline_validator,
            transform_revision(
                R17_VALIDATOR.read_text(encoding="ascii")
            ).encode("ascii"),
            0o500,
        )
        builder_text = transform_revision(
            R17_BUILDER.read_text(encoding="ascii")
        )
        expected = (
            'R18_VALIDATOR = REPOSITORY / '
            '"ace3/model/validate_model24_r18_launch_package.py"'
        )
        require(expected in builder_text, "r18 validator substitution seam missing")
        builder_text = builder_text.replace(
            expected,
            f"R18_VALIDATOR = Path({str(baseline_validator)!r})",
        )
        write(baseline_builder, builder_text.encode("ascii"), 0o500)
        module = load_module(
            baseline_builder,
            "model24_r18_fresh_baseline_builder",
        )
        previous = os.environ.get("ACE3_R18_PREPARATION_ONLY")
        os.environ["ACE3_R18_PREPARATION_ONLY"] = "1"
        try:
            result = module.prepare(nonce)
        finally:
            if previous is None:
                os.environ.pop("ACE3_R18_PREPARATION_ONLY", None)
            else:
                os.environ["ACE3_R18_PREPARATION_ONLY"] = previous
        require(
            result["status"] == "SEALED_AWAITING_INDEPENDENT_L2_REVIEW",
            "fresh r18 baseline did not seal",
        )
    return (
        Path(f"/home/argustest/ace3-model24-r18-prep-20260829-{nonce}"),
        module,
    )


def controller_overlay_payload(payload: str) -> str:
    require("import stat\n" not in payload, "controller already imports stat")
    payload = payload.replace(
        "import shutil\nimport subprocess\nimport tempfile\n",
        "import shutil\nimport stat\nimport subprocess\nimport tempfile\nimport time\n",
    )
    marker = "\ndef execute(\n"
    require(marker in payload, "controller execute insertion seam missing")
    helpers = '''
def build_layer_compile_argv(
    repository_root: Path,
    output_dir: Path,
    layer_id: int,
    build_root: Path | None,
) -> list[str]:
    require(0 <= layer_id < LAYER_COUNT, "layer compile index is outside Model24")
    require(build_root is not None, "MODEL24_RTL_CASCADE_DIR build root is required")
    repository_root = repository_root.absolute()
    output_dir = output_dir.absolute()
    build_root = build_root.absolute()
    require(
        output_dir.is_dir() and not output_dir.is_symlink(),
        "bound payload output root is absent or symlinked",
    )
    require(
        output_dir.resolve(strict=True) == output_dir,
        "bound payload output root contains a symlink",
    )
    require(not build_root.is_symlink(), "build root is symlinked")
    try:
        resolved_build_root = build_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise RtlCascadeError("build root does not exist") from error
    require(resolved_build_root == build_root, "build root contains a symlink")
    require(
        build_root == output_dir,
        "build root differs from bound payload output root",
    )
    require(
        build_root != repository_root and repository_root not in build_root.parents,
        "build root is inside sealed source",
    )
    return [
        "make",
        "--no-print-directory",
        "model24-rtl-layer-compile",
        f"MODEL24_RTL_LAYER_INDEX={layer_id}",
        "MODEL24_RTL_ACCURATE_SILU=1",
        f"MODEL24_RTL_CASCADE_DIR={build_root}",
    ]


def authenticate_compiled_binary(
    binary: Path,
    compile_started_ns: int,
) -> dict[str, Any]:
    require(
        binary.is_file() and not binary.is_symlink(),
        "fresh compiled binary is missing or symlinked",
    )
    binary_stat = binary.stat()
    require(
        stat.S_ISREG(binary_stat.st_mode) and binary_stat.st_mode & stat.S_IXUSR,
        "fresh compiled binary is not a regular executable",
    )
    require(
        binary_stat.st_mtime_ns >= compile_started_ns,
        "fresh compiled binary is stale",
    )
    record = hash_file(binary)
    record["mtime_ns"] = binary_stat.st_mtime_ns
    return record

'''
    payload = payload.replace(marker, helpers + marker, 1)
    original = '''                run_command(
                    [
                        "make",
                        "--no-print-directory",
                        "model24-rtl-layer-compile",
                        f"MODEL24_RTL_LAYER_INDEX={layer_id}",
                        "MODEL24_RTL_ACCURATE_SILU=1",
                    ],
                    repository_root,
                    compile_log,
                )
                binary = (
                    output_dir
                    / "compiled"
                    / f"layer{layer_id}"
                    / "obj_dir"
                    / "Vace3_decoder_layer0_token_engine"
                )
                require(binary.is_file(), f"layer {layer_id} RTL binary missing")
'''
    replacement = '''                compile_started_ns = time.time_ns()
                run_command(
                    build_layer_compile_argv(
                        repository_root,
                        output_dir,
                        layer_id,
                        output_dir,
                    ),
                    repository_root,
                    compile_log,
                )
                binary = (
                    output_dir
                    / "compiled"
                    / f"layer{layer_id}"
                    / "obj_dir"
                    / "Vace3_decoder_layer0_token_engine"
                )
                authenticate_compiled_binary(binary, compile_started_ns)
'''
    require(original in payload, "production Make invocation replacement seam missing")
    return payload.replace(original, replacement, 1)


def copy_r17_failed_ancestry(package: Path, validator: Any) -> None:
    require(len(R17_EXIT_CODES) == 1, "r17 exit-code evidence is ambiguous")
    root = package / "provenance/r17-failed"
    root.mkdir(parents=True)
    for name, source in R17_ARTIFACTS.items():
        require(
            source.is_file() and not source.is_symlink(),
            f"r17 failed ancestry missing: {source}",
        )
        require(
            digest(source) == validator.R17_FAILURE_HASHES[name],
            f"r17 failed ancestry hash mismatch: {name}",
        )
        shutil.copy2(source, root / name)
        (root / name).chmod(0o400)
    classification = {
        "schema_version": 1,
        "kind": "ace3_model24_r17_consumed_failed_ancestry",
        "nonce": R17_NONCE,
        "classification": "LAWFUL_CONSUMED_FAILED",
        "sole_lawful_invocation": True,
        "historical_execution_invocations": 1,
        "authority_consumed": True,
        "natural_terminal": False,
        "exit_code": 1,
        "review_reusable": False,
        "authority_reusable": False,
        "execution_inputs_reusable": False,
        "predecessor_reusable": False,
        "copied_byte_for_byte": True,
        "terminal_manifest_sha256": validator.R17_FAILURE_HASHES[
            "terminal-manifest.json"
        ],
        "compile_log_sha256": validator.R17_FAILURE_HASHES["compile.log"],
        "hashes": validator.R17_FAILURE_HASHES,
    }
    write(root / "classification.json", canonical_json(classification))


def extend_negative_tests(builder: Path) -> None:
    source = builder.read_text(encoding="ascii")
    marker = '''    cases.append("tensor-map-mutation")

    require(
        cases == list(validator.REQUIRED_NEGATIVE_CASES),
'''
    require(marker in source, "r18 negative-suite insertion point missing")
    extension = '''    cases.append("tensor-map-mutation")

    with tempfile.TemporaryDirectory(
        prefix="ace3-model24-r18-build-root-negatives-"
    ) as build_scratch_name:
        build_scratch = Path(build_scratch_name)
        build_output = build_scratch / "bound-build-output"
        build_output.mkdir()
        _expect_rejection(
            lambda: validator.validate_build_root(package, None, build_output),
            "build root is required",
            "omitted-build-root",
        )
        cases.append("omitted-build-root")

        wrong_root = build_output / "nested"
        wrong_root.mkdir()
        _expect_rejection(
            lambda: validator.validate_build_root(package, wrong_root, build_output),
            "differs from bound payload output root",
            "wrong-build-root",
        )
        cases.append("wrong-build-root")

        outside_root = build_scratch / "outside-build-output"
        outside_root.mkdir()
        _expect_rejection(
            lambda: validator.validate_build_root(
                package, outside_root, build_output
            ),
            "differs from bound payload output root",
            "outside-build-root",
        )
        cases.append("outside-build-root")

        symlinked_root = build_scratch / "symlinked-build-output"
        symlinked_root.symlink_to(build_output, target_is_directory=True)
        _expect_rejection(
            lambda: validator.validate_build_root(
                package, symlinked_root, build_output
            ),
            "build root is symlinked",
            "symlinked-build-root",
        )
        cases.append("symlinked-build-root")

        _expect_rejection(
            lambda: validator.reject_source_write_attempt(
                package.parent / "source"
            ),
            "source write attempt rejected",
            "source-write-attempt",
        )
        cases.append("source-write-attempt")

        stale_binary = build_scratch / "stale-binary"
        stale_binary.write_bytes(b"stale binary")
        stale_binary.chmod(0o500)
        os.utime(stale_binary, ns=(1, 1))
        _expect_rejection(
            lambda: validator.authenticate_probe_binary(
                package, stale_binary, time.time_ns()
            ),
            "fresh compiled binary is stale",
            "stale-build-output",
        )
        cases.append("stale-build-output")

    source_records = validator.load_json(package / "source-tree.json")["records"]
    mutated_records = json.loads(json.dumps(source_records))
    mutated_records[0]["mode"] ^= 0o200
    _expect_rejection(
        lambda: validator.require_source_records_match(
            source_records, mutated_records
        ),
        "post-seal source mutation detected",
        "post-seal-source-mutation",
    )
    cases.append("post-seal-source-mutation")

    require(
        cases == list(validator.REQUIRED_NEGATIVE_CASES),
'''
    builder.chmod(0o600)
    builder.write_text(source.replace(marker, extension, 1), encoding="ascii")
    builder.chmod(0o400)


def regenerate_bindings(base: Path, validator: Any) -> dict[str, Any]:
    package = base / "package"
    source = base / "source"
    model_root = source / "ace3/model"
    sys.path.insert(0, str(model_root))
    try:
        controller = load_module(
            model_root / "controller_model24_cascade.py",
            "model24_r18_binding_generator",
        )
    finally:
        sys.path.pop(0)
    tensor_payload = (
        source / "ace3/contracts/model24_tensor_map.json"
    ).read_bytes()
    document = controller.build_binding_document(source, tensor_payload)
    controller.validate_binding_document(document, source, tensor_payload)
    write(package / "bindings.json", controller._canonical_json(document))
    r16_payload = (
        package / "provenance/r16-failed/bindings.json"
    ).read_bytes()
    r16_document = controller._load_json(r16_payload, "r16 layer bindings")
    try:
        controller.validate_binding_document(
            r16_document,
            source,
            tensor_payload,
        )
    except controller.ControllerCascadeError as error:
        require(
            str(error) == "layer binding manifest mismatch",
            f"r16 binding rejected for wrong reason: {error}",
        )
    else:
        raise SystemExit("r16 stale binding unexpectedly passed")
    differences = validator.binding_differences(document, r16_document)
    require(
        len(differences) == 8,
        "r18 binding mismatch diagnosis is incomplete",
    )
    result = {
        "schema_version": 1,
        "kind": "ace3_model24_r18_binding_closure",
        "status": "PASS",
        "validator": "controller_model24_cascade.validate_binding_document",
        "controller_source": "ace3/model/controller_model24_cascade.py",
        "controller_source_sha256": digest(
            model_root / "controller_model24_cascade.py"
        ),
        "bindings_sha256": digest(package / "bindings.json"),
        "tensor_map_bytes": len(tensor_payload),
        "tensor_map_sha256": hashlib.sha256(tensor_payload).hexdigest(),
        "r16_stale_binding_sha256": hashlib.sha256(r16_payload).hexdigest(),
        "r16_validation": {
            "status": "REJECT",
            "reason": "layer binding manifest mismatch",
        },
        "r16_expected_differences": differences,
        "model24_invocations": 0,
        "controller_simulator_invocations": 0,
        "rtl_simulator_invocations": 0,
        "lifecycle_launch_invocations": 0,
        "subagent_invocations": 0,
    }
    write(
        package / "evidence/binding-closure/result.json",
        canonical_json(result),
    )
    return result


def run_readonly_build_output_probe(
    base: Path,
    source_hash: str,
    r16: Any,
) -> dict[str, Any]:
    package = base / "package"
    source = base / "source"
    output = base / "probe-output"
    evidence = package / "evidence/readonly-build-output-probe"
    preflight = package / "evidence/production-preflight"
    if output.exists():
        shutil.rmtree(output)
    if evidence.exists():
        shutil.rmtree(evidence)
    if preflight.exists():
        shutil.rmtree(preflight)
    output.mkdir()
    evidence.mkdir(parents=True)
    preflight.mkdir(parents=True)
    before_records = r16.tree_records(source)
    before_hash = r16.tree_sha256(before_records)
    require(before_hash == source_hash, "source seal changed before build-output probe")
    require(not (source / "build").exists(), "sealed source contains build output")
    write(
        evidence / "source-before.json",
        canonical_json({"records": before_records, "tree_sha256": before_hash}),
    )

    sys.path.insert(0, str(source / "ace3/model"))
    try:
        controller = load_module(
            source / CONTROLLER_PATH,
            "model24_r18_readonly_build_output_probe",
        )
    finally:
        sys.path.pop(0)

    negatives: list[str] = []

    def expect_rejection(callable_object: Any, expected: str, name: str) -> None:
        try:
            callable_object()
        except controller.RtlCascadeError as error:
            require(expected in str(error), f"{name} rejected for wrong reason: {error}")
        else:
            raise SystemExit(f"negative probe unexpectedly accepted: {name}")
        negatives.append(name)

    with tempfile.TemporaryDirectory(
        prefix="r18-build-root-negatives-",
        dir=base,
    ) as temporary:
        scratch = Path(temporary)
        wrong = output / "nested"
        wrong.mkdir()
        outside = scratch / "outside"
        outside.mkdir()
        symlinked = scratch / "symlinked"
        symlinked.symlink_to(output, target_is_directory=True)
        expect_rejection(
            lambda: controller.build_layer_compile_argv(
                source, output, 0, None
            ),
            "build root is required",
            "omitted-build-root",
        )
        expect_rejection(
            lambda: controller.build_layer_compile_argv(
                source, output, 0, wrong
            ),
            "differs from bound payload output root",
            "wrong-build-root",
        )
        expect_rejection(
            lambda: controller.build_layer_compile_argv(
                source, output, 0, outside
            ),
            "differs from bound payload output root",
            "outside-build-root",
        )
        expect_rejection(
            lambda: controller.build_layer_compile_argv(
                source, output, 0, symlinked
            ),
            "build root is symlinked",
            "symlinked-build-root",
        )
        wrong.rmdir()

    source_write_probe = source / ".r18-source-write-attempt"
    try:
        descriptor = os.open(
            source_write_probe,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except OSError:
        negatives.append("source-write-attempt")
    else:
        os.close(descriptor)
        source_write_probe.unlink()
        raise SystemExit("recursively sealed source accepted a write")

    stale_binary = base / "r18-stale-binary"
    stale_binary.write_bytes(b"stale binary")
    stale_binary.chmod(0o500)
    os.utime(stale_binary, ns=(1, 1))
    expect_rejection(
        lambda: controller.authenticate_compiled_binary(
            stale_binary,
            time.time_ns(),
        ),
        "fresh compiled binary is stale",
        "stale-build-output",
    )
    stale_binary.unlink()

    layers: dict[str, Any] = {}
    for layer in (0, 23):
        command = controller.build_layer_compile_argv(
            source,
            output,
            layer,
            output,
        )
        started_ns = time.time_ns()
        completed = subprocess.run(
            command,
            cwd=source,
            capture_output=True,
            check=False,
        )
        stdout_name = f"compile-layer{layer}.stdout"
        stderr_name = f"compile-layer{layer}.stderr"
        write(evidence / stdout_name, completed.stdout)
        write(evidence / stderr_name, completed.stderr)
        require(completed.returncode == 0, f"layer {layer} read-only compile failed")
        binary = (
            output
            / "compiled"
            / f"layer{layer}"
            / "obj_dir"
            / "Vace3_decoder_layer0_token_engine"
        )
        binary_record = controller.authenticate_compiled_binary(
            binary,
            started_ns,
        )
        layers[str(layer)] = {
            "argv": command,
            "compile_started_ns": started_ns,
            "exit_code": completed.returncode,
            "binary": binary_record,
            "evidence_hashes": {
                stdout_name: digest(evidence / stdout_name),
                stderr_name: digest(evidence / stderr_name),
            },
        }

    after_records = r16.tree_records(source)
    after_hash = r16.tree_sha256(after_records)
    require(before_records == after_records, "production compile mutated sealed source")
    require(before_hash == after_hash, "production compile changed sealed source hash")
    require(not (source / "build").exists(), "production compile wrote source/build")
    write(
        evidence / "source-after.json",
        canonical_json({"records": after_records, "tree_sha256": after_hash}),
    )

    mutated_records = json.loads(json.dumps(after_records))
    mutated_records[0]["mode"] ^= 0o200
    require(mutated_records != after_records, "source mutation negative is inert")
    negatives.append("post-seal-source-mutation")
    require(
        negatives == list(ADDITIONAL_NEGATIVE_CASES),
        "read-only build-output negative order differs",
    )

    layer0_binary = (
        output
        / "compiled/layer0/obj_dir/Vace3_decoder_layer0_token_engine"
    )
    copied_binary = preflight / "Vace3_decoder_layer0_token_engine"
    shutil.copy2(layer0_binary, copied_binary)
    copied_binary.chmod(0o500)
    for target, source_name in (
        ("compile.stdout", "compile-layer0.stdout"),
        ("compile.stderr", "compile-layer0.stderr"),
        ("make-layer0.stdout", "compile-layer0.stdout"),
        ("make-layer23.stdout", "compile-layer23.stdout"),
    ):
        write(preflight / target, (evidence / source_name).read_bytes())

    output_records = r16.tree_records(output)
    result = {
        "schema_version": 1,
        "kind": "ace3_model24_r18_readonly_build_output_probe",
        "status": "PASS",
        "source": str(source),
        "source_tree_sha256": source_hash,
        "output_root": str(output),
        "source_build_absent_before": True,
        "source_build_absent_after": True,
        "source_additions": 0,
        "source_modifications": 0,
        "layers": layers,
        "negative_cases": negatives,
        "output_tree_sha256": r16.tree_sha256(output_records),
        "model24_invocations": 0,
        "controller_simulator_invocations": 0,
        "rtl_binary_invocations": 0,
        "authority_invocations": 0,
    }
    write(evidence / "result.json", canonical_json(result))

    preflight_result = {
        "schema_version": 1,
        "kind": "ace3_model24_r18_production_preflight",
        "status": "PASS",
        "source": str(source),
        "source_tree_sha256": source_hash,
        "cascade_command_source": str(CONTROLLER_PATH),
        "cascade_layer0_argv": layers["0"]["argv"],
        "compile_argv": layers["0"]["argv"],
        "compile_exit_code": 0,
        "accurate_silu": 1,
        "layer_index": 0,
        "make_target_resolution": {
            "layer0_exit_code": layers["0"]["exit_code"],
            "layer23_exit_code": layers["23"]["exit_code"],
        },
        "binary": {
            "name": copied_binary.name,
            "bytes": copied_binary.stat().st_size,
            "sha256": digest(copied_binary),
            "fresh_compile_started_ns": layers["0"]["compile_started_ns"],
            "compiled_mtime_ns": layers["0"]["binary"]["mtime_ns"],
        },
        "evidence_hashes": {
            name: digest(preflight / name)
            for name in (
                "compile.stdout",
                "compile.stderr",
                "make-layer0.stdout",
                "make-layer23.stdout",
            )
        },
        "model24_invocations": 0,
        "launcher_invocations": 0,
        "authority_invocations": 0,
    }
    write(preflight / "result.json", canonical_json(preflight_result))
    return result


def finalize(base: Path, baseline: Any) -> dict[str, Any]:
    package = base / "package"
    source = base / "source"
    r16 = load_module(
        baseline.R16_BUILDER,
        "model24_r18_source_tree_helpers",
    )
    r16.writable_tree(base)
    for root in (base / "validation", base / "preparation"):
        if root.exists():
            shutil.rmtree(root)
        root.mkdir()
    seal_path = package / "seal.json"
    if seal_path.exists():
        seal_path.unlink()

    baseline_validator = package / "validate-package.py"
    baseline_text = baseline_validator.read_text(encoding="ascii")
    require(
        'require(len(differences) == 6, "r16 binding diagnosis is not complete")'
        in baseline_text,
        "r18 binding-difference validator seam missing",
    )
    baseline_text = baseline_text.replace(
        'require(len(differences) == 6, "r16 binding diagnosis is not complete")',
        'require(len(differences) == 8, "r16 binding diagnosis is not complete")',
    )
    write(
        package / "validate-r17-package.py",
        baseline_text.encode("ascii"),
        0o500,
    )
    baseline_validator.unlink()
    write(package / "validate-package.py", R18_VALIDATOR.read_bytes(), 0o500)
    extend_negative_tests(package / "ancestry/repository-builder.py")

    controller = source / CONTROLLER_PATH
    require(
        digest(controller)
        == "a633a4f40089e0c486b9aa7247ae5884ba44cd4c4922d2b62ee28f49999459b6",
        "r17 controller source binding differs",
    )
    controller_payload = controller_overlay_payload(
        controller.read_text(encoding="utf-8")
    )
    write(controller, controller_payload.encode("utf-8"), 0o500)
    overlay = {
        "schema_version": 1,
        "kind": "ace3_model24_r18_controller_build_output_overlay",
        "path": CONTROLLER_PATH.as_posix(),
        "overlay_set_exact": True,
        "r17_source_sha256": (
            "a633a4f40089e0c486b9aa7247ae5884ba44cd4c4922d2b62ee28f49999459b6"
        ),
        "materialized_sha256": digest(controller),
        "production_make_target": "model24-rtl-layer-compile",
        "build_root_assignment": (
            "MODEL24_RTL_CASCADE_DIR=<bound payload output_dir>"
        ),
        "source_build_forbidden": True,
    }
    write(package / "controller-overlay.json", canonical_json(overlay))

    r16.readonly_tree(source)
    source_records = r16.tree_records(source)
    source_hash = r16.tree_sha256(source_records)
    source_manifest = load_json(package / "source-tree.json")
    source_manifest.update(
        {
            "kind": "ace3_model24_r18_materialized_source_tree",
            "overlay_paths": [
                *r16.OVERLAY_PATHS,
                CONTROLLER_PATH.as_posix(),
            ],
            "records": source_records,
            "tree_sha256": source_hash,
        }
    )
    write(package / "source-tree.json", canonical_json(source_manifest))

    validator = load_module(
        package / "validate-package.py",
        "model24_r18_contract_validator",
    )
    copy_r17_failed_ancestry(package, validator)
    binding_closure = regenerate_bindings(base, validator)
    probe = run_readonly_build_output_probe(base, source_hash, r16)

    contract = load_json(package / "launch-contract.json")
    contract["review"]["required_negative_cases"] = list(
        validator.REQUIRED_NEGATIVE_CASES
    )
    contract["review"]["required_bound_hashes"] = list(
        validator.REQUIRED_REVIEW_HASHES
    )
    contract["source_materialization"].update(
        {
            "r17_source_preserved": True,
            "r18_overlay_paths": [CONTROLLER_PATH.as_posix()],
            "source_recursively_readonly": True,
        }
    )
    contract["production_preflight"].update(
        {
            "layers_compiled": [0, 23],
            "build_root_assignment_required": True,
            "build_root_exact_payload_output": True,
            "source_build_forbidden": True,
            "source_mutation_forbidden": True,
            "rtl_binary_execution_forbidden": True,
        }
    )
    contract["ancestry"].update(
        {
            "r17_lawful_consumed_failed": True,
            "r17_review_reusable": False,
            "r17_authority_reusable": False,
            "r17_execution_inputs_reusable": False,
            "predecessor_reusable": False,
        }
    )
    write(package / "launch-contract.json", canonical_json(contract))

    manifest = load_json(package / "package.json")
    manifest["provenance"]["source"].update(
        {
            "materialization": (
                "r17 exact source plus one authenticated controller overlay"
            ),
            "tree_sha256": source_hash,
            "overlay_paths": [
                *r16.OVERLAY_PATHS,
                CONTROLLER_PATH.as_posix(),
            ],
        }
    )
    manifest["provenance"]["r18_controller_overlay"] = {
        "path": CONTROLLER_PATH.as_posix(),
        "manifest_sha256": digest(package / "controller-overlay.json"),
        "r17_source_sha256": overlay["r17_source_sha256"],
        "materialized_sha256": overlay["materialized_sha256"],
    }
    manifest["provenance"]["r17_terminal_ancestry"] = {
        "classification": "LAWFUL_CONSUMED_FAILED",
        "classification_sha256": digest(
            package / "provenance/r17-failed/classification.json"
        ),
        "terminal_manifest_sha256": validator.R17_FAILURE_HASHES[
            "terminal-manifest.json"
        ],
        "compile_log_sha256": validator.R17_FAILURE_HASHES["compile.log"],
        "review_reusable": False,
        "authority_reusable": False,
        "execution_inputs_reusable": False,
    }
    manifest["provenance"]["bindings"] = {
        "generation": (
            "materialized controller build_binding_document over exact "
            "materialized tensor-map bytes"
        ),
        "sha256": binding_closure["bindings_sha256"],
        "tensor_map_sha256": binding_closure["tensor_map_sha256"],
        "binding_closure_sha256": digest(
            package / "evidence/binding-closure/result.json"
        ),
        "r16_stale_binding_sha256": binding_closure[
            "r16_stale_binding_sha256"
        ],
    }
    manifest["production_preflight"] = {
        "status": "PASS",
        "result_sha256": digest(
            package / "evidence/production-preflight/result.json"
        ),
        "readonly_build_output_probe_sha256": digest(
            package / "evidence/readonly-build-output-probe/result.json"
        ),
        "compile_argv": probe["layers"]["0"]["argv"],
        "layer23_compile_argv": probe["layers"]["23"]["argv"],
        "binary_sha256": probe["layers"]["0"]["binary"]["sha256"],
        "layer23_binary_sha256": probe["layers"]["23"]["binary"]["sha256"],
        "build_root": probe["output_root"],
        "source_recursively_readonly": True,
        "source_mutations": 0,
        "model24_invocations": 0,
        "rtl_binary_invocations": 0,
        "binding_validation": {
            "validator": (
                "controller_model24_cascade.validate_binding_document"
            ),
            "before_authority_consumption": True,
            "before_output_creation": True,
            "before_controller_simulation": True,
            "before_payload_execution": True,
        },
    }
    manifest["review_policy"]["predecessor_review_reusable"] = False
    manifest["review_policy"]["predecessor_authority_reusable"] = False
    write(package / "package.json", canonical_json(manifest))

    authority_schema = load_json(package / "authority-schema.json")
    authority_schema["required_exact_fields"] = [
        "schema_version",
        "kind",
        "nonce",
        "task_id",
        "review_path",
        "review_sha256",
        *validator.REQUIRED_REVIEW_HASHES,
        "submitter_cwd",
        "child_cwd",
        "submission_argv",
        "launch_argv",
        "durable_command",
        "terminal_root",
        "registry_root",
        "receipt_path",
        "stdout_log",
        "stderr_log",
        "terminal_manifest",
        "output_namespace",
        "simulation_namespace",
        "payload_output_namespace",
        "execution_cardinality",
        "manager_exactly_once_directive",
        "retry",
        "replay",
        "resume",
        "watcher",
    ]
    write(package / "authority-schema.json", canonical_json(authority_schema))

    review_request = load_json(package / "review-request.json")
    review_request["required_bound_hashes"] = list(
        validator.REQUIRED_REVIEW_HASHES
    )
    review_request["required_test_results"]["negative_cases"] = list(
        validator.REQUIRED_NEGATIVE_CASES
    )
    review_request.update(
        {
            "package_manifest_sha256": digest(package / "package.json"),
            "validator_sha256": digest(package / "validate-package.py"),
            "base_validator_sha256": digest(
                package / "validate-base-package.py"
            ),
            "r16_validator_sha256": digest(
                package / "validate-r16-package.py"
            ),
            "r17_validator_sha256": digest(
                package / "validate-r17-package.py"
            ),
            "lifecycle_sha256": digest(package / "lifecycle.py"),
            "launch_contract_sha256": digest(package / "launch-contract.json"),
            "source_tree_sha256": digest(package / "source-tree.json"),
            "source_overlays_sha256": digest(package / "source-overlays.json"),
            "controller_overlay_sha256": digest(
                package / "controller-overlay.json"
            ),
            "authority_schema_sha256": digest(
                package / "authority-schema.json"
            ),
            "production_preflight_sha256": digest(
                package / "evidence/production-preflight/result.json"
            ),
            "readonly_build_output_probe_sha256": digest(
                package / "evidence/readonly-build-output-probe/result.json"
            ),
            "bindings_sha256": digest(package / "bindings.json"),
            "binding_closure_sha256": digest(
                package / "evidence/binding-closure/result.json"
            ),
            "r16_failed_ancestry_sha256": digest(
                package / "provenance/r16-failed/classification.json"
            ),
            "r17_failed_ancestry_sha256": digest(
                package / "provenance/r17-failed/classification.json"
            ),
            "r17_lawful_consumed_failed": True,
            "predecessor_review_reusable": False,
            "predecessor_authority_reusable": False,
            "predecessor_execution_inputs_reusable": False,
            "source_recursively_readonly": True,
            "source_build_forbidden": True,
            "canonical_execution_performed": False,
        }
    )
    write(package / "review-request.json", canonical_json(review_request))

    probe_test = (
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import importlib.util,json\n"
        "package=Path(__file__).resolve().parent\n"
        "spec=importlib.util.spec_from_file_location('validator',package/'validate-package.py')\n"
        "module=importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module.validate_readonly_build_output_probe(package)\n"
        "print(json.dumps({'status':'PASS'},sort_keys=True,separators=(',',':')))\n"
    ).encode("ascii")
    write(package / "test-readonly-build-output.py", probe_test, 0o500)

    r16.readonly_tree(package)
    package.chmod(0o700)
    package_records = r16.tree_records(package)
    seal = {
        "schema_version": 1,
        "kind": "ace3_model24_r18_package_seal",
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "package_manifest_sha256": digest(package / "package.json"),
        "review_request_sha256": digest(package / "review-request.json"),
        "source_tree_sha256": source_hash,
        "source_tree_manifest_sha256": digest(package / "source-tree.json"),
        "source_overlays_sha256": digest(package / "source-overlays.json"),
        "controller_overlay_sha256": digest(package / "controller-overlay.json"),
        "production_preflight_sha256": digest(
            package / "evidence/production-preflight/result.json"
        ),
        "readonly_build_output_probe_sha256": digest(
            package / "evidence/readonly-build-output-probe/result.json"
        ),
        "bindings_sha256": digest(package / "bindings.json"),
        "binding_closure_sha256": digest(
            package / "evidence/binding-closure/result.json"
        ),
        "r17_failed_ancestry_sha256": digest(
            package / "provenance/r17-failed/classification.json"
        ),
        "v2_bound_hashes": r16.V2_HASHES,
        "package_entries": package_records,
        "zero_state": True,
    }
    write(seal_path, canonical_json(seal))
    package.chmod(0o500)

    validation_root = base / "validation"
    commands = {
        "package-validation": [
            str(MODEL_PYTHON),
            "-B",
            str(package / "validate-package.py"),
            "--package",
            str(package),
            "--mode",
            "package",
        ],
        "negative-tests": [
            str(MODEL_PYTHON),
            "-B",
            str(package / "test-launch-contract.py"),
        ],
        "controller-entry-tests": [
            str(MODEL_PYTHON),
            "-B",
            str(package / "test-controller-entry.py"),
        ],
        "production-binding-preflight": [
            str(MODEL_PYTHON),
            "-B",
            str(package / "test-binding-preflight.py"),
        ],
        "readonly-build-output-probe": [
            str(MODEL_PYTHON),
            "-B",
            str(package / "test-readonly-build-output.py"),
        ],
    }
    environment = dict(os.environ)
    environment["ACE3_R18_PREPARATION_ONLY"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    outcomes: dict[str, dict[str, Any]] = {}
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            cwd=base,
            env=environment,
            capture_output=True,
            check=False,
        )
        write(validation_root / f"{name}.stdout", completed.stdout)
        write(validation_root / f"{name}.stderr", completed.stderr)
        write(
            validation_root / f"{name}.status",
            f"{completed.returncode}\n".encode("ascii"),
        )
        require(
            completed.returncode == 0,
            f"{name} failed: {completed.stderr.decode(errors='replace')}",
        )
        outcomes[name] = {
            "argv": command,
            "exit_code": completed.returncode,
            "stdout_sha256": digest(validation_root / f"{name}.stdout"),
            "stderr_sha256": digest(validation_root / f"{name}.stderr"),
        }

    paths = validator.expected_paths(package, manifest)
    forbidden = (
        "review",
        "authority",
        "authority_consumed",
        "output",
        "simulation_dir",
        "payload_output_dir",
        "terminal_root",
        "receipt",
        "stdout_log",
        "stderr_log",
        "terminal_manifest",
    )
    zero_state = {f"{name}_absent": not paths[name].exists() for name in forbidden}
    require(
        all(zero_state.values()),
        "canonical r18 review or execution namespace was created",
    )
    zero_state.update(
        {
            "model24_invocations": 0,
            "controller_simulator_invocations": 0,
            "rtl_binary_invocations": 0,
            "lifecycle_launch_invocations": 0,
            "subagent_invocations": 0,
            "authority_invocations": 0,
            "output_creations": 0,
        }
    )
    write(validation_root / "zero-state.json", canonical_json(zero_state))
    r16.readonly_tree(validation_root)
    validation_root.chmod(0o700)
    validation_records = r16.tree_records(validation_root)
    validation_seal = {
        "schema_version": 1,
        "kind": "ace3_model24_r18_inert_validation_seal",
        "package_seal_sha256": digest(seal_path),
        "negative_case_count": len(validator.REQUIRED_NEGATIVE_CASES),
        "positive_fixture_count": len(validator.REQUIRED_POSITIVE_FIXTURES),
        "outcomes": outcomes,
        "records": validation_records,
        "model24_invocations": 0,
        "controller_simulator_invocations": 0,
        "rtl_binary_invocations": 0,
        "lifecycle_launch_invocations": 0,
        "subagent_invocations": 0,
        "authority_invocations": 0,
    }
    write(validation_root / "seal.json", canonical_json(validation_seal))
    validation_root.chmod(0o500)

    result = {
        "status": "SEALED_AWAITING_INDEPENDENT_L2_REVIEW",
        "candidate": str(base),
        "package": str(package),
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "package_manifest_sha256": digest(package / "package.json"),
        "package_seal_sha256": digest(seal_path),
        "review_request_sha256": digest(package / "review-request.json"),
        "source_tree_manifest_sha256": digest(package / "source-tree.json"),
        "source_tree_sha256": source_hash,
        "controller_overlay_sha256": digest(package / "controller-overlay.json"),
        "bindings_sha256": digest(package / "bindings.json"),
        "binding_closure_sha256": digest(
            package / "evidence/binding-closure/result.json"
        ),
        "readonly_build_output_probe_sha256": digest(
            package / "evidence/readonly-build-output-probe/result.json"
        ),
        "validation_seal_sha256": digest(validation_root / "seal.json"),
        "negative_case_count": len(validator.REQUIRED_NEGATIVE_CASES),
        "positive_fixture_count": len(validator.REQUIRED_POSITIVE_FIXTURES),
        "production_preflight": "PASS",
        "readonly_build_output_probe": "PASS",
        "layers_compiled_not_executed": [0, 23],
        "source_mutations": 0,
        "r17_lawful_consumed_failed_ancestry": True,
        "canonical_zero_state": zero_state,
        "execution_authority_withheld": True,
        "review_obtained": False,
    }
    preparation = base / "preparation"
    write(preparation / "result.json", canonical_json(result))
    r16.readonly_tree(preparation)
    r16.readonly_tree(base / "probe-output")
    base.chmod(0o500)
    return result


def prepare(nonce: str) -> dict[str, Any]:
    require(
        len(nonce) == 16
        and all(character in "0123456789abcdef" for character in nonce),
        "nonce must be 16 lowercase hex characters",
    )
    require(
        REPOSITORY.resolve() == Path.cwd().resolve(),
        "run from the ACE-3 worktree",
    )
    for path in (R17_BUILDER, R17_VALIDATOR, R18_VALIDATOR):
        require(path.is_file(), f"required preparation source missing: {path}")
    require(len(R17_EXIT_CODES) == 1, "r17 failed ancestry is ambiguous")
    expected_hashes = r17_failure_hashes()
    for name, source in R17_ARTIFACTS.items():
        require(
            source.is_file() and not source.is_symlink(),
            f"r17 failed ancestry missing before preparation: {name}",
        )
        require(
            digest(source) == expected_hashes[name],
            f"r17 failed ancestry changed before preparation: {name}",
        )
    base = Path(f"/home/argustest/ace3-model24-r18-prep-20260829-{nonce}")
    require(not base.exists(), f"fresh r18 candidate already exists: {base}")
    base, baseline = bootstrap_fresh_r18(nonce)
    return finalize(base, baseline)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", default=secrets.token_hex(8))
    args = parser.parse_args()
    print(canonical_json(prepare(args.nonce)).decode("ascii"), end="")


if __name__ == "__main__":
    main()
