#!/usr/bin/env python3
"""Prepare and inertly validate the Model24 r17 binding-closure package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any


REPOSITORY = Path("/home/argustest/ace3-argus")
R15_BUILDER = REPOSITORY / "ace3/model/prepare_model24_r15_launch_package.py"
R15_VALIDATOR = REPOSITORY / "ace3/model/validate_model24_r15_launch_package.py"
R15_LIFECYCLE = REPOSITORY / "ace3/model/model24_r15_lifecycle.py"
R15_CONTRACT = REPOSITORY / "ace3/contracts/model24_r15_durable_launch_contract.json"
R16_BUILDER = REPOSITORY / "ace3/model/prepare_model24_r16_launch_package.py"
R16_VALIDATOR = REPOSITORY / "ace3/model/validate_model24_r16_launch_package.py"
R17_VALIDATOR = REPOSITORY / "ace3/model/validate_model24_r17_launch_package.py"
MODEL_PYTHON = Path("/home/argustest/miniconda3/bin/python3")
R14_NONCE = "03820ca8ec7835ca"
R15_NONCE = "f0742dfe9b4caa45"
R16_NONCE = "bbcf2022e008a933"
R16_BASE = Path(f"/home/argustest/ace3-model24-r16-prep-20260829-{R16_NONCE}")
R16_TASK_ID = f"ace3-model24-r16-42c895c-{R16_NONCE}"
R16_TERMINAL = Path(
    f"/home/argustest/ace3-model24-r16-terminal-20260829-{R16_NONCE}"
)
R16_LOGS = (
    R16_TERMINAL
    / ".argus_subagents"
    / f"{R16_TASK_ID}_logs"
)
R16_EXIT_CODES = list(R16_LOGS.glob("exit_code.*"))
V2_SOURCE_TREE_SHA256 = (
    "b5228db8d3513ab29059bc55d446186fa8440f0cfc2283da28c61e1fe9eb156c"
)
R16_ARTIFACTS = {
    "package.json": R16_BASE / "package/package.json",
    "seal.json": R16_BASE / "package/seal.json",
    "review.json": Path(
        f"/home/argustest/ace3-model24-r16-review-20260829-{R16_NONCE}/review.json"
    ),
    "authority.json": Path(
        f"/home/argustest/ace3-model24-r16-authority-20260829-{R16_NONCE}.json"
    ),
    "authority-consumed.json": Path(
        f"/home/argustest/ace3-model24-r16-authority-20260829-{R16_NONCE}.json.consumed"
    ),
    "launch-terminal.json": Path(
        f"/home/argustest/ace3-model24-r16-output-20260829-{R16_NONCE}/launch-terminal.json"
    ),
    "terminal-manifest.json": R16_TERMINAL / "manifest.json",
    "runner-receipt.json": (
        R16_TERMINAL / ".argus_subagents" / f"{R16_TASK_ID}.json"
    ),
    "stdout.log": R16_LOGS / "stdout.log",
    "stderr.log": R16_LOGS / "stderr.log",
    "exit-code.txt": R16_EXIT_CODES[0] if len(R16_EXIT_CODES) == 1 else Path(),
    "bindings.json": R16_BASE / "package/bindings.json",
}


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


def transform_revision(payload: str, source: str, target: str) -> str:
    return payload.replace(source, target)


def preparation_only_controller_fixture(builder_text: str) -> str:
    original = """        controller_artifacts = lifecycle.materialize_controller_simulation(
            materializer_manifest,
            positive_paths,
            environment,
        )
"""
    replacement = """        if os.environ.get("ACE3_R17_PREPARATION_ONLY") == "1":
            vectors = positive_simulation / "vectors"
            verilator = positive_simulation / "verilator"
            vectors.mkdir()
            verilator.mkdir()
            (positive_simulation / "terminal.txt").write_text(
                lifecycle.CONTROLLER_TERMINAL,
                encoding="ascii",
            )
            (positive_simulation / "controller_events.hex").write_bytes(
                lifecycle.expected_controller_events()
            )
            (vectors / "cascade_events.hex").write_bytes(b"inert fixture\\n")
            (vectors / "manifest.json").write_bytes(b"{}\\n")
            simulator = verilator / "Vace3_model24_layer_controller"
            simulator.write_bytes(b"inert preparation fixture\\n")
            simulator.chmod(0o500)
            controller_artifacts = (
                lifecycle.validate_controller_simulation_artifacts(
                    positive_simulation
                )
            )
        else:
            controller_artifacts = lifecycle.materialize_controller_simulation(
                materializer_manifest,
                positive_paths,
                environment,
            )
"""
    require(original in builder_text, "controller preparation fixture seam missing")
    return builder_text.replace(original, replacement)


def bootstrap_fresh_r17(nonce: str) -> Path:
    for path in (
        R15_BUILDER,
        R15_VALIDATOR,
        R15_LIFECYCLE,
        R15_CONTRACT,
    ):
        require(path.is_file(), f"required construction source missing: {path}")
    with tempfile.TemporaryDirectory(
        prefix="ace3-model24-r17-scaffold-"
    ) as scratch_name:
        scratch = Path(scratch_name)
        validator = scratch / "validate-package.py"
        lifecycle = scratch / "lifecycle.py"
        contract = scratch / "launch-contract.json"
        builder = scratch / "repository-builder.py"
        write(
            validator,
            transform_revision(
                R15_VALIDATOR.read_text(encoding="ascii"),
                "r15",
                "r17",
            ).encode("ascii"),
            0o500,
        )
        write(
            lifecycle,
            transform_revision(
                R15_LIFECYCLE.read_text(encoding="ascii"),
                "r15",
                "r17",
            ).encode("ascii"),
            0o500,
        )
        write(
            contract,
            transform_revision(
                R15_CONTRACT.read_text(encoding="ascii"),
                "r15",
                "r17",
            ).encode("ascii"),
        )
        builder_text = transform_revision(
            R15_BUILDER.read_text(encoding="ascii"),
            "r15",
            "r17",
        )
        builder_text = builder_text.replace(
            'CONTRACT = REPOSITORY / "ace3/contracts/model24_r17_durable_launch_contract.json"',
            f"CONTRACT = Path({str(contract)!r})",
        ).replace(
            'VALIDATOR = REPOSITORY / "ace3/model/validate_model24_r17_launch_package.py"',
            f"VALIDATOR = Path({str(validator)!r})",
        ).replace(
            'LIFECYCLE = REPOSITORY / "ace3/model/model24_r17_lifecycle.py"',
            f"LIFECYCLE = Path({str(lifecycle)!r})",
        ).replace(
            'write_new(package / "bindings.json", BINDINGS_SOURCE.read_bytes(), 0o400)',
            'write_new(package / "bindings.json", b"{}\\n", 0o400)',
        )
        builder_text = preparation_only_controller_fixture(builder_text)
        builder_text = builder_text.replace(
            '"simulator_invocations": 1,',
            '"simulator_invocations": ('
            '0 if os.environ.get("ACE3_R17_PREPARATION_ONLY") == "1" else 1'
            '),',
        )
        write(builder, builder_text.encode("ascii"), 0o500)
        module = load_module(builder, "model24_r17_fresh_scaffold_builder")

        def inert_runner_probe(
            package: Path,
            unused_nonce: str,
            runner_sources: list[dict[str, Any]],
        ) -> dict[str, Any]:
            del unused_nonce
            root = package / "inert-runner-probe"
            root.mkdir()
            evidence = {
                "status": "PREPARATION_ONLY_NOT_SUBMITTED",
                "payload_invocation_count": 1,
                "runner_sources": runner_sources,
                "subagent_invocations": 0,
            }
            module.write_new(
                root / "evidence.json",
                module.canonical_json(evidence),
                0o400,
            )
            return evidence

        module.run_installed_runner_probe = inert_runner_probe
        previous = os.environ.get("ACE3_R17_PREPARATION_ONLY")
        os.environ["ACE3_R17_PREPARATION_ONLY"] = "1"
        try:
            result = module.prepare(nonce)
        finally:
            if previous is None:
                os.environ.pop("ACE3_R17_PREPARATION_ONLY", None)
            else:
                os.environ["ACE3_R17_PREPARATION_ONLY"] = previous
        require(
            result["status"] == "SEALED_AWAITING_INDEPENDENT_L2_REVIEW",
            "fresh r17 scaffold did not seal",
        )
    return Path(f"/home/argustest/ace3-model24-r17-prep-20260829-{nonce}")


def install_production_binding_preflight(lifecycle: Path) -> None:
    source = lifecycle.read_text(encoding="ascii")
    require("import sys\n" not in source, "lifecycle already contains sys import")
    source = source.replace("import subprocess\n", "import subprocess\nimport sys\n")
    marker = "\ndef submit(package: Path, review: Path, authority: Path) -> None:\n"
    require(marker in source, "lifecycle preflight insertion point missing")
    preflight = '''
def validate_production_bindings(package: Path) -> dict[str, str]:
    source = package.parent / "source"
    model_root = source / "ace3/model"
    controller_path = model_root / "controller_model24_cascade.py"
    spec = importlib.util.spec_from_file_location(
        "model24_r17_production_binding_preflight",
        controller_path,
    )
    if spec is None or spec.loader is None:
        raise SystemExit("materialized binding controller is unavailable")
    sys.path.insert(0, str(model_root))
    try:
        controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller)
    finally:
        sys.path.pop(0)
    bindings_path = package / "bindings.json"
    tensor_map_path = source / "ace3/contracts/model24_tensor_map.json"
    if not bindings_path.is_file() or bindings_path.is_symlink():
        raise SystemExit("final sealed binding is missing")
    if not tensor_map_path.is_file() or tensor_map_path.is_symlink():
        raise SystemExit("exact materialized tensor map is missing")
    binding_payload = bindings_path.read_bytes()
    tensor_map_payload = tensor_map_path.read_bytes()
    document = controller._load_json(binding_payload, "layer bindings")
    controller.validate_binding_document(document, source, tensor_map_payload)
    return {
        "status": "PASS",
        "validator": "controller_model24_cascade.validate_binding_document",
        "bindings_sha256": hashlib.sha256(binding_payload).hexdigest(),
        "tensor_map_sha256": hashlib.sha256(tensor_map_payload).hexdigest(),
    }

'''
    source = source.replace(marker, preflight + marker)
    launch_marker = """    manifest = validator.load_json(package / "package.json")
    paths = validator.expected_paths(package, manifest)
    consumed = paths["authority_consumed"]
"""
    launch_replacement = """    manifest = validator.load_json(package / "package.json")
    paths = validator.expected_paths(package, manifest)
    validate_production_bindings(package)
    consumed = paths["authority_consumed"]
"""
    require(launch_marker in source, "launch binding preflight seam missing")
    lifecycle.write_text(
        source.replace(launch_marker, launch_replacement, 1),
        encoding="ascii",
    )
    lifecycle.chmod(0o500)


def extend_binding_negative_tests(builder: Path) -> None:
    source = builder.read_text(encoding="ascii")
    marker = """    cases.append("predecessor-review-or-authority-reuse")

    require(
        cases == list(validator.REQUIRED_NEGATIVE_CASES),
"""
    require(marker in source, "binding negative-suite insertion point missing")
    extension = """    cases.append("predecessor-review-or-authority-reuse")

    _expect_rejection(
        lambda: validator.validate_binding_payload(
            package,
            binding_path=package / "missing-bindings.json",
        ),
        "final sealed binding is missing",
        "missing-binding",
    )
    cases.append("missing-binding")

    substituted = validator.load_json(package / "bindings.json")
    substituted["sources"][0]["sha256"] = "0" * 64
    _expect_rejection(
        lambda: validator.validate_binding_payload(
            package,
            binding_payload=validator.canonical_json(substituted),
        ),
        "layer binding manifest mismatch",
        "substituted-binding",
    )
    cases.append("substituted-binding")

    _expect_rejection(
        lambda: validator.validate_binding_payload(
            package,
            binding_payload=(
                package / "provenance/r16-failed/bindings.json"
            ).read_bytes(),
        ),
        "layer binding manifest mismatch",
        "r16-stale-binding",
    )
    cases.append("r16-stale-binding")

    tensor_map = (
        package.parent / "source/ace3/contracts/model24_tensor_map.json"
    ).read_bytes()
    _expect_rejection(
        lambda: validator.validate_binding_payload(
            package,
            tensor_map_payload=tensor_map + b" ",
        ),
        "tensor map SHA256 mismatch",
        "tensor-map-mutation",
    )
    cases.append("tensor-map-mutation")

    require(
        cases == list(validator.REQUIRED_NEGATIVE_CASES),
"""
    builder.chmod(0o600)
    builder.write_text(source.replace(marker, extension), encoding="ascii")
    builder.chmod(0o400)


def copy_r16_failed_ancestry(package: Path, validator: Any) -> dict[str, Any]:
    require(len(R16_EXIT_CODES) == 1, "r16 exit-code evidence is ambiguous")
    root = package / "provenance/r16-failed"
    root.mkdir(parents=True)
    for name, source in R16_ARTIFACTS.items():
        require(
            source.is_file() and not source.is_symlink(),
            f"r16 failed ancestry missing: {source}",
        )
        require(
            digest(source) == validator.R16_FAILURE_HASHES[name],
            f"r16 failed ancestry hash mismatch: {name}",
        )
        shutil.copy2(source, root / name)
        (root / name).chmod(0o400)
    classification = {
        "schema_version": 1,
        "kind": "ace3_model24_r16_consumed_failed_ancestry",
        "nonce": R16_NONCE,
        "classification": "LAWFUL_CONSUMED_FAILED",
        "sole_lawful_invocation": True,
        "historical_execution_invocations": 1,
        "authority_consumed": True,
        "natural_terminal": False,
        "exit_code": 1,
        "review_reusable": False,
        "authority_reusable": False,
        "execution_inputs_reusable": False,
        "copied_byte_for_byte": True,
        "hashes": validator.R16_FAILURE_HASHES,
    }
    write(root / "classification.json", canonical_json(classification))
    return classification


def materialize_r16_source_closure(base: Path, r16: Any) -> tuple[str, dict[str, Any]]:
    package = base / "package"
    source = base / "source"
    r16.writable_tree(base)
    baseline = load_json(package / "source-tree.json")
    write(
        package / "provenance/accepted-source-tree.json",
        canonical_json(baseline),
    )
    for name, source_path in r16.V2_SOURCE_FILES.items():
        require(
            digest(source_path) == r16.V2_HASHES[name],
            f"supplied v2 hash mismatch: {name}",
        )
        write(
            package / "provenance/v2-compile-source" / name,
            source_path.read_bytes(),
            0o500 if name == "validate-package.py" else 0o400,
        )
    overlays = r16.authenticated_overlays(base)
    overlay_document = load_json(package / "source-overlays.json")
    overlay_document["kind"] = "ace3_model24_r17_exact_source_overlays"
    write(package / "source-overlays.json", canonical_json(overlay_document))
    r16.copy_unauthorized_ancestry(package, 14, R14_NONCE)
    r16.copy_unauthorized_ancestry(package, 15, R15_NONCE)

    r16.readonly_tree(source)
    source_records = r16.tree_records(source)
    source_hash = r16.tree_sha256(source_records)
    r16.writable_tree(source)
    preflight = r16.production_preflight(base, source_hash)
    preflight_path = package / "evidence/production-preflight/result.json"
    preflight["kind"] = "ace3_model24_r17_production_preflight"
    write(preflight_path, canonical_json(preflight))
    r16.readonly_tree(source)
    require(
        r16.tree_records(source) == source_records,
        "production preflight changed sealed source",
    )
    source_manifest = {
        "schema_version": 1,
        "kind": "ace3_model24_r17_materialized_source_tree",
        "base_commit": r16.ACCEPTED_COMMIT,
        "base_tree_sha256": baseline["tree_sha256"],
        "overlay_paths": list(r16.OVERLAY_PATHS),
        "records": source_records,
        "tree_sha256": source_hash,
    }
    write(package / "source-tree.json", canonical_json(source_manifest))
    return source_hash, {"records": overlays, "preflight": preflight}


def generate_binding_closure(base: Path, validator: Any) -> dict[str, Any]:
    package = base / "package"
    source = base / "source"
    model_root = source / "ace3/model"
    sys.path.insert(0, str(model_root))
    try:
        controller = load_module(
            model_root / "controller_model24_cascade.py",
            "model24_r17_binding_generator",
        )
    finally:
        sys.path.pop(0)
    tensor_map = source / "ace3/contracts/model24_tensor_map.json"
    tensor_payload = tensor_map.read_bytes()
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
    require(len(differences) == 6, "r16 binding mismatch diagnosis is incomplete")
    result = {
        "schema_version": 1,
        "kind": "ace3_model24_r17_binding_closure",
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


def seal_and_validate(base: Path, source_hash: str, r16: Any) -> dict[str, Any]:
    package = base / "package"
    source = base / "source"
    validation_root = base / "validation"
    preparation = base / "preparation"
    for root in (validation_root, preparation):
        if root.exists():
            shutil.rmtree(root)
        root.mkdir()
    for root in (
        package / "inert-runner-probe",
        package / "inert-controller-entry-probe",
    ):
        if root.exists():
            shutil.rmtree(root)
    seal_path = package / "seal.json"
    if seal_path.exists():
        seal_path.unlink()

    base_validator = package / "validate-package.py"
    base_validator.rename(package / "validate-base-package.py")
    adapted_r16_validator = transform_revision(
        R16_VALIDATOR.read_text(encoding="ascii"),
        "r16",
        "r17",
    )
    write(
        package / "validate-r16-package.py",
        adapted_r16_validator.encode("ascii"),
        0o500,
    )
    write(package / "validate-package.py", R17_VALIDATOR.read_bytes(), 0o500)
    install_production_binding_preflight(package / "lifecycle.py")
    validator = load_module(
        package / "validate-package.py",
        "model24_r17_contract_validator",
    )
    copy_r16_failed_ancestry(package, validator)
    binding_closure = generate_binding_closure(base, validator)
    r16.extend_negative_tests(package / "ancestry/repository-builder.py")
    extend_binding_negative_tests(package / "ancestry/repository-builder.py")

    contract = load_json(package / "launch-contract.json")
    contract["review"]["required_negative_cases"] = list(
        validator.REQUIRED_NEGATIVE_CASES
    )
    contract["review"]["required_bound_hashes"] = list(
        validator.REQUIRED_REVIEW_HASHES
    )
    contract["source_materialization"] = {
        "accepted_commit": r16.ACCEPTED_COMMIT,
        "v2_interpretation": "compile_evidence_only",
        "external_wrapper": False,
        "overlay_paths": list(r16.OVERLAY_PATHS),
    }
    contract["production_preflight"] = {
        "required_before_model24": True,
        "layer0_compile_required": True,
        "accurate_silu": 1,
        "cascade_make_target_resolution_required": True,
        "fresh_compiled_binary_authentication_required": True,
        "binding_validator": (
            "controller_model24_cascade.validate_binding_document"
        ),
        "binding_validation_before_authority_consumption": True,
        "binding_validation_before_output_creation": True,
        "binding_validation_before_controller_simulation": True,
        "binding_validation_before_payload_execution": True,
    }
    contract["ancestry"] = {
        "r12_terminal_preserved": True,
        "r13_terminal_preserved": True,
        "r14_execution_authorized": False,
        "r15_execution_authorized": False,
        "r16_lawful_consumed_failed": True,
        "r16_review_reusable": False,
        "r16_authority_reusable": False,
        "r16_execution_inputs_reusable": False,
        "predecessor_review_reusable": False,
        "predecessor_authority_reusable": False,
    }
    write(package / "launch-contract.json", canonical_json(contract))

    manifest = load_json(package / "package.json")
    manifest["provenance"]["source"].update(
        {
            "materialization": "git archive plus exact authenticated v2 overlays",
            "tree_sha256": source_hash,
            "overlay_paths": list(r16.OVERLAY_PATHS),
        }
    )
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
    manifest["provenance"]["v2_compile_source"] = {
        "candidate": str(r16.V2),
        "interpretation": "compile_evidence_only",
        "external_wrapper": False,
        "hashes": r16.V2_HASHES,
        "source_tree_sha256": V2_SOURCE_TREE_SHA256,
    }
    manifest["provenance"]["source_overlays"] = {
        "paths": list(r16.OVERLAY_PATHS),
        "records": load_json(package / "source-overlays.json")["records"],
        "manifest_sha256": digest(package / "source-overlays.json"),
    }
    for revision in (14, 15):
        manifest["provenance"][f"r{revision}_terminal_ancestry"] = {
            "classification": "UNAUTHORIZED_NO_MANAGER_DIRECTIVE",
            "classification_sha256": digest(
                package
                / f"provenance/r{revision}-unauthorized/classification.json"
            ),
        }
    manifest["provenance"]["r16_terminal_ancestry"] = {
        "classification": "LAWFUL_CONSUMED_FAILED",
        "classification_sha256": digest(
            package / "provenance/r16-failed/classification.json"
        ),
        "review_reusable": False,
        "authority_reusable": False,
        "execution_inputs_reusable": False,
    }
    preflight = load_json(
        package / "evidence/production-preflight/result.json"
    )
    manifest["production_preflight"] = {
        "status": "PASS",
        "result_sha256": digest(
            package / "evidence/production-preflight/result.json"
        ),
        "compile_argv": r16.COMPILE_ARGV,
        "binary_sha256": preflight["binary"]["sha256"],
        "binding_validation": {
            "validator": (
                "controller_model24_cascade.validate_binding_document"
            ),
            "before_authority_consumption": True,
            "before_output_creation": True,
            "before_controller_simulation": True,
            "before_payload_execution": True,
        },
        "model24_invocations": 0,
    }
    manifest["review_policy"]["predecessor_review_reusable"] = False
    manifest["review_policy"]["predecessor_authority_reusable"] = False
    write(package / "package.json", canonical_json(manifest))

    paths = validator.expected_paths(package, manifest)
    authority_schema = {
        "schema_version": 1,
        "kind": "ace3_model24_r17_exact_manager_authority_schema",
        "document_kind": "ace3_model24_r17_manager_exactly_once_authority",
        "additional_fields_allowed": False,
        "required_exact_fields": [
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
        ],
        "canonical_paths": {
            "review_path": str(paths["review"] / "review.json"),
            "authority_path": str(paths["authority"]),
            "authority_consumed_path": str(paths["authority_consumed"]),
            "terminal_root": str(paths["terminal_root"]),
            "registry_root": str(paths["registry_root"]),
            "receipt_path": str(paths["receipt"]),
            "stdout_log": str(paths["stdout_log"]),
            "stderr_log": str(paths["stderr_log"]),
            "terminal_manifest": str(paths["terminal_manifest"]),
            "output_namespace": str(paths["output"]),
            "simulation_namespace": str(paths["simulation_dir"]),
            "payload_output_namespace": str(paths["payload_output_dir"]),
        },
        "exact_policy": {
            "execution_cardinality": 1,
            "manager_exactly_once_directive": True,
            "retry": False,
            "replay": False,
            "resume": False,
            "watcher": False,
        },
    }
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
            "lifecycle_sha256": digest(package / "lifecycle.py"),
            "launch_contract_sha256": digest(package / "launch-contract.json"),
            "source_tree_sha256": digest(package / "source-tree.json"),
            "source_overlays_sha256": digest(package / "source-overlays.json"),
            "authority_schema_sha256": digest(
                package / "authority-schema.json"
            ),
            "production_preflight_sha256": digest(
                package / "evidence/production-preflight/result.json"
            ),
            "bindings_sha256": digest(package / "bindings.json"),
            "binding_closure_sha256": digest(
                package / "evidence/binding-closure/result.json"
            ),
            "r16_failed_ancestry_sha256": digest(
                package / "provenance/r16-failed/classification.json"
            ),
            "v2_required_hashes": r16.V2_HASHES,
            "v2_interpretation": "compile_evidence_only",
            "external_wrapper": False,
            "r14_execution_authorized": False,
            "r15_execution_authorized": False,
            "r16_lawful_consumed_failed": True,
            "predecessor_review_reusable": False,
            "predecessor_authority_reusable": False,
        }
    )
    write(package / "review-request.json", canonical_json(review_request))

    binding_test = (
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import importlib.util\n"
        "package=Path(__file__).resolve().parent\n"
        "spec=importlib.util.spec_from_file_location('lifecycle',package/'lifecycle.py')\n"
        "module=importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "result=module.validate_production_bindings(package)\n"
        "print(module.json.dumps(result,sort_keys=True,separators=(',',':')))\n"
    ).encode("ascii")
    write(package / "test-binding-preflight.py", binding_test, 0o500)

    r16.readonly_tree(package)
    package.chmod(0o700)
    package_records = r16.tree_records(package)
    seal = {
        "schema_version": 1,
        "kind": "ace3_model24_r17_package_seal",
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "package_manifest_sha256": digest(package / "package.json"),
        "review_request_sha256": digest(package / "review-request.json"),
        "source_tree_sha256": source_hash,
        "source_tree_manifest_sha256": digest(package / "source-tree.json"),
        "source_overlays_sha256": digest(package / "source-overlays.json"),
        "production_preflight_sha256": digest(
            package / "evidence/production-preflight/result.json"
        ),
        "bindings_sha256": digest(package / "bindings.json"),
        "binding_closure_sha256": digest(
            package / "evidence/binding-closure/result.json"
        ),
        "v2_bound_hashes": r16.V2_HASHES,
        "package_entries": package_records,
        "zero_state": True,
    }
    write(seal_path, canonical_json(seal))
    package.chmod(0o500)

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
        "production-binding-preflight": [
            str(MODEL_PYTHON),
            "-B",
            str(package / "test-binding-preflight.py"),
        ],
    }
    outcomes: dict[str, dict[str, Any]] = {}
    environment = dict(os.environ)
    environment["ACE3_R17_PREPARATION_ONLY"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
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
        "canonical r17 review or execution namespace was created",
    )
    zero_state.update(
        {
            "model24_invocations": 0,
            "controller_simulator_invocations": 0,
            "rtl_simulator_invocations": 0,
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
        "kind": "ace3_model24_r17_inert_validation_seal",
        "package_seal_sha256": digest(seal_path),
        "negative_case_count": len(validator.REQUIRED_NEGATIVE_CASES),
        "positive_fixture_count": len(validator.REQUIRED_POSITIVE_FIXTURES),
        "outcomes": outcomes,
        "records": validation_records,
        **{
            key: zero_state[key]
            for key in (
                "model24_invocations",
                "controller_simulator_invocations",
                "rtl_simulator_invocations",
                "lifecycle_launch_invocations",
                "subagent_invocations",
                "authority_invocations",
            )
        },
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
        "bindings_sha256": digest(package / "bindings.json"),
        "binding_closure_sha256": digest(
            package / "evidence/binding-closure/result.json"
        ),
        "validation_seal_sha256": digest(validation_root / "seal.json"),
        "negative_case_count": len(validator.REQUIRED_NEGATIVE_CASES),
        "positive_fixture_count": len(validator.REQUIRED_POSITIVE_FIXTURES),
        "production_preflight": "PASS",
        "binding_preflight": "PASS",
        "r16_stale_binding": "REJECT",
        "r16_lawful_consumed_failed_ancestry": True,
        "canonical_zero_state": zero_state,
        "execution_authority_withheld": True,
        "review_obtained": False,
    }
    write(preparation / "result.json", canonical_json(result))
    r16.readonly_tree(preparation)
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
    for path in (R16_BUILDER, R16_VALIDATOR, R17_VALIDATOR):
        require(path.is_file(), f"required preparation source missing: {path}")
    require(
        len(R16_EXIT_CODES) == 1,
        "r16 failed ancestry has ambiguous exit evidence",
    )
    base = Path(f"/home/argustest/ace3-model24-r17-prep-20260829-{nonce}")
    require(not base.exists(), f"fresh r17 candidate already exists: {base}")
    r16 = load_module(R16_BUILDER, "model24_r17_r16_source_closure")
    for name, source in R16_ARTIFACTS.items():
        require(
            source.is_file() and not source.is_symlink(),
            f"r16 failed ancestry missing before preparation: {name}",
        )
    base = bootstrap_fresh_r17(nonce)
    source_hash, _ = materialize_r16_source_closure(base, r16)
    return seal_and_validate(base, source_hash, r16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", default=secrets.token_hex(8))
    args = parser.parse_args()
    print(canonical_json(prepare(args.nonce)).decode("ascii"), end="")


if __name__ == "__main__":
    main()
