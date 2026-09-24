#!/usr/bin/env python3
"""Accept an authenticated position-3 package into an inert preflight record."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from prepare_position3_continuation import (
    CHECKPOINT_SHA256,
    HIDDEN_SIZE,
    LAYER_COUNT,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    POSITION0_TOKEN_ID,
    POSITION1_TOKEN_ID,
    POSITION2_TOKEN_ID,
    POSITION3,
    PreparationError,
    canonical_json,
    file_record,
    load_json,
    validate as validate_prepared_package,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_DIR = (
    ROOT / "build/model24_selected_token_position3_preparation/package"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "build/model24_selected_token_position3_launch_acceptance/preflight"
)
MANIFEST_NAME = "launch_manifest.json"
EMBEDDING_NAME = "position3_input.hex"
PREFLIGHT_NAME = "launch_preflight.json"
NUMERIC_PROFILE = (
    "native asymmetric packed INT4 AWQ W4A16 G128, no qzero plus-one "
    "adjustment, FP16 activations and FP16 K/V"
)
TRAVERSAL_OPERATION = "selected-token-position3-full-traversal"
RECHECK_CONDITION = (
    "recheck when "
    "build/model24_selected_token_position3_preparation/package/"
    "launch_manifest.json exists with status READY and passes fresh validation "
    "of its exact embedding, token history, 24-layer FP16 K/V parentage, "
    "position-2 parents, source/artifact closure, and "
    "selected-token-position3-full-traversal operation"
)
SOURCE_PATHS = {
    "position3_launch_acceptor": (
        "ace3/model/accept_position3_traversal_launch.py"
    ),
    "focused_tests": (
        "ace3/model/tests/test_accept_position3_traversal_launch.py"
    ),
    "contract": "ace3/contracts/position3_traversal_launch_acceptance.json",
    "position3_package_preparer": (
        "ace3/model/prepare_position3_continuation.py"
    ),
    "position3_package_contract": (
        "ace3/contracts/position3_continuation_preparation.json"
    ),
    "makefile": "Makefile",
}

PackageAuthenticator = Callable[[Path], dict[str, Any]]


class LaunchAcceptanceError(RuntimeError):
    """Raised when a package cannot receive inert position-3 preflight."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchAcceptanceError(message)


def record_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": record.get("path"),
        "bytes": record.get("bytes"),
        "sha256": record.get("sha256"),
    }


def authenticate_record(
    record: Any,
    label: str,
    *,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    require(isinstance(record, dict), f"{label} record is malformed")
    require(isinstance(record.get("path"), str), f"{label} path is missing")
    path = Path(record["path"])
    require(path.is_file(), f"{label} is missing: {path}")
    actual = file_record(path)
    require(record_fields(record) == actual, f"{label} content binding mismatch")
    if expected_path is not None:
        require(
            actual["path"] == str(expected_path.resolve()),
            f"{label} substituted path binding",
        )
    return actual


def authenticate_record_closure(records: Any, label: str) -> dict[str, Any]:
    require(isinstance(records, dict) and records, f"{label} closure is incomplete")
    return {
        name: authenticate_record(record, f"{label} {name}")
        for name, record in sorted(records.items())
    }


def source_records(
    repository_root: Path = ROOT,
    paths: Mapping[str, str] = SOURCE_PATHS,
) -> dict[str, dict[str, Any]]:
    return {
        label: file_record(repository_root / relative)
        for label, relative in paths.items()
    }


def authenticate_ready_package(package_dir: Path) -> dict[str, Any]:
    """Run the package preparer's fresh validator against its exact parents."""
    require(
        package_dir.is_dir(),
        f"position-3 READY package is missing: {package_dir}",
    )
    manifest_path = package_dir / MANIFEST_NAME
    require(
        manifest_path.is_file(),
        f"position-3 READY launch manifest is missing: {manifest_path}",
    )
    candidate = load_json(manifest_path)
    parents = candidate.get("parents")
    require(isinstance(parents, dict), "position-3 package parents are missing")
    traversal = parents.get("position2_traversal")
    lm_head = parents.get("position2_lm_head")
    require(
        isinstance(traversal, dict)
        and isinstance(traversal.get("path"), str)
        and isinstance(lm_head, dict)
        and isinstance(lm_head.get("path"), str),
        "position-3 package parent paths are missing",
    )
    return validate_prepared_package(
        Path(traversal["path"]),
        Path(lm_head["path"]),
        package_dir,
    )


def require_embedding(path: Path) -> None:
    payload = path.read_bytes()
    require(payload.endswith(b"\n"), "position-3 embedding lacks final newline")
    lines = payload.splitlines()
    require(
        len(lines) == HIDDEN_SIZE,
        "position-3 embedding element count mismatch",
    )
    require(
        payload == b"".join(line + b"\n" for line in lines),
        "position-3 embedding row encoding mismatch",
    )
    for index, line in enumerate(lines):
        try:
            row = line.decode("ascii")
            row_index = int(row[2:6], 16)
            int(row[6:10], 16)
        except (UnicodeDecodeError, ValueError) as error:
            raise LaunchAcceptanceError(
                f"position-3 embedding row {index} is malformed"
            ) from error
        require(
            len(row) == 10 and row[:2] == "00" and row_index == index,
            f"position-3 embedding row {index} ordering mismatch",
        )


def authenticate_manifest(
    package_dir: Path,
    package_authenticator: PackageAuthenticator,
    *,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
) -> dict[str, Any]:
    require(
        package_dir.is_dir(),
        f"position-3 READY package is missing: {package_dir}",
    )
    require(
        {path.name for path in package_dir.iterdir()}
        == {MANIFEST_NAME, EMBEDDING_NAME},
        "position-3 package artifact closure mismatch",
    )
    manifest_path = package_dir / MANIFEST_NAME
    manifest = package_authenticator(package_dir)
    require(isinstance(manifest, dict), "package authenticator returned no manifest")
    require(
        manifest == load_json(manifest_path),
        "authenticated package manifest does not match stored manifest",
    )
    require(
        manifest.get("schema_version") == 1
        and manifest.get("kind")
        == "ace3_selected_token_position3_traversal_launch_package"
        and manifest.get("status") == "READY",
        "position-3 package is not READY",
    )
    require(
        manifest.get("model")
        == {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "numeric_profile": NUMERIC_PROFILE,
        },
        "position-3 model binding mismatch",
    )

    position3_input = manifest.get("position3_input")
    require(isinstance(position3_input, dict), "position-3 input is missing")
    selected_token = position3_input.get("selected_token_id")
    selected_logit = position3_input.get("selected_logit_f16_bits")
    prompt_history = [POSITION0_TOKEN_ID, POSITION1_TOKEN_ID, POSITION2_TOKEN_ID]
    require(
        position3_input.get("position") == POSITION3
        and isinstance(selected_token, int)
        and 0 <= selected_token < 151936
        and isinstance(selected_logit, int)
        and 0 <= selected_logit <= 0xFFFF,
        "position-3 selected-token input binding mismatch",
    )
    require(
        position3_input.get("prompt_token_history") == prompt_history
        and position3_input.get("traversal_token_history")
        == [*prompt_history, selected_token],
        "position-3 token history binding mismatch",
    )
    embedding = position3_input.get("embedding")
    expected_embedding = package_dir / EMBEDDING_NAME
    embedding_record = authenticate_record(
        embedding,
        "position-3 embedding",
        expected_path=expected_embedding,
    )
    require(
        embedding.get("dtype") == "FP16"
        and embedding.get("elements") == HIDDEN_SIZE
        and embedding.get("tensor") == "model.embed_tokens.weight"
        and embedding.get("token_id") == selected_token,
        "position-3 embedding metadata binding mismatch",
    )
    require_embedding(expected_embedding)

    parents = manifest.get("parents")
    require(
        isinstance(parents, dict)
        and set(parents) == {"position2_traversal", "position2_lm_head"},
        "position-3 parent closure mismatch",
    )
    for name, record in parents.items():
        authenticate_record(record, f"position-3 parent {name}")

    source_bindings = manifest.get("source_bindings")
    require(
        isinstance(source_bindings, dict)
        and set(source_bindings)
        == {"preparation", "position2_traversal", "position2_lm_head"},
        "position-3 source binding closure mismatch",
    )
    for name, records in source_bindings.items():
        authenticate_record_closure(records, f"position-3 sources {name}")

    kv_parentage = manifest.get("layer_kv_parentage")
    require(isinstance(kv_parentage, dict), "position-3 K/V parentage is missing")
    layers = kv_parentage.get("layers")
    require(
        kv_parentage.get("source_position") == 2
        and kv_parentage.get("target_position") == POSITION3
        and kv_parentage.get("layer_order") == list(range(LAYER_COUNT))
        and isinstance(layers, list)
        and len(layers) == LAYER_COUNT,
        "position-3 K/V parentage header mismatch",
    )

    artifacts = manifest.get("consumed_artifacts")
    require(
        isinstance(artifacts, dict)
        and set(artifacts)
        == {
            "official_checkpoint",
            "position2_lm_head_artifacts",
            "position2_lm_head_binary",
            "position2_lm_head_terminal_log",
            "position2_layer_states",
        },
        "position-3 consumed artifact closure mismatch",
    )
    checkpoint = authenticate_record(
        artifacts["official_checkpoint"], "official checkpoint"
    )
    require(
        checkpoint["sha256"] == expected_checkpoint_sha256,
        "official checkpoint SHA-256 mismatch",
    )
    authenticate_record_closure(
        artifacts["position2_lm_head_artifacts"],
        "position-2 lm_head artifacts",
    )
    authenticate_record(
        artifacts["position2_lm_head_binary"], "position-2 lm_head binary"
    )
    authenticate_record(
        artifacts["position2_lm_head_terminal_log"],
        "position-2 lm_head terminal log",
    )
    layer_artifacts = artifacts["position2_layer_states"]
    require(
        isinstance(layer_artifacts, list)
        and len(layer_artifacts) == LAYER_COUNT,
        "position-2 layer artifact closure mismatch",
    )
    state_names = (
        "position1_predecessor_state",
        "position2_output_state",
        "position2_output_hidden",
    )
    for layer_index, (layer, layer_artifact) in enumerate(
        zip(layers, layer_artifacts)
    ):
        require(
            isinstance(layer, dict)
            and layer.get("layer_index") == layer_index
            and layer.get("parent_position") == 2
            and layer.get("next_position") == POSITION3,
            f"position-3 layer {layer_index} K/V parentage mismatch",
        )
        require(
            isinstance(layer_artifact, dict)
            and layer_artifact.get("layer_index") == layer_index,
            f"position-3 layer {layer_index} artifact ordering mismatch",
        )
        for state_name in state_names:
            require(
                layer.get(state_name) == layer_artifact.get(state_name),
                f"position-3 layer {layer_index} {state_name} binding mismatch",
            )
            authenticate_record(
                layer[state_name],
                f"position-3 layer {layer_index} {state_name}",
            )

    launch = manifest.get("launch")
    require(
        launch
        == {
            "operation": TRAVERSAL_OPERATION,
            "input_position": POSITION3,
            "layer_order": list(range(LAYER_COUNT)),
            "required_embedding": embedding_record["path"],
            "required_fp16_kv_state_count": LAYER_COUNT,
            "execution_performed": False,
            "execution_authority": False,
        },
        "position-3 intended traversal operation mismatch",
    )
    claim_boundary = manifest.get("claim_boundary")
    require(
        isinstance(claim_boundary, dict)
        and claim_boundary.get("position3_traversal") == "not executed"
        and claim_boundary.get("dialogue") == "not claimed",
        "position-3 package claim boundary mismatch",
    )
    return manifest


def assemble_preflight(
    package_dir: Path,
    manifest: Mapping[str, Any],
    own_sources: Mapping[str, Any],
) -> dict[str, Any]:
    position3_input = manifest["position3_input"]
    return {
        "schema_version": 1,
        "kind": "ace3_position3_traversal_inert_launch_authority_preflight",
        "status": "ACCEPTED",
        "model": manifest["model"],
        "accepted_package": {
            "launch_manifest": file_record(package_dir / MANIFEST_NAME),
            "position3_embedding": record_fields(position3_input["embedding"]),
            "parents": manifest["parents"],
        },
        "validated_inputs": {
            "position": POSITION3,
            "selected_token_id": position3_input["selected_token_id"],
            "selected_logit_f16_bits": position3_input[
                "selected_logit_f16_bits"
            ],
            "prompt_token_history": position3_input["prompt_token_history"],
            "traversal_token_history": position3_input[
                "traversal_token_history"
            ],
            "kv_source_position": 2,
            "kv_target_position": POSITION3,
            "kv_parent_layers": LAYER_COUNT,
            "package_source_bindings": manifest["source_bindings"],
            "package_artifact_bindings": manifest["consumed_artifacts"],
        },
        "acceptance_source_bindings": dict(own_sources),
        "launch_authority": {
            "operation": TRAVERSAL_OPERATION,
            "state": "INERT_PREFLIGHT_ONLY",
            "all_required_bindings_valid": True,
            "execution_authority": False,
            "durable_launch_authority": False,
            "execution_performed": False,
            "traversal_output_created": False,
        },
        "claim_boundary": {
            "demonstrated": (
                "fresh fail-closed acceptance of an authenticated READY "
                "position-3 launch package into an inert preflight record"
            ),
            "position3_traversal": "not executed",
            "dialogue": "not claimed",
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
            "latency": "not measured",
            "throughput": "not measured",
        },
    }


def accept(
    package_dir: Path,
    output_dir: Path,
    *,
    package_authenticator: PackageAuthenticator = authenticate_ready_package,
    repository_root: Path = ROOT,
    own_source_paths: Mapping[str, str] = SOURCE_PATHS,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
) -> dict[str, Any]:
    require(not output_dir.exists(), f"preflight output already exists: {output_dir}")
    manifest = authenticate_manifest(
        package_dir,
        package_authenticator,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    own_sources = source_records(repository_root, own_source_paths)
    document = assemble_preflight(package_dir, manifest, own_sources)
    output_dir.mkdir(parents=True)
    (output_dir / PREFLIGHT_NAME).write_bytes(canonical_json(document))
    return document


def validate(
    package_dir: Path,
    output_dir: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    require(
        output_dir.is_dir()
        and {path.name for path in output_dir.iterdir()} == {PREFLIGHT_NAME},
        "position-3 preflight artifact closure mismatch",
    )
    stored = load_json(output_dir / PREFLIGHT_NAME)
    manifest = authenticate_manifest(
        package_dir,
        kwargs.get("package_authenticator", authenticate_ready_package),
        expected_checkpoint_sha256=kwargs.get(
            "expected_checkpoint_sha256", CHECKPOINT_SHA256
        ),
    )
    fresh = assemble_preflight(
        package_dir,
        manifest,
        source_records(
            kwargs.get("repository_root", ROOT),
            kwargs.get("own_source_paths", SOURCE_PATHS),
        ),
    )
    require(stored == fresh, "stored position-3 preflight record is stale")
    return stored


def print_summary(document: Mapping[str, Any], output_dir: Path) -> None:
    authority = document["launch_authority"]
    print(
        "POSITION3_TRAVERSAL_LAUNCH_PREFLIGHT_PASS "
        f"status={document['status']} "
        f"selected_token={document['validated_inputs']['selected_token_id']} "
        f"kv_parent_layers={document['validated_inputs']['kv_parent_layers']} "
        f"execution_authority={int(authority['execution_authority'])} "
        f"execution_performed={int(authority['execution_performed'])} "
        f"record={output_dir / PREFLIGHT_NAME}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("accept", "validate"))
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        if args.operation == "accept":
            document = accept(
                args.package_dir.resolve(),
                args.output_dir.resolve(),
            )
        else:
            document = validate(
                args.package_dir.resolve(strict=True),
                args.output_dir.resolve(strict=True),
            )
    except (
        LaunchAcceptanceError,
        PreparationError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        raise SystemExit(
            f"POSITION3_TRAVERSAL_LAUNCH_NOT_READY {error}; "
            f"{RECHECK_CONDITION}"
        ) from error
    print_summary(document, args.output_dir.resolve())


if __name__ == "__main__":
    main()
