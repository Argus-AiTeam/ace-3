#!/usr/bin/env python3
"""Prepare the Model24 r19 reboot-resilient full launch package."""

from __future__ import annotations

import argparse
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
R16_BUILDER = REPOSITORY / "ace3/model/prepare_model24_r16_launch_package.py"
R18_BUILDER = REPOSITORY / "ace3/model/prepare_model24_r18_launch_package.py"
R18_VALIDATOR = REPOSITORY / "ace3/model/validate_model24_r18_launch_package.py"
R19_VALIDATOR = REPOSITORY / "ace3/model/validate_model24_r19_launch_package.py"
MODEL_PYTHON = Path("/home/argustest/miniconda3/bin/python3")
R18_NONCE = "33836f56ad76c8e0"
R18_TASK_ID = f"ace3-model24-r18-42c895c-{R18_NONCE}"
R18_BASE = Path(f"/home/argustest/ace3-model24-r18-prep-20260829-{R18_NONCE}")
R18_REVIEW = Path(
    f"/home/argustest/ace3-model24-r18-review-20260829-{R18_NONCE}"
)
R18_AUTHORITY = Path(
    f"/home/argustest/ace3-model24-r18-authority-20260829-{R18_NONCE}.json"
)
R18_OUTPUT = Path(
    f"/home/argustest/ace3-model24-r18-output-20260829-{R18_NONCE}"
)
R18_TERMINAL = Path(
    f"/home/argustest/ace3-model24-r18-terminal-20260829-{R18_NONCE}"
)
R18_RECEIPT_SHA256 = (
    "fd626aff158020a7db1a000c559e2db11ee4ad2a3ee9f1bf991ad8d471552a71"
)
R18_AUTHORITY_SHA256 = (
    "7f494c4e4fa1e56961f05ace96ccd3de454ce18b42dfaa77898ae42c504513f7"
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
    return payload.replace("R18", "R19").replace("r18", "r19")


def bootstrap_fresh_r19(nonce: str) -> tuple[Path, Any]:
    with tempfile.TemporaryDirectory(
        prefix="ace3-model24-r19-scaffold-"
    ) as scratch_name:
        scratch = Path(scratch_name)
        baseline_validator = scratch / "validate-package.py"
        baseline_builder = scratch / "repository-builder.py"
        write(
            baseline_validator,
            transform_revision(
                R18_VALIDATOR.read_text(encoding="ascii")
            ).encode("ascii"),
            0o500,
        )
        builder_text = transform_revision(
            R18_BUILDER.read_text(encoding="ascii")
        )
        expected = (
            'R19_VALIDATOR = REPOSITORY / '
            '"ace3/model/validate_model24_r19_launch_package.py"'
        )
        require(expected in builder_text, "r19 validator substitution seam missing")
        builder_text = builder_text.replace(
            expected,
            f"R19_VALIDATOR = Path({str(baseline_validator)!r})",
        )
        write(baseline_builder, builder_text.encode("ascii"), 0o500)
        module = load_module(
            baseline_builder,
            "model24_r19_fresh_baseline_builder",
        )
        previous = os.environ.get("ACE3_R19_PREPARATION_ONLY")
        os.environ["ACE3_R19_PREPARATION_ONLY"] = "1"
        try:
            result = module.prepare(nonce)
        finally:
            if previous is None:
                os.environ.pop("ACE3_R19_PREPARATION_ONLY", None)
            else:
                os.environ["ACE3_R19_PREPARATION_ONLY"] = previous
        require(
            result["status"] == "SEALED_AWAITING_INDEPENDENT_L2_REVIEW",
            "fresh r19 baseline did not seal",
        )
    return (
        Path(f"/home/argustest/ace3-model24-r19-prep-20260829-{nonce}"),
        module,
    )


def copy_tree_exact(source: Path, target: Path, r16: Any) -> dict[str, Any]:
    before = r16.tree_records(source)
    require(not target.exists(), f"r18 ancestry target exists: {target}")
    shutil.copytree(source, target)
    after = r16.tree_records(source)
    copied = r16.tree_records(target)
    require(before == after, f"r18 ancestry changed while copying: {source}")
    require(
        [
            {key: value for key, value in record.items() if key != "mode"}
            for record in before
        ]
        == [
            {key: value for key, value in record.items() if key != "mode"}
            for record in copied
        ],
        f"r18 ancestry copy differs: {source}",
    )
    r16.readonly_tree(target)
    records = r16.tree_records(target)
    return {"tree_sha256": r16.tree_sha256(records), "records": records}


def copy_r18_crash_ancestry(
    package: Path,
    r16: Any,
    validator: Any,
) -> dict[str, Any]:
    root = package / "provenance/r18-crashed"
    root.mkdir(parents=True)
    sources = {
        "package": R18_BASE / "package",
        "review": R18_REVIEW,
        "output": R18_OUTPUT,
        "terminal": R18_TERMINAL,
    }
    for path in sources.values():
        require(path.is_dir() and not path.is_symlink(), f"r18 source missing: {path}")
    roots = {
        name: copy_tree_exact(source, root / name, r16)
        for name, source in sources.items()
    }
    authority_root = root / "authority"
    authority_root.mkdir()
    consumed = Path(f"{R18_AUTHORITY}.consumed")
    require(
        digest(R18_AUTHORITY) == R18_AUTHORITY_SHA256
        and digest(consumed) == R18_AUTHORITY_SHA256,
        "r18 authority hash differs",
    )
    shutil.copy2(R18_AUTHORITY, authority_root / "original.json")
    shutil.copy2(consumed, authority_root / "consumed.json")
    r16.readonly_tree(authority_root)
    authority_records = r16.tree_records(authority_root)
    roots["authority"] = {
        "tree_sha256": r16.tree_sha256(authority_records),
        "records": authority_records,
    }
    controller_root = root / "controller"
    controller_root.mkdir()
    controller = (
        R18_BASE / "source/ace3/model/controller_model24_rtl_cascade.py"
    )
    shutil.copy2(controller, controller_root / controller.name)
    r16.readonly_tree(controller_root)
    controller_records = r16.tree_records(controller_root)
    roots["controller"] = {
        "tree_sha256": r16.tree_sha256(controller_records),
        "records": controller_records,
    }

    receipt = (
        root / "terminal/.argus_subagents" / f"{R18_TASK_ID}.json"
    )
    require(digest(receipt) == R18_RECEIPT_SHA256, "r18 crash receipt hash differs")
    receipt_payload = load_json(receipt)
    require(
        receipt_payload["state"] == "crashed"
        and "exit_code" not in receipt_payload,
        "r18 receipt is not an exit-code-free crash receipt",
    )
    sidecar = Path(receipt_payload["exit_status_path"])
    require(not sidecar.exists(), "r18 crash unexpectedly has an exit sidecar")
    require(
        not (R18_OUTPUT / "launch-terminal.json").exists()
        and not (R18_TERMINAL / "manifest.json").exists(),
        "r18 crash unexpectedly has lifecycle terminal evidence",
    )

    completion_records: dict[str, dict[str, Any]] = {}
    copied_layers = root / "output/rtl-cascade/layers"
    for index in range(9):
        record_path = copied_layers / f"layer{index:02d}/record.json"
        record = load_json(record_path)
        require(
            record["layer_index"] == index
            and record["comparison"]["within_tolerance"] is True
            and record["comparison"]["failure_count"] == 0,
            f"r18 layer {index} is not authenticated complete evidence",
        )
        completion_records[str(index)] = {
            "bytes": record_path.stat().st_size,
            "sha256": digest(record_path),
            "output_raw_sha256": record["output_raw_sha256"],
        }
    require(
        not (copied_layers / "layer09/record.json").exists(),
        "r18 layer 9 unexpectedly completed",
    )
    partial_layers = {
        "authenticated_complete_layers": list(range(9)),
        "completion_records": completion_records,
        "incomplete_layer": 9,
        "layer9_completion_record_present": False,
        "resumable": False,
        "execution_output_reusable": False,
    }
    validator.validate_partial_layers(root / "output", partial_layers)
    classification = {
        "schema_version": 1,
        "kind": "ace3_model24_r18_consumed_crash_ancestry",
        "nonce": R18_NONCE,
        "task_id": R18_TASK_ID,
        "classification": "LAWFUL_CONSUMED_INFRASTRUCTURE_CRASH",
        "authority_consumed": True,
        "receipt_sha256": R18_RECEIPT_SHA256,
        "authority_sha256": R18_AUTHORITY_SHA256,
        "exit_code_observed": False,
        "exit_code": None,
        "exit_sidecar_present": False,
        "launch_terminal_present": False,
        "nonce_reusable": False,
        "task_id_reusable": False,
        "review_reusable": False,
        "authority_reusable": False,
        "execution_output_reusable": False,
        "terminal_reusable": False,
        "predecessor_reusable": False,
        "copied_byte_for_byte": True,
        "partial_layers": partial_layers,
        "roots": roots,
    }
    write(root / "classification.json", canonical_json(classification))
    return classification


def lifecycle_overlay_payload(payload: str) -> str:
    marker = "\ndef finalize(package: Path) -> None:\n"
    require(marker in payload, "lifecycle finalizer insertion seam missing")
    helper = '''
def finalize_infrastructure_crash(
    package: Path,
    validator: Any,
    manifest: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    artifacts = validator.validate_infrastructure_crash_layout(paths, manifest)
    records = {}
    for name, path in (
        ("runner_receipt", artifacts["receipt_path"]),
        ("stdout_log", artifacts["stdout"]),
        ("stderr_log", artifacts["stderr"]),
    ):
        records[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    terminal = {
        "schema_version": 1,
        "kind": "ace3_model24_r19_terminal_evidence",
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "run_id": artifacts["receipt"]["run_id"],
        "runner_state": "crashed",
        "terminal_classification": "INFRASTRUCTURE_CRASH",
        "runner_exit_code": None,
        "exit_code_observed": False,
        "payload_invocation_count": 0,
        "records": records,
        "canonical_stdout_stderr_reconstructed": False,
    }
    write_exclusive(paths["terminal_manifest"], terminal)
    fsync_directory(paths["terminal_root"])

'''
    payload = payload.replace(marker, helper + marker, 1)
    strict = '''    artifacts = validator.validate_terminal_layout(
        paths["terminal_root"],
        manifest["identity"]["task_id"],
        manifest["execution"]["durable_command"],
        paths["child_cwd"],
    )
'''
    replacement = '''    if paths["receipt"].is_file():
        receipt = validator.load_json(paths["receipt"])
        if receipt.get("state") == "crashed":
            finalize_infrastructure_crash(package, validator, manifest, paths)
            return
    artifacts = validator.validate_terminal_layout(
        paths["terminal_root"],
        manifest["identity"]["task_id"],
        manifest["execution"]["durable_command"],
        paths["child_cwd"],
    )
'''
    require(strict in payload, "strict terminal validation seam missing")
    return payload.replace(strict, replacement, 1)


def reboot_crash_test_payload() -> bytes:
    return r'''#!/usr/bin/env python3
from pathlib import Path
import importlib.util,json,tempfile

package=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("validator",package/"validate-package.py")
validator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
life_spec=importlib.util.spec_from_file_location("lifecycle",package/"lifecycle.py")
lifecycle=importlib.util.module_from_spec(life_spec)
life_spec.loader.exec_module(lifecycle)
cases=[]

def rejected(call, text, name):
    try:
        call()
    except SystemExit as error:
        if text not in str(error):
            raise SystemExit(f"{name} rejected for wrong reason: {error}")
    else:
        raise SystemExit(f"{name} unexpectedly accepted")
    cases.append(name)

with tempfile.TemporaryDirectory(prefix="r19-reboot-crash-") as temporary:
    root=Path(temporary)
    terminal=root/"terminal"
    registry=terminal/".argus_subagents"
    logs=registry/"r19-task_logs"
    output=root/"output"
    logs.mkdir(parents=True)
    output.mkdir()
    (logs/"stdout.log").write_bytes(b"")
    (logs/"stderr.log").write_bytes(b"")
    receipt_path=registry/"r19-task.json"
    receipt={
        "state":"crashed","task_id":"r19-task","run_id":"r19-task-1",
        "command":"python lifecycle.py launch","pid":11,"worker_pid":10,
        "mode":"direct","cwd":str(root/"source"),
        "exit_status_path":str(logs/"exit_code.r19-task-1"),
        "stdout_log":".argus_subagents/r19-task_logs/stdout.log",
        "stderr_log":".argus_subagents/r19-task_logs/stderr.log",
    }
    manifest={
        "identity":{"task_id":"r19-task"},
        "execution":{"durable_command":"python lifecycle.py launch"},
    }
    paths={
        "receipt":receipt_path,"log_dir":logs,
        "stdout_log":logs/"stdout.log","stderr_log":logs/"stderr.log",
        "terminal_manifest":terminal/"manifest.json",
        "output":output,"child_cwd":root/"source",
    }
    def store(value):
        receipt_path.write_text(json.dumps(value),encoding="ascii")
    store(receipt)
    validator.validate_infrastructure_crash_layout(paths,manifest)
    cases.append("reboot-crash-positive")
    changed=dict(receipt); changed["exit_code"]=1; store(changed)
    rejected(lambda:validator.validate_infrastructure_crash_layout(paths,manifest),"fabricated","fabricated-exit-code")
    store(receipt); sidecar=logs/"exit_code.r19-task-1"; sidecar.write_text("1\n")
    rejected(lambda:validator.validate_infrastructure_crash_layout(paths,manifest),"has an exit sidecar","unexpected-exit-sidecar")
    sidecar.unlink(); changed=dict(receipt); changed["exit_status_path"]=str(logs/"exit_code.other"); store(changed)
    rejected(lambda:validator.validate_infrastructure_crash_layout(paths,manifest),"substituted","substituted-exit-sidecar")
    changed=dict(receipt); changed["state"]="error"; store(changed)
    rejected(lambda:validator.validate_infrastructure_crash_layout(paths,manifest),"state","non-crashed-receipt")
    changed=dict(receipt); changed["pid"]=0; store(changed)
    rejected(lambda:validator.validate_infrastructure_crash_layout(paths,manifest),"process identity","wrong-process-identity")
    changed=dict(receipt); changed["run_id"]="foreign"; store(changed)
    rejected(lambda:validator.validate_infrastructure_crash_layout(paths,manifest),"run identity","wrong-run-identity")
    changed=dict(receipt); changed["task_id"]="r18-task"; store(changed)
    rejected(lambda:validator.validate_infrastructure_crash_layout(paths,manifest),"task identity","mutated-receipt")

    ancestry=package/"provenance/r18-crashed"
    changed_authority=root/"authority"
    changed_authority.mkdir()
    for source,name in ((ancestry/"authority/original.json","original.json"),(ancestry/"authority/consumed.json","consumed.json")):
        (changed_authority/name).write_bytes(source.read_bytes())
    (changed_authority/"consumed.json").write_bytes(b"{}\n")
    rejected(lambda:validator.validate_authority_pair(root),"authority binding","mutated-authority")

    summary=dict(validator.load_json(ancestry/"classification.json")["partial_layers"])
    summary["authenticated_complete_layers"]=[0,1]
    rejected(lambda:validator.validate_partial_summary(summary),"layer set","partial-layer-tampering")
    summary=dict(validator.load_json(ancestry/"classification.json")["partial_layers"])
    summary["layer9_completion_record_present"]=True
    rejected(lambda:validator.validate_partial_summary(summary),"layer 9","false-layer9-completion")
    rejected(lambda:validator.validate_no_r18_reuse([validator.R18_NONCE]),"reuse rejected","r18-identity-reuse")

    seal=root/"duplicate.json"
    lifecycle.write_exclusive(seal,{"once":True})
    try:
        lifecycle.write_exclusive(seal,{"twice":True})
    except FileExistsError:
        cases.append("duplicate-terminal-sealing")
    else:
        raise SystemExit("duplicate terminal seal unexpectedly accepted")

if cases != list(validator.REBOOT_CRASH_CASES):
    raise SystemExit(f"reboot crash case order differs: {cases!r}")
print(json.dumps({"cases":cases,"status":"PASS"},sort_keys=True,separators=(",",":")))
'''.encode("ascii")


def run_postseal_compile_probe(
    base: Path,
    package: Path,
    validation_root: Path,
    validator: Any,
    r16: Any,
) -> dict[str, Any]:
    source = base / "source"
    output = base / "postseal-probe-output"
    require(not output.exists(), "post-seal probe output already exists")
    output.mkdir()
    before = r16.tree_records(source)
    controller = validator.load_rtl_controller(package)
    layers: dict[str, Any] = {}
    for index in (0, 23):
        command = controller.build_layer_compile_argv(
            source, output, index, output
        )
        started_ns = time.time_ns()
        completed = subprocess.run(
            command,
            cwd=source,
            capture_output=True,
            check=False,
        )
        write(
            validation_root / f"postseal-layer{index}.stdout",
            completed.stdout,
        )
        write(
            validation_root / f"postseal-layer{index}.stderr",
            completed.stderr,
        )
        require(completed.returncode == 0, f"post-seal layer {index} compile failed")
        binary = (
            output
            / f"compiled/layer{index}/obj_dir/Vace3_decoder_layer0_token_engine"
        )
        record = controller.authenticate_compiled_binary(binary, started_ns)
        layers[str(index)] = {
            "argv": command,
            "exit_code": completed.returncode,
            "binary": record,
        }
    after = r16.tree_records(source)
    require(before == after, "post-seal compile changed read-only source")
    result = {
        "schema_version": 1,
        "kind": "ace3_model24_r19_postseal_readonly_compile_probe",
        "status": "PASS",
        "layers": layers,
        "source_unchanged": True,
        "source_build_absent": not (source / "build").exists(),
        "rtl_binary_invocations": 0,
        "model24_invocations": 0,
        "authority_invocations": 0,
    }
    write(
        validation_root / "postseal-readonly-compile-probe.json",
        canonical_json(result),
    )
    r16.readonly_tree(output)
    return result


def finalize(base: Path, baseline: Any) -> dict[str, Any]:
    package = base / "package"
    source = base / "source"
    r16 = load_module(R16_BUILDER, "model24_r19_source_tree_helpers")
    r16.writable_tree(base)
    for root in (base / "validation", base / "preparation"):
        if root.exists():
            shutil.rmtree(root)
        root.mkdir()
    seal_path = package / "seal.json"
    seal_path.unlink()

    inherited_validator = package / "validate-package.py"
    write(
        package / "validate-r18-package.py",
        inherited_validator.read_bytes(),
        0o500,
    )
    inherited_validator.unlink()
    write(package / "validate-package.py", R19_VALIDATOR.read_bytes(), 0o500)

    lifecycle = package / "lifecycle.py"
    inherited_lifecycle_sha256 = digest(lifecycle)
    write(
        lifecycle,
        lifecycle_overlay_payload(
            lifecycle.read_text(encoding="ascii")
        ).encode("ascii"),
        0o500,
    )
    lifecycle_overlay = {
        "schema_version": 1,
        "kind": "ace3_model24_r19_reboot_crash_lifecycle_overlay",
        "inherited_r18_sha256": inherited_lifecycle_sha256,
        "materialized_sha256": digest(lifecycle),
        "normal_terminal_contract_preserved": True,
        "crashed_receipt_requires_absent_exit_sidecar": True,
        "crashed_receipt_requires_absent_launch_terminal": True,
        "exit_code_representation": "unobserved-null",
    }
    write(
        package / "lifecycle-overlay.json",
        canonical_json(lifecycle_overlay),
    )

    validator = load_module(
        package / "validate-package.py",
        "model24_r19_contract_validator",
    )
    classification = copy_r18_crash_ancestry(package, r16, validator)

    contract = load_json(package / "launch-contract.json")
    contract["review"]["required_bound_hashes"] = list(
        validator.REQUIRED_REVIEW_HASHES
    )
    contract["review"]["required_reboot_crash_cases"] = list(
        validator.REBOOT_CRASH_CASES
    )
    contract["ancestry"].update(
        {
            "r18_lawful_consumed_infrastructure_crash": True,
            "r18_authenticated_complete_layers": list(range(9)),
            "r18_incomplete_layer": 9,
            "r18_nonce_reusable": False,
            "r18_review_reusable": False,
            "r18_authority_reusable": False,
            "r18_execution_output_reusable": False,
            "predecessor_reusable": False,
        }
    )
    contract["terminal_evidence"]["infrastructure_crash"] = {
        "runner_state": "crashed",
        "exit_sidecar_required_absent": True,
        "launch_terminal_required_absent": True,
        "exit_code_observed": False,
        "exit_code": None,
        "normal_terminal_contract_preserved": True,
    }
    write(package / "launch-contract.json", canonical_json(contract))

    manifest = load_json(package / "package.json")
    manifest["provenance"]["r18_crash_ancestry"] = {
        "classification": classification["classification"],
        "receipt_sha256": R18_RECEIPT_SHA256,
        "authority_sha256": R18_AUTHORITY_SHA256,
        "authenticated_complete_layers": list(range(9)),
        "incomplete_layer": 9,
        "classification_sha256": digest(
            package / "provenance/r18-crashed/classification.json"
        ),
        "execution_output_reusable": False,
    }
    manifest["terminal_accountability"] = {
        "reboot_or_power_loss": "INFRASTRUCTURE_CRASH",
        "exit_code_observed": False,
        "exit_code": None,
        "strict_normal_terminals_preserved": True,
    }
    manifest["review_policy"]["predecessor_review_reusable"] = False
    manifest["review_policy"]["predecessor_authority_reusable"] = False
    write(package / "package.json", canonical_json(manifest))

    authority_schema = load_json(package / "authority-schema.json")
    authority_fields = authority_schema["required_exact_fields"]
    submitter_index = authority_fields.index("submitter_cwd")
    require(
        not any(
            field in authority_fields
            for field in validator.ADDITIONAL_REVIEW_HASHES
        ),
        "r19 authority schema already contains successor-only fields",
    )
    authority_fields[submitter_index:submitter_index] = list(
        validator.ADDITIONAL_REVIEW_HASHES
    )
    write(
        package / "authority-schema.json",
        canonical_json(authority_schema),
    )

    review_request = load_json(package / "review-request.json")
    review_request["required_bound_hashes"] = list(
        validator.REQUIRED_REVIEW_HASHES
    )
    review_request["required_test_results"]["reboot_crash_cases"] = list(
        validator.REBOOT_CRASH_CASES
    )
    review_request.update(
        {
            "package_manifest_sha256": digest(package / "package.json"),
            "validator_sha256": digest(package / "validate-package.py"),
            "r18_validator_sha256": digest(
                package / "validate-r18-package.py"
            ),
            "lifecycle_sha256": digest(package / "lifecycle.py"),
            "lifecycle_overlay_sha256": digest(
                package / "lifecycle-overlay.json"
            ),
            "launch_contract_sha256": digest(
                package / "launch-contract.json"
            ),
            "r18_crash_ancestry_sha256": digest(
                package / "provenance/r18-crashed/classification.json"
            ),
            "r18_receipt_sha256": R18_RECEIPT_SHA256,
            "r18_authority_sha256": R18_AUTHORITY_SHA256,
            "r18_execution_output_reusable": False,
            "canonical_execution_performed": False,
        }
    )
    write(package / "review-request.json", canonical_json(review_request))
    write(
        package / "test-reboot-crash.py",
        reboot_crash_test_payload(),
        0o500,
    )

    r16.readonly_tree(source)
    r16.readonly_tree(package)
    package.chmod(0o700)
    package_records = r16.tree_records(package)
    inherited_seal = load_json(
        package / "provenance/r18-crashed/package/seal.json"
    )
    seal = {
        "schema_version": 1,
        "kind": "ace3_model24_r19_package_seal",
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "package_manifest_sha256": digest(package / "package.json"),
        "review_request_sha256": digest(package / "review-request.json"),
        "source_tree_sha256": manifest["provenance"]["source"]["tree_sha256"],
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
        "r17_failed_ancestry_sha256": inherited_seal[
            "r17_failed_ancestry_sha256"
        ],
        "lifecycle_overlay_sha256": digest(
            package / "lifecycle-overlay.json"
        ),
        "r18_crash_ancestry_sha256": digest(
            package / "provenance/r18-crashed/classification.json"
        ),
        "v2_bound_hashes": inherited_seal["v2_bound_hashes"],
        "package_entries": package_records,
        "zero_state": True,
    }
    write(seal_path, canonical_json(seal))
    package.chmod(0o500)

    validation_root = base / "validation"
    commands = {
        "package-validation": [
            str(MODEL_PYTHON), "-B", str(package / "validate-package.py"),
            "--package", str(package), "--mode", "package",
        ],
        "negative-tests": [
            str(MODEL_PYTHON), "-B", str(package / "test-launch-contract.py"),
        ],
        "controller-entry-tests": [
            str(MODEL_PYTHON), "-B", str(package / "test-controller-entry.py"),
        ],
        "production-binding-preflight": [
            str(MODEL_PYTHON), "-B", str(package / "test-binding-preflight.py"),
        ],
        "readonly-build-output-probe": [
            str(MODEL_PYTHON), "-B",
            str(package / "test-readonly-build-output.py"),
        ],
        "reboot-crash-tests": [
            str(MODEL_PYTHON), "-B", str(package / "test-reboot-crash.py"),
        ],
    }
    environment = dict(os.environ)
    environment["ACE3_R19_PREPARATION_ONLY"] = "1"
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

    compile_probe = run_postseal_compile_probe(
        base, package, validation_root, validator, r16
    )
    paths = validator.expected_paths(package, manifest)
    forbidden = (
        "review", "authority", "authority_consumed", "output",
        "simulation_dir", "payload_output_dir", "terminal_root", "receipt",
        "stdout_log", "stderr_log", "terminal_manifest",
    )
    zero_state = {f"{name}_absent": not paths[name].exists() for name in forbidden}
    require(all(zero_state.values()), "canonical r19 execution state is not zero")
    zero_state.update(
        {
            "model24_invocations": 0,
            "controller_simulator_invocations": 0,
            "rtl_binary_invocations": 0,
            "lifecycle_launch_invocations": 0,
            "durable_submissions": 0,
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
        "kind": "ace3_model24_r19_inert_validation_seal",
        "package_seal_sha256": digest(seal_path),
        "negative_case_count": len(validator.REQUIRED_NEGATIVE_CASES),
        "positive_fixture_count": len(validator.REQUIRED_POSITIVE_FIXTURES),
        "reboot_crash_case_count": len(validator.REBOOT_CRASH_CASES),
        "outcomes": outcomes,
        "postseal_compile_probe": compile_probe,
        "records": validation_records,
        "model24_invocations": 0,
        "controller_simulator_invocations": 0,
        "rtl_binary_invocations": 0,
        "lifecycle_launch_invocations": 0,
        "durable_submissions": 0,
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
        "validation_seal_sha256": digest(validation_root / "seal.json"),
        "negative_case_count": len(validator.REQUIRED_NEGATIVE_CASES),
        "positive_fixture_count": len(validator.REQUIRED_POSITIVE_FIXTURES),
        "reboot_crash_case_count": len(validator.REBOOT_CRASH_CASES),
        "r18_lawful_consumed_infrastructure_crash": True,
        "r18_authenticated_complete_layers": list(range(9)),
        "r18_incomplete_layer": 9,
        "r18_execution_output_reusable": False,
        "postseal_readonly_compile_layers": [0, 23],
        "canonical_zero_state": zero_state,
        "execution_authority_withheld": True,
        "review_obtained": False,
    }
    preparation = base / "preparation"
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
    for path in (
        R16_BUILDER, R18_BUILDER, R18_VALIDATOR, R19_VALIDATOR,
        R18_BASE / "package", R18_REVIEW, R18_OUTPUT, R18_TERMINAL,
    ):
        require(path.exists(), f"required preparation source missing: {path}")
    require(
        digest(R18_AUTHORITY) == R18_AUTHORITY_SHA256
        and digest(Path(f"{R18_AUTHORITY}.consumed")) == R18_AUTHORITY_SHA256,
        "r18 consumed authority changed before preparation",
    )
    receipt = (
        R18_TERMINAL / ".argus_subagents" / f"{R18_TASK_ID}.json"
    )
    require(
        digest(receipt) == R18_RECEIPT_SHA256,
        "r18 crashed receipt changed before preparation",
    )
    base = Path(f"/home/argustest/ace3-model24-r19-prep-20260829-{nonce}")
    require(not base.exists(), f"fresh r19 candidate already exists: {base}")
    base, baseline = bootstrap_fresh_r19(nonce)
    return finalize(base, baseline)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", default=secrets.token_hex(8))
    args = parser.parse_args()
    print(canonical_json(prepare(args.nonce)).decode("ascii"), end="")


if __name__ == "__main__":
    main()
