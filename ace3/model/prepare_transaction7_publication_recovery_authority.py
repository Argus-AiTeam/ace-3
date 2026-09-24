#!/usr/bin/env python3
"""Prepare transaction007's publication-only recovery authority sources."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path("/home/argustest/ace3-argus")
RUNTIME = (
    ROOT
    / "build/model24_selected_token_position3_runs"
    / "ace3-position3-fresh-r11-20260831t215500z"
)
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
RECOVERY = (
    ROOT
    / "build/model24_selected_token_position3_transaction7_recovery"
    / "r1-postconsume-natural-terminal-mapping"
)
RECOVERY_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction7_recovery"
    / "reviews/r1-postconsume-natural-terminal-mapping/independent-review.json"
)
TEMPLATE = (
    ROOT
    / "build/model24_selected_token_position3_transaction6_recovery_authorities"
    / "r1-manager-direct-recovery-bound"
)
AUDIT = ROOT / "build/tx007-layer06-publication-recovery/audit"
PUBLISH_SOURCE = AUDIT / "transaction7-publication-authority-source.py"
REVIEW_SOURCE = AUDIT / "transaction7-publication-authority-review-source.py"
AUTHORITY_COLLECTION = (
    ROOT
    / "build/model24_selected_token_position3_transaction7_recovery_authorities"
)
AUTHORITY_PACKAGE = AUTHORITY_COLLECTION / "r1-manager-direct-recovery-bound"
AUTHORITY_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction7_recovery_authority_reviews"
    / "r1-manager-direct-recovery-bound/independent-review.json"
)
MISSION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/5e612b01490f/mission.json"
)
REVIEWED_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/5e612b01490f/round-0004.json"
)
TEMPLATE_HASHES = {
    "publication-recovery-authority.py": (
        "75059b92f6d15483145262b432041aeea37522e41e4bc0d4a6b2b0abc009f442"
    ),
    "review-emitter.py": (
        "3f28bda4c58ba12174cca6074aee0566f68d8a0ac816617710f63c3e015977e8"
    ),
}
NEXT_ACTION = (
    "Using the recorded independent recovery PASS, prepare and seal a "
    "publication-only authority with immutable argv that can atomically publish "
    "generation8/checkpoint007/cursor8 from the reconstructed transaction007 "
    "receipt without any RTL, oracle, model, retry, replay, resume, synthesis, "
    "U280, or transaction008-025 execution; stop for independent authority review "
    "before invocation."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def quote_lines(value: str, indent: str) -> str:
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > 74:
            lines.append(current + " ")
            current = word
        else:
            current = candidate
    lines.append(current)
    return "\n".join(f'{indent}"{line}"' for line in lines)


def fixed_binding_block() -> str:
    generation7 = ADOPTION / "state-generations/generation-0000000007"
    future = RUNTIME / "transaction7-authoritative-generation8"
    paths = [
        (RECOVERY / "adjudication.json", 'RECOVERY_PACKAGE / "adjudication.json"'),
        (
            RECOVERY / "package-manifest.json",
            'RECOVERY_PACKAGE / "package-manifest.json"',
        ),
        (RECOVERY / "package-seal.json", 'RECOVERY_PACKAGE / "package-seal.json"'),
        (
            RECOVERY / "recovery-adjudication.py",
            'RECOVERY_PACKAGE / "recovery-adjudication.py"',
        ),
        (RECOVERY_REVIEW, "RECOVERY_REVIEW"),
        (ADOPTION / "authoritative-state.json", "POINTER"),
        (
            generation7 / "generation-manifest.json",
            'GENERATION7 / "generation-manifest.json"',
        ),
        (generation7 / "ledger.json", 'GENERATION7 / "ledger.json"'),
        (
            generation7 / "checkpoints/transaction-006.json",
            'GENERATION7 / "checkpoints/transaction-006.json"',
        ),
        (
            future / "manager-authorization-consumption.json",
            "ORIGINAL_CONSUMPTION",
        ),
        (future / "fail-closed-terminal.json", "ORIGINAL_FAIL_CLOSED_TERMINAL"),
        (
            RUNTIME / "transactions/transaction-007/position003/simulation.log",
            "SIMULATION_LOG",
        ),
        (MISSION, "MANAGER_MISSION"),
        (REVIEWED_HANDOFF, "REVIEWED_HANDOFF"),
    ]
    entries = "\n".join(
        f'    {expression}: "{sha256_file(path)}",' for path, expression in paths
    )
    counters = """RECOVERY_ZERO_ACTIVITY = {
    "authority_consumption": 0,
    "generation8_publication": 0,
    "model_execution": 0,
    "oracle_execution": 0,
    "rtl_compile": 0,
    "rtl_simulation": 0,
    "synthesis": 0,
    "transaction_execution": 0,
    "transaction_replay": 0,
    "transaction_resume": 0,
    "transaction_retry": 0,
    "transactions008_025_execution": 0,
    "u280": 0,
}"""
    return f"FIXED_SHA256 = {{\n{entries}\n}}\n{counters}"


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence, found {count}: {old!r}")
    return source.replace(old, new)


def replace_function(source: str, name: str, next_name: str, body: str) -> str:
    pattern = rf"def {name}\(.*?\n(?=def {next_name}\()"
    transformed, count = re.subn(
        pattern,
        body.rstrip() + "\n\n",
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"function replacement failed: {name}")
    return transformed


def accepted_recovery_validation() -> str:
    return '''def validate_accepted_recovery() -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        RECOVERY_PACKAGE.is_dir()
        and not RECOVERY_PACKAGE.is_symlink()
        and stat.S_IMODE(RECOVERY_PACKAGE.stat().st_mode) & 0o222 == 0,
        "accepted r1 recovery package is absent or writable",
    )
    for path, expected in FIXED_SHA256.items():
        require(sha256_file(path) == expected, f"fixed input hash differs: {path}")
    require(
        stat.S_IMODE(RECOVERY_REVIEW.stat().st_mode) & 0o222 == 0,
        "accepted r1 recovery review is writable",
    )
    seal = load_json(RECOVERY_PACKAGE / "package-seal.json")
    require(
        seal.get("kind")
        == "ace3_transaction7_publication_recovery_adjudication_seal"
        and seal.get("status") == "SEALED_REVIEW_REQUIRED"
        and seal.get("activity_counters") == RECOVERY_ZERO_ACTIVITY
        and seal.get("generation8_publication_authorized") is False
        and seal.get("generation8_publication_performed") is False
        and set(seal.get("members", {})) == RECOVERY_MEMBERS,
        "accepted r1 recovery seal differs",
    )
    for name, record in seal["members"].items():
        require(
            record.get("path") == str(RECOVERY_PACKAGE / name),
            f"accepted r1 member path differs: {name}",
        )
        authenticate(record, f"accepted r1 member {name}")
    review = load_json(RECOVERY_REVIEW)
    require(
        review.get("kind")
        == "ace3_transaction7_publication_recovery_independent_review"
        and review.get("status") == "PASS"
        and review.get("producer_role") == "reviewer"
        and review.get("preparation_participation") is False
        and review.get("activity_counters") == RECOVERY_ZERO_ACTIVITY
        and review.get("natural_rtl_terminal") is True
        and review.get("final_output_count") == 896
        and review.get("integer_oracle_mismatches") == 0
        and review.get("transaction007_retry_replay_resume") is False
        and review.get("generations3_7_preserved") is True
        and review.get("transactions000_006_preserved") is True
        and review.get("transactions008_025_absent") is True
        and review.get("generation8_publication_authorized") is False
        and review.get("generation8_publication_performed") is False
        and review.get("separately_reviewed_publication_authority_required") is True
        and review.get("synthesis_or_u280") is False
        and review.get("package_seal")
        == file_record(RECOVERY_PACKAGE / "package-seal.json")
        and review.get("adjudication")
        == file_record(RECOVERY_PACKAGE / "adjudication.json"),
        "accepted r1 recovery review differs",
    )
    adjudication = load_json(RECOVERY_PACKAGE / "adjudication.json")
    boundary = adjudication.get("recovery_boundary", {})
    require(
        adjudication.get("kind")
        == "ace3_transaction7_postconsume_publication_recovery_adjudication"
        and adjudication.get("status") == "PASS"
        and adjudication.get("transaction_index") == 7
        and adjudication.get("layer_index") == 6
        and boundary.get("activity_counters") == RECOVERY_ZERO_ACTIVITY
        and boundary.get("authority_consumed_before_recovery") is True
        and boundary.get("authority_reusable") is False
        and boundary.get("new_authority_created") is False
        and boundary.get("generation8_publication_authorized") is False
        and boundary.get("generation8_publication_performed") is False
        and boundary.get("separately_reviewed_publication_authority_required")
        is True
        and boundary.get("transactions008_025_absent") is True,
        "accepted r1 adjudication differs",
    )
    finding = adjudication.get("finding", {})
    receipt = adjudication.get("reconstructed_receipt", {})
    require(
        finding.get("natural_terminal") == 1
        and finding.get("rtl_exit_code") == 0
        and finding.get("done_count") == 1
        and finding.get("final_count") == 896
        and finding.get("integer_mismatches") == 0
        and finding.get("exact_oracle_agreement") is True
        and finding.get("generation8_publication_attempted") is False
        and receipt.get("status") == "COMPLETE"
        and receipt.get("transaction_index") == 7
        and receipt.get("result", {}).get("natural_rtl_terminal") is True
        and receipt.get("result", {}).get("exact_integer_oracle_match") is True
        and receipt.get("result", {}).get("output_hidden_elements") == 896
        and receipt.get("rtl_reference_agreement", {}).get("mismatches") == 0,
        "accepted r1 claim-bearing evidence differs",
    )
    parent = adjudication["authoritative_parent"]
    require(
        parent.get("generation") == 7
        and parent.get("cursor") == 7
        and parent.get("pointer") == file_record(POINTER)
        and parent.get("ledger") == file_record(GENERATION7 / "ledger.json")
        and parent.get("checkpoint006")
        == file_record(GENERATION7 / "checkpoints/transaction-006.json"),
        "accepted r1 generation7/cursor7 parent differs",
    )
    for label, record in frozen_evidence(adjudication).items():
        if isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}:
            authenticate(record, f"frozen transaction007 {label}")
    return adjudication, review
'''


def recovery_review_validation() -> str:
    return '''def validate_recovery_review(document: Mapping[str, Any]) -> None:
    require(
        document.get("kind")
        == "ace3_transaction7_publication_recovery_independent_review"
        and document.get("status") == "PASS"
        and document.get("producer_role") == "reviewer"
        and document.get("preparation_participation") is False
        and document.get("activity_counters") == RECOVERY_ZERO_ACTIVITY
        and document.get("natural_rtl_terminal") is True
        and document.get("final_output_count") == 896
        and document.get("integer_oracle_mismatches") == 0
        and document.get("transaction007_retry_replay_resume") is False
        and document.get("generations3_7_preserved") is True
        and document.get("transactions000_006_preserved") is True
        and document.get("transactions008_025_absent") is True
        and document.get("generation8_publication_authorized") is False
        and document.get("generation8_publication_performed") is False
        and document.get("separately_reviewed_publication_authority_required") is True
        and document.get("synthesis_or_u280") is False
        and document.get("package_seal")
        == file_record(RECOVERY_PACKAGE / "package-seal.json")
        and document.get("adjudication")
        == file_record(RECOVERY_PACKAGE / "adjudication.json"),
        "accepted r1 recovery review differs",
    )
'''


def transform(source: str, *, reviewer: bool) -> str:
    source = source.replace(
        "build/tx006-layer05-execution/audit/"
        "transaction6-publication-authority-review-source.py",
        "build/tx007-layer06-publication-recovery/audit/"
        "transaction7-publication-authority-review-source.py",
    )
    for old, temporary in [
        ("GENERATION6", "__PARENT_GENERATION_SYMBOL__"),
        ("generation-0000000006", "__PARENT_GENERATION_PATH__"),
        ("generation6", "__PARENT_GENERATION_WORD__"),
        ("checkpoint005", "__PARENT_CHECKPOINT_WORD__"),
        ("cursor6", "__PARENT_CURSOR_WORD__"),
        ("transaction-005", "__PARENT_TRANSACTION_PATH__"),
        ('"generation": 6', '"generation": __PARENT_GENERATION_NUMBER__'),
        ("for index in range(6)", "for index in range(__PARENT_CHECKPOINT_COUNT__)"),
    ]:
        source = source.replace(old, temporary)
    source = source.replace("GENERATION7", "GENERATION8")
    source = source.replace("generation-0000000007", "generation-0000000008")
    source = source.replace("generation7", "generation8")
    source = source.replace("checkpoint006", "checkpoint007")
    source = source.replace("cursor7", "cursor8")
    source = source.replace("transaction-006", "transaction-007")
    for temporary, new in [
        ("__PARENT_GENERATION_SYMBOL__", "GENERATION7"),
        ("__PARENT_GENERATION_PATH__", "generation-0000000007"),
        ("__PARENT_GENERATION_WORD__", "generation7"),
        ("__PARENT_CHECKPOINT_WORD__", "checkpoint006"),
        ("__PARENT_CURSOR_WORD__", "cursor7"),
        ("__PARENT_TRANSACTION_PATH__", "transaction-006"),
    ]:
        source = source.replace(temporary, new)
    for old, new in [
        ("TRANSACTION006", "TRANSACTION007"),
        ("transaction006", "transaction007"),
        ("Transaction006", "Transaction007"),
        ("transaction6", "transaction7"),
        ("tx006-layer05-execute-once", "tx007-layer06-execute-once"),
        ("layer05", "layer06"),
        ("layer5", "layer6"),
        ("f0576bd98422", "5e612b01490f"),
        ("round-0002.json", "round-0004.json"),
        ("r2-correct-generation7-checkpoint006-parent", RECOVERY.name),
        ("accepted_r2", "accepted_r1"),
        ("Accepted r2", "Accepted r1"),
        ("accepted r2", "accepted r1"),
        ("transactions007_025", "transactions008_025"),
        ("transaction007-025", "transaction008-025"),
        ("range(7, 26)", "range(8, 26)"),
        ("generation=7 cursor=7", "generation=8 cursor=8"),
        ('"transaction_index": 6', '"transaction_index": 7'),
        ('transaction_index") == 6', 'transaction_index") == 7'),
        ('"layer_index": 5', '"layer_index": 6'),
        ('layer_index") == 5', 'layer_index") == 6'),
        ('"parent_generation": 6', '"parent_generation": 7'),
        ('pointer.get("generation") == 6', 'pointer.get("generation") == 7'),
        ('parent.get("generation") == 6', 'parent.get("generation") == 7'),
        ('parent.get("cursor") == 6', 'parent.get("cursor") == 7'),
        ('"cursor": 6', '"cursor": 7'),
        (
            'parent.get("latest_checkpoint_index") == 5',
            'parent.get("latest_checkpoint_index") == 6',
        ),
        ('"latest_checkpoint_index": 5', '"latest_checkpoint_index": 6'),
        ('"generation": 7', '"generation": 8'),
        ('"completed_transaction_count": 7', '"completed_transaction_count": 8'),
        ('"next_transaction_index": 7', '"next_transaction_index": 8'),
        ('"state_generation": 7', '"state_generation": 8'),
        ('completed_transaction_count") == 6', 'completed_transaction_count") == 7'),
        ('next_transaction_index") == 6', 'next_transaction_index") == 7'),
        ('state_generation") == 6', 'state_generation") == 7'),
        ('len(ledger.get("completed_receipts", [])) == 6', 'len(ledger.get("completed_receipts", [])) == 7'),
        ("for index in range(7)", "for index in range(8)"),
        ("ledger5", "ledger7_parent"),
        ("ledger6", "ledger8"),
        ("pointer6", "pointer8"),
    ]:
        source = source.replace(old, new)
    source = source.replace(
        '"generation": __PARENT_GENERATION_NUMBER__', '"generation": 7'
    )
    source = source.replace(
        "for index in range(__PARENT_CHECKPOINT_COUNT__)",
        "for index in range(7)",
    )
    source = re.sub(
        r'and review\.get\("next_action"\)\n\s*== \(\n(?:\s*".*"\n)+\s*\),',
        'and review.get("next_action")\n        == (\n'
        + quote_lines(NEXT_ACTION, "            ")
        + "\n        ),",
        source,
        count=1,
    )
    source, count = re.subn(
        r"FIXED_SHA256 = \{.*?\n\}\nRECOVERY_ZERO_ACTIVITY = \{.*?\n\}",
        fixed_binding_block(),
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("fixed binding replacement failed")
    source = replace_once(
        source,
        '    "RTL simulation",\n',
        '    "RTL simulation",\n    "synthesis",\n    "U280 execution",\n',
    )
    if reviewer:
        source = replace_function(
            source,
            "validate_recovery_review",
            "validate_authority_document",
            recovery_review_validation(),
        )
        source = replace_once(
            source,
            'lambda item: item["authoritative_parent"].update({"cursor": 5}),',
            'lambda item: item["authoritative_parent"].update({"cursor": 6}),',
        )
        source = replace_once(
            source,
            "later_transaction_indices=[6],",
            "later_transaction_indices=[7],",
        )
        source = replace_once(
            source,
            '        "model_oracle_rtl_execution": False,\n',
            '        "model_oracle_rtl_execution": False,\n'
            '        "synthesis_or_u280": False,\n',
        )
    else:
        source = replace_once(
            source,
            '                "compile or simulate RTL",\n',
            '                "compile or simulate RTL",\n'
            '                "run synthesis or U280",\n',
        )
        source = replace_function(
            source,
            "validate_accepted_recovery",
            "authorized_publish_argv",
            accepted_recovery_validation(),
        )
        source = replace_once(
            source,
            '        and review.get("model_oracle_rtl_execution") is False\n',
            '        and review.get("model_oracle_rtl_execution") is False\n'
            '        and review.get("synthesis_or_u280") is False\n',
        )
    return source


def write_new(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(payload)


def main() -> None:
    if AUTHORITY_PACKAGE.exists() or AUTHORITY_REVIEW.exists():
        raise RuntimeError("transaction007 publication authority namespace is not fresh")
    for name, expected in TEMPLATE_HASHES.items():
        path = TEMPLATE / name
        if sha256_file(path) != expected:
            raise RuntimeError(f"authority template differs: {path}")
    publisher = transform(
        (TEMPLATE / "publication-recovery-authority.py").read_text(encoding="utf-8"),
        reviewer=False,
    )
    reviewer = transform(
        (TEMPLATE / "review-emitter.py").read_text(encoding="utf-8"),
        reviewer=True,
    )
    write_new(PUBLISH_SOURCE, publisher)
    write_new(REVIEW_SOURCE, reviewer)
    print(
        "TRANSACTION007_PUBLICATION_RECOVERY_AUTHORITY_SOURCES "
        f"publisher={PUBLISH_SOURCE} reviewer={REVIEW_SOURCE}"
    )


if __name__ == "__main__":
    main()
