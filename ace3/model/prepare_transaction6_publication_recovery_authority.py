#!/usr/bin/env python3
"""Prepare transaction006's publication-only recovery authority sources."""

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
    / "build/model24_selected_token_position3_transaction6_recovery"
    / "r2-correct-generation6-checkpoint005-parent"
)
RECOVERY_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction6_recovery"
    / "reviews/r2-correct-generation6-checkpoint005-parent/independent-review.json"
)
TEMPLATE = (
    ROOT
    / "build/model24_selected_token_position3_transaction5_recovery_authorities"
    / "r1-manager-direct-recovery-bound"
)
AUDIT = ROOT / "build/tx006-layer05-execution/audit"
REVIEW_SOURCE = AUDIT / "transaction6-publication-authority-review-source.py"
AUTHORITY_COLLECTION = (
    ROOT
    / "build/model24_selected_token_position3_transaction6_recovery_authorities"
)
AUTHORITY_PACKAGE = AUTHORITY_COLLECTION / "r1-manager-direct-recovery-bound"
AUTHORITY_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction6_recovery_authority_reviews"
    / "r1-manager-direct-recovery-bound/independent-review.json"
)
MISSION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/f0576bd98422/mission.json"
)
REVIEWED_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/f0576bd98422/round-0002.json"
)
TEMPLATE_HASHES = {
    "publication-recovery-authority.py": (
        "71cae1fa426ca0eb7bb0e589a6da0fba78273d6d70634392c180f6fcc4469bbe"
    ),
    "review-emitter.py": (
        "fe912b0d80166535d6f4839362ce2e292c08a563ccd407165bac5dc18995f674"
    ),
}
NEXT_ACTION = (
    "Complete independent review of the publication-only authority, then consume "
    "it once and invoke only its sealed argv to atomically publish "
    "generation7/checkpoint006/cursor7 from accepted r2 evidence without "
    "replaying transaction006; preserve generations3-6 and transactions000-005 "
    "and keep transactions007-025 absent."
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


def fixed_hash_block() -> str:
    generation6 = ADOPTION / "state-generations/generation-0000000006"
    future = RUNTIME / "transaction6-authoritative-generation7"
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
            generation6 / "generation-manifest.json",
            'GENERATION6 / "generation-manifest.json"',
        ),
        (generation6 / "ledger.json", 'GENERATION6 / "ledger.json"'),
        (
            generation6 / "checkpoints/transaction-005.json",
            'GENERATION6 / "checkpoints/transaction-005.json"',
        ),
        (
            future / "manager-authorization-consumption.json",
            "ORIGINAL_CONSUMPTION",
        ),
        (future / "fail-closed-terminal.json", "ORIGINAL_FAIL_CLOSED_TERMINAL"),
        (
            RUNTIME / "transactions/transaction-006/position003/simulation.log",
            "SIMULATION_LOG",
        ),
        (MISSION, "MANAGER_MISSION"),
        (REVIEWED_HANDOFF, "REVIEWED_HANDOFF"),
    ]
    entries = "\n".join(
        f'    {expression}: "{sha256_file(path)}",' for path, expression in paths
    )
    return f"FIXED_SHA256 = {{\n{entries}\n}}\nRECOVERY_ZERO_ACTIVITY ="


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence, found {count}: {old!r}")
    return source.replace(old, new)


def transform(source: str, *, reviewer: bool) -> str:
    source = source.replace(
        "ace3/model/emit_transaction5_publication_recovery_authority_review.py",
        "build/tx006-layer05-execution/audit/"
        "transaction6-publication-authority-review-source.py",
    )
    for old, temporary in [
        ("GENERATION5", "__PARENT_GENERATION_SYMBOL__"),
        ("generation-0000000005", "__PARENT_GENERATION_PATH__"),
        ("generation5", "__PARENT_GENERATION_WORD__"),
        ("checkpoint004", "__PARENT_CHECKPOINT_WORD__"),
        ("cursor5", "__PARENT_CURSOR_WORD__"),
        ("transaction-004", "__PARENT_TRANSACTION_PATH__"),
        ('"generation": 5', '"generation": __PARENT_GENERATION_NUMBER__'),
        ("for index in range(5)", "for index in range(__PARENT_CHECKPOINT_COUNT__)"),
    ]:
        source = source.replace(old, temporary)
    source = source.replace("GENERATION6", "GENERATION7")
    source = source.replace("generation-0000000006", "generation-0000000007")
    source = source.replace("generation6", "generation7")
    source = source.replace("checkpoint005", "checkpoint006")
    source = source.replace("cursor6", "cursor7")
    source = source.replace("transaction-005", "transaction-006")
    for temporary, new in [
        ("__PARENT_GENERATION_SYMBOL__", "GENERATION6"),
        ("__PARENT_GENERATION_PATH__", "generation-0000000006"),
        ("__PARENT_GENERATION_WORD__", "generation6"),
        ("__PARENT_CHECKPOINT_WORD__", "checkpoint005"),
        ("__PARENT_CURSOR_WORD__", "cursor6"),
        ("__PARENT_TRANSACTION_PATH__", "transaction-005"),
    ]:
        source = source.replace(temporary, new)
    for old, new in [
        ("TRANSACTION005", "TRANSACTION006"),
        ("transaction005", "transaction006"),
        ("Transaction005", "Transaction006"),
        ("transaction5", "transaction6"),
        ("tx005-layer05-execute-once", "tx006-layer05-execute-once"),
        ("layer04", "layer05"),
        ("layer4", "layer5"),
        ("b66531b35910", "f0576bd98422"),
        ("round-0004.json", "round-0002.json"),
        ("r1-postconsume-none-to-false-sourcebound", RECOVERY.name),
        ("accepted_r1", "accepted_r2"),
        ("Accepted r1", "Accepted r2"),
        ("accepted r1", "accepted r2"),
        ("transactions006_025", "transactions007_025"),
        ("transaction006-025", "transaction007-025"),
        ("range(6, 26)", "range(7, 26)"),
        ("generation=6 cursor=6", "generation=7 cursor=7"),
        ('"transaction_index": 5', '"transaction_index": 6'),
        ('transaction_index") == 5', 'transaction_index") == 6'),
        ('"layer_index": 4', '"layer_index": 5'),
        ('layer_index") == 4', 'layer_index") == 5'),
        ('"parent_generation": 5', '"parent_generation": 6'),
        ('pointer.get("generation") == 5', 'pointer.get("generation") == 6'),
        ('parent.get("generation") == 5', 'parent.get("generation") == 6'),
        ('parent.get("cursor") == 5', 'parent.get("cursor") == 6'),
        ('"cursor": 5', '"cursor": 6'),
        (
            'parent.get("latest_checkpoint_index") == 4',
            'parent.get("latest_checkpoint_index") == 5',
        ),
        ('"latest_checkpoint_index": 4', '"latest_checkpoint_index": 5'),
        ('"generation": 6', '"generation": 7'),
        ('"completed_transaction_count": 6', '"completed_transaction_count": 7'),
        ('"next_transaction_index": 6', '"next_transaction_index": 7'),
        ('"state_generation": 6', '"state_generation": 7'),
        ('completed_transaction_count") == 5', 'completed_transaction_count") == 6'),
        ('next_transaction_index") == 5', 'next_transaction_index") == 6'),
        ('state_generation") == 5', 'state_generation") == 6'),
        ('len(ledger.get("completed_receipts", [])) == 5', 'len(ledger.get("completed_receipts", [])) == 6'),
        ("for index in range(6)", "for index in range(7)"),
    ]:
        source = source.replace(old, new)
    source = source.replace(
        '"generation": __PARENT_GENERATION_NUMBER__', '"generation": 6'
    )
    source = source.replace(
        "for index in range(__PARENT_CHECKPOINT_COUNT__)",
        "for index in range(6)",
    )
    source = re.sub(
        r'and review\.get\("next_action"\)\n\s*== \(\n(?:\s*".*"\n)+\s*\),',
        'and review.get("next_action")\n        == (\n'
        + quote_lines(NEXT_ACTION, "            ")
        + "\n        ),",
        source,
        count=1,
    )
    source = re.sub(
        r"FIXED_SHA256 = \{.*?\n\}\nRECOVERY_ZERO_ACTIVITY =",
        fixed_hash_block(),
        source,
        count=1,
        flags=re.DOTALL,
    )
    if reviewer:
        source = replace_once(
            source,
            'lambda item: item["authoritative_parent"].update({"cursor": 6}),',
            'lambda item: item["authoritative_parent"].update({"cursor": 5}),',
        )
    return source


def write_new(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(payload)


def main() -> None:
    if AUTHORITY_PACKAGE.exists() or AUTHORITY_REVIEW.exists():
        raise RuntimeError("transaction006 publication authority namespace is not fresh")
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
    write_new(AUDIT / "transaction6-publication-authority-source.py", publisher)
    write_new(REVIEW_SOURCE, reviewer)
    print(
        "TRANSACTION006_PUBLICATION_RECOVERY_AUTHORITY_SOURCES "
        f"publisher={AUDIT / 'transaction6-publication-authority-source.py'} "
        f"reviewer={REVIEW_SOURCE}"
    )


if __name__ == "__main__":
    main()
