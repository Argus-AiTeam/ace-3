"""Separate primitive and authenticated decoder codecs for exact-grid state.

Decoder serialization retains the 8064-byte <qB payload and JSON encoding.
Only explicit decoder entry points accept the decoder state/policy/public-ABI
tuple. Decoding checks the separately trusted producer digest, full owner
metadata and every rounded H view; a root additionally has no remainder.
This portable payload is not a generated simulator save or a KV snapshot.
The caller must authenticate the root embedding and the producing execution;
encoding an internally consistent tuple is not provenance or admission.
"""

import hashlib
import json
import re
import struct
from collections.abc import Mapping, Sequence


CONSTANTS = {
    "state_id": "ace3-residual-exact-grid-q24-state-v1",
    "json_encoding": "ace3-residual-exact-grid-q24-json-v1",
    "arithmetic_id": "ace3-residual-exact-grid-q24-v1",
    "policy_id": "ace3-residual-exact-grid-q24-local-global-policy-v1",
    "public_abi": "ace3-residual-exact-grid-q24-public-abi-v1",
    "width": 64,
    "fraction_bits": 24,
    "hidden_size": 896,
    "completed": True,
    "idle": True,
}
IDENTITIES = ("model_id", "token_history_id", "root_id", "producer_id", "kv_lineage_id")
RANGES = {"slot": (0, 3), "position": (0, 32767), "next_layer": (0, 24)}
RECORD = struct.Struct("<qB")
DECODER_CONSTANTS = dict(
    CONSTANTS,
    state_id="ace3-decoder-residual-exact-grid-q24-state-v1",
    policy_id="ace3-decoder-residual-exact-grid-q24-local-global-policy-v1",
    public_abi="ace3-decoder-q24-public-abi-v1",
)


def _metadata(metadata: Mapping[str, object], *, decoder: bool = False) -> dict[str, object]:
    constants = DECODER_CONSTANTS if decoder else CONSTANTS
    if set(metadata) != set(constants) | set(IDENTITIES) | set(RANGES):
        raise ValueError("missing or unsupported Q24 portable-state metadata fields")
    for key, expected in constants.items():
        if type(metadata[key]) is not type(expected) or metadata[key] != expected:
            raise ValueError(f"unsupported portable-state field: {key}")
    for key, (minimum, maximum) in RANGES.items():
        value = metadata[key]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"invalid portable-state owner: {key}")
    for key in IDENTITIES:
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f"missing portable-state identity: {key}")
    return dict(metadata)


def _record(integer: int, tag: int) -> None:
    if type(integer) is not int or not -(1 << 63) <= integer < (1 << 63):
        raise ValueError("portable residual is not a signed 64-bit integer")
    if type(tag) is not int or tag not in (0, 1) or (integer != 0 and tag != 0):
        raise ValueError("noncanonical portable residual zero tag")
    if abs(integer) >= 65520 * (1 << 24):
        raise ValueError("portable residual does not have a finite FP16 view")


def encode(records: Sequence[tuple[int, int]], metadata: Mapping[str, object]) -> str:
    document = _metadata(metadata)
    if len(records) != 896:
        raise ValueError("portable Q24 state requires exactly 896 ordered coordinates")
    payload = bytearray()
    for integer, tag in records:
        _record(integer, tag)
        payload.extend(RECORD.pack(integer, tag))
    document["payload_hex"] = payload.hex()
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate portable-state field: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def decode(text: str, *, trusted_metadata: Mapping[str, object]) -> tuple[tuple[int, int], ...]:
    expected = _metadata(trusted_metadata)
    document = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(document, dict) or "payload_hex" not in document:
        raise ValueError("portable state must be a JSON object with payload_hex")
    encoded = document.pop("payload_hex")
    actual = _metadata(document)
    if actual != expected:
        raise ValueError("portable state does not match separately trusted owner/producer context")
    if not isinstance(encoded, str) or re.fullmatch(r"[0-9a-f]{16128}", encoded) is None:
        raise ValueError("portable payload must contain exactly 8064 canonical hex-encoded bytes")
    records = tuple(RECORD.iter_unpack(bytes.fromhex(encoded)))
    for integer, tag in records:
        _record(integer, tag)
    return records


def fp16_view(integer: int, tag: int) -> int:
    _record(integer, tag)
    if integer == 0:
        return tag << 15
    # Every admitted Q24 integer is below 2**40, hence exact in binary64.
    return int.from_bytes(struct.pack("<e", integer / (1 << 24)), "little")


def _paired_hidden(records: Sequence[tuple[int, int]], hidden: Sequence[int]) -> None:
    if len(records) != 896 or len(hidden) != 896:
        raise ValueError("decoder state requires exactly 896 paired I/Z/H coordinates")
    for index, ((integer, tag), word) in enumerate(zip(records, hidden, strict=True)):
        if type(word) is not int or not 0 <= word <= 65535:
            raise ValueError(f"invalid decoder FP16 word at coordinate {index}")
        if fp16_view(integer, tag) != word:
            raise ValueError(f"decoder H/state disagreement at coordinate {index}")


def encode_decoder(records: Sequence[tuple[int, int]], metadata: Mapping[str, object],
                   *, hidden: Sequence[int]) -> str:
    document = _metadata(metadata, decoder=True)
    _paired_hidden(records, hidden)
    document["payload_hex"] = b"".join(RECORD.pack(i, z) for i, z in records).hex()
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode_decoder(text: str, *, trusted_metadata: Mapping[str, object],
                   trusted_sha256: str, hidden: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """The digest and owner must come from the admitted producer, not this JSON."""
    expected = _metadata(trusted_metadata, decoder=True)
    if not isinstance(trusted_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", trusted_sha256) is None:
        raise ValueError("missing separately authenticated decoder artifact digest")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != trusted_sha256:
        raise ValueError("decoder state artifact digest mismatch")
    document = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(document, dict) or "payload_hex" not in document:
        raise ValueError("missing decoder paired-state payload; FP16-only seeds are forbidden")
    payload = document.pop("payload_hex")
    if _metadata(document, decoder=True) != expected:
        raise ValueError("decoder state does not match its admitted producer/owner")
    if not isinstance(payload, str) or re.fullmatch(r"[0-9a-f]{16128}", payload) is None:
        raise ValueError("decoder payload must contain exactly 8064 canonical bytes")
    records = tuple(RECORD.iter_unpack(bytes.fromhex(payload)))
    _paired_hidden(records, hidden)
    if expected["next_layer"] == 0:
        for index, ((integer, tag), word) in enumerate(zip(records, hidden, strict=True)):
            value = struct.unpack("<e", word.to_bytes(2, "little"))[0]
            if integer != int(value * (1 << 24)) or tag != int(word == 0x8000):
                raise ValueError(f"invented model-root remainder at coordinate {index}")
    return records
