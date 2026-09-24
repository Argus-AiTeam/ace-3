"""Isolated Q24 expectations: Python binary16 codec versus rational oracle."""

from fractions import Fraction
import json
from pathlib import Path
import random
import struct
import sys
import unittest

from ace3.model.candidates import residual_exact_grid_q24_reference_v1 as oracle
from ace3.model.candidates import residual_exact_grid_q24_runtime_v1 as codec


GRID = 1 << 24
LIMIT = 65520 * GRID
SEED = 0xACE324


def lift(word):
    value = struct.unpack("<e", word.to_bytes(2, "little"))[0]
    return int(Fraction.from_float(value) * GRID)


def nearest(integer, tag=0):
    if integer == 0:
        return tag << 15
    # In the finite-view range every integer here is exactly representable in
    # binary64. struct's binary16 RNE is independent of the rational search.
    return int.from_bytes(struct.pack("<e", integer / GRID), "little")


def rounding_states():
    values = {0, 1, 1023, 1024, 1025, 65504 * GRID, LIMIT - 1}
    for exponent in range(1, 31):
        for fraction in (0, 1, 511, 512, 1022, 1023):
            word = (exponent << 10) | fraction
            if word >= 0x7BFF:
                continue
            a, b = lift(word), lift(word + 1)
            if (a + b) % 2 == 0:
                midpoint = (a + b) // 2
                values.update((midpoint - 1, midpoint, midpoint + 1))
    rng = random.Random(SEED)
    values.update(rng.randrange(LIMIT) for _ in range(128))
    return sorted(values | {-value for value in values})


def vector_rows():
    for word in range(65536):
        if word & 0x7C00 != 0x7C00:
            tag = int(word == 0x8000)
            yield 0, tag, word, lift(word), tag, word, 0
    for integer in rounding_states():
        yield integer, 0, 0, integer, 0, nearest(integer), 0
    for tag in (0, 1):
        for word in (0, 0x8000):
            result_tag = int(tag == 1 and word == 0x8000)
            yield 0, tag, word, 0, result_tag, result_tag << 15, 0
    for word in (1, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0x8001, 0xBC00, 0xFBFF):
        yield -lift(word), 0, word, 0, 0, 0, 0
    for integer, tag, word, code in (
        (0, 0, 0x7C00, 1), (0, 0, 0xFC00, 1),
        (0, 0, 0x7E01, 1), (1, 1, 0xFC01, 1),
        (1, 1, 0, 2), (-1, 1, 0, 2),
        ((1 << 63) - 1, 1, 1, 2),
        ((1 << 63) - 1, 0, 1, 3), (-(1 << 63), 0, 0x8001, 3),
        ((1 << 63) - 1, 0, 0, 4), (-(1 << 63), 0, 0, 4),
        (LIMIT - 1, 0, 1, 4), (-LIMIT + 1, 0, 0x8001, 4),
        (LIMIT, 0, 0, 4), (-LIMIT, 0, 0, 4),
        (LIMIT + 1, 0, 0, 4), (-LIMIT - 1, 0, 0, 4),
    ):
        yield integer, tag, word, 0, 0, 0, code


def metadata():
    return dict(codec.CONSTANTS, slot=0, position=0, next_layer=0,
                model_id="isolated-fixture-model", token_history_id="fixture-history",
                root_id="fixture-root", producer_id="fixture-producer",
                kv_lineage_id="separate-fp16-kv-fixture")


class Q24ReferenceTests(unittest.TestCase):
    def test_u01_all_finite_lifts_and_roundtrips(self):
        count = 0
        for word in range(65536):
            if word & 0x7C00 == 0x7C00:
                with self.assertRaises(oracle.PrimitiveFault) as caught:
                    oracle.root(word)
                self.assertEqual(caught.exception.code, 1)
                continue
            state = (lift(word), int(word == 0x8000))
            self.assertEqual(oracle.root(word), state, hex(word))
            self.assertEqual(oracle.project(*state), word, hex(word))
            count += 1
        self.assertEqual(count, 63488)

    def test_u02_persistent_remainder_not_rounded_view(self):
        state = oracle.root(0x3C00)
        for step in range(1, 9):
            integer, tag, view = oracle.add(*state, 0x1000)
            self.assertEqual(integer, GRID + step * 8192)
            self.assertEqual(tag, 0)
            self.assertEqual(view, nearest(GRID + step * 8192))
            state = integer, tag
        self.assertEqual(oracle.project(GRID, 0), oracle.project(GRID + 8192, 0))
        self.assertNotEqual(oracle.add(GRID, 0, 0x1000),
                            oracle.add(GRID + 8192, 0, 0x1000))

    def test_u03_u04_u05_general_vectors_and_distinct_faults(self):
        for integer, tag, word, expected, expected_tag, view, code in vector_rows():
            if code:
                with self.assertRaises(oracle.PrimitiveFault) as caught:
                    oracle.add(integer, tag, word)
                self.assertEqual(caught.exception.code, code, (integer, tag, word))
            elif integer != 0 or word in (0, 0x8000):
                self.assertEqual(oracle.add(integer, tag, word),
                                 (expected, expected_tag, view), (integer, tag, word))

    def test_u05_invalid_python_encodings(self):
        for word in (-1, 65536, True, 1.0, None):
            with self.assertRaises(oracle.PrimitiveFault) as caught:
                oracle.root(word)
            self.assertEqual(caught.exception.code, 1)
        for integer in (-(1 << 63) - 1, 1 << 63, True, 0.0):
            with self.assertRaises(oracle.PrimitiveFault) as caught:
                oracle.project(integer, 0)
            self.assertEqual(caught.exception.code, 1)
        for integer, tag in ((0, -1), (0, 2), (0, True), (1, 1)):
            with self.assertRaises(oracle.PrimitiveFault) as caught:
                oracle.project(integer, tag)
            self.assertEqual(caught.exception.code, 2)


class Q24CodecTests(unittest.TestCase):
    def setUp(self):
        self.meta = metadata()
        self.records = tuple(((index - 448) * 8193, 0) for index in range(896))
        self.records = ((0, 1), (LIMIT - 1, 0), (-LIMIT + 1, 0)) + self.records[3:]
        self.text = codec.encode(self.records, self.meta)

    def test_u11_exact_portable_bytes_and_restore(self):
        document = json.loads(self.text)
        payload = bytes.fromhex(document["payload_hex"])
        independent = b"".join(integer.to_bytes(8, "little", signed=True) + bytes([tag])
                               for integer, tag in self.records)
        self.assertEqual(len(payload), 8064)
        self.assertEqual(len(document["payload_hex"]), 16128)
        self.assertEqual(payload, independent)
        restored = codec.decode(self.text, trusted_metadata=self.meta)
        self.assertEqual(restored, self.records)
        self.assertEqual(codec.encode(restored, self.meta), self.text)
        for (integer, tag), recovered in zip(self.records[3:], restored[3:]):
            expected = integer + lift(0x1000)
            self.assertEqual(oracle.add(*recovered, 0x1000),
                             (expected, 0, nearest(expected)))

    def test_u11_metadata_rejection(self):
        for key, value in (
            ("public_abi", "wrong"), ("arithmetic_id", "wrong"),
            ("state_id", "wrong"), ("policy_id", "wrong"), ("json_encoding", "wrong"),
            ("root_id", "wrong"), ("producer_id", "wrong"),
            ("model_id", "wrong"), ("token_history_id", "wrong"),
            ("kv_lineage_id", "wrong"), ("completed", False), ("idle", False),
            ("slot", 4), ("position", -1), ("next_layer", 25),
            ("width", 32), ("fraction_bits", 23), ("hidden_size", 895),
            ("slot", True), ("root_id", ""), ("idle", 1),
        ):
            document = json.loads(self.text)
            document[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                codec.decode(json.dumps(document), trusted_metadata=self.meta)
        for key in self.meta:
            document = json.loads(self.text)
            del document[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                codec.decode(json.dumps(document), trusted_metadata=self.meta)
        for change in ({"extra": 1}, {"idle": False}, {"slot": -1},
                       {"producer_id": ""}, {"fraction_bits": 25}):
            with self.assertRaises(ValueError):
                codec.encode(self.records, dict(self.meta, **change))

    def test_u11_payload_and_json_rejection(self):
        original = json.loads(self.text)
        payload = original["payload_hex"]
        bad_payloads = (payload[:-2], payload + "00", payload.upper(),
                        "gg" + payload[2:], " " + payload[1:], None,
                        (1).to_bytes(8, "little").hex() + "01" + payload[18:],
                        "00" * 8 + "02" + payload[18:],
                        LIMIT.to_bytes(8, "little", signed=True).hex() + "00" + payload[18:],
                        (-(1 << 63)).to_bytes(8, "little", signed=True).hex() + "00" + payload[18:])
        for invalid in bad_payloads:
            with self.assertRaises(ValueError):
                codec.decode(json.dumps(dict(original, payload_hex=invalid)),
                             trusted_metadata=self.meta)
        for text in ("[]", "null", "{}", self.text + "{}",
                     '{"idle":true,' + self.text[1:],
                     self.text.replace('"slot":0', '"slot":NaN'),
                     self.text.replace('"slot":0', '"slot":Infinity')):
            with self.assertRaises(ValueError):
                codec.decode(text, trusted_metadata=self.meta)
        for records in (self.records[:-1], self.records + ((0, 0),),
                        ((1, 1),) + self.records[1:],
                        ((LIMIT, 0),) + self.records[1:],
                        ((1 << 63, 0),) + self.records[1:]):
            with self.assertRaises(ValueError):
                codec.encode(records, self.meta)


def write_vectors(path):
    rows = list(vector_rows())
    with Path(path).open("x") as output:
        output.write(f"{len(rows)}\n")
        for integer, tag, word, expected, expected_tag, view, code in rows:
            output.write(f"{integer & ((1 << 64) - 1):016x} {tag:x} {word:04x} "
                         f"{expected & ((1 << 64) - 1):016x} {expected_tag:x} "
                         f"{view:04x} {code:x}\n")
    print(f"Q24_VECTORS rows={len(rows)} finite_lifts=63488 seed={SEED}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--vectors":
        write_vectors(sys.argv[2])
    else:
        unittest.main()
