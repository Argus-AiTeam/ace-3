#!/usr/bin/env python3
"""Focused tests for nonexecuting, fail-closed reviewed-parent admission."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepare_token358_position3_preflight as preflight


class EmbeddingAndPublicationTests(unittest.TestCase):
    def test_embedding_records_preserve_all_fp16_bits_and_indices(self):
        bits = [0x8000, 0x0001, 0x3c00, 0xbc00] * 224
        payload = preflight.embedding.embedding_payload(bits)
        rows = payload.decode().splitlines()
        self.assertEqual(len(rows), 896)
        self.assertEqual([int(row[2:6], 16) for row in rows], list(range(896)))
        self.assertEqual([int(row[6:10], 16) for row in rows], bits)

    def test_exclusive_publication_preserves_existing_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            preflight.write_new(path, {"accepted": True})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                preflight.write_new(path, {"accepted": False})
            self.assertEqual(path.read_bytes(), original)

    def test_embedding_drift_rejected_before_recording(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "embedding.hex"
            path.write_bytes(preflight.embedding.embedding_payload([0] * 896))
            context = {"embedding_bits": [1] * 896, "package": {}}
            with self.assertRaisesRegex(RuntimeError, "materialized embedding mismatch"):
                preflight.bind_embedding(context, path)

    def test_execution_and_consumed_authority_rejected(self):
        document = {"runtime_tail_invocations": 0,
                    "position3_execution_performed": False,
                    "authority_consumed": False, "execution_authorized": False}
        for field in document:
            candidate = copy.deepcopy(document)
            candidate[field] = 1 if field == "runtime_tail_invocations" else True
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "consumed authority or execution"):
                    preflight.validate_document(candidate, document, None)


if __name__ == "__main__":
    unittest.main()
