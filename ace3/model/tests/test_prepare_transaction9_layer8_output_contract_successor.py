from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ace3/model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

import prepare_transaction9_layer8_output_contract_successor as successor  # noqa: E402


class Transaction9Layer8OutputContractSuccessorTest(unittest.TestCase):
    def test_parent_outputs_separate_file_identity_from_semantics(self) -> None:
        contract = successor.generation9_parent_receipt_contract()

        self.assertEqual(
            {
                name: set(record)
                for name, record in contract["outputs"].items()
            },
            {
                "hidden": {"path", "bytes", "sha256"},
                "state": {"path", "bytes", "sha256"},
            },
        )
        self.assertEqual(
            contract["output_semantics"]["hidden"]["dtype"],
            "FP16",
        )
        self.assertEqual(
            contract["output_semantics"]["hidden"]["elements"],
            896,
        )
        self.assertEqual(
            contract["output_semantics"]["state"],
            {"layer_index": 7, "position": 4},
        )

    def test_sealed_contract_forbids_fresh_r14_authority(self) -> None:
        conflicts = successor.sealed_authority_conflicts()

        self.assertEqual(
            [conflict["reason_code"] for conflict in conflicts],
            [
                "R14_PACKAGE_NON_EXECUTABLE",
                "R14_REVIEW_WITHHOLDS_AUTHORITY",
                "R12_AUTHORITY_CONSUMED_NO_RETRY",
            ],
        )
        self.assertFalse(successor.AUTHORITY.exists())
        self.assertFalse(successor.EXECUTION_BINDING.exists())
        self.assertFalse(successor.READINESS.exists())


if __name__ == "__main__":
    unittest.main()
