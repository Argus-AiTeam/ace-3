import unittest

from trace_lineage_a_layer06_stage16_activation import (
    activation_chain, components, integer_norm, norm_producer_chains, projection_chain, true_silu,
)


class Stage16ActivationTests(unittest.TestCase):
    def test_integer_norm_signs_subnormals_and_global_denominator(self):
        hidden = [0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x4000, 0x7bff]
        result = integer_norm(hidden, [0x3c00] * len(hidden))
        self.assertEqual(result[:2], [0, 0x8000])
        self.assertEqual(result[4] ^ 0x8000, result[5])
        self.assertNotEqual(integer_norm([0x3c00, 0x4000], [0x3c00] * 2)[0],
                            integer_norm([0x3c00, 0x4400], [0x3c00] * 2)[0])
        with self.assertRaises(ValueError):
            integer_norm([0x7c00], [0x3c00])

    def test_true_silu_signed_zero_and_signs(self):
        result, disagreements = true_silu([0, 0x8000, 0x3c00, 0xbc00],
                                         [0x3c00, 0x3c00, 0xbc00, 0xbc00])
        self.assertEqual(result, [0, 0x8000, 0xb9d9, 0x344e])
        self.assertEqual(disagreements, [])

    def test_both_orders_preserve_interaction_and_close(self):
        edges = projection_chain([("norm_arithmetic", "A", "R")])
        chains = {name: activation_chain(edges, first)
                  for name, first in (("gate_first", True), ("up_first", False))}
        projections = {"A": 0x4000, "P:A": 0x3e00, "P:R": 0x3c00, "R": 0x3800}
        cases = {"A": [0x4400], "own": [0x4200], "R": [0x3400]}
        for g, gb in projections.items():
            for u, ub in projections.items():
                cases[f"G:{g}/U:{u}"] = true_silu([gb], [ub])[0]
        parts = components(cases, chains, 0)
        self.assertEqual(sum(parts["gate_first"].values()), sum(parts["up_first"].values()))
        self.assertNotEqual(parts["gate_first"]["gate/norm_arithmetic"],
                            parts["up_first"]["gate/norm_arithmetic"])
        for chain in chains.values():
            self.assertEqual(chain[0][1], "A")
            self.assertEqual(chain[-1][2], "R")
            self.assertTrue(all(a[2] == b[1] for a, b in zip(chain, chain[1:])))

    def test_norm_endpoints_use_shared_component_contract(self):
        cases = {key: [0x3c00 + i] for i, key in enumerate(
            ("A", "own", "n:A", "n:AA", "n:AR", "n:RA", "n:RR", "n:R", "R"))}
        parts = components(cases, norm_producer_chains(), 0)
        self.assertEqual(sum(parts["incoming_first"].values()),
                         sum(parts["O_first"].values()))


if __name__ == "__main__":
    unittest.main()
