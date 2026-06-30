import unittest

from localbooster.metrics import GenerationMetrics


class MetricsTest(unittest.TestCase):
    def test_metrics_compute_cost_and_acceptance_ratio(self):
        metrics = GenerationMetrics(
            backend="fake",
            sampler="power",
            model="fake-model",
            latency_seconds=1.2,
            generated_tokens=4,
            sampled_tokens=12,
            accepted_proposals=2,
            attempted_proposals=5,
        )

        self.assertEqual(metrics.cost_multiplier, 3)
        self.assertEqual(metrics.acceptance_ratio, 0.4)
        self.assertEqual(metrics.to_dict()["cost_multiplier"], 3)


if __name__ == "__main__":
    unittest.main()
