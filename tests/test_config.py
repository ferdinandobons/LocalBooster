import unittest
from dataclasses import replace

from localbooster.config import GenerationConfig, get_preset


class ConfigTest(unittest.TestCase):
    def test_preset_returns_power_config(self):
        config = get_preset("fast", max_new_tokens=32, backend="mlx")

        self.assertEqual(config.sampler, "power")
        self.assertEqual(config.backend, "mlx")
        self.assertEqual(config.max_new_tokens, 32)
        self.assertEqual(config.proposal_temperature, 0.5)

    def test_proposal_temperature_rejects_invalid_alpha(self):
        config = replace(GenerationConfig(), alpha=0)

        with self.assertRaisesRegex(ValueError, "alpha"):
            _ = config.proposal_temperature


if __name__ == "__main__":
    unittest.main()
