import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from localbooster.cli import main


class CliTest(unittest.TestCase):
    def test_report_preserves_requested_sampler_alias(self):
        record = {
            "sampler": "power-fast",
            "result": {
                "metrics": {
                    "backend": "mlx",
                    "model": "fake",
                    "sampler": "power",
                    "latency_seconds": 1.0,
                    "cost_multiplier": 2.0,
                    "acceptance_ratio": 0.5,
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "results.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(["report", str(path)])

        self.assertEqual(exit_code, 0)
        self.assertIn("power-fast", output.getvalue())

    def test_report_includes_accuracy_when_scores_are_present(self):
        records = [
            {
                "sampler": "standard",
                "score": {"correct": True},
                "result": {
                    "metrics": {
                        "backend": "mlx",
                        "model": "fake",
                        "sampler": "standard",
                        "latency_seconds": 1.0,
                        "cost_multiplier": 1.0,
                        "acceptance_ratio": None,
                    }
                },
            },
            {
                "sampler": "standard",
                "score": {"correct": False},
                "result": {
                    "metrics": {
                        "backend": "mlx",
                        "model": "fake",
                        "sampler": "standard",
                        "latency_seconds": 3.0,
                        "cost_multiplier": 1.0,
                        "acceptance_ratio": None,
                    }
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "results.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(["report", str(path)])

        self.assertEqual(exit_code, 0)
        self.assertIn("Accuracy", output.getvalue())
        self.assertIn("0.50", output.getvalue())


if __name__ == "__main__":
    unittest.main()
