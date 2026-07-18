import json
import tempfile
import unittest
from pathlib import Path

from gnn_pipeline.artifacts import sha256_file, write_json_atomic
from gnn_pipeline.evaluation.recommendation.final_test import (
    FinalTestAlreadyRun,
    FinalTestRegistry,
)
from gnn_pipeline.evaluation.recommendation.tuning import TrialOutcome, ValidationOnlyTuner


class Phase3GovernanceTests(unittest.TestCase):
    def test_validation_tuning_freezes_five_seed_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def callback(config, seed):
                path = root / f"model_{config['width']}_{seed}.bin"
                path.write_text(f"{config['width']}:{seed}")
                quality = float(config["width"]) / 10.0
                return TrialOutcome(
                    metrics={"ndcg@10": quality, "recall@20": quality / 2},
                    artifact_path=str(path.resolve()),
                    artifact_sha256=sha256_file(str(path)),
                )

            tuner = ValidationOnlyTuner(
                experiment_name="phase3:test",
                benchmark_id="benchmark",
                max_coarse_trials=2,
            )
            selection = tuner.tune(
                search_space={"width": [1, 2]},
                callback=callback,
                seeds=[13, 42, 77, 101, 202],
                top_configurations=1,
                output_dir=str(root / "tuning"),
            )
            self.assertEqual(selection["winner_config"], {"width": 2})
            self.assertEqual(len(selection["winner_artifacts"]), 5)
            self.assertEqual(selection["selection_split"], "validation")

    def test_test_configuration_is_rejected_during_tuning(self):
        tuner = ValidationOnlyTuner(experiment_name="x", benchmark_id="b")
        with self.assertRaisesRegex(ValueError, "Test-related"):
            tuner._run(
                lambda config, seed: None,
                {"test_threshold": 1},
                1,
                "coarse",
            )

    def test_final_registry_runs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = []
            for seed in [13, 42, 77, 101, 202]:
                path = root / f"checkpoint_{seed}.pt"
                path.write_text(str(seed))
                artifacts.append(
                    {
                        "seed": seed,
                        "artifact_path": str(path.resolve()),
                        "artifact_sha256": sha256_file(str(path)),
                    }
                )
            selection = {
                "experiment_name": "phase3:model",
                "benchmark_id": "benchmark",
                "selection_split": "validation",
                "repeated_seeds": [13, 42, 77, 101, 202],
                "winner_artifacts": artifacts,
                "winner_config": {"hidden": 64},
                "selection_id": "selection",
            }
            selection_path = root / "selection.json"
            write_json_atomic(str(selection_path), selection)
            baseline_path = root / "baseline.json"
            write_json_atomic(str(baseline_path), {"run_id": "baseline"})

            registry = FinalTestRegistry(str(root / "registry"))
            plan = registry.create_plan(
                frozen_selection_path=str(selection_path),
                frozen_baseline_manifest_path=str(baseline_path),
                final_context={"architecture": "model"},
            )

            def callback(plan, output_dir):
                result = Path(output_dir) / "predictions.csv"
                result.write_text("query,candidate\n")
                return {"predictions": str(result)}

            completion = registry.run_once(plan, callback)
            self.assertEqual(completion["status"], "complete")
            with self.assertRaises(FinalTestAlreadyRun):
                registry.run_once(plan, callback)


if __name__ == "__main__":
    unittest.main()
