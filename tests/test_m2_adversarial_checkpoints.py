import hashlib
import json
from pathlib import Path
import unittest
import pandas as pd
import torch

from gnn_pipeline.evaluation.recommendation.graph_models import TemporalGraphRanker
from gnn_pipeline.evaluation.recommendation.graph_training import load_temporal_graph_ranker
from gnn_pipeline.evaluation.recommendation.temporal_graph import (
    load_temporal_snapshot,
    load_temporal_graph_snapshot,
)


class Phase3AdversarialCheckpointVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.frozen_path = (
            cls.project_root
            / "saved_results"
            / "phase3_validation_v2"
            / "minilm"
            / "hetero_sage"
            / "full"
            / "frozen_selection.json"
        )
        cls.checkpoints_dir = (
            cls.project_root
            / "saved_results"
            / "phase3_validation_v2"
            / "minilm"
            / "hetero_sage"
            / "full"
            / "checkpoints"
        )
        cls.temporal_graphs_dir = (
            cls.project_root / "saved_results" / "temporal_graphs" / "minilm"
        )
        cls.validation_summary_path = (
            cls.project_root
            / "saved_results"
            / "phase3_validation_v2"
            / "minilm"
            / "hetero_sage"
            / "full"
            / "validation_summary.csv"
        )
        cls.validation_trials_path = (
            cls.project_root
            / "saved_results"
            / "phase3_validation_v2"
            / "minilm"
            / "hetero_sage"
            / "full"
            / "validation_trials.csv"
        )

    def test_frozen_selection_metadata(self):
        self.assertTrue(
            self.frozen_path.exists(),
            f"frozen_selection.json not found at {self.frozen_path}",
        )
        with open(self.frozen_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(
            data.get("selection_split"),
            "validation",
            f"selection_split must be 'validation', got {data.get('selection_split')}",
        )
        self.assertEqual(
            data.get("primary_metric"),
            "ndcg@10",
            f"primary_metric must be 'ndcg@10', got {data.get('primary_metric')}",
        )
        self.assertEqual(
            data.get("tie_breaker"),
            "recall@20",
            f"tie_breaker must be 'recall@20', got {data.get('tie_breaker')}",
        )
        self.assertEqual(
            len(data.get("repeated_seeds", [])),
            5,
            "Expected 5 repeated seeds",
        )
        self.assertEqual(
            len(data.get("winner_artifacts", [])),
            5,
            "Expected 5 winner artifacts",
        )

    def test_winner_checkpoints_sha256_and_loading(self):
        with open(self.frozen_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load the 2022 validation snapshot
        snapshot = load_temporal_graph_snapshot(
            str(self.temporal_graphs_dir), 2022
        )
        self.assertEqual(snapshot.cutoff_year, 2022)

        for artifact in data["winner_artifacts"]:
            rel_path = artifact["artifact_path"]
            if rel_path.startswith("/app/"):
                rel_path = rel_path[len("/app/") :]
            pt_path = self.project_root / rel_path
            self.assertTrue(
                pt_path.exists(), f"Checkpoint file does not exist: {pt_path}"
            )

            # Check sha256
            with open(pt_path, "rb") as f_bin:
                actual_hash = hashlib.sha256(f_bin.read()).hexdigest()
            self.assertEqual(
                actual_hash,
                artifact["artifact_sha256"],
                f"SHA256 mismatch for {pt_path}",
            )

            # Check model.json
            meta_path = pt_path.parent / "model.json"
            self.assertTrue(
                meta_path.exists(), f"Metadata JSON missing: {meta_path}"
            )
            with open(meta_path, "r", encoding="utf-8") as f_meta:
                meta = json.load(f_meta)
            self.assertEqual(meta.get("checkpoint_sha256"), actual_hash)
            self.assertEqual(meta.get("checkpoint_version"), 2)

            model_config = meta.get("model_config", {})
            self.assertEqual(
                model_config.get("architecture"),
                "hetero_sage",
                f"Expected hetero_sage architecture, got {model_config.get('architecture')}",
            )
            self.assertEqual(
                model_config.get("model_version"),
                2,
                f"Expected model_version 2, got {model_config.get('model_version')}",
            )

            # Load model with load_temporal_graph_ranker
            model, payload = load_temporal_graph_ranker(
                str(pt_path), snapshot=snapshot, device="cpu"
            )
            self.assertIsInstance(model, TemporalGraphRanker)
            self.assertEqual(model.architecture, "hetero_sage")
            self.assertEqual(model.model_version, 2)
            self.assertFalse(model.training)

            # Test forward evaluation
            with torch.no_grad():
                cand_embs = model.encode_candidates(snapshot.data)
                self.assertEqual(cand_embs.shape[0], len(snapshot.paper_ids))
                self.assertEqual(cand_embs.shape[1], model_config["text_dim"])
                self.assertTrue(torch.isfinite(cand_embs).all())

                # Test query encoding
                dummy_queries = torch.randn(4, model_config["text_dim"])
                query_embs = model.encode_queries(dummy_queries)
                self.assertEqual(query_embs.shape, (4, model_config["text_dim"]))
                self.assertTrue(torch.isfinite(query_embs).all())

                # Test scoring
                scores = model.full_scores(dummy_queries, cand_embs)
                self.assertEqual(scores.shape, (4, len(snapshot.paper_ids)))
                self.assertTrue(torch.isfinite(scores).all())

    def test_all_checkpoints_in_directory_are_valid(self):
        ckpt_dirs = [p for p in self.checkpoints_dir.iterdir() if p.is_dir()]
        self.assertEqual(
            len(ckpt_dirs),
            20,
            f"Expected 20 checkpoint directories (4 configs x 5 seeds), found {len(ckpt_dirs)}",
        )
        for ckpt_dir in ckpt_dirs:
            pt_path = ckpt_dir / "model.pt"
            meta_path = ckpt_dir / "model.json"
            history_path = ckpt_dir / "history.csv"

            self.assertTrue(pt_path.exists(), f"Missing model.pt in {ckpt_dir}")
            self.assertTrue(meta_path.exists(), f"Missing model.json in {ckpt_dir}")
            self.assertTrue(history_path.exists(), f"Missing history.csv in {ckpt_dir}")

            with open(pt_path, "rb") as f_bin:
                actual_hash = hashlib.sha256(f_bin.read()).hexdigest()
            with open(meta_path, "r", encoding="utf-8") as f_meta:
                meta = json.load(f_meta)
            self.assertEqual(
                meta.get("checkpoint_sha256"),
                actual_hash,
                f"Hash mismatch in {ckpt_dir}",
            )
            self.assertEqual(
                meta.get("model_config", {}).get("architecture"), "hetero_sage"
            )
            self.assertEqual(
                meta.get("model_config", {}).get("model_version"), 2
            )

    def test_temporal_snapshots_omits_2023_cutoff(self):
        self.assertTrue((self.temporal_graphs_dir / "2020").exists())
        self.assertTrue((self.temporal_graphs_dir / "2021").exists())
        self.assertTrue((self.temporal_graphs_dir / "2022").exists())
        self.assertFalse(
            (self.temporal_graphs_dir / "2023").exists(),
            "2023 cutoff must not exist in M2 Phase 3 Model Tuning",
        )

    def test_validation_summary_and_trials_consistency(self):
        self.assertTrue(self.validation_summary_path.exists())
        self.assertTrue(self.validation_trials_path.exists())

        summary_df = pd.read_csv(self.validation_summary_path)
        trials_df = pd.read_csv(self.validation_trials_path)

        self.assertEqual(len(summary_df), 4)  # 4 configs
        self.assertEqual(len(trials_df), 20)  # 20 trials

        with open(self.frozen_path, "r", encoding="utf-8") as f:
            frozen = json.load(f)
        winner_cfg = frozen["winner_config"]

        # Check winner config matches highest ranked row in summary_df
        top_row = summary_df.iloc[0]
        self.assertEqual(int(top_row["hidden_dim"]), winner_cfg["hidden_dim"])
        self.assertEqual(int(top_row["num_layers"]), winner_cfg["num_layers"])
        self.assertEqual(float(top_row["dropout"]), winner_cfg["dropout"])
        self.assertAlmostEqual(float(top_row["learning_rate"]), winner_cfg["learning_rate"])


if __name__ == "__main__":
    unittest.main()
