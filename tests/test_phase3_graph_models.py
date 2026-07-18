import tempfile
import unittest
from pathlib import Path

import torch

from gnn_pipeline.evaluation.recommendation.graph_models import (
    TemporalGraphRanker,
    TextDualEncoder,
)
from gnn_pipeline.evaluation.recommendation.temporal_graph import (
    RelationConfig,
    build_temporal_graph_snapshot,
    fit_training_topic_model,
    load_temporal_snapshot,
    load_temporal_topic_model,
    relation_ablation_configs,
    save_temporal_snapshot,
    save_temporal_topic_model,
)
from tests.recommendation_fixture import build_fixture, fixture_embedding_table


class Phase3TemporalGraphTests(unittest.TestCase):
    def _context(self, root):
        benchmark = build_fixture(root)
        table = fixture_embedding_table(benchmark)
        topics = fit_training_topic_model(
            benchmark, table, training_cutoff=2022, n_clusters=2, random_seed=3
        )
        return benchmark, table, topics

    def test_snapshot_has_no_future_nodes_and_relation_ablation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark, table, topics = self._context(root)
            snapshot = build_temporal_graph_snapshot(
                benchmark=benchmark,
                raw_csv_path=str(root / "papers.csv"),
                embedding_table=table,
                topic_model=topics,
                cutoff_year=2021,
                relation_config=relation_ablation_configs()["authorship_only"],
            )
            self.assertEqual(snapshot.paper_ids, ["P0", "P1"])
            self.assertEqual(snapshot.data.temporal_cutoff_year, 2021)
            self.assertIn(("author", "authored", "paper"), snapshot.data.edge_types)
            self.assertNotIn(("paper", "cites", "paper"), snapshot.data.edge_types)
            self.assertNotIn(("paper", "has_topic", "topic"), snapshot.data.edge_types)
            graph_root = root / "graphs"
            save_temporal_snapshot(snapshot, str(graph_root))
            snapshot_dir = next(graph_root.glob("2021/*"))
            loaded = load_temporal_snapshot(str(snapshot_dir))
            self.assertEqual(loaded.snapshot_id, snapshot.snapshot_id)
            self.assertEqual(loaded.embedding_cache_key, table.manifest["cache_key"])

            topic_root = root / "topics"
            save_temporal_topic_model(topics, str(topic_root))
            topic_dir = next((topic_root / "topic_models").iterdir())
            loaded_topics = load_temporal_topic_model(str(topic_dir))
            self.assertEqual(loaded_topics.model_id, topics.model_id)

    def test_all_graph_architectures_produce_candidate_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark, table, topics = self._context(root)
            snapshot = build_temporal_graph_snapshot(
                benchmark=benchmark,
                raw_csv_path=str(root / "papers.csv"),
                embedding_table=table,
                topic_model=topics,
                cutoff_year=2022,
                relation_config=RelationConfig(),
            )
            for architecture in ("citation_sage", "hetero_sage", "hgt"):
                model = TemporalGraphRanker(
                    architecture=architecture,
                    metadata=snapshot.data.metadata(),
                    text_dim=table.dimension,
                    hidden_dim=4,
                    num_layers=1,
                    dropout=0.0,
                    hgt_heads=2,
                )
                with torch.inference_mode():
                    candidates = model.encode_candidates(snapshot.data)
                    scores = model.full_scores(
                        torch.from_numpy(table.lookup(["P3"])).float(), candidates
                    )
                self.assertEqual(
                    tuple(candidates.shape),
                    (len(snapshot.paper_ids), table.dimension),
                )
                self.assertEqual(tuple(scores.shape), (1, len(snapshot.paper_ids)))
                self.assertTrue(torch.isfinite(scores).all())
                expected = torch.nn.functional.normalize(snapshot.data["paper"].x, dim=-1)
                self.assertTrue(torch.allclose(candidates, expected, atol=1e-6))

            text_model = TextDualEncoder(
                table.dimension, hidden_dim=7, dropout=0.0, model_version=2
            ).eval()
            values = torch.from_numpy(table.lookup(["P0", "P1"])).float()
            expected = torch.nn.functional.normalize(values, dim=-1)
            self.assertTrue(
                torch.allclose(text_model.encode_queries(values), expected, atol=1e-6)
            )
            self.assertTrue(
                torch.allclose(text_model.encode_candidates(values), expected, atol=1e-6)
            )


if __name__ == "__main__":
    unittest.main()
