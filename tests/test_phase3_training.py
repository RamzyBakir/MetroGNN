import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gnn_pipeline.evaluation.recommendation.baselines import TfidfRecommender
from gnn_pipeline.evaluation.recommendation.feature_models import PairwiseFeatureRecommender
from gnn_pipeline.evaluation.recommendation.fusion import ValidationFusionEvaluator
from gnn_pipeline.evaluation.recommendation.features import FEATURE_NAMES, TemporalFeatureStore
from gnn_pipeline.evaluation.recommendation.graph_models import TemporalGraphRanker
from gnn_pipeline.evaluation.recommendation.graph_training import (
    TemporalGraphTrainer,
    load_temporal_graph_ranker,
)
from gnn_pipeline.evaluation.recommendation.runner import run_recommenders
from gnn_pipeline.evaluation.recommendation.sampling import build_pairwise_samples
from gnn_pipeline.evaluation.recommendation.temporal_graph import (
    RelationConfig,
    build_temporal_graph_snapshot,
    fit_training_topic_model,
)
from gnn_pipeline.evaluation.recommendation.text_training import (
    TextDualEncoderTrainer,
    load_text_dual_encoder,
)
from tests.recommendation_fixture import build_fixture, fixture_embedding_table


class Phase3TrainingTests(unittest.TestCase):
    def _setup(self, root):
        benchmark = build_fixture(root)
        table = fixture_embedding_table(benchmark)
        features = TemporalFeatureStore(
            benchmark,
            raw_csv_path=str(root / "papers.csv"),
            embedding_table=table,
            include_tfidf=True,
            include_bm25=True,
        )
        samples = build_pairwise_samples(
            benchmark,
            feature_store=features,
            random_negatives=1,
            hard_negatives=1,
            seed=9,
        )
        return benchmark, table, features, samples

    def test_pair_features_sampling_and_feature_rankers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark, table, features, samples = self._setup(root)
            self.assertFalse(samples.triples.empty)
            self.assertFalse(
                (samples.triples["positive_id"] == samples.triples["negative_id"]).any()
            )
            query = benchmark.split_queries("validation").iloc[0]
            candidates = benchmark.candidates_for_query(query)
            matrix = features.candidate_features(
                query,
                candidates,
                visible_citation_edges=benchmark.visible_citation_edges(query),
            )
            self.assertEqual(matrix.shape, (len(candidates), len(FEATURE_NAMES)))
            self.assertTrue(np.isfinite(matrix).all())

            for architecture in ("linear", "mlp"):
                model = PairwiseFeatureRecommender(
                    features,
                    samples,
                    architecture=architecture,
                    hidden_dim=4,
                    epochs=2,
                    seed=3,
                ).fit(benchmark.training_data())
                ranking = model.rank(
                    query,
                    candidates,
                    3,
                    3,
                    visible_citation_edges=benchmark.visible_citation_edges(query),
                )
                self.assertEqual(len(ranking), 3)
                checkpoint = root / f"feature_{architecture}.pt"
                model.save(str(checkpoint))
                loaded = PairwiseFeatureRecommender.load(
                    str(checkpoint), feature_store=features, samples=samples
                )
                loaded.fit(benchmark.training_data())
                loaded_ranking = loaded.rank(
                    query,
                    candidates,
                    3,
                    3,
                    visible_citation_edges=benchmark.visible_citation_edges(query),
                )
                self.assertEqual(loaded_ranking, ranking)

    def test_text_and_graph_trainers_use_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark, table, features, samples = self._setup(root)
            fusion = ValidationFusionEvaluator(benchmark)
            tfidf_run = run_recommenders(
                benchmark,
                [TfidfRecommender()],
                splits=["validation"],
                seeds=[2],
                top_k=20,
                metric_config={"ndcg": [10], "recall": [20]},
            )
            recovered = fusion.evaluate(
                torch.zeros_like(torch.from_numpy(fusion.lexical_scores)), [0.0]
            )
            baseline = tfidf_run.aggregate_metrics.iloc[0]
            self.assertAlmostEqual(recovered["ndcg@10"], baseline["ndcg@10"])
            self.assertAlmostEqual(recovered["recall@20"], baseline["recall@20"])

            text_result = TextDualEncoderTrainer(
                benchmark=benchmark,
                embedding_table=table,
                samples=samples,
                fusion_evaluator=fusion,
            ).train(hidden_dim=4, max_epochs=2, patience=1, seed=2)
            self.assertGreaterEqual(text_result.best_epoch, 0)
            self.assertGreaterEqual(
                text_result.best_validation["ndcg@10"], baseline["ndcg@10"]
            )
            text_checkpoint = root / "text_model.pt"
            text_result.save(
                str(text_checkpoint),
                embedding_cache_key=table.manifest["cache_key"],
                sample_id=samples.manifest["sample_id"],
            )
            loaded_text, _ = load_text_dual_encoder(str(text_checkpoint))
            self.assertEqual(loaded_text.query_projection.out_features, 4)
            self.assertEqual(
                loaded_text.fusion_weight,
                text_result.best_validation["fusion_weight"],
            )

            topics = fit_training_topic_model(
                benchmark, table, training_cutoff=2022, n_clusters=2, random_seed=2
            )
            snapshot_2021 = build_temporal_graph_snapshot(
                benchmark=benchmark,
                raw_csv_path=str(root / "papers.csv"),
                embedding_table=table,
                topic_model=topics,
                cutoff_year=2021,
                relation_config=RelationConfig(),
            )
            validation = build_temporal_graph_snapshot(
                benchmark=benchmark,
                raw_csv_path=str(root / "papers.csv"),
                embedding_table=table,
                topic_model=topics,
                cutoff_year=2022,
                relation_config=RelationConfig(),
            )
            trainer = TemporalGraphTrainer(
                benchmark=benchmark,
                embedding_table=table,
                snapshots={2021: snapshot_2021},
                pairwise_samples=samples,
                validation_snapshot=validation,
                fusion_evaluator=fusion,
            )
            model = TemporalGraphRanker(
                architecture="citation_sage",
                metadata=validation.data.metadata(),
                text_dim=table.dimension,
                hidden_dim=4,
                num_layers=1,
                dropout=0.0,
            )
            result = trainer.train(model, max_epochs=2, patience=1, seed=2)
            self.assertGreaterEqual(result.best_epoch, 0)
            self.assertGreaterEqual(
                result.best_validation["ndcg@10"], baseline["ndcg@10"]
            )
            graph_checkpoint = root / "graph_model.pt"
            result.save(
                str(graph_checkpoint),
                snapshot_ids={2021: snapshot_2021.snapshot_id, 2022: validation.snapshot_id},
                sample_id=samples.manifest["sample_id"],
            )
            loaded_graph, _ = load_temporal_graph_ranker(
                str(graph_checkpoint), snapshot=validation
            )
            self.assertEqual(loaded_graph.get_config()["architecture"], "citation_sage")
            self.assertEqual(
                loaded_graph.fusion_weight,
                result.best_validation["fusion_weight"],
            )


if __name__ == "__main__":
    unittest.main()
