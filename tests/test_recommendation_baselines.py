import tempfile
import unittest
from pathlib import Path

from gnn_pipeline.evaluation.recommendation.baselines import (
    BM25Recommender,
    HistoricalPopularityRecommender,
    RandomRecommender,
    RecencyRecommender,
    SemanticSeededPPRRecommender,
    TfidfRecommender,
)
from gnn_pipeline.evaluation.recommendation.runner import run_recommenders
from tests.recommendation_fixture import build_fixture


class RecommendationBaselineTests(unittest.TestCase):
    def test_text_and_temporal_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = build_fixture(Path(tmp))
            query = benchmark.split_queries("test").iloc[0]
            candidates = benchmark.candidates_for_query(query)
            training = benchmark.training_data()
            visible_edges = benchmark.visible_citation_edges(query)

            popularity = HistoricalPopularityRecommender().fit(training)
            self.assertEqual(
                popularity.rank(
                    query, candidates, 1, 42, visible_citation_edges=visible_edges
                )[0].paper_id,
                "P0",
            )
            self.assertEqual(
                RecencyRecommender().rank(query, candidates, 1, 42)[0].paper_id,
                "P3",
            )
            self.assertEqual(
                TfidfRecommender().rank(query, candidates, 1, 42)[0].paper_id,
                "P2",
            )
            self.assertEqual(
                BM25Recommender().rank(query, candidates, 1, 42)[0].paper_id,
                "P2",
            )

    def test_random_is_query_seed_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = build_fixture(Path(tmp))
            query = benchmark.split_queries("test").iloc[0]
            candidates = benchmark.candidates_for_query(query)
            baseline = RandomRecommender()
            first = baseline.rank(query, candidates, 5, 13)
            second = baseline.rank(query, candidates, 5, 13)
            self.assertEqual(first, second)

    def test_seeded_ppr_returns_only_temporal_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = build_fixture(Path(tmp))
            query = benchmark.split_queries("test").iloc[0]
            candidates = benchmark.candidates_for_query(query)
            baseline = SemanticSeededPPRRecommender(seed_count=2).fit(
                benchmark.training_data()
            )
            result = baseline.rank(
                query,
                candidates,
                3,
                42,
                visible_citation_edges=benchmark.visible_citation_edges(query),
            )
            self.assertEqual(len(result), 3)
            self.assertTrue(
                {item.paper_id for item in result}.issubset(set(candidates["paper_id"]))
            )

    def test_runner_repeats_only_stochastic_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = build_fixture(Path(tmp))
            run = run_recommenders(
                benchmark,
                [RandomRecommender(), RecencyRecommender()],
                splits=["test"],
                seeds=[13, 42],
                top_k=3,
                metric_config={"recall": [3], "ndcg": [3]},
            )
            random_seeds = run.predictions[run.predictions["method"] == "random"][
                "seed"
            ].unique()
            recency_seeds = run.predictions[
                run.predictions["method"] == "recency"
            ]["seed"].unique()
            self.assertEqual(set(random_seeds), {13, 42})
            self.assertEqual(recency_seeds.tolist(), [13])
            self.assertEqual(len(run.per_query_metrics), 3)


if __name__ == "__main__":
    unittest.main()
