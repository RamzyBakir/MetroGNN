import unittest

import pandas as pd

from gnn_pipeline.evaluation.recommendation.metrics import (
    average_precision_at_k,
    evaluate_rankings,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


class RecommendationMetricTests(unittest.TestCase):
    def test_known_binary_ranking_values(self):
        ranking = ["X", "A", "Y"]
        qrels = {"A": 1.0}
        self.assertEqual(recall_at_k(ranking, qrels, 1), 0.0)
        self.assertEqual(recall_at_k(ranking, qrels, 2), 1.0)
        self.assertAlmostEqual(reciprocal_rank_at_k(ranking, qrels, 10), 0.5)
        self.assertAlmostEqual(average_precision_at_k(ranking, qrels, 10), 0.5)
        self.assertAlmostEqual(ndcg_at_k(ranking, qrels, 10), 1.0 / 1.5849625007)

    def test_metrics_are_computed_from_persisted_rankings(self):
        predictions = pd.DataFrame(
            [
                {
                    "query_id": "Q",
                    "candidate_id": "X",
                    "rank": 1,
                    "score": 0.9,
                    "method": "method",
                    "seed": 42,
                    "split": "test",
                    "benchmark_id": "benchmark",
                },
                {
                    "query_id": "Q",
                    "candidate_id": "A",
                    "rank": 2,
                    "score": 0.8,
                    "method": "method",
                    "seed": 42,
                    "split": "test",
                    "benchmark_id": "benchmark",
                },
            ]
        )
        qrels = pd.DataFrame(
            [{"query_id": "Q", "candidate_id": "A", "relevance": 1.0}]
        )
        metrics = evaluate_rankings(
            predictions,
            qrels,
            {"recall": [1, 2], "ndcg": [2], "mrr": [2], "map": [2]},
        ).iloc[0]
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertAlmostEqual(metrics["mrr@2"], 0.5)

    def test_duplicate_predictions_are_rejected(self):
        row = {
            "query_id": "Q",
            "candidate_id": "A",
            "rank": 1,
            "score": 1.0,
            "method": "m",
            "seed": 1,
            "split": "test",
            "benchmark_id": "b",
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_rankings(
                pd.DataFrame([row, {**row, "rank": 2}]),
                pd.DataFrame(
                    [{"query_id": "Q", "candidate_id": "A", "relevance": 1.0}]
                ),
            )


if __name__ == "__main__":
    unittest.main()
