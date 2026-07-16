import unittest

import pandas as pd

from gnn_pipeline.evaluation.recommendation.statistics import (
    bootstrap_mean_ci,
    compare_paired_methods,
    holm_adjust,
    paired_bootstrap_difference,
)


class RecommendationStatisticsTests(unittest.TestCase):
    def test_bootstrap_uses_observed_values(self):
        estimate, lower, upper = bootstrap_mean_ci(
            [0.0, 0.5, 1.0], n_resamples=2000, seed=7
        )
        self.assertAlmostEqual(estimate, 0.5)
        self.assertLessEqual(lower, estimate)
        self.assertGreaterEqual(upper, estimate)

        difference = paired_bootstrap_difference(
            [0.4, 0.6, 0.8], [0.2, 0.4, 0.6], n_resamples=1000, seed=7
        )
        self.assertAlmostEqual(difference[0], 0.2)
        self.assertAlmostEqual(difference[1], 0.2)
        self.assertAlmostEqual(difference[2], 0.2)

    def test_holm_correction(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual([round(value, 2) for value in adjusted], [0.03, 0.06, 0.06])

    def test_comparison_pairs_queries_and_averages_seeds(self):
        rows = []
        for query_index in range(5):
            query = f"Q{query_index}"
            rows.append(
                {
                    "query_id": query,
                    "method": "baseline",
                    "seed": 1,
                    "split": "test",
                    "benchmark_id": "b",
                    "ndcg@10": 0.2,
                    "recall@20": 0.4,
                }
            )
            for seed, delta in [(1, 0.1), (2, 0.3)]:
                rows.append(
                    {
                        "query_id": query,
                        "method": "proposed",
                        "seed": seed,
                        "split": "test",
                        "benchmark_id": "b",
                        "ndcg@10": 0.2 + delta,
                        "recall@20": 0.4 + delta,
                    }
                )
        comparison = compare_paired_methods(
            pd.DataFrame(rows),
            proposed_method="proposed",
            baseline_method="baseline",
            metrics=["ndcg@10", "recall@20"],
            bootstrap_resamples=1000,
            permutation_resamples=1000,
            seed=3,
        )
        self.assertEqual(comparison["num_queries"].tolist(), [5, 5])
        self.assertTrue((comparison["mean_difference"].round(8) == 0.2).all())
        self.assertTrue((comparison["wins"] == 5).all())
        self.assertTrue((comparison["ties"] == 0).all())


if __name__ == "__main__":
    unittest.main()
