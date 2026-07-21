import unittest

import pandas as pd

from gnn_pipeline.data.feature_engineering import create_paper_features
from gnn_pipeline.data.validation import ensure_paper_identity_columns
from gnn_pipeline.models.training import create_author_split_indices


class TrainingSplitAndFeatureTests(unittest.TestCase):
    def test_author_splits_include_held_out_test_set(self):
        train_idx, val_idx, test_idx = create_author_split_indices(100, seed=7)
        self.assertEqual(len(train_idx), 70)
        self.assertEqual(len(val_idx), 15)
        self.assertEqual(len(test_idx), 15)

        combined = (
            train_idx.tolist()
            + val_idx.tolist()
            + test_idx.tolist()
        )
        self.assertEqual(sorted(combined), list(range(100)))
        self.assertEqual(len(set(combined)), 100)

    def test_recency_uses_dataset_observation_year(self):
        df = pd.DataFrame(
            {
                "n_citation": [1, 1, 1],
                "year": [2026, 2024, 2010],
                "authors_parsed": [["a"], ["b"], ["c"]],
                "title": ["new", "recent", "old"],
                "topic_id": [0, 0, 0],
            }
        )
        features = create_paper_features(df, {0: {"name": "Topic"}})
        self.assertAlmostEqual(features[0, 1], 1.0)
        self.assertAlmostEqual(features[1, 1], 0.8)
        self.assertAlmostEqual(features[2, 1], 0.0)

    def test_legacy_csv_gets_stable_identity_columns(self):
        df = pd.DataFrame(
            {
                "title": ["paper a", "paper b"],
                "authors": ["[]", "[]"],
                "year": [2024, 2025],
                "n_citation": [0, 1],
                "venue": ["v", "v"],
            }
        )
        normalized = ensure_paper_identity_columns(df)
        self.assertEqual(normalized["id"].tolist(), ["legacy-row:0", "legacy-row:1"])
        self.assertEqual(normalized["references"].tolist(), ["[]", "[]"])
        self.assertNotIn("id", df.columns)


if __name__ == "__main__":
    unittest.main()
