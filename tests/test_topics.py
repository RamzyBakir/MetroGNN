import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_pipeline.data.topic_assignment import (
    TopicAssigner,
    load_topic_artifacts,
    prepare_topics,
    save_topic_artifacts,
)


class TopicArtifactTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "id": ["P1", "P2", "P3"],
                "title": ["one", "two", "three"],
                "topic_id": [0, 1, 0],
                "topic_name": ["A", "B", "A"],
            }
        )
        self.info = {
            0: {"topic_id": 0, "name": "A", "top_terms": ["alpha"]},
            1: {"topic_id": 1, "name": "B", "top_terms": ["beta"]},
        }
        self.assigner = TopicAssigner(n_clusters=2, model_name="test-model")
        self.assigner.cluster_centers_ = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def _paths(self, tmp):
        return (
            str(Path(tmp) / "assignments.csv"),
            str(Path(tmp) / "info.json"),
            str(Path(tmp) / "centroids.npz"),
        )

    def test_round_trip_and_dataset_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            assignments, info, centroids = self._paths(tmp)
            save_topic_artifacts(
                self.df,
                self.info,
                self.assigner,
                assignments_path=assignments,
                topic_info_path=info,
                centroids_path=centroids,
                dataset_sha256="abc",
            )
            loaded, loaded_info, loaded_assigner, generated = prepare_topics(
                self.df.drop(columns=["topic_id", "topic_name"]),
                n_clusters=2,
                assignments_path=assignments,
                topic_info_path=info,
                centroids_path=centroids,
                dataset_sha256="abc",
            )
            self.assertFalse(generated)
            self.assertEqual(loaded["topic_id"].tolist(), [0, 1, 0])
            self.assertEqual(loaded_info[1]["name"], "B")
            np.testing.assert_allclose(loaded_assigner.cluster_centers_, self.assigner.cluster_centers_)

    def test_partial_artifact_set_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            assignments, info, centroids = self._paths(tmp)
            Path(assignments).write_text("paper_id,row_index,topic_id\n")
            with self.assertRaisesRegex(ValueError, "Incomplete topic artifact"):
                prepare_topics(
                    self.df.drop(columns=["topic_id", "topic_name"]),
                    n_clusters=2,
                    assignments_path=assignments,
                    topic_info_path=info,
                    centroids_path=centroids,
                    dataset_sha256="abc",
                )

    def test_stale_paper_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            assignments, info, centroids = self._paths(tmp)
            save_topic_artifacts(
                self.df,
                self.info,
                self.assigner,
                assignments_path=assignments,
                topic_info_path=info,
                centroids_path=centroids,
            )
            changed = self.df.iloc[::-1].reset_index(drop=True)
            with self.assertRaisesRegex(ValueError, "paper IDs"):
                load_topic_artifacts(
                    changed,
                    assignments_path=assignments,
                    topic_info_path=info,
                    centroids_path=centroids,
                )

    def test_single_topic_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            assignments, info, centroids = self._paths(tmp)
            broken = self.df.copy()
            broken["topic_id"] = 0
            save_topic_artifacts(
                broken,
                self.info,
                self.assigner,
                assignments_path=assignments,
                topic_info_path=info,
                centroids_path=centroids,
            )
            with self.assertRaisesRegex(ValueError, "fewer than two"):
                load_topic_artifacts(
                    broken.drop(columns=["topic_id", "topic_name"]),
                    assignments_path=assignments,
                    topic_info_path=info,
                    centroids_path=centroids,
                )


if __name__ == "__main__":
    unittest.main()
