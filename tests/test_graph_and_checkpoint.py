import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn_pipeline.artifacts import (
    ArtifactCompatibilityError,
    create_pipeline_manifest,
    validate_pipeline_manifest,
    write_json_atomic,
)
from gnn_pipeline.data.graph_builder import build_knowledge_graph_from_dblp
from gnn_pipeline.data.validation import validate_knowledge_graph
from gnn_pipeline.models.checkpoint import (
    load_checkpoint_payload,
    load_model_for_pipeline,
    save_model_checkpoint,
)
from gnn_pipeline.models.gnn_model import AdvancedHeteroGCN


class GraphAndCheckpointTests(unittest.TestCase):
    def _build_graph(self, graph_path=None):
        df = pd.DataFrame(
            {
                "id": ["P1", "P2", "P3"],
                "title": ["one", "two", "three"],
                "references": ["['P2']", "['P1', 'outside']", "[]"],
                "author_indices": [[0, 1], [1], [2]],
                "topic_id": [0, 1, 0],
            }
        )
        topics = {
            0: {"name": "A", "top_terms": ["a"]},
            1: {"name": "B", "top_terms": ["b"]},
        }
        return build_knowledge_graph_from_dblp(
            df,
            {"a": 0, "b": 1, "c": 2},
            topics,
            np.ones((3, 7), dtype=np.float32),
            np.ones((3, 6), dtype=np.float32),
            np.eye(2, dtype=np.float32),
            save_path=graph_path,
        )[0]

    def test_topic_and_citation_edges(self):
        graph = self._build_graph()
        stats = validate_knowledge_graph(
            graph, expected_papers=3, expected_authors=3, expected_topics=2
        )
        self.assertEqual(stats["populated_topics"], 2)
        self.assertEqual(stats["citation_edges"], 2)
        cites = {tuple(edge) for edge in graph["paper", "cites", "paper"].edge_index.t().tolist()}
        self.assertEqual(cites, {(0, 1), (1, 0)})

    def test_legacy_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pth"
            torch.save({"weight": torch.ones(1)}, path)
            with self.assertRaisesRegex(ArtifactCompatibilityError, "Legacy"):
                load_checkpoint_payload(str(path))

    def test_versioned_checkpoint_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "papers.csv"
            csv_path.write_text("id\nP1\n")
            graph_path = root / "graph.pt"
            graph = self._build_graph(str(graph_path))
            author_path = root / "authors.json"
            assignments_path = root / "topics.csv"
            info_path = root / "topics.json"
            centroids_path = root / "centroids.npz"
            author_path.write_text("[]")
            assignments_path.write_text("paper_id,topic_id\nP1,0\n")
            info_path.write_text("{}")
            np.savez_compressed(centroids_path, cluster_centers=np.eye(2))

            manifest = create_pipeline_manifest(
                csv_path=str(csv_path),
                graph_path=str(graph_path),
                author_mapping_path=str(author_path),
                topic_assignments_path=str(assignments_path),
                topic_info_path=str(info_path),
                topic_centroids_path=str(centroids_path),
                embedding_model="test",
                n_clusters=2,
                graph_stats={"citation_edges": 2},
            )
            manifest_path = root / "manifest.json"
            write_json_atomic(str(manifest_path), manifest)

            model = AdvancedHeteroGCN(hidden_channels=8, num_layers=2, num_topics=2)
            with torch.no_grad():
                model(graph.x_dict, graph.edge_index_dict)
            checkpoint_path = root / "model.pth"
            save_model_checkpoint(
                model, str(checkpoint_path), pipeline_id=manifest["pipeline_id"]
            )
            loaded, payload = load_model_for_pipeline(
                str(checkpoint_path), str(manifest_path)
            )
            self.assertEqual(payload["pipeline_id"], manifest["pipeline_id"])
            self.assertEqual(loaded.get_config()["hidden_channels"], 8)

            assignments_path.write_text("corrupted")
            with self.assertRaises(ArtifactCompatibilityError):
                validate_pipeline_manifest(str(manifest_path))


if __name__ == "__main__":
    unittest.main()
