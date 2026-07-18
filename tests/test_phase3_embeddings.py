import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gnn_pipeline.evaluation.recommendation.embeddings import (
    EmbeddingSpec,
    build_embedding_cache,
    load_embedding_cache,
)
from tests.recommendation_fixture import build_fixture


class _FakeEncoder:
    def __init__(self, model_name, device=None):
        self.model_name = model_name

    def encode(self, texts, **kwargs):
        values = np.asarray(
            [[len(text), sum(character.isalpha() for character in text), 1.0] for text in texts],
            dtype=np.float32,
        )
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        return values


class Phase3EmbeddingTests(unittest.TestCase):
    def test_versioned_embedding_cache_round_trip_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = build_fixture(root)
            spec = EmbeddingSpec(name="fake", model_name="fake/model")
            with patch("sentence_transformers.SentenceTransformer", _FakeEncoder):
                table = build_embedding_cache(
                    benchmark, spec, cache_root=str(root / "cache")
                )
            loaded = load_embedding_cache(
                benchmark, spec, cache_root=str(root / "cache")
            )
            self.assertEqual(table.dimension, 3)
            np.testing.assert_allclose(
                loaded.lookup(["P0", "P4"]), table.lookup(["P0", "P4"])
            )

            archive = next((root / "cache").glob("**/embeddings.npz"))
            archive.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_embedding_cache(
                    benchmark, spec, cache_root=str(root / "cache")
                )


if __name__ == "__main__":
    unittest.main()
