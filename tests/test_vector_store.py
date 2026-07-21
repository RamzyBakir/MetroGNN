import tempfile
import unittest
from pathlib import Path

from gnn_pipeline.artifacts import (
    ArtifactCompatibilityError,
    VECTOR_STORE_SCHEMA_VERSION,
    sha256_file,
    write_json_atomic,
)
from gnn_pipeline.data.vector_store import COLLECTION_NAME, EMBED_MODEL_NAME, PaperVectorStore


class VectorStoreManifestTests(unittest.TestCase):
    def test_compatibility_manifest_detects_stale_topics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "papers.csv"
            topics_path = root / "topics.csv"
            manifest_path = root / "vector_manifest.json"
            csv_path.write_text("id,title\nP1,Example\n")
            topics_path.write_text("paper_id,topic_id\nP1,0\n")

            with PaperVectorStore(
                persist_dir=str(root / "chroma"),
                manifest_path=str(manifest_path),
            ) as store:
                store.collection.upsert(
                    ids=["0"],
                    embeddings=[[1.0, 0.0]],
                    documents=["Example"],
                    metadatas=[{"topic_id": 0}],
                )
                write_json_atomic(
                    str(manifest_path),
                    {
                        "vector_store_schema_version": VECTOR_STORE_SCHEMA_VERSION,
                        "pipeline_id": "pipeline",
                        "dataset_sha256": sha256_file(str(csv_path)),
                        "topic_assignments_sha256": sha256_file(str(topics_path)),
                        "embedding_model": EMBED_MODEL_NAME,
                        "collection_name": COLLECTION_NAME,
                        "paper_count": 1,
                    },
                )
                store.validate_compatibility(
                    csv_path=str(csv_path),
                    topic_assignments_path=str(topics_path),
                    pipeline_id="pipeline",
                )
                topics_path.write_text("paper_id,topic_id\nP1,1\n")
                with self.assertRaises(ArtifactCompatibilityError):
                    store.validate_compatibility(
                        csv_path=str(csv_path),
                        topic_assignments_path=str(topics_path),
                        pipeline_id="pipeline",
                    )


if __name__ == "__main__":
    unittest.main()
