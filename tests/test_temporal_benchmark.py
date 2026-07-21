import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gnn_pipeline.evaluation.recommendation.benchmark import (
    TemporalCitationBenchmark,
    build_temporal_citation_benchmark,
)
from tests.recommendation_fixture import build_fixture, write_papers


class TemporalBenchmarkTests(unittest.TestCase):
    def test_strict_temporal_splits_and_anomalies(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = build_fixture(Path(tmp))
            self.assertEqual(len(benchmark.split_queries("train")), 2)
            self.assertEqual(len(benchmark.split_queries("validation")), 1)
            self.assertEqual(len(benchmark.split_queries("test")), 1)
            self.assertEqual(len(benchmark.qrels), 4)
            self.assertEqual(benchmark.metadata["anomalies"]["self_references"], 1)
            self.assertEqual(benchmark.metadata["anomalies"]["future_edges"], 1)
            self.assertEqual(benchmark.metadata["anomalies"]["same_year_edges"], 1)

            query = benchmark.split_queries("test").iloc[0]
            candidates = benchmark.candidates_for_query(query)
            self.assertEqual(set(candidates["paper_id"]), {"P0", "P1", "P2", "P2B", "P3"})
            self.assertEqual(benchmark.qrels_for_query("P4"), {"P2": 1.0})

            training = benchmark.training_data()
            self.assertLessEqual(training.papers["year"].max(), 2022)
            self.assertLessEqual(training.citation_edges["source_year"].max(), 2022)
            self.assertEqual(set(training.train_qrels["query_id"]), {"P1", "P2"})
            visible = benchmark.visible_citation_edges(query)
            self.assertTrue((visible["source_year"] < 2024).all())

    def test_arbitrary_free_text_query_ids_and_graded_qrels_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = build_fixture(Path(tmp))
            benchmark.queries.loc[
                benchmark.queries["query_id"] == "P4", "query_id"
            ] = "human-query-1"
            benchmark.qrels.loc[
                benchmark.qrels["query_id"] == "P4", "query_id"
            ] = "human-query-1"
            benchmark.qrels.loc[
                benchmark.qrels["query_id"] == "human-query-1", "relevance"
            ] = 3.0
            self.assertEqual(benchmark.validate()["queries"], 4)

    def test_persisted_benchmark_checksums_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = build_fixture(root)
            output = root / "benchmark"
            manifest = benchmark.save(str(output))
            loaded = TemporalCitationBenchmark.load(str(output))
            self.assertEqual(loaded.benchmark_id, benchmark.benchmark_id)
            self.assertEqual(loaded.validate()["qrels"], 4)

            (output / manifest["files"]["qrels"]).write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                TemporalCitationBenchmark.load(str(output))

    def test_optional_abstract_column_can_be_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = write_papers(root / "papers.csv")
            frame = pd.read_csv(csv_path).drop(columns=["abstract"])
            frame.to_csv(csv_path, index=False)

            benchmark = build_temporal_citation_benchmark(
                str(csv_path),
                splits={"train": [2021, 2022], "validation": [2023], "test": [2024]},
                query_text_fields=["title", "abstract", "keywords"],
            )
            self.assertEqual(benchmark.validate()["queries"], 4)

    def test_legacy_proxy_qrels_require_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = write_papers(root / "papers.csv")
            frame = pd.read_csv(csv_path).drop(columns=["references"])
            frame.to_csv(csv_path, index=False)

            splits = {"train": [2021, 2022], "validation": [2023], "test": [2024]}
            with self.assertRaisesRegex(ValueError, "No strictly-prior internal"):
                build_temporal_citation_benchmark(
                    str(csv_path),
                    splits=splits,
                    query_text_fields=["title", "abstract", "keywords"],
                )

            benchmark = build_temporal_citation_benchmark(
                str(csv_path),
                splits=splits,
                query_text_fields=["title", "abstract", "keywords"],
                allow_legacy_proxy_qrels=True,
                proxy_qrels_per_query=2,
            )
            self.assertEqual(
                benchmark.metadata["relevance_source"],
                "legacy_metadata_citation_proxy",
            )
            self.assertTrue(benchmark.metadata["legacy_proxy_qrels"])
            self.assertGreater(len(benchmark.qrels), 0)
            self.assertTrue(
                (benchmark.qrels["target_year"] < benchmark.qrels["query_year"]).all()
            )
            self.assertEqual(set(benchmark.queries["query_year"]), {2021, 2022, 2023, 2024})
            self.assertEqual(benchmark.validate()["queries"], len(benchmark.queries))


if __name__ == "__main__":
    unittest.main()
