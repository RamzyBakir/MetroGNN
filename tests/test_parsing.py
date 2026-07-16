import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gnn_pipeline.data.parsing import (
    build_author_index,
    canonicalize_author,
    load_author_mapping,
    parse_authors,
    parse_list_field,
    save_author_mapping,
)


class ParsingTests(unittest.TestCase):
    def test_literal_parser_does_not_execute_input(self):
        self.assertEqual(parse_list_field("__import__('os').system('false')"), [])

    def test_dblp_id_wins_over_metadata_variants(self):
        df = pd.DataFrame(
            {
                "authors": [
                    "[{'id': 'A1', 'name': 'Ada', 'org': 'One'}]",
                    "[{'id': 'A1', 'name': 'Ada Lovelace', 'org': 'Two'}]",
                ]
            }
        )
        mapping, records, paper_indices, stats = build_author_index(df)
        self.assertEqual(mapping, {"dblp:A1": 0})
        self.assertEqual(paper_indices, [[0], [0]])
        self.assertEqual(stats["num_authors"], 1)
        self.assertEqual(records[0]["dblp_id"], "A1")

    def test_fallback_is_normalized_and_conservative(self):
        first = canonicalize_author({"id": "", "name": "  Jane  Doe ", "org": "Uni X"})
        second = canonicalize_author({"name": "jane doe", "org": " uni x "})
        different_org = canonicalize_author({"name": "Jane Doe", "org": "Uni Y"})
        self.assertEqual(first["author_key"], second["author_key"])
        self.assertNotEqual(first["author_key"], different_org["author_key"])

    def test_duplicate_author_on_paper_is_removed(self):
        authors = parse_authors(
            "[{'id':'A1','name':'Ada'}, {'id':'A1','name':'Ada'}]"
        )
        self.assertEqual(len(authors), 1)

    def test_mapping_round_trip(self):
        records = [
            {
                "author_idx": 0,
                "author_key": "dblp:A1",
                "dblp_id": "A1",
                "name": "Ada",
                "organization": "Example",
                "identity_source": "dblp_id",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "authors.json")
            save_author_mapping(records, path)
            self.assertEqual(load_author_mapping(path)[0]["name"], "Ada")


if __name__ == "__main__":
    unittest.main()
