from pathlib import Path

import pandas as pd

from gnn_pipeline.evaluation.recommendation.benchmark import (
    build_temporal_citation_benchmark,
)


def write_papers(path: Path) -> Path:
    rows = [
        {
            "id": "P0",
            "title": "Foundations of graph learning",
            "abstract": "graph representation foundations",
            "keywords": "['graph']",
            "year": 2020,
            "authors": "[{'id':'A0','name':'A'}]",
            "references": "[]",
            "venue": "V",
            "n_citation": 10,
        },
        {
            "id": "P1",
            "title": "Graph neural networks",
            "abstract": "neural message passing",
            "keywords": "['graph neural network']",
            "year": 2021,
            "authors": "[{'id':'A1','name':'B'}]",
            "references": "['P0']",
            "venue": "V",
            "n_citation": 8,
        },
        {
            "id": "P2",
            "title": "Quantum optimization methods",
            "abstract": "quantum optimization algorithm",
            "keywords": "['quantum optimization']",
            "year": 2022,
            "authors": "[{'id':'A2','name':'C'}]",
            "references": "['P0', 'P2', 'P3']",
            "venue": "V",
            "n_citation": 5,
        },
        {
            "id": "P2B",
            "title": "Contemporary graph systems",
            "abstract": "graph systems",
            "keywords": "['graph']",
            "year": 2022,
            "authors": "[{'id':'A3','name':'D'}]",
            "references": "['P2']",
            "venue": "V",
            "n_citation": 3,
        },
        {
            "id": "P3",
            "title": "Scalable graph networks",
            "abstract": "scalable graph neural models",
            "keywords": "['graph']",
            "year": 2023,
            "authors": "[{'id':'A4','name':'E'}]",
            "references": "['P1']",
            "venue": "V",
            "n_citation": 2,
        },
        {
            "id": "P4",
            "title": "Applications of quantum optimization",
            "abstract": "new quantum optimization applications",
            "keywords": "['quantum optimization']",
            "year": 2024,
            "authors": "[{'id':'A5','name':'F'}]",
            "references": "['P2']",
            "venue": "V",
            "n_citation": 1,
        },
        {
            "id": "P5",
            "title": "Incomplete future record",
            "abstract": "future",
            "keywords": "['future']",
            "year": 2025,
            "authors": "[{'id':'A6','name':'G'}]",
            "references": "['P1']",
            "venue": "V",
            "n_citation": 0,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_fixture(root: Path):
    csv_path = write_papers(root / "papers.csv")
    return build_temporal_citation_benchmark(
        str(csv_path),
        splits={"train": [2021, 2022], "validation": [2023], "test": [2024]},
        query_text_fields=["title", "abstract", "keywords"],
    )
