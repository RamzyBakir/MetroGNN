from typing import Dict, List, Tuple

import pandas as pd

from gnn_pipeline.data.parsing import build_author_index, parse_authors as parse_author_records
from gnn_pipeline.data.validation import validate_papers_dataframe


def load_and_clean_dblp_data(
    csv_path: str,
) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """
    Load DBLP subset data and perform cleaning operations.

    Args:
        csv_path: Path to dblp_subset.csv

    Returns:
        df: Cleaned DataFrame with papers
        author_to_idx: Mapping from stable author keys to unique indices
        author_list: Stable author keys in graph index order
    """
    df = pd.read_csv(csv_path)
    validate_papers_dataframe(df)

    print(f"Original dataset: {len(df)} papers")

    # Keep row order immutable: graph node IDs and vector-store IDs share it.
    print(f"Papers with missing venues: {int(df['venue'].isna().sum())}")

    # Resolve stable identities before filtering so paper IDs remain aligned with
    # graph and vector-store node IDs.
    author_to_idx, author_records, paper_author_indices, identity_stats = (
        build_author_index(df)
    )
    df["authors_parsed"] = df["authors"].apply(parse_authors)
    df["author_indices"] = paper_author_indices
    print(f"Found {len(author_records)} stable authors")
    print(f"Author identity statistics: {identity_stats}")

    author_list = [record["author_key"] for record in author_records]
    return df, author_to_idx, author_list


def parse_authors(authors_str: str) -> List[str]:
    """
    Parse authors into stable identity keys without executing input.

    Args:
        authors_str: String representation of author list

    Returns:
        List of author names
    """
    return [record["author_key"] for record in parse_author_records(authors_str)]


def get_dataset_statistics(df: pd.DataFrame, author_to_idx: Dict[str, int]) -> Dict:
    """
    Calculate basic statistics about the dataset.

    Args:
        df: Cleaned DataFrame
        author_to_idx: Author to index mapping

    Returns:
        Dictionary of statistics
    """
    stats = {
        "num_papers": len(df),
        "num_unique_authors": len(author_to_idx),
        "avg_authors_per_paper": df["authors_parsed"].apply(len).mean(),
        "total_authors": df["authors_parsed"].apply(len).sum(),
        "year_range": (df["year"].min(), df["year"].max()),
        "citation_stats": {
            "mean": df["n_citation"].mean(),
            "median": df["n_citation"].median(),
            "max": df["n_citation"].max(),
            "min": df["n_citation"].min(),
        },
        "venues": df["venue"].nunique(),
    }

    return stats


if __name__ == "__main__":
    # Test the loader
    from gnn_pipeline.config import get_path

    csv_path = get_path("data", "csv_path")

    print("=" * 60)
    print("Loading DBLP Data")
    print("=" * 60)

    df, author_to_idx, author_list = load_and_clean_dblp_data(csv_path)

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    stats = get_dataset_statistics(df, author_to_idx)

    print(f"\nPapers: {stats['num_papers']}")
    print(f"Unique Authors: {stats['num_unique_authors']}")
    print(f"Average Authors per Paper: {stats['avg_authors_per_paper']:.2f}")
    print(f"Total Author-Paper Links: {stats['total_authors']}")
    print(f"Year Range: {stats['year_range'][0]} - {stats['year_range'][1]}")
    print(f"Unique Venues: {stats['venues']}")

    print(f"\nCitation Statistics:")
    print(f"  Mean: {stats['citation_stats']['mean']:.2f}")
    print(f"  Median: {stats['citation_stats']['median']:.2f}")
    print(f"  Max: {stats['citation_stats']['max']}")
    print(f"  Min: {stats['citation_stats']['min']}")

    print(f"\nFirst 5 Papers:")
    print(df[["title", "year", "n_citation"]].head())

    print(f"\nSample Authors (first 10):")
    print(author_list[:10])
