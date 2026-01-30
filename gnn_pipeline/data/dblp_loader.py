import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_and_clean_dblp_data(
    csv_path: str,
) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """
    Load DBLP subset data and perform cleaning operations.

    Args:
        csv_path: Path to dblp_subset.csv

    Returns:
        df: Cleaned DataFrame with papers
        author_to_idx: Mapping from author names to unique indices
        author_list: List of author names in index order
    """
    df = pd.read_csv(csv_path)

    print(f"Original dataset: {len(df)} papers")

    # Filter out papers with NaN venues
    df = df.dropna(subset=["venue"])
    print(f"After removing papers with NaN venues: {len(df)} papers")

    # Parse authors column (stored as stringified lists)
    df["authors_parsed"] = df["authors"].apply(parse_authors)

    # Filter out papers with empty or invalid author lists
    df = df[df["authors_parsed"].apply(lambda x: len(x) > 0)]
    print(f"After removing papers with invalid authors: {len(df)} papers")

    # Extract all unique authors
    all_authors = []
    for author_list in df["authors_parsed"]:
        all_authors.extend(author_list)

    unique_authors = list(sorted(set(all_authors)))
    print(f"Found {len(unique_authors)} unique authors")

    # Create author to index mapping
    author_to_idx = {author: idx for idx, author in enumerate(unique_authors)}
    author_list = unique_authors

    # Add author indices to dataframe
    df["author_indices"] = df["authors_parsed"].apply(
        lambda authors: [author_to_idx[a] for a in authors]
    )

    return df, author_to_idx, author_list


def parse_authors(authors_str: str) -> List[str]:
    """
    Parse authors column which is stored as a stringified list.

    Args:
        authors_str: String representation of author list

    Returns:
        List of author names
    """
    try:
        # Try to parse as JSON
        authors = json.loads(authors_str.replace("'", '"'))
        if isinstance(authors, list):
            return [str(a).strip() for a in authors]
        return []
    except:
        # Fallback: try to extract using eval (less safe but handles Python syntax)
        try:
            authors = eval(authors_str)
            if isinstance(authors, list):
                return [str(a).strip() for a in authors]
            return []
        except:
            return []


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
    csv_path = "/home/ramzy/Documents/Uni/metrognn/data/dblp_subset.csv"

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
