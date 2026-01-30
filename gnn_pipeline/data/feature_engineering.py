import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import json


def create_paper_features(df: pd.DataFrame, topic_info_dict: Dict, tfidf_vectors: np.ndarray = None) -> np.ndarray:
    """
    Create feature vectors for papers based on available data.
    
    Features (6 dims + 2 reserved):
    0. n_citation: Citation count from dataset
    1. recency: 2025 - year (more recent = higher)
    2. author_count: Number of authors
    3. title_length: Character count of title
    4. topic_relevance: Max TF-IDF score for paper's topic
    5-6. reserved: Placeholder features (zeros)
    
    Args:
        df: DataFrame with papers (must have topic_id column)
        topic_info_dict: Dictionary with topic information
        tfidf_vectors: TF-IDF vectors (optional, for topic relevance)
        
    Returns:
        paper_features: NumPy array of shape (num_papers, 7)
    """
    num_papers = len(df)
    paper_features = np.zeros((num_papers, 7), dtype=np.float32)
    
    # Feature 0: n_citation
    paper_features[:, 0] = df['n_citation'].values.astype(np.float32)
    
    # Feature 1: recency (2025 - year, normalized)
    current_year = 2025
    max_age = 10  # Assume papers older than 10 years get max penalty
    recency = (current_year - df['year'].values).astype(np.float32)
    paper_features[:, 1] = np.clip(1 - (recency / max_age), 0, 1)  # Newer = higher
    
    # Feature 2: author_count
    if 'authors_parsed' in df.columns:
        paper_features[:, 2] = df['authors_parsed'].apply(len).values.astype(np.float32)
    else:
        paper_features[:, 2] = 1.0  # Default if no author info
    
    # Feature 3: title_length (normalized)
    title_lengths = df['title'].str.len().values.astype(np.float32)
    max_title_len = 200  # Reasonable max for normalization
    paper_features[:, 3] = np.clip(title_lengths / max_title_len, 0, 1)
    
    # Feature 4: topic_relevance (max TF-IDF score for assigned topic)
    if tfidf_vectors is not None:
        for idx, row in df.iterrows():
            topic_id = row['topic_id']
            # Get TF-IDF vector for this paper
            paper_tfidf = tfidf_vectors[idx]
            # Use TF-IDF score as relevance (simplified approach)
            # In real scenario, would use cluster centroid distance
            paper_features[idx, 4] = paper_tfidf.max()
    else:
        paper_features[:, 4] = 0.5  # Default if no TF-IDF
    
    # Features 5-6: Reserved (zeros for future use)
    paper_features[:, 5] = 0.0
    paper_features[:, 6] = 0.0
    
    return paper_features


def create_author_features(df: pd.DataFrame, author_to_idx: Dict[str, int]) -> np.ndarray:
    """
    Create feature vectors for authors by aggregating paper-level data.
    
    Features (8 dims + 2 reserved):
    0. paper_count: Number of papers by this author
    1. collaboration_count: Number of unique coauthors
    2. topic_diversity: Number of distinct topics worked on
    3. years_active: max_year - min_year + 1
    4. total_citations: Sum of all paper citations
    5. avg_citations: total / paper_count
    6. h_index_approx: Approximate h-index (simplified)
    7-8. reserved: Placeholder features (zeros)
    
    Args:
        df: DataFrame with papers (must have author_indices and topic_id)
        author_to_idx: Mapping from author name to index
        
    Returns:
        author_features: NumPy array of shape (num_authors, 9)
    """
    num_authors = len(author_to_idx)
    author_features = np.zeros((num_authors, 9), dtype=np.float32)
    
    # Initialize data structures
    author_papers = {author_idx: [] for author_idx in range(num_authors)}
    author_coauthors = {author_idx: set() for author_idx in range(num_authors)}
    author_topics = {author_idx: set() for author_idx in range(num_authors)}
    author_citations = {author_idx: [] for author_idx in range(num_authors)}
    author_years = {author_idx: [] for author_idx in range(num_authors)}
    
    # Aggregate data from papers
    for idx, row in df.iterrows():
        author_indices = row.get('author_indices', [])
        topic_id = row.get('topic_id', -1)
        citations = row['n_citation']
        year = row['year']
        
        for author_idx in author_indices:
            author_papers[author_idx].append(idx)
            author_citations[author_idx].append(citations)
            author_years[author_idx].append(year)
            if topic_id >= 0:
                author_topics[author_idx].add(topic_id)
            
            # Track coauthors
            for coauthor_idx in author_indices:
                if coauthor_idx != author_idx:
                    author_coauthors[author_idx].add(coauthor_idx)
    
    # Compute features for each author
    for author_idx in range(num_authors):
        papers = author_papers[author_idx]
        coauthors = author_coauthors[author_idx]
        topics = author_topics[author_idx]
        citations = author_citations[author_idx]
        years = author_years[author_idx]
        
        # Feature 0: paper_count
        author_features[author_idx, 0] = len(papers)
        
        # Feature 1: collaboration_count
        author_features[author_idx, 1] = len(coauthors)
        
        # Feature 2: topic_diversity
        author_features[author_idx, 2] = len(topics)
        
        # Feature 3: years_active
        if len(years) > 0:
            author_features[author_idx, 3] = max(years) - min(years) + 1
        else:
            author_features[author_idx, 3] = 1
        
        # Feature 4: total_citations
        author_features[author_idx, 4] = sum(citations)
        
        # Feature 5: avg_citations
        if len(citations) > 0:
            author_features[author_idx, 5] = sum(citations) / len(citations)
        else:
            author_features[author_idx, 5] = 0
        
        # Feature 6: h_index_approx (simplified)
        # Real h-index: max h such that author has h papers with >= h citations
        if len(citations) > 0:
            sorted_citations = sorted(citations, reverse=True)
            h_approx = 0
            for i, cit in enumerate(sorted_citations):
                if cit >= i + 1:
                    h_approx = i + 1
                else:
                    break
            author_features[author_idx, 6] = h_approx
        else:
            author_features[author_idx, 6] = 0
        
        # Features 7-8: Reserved (zeros for future use)
        author_features[author_idx, 7] = 0.0
        author_features[author_idx, 8] = 0.0
    
    return author_features


def create_topic_features(topic_info_dict: Dict) -> np.ndarray:
    """
    Create feature vectors for topics using one-hot encoding.
    
    Args:
        topic_info_dict: Dictionary with topic information
        
    Returns:
        topic_features: NumPy array of shape (num_topics, num_topics)
                       (one-hot encoding: torch.eye(num_topics))
    """
    num_topics = len(topic_info_dict)
    topic_features = np.eye(num_topics, dtype=np.float32)
    
    return topic_features


def create_all_features(
    df: pd.DataFrame,
    topic_info_dict: Dict,
    author_to_idx: Dict[str, int],
    tfidf_vectors: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Create all feature matrices for the knowledge graph.
    
    Args:
        df: DataFrame with papers and topic assignments
        topic_info_dict: Dictionary with topic information
        author_to_idx: Mapping from author name to index
        tfidf_vectors: TF-IDF vectors (optional)
        
    Returns:
        paper_features: Shape (num_papers, 7)
        author_features: Shape (num_authors, 9)
        topic_features: Shape (num_topics, num_topics)
        feature_info: Dictionary with feature descriptions
    """
    # Create paper features
    paper_features = create_paper_features(df, topic_info_dict, tfidf_vectors)
    
    # Create author features
    author_features = create_author_features(df, author_to_idx)
    
    # Create topic features
    topic_features = create_topic_features(topic_info_dict)
    
    # Create feature descriptions
    feature_info = {
        'paper_features': {
            'dims': paper_features.shape[1],
            'description': [
                'n_citation: Citation count',
                'recency: 1 - (2025 - year) / 10 (normalized)',
                'author_count: Number of authors',
                'title_length: Title length / 200 (normalized)',
                'topic_relevance: Max TF-IDF score',
                'reserved_0: Placeholder',
                'reserved_1: Placeholder'
            ]
        },
        'author_features': {
            'dims': author_features.shape[1],
            'description': [
                'paper_count: Number of papers',
                'collaboration_count: Unique coauthors',
                'topic_diversity: Distinct topics worked on',
                'years_active: max_year - min_year + 1',
                'total_citations: Sum of all citations',
                'avg_citations: Average citations per paper',
                'h_index_approx: Simplified h-index',
                'reserved_0: Placeholder',
                'reserved_1: Placeholder'
            ]
        },
        'topic_features': {
            'dims': topic_features.shape[1],
            'description': [
                f'one_hot_{i}: Topic {i} indicator' for i in range(topic_features.shape[1])
            ]
        },
        'statistics': {
            'num_papers': paper_features.shape[0],
            'num_authors': author_features.shape[0],
            'num_topics': topic_features.shape[0]
        }
    }
    
    return paper_features, author_features, topic_features, feature_info


def print_feature_statistics(feature_info: Dict):
    """
    Print summary of feature statistics.
    
    Args:
        feature_info: Dictionary with feature descriptions and statistics
    """
    print("\n" + "=" * 70)
    print("FEATURE STATISTICS")
    print("=" * 70)
    
    print(f"\nNodes in Graph:")
    print(f"  Papers: {feature_info['statistics']['num_papers']}")
    print(f"  Authors: {feature_info['statistics']['num_authors']}")
    print(f"  Topics: {feature_info['statistics']['num_topics']}")
    
    print(f"\nPaper Features ({feature_info['paper_features']['dims']} dimensions):")
    for i, desc in enumerate(feature_info['paper_features']['description']):
        print(f"  [{i}] {desc}")
    
    print(f"\nAuthor Features ({feature_info['author_features']['dims']} dimensions):")
    for i, desc in enumerate(feature_info['author_features']['description']):
        print(f"  [{i}] {desc}")
    
    print(f"\nTopic Features ({feature_info['topic_features']['dims']} dimensions):")
    for i, desc in enumerate(feature_info['topic_features']['description']):
        print(f"  [{i}] {desc}")


if __name__ == "__main__":
    print("This module provides feature engineering functionality.")
    print("Use create_all_features() function with your DBLP DataFrame.")