import torch
import numpy as np
from torch_geometric.data import HeteroData
import pandas as pd
from typing import Dict, Tuple, Optional

from gnn_pipeline.artifacts import GRAPH_SCHEMA_VERSION
from gnn_pipeline.data.parsing import parse_list_field


def create_meaningful_node_features(
    num_researchers=20, num_papers=30, num_topics=8, observation_year=2026
):

    researcher_features = []
    for i in range(num_researchers):
        paper_count = np.random.poisson(8) + 1
        collaboration_count = np.random.poisson(5) + 1
        topic_diversity = np.random.beta(2, 5)
        years_active = np.random.uniform(1, 20)
        citation_impact = np.random.gamma(2, 2)

        researcher_features.append(
            [
                paper_count,
                collaboration_count,
                topic_diversity,
                years_active,
                citation_impact,
                0,
                0,
                0,
            ]
        )

    paper_features = []
    for i in range(num_papers):
        citation_count = np.random.poisson(10)
        topic_relevance = np.random.uniform(0.5, 1.0)
        recency = observation_year - np.random.uniform(2010, observation_year)
        author_count = np.random.poisson(3) + 1

        paper_features.append(
            [citation_count, topic_relevance, recency, author_count, 0, 0, 0, 0, 0, 0]
        )

    topic_features = torch.eye(num_topics)

    return (
        torch.tensor(researcher_features, dtype=torch.float32),
        torch.tensor(paper_features, dtype=torch.float32),
        topic_features,
    )


def build_enhanced_knowledge_graph():

    data = HeteroData()

    num_researchers = 20
    num_papers = 30
    num_topics = 8

    researcher_feats, paper_feats, topic_feats = create_meaningful_node_features(
        num_researchers, num_papers, num_topics
    )

    data["researcher"].x = researcher_feats
    data["paper"].x = paper_feats
    data["topic"].x = topic_feats

    authorship_edges = []
    paper_assignments = {}

    for paper_id in range(num_papers):
        author_weights = researcher_feats[:, 0].numpy() + 1
        author_weights = author_weights / author_weights.sum()
        num_authors = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
        authors = np.random.choice(
            num_researchers, size=num_authors, p=author_weights, replace=False
        )

        for author in authors:
            authorship_edges.append([author, paper_id])
        paper_assignments[paper_id] = authors

    data["researcher", "authored", "paper"].edge_index = torch.tensor(
        authorship_edges, dtype=torch.long
    ).t()

    authored_edge_index = data["researcher", "authored", "paper"].edge_index
    data["paper", "authored_by", "researcher"].edge_index = torch.stack(
        [authored_edge_index[1], authored_edge_index[0]]
    )

    paper_topic_edges = []
    for paper_id in range(num_papers):
        num_topics_for_paper = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        topics = np.random.choice(num_topics, size=num_topics_for_paper, replace=False)
        for topic in topics:
            paper_topic_edges.append([paper_id, topic])

    data["paper", "has_topic", "topic"].edge_index = torch.tensor(
        paper_topic_edges, dtype=torch.long
    ).t()

    has_topic_edge_index = data["paper", "has_topic", "topic"].edge_index
    data["topic", "has_paper", "paper"].edge_index = torch.stack(
        [has_topic_edge_index[1], has_topic_edge_index[0]]
    )

    collaboration_edges = []
    for i in range(num_researchers):
        num_collabs = np.random.randint(2, 9)
        potential_collabs = list(set(range(num_researchers)) - {i})
        collab_weights = np.ones(len(potential_collabs))

        for j, collab in enumerate(potential_collabs):
            coauthored = any(
                set(paper_assignments.get(p, [])) == {i, collab}
                for p in range(num_papers)
            )
            if coauthored:
                collab_weights[j] += 5

        if len(potential_collabs) > 0:
            collab_weights = collab_weights / collab_weights.sum()
            num_to_sample = min(num_collabs, len(potential_collabs))
            collaborators = np.random.choice(
                potential_collabs, size=num_to_sample, p=collab_weights, replace=False
            )

            for collab in collaborators:
                collaboration_edges.append([i, collab])
                collaboration_edges.append([collab, i])

    if collaboration_edges:
        collaboration_tensor = (
            torch.tensor(collaboration_edges, dtype=torch.long).t().contiguous()
        )
        data[
            "researcher", "collaborates_with", "researcher"
        ].edge_index = collaboration_tensor
    else:
        collaboration_edges = []
        for i in range(num_researchers):
            for j in range(i + 1, min(i + 3, num_researchers)):
                collaboration_edges.append([i, j])
                collaboration_edges.append([j, i])
        data["researcher", "collaborates_with", "researcher"].edge_index = (
            torch.tensor(collaboration_edges, dtype=torch.long).t().contiguous()
        )

    topic_names = {
        0: "Artificial Intelligence",
        1: "Machine Learning",
        2: "Deep Learning",
        3: "Natural Language Processing",
        4: "Computer Vision",
        5: "Data Mining",
        6: "Cloud Computing",
        7: "Cybersecurity",
    }

    return data, topic_names


def build_knowledge_graph_from_dblp(
    df: pd.DataFrame,
    author_to_idx: Dict[str, int],
    topic_info_dict: Dict,
    paper_features: np.ndarray,
    author_features: np.ndarray,
    topic_features: np.ndarray,
    save_path: Optional[str] = None,
) -> Tuple[HeteroData, Dict, Dict]:
    """
    Build HeteroData from DBLP data and computed features.

    Args:
        df: DataFrame with stable paper IDs, references, author_indices, and topic_id
        author_to_idx: Mapping from stable author key to index
        topic_info_dict: Topic information dictionary
        paper_features: Precomputed paper features (num_papers, 7)
        author_features: Precomputed author features (num_authors, 6)
        topic_features: Precomputed topic features (num_topics, num_topics)
        save_path: Optional path to save graph

    Returns:
        data: PyTorch Geometric HeteroData object
        topic_names: Dictionary mapping topic_id to topic name
        topic_info_dict: Same as input (for reference)
    """
    data = HeteroData()
    data.pipeline_schema_version = GRAPH_SCHEMA_VERSION

    # Set node features
    data["paper"].x = torch.FloatTensor(paper_features)
    data["author"].x = torch.FloatTensor(author_features)
    data["topic"].x = torch.FloatTensor(topic_features)

    # Build authorship edges (author -> paper)
    authored_edges = []
    # Reset index to ensure sequential paper IDs
    df_reset = df.reset_index(drop=True)
    for idx, row in df_reset.iterrows():
        paper_id = idx
        author_indices = row.get("author_indices", [])
        for author_idx in author_indices:
            if paper_id < paper_features.shape[0]:  # Sanity check
                authored_edges.append([author_idx, paper_id])

    data["author", "authored", "paper"].edge_index = (
        torch.tensor(authored_edges, dtype=torch.long).t().contiguous()
    )

    # Add reverse edges (paper -> author)
    authored_edge_index = data["author", "authored", "paper"].edge_index
    data["paper", "authored_by", "author"].edge_index = torch.stack(
        [authored_edge_index[1], authored_edge_index[0]]
    ).contiguous()

    # Build paper-topic edges (paper -> topic)
    if "topic_id" not in df_reset.columns:
        raise ValueError("The paper DataFrame has no persisted topic assignments")
    paper_topic_edges = []
    for paper_id, row in df_reset.iterrows():
        topic_id = int(row["topic_id"])
        if not 0 <= topic_id < topic_features.shape[0]:
            raise ValueError(f"Invalid topic_id={topic_id} for paper {paper_id}")
        paper_topic_edges.append([paper_id, topic_id])

    data["paper", "has_topic", "topic"].edge_index = (
        torch.tensor(paper_topic_edges, dtype=torch.long).t().contiguous()
    )

    # Add reverse edges (topic -> paper)
    has_topic_edge_index = data["paper", "has_topic", "topic"].edge_index
    data["topic", "has_paper", "paper"].edge_index = torch.stack(
        [has_topic_edge_index[1], has_topic_edge_index[0]]
    ).contiguous()

    # Build in-subset citation edges (paper -> paper).
    paper_id_to_idx = {
        str(paper_id): idx for idx, paper_id in enumerate(df_reset["id"].tolist())
    }
    citation_edges = set()
    for source_idx, row in df_reset.iterrows():
        for referenced_id in parse_list_field(row.get("references", [])):
            target_idx = paper_id_to_idx.get(str(referenced_id))
            if target_idx is not None and target_idx != source_idx:
                citation_edges.add((int(source_idx), int(target_idx)))

    sorted_citations = sorted(citation_edges)
    if sorted_citations:
        citation_tensor = torch.tensor(sorted_citations, dtype=torch.long).t().contiguous()
    else:
        citation_tensor = torch.empty((2, 0), dtype=torch.long)
    data["paper", "cites", "paper"].edge_index = citation_tensor
    data["paper", "cited_by", "paper"].edge_index = torch.stack(
        [citation_tensor[1], citation_tensor[0]]
    ).contiguous()

    # Build collaboration edges (author -> author)
    collab_edges = []
    collab_set = set()

    for idx, row in df_reset.iterrows():
        author_indices = row.get("author_indices", [])
        # Add all pairs of coauthors (both directions)
        for i in range(len(author_indices)):
            for j in range(len(author_indices)):
                if i != j:
                    author_i = author_indices[i]
                    author_j = author_indices[j]
                    edge = tuple(sorted([author_i, author_j]))
                    if edge not in collab_set:
                        collab_set.add(edge)
                        collab_edges.append([author_i, author_j])
                        collab_edges.append([author_j, author_i])

    if collab_edges:
        collab_tensor = torch.tensor(collab_edges, dtype=torch.long).t().contiguous()
    else:
        collab_tensor = torch.empty((2, 0), dtype=torch.long)
    data["author", "collaborates_with", "author"].edge_index = collab_tensor

    # Create topic names dictionary (use dynamic topic names)
    topic_names = {tid: topic_info_dict[tid]["name"] for tid in topic_info_dict.keys()}

    # Optionally save graph
    if save_path:
        torch.save(data, save_path)
        print(f"Graph saved to: {save_path}")

    return data, topic_names, topic_info_dict
