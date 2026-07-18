"""Cold-query neural and graph architecture baselines."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, HeteroConv, SAGEConv


class TextDualEncoder(nn.Module):
    """Text ranker with a versioned residual encoder that preserves input geometry."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        *,
        model_version: int = 2,
        fusion_weight: float = 1.0,
    ):
        super().__init__()
        if model_version not in (1, 2):
            raise ValueError("Unsupported text dual-encoder model version")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout)
        self.model_version = int(model_version)
        self.fusion_weight = float(fusion_weight)
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.candidate_projection = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if self.model_version == 2:
            self.query_output = nn.Linear(hidden_dim, input_dim)
            self.candidate_output = nn.Linear(hidden_dim, input_dim)
            nn.init.zeros_(self.query_output.weight)
            nn.init.zeros_(self.query_output.bias)
            nn.init.zeros_(self.candidate_output.weight)
            nn.init.zeros_(self.candidate_output.bias)

    def encode_queries(self, values):
        projected = self.query_projection(values)
        if self.model_version == 1:
            return F.normalize(self.dropout(projected), dim=-1)
        hidden = self.dropout(F.relu(projected))
        return F.normalize(values + self.query_output(hidden), dim=-1)

    def encode_candidates(self, values):
        projected = self.candidate_projection(values)
        if self.model_version == 1:
            return F.normalize(self.dropout(projected), dim=-1)
        hidden = self.dropout(F.relu(projected))
        return F.normalize(values + self.candidate_output(hidden), dim=-1)

    def score(self, query_values, candidate_values):
        return (self.encode_queries(query_values) * self.encode_candidates(candidate_values)).sum(-1)


class CitationGraphSAGEEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = float(dropout)

    def forward(self, data):
        x = self.input_projection(data["paper"].x)
        edge_parts = []
        for edge_type in (
            ("paper", "cites", "paper"),
            ("paper", "cited_by", "paper"),
        ):
            if edge_type in data.edge_types:
                edge_parts.append(data[edge_type].edge_index)
        if not edge_parts:
            raise ValueError("Citation GraphSAGE requires citation relations")
        edge_index = torch.cat(edge_parts, dim=1)
        for layer, norm in zip(self.layers, self.norms):
            update = layer(x, edge_index)
            x = norm(x + F.dropout(F.relu(update), p=self.dropout, training=self.training))
        return x


class HeteroGraphSAGEEncoder(nn.Module):
    def __init__(self, metadata, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.input_projections = nn.ModuleDict(
            {node_type: nn.LazyLinear(hidden_dim) for node_type in node_types}
        )
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv((hidden_dim, hidden_dim), hidden_dim)
                        for edge_type in edge_types
                    },
                    aggr="mean",
                )
            )
            self.norms.append(
                nn.ModuleDict(
                    {node_type: nn.LayerNorm(hidden_dim) for node_type in node_types}
                )
            )
        self.dropout = float(dropout)

    def forward(self, data):
        x_dict = {
            node_type: self.input_projections[node_type](data[node_type].x)
            for node_type in self.node_types
        }
        for layer, norms in zip(self.layers, self.norms):
            updates = layer(x_dict, data.edge_index_dict)
            x_dict = {
                node_type: norms[node_type](
                    x_dict[node_type]
                    + F.dropout(
                        F.relu(updates.get(node_type, torch.zeros_like(x_dict[node_type]))),
                        p=self.dropout,
                        training=self.training,
                    )
                )
                for node_type in self.node_types
            }
        return x_dict["paper"]


class HGTEncoder(nn.Module):
    def __init__(
        self,
        metadata,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        heads: int = 2,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("HGT hidden_dim must be divisible by heads")
        node_types, _ = metadata
        self.node_types = list(node_types)
        self.input_projections = nn.ModuleDict(
            {node_type: nn.LazyLinear(hidden_dim) for node_type in node_types}
        )
        self.layers = nn.ModuleList(
            [
                HGTConv(
                    hidden_dim,
                    hidden_dim,
                    metadata,
                    heads=heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {node_type: nn.LayerNorm(hidden_dim) for node_type in node_types}
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = float(dropout)

    def forward(self, data):
        x_dict = {
            node_type: self.input_projections[node_type](data[node_type].x)
            for node_type in self.node_types
        }
        for layer, norms in zip(self.layers, self.norms):
            updates = layer(x_dict, data.edge_index_dict)
            x_dict = {
                node_type: norms[node_type](
                    x_dict[node_type]
                    + F.dropout(
                        F.relu(updates.get(node_type, torch.zeros_like(x_dict[node_type]))),
                        p=self.dropout,
                        training=self.training,
                    )
                )
                for node_type in self.node_types
            }
        return x_dict["paper"]


class TemporalGraphRanker(nn.Module):
    """Encode historical candidates with a graph and cold queries with text."""

    def __init__(
        self,
        *,
        architecture: str,
        metadata,
        text_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        hgt_heads: int = 2,
        model_version: int = 2,
        fusion_weight: float = 1.0,
    ):
        super().__init__()
        self.architecture = architecture
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        if model_version not in (1, 2):
            raise ValueError("Unsupported temporal graph model version")
        self.hgt_heads = int(hgt_heads)
        self.model_version = int(model_version)
        self.fusion_weight = float(fusion_weight)
        self.query_projection = nn.Linear(text_dim, hidden_dim)
        if architecture == "citation_sage":
            self.encoder = CitationGraphSAGEEncoder(
                text_dim, hidden_dim, num_layers, dropout
            )
        elif architecture == "hetero_sage":
            self.encoder = HeteroGraphSAGEEncoder(
                metadata, hidden_dim, num_layers, dropout
            )
        elif architecture == "hgt":
            self.encoder = HGTEncoder(
                metadata, hidden_dim, num_layers, dropout, heads=hgt_heads
            )
        else:
            raise ValueError(f"Unknown temporal graph architecture: {architecture}")
        if self.model_version == 2:
            self.query_output = nn.Linear(hidden_dim, text_dim)
            self.candidate_output = nn.Linear(hidden_dim, text_dim)
            nn.init.zeros_(self.query_output.weight)
            nn.init.zeros_(self.query_output.bias)
            nn.init.zeros_(self.candidate_output.weight)
            nn.init.zeros_(self.candidate_output.bias)

    def get_config(self) -> Dict:
        return {
            "architecture": self.architecture,
            "text_dim": self.text_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "hgt_heads": self.hgt_heads,
            "model_version": self.model_version,
            "fusion_weight": self.fusion_weight,
        }

    def encode_queries(self, query_embeddings):
        hidden = self.query_projection(query_embeddings)
        if self.model_version == 1:
            return F.normalize(hidden, dim=-1)
        residual = self.query_output(F.dropout(
            F.relu(hidden), p=self.dropout, training=self.training
        ))
        return F.normalize(query_embeddings + residual, dim=-1)

    def encode_candidates(self, data):
        hidden = self.encoder(data)
        if self.model_version == 1:
            return F.normalize(hidden, dim=-1)
        residual = self.candidate_output(hidden)
        return F.normalize(data["paper"].x + residual, dim=-1)

    def pair_scores(self, query_embeddings, candidate_embeddings):
        queries = self.encode_queries(query_embeddings)
        return (queries * candidate_embeddings).sum(dim=-1)

    def full_scores(self, query_embeddings, candidate_embeddings):
        queries = self.encode_queries(query_embeddings)
        return queries @ candidate_embeddings.T
