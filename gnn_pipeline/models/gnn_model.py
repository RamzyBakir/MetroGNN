import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear


class AdvancedHeteroGCN(nn.Module):
    def __init__(
        self,
        hidden_channels=64,
        out_channels=32,
        num_layers=4,
        num_classes=3,
        num_topics=8,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_topics = num_topics
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            conv = HeteroConv(
                {
                    ("author", "authored", "paper"): SAGEConv(
                        (-1, -1), hidden_channels
                    ),
                    ("paper", "authored_by", "author"): SAGEConv(
                        (-1, -1), hidden_channels
                    ),
                    ("paper", "has_topic", "topic"): SAGEConv(
                        (-1, -1), hidden_channels
                    ),
                    ("topic", "has_paper", "paper"): SAGEConv(
                        (-1, -1), hidden_channels
                    ),
                    ("author", "collaborates_with", "author"): SAGEConv(
                        (-1, -1), hidden_channels
                    ),
                },
                aggr="mean",
            )
            self.convs.append(conv)

        self.importance_head = Linear(hidden_channels, num_classes)
        self.role_head = Linear(hidden_channels, 4)
        self.influence_head = Linear(hidden_channels, 1)

        self.paper_quality = nn.Sequential(nn.Linear(hidden_channels, 1), nn.Sigmoid())

        self.attention_weights = nn.Parameter(torch.randn(3))  # author, paper, topic

    def forward(self, x_dict, edge_index_dict):

        previous_x_dict = x_dict

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

            if i % 2 == 1 and i > 0:
                for key in x_dict.keys():
                    if (
                        key in previous_x_dict
                        and x_dict[key].shape == previous_x_dict[key].shape
                    ):
                        x_dict[key] = x_dict[key] + previous_x_dict[key]

            if i < self.num_layers - 1:
                previous_x_dict = x_dict

        if "paper" in x_dict:
            paper_scores = self.paper_quality(x_dict["paper"])
            x_dict["paper_quality"] = paper_scores

        if "author" in x_dict and "paper" in x_dict and "topic" in x_dict:
            author_emb = x_dict["author"]
            paper_emb = x_dict["paper"]
            topic_emb = x_dict["topic"]

            paper_context = self.aggregate_paper_context(
                author_emb, paper_emb, edge_index_dict
            )
            topic_context = self.aggregate_topic_context(
                author_emb, paper_emb, topic_emb, edge_index_dict
            )

            attention_weights = F.softmax(self.attention_weights, dim=0)

            enhanced_embeddings = (
                attention_weights[0] * author_emb
                + attention_weights[1] * paper_context
                + attention_weights[2] * topic_context
            )

            x_dict["author_enhanced"] = enhanced_embeddings

            importance_logits = self.importance_head(enhanced_embeddings)
            role_logits = self.role_head(enhanced_embeddings)
            influence_scores = self.influence_head(enhanced_embeddings)

            return x_dict, importance_logits, role_logits, influence_scores
        elif "author" in x_dict:
            author_features = x_dict["author"]
            importance_logits = self.importance_head(author_features)
            role_logits = self.role_head(author_features)
            influence_scores = self.influence_head(author_features)
            return x_dict, importance_logits, role_logits, influence_scores
        return x_dict, None, None

    def aggregate_paper_context(self, author_emb, paper_emb, edge_index_dict):
        authored_edges = edge_index_dict[("author", "authored", "paper")]
        author_idx = authored_edges[0]
        paper_idx = authored_edges[1]

        num_authors = author_emb.shape[0]
        hidden_dim = author_emb.shape[1]
        device = author_emb.device

        paper_feats = paper_emb[paper_idx]

        idx_expanded = author_idx.unsqueeze(1).expand_as(paper_feats)
        result = torch.zeros(num_authors, hidden_dim, device=device)
        result.scatter_add_(0, idx_expanded, paper_feats)

        counts = torch.zeros(num_authors, 1, device=device)
        counts.scatter_add_(
            0,
            author_idx.unsqueeze(1),
            torch.ones(author_idx.shape[0], 1, device=device),
        )

        return result / counts.clamp(min=1)

    def aggregate_topic_context(
        self, author_emb, paper_emb, topic_emb, edge_index_dict
    ):
        authored_edges = edge_index_dict[("author", "authored", "paper")]
        paper_topic_edges = edge_index_dict[("paper", "has_topic", "topic")]

        num_papers = paper_emb.shape[0]
        num_authors = author_emb.shape[0]
        hidden_dim = author_emb.shape[1]
        device = author_emb.device

        pt_paper = paper_topic_edges[0]
        pt_topic = paper_topic_edges[1]
        topic_feats = topic_emb[pt_topic]

        idx_expanded = pt_paper.unsqueeze(1).expand_as(topic_feats)
        paper_topic_ctx = torch.zeros(num_papers, hidden_dim, device=device)
        paper_topic_ctx.scatter_add_(0, idx_expanded, topic_feats)

        paper_counts = torch.zeros(num_papers, 1, device=device)
        paper_counts.scatter_add_(
            0, pt_paper.unsqueeze(1), torch.ones(pt_paper.shape[0], 1, device=device)
        )
        paper_topic_ctx = paper_topic_ctx / paper_counts.clamp(min=1)

        ap_author = authored_edges[0]
        ap_paper = authored_edges[1]
        author_feats = paper_topic_ctx[ap_paper]

        idx_expanded = ap_author.unsqueeze(1).expand_as(author_feats)
        result = torch.zeros(num_authors, hidden_dim, device=device)
        result.scatter_add_(0, idx_expanded, author_feats)

        author_counts = torch.zeros(num_authors, 1, device=device)
        author_counts.scatter_add_(
            0, ap_author.unsqueeze(1), torch.ones(ap_author.shape[0], 1, device=device)
        )

        return result / author_counts.clamp(min=1)
