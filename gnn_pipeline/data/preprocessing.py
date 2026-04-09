
import torch
import numpy as np


def create_enhanced_training_targets(data):

    num_researchers = data['author'].x.shape[0]
    researcher_feats = data['author'].x

    importance_scores = (
        researcher_feats[:, 0] * 0.3 +
        researcher_feats[:, 1] * 0.2 +
        researcher_feats[:, 4] * 0.5
    )

    quantiles = torch.quantile(importance_scores, torch.tensor([0.25, 0.75]))
    importance_targets = torch.zeros(num_researchers, dtype=torch.long)
    importance_targets[importance_scores <= quantiles[0]] = 0
    importance_targets[(importance_scores > quantiles[0]) & (importance_scores <= quantiles[1])] = 1
    importance_targets[importance_scores > quantiles[1]] = 2

    role_targets = torch.zeros(num_researchers, dtype=torch.long)
    collab_edges = data['author', 'collaborates_with', 'author'].edge_index

    for i in range(num_researchers):
        collab_count = (collab_edges[0] == i).sum().item()
        if collab_count == 0:
            role_targets[i] = 3
        elif collab_count > 6:
            role_targets[i] = 0
        elif collab_count > 3:
            role_targets[i] = 1
        else:
            role_targets[i] = 2

    influence_targets = (importance_scores - importance_scores.min()) / (
    importance_scores.max() - importance_scores.min())

    return importance_targets, role_targets, influence_targets


def normalize_features(data):

    for node_type in data.node_types:
        if data[node_type].x is not None:
            features = data[node_type].x

            if features.shape[1] > 1:

                mean = features.mean(dim=0, keepdim=True)
                std = features.std(dim=0, keepdim=True)
                std[std == 0] = 1
                data[node_type].x = (features - mean) / std
    return data


def get_node_statistics(data):

    stats = {}
    for node_type in data.node_types:
        stats[node_type] = {
            'count': data[node_type].x.shape[0],
            'features': data[node_type].x.shape[1],
            'mean': data[node_type].x.mean(dim=0).numpy(),
            'std': data[node_type].x.std(dim=0).numpy()
        }
    return stats