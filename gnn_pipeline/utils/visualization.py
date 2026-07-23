
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch


def plot_training_history(train_losses, val_metrics, filename=None):

    plt.figure(figsize=(15, 5))


    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.7, color='blue', linewidth=2)
    val_losses = [m[3] for m in val_metrics] if val_metrics else []
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', alpha=0.7, color='red', linewidth=2)

        best_epoch = np.argmin(val_losses)
        plt.scatter(best_epoch, val_losses[best_epoch], color='green', s=100, label='Best Model', zorder=5)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)


    plt.subplot(1, 3, 2)
    val_imp_accs = [m[0] for m in val_metrics] if val_metrics else []
    val_role_accs = [m[1] for m in val_metrics] if val_metrics else []

    if val_imp_accs:
        plt.plot(val_imp_accs, label='Importance Accuracy', alpha=0.7, color='green', linewidth=2)
    if val_role_accs:
        plt.plot(val_role_accs, label='Role Accuracy', alpha=0.7, color='orange', linewidth=2)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)


    plt.subplot(1, 3, 3)
    val_inf_maes = [m[2] for m in val_metrics] if val_metrics else []
    if val_inf_maes:
        plt.plot(val_inf_maes, label='Influence MAE', color='red', alpha=0.7, linewidth=2)

        best_mae_epoch = np.argmin(val_inf_maes)
        plt.scatter(best_mae_epoch, val_inf_maes[best_mae_epoch], color='purple', s=100, label='Best MAE', zorder=5)
    plt.title('Influence Score MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" Training history plot saved to: {filename}")
    plt.show()


def visualize_embeddings(embeddings, labels=None, title="Embedding Visualization", filename=None):

    if embeddings is None:
        print("No embeddings to visualize")
        return


    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()


    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    if labels is not None:

        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        color=colors[i], label=f'Class {label}', alpha=0.7, s=50)
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=50)

    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" Embedding visualization saved to: {filename}")
    plt.show()


def visualize_community_network(community_data, filename=None, collaboration_edges=None):

    import networkx as nx

    key_members = community_data.get('key_members', [])
    other_members = community_data.get('other_members', [])
    key_ids = {member['id'] for member in key_members}
    members = {member['id']: member for member in key_members + other_members}

    G = nx.Graph()
    for member_id, member in members.items():
        G.add_node(member_id,
                   influence=member.get('global_influence', 0.5),
                   role=member.get('role', 'Contributor'),
                   importance=member.get('importance', 'Medium'),
                   is_key=member_id in key_ids)

    # Edges come from the real author collaboration graph; only links between
    # members of this community are drawn.
    if collaboration_edges is not None and len(collaboration_edges):
        idx_to_id = {int(member_id[1:]): member_id for member_id in members}
        for src, dst in zip(collaboration_edges[0], collaboration_edges[1]):
            src_id = idx_to_id.get(int(src))
            dst_id = idx_to_id.get(int(dst))
            if src_id is not None and dst_id is not None and src_id != dst_id:
                G.add_edge(src_id, dst_id)

    if G.number_of_nodes() == 0:
        print("No community members to visualize")
        return

    plt.figure(figsize=(12, 10))

    # Node colors based on role
    role_colors = {
        'Community Leader': 'red',
        'Core Member': 'orange',
        'Contributor': 'green',
        'Sole Expert': 'blue'
    }

    node_colors = []
    node_sizes = []
    edge_colors = []
    for node in G.nodes():
        attrs = G.nodes[node]
        node_colors.append(role_colors.get(attrs['role'], 'gray'))
        node_sizes.append(100 + attrs['influence'] * 500)
        edge_colors.append('black' if attrs['is_key'] else 'none')

    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.8, edgecolors=edge_colors, linewidths=1.5)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)
    labels = {node: node for node in G.nodes() if G.nodes[node]['is_key']}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    legend_patches = [Patch(color=color, label=role) for role, color in role_colors.items()]
    legend_patches.append(Patch(facecolor='white', edgecolor='black', label='Key member'))
    plt.legend(handles=legend_patches, loc='upper right')

    plt.title(f"Community Network: {community_data.get('community_name', 'Unknown')}")
    plt.axis('off')

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" Community network visualization saved to: {filename}")
    plt.show()


def plot_quality_metrics(community_data, filename=None):

    metrics = community_data['enhanced_analysis']


    radar_metrics = ['research_quality', 'collaboration_potential',
                     'topic_alignment', 'research_productivity']

    values = [metrics.get(m, 0) for m in radar_metrics]
    labels = ['Research Quality', 'Collaboration', 'Topic Alignment', 'Productivity']


    N = len(radar_metrics)


    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]


    values += values[:1]


    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))


    plt.xticks(angles[:-1], labels, size=12)


    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=10)
    plt.ylim(0, 1)


    ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')
    ax.fill(angles, values, 'b', alpha=0.1)


    plt.title(f"Community Quality Metrics\n{community_data.get('community_name', 'Unknown')}",
              size=15, y=1.1)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" Quality metrics plot saved to: {filename}")
    plt.show()