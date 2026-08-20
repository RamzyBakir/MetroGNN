
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


def visualize_community_network(community_data, filename=None, collaboration_edges=None, max_viz_nodes=250):
    """
    Publication-grade multi-panel community network visualizer with:
    - Force-directed collaboration graph layout
    - Role-coded nodes sized by global influence and centrality
    - Distinct halos on key community leaders
    - Prominent callout annotations on top researchers
    - Integrated side-panel showing community metrics and role breakdown
    """
    import networkx as nx
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec

    key_members = community_data.get('key_members', [])
    other_members = community_data.get('other_members', [])

    if len(key_members) > max_viz_nodes:
        displayed_key = sorted(
            key_members,
            key=lambda m: m.get('global_influence', 0),
            reverse=True,
        )[:max_viz_nodes]
        displayed_others = []
    else:
        displayed_key = key_members
        rem = max_viz_nodes - len(displayed_key)
        if rem > 0 and other_members:
            displayed_others = sorted(
                other_members,
                key=lambda m: m.get('global_influence', 0),
                reverse=True,
            )[:rem]
        else:
            displayed_others = []

    key_ids = {member['id'] for member in key_members}
    members = {member['id']: member for member in displayed_key + displayed_others}

    G = nx.Graph()
    for member_id, member in members.items():
        G.add_node(
            member_id,
            influence=member.get('global_influence', 0.5),
            role=member.get('role', 'Contributor'),
            importance=member.get('importance', 'Medium'),
            is_key=member_id in key_ids,
            composite=member.get('composite_score', 0.5),
            centrality=member.get('centrality', 0.0),
        )

    # Real collaboration edges
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

    # Palette
    role_colors = {
        'Community Leader': '#E63946',  # Crimson Red
        'Core Member': '#F4A261',       # Amber Gold
        'Contributor': '#2A9D8F',       # Emerald Teal
        'Sole Expert': '#457B9D'        # Steel Blue
    }

    # Setup multi-panel figure
    fig = plt.figure(figsize=(16, 9.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[3.2, 1.2], height_ratios=[1.2, 1.0], wspace=0.22, hspace=0.28)

    ax_net = fig.add_subplot(gs[:, 0])
    ax_card = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, 1])

    # 1. Main Network Plot
    node_colors = []
    node_sizes = []
    edge_colors = []
    linewidths = []

    for node in G.nodes():
        attrs = G.nodes[node]
        node_colors.append(role_colors.get(attrs['role'], '#6C757D'))
        # Non-linear size scaling based on influence & centrality
        inf = attrs['influence']
        cent = attrs.get('centrality', 0.0)
        size = 80 + (inf ** 1.3) * 750 + cent * 300
        node_sizes.append(size)
        edge_colors.append('#1D3557' if attrs['is_key'] else '#6C757D')
        linewidths.append(2.0 if attrs['is_key'] else 0.6)

    # Force-directed layout
    k_dist = 1.3 / np.sqrt(max(G.number_of_nodes(), 1))
    pos = nx.spring_layout(G, k=k_dist, iterations=70, seed=42)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax_net,
        alpha=0.32,
        edge_color='#8D99AE',
        width=1.0,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax_net,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.88,
        edgecolors=edge_colors,
        linewidths=linewidths,
    )

    # Annotate top 10 most influential leaders with badge callouts
    top_leaders = sorted(
        [n for n in G.nodes() if G.nodes[n]['is_key']],
        key=lambda n: G.nodes[n]['influence'],
        reverse=True
    )[:10]

    for node in top_leaders:
        x, y = pos[node]
        r_name = G.nodes[node]['role'].replace("Community ", "")
        ax_net.annotate(
            f"{node}\n({r_name})",
            (x, y),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#0B090A",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=role_colors.get(G.nodes[node]['role'], '#333333'),
                alpha=0.90,
                linewidth=1.2,
            ),
            zorder=6,
        )

    # Network Title & Legend
    comm_name = community_data.get('community_name', 'Research Community')
    ax_net.set_title(
        f"Community Collaboration Subgraph: {comm_name}\n"
        f"(Displaying top {G.number_of_nodes()} members, {G.number_of_edges()} collaboration edges)",
        fontsize=13, fontweight="bold", pad=12
    )
    ax_net.axis('off')
    ax_net.set_facecolor('#FAFAFA')

    legend_patches = [
        Patch(facecolor=color, edgecolor='none', label=role)
        for role, color in role_colors.items()
    ]
    legend_patches.append(Patch(facecolor='white', edgecolor='#1D3557', linewidth=1.8, label='Key Member (Border)'))
    ax_net.legend(handles=legend_patches, loc='lower left', framealpha=0.92, facecolor='#FFFFFF', edgecolor='#CCCCCC', fontsize=9)

    # 2. Side Panel 1: Community Overview Card
    ax_card.axis('off')
    ax_card.set_facecolor('#F8F9FA')

    metrics = community_data.get('community_metrics', {})
    enhanced = community_data.get('enhanced_analysis', {})
    total_size = len(key_members) + len(other_members)
    key_count = len(key_members)

    card_text = (
        f"COMMUNITY DIAGNOSTICS\n"
        f"─────────────────────────────\n"
        f"• Total Members:       {total_size:,}\n"
        f"• Key Leaders:         {key_count:,} ({key_count / max(total_size, 1) * 100:.1f}%)\n"
        f"• Graph Density:       {metrics.get('cohesion', 0):.4f}\n"
        f"• Avg Influence:       {enhanced.get('avg_influence', 0):.4f}\n"
        f"• Research Quality:    {enhanced.get('research_quality', 0):.4f}\n"
        f"• Topic Alignment:     {enhanced.get('topic_alignment', 0):.4f}\n"
        f"• Productivity:        {enhanced.get('research_productivity', 0):.4f}\n"
        f"• Network Health:      {enhanced.get('network_health', 0):.4f}\n"
    )

    ax_card.text(
        0.05, 0.95, card_text,
        transform=ax_card.transAxes,
        fontsize=9.5,
        fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFFFFF', edgecolor='#D0D7DE', alpha=0.95)
    )
    ax_card.set_title("Community Metrics", fontsize=11, fontweight="bold")

    # 3. Side Panel 2: Role Distribution Breakdown
    all_roles = [m.get('role', 'Contributor') for m in key_members + other_members]
    role_counts = {r: all_roles.count(r) for r in role_colors}

    y_pos = np.arange(len(role_colors))
    counts = [role_counts[r] for r in role_colors]
    bar_colors = [role_colors[r] for r in role_colors]

    bars = ax_dist.barh(y_pos, counts, color=bar_colors, edgecolor='#333333', linewidth=0.5, alpha=0.85)
    ax_dist.set_yticks(y_pos)
    ax_dist.set_yticklabels([r.replace("Community ", "") for r in role_colors], fontsize=9)
    ax_dist.invert_yaxis()
    ax_dist.set_xlabel("Member Count", fontsize=9)
    ax_dist.set_title("Role Distribution", fontsize=11, fontweight="bold")
    ax_dist.grid(axis='x', linestyle='--', alpha=0.3)
    ax_dist.set_facecolor('#FAFAFA')

    for bar in bars:
        w = bar.get_width()
        pct = (w / max(total_size, 1)) * 100.0
        ax_dist.text(w + max(counts) * 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{w:,} ({pct:.1f}%)", va='center', fontsize=8, fontweight='bold', color='#333333')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" Community network visualization saved to: {filename}")
    plt.show()


def plot_quality_metrics(community_data, filename=None):
    """
    Publication-grade 2-panel quality metrics visualizer:
    - Polar radar chart with percentage vertices and shaded polygons
    - Horizontal benchmark comparison chart across all core dimensions
    """
    metrics = community_data.get('enhanced_analysis', {})

    radar_keys = [
        'research_quality', 'collaboration_potential',
        'topic_alignment', 'research_productivity',
        'influence_concentration', 'network_health'
    ]
    labels = [
        'Research Quality', 'Collaboration',
        'Topic Alignment', 'Productivity',
        'Influence Concen.', 'Network Health'
    ]

    values = [float(metrics.get(k, 0.5)) for k in radar_keys]
    N = len(radar_keys)

    fig = plt.figure(figsize=(15, 6.5), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0], wspace=0.30)

    # 1. Radar Chart (Left)
    ax_radar = fig.add_subplot(gs[0, 0], projection='polar')

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    plot_values = values + values[:1]

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, size=10, fontweight="bold", color="#1D3557")

    ax_radar.set_rlabel_position(0)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color="#6C757D", size=8.5)
    ax_radar.set_ylim(0, 1.05)

    # Draw polygon with gradient/alpha
    ax_radar.plot(angles, plot_values, linewidth=2.5, linestyle='solid', color='#1D3557', marker='o', markersize=6)
    ax_radar.fill(angles, plot_values, color='#457B9D', alpha=0.30)

    # Annotate vertices with exact percentages
    for angle, val in zip(angles[:-1], values):
        ax_radar.text(angle, val + 0.08, f"{val * 100:.1f}%",
                      ha='center', va='center', size=8.5, fontweight='bold', color='#1D3557',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#457B9D', alpha=0.85, linewidth=0.8))

    comm_name = community_data.get('community_name', 'Research Community')
    ax_radar.set_title(f"Community Capability Profile\n{comm_name}", size=12, fontweight="bold", pad=20)
    ax_radar.grid(color='#D0D7DE', linestyle='--', linewidth=0.8)

    # 2. Horizontal Benchmark Comparison Bar Chart (Right)
    ax_bar = fig.add_subplot(gs[0, 1])

    y_pos = np.arange(len(labels))
    bar_colors = ['#2A9D8F' if v >= 0.5 else '#E76F51' for v in values]

    bars = ax_bar.barh(y_pos, values, color=bar_colors, edgecolor='#333333', linewidth=0.6, alpha=0.85)
    ax_bar.axvline(0.5, color='#E63946', linestyle='--', linewidth=1.2, label='Baseline Reference (0.50)')

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=9.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_xlabel("Metric Score (0.0 to 1.0)", fontsize=10)
    ax_bar.set_title("Dimensional Scores & Baseline Comparison", fontsize=12, fontweight="bold")
    ax_bar.grid(axis='x', linestyle='--', alpha=0.35)
    ax_bar.set_facecolor('#FAFAFA')
    ax_bar.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

    for bar, val in zip(bars, values):
        w = bar.get_width()
        ax_bar.text(w + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va='center', fontsize=9, fontweight='bold', color='#1D3557')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" Quality metrics plot saved to: {filename}")
    plt.show()