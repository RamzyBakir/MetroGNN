import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pandas as pd
import os


try:
    from gnn_pipeline.config import get_path

    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    get_path = lambda *args: "models/best_model.pth"

try:
    from gnn_pipeline.utils.visualization import plot_training_history

    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

    def plot_training_history(*args, **kwargs):
        pass


# ==============================================================================
# EXTERNAL DATA REQUIRED
# ==============================================================================

_external_df = None
_external_author_to_idx = None


def set_external_data(df: pd.DataFrame, author_to_idx: dict):
    """
    REQUIRED: Set external data for valid importance prediction.
    Must be called BEFORE train_enhanced_model().
    """
    global _external_df, _external_author_to_idx
    _external_df = df
    _external_author_to_idx = author_to_idx
    print(f"External data set: {len(df)} papers, {len(author_to_idx)} authors")


def create_enhanced_training_targets(data):
    """
    Create training targets from EXTERNAL data ONLY.

    IMPORTANT: This function NO LONGER uses author_feats to create targets.
    Targets come exclusively from paper citation data (n_citation, year).

    Raises:
        RuntimeError: If set_external_data() was not called first.
    """
    global _external_df, _external_author_to_idx

    # ================================================================
    # REQUIRE EXTERNAL DATA - NO FALLBACK TO CIRCULAR DEPENDENCY
    # ================================================================
    if _external_df is None or _external_author_to_idx is None:
        raise RuntimeError(
            "\n" + "=" * 70 + "\n"
            "ERROR: External data not set!\n"
            "You MUST call set_external_data(df, author_to_idx) BEFORE training.\n"
            "This ensures targets come from external citation data, not input features.\n"
            "Example:\n"
            "  from training import set_external_data, train_enhanced_model\n"
            "  set_external_data(df, author_to_idx)\n"
            "  model, losses, ... = train_enhanced_model(model, data)\n"
            "=" * 70
        )

    print(
        "\n[INFO] Creating targets from EXTERNAL citation data (no circular dependency)"
    )

    df = _external_df
    author_to_idx = _external_author_to_idx

    num_authors = len(author_to_idx)

    # Get authorship edges from graph
    authored_edges = data["author", "authored", "paper"].edge_index
    paper_ids = authored_edges[1].cpu().numpy()
    author_ids = authored_edges[0].cpu().numpy()

    # ================================================================
    # IMPORTANCE TARGETS (from paper citations - EXTERNAL)
    # ================================================================
    paper_citations = pd.to_numeric(df["n_citation"], errors="coerce").fillna(0).values
    paper_years = pd.to_numeric(df["year"], errors="coerce").fillna(2025).values

    current_year = 2025
    citation_velocity = np.zeros(len(paper_citations))
    for i, year in enumerate(paper_years):
        age = max(1, int(current_year - year))
        citation_velocity[i] = paper_citations[i] / age

    # Paper importance (ONLY from external data - NOT from author features)
    paper_importance = paper_citations * 0.6 + citation_velocity * 0.4

    # Aggregate to authors (sum of paper importance)
    author_importance = np.zeros(num_authors)
    for author_id, paper_id in zip(author_ids, paper_ids):
        if paper_id < len(paper_importance):
            author_importance[author_id] += paper_importance[paper_id]

    # Percentile-based labels (3 classes)
    percentiles = np.argsort(np.argsort(author_importance)) / num_authors

    importance_targets = torch.zeros(num_authors, dtype=torch.long)
    importance_targets[percentiles <= 0.33] = 0  # Low
    importance_targets[(percentiles > 0.33) & (percentiles <= 0.66)] = 1  # Medium
    importance_targets[percentiles > 0.66] = 2  # High

    # ================================================================
    # ROLE TARGETS (from graph degree - EXTERNAL metric)
    # ================================================================
    role_targets = torch.zeros(num_authors, dtype=torch.long)
    collab_edges = data["author", "collaborates_with", "author"].edge_index

    degree = torch.zeros(num_authors)
    for i in range(collab_edges.shape[1]):
        src = collab_edges[0, i].item()
        degree[src] += 1

    degree_percentiles = np.percentile(degree.numpy(), [50, 75])
    for i in range(num_authors):
        deg = degree[i].item()
        if deg == 0:
            role_targets[i] = 3  # Sole Expert
        elif deg > degree_percentiles[1]:
            role_targets[i] = 0  # Leader
        elif deg > degree_percentiles[0]:
            role_targets[i] = 1  # Core
        else:
            role_targets[i] = 2  # Contributor

    # ================================================================
    # INFLUENCE TARGETS (from citation velocity - EXTERNAL)
    # ================================================================
    author_velocity = np.zeros(num_authors)
    for author_id, paper_id in zip(author_ids, paper_ids):
        if paper_id < len(citation_velocity):
            author_velocity[author_id] += citation_velocity[paper_id]

    if author_velocity.max() > author_velocity.min():
        influence_targets = (author_velocity - author_velocity.min()) / (
            author_velocity.max() - author_velocity.min()
        )
    else:
        influence_targets = np.zeros(num_authors)
    influence_targets = torch.tensor(influence_targets, dtype=torch.float32)

    # Print summary
    print(
        f"\n  Importance: Low={sum(importance_targets == 0)}, Medium={sum(importance_targets == 1)}, High={sum(importance_targets == 2)}"
    )
    print(
        f"  Role: Leader={sum(role_targets == 0)}, Core={sum(role_targets == 1)}, Contributor={sum(role_targets == 2)}, Sole={sum(role_targets == 3)}"
    )
    print(
        f"  Influence: min={influence_targets.min():.3f}, max={influence_targets.max():.3f}, mean={influence_targets.mean():.3f}"
    )

    return importance_targets, role_targets, influence_targets


def train_enhanced_model(model, data, epochs=200, checkpoint_path=None):
    """
    Train MetroGNN model with multi-task learning using EXTERNAL ground truth.

    IMPORTANT: Call set_external_data(df, author_to_idx) BEFORE this function.
    """
    if checkpoint_path is None and HAS_CONFIG:
        checkpoint_path = get_path("models", "best_model")
    elif checkpoint_path is None:
        os.makedirs("models", exist_ok=True)
        checkpoint_path = "models/best_model.pth"

    print("=" * 60)
    print("Starting enhanced training with multi-task learning")
    print("Using EXTERNAL ground truth (no circular dependency)")
    print("=" * 60)

    # Create targets from EXTERNAL data only
    importance_targets, role_targets, influence_targets = (
        create_enhanced_training_targets(data)
    )

    num_researchers = len(importance_targets)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(num_researchers, generator=generator)
    train_idx = indices[: int(0.7 * num_researchers)]
    val_idx = indices[int(0.7 * num_researchers) :]

    if HAS_CONFIG:
        try:
            splits_path = get_path("models", "splits_path")
            os.makedirs(os.path.dirname(splits_path), exist_ok=True)
            torch.save({"train_idx": train_idx, "val_idx": val_idx}, splits_path)
            print(f"Split indices saved to {splits_path}")
        except Exception as e:
            print(f"Warning: Could not save split indices: {e}")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )

    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_reg = torch.nn.MSELoss()

    train_losses = []
    val_metrics = []
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 50
    grad_clip_value = 1.0

    print(
        f"\nTraining on {len(train_idx)} researchers, validating on {len(val_idx)} researchers"
    )
    print(
        f"Class distribution - Importance: {torch.bincount(importance_targets).tolist()}"
    )
    print(f"Class distribution - Roles: {torch.bincount(role_targets).tolist()}")

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        try:
            updated_embeddings, importance_logits, role_logits, influence_scores = (
                model(data.x_dict, data.edge_index_dict)
            )
        except Exception as e:
            print(f"Error in forward pass at epoch {epoch}: {e}")
            continue

        if (
            importance_logits is not None
            and role_logits is not None
            and influence_scores is not None
        ):
            loss_importance = criterion_class(
                importance_logits[train_idx], importance_targets[train_idx]
            )
            loss_role = criterion_class(role_logits[train_idx], role_targets[train_idx])
            loss_influence = criterion_reg(
                influence_scores[train_idx].squeeze(), influence_targets[train_idx]
            )

            total_loss = loss_importance * 0.5 + loss_role * 0.3 + loss_influence * 0.2

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                try:
                    _, val_imp_logits, val_role_logits, val_inf_scores = model(
                        data.x_dict, data.edge_index_dict
                    )

                    val_imp_acc = accuracy_score(
                        importance_targets[val_idx].numpy(),
                        val_imp_logits[val_idx].argmax(dim=1).numpy(),
                    )
                    val_role_acc = accuracy_score(
                        role_targets[val_idx].numpy(),
                        val_role_logits[val_idx].argmax(dim=1).numpy(),
                    )
                    val_inf_mae = torch.abs(
                        val_inf_scores[val_idx].squeeze() - influence_targets[val_idx]
                    ).mean()

                    val_loss_importance = criterion_class(
                        val_imp_logits[val_idx], importance_targets[val_idx]
                    )
                    val_loss_role = criterion_class(
                        val_role_logits[val_idx], role_targets[val_idx]
                    )
                    val_loss_influence = criterion_reg(
                        val_inf_scores[val_idx].squeeze(), influence_targets[val_idx]
                    )
                    val_total_loss = (
                        val_loss_importance * 0.5
                        + val_loss_role * 0.3
                        + val_loss_influence * 0.2
                    )

                except Exception as e:
                    print(f"Error in validation at epoch {epoch}: {e}")
                    val_imp_acc, val_role_acc, val_inf_mae, val_total_loss = (
                        0,
                        0,
                        1.0,
                        float("inf"),
                    )

            model.train()

            train_losses.append(total_loss.item())
            val_metrics.append(
                (val_imp_acc, val_role_acc, val_inf_mae.item(), val_total_loss.item())
            )

            scheduler.step(val_total_loss)

            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1

            if epoch % 20 == 0 or epoch == epochs - 1:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch:03d}/{epochs}:")
                print(
                    f"  Loss: {total_loss:.4f} (Imp: {loss_importance:.4f}, Role: {loss_role:.4f}, Inf: {loss_influence:.4f})"
                )
                print(f"  Val Loss: {val_total_loss:.4f}, LR: {current_lr:.6f}")
                print(
                    f"  Val Acc - Imp: {val_imp_acc:.4f}, Role: {val_role_acc:.4f}, Inf MAE: {val_inf_mae:.4f}"
                )

                if patience_counter > 0:
                    print(f"  Early stopping: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        else:
            print(f"Warning: No predictions returned at epoch {epoch}")

    # Load best model
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded best model from training")
    except:
        print("Could not load best model, using final model")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        try:
            _, final_imp_logits, final_role_logits, final_inf_scores = model(
                data.x_dict, data.edge_index_dict
            )

            final_imp_acc = accuracy_score(
                importance_targets.numpy(), final_imp_logits.argmax(dim=1).numpy()
            )
            final_role_acc = accuracy_score(
                role_targets.numpy(), final_role_logits.argmax(dim=1).numpy()
            )
            final_inf_mae = torch.abs(
                final_inf_scores.squeeze() - influence_targets
            ).mean()

            print(f"\n=== FINAL PERFORMANCE (External Ground Truth) ===")
            print(f"Importance Accuracy: {final_imp_acc:.4f} ({final_imp_acc:.1%})")
            print(f"Role Accuracy: {final_role_acc:.4f} ({final_role_acc:.1%})")
            print(f"Influence MAE: {final_inf_mae:.4f}")
            print(f"Final Training Loss: {train_losses[-1]:.6f}")

            return (
                model,
                train_losses,
                val_metrics,
                final_imp_logits,
                final_role_logits,
                final_inf_scores,
            )

        except Exception as e:
            print(f"Error in final evaluation: {e}")
            return model, train_losses, val_metrics, None, None, None
