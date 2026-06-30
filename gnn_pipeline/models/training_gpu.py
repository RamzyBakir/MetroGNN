"""GPU-optimized training with mixed precision and mini-batch support"""

import os
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import scatter

from gnn_pipeline.config import get_path
from gnn_pipeline.utils.visualization import plot_training_history
from gnn_pipeline.models.training import (
    set_external_data,
    _external_df,
    _external_author_to_idx,
)


def create_enhanced_training_targets(data):
    from gnn_pipeline.models.training import create_enhanced_training_targets as _create

    return _create(data)


def train_enhanced_model_gpu(
    model,
    data,
    epochs=200,
    device=None,
    batch_size=1024,
    num_neighbors=15,
    use_amp=True,
    checkpoint_path=None,
):
    """
    GPU-optimized training with mixed precision and mini-batch support.

    Args:
        model: GNN model (GPU-optimized version)
        data: HeteroData graph
        epochs: Number of training epochs
        device: torch.device (auto-detected if None)
        batch_size: Mini-batch size (default: 1024)
        num_neighbors: Number of neighbors to sample (default: 15)
        use_amp: Use automatic mixed precision (default: True)

    Returns:
        model, train_losses, val_metrics, final_logits
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = get_path("models", "best_model_gpu")

    print("Starting GPU-optimized training with mixed precision")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Neighbors: {num_neighbors}")
    print(f"AMP: {use_amp}")

    importance_targets, role_targets, influence_targets = (
        create_enhanced_training_targets(data)
    )

    num_researchers = len(importance_targets)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(num_researchers, generator=generator)
    train_idx = indices[: int(0.7 * num_researchers)]
    val_idx = indices[int(0.7 * num_researchers) :]

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

    grad_clip_value = 1.0

    train_losses = []
    val_metrics = []
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 50

    print(
        f"Training on {len(train_idx)} researchers, validating on {len(val_idx)} researchers"
    )
    print(
        f"Class distribution - Importance: {torch.bincount(importance_targets).tolist()}"
    )
    print(f"Class distribution - Roles: {torch.bincount(role_targets).tolist()}")

    num_neighbors_list = [num_neighbors] * 4

    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors_list,
        batch_size=batch_size,
        input_nodes=("author", train_idx),
        shuffle=True,
        num_workers=0,
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors_list,
        batch_size=batch_size,
        input_nodes=("author", val_idx),
        shuffle=False,
        num_workers=0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        optimizer.zero_grad()

        for batch in train_loader:
            batch = batch.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                try:
                    (
                        updated_embeddings,
                        importance_logits,
                        role_logits,
                        influence_scores,
                    ) = model(batch.x_dict, batch.edge_index_dict)
                except Exception as e:
                    print(f"Error in forward pass at epoch {epoch}: {e}")
                    continue

            if (
                importance_logits is not None
                and role_logits is not None
                and influence_scores is not None
            ):
                batch_train_idx = batch["author"].batch_size

                loss_importance = criterion_class(
                    importance_logits[:batch_train_idx],
                    importance_targets[batch["author"].input_n_id[:batch_train_idx]],
                )
                loss_role = criterion_class(
                    role_logits[:batch_train_idx],
                    role_targets[batch["author"].input_n_id[:batch_train_idx]],
                )
                loss_influence = criterion_reg(
                    influence_scores[:batch_train_idx].squeeze(),
                    influence_targets[batch["author"].input_n_id[:batch_train_idx]],
                )

                total_loss = (
                    loss_importance * 0.5 + loss_role * 0.3 + loss_influence * 0.2
                )

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_value
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                epoch_loss += total_loss.item()
                num_batches += 1

        model.eval()
        val_total_loss = 0.0
        val_num_batches = 0

        all_val_imp_preds = []
        all_val_imp_targets = []
        all_val_role_preds = []
        all_val_role_targets = []
        all_val_inf_preds = []
        all_val_inf_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                try:
                    _, val_imp_logits, val_role_logits, val_inf_scores = model(
                        batch.x_dict, batch.edge_index_dict
                    )
                except Exception as e:
                    print(f"Error in validation at epoch {epoch}: {e}")
                    continue

                batch_val_idx = batch["author"].batch_size

                val_loss_importance = criterion_class(
                    val_imp_logits[:batch_val_idx],
                    importance_targets[batch["author"].input_n_id[:batch_val_idx]],
                )
                val_loss_role = criterion_class(
                    val_role_logits[:batch_val_idx],
                    role_targets[batch["author"].input_n_id[:batch_val_idx]],
                )
                val_loss_influence = criterion_reg(
                    val_inf_scores[:batch_val_idx].squeeze(),
                    influence_targets[batch["author"].input_n_id[:batch_val_idx]],
                )
                val_batch_loss = (
                    val_loss_importance * 0.5
                    + val_loss_role * 0.3
                    + val_loss_influence * 0.2
                )

                val_total_loss += val_batch_loss.item()
                val_num_batches += 1

                all_val_imp_preds.append(
                    val_imp_logits[:batch_val_idx].argmax(dim=1).cpu()
                )
                all_val_imp_targets.append(
                    importance_targets[batch["author"].input_n_id[:batch_val_idx]].cpu()
                )
                all_val_role_preds.append(
                    val_role_logits[:batch_val_idx].argmax(dim=1).cpu()
                )
                all_val_role_targets.append(
                    role_targets[batch["author"].input_n_id[:batch_val_idx]].cpu()
                )
                all_val_inf_preds.append(val_inf_scores[:batch_val_idx].squeeze().cpu())
                all_val_inf_targets.append(
                    influence_targets[batch["author"].input_n_id[:batch_val_idx]].cpu()
                )

        model.train()

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_val_loss = val_total_loss / val_num_batches if val_num_batches > 0 else 1.0

        train_losses.append(avg_train_loss)

        all_val_imp_preds = torch.cat(all_val_imp_preds, dim=0)
        all_val_imp_targets = torch.cat(all_val_imp_targets, dim=0)
        all_val_role_preds = torch.cat(all_val_role_preds, dim=0)
        all_val_role_targets = torch.cat(all_val_role_targets, dim=0)
        all_val_inf_preds = torch.cat(all_val_inf_preds, dim=0)
        all_val_inf_targets = torch.cat(all_val_inf_targets, dim=0)

        val_imp_acc = accuracy_score(
            all_val_imp_targets.numpy(), all_val_imp_preds.numpy()
        )
        val_role_acc = accuracy_score(
            all_val_role_targets.numpy(), all_val_role_preds.numpy()
        )
        val_inf_mae = torch.abs(all_val_inf_preds - all_val_inf_targets).mean().item()

        val_metrics.append((val_imp_acc, val_role_acc, val_inf_mae, avg_val_loss))

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if (
            epoch % 20 == 0
            or epoch == epochs - 1
            or patience_counter == 0
            or patience_counter >= patience - 5
        ):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
            print(
                f"  Val Acc - Imp: {val_imp_acc:.4f}, Role: {val_role_acc:.4f}, Inf MAE: {val_inf_mae:.4f}"
            )

            if patience_counter > 0:
                print(f"  Early stopping: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded best model from training")
    except:
        print("Could not load best model, using final model")

    data_full = data.to(device)
    model.eval()
    with torch.no_grad():
        try:
            _, final_imp_logits, final_role_logits, final_inf_scores = model(
                data_full.x_dict, data_full.edge_index_dict
            )

            final_imp_acc = accuracy_score(
                importance_targets.cpu().numpy(),
                final_imp_logits.cpu().argmax(dim=1).numpy(),
            )
            final_role_acc = accuracy_score(
                role_targets.cpu().numpy(),
                final_role_logits.cpu().argmax(dim=1).numpy(),
            )
            final_inf_mae = (
                torch.abs(final_inf_scores.cpu().squeeze() - influence_targets.cpu())
                .mean()
                .item()
            )

            print(f"\n=== ENHANCED FINAL PERFORMANCE ===")
            print(f"Importance Accuracy: {final_imp_acc:.4f} ({final_imp_acc:.1%})")
            print(f"Role Accuracy: {final_role_acc:.4f} ({final_role_acc:.1%})")
            print(f"Influence MAE: {final_inf_mae:.4f}")
            print(f"Final Training Loss: {train_losses[-1]:.6f}")

            if final_imp_acc >= 0.8 and final_role_acc >= 0.8 and final_inf_mae <= 0.15:
                print(" EXCELLENT: Model achieved high performance across all tasks!")
            elif (
                final_imp_acc >= 0.7 and final_role_acc >= 0.7 and final_inf_mae <= 0.2
            ):
                print(" GOOD: Model performed well on primary tasks")
            else:
                print(" MODERATE: Consider tuning hyperparameters or architecture")

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
            final_imp_acc, final_role_acc, final_inf_mae = 0, 0, 1.0
            return model, train_losses, val_metrics, None, None, None
