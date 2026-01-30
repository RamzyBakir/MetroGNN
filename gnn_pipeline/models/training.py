import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score

from gnn_pipeline.utils.visualization import plot_training_history


def create_enhanced_training_targets(data):

    num_authors = data['author'].x.shape[0]
    author_feats = data['author'].x

    importance_scores = (
        author_feats[:, 0] * 0.3 +
        author_feats[:, 1] * 0.2 +
        author_feats[:, 4] * 0.5
    )

    quantiles = torch.quantile(importance_scores, torch.tensor([0.25, 0.75]))
    importance_targets = torch.zeros(num_authors, dtype=torch.long)
    importance_targets[importance_scores <= quantiles[0]] = 0
    importance_targets[(importance_scores > quantiles[0]) & (importance_scores <= quantiles[1])] = 1
    importance_targets[importance_scores > quantiles[1]] = 2

    role_targets = torch.zeros(num_authors, dtype=torch.long)
    collab_edges = data['author', 'collaborates_with', 'author'].edge_index

    for i in range(num_authors):
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

def train_enhanced_model(model, data, epochs=200):

    print("Starting enhanced training with multi-task learning")


    importance_targets, role_targets, influence_targets = create_enhanced_training_targets(data)


    num_researchers = len(importance_targets)
    indices = torch.randperm(num_researchers)
    train_idx = indices[:int(0.7 * num_researchers)]
    val_idx = indices[int(0.7 * num_researchers):]


    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )

    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_reg = torch.nn.MSELoss()

    train_losses = []
    val_metrics = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50
    grad_clip_value = 1.0

    print(f"Training on {len(train_idx)} researchers, validating on {len(val_idx)} researchers")
    print(f"Class distribution - Importance: {torch.bincount(importance_targets).tolist()}")
    print(f"Class distribution - Roles: {torch.bincount(role_targets).tolist()}")

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()


        try:
            updated_embeddings, importance_logits, role_logits, influence_scores = model(
                data.x_dict, data.edge_index_dict
            )
        except Exception as e:
            print(f"Error in forward pass at epoch {epoch}: {e}")
            continue

        if importance_logits is not None and role_logits is not None and influence_scores is not None:

            loss_importance = criterion_class(importance_logits[train_idx], importance_targets[train_idx])
            loss_role = criterion_class(role_logits[train_idx], role_targets[train_idx])
            loss_influence = criterion_reg(influence_scores[train_idx].squeeze(), influence_targets[train_idx])


            total_loss = loss_importance * 0.5 + loss_role * 0.3 + loss_influence * 0.2


            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
            optimizer.step()


            model.eval()
            with torch.no_grad():
                try:
                    _, val_imp_logits, val_role_logits, val_inf_scores = model(
                        data.x_dict, data.edge_index_dict
                    )


                    val_imp_acc = accuracy_score(
                        importance_targets[val_idx].numpy(),
                        val_imp_logits[val_idx].argmax(dim=1).numpy()
                    )
                    val_role_acc = accuracy_score(
                        role_targets[val_idx].numpy(),
                        val_role_logits[val_idx].argmax(dim=1).numpy()
                    )
                    val_inf_mae = torch.abs(val_inf_scores[val_idx].squeeze() - influence_targets[val_idx]).mean()


                    val_loss_importance = criterion_class(val_imp_logits[val_idx], importance_targets[val_idx])
                    val_loss_role = criterion_class(val_role_logits[val_idx], role_targets[val_idx])
                    val_loss_influence = criterion_reg(val_inf_scores[val_idx].squeeze(), influence_targets[val_idx])
                    val_total_loss = val_loss_importance * 0.5 + val_loss_role * 0.3 + val_loss_influence * 0.2

                except Exception as e:
                    print(f"Error in validation at epoch {epoch}: {e}")
                    val_imp_acc, val_role_acc, val_inf_mae, val_total_loss = 0, 0, 1.0, float('inf')

            model.train()


            train_losses.append(total_loss.item())
            val_metrics.append((val_imp_acc, val_role_acc, val_inf_mae.item(), val_total_loss.item()))


            scheduler.step(val_total_loss)


            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0

                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1


            if epoch % 20 == 0 or epoch == epochs - 1 or patience_counter == 0 or patience_counter >= patience - 5:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch:03d}/{epochs}:')
                print(
                    f'  Loss: {total_loss:.4f} (Imp: {loss_importance:.4f}, Role: {loss_role:.4f}, Inf: {loss_influence:.4f})')
                print(f'  Val Loss: {val_total_loss:.4f}, LR: {current_lr:.6f}')
                print(f'  Val Acc - Imp: {val_imp_acc:.4f}, Role: {val_role_acc:.4f}, Inf MAE: {val_inf_mae:.4f}')


                if patience_counter > 0:
                    print(f'  Early stopping: {patience_counter}/{patience}')


                if len(train_losses) > 10:
                    recent_losses = train_losses[-10:]
                    loss_std = np.std(recent_losses)
                    if loss_std < 0.01:
                        print(f'Training stabilized (loss std: {loss_std:.4f})')


            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        else:
            print(f"Warning: No predictions returned at epoch {epoch}")


    try:
        model.load_state_dict(torch.load('best_model.pth'))
        print("Loaded best model from training")
    except:
        print("Could not load best model, using final model")


    model.eval()
    with torch.no_grad():
        try:
            _, final_imp_logits, final_role_logits, final_inf_scores = model(
                data.x_dict, data.edge_index_dict
            )


            final_imp_acc = accuracy_score(
                importance_targets.numpy(),
                final_imp_logits.argmax(dim=1).numpy()
            )
            final_role_acc = accuracy_score(
                role_targets.numpy(),
                final_role_logits.argmax(dim=1).numpy()
            )
            final_inf_mae = torch.abs(final_inf_scores.squeeze() - influence_targets).mean()


            print(f"\n=== ENHANCED FINAL PERFORMANCE ===")
            print(f"Importance Accuracy: {final_imp_acc:.4f} ({final_imp_acc:.1%})")
            print(f"Role Accuracy: {final_role_acc:.4f} ({final_role_acc:.1%})")
            print(f"Influence MAE: {final_inf_mae:.4f}")
            print(f"Final Training Loss: {train_losses[-1]:.6f}")


            if final_imp_acc >= 0.8 and final_role_acc >= 0.8 and final_inf_mae <= 0.15:
                print(" EXCELLENT: Model achieved high performance across all tasks!")
            elif final_imp_acc >= 0.7 and final_role_acc >= 0.7 and final_inf_mae <= 0.2:
                print(" GOOD: Model performed well on primary tasks")
            else:
                print("️ MODERATE: Consider tuning hyperparameters or architecture")


            return model, train_losses, val_metrics, final_imp_logits, final_role_logits, final_inf_scores

        except Exception as e:
            print(f"Error in final evaluation: {e}")
            final_imp_acc, final_role_acc, final_inf_mae = 0, 0, 1.0
            return model, train_losses, val_metrics, None, None, None


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
    plt.show()
    plot_training_history(train_losses, val_metrics)

    return model, train_losses, val_metrics, final_imp_logits, final_role_logits, final_inf_scores
