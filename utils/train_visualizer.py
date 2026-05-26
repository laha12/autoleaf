from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix


def plot_training_curves(stage_csv_paths, output_png, title):
    frames = []
    epoch_offset = 0
    for csv_path in stage_csv_paths:
        p = Path(csv_path)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        df["global_epoch"] = df["epoch"] + epoch_offset
        epoch_offset = int(df["global_epoch"].max())
        frames.append(df)
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(merged["global_epoch"], merged["train_loss"], label="train_loss", color="#1f77b4", linewidth=2)
    ax1.plot(merged["global_epoch"], merged["val_loss"], label="val_loss", color="#ff7f0e", linewidth=2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False)

    ax2.plot(merged["global_epoch"], merged["train_acc"], label="train_acc", color="#1f77b4", linewidth=2)
    ax2.plot(merged["global_epoch"], merged["val_acc"], label="val_acc", color="#ff7f0e", linewidth=2)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False)

    ax3.plot(merged["global_epoch"], merged["lr"], color="#2ca02c", linewidth=2)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("learning rate")
    ax3.set_title("Learning Rate")
    ax3.grid(alpha=0.3)

    ax4.axis("off")
    best_idx = merged["val_acc"].idxmax()
    best_epoch = int(merged.loc[best_idx, "global_epoch"])
    best_val_acc = float(merged.loc[best_idx, "val_acc"])
    summary = f"Best ValAcc: {best_val_acc:.2f}%\nBest Epoch: {best_epoch}\nTotal Epoch: {int(merged['global_epoch'].max())}"
    ax4.text(0.05, 0.6, summary, fontsize=12)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(model, data_loader, class_names, device, output_png):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    cm = cm.astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    cm_norm = cm / row_sum

    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalized Confusion Matrix")
    if len(class_names) <= 30:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
