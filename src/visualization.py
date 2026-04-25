from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hw1_fashion_mnist_mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss Curves")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["val_accuracy"], label="Validation Accuracy", linewidth=2, color="tab:green")
    axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy", linewidth=2, color="tab:orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() * 0.6 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_first_layer_weights(
    weight_matrix: np.ndarray,
    output_path: str | Path,
    max_units: int = 64,
) -> None:
    if weight_matrix.ndim != 2 or weight_matrix.shape[0] != 28 * 28:
        raise ValueError("Expected first-layer weights with shape (784, hidden_dim)")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unit_count = min(weight_matrix.shape[1], max_units)
    grid_size = math.ceil(math.sqrt(unit_count))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.0, grid_size * 2.0))
    axes = np.asarray(axes).reshape(-1)

    for axis in axes:
        axis.axis("off")

    for index in range(unit_count):
        axes[index].imshow(weight_matrix[:, index].reshape(28, 28), cmap="coolwarm")
        axes[index].set_title(f"Unit {index}", fontsize=8)
        axes[index].axis("off")

    fig.suptitle("First Hidden Layer Weights", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_examples(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    max_examples: int = 16,
) -> None:
    mistakes = np.where(predictions != labels)[0]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mistakes.size == 0:
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.text(0.5, 0.5, "No misclassified examples found.", ha="center", va="center", fontsize=14)
        axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    example_count = min(mistakes.size, max_examples)
    grid_size = math.ceil(math.sqrt(example_count))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.5, grid_size * 2.5))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")

    for plot_index, image_index in enumerate(mistakes[:example_count]):
        axes[plot_index].imshow(images[image_index].reshape(28, 28), cmap="gray")
        axes[plot_index].set_title(
            f"T: {class_names[int(labels[image_index])]}\nP: {class_names[int(predictions[image_index])]}",
            fontsize=8,
        )
        axes[plot_index].axis("off")

    fig.suptitle("Misclassified Test Examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
