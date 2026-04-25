from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .autograd import Tensor, cross_entropy, l2_penalty
from .data import download_fashion_mnist, iterate_minibatches, load_fashion_mnist, train_val_split
from .metrics import accuracy_score
from .nn import MLP
from .optim import SGD, build_scheduler
from .utils import ensure_dir, save_json, set_seed, to_serializable
from .visualization import plot_training_curves


@dataclass
class TrainingConfig:
    data_dir: str
    output_dir: str
    hidden_dims: tuple[int, ...] = (128,)
    activation: str = "relu"
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 0.1
    lr_scheduler: str = "step"
    lr_gamma: float = 0.5
    lr_step_size: int = 5
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    seed: int = 42
    download: bool = True
    input_dim: int = 28 * 28
    num_classes: int = 10
    verbose: bool = True


def build_model(config: TrainingConfig) -> MLP:
    return MLP(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        num_classes=config.num_classes,
        activation=config.activation,
    )


def _weight_parameters(model: MLP):
    return [parameter for name, parameter in model.named_parameters() if name.endswith("weight")]


def _forward_pass(model: MLP, images: np.ndarray) -> Tensor:
    return model(Tensor(images.astype(np.float32), requires_grad=False))


def evaluate_split(
    model: MLP,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray | float]:
    total_loss = 0.0
    all_predictions: list[np.ndarray] = []

    for batch_images, batch_labels in iterate_minibatches(images, labels, batch_size=batch_size, shuffle=False):
        logits = _forward_pass(model, batch_images)
        loss = cross_entropy(logits, batch_labels)
        predictions = logits.data.argmax(axis=1)
        total_loss += loss.item() * len(batch_labels)
        all_predictions.append(predictions)

    predictions = np.concatenate(all_predictions)
    return {
        "loss": total_loss / len(labels),
        "accuracy": accuracy_score(predictions, labels),
        "predictions": predictions,
    }


def _run_training_epoch(
    model: MLP,
    images: np.ndarray,
    labels: np.ndarray,
    optimizer: SGD,
    batch_size: int,
    weight_decay: float,
    seed: int,
) -> dict[str, float]:
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for step, (batch_images, batch_labels) in enumerate(
        iterate_minibatches(images, labels, batch_size=batch_size, shuffle=True, seed=seed),
        start=1,
    ):
        optimizer.zero_grad()
        logits = _forward_pass(model, batch_images)
        ce_loss = cross_entropy(logits, batch_labels)
        objective = ce_loss
        if weight_decay > 0.0:
            objective = objective + l2_penalty(_weight_parameters(model), weight_decay)
        objective.backward()
        optimizer.step()

        predictions = logits.data.argmax(axis=1)
        total_loss += ce_loss.item() * len(batch_labels)
        total_correct += int((predictions == batch_labels).sum())
        total_examples += len(batch_labels)
        del step

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def _save_state_dict(path: Path, model: MLP) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **model.state_dict())


def _load_state_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as state:
        return {name: state[name] for name in state.files}


def load_model_from_artifact(artifact_dir: str | Path) -> tuple[TrainingConfig, MLP]:
    artifact_dir = Path(artifact_dir)
    config_payload = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    config_payload["hidden_dims"] = tuple(config_payload["hidden_dims"])
    config = TrainingConfig(**config_payload)
    model = build_model(config)
    model.load_state_dict(_load_state_dict(artifact_dir / "best_model.npz"))
    return config, model


def train_model(config: TrainingConfig) -> dict[str, object]:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    data_dir = ensure_dir(config.data_dir)

    if config.download:
        download_fashion_mnist(data_dir)

    train_images, train_labels, test_images, test_labels = load_fashion_mnist(data_dir)
    train_images, train_labels, val_images, val_labels = train_val_split(
        train_images,
        train_labels,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    model = build_model(config)
    optimizer = SGD(model.parameters(), lr=config.learning_rate)
    scheduler = build_scheduler(
        optimizer=optimizer,
        mode=config.lr_scheduler,
        gamma=config.lr_gamma,
        step_size=config.lr_step_size,
    )

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    best_val_accuracy = -1.0
    best_epoch = -1
    best_model_path = output_dir / "best_model.npz"

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_training_epoch(
            model=model,
            images=train_images,
            labels=train_labels,
            optimizer=optimizer,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            seed=config.seed + epoch,
        )
        val_metrics = evaluate_split(model, val_images, val_labels, batch_size=config.batch_size)
        current_lr = optimizer.lr

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_metrics["loss"]))
        history["train_accuracy"].append(float(train_metrics["accuracy"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["learning_rate"].append(float(current_lr))

        if float(val_metrics["accuracy"]) > best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_epoch = epoch
            _save_state_dict(best_model_path, model)

        if config.verbose:
            print(
                f"[epoch {epoch:02d}/{config.epochs:02d}] "
                f"lr={current_lr:.5f} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}",
                flush=True,
            )

        if scheduler is not None:
            scheduler.step(epoch)

    plot_training_curves(history, output_dir / "training_curves.png")
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "config.json", to_serializable(config))

    model.load_state_dict(_load_state_dict(best_model_path))
    test_metrics = evaluate_split(model, test_images, test_labels, batch_size=config.batch_size)
    summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": float(test_metrics["accuracy"]),
        "output_dir": str(output_dir),
        "model_path": str(best_model_path),
    }
    save_json(output_dir / "summary.json", summary)
    return {
        "config": config,
        "history": history,
        "summary": summary,
        "train_examples": len(train_labels),
        "val_examples": len(val_labels),
        "test_examples": len(test_labels),
    }
