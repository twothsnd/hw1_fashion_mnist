from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autograd import Tensor, cross_entropy
from src.training import TrainingConfig, train_model
from src.utils import set_seed


def _make_synthetic_dataset(root: Path) -> None:
    import gzip
    import struct

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    def write_images(path: Path, images: np.ndarray) -> None:
        with gzip.open(path, "wb") as handle:
            handle.write(struct.pack(">IIII", 2051, images.shape[0], 28, 28))
            handle.write(images.astype(np.uint8).tobytes())

    def write_labels(path: Path, labels: np.ndarray) -> None:
        with gzip.open(path, "wb") as handle:
            handle.write(struct.pack(">II", 2049, labels.shape[0]))
            handle.write(labels.astype(np.uint8).tobytes())

    rng = np.random.default_rng(0)
    prototypes = np.zeros((10, 28 * 28), dtype=np.float32)
    for label in range(10):
        prototypes[label, label * 8 : label * 8 + 32] = 255.0

    def sample_split(num_examples: int) -> tuple[np.ndarray, np.ndarray]:
        labels = np.repeat(np.arange(10), math.ceil(num_examples / 10))[:num_examples]
        rng.shuffle(labels)
        images = prototypes[labels] + rng.normal(0, 15.0, size=(num_examples, 28 * 28))
        images = np.clip(images, 0, 255).reshape(num_examples, 28, 28)
        return images, labels

    train_images, train_labels = sample_split(500)
    test_images, test_labels = sample_split(200)
    write_images(data_dir / "train-images-idx3-ubyte.gz", train_images)
    write_labels(data_dir / "train-labels-idx1-ubyte.gz", train_labels)
    write_images(data_dir / "t10k-images-idx3-ubyte.gz", test_images)
    write_labels(data_dir / "t10k-labels-idx1-ubyte.gz", test_labels)


def _gradient_check() -> None:
    set_seed(1)
    logits = Tensor(np.array([[0.3, -0.1, 0.8]], dtype=np.float32), requires_grad=True)
    loss = cross_entropy(logits, np.array([2], dtype=np.int64))
    loss.backward()
    analytical = logits.grad.copy()
    numerical = np.zeros_like(logits.data)
    epsilon = 1e-3
    for index in range(logits.data.size):
        perturb = np.zeros_like(logits.data)
        perturb.reshape(-1)[index] = epsilon
        plus = cross_entropy(Tensor(logits.data + perturb), np.array([2], dtype=np.int64)).item()
        minus = cross_entropy(Tensor(logits.data - perturb), np.array([2], dtype=np.int64)).item()
        numerical.reshape(-1)[index] = (plus - minus) / (2 * epsilon)
    if not np.allclose(analytical, numerical, atol=1e-3):
        raise AssertionError(f"Gradient check failed\nanalytical={analytical}\nnumerical={numerical}")


def main() -> None:
    _gradient_check()
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _make_synthetic_dataset(root)
        config = TrainingConfig(
            data_dir=str(root / "data"),
            output_dir=str(root / "artifacts"),
            hidden_dims=(64,),
            activation="relu",
            epochs=8,
            batch_size=64,
            learning_rate=0.2,
            lr_scheduler="step",
            lr_gamma=0.5,
            lr_step_size=4,
            weight_decay=1e-4,
            val_ratio=0.1,
            seed=7,
            download=False,
        )
        result = train_model(config)
        if result["summary"]["best_val_accuracy"] < 0.95:
            raise AssertionError(f"Smoke test accuracy too low: {result['summary']['best_val_accuracy']:.4f}")
    print("smoke_test_passed")


if __name__ == "__main__":
    main()
