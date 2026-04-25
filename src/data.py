from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


FASHION_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

FASHION_MNIST_MIRRORS = [
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com",
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion",
]

FASHION_MNIST_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "hw1-fashion-mnist-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as handle:
        handle.write(response.read())


def download_fashion_mnist(root: str | Path, overwrite: bool = False) -> dict[str, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    local_paths: dict[str, Path] = {}
    for key, filename in FASHION_MNIST_FILES.items():
        destination = root / filename
        local_paths[key] = destination
        if destination.exists() and not overwrite:
            continue
        errors: list[str] = []
        for mirror in FASHION_MNIST_MIRRORS:
            try:
                _download_file(f"{mirror}/{filename}", destination)
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{mirror}: {exc}")
                if destination.exists():
                    destination.unlink()
        else:
            raise RuntimeError(f"Failed to download {filename}. Tried mirrors: {errors}")
    return local_paths


def _read_idx_images(path: str | Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number in {path}: {magic}")
        buffer = handle.read()
    images = np.frombuffer(buffer, dtype=np.uint8).reshape(count, rows * cols)
    return images.astype(np.float32) / 255.0


def _read_idx_labels(path: str | Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number in {path}: {magic}")
        buffer = handle.read()
    labels = np.frombuffer(buffer, dtype=np.uint8)
    if labels.shape[0] != count:
        raise ValueError(f"Label count mismatch in {path}: expected {count}, got {labels.shape[0]}")
    return labels.astype(np.int64)


def load_fashion_mnist(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(root)
    required = {key: root / filename for key, filename in FASHION_MNIST_FILES.items()}
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Fashion-MNIST files are missing. Run download_fashion_mnist() or use --download. "
            f"Missing: {missing}"
        )
    train_images = _read_idx_images(required["train_images"])
    train_labels = _read_idx_labels(required["train_labels"])
    test_images = _read_idx_images(required["test_images"])
    test_labels = _read_idx_labels(required["test_labels"])
    return train_images, train_labels, test_images, test_labels


def train_val_split(
    images: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []
    for label in np.unique(labels):
        class_indices = np.where(labels == label)[0]
        rng.shuffle(class_indices)
        val_count = max(1, int(len(class_indices) * val_ratio))
        val_indices.append(class_indices[:val_count])
        train_indices.append(class_indices[val_count:])
    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return images[train_idx], labels[train_idx], images[val_idx], labels[val_idx]


def iterate_minibatches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = np.arange(images.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield images[batch_indices], labels[batch_indices]
