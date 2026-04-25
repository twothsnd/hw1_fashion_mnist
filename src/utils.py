from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_hidden_dims(raw_value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not dims:
        raise ValueError("hidden_dims must contain at least one integer")
    return dims
