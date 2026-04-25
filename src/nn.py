from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .autograd import Tensor


class Parameter(Tensor):
    def __init__(self, data: np.ndarray, name: str | None = None) -> None:
        super().__init__(data=data.astype(np.float32, copy=False), requires_grad=True, name=name)


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _iter_named_parameters(self, value, prefix: str, seen: set[int]):
        if isinstance(value, Parameter):
            if id(value) not in seen:
                seen.add(id(value))
                yield prefix, value
            return
        if isinstance(value, Module):
            for name, member in value.__dict__.items():
                if name.startswith("_"):
                    continue
                child_prefix = f"{prefix}.{name}" if prefix else name
                yield from self._iter_named_parameters(member, child_prefix, seen)
            return
        if isinstance(value, (list, tuple)):
            for index, member in enumerate(value):
                child_prefix = f"{prefix}.{index}" if prefix else str(index)
                yield from self._iter_named_parameters(member, child_prefix, seen)
            return
        if isinstance(value, dict):
            for key, member in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from self._iter_named_parameters(member, child_prefix, seen)

    def named_parameters(self):
        seen: set[int] = set()
        for name, value in self.__dict__.items():
            if name.startswith("_"):
                continue
            yield from self._iter_named_parameters(value, name, seen)

    def parameters(self) -> list[Parameter]:
        return [parameter for _, parameter in self.named_parameters()]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: parameter.detach() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        missing = []
        for name, parameter in self.named_parameters():
            if name not in state_dict:
                missing.append(name)
                continue
            value = np.asarray(state_dict[name], dtype=np.float32)
            if value.shape != parameter.data.shape:
                raise ValueError(f"Shape mismatch for {name}: expected {parameter.data.shape}, got {value.shape}")
            parameter.data[...] = value
        if missing:
            raise KeyError(f"Missing parameters in state_dict: {missing}")


def _activation(tensor: Tensor, name: str) -> Tensor:
    if name == "relu":
        return tensor.relu()
    if name == "sigmoid":
        return tensor.sigmoid()
    if name == "tanh":
        return tensor.tanh()
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class Linear(Module):
    in_features: int
    out_features: int
    init_scale: float

    def __post_init__(self) -> None:
        weight = np.random.randn(self.in_features, self.out_features).astype(np.float32) * self.init_scale
        bias = np.zeros((1, self.out_features), dtype=np.float32)
        self.weight = Parameter(weight, name="weight")
        self.bias = Parameter(bias, name="bias")

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.weight) + self.bias


class MLP(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        num_classes: int,
        activation: str = "relu",
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.num_classes = num_classes
        self.activation = activation.lower()
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer")

        self.hidden_layers: list[Linear] = []
        previous_dim = input_dim
        for hidden_dim in self.hidden_dims:
            scale = np.sqrt(2.0 / previous_dim) if self.activation == "relu" else np.sqrt(1.0 / previous_dim)
            self.hidden_layers.append(Linear(previous_dim, hidden_dim, init_scale=scale))
            previous_dim = hidden_dim
        self.output_layer = Linear(previous_dim, num_classes, init_scale=np.sqrt(1.0 / previous_dim))

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.hidden_layers:
            out = _activation(layer(out), self.activation)
        return self.output_layer(out)

