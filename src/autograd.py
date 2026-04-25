from __future__ import annotations

from typing import Iterable

import numpy as np


ArrayLike = np.ndarray | list[float] | tuple[float, ...] | float | int


def _to_numpy(data: ArrayLike | "Tensor") -> np.ndarray:
    if isinstance(data, Tensor):
        return data.data
    return np.asarray(data, dtype=np.float32)


def _sum_to_shape(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if grad.shape == shape:
        return grad
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __array_priority__ = 100.0

    def __init__(
        self,
        data: ArrayLike | np.ndarray,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> None:
        self.data = _to_numpy(data).astype(np.float32, copy=False)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self.name = name
        self._prev: tuple[Tensor, ...] = ()
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def detach(self) -> np.ndarray:
        return self.data.copy()

    def item(self) -> float:
        return float(self.data)

    def _accumulate_grad(self, grad: np.ndarray) -> None:
        if not self.requires_grad:
            return
        grad = grad.astype(np.float32, copy=False)
        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad += grad

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

    def backward(self, grad: ArrayLike | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("backward() requires a tensor with requires_grad=True")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be provided for non-scalar tensors")
            grad_array = np.ones_like(self.data, dtype=np.float32)
        else:
            grad_array = _to_numpy(grad).astype(np.float32, copy=False)

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        self.grad = grad_array.copy()
        for node in reversed(topo):
            node._backward()

    def __add__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(_sum_to_shape(out.grad, self.shape))
            if other.requires_grad:
                other._accumulate_grad(_sum_to_shape(out.grad, other.shape))

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike | "Tensor") -> "Tensor":
        return self + other

    def __sub__(self, other: ArrayLike | "Tensor") -> "Tensor":
        return self + (-other)

    def __rsub__(self, other: ArrayLike | "Tensor") -> "Tensor":
        return other + (-self)

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(-out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(_sum_to_shape(out.grad * other.data, self.shape))
            if other.requires_grad:
                other._accumulate_grad(_sum_to_shape(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike | "Tensor") -> "Tensor":
        return self * other

    def __truediv__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(_sum_to_shape(out.grad / other.data, self.shape))
            if other.requires_grad:
                other_grad = -out.grad * self.data / (other.data ** 2)
                other._accumulate_grad(_sum_to_shape(other_grad, other.shape))

        out._backward = _backward
        return out

    def __rtruediv__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self

    def __matmul__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad @ other.data.T)
            if other.requires_grad:
                other._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            if axis is None:
                grad = np.broadcast_to(grad, self.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = tuple(ax if ax >= 0 else ax + self.ndim for ax in axes)
                if not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)
                grad = np.broadcast_to(grad, self.shape)
            self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            denominator = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            denominator = 1
            for ax in axes:
                denominator *= self.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) / denominator

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, axes: tuple[int, ...] | None = None) -> "Tensor":
        axes = axes if axes is not None else tuple(reversed(range(self.ndim)))
        out = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            inverse_axes = np.argsort(axes)
            self._accumulate_grad(out.grad.transpose(inverse_axes))

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * (self.data > 0.0))

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(sig, requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * sig * (1.0 - sig))

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        tanh_data = np.tanh(self.data)
        out = Tensor(tanh_data, requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * (1.0 - tanh_data**2))

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        exp_data = np.exp(self.data)
        out = Tensor(exp_data, requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * exp_data)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad / self.data)

        out._backward = _backward
        return out


def tensor(data: ArrayLike, requires_grad: bool = False, name: str | None = None) -> Tensor:
    return Tensor(data, requires_grad=requires_grad, name=name)


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("cross_entropy expects logits with shape (batch_size, num_classes)")
    targets = np.asarray(targets, dtype=np.int64)
    if targets.ndim != 1:
        raise ValueError("targets must have shape (batch_size,)")
    batch_size = logits.data.shape[0]
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    losses = -np.log(probabilities[np.arange(batch_size), targets] + 1e-12)
    out = Tensor(losses.mean(), requires_grad=logits.requires_grad)
    out._prev = (logits,)

    def _backward() -> None:
        if out.grad is None or not logits.requires_grad:
            return
        grad = probabilities.copy()
        grad[np.arange(batch_size), targets] -= 1.0
        grad /= batch_size
        logits._accumulate_grad(grad * float(out.grad))

    out._backward = _backward
    return out


def l2_penalty(parameters: Iterable[Tensor], strength: float) -> Tensor:
    penalty = Tensor(0.0, requires_grad=False)
    for parameter in parameters:
        penalty = penalty + 0.5 * strength * (parameter * parameter).sum()
    return penalty
