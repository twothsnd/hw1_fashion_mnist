from __future__ import annotations

from dataclasses import dataclass

from .nn import Parameter


class SGD:
    def __init__(self, parameters: list[Parameter], lr: float = 0.1) -> None:
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            parameter.data -= self.lr * parameter.grad


@dataclass
class StepLRScheduler:
    optimizer: SGD
    step_size: int
    gamma: float

    def step(self, epoch: int) -> float:
        if self.step_size > 0 and epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
        return self.optimizer.lr


@dataclass
class ExponentialLRScheduler:
    optimizer: SGD
    gamma: float

    def step(self, epoch: int) -> float:
        del epoch
        self.optimizer.lr *= self.gamma
        return self.optimizer.lr


def build_scheduler(
    optimizer: SGD,
    mode: str,
    gamma: float,
    step_size: int,
):
    mode = mode.lower()
    if mode == "none":
        return None
    if mode == "step":
        return StepLRScheduler(optimizer=optimizer, step_size=step_size, gamma=gamma)
    if mode == "exp":
        return ExponentialLRScheduler(optimizer=optimizer, gamma=gamma)
    raise ValueError(f"Unsupported lr scheduler mode: {mode}")
