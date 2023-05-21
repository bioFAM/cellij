from typing import Callable

import numpy as np


class EarlyStopper(object):
    """Class to manage early stopping of model training.

    Adapted from https://gist.github.com/stefanonardo
    """

    def __init__(
        self,
        mode: str = "min",
        min_delta: float = 0.0,
        patience: int = 10,
        percentage: bool = False,
    ):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs: int = 0
        self.is_better: Callable[[float, float], bool]
        self.step: Callable[..., bool]
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: 1 < 0

    def step(self, metrics: float) -> bool:  # type: ignore
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if np.isinf(metrics):
            self.num_bad_epochs += 1

        elif self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics

        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (
                    abs(best) * min_delta / 100
                )
            if mode == "max":
                self.is_better = lambda a, best: a > best + (
                    abs(best) * min_delta / 100
                )
