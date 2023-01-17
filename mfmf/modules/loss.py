from __future__ import annotations

import pyro


class Loss:
    """Wrapper around Pyro Inference Loss Functions."""

    def __init__(
        self,
        loss_fn: pyro.infer.elbo.ELBO,
        epochs: int = 1000,
        report_after_n_epochs: int = 100,
    ) -> None:

        self.loss_fn = loss_fn
        self.epochs = epochs
        self.report_after_n_epochs = report_after_n_epochs

    @property
    def loss_fn(self):
        return self._loss_fn

    @loss_fn.setter
    def loss_fn(self, loss_fn):
        self._loss_fn = loss_fn

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, epochs):
        if not isinstance(epochs, int):
            raise TypeError("Parameter 'epochs' must be a positive integer.")
        elif epochs <= 0:
            raise ValueError("Parameter 'epochs' must be a positive integer.")
        else:
            self._epochs = epochs

    @property
    def report_after_n_epochs(self):
        return self._report_after_n_epochs

    @report_after_n_epochs.setter
    def report_after_n_epochs(self, report_after_n_epochs):
        self._report_after_n_epochs = report_after_n_epochs


class EarlyStoppingLoss(Loss):
    """Wrapper around Pyro Loss Functions to facilitate early stopping.

    Adds an early-stopping logic to the pyro.infer.elbo.ELBO loss functions
    based on comparing the current interval of width 'report_after_n_epochs' with
    the previous one. If a number of consecutive 'max_flat_intervals'
    don't differ by at least 'min_decrease' percent, the early stop is triggered.

    Args:
        loss:
            A loss function that inherits from pyro.infer.elbo.ELBO.
        epochs:
            The maximum number of epochs that will be iterated.
        report_after_n_epochs:
            After how many epochs to print the loss and save it to the loss_dict.
        min_decrease:
            Minimal percentage that an interval n has to be smaller than the
            interval n-1 as to not count as a plateau.
        max_flat_intervals:
            The number of consecutive intervals which don't fulfill the
            'min_decrease' criteria so that the early stop is triggered.

    """

    def __init__(
        self,
        loss_fn: pyro.infer.elbo.ELBO,
        epochs: int,
        report_after_n_epochs: int = 100,
        min_decrease: float = 0.01,
        max_flat_intervals: int = 3,
    ) -> None:

        super(self.__class__, self).__init__(
            loss_fn=loss_fn,
            epochs=epochs,
            report_after_n_epochs=report_after_n_epochs,
        )
        self.min_decrease = min_decrease
        self.max_flat_intervals = max_flat_intervals

    @property
    def min_decrease(self):
        return self._min_decrease

    @min_decrease.setter
    def min_decrease(self, min_decrease):
        self._min_decrease = min_decrease

    @property
    def max_flat_intervals(self):
        return self._max_flat_intervals

    @max_flat_intervals.setter
    def max_flat_intervals(self, max_flat_intervals):
        self._max_flat_intervals = max_flat_intervals
