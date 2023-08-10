import os
import pickle
import random
from typing import Optional

import numpy as np
import pyro
import torch
from sklearn.impute import KNNImputer

from cellij._logging import logger

class EarlyStopper:
    """Class to manage early stopping of model training.

    Adapted from https://gist.github.com/stefanonardo.
    """

    def __init__(
        self,
        mode: str = "min",
        min_delta: float = 0.0,
        patience: int = 10,
        percentage: bool = False,
    ):
        """Initialize the EarlyStopper."""
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best: Optional[float] = None
        self.num_bad_epochs: int = 0
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda metrics: False

    def step(self, metrics: float) -> bool:
        """Determine if the training should stop."""
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

        return self.num_bad_epochs >= self.patience

    def _init_is_better(self, mode: str, min_delta: float, percentage: bool) -> None:
        """Initialize the comparator based on the mode."""
        if mode not in {"min", "max"}:
            raise ValueError(f"mode {mode} is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            delta = min_delta / 100
            if mode == "min":
                self.is_better = lambda a, best: a < best - abs(best) * delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + abs(best) * delta


def nanstd(data: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Calculate standard deviation ignoring NaN values."""
    if dim is None:
        flattened_data = data.flatten()
        mean_tensor = torch.mean(flattened_data)
        differences = flattened_data - mean_tensor
        squared_diff = torch.square(differences)
        mean_squared_diff = torch.mean(squared_diff)
        std = torch.sqrt(mean_squared_diff)
    else:
        mean_tensor = torch.mean(data, dim=dim)
        differences = data - mean_tensor.unsqueeze(dim)
        squared_diff = torch.square(differences)
        mean_squared_diff = torch.mean(squared_diff, dim=dim)
        std = torch.sqrt(mean_squared_diff)
    return std


def load_model(filename: str) -> torch.nn.Module:
    """Load a model from a file."""
    if not isinstance(filename, str):
        raise TypeError("Parameter 'filename' must be a string.")

    with open(filename, "rb") as f:
        model = pickle.load(f)

    _, file_ending = os.path.splitext(filename)
    state_dict_name = filename.replace(file_ending, ".state_dict")
    try:
        model.load_state_dict(torch.load(state_dict_name))
    except FileNotFoundError:
        logger.warning(
            f"No state_dict with name '{state_dict_name}' found, loading model without."
        )

    return model


def impute_data(data: np.ndarray, strategy: str, **kwargs) -> np.ndarray:
    """Impute missing data based on the specified strategy."""
    if strategy in ["knn", "knn_by_features", "knn_by_observations"]:
        k = kwargs.get("k", int(np.round(np.sqrt(data.shape[0]))))
        imputer = KNNImputer(n_neighbors=k)

    if strategy == "knn":
        data_imputed = imputer.fit_transform(data.values.ravel().reshape(-1, 1)).ravel()
        result = data_imputed.reshape(data.shape)

    elif strategy == "knn_by_features":
        data_imputed = imputer.fit_transform(data.values)
        result = data_imputed

    elif strategy == "knn_by_observations":
        data_imputed = imputer.fit_transform(data.T.values).T
        result = data_imputed

    elif strategy == "mean":
        mean = np.nanmean(data.values)
        result = np.where(np.isnan(data.values), mean, data.values)

    elif strategy == "mean_by_features":
        col_means = np.nanmean(data.values, axis=0)
        col_means[np.isnan(col_means)] = 0
        col_means = np.repeat(col_means[np.newaxis, :], data.values.shape[0], axis=0)
        result = np.where(np.isnan(data.values), col_means, data.values)

    elif strategy == "mean_by_observations":
        row_means = np.nanmean(data.values, axis=1)
        row_means[np.isnan(row_means)] = 0
        row_means = np.repeat(row_means[:, np.newaxis], data.values.shape[1], axis=1)
        result = np.where(np.isnan(data.values), row_means, data.values)

    else:
        raise NotImplementedError("Unknown imputation strategy %s" % strategy)

    logger.info(
        f"Found {np.isnan(data.values).sum()} missing values, imputed them using '{strategy}'."
    )

    return result


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    pyro.set_rng_seed(seed)
    random.seed(seed)

    np.random.default_rng(seed)  # Updated

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False