import os
import pickle
from typing import Optional

import numpy as np
import torch
from sklearn.impute import KNNImputer

from cellij._logging import logger


def nanstd(data, dim: Optional[int] = None):
    """Torch doesn't have a torch.nonstd() method so here it is."""
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


def load_model(filename: str):
    if not isinstance(filename, str):
        raise TypeError("Parameter 'filename' must be a string.")

    try:
        with open(filename, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError as e:
        raise e

    # Try to load the state_dict corresponding to the model
    _, file_ending = os.path.splitext(filename)
    state_dict_name = filename.replace(file_ending, ".state_dict")
    try:
        model.load_state_dict(torch.load(state_dict_name))
    except FileNotFoundError as e:
        logger.warning(f"No state_dict with name '{state_dict_name}' found, loading model without. {e}")

    return model


def impute_data(data, strategy: str, **kwargs):
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

    logger.info(f"Found {np.isnan(data.values).sum()} missing values, imputed them using '{strategy}'.")

    return result
