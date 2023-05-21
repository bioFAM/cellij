import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances


def compute_factor_correlation(actuals, predictions, replace_nan=0):
    """Compute correlation values for each factor.

    actuals: np.ndarray
        Array of shape (n_features, n_factors) containing the actual values.
    predictions: np.ndarray
        Array of shape (n_features, n_factors) containing the predicted values.
    replace_nan: float
        Value to replace NaNs with.
        If None, NaNs are not replaced and an error is raised if there are any.
    """
    # Because the factors might not be in the same order, we need to find the
    # permutation that maximizes the correlation values.
    correlations = pairwise_distances(
        actuals.T,
        predictions.T,
        metric=lambda a, b: pearsonr(a, b)[0],
        force_all_finite=False,
    )

    if replace_nan is not None:
        correlations = np.nan_to_num(correlations, nan=replace_nan)
    else:
        if np.isnan(correlations).any():
            raise ValueError("NaNs found in factor correlations.")

    row_idx, col_idx = linear_sum_assignment(-np.abs(correlations))

    factorwise_correlation = correlations[row_idx, col_idx]
    mean_correlation = np.abs(factorwise_correlation).mean()

    return mean_correlation, factorwise_correlation, row_idx, col_idx
