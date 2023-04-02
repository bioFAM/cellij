import pytest
import numpy as np
import cellij
import torch

torch.manual_seed(8)
np.random.seed(8)


# Test correct number of factor correlations
def test_correct_number_of_factor_correlations():
    actuals = np.random.rand(10, 4)
    predictions = np.random.rand(10, 8)
    assert (
        len(cellij.evaluation.compute_factor_correlation(actuals, predictions)[1]) == 4
    )


def test_raise_on_nan():
    # Factor is constant, hence, during pearsonr computation we try to divide by zero, leading to nan
    actuals = np.ones((10, 4))
    predictions = np.random.rand(10, 8)

    with pytest.raises(ValueError):
        cellij.evaluation.compute_factor_correlation(actuals, predictions, replace_nan=None)


def test_replacement_of_nan():
    # Factor is constant, hence, during pearsonr computation we try to divide by zero, leading to nan
    actuals = np.ones((10, 4))
    predictions = np.random.rand(10, 8)

    assert (
        len(cellij.evaluation.compute_factor_correlation(actuals, predictions, replace_nan=0)[1]) == 4
    )
