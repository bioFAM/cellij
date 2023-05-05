import unittest

import anndata
import numpy as np
import pandas as pd
import torch

import cellij

torch.manual_seed(8)
np.random.seed(8)


class core_TestClass(unittest.TestCase):
    def setUp(self):
        self.model = cellij.core.models.MOFA(n_factors=2)

    def test_na_strategy_mean(self):
        data = anndata.AnnData(
            pd.DataFrame(
                data=np.array([1, 2, 6, np.nan, np.nan, np.nan, 1, 2, 6]).reshape(3, 3),
                columns=["feat1", "feat2", "feat3"],
                index=["obs1", "obs2", "obs3"],
            )
        )
        self.model.add_data(name="toydata", data=data, na_strategy="mean")

        assert all(self.model.data.values[1, :] == [3, 3, 3])

    def test_na_strategy_mean_by_features(self):
        data = anndata.AnnData(
            pd.DataFrame(
                data=np.array([1, 2, 6, np.nan, np.nan, np.nan, 1, 2, 6]).reshape(3, 3),
                columns=["feat1", "feat2", "feat3"],
                index=["obs1", "obs2", "obs3"],
            )
        )
        self.model.add_data(name="toydata", data=data, na_strategy="mean_by_features")

        assert all(self.model.data.values[1, :] == [1, 2, 6])

    def test_na_strategy_mean_by_observations(self):
        data = anndata.AnnData(
            pd.DataFrame(
                data=np.array([1, np.nan, 1, 2, np.nan, 2, 6, np.nan, 6]).reshape(3, 3),
                columns=["feat1", "feat2", "feat3"],
                index=["obs1", "obs2", "obs3"],
            )
        )
        self.model.add_data(name="toydata", data=data, na_strategy="mean_by_observations")

        assert all(self.model.data.values[:, 1] == [1, 2, 6])

    def test_na_strategy_knn(self):
        data = np.round(np.random.randn(100).reshape(10, 10))
        data[np.where(data > 2)] = np.nan
        data = anndata.AnnData(
            pd.DataFrame(data=data, columns=[f"feat{i}" for i in range(10)], index=[f"obs{i}" for i in range(10)])
        )
        self.model.add_data(name="toydata", data=data, na_strategy="knn")

        np.testing.assert_almost_equal(np.round(self.model.data.values[9, 4], 4), 0.0101)

    def test_na_strategy_knn_by_features(self):
        data = np.round(np.random.randn(100).reshape(10, 10))
        data[np.where(data > 2)] = np.nan
        data = anndata.AnnData(
            pd.DataFrame(data=data, columns=[f"feat{i}" for i in range(10)], index=[f"obs{i}" for i in range(10)])
        )
        self.model.add_data(name="toydata", data=data, na_strategy="knn_by_features")

        np.testing.assert_almost_equal(np.round(self.model.data.values[9, 4], 4), -1.3333)

    def test_na_strategy_knn_by_observations(self):
        data = np.round(np.random.randn(100).reshape(10, 10))
        data[np.where(data > 2)] = np.nan
        data = anndata.AnnData(
            pd.DataFrame(data=data, columns=[f"feat{i}" for i in range(10)], index=[f"obs{i}" for i in range(10)])
        )
        self.model.add_data(name="toydata", data=data, na_strategy="knn_by_observations")

        np.testing.assert_almost_equal(np.round(self.model.data.values[9, 4], 4), -0.6667)
