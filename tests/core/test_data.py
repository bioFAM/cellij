import unittest

import anndata
import muon as mu
import numpy as np
import pandas as pd
import torch

import cellij

torch.manual_seed(8)
np.random.seed(8)


class core_TestClass(unittest.TestCase):
    def setUp(self):
        self.model = cellij.core.models.MOFA(n_factors=2)
        self.rng = np.random.default_rng(8)

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
        self.model.add_data(
            name="toydata", data=data, na_strategy="mean_by_observations"
        )

        assert all(self.model.data.values[:, 1] == [1, 2, 6])

    def test_na_strategy_knn(self):
        data = np.round(self.rng.random(size=(10, 10)), 1)
        data[np.where(data == 1)] = np.nan
        data = anndata.AnnData(
            pd.DataFrame(
                data=data,
                columns=[f"feat{i}" for i in range(10)],
                index=[f"obs{i}" for i in range(10)],
            )
        )
        self.model.add_data(name="toydata", data=data, na_strategy="knn")

        np.testing.assert_almost_equal(
            np.round(self.model.data.values[0, 1], 4), 0.4242
        )

    def test_na_strategy_knn_by_features(self):
        data = np.round(self.rng.random(size=(10, 10)), 1)
        data[np.where(data == 1)] = np.nan
        data = anndata.AnnData(
            pd.DataFrame(
                data=data,
                columns=[f"feat{i}" for i in range(10)],
                index=[f"obs{i}" for i in range(10)],
            )
        )
        self.model.add_data(name="toydata", data=data, na_strategy="knn_by_features")

        np.testing.assert_almost_equal(
            np.round(self.model.data.values[0, 1], 4), 0.2333
        )

    def test_na_strategy_knn_by_observations(self):
        data = np.round(self.rng.random(size=(10, 10)), 1)
        data[np.where(data == 1)] = np.nan
        data = anndata.AnnData(
            pd.DataFrame(
                data=data,
                columns=[f"feat{i}" for i in range(10)],
                index=[f"obs{i}" for i in range(10)],
            )
        )
        self.model.add_data(
            name="toydata", data=data, na_strategy="knn_by_observations"
        )

        np.testing.assert_almost_equal(
            np.round(self.model.data.values[0, 1], 4), 0.3667
        )

    def test_can_create_Importer(self):
        imp = cellij.core.Importer()

        assert isinstance(imp, cellij.core.Importer)

    def test_Importer_can_load_CLL_data(self):
        imp = cellij.core.Importer()
        data = imp.load_CLL()

        assert isinstance(data, mu.MuData)
        assert data.n_obs == 200
        assert data.n_vars == 9627

    def test_Importer_can_load_MEFISTO_data(self):
        imp = cellij.core.Importer()
        data = imp.load_MEFISTO()

        assert isinstance(data, mu.MuData)
        assert data.n_obs == 200
        assert data.n_vars == 800

    def test_Importer_can_load_Guo2010_data(self):
        imp = cellij.core.Importer()
        data = imp.load_Guo2010()

        assert isinstance(data, mu.MuData)
        assert data.n_obs == 437
        assert data.n_vars == 48
        assert all(
            column in data.obs.columns
            for column in ["n_cells", "division", "division_scaled", "label"]
        )

    def test_obsnames_are_not_sorted_on_change(self):
        """Checks that indicies are not alphabetically sorted.

        The previous sorting logic would result in indicies like ["0", "1", "10", "2"],
        which could be confusing to the user. So now this order is retained so that it
        looks like ["0", "1", "2", "10"].
        """
        n_samples = 11
        n_features = [1, 1, 1]
        ad_dict = {}
        for offset, nf in enumerate(n_features):
            arr = np.tile(np.array(list(range(n_samples))), (nf, 1)) + offset * 10
            adata = anndata.AnnData(arr.T, dtype=np.float32)
            adata.var_names = [f"{offset}_{name}" for name in adata.var_names]
            ad_dict[f"view_{offset}"] = adata

        mdata = mu.MuData(ad_dict)
        model = cellij.core.models.MOFA(n_factors=5)
        model.add_data(data=mdata, na_strategy=None)

        # verify that the last element hasn't been moved to the front
        assert all(model._data.values[:,0] == list(range(n_samples)))
