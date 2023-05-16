import unittest

import numpy as np
import torch

import cellij

torch.manual_seed(8)


class core_TestClass(unittest.TestCase):
    def setUp(self):
        self.n_epochs = 2
        self.n_factors = 2
        self.mdata = cellij._data.Importer().load_CLL()["mrna"]

    def test_center_features_works(self):
        model = cellij.core.models.MOFA(n_factors=self.n_factors)
        model.add_data(data=self.mdata, name="mrna")
        np.testing.assert_almost_equal(np.round(np.mean(model.data.values), 3), 6.163)
        np.testing.assert_almost_equal(np.round(np.std(model.data.values), 3), 3.408)
        model.fit(
            likelihoods="Normal",
            epochs=self.n_epochs,
            center_features=True,
            scale_features=False,
            scale_views=False,
        )
        np.testing.assert_almost_equal(np.round(np.mean(model.data.values), 3), 0.0)
        np.testing.assert_almost_equal(np.round(np.std(model.data.values), 3), 1.342)

    def test_scale_features_works(self):
        model = cellij.core.models.MOFA(n_factors=self.n_factors)
        model.add_data(data=self.mdata, name="mrna")
        np.testing.assert_almost_equal(np.round(np.mean(model.data.values), 3), 6.163)
        np.testing.assert_almost_equal(np.round(np.std(model.data.values), 3), 3.408)
        model.fit(
            likelihoods="Normal",
            epochs=self.n_epochs,
            center_features=True,
            scale_features=True,
            scale_views=False,
        )
        np.testing.assert_almost_equal(np.round(np.mean(model.data.values), 3), 0.0)
        np.testing.assert_almost_equal(np.round(np.std(model.data.values), 3), 1.0)

    def test_scale_views_works(self):
        model = cellij.core.models.MOFA(n_factors=self.n_factors)
        model.add_data(data=self.mdata, name="mrna")
        np.testing.assert_almost_equal(np.round(np.mean(model.data.values), 3), 6.163)
        np.testing.assert_almost_equal(np.round(np.std(model.data.values), 3), 3.408)
        model.fit(
            likelihoods="Normal",
            epochs=self.n_epochs,
            center_features=False,
            scale_features=False,
            scale_views=True,
        )
        np.testing.assert_almost_equal(np.round(np.mean(model.data.values), 3), 1.808)
        np.testing.assert_almost_equal(np.round(np.std(model.data.values), 3), 1.0)
