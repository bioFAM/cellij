import unittest

import cellij
import torch

torch.manual_seed(8)


class core_TestClass(unittest.TestCase):
    def setUp(self):
        self.n_epochs = 2
        self.n_factors = 2
        self.mdata = cellij._data.Importer().load_CLL()

    def test_model_is_untrained_before_fit_function(self):

        model = cellij.core.models.MOFA(n_factors=self.n_factors)
        model.add_data(data=self.mdata, na_strategy="knn_by_obs")

        assert model.is_trained == False

    def test_model_is_trained_before_fit_function(self):

        model = cellij.core.models.MOFA(n_factors=self.n_factors)
        model.add_data(data=self.mdata, na_strategy="knn_by_obs")
        model.fit(likelihood="Normal", epochs=self.n_epochs)

        assert model.is_trained == True
