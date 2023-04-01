import unittest

import pyro
import cellij
import torch

torch.manual_seed(8)


class tools_inspect_TestClass(unittest.TestCase):
    def setUp(self):

        self.n_epochs = 2
        self.n_factors = 2
        self.mdata = cellij._data.Importer().load_CLL()
        self.model = cellij.core.models.MOFA(n_factors=self.n_factors)
        self.model.add_data(data=self.mdata, na_strategy="knn_by_obs")
        self.model.fit(likelihood="Normal", epochs=self.n_epochs)

        assert self.model.is_trained

    def test_can_get_model_params_from_param_storage(self):
        # Using Normal distribution, so we expect [loc, scale] for [w, z]

        keys = cellij.utils._get_keys_from_model(self.model)

        assert "FactorModel._guide.locs.w" in keys
        assert "FactorModel._guide.locs.z" in keys
        assert "FactorModel._guide.scales.w" in keys
        assert "FactorModel._guide.scales.z" in keys
