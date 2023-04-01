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

    def test_factormodel_populates_param_storage_with_correct_prefix(self):

        prefix = cellij.utils._get_param_storage_key_prefix()
        keys = list(pyro.get_param_store().keys())
        keys_with_prefix = [s for s in keys if s.startswith(prefix)]

        assert len(keys_with_prefix) > 0
