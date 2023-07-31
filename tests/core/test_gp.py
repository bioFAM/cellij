import unittest

import anndata
import muon as mu
import numpy as np
import pandas as pd
import torch

import cellij

SEED = 123

torch.manual_seed(SEED)
np.random.seed(SEED)


class core_TestClass(unittest.TestCase):
    def setUp(self):
        self.model = cellij.core.models.SimpleGP(n_factors=2)
        self.rng = np.random.default_rng(SEED)
        self.guo2010 = cellij.core.Importer().load_Guo2010()

    def test_can_run_1D_gp(self):
        self.model.add_data(data=self.guo2010)
        self.model.add_covariate(self.guo2010.obs[["division_scaled"]])
        self.model.fit(
            likelihoods="Normal",
            epochs=2,
            verbose_epochs=1,
            learning_rate=0.01,
        )

        factor_means = []
        with torch.no_grad():
            dist = self.model.gp(self.model.covariate)
            samples = dist(torch.Size([10])).cpu()
            mean = samples.mean(dim=0)
            mean = torch.transpose(mean, -1, -2)
            factor_means.append(torch.transpose(mean, -1, -2))

        factor_means = torch.stack(factor_means, dim=0)

        assert round(self.model.train_loss_elbo[-1], 3) == 2.592
