from cellij.core._factormodel import FactorModel
from cellij.core._pyro_models import MOFA_Model


class MOFA(FactorModel):
    """Model for Multi-Omics Factor Analysis."""

    def __init__(self, n_factors, sparsity_prior="spikeandslab", **kwargs):
        # If default variable is provided in kwargs, overwrite it
        mofa_defaults = {
            "model": MOFA_Model(n_factors=n_factors, sparsity_prior=sparsity_prior),
            "guide": "AutoNormal",
            "trainer": "Adam",
        }

        kwargs = {**mofa_defaults, **kwargs}

        super(MOFA, self).__init__(n_factors=n_factors, **kwargs)
