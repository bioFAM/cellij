from cellij.core._factormodel import FactorModel

# from cellij.core._pyro_models import MOFA_Model


class MOFA(FactorModel):
    """Model for Multi-Omics Factor Analysis.

    Based on:
    - Multi-Omics Factor Analysis-a framework for unsupervised integration of multi-omics data sets
        by Argelaguet, R. et al. (2018)
    - MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data
        by Argelaguet, R. et al. (2020)
    """

    def __init__(self, n_factors, sparsity_prior="Spikeandslab-Beta", **kwargs):
        # If default variable is provided in kwargs, overwrite it
        mofa_defaults = {
            "model": None,
            # "model": MOFA_Model(n_factors=n_factors, sparsity_prior=sparsity_prior),
            "guide": "AutoNormal",
            "trainer": "Adam",
        }

        kwargs = {**mofa_defaults, **kwargs}

        super(MOFA, self).__init__(n_factors=n_factors, **kwargs)
