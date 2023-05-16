from cellij.core._factormodel import FactorModel
from cellij.core._pyro_models import (
    HorseshoeGenerative,
    # HorseshoePlusGenerative,
    LassoGenerative,
    NonnegativeGenerative,
    NormalGenerative,
    SpikeAndSlabGenerative,
    # SpikeNSlabLassoGenerative,
)


class MOFA(FactorModel):
    """Model for Multi-Omics Factor Analysis with additional sparsity priors.

    Based on:
    - Multi-Omics Factor Analysis-a framework for unsupervised integration of multi-omics data sets
        by Argelaguet, R. et al. (2018)
    - MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data
        by Argelaguet, R. et al. (2020)
    """

    def __init__(self, n_factors, sparsity_prior="SpikeNSlab", **kwargs):
        prior = NormalGenerative
        # guide = NormalGuide
        if sparsity_prior == "Lasso":
            prior = LassoGenerative
        if sparsity_prior == "Horseshoe":
            prior = HorseshoeGenerative
        if sparsity_prior == "SpikeNSlab":
            prior = SpikeAndSlabGenerative
        # if sparsity_prior == "SpikeNSlabLasso":
        #     prior = SpikeNSlabLassoGenerative
        if sparsity_prior == "Nonnegative":
            prior = NonnegativeGenerative
        # if sparsity_prior == "HorseshoePlus":
        #     prior = HorseshoePlusGenerative

        mofa_defaults = {
            "model": prior,
            # TODO: guides not implemented yet
            "guide": "AutoNormal",
        }

        kwargs = {**mofa_defaults, **kwargs}

        super(MOFA, self).__init__(n_factors=n_factors, **kwargs)
