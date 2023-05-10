from cellij.core._factormodel import FactorModel
from cellij.core._pyro_guides import HorseshoeGuide, LassoGuide
from cellij.core._pyro_models import (
    HorseshoeGenerative,
    LassoGenerative,
    SpikeNSlabGenerative,
)


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
        if sparsity_prior == "Lasso":
            prior = LassoGenerative
            guide = LassoGuide
        elif sparsity_prior == "Horseshoe":
            prior = HorseshoeGenerative
            guide = HorseshoeGuide
        elif sparsity_prior == "SpikeNSlab":
            prior = SpikeNSlabGenerative
            guide = "AutoNormal"

        mofa_defaults = {
            "model": prior,
            "guide": guide,
        }

        kwargs = {**mofa_defaults, **kwargs}

        super(MOFA, self).__init__(n_factors=n_factors, **kwargs)
