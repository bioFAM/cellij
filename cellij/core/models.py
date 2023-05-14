from cellij.core._factormodel import FactorModel
from cellij.core._pyro_guides import (
    HorseshoeGuide,
    LassoGuide,
    NonnegativityGuide,
)
from cellij.core._pyro_models import (
    HorseshoeGenerative,
    LassoGenerative,
    NonnegativityGenerative,
    NormalGenerative,
    SpikeNSlabGenerative,
    SpikeNSlabLassoGenerative,
)


class MOFA(FactorModel):
    """Model for Multi-Omics Factor Analysis with additional sparsity priors.

    Based on:
    - Multi-Omics Factor Analysis-a framework for unsupervised integration of multi-omics data sets
        by Argelaguet, R. et al. (2018)
    - MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data
        by Argelaguet, R. et al. (2020)
    """

    def __init__(self, n_factors, sparsity_prior="Spikeandslab-Beta", **kwargs):
        if sparsity_prior == "Lasso":
            prior = LassoGenerative
            guide = LassoGuide
        elif sparsity_prior == "Horseshoe":
            prior = HorseshoeGenerative
            guide = HorseshoeGuide
        elif sparsity_prior == "SpikeNSlab":
            prior = SpikeNSlabGenerative
            guide = "AutoNormal"
        elif sparsity_prior == "SpikeNSlabLasso":
            prior = SpikeNSlabLassoGenerative
            guide = "AutoNormal"
        elif sparsity_prior == "Nonnegativity":
            prior = NonnegativityGenerative
            guide = NonnegativityGuide
        elif sparsity_prior is None:
            prior = NormalGenerative
            guide = "AutoNormal"

        mofa_defaults = {
            "model": prior,
            "guide": guide,
        }

        kwargs = {**mofa_defaults, **kwargs}

        super(MOFA, self).__init__(n_factors=n_factors, **kwargs)
