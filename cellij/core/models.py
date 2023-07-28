from cellij.core._factormodel import FactorModel
# from cellij.core._pyro_guides import HorseshoeGuide, NormalGuide
# from cellij.core._pyro_models import (
#     # HorseshoeStandaloneGenerative,
#     HorseshoeGenerative,
#     LassoGenerative,
#     # NonnegativeGenerative,
#     NormalGenerative,
#     SpikeAndSlabGenerative,
#     SpikeAndSlabLassoGenerative,
# )


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
        guide = "AutoNormal"
        if sparsity_prior == "Lasso":
            prior = LassoGenerative
            guide = NormalGuide
        if sparsity_prior == "HorseshoeStandalone":
            prior = HorseshoeStandaloneGenerative
            guide = NormalGuide
        if sparsity_prior == "Horseshoe":
            prior = HorseshoeGenerative
            guide = HorseshoeGuide
        if sparsity_prior == "SpikeAndSlab":
            prior = SpikeAndSlabGenerative
        if sparsity_prior == "SpikeAndSlabLasso":
            prior = SpikeAndSlabLassoGenerative
        if sparsity_prior == "Nonnegative":
            prior = NonnegativeGenerative
        # if sparsity_prior == "HorseshoePlus":
        #     prior = HorseshoePlusGenerative

        mofa_defaults = {
            "model": prior,
            "guide": guide,
        }

        kwargs = {**mofa_defaults, **kwargs}

        super(MOFA, self).__init__(n_factors=n_factors, **kwargs)


class MOFAPLUS(FactorModel):
    """Model for Multi-Omics Factor Analysis with additional sparsity priors.

    Based on:
    - Multi-Omics Factor Analysis-a framework for unsupervised integration of multi-omics data sets
        by Argelaguet, R. et al. (2018)
    - MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data
        by Argelaguet, R. et al. (2020)
    """

    def __init__(self, n_factors, sparsity_prior="SpikeNSlab", **kwargs):
        prior = NormalGenerative
        guide = "AutoNormal"
        if sparsity_prior == "Lasso":
            prior = LassoGenerative
            guide = NormalGuide
        if sparsity_prior == "HorseshoeStandalone":
            prior = HorseshoeStandaloneGenerative
            guide = NormalGuide
        if sparsity_prior == "Horseshoe":
            prior = HorseshoeGenerative
            guide = HorseshoeGuide
        if sparsity_prior == "SpikeAndSlab":
            prior = SpikeAndSlabGenerative
        if sparsity_prior == "SpikeAndSlabLasso":
            prior = SpikeAndSlabLassoGenerative
        if sparsity_prior == "Nonnegative":
            prior = NonnegativeGenerative
        # if sparsity_prior == "HorseshoePlus":
        #     prior = HorseshoePlusGenerative

        mofa_defaults = {
            "model": prior,
            "guide": guide,
        }

        kwargs = {**mofa_defaults, **kwargs}

        super(MOFA, self).__init__(n_factors=n_factors, **kwargs)
