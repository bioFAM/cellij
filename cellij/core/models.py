from cellij.core._factormodel import FactorModel
from cellij.core._pyro_models import MOFA_Model


class MOFA(FactorModel):
    """Model for Multi-Omics Factor Analysis."""

    def __init__(self, n_factors, **kwargs):
        # If default variable is provided in kwargs, overwrite it
        mofa_defaults = {
            "model": MOFA_Model(n_factors=n_factors),
            "guide": "AutoNormal",
            "trainer": "Adam",
            "sparsity_prior": "spikeandslab"
        }

        kwargs = {**mofa_defaults, **kwargs}

        # # We do not type check kwargs, because we rely on type checking
        # # in the FactorModel class
        # if "guide" in kwargs:
        #     mofa_defaults["guide"] = kwargs["guide"]
        #     kwargs.pop("guide")
        # if "trainer" in kwargs:
        #     mofa_defaults["trainer"] = kwargs["trainer"]
        #     kwargs.pop("trainer")
        # if "sparsity_prior" in kwargs:
        #     mofa_defaults["sparsity_prior"] = kwargs["sparsity_prior"]
        #     kwargs.pop("sparsity_prior")

        super(MOFA, self).__init__(
            n_factors=n_factors,
            **kwargs
        )
