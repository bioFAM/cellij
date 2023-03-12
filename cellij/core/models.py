from typing import Optional, Union

import anndata
import muon
import numpy
import pandas
import torch

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

    def add_data(
        self,
        data: Union[
            numpy.ndarray, pandas.DataFrame, torch.Tensor, anndata.AnnData, muon.MuData
        ],
        name: Optional[Union[str, None]] = None,
        **kwargs,
    ):
        """Adds data to the model.

        Parameters
        ----------
        data : torch.Tensor
            The data to be added
        name : str
            The name of the data
        """

        super().add_data(data=data, name=name, **kwargs)
