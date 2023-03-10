from typing import Optional, Union

import numpy
import pandas
import torch
import anndata
import muon

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
        }
        # We do not type check kwargs, because we rely on type checking
        # in the FactorModel class
        if "guide" in kwargs:
            mofa_defaults["guide"] = kwargs["guide"]
            kwargs.pop("guide")
        if "trainer" in kwargs:
            mofa_defaults["trainer"] = kwargs["trainer"]
            kwargs.pop("trainer")

        super(MOFA, self).__init__(
            n_factors=n_factors,
            **mofa_defaults,
            **kwargs,
        )

    def add_data(
        self,
        data: Union[numpy.ndarray, pandas.DataFrame, torch.Tensor, anndata.AnnData, muon.MuData],
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