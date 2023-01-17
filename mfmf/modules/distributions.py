from __future__ import annotations
from numpy import isin

import pyro
import torch
import numpy as np


class Distribution:
    def __init__(self, distribution: pyro.distribution, params: dict):
        """Thin wrapper around pyro.distributions.

        The class manages the extension of a pyro.distributions object to include
        optional parameters, such as sparsity priors.

        Args:
            distribution (pyro.distribution): A likelihood distribution
            params (dict): A dict of parameters for the given distribution
        """

        if "pyro.distributions" not in distribution.__module__:
            raise TypeError(
                "Parameter 'distribution' must be come from 'pyro.distributions'."
            )

        if not isinstance(params, dict):
            raise TypeError("Parameter 'params' must be of type 'dict'.")

        self.distribution = distribution
        self.params = params

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        self._distribution = distribution

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    @property
    def likelihood(self):

        if (
            self.distribution is pyro.distributions.Normal
            or self.distribution == "normal"
        ):

            if "loc" in self.params.keys():
                if not isinstance(
                    self.params["loc"], (int, float, np.float32, torch.Tensor)
                ):
                    raise TypeError(
                        "Parameter \"params['loc']\" must be numeric or 'pyro.param'."
                    )

            elif "loc" not in self.params.keys():
                raise ValueError(
                    "Parameter 'loc' is required for 'pyro.distributions.Normal'."
                )

            if "scale" not in self.params.keys():
                raise ValueError(
                    "Parameter 'scale' is required for 'pyro.distributions.Normal'."
                )

            if not isinstance(
                self.params["scale"], (int, float, np.float32, torch.Tensor)
            ):
                raise TypeError("Parameter \"params['scale']\" must be numeric.")

            self._likelihood = pyro.distributions.Normal(
                loc=self.params["loc"], scale=self.params["scale"]
            )

        elif (
            self.distribution is pyro.distributions.LogNormal
            or self.distribution == "lognormal"
        ):

            if "loc" in self.params.keys():
                if not isinstance(self.params["loc"], (int, float, torch.Tensor)):
                    raise TypeError(
                        "Parameter \"params['loc']\" must be numeric or 'pyro.param'."
                    )

            if "loc" not in self.params.keys():
                raise ValueError(
                    "Parameter 'loc' is required for 'pyro.distributions.LogNormal'."
                )

            if "scale" not in self.params.keys():
                raise ValueError(
                    "Parameter 'scale' is required for 'pyro.distributions.LogNormal'."
                )

            if not isinstance(self.params["scale"], (int, float, torch.Tensor)):
                raise TypeError("Parameter \"params['scale']\" must be numeric.")

            self._likelihood = pyro.distributions.LogNormal(
                loc=self.params["loc"], scale=self.params["scale"]
            )

        else:
            print(self.distribution)
            raise NotImplementedError("Only normal plz")

        return self._likelihood

    @likelihood.setter
    def likelihood(self):

        raise NotImplementedError(
            "Can't set, is deducted from parameters 'distribution' and 'params'."
        )
