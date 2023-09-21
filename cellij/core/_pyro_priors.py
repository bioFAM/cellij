import logging
from typing import Any, Optional

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from torch.types import _device, _size

logger = logging.getLogger(__name__)


class PriorDist(PyroModule):
    def __init__(
        self, name: str, site_name: str, device: _device, **kwargs: dict[str, Any]
    ):
        """Instantiate a base prior distribution.

        Parameters
        ----------
        name : str
            Module name
        site_name : str
            Site name for the pyro.sample statement
        device : _device
            Torch device
        """
        super().__init__(name)
        self.site_name = site_name
        self.device = device
        self.to(self.device)

        self.sample_dict: dict[str, torch.Tensor] = {}

    def _zeros(self, size: _size) -> torch.Tensor:
        """Generate a tensor of zeros.

        Parameters
        ----------
        size : _size
            Size of the tensor

        Returns
        -------
        torch.Tensor
            Tensor of zeros
        """
        return torch.zeros(size, device=self.device)

    def _ones(self, size: _size) -> torch.Tensor:
        """Generate a tensor of ones.

        Parameters
        ----------
        size : _size
            Size of the tensor

        Returns
        -------
        torch.Tensor
            Tensor of ones
        """
        return torch.ones(size, device=self.device)

    def _const(self, value: float, size: _size = (1,)) -> torch.Tensor:
        """Generate a tensor of constant values.

        Parameters
        ----------
        value : float
            Constant value
        size : _size
            Size of the tensor

        Returns
        -------
        torch.Tensor
            Tensor of constant values
        """
        return value * self._ones(size)

    def _sample(
        self,
        site_name: str,
        dist: dist.Distribution,
        dist_kwargs: dict[str, torch.Tensor],
        constraint_fun: Optional[callable] = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Sample from a distribution.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        dist : dist.Distribution
            Distribution class
        dist_kwargs : dict[str, torch.Tensor]
            Distribution keyword arguments

        Returns
        -------
        torch.Tensor
            Sampled values
        """
        self.sample_dict[site_name] = pyro.sample(
            site_name, dist(**dist_kwargs), **kwargs
        )
        if constraint_fun:
            self.sample_dict[site_name] = constraint_fun(self.sample_dict[site_name])
        return self.sample_dict[site_name]

    def _deterministic(
        self, site_name: str, value: torch.Tensor, event_dim: Optional[int] = None
    ) -> torch.Tensor:
        """Sample deterministic values.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        value : torch.Tensor
            Value to be sampled
        event_dim : Optional[int], optional
            Event dimension, by default None

        Returns
        -------
        torch.Tensor
            Sampled values
        """
        self.sample_dict[site_name] = pyro.deterministic(site_name, value, event_dim)
        return self.sample_dict[site_name]

    def sample_global(self) -> Optional[torch.Tensor]:
        """Sample global variables.

        Returns
        -------
        Optional[torch.Tensor]
            Sampled values
        """
        return None

    def sample_inter(self) -> Optional[torch.Tensor]:
        """Sample intermediate variables, typically across the factor dimension.

        Returns
        -------
        Optional[torch.Tensor]
            Sampled values
        """
        return None

    def forward(self, *args: Any, **kwargs: dict[str, Any]) -> Optional[torch.Tensor]:
        """Sample local variables.

        Returns
        -------
        Optional[torch.Tensor]
            Sampled values
        """
        return None


class InverseGammaPrior(PriorDist):
    def __init__(self, site_name: str, device: _device, **kwargs: dict[str, Any]):
        """Instantiate an Inverse Gamma prior.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        device : _device
            Torch device
        """
        super().__init__("InverseGammaP", site_name, device)

    def forward(self, *args: Any, **kwargs: dict[str, Any]) -> Optional[torch.Tensor]:
        return self._sample(
            self.site_name,
            dist.InverseGamma,
            dist_kwargs={"concentration": self._ones((1,)), "rate": self._ones((1,))},
        )


class NormalPrior(PriorDist):
    def __init__(self, site_name: str, device: _device, **kwargs: dict[str, Any]):
        """Instantiate a Normal prior.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        device : _device
            Torch device
        """
        super().__init__("NormalP", site_name, device)

    def forward(self, *args: Any, **kwargs: dict[str, Any]) -> Optional[torch.Tensor]:
        return self._sample(
            self.site_name,
            dist.Normal,
            dist_kwargs={"loc": self._zeros((1,)), "scale": self._ones((1,))},
        )


class GaussianProcessPrior(PriorDist):
    def __init__(self, site_name: str, device: _device, **kwargs: dict[str, Any]):
        """Instantiate a Gaussian Process prior.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        device : _device
            Torch device
        """
        super().__init__("GaussianProcessP", site_name, device)
        self.gp = PseudotimeGPrior(**kwargs)

    def forward(self, *args: Any, **kwargs: dict[str, Any]) -> Optional[torch.Tensor]:
        covariate = args[0]
        return self._sample(
            self.site_name,
            self.gp.pyro_model,
            dist_kwargs={"input": covariate},
        )


class LaplacePrior(PriorDist):
    def __init__(
        self,
        site_name: str,
        device: _device,
        scale: float = 0.1,
        **kwargs: dict[str, Any],
    ):
        """Instantiate a Laplace prior.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        device : _device
            Torch device
        scale : float, optional
            Scale for the Laplace distribution, smaller leads to sparser solutions,
            by default 0.1
        """
        super().__init__("LaplaceP", site_name, device)
        self.scale = self._const(scale)

    def forward(self, *args: Any, **kwargs: dict[str, Any]) -> Optional[torch.Tensor]:
        return self._sample(
            self.site_name,
            dist.SoftLaplace,
            dist_kwargs={"loc": self._zeros((1,)), "scale": self.scale},
        )


class NonnegativePrior(PriorDist):
    def __init__(
        self,
        site_name: str,
        device: _device,
        pos_fun: callable = torch.nn.Softplus(),  # FIXME: Add beta, e.g., 20 or RELU
        **kwargs: dict[str, Any],
    ):
        """Instantiate a Non-Negativty prior.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        device : _device
            Torch device
        pos_fun: callable, optional
            Function to apply to the sampled values to enforce positive values,
            by default torch.nn.Softplus
        """
        super().__init__("NonnegativeP", site_name, device)
        self.pos_fun = pos_fun

    def forward(self, *args: Any, **kwargs: dict[str, Any]) -> Optional[torch.Tensor]:
        return self._sample(
            self.site_name,
            dist.Normal,
            dist_kwargs={"loc": self._zeros((1,)), "scale": self._ones((1,))},
            constraint_fun=self.pos_fun,
        )


class HorseshoePrior(PriorDist):
    def __init__(
        self,
        site_name: str,
        device: _device,
        tau_scale: Optional[float] = 1.0,
        tau_delta: Optional[float] = None,
        lambdas_scale: float = 1.0,
        thetas_scale: float = 1.0,
        regularized: bool = False,
        ard: bool = True,
        **kwargs: dict[str, Any],
    ):
        """Instantiate a Horseshoe prior.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        device : _device
            Torch device
        tau_scale : Optional[float], optional
            Scale of the Cauchy+ for the global scale samples,
            by default 1.0
        tau_delta : Optional[float], optional
            Deterministic scale, by default None
        lambdas_scale : float, optional
            Local scale, by default 1.0
        thetas_scale : float, optional
            Factor scale, irrelevant if ard set to False,
            by default 1.0
        regularized : bool, optional
            Whether to apply the Finnish horseshoe prior,
            by default False
        ard : bool, optional
            Whether to sparsify whole components (factors),
            by default True

        Raises
        ------
        ValueError
            If both `tau_scale` and `tau_delta` are specified
        """
        super().__init__("HorseshoeP", site_name, device)

        self.tau_site_name = self.site_name + "_tau"
        self.thetas_site_name = self.site_name + "_thetas"
        self.caux_site_name = self.site_name + "_caux"
        self.lambdas_site_name = self.site_name + "_lambdas"

        if (tau_scale is None) == (tau_delta is None):
            raise ValueError(
                "Either `tau_scale` or `tau_delta` must be specified, but not both."
            )

        if tau_scale is not None:
            self.tau_scale = self._const(tau_scale)
        if tau_delta is not None:
            self.tau_delta = self._const(tau_delta)
        self.lambdas_scale = self._const(lambdas_scale)
        self.thetas_scale = self._const(thetas_scale)
        self.regularized = regularized
        self.ard = ard

    def sample_global(self) -> Optional[torch.Tensor]:
        if hasattr(self, "tau_delta"):
            return self._deterministic(self.tau_site_name, self.tau_delta)
        return self._sample(
            self.tau_site_name, dist.HalfCauchy, dist_kwargs={"scale": self.tau_scale}
        )

    def sample_inter(self) -> Optional[torch.Tensor]:
        if not self.ard:
            return self._deterministic(self.thetas_site_name, self._ones((1,)))
        return self._sample(
            self.thetas_site_name,
            dist.HalfCauchy,
            dist_kwargs={"scale": self.thetas_scale},
        )

    def forward(self, *args: Any, **kwargs: dict[str, Any]) -> Optional[torch.Tensor]:
        lambdas_samples = self._sample(
            self.lambdas_site_name,
            dist.HalfCauchy,
            dist_kwargs={"scale": self.lambdas_scale},
        )

        lambdas_samples = (
            lambdas_samples
            * self.sample_dict[self.thetas_site_name]
            * self.sample_dict[self.tau_site_name]
        )

        if self.regularized:
            caux_samples = self._sample(
                self.caux_site_name,
                dist.InverseGamma,
                dist_kwargs={
                    "concentration": self._const(0.5),
                    "rate": self._const(0.5),
                },
            )
            lambdas_samples = (torch.sqrt(caux_samples) * lambdas_samples) / torch.sqrt(
                caux_samples + lambdas_samples**2
            )

        return self._sample(
            self.site_name,
            dist.Normal,
            dist_kwargs={"loc": self._zeros((1,)), "scale": lambdas_samples},
        )


class SpikeAndSlabPrior(PriorDist):
    def __init__(
        self,
        site_name: str,
        device: _device,
        relaxed_bernoulli: bool = True,
        temperature: float = 0.1,
        ard: bool = True,
        **kwargs: dict[str, Any],
    ):
        """Instantiate a Spike and Slab prior.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        device : _device
            Torch device
        relaxed_bernoulli : bool, optional
            Whether to use the relaxed Bernoulli
            as opposed to the continuous Bernoulli, by default True
        temperature : float, optional
            Temperature for the relaxed Bernoulli,
            approaching zero gets closer to the discrete Bernoulli, by default 0.1
        ard : bool, optional
            Whether to sparsify whole components (factors),
            by default True
        """
        super().__init__("SpikeAndSlabP", site_name, device)

        self.thetas_site_name = self.site_name + "_thetas"
        self.alphas_site_name = self.site_name + "_alphas"
        self.lambdas_site_name = self.site_name + "_lambdas"
        self.untransformed_site_name = self.site_name + "_untransformed"

        self.relaxed_bernoulli = relaxed_bernoulli
        self.temperature = temperature
        self.ard = ard

        self.lambdas_dist = dist.ContinuousBernoulli
        if self.relaxed_bernoulli:
            self.lambdas_dist = dist.RelaxedBernoulliStraightThrough

    def sample_inter(self) -> Optional[torch.Tensor]:
        if self.ard:
            self._sample(
                self.alphas_site_name,
                dist.InverseGamma,
                dist_kwargs={
                    "concentration": self._const(0.5),
                    "rate": self._const(0.5),
                },
            )
        else:
            self._deterministic(self.alphas_site_name, self._ones((1,)))

        # how can we also return alphas...
        # they are still accessible via self.sample_dict,
        # but still would be nice to return them
        return self._sample(
            self.thetas_site_name,
            dist.Beta,
            dist_kwargs={
                "concentration1": self._const(0.5),
                "concentration0": self._const(0.5),
            },
        )

    def forward(self, *args: Any, **kwargs: dict[str, Any]) -> Optional[torch.Tensor]:
        dist_kwargs = {"probs": self.sample_dict[self.thetas_site_name]}
        if self.relaxed_bernoulli:
            dist_kwargs["temperature"] = self._const(self.temperature)

        lambdas_samples = self._sample(
            self.lambdas_site_name,
            self.lambdas_dist,
            dist_kwargs=dist_kwargs,
        )

        untransformed_samples = self._sample(
            self.untransformed_site_name,
            dist.Normal,
            dist_kwargs={
                "loc": self._zeros((1,)),
                "scale": self.sample_dict[self.alphas_site_name],
            },
        )
        return self._deterministic(
            self.site_name, untransformed_samples * lambdas_samples
        )


PRIOR_MAP = {
    "InverseGamma": InverseGammaPrior,
    "Normal": NormalPrior,
    "GaussianProcess": GaussianProcessPrior,
    "Laplace": LaplacePrior,
    "Nonnegative": NonnegativePrior,
    "Horseshoe": HorseshoePrior,
    "SpikeAndSlab": SpikeAndSlabPrior,
}
