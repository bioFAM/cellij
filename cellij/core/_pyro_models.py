import logging
from typing import Any, Dict, Optional

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from torch.types import _device, _size

from cellij.core._gp import PseudotimeGP

logger = logging.getLogger(__name__)


class PDist(PyroModule):
    def __init__(
        self, name: str, site_name: str, device: _device, **kwargs: Dict[str, Any]
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

        self.sample_dict: Dict[str, torch.Tensor] = {}

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
        dist_kwargs: Dict[str, torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Sample from a distribution.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        dist : dist.Distribution
            Distribution class
        dist_kwargs : Dict[str, torch.Tensor]
            Distribution keyword arguments

        Returns
        -------
        torch.Tensor
            Sampled values
        """
        self.sample_dict[site_name] = pyro.sample(
            site_name, dist(**dist_kwargs), **kwargs
        )
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

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Sample local variables.

        Returns
        -------
        Optional[torch.Tensor]
            Sampled values
        """
        return None


class InverseGammaP(PDist):
    def __init__(self, site_name: str, device: _device, **kwargs: Dict[str, Any]):
        """Instantiate an Inverse Gamma prior.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        device : _device
            Torch device
        """
        super().__init__("InverseGammaP", site_name, device)

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        return self._sample(
            self.site_name,
            dist.InverseGamma,
            dist_kwargs={"concentration": self._ones((1,)), "rate": self._ones((1,))},
        )


class NormalP(PDist):
    def __init__(self, site_name: str, device: _device, **kwargs: Dict[str, Any]):
        """Instantiate a Normal prior.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        device : _device
            Torch device
        """
        super().__init__("NormalP", site_name, device)

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        return self._sample(
            self.site_name,
            dist.Normal,
            dist_kwargs={"loc": self._zeros((1,)), "scale": self._ones((1,))},
        )


class GaussianProcessP(PDist):
    def __init__(self, site_name: str, device: _device, **kwargs: Dict[str, Any]):
        """Instantiate a Gaussian Process prior.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        device : _device
            Torch device
        """
        super().__init__("GaussianProcessP", site_name, device)
        self.gp = PseudotimeGP(**kwargs)

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        covariate = args[0]
        return self._sample(
            self.site_name,
            self.gp.pyro_model,
            dist_kwargs={"input": covariate},
        )


class LaplaceP(PDist):
    def __init__(
        self,
        site_name: str,
        device: _device,
        scale: float = 0.1,
        **kwargs: Dict[str, Any],
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

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        return self._sample(
            self.site_name,
            dist.SoftLaplace,
            dist_kwargs={"loc": self._zeros((1,)), "scale": self.scale},
        )


class HorseshoeP(PDist):
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
        **kwargs: Dict[str, Any],
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

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
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


class SpikeAndSlabP(PDist):
    def __init__(
        self,
        site_name: str,
        device: _device,
        relaxed_bernoulli: bool = True,
        temperature: float = 0.1,
        ard: bool = True,
        **kwargs: Dict[str, Any],
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

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
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


class Generative(PyroModule):
    def __init__(
        self,
        n_factors: int,
        obs_dict: Dict[str, int],
        feature_dict: Dict[str, int],
        likelihoods: Dict[str, str],
        factor_priors: Dict[str, str],
        weight_priors: Dict[str, str],
        device: _device,
    ):
        """Instantiate a generative model for the multi-group and multi-view FA.

        Parameters
        ----------
        n_factors : int
            Number of factors
        obs_dict : Dict[str, int]
            Dictionary of observations per group
        feature_dict : Dict[str, int]
            Dictionary of features per view
        likelihoods : Dict[str, str]
            Likelihoods per view
        factor_priors : Dict[str, str]
            Prior distributions for the factor scores
        weight_priors : Dict[str, str]
            Prior distributions for the factor loadings
        device : _device
            Torch device
        """
        super().__init__("Generative")
        self.n_samples = sum(obs_dict.values())
        self.n_factors = n_factors
        self.obs_dict = obs_dict
        self.feature_dict = feature_dict
        self.n_feature_groups = len(feature_dict)
        self.n_obs_groups = len(obs_dict)
        self.likelihoods = likelihoods

        self.device = device
        self.to(self.device)

        self.sigma_priors = self._get_priors(
            {k: "InverseGamma" for k in weight_priors}, "sigma"
        )
        self.factor_priors = self._get_priors(factor_priors, "z")
        self.weight_priors = self._get_priors(weight_priors, "w")

        self.sample_dict: Dict[str, torch.Tensor] = {}

    def _get_priors(
        self, priors: Dict[str, str], site_name: str, **kwargs: Dict[str, Any]
    ) -> Dict[str, PDist]:
        prior_map = {
            "InverseGamma": InverseGammaP,
            "Normal": NormalP,
            "GaussianProcess": GaussianProcessP,
            "Laplace": LaplaceP,
            "Horseshoe": HorseshoeP,
            "SpikeAndSlab": SpikeAndSlabP,
        }
        _priors = {}

        for group, prior_config in priors.items():
            # Replace prior config with actuals priors
            prior = None
            prior_kwargs = {}
            if isinstance(prior_config, str):
                prior = prior_map[prior_config]
            if isinstance(prior_config, dict):
                if "name" not in prior_config:
                    raise ValueError(
                        f"Prior `{prior_config}` must contain a `name` key."
                    )
                try:
                    prior = prior_map[prior_config["name"]]
                except KeyError:
                    logger.warning(
                        f"Prior `{prior_config['name']}` is not supported. "
                        "Using `Normal` prior instead."
                    )
                    prior = prior_map["Normal"]
                prior_kwargs = prior_config
            if prior is None:
                raise ValueError(
                    f"Prior `{prior_config}` is not supported, "
                    "please provide a string or a dictionary."
                )
            _priors[group] = prior(
                site_name=f"{site_name}_{group}", device=self.device, **prior_kwargs
            )

        return _priors

    def get_plates(self) -> Dict[str, pyro.plate]:
        plates = {
            "factor": pyro.plate("factor", self.n_factors, dim=-2),
        }
        for obs_group, n_obs in self.obs_dict.items():
            plates[f"obs_{obs_group}"] = pyro.plate(obs_group, n_obs, dim=-3)
        for feature_group, n_features in self.feature_dict.items():
            plates[f"feature_{feature_group}"] = pyro.plate(
                feature_group, n_features, dim=-1
            )

        return plates

    def forward(
        self,
        data: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        covariate: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        plates = self.get_plates()

        for obs_group, factor_prior in self.factor_priors.items():
            factor_prior.sample_global()
            with plates["factor"]:
                factor_prior.sample_inter()
                with plates[f"obs_{obs_group}"]:
                    self.sample_dict[f"z_{obs_group}"] = factor_prior(covariate)

        for feature_group, weight_prior in self.weight_priors.items():
            weight_prior.sample_global()
            with plates["factor"]:
                weight_prior.sample_inter()
                with plates[f"feature_{feature_group}"]:
                    self.sample_dict[f"w_{feature_group}"] = weight_prior()

            with plates[f"feature_{feature_group}"]:
                self.sample_dict[f"sigma_{feature_group}"] = self.sigma_priors[
                    feature_group
                ]()

        for obs_group, n_obs in self.obs_dict.items():
            for feature_group, n_features in self.feature_dict.items():
                with plates[f"obs_{obs_group}"], plates[f"feature_{feature_group}"]:
                    z_shape = (-1, n_obs, self.n_factors, 1)
                    w_shape = (-1, 1, self.n_factors, n_features)
                    sigma_shape = (-1, 1, 1, n_features)
                    obs_shape = (-1, n_obs, 1, n_features)

                    obs = None
                    if data is not None:
                        obs = data[obs_group][feature_group].view(obs_shape)

                    z = self.sample_dict[f"z_{obs_group}"].view(z_shape)
                    w = self.sample_dict[f"w_{feature_group}"].view(w_shape)

                    loc = torch.einsum("...ikj,...ikj->...ij", z, w).view(obs_shape)

                    scale = torch.sqrt(self.sample_dict[f"sigma_{feature_group}"]).view(
                        sigma_shape
                    )

                    site_name = f"x_{obs_group}_{feature_group}"
                    obs_mask = torch.ones_like(loc, dtype=torch.bool)
                    if obs is not None:
                        obs_mask = torch.logical_not(torch.isnan(obs))
                    with pyro.poutine.mask(mask=obs_mask):
                        if obs is not None:
                            obs = torch.nan_to_num(obs, nan=0.0)

                        self.sample_dict[site_name] = pyro.sample(
                            site_name,
                            dist.Normal(loc, scale),
                            obs=obs,
                            infer={"is_auxiliary": True},
                        )

        return self.sample_dict
