import logging
from typing import Any, Dict, Optional, Tuple

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.infer.autoguide.guides import deep_getattr, deep_setattr
from pyro.nn import PyroModule, PyroParam
from torch.types import _size

from cellij.core._pyro_models import Generative
from cellij.core._priors import PriorDist

logger = logging.getLogger(__name__)


class QDist(PyroModule):
    def __init__(
        self, name: str, prior: PriorDist, init_loc: float = 0.0, init_scale: float = 0.1
    ):
        """Instantiate a base class for a variational distribution.

        Parameters
        ----------
        name : str
            Module name
        prior : PriorDist
            Prior distribution
        init_loc : float, optional
            Initial value for the loc (mean) of the distribution, by default 0.0
        init_scale : float, optional
            Initial value for the scale (std) of the distribution, by default 0.1
        """
        super().__init__(name)

        self.locs = PyroModule()
        self.scales = PyroModule()

        self.prior = prior
        self.init_loc = init_loc
        self.init_scale = init_scale
        self.device = prior.device
        self.to(self.device)

        self.setup_sites()

        self.sample_dict: dict[str, torch.Tensor] = {}

    def _set_loc_and_scale(self, site_name: str, site_shape: _size) -> None:
        """Initialize loc and scale parameters.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        site_shape : _size
            Shape of the site
        """
        deep_setattr(
            self.locs,
            site_name,
            PyroParam(
                self.init_loc * torch.ones(site_shape, device=self.device),
                constraints.real,
            ),
        )
        deep_setattr(
            self.scales,
            site_name,
            PyroParam(
                self.init_scale * torch.ones(site_shape, device=self.device),
                constraints.softplus_positive,
            ),
        )

    def _get_loc_and_scale(self, site_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get loc and scale parameters.

        Parameters
        ----------
        site_name : str
            Name of the site

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            loc and scale parameters
        """
        site_loc = deep_getattr(self.locs, site_name)
        site_scale = deep_getattr(self.scales, site_name)
        return (site_loc, site_scale)

    def _sample(self, site_name: str, dist: dist.Distribution) -> torch.Tensor:
        """Sample from a variational distribution.

        Parameters
        ----------
        site_name : str
            Site name for the pyro.sample statement
        dist : dist.Distribution
            Distribution class

        Returns
        -------
        torch.Tensor
            Sampled values
        """
        loc, scale = self._get_loc_and_scale(site_name)
        self.sample_dict[site_name] = pyro.sample(site_name, dist(loc, scale))
        return self.sample_dict[site_name]

    def _sample_normal(self, site_name: str) -> torch.Tensor:
        return self._sample(site_name, dist.Normal)

    def _sample_log_normal(self, site_name: str) -> torch.Tensor:
        return self._sample(site_name, dist.LogNormal)

    def setup_sites(self) -> None:
        """Set up the sample sites."""
        for site_name, site_samples in self.prior.sample_dict.items():
            self._set_loc_and_scale(site_name, site_samples.shape)

    def _mean_normal(self, site_name: str) -> torch.Tensor:
        loc, _ = self._get_loc_and_scale(site_name)
        return loc

    def _median_normal(self, site_name: str) -> torch.Tensor:
        return self._mean_normal(site_name)

    def _mode_normal(self, site_name: str) -> torch.Tensor:
        return self._mean_normal(site_name)

    def _mean_log_normal(self, site_name: str) -> torch.Tensor:
        loc, scale = self._get_loc_and_scale(site_name)
        return (loc + scale.pow(2) / 2).exp()

    def _median_log_normal(self, site_name: str) -> torch.Tensor:
        loc, _ = self._get_loc_and_scale(site_name)
        return loc.exp()

    def _mode_log_normal(self, site_name: str) -> torch.Tensor:
        loc, scale = self._get_loc_and_scale(site_name)
        return (loc - scale.square()).exp()

    @torch.no_grad()
    def mean(self) -> torch.Tensor:
        """Get the mean of the variational distribution.

        Returns
        -------
        torch.Tensor
            Mean of the variational distribution
        """
        return self._mean_normal(self.prior.site_name)

    @torch.no_grad()
    def median(self) -> torch.Tensor:
        """Get the median of the variational distribution.

        Returns
        -------
        torch.Tensor
            Median of the variational distribution
        """
        return self._median_normal(self.prior.site_name)

    @torch.no_grad()
    def mode(self) -> torch.Tensor:
        """Get the mode of the variational distribution.

        Returns
        -------
        torch.Tensor
            Mode of the variational distribution
        """
        return self._mode_normal(self.prior.site_name)

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


class InverseGammaQ(QDist):
    def __init__(self, prior: PriorDist, init_loc: float = 0, init_scale: float = 0.1):
        super().__init__("InverseGammaQ", prior, init_loc, init_scale)

    @torch.no_grad()
    def mean(self) -> torch.Tensor:
        return self._mean_log_normal(self.prior.site_name)

    @torch.no_grad()
    def median(self) -> torch.Tensor:
        return self._median_log_normal(self.prior.site_name)

    @torch.no_grad()
    def mode(self) -> torch.Tensor:
        return self._mode_log_normal(self.prior.site_name)

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        return self._sample_log_normal(self.prior.site_name)


class NormalQ(QDist):
    def __init__(self, prior: PriorDist, init_loc: float = 0, init_scale: float = 0.1):
        super().__init__("NormalQ", prior, init_loc, init_scale)

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        return self._sample_normal(self.prior.site_name)


class GaussianProcessQ(QDist):
    def __init__(self, prior: PriorDist, init_loc: float = 0, init_scale: float = 0.1):
        super().__init__("GaussianProcessQ", prior, init_loc, init_scale)

    def _sample_gp(self, site_name: str, covariate: torch.Tensor) -> torch.Tensor:
        self.sample_dict[site_name] = pyro.sample(
            site_name, self.prior.gp.pyro_guide(covariate)
        )
        return self.sample_dict[site_name]

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        covariate = args[0]
        return self._sample_gp(self.prior.site_name, covariate)


class LaplaceQ(QDist):
    def __init__(self, prior: PriorDist, init_loc: float = 0, init_scale: float = 0.1):
        super().__init__("LaplaceQ", prior, init_loc, init_scale)

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        return self._sample_normal(self.prior.site_name)


class HorseshoeQ(QDist):
    def __init__(self, prior: PriorDist, init_loc: float = 0, init_scale: float = 0.1):
        super().__init__("HorseshoeQ", prior, init_loc, init_scale)

    def sample_global(self) -> Optional[torch.Tensor]:
        if hasattr(self.prior, "tau_delta"):
            return None
        return self._sample_log_normal(self.prior.tau_site_name)

    def sample_inter(self) -> Optional[torch.Tensor]:
        if not self.prior.ard:
            return None
        return self._sample_log_normal(self.prior.thetas_site_name)

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        self._sample_log_normal(self.prior.lambdas_site_name)
        if self.prior.regularized:
            self._sample_log_normal(self.prior.caux_site_name)
        return self._sample_normal(self.prior.site_name)


class SpikeAndSlabQ(QDist):
    def __init__(self, prior: PriorDist, init_loc: float = 0, init_scale: float = 0.1):
        super().__init__("SpikeAndSlabQ", prior, init_loc, init_scale)
        self.sigmoid_transform = dist.transforms.SigmoidTransform()

    def _sample_transformed_beta(self, site_name: str) -> torch.Tensor:
        loc, scale = self._get_loc_and_scale(site_name)
        unconstrained_latent = pyro.sample(
            site_name + "_unconstrained",
            dist.Normal(
                loc,
                scale,
            ),
            infer={"is_auxiliary": True},
        )

        value = self.sigmoid_transform(unconstrained_latent)
        log_density = self.sigmoid_transform.inv.log_abs_det_jacobian(
            value,
            unconstrained_latent,
        )
        delta_dist = dist.Delta(value, log_density=log_density)

        return pyro.sample(site_name, delta_dist)

    def _get_means(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._mean_normal(self.prior.untransformed_site_name), self._mean_normal(
            self.prior.lambdas_site_name
        )

    @torch.no_grad()
    def mean(self) -> torch.Tensor:
        untransformed_mean, lambdas_mean = self._get_means()
        return untransformed_mean * lambdas_mean

    @torch.no_grad()
    def median(self) -> torch.Tensor:
        untransformed_mean, lambdas_mean = self._get_means()
        return untransformed_mean * (lambdas_mean > 0.0)

    @torch.no_grad()
    def mode(self) -> torch.Tensor:
        return self.median()

    def sample_inter(self) -> Optional[torch.Tensor]:
        if self.prior.ard:
            self._sample_log_normal(self.prior.alphas_site_name)
        return self._sample_transformed_beta(self.prior.thetas_site_name)

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        self._sample_transformed_beta(self.prior.lambdas_site_name)
        return self._sample_normal(self.prior.untransformed_site_name)


class Guide(PyroModule):
    def __init__(self, model: Generative):
        """Approximate variational distribution for a generative model.

        Parameters
        ----------
        model : Generative
            Generative model
        """
        super().__init__("Guide")
        self.model = model
        self.device = model.device
        self.to(self.device)

        # dry run to setup shapes
        # call before setting up q_dists
        self.model()

        self.sigma_q_dists = self._get_q_dists(self.model.sigma_priors)
        self.factor_q_dists = self._get_q_dists(self.model.factor_priors)
        self.weight_q_dists = self._get_q_dists(self.model.weight_priors)

        self.sample_dict: dict[str, torch.Tensor] = {}

    def _get_q_dists(self, priors: Dict[str, PriorDist]) -> Dict[str, QDist]:
        _q_dists = {}

        for group, prior in priors.items():
            # Replace strings with actual Q distributions
            _q_dists[group] = {
                "InverseGammaP": InverseGammaQ,
                "NormalP": NormalQ,
                "GaussianProcessQ": GaussianProcessQ,
                "LaplaceP": LaplaceQ,
                "HorseshoeP": HorseshoeQ,
                "SpikeAndSlabP": SpikeAndSlabQ,
            }[prior._pyro_name](prior=prior)

        return _q_dists

    def forward(
        self,
        data: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        covariate: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Approximate posterior."""
        plates = self.model.get_plates()

        for obs_group, factor_q_dist in self.factor_q_dists.items():
            factor_q_dist.sample_global()
            with plates["factor"]:
                factor_q_dist.sample_inter()
                with plates[f"obs_{obs_group}"]:
                    self.sample_dict[
                        self.model.factor_priors[obs_group].site_name
                    ] = factor_q_dist()

        for feature_group, weight_q_dist in self.weight_q_dists.items():
            weight_q_dist.sample_global()
            with plates["factor"]:
                weight_q_dist.sample_inter()
                with plates[f"feature_{feature_group}"]:
                    self.sample_dict[
                        self.model.weight_priors[feature_group].site_name
                    ] = weight_q_dist()

            with plates[f"feature_{feature_group}"]:
                self.sample_dict[
                    self.model.sigma_priors[feature_group].site_name
                ] = self.sigma_q_dists[feature_group]()

        return self.sample_dict
