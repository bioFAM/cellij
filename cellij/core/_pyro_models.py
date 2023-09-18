import logging
from typing import Any, Optional, Union

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from torch.types import _device

from cellij.core._pyro_priors import PRIOR_MAP, PriorDist

logger = logging.getLogger(__name__)


class Generative(PyroModule):
    def __init__(
        self,
        n_factors: int,
        obs_dict: dict[str, int],
        feature_dict: dict[str, int],
        likelihoods: dict[str, str],
        factor_priors: dict[str, str],
        weight_priors: dict[str, str],
        device: _device,
    ):
        """Instantiate a generative model for the multi-group and multi-view FA.

        Parameters
        ----------
        n_factors : int
            Number of factors
        obs_dict : dict[str, int]
            Dictionary of observations per group
        feature_dict : dict[str, int]
            Dictionary of features per view
        likelihoods : dict[str, str]
            Likelihoods per view
        factor_priors : dict[str, str]
            Prior distributions for the factor scores
        weight_priors : dict[str, str]
            Prior distributions for the factor loadings
        device : _device
            Torch device
        """
        super().__init__("Generative")
        self.n_samples = sum(obs_dict.values())
        # self.n_samples = sum(len(obs) for obs in obs_dict.values())  # type: ignore
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

        self.sample_dict: dict[str, torch.Tensor] = {}

    def _get_prior(
        self, prior_config: Union[PriorDist, str, dict[str, Any]], site_name: str
    ) -> PriorDist:
        if isinstance(prior_config, PriorDist):
            return prior_config

        if isinstance(prior_config, str):
            prior_config = {"name": prior_config}
        elif isinstance(prior_config, dict):
            if "name" not in prior_config:
                raise ValueError(f"Prior `{prior_config}` must contain a `name` key.")
        else:
            raise ValueError(
                f"Prior `{prior_config}` is not supported, please provide a string or a dictionary."
            )

        if prior_config["name"] not in PRIOR_MAP:
            raise ValueError(
                f"Prior `{prior_config['name']}` is not supported, "
                f"please provide one of {', '.join(PRIOR_MAP.keys())}."
            )

        prior = PRIOR_MAP[prior_config["name"]]
        prior_config.pop("name")
        return prior(site_name=site_name, device=self.device, **prior_config)

    def _get_priors(
        self,
        priors: Union[dict[str, str], dict[str, dict[str, Any]]],
        site_name: str,
        **kwargs: dict[str, Any],
    ) -> dict[str, PriorDist]:
        return {
            group: self._get_prior(prior_config, f"{site_name}_{group}")
            for group, prior_config in priors.items()
        }

    def get_plates(self) -> dict[str, pyro.plate]:
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
        data: Optional[dict[str, dict[str, torch.Tensor]]] = None,
        covariate: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
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

                    # Likelihoods are defined per view
                    # - if Bernoulli, we treat them as logits, hence we don't apply a non-linearity
                    if self.likelihoods[feature_group] == "Poisson":
                        loc = torch.nn.Softplus()(loc)
                    elif self.likelihoods[feature_group] == "Gamma":
                        loc = torch.nn.Softplus()(loc) + 1e-6
                        scale = torch.exp(scale)

                    site_name = f"x_{obs_group}_{feature_group}"
                    obs_mask = torch.ones_like(loc, dtype=torch.bool)
                    if obs is not None:
                        obs_mask = torch.logical_not(torch.isnan(obs))
                    with pyro.poutine.mask(mask=obs_mask):
                        if obs is not None:
                            obs = torch.nan_to_num(obs, nan=0.5)

                        if self.likelihoods[feature_group] == "Normal":
                            dist_parametrized = dist.Normal(loc, scale + 1e-6)  # FIXME: Only a HOTFIX
                        elif self.likelihoods[feature_group] == "Bernoulli":
                            dist_parametrized = dist.ContinuousBernoulli(logits=loc)
                        elif self.likelihoods[feature_group] == "Gamma":
                            dist_parametrized = dist.Gamma(loc, scale)
                        elif self.likelihoods[feature_group] == "Poisson":
                            dist_parametrized = dist.Poisson(loc)

                        self.sample_dict[site_name] = pyro.sample(
                            site_name,
                            dist_parametrized,
                            obs=obs,
                            infer={"is_auxiliary": True},
                        )

        return self.sample_dict
