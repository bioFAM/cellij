import logging
from typing import Dict

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule

logger = logging.getLogger(__name__)


class PDist(PyroModule):
    def __init__(self, name, site_name: str, device=None):
        super().__init__(name)
        self.site_name = site_name
        self.device = device
        self.to(self.device)

        self.sample_dict = {}

    def _zeros(self, size):
        return torch.zeros(size, device=self.device)

    def _ones(self, size):
        return torch.ones(size, device=self.device)

    def _const(self, value, size=1):
        return value * self._ones(size)

    def _sample(self, site_name, dist, dist_kwargs={}, sample_kwargs={}):
        self.sample_dict[site_name] = pyro.sample(
            site_name, dist(**dist_kwargs), **sample_kwargs
        )
        return self.sample_dict[site_name]

    def _deterministic(self, site_name, value, event_dim=None):
        self.sample_dict[site_name] = pyro.deterministic(site_name, value, event_dim)
        return self.sample_dict[site_name]

    def sample_global(self):
        return None

    def sample_inter(self):
        return None

    def forward(self):
        return None


class InverseGammaP(PDist):
    def __init__(self, site_name: str, device=None):
        super().__init__("InverseGammaP", site_name, device)

    def forward(self):
        return self._sample(
            self.site_name,
            dist.InverseGamma,
            dist_kwargs={"concentration": self._ones(1), "rate": self._ones(1)},
        )


class NormalP(PDist):
    def __init__(self, site_name: str, device=None):
        super().__init__("NormalP", site_name, device)

    def forward(self):
        return self._sample(
            self.site_name,
            dist.Normal,
            dist_kwargs={"loc": self._zeros(1), "scale": self._ones(1)},
        )


class LaplaceP(PDist):
    def __init__(self, site_name: str, scale: float = 0.1, device=None):
        super().__init__("LaplaceP", site_name, device)
        self.scale = self._const(scale)

    def forward(self):
        return self._sample(
            self.site_name,
            dist.SoftLaplace,
            dist_kwargs={"loc": self._zeros(1), "scale": self.scale},
        )


class HorseshoeP(PDist):
    def __init__(
        self,
        site_name: str,
        tau_scale: float = 1.0,
        tau_delta: float = None,
        lambdas_scale: float = 1.0,
        thetas_scale: float = 1.0,
        regularized: bool = False,
        ard: bool = True,
        device=None,
    ):
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

    def sample_global(self):
        if hasattr(self, "tau_delta"):
            return self._deterministic(self.tau_site_name, self.tau_delta)
        return self._sample(
            self.tau_site_name, dist.HalfCauchy, dist_kwargs={"scale": self.tau_scale}
        )

    def sample_inter(self):
        if not self.ard:
            return self._deterministic(self.thetas_site_name, self._ones(1))
        return self._sample(
            self.thetas_site_name,
            dist.HalfCauchy,
            dist_kwargs={"scale": self.thetas_scale},
        )

    def forward(self):
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
            dist_kwargs={"loc": self._zeros(1), "scale": lambdas_samples},
        )


class SpikeAndSlabP(PDist):
    def __init__(
        self,
        site_name: str,
        relaxed_bernoulli: bool = True,
        temperature: float = 0.1,
        ard: bool = True,
        device=None,
    ):
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

    def sample_inter(self):
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
            self._deterministic(self.alphas_site_name, self._ones(1))

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

    def forward(self):
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
                "loc": self._zeros(1),
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
        factor_priors: Dict[str, str] = None,
        weight_priors: Dict[str, str] = None,
        device=None,
    ):
        super().__init__("Generative")
        self.n_samples = sum(obs_dict.values())
        self.n_factors = n_factors
        self.feature_dict = feature_dict
        self.obs_dict = obs_dict
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

        self.sample_dict = {}

    def _get_priors(self, priors, site_name):
        _priors = {}

        for group, prior in priors.items():
            # Replace strings with actuals priors
            _priors[group] = {
                "InverseGamma": InverseGammaP,
                "Normal": NormalP,
                "Laplace": LaplaceP,
                "Horseshoe": HorseshoeP,
                "SpikeAndSlab": SpikeAndSlabP,
            }[prior](site_name=f"{site_name}_{group}", device=self.device)

        return _priors

    def get_plates(self):
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

    def forward(self, data: torch.Tensor = None):
        plates = self.get_plates()

        for obs_group, factor_prior in self.factor_priors.items():
            factor_prior.sample_global()
            with plates["factor"]:
                factor_prior.sample_inter()
                with plates[f"obs_{obs_group}"]:
                    self.sample_dict[f"z_{obs_group}"] = factor_prior()

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

        for obs_group, factor_prior in self.factor_priors.items():
            for feature_group, weight_prior in self.weight_priors.items():
                n_obs = self.obs_dict[obs_group]
                n_features = self.feature_dict[feature_group]

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
                    obs_mask = obs is None
                    if not obs_mask:
                        obs_mask = ~torch.isnan(obs)
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


if __name__ == "__main__":
    hs = HorseshoeP(
        "z",
        tau_scale=1.0,
        tau_delta=None,
        lambdas_scale=1.0,
        thetas_scale=1.0,
        regularized=True,
        ard=True,
        device=torch.device("cpu"),
    )

    hs.sample_global()
    hs.sample_inter()
    hs()

    model = Generative(
        n_factors=3,
        obs_dict={"group_0": 5, "group_1": 7, "group_2": 9},
        feature_dict={"view_0": 15, "view_1": 17, "view_2": 19},
        likelihoods={"group_0": "Normal", "group_1": "Normal", "group_2": "Normal"},
        factor_priors={
            "group_0": "Laplace",
            "group_1": "Horseshoe",
            "group_2": "SpikeAndSlab",
        },
        weight_priors={
            "view_0": "Laplace",
            "view_1": "Horseshoe",
            "view_2": "SpikeAndSlab",
        },
        device=torch.device("cpu"),
    )
    for k, v in model().items():
        print(k, v.shape)
