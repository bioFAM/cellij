import logging
from typing import Dict

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule

logger = logging.getLogger(__name__)


class Generative(PyroModule):
    def __init__(
        self,
        n_samples: int,
        n_factors: int,
        feature_dict: Dict[str, int],
        likelihoods: Dict[str, str],
        device: str = None,
    ):
        super().__init__(name="Generative")
        self.n_samples = n_samples
        self.n_factors = n_factors
        self.feature_dict = feature_dict
        self.n_feature_groups = len(feature_dict)
        self.likelihoods = likelihoods
        self.device = device

        self.sample_dict = {}

    def get_n_features(self, feature_group: str = None):
        return self.feature_dict[feature_group]

    def get_plates(self):
        plates = {
            "sample": pyro.plate("sample", self.n_samples, dim=-3),
            "factor": pyro.plate("factor", self.n_factors, dim=-2),
        }

        for feature_group, n_features in self.feature_dict.items():
            plates[f"feature_{feature_group}"] = pyro.plate(
                feature_group, n_features, dim=-1
            )

        return plates

    def get_latent_shape(self):
        return (-1, self.n_samples, self.n_factors, 1)

    def get_feature_group_shape(self):
        return (-1, 1, 1, 1)

    def get_factor_shape(self):
        return (-1, 1, self.n_factors, 1)

    def get_weight_shape(self, feature_group: str = None):
        return (-1, 1, self.n_factors, self.get_n_features(feature_group))

    def get_feature_shape(self, feature_group: str = None):
        return (-1, 1, 1, self.get_n_features(feature_group))

    def get_obs_shape(self, feature_group: str = None):
        return (-1, self.n_samples, 1, self.get_n_features(feature_group))

    def _sample_site(
        self,
        site_name,
        dist,
        dist_kwargs={},
        sample_kwargs={},
        link_fn=None,
        out_shape=None,
    ):
        samples = pyro.sample(site_name, dist(**dist_kwargs), **sample_kwargs)
        if out_shape is not None:
            samples = samples.view(out_shape)
        if link_fn is not None:
            samples = link_fn(samples)
        self.sample_dict[site_name] = samples
        return samples

    def sample_latent(self):
        return None

    def sample_feature_group(self, feature_group: str = None):
        return None

    def sample_feature_group_factor(self, feature_group: str = None):
        return None

    def sample_weight(self, feature_group: str = None):
        return None

    def sample_feature(self, feature_group: str = None):
        return None

    def sample_obs(self, data, feature_group: str = None):
        return None

    def forward(self, data: torch.Tensor = None, **kwargs):
        plates = self.get_plates()

        with plates["sample"], plates["factor"]:
            self.sample_latent()

        for feature_group, _ in self.feature_dict.items():
            self.sample_feature_group(feature_group=feature_group)
            with plates["factor"]:
                self.sample_feature_group_factor(feature_group=feature_group)
                with plates[f"feature_{feature_group}"]:
                    self.sample_weight(feature_group=feature_group)

            with plates[f"feature_{feature_group}"]:
                self.sample_feature(feature_group=feature_group)

            with plates["sample"]:
                self.sample_obs(data, feature_group=feature_group)

        return self.sample_dict


class NormalGenerative(Generative):
    def __init__(
        self,
        n_samples: int,
        n_factors: int,
        feature_dict: Dict[str, int],
        likelihoods: Dict[str, str],
        device: str = None,
    ):
        super().__init__(n_samples, n_factors, feature_dict, likelihoods, device)

    def sample_z(self):
        return self._sample_site(
            "z",
            dist.Normal,
            dist_kwargs={"loc": torch.zeros(1), "scale": torch.ones(1)},
            out_shape=self.get_latent_shape(),
        )

    def sample_w(self, feature_group: str = None):
        return self._sample_site(
            f"w_{feature_group}",
            dist.Normal,
            dist_kwargs={"loc": torch.zeros(1), "scale": torch.ones(1)},
            out_shape=self.get_weight_shape(feature_group),
        )

    def sample_sigma(self, feature_group: str = None):
        return self._sample_site(
            f"sigma_{feature_group}",
            dist.InverseGamma,
            dist_kwargs={"concentration": torch.tensor(1.0), "rate": torch.tensor(1.0)},
            out_shape=self.get_feature_shape(feature_group),
        )

    def sample_latent(self):
        return self.sample_z()

    def sample_weight(self, feature_group: str = None):
        return self.sample_w(feature_group=feature_group)

    def sample_feature(self, feature_group: str = None):
        return self.sample_sigma(feature_group=feature_group)

    def sample_obs(self, data, feature_group: str = None):
        obs = None
        if data is not None:
            obs = data[feature_group].view(self.get_obs_shape(feature_group))

        loc = torch.einsum(
            "...ikj,...ikj->...ij",
            self.sample_dict["z"],
            self.sample_dict[f"w_{feature_group}"],
        ).view(self.get_obs_shape(feature_group))
        scale = torch.sqrt(self.sample_dict[f"sigma_{feature_group}"])

        site_name = f"x_{feature_group}"

        self.sample_dict[site_name] = pyro.sample(
            site_name,
            dist.Normal(loc, scale),
            obs=obs,
            infer={"is_auxiliary": True},
        )


class LassoGenerative(NormalGenerative):
    def __init__(
        self,
        n_samples: int,
        n_factors: int,
        feature_dict: Dict[str, int],
        likelihoods: Dict[str, str],
        lasso_scale=0.1,
        device: str = None,
    ):
        super().__init__(n_samples, n_factors, feature_dict, likelihoods, device)
        self.lasso_scale = lasso_scale

    def sample_w(self, feature_group: str = None):
        return self._sample_site(
            f"w_{feature_group}",
            dist.SoftLaplace,
            dist_kwargs={
                "loc": torch.zeros(1),
                "scale": torch.tensor(self.lasso_scale),
            },
            out_shape=self.get_weight_shape(feature_group),
        )


class NonnegativeGenerative(NormalGenerative):
    def __init__(
        self,
        n_samples: int,
        n_factors: int,
        feature_dict: Dict[str, int],
        likelihoods: Dict[str, str],
        device: str = None,
    ):
        super().__init__(n_samples, n_factors, feature_dict, likelihoods, device)

    def sample_w(self, feature_group: str = None):
        return self._sample_site(
            f"w_{feature_group}",
            dist.Normal,
            dist_kwargs={"loc": torch.zeros(1), "scale": torch.ones(1)},
            link_fn=torch.nn.functional.relu,
            out_shape=self.get_weight_shape(feature_group),
        )


class HorseshoeGenerative(NormalGenerative):
    def __init__(
        self,
        n_samples: int,
        n_factors: int,
        feature_dict: Dict[str, int],
        likelihoods: Dict[str, str],
        tau_scale: float = 1.0,
        lambda_scale: float = 1.0,
        theta_scale: float = 1.0,
        delta_tau: bool = False,
        regularized: bool = False,
        ard: bool = False,
        device: str = None,
    ):
        super().__init__(n_samples, n_factors, feature_dict, likelihoods, device)
        self.tau_scale = tau_scale
        self.lambda_scale = lambda_scale
        self.theta_scale = theta_scale
        self.delta_tau = delta_tau
        self.regularized = regularized
        self.ard = ard

    def sample_tau(self, feature_group: str = None):
        site_name = f"tau_{feature_group}"
        if self.delta_tau:
            self.sample_dict[site_name] = pyro.deterministic(
                site_name, torch.tensor(self.tau_scale)
            )
        else:
            self._sample_site(
                site_name,
                dist.HalfCauchy,
                dist_kwargs={"scale": torch.tensor(self.tau_scale)},
                out_shape=self.get_feature_group_shape(),
            )
        return self.sample_dict[site_name]

    def sample_theta(self, feature_group: str = None):
        return self._sample_site(
            f"theta_{feature_group}",
            dist.HalfCauchy,
            dist_kwargs={"scale": torch.tensor(self.theta_scale)},
            out_shape=self.get_factor_shape(),
        )

    def sample_lambda(self, feature_group: str = None):
        return self._sample_site(
            f"lambda_{feature_group}",
            dist.HalfCauchy,
            dist_kwargs={"scale": torch.tensor(self.lambda_scale)},
            out_shape=self.get_weight_shape(feature_group=feature_group),
        )

    def sample_caux(self, feature_group: str = None):
        return self._sample_site(
            f"caux_{feature_group}",
            dist.InverseGamma,
            dist_kwargs={"concentration": torch.tensor(0.5), "rate": torch.tensor(0.5)},
            out_shape=self.get_weight_shape(feature_group=feature_group),
        )

    def sample_w(self, feature_group: str = None):
        return self._sample_site(
            f"w_{feature_group}",
            dist.Normal,
            dist_kwargs={
                "loc": torch.zeros(1),
                "scale": self.sample_dict[f"lambda_{feature_group}"],
            },
            out_shape=self.get_weight_shape(feature_group=feature_group),
        )

    def sample_feature_group(self, feature_group: str = None):
        return self.sample_tau(feature_group=feature_group)

    def sample_feature_group_factor(self, feature_group: str = None):
        return self.sample_theta(feature_group=feature_group)

    def sample_weight(self, feature_group: str = None):
        lmbda = (
            self.sample_lambda(feature_group=feature_group)
            * self.sample_dict[f"tau_{feature_group}"]
        )

        if self.ard:
            lmbda = lmbda * self.sample_dict[f"theta_{feature_group}"]

        if self.regularized:
            caux = self.sample_caux(feature_group=feature_group)
            lmbda = (torch.sqrt(caux) * lmbda) / torch.sqrt(caux + lmbda**2)

        self.sample_dict[f"lambda_{feature_group}"] = lmbda
        return self.sample_w(feature_group=feature_group)


class SpikeAndSlabGenerative(NormalGenerative):
    def __init__(
        self,
        n_samples: int,
        n_factors: int,
        feature_dict: Dict[str, int],
        likelihoods: Dict[str, str],
        relaxed_bernoulli: bool = True,
        temperature: float = 0.1,
        device: str = None,
    ):
        super().__init__(n_samples, n_factors, feature_dict, likelihoods, device)
        self.relaxed_bernoulli = relaxed_bernoulli
        self.temperature = temperature

        self.bernoulli_dist = dist.ContinuousBernoulli
        if self.relaxed_bernoulli:
            self.bernoulli_dist = dist.RelaxedBernoulliStraightThrough

    def sample_theta(self, feature_group: str = None):
        return self._sample_site(
            f"theta_{feature_group}",
            dist.Beta,
            dist_kwargs={
                "concentration1": torch.tensor(0.5),
                "concentration0": torch.tensor(0.5),
            },
            out_shape=self.get_factor_shape(),
        )

    def sample_alpha(self, feature_group: str = None):
        return self._sample_site(
            f"alpha_{feature_group}",
            dist.InverseGamma,
            dist_kwargs={"concentration": torch.tensor(0.5), "rate": torch.tensor(0.5)},
            out_shape=self.get_factor_shape(),
        )

    def sample_lambda(self, feature_group: str = None):
        dist_kwargs = {"probs": self.sample_dict[f"theta_{feature_group}"]}
        if self.relaxed_bernoulli:
            dist_kwargs["temperature"] = torch.tensor(self.temperature)

        return self._sample_site(
            f"lambda_{feature_group}",
            self.bernoulli_dist,
            dist_kwargs=dist_kwargs,
            out_shape=self.get_weight_shape(feature_group=feature_group),
        )

    def sample_w(self, feature_group: str = None):
        return self._sample_site(
            f"w_{feature_group}",
            dist.Normal,
            dist_kwargs={
                "loc": torch.zeros(1),
                "scale": self.sample_dict[f"alpha_{feature_group}"],
            },
            out_shape=self.get_weight_shape(feature_group=feature_group),
        )

    def sample_feature_group_factor(self, feature_group: str = None):
        self.sample_theta(feature_group=feature_group)
        self.sample_alpha(feature_group=feature_group)

    def sample_weight(self, feature_group: str = None):
        lmbda = self.sample_lambda(feature_group=feature_group)
        w = self.sample_w(feature_group=feature_group)
        return w * lmbda



# class SpikeNSlabLassoGenerative(SpikeNSlabGenerative):
#     def __init__(
#         self,
#         n_samples: int,
#         n_factors: int,
#         feature_dict: dict,
#         likelihoods,
#         lambda_spike=20.0,
#         lambda_slab=1.0,
#         relaxed_bernoulli=True,
#         temperature=0.1,
#         device=None,
#     ):
#         super().__init__(
#             n_samples,
#             n_factors,
#             feature_dict,
#             likelihoods,
#             relaxed_bernoulli,
#             temperature,
#             device,
#         )
#         self.lambda_spike = lambda_spike
#         self.lambda_slab = lambda_slab

#     def sample_laplace(
#         self, site_name="lambda", feature_group: str = None, is_spike=True
#     ):
#         spike_name = "spike" if is_spike else "slab"
#         scale = self.lambda_spike if is_spike else self.lambda_slab
#         return self._sample_site(
#             f"{site_name}_{spike_name}_{feature_group}",
#             self.get_w_shape(feature_group),
#             dist.Laplace,
#             dist_kwargs={
#                 "loc": torch.zeros(1),
#                 "scale": torch.tensor(scale),
#             },
#         )

#     def sample_w(self, site_name="w", feature_group: str = None):
#         self.sample_theta(feature_group=feature_group)
#         lmbda = self.sample_lambda(feature_group=feature_group)
#         lmbda_spike = self.sample_laplace(feature_group=feature_group, is_spike=True)
#         lmbda_slab = self.sample_laplace(feature_group=feature_group, is_spike=False)

#         w = (1 - lmbda) * lmbda_spike + lmbda * lmbda_slab
#         self.sample_dict[f"{site_name}_{feature_group}"] = w
#         return w


if __name__ == "__main__":
    model = HorseshoeGenerative(100, 5, {"a": 10, "b": 20}, {})
    for k, v in model().items():
        print(k, v.shape)
