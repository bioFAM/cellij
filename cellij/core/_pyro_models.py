import logging

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
        feature_dict: dict,
        likelihoods,
        device=None,
    ):
        super().__init__(name="Generative")
        self.n_samples = n_samples
        self.n_factors = n_factors
        self.feature_dict = feature_dict
        self.n_feature_groups = len(feature_dict)
        self.likelihoods = likelihoods
        self.device = device

        self.sample_dict = {}

    def get_n_features(self, feature_group=None):
        return self.feature_dict[feature_group]
        # n_features = self.n_features
        # if feature_group is not None:
        #     n_features = self.feature_dict[feature_group]
        # return n_features

    def get_plates(self):
        plates = {
            "obs": pyro.plate("obs", self.n_samples, dim=-3),
            "factors": pyro.plate("factors", self.n_factors, dim=-2),
            "feature_groups": pyro.plate(
                "feature_groups", self.n_feature_groups, dim=-1
            ),
        }

        for feature_group, n_features in self.feature_dict.items():
            plates[f"features_{feature_group}"] = pyro.plate(
                feature_group, n_features, dim=-1
            )

        return plates

    def get_w_shape(self, feature_group=None):
        return (-1, 1, self.n_factors, self.get_n_features(feature_group))

    def get_tau_shape(self):
        return (-1, 1, 1, 1)

    def get_z_shape(self):
        return (-1, self.n_samples, self.n_factors, 1)

    def get_sigma_shape(self, feature_group=None):
        return (-1, 1, 1, self.get_n_features(feature_group))

    def get_x_shape(self, feature_group):
        return (-1, self.n_samples, 1, self.get_n_features(feature_group))

    def _sample_site(
        self, site_name, out_shape, dist, dist_kwargs={}, sample_kwargs={}, link_fn=None
    ):
        samples = pyro.sample(site_name, dist(**dist_kwargs), **sample_kwargs).view(
            out_shape
        )
        if link_fn is not None:
            samples = link_fn(samples)
        self.sample_dict[site_name] = samples
        return samples

    def sample_z(self, site_name="z"):
        return self._sample_site(
            site_name,
            self.get_z_shape(),
            dist.Normal,
            dist_kwargs={"loc": torch.zeros(1), "scale": torch.ones(1)},
        )

    def sample_tau(self, site_name="tau", feature_group=None):
        return None

    def sample_w(self, site_name="w", feature_group=None):
        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_w_shape(feature_group),
            dist.Normal,
            dist_kwargs={"loc": torch.zeros(1), "scale": torch.ones(1)},
        )

    def sample_sigma(self, site_name="sigma", feature_group=None):
        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_sigma_shape(feature_group),
            dist.InverseGamma,
            dist_kwargs={"concentration": torch.tensor(1.0), "rate": torch.tensor(1.0)},
        )

    def forward(self, data: torch.Tensor = None, **kwargs):
        plates = self.get_plates()

        with plates["obs"], plates["factors"]:
            self.sample_z()

        for feature_group, _ in self.feature_dict.items():
            self.sample_tau(feature_group=feature_group)
            with plates["factors"], plates[f"features_{feature_group}"]:
                self.sample_w(feature_group=feature_group)

            with plates[f"features_{feature_group}"]:
                self.sample_sigma(feature_group=feature_group)

            with plates["obs"]:
                data_view = None
                if data is not None:
                    data_view = data[feature_group].view(
                        self.get_x_shape(feature_group)
                    )

                with pyro.poutine.mask(mask=torch.isnan(data_view) == 0):
                    # https://forum.pyro.ai/t/poutine-nan-mask-not-working/3489
                    # Assign temporary values to the missing data, not used
                    # anyway due to masking.
                    masked_data = torch.nan_to_num(data_view, nan=1.0)

                    self.sample_dict[f"x_{feature_group}"] = pyro.sample(
                        f"x_{feature_group}",
                        dist.Normal(
                            torch.einsum(
                                "...ikj,...ikj->...ij",
                                self.sample_dict["z"],
                                self.sample_dict[f"w_{feature_group}"],
                            ).view(self.get_x_shape(feature_group)),
                            torch.sqrt(self.sample_dict[f"sigma_{feature_group}"]),
                        ),
                        obs=masked_data,
                        infer={"is_auxiliary": True},
                    )

        return self.sample_dict


class HorseshoeGenerative(Generative):
    def __init__(
        self,
        n_samples: int,
        n_factors: int,
        feature_dict: dict,
        likelihoods,
        tau_scale=1.0,
        lambda_scale=1.0,
        delta_tau=False,
        regularized=False,
        device=None,
    ):
        super().__init__(n_samples, n_factors, feature_dict, likelihoods, device)
        self.tau_scale = tau_scale
        self.lambda_scale = lambda_scale
        self.delta_tau = delta_tau
        self.regularized = regularized

    def sample_tau(self, site_name="tau", feature_group=None):
        site_name = f"{site_name}_{feature_group}"
        if self.delta_tau:
            self.sample_dict[site_name] = pyro.deterministic(
                site_name, torch.tensor(self.tau_scale)
            )
        else:
            self._sample_site(
                site_name,
                self.get_tau_shape(),
                dist.HalfCauchy,
                dist_kwargs={"scale": torch.tensor(self.tau_scale)},
            )
        return self.sample_dict[site_name]

    def sample_lambda(self, site_name="lambda", feature_group=None):
        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_w_shape(feature_group),
            dist.HalfCauchy,
            dist_kwargs={"scale": torch.tensor(self.lambda_scale)},
        )

    def sample_caux(self, site_name="caux", feature_group=None):
        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_w_shape(feature_group),
            dist.InverseGamma,
            dist_kwargs={"concentration": torch.tensor(0.5), "rate": torch.tensor(0.5)},
        )

    def sample_w(self, site_name="w", feature_group=None):
        lmbda = self.sample_lambda(feature_group=feature_group)
        if self.regularized:
            caux = self.sample_caux(feature_group=feature_group)
            lmbda = (torch.sqrt(caux) * lmbda) / torch.sqrt(caux + lmbda**2)

        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_w_shape(feature_group),
            dist.Normal,
            dist_kwargs={
                "loc": torch.zeros(1),
                "scale": self.sample_dict[f"tau_{feature_group}"] * lmbda,
            },
        )


class HorseshoePlusGenerative(Generative):
    def __init__(
        self,
        n_samples: int,
        n_factors: int,
        feature_dict: dict,
        likelihoods,
        tau_const=1.0,
        eta_scale=1.0,
        device=None,
    ):
        super().__init__(n_samples, n_factors, feature_dict, likelihoods, device)
        self.tau_const = tau_const
        self.eta_scale = eta_scale

    def sample_tau(self, site_name="tau", feature_group=None):
        site_name = f"{site_name}_{feature_group}"
        self.sample_dict[site_name] = pyro.deterministic(
            site_name, torch.tensor(self.tau_const)
        )
        return self.sample_dict[site_name]

    def sample_eta(self, site_name="eta", feature_group=None):
        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_w_shape(feature_group),
            dist.HalfCauchy,
            dist_kwargs={"scale": torch.tensor(self.eta_scale)},
        )

    def sample_lambda(self, site_name="lambda", feature_group=None):
        eta = self.sample_eta(feature_group=feature_group)
        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_w_shape(feature_group),
            dist.HalfCauchy,
            dist_kwargs={"scale": eta * self.sample_dict[f"tau_{feature_group}"]},
        )

    def sample_w(self, site_name="w", feature_group=None):
        lmbda = self.sample_lambda(feature_group=feature_group)

        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_w_shape(feature_group),
            dist.Normal,
            dist_kwargs={
                "loc": torch.zeros(1),
                "scale": lmbda,
            },
        )


class NormalGenerative(Generative):
    def __init__(
        self,
        n_samples: int,
        n_factors: int,
        feature_dict: dict,
        likelihoods,
        device=None,
        **kwargs,
    ):
        super().__init__(n_samples, n_factors, feature_dict, likelihoods, device)


class LassoGenerative(Generative):
    def __init__(
        self,
        n_samples: int,
        n_factors: int,
        feature_dict: dict,
        likelihoods,
        lasso_scale=0.1,
        device=None,
        **kwargs,
    ):
        super().__init__(n_samples, n_factors, feature_dict, likelihoods, device)

        self.lasso_scale = lasso_scale

    def sample_w(self, site_name="w", feature_group=None):
        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_w_shape(feature_group),
            dist.SoftLaplace,
            dist_kwargs={
                "loc": torch.zeros(1),
                "scale": torch.tensor(self.lasso_scale),
            },
        )


class NonnegativityGenerative(Generative):
    def __init__(
        self,
        n_samples: int,
        n_factors: int,
        feature_dict: dict,
        likelihoods,
        device=None,
        **kwargs,
    ):
        super().__init__(n_samples, n_factors, feature_dict, likelihoods, device)

    def sample_w(self, site_name="w", feature_group=None):
        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_w_shape(feature_group),
            dist.Normal,
            dist_kwargs={
                "loc": torch.zeros(1),
                "scale": torch.ones(1),
            },
            link_fn=torch.nn.functional.relu,
        )


class SpikeNSlabGenerative(Generative):
    def __init__(
        self,
        n_samples: int,
        n_factors: int,
        feature_dict: dict,
        likelihoods,
        relaxed_bernoulli=True,
        temperature=0.1,
        device=None,
    ):
        super().__init__(n_samples, n_factors, feature_dict, likelihoods, device)

        self.relaxed_bernoulli = relaxed_bernoulli
        if self.relaxed_bernoulli:
            self.bernoulli_dist = dist.RelaxedBernoulliStraightThrough
        else:
            self.bernoulli_dist = dist.ContinuousBernoulli
        self.temperature = temperature

    def sample_theta(self, site_name="theta", feature_group=None):
        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_w_shape(feature_group),
            dist.Beta,
            dist_kwargs={
                "concentration1": torch.tensor(0.5),
                "concentration0": torch.tensor(0.5),
            },
        )

    def sample_lambda(self, site_name="lambda", feature_group=None):
        dist_kwargs = {"probs": self.sample_dict[f"theta_{feature_group}"]}
        if self.relaxed_bernoulli:
            dist_kwargs["temperature"] = torch.tensor(self.temperature)
        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_w_shape(feature_group),
            self.bernoulli_dist,
            dist_kwargs=dist_kwargs,
        )

    def sample_w(self, site_name="w", feature_group=None):
        self.sample_theta(feature_group=feature_group)
        lmbda = self.sample_lambda(feature_group=feature_group)

        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_w_shape(feature_group),
            dist.Normal,
            dist_kwargs={
                "loc": torch.zeros(1),
                "scale": lmbda,
            },
        )


class SpikeNSlabLassoGenerative(SpikeNSlabGenerative):
    def __init__(
        self,
        n_samples: int,
        n_factors: int,
        feature_dict: dict,
        likelihoods,
        lambda_spike=20.0,
        lambda_slab=1.0,
        relaxed_bernoulli=True,
        temperature=0.1,
        device=None,
    ):
        super().__init__(
            n_samples,
            n_factors,
            feature_dict,
            likelihoods,
            relaxed_bernoulli,
            temperature,
            device,
        )
        self.lambda_spike = lambda_spike
        self.lambda_slab = lambda_slab

    def sample_laplace(self, site_name="lambda", feature_group=None, is_spike=True):
        spike_name = "spike" if is_spike else "slab"
        scale = self.lambda_spike if is_spike else self.lambda_slab
        return self._sample_site(
            f"{site_name}_{spike_name}_{feature_group}",
            self.get_w_shape(feature_group),
            dist.Laplace,
            dist_kwargs={
                "loc": torch.zeros(1),
                "scale": torch.tensor(scale),
            },
        )

    def sample_w(self, site_name="w", feature_group=None):
        self.sample_theta(feature_group=feature_group)
        lmbda = self.sample_lambda(feature_group=feature_group)
        lmbda_spike = self.sample_laplace(feature_group=feature_group, is_spike=True)
        lmbda_slab = self.sample_laplace(feature_group=feature_group, is_spike=False)

        w = (1 - lmbda) * lmbda_spike + lmbda * lmbda_slab
        self.sample_dict[f"{site_name}_{feature_group}"] = w
        return w


if __name__ == "__main__":
    model = HorseshoeGenerative(100, 5, {"a": 10, "b": 20}, {})
    for k, v in model().items():
        print(k, v.shape)
