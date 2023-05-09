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

    def get_lambda_shape(self, feature_group=None):
        return (-1, 1, self.n_factors, self.get_n_features(feature_group))

    def get_tau_shape(self):
        return (-1, 1)

    def get_z_shape(self):
        return (-1, self.n_samples, self.n_factors, 1)

    def get_sigma_shape(self, feature_group=None):
        return (-1, 1, 1, self.get_n_features(feature_group))

    def get_x_shape(self, feature_group):
        return (-1, self.n_samples, 1, self.get_n_features(feature_group))

    def _sample_site(
        self, site_name, out_shape, dist, dist_kwargs={}, sample_kwargs={}
    ):
        samples = pyro.sample(site_name, dist(**dist_kwargs), **sample_kwargs).view(
            out_shape
        )
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
                    obs=data_view,
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
        device=None,
    ):
        self.tau_scale = tau_scale
        self.lambda_scale = lambda_scale
        super().__init__(n_samples, n_factors, feature_dict, likelihoods, device)

    def sample_tau(self, site_name="tau", feature_group=None):
        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_tau_shape(),
            dist.HalfCauchy,
            dist_kwargs={"scale": torch.tensor(self.tau_scale)},
        )

    def sample_lambda(self, site_name="lambda", feature_group=None):
        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_lambda_shape(feature_group),
            dist.HalfCauchy,
            dist_kwargs={"scale": torch.tensor(self.lambda_scale)},
        )

    def sample_w(self, site_name="w", feature_group=None):
        lmbda = self.sample_lambda(feature_group=feature_group)

        return self._sample_site(
            f"{site_name}_{feature_group}",
            self.get_w_shape(feature_group),
            dist.Normal,
            dist_kwargs={
                "loc": torch.zeros(1),
                "scale": self.sample_dict[f"tau_{feature_group}"] * lmbda,
            },
        )


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


if __name__ == "__main__":
    model = HorseshoeGenerative(100, 5, {"a": 10, "b": 20}, {})
    for k, v in model().items():
        print(k, v.shape)
