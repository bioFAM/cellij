import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule


class MOFA_Model(PyroModule):
    def __init__(self, n_factors: int):
        super().__init__(name="MOFA_Model")
        self.n_factors = n_factors
        # self.n_features = None
        # self.n_feature_groups = None
        # self.n_obs = None
        # self.feature_offsets = None

    def _setup(self, data):
        # TODO: at some point replace n_obs with obs_offsets
        self.n_obs = data.n_obs
        self.n_features = data.n_features
        self.n_views = len(data.names)
        self.views = data.names
        self.obs_idx = data._obs_idx
        self.feature_idx = data._feature_idx

    def forward(self, X):
        """Generative model for MOFA."""
        plates = self.get_plates()

        with plates["feature_groups"]:
            feature_group_scale = pyro.sample("feature_group_scale", dist.HalfCauchy(torch.ones(1))).view(
                -1, self.n_views
            )

        with plates["obs"], plates["factors"]:
            z = pyro.sample("z", dist.Normal(torch.zeros(1), torch.ones(1))).view(-1, self.n_obs, self.n_factors, 1)

        with plates["features"], plates["factors"]:
            # implement the horseshoe prior
            w_shape = (-1, 1, self.n_factors, self.n_features)
            w_scale = pyro.sample("w_scale", dist.HalfCauchy(torch.ones(1))).view(w_shape)
            w_scale = torch.cat(
                [
                    w_scale[..., self.feature_idx[self.views[m]]]
                    * feature_group_scale[..., m : m + 1]
                    for m in range(self.n_views)
                ],
                dim=-1,
            )
            w = pyro.sample("w", dist.Normal(torch.zeros(1), w_scale)).view(w_shape)

        with plates["features"]:
            sigma = pyro.sample("sigma", dist.InverseGamma(torch.tensor(3.0), torch.tensor(1.0))).view(
                -1, 1, 1, self.n_features
            )

        with plates["obs"]:
            prod = torch.einsum("...ikj,...ikj->...ij", z, w).view(-1, self.n_obs, 1, self.n_features)
            y = pyro.sample("data", dist.Normal(prod, torch.sqrt(sigma)), obs=X.view(-1, self.n_obs, 1, self.n_features))

        return {"z": z, "w": w, "sigma": sigma, "y": y}

    def get_plates(self):
        return {
            "obs": pyro.plate("obs", self.n_obs, dim=-3),
            "factors": pyro.plate("factors", self.n_factors, dim=-2),
            "features": pyro.plate("features", self.n_features, dim=-1),
            "feature_groups": pyro.plate("feature_groups", self.n_views, dim=-1),
        }
