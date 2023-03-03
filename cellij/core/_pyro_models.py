import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule

# from cellij.core.priors import Horseshoe


class MOFA_Model(PyroModule):
    def __init__(self, n_factors: int):
        super().__init__(name="MOFA_Model")
        self.n_factors = n_factors
        self.n_features = None
        self.n_feature_groups = None
        self.n_obs = None
        self.feature_offsets = None

    def _setup(self, n_obs, feature_offsets, sparsity_prior):
        # TODO: at some point replace n_obs with obs_offsets
        self.n_obs = n_obs
        self.n_features = feature_offsets[-1]
        self.n_feature_groups = len(feature_offsets) - 1
        self.feature_offsets = feature_offsets
        self.sparsity_prior = sparsity_prior

    def forward(self, X):
        """Generative model for MOFA."""
        plates = self.get_plates()

        with plates["obs"], plates["factors"]:
            z = pyro.sample("z", dist.Normal(torch.zeros(1), torch.ones(1))).view(-1, self.n_obs, self.n_factors, 1)

        with plates["features"], plates["factors"]:
            if self.sparsity_prior == "Spikeandslab-Beta":
                a_beta = torch.tensor(1e-3)
                b_beta = torch.tensor(1e-3)
                # a_gamma = pyro.param(torch.tensor(1e-3))
                # b_gamma = pyro.param(torch.tensor(1e-3))
                # tau = pyro.sample("tau", dist.Gamma(a_gamma, b_gamma)).view(-1, 1, self.n_factors, self.n_features)
                w_scale = pyro.sample("w_scale", dist.Beta(a_beta, b_beta)).view(-1, 1, self.n_factors, self.n_features)
                # slab = pyro.sample("slab", dist.Normal(torch.zeros(1), torch.tensor(1.0))).view(
                #     -1, 1, self.n_factors, self.n_features
                # )
                # w = pi * slab
            elif self.sparsity_prior == "Spikeandslab-ContinuousBernoulli":
                raise NotImplementedError()
            elif self.sparsity_prior == "Spikeandslab-RelaxedBernoulli":
                raise NotImplementedError()
            elif self.sparsity_prior == "Spikeandslab-Enumeration":
                raise NotImplementedError()
            elif self.sparsity_prior == "Spikeandslab-Lasso":
                raise NotImplementedError()
            elif self.sparsity_prior == "Horseshoe":
                w_scale = pyro.sample("w", self.sampling_dist(torch.tensor(1.0))).view(
                    -1, 1, self.n_factors, self.n_features
                )
            elif self.sparsity_prior == "Lasso":
                raise NotImplementedError()
            elif self.sparsity_prior == "Nonnegative":
                raise NotImplementedError()
            else:
                w_scale = torch.tensor(1.0)

            w = pyro.sample("w", dist.Normal(torch.zeros(1), w_scale)).view(-1, 1, self.n_factors, self.n_features)

        with plates["features"]:
            sigma = pyro.sample("sigma", dist.InverseGamma(torch.tensor(3.0), torch.tensor(1.0))).view(
                -1, 1, 1, self.n_features
            )

        with plates["obs"]:
            prod = torch.einsum("...ikj,...ikj->...ij", z, w).view(-1, self.n_obs, 1, self.n_features)
            y = pyro.sample("data", dist.Normal(prod, torch.sqrt(sigma)), obs=X.view(1, self.n_obs, 1, self.n_features))

        return {"z": z, "w": w, "sigma": sigma, "y": y}

    def get_plates(self):
        return {
            "obs": pyro.plate("obs", self.n_obs, dim=-3),
            "factors": pyro.plate("factors", self.n_factors, dim=-2),
            "features": pyro.plate("features", self.n_features, dim=-1),
            "feature_groups": pyro.plate("feature_groups", self.n_feature_groups, dim=-1),
        }
