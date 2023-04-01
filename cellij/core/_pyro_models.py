import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule


class MOFA_Model(PyroModule):
    def __init__(self, n_factors: int, sparsity_prior: str):
        super().__init__(name="MOFA_Model")
        self.n_factors = n_factors
        # TODO: We should check if the prior is defined. Having a typo in the name
        # of the prior will not raise an error, but run the model without a prior.
        self.sparsity_prior = sparsity_prior

    def _setup(self, data):
        # TODO: at some point replace n_obs with obs_offsets
        self.values = torch.Tensor(data.values)
        self.n_obs = data.n_obs
        self.n_features = data.n_features
        self.n_feature_groups = len(data.names)
        self.feature_group_names = data.names
        self.obs_idx = data._obs_idx
        self.feature_idx = data._feature_idx

    def forward(self, X):
        """Generative model for MOFA."""
        plates = self.get_plates()

        with plates["obs"], plates["factors"]:
            z = pyro.sample("z", dist.Normal(torch.zeros(1), torch.ones(1))).view(
                -1, self.n_obs, self.n_factors, 1
            )

        if self.sparsity_prior == "Horseshoe":
            with plates["feature_groups"]:
                feature_group_scale = pyro.sample(
                    "feature_group_scale", dist.HalfCauchy(torch.ones(1))  # type: ignore
                ).view(-1, self.n_feature_groups)

        with plates["features"], plates["factors"]:

            if self.sparsity_prior == "Spikeandslab-Beta":

                w_scale = pyro.sample(
                    "w_scale", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
                ).view(-1, 1, self.n_factors, self.n_features)

            elif self.sparsity_prior == "Spikeandslab-ContinuousBernoulli":

                pi = pyro.sample(
                    "pi", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
                ).view(-1, 1, self.n_factors, self.n_features)

                w_scale = pyro.sample(
                    "w_scale", dist.ContinuousBernoulli(probs=pi)  # type: ignore
                ).view(-1, 1, self.n_factors, self.n_features)

            elif self.sparsity_prior == "Spikeandslab-RelaxedBernoulli":

                pi = pyro.sample(
                    "pi", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
                ).view(-1, 1, self.n_factors, self.n_features)

                w_scale = pyro.sample(
                    "w_scale",
                    dist.RelaxedBernoulliStraightThrough(
                        temperature=torch.tensor(0.1), probs=pi
                    ),
                ).view(-1, 1, self.n_factors, self.n_features)

            elif self.sparsity_prior == "Spikeandslab-Enumeration":

                raise NotImplementedError()

            elif self.sparsity_prior == "Spikeandslab-Lasso":

                raise NotImplementedError()

            elif self.sparsity_prior == "Horseshoe":

                # implement the horseshoe prior
                w_shape = (-1, 1, self.n_factors, self.n_features)
                w_scale = pyro.sample("w_scale", dist.HalfCauchy(torch.ones(1))).view(  # type: ignore
                    w_shape
                )
                w_scale = torch.cat(
                    [
                        w_scale[..., self.feature_idx[view]]
                        * feature_group_scale[..., idx: idx + 1]
                        for idx, view in enumerate(self.feature_group_names)
                    ],
                    dim=-1,
                )

            elif self.sparsity_prior == "Lasso":
                # TODO: Add source paper
                # TODO: Parametrize scale
                # Approximation to the Laplace density with a SoftLaplace,
                # see https://docs.pyro.ai/en/stable/_modules/pyro/distributions/softlaplace.html#SoftLaplace
                #
                # Unlike the Laplace distribution, this distribution is infinitely differentiable everywhere
                w_scale = pyro.sample(
                    "w_scale", dist.SoftLaplace(torch.tensor(0.0), torch.tensor(1.0))
                ).view(-1, 1, self.n_factors, self.n_features)

            elif self.sparsity_prior == "Nonnegative":

                w_scale = pyro.sample(
                    "w_scale", dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
                ).view(-1, 1, self.n_factors, self.n_features)

            else:
                w_scale = torch.ones(1)

            w = pyro.sample("w", dist.Normal(torch.zeros(1), torch.ones(1))).view(
                -1, 1, self.n_factors, self.n_features

            )

            w = w_scale * w

            if self.sparsity_prior == "Nonnegative":
                w = torch.nn.Softplus()(w)

        with plates["features"]:

            sigma = pyro.sample(
                "sigma", dist.InverseGamma(torch.tensor(3.0), torch.tensor(1.0))
            ).view(-1, 1, 1, self.n_features)

        with plates["obs"]:
            prod = torch.einsum("...ikj,...ikj->...ij", z, w).view(
                -1, self.n_obs, 1, self.n_features
            )
            y = pyro.sample(
                "data",
                dist.Normal(prod, torch.sqrt(sigma)),
                obs=X.view(-1, self.n_obs, 1, self.n_features),
            )

        return {"z": z, "w": w, "sigma": sigma, "y": y}

    def get_plates(self):
        return {
            "obs": pyro.plate("obs", self.n_obs, dim=-3),
            "factors": pyro.plate("factors", self.n_factors, dim=-2),
            "features": pyro.plate("features", self.n_features, dim=-1),
            "feature_groups": pyro.plate("feature_groups", self.n_feature_groups, dim=-1),
        }
