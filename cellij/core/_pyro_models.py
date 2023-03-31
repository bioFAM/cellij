import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule


class MOFA_Model(PyroModule):
    def __init__(self, n_factors: int, sparsity_prior: str):
        super().__init__(name="MOFA_Model")
        self.n_factors = n_factors
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
        self.params: dict[str, torch.Tensor] = {}

    def forward(self, X):
        """Generative model for MOFA."""
        plates = self.get_plates()

        with plates["obs"], plates["factors"]:
            self.params["z"] = pyro.sample("z", dist.Normal(torch.zeros(1), torch.ones(1))).view(
                -1, self.n_obs, self.n_factors, 1
            )

        if self.sparsity_prior == "Horseshoe":

            with plates["feature_groups"]:

                self.params["feature_group_scale"] = pyro.sample(
                    "feature_group_scale", dist.HalfCauchy(torch.ones(1))  # type: ignore
                ).view(-1, self.n_feature_groups)

        with plates["features"], plates["factors"]:

            if self.sparsity_prior == "Spikeandslab-Beta":

                self.params["w_scale"] = pyro.sample(
                    "w_scale", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
                ).view(-1, 1, self.n_factors, self.n_features)

            elif self.sparsity_prior == "Spikeandslab-ContinuousBernoulli":

                self.params["pi"] = pyro.sample(
                    "pi", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
                ).view(-1, 1, self.n_factors, self.n_features)

                self.params["w_scale"] = pyro.sample(
                    "w_scale", dist.ContinuousBernoulli(probs=self.params["pi"])  # type: ignore
                ).view(-1, 1, self.n_factors, self.n_features)

            elif self.sparsity_prior == "Spikeandslab-RelaxedBernoulli":

                self.params["pi"] = pyro.sample(
                    "pi", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
                ).view(-1, 1, self.n_factors, self.n_features)

                self.params["w_scale"] = pyro.sample(
                    "w_scale",
                    dist.RelaxedBernoulliStraightThrough(
                        temperature=torch.tensor(0.1), probs=self.params["pi"]
                    ),
                ).view(-1, 1, self.n_factors, self.n_features)

            elif self.sparsity_prior == "Spikeandslab-Enumeration":

                raise NotImplementedError()

            elif self.sparsity_prior == "Spikeandslab-Lasso":

                raise NotImplementedError()

            elif self.sparsity_prior == "Horseshoe":

                # implement the horseshoe prior
                w_shape = (-1, 1, self.n_factors, self.n_features)
                self.params["w_scale"] = pyro.sample("w_scale", dist.HalfCauchy(torch.ones(1))).view(  # type: ignore
                    w_shape
                )
                self.params["w_scale"] = torch.cat(
                    [
                        self.params["w_scale"][..., self.feature_idx[view]]
                        * self.params["feature_group_scale"][..., idx: idx + 1]
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
                self.params["w_scale"] = pyro.sample(
                    "w_scale", dist.SoftLaplace(torch.tensor(0.0), torch.tensor(1.0))
                ).view(-1, 1, self.n_factors, self.n_features)

            elif self.sparsity_prior == "Nonnegative":

                self.params["w_scale"] = pyro.sample(
                    "w_scale", dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
                ).view(-1, 1, self.n_factors, self.n_features)

            else:
                self.params["w_scale"] = torch.ones(1)

            self.params["w"] = pyro.sample("w", dist.Normal(torch.zeros(1), torch.ones(1))).view(
                -1, 1, self.n_factors, self.n_features

            )

            self.params["w"] = self.params["w_scale"] * self.params["w"]

            if self.sparsity_prior == "Nonnegative":
                self.params["w"] = torch.nn.Softplus()(self.params["w"])

        with plates["features"]:

            self.params["sigma"] = pyro.sample(
                "sigma", dist.InverseGamma(torch.tensor(3.0), torch.tensor(1.0))
            ).view(-1, 1, 1, self.n_features)

        with plates["obs"]:
            prod = torch.einsum("...ikj,...ikj->...ij", self.params["z"], self.params["w"]).view(
                -1, self.n_obs, 1, self.n_features
            )
            self.params["y"] = pyro.sample(
                "data",
                dist.Normal(prod, torch.sqrt(self.params["sigma"])),
                obs=X.view(-1, self.n_obs, 1, self.n_features),
            )

        return {"z": self.params["z"], "w": self.params["w"], "sigma": self.params["sigma"], "y": self.params["y"]}

    def get_plates(self):
        return {
            "obs": pyro.plate("obs", self.n_obs, dim=-3),
            "factors": pyro.plate("factors", self.n_factors, dim=-2),
            "features": pyro.plate("features", self.n_features, dim=-1),
            "feature_groups": pyro.plate("feature_groups", self.n_feature_groups, dim=-1),
        }
