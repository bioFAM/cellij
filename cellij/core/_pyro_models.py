import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from cellij.core.constants import DISTRIBUTIONS_PROPERTIES


class MOFA_Model(PyroModule):
    def __init__(self, n_factors: int, sparsity_prior: str):
        super().__init__(name="MOFA_Model")
        self.n_factors = n_factors
        # TODO: We should check if the prior is defined. Having a typo in the name
        # of the prior will not raise an error, but run the model without a prior.
        self.sparsity_prior = sparsity_prior
        self.f_positive = torch.nn.Softplus()
        self.f_unity = torch.nn.Sigmoid()

    def _setup(self, data, likelihoods):
        # TODO: at some point replace n_obs with obs_offsets
        self.values = torch.Tensor(data.values)
        self.n_obs = data.n_obs
        self.n_features = data.n_features
        self.n_feature_groups = len(data.names)
        self.feature_group_names = data.names
        self.obs_idx = data._obs_idx
        self.feature_idx = data._feature_idx
        self.likelihoods = likelihoods
        self.n_feature_per_groups = [len(x) for x in data._feature_idx.values()]
        self.enum_data = {i: j for j, i in enumerate(data.names)}

        self.n_parameters_per_group = [
            DISTRIBUTIONS_PROPERTIES[l]["params"] for l in likelihoods.values()
        ]

        # Now, we prepare some variables we need in the forward function repeatedly

        # In addition to the z*w, for some distributions we have to esitmate an additional parameter
        # we are storing them here
        self.p2 = {}
        # contains idx of feature group where we need to estimate a 2nd parameter
        self.estimate_p2 = []
        for name, llh in self.likelihoods.items():
            if DISTRIBUTIONS_PROPERTIES[llh]["params"] > 1:
                self.estimate_p2.append(name)

        # Prepare constraints
        self.constrain_p1 = {}
        for name, llh in self.likelihoods.items():
            if DISTRIBUTIONS_PROPERTIES[llh]["constraints"][0] == "positive":
                self.constrain_p1[name] = self.f_positive
            if DISTRIBUTIONS_PROPERTIES[llh]["constraints"][0] == "unit_interval":
                self.constrain_p1[name] = self.f_unity

    def forward(self, X):
        """Generative model for MOFA."""
        # TODO: Is it helpful to move this into the _setup()?
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
                        * feature_group_scale[..., idx : idx + 1]
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

        # For a Normal distribution, we have to estimate a noise parameter
        for name in self.estimate_p2:
            i = self.enum_data[name]
            with plates[f"features_{i}"]:
                self.p2[name] = pyro.sample(
                    f"p2_{i}",
                    dist.InverseGamma(torch.tensor(3.0), torch.tensor(1.0)),
                ).view(-1, 1, 1, self.n_feature_per_groups[i])

        with plates["obs"]:
            # We assume that the first parameter of each distribution is modelled as the product of
            # factor weights and loadings, aka z * w
            p1 = torch.einsum("...ikj,...ikj->...ij", z, w).view(
                -1, self.n_obs, 1, self.n_features
            )

            # Apply constraints to parts of the product if necessary
            # TODO: Also apply to other parameters of the distribution
            for name, f in self.constrain_p1.items():
                p1[..., self.feature_idx[name]] = f(p1[..., self.feature_idx[name]])

            # Iterate through all LLHs and sample from the appropriate distribution
            for name, llh in self.likelihoods.items():
                if llh.__name__ == "Normal":
                    y = pyro.sample(
                        name,
                        self.likelihoods[name](
                            p1[..., self.feature_idx[name]], torch.sqrt(self.p2[name])
                        ),
                        obs=X[..., self.feature_idx[name]],
                    )
                elif llh.__name__ == "Bernoulli":
                    y = pyro.sample(
                        name,
                        self.likelihoods[name](probs=p1[..., self.feature_idx[name]]),
                        obs=X[..., self.feature_idx[name]],
                    )
                elif llh.__name__ == "Poisson":
                    y = pyro.sample(
                        name,
                        self.likelihoods[name](p1[..., self.feature_idx[name]]),
                        obs=X[..., self.feature_idx[name]],
                    )

        return {"z": z, "w": w, "p2": self.p2, "y": y}

    def get_plates(self):
        # plates without and index '_i' cover the sum of all items
        plates = {
            "obs": pyro.plate("obs", self.n_obs, dim=-3),
            "factors": pyro.plate("factors", self.n_factors, dim=-2),
            "features": pyro.plate("features", self.n_features, dim=-1),
            "feature_groups": pyro.plate(
                "feature_groups", self.n_feature_groups, dim=-1
            ),
        }
        # Add one feature plate for each group
        for i, feature_set in enumerate(self.feature_idx.values()):
            plates[f"features_{i}"] = pyro.plate(
                f"features_{i}", len(feature_set), dim=-1
            )

        return plates
