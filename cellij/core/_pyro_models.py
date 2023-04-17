import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from pyro.distributions import constraints


class MOFA_Model(PyroModule):
    def __init__(self, n_factors: int, sparsity_prior: str):
        super().__init__(name="MOFA_Model")
        self.n_factors = n_factors
        # TODO: We should check if the prior is defined. Having a typo in the name
        # of the prior will not raise an error, but run the model without a prior.
        self.sparsity_prior = sparsity_prior
        # Default function to enforce constraints on the distributon parameters
        self.f_positive = torch.nn.Softplus()
        self.f_unit_interval = torch.nn.Sigmoid()

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

        # Create a dict of
        #   modality name : constraints of parameters
        # TODO: Remove the llh name in the Tuple
        self.distr_properties = {}
        for name, llh in likelihoods.items():
            # For some distributions you can pass one (!) out of multiple arguments, e.g., probs or logits
            # but you must not pass both. For probs vs. logits, we stick to logits, because it's simpler
            self.distr_properties[name] = (
                llh.__name__,
                {
                    k: v
                    for k, v in llh.arg_constraints.items()
                    if not ((k == "probs") and (set(["probs", "logits"]).issubset(llh.arg_constraints.keys())))
                },
            )

    def forward(self, X):
        """Generative model for MOFA."""
        # TODO: Is it helpful to move this into the _setup()?
        plates = self.get_plates()

        # We store all distributional parameters for the final likelihoods in a dict
        # keys are the modalities, values are dictionaries with keys being the distributional parameter names
        # and the values tensors
        params = {}
        for mod_name, (_, moment_constraints) in self.distr_properties.items():
            params[mod_name] = {k: None for k in moment_constraints.keys()}

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

        # Estimate additional distributional parameters
        for mod_name, (_, moment_constraints) in self.distr_properties.items():
            for idx, k in enumerate(moment_constraints.keys()):
                # The first parameter will be the result of z*w, hence we skip it here
                if idx == 0:
                    continue

                i = self.enum_data[mod_name]
                # TODO: Look-up conjugate prior based on distr_name
                with plates[f"features_{i}"]:
                    params[mod_name][k] = pyro.sample(
                        f"p{k}_{i}",
                        dist.InverseGamma(torch.tensor(3.0), torch.tensor(1.0)),
                    ).view(-1, 1, 1, self.n_feature_per_groups[i])

        with plates["obs"]:
            # We assume that the first parameter of each distribution is modelled as the product of
            # factor weights and loadings, aka z * w
            prod = torch.einsum("...ikj,...ikj->...ij", z, w).view(
                -1, self.n_obs, 1, self.n_features
            )
            # Split prod according to the modalities and assign it to the first estimated parameter
            for mod_name, (_, moment_constraints) in self.distr_properties.items():
                params[mod_name][next(iter(moment_constraints))] = prod[
                    ..., self.feature_idx[mod_name]
                ]

            # Apply constraints to parameters if necessary
            # Loop over all modalities and all distributional parameters
            # Apply constraints if necessary
            # and match against observed data
            for mod_name, (_, moment_constraints) in self.distr_properties.items():
                for k, (moment, constraint) in enumerate(moment_constraints.items()):
                    if constraint == constraints.positive:
                        params[mod_name][moment] = self.f_positive(
                            params[mod_name][moment]
                        )
                    elif constraint == constraints.unit_interval:
                        params[mod_name][moment] = self.f_unit_interval(
                            params[mod_name][moment]
                        )

            # Manual post-processing
            # - For normal distributions: estimate standard deviation, not variance
            for mod_name, (distr_name, _) in self.distr_properties.items():
                if distr_name == "Normal":
                    params[mod_name]["scale"] = torch.sqrt(params[mod_name]["scale"])

            for mod_name in self.distr_properties.keys():
                pyro.sample(
                    mod_name,
                    self.likelihoods[mod_name](**params[mod_name]),
                    obs=X[..., self.feature_idx[mod_name]],
                )

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
