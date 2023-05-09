import logging

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.nn import PyroModule

logger = logging.getLogger(__name__)


class MOFA_Model(PyroModule):
    def __init__(self, n_factors: int, sparsity_prior: str):
        super().__init__(name="MOFA_Model")
        self.n_factors = n_factors
        self.sparsity_prior = sparsity_prior
        # Default function to enforce constraints on the distributon parameters
        self.f_positive = torch.nn.Softplus()
        self.f_unit_interval = torch.nn.Sigmoid()

        valid_priors = [
            "Horseshoe",
            # "Spikeandslab-Beta",
            "Spikeandslab-ContinuousBernoulli",
            "Spikeandslab-RelaxedBernoulli",
            # "Spikeandslab-Enumeration",
            "Spikeandslab-Lasso",
            "Lasso",
            "Nonnegative",
        ]
        if self.sparsity_prior not in valid_priors:
            logger.warning("No prior for inference selected.")

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
        self.data_dict = {i: j for j, i in enumerate(data.names)}

        # Create a dict of
        #   modality name : constraints of parameters
        # TODO: Remove the llh name in the Tuple
        self.distr_properties = {}
        for name, llh in likelihoods.items():
            # For some distributions you can pass one (!) out of multiple arguments, e.g., probs or logits
            # but you must not pass both.
            # - For probs vs. logits, we stick to logits, because it's simpler
            self.distr_properties[name] = (
                llh.__name__,
                {
                    k: v
                    for k, v in llh.arg_constraints.items()
                    if not (
                        (k == "probs")
                        and (["probs", "logits"] <= list(llh.arg_constraints.keys()))
                    )
                },
            )

    def forward(self, data: torch.Tensor, **kwargs):
        """Generative model for MOFA+."""
        plates = self.get_plates()

        # # #### !!! DEBUG !!!
        # with plates["obs"], plates["factors"]:
        #     z = pyro.sample("z", dist.Normal(torch.zeros(1), torch.ones(1))).view(
        #         -1, self.n_obs, self.n_factors, 1
        #     )

        # with plates["features"], plates["factors"]:
        #     w = pyro.sample("w", dist.Normal(torch.zeros(1), torch.ones(1))).view(
        #         -1, 1, self.n_factors, self.n_features
        #     )

        # with plates["features"]:
        #     sigma = pyro.sample(
        #         "sigma", dist.InverseGamma(torch.tensor(1.0), torch.tensor(1.0))
        #     ).view(-1, 1, 1, self.n_features)

        # with plates["obs"]:
        #     # We assume that the first parameter of each distribution is modelled as the product of
        #     # factor weights and loadings, aka z * w
        #     prod = torch.einsum("...ikj,...ikj->...ij", z, w).view(-1, self.n_obs, 1, self.n_features)

        #     # prod = torch.matmul(z, w)

        #     x = pyro.sample(
        #         "x",
        #         dist.Normal(prod, torch.sqrt(sigma)),
        #         obs=data.view(-1, self.n_obs, 1, self.n_features),
        #     )

        # return
        # # #### !!! DEBUG !!!

        # We store all distributional parameters for the final likelihoods in a dict, called `params`.
        # Keys are the modalities, values are dictionaries with keys being the distributional parameter names
        # and the values the corresponding tensors
        params = {}
        for mod_name, (_, moment_constraints) in self.distr_properties.items():
            params[mod_name] = {k: None for k in moment_constraints}

            # Estimate distributional parameters in addition to z*w if necessary
            for idx, k in enumerate(moment_constraints.keys()):
                # The first parameter will be the result of z*w, hence we skip it here
                if idx > 0:
                    i = self.data_dict[mod_name]
                    # TODO: Look-up conjugate prior based on distr_name
                    with plates[f"features_{i}"]:
                        params[mod_name][k] = pyro.sample(
                            f"p{k}_{i}",
                            dist.InverseGamma(torch.tensor(1.0), torch.tensor(1.0)),
                        ).view(-1, 1, 1, len(self.feature_idx[mod_name]))

        with plates["obs"], plates["factors"]:
            z = pyro.sample("z", dist.Normal(torch.zeros(1), torch.ones(1))).view(
                -1, self.n_obs, self.n_factors, 1
            )

        # Priors

        shape = (-1, 1, self.n_factors, self.n_features)

        if self.sparsity_prior == "Horseshoe":
            # Horseshoe
            # Reference: "The horseshoe estimator for sparse signals.", C. M. Carvalho, N. G. Polson,
            # J. G. S. (2010).
            with plates["feature_groups"]:
                feature_group_scale = pyro.sample(
                    "feature_group_scale", dist.HalfCauchy(torch.ones(1))  # type: ignore[call-overload]
                ).view(-1, self.n_feature_groups)

            with plates["features"], plates["factors"]:
                w_scale = pyro.sample("w_scale", dist.HalfCauchy(torch.ones(1))).view(shape)  # type: ignore
                w_scale = torch.cat(
                    [
                        w_scale[..., self.feature_idx[view]]
                        * feature_group_scale[..., idx : idx + 1]
                        for idx, view in enumerate(self.feature_group_names)
                    ],
                    dim=-1,
                )

                # caux = pyro.sample(
                #     "caux",
                #     dist.InverseGamma(torch.tensor(0.5), torch.tensor(0.5)),
                # )
                # w_scale = (torch.sqrt(caux) * w_scale) / torch.sqrt(caux + w_scale**2)

                # unscaled_w = pyro.sample(
                #     "unscaled_w", dist.Normal(torch.zeros(1), torch.ones(1))
                # )
                # w = pyro.deterministic("w", unscaled_w * w_scale)

                w = pyro.sample("w", dist.Normal(torch.zeros(1), w_scale)).view(shape)
                w = w.view(shape)

        else:
            if self.sparsity_prior == "Lasso":
                with plates["features"], plates["factors"]:
                    # Bayesian Lasso
                    # Reference: "The Bayesian lasso.", Park, T. and Casella, G. (2008).
                    # Journal of the American Statistical Association
                    #
                    # Approximation to the Laplace density with a SoftLaplace,
                    # see https://docs.pyro.ai/en/stable/_modules/pyro/distributions/softlaplace.html#SoftLaplace
                    lambda_ = kwargs.get("lambda", 0.1)
                    w = pyro.sample(
                        "w",
                        dist.SoftLaplace(torch.tensor(0.0), torch.tensor(lambda_)),
                    ).view(shape)

            elif self.sparsity_prior == "Spikeandslab-ContinuousBernoulli":
                # Spike-and-Slab with a continuous relaxation of the Bernoulli distribution
                alpha_beta = kwargs.get("alpha_beta", 1.0)
                beta_beta = kwargs.get("beta_beta", 1.0)
                alpha_gamma = kwargs.get("alpha_gamma", 0.001)
                beta_gamma = kwargs.get("beta_gamma", 0.001)
                # We estimate one value per factor and per modality
                with plates["feature_groups"], plates["factors"]:
                    theta = pyro.sample(
                        "theta",
                        dist.Beta(torch.tensor(alpha_beta), torch.tensor(beta_beta)),
                    )
                theta = theta.view(-1, self.n_feature_groups, self.n_factors, 1)

                with plates["feature_groups"], plates["factors"]:
                    alpha = pyro.sample(
                        "alpha",
                        dist.Gamma(torch.tensor(alpha_gamma), torch.tensor(beta_gamma)),
                    )
                alpha = alpha.view(-1, self.n_feature_groups, self.n_factors, 1)

                samples = []
                for idx, _ in enumerate(self.feature_group_names):
                    with plates[f"features_{idx}"], plates["factors"]:
                        samples_normal = pyro.sample(
                            f"samples_normal_{idx}",
                            dist.Normal(loc=0, scale=alpha[0, idx]),
                        )
                        samples_bernoulli = pyro.sample(
                            f"samples_bernoulli_{idx}",
                            dist.ContinuousBernoulli(probs=theta[0, idx]),
                        )
                    samples.append(samples_normal * samples_bernoulli)

                w = torch.cat(samples, axis=-1).view(shape)

            elif self.sparsity_prior == "Spikeandslab-RelaxedBernoulli":
                # Spike-and-Slab with a continuous relaxation of the Bernoulli distribution
                alpha_beta = kwargs.get("alpha_beta", 1.0)
                beta_beta = kwargs.get("beta_beta", 1.0)
                alpha_gamma = kwargs.get("alpha_gamma", 0.001)
                beta_gamma = kwargs.get("beta_gamma", 0.001)
                temperature = kwargs.get("temperature", 0.1)
                # We estimate one value per factor and per modality
                with plates["feature_groups"], plates["factors"]:
                    theta = pyro.sample(
                        "theta",
                        dist.Beta(torch.tensor(alpha_beta), torch.tensor(beta_beta)),
                    )
                theta = theta.view(-1, self.n_feature_groups, self.n_factors, 1)

                with plates["feature_groups"], plates["factors"]:
                    alpha = pyro.sample(
                        "alpha",
                        dist.Gamma(torch.tensor(alpha_gamma), torch.tensor(beta_gamma)),
                    )
                alpha = alpha.view(-1, self.n_feature_groups, self.n_factors, 1)

                samples = []
                for idx, _ in enumerate(self.feature_group_names):
                    with plates[f"features_{idx}"], plates["factors"]:
                        samples_normal = pyro.sample(
                            f"samples_normal_{idx}",
                            dist.Normal(loc=0, scale=alpha[0, idx]),
                        )
                        samples_bernoulli = pyro.sample(
                            f"samples_bernoulli_{idx}",
                            dist.RelaxedBernoulliStraightThrough(
                                temperature=torch.tensor(temperature),
                                probs=theta[0, idx],
                            ),
                        )
                    samples.append(samples_normal * samples_bernoulli)

                w = torch.cat(samples, axis=-1).view(shape)

            elif self.sparsity_prior == "Spikeandslab-Enumeration":
                raise NotImplementedError()

            elif self.sparsity_prior == "Spikeandslab-Lasso":
                # Spike-and-Slab with a Laplace distribution for the spike and slab
                lambda_spike = kwargs.get("lambda_spike", 20)
                lambda_slab = kwargs.get("lambda_slab", 1)

                samples = []
                for idx, (_, features) in enumerate(self.feature_idx.items()):
                    # The hyperparamters in the beta prior are a=1 and b=#number of features
                    with plates["feature_groups"], plates["factors"]:
                        samples_beta = pyro.sample(
                            f"samples_beta_{idx}",
                            dist.Beta(1, len(features)),
                        )
                    samples_beta = samples_beta.view(
                        -1, self.n_feature_groups, self.n_factors, 1
                    )

                    with plates[f"features_{idx}"], plates["factors"]:
                        samples_bernoulli = pyro.sample(
                            f"samples_bernoulli_{idx}",
                            dist.ContinuousBernoulli(probs=samples_beta[0, idx]),
                        )

                        samples_lambda_spike = pyro.sample(
                            f"samples_lambda_spike_{idx}",
                            dist.SoftLaplace(
                                torch.tensor(0.0), torch.tensor(lambda_spike)
                            ),
                        )
                        samples_lambda_slab = pyro.sample(
                            f"samples_lambda_slab_{idx}",
                            dist.SoftLaplace(
                                torch.tensor(0.0), torch.tensor(lambda_slab)
                            ),
                        )

                    samples.append(
                        (1 - samples_bernoulli) * samples_lambda_spike
                        + samples_bernoulli * samples_lambda_slab
                    )

                w = torch.cat(samples, axis=-1).view(shape)

            elif self.sparsity_prior == "Nonnegative":
                with plates["features"], plates["factors"]:
                    w = pyro.sample(
                        "w", dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
                    ).view(shape)
                    w = torch.nn.Softplus()(w)

            else:
                with plates["features"], plates["factors"]:
                    w = pyro.sample(
                        "w", dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
                    ).view(shape)

        with plates["obs"]:
            # We assume that the first parameter of each distribution is modelled as the product of
            # factor weights and loadings, aka z * w
            prod = torch.einsum("...ikj,...ikj->...ij", z, w).view(
                -1, self.n_obs, 1, self.n_features
            )

            # Apply constraints to parameters if necessary
            # Loop over all modalities and all distributional parameters
            # Apply constraints if necessary
            # and match against observed data
            for mod_name, (
                distr_name,
                moment_constraints,
            ) in self.distr_properties.items():
                # Split prod according to the modalities and assign it to the first estimated parameter
                params[mod_name][next(iter(moment_constraints))] = prod[
                    ..., self.feature_idx[mod_name]
                ]

                for moment, constraint in moment_constraints.items():
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
                if distr_name == "Normal":
                    params[mod_name]["scale"] = torch.sqrt(params[mod_name]["scale"])

            for mod_name in self.distr_properties:
                mod_data = data[..., self.feature_idx[mod_name]]

                with pyro.poutine.mask(mask=~torch.isnan(mod_data)):
                    # https://forum.pyro.ai/t/poutine-nan-mask-not-working/3489
                    # Assign temporary values to the missing data, not used
                    # anyway due to masking.
                    masked_data = torch.nan_to_num(mod_data, nan=0.0)

                    pyro.sample(
                        mod_name,
                        self.likelihoods[mod_name](**params[mod_name]),
                        obs=masked_data.view(
                            -1, self.n_obs, 1, len(self.feature_idx[mod_name])
                        ),
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
