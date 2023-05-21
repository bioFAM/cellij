from typing import Callable, Optional

import pyro
import pyro.distributions as dist
import torch


def get_prior_function(
    sparsity_prior: Optional[str],
    n_factors: int,
    n_features: int,
    feature_idx: dict,
    feature_group_scale: dict,
    feature_group_names: list,
) -> Callable:
    """Return a function that samples the sparsity prior for each feature group.

    Args:
        sparsity_prior: The sparsity prior to use. If None, no sparsity prior is used.
        n_factors: The number of factors.
        n_features: The number of features.
        feature_idx: A dictionary mapping feature group names to the indices of the features in the feature group.
        feature_group_scale: A dictionary mapping feature group names to the scale of the features in the feature group.
        feature_group_names: A list of feature group names.

    Returns
    -------
        A function that samples the sparsity prior for each feature group.
    """
    view_shape = (-1, 1, n_factors, n_features)

    def spikeandslabbeta_sample():
        return pyro.sample(
            "w_scale", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
        ).view(view_shape)

    def spikeandslab_continousbernoulli_sample():
        pi = pyro.sample(
            "pi", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
        ).view(view_shape)
        return pyro.sample("w_scale", dist.ContinuousBernoulli(probs=pi)).view(
            view_shape
        )

    def spikeandslab_relaxedbernoulli_sample():
        pi = pyro.sample(
            "pi", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
        ).view(view_shape)
        return pyro.sample(
            "w_scale",
            dist.RelaxedBernoulliStraightThrough(
                temperature=torch.tensor(0.1), probs=pi
            ),
        ).view(view_shape)

    def horseshoe_sample():
        w_shape = view_shape
        w_scale = pyro.sample("w_scale", dist.HalfCauchy(torch.ones(1))).view(w_shape)
        return torch.cat(
            [
                w_scale[..., feature_idx[view]]
                * feature_group_scale[..., idx : idx + 1]
                for idx, view in enumerate(feature_group_names)
            ],
            dim=-1,
        )

    def lasso_sample():
        # TODO: Add source paper
        # Approximation to the Laplace density with a SoftLaplace,
        # see https://docs.pyro.ai/en/stable/_modules/pyro/distributions/softlaplace.html#SoftLaplace
        #
        # Unlike the Laplace distribution, this distribution is infinitely differentiable everywhere
        return pyro.sample(
            "w_scale", dist.SoftLaplace(torch.tensor(0.0), torch.tensor(1.0))
        ).view(view_shape)

    def nonnegative_sample():
        return pyro.sample(
            "w_scale", dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
        ).view(view_shape)

    def no_sample():
        return torch.ones(1)

    prior_functions = {
        "Spikeandslab-Beta": spikeandslabbeta_sample,
        "Spikeandslab-ContinuousBernoulli": spikeandslab_continousbernoulli_sample,
        "Spikeandslab-RelaxedBernoulli": spikeandslab_relaxedbernoulli_sample,
        "Horseshoe": horseshoe_sample,
        "Lasso": lasso_sample,
        "Nonnegative": nonnegative_sample,
        None: no_sample,
    }

    if not any([sparsity_prior == prior for prior in prior_functions]):
        valid_priors = list(prior_functions.keys())
        raise ValueError(
            f"Sparsity prior '{sparsity_prior}' is not valid. Valid priors are {valid_priors}."
        )

    return prior_functions[sparsity_prior]
