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
    """Returns a function that samples the sparsity prior for each feature group.

    Args:
        sparsity_prior: The sparsity prior to use. If None, no sparsity prior is used.
        n_factors: The number of factors.
        n_features: The number of features.
        feature_idx: A dictionary mapping feature group names to the indices of the features in the feature group.
        feature_group_scale: A dictionary mapping feature group names to the scale of the features in the feature group.
        feature_group_names: A list of feature group names.

    Returns:
        A function that samples the sparsity prior for each feature group.
    """

    view_shape = (-1, 1, n_factors, n_features)
    if sparsity_prior == "Spikeandslab-Beta":

        def spikeandslabbeta_sample():
            return pyro.sample(
                "w_scale", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
            ).view(view_shape)

        return spikeandslabbeta_sample

    elif sparsity_prior == "Spikeandslab-ContinuousBernoulli":

        def spikeandslab_continousbernoulli_sample():
            pi = pyro.sample(
                "pi", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
            ).view(view_shape)

            return pyro.sample(
                "w_scale", dist.ContinuousBernoulli(probs=pi)  # type: ignore
            ).view(view_shape)

        return spikeandslab_continousbernoulli_sample

    elif sparsity_prior == "Spikeandslab-RelaxedBernoulli":

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

        return spikeandslab_relaxedbernoulli_sample

    elif sparsity_prior == "Spikeandslab-Enumeration":
        raise NotImplementedError()

    elif sparsity_prior == "Spikeandslab-Lasso":
        raise NotImplementedError()

    elif sparsity_prior == "Horseshoe":

        def horseshoe_sample():
            w_shape = view_shape
            w_scale = pyro.sample("w_scale", dist.HalfCauchy(torch.ones(1))).view(  # type: ignore
                w_shape
            )
            return torch.cat(
                [
                    w_scale[..., feature_idx[view]]
                    * feature_group_scale[..., idx : idx + 1]
                    for idx, view in enumerate(feature_group_names)
                ],
                dim=-1,
            )

        return horseshoe_sample

    elif sparsity_prior == "Lasso":
        # TODO: Add source paper
        # Approximation to the Laplace density with a SoftLaplace,
        # see https://docs.pyro.ai/en/stable/_modules/pyro/distributions/softlaplace.html#SoftLaplace
        #
        # Unlike the Laplace distribution, this distribution is infinitely differentiable everywhere
        def lasso_sample():
            return pyro.sample(
                "w_scale", dist.SoftLaplace(torch.tensor(0.0), torch.tensor(1.0))
            ).view(view_shape)

        return lasso_sample

    elif sparsity_prior == "Nonnegative":

        def nonnegative_sample():
            return pyro.sample(
                "w_scale", dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
            ).view(view_shape)

        return nonnegative_sample

    elif sparsity_prior is None:

        def no_sample():

            return torch.ones(1)

        return no_sample

    else:
        valid_priors = [
            None,
            "Spikeandslab-Beta",
            "Spikeandslab-ContinuousBernoulli",
            "Spikeandslab-RelaxedBernoulli",
            "Spikeandslab-Enumeration",
            "Spikeandslab-Lasso",
            "Lasso",
            "Horseshoe",
            "Nonnegative",
        ]
        raise ValueError(
            f"Sparsity prior '{sparsity_prior}' is not valid. Valid priors are {valid_priors}."
        )
