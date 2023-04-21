from typing import List

import pyro
import pyro.distributions as dist
import torch


def get_prior_function(
    sparsity_prior: str,
    n_factors: int,
    n_features: int,
    feature_idx: dict,
    feature_group_scale: dict,
    feature_group_names: List,
):
    view_shape = (-1, 1, n_factors, n_features)
    if sparsity_prior == "Spikeandslab-Beta":

        def prior_sample():
            return pyro.sample(
                "w_scale", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
            ).view(view_shape)

        return prior_sample

    elif sparsity_prior == "Spikeandslab-ContinuousBernoulli":

        def prior_sample():
            pi = pyro.sample(
                "pi", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
            ).view(view_shape)

            return pyro.sample(
                "w_scale", dist.ContinuousBernoulli(probs=pi)  # type: ignore
            ).view(view_shape)

        return prior_sample

    elif sparsity_prior == "Spikeandslab-RelaxedBernoulli":

        def prior_sample():
            pi = pyro.sample(
                "pi", dist.Beta(torch.tensor(0.001), torch.tensor(0.001))
            ).view(view_shape)

            return pyro.sample(
                "w_scale",
                dist.RelaxedBernoulliStraightThrough(
                    temperature=torch.tensor(0.1), probs=pi
                ),
            ).view(view_shape)

        return prior_sample

    elif sparsity_prior == "Spikeandslab-Enumeration":
        raise NotImplementedError()

    elif sparsity_prior == "Spikeandslab-Lasso":
        raise NotImplementedError()

    elif sparsity_prior == "Horseshoe":

        def prior_sample():
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

        return prior_sample

    elif sparsity_prior == "Lasso":
        # TODO: Add source paper
        # Approximation to the Laplace density with a SoftLaplace,
        # see https://docs.pyro.ai/en/stable/_modules/pyro/distributions/softlaplace.html#SoftLaplace
        #
        # Unlike the Laplace distribution, this distribution is infinitely differentiable everywhere
        def prior_sample():
            return pyro.sample(
                "w_scale", dist.SoftLaplace(torch.tensor(0.0), torch.tensor(1.0))
            ).view(view_shape)

        return prior_sample

    elif sparsity_prior == "Nonnegative":

        def prior_sample():
            return pyro.sample(
                "w_scale", dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
            ).view(view_shape)

        return prior_sample
