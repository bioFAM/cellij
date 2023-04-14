import pyro.distributions as dist

DISTRIBUTIONS_PROPERTIES = {
    dist.torch.Normal: {
        "params": 2,
        "constraints": ["real", "positive"],
    },
    dist.torch.Poisson: {
        "params": 1,
        "constraints": ["positive"],
    },
    dist.torch.Bernoulli: {
        "params": 1,
        "constraints": ["unit_interval"],
    },
    dist.torch.Gamma: {
        "params": 2,
        "constraints": ["positive", "positive"],
    },
}
