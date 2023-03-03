import torch
import numpy as np
from torch.distributions import constraints
from torch.nn.functional import softplus

from pyro.distributions import TorchDistribution, Normal, HalfCauchy


class Horseshoe(TorchDistribution):
    arg_constraints = {"scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, scale: float = 1.0, *, validate_args=None):  # feature_offsets, n_factors,
        # self.n_factors = n_factors
        # self.feature_offsets = feature_offsets

        # tensor_size = (1,1)  # (self.n_factors, self.feature_offsets[-1])
        self.scale = scale  #  * torch.ones(tensor_size)
        self._half_cauchy = HalfCauchy(self.scale)  # Lambda
        self._normal = Normal(torch.zeros_like(self.scale), torch.ones_like(self.scale))
        super().__init__(self.scale.shape, validate_args=validate_args)

    # def expand(self, batch_shape, _instance=None):
    #     new = self._get_checked_instance(Horseshoe, _instance)
    #     batch_shape = torch.Size(batch_shape)
    #     new.scale = self.scale.expand(batch_shape)
    #     super(Horseshoe, new).__init__(batch_shape, validate_args=False)
    #     new._validate_args = self._validate_args
    #     return new

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # The exact HalfCauchy-Normal marginal log-density is analytically
        # intractable; we compute a (relatively accurate) numerical
        # approximation. This is a log space version of ref[2] from class docstring.
        xx = (value / self.scale) ** 2 / 2
        g = 0.5614594835668851  # torch.exp(-0.5772156649015328606)
        b = 1.0420764938351215  # torch.sqrt(2 * (1-g) / (g * (2-g)))
        h_inf = 1.0801359952503342  #  (1-g)*(g*g-6*g+12) / (3*g * (2-g)**2 * b)
        q = 20.0 / 47.0 * xx**1.0919284281983377
        h = 1.0 / (1 + xx ** (1.5)) + h_inf * q / (1 + q)
        c = -0.5 * np.log(2 * np.pi**3) - torch.log(g * self.scale)
        z = np.log1p(-g) - np.log(g)
        eps = 1e-8
        return -softplus(z - xx / (1 - g)) + torch.log(torch.log1p(g / xx - (1 - g) / (h + b * xx) ** 2) + eps) + c

    def rsample(self, sample_shape=torch.Size()):
        local_shrinkage = self._half_cauchy.sample(sample_shape)
        shrinkage = self.scale * local_shrinkage
        sampled = self._normal.sample(sample_shape)
        return sampled * shrinkage

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def cdf(self, value):
        raise NotImplementedError()

    @property
    def mean(self):
        return torch.zeros_like(self.scale)

    @property
    def mode(self):
        # really?!
        return self.mean()
