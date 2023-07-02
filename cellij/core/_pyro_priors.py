import logging

import numpy as np
import torch
from pyro.distributions import HalfCauchy, Normal, TorchDistribution
from torch.distributions import constraints

logger = logging.getLogger(__name__)


class Horseshoe(TorchDistribution):
    arg_constraints = {"scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, scale, *, validate_args=None):
        self.scale = scale
        self._half_cauchy = HalfCauchy(torch.ones_like(self.scale))
        self._normal = Normal(torch.zeros_like(self.scale), torch.ones_like(self.scale))
        super().__init__(self.scale.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Horseshoe, _instance)
        Horseshoe.__init__(
            new, scale=self.scale.expand(torch.Size(batch_shape)), validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        xx = (value / self.scale) ** 2 / 2
        g = 0.5614594835668851  # tf.exp(-0.5772156649015328606)
        b = 1.0420764938351215  # tf.sqrt(2 * (1-g) / (g * (2-g)))
        h_inf = 1.0801359952503342  #  (1-g)*(g*g-6*g+12) / (3*g * (2-g)**2 * b)
        q = 20.0 / 47.0 * xx**1.0919284281983377
        h = 1.0 / (1 + xx ** (1.5)) + h_inf * q / (1 + q)
        c = -0.5 * np.log(2 * np.pi**3) - torch.log(g * self.scale)
        z = np.log1p(-g) - np.log(g)
        softplus_bij = torch.nn.Softplus()
        return (
            -softplus_bij(z - xx / (1 - g))
            + torch.log(torch.log1p(g / xx - (1 - g) / (h + b * xx) ** 2))
            + c
        )

    def rsample(self, sample_shape=torch.Size()):
        local_shrinkage = self._half_cauchy.sample(sample_shape)
        shrinkage = self.scale * local_shrinkage
        sampled = self._normal.sample(sample_shape)
        return sampled * shrinkage

    def cdf(self, value):
        raise NotImplementedError()

    @property
    def mean(self):
        return torch.zeros_like(self.scale)

    @property
    def mode(self):
        # TODO: really?
        return self.mean()
