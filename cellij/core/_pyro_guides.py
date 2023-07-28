import logging

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.infer.autoguide.guides import deep_getattr, deep_setattr
from pyro.nn import PyroModule, PyroParam

logger = logging.getLogger(__name__)


class Q(PyroModule):
    def __init__(self, prior, init_loc: float = 0.0, init_scale: float = 0.1):
        super().__init__("Q")

        self.locs = PyroModule()
        self.scales = PyroModule()

        self.prior = prior
        self.init_loc = init_loc
        self.init_scale = init_scale
        self.device = prior.device
        self.to(self.device)

        self.setup_sites()

        self.sample_dict = {}

    def _set_loc_and_scale(self, site_name: str, site_shape):
        deep_setattr(
            self.locs,
            site_name,
            PyroParam(
                self.init_loc * torch.ones(site_shape, device=self.device),
                constraints.real,
            ),
        )
        deep_setattr(
            self.scales,
            site_name,
            PyroParam(
                self.init_scale * torch.ones(site_shape, device=self.device),
                constraints.softplus_positive,
            ),
        )

    def _get_loc_and_scale(self, site_name: str):
        """Get loc and scale parameters.

        Parameters
        ----------
        site_name : str
            Name of the sampling site

        Returns
        -------
        tuple
            Tuple of (loc, scale)
        """
        site_loc = deep_getattr(self.locs, site_name)
        site_scale = deep_getattr(self.scales, site_name)
        return site_loc, site_scale

    def _sample(self, site_name, dist):
        loc, scale = self._get_loc_and_scale(site_name)
        self.sample_dict[site_name] = pyro.sample(site_name, dist(loc, scale))
        return self.sample_dict[site_name]

    def _sample_normal(self, site_name: str):
        return self._sample(site_name, dist.Normal)

    def _sample_log_normal(self, site_name: str):
        return self._sample(site_name, dist.LogNormal)

    def setup_shapes(self):
        for site_name, site_samples in self.prior.sample_dict.items():
            self.site_to_shape[site_name] = site_samples.shape

    def setup_sites(self):
        for site_name, site_samples in self.prior.sample_dict.items():
            self._set_loc_and_scale(site_name, site_samples.shape)

    def sample_global(self):
        return None

    def sample_inter(self):
        return None

    def _mean_normal(self, site_name):
        return self.locs[site_name]

    def _median_normal(self, site_name):
        return self._mean_normal(site_name)

    def _mode_normal(self, site_name):
        return self._mean_normal(site_name)

    def _mean_log_normal(self, site_name):
        loc, scale = self._get_loc_and_scale(site_name)
        return (loc + scale.pow(2) / 2).exp()

    def _median_log_normal(self, site_name):
        loc, scale = self._get_loc_and_scale(site_name)
        return loc.exp()

    def _mode_log_normal(self, site_name):
        loc, scale = self._get_loc_and_scale(site_name)
        return (loc - scale.square()).exp()

    @torch.no_grad()
    def mean(self):
        return None

    @torch.no_grad()
    def median(self):
        return None

    @torch.no_grad()
    def mode(self):
        return None

    def forward(self):
        return None


class InverseGammaQ(Q):
    def __init__(self, prior, init_loc: float = 0, init_scale: float = 0.1):
        super().__init__(prior, init_loc, init_scale)

    @torch.no_grad()
    def mean(self):
        return self._mean_log_normal(self.prior.site_name)

    @torch.no_grad()
    def median(self):
        return self._median_log_normal(self.prior.site_name)

    @torch.no_grad()
    def mode(self):
        return self._mode_log_normal(self.prior.site_name)

    def forward(self):
        return self._sample_log_normal(self.prior.site_name)


class NormalQ(Q):
    def __init__(self, prior, init_loc: float = 0, init_scale: float = 0.1):
        super().__init__(prior, init_loc, init_scale)

    @torch.no_grad()
    def mean(self):
        return self._mean_normal(self.prior.site_name)

    @torch.no_grad()
    def median(self):
        return self._median_normal(self.prior.site_name)

    @torch.no_grad()
    def mode(self):
        return self._mode_normal(self.prior.site_name)

    def forward(self):
        return self._sample_normal(self.prior.site_name)


class LaplaceQ(NormalQ):
    def __init__(self, prior, init_loc: float = 0, init_scale: float = 0.1):
        super().__init__(prior, init_loc, init_scale)


class HorseshoeQ(NormalQ):
    def __init__(self, prior, init_loc: float = 0, init_scale: float = 0.1):
        super().__init__(prior, init_loc, init_scale)

    def sample_global(self):
        if hasattr(self.prior, "tau_delta"):
            return None
        return self._sample_log_normal(self.prior.tau_site_name)

    def sample_inter(self):
        if not self.prior.ard:
            return None
        return self._sample_log_normal(self.prior.thetas_site_name)

    def forward(self):
        self._sample_log_normal(self.prior.lambdas_site_name)
        if self.prior.regularized:
            self._sample_log_normal(self.prior.caux_site_name)
        return self._sample_normal(self.prior.site_name)


class Guide(PyroModule):
    def __init__(self, model):
        super().__init__("Guide")
        self.model = model
        self.device = model.device
        self.to(self.device)

        # dry run to setup shapes
        # call before setting up q_dists
        self.model()

        self.sigma_q_dists = self._get_q_dists(self.model.sigma_priors)
        self.factor_q_dists = self._get_q_dists(self.model.factor_priors)
        self.weight_q_dists = self._get_q_dists(self.model.weight_priors)

        self.sample_dict = {}

    def _get_q_dists(self, priors):
        _q_dists = {}

        for group, prior in priors.items():
            # Replace strings with actuals priors
            _q_dists[group] = {
                "InverseGamma": InverseGammaQ,
                "Normal": NormalQ,
                "Laplace": LaplaceQ,
                "Horseshoe": HorseshoeQ,
                # "SpikeAndSlab": SpikeAndSlabQ,
            }[prior._pyro_name](prior=prior)

        return _q_dists

    def forward(self, data: torch.Tensor = None):
        """Approximate posterior."""
        plates = self.model.get_plates()

        for obs_group, factor_q_dist in self.factor_q_dists.items():
            factor_q_dist.sample_global()
            with plates["factor"]:
                factor_q_dist.sample_inter()
                with plates[f"obs_{obs_group}"]:
                    self.sample_dict[
                        self.model.factor_priors[obs_group].site_name
                    ] = factor_q_dist()

        for feature_group, weight_q_dist in self.weight_q_dists.items():
            weight_q_dist.sample_global()
            with plates["factor"]:
                weight_q_dist.sample_inter()
                with plates[f"feature_{feature_group}"]:
                    self.sample_dict[
                        self.model.weight_priors[feature_group].site_name
                    ] = weight_q_dist()

            with plates[f"feature_{feature_group}"]:
                self.sample_dict[
                    self.model.sigma_priors[feature_group].site_name
                ] = self.sigma_q_dists[feature_group]()

        return self.sample_dict
