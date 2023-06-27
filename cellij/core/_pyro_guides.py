import logging

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.infer.autoguide.guides import deep_getattr, deep_setattr
from pyro.nn import PyroModule, PyroParam

logger = logging.getLogger(__name__)


class Guide(PyroModule):
    def __init__(
        self,
        model,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
    ):
        super().__init__(name="Guide")
        self.model = model
        self.locs = PyroModule()
        self.scales = PyroModule()

        self.init_loc = init_loc
        self.init_scale = init_scale
        self.device = model.device

        self.site_to_shape = {}
        self.sample_dict = {}

        self.setup_shapes()
        self.setup_sites()

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

    def setup_shapes(self):
        """Setup parameters and sampling sites."""
        return self.site_to_shape

    def _setup_site(self, site_name, shape):
        deep_setattr(
            self.locs,
            site_name,
            PyroParam(
                self.init_loc * torch.ones(shape, device=self.device),
                constraints.real,
            ),
        )
        deep_setattr(
            self.scales,
            site_name,
            PyroParam(
                self.init_scale * torch.ones(shape, device=self.device),
                constraints.softplus_positive,
            ),
        )

    def setup_sites(self):
        """Setup parameters and sampling sites."""
        for site_name, shape in self.site_to_shape.items():
            self._setup_site(site_name, shape)

    @torch.no_grad()
    def mode(self):
        """Get the MAP estimates.

        Returns
        -------
        torch.Tensor
            MAP estimate
        """
        modes = {}
        for site_name, _ in self.site_to_shape.items():
            mode, scale = self._get_loc_and_scale(site_name)
            # TODO: This is a hack, but it works for now. Register
            # sample sites with a log-normal distribution before and pull it here
            not_z = site_name != "z"
            not_w = "w_" not in site_name
            not_nonnegative = isinstance(self, NonnegativeGuide)
            if not_w and not_z and not_nonnegative:
                mode = (mode - scale.pow(2)).exp()
            modes[site_name] = mode.clone()

        return modes

    @torch.no_grad()
    def _get_map_estimate(self, param_name: str):
        param = self.mode(param_name)
        param = param.cpu().detach().numpy()
        if param_name == "sigma":
            param = param[None, :]
        return param

    def get_z(self):
        """Get the factor scores."""
        return self._get_map_estimate("z")

    def get_w(self):
        """Get the factor loadings."""
        return self._get_map_estimate("w")

    def get_sigma(self):
        """Get the marginal feature scales."""
        return self._get_map_estimate("sigma")

    def _sample_site(
        self,
        site_name,
        dist,
    ):
        loc, scale = self._get_loc_and_scale(site_name)
        samples = pyro.sample(site_name, dist(loc, scale))
        self.sample_dict[site_name] = samples
        return samples

    def _sample_normal(self, site_name: str):
        return self._sample_site(site_name, dist.Normal)

    def _sample_log_normal(self, site_name: str):
        return self._sample_site(site_name, dist.LogNormal)

    def sample_latent(self):
        return None

    def sample_feature_group(self, feature_group: str = None):
        return None

    def sample_feature_group_factor(self, feature_group: str = None):
        return None

    def sample_weight(self, feature_group: str = None):
        return None

    def sample_feature(self, feature_group: str = None):
        return None

    def forward(
        self,
        data: torch.Tensor,
    ):
        """Approximate posterior."""
        plates = self.model.get_plates()

        with plates["factor"], plates["sample"]:
            self.sample_latent()

        for feature_group, _ in self.model.feature_dict.items():
            self.sample_feature_group(feature_group=feature_group)
            with plates["factor"]:
                self.sample_feature_group_factor(feature_group=feature_group)
                with plates[f"feature_{feature_group}"]:
                    self.sample_weight(feature_group=feature_group)

            with plates[f"feature_{feature_group}"]:
                self.sample_feature(feature_group=feature_group)

        return self.sample_dict


class NormalGuide(Guide):
    def __init__(
        self, model, init_loc: float = 0, init_scale: float = 0.1, device=None, gp = None, covariate = None
    ):
        super().__init__(model, init_loc, init_scale, device)
        self.gp = gp
        self.covariate = covariate

    def setup_shapes(self):
        """Setup parameters and sampling sites."""
        self.site_to_shape["z"] = self.model.get_latent_shape()[1:]

        for feature_group in self.model.feature_dict.keys():
            self.site_to_shape[f"w_{feature_group}"] = self.model.get_weight_shape(
                feature_group
            )[1:]
            self.site_to_shape[f"sigma_{feature_group}"] = self.model.get_feature_shape(
                feature_group
            )[1:]

        return super().setup_shapes()

    def sample_z(self):
        if self.gp is None:
            return self._sample_normal("z")
        else:
            pyro.sample(
                    "z",
                    self.gp.pyro_guide(self.covariate),
            )

    def sample_w(self, feature_group=None):
        return self._sample_normal(f"w_{feature_group}")

    def sample_sigma(self, feature_group=None):
        return self._sample_log_normal(f"sigma_{feature_group}")

    def sample_latent(self):
        return self.sample_z()

    def sample_weight(self, feature_group: str = None):
        return self.sample_w(feature_group=feature_group)

    def sample_feature(self, feature_group: str = None):
        return self.sample_sigma(feature_group=feature_group)


class HorseshoeGuide(NormalGuide):
    def __init__(self, model, init_loc: float = 0, init_scale: float = 0.1):
        super().__init__(model, init_loc, init_scale)

    def setup_shapes(self):
        for feature_group, _ in self.model.feature_dict.items():
            self.site_to_shape[
                f"tau_{feature_group}"
            ] = self.model.get_feature_group_shape()[1:]
            self.site_to_shape[
                f"theta_{feature_group}"
            ] = self.model.get_factor_shape()[1:]
            self.site_to_shape[f"caux_{feature_group}"] = self.model.get_weight_shape(
                feature_group
            )[1:]
            self.site_to_shape[f"lambda_{feature_group}"] = self.model.get_weight_shape(
                feature_group
            )[1:]
        return super().setup_shapes()

    def sample_tau(self, feature_group=None):
        if not self.model.delta_tau:
            self._sample_log_normal(f"tau_{feature_group}")

    def sample_theta(self, feature_group=None):
        self._sample_log_normal(f"theta_{feature_group}")

    def sample_caux(self, feature_group=None):
        self._sample_log_normal(f"caux_{feature_group}")

    def sample_lambda(self, feature_group=None):
        self._sample_log_normal(f"lambda_{feature_group}")

    def sample_feature_group(self, feature_group: str = None):
        return self.sample_tau(feature_group=feature_group)

    def sample_feature_group_factor(self, feature_group: str = None):
        if self.model.ard:
            return self.sample_theta(feature_group=feature_group)
        return super().sample_feature_group_factor(feature_group=feature_group)

    def sample_weight(self, feature_group: str = None):
        self.sample_lambda(feature_group=feature_group)
        if self.model.regularized:
            self.sample_caux(feature_group=feature_group)
        return super().sample_weight(feature_group=feature_group)


class NonnegativeGuide(NormalGuide):
    def __init__(self, model, init_loc: float = 0, init_scale: float = 0.1):
        super().__init__(model, init_loc, init_scale)

    def sample_w(self, feature_group=None):
        return self._sample_log_normal(f"w_{feature_group}")
