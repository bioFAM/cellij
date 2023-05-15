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
        device=None,
    ):
        super().__init__(name="Guide")
        self.model = model
        self.locs = PyroModule()
        self.scales = PyroModule()

        self.init_loc = init_loc
        self.init_scale = init_scale
        self.device = device

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
        self.site_to_shape["z"] = self.model.get_z_shape()[1:]

        for feature_group, _ in self.model.feature_dict.items():
            self.site_to_shape[f"w_{feature_group}"] = self.model.get_w_shape(
                feature_group
            )[1:]
            self.site_to_shape[f"sigma_{feature_group}"] = self.model.get_sigma_shape(
                feature_group
            )[1:]

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
            if "sigma" in site_name or "w_scale" in site_name:
                mode = (mode - scale.pow(2)).exp()
            modes[site_name] = mode.clone()

        return modes

    @torch.no_grad()
    def _get_map_estimate(self, param_name: str, as_list: bool):
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

    def sample_z(self, site_name="z"):
        return self._sample_normal(site_name)

    def sample_w(self, site_name="w", feature_group=None):
        return self._sample_normal(f"{site_name}_{feature_group}")

    def sample_w_positive(self, site_name="w", feature_group=None):
        return self._sample_log_normal(f"{site_name}_{feature_group}")

    def sample_tau(self, site_name="tau", feature_group=None):
        return None

    def sample_sigma(self, site_name="sigma", feature_group=None):
        return self._sample_log_normal(f"{site_name}_{feature_group}")

    def forward(
        self,
        data: torch.Tensor,
    ):
        """Approximate posterior."""
        plates = self.model.get_plates()

        with plates["obs"], plates["factors"]:
            self.sample_z()

        for feature_group, _ in self.model.feature_dict.items():
            self.sample_tau(feature_group=feature_group)
            with plates["factors"], plates[f"features_{feature_group}"]:
                self.sample_w(feature_group=feature_group)

            with plates[f"features_{feature_group}"]:
                self.sample_sigma(feature_group=feature_group)

        return self.sample_dict


class NormalGuide(Guide):
    def __init__(
        self, model, init_loc: float = 0, init_scale: float = 0.1, device=None
    ):
        super().__init__(model, init_loc, init_scale, device)


class HorseshoeGuide(Guide):
    def __init__(
        self, model, init_loc: float = 0, init_scale: float = 0.1, device=None
    ):
        super().__init__(model, init_loc, init_scale, device)

    def setup_shapes(self):
        for feature_group, _ in self.model.feature_dict.items():
            self.site_to_shape[f"tau_{feature_group}"] = self.model.get_tau_shape()[1:]
            self.site_to_shape[f"caux_{feature_group}"] = self.model.get_w_shape(
                feature_group
            )[1:]
            self.site_to_shape[f"lambda_{feature_group}"] = self.model.get_w_shape(
                feature_group
            )[1:]
        return super().setup_shapes()

    def sample_tau(self, site_name="tau", feature_group=None):
        if self.model.delta_tau:
            return None
        self._sample_log_normal(f"{site_name}_{feature_group}")

    def sample_caux(self, site_name="caux", feature_group=None):
        self._sample_log_normal(f"{site_name}_{feature_group}")

    def sample_lambda(self, site_name="lambda", feature_group=None):
        self._sample_log_normal(f"{site_name}_{feature_group}")

    def sample_w(self, site_name="w", feature_group=None):
        self.sample_lambda(feature_group=feature_group)
        if self.model.regularized:
            self.sample_caux(feature_group=feature_group)
        return super().sample_w(site_name, feature_group)


class HorseshoePlusGuide(Guide):
    def __init__(
        self, model, init_loc: float = 0, init_scale: float = 0.1, device=None
    ):
        super().__init__(model, init_loc, init_scale, device)

    def setup_shapes(self):
        for feature_group, _ in self.model.feature_dict.items():
            self.site_to_shape[f"tau_{feature_group}"] = self.model.get_tau_shape()[1:]
            self.site_to_shape[f"eta_{feature_group}"] = self.model.get_w_shape(
                feature_group
            )[1:]
            self.site_to_shape[f"lambda_{feature_group}"] = self.model.get_w_shape(
                feature_group
            )[1:]
        return super().setup_shapes()

    def sample_tau(self, site_name="tau", feature_group=None):
        return None

    def sample_eta(self, site_name="eta", feature_group=None):
        self._sample_log_normal(f"{site_name}_{feature_group}")

    def sample_lambda(self, site_name="lambda", feature_group=None):
        self._sample_log_normal(f"{site_name}_{feature_group}")

    def sample_w(self, site_name="w", feature_group=None):
        self.sample_eta(feature_group=feature_group)
        self.sample_lambda(feature_group=feature_group)
        return super().sample_w(site_name, feature_group)


class LassoGuide(Guide):
    def __init__(
        self, model, init_loc: float = 0, init_scale: float = 0.1, device=None
    ):
        super().__init__(model, init_loc, init_scale, device)


class NonnegativityGuide(Guide):
    def __init__(
        self, model, init_loc: float = 0, init_scale: float = 0.1, device=None
    ):
        super().__init__(model, init_loc, init_scale, device)

    def sample_w(self, site_name="w", feature_group=None):
        return super().sample_w_positive(site_name, feature_group)
