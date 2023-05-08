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
        
        self.sample_dict = {}

        self.setup()

    def _get_loc_and_scale(self, name: str):
        """Get loc and scale parameters.

        Parameters
        ----------
        name : str
            Name of the sampling site

        Returns
        -------
        tuple
            Tuple of (loc, scale)
        """
        site_loc = deep_getattr(self.locs, name)
        site_scale = deep_getattr(self.scales, name)
        return site_loc, site_scale

    def setup(self):
        """Setup parameters and sampling sites."""
        n_factors = self.model.n_factors
        
        # TODO: maybe we can read this from the model?
        site_to_shape = {
            "z": (self.model.n_samples, n_factors, 1),
        }
        
        for feature_group, n_features in self.model.feature_dict.items():
            site_to_shape[f"w_{feature_group}"] = (n_factors, n_features)
            site_to_shape[f"w_scale_{feature_group}"] = (n_factors, n_features)
            site_to_shape[f"sigma_{feature_group}"] = n_features

        for name, shape in site_to_shape.items():
            deep_setattr(
                self.locs,
                name,
                PyroParam(
                    self.init_loc * torch.ones(shape, device=self.device),
                    constraints.real,
                ),
            )
            deep_setattr(
                self.scales,
                name,
                PyroParam(
                    self.init_scale * torch.ones(shape, device=self.device),
                    constraints.softplus_positive,
                ),
            )

    @torch.no_grad()
    def mode(self, name: str):
        """Get the MAP estimates.

        Parameters
        ----------
        name : str
            Name of the sampling site

        Returns
        -------
        torch.Tensor
            MAP estimate
        """
        loc, scale = self._get_loc_and_scale(name)
        mode = loc
        # if name not in ["z", "w"]:
        # TODO! This is a hack, but it works for now
        if 'sigma' in name or 'w_scale' in name:
            mode = (loc - scale.pow(2)).exp()
        return mode.clone()

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

    def _sample_normal(self, name: str, index=None):
        # duplicate code below, might update later
        loc, scale = self._get_loc_and_scale(name)
        if index is not None:
            # indices = indices.to(self.device)
            loc = loc.index_select(0, index)
            scale = scale.index_select(0, index)
        return pyro.sample(name, dist.Normal(loc, scale))

    def _sample_log_normal(self, name: str, index=None):
        loc, scale = self._get_loc_and_scale(name)
        if index is not None:
            loc = loc.index_select(0, index)
            scale = scale.index_select(0, index)
        return pyro.sample(name, dist.LogNormal(loc, scale))

    def forward(
        self,
        data: torch.Tensor,
    ):
        """Approximate posterior."""

        plates = self.model.get_plates()
        
        with plates["obs"], plates["factors"]:
            self.sample_dict['z'] = self._sample_normal("z")

        for feature_group, _ in self.model.feature_dict.items():
            with plates["factors"], plates[f"features_{feature_group}"]:
                self.sample_dict[f"w_scale_{feature_group}"] = self._sample_log_normal(f"w_scale_{feature_group}")
                self.sample_dict[f"w_{feature_group}"] = self._sample_normal(f"w_{feature_group}")

            with plates[f"features_{feature_group}"]:
                site_name = f"sigma_{feature_group}"
                self.sample_dict[site_name] = self._sample_log_normal(site_name)
                
        return self.sample_dict