import logging
import os
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union

import anndata
import muon
import numpy as np
import pandas
import pyro
import torch
from pyro.infer import SVI
from pyro.nn import PyroModule
from tqdm import trange
import pandas as pd

import cellij
from cellij.core._data import DataContainer
from cellij.core._pyro_guides import Guide
from cellij.core._pyro_models import Generative
from cellij.core.utils import EarlyStopper

logger = logging.getLogger(__name__)


class FactorModel(PyroModule):
    """Base class for all estimators in Cellij.

    Attributes
    ----------
    model : cellij.model
        The generative model
    guide : Union[str, pyro.infer.autoguide.initialization, cellij.guide]
        The variational distribution
    n_factors : int
        The number of factors
    dtype : torch.dtype, default = torch.float32
        The data type of the model
    device : str, default = "cpu"
        The device on which the model is run


    Methods
    -------
    add_data(name, data, **kwargs)
        Adds data to the model
    set_data(name, data, **kwargs)
        Overwrites data with the same name
    remove_data(name, **kwargs)
        Removes a data from the model by its name

    add_feature_group(name, features, **kwargs)
        Delegates to _add_group(..., level = 'feature')
    set_feature_group(name, features, **kwargs)
        Delegates to _set_group(..., level = 'feature')
    remove_feature_group(name, **kwargs)
        Delegates to _remove_group(..., level = 'feature')

    add_obs_group(name, features, **kwargs)
        Delegates to _add_group(..., level = 'obs')
    set_obs_group(name, features, **kwargs)
        Delegates to _set_group(..., level = 'obs')
    remove_obs_group(name, **kwargs)
        Delegates to _remove_group(..., level = 'obs')

    _add_group(name, group, level, **kwargs)
        Adds a group to the model
    _set_group(name, group, level, **kwargs)
        Overwrites a group with the same name
    _remove_group(name, level, **kwargs)
        Removes a group from the model by its name

    fit(dry_run=False, **kwargs)

    """

    def __init__(
        self,
        # model,
        # guide,
        n_factors,
        dtype: torch.dtype = torch.float32,
        device="cpu",
        **kwargs,
    ):
        super().__init__(name="FactorModel")

        # self._model = model
        # self._guide, self._guide_kwargs = self._setup_guide(guide, kwargs)
        self._n_factors = n_factors
        self._dtype = dtype

        self.device = self._setup_device(device)
        self.to(self.device)

        self._data = DataContainer()
        self._is_trained = False
        self._feature_groups = {}
        self._obs_groups = {}

        self._data_options: dict[str, Any] = {}
        self._model_options: dict[str, Any] = {}
        self._training_options: dict[str, Any] = {}

        # Save kwargs for later
        self._kwargs = kwargs

    def __repr__(self):

        self._init_from_config(
            data_options=self._data_options,
            model_options=self._model_options,
            training_options=self._training_options,
        )

        output = []
        output.append(f"FactorModel(n_factors={self.n_factors})")

        output.append("├─ data")
        if len(self._data._names) == 0:
            output.append("│  └─ no data added yet")
        else:
            for name, adata in self._data.feature_groups.items():
                branch_char = "├" if name != list(self._data.feature_groups.keys())[-1] else "└"
                output.append(f"│  {branch_char}─ {name}: {adata.n_obs} observations × {adata.n_vars} features")
                output.append(f"│     ├ likelihood: {self._model_options['likelihoods'][name]}")
                output.append(f"│     └ weight_prior: {self._model_options['weight_priors'][name]}")

        output.append("├─ groups")        
        if len(self._data._names) == 0:
            output.append("│  └─ no data added yet")
        else:
            for name, obs in self.obs_groups.items():
                branch_char = "├" if name != list(self.obs_groups.keys())[-1] else "└"
                output.append(f"│  {branch_char}─ {name}: {len(obs)} observations")
                output.append(f"│     └ factor_prior: {self._model_options['factor_priors'][name]}")

        if self._model_options["covariates"] is not None:
            output.append("├─ covariates")
            
            output.append(f"│  └─ {self._model_options['covariates'].shape[1]}D covariate with {self._model_options['covariates'].shape[0]} observations")

        output.append("└─ config")

        output.append(f"   ├─ data options")
        for key, value in self._data_options.items():
            branch_char = "├" if key != list(self._data_options.keys())[-1] else "└"
            output.append(f"   │  {branch_char}─ {key}: {value}")

        output.append(f"   └─ training options")
        for key, value in self._training_options.items():
            branch_char = "├" if key != list(self._training_options.keys())[-1] else "└"
            output.append(f"      {branch_char}─ {key}: {value}")

        return "\n".join(output)

    @property
    def n_factors(self):
        return self._n_factors

    @n_factors.setter
    def n_factors(self, n_factors):
        self._n_factors = n_factors

    @property
    def dtype(self):
        pass

    @dtype.setter
    def dtype(self, dtype):
        pass

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, *args):
        raise AttributeError(
            "Use `add_data()`, `set_data` or `remove_data()` to modify this property."
        )

    @property
    def is_trained(self):
        return self._is_trained

    @is_trained.setter
    def is_trained(self, *args):
        raise AttributeError("This property is read-only.")

    @property
    def feature_groups(self):
        return {
            group_name: list(group.var_names)
            for group_name, group in self.data._feature_groups.items()
        }

    @feature_groups.setter
    def feature_groups(self, *args):
        raise AttributeError(
            "Use `add_feature_group()`, `set_feature_group` or `remove_feature_group()` "
            "to modify this property."
        )

    @property
    def model(self):
        return self._model[0]

    @property
    def guide(self):
        return self._guide[0]

    @property
    def obs_groups(self):
        try:
            groups = self._model_options["groups"]
        except KeyError:
            groups = None

        if groups is None:
            return {"all_observations": self._data._merged_obs_names}
        elif isinstance(groups, dict):
            return groups

    @obs_groups.setter
    def obs_groups(self, *args):
        raise AttributeError(
            "Use `add_obs_group()`, `set_obs_group` or `remove_obs_group()` to modify this property."
        )

    def _setup_guide(self, guide, kwargs):
        if isinstance(guide, str):
            # Implement some default guides
            if guide == "AutoDelta":
                guide = pyro.infer.autoguide.AutoDelta
            if guide == "AutoNormal":
                guide = pyro.infer.autoguide.AutoNormal
            if guide == "AutoLowRankMultivariateNormal":
                guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal

            # TODO: return proper kwargs
            return guide, {}

        guide_kwargs = {}
        # TODO: implement init_loc_fn instead of init_loc
        for arg in ["init_loc", "init_scale"]:
            if arg in kwargs:
                guide_kwargs[arg] = kwargs[arg]

        if issubclass(guide, cellij.core._pyro_guides.Guide):
            logger.info("Using custom guide.")
            return guide, guide_kwargs

        raise ValueError(f"Unknown guide: {guide}")

    def _setup_device(self, device):
        cuda_available = torch.cuda.is_available()

        try:
            mps_available = torch.backends.mps.is_available()
        except AttributeError:
            mps_available = False

        device = str(device).lower()
        if ("cuda" in device and not cuda_available) or (
            device == "mps" and not mps_available
        ):
            logger.warning(f"`{device}` not available...")
            device = "cpu"

        logger.info(f"Running all computations on `{device}`.")
        return torch.device(device)

    @property
    def data_options(self):
        return self._data_options

    @data_options.setter
    def data_options(self, *args):
        raise AttributeError("Use `set_data_options()` to modify this property.")

    def set_data_options(
        self,
        scale_views: bool = False,
        scale_groups: bool = False,
        center_groups: bool = True,
        preview: bool = False,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(scale_views, bool):
            raise TypeError(
                f"Parameter 'scale_views' must be bool, got '{type(scale_views)}'."
            )

        if not isinstance(scale_groups, bool):
            raise TypeError(
                f"Parameter 'scale_groups' must be bool, got '{type(scale_groups)}'."
            )

        if not isinstance(center_groups, bool):
            raise TypeError(
                f"Parameter 'center_groups' must be bool, got '{type(center_groups)}'."
            )

        if not isinstance(preview, bool):
            raise TypeError(f"Parameter 'preview' must be bool, got '{type(preview)}'.")

        options = {
            "scale_views": scale_views,
            "scale_features": scale_groups,
            "center_features": center_groups,
        }

        if not preview:
            self._data_options = options
            return None

        return options

    @property
    def model_options(self):
        return self._model_options

    @model_options.setter
    def model_options(self, *args):
        raise AttributeError("Use `set_model_options()` to modify this property.")

    def set_model_options(
        self,
        likelihoods: Union[str, dict[str, str]] = "Normal",
        factor_priors: Optional[Union[str, dict[str, str]]] = None,
        weight_priors: Optional[Union[str, dict[str, str]]] = None,
        groups: Optional[dict[str, Iterable]] = None,
        regress_out: Optional[pd.DataFrame] = None,
        covariates: Optional[pd.DataFrame] = None,
        preview: bool = False,
    ) -> Optional[dict[str, Any]]:
        if isinstance(likelihoods, str):
            likelihoods = {view: likelihoods for view in self.feature_groups}
        elif isinstance(likelihoods, dict) and not all(
            key in self._data._names for key in likelihoods
        ):
            raise ValueError(
                "All views must have a likelihood set.\n"
                + "  - Actual: "
                + ", ".join(likelihoods.keys())
                + "\n"
                + "  - Expected: "
                + ", ".join(self._data._names)
            )

        for name, distribution in likelihoods.items():
            if isinstance(distribution, str):
                # Replace likelihood string with common synonyms used in Pyro
                distribution = distribution.title()
                if distribution == "Gaussian":
                    distribution = "Normal"

                try:
                    likelihoods[name] = distribution
                except AttributeError as e:
                    raise AttributeError(
                        f"Could not find valid Pyro distribution for {distribution}."
                    ) from e

        groups = self.obs_groups if groups is None else groups

        if factor_priors is None:
            factor_priors = {group: "Normal" for group in groups}
        elif isinstance(factor_priors, str):
            factor_priors = {view: factor_priors for view in groups}
        elif isinstance(factor_priors, dict) and not all(
            key in groups for key in factor_priors
        ):
            raise ValueError(
                "When manually specifying 'factor_priors', all views must be assigned a prior.\n"
                + "  - Actual: "
                + ", ".join(factor_priors.keys())
                + "\n"
                + "  - Expected: "
                + ", ".join(groups)
            )

        for name, prior in factor_priors.items():
            if isinstance(prior, str):
                # Replace likelihood string with common synonyms used in Pyro
                prior = prior.title()
                if prior == "Gaussian":
                    prior = "Normal"

                try:
                    factor_priors[name] = prior
                except AttributeError as e:
                    raise AttributeError(
                        f"Could not find valid prior for '{prior}'."
                    ) from e

        if weight_priors is None:
            weight_priors = "Normal"

        if isinstance(weight_priors, str):
            weight_priors = {group: weight_priors for group in self.feature_groups}
        elif isinstance(weight_priors, dict) and not all(
            prior_group in self.feature_groups for prior_group in weight_priors
        ):
            raise ValueError(
                "When manually specifying 'weight_priors', all groups must be assigned a prior.\n"
                + "  - Actual: "
                + ", ".join(weight_priors.keys())
                + "\n"
                + "  - Expected: "
                + ", ".join(self.feature_groups)
            )

        for name, prior in weight_priors.items():
            if isinstance(prior, str):
                # prior = prior.lower()
                if prior == "Gaussian":
                    prior = "Normal"

                try:
                    weight_priors[name] = prior
                except AttributeError as e:
                    raise AttributeError(
                        f"Could not find valid prior for '{prior}'."
                    ) from e

        groups = self.obs_groups if groups is None else groups
        
        #TODO: Implement logic for `regress_out`
        
        if covariates is not None:
            if not isinstance(covariates, (pd.DataFrame, pd.Series)):
                raise TypeError(
                    f"Parameter 'covariates' must be pd.DataFrame, got '{type(covariates)}'."
                )
            
            if (covariates.reset_index().duplicated()).any():
                raise ValueError(
                    f"Parameter 'covariates' contains duplicate columns."
                )
                
            for group in factor_priors.keys():
                factor_priors[group] = "GaussianProcess"

        options = {
            "likelihoods": likelihoods,
            "factor_priors": factor_priors,
            "weight_priors": weight_priors,
            "covariates": covariates,
            "groups": groups,
        }

        if not preview:
            self._model_options = options
            return None

        return options

    @property
    def training_options(self):
        return self._training_options

    @training_options.setter
    def training_options(self, *args):
        raise AttributeError("Use `set_training_options()` to modify this property.")

    def set_training_options(
        self,
        early_stopping: bool = True,
        verbose_epochs: int = 100,
        patience: int = 500,
        min_delta: float = 0.1,
        percentage: bool = True,
        scale_gradients: bool = True,
        optimizer: str = "ClippedAdam",
        num_particles: int = 1,
        learning_rate: float = 0.003,
        preview: bool = False,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(early_stopping, bool):
            raise TypeError(
                f"Parameter 'early_stopping' must be bool, got '{type(early_stopping)}'."
            )

        if not isinstance(verbose_epochs, int):
            raise TypeError(
                f"Parameter 'verbose_epochs' must be int, got '{type(verbose_epochs)}'."
            )

        if not isinstance(patience, int):
            raise TypeError(
                f"Parameter 'patience' must be int, got '{type(patience)}'."
            )

        if not isinstance(min_delta, float):
            raise TypeError(
                f"Parameter 'min_delta' must be float, got '{type(min_delta)}'."
            )

        if not isinstance(percentage, bool):
            raise TypeError(
                f"Parameter 'percentage' must be bool, got '{type(percentage)}'."
            )

        if not isinstance(scale_gradients, bool):
            raise TypeError(
                f"Parameter 'scale_gradients' must be bool, got '{type(scale_gradients)}'."
            )

        if not isinstance(optimizer, str):
            raise TypeError(
                f"Parameter 'optimizer' must be str, got '{type(optimizer)}'."
            )

        valid_optimizer = ["adam", "clippedadam"]
        if optimizer.lower() not in valid_optimizer:
            raise NotImplementedError(
                "Currently, only the following optimizers are supported: "
                + ", ".join(valid_optimizer)
            )

        if not isinstance(num_particles, int):
            raise TypeError(
                f"Parameter 'num_particles' must be int, got '{type(num_particles)}'."
            )

        if not isinstance(learning_rate, float):
            raise TypeError(
                f"Parameter 'learning_rate' must be float, got '{type(learning_rate)}'."
            )

        if not isinstance(preview, bool):
            raise TypeError(f"Parameter 'preview' must be bool, got '{type(preview)}'.")

        for param_name, param_value in zip(
            [
                "verbose_epochs",
                "patience",
                "min_delta",
                "num_particles",
                "learning_rate",
            ],
            [verbose_epochs, patience, min_delta, num_particles, learning_rate],
        ):
            if param_value <= 0:
                raise ValueError(
                    f"Parameter '{param_name}' must be positive, got '{param_value}'."
                )

        options = {
            "early_stopping": early_stopping,
            "verbose_epochs": verbose_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "percentage": percentage,
            "scale_gradients": scale_gradients,
            "optimizer": optimizer,
            "num_particles": num_particles,
            "learning_rate": learning_rate,
        }

        if not preview:
            self._training_options = options
            return None

        return options

    def _init_from_config(
        self,
        data_options: dict[str, Any],
        model_options: dict[str, Any],
        training_options: dict[str, Any],
    ) -> None:
        data_option_defaults = self.set_data_options(preview=True)
        model_option_defaults = self.set_model_options(preview=True)
        training_option_defaults = self.set_training_options(preview=True)

        for key in data_option_defaults:
            if key not in data_options.keys():
                self._data_options[key] = data_option_defaults[key]

        for key in model_option_defaults:
            if key not in model_options.keys():
                self._model_options[key] = model_option_defaults[key]

        for key in training_option_defaults:
            if key not in training_options.keys():
                self._training_options[key] = training_option_defaults[key]

    def add_data(
        self,
        data: Union[pandas.DataFrame, anndata.AnnData, muon.MuData],
        name: Optional[str] = None,
        merge: bool = True,
        **kwargs,
    ):
        # TODO: Add a check that no name is "all"
        valid_types = (pandas.DataFrame, anndata.AnnData, muon.MuData)
        metadata = None

        if not isinstance(data, valid_types):
            raise TypeError(
                f"Expected data to be one of {valid_types}, got {type(data)}."
            )

        if not isinstance(data, muon.MuData) and not isinstance(name, str):
            raise ValueError(
                "When adding data that is not a MuData object, a name must be provided."
            )

        if isinstance(data, pandas.DataFrame):
            data = anndata.AnnData(data)
            self._add_data(data=data, name=name)

        elif isinstance(data, anndata.AnnData):
            self._add_data(data=data, name=name)

        elif isinstance(data, muon.MuData):
            if not data.obs.empty:
                metadata = data.obs

            # call again for each anndata contained, but non-merging
            for modality_name, anndata_object in data.mod.items():
                if metadata is not None:
                    anndata_object.obs = anndata_object.obs.merge(
                        metadata, how="left", left_index=True, right_index=True
                    )

                self.add_data(name=modality_name, data=anndata_object, merge=False)

        if merge:
            self._data.merge_data(**kwargs)

    def _add_data(
        self,
        data: anndata.AnnData,
        name: str,
    ):
        self._data.add_data(data=data, name=name)

    def add_feature_group(self, name, features, **kwargs):
        self._add_group(name=name, features=features, level="feature", **kwargs)

    def set_feature_group(self, name, features, **kwargs):
        self._set_group(name=name, features=features, level="feature", **kwargs)

    def remove_feature_group(self, name, **kwargs):
        self._remove_group(name=name, level="feature", **kwargs)

    def add_obs_group(self, name, features, **kwargs):
        self._add_group(name=name, features=features, level="obs", **kwargs)

    def set_obs_group(self, name, features, **kwargs):
        self._set_group(name=name, features=features, level="obs", **kwargs)

    def remove_obs_group(self, name, **kwargs):
        self._remove_group(name=name, level="obs", **kwargs)

    def _add_group(self, name, group, level, **kwargs):
        if name in self._feature_groups or name in self._obs_groups:
            raise ValueError(f"A group with the name {name} already exists.")

        if level == "feature":
            self._feature_groups[name] = group
        elif level == "obs":
            self._obs_groups[name] = group
        else:
            raise ValueError(f"Level must be 'feature' or 'obs', not {level}")

    def _set_group(self, name, group, level, **kwargs):
        if level == "feature":
            self._feature_groups[name] = group
        elif level == "obs":
            self._obs_groups[name] = group
        else:
            raise ValueError(f"Level must be 'feature' or 'obs', not {level}")

    def _remove_group(self, name, level, **kwargs):
        if (
            name not in self._feature_groups.keys()
            and name not in self._obs_groups.keys()
        ):
            raise ValueError(f"No group with the name {name} exists.")

        if level == "feature":
            del self._feature_groups[name]
        elif level == "obs":
            del self._obs_groups[name]
        else:
            raise ValueError(f"Level must be 'feature' or 'obs', not {level}")

    def _get_from_param_storage(
        self,
        name: str,
        param: str = "locs",
        views: Optional[Union[str, list[str]]] = "all",
        groups: Optional[Union[str, list[str]]] = "all",
        moment: str = "median",
        format: str = "numpy",
    ) -> np.ndarray:
        """Pull a parameter from the pyro parameter storage.

        Parameters
        ----------
        name : str
            The name of the parameter to be pulled.
        moment : str
            The moment statistic to be pulled.
        format : str
            The format in which the parameter should be returned.
            Options are: "numpy", "torch".

        Returns
        -------
        parameter : dict
            The parameters pulled from the pyro parameter storage.
            Key Value Pairs are view name : variational param
        """
        if not isinstance(name, str):
            raise TypeError("Parameter 'name' must be of type str.")

        if not isinstance(param, str):
            raise TypeError("Parameter 'param' must be of type str.")

        if param not in ["locs", "scales"]:
            raise ValueError("Parameter 'param' must be in ['locs', 'scales'].")

        if views is None and groups is None:
            raise ValueError("Parameters 'views' and 'groups' cannot both be None.")

        if views is not None:
            if not isinstance(views, (str, list)):
                raise TypeError("Parameter 'views' must be of type str or list.")

            if isinstance(views, list) and not all(
                isinstance(view, str) for view in views
            ):
                raise TypeError("Parameter 'views' must be a list of strings.")

        if groups is not None:
            if not isinstance(groups, (str, list)):
                raise TypeError("Parameter 'groups' must be of type str or list.")

            if isinstance(groups, list) and not all(
                isinstance(view, str) for view in groups
            ):
                raise TypeError("Parameter 'groups' must be a list of strings.")

        if not isinstance(format, str):
            raise TypeError("Parameter 'format' must be of type str.")

        if format not in ["numpy", "torch"]:
            raise ValueError("Parameter 'format' must be in ['numpy', 'torch'].")

        results = {}

        if name == "weights":
            if views == "all":
                views = self.data._names
            elif isinstance(views, str):
                if views not in self.data._names:
                    raise ValueError(
                        f"Parameter 'views' must be in {list(self.data._names)}."
                    )
                views = [views]
            elif isinstance(views, list) and not all(
                [view in self.data._names for view in views]
            ):
                raise ValueError(
                    f"All elements in 'views' must be in {list(self.data._names)}."
                )

            for view in views:
                q_dist = self._guide[0].weight_q_dists[view]
                if moment == "median":
                    results[view] = q_dist.median().squeeze()
                elif moment == "mean":
                    results[view] = q_dist.mean().squeeze()
                elif moment == "mode":
                    results[view] = q_dist.mode().squeeze()
                else:
                    raise NotImplementedError(
                        "Parameter 'moment' must be in ['median', 'mean', 'mode']."
                    )

        elif name == "factors":
            if groups == "all":
                groups = list(self._model_options["groups"].keys())
            elif isinstance(groups, str):
                if groups not in list(self._model_options["groups"].keys()):
                    raise ValueError(
                        f"Parameter 'views' must be in {list(self._model_options['groups'].keys())}."
                    )
                groups = [groups]
            elif isinstance(groups, list) and not all(
                [
                    group in list(self._model_options["groups"].keys())
                    for group in groups
                ]
            ):
                raise ValueError(
                    f"All elements in 'views' must be in {list(list(self._model_options['groups'].keys()))}."
                )

            for group in groups:
                q_dist = self._guide[0].factor_q_dists[group]
                if moment == "median":
                    results[group] = q_dist.median().squeeze()
                elif moment == "mean":
                    results[group] = q_dist.mean().squeeze()
                elif moment == "mode":
                    results[group] = q_dist.mode().squeeze()
                else:
                    raise NotImplementedError(
                        "Parameter 'moment' must be in ['median', 'mean', 'mode']."
                    )

        elif name == "residuals":
            raise NotImplementedError()

        else:
            raise NotImplementedError()

        if format == "numpy":
            for k, v in results.items():
                if v.is_cuda:
                    v = v.cpu()
                if isinstance(v, torch.Tensor):
                    results[k] = v.detach().numpy()

        return results

    def get_weights(
        self,
        views: Union[str, list[str]] = "all",
        moment: str = "median",
        format: str = "numpy",
    ):
        """Return a dictionary of the weight estimates for each requested view.

        Parameters
        ----------
        views : Union[str, list[str]], default = "all"
            The views for which the weights should be returned.
        moment : str, default = "median"
            The moment statistic to be returned.
        format : str, default = "numpy"
            The format in which the weights should be returned.
            Options are: "numpy", "torch".

        Returns
        -------
        weights : dict
            The weights for each requested view.
        """
        return self._get_from_param_storage(
            name="weights",
            param="locs",
            views=views,
            groups=None,
            moment=moment,
            format=format,
        )

    def get_factors(
        self,
        groups: Union[str, list[str]] = "all",
        moment: str = "median",
        format: str = "numpy",
    ):
        """Return a dictionary of the factor estimates for each requested group.

        Parameters
        ----------
        groups : Union[str, list[str]], default = "all"
            The groups for which the weights should be returned.
        moment : str, default = "median"
            The moment statistic to be returned.
        format : str, default = "numpy"
            The format in which the weights should be returned.
            Options are: "numpy", "torch".

        Returns
        -------
        weights : dict
            The weights for each requested view.
        """
        return self._get_from_param_storage(
            name="factors",
            param="locs",
            views=None,
            groups=groups,
            moment=moment,
            format=format,
        )

    def fit(
        self,
        epochs: int = 1000,
        verbose: int = 1,
    ):
        # Clear pyro param
        pyro.clear_param_store()

        # Initialize model from config dicts
        self._init_from_config(
            data_options=self._data_options,
            model_options=self._model_options,
            training_options=self._training_options,
        )

        if len(self.feature_groups) == 0:
            raise ValueError("No data has been added yet.")

        obs_dict = {
            f"{name}": len(obs_names) for name, obs_names in self.obs_groups.items()
        }
        feature_dict = {
            f"{name}": len(var_names) for name, var_names in self.feature_groups.items()
        }

        if self._model_options["covariates"] is not None:
            covariates = torch.tensor(self._model_options["covariates"].values, dtype=self._dtype)
        else:
            covariates = None

        model = Generative(
            n_factors=self.n_factors,
            obs_dict=obs_dict,
            feature_dict=feature_dict,
            likelihoods=self._model_options["likelihoods"],
            factor_priors=self._model_options["factor_priors"],
            weight_priors=self._model_options["weight_priors"],
            covariates=covariates,
            device=self.device,
        )
        guide = Guide(model)
        model = (model,)
        guide = (guide,)

        self._model = model
        self._guide = guide

        # If early stopping is set, check if it is a valid value
        if self._training_options["early_stopping"]:
            earlystopper = EarlyStopper(
                patience=self._training_options["patience"],
                min_delta=self._training_options["min_delta"],
                percentage=self._training_options["percentage"],
            )
        else:
            earlystopper = None

        # Checks
        if self._data is None:
            raise ValueError("No data set.")

        # We scale the gradients by the number of total samples to allow a better comparison across
        # models/datasets
        scaling_constant = 1.0
        if self._training_options["scale_gradients"]:
            scaling_constant = 1.0 / self._data._values.shape[1]

        if self._training_options["optimizer"].lower() == "adam":
            optim = pyro.optim.Adam(
                {"lr": self._training_options["learning_rate"], "betas": (0.95, 0.999)}
            )
        elif self._training_options["optimizer"].lower() == "clippedadam":
            gamma = 0.1
            lrd = gamma ** (1 / epochs)
            optim = pyro.optim.ClippedAdam(
                {"lr": self._training_options["learning_rate"], "lrd": lrd}
            )

        svi = SVI(
            model=pyro.poutine.scale(self.model, scale=scaling_constant),
            guide=pyro.poutine.scale(self.guide, scale=scaling_constant),
            optim=optim,
            loss=pyro.infer.Trace_ELBO(
                retain_graph=True,
                num_particles=self._training_options["num_particles"],
                vectorize_particles=True,
            ),
        )

        # get indicies for each view and group and subset merged data with them
        obs_idx = {
            group_name: [self._data._merged_obs_names.index(obs) for obs in obs_list]
            for group_name, obs_list in self.obs_groups.items()
        }
        feature_idx = {
            view_name: [
                self._data._merged_feature_names.index(feature)
                for feature in feature_list
            ]
            for view_name, feature_list in self.feature_groups.items()
        }
        data = {
            group_name: {
                view_name: torch.tensor(
                    [
                        [self._data._values[i][j] for j in feature_idx[view_name]]
                        for i in obs_idx[group_name]
                    ]
                )
                for view_name in feature_idx
            }
            for group_name in obs_idx
        }

        self.train_loss_elbo = []
        loss: int = 0

        with trange(
            epochs,
            unit="epoch",
            miniters=self._training_options["verbose_epochs"],
            maxinterval=99999,
        ) as pbar:
            pbar.set_description("Training")
            for i in pbar:
                loss = svi.step(data=data, covariate=covariates)
                self.train_loss_elbo.append(loss)

                if self._training_options["early_stopping"] and earlystopper.step(loss):
                    logger.info(
                        f"Early stopping of training due to convergence at step {i}"
                    )
                    break

                if i % self._training_options["verbose_epochs"] == 0:
                    if i == 0:
                        decrease_pct = 0.0
                    else:
                        decrease_pct = (
                            100
                            - 100
                            * self.train_loss_elbo[i]
                            / self.train_loss_elbo[
                                i - self._training_options["verbose_epochs"]
                            ]
                        )

                    pbar.set_postfix(
                        loss=f"{loss:>14.2f}", decrease=f"{decrease_pct:>6.2f} %"
                    )

        self._is_trained = True

    def save(self, filename: str, overwrite: bool = False):
        if not self._is_trained:
            raise ValueError("Model must be trained before saving.")

        if not isinstance(filename, str):
            raise ValueError(
                f"Parameter 'filename' must be a string, got {type(filename)}."
            )

        # Verify that the user used a file ending
        _, file_ending = os.path.splitext(filename)

        if file_ending == "":
            raise ValueError(
                "No file ending provided. Please provide a file ending such as '.pkl'."
            )

        file_name = filename.replace(".pkl", ".state_dict")
        if Path(file_name).exists() and not overwrite:
            raise ValueError(
                f"File {filename} already exists. Set 'overwrite' to True to overwrite the file."
            )

        # AnnData causes some issues with old Pytorch versions
        self._data = None

        with open(filename, "wb") as f:
            pickle.dump(self, f)

        torch.save(self.state_dict(), filename.replace(".pkl", ".state_dict"))
