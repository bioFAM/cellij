import logging
import os
import pickle
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

import cellij
from cellij.core._data import DataContainer
from cellij.core.utils_training import EarlyStopper

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
        model,
        guide,
        n_factors,
        dtype: torch.dtype = torch.float32,
        device="cpu",
        **kwargs,
    ):
        super().__init__(name="FactorModel")

        self._model = model
        self._guide, self._guide_kwargs = self._setup_guide(guide, kwargs)
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

    @property
    def model(self):
        pass

    @model.setter
    def model(self, model):
        pass

    @property
    def guide(self):
        pass

    @guide.setter
    def guide(self, guide):
        self._guide = guide

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
        return self._feature_groups

    @feature_groups.setter
    def feature_groups(self, *args):
        raise AttributeError(
            "Use `add_feature_group()`, `set_feature_group` or `remove_feature_group()` "
            "to modify this property."
        )

    @property
    def obs_groups(self):
        return self._obs_groups

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
        scale_features: bool = False,
        center_features: bool = True,
        preview: bool = False,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(scale_views, bool):
            raise TypeError(
                f"Parameter 'scale_views' must be bool, got '{type(scale_views)}'."
            )

        if not isinstance(scale_features, bool):
            raise TypeError(
                f"Parameter 'scale_features' must be bool, got '{type(scale_features)}'."
            )

        if not isinstance(center_features, bool):
            raise TypeError(
                f"Parameter 'center_features' must be bool, got '{type(center_features)}'."
            )

        if not isinstance(preview, bool):
            raise TypeError(f"Parameter 'preview' must be bool, got '{type(preview)}'.")

        options = {
            "scale_views": scale_views,
            "scale_features": scale_features,
            "center_features": center_features,
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
        likelihoods: Optional[Union[str, dict[str, str]]] = None,
    ) -> None:
        if likelihoods is not None:
            if isinstance(likelihoods, str):
                likelihoods = {
                    modality: likelihoods for modality in self._data.feature_groups
                }
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
                    # Replace likelihood string with common synonyms and correct for align with Pyro
                    distribution = distribution.title()
                    if distribution == "Gaussian":
                        distribution = "Normal"

                    try:
                        likelihoods[name] = getattr(pyro.distributions, distribution)
                    except AttributeError as e:
                        raise AttributeError(
                            f"Could not find valid Pyro distribution for {distribution}."
                        ) from e

        options = {
            "likelihoods": likelihoods,
        }

        self._model_options = options

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
        optimizer: str = "Clipped",
        num_particles: int = 1,
        learning_rate: float = 0.003,
        preview: bool = False,
    ) -> None:
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

        valid_optimizer = ["Adam", "Clipped"]
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
        self._data(data=data, name=name)

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
        format: str = "numpy",
    ) -> np.ndarray:
        """Pull a parameter from the pyro parameter storage.

        TODO: Get all parameters, but in a dict.
        TODO: Add full support for group selection.
        TODO: Add torch.FloatTensor return type hint

        Parameters
        ----------
        name : str
            The name of the parameter to be pulled.
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
                (isinstance(view, str) for view in views)
            ):
                raise TypeError("Parameter 'views' must be a list of strings.")

        if groups is not None:
            if not isinstance(groups, (str, list)):
                raise TypeError("Parameter 'groups' must be of type str or list.")

            if isinstance(groups, list) and not all(
                (isinstance(view, str) for view in groups)
            ):
                raise TypeError("Parameter 'groups' must be a list of strings.")

        if not isinstance(format, str):
            raise TypeError("Parameter 'format' must be of type str.")

        if format not in ["numpy", "torch"]:
            raise ValueError("Parameter 'format' must be in ['numpy', 'torch'].")

        if views is not None:
            result = {}

            if views == "all":
                views = self.data._names
            elif isinstance(views, str):
                if views not in self.data._names:
                    raise ValueError(
                        f"Parameter 'views' must be in {list(self.data._names)}."
                    )
            elif isinstance(views, list) and not all(
                [view in self.data._names for view in views]
            ):
                raise ValueError(
                    f"All elements in 'views' must be in {list(self.data._names)}."
                )

            for view in views:
                key = "FactorModel._guide." + param + "." + name
                if name == "w":
                    key += "_" + view

                if key not in list(pyro.get_param_store().keys()):
                    raise ValueError(
                        f"Parameter '{key}' not found in parameter storage. "
                        f"Available choices are: {', '.join(list(pyro.get_param_store().keys()))}"
                    )

                data = pyro.get_param_store()[key]

                result[view] = data.squeeze()

        if format == "numpy":
            for k, v in result.items():
                if v.is_cuda:
                    v = v.cpu()
                if isinstance(v, torch.Tensor):
                    result[k] = v.detach().numpy()

        return result

    def get_weights(self, views: Union[str, list[str]] = "all", format: str = "numpy"):
        return self._get_from_param_storage(
            name="w", param="locs", views=views, groups=None, format=format
        )

    def get_factors(self, groups: Union[str, list[str]] = "all", format: str = "numpy"):
        return self._get_from_param_storage(
            name="z", param="locs", views=None, groups=groups, format=format
        )

    def fit(
        self,
        epochs: int = 1000,
        # likelihoods: Union[str, dict],
        # num_particles: int = 1,
        # # learning_rate: float = 0.003,
        # optimizer: str = "Clipped",
        # verbose_epochs: int = 100,
        # early_stopping: bool = True,
        # patience: int = 500,
        # min_delta: float = 0.1,
        # percentage: bool = True,
        # scale_gradients: bool = True,
        # center_features: bool = True,
        # scale_features: bool = False,
        # scale_views: bool = False,
        sample_groups: str = None,
    ):
        # Clear pyro param
        pyro.clear_param_store()

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

        # Prepare likelihoods
        # TODO: If string is passed, check if string corresponds to a valid pyro distribution
        # TODO: If custom distribution is passed, check if it provides arg_constraints parameter

        # Provide data information to generative model
        feature_dict = {
            k: len(feature_idx) for k, feature_idx in self._data._feature_idx.items()
        }
        data_dict = {
            k: torch.tensor(self._data._values[:, feature_idx], device=self.device)
            for k, feature_idx in self._data._feature_idx.items()
        }

        # Initialize class objects with correct data-related parameters
        if not self._is_trained:
            self._model = self._model(
                n_samples=self._data._values.shape[0],
                n_factors=self.n_factors,
                feature_dict=feature_dict,
                likelihoods=None,
                device=self.device,
                **self._kwargs,
            )

            self._guide = self._guide(self._model, **self._guide_kwargs)

        # # Provide data information to generative model
        # self._model._setup(
        #     data=self._data,
        #     likelihoods=likelihoods,
        #     center_features=center_features,
        #     scale_features=scale_features,
        #     scale_views=scale_views,
        # )

        # Prepare likelihoods
        # TODO: If string is passed, check if string corresponds to a valid pyro distribution
        # TODO: If custom distribution is passed, check if it provides arg_constraints parameter

        # We scale the gradients by the number of total samples to allow a better comparison across
        # models/datasets
        scaling_constant = 1.0
        if self._training_options["scale_gradients"]:
            scaling_constant = 1.0 / self._data.values.shape[1]

        if self._training_options["optimizer"].lower() == "adam":
            optim = pyro.optim.Adam(
                {"lr": self._training_options["learning_rate"], "betas": (0.95, 0.999)}
            )
        elif self._training_options["optimizer"].lower() == "clipped":
            gamma = 0.1
            lrd = gamma ** (1 / epochs)
            optim = pyro.optim.ClippedAdam(
                {"lr": self._training_options["learning_rate"], "lrd": lrd}
            )

        svi = SVI(
            model=pyro.poutine.scale(self._model, scale=scaling_constant),
            guide=pyro.poutine.scale(self._guide, scale=scaling_constant),
            optim=optim,
            loss=pyro.infer.Trace_ELBO(
                retain_graph=True,
                num_particles=self._training_options["num_particles"],
                vectorize_particles=True,
            ),
        )

        # TOOD: Preprocess data
        data = data_dict
        # data = self._model.values

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
                loss = svi.step(data=data)
                self.train_loss_elbo.append(loss)

                if self._training_options["early_stopping"] and earlystopper.step(loss):
                    logger.warning(
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
