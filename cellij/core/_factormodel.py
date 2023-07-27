import logging
import os
import pickle
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional, Union

import anndata
import gpytorch
import muon
import numpy as np
import pandas as pd
import pyro
import torch
from pyro.infer import SVI
from pyro.nn import PyroModule

import cellij
from cellij.core._data import DataContainer
from cellij.core.utils_training import EarlyStopper

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class FactorModel(PyroModule, gpytorch.Module):
    """Base class for all estimators in cellij.

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
        self._covariate = None

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

    @property
    def covariate(self):
        return self._covariate

    @covariate.setter
    def covariate(self, *args):
        raise AttributeError("Use `add_covariate()` to modify this property.")

    def add_covariate(
        self,
        covariate: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
        num_inducing_points: int = 100,
    ):
        """
        Add a covariate to the model, replacing any existing covariate if necessary.

        Parameters
        ----------
        covariate : Any
            The covariate to be added to the model. Should be a 1D or 2D matrix-like object.
            If 2D, it must not have more than 2 columns.
        num_inducing_points : int, optional
            The number of inducing points to keep. Default is 100.

        Raises
        ------
        TypeError
            If the 'covariate' is not a matrix-like object.
        ValueError
            If the 'covariate' is not a 1D or 2D matrix, or if it is a 2D matrix with more than 2 columns.

        Attributes updated
        ------------------
        _covariate : torch.Tensor
            The covariate data stored as a PyTorch tensor.
        _inducing_points : torch.Tensor
            Unique values from the covariate data, stored as a PyTorch tensor.
        """
        if self.covariate is not None:
            logger.info(
                "Currently, only one covariate is supported. Overwriting existing covariate."
            )
            self.covariate = None

        if not isinstance(
            covariate, (np.ndarray, pd.DataFrame, pd.Series, torch.Tensor)
        ):
            raise TypeError(
                f"Parameter 'covariate' must be a matrix-like object, got {type(covariate)}."
            )

        if isinstance(covariate, np.ndarray):
            covariate = pd.DataFrame(covariate.tolist())
        elif isinstance(covariate, torch.Tensor):
            covariate = pd.DataFrame(covariate.numpy().tolist())
        elif isinstance(covariate, pd.Series):
            covariate = covariate.to_frame()

        covariate_shape_len = len(covariate.shape)
        try:
            if covariate_shape_len == 1:
                rows = len(covariate)
                cols = 1
            elif covariate_shape_len == 2:
                rows, cols = covariate.shape
            else:
                raise ValueError(
                    f"Parameter 'covariate' must be a 1D or 2D matrix, got shape '{covariate.shape}'."
                )
        except AttributeError as e:
            raise TypeError(
                f"Paramter 'covariate' must be a matrix-like object, got {type(covariate)}."
            ) from e

        if cols == 0:
            raise ValueError(
                f"Parameter 'covariate' must have at least one column, got {cols}."
            )

        if cols > 2:
            raise ValueError(
                f"Parameter 'covariate' must have 1 or 2 columns, got {cols}."
            )

        self._covariate = torch.Tensor(covariate.values)

        if cols == 1:
            unique_points = covariate.drop_duplicates().values
            if len(unique_points) > num_inducing_points:
                unique_points = unique_points[
                    rng.choice(
                        len(unique_points), size=num_inducing_points, replace=False
                    )
                ]
            self._inducing_points = torch.Tensor(unique_points)

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

    def add_data(
        self,
        data: Union[pd.DataFrame, anndata.AnnData, muon.MuData],
        name: Optional[str] = None,
        merge: bool = True,
        **kwargs,
    ):
        # TODO: Add a check that no name is "all"
        valid_types = (pd.DataFrame, anndata.AnnData, muon.MuData)
        metadata = None

        if not isinstance(data, valid_types):
            raise TypeError(
                f"Expected data to be one of {valid_types}, got {type(data)}."
            )

        if not isinstance(data, muon.MuData) and not isinstance(name, str):
            raise ValueError(
                "When adding data that is not a MuData object, a name must be provided."
            )

        if isinstance(data, pd.DataFrame):
            data = anndata.AnnData(
                X=data.values,
                obs=pd.DataFrame(data.index),
                var=pd.DataFrame(data.columns),
                dtype=self._dtype,
            )

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

    def remove_data(self, name, **kwargs):
        pass

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
                [isinstance(view, str) for view in views]
            ):
                raise TypeError("Parameter 'views' must be a list of strings.")

        if groups is not None:
            if not isinstance(groups, (str, list)):
                raise TypeError("Parameter 'groups' must be of type str or list.")

            if isinstance(groups, list) and not all(
                [isinstance(view, str) for view in groups]
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
                        f"Parameter '{key}' not found in parameter storage. ",
                        f"Available choices are: {', '.join(list(pyro.get_param_store().keys()))}",
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
        likelihoods: Union[str, dict],
        epochs: int = 1000,
        num_particles: int = 1,
        learning_rate: float = 0.003,
        optimizer: str = "Clipped",
        verbose_epochs: int = 100,
        early_stopping: bool = True,
        patience: int = 500,
        min_delta: float = 0.1,
        percentage: bool = True,
        scale_gradients: bool = True,
        center_features: bool = True,
        scale_features: bool = False,
        scale_views: bool = False,
    ):
        # Clear pyro param
        pyro.clear_param_store()

        # If early stopping is set, check if it is a valid value
        if early_stopping:
            if min_delta < 0:
                raise ValueError("min_delta must be positive.")
            earlystopper = EarlyStopper(
                patience=patience, min_delta=min_delta, percentage=percentage
            )
        else:
            earlystopper = None

        # Checks
        if self._data is None:
            raise ValueError("No data set.")

        if not isinstance(likelihoods, (str, dict)):
            raise ValueError(
                f"Parameter 'likelihoods' must either be a string or a dictionary "
                f"mapping the modalities to strings, got {type(likelihoods)}."
            )

        for arg_name, arg in zip(
            [
                "early_stopping",
                "scale_gradients",
                "center_features",
                "scale_features",
                "scale_views",
            ],
            [
                early_stopping,
                scale_gradients,
                center_features,
                scale_features,
                scale_views,
            ],
        ):
            if not isinstance(arg, bool):
                raise TypeError(f"Parameter '{arg_name}' must be of type bool.")

        # Prepare likelihoods
        # TODO: If string is passed, check if string corresponds to a valid pyro distribution
        # TODO: If custom distribution is passed, check if it provides arg_constraints parameter

        # If user passed strings, replace the likelihood strings with the actual distributions
        if isinstance(likelihoods, str):
            likelihoods = {
                modality: likelihoods for modality in self._data.feature_groups
            }

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

        # Raise error if likelihoods are not set for all modalities
        if len(likelihoods.keys()) != len(self._data.feature_groups):
            raise ValueError(
                f"Likelihoods must be set for all modalities. Got {len(likelihoods.keys())} likelihood "
                f"and {len(self._data.feature_groups)} data modalities."
            )

        # Provide data information to generative model
        feature_dict = {
            k: len(feature_idx) for k, feature_idx in self._data._feature_idx.items()
        }
        data_dict = {
            k: torch.tensor(self._data._values[:, feature_idx], device=self.device)
            for k, feature_idx in self._data._feature_idx.items()
        }

        if self.covariate is not None:
            self.gp = cellij.core._gp.PseudotimeGP(
                inducing_points=self._inducing_points, n_factors=self.n_factors
            )

            self._kwargs["gp"] = self.gp
            self._kwargs["covariate"] = self.covariate

        # Initialize class objects with correct data-related parameters
        self._model = self._model(
            n_samples=self._data._values.shape[0],
            n_factors=self.n_factors,
            feature_dict=feature_dict,
            likelihoods=None,
            device=self.device,
            **self._kwargs,
        )

        for key, value in self._kwargs.items():
            if key in ["init_loc", "init_scale"]:
                self._guide_kwargs[key] = value

        if self.covariate is not None:
            self._guide_kwargs["gp"] = self.gp
            self._guide_kwargs["covariate"] = self.covariate

        self._guide = self._guide(self._model, **self._guide_kwargs)

        if not isinstance(likelihoods, (str, dict)):
            raise ValueError(
                "Parameter 'likelihoods' must either be a string or a dictionary "
                f"mapping the modalities to strings, got {type(likelihoods)}."
            )

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
        if scale_gradients:
            scaling_constant = 1.0 / self._data.values.shape[1]

        optim = pyro.optim.Adam({"lr": learning_rate, "betas": (0.95, 0.999)})
        if optimizer.lower() == "clipped":
            gamma = 0.1
            lrd = gamma ** (1 / epochs)
            optim = pyro.optim.ClippedAdam({"lr": learning_rate, "lrd": lrd})

        svi = SVI(
            model=pyro.poutine.scale(self._model, scale=scaling_constant),
            guide=pyro.poutine.scale(self._guide, scale=scaling_constant),
            optim=optim,
            loss=pyro.infer.Trace_ELBO(
                retain_graph=True,
                num_particles=num_particles,
                vectorize_particles=True,
            ),
        )

        # TOOD: Preprocess data
        data = data_dict
        # data = self._model.values

        self.train_loss_elbo = []
        time_start = timer()
        verbose_time_start = time_start
        print("Training Model...")
        for i in range(epochs + 1):
            loss = svi.step(data=data)
            self.train_loss_elbo.append(loss)

            if early_stopping and earlystopper.step(loss):
                print(f"Early stopping of training due to convergence at step {i}")
                break

            if i % verbose_epochs == 0:
                log = f"- Epoch {i:>6}/{epochs} | Train Loss: {loss:>14.2f} \t"
                if i >= 1:
                    decrease_pct = (
                        100
                        - 100
                        * self.train_loss_elbo[i]
                        / self.train_loss_elbo[i - verbose_epochs]
                    )
                    log += f"| Decrease: {decrease_pct:>6.2f}%\t"
                    log += f"| Time: {(timer() - verbose_time_start):>6.2f}s"

                verbose_time_start = timer()

                print(log)

        self._is_trained = True
        print("Training finished.")
        print(f"- Final loss: {loss:.2f}")
        print(f"- Training took {(timer() - time_start):.2f}s")

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
