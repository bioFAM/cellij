import os
import pickle
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Optional, Union

import anndata
import muon
import numpy as np
import pandas
import pyro
import torch
from pyro.infer import SVI
from pyro.nn import PyroModule

import cellij
from cellij.core._data import DataContainer
from cellij.core.utils_training import EarlyStopper


class FactorModel(PyroModule):
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
        self._n_factors = n_factors
        self._dtype = dtype
        self._device = device
        self._data = DataContainer()
        self._is_trained = False
        self._feature_groups = {}
        self._obs_groups = {}

        # # Setup
        if isinstance(guide, str):
            # Implement some default guides
            guide_args = {}
            if "init_loc_fn" in kwargs:
                guide_args["init_loc_fn"] = kwargs["init_loc_fn"]

            if guide == "AutoDelta":
                self._guide = pyro.infer.autoguide.AutoDelta  # type: ignore
            elif guide == "AutoNormal":
                if "init_scale" in kwargs:
                    guide_args["init_scale"] = kwargs["init_scale"]
                self._guide = pyro.infer.autoguide.AutoNormal  # type: ignore
            elif guide == "AutoLowRankMultivariateNormal":
                if "init_scale" in kwargs:
                    guide_args["init_scale"] = kwargs["init_scale"]
                if "rank" in kwargs:
                    guide_args["rank"] = kwargs["rank"]
                self._guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal  # type: ignore
        elif isinstance(guide, pyro.infer.autoguide.AutoGuide):  # type: ignore
            self._guide = guide
        elif issubclass(guide, cellij.core._pyro_guides.Guide):
            print("Using custom guide.")
            self._guide = guide
        else:
            raise ValueError(f"Unknown guide: {guide}")
        
        self.model_kwargs = {k:v for k,v in kwargs.items() if k.startswith('model_')}
        self.guide_kwargs = {k:v for k,v in kwargs.items() if k.startswith('guide_')}
        # remove model_ and guide_ from kwargs
        # for k in self.model_kwargs.keys():
        #     del kwargs[k]
        # for k in self.guide_kwargs.keys():
        #     del kwargs[k]

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
        pass

    @device.setter
    def device(self, device):
        pass

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, *args):
        raise AttributeError("Use `add_data()`, `set_data` or `remove_data()` to modify this property.")

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
            "Use `add_feature_group()`, `set_feature_group` or `remove_feature_group()` " "to modify this property."
        )

    @property
    def obs_groups(self):
        return self._obs_groups

    @obs_groups.setter
    def obs_groups(self, *args):
        raise AttributeError("Use `add_obs_group()`, `set_obs_group` or `remove_obs_group()` to modify this property.")

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
            raise TypeError(f"Expected data to be one of {valid_types}, got {type(data)}.")

        if not isinstance(data, muon.MuData) and not isinstance(name, str):
            raise ValueError("When adding data that is not a MuData object, a name must be provided.")

        if isinstance(data, pandas.DataFrame):
            data = anndata.AnnData(
                X=data.values,
                obs=pandas.DataFrame(data.index),
                var=pandas.DataFrame(data.columns),
                dtype=self._dtype,  # type: ignore
            )

        elif isinstance(data, anndata.AnnData):
            self._add_data(data=data, name=name)  # type: ignore

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
        if name in self._feature_groups.keys() or name in self._obs_groups.keys():
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
        if name not in self._feature_groups.keys() and name not in self._obs_groups.keys():
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
        views: Optional[Union[str, List[str]]] = "all",
        groups: Optional[Union[str, List[str]]] = "all",
        format: str = "numpy",
    ) -> np.ndarray:
        """Pulls a parameter from the pyro parameter storage.

        TODO: Get all parameters, but in a dict.

        Parameters
        ----------
        name : str
            The name of the parameter to be pulled.
        format : str
            The format in which the parameter should be returned.
            Options are: "numpy", "torch".

        Returns
        -------
        parameter : torch.Tensor or numpy.ndarray
            The parameter pulled from the pyro parameter storage.
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

            if isinstance(views, list):
                if not all([isinstance(view, str) for view in views]):
                    raise TypeError("Parameter 'views' must be a list of strings.")

        if groups is not None:
            if not isinstance(groups, (str, list)):
                raise TypeError("Parameter 'groups' must be of type str or list.")

            if isinstance(groups, list):
                if not all([isinstance(view, str) for view in groups]):
                    raise TypeError("Parameter 'groups' must be a list of strings.")

        if not isinstance(format, str):
            raise TypeError("Parameter 'format' must be of type str.")

        if format not in ["numpy", "torch"]:
            raise ValueError("Parameter 'format' must be in ['numpy', 'torch'].")

        key = "FactorModel._guide." + param + "." + name

        if key not in list(pyro.get_param_store().keys()):
            raise ValueError(
                f"Parameter '{key}' not found in parameter storage. Available choices are: {', '.join(list(pyro.get_param_store().keys()))}"
            )

        data = pyro.get_param_store()[key]

        # TODO: Add full support for group selection.

        if views is not None:
            if views != "all":
                if isinstance(views, str):
                    if views not in self.data._names:
                        raise ValueError(f"Parameter 'views' must be in {list(self.data._names)}.")

                    result = data[..., self.data._feature_idx[views]]

                elif isinstance(views, list):
                    if not all([view in self.data._names for view in views]):
                        raise ValueError(f"All elements in 'views' must be in {list(self.data._names)}.")

                    result = {}
                    for view in views:
                        result[view] = data[..., self.data._feature_idx[view]]

            elif views == "all":
                result = data

        if groups is not None:
            if groups != "all":
                raise NotImplementedError()
            elif groups == "all":
                result = data

        if format == "numpy":
            if result.is_cuda:
                result = result.cpu()
            result = result.detach().numpy()

        return result.squeeze()

    def get_weights(self, views: Union[str, List[str]] = "all", format="numpy"):
        return self._get_from_param_storage(name="w", param="locs", views=views, groups=None, format=format)

    def get_factors(self, groups: Union[str, List[str]] = "all", format="numpy"):
        return self._get_from_param_storage(name="z", param="locs", views=None, groups=groups, format=format)

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
        scale: bool = True,
    ):
        # Clear pyro param
        pyro.clear_param_store()

        # If early stopping is set, check if it is a valid value
        if early_stopping:
            if min_delta < 0:
                raise ValueError("min_delta must be positive.")
            earlystopper = EarlyStopper(patience=patience, min_delta=min_delta, percentage=percentage)
        else:
            earlystopper = None

        # Checks
        if self._data is None:
            raise ValueError("No data set.")

        # if not isinstance(likelihoods, (str, dict)):
        #     raise ValueError(
        #         f"Parameter 'likelihoods' must either be a string or a dictionary mapping the modalities to strings, got {type(likelihoods)}."
        #     )

        # # Prepare likelihoods
        # # TODO: If string is passed, check if string corresponds to a valid pyro distribution
        # # TODO: If custom distribution is passed, check if it provides arg_constraints parameter

        # # If user passed strings, replace the likelihood strings with the actual distributions
        # if isinstance(likelihoods, str):
        #     likelihoods = {
        #         modality: likelihoods for modality in self._data.feature_groups
        #     }

        # for name, distribution in likelihoods.items():
        #     if isinstance(distribution, str):
        #         # Replace likelihood string with common synonyms and correct for align with Pyro
        #         distribution = distribution.title()
        #         if distribution == "Gaussian":
        #             distribution = "Normal"

        #         try:
        #             likelihoods[name] = getattr(pyro.distributions, distribution)  # type: ignore
        #         except AttributeError:
        #             raise AttributeError(
        #                 f"Could not find valid Pyro distribution for {distribution}."
        #             )

        # # Raise error if likelihoods are not set for all modalities
        # if len(likelihoods.keys()) != len(self._data.feature_groups):
        #     raise ValueError(
        #         f"Likelihoods must be set for all modalities. Got {len(likelihoods.keys())} likelihood "
        #         f"and {len(self._data.feature_groups)} data modalities."
        #     )

        # # Provide data information to generative model
        # self._model._setup(data=self._data, likelihoods=likelihoods)
        n_views = len(self.data._feature_groups)
        n_features = [len(x) for x in self.data._feature_idx.values()]
        feature_dict = dict(zip([f"view_{m}" for m in range(n_views)], n_features))
        data_dict = {}
        for m, (view_name, _) in enumerate(feature_dict.items()):
            # TODO: This works only for a sinlge group as of now
            data_dict[view_name] = torch.Tensor(self._data._values[:, self._data._feature_idx[f"feature_group_{m}"]])  #.to(device)

        # Initialize class objects with correct data-related parameters
        self._model = self._model(
            n_samples=self._data._values.shape[0],
            n_factors=self.n_factors,
            feature_dict=feature_dict,
            likelihoods=None,
            **self.model_kwargs
        )
        self._guide = self._guide(self._model, **self.guide_kwargs)

        # We scale the gradients by the number of total samples to allow a better comparison across
        # models/datasets
        scaling_constant = 1.0
        if scale:
            # scaling_constant = 1.0 / self._data.values.shape[1]
            # TODO!! remove hardcoding
            scaling_constant = 1.0 / 200

        optim = pyro.optim.Adam({"lr": learning_rate, "betas": (0.95, 0.999)})
        if optimizer.lower() == "clipped":
            gamma = 0.1
            lrd = gamma ** (1 / epochs)
            optim = pyro.optim.ClippedAdam({"lr": learning_rate, "lrd": lrd})

        svi = SVI(
            model=pyro.poutine.scale(self._model, scale=scaling_constant),
            guide=pyro.poutine.scale(self._guide, scale=scaling_constant),
            optim=optim,
            # loss=pyro.infer.Trace_ELBO(),  # type: ignore
            loss=pyro.infer.TraceMeanField_ELBO(
                retain_graph=True,
                num_particles=num_particles,
                vectorize_particles=True,
            ),
        )

        # TOOD: Preprocess data
        data = data_dict
        # data = self._model.values

        self.losses = []
        time_start = timer()
        verbose_time_start = time_start
        for i in range(epochs + 1):
            loss = svi.step(data=data)
            self.losses.append(loss)

            if early_stopping:
                if earlystopper.step(loss):
                    print(f"Early stopping of training due to convergence at step {i}")
                    break

            if i % verbose_epochs == 0:
                log = f"Epoch {i:>6}: {loss:>14.2f} \t"
                if i >= 1:
                    log += f"| {100 - 100*self.losses[i]/self.losses[i - verbose_epochs]:>6.2f}%\t"
                    log += f"| {(timer() - verbose_time_start):>6.2f}s"

                verbose_time_start = timer()

                print(log)

        self._is_trained = True
        print("Training finished.")
        
        return self.losses

    def save(self, filename: str, overwrite : bool = True):
        if not self._is_trained:
            raise ValueError("Model must be trained before saving.")

        if not isinstance(filename, str):
            raise ValueError(f"Parameter 'filename' must be a string, got {type(filename)}.")

        # Verify that the user used a file ending
        _, file_ending = os.path.splitext(filename)

        if file_ending == "":
            raise ValueError("No file ending provided. Please provide a file ending such as '.pkl'.")
        
        file_name = filename.replace(".pkl", ".state_dict")
        if Path(file_name).exists() and not overwrite:
            raise ValueError(f"File {filename} already exists. Set 'overwrite' to True to overwrite the file.")

        with open(filename, "wb") as f:
            pickle.dump(self, f)

        torch.save(self.state_dict(), file_name)
