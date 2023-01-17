from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from importlib.metadata import version
from typing import List, Optional, Union, Dict
import itertools
import anndata
import subprocess
import matplotlib.pyplot as plt
import mfmf
import muon
import numpy as np
import pandas as pd
import pyro
import torch
from pyro.infer import SVI
import os
import time


class FactorModel:
    def __init__(
        self,
        n_factors: Union[int, Iterable[int]],
        optimizer: pyro.optim.optim.PyroOptim = "default",
        loss: Union[pyro.infer.ELBO, mfmf.loss.Loss] = "default",
        guide: Union[str, pyro.infer.autoguide.AutoGuide] = "AutoNormal",
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):

        if not (
            isinstance(n_factors, int)
            or n_factors == "auto"
            or (
                isinstance(n_factors, list)
                and all(isinstance(item, int) for item in n_factors)
            )
        ):
            # TODO: implement fit for list of n_factors
            raise TypeError(
                "Parameter 'n_factors' must be a single integer, a list of integers or 'auto'."
            )

        if optimizer == "default":
            optimizer = pyro.optim.Adam({"lr": 0.01, "betas": (0.90, 0.999)})

        if not isinstance(optimizer, pyro.optim.optim.PyroOptim):
            raise TypeError(
                "Parameter 'optimizer' must be type 'pyro.optim.optim.PyroOptim'."
            )

        if loss == "default":
            loss = mfmf.loss.EarlyStoppingLoss(
                loss_fn=pyro.infer.Trace_ELBO(
                    num_particles=1, vectorize_particles=True
                ),
                epochs=999999,
                report_after_n_epochs=200,
                min_decrease=0.001,
                max_flat_intervals=3,
            )

        if not (isinstance(loss, pyro.infer.ELBO) or isinstance(loss, mfmf.loss.Loss)):
            raise TypeError(
                "Parameter 'loss' must be type 'pyro.infer.ELBO' or 'mfmf.loss.Loss'."
            )

        valid_guide_strings = [
            "AutoNormal",
            "AutoDelta",
            "AutoGuideList",
            "AutoMultivariateNormal",
            "AutoLowRankMultivariateNormal",
            "AutoHierarchicalNormalMessenger",
        ]

        if isinstance(guide, str):
            if guide not in valid_guide_strings:
                raise ValueError(
                    "Parameter 'guide' must be one of ['"
                    + "', '".join(valid_guide_strings)
                    + "']."
                )
        elif isinstance(guide, pyro.infer.autoguide.AutoGuide):
            if guide.__class__.__name__ not in valid_guide_strings:
                raise ValueError(
                    "Parameter 'guide' must be one of ['"
                    + "', '".join(valid_guide_strings)
                    + "']."
                )

        if not isinstance(dtype, torch.dtype):
            raise TypeError("Parameter 'dtype' must be type 'torch.dtype'.")

        if not (("cpu" in device) or ("cuda" in device)):
            raise ValueError("Parameter 'device' must be either 'cpu' or 'cuda'.")

        # Store information for object traceability
        self.pytorch_version = version("torch") if "torch" in sys.modules else "unknown"
        self.pyro_version = version("pyro-ppl") if "pyro" in sys.modules else "unknown"

        # Store parameters passed to object
        self.n_factors = n_factors
        self.nfactors = n_factors  # for mofax, won't use otherwise
        self.optimizer = optimizer
        self.loss = loss
        self.guide = guide
        self.views = {}
        self._covariates = []
        self.dtype = dtype
        self.device = device
        self._shape = [0, 0]
        self._feature_groups = []
        self._obs_groups = []
        self.metadata = pd.DataFrame()

        # Adapted from http://pyro.ai/examples/sir_hmc.html?highlight=cuda
        if dtype is torch.float64:
            if "cuda" in device:
                torch.set_default_tensor_type(torch.cuda.DoubleTensor)
            else:
                torch.set_default_tensor_type(torch.DoubleTensor)
        elif "cuda" in device:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)

        torch.device(self.device)

        if logging and "cuda" in self.device:
            gpu_usage = mfmf.utils.get_current_gpu_usage(output="formatted")
            print(f"Before training: {gpu_usage}")

        # Store defaults and slots-to-be-filled
        self.loss_during_training = pd.DataFrame()
        self._is_trained = False

    @property
    def n_factors(self):
        return self._n_factors

    @n_factors.setter
    def n_factors(self, n_factors):
        if n_factors == "auto":
            # TODO(@ttreis): Write automatic factor robustness stuff
            raise NotImplementedError("Not yet implemented.")
        elif not isinstance(n_factors, int):
            raise TypeError(
                "Parameter 'n_factors' must be 'auto' or a positive integer."
            )
        elif isinstance(n_factors, int) and n_factors <= 0:
            raise ValueError(
                "Parameter 'n_factors' must be 'auto' or a positive integer."
            )
        else:
            self._n_factors = n_factors

    @property
    def likelihood(self):
        return self._likelihood

    @likelihood.setter
    def likelihood(self, likelihood):
        self._likelihood = likelihood

    @property
    def likelihoods(self):

        # if not self.is_trained:
        #     raise RuntimeError("Model must be trained.")

        output = {}

        for fg in self.feature_groups:

            for og in self.obs_groups:

                output[fg["associated_view"]] = self.internals[fg["id"]][og["id"]][
                    "likelihood"
                ]

        return output

    @likelihoods.setter
    def likelihoods(self, *args):

        raise AttributeError("The likelihoods of a trained model cannot be changed.")

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss):
        if isinstance(loss, pyro.infer.ELBO):
            self._loss = mfmf.loss.Loss(
                loss_fn=loss, epochs=1000, report_after_n_epochs=100
            )
            logging.warning(
                (
                    "Neither a maximum number of epochs to train, nor a stopping "
                    "criteria was specified. Assuming 1000 epochs total, reporting "
                    "every 100 epochs. You can usese mfmf.loss.Loss or "
                    "mfmf.loss.EarlyStoppingLoss to modify those parameters."
                )
            )
        elif isinstance(loss, mfmf.loss.Loss):
            self._loss = loss

    @property
    def guide(self):
        return self._guide

    @guide.setter
    def guide(self, guide):

        if guide == "AutoNormal":
            self._guide = pyro.infer.autoguide.guides.AutoNormal(self.model) 
        elif guide == "AutoDelta":
            self._guide = pyro.infer.autoguide.guides.AutoDelta(self.model)
        elif guide == "AutoMultivariateNormal":
            self._guide = pyro.infer.autoguide.guides.AutoMultivariateNormal(self.model)
        elif guide == "AutoLowRankMultivariateNormal":
            self._guide = pyro.infer.autoguide.guides.AutoLowRankMultivariateNormal(
                self.model
            )
        elif guide == "AutoHierarchicalNormalMessenger":
            self._guide = pyro.infer.autoguide.guides.AutoHierarchicalNormalMessenger(
                self.model
            )
        elif guide == "AutoGuideList":
            self._guide = pyro.infer.autoguide.guides.AutoGuideList(self.model)
        elif guide.__name__ in dir(pyro.infer.autoguide):
            self._guide = guide(self.model)

    @property
    def feature_reg(self):
        return self._feature_reg

    @feature_reg.setter
    def feature_reg(self, feature_reg):
        self._feature_reg = feature_reg

    @property
    def sample_reg(self):
        return self._sample_reg

    @sample_reg.setter
    def sample_reg(self, sample_reg):
        self._sample_reg = sample_reg

    @property
    def covariates(self):

        return self._covariates

    @covariates.setter
    def covariates(self, *args):

        raise AttributeError(
            "Please use 'FactorModel.add_covariates()' to add covariates."
        )

    @property
    def views(self):
        return self._views

    @views.setter
    def views(self, views):
        self._views = views

    @property
    def n_views(self):
        return len(self.views)

    @n_views.setter
    def n_views(self, *args):

        raise AttributeError(
            "Parameter 'n_views' is derived from views added to the model."
        )

    @property
    def view_names(self):
        return list(self.views.keys())

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, n_features):
        self._n_features = n_features

    @property
    def n_obs(self):

        if self.n_views == 0:
            return 0
        else:
            obs_names = []
            for view in self.views.values():
                obs_names += view.obs_names

            return len(set(obs_names))

    @n_obs.setter
    def n_obs(self, *args):

        raise AttributeError(
            "Parameter 'n_obs' is derived from views added to the model."
        )

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, obs):
        self._obs = obs

    @property
    def obs_names(self):

        if self.n_views == 0:
            return 0
        else:
            obs_names = []
            for view in self.views.values():
                obs_names += view.obs_names

            return list(set(obs_names))

    @property
    def feature_names(self):

        if self.n_views == 0:
            return 0
        else:
            feature_names = []
            for view in self.views.values():
                feature_names += view.feature_names

            return list(set(feature_names))

    @property
    def n_features(self):

        if self.n_views == 0:
            return 0
        else:
            if self.is_trained:
                return self._n_features
            else:
                n_features = 0
                for view in self.views.values():
                    n_features += view.n_features
                return n_features

    @property
    def shape(self):

        if len(self.views.keys()) == 0:
            return (0, 0)
        elif len(self.views.keys()) == 1:
            return self.views[list(self.views.keys())[0]].data.shape
        else:
            return (self.n_obs, self.n_features)

    @shape.setter
    def shape(self, *args):

        raise AttributeError(
            "Parameter 'shape' is derived from the views and can't be set."
        )

    @property
    def groups(self):

        # is for mofax compatibility

        if len(self.obs_groups) > 0:

            found_groups = [og["group_name"] for og in self.obs_groups]

            return found_groups

        else:

            return None

    @groups.setter
    def groups(self, *args):

        raise AttributeError(
            "Parameter 'groups' is derived from the covariates and can't be set."
        )

    @property
    def feature_groups(self):
        """List that holds the feature groups of the model.

        Keys per member:
        - 'associated_view' (f.e. 'mrna')
        - 'group_name' (f.e. 'all_features')
        - 'features' (f.e. ['gene1', 'gene2', 'gene3'])
        - 'id' (f.e. 'mrna_all_features')

        Returns:
            list: List of dictionaries, one per feature group (either full view or subset)
        """

        return self._feature_groups

    @feature_groups.setter
    def feature_groups(self, *args):

        raise AttributeError(
            "Please use 'FactorModel.add_feature_group()' to add feature groupings."
        )

    @property
    def obs_groups(self):
        """List that holds the obs groups of the model.

        Keys per member:
        - 'group_name' (f.e. 'all_observations')
        - 'observations' (f.e. ['obs1', 'obs2', 'obs3'])
        - 'id' (same as 'group_name', exists for more similar downstream code)

        Returns:
            list: List of dictionaries, one per obs group
        """

        return self._obs_groups

    @obs_groups.setter
    def obs_groups(self, *args):

        raise AttributeError(
            "Please use 'FactorModel.add_obs_group()' to add observation groupings."
        )

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):

        if not isinstance(metadata, pd.DataFrame):
            raise TypeError("Parameter 'metadata' must be a pandas DataFrame.")

        self._metadata = metadata

    @property
    def is_trained(self):
        return self._is_trained

    @is_trained.setter
    def is_trained(self, is_trained):
        self._is_trained = is_trained

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    def add_view(
        self,
        name: str,
        data: Union[np.ndarray, torch.Tensor, anndata.AnnData],
    ) -> None:

        self.views[name] = mfmf.modules.View(
            name=name,
            data=data,
        )

        if isinstance(data, anndata.AnnData):

            self.metadata = data.obs

    def remove_view(self, name: str) -> None:

        del self.views[name]

    def add_views(
        self,
        data: muon.MuData,
    ) -> None:

        if not isinstance(data, muon.MuData):
            raise TypeError("Paramter 'data' must be a muon.MuData object.")

        for modality_name, anndata_object in data.mod.items():
            self.add_view(name=modality_name, data=anndata_object)

        self.metadata = data.obs

    def get_merged_view(self, views: Union[str, List[str]] = "all"):

        invalid_view_passed = False
        valid_views = [view_name for view_name, _ in self.views.items()]
        valid_views += ["all_features"]

        if isinstance(views, str):
            views = [views]

        for v in views:
            if v not in valid_views:
                logging.error(f"View '{v}' is not a valid view.")
                invalid_view_passed = True

        if invalid_view_passed:
            w = "One or more invalid views were passed. Valid choices are ['"
            w += "', '".join([view_name for view_name, _ in self.views.items()])
            w += "']"
            logging.error(w)
            raise ValueError(w)

        if "all" in views:
            views = valid_views
            valid_views.remove("all")

        views_for_merge = [self.views[view_name] for view_name in views]

        merged_view = mfmf.utils.merge_views(
            views=views_for_merge,
        )

        return merged_view

    def fit(
        self,
        likelihood: Union[
            pyro.distributions.distribution.DistributionMeta,
            dict,
        ] = pyro.distributions.Normal,
        feature_reg: Union[str, dict] = None,
        sample_reg: Union[str, dict] = None,
        store_params: bool = True,
        dry_run: bool = False,
        logging: bool = False,
        **kwargs,
    ):

        if not isinstance(dry_run, bool):
            raise TypeError("Parameter 'dry_run' must be a boolean.")

        if not isinstance(logging, bool):
            raise TypeError("Parameter 'logging' must be a boolean.")

        # Complicated input check, hence this comment. The user can theoretically
        # specify 'likelihood', 'feature_reg' and 'sample_reg' for each element
        # in a [view x covariate group] matrix separately. In the simplest case,
        # this is done by assigning the same likelihood/sparsity to all elements.
        # However, assignment can also happen on a per-view or per-covariate group
        # basis or by specifying an exact location in the theoretical matrix.

        # 1) Construction of a MultiIndex Matrix to hold all three parameters
        #    which more or less looks like this
        #
        # +-------------+-------------+----------------+-----+----------------+
        # | cov_names   | param_names |     view_1     | ... |     view_n     |
        # +-------------+-------------+----------------+-----+----------------+
        # | cov_group_1 | likelihood  | likelihood     | ... | likelihood     |
        # |             | feature_reg | sparsity_prior | ... | sparsity_prior |
        # |             | sample_reg  | sparsity_prior | ... | sparsity_prior |
        # |             |             |                |     |                |
        # | ...         | ...         | ...            | ... | ...            |
        # |             |             |                |     |                |
        # | cov_group_n | likelihood  | likelihood     | ... | likelihood     |
        # |             | feature_reg | sparsity_prior | ... | sparsity_prior |
        # |             | sample_reg  | sparsity_prior | ... | sparsity_prior |
        # +-------------+-------------+----------------+-----+----------------+

        param_names = ["likelihood", "feature_reg", "sample_reg"]

        views_with_feature_group = [
            f_group["associated_view"] for f_group in self.feature_groups
        ]

        # Add complete groups if no subgroups are defined
        for view_name in self.view_names:

            if view_name not in views_with_feature_group:

                self.add_feature_group(
                    associated_view=view_name,
                    group_name="all_features",
                    features=self.views[view_name].feature_names,
                )

        if len(self.obs_groups) == 0:

            self.add_obs_group(
                group_name="all_observations",
                observations=self.obs_names,
            )

        row_indices = [
            np.array(
                [
                    og["group_name"]
                    for og in self.obs_groups
                    for _ in range(len(param_names))
                ]
            ),
            np.array(param_names * len(self.obs_groups)),
        ]

        col_indices = [
            np.array([x["associated_view"] for x in self.feature_groups]),
            np.array([x["group_name"] for x in self.feature_groups]),
        ]

        info_df = pd.DataFrame(
            np.full(
                shape=(
                    len(row_indices[0]),
                    len(col_indices[0]),
                ),
                fill_value=np.nan,
            ),
        )
        info_df.index = pd.MultiIndex.from_arrays(row_indices)
        info_df.columns = pd.MultiIndex.from_arrays(col_indices)

        obs_group_names = [obs_group["group_name"] for obs_group in self.obs_groups]

        # 2a) Easiest case: 'likelihood' / 'feature_reg' / 'sample_reg' is the
        #     same across all elements per parameter
        if isinstance(likelihood, pyro.distributions.distribution.DistributionMeta):
            for obs_group in self.obs_groups:
                info_df.loc[obs_group["group_name"], "likelihood"] = likelihood.__name__

        if isinstance(feature_reg, str):
            for obs_group in self.obs_groups:
                info_df.loc[obs_group["group_name"], "feature_reg"] = feature_reg

        if isinstance(sample_reg, str):
            for obs_group in self.obs_groups:
                info_df.loc[obs_group["group_name"], "sample_reg"] = sample_reg

        # 2b) More complicated case: 'likelihood' / 'feature_reg' / 'sample_reg'
        #     is specified on a per-covariance-group xor per-view basis
        if isinstance(likelihood, dict):
            for k, v in likelihood.items():
                if not isinstance(v, dict):
                    if k in obs_group_names:
                        info_df.loc[k, "likelihood"] = v.__name__
                    elif k in self.view_names:
                        for obs_group in self.obs_groups:
                            info_df.loc[
                                (obs_group["group_name"], "likelihood"), k
                            ] = v.__name__
                    else:
                        raise ValueError(
                            f"Key '{k}' in parameter 'likelihood' is not a valid view or covariance group."
                        )

        if isinstance(feature_reg, dict):
            for k, v in feature_reg.items():
                if not isinstance(v, dict):
                    if k in obs_group_names:
                        info_df.loc[k, "feature_reg"] = feature_reg[k]
                    elif k in self.view_names:
                        for obs_group in self.obs_groups:
                            info_df.loc[
                                (obs_group["group_name"], "feature_reg"), k
                            ] = feature_reg[k]
                    else:
                        raise ValueError(
                            f"Key '{k}' in parameter 'feature_reg' is not a valid view or covariance group."
                        )

        if isinstance(sample_reg, dict):
            for k, v in sample_reg.items():
                if not isinstance(v, dict):
                    if k in obs_group_names:
                        info_df.loc[k, "sample_reg"] = sample_reg[k]
                    elif k in self.view_names:
                        for obs_group in self.obs_groups:
                            info_df.loc[obs_group["group_name"], "sample_reg"][
                                k
                            ] = sample_reg[k]
                    else:
                        raise ValueError(
                            f"Key '{k}' in parameter 'sample_reg' is not a valid view or covariance group."
                        )

        # 2c) Most granular case: 'likelihood' / 'feature_reg' / 'sample_reg'
        #     is specified for a single element in the matrix
        if isinstance(likelihood, dict):
            for k1, v1 in likelihood.items():
                if isinstance(v1, dict):
                    for k2 in v1.keys():
                        if k1 in obs_group_names and k2 in self.view_names:
                            info_df.loc[k1, "likelihood"][k2] = likelihood[k1][
                                k2
                            ].__name__
                        elif k1 in self.view_names and k2 in obs_group_names:
                            info_df.loc[k2, "likelihood"][k1] = likelihood[k1][
                                k2
                            ].__name__
                        else:
                            raise ValueError(
                                f"Key '{k1}' or '{k2}' in parameter 'likelihood' is not a valid view or covariance group."
                            )

        if isinstance(feature_reg, dict):
            for k1, v1 in feature_reg.items():
                if isinstance(v1, dict):
                    for k2 in v1.keys():
                        if k1 in obs_group_names and k2 in self.view_names:
                            info_df.loc[k1, "feature_reg"][k2] = feature_reg[k1][k2]
                        elif k1 in self.view_names and k2 in obs_group_names:
                            info_df.loc[k2, "feature_reg"][k1] = feature_reg[k1][k2]
                        else:
                            raise ValueError(
                                f"Key '{k1}' or '{k2}' in parameter 'feature_reg' is not a valid view or covariance group."
                            )

        if isinstance(sample_reg, dict):
            for k1, v1 in sample_reg.items():
                if isinstance(v1, dict):
                    for k2 in v1.keys():
                        if k1 in obs_group_names and k2 in self.view_names:
                            info_df.loc[k1, "sample_reg"][k2] = sample_reg[k1][k2]
                        elif k1 in self.view_names and k2 in obs_group_names:
                            info_df.loc[k2, "sample_reg"][k1] = sample_reg[k1][k2]
                        else:
                            raise ValueError(
                                f"Key '{k1}' or '{k2}' in parameter 'sample_reg' is not a valid view or covariance group."
                            )

        self._info_df = info_df

        if not dry_run:

            if not len(self.views) >= 1:
                raise ValueError(
                    "At least one view must be added to the model before fitting."
                )

            if not isinstance(store_params, bool):
                raise TypeError("Parameter 'store_params' must be 'True' or 'False'.")

            self.likelihood = likelihood
            self.feature_reg = feature_reg
            self.sample_reg = sample_reg
            self._merged_view = self.get_merged_view(views=self.views)
            self.feature_offsets = self._merged_view.feature_offsets
            self.obs = self._merged_view.data
            self._n_features = self._merged_view.n_features
            self._n_obs = self._merged_view.n_obs
            self._feature_names = self._merged_view.feature_names
            self._obs_names = self._merged_view.obs_names
            self.internals = {}
            self.kwargs = kwargs

            all_features = pd.DataFrame(self._feature_names)
            all_obs = pd.DataFrame(self._obs_names)

            feature_group_indices = {}
            for feature_group in self.feature_groups:
                feature_group_key = (
                    feature_group["associated_view"] + "_" + feature_group["group_name"]
                )
                feature_group_indices[feature_group_key] = (
                    all_features.reset_index()
                    .merge(pd.DataFrame(feature_group["features"]), how="inner")
                    .set_index("index")
                    .index.values
                )

            obs_group_indices = {}
            for obs_group in self.obs_groups:
                obs_group_key = obs_group["group_name"]
                obs_group_indices[obs_group_key] = (
                    all_obs.reset_index()
                    .merge(pd.DataFrame(obs_group["observations"]), how="inner")
                    .set_index("index")
                    .index.values
                )

            # Prepare dict with feature / obs group permutations
            for fg_name in feature_group_indices.keys():

                self.internals[fg_name] = {}

                for og_name in obs_group_indices.keys():

                    self.internals[og_name] = {}
                    self.internals[fg_name][og_name] = {}

            # Prepare feature and obs plates
            self.internals["feature_plates"] = {}
            self.internals["obs_plates"] = {}

            for fg_name, fg_indicies in feature_group_indices.items():

                self.internals["feature_plates"][fg_name] = pyro.plate(
                    name=f"feature_plate_{fg_name}",
                    dim=-1,
                    size=len(fg_indicies),
                )

            for og_name, og_indicies in obs_group_indices.items():

                if "obs_subsample_perc" in self.kwargs:
                    if self.kwargs["obs_subsample_perc"] is not None:
                        subsample_size = int(len(og_indicies) * 0.5)
                        self.internals["obs_plates"][og_name] = pyro.plate(
                            name=f"obs_plate_{og_name}",
                            dim=-2,
                            size=len(og_indicies),
                            subsample_size=subsample_size,
                        )
                else:

                    self.internals["obs_plates"][og_name] = pyro.plate(
                        name=f"obs_plate_{og_name}",
                        dim=-2,
                        size=len(og_indicies),
                    )

            # Extract likelihoods and prepare views
            for fg_name, fg_idxs in feature_group_indices.items():

                fg_view = fg_name.split("_")[0]
                fg_group = "_".join(fg_name.split("_")[1:])

                for og_name, og_idxs in obs_group_indices.items():

                    self.internals[fg_name][og_name]["obs"] = self.obs[:, fg_idxs][
                        og_idxs, :
                    ]

                    self.internals[fg_name][og_name]["likelihood"] = self._info_df[
                        (fg_view, fg_group)
                    ][(og_name, "likelihood")]

            # scale view if needed
            for view_name in self.view_names:
                if self.likelihoods[view_name] != "Bernoulli":
                    data_mean = self.views[view_name].data.nanmean()
                    data_std = torch.tensor(
                        np.nanstd(self.views[view_name].data.cpu().detach().numpy())
                    )
                    self.views[view_name].data = (
                        self.views[view_name].data - data_mean
                    ) / data_std

            # assert that data fits the likelihood
            for fg_name, fg_idxs in feature_group_indices.items():

                fg_view = fg_name.split("_")[0]
                fg_group = "_".join(fg_name.split("_")[1:])

                for og_name, og_idxs in obs_group_indices.items():

                    if self.internals[fg_name][og_name]["likelihood"] == "Poisson":

                        constraint = pyro.distributions.Poisson.support
                        # Replacing NaNs with a valid value to avoid errors in constraint check, will be masked later
                        data_to_check = torch.nan_to_num(
                            self.internals[fg_name][og_name]["obs"], nan=1
                        )

                        if not torch.all(constraint.check(data_to_check)):
                            raise ValueError(
                                f"Error when using Poisson likelihood for view '({fg_view}, {fg_group})': Data can only be positive integers."
                            )

                    elif self.internals[fg_name][og_name]["likelihood"] == "Bernoulli":

                        constraint = pyro.distributions.ContinuousBernoulli.support
                        # Replacing NaNs with a valid value to avoid errors in constraint check, will be masked later
                        data_to_check = torch.nan_to_num(
                            self.internals[fg_name][og_name]["obs"], nan=0.0
                        )

                        if not torch.all(constraint.check(data_to_check)):
                            raise ValueError(
                                f"Error when using Bernoulli likelihood for view '({fg_view}, {fg_group})': Data must be in [0.0, 1.0]."
                            )

                    elif (
                        self.internals[fg_name][og_name]["likelihood"]
                        == "NegativeBinomial"
                    ):

                        constraint = pyro.distributions.GammaPoisson.support
                        # Replacing NaNs with a valid value to avoid errors in constraint check, will be masked later
                        data_to_check = torch.nan_to_num(
                            self.internals[fg_name][og_name]["obs"], nan=1
                        )

                        if not torch.all(constraint.check(data_to_check)):
                            raise ValueError(
                                f"Error when using NegativeBinomial likelihood for view '({fg_view}, {fg_group})': Data can only be integers greater or equal than 0."
                            )

            scaler = 1.0 / (self._n_obs * self._n_features)

            self.svi = SVI(
                model=pyro.poutine.scale(self.model, scale=scaler),
                guide=pyro.poutine.scale(self.guide, scale=scaler),
                # model=self.model,
                # guide=self.guide,
                optim=self.optimizer,
                loss=self.loss.loss_fn,
            )

            # Flush old training data
            if not self.is_trained:
                pyro.clear_param_store()
            self.loss_during_training = pd.DataFrame()

            is_first_step = True
            is_first_interval = True
            is_flat_interval = False
            early_stop_possible = (
                True if isinstance(self.loss, mfmf.loss.EarlyStoppingLoss) else False
            )
            stop_counter = 0
            start = time.time()
            for epoch in range(self.loss.epochs + 1):

                cur_loss = self.svi.step()
                self.loss_during_training = pd.concat(
                    [
                        self.loss_during_training,
                        pd.DataFrame(
                            {
                                "epoch": [epoch],
                                "loss": [cur_loss],
                            }
                        ),
                    ],
                    axis=0,
                )
                if epoch % self.loss.report_after_n_epochs == 0:

                    if is_first_step:

                        first_loss = cur_loss

                    else:

                        avg_loss_last_interval = np.median(
                            self.loss_during_training.loss.tail(
                                self.loss.report_after_n_epochs
                            )
                        )

                        formatted_iteration = mfmf.utils.get_formatted_number(
                            number=epoch,
                            n_before_separator=len(str(int(self.loss.epochs))),
                            n_after_separator=0,
                        )
                        left_pad_for_loss = len(str(int(first_loss))) + 1
                        formatted_loss = mfmf.utils.get_formatted_number(
                            number=avg_loss_last_interval,
                            n_before_separator=left_pad_for_loss + 1,
                            n_after_separator=3,
                        )

                        abs_perc_decrease_num = 100 - (
                            (avg_loss_last_interval / first_loss) * 100
                        )
                        abs_perc_decrease = mfmf.utils.get_formatted_number(
                            number=abs(abs_perc_decrease_num),
                            n_before_separator=4,
                            n_after_separator=6,
                        )

                        if abs_perc_decrease_num > 0:

                            abs_perc_decrease = f"-{str(abs_perc_decrease)} %"

                        else:

                            abs_perc_decrease = f"+{str(abs_perc_decrease)} %"

                        if is_first_interval:

                            last_perc_decrease = ""

                        else:

                            last_perc_decrease_num = (
                                abs_perc_decrease_num - prev_interval_loss
                            )

                            if early_stop_possible:

                                is_flat_interval = (
                                    True
                                    if last_perc_decrease_num
                                    < (self.loss.min_decrease * 100)
                                    else False
                                )

                            last_perc_decrease = mfmf.utils.get_formatted_number(
                                number=abs(last_perc_decrease_num),
                                n_before_separator=4,
                                n_after_separator=2,
                            )

                            if last_perc_decrease_num > 0:

                                last_perc_decrease = f"-{str(last_perc_decrease)} %"

                            else:

                                last_perc_decrease = f"+{str(last_perc_decrease)} %"

                        gpu_usage = None
                        if logging and "cuda" in self.device:

                            gpu_usage = (
                                "["
                                + mfmf.utils.get_current_gpu_usage(output="formatted")
                                + "]"
                            )

                        # get and format elapsed time
                        elapsed = time.time() - start
                        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))

                        msg = "[iteration {}] [{}]{} median ELBO in last interval: {}  |  {}  |  {}  ".format(
                            formatted_iteration,
                            formatted_time,
                            gpu_usage,
                            formatted_loss,
                            abs_perc_decrease,
                            last_perc_decrease,
                        )

                        if is_flat_interval:

                            msg += "|  FLAT INTERVAL"
                            stop_counter += 1

                        else:

                            stop_counter = 0

                        print(msg) if logging else None
                        is_first_interval = False
                        prev_interval_loss = abs_perc_decrease_num

                    if (
                        early_stop_possible
                        and stop_counter >= self.loss.max_flat_intervals
                    ):
                        print(
                            "BREAK - Early stopping criteria reached."
                        ) if logging else None
                        break

                is_first_step = False

            self.is_trained = True

            self.results = {}
            for fg in self.feature_groups:
                self.results[fg["associated_view"]] = {}
                self.results[fg["associated_view"]]["w"] = (
                    self.internals[fg["id"]]["w"].cpu().detach().numpy().squeeze()
                )
            for og in self.obs_groups:
                self.results[og["group_name"]] = {}
                self.results[og["group_name"]]["z"] = (
                    self.internals[og["id"]]["z"].cpu().detach().numpy().squeeze()
                )
            for fg in self.feature_groups:
                for og in self.obs_groups:
                    self.results[fg["associated_view"]][og["group_name"]] = {}
                    self.results[fg["associated_view"]][og["group_name"]]["obs"] = (
                        self.internals[fg["id"]][og["id"]]["obs"].cpu().detach().numpy()
                    )

            # Add obs group information to metadata
            obs_id_to_og_df = pd.DataFrame()
            for og in self.obs_groups:

                og_df = pd.DataFrame(
                    {
                        "sample": og["observations"],
                        "group": [og["group_name"]] * len(og["observations"]),
                    }
                )
                obs_id_to_og_df = pd.concat([obs_id_to_og_df, og_df], axis=0)

            self.metadata = pd.merge(
                left=self.metadata,
                left_index=True,
                right=obs_id_to_og_df,
                right_on="sample",
            )
            self.metadata.set_index("sample", inplace=True)

        elif dry_run:

            # Dry-run case in which only a preview of the likelihoods,
            # regularisation priors, covariates and views of the model
            # is shown

            try:
                from tabulate import tabulate
            except ImportError as ie:
                print("This feature requires the package 'tabulate', please install.")
                raise ie

            if not len(self.views.keys()) >= 1:
                raise ValueError("No views have been added yet.")

            info_df_reduced = self._info_df.dropna()

            print(
                tabulate(
                    info_df_reduced,
                    headers=[""] + [(t[0], t[1]) for t in info_df_reduced.columns],
                    tablefmt="fancy_grid",
                )
            )

    def model(self):

        self.obs_factor_plate = pyro.plate(
            name="obs_factor_plate", dim=-1, size=self.n_factors
        )
        self.feature_factor_plate = pyro.plate(
            name="feature_factor_plate", dim=-2, size=self.n_factors
        )

        for fg in self.feature_groups:

            if self.feature_reg == "horseshoe":

                self.internals[fg["id"]].update(
                    mfmf.utils.get_matrix_with_prior(
                        id=fg["id"],
                        prior="horseshoe",
                        feature_plate=self.internals["feature_plates"][fg["id"]],
                        factor_plate=self.feature_factor_plate,
                        feature_reg_kwargs=self.kwargs,
                    )
                )

            elif self.feature_reg == "finnish_horseshoe":

                self.internals[fg["id"]].update(
                    mfmf.utils.get_matrix_with_prior(
                        id=fg["id"],
                        prior="finnish_horseshoe",
                        feature_plate=self.internals["feature_plates"][fg["id"]],
                        factor_plate=self.feature_factor_plate,
                        feature_reg_kwargs=self.kwargs,
                    )
                )

            elif self.feature_reg == "ard":

                self.internals[fg["id"]].update(
                    mfmf.utils.get_matrix_with_prior(
                        id=fg["id"],
                        prior="ard",
                        feature_plate=self.internals["feature_plates"][fg["id"]],
                        factor_plate=self.feature_factor_plate,
                        feature_reg_kwargs=self.kwargs,
                    )
                )

            elif self.feature_reg == "spike-and-slab":

                self.internals[fg["id"]].update(
                    mfmf.utils.get_matrix_with_prior(
                        id=fg["id"],
                        prior="spike-and-slab",
                        feature_plate=self.internals["feature_plates"][fg["id"]],
                        factor_plate=self.feature_factor_plate,
                        feature_reg_kwargs=self.kwargs,
                    )
                )

            elif self.feature_reg == "ard_spike-and-slab":

                self.internals[fg["id"]].update(
                    mfmf.utils.get_matrix_with_prior(
                        id=fg["id"],
                        prior="ard_spike-and-slab",
                        feature_plate=self.internals["feature_plates"][fg["id"]],
                        factor_plate=self.feature_factor_plate,
                        feature_reg_kwargs=self.kwargs,
                    )
                )

            else:

                self.internals[fg["id"]].update(
                    mfmf.utils.get_matrix_with_prior(
                        id=fg["id"],
                        prior="none",
                        feature_plate=self.internals["feature_plates"][fg["id"]],
                        factor_plate=self.feature_factor_plate,
                        feature_reg_kwargs=self.kwargs,
                    )
                )

        tau = pyro.sample(
            name="tau",
            fn=pyro.distributions.InverseGamma(
                concentration=torch.tensor([1.0]), rate=torch.tensor([1.0])
            ),
        )

        y_scale = torch.sqrt(tau)

        for og in self.obs_groups:

            if self.sample_reg == "horseshoe":

                self.internals[og["id"]].update(
                    mfmf.utils.get_matrix_with_prior(
                        id=og["id"],
                        prior="horseshoe",
                        obs_plate=self.internals["obs_plates"][og["id"]],
                        factor_plate=self.obs_factor_plate,
                        sample_reg_kwargs=self.kwargs,
                    )
                )

            elif self.sample_reg == "finnish_horseshoe":

                self.internals[og["id"]].update(
                    mfmf.utils.get_matrix_with_prior(
                        id=og["id"],
                        prior="finnish_horseshoe",
                        obs_plate=self.internals["obs_plates"][og["id"]],
                        factor_plate=self.obs_factor_plate,
                        sample_reg_kwargs=self.kwargs,
                    )
                )

            elif self.sample_reg == "ard":

                self.internals[og["id"]].update(
                    mfmf.utils.get_matrix_with_prior(
                        id=og["id"],
                        prior="ard",
                        obs_plate=self.internals["obs_plates"][og["id"]],
                        factor_plate=self.obs_factor_plate,
                        sample_reg_kwargs=self.kwargs,
                    )
                )

            elif self.sample_reg == "spike-and-slab":

                self.internals[og["id"]].update(
                    mfmf.utils.get_matrix_with_prior(
                        id=og["id"],
                        prior="spike-and-slab",
                        obs_plate=self.internals["obs_plates"][og["id"]],
                        factor_plate=self.obs_factor_plate,
                        sample_reg_kwargs=self.kwargs,
                    )
                )

            elif self.sample_reg == "ard_spike-and-slab":

                self.internals[og["id"]].update(
                    mfmf.utils.get_matrix_with_prior(
                        id=og["id"],
                        prior="ard_spike-and-slab",
                        obs_plate=self.internals["obs_plates"][og["id"]],
                        factor_plate=self.obs_factor_plate,
                        sample_reg_kwargs=self.kwargs,
                    )
                )

            else:

                self.internals[og["id"]].update(
                    mfmf.utils.get_matrix_with_prior(
                        id=og["id"],
                        prior="none",
                        obs_plate=self.internals["obs_plates"][og["id"]],
                        factor_plate=self.obs_factor_plate,
                    )
                )

        for og in self.obs_groups:

            with self.internals["obs_plates"][og["id"]] as ind:

                for fg in self.feature_groups:

                    with self.internals["feature_plates"][fg["id"]]:

                        self.internals[fg["id"]][og["id"]]["y_mean"] = torch.matmul(
                            self.internals[og["id"]]["z"].squeeze(),
                            self.internals[fg["id"]]["w"].squeeze(),
                        )

                        with pyro.poutine.mask(
                            mask=torch.isnan(self.internals[fg["id"]][og["id"]]["obs"])[
                                ind
                            ]
                            == 0
                        ):

                            # https://forum.pyro.ai/t/poutine-nan-mask-not-working/3489
                            # Assign temporary values to the missing data, not used
                            # anyway due to masking. Value needs to satisfy all constraints
                            # of the used distributions:
                            #   a) be positive integer for Poisson
                            #   b) be in [0.0, 1.0] for ContinuousBernoulli
                            self.internals[fg["id"]][og["id"]][
                                "obs"
                            ] = torch.nan_to_num(
                                self.internals[fg["id"]][og["id"]]["obs"], nan=1.0
                            )

                            if (
                                self.internals[fg["id"]][og["id"]]["likelihood"]
                                == "Normal"
                            ):

                                pyro.sample(
                                    name=f"y_{fg['id']}_{og['id']}",
                                    fn=pyro.distributions.Normal(
                                        loc=self.internals[fg["id"]][og["id"]][
                                            "y_mean"
                                        ],
                                        scale=y_scale,
                                    ),
                                    obs=self.internals[fg["id"]][og["id"]]["obs"][ind],
                                )

                            elif (
                                self.internals[fg["id"]][og["id"]]["likelihood"]
                                == "Poisson"
                            ):

                                pyro.sample(
                                    name=f"y_{fg['id']}_{og['id']}",
                                    fn=pyro.distributions.Poisson(
                                        rate=torch.exp(
                                            self.internals[fg["id"]][og["id"]]["y_mean"]
                                        ),
                                    ),
                                    obs=self.internals[fg["id"]][og["id"]]["obs"][ind],
                                )

                            elif (
                                self.internals[fg["id"]][og["id"]]["likelihood"]
                                == "Bernoulli"
                            ):

                                pyro.sample(
                                    name=f"y_{fg['id']}_{og['id']}",
                                    fn=pyro.distributions.ContinuousBernoulli(
                                        probs=torch.sigmoid(
                                            self.internals[fg["id"]][og["id"]]["y_mean"]
                                        )
                                    ),
                                    obs=self.internals[fg["id"]][og["id"]]["obs"][ind],
                                )

                            elif (
                                self.internals[fg["id"]][og["id"]]["likelihood"]
                                == "NegativeBinomial"
                            ):

                                pyro.sample(
                                    name=f"y_{fg['id']}_{og['id']}",
                                    fn=pyro.distributions.NegativeBinomial(
                                        total_count=torch.ones(1),
                                        probs=torch.sigmoid(
                                            self.internals[fg["id"]][og["id"]]["y_mean"]
                                        ),
                                    ),
                                    obs=self.internals[fg["id"]][og["id"]]["obs"][ind],
                                )

    def add_feature_group(self, associated_view: str, group_name: str, features: list):

        if not len(self.view_names) > 0:
            raise ValueError("Please add views first.")

        if not isinstance(associated_view, str):
            raise TypeError("Parameter 'associated_view' must be a string.")

        if associated_view not in self.view_names:
            raise ValueError(
                "Parameter 'associated_view' is invalid. Valid choices are: "
                + str(self.view_names)
            )

        if not isinstance(group_name, str):
            raise TypeError("Parameter 'group_name' must be a string.")
        if not isinstance(features, list):
            raise TypeError("Parameter 'features' must be a list.")

        if not all(isinstance(x, str) for x in features):
            raise TypeError("Parameter 'features' must be a list of strings.")

        self._feature_groups.append(
            {
                "associated_view": associated_view,
                "group_name": group_name,
                "features": features,
                "id": associated_view + "_" + group_name,  # for easy access
            }
        )

    def add_obs_group(self, group_name: str, observations: list):

        if not len(self.obs_names) > 0:
            raise ValueError("Please add views first.")

        if not isinstance(group_name, str):
            raise TypeError("Parameter 'group_name' must be a string.")

        if not isinstance(observations, list):
            raise TypeError("Parameter 'observations' must be a list.")

        if not all(isinstance(x, str) for x in observations):
            raise TypeError("Parameter 'observations' must be a list of strings.")

        self._obs_groups.append(
            {
                "group_name": group_name,
                "observations": observations,
                "id": group_name,  # yes, redundant, but then more similar to feature groups
            }
        )

        self._obs_groups = sorted(self._obs_groups, key=lambda k: k["group_name"])

    def add_covariates(
        self, level: str, flavour: str, covariates: dict, view: str = None
    ):

        if level not in ["features", "obs"]:
            raise ValueError("Parameter 'level' must be either 'features' or 'obs'.")

        if flavour not in ["unordered", "ordered", "ordered_spaced", "ordered_2D"]:
            raise ValueError(
                "Parameter 'flavour' must be in ['unordered', 'ordered', 'ordered_spaced', 'ordered_2D']."
            )

        if not isinstance(covariates, dict):
            raise ValueError("Parameter 'covariates' must be a dictionary.")

        if len(self.views.keys()) == 0:
            raise ValueError("No views have been added yet.")

        if level == "features" and view is None:
            raise ValueError(
                "Parameter 'view' must be specified when 'flavour' is 'features'."
            )

        if level == "features" and view not in self.views.keys():
            raise ValueError(f"View '{view}' not found.")

        if level == "obs":

            if flavour == "unordered":

                if not len(covariates.keys()) >= 2:
                    raise ValueError(
                        "Parameter 'covariates' (unordered) must have at least 2 groups."
                    )

                n_obs_in_covariates = sum([len(v) for v in covariates.values()])
                if not n_obs_in_covariates == self.n_obs:
                    raise ValueError(
                        "The total number of observations in the specified groups does not match the number of observations across all views: {} != {}".format(
                            n_obs_in_covariates, self.n_obs
                        )
                    )

                for group_name, observations in covariates.items():

                    self.add_obs_group(group_name=group_name, observations=observations)

            elif flavour == "ordered":

                raise NotImplementedError("Not yet implemented.")

            elif flavour == "ordered_spaced":

                raise NotImplementedError("Not yet implemented.")

            elif flavour == "ordered_2D":

                raise NotImplementedError("Not yet implemented.")

        elif level == "features":

            if flavour == "unordered":

                if not len(covariates.keys()) >= 2:
                    raise ValueError(
                        "Parameter 'covariates' (unordered) must have at least 2 groups."
                    )

                n_features_in_covariates = sum([len(v) for v in covariates.values()])
                if not n_features_in_covariates == self.views[view].n_features:
                    raise ValueError(
                        "The total number of observations in the specified groups does not match the number of observations across all views: {} != {}".format(
                            n_features_in_covariates, self.n_features
                        )
                    )

                for group_name, features in covariates.items():

                    self.add_feature_group(
                        associated_view=view, group_name=group_name, features=features
                    )

            elif flavour == "ordered":

                raise NotImplementedError("Not yet implemented.")

            elif flavour == "ordered_spaced":

                raise NotImplementedError("Not yet implemented.")

            elif flavour == "ordered_2D":

                raise NotImplementedError("Not yet implemented.")

    def get_input_data(self, format: str = "df", indicies: str = "multi"):

        if not isinstance(format, str):
            raise ValueError("Parameter 'format' must be a string.")

        if format not in ["df", "numpy", "anndata", "tidy"]:
            raise ValueError(
                "Parameter 'format' must be either 'df', 'numpy', 'anndata' or 'tidy'."
            )

        if not isinstance(indicies, str):
            raise ValueError("Parameter 'indicies' must be a string.")

        if indicies not in ["multi", "merged"]:
            raise ValueError("Parameter 'indicies' must be either 'multi' or 'merged'.")

        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")

        data = self._merged_view.data.cpu().detach().numpy()

        # Repeat name of view once per feature contained in respective view
        repeated_view_names = [
            [fg["associated_view"]] * len(fg["features"]) for fg in self.feature_groups
        ]
        feature_names = [fg["features"] for fg in self.feature_groups]

        if indicies == "multi":

            indices = [
                np.array(
                    [i for i in itertools.chain.from_iterable(repeated_view_names)]
                ),
                np.array([i for i in itertools.chain.from_iterable(feature_names)]),
            ]

        elif indicies == "merged":

            # concatenate view and feature name, as in "view1_gene1"
            indicies = [
                np.array(
                    [
                        view_name + "_" + feature_name
                        for view_name, feature_name in zip(
                            repeated_view_names, self.feature_names
                        )
                    ]
                )
            ]

        if format == "df" or format == "tidy":

            # df will later be 'melted' to tidy format if format == 'tidy'

            data = pd.DataFrame(
                data=data,
                index=self._merged_view.obs_names,
                columns=indices,
            )

        elif format == "anndata":

            data = anndata.AnnData(X=data, obs=self._merged_view.obs_names, var=indices)

        if format == "tidy":

            # Convert data frame to tidy long format
            data = pd.melt(data, ignore_index=False)
            data = data.reset_index(drop=False)

            view_col_name = None
            feature_col_name = None

            # Robustly find which column contains the view names
            for col in data.columns:
                if set(data[col].unique()) == set(self.view_names):
                    view_col_name = col

            # Robustly find which column contains the feature names
            for col in data.columns:
                if set(data[col].unique()) == set(self.feature_names):
                    feature_col_name = col

            # The following two should never trigger, but just in case
            if not isinstance(view_col_name, str):
                raise ValueError("Could not find column containing view names.")

            if not isinstance(feature_col_name, str):
                raise ValueError("Could not find column containing feature names.")

            # Rename to MOFA-compatible names
            data = data.rename(
                columns={
                    "index": "sample",
                    view_col_name: "view",
                    feature_col_name: "feature",
                }
            )

        # Generate data frame with view/obs associations
        view_obs_assoc_df = pd.DataFrame()

        if len(self.obs_groups) > 0:

            # TODO(ttreis): Probably wonky, need to evaluate more complex covs
            repeated_group_names = [
                [x["group_name"]] * len(x["observations"]) for x in self.obs_groups
            ]
            obs_names = [x["observations"] for x in self.obs_groups]

            if len(self.obs_groups) > 0:

                tmp = pd.DataFrame(
                    {
                        "group": [
                            i
                            for i in itertools.chain.from_iterable(repeated_group_names)
                        ],
                        "sample": [i for i in itertools.chain.from_iterable(obs_names)],
                    }
                )

            # Join on sample name
            data = data.merge(tmp, on="sample", how="left")

        return data

    # def plot_epsilon(self, abs=True, method: str = "sum", ax: plt.Axes = None):

    #     if not self.is_trained:

    #         raise NotImplementedError("Must be a trained model.")

    #     ws = []
    #     zs = []
    #     x_vals = []

    #     for k, v in self.pyro_params["AutoNormal.locs.w"].items():

    #         ws.append(v.squeeze())

    #     for k, v in self.pyro_params["AutoNormal.locs.z"].items():

    #         zs.append(v.squeeze())
    #         x_vals.append(int(k))

    #     ds = [np.matmul(ws[i], zs[i]) for i in range(len(x_vals))]
    #     es = [
    #         ds[i] - self.obs.cpu().squeeze().detach().numpy()
    #         for i in range(len(x_vals))
    #     ]

    #     if abs:

    #         if method == "sum":

    #             y_vals = [np.sum(np.abs(es[i])) for i in range(len(x_vals))]

    #     else:

    #         y_vals = [np.sum(es[i]) for i in range(len(x_vals))]

    #     ax.plot(x_vals, y_vals, linewidth=2.5)
    #     ax.set_title("Epsilon")
    #     ax.set_xlabel("Epochs")
    #     ax.set_ylabel("sum(abs(epsilon))")

    def plot_elbo(self, window: int = 100, method: str = "median"):

        self.elbo_plot = mfmf.metrics.plot_elbo(self, window=window, method=method)

    def plot_diagnostics(
        self,
        metrics: Union[str, List[str]] = "ELBO",
        window: int = 100,
        method: str = "median",
        axs: List[plt.Axes] = None,
    ):

        if isinstance(metrics, str):
            metrics = [metrics]

        if axs is not None and (len(metrics) != len(axs)):
            raise ValueError("The lengths of 'metrics' and 'axs' must match.")

        if axs is None:
            fig, axs = plt.subplots(1, len(metrics))

        if not isinstance(axs, np.ndarray):
            axs = [axs]

        for idx, m in enumerate(metrics):

            if m == "ELBO":

                axs[idx] = mfmf.metrics.plot_elbo(
                    self, window=window, method=method, ax=axs[idx]
                )

            elif "norm" in m:

                k = m.replace("norm(", "").replace(")", "").strip()
                axs[idx] = mfmf.metrics.plot_norm(self, k, ax=axs[idx])

        for idx, m in enumerate(metrics):

            axs[idx].set_title(m)

    def preview(self):

        self.fit(dry_run=True)

    def render_model(
        self,
        render_params: bool = True,
        render_distributions: bool = True,
    ):

        self.model_rendering = mfmf.metrics.render_model(
            model=self,
            render_params=render_params,
            render_distributions=render_distributions,
        )

        return self.model_rendering

    @property
    def features_metadata(self):

        features_metadata = pd.DataFrame(
            [
                [feature, view_name]
                for view_name, feature_list in self.views.items()
                for feature in feature_list.feature_names
            ],
            columns=["feature", "view"],
        )

        features_metadata = features_metadata.set_index("feature")

        return features_metadata

    def save_to_hdf5(self, path: str, quiet: bool = True):

        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")

        import h5py

        # if file exists, we'll overwrite it
        try:
            os.remove(path)
            print("File already exists, overwriting.")
        except OSError:
            pass

        hf = h5py.File(path, "a")

        # add data ------------------------------------------------------------
        for fg in self.feature_groups:

            hf_view = hf.create_group("data/" + fg["associated_view"])

            for og in self.obs_groups:

                from_idx, to_idx = self.feature_offsets[fg["associated_view"]]
                tmp = self._merged_view.data.cpu().detach().numpy()[:, from_idx:to_idx]

                hf_view.create_dataset(
                    name=og["group_name"],
                    data=tmp,
                    fillvalue=None,
                )

        # add W per view ------------------------------------------------------
        # MOFA doesn't support split views so we also won't here
        if len(self.feature_groups) == len(self.view_names):

            hf_expectations_w = hf.create_group("expectations/W/")

            for view_name in self.view_names:

                hf_expectations_w.create_dataset(
                    name=view_name,
                    data=self.results[view_name]["w"],
                    fillvalue=None,
                )

        else:

            raise NotImplementedError(
                "Saving models with feature groups is not yet supported."
            )

        # add Z per group -----------------------------------------------------
        hf_expectations_z = hf.create_group("expectations/Z/")

        for og in self.obs_groups:

            hf_expectations_z.create_dataset(
                name=og["group_name"],
                data=self.results[og["group_name"]]["z"].squeeze().T,
                fillvalue=None,
            )

        # add feature names per view ------------------------------------------
        hf_features = hf.create_group("features/")

        for fg in self.feature_groups:

            hf_features.create_dataset(
                name=fg["associated_view"],
                data=fg["features"],
                fillvalue=None,
            )

        # add groups ----------------------------------------------------------
        hf_groups = hf.create_group("groups/")

        hf_groups.create_dataset(
            name="groups",
            data=self.groups,
            fillvalue=None,
        )

        # add intercepts ------------------------------------------------------
        for fg in self.feature_groups:

            hf_intercepts = hf.create_group("intercepts/" + fg["associated_view"])

            for og in self.obs_groups:

                # TODO(ttreis): figure out how to calculate those
                hf_intercepts.create_dataset(
                    name=og["group_name"],
                    data=np.random.randint(0, 100, size=(len(fg["features"]),)),
                    fillvalue=None,
                )

        # add model options ---------------------------------------------------
        hf_model_opts = hf.create_group("model_options/")

        # TODO(ttreis): extend mofax to allow that

        hf_model_opts.create_dataset(
            name="likelihoods",
            data=[x for x in self.likelihoods],
            fillvalue=None,
        )

        hf_model_opts.create_dataset(
            name="ard_factors",
            data=["True"],
            fillvalue=None,
        )

        hf_model_opts.create_dataset(
            name="ard_weights",
            data=["True"],
        )

        hf_model_opts.create_dataset(
            name="spikeslab_factors",
            data=["True"],
            fillvalue=None,
        )

        hf_model_opts.create_dataset(
            name="spikeslab_weights",
            data=["True"],
            fillvalue=None,
        )

        # for fg in self.feature_groups:

        #     for og in self.obs_groups:

        #         hf_opts = hf.create_group("model_options/" + "/" + fg["associated_view"] + "/" + og["group_name"])

        #         # TODO(ttreis): figure out how to calculate those
        #         hf_opts.create_dataset(
        #             name=og["group_name"],
        #             data=np.random.randint(0, 100, size=(fg["features"].shape[0],)),
        #         )

        # add samples ---------------------------------------------------------
        hf_samples = hf.create_group("samples/")

        for og in self.obs_groups:

            hf_samples.create_dataset(
                name=og["group_name"],
                data=og["observations"],
                fillvalue=None,
            )

        # add samples metadata ------------------------------------------------
        for og in self.obs_groups:

            hf_samples_meta = hf.create_group("samples_metadata/" + og["group_name"])

            metadata_per_og = self.metadata.loc[
                self.metadata.index.isin(og["observations"])
            ]

            hf_samples_meta.create_dataset(
                name="sample",
                data=metadata_per_og.index.values.tolist(),
                fillvalue=None,
            )

            for col in metadata_per_og.columns:

                try:
                    hf_samples_meta.create_dataset(
                        name=col,
                        data=np.array(
                            [i for i in metadata_per_og[col].astype("|S").values]
                        ),
                        fillvalue=None,
                    )
                except (UnicodeDecodeError, AttributeError):
                    print("Could not decode column " + col + " to utf-8. Skipping.")

        # add training options ------------------------------------------------
        # TODO(ttreis): figure out what mofax puts here
        hf.create_dataset(
            name="training_opts",
            data=np.random.randint(0, 100, size=(5,)),
            fillvalue=None,
        )

        # add training stats --------------------------------------------------
        hf_training_stats = hf.create_group("training_stats/")

        hf_training_stats.create_dataset(
            name="elbo",
            data=[
                self.loss_during_training.loss.head(1).values,
                self.loss_during_training.loss.tail(1).values,
            ],
            fillvalue=None,
        )

        # TODO(ttreis): figure out what mofax puts here
        hf_training_stats.create_dataset(
            name="time",
            data=[0, 0],
            fillvalue=None,
        )

        # TODO(ttreis): figure out why mofax puts 2 values here
        hf_training_stats.create_dataset(
            name="number_factors",
            data=[self.n_factors, np.nan],
            fillvalue=None,
        )

        # add variance explained ----------------------------------------------
        hf_variance_total = hf.create_group("variance_explained/r2_total/")
        hf_variance_per_factor = hf.create_group("variance_explained/r2_per_factor/")

        for og in self.obs_groups:

            hf_variance_total.create_dataset(
                name=og["group_name"],
                data=self.calculate_variance_explained(
                    groups=og["group_name"]
                ).R2.values,
                fillvalue=None,
            )

            r2_per_factor = [
                self.get_variance_explained(
                    groups=og["group_name"], views=[fg["associated_view"]]
                ).R2.values
                for fg in self.feature_groups
            ]

            hf_variance_per_factor.create_dataset(
                name=og["group_name"],
                data=r2_per_factor,
                fillvalue=None,
            )

        # add views -----------------------------------------------------------
        hf_views = hf.create_group("views")

        # using self.likelihoods so that the order is the same
        hf_views.create_dataset(
            name="views",
            data=[x for x in self.likelihoods.keys()],
            fillvalue=None,
        )

        hf.close()

    ###############################################################################
    # functions for mofax support #################################################

    def get_shape(self, groups=None, views=None):
        """
        Get the shape of all the data, samples (cells) and features pulled across groups and views.
        Parameters
        ----------
        groups : optional
            List of groups to consider
        views : optional
            List of views to consider
        """

        shape = mfmf.mofax_support.get_shape(model=self, groups=groups, views=views)

        return shape

    def get_samples(self, groups=None):
        """
        Get the sample metadata table (sample ID and its respective group)
        Parameters
        ----------
        groups : optional
            List of groups to consider
        """

        samples = mfmf.mofax_support.get_samples(model=self, groups=groups)

        return samples

    # Alias samples as cells
    def get_cells(self, groups=None):
        """
        Get the cell metadata table (cell ID and its respective group)
        Parameters
        ----------
        groups : optional
            List of groups to consider
        """

        cells = mfmf.mofax_support.get_cells(model=self, groups=groups)

        return cells

    def get_features(self, views=None):
        """
        Get the features metadata table (feature name and its respective view)
        Parameters
        ----------
        views : optional
            List of views to consider
        """

        features = mfmf.mofax_support.get_features(model=self, views=views)

        return features

    def get_groups(self):
        """
        Get the groups names
        """
        return self.groups

    def get_views(self):
        """
        Get the views names
        """
        return self.views

    def get_top_features(
        self,
        factors: Union[int, List[int]] = None,
        views: Union[str, int, List[str], List[int]] = None,
        n_features: int = None,
        clip_threshold: float = None,
        scale: bool = False,
        absolute_values: bool = False,
        only_positive: bool = False,
        only_negative: bool = False,
        per_view: bool = True,
        df: bool = False,
    ):
        """
        Fetch a list of top feature names
        Parameters
        ----------
        factors : optional
            Factors to use (all factors in the model by default)
        view : options
            The view to get the factor weights for (first view by default)
        n_features : optional
            Number of features for each factor by their absolute value (10 by default)
        clip_threshold : optional
            Absolute weight threshold to clip all values to (no threshold by default)
        absolute_values : optional
            If to fetch absolute weight values
        only_positive : optional
            If to fetch only positive weights
        only_negative : optional
            If to fetch only negative weights
        per_view : optional
            Get n_features per view rather than globally (True by default)
        df : optional
            Boolean value if to return a DataFrame
        """

        features = mfmf.mofax_support.get_top_features(
            model=self,
            factors=factors,
            views=views,
            n_features=n_features,
            clip_threshold=clip_threshold,
            scale=scale,
            absolute_values=absolute_values,
            only_positive=only_positive,
            only_negative=only_negative,
            per_view=per_view,
            df=df,
        )

        return features

    def get_factors(
        self,
        groups: Union[str, int, List[str], List[int]] = None,
        factors: Optional[Union[int, List[int], str, List[str]]] = None,
        df: bool = False,
        concatenate_groups: bool = True,
        scale: bool = False,
        absolute_values: bool = False,
    ):
        """
        Get the matrix with factors as a NumPy array or as a DataFrame (df=True).
        Parameters
        ----------
        groups : optional
            List of groups to consider
        factors : optional
            Indices of factors to consider
        df : optional
            Boolean value if to return the factor matrix Z as a (wide) pd.DataFrame
        concatenate_groups : optional
            If concatenate Z matrices (True by default)
        scale : optional
            If return values scaled to zero mean and unit variance
            (per factor when concatenated or per factor and per group otherwise)
        absolute_values : optional
            If return absolute values for weights
        """

        z = mfmf.mofax_support.get_factors(
            model=self,
            groups=groups,
            factors=factors,
            df=df,
            concatenate_groups=concatenate_groups,
            scale=scale,
            absolute_values=absolute_values,
        )

        return z

    # def get_interpolated_factors(
    #     self,
    #     groups: Union[str, int, List[str], List[int]] = None,
    #     factors: Optional[Union[int, List[int], str, List[str]]] = None,
    #     df: bool = False,
    #     df_long: bool = False,
    #     concatenate_groups: bool = True,
    #     scale: bool = False,
    #     absolute_values: bool = False,
    # ):
    #     """
    #     Get the matrix with interpolated factors.
    #     If df_long is False, a dictionary with keys ("mean", "variance") is returned
    #     with NumPy arrays (df=False) or DataFrames (df=True) as values.
    #     If df_long is True, a DataFrame with columns ("new_value", "factor", "mean", "variance")
    #     is returned.
    #     Parameters
    #     ----------
    #     groups : optional
    #         List of groups to consider
    #     factors : optional
    #         Indices of factors to consider
    #     df : optional
    #         Boolean value if to return mean and variance matrices as (wide) DataFrames
    #         (can be superseded by df_long=True)
    #     df_long : optional
    #         Boolean value if to return a single long DataFrame
    #         (supersedes df=False and concatenate_groups=False)
    #     concatenate_groups : optional
    #         If concatenate Z matrices (True by default, can be superseded by df_long=True)
    #     scale : optional
    #         If return values scaled to zero mean and unit variance
    #         (per factor when concatenated or per factor and per group otherwise)
    #     absolute_values : optional
    #         If return absolute values for weights
    #     """

    #     groups = self._check_groups(groups)
    #     factor_indices, factors = self._check_factors(factors)

    #     z_interpolated = dict()
    #     new_values_names = tuple()
    #     if self.covariates_names:
    #         new_values_names = tuple(
    #             [f"{value}_transformed" for value in self.covariates_names]
    #         )
    #     else:
    #         new_values_names = tuple(
    #             [
    #                 f"new_value{i}"
    #                 for i in range(self.interpolated_factors["new_values"].shape[1])
    #             ]
    #         )

    #     for stat in ["mean", "variance"]:
    #         # get factors
    #         z = list(
    #             np.array(self.interpolated_factors[stat][g])[:, factor_indices]
    #             for g in groups
    #         )

    #         # consider transformations
    #         for g in range(len(groups)):
    #             if not concatenate_groups:
    #                 if scale:
    #                     z[g] = (z[g] - z[g].mean(axis=0)) / z[g].std(axis=0)
    #                 if absolute_values:
    #                     z[g] = np.absolute(z[g])
    #             if df or df_long:
    #                 z[g] = pd.DataFrame(z[g])
    #                 z[g].columns = factors
    #                 z[g]["group"] = self.groups[g]

    #                 if "new_values" in self.interpolated_factors:
    #                     new_values = np.array(self.interpolated_factors["new_values"])
    #                 else:
    #                     new_values = np.arange(z[g].shape[0]).reshape(-1, 1)

    #                 new_values = pd.DataFrame(new_values, columns=new_values_names)

    #                 z[g] = pd.concat([z[g], new_values], axis=1)

    #                 # If groups are to be concatenated (but not in a long DataFrame),
    #                 # index has to be made unique per group
    #                 new_samples = [
    #                     f"{groups[g]}_{'_'.join(value.astype(str))}"
    #                     for _, value in new_values.iterrows()
    #                 ]

    #                 z[g].index = new_samples

    #                 # Create an index for new values
    #                 z[g]["new_value"] = np.arange(z[g].shape[0])

    #         # concatenate views if requested
    #         if concatenate_groups:
    #             z = pd.concat(z) if df or df_long else np.concatenate(z)
    #             if scale:
    #                 z = (z - z.mean(axis=0)) / z.std(axis=0)
    #             if absolute_values:
    #                 z = np.absolute(z)

    #         # melt DataFrames
    #         if df_long:
    #             if not concatenate_groups:  # supersede
    #                 z = pd.concat(z)
    #             z = (
    #                 z.rename_axis("new_sample", axis=0)
    #                 .reset_index()
    #                 .melt(
    #                     id_vars=["new_sample", *new_values_names, "group"],
    #                     var_name="factor",
    #                     value_name=stat,
    #                 )
    #             )

    #         z_interpolated[stat] = z

    #     if df_long:
    #         z_interpolated = (
    #             z_interpolated["mean"]
    #             .set_index(["new_sample", *new_values_names, "group", "factor"])
    #             .merge(
    #                 z_interpolated["variance"],
    #                 on=("new_sample", *new_values_names, "group", "factor"),
    #             )
    #         )

    # #     return z_interpolated

    # def get_group_kernel(self):
    #     model_groups = False
    #     if (
    #         self.options
    #         and "smooth" in self.options
    #         and "model_groups" in self.options["smooth"]
    #     ):
    #         model_groups = bool(self.options["smooth"]["model_groups"].item().decode())

    #     kernels = list()
    #     if not model_groups or self.ngroups == 1:
    #         Kg = np.ones(shape=(self.nfactors, self.ngroups, self.ngroups))
    #         return Kg
    #     else:
    #         if self.training_stats and "Kg" in self.training_stats:
    #             return self.training_stats["Kg"]
    #         else:
    #             raise ValueError(
    #                 "No group kernel was saved. Specify the covariates and train the MEFISTO model with the option 'model_groups' set to True."
    #             )

    def get_weights(
        self,
        views: Union[str, int, List[str], List[int]] = None,
        factors: Union[int, List[int]] = None,
        df: bool = False,
        scale: bool = False,
        concatenate_views: bool = True,
        absolute_values: bool = False,
    ):
        """
        Fetch the weight matrices
        Parameters
        ----------
        views : optional
            List of views to consider
        factors : optional
            Indices of factors to use
        df : optional
            Boolean value if to return W matrix as a (wide) pd.DataFrame
        scale : optional
            If return values scaled to zero mean and unit variance
            (per factor when concatenated or per factor and per view otherwise)
        concatenate_weights : optional
            If concatenate W matrices (True by default)
        absolute_values : optional
            If return absolute values for weights
        """

        w = mfmf.mofax_support.get_weights(
            model=self,
            views=views,
            factors=factors,
            df=df,
            scale=scale,
            concatenate_views=concatenate_views,
            absolute_values=absolute_values,
        )

        return w

    def get_data(
        self,
        views: Optional[Union[str, int]] = None,
        features: Optional[Union[str, List[str]]] = None,
        groups: Optional[Union[str, int, List[str], List[int]]] = None,
        df: bool = False,
    ):
        """
        Fetch the training data
        Parameters
        ----------
        view : optional
            view to consider
        features : optional
            Features to consider (from one view)
        groups : optional
            groups to consider
        df : optional
            Boolean value if to return Y matrix as a DataFrame
        """

        y = mfmf.mofax_support.get_data(
            model=self,
            views=views,
            features=features,
            groups=groups,
            df=df,
        )

        return y

    # def fetch_values(self, variables: Union[str, List[str]], unique: bool = True):
    #     """
    #     Fetch metadata column, factors, or feature values
    #     as well as covariates.
    #     Shorthand to get_data, get_factors, metadata, and covariates calls.
    #     Parameters
    #     ----------
    #     variables : str
    #         Features, metadata columns, or factors (FactorN) to fetch.
    #         For MEFISTO models with covariates, covariates are accepted
    #         such as 'covariate1' or 'covariate1_transformed'.
    #     """
    #     # If a sole variable name is used, wrap it in a list
    #     if not isinstance(variables, Iterable) or isinstance(variables, str):
    #         variables = [variables]

    #     # Remove None values and duplicates
    #     variables = [i for i in variables if i is not None]
    #     # Transform integers to factors
    #     variables = maybe_factor_indices_to_factors(variables)
    #     if unique:
    #         variables = pd.Series(variables).drop_duplicates().tolist()

    #     var_meta = list()
    #     var_features = list()
    #     var_factors = list()
    #     var_covariates = list()

    #     # Split all the variables into metadata and features
    #     for i, var in enumerate(variables):
    #         if var in self.metadata.columns:
    #             var_meta.append(var)
    #         elif var.capitalize().startswith("Factor"):
    #             # Unify factor naming
    #             variables[i] = var.capitalize()
    #             var_factors.append(var.capitalize())
    #         elif (
    #             self.covariates_names is not None
    #             and (
    #                 var in self.covariates_names
    #                 or var in [f"{cov}_transformed" for cov in self.covariates_names]
    #             )
    #             and self.covariates is not None
    #         ):
    #             var_covariates.append(var)
    #         else:
    #             var_features.append(var)

    #     var_list = list()
    #     if len(var_meta) > 0:
    #         var_list.append(self.metadata[var_meta])
    #     if len(var_features) > 0:
    #         var_list.append(self.get_data(features=var_features, df=True))
    #     if len(var_factors) > 0:
    #         var_list.append(self.get_factors(factors=var_factors, df=True))
    #     if len(var_covariates) > 0:
    #         var_list.append(self.covariates[var_covariates])

    #     # Return a DataFrame with columns ordered as requested
    #     return pd.concat(var_list, axis=1).loc[:, variables]

    def _check_views(self, views):

        views = mfmf.mofax_support._check_views(model=self, views=views)

        return views

    def _check_groups(self, groups):

        groups = mfmf.mofax_support._check_groups(model=self, groups=groups)

        return groups

    def _check_factors(self, factors, unique=False):

        factor_indices, factors = mfmf.mofax_support._check_factors(
            model=self, factors=factors, unique=unique
        )

        return (factor_indices, factors)

    def calculate_variance_explained(
        self,
        # factor_index: int,
        factors: Optional[Union[int, List[int], str, List[str]]] = None,
        groups: Optional[Union[str, int, List[str], List[int]]] = None,
        views: Optional[Union[str, int, List[str], List[int]]] = None,
        group_label: Optional[str] = None,
        per_factor: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Calculate the variance explained estimates for each factor in each view and/or group.
        Allow also for predefined groups
        Parameters
        ----------
        factors : optional
            List of factors to consider (default is None, all factors)
        groups : optional
            List of groups to consider (default is None, all groups)
        views : optional
            List of views to consider (default is None, all views)
        group_label : optional
            Group label to split samples by (default is None)
        per_factor : optional
            If calculate R2 per factor or for all factors (default)
        """

        r2_df = mfmf.mofax_support.calculate_variance_explained(
            model=self,
            factors=factors,
            groups=groups,
            views=views,
            group_label=group_label,
            per_factor=per_factor,
        )

        return r2_df

    def get_variance_explained(
        self,
        factors: Optional[Union[int, List[int], str, List[str]]] = None,
        groups: Optional[Union[str, int, List[str], List[int]]] = None,
        views: Optional[Union[str, int, List[str], List[int]]] = None,
    ) -> pd.DataFrame:
        """
        Get variance explained estimates (R2) for each factor across  view(s) and/or group(s).
        factors : optional
            List of factors to consider (all by default)
        groups : optional
            List of groups to consider (all by default)
        views : optional
            List of views to consider (all by default)
        """

        r2 = mfmf.mofax_support.get_variance_explained(
            model=self,
            factors=factors,
            groups=groups,
            views=views,
        )

        return r2

    def _get_factor_r2_null(
        self,
        factor_index: int,
        groups_df: Optional[pd.DataFrame],
        group_label: Optional[str],
        n_iter=100,
        return_full=False,
        return_true=False,
        return_pvalues=True,
        fdr=True,
    ) -> pd.DataFrame:

        r2 = mfmf.mofax_support._get_factor_r2_null(
            model=self,
            factors_index=factor_index,
            groups_df=groups_df,
            group_label=group_label,
            n_iter=n_iter,
            return_full=return_full,
            return_true=return_true,
            return_pvalues=return_pvalues,
            fdr=fdr,
        )

        return r2

    def _get_r2_null(
        self,
        factors: Union[int, List[int], str, List[str]] = None,
        n_iter: int = 100,
        groups_df: Optional[pd.DataFrame] = None,
        group_label: Optional[str] = None,
        return_full=False,
        return_pvalues=True,
        fdr=True,
    ) -> pd.DataFrame:

        r2 = mfmf.mofax_support._get_r2_null(
            model=self,
            factors=factors,
            n_iter=n_iter,
            groups_df=groups_df,
            group_label=group_label,
            return_full=return_full,
            return_pvalues=return_pvalues,
            fdr=fdr,
        )

        return r2

    def get_sample_r2(
        self,
        factors: Optional[Union[str, int, List[str], List[int]]] = None,
        groups: Optional[Union[str, int, List[str], List[int]]] = None,
        views: Optional[Union[str, int, List[str], List[int]]] = None,
        df: bool = True,
    ) -> pd.DataFrame:

        r2 = mfmf.mofax_support.get_sample_r2(
            model=self,
            factors=factors,
            groups=groups,
            views=views,
            df=df,
        )

        return r2

    def project_data(
        self,
        data,
        view: Union[str, int] = None,
        factors: Union[int, List[int], str, List[str]] = None,
        df: bool = False,
        feature_intersection: bool = False,
    ):
        """
        Project new data onto the factor space of the model.
        For the projection, a pseudo-inverse of the weights matrix is calculated
        and its product with the provided data matrix is calculated.
        Parameters
        ----------
        data
            Numpy array or Pandas DataFrame with the data matching the number of features
        view : optional
            A view of the model to consider (first view by default)
        factors : optional
            Indices of factors to use for the projection (all factors by default)
        """

        zpred = mfmf.mofax_support.project_data(
            model=self,
            data=data,
            view=view,
            factors=factors,
            df=df,
            feature_intersection=feature_intersection,
        )

        return zpred

    def get_views_contributions(self, scaled: bool = True):
        """
        Project new data onto the factor space of the model.
        For the projection, a pseudo-inverse of the weights matrix is calculated
        and its product with the provided data matrix is calculated.
        Parameters
        ----------
        scaled : bool, optional
            Whether to scale contributions scores per sample
            so that they sum up to 1 (True by default)
        Returns
        -------
        pd.DataFrame
            Dataframe with view contribution scores, samples in rows and views in columns
        """

        view_contribution = mfmf.mofax_support.get_views_contributions(
            model=self,
            scaled=scaled,
        )

        return view_contribution

    def run_umap(
        self,
        groups: Union[str, int, List[str], List[int]] = None,
        factors: Union[int, List[int]] = None,
        n_neighbors: int = 10,
        min_dist: float = 0.5,
        spread: float = 1.0,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        """
        Run UMAP on the factor space
        Parameters
        ----------
        n_neighbors : optional
            UMAP parameter: number of neighbors.
        min_dist
            UMAP parameter: the effective minimum distance between embedded points. Smaller values
            will result in a more clustered/clumped embedding where nearby points on
            the manifold are drawn closer together, while larger values will result
            on a more even dispersal of points. The value should be set relative to
            the ``spread`` value, which determines the scale at which embedded
            points will be spread out.
        spread
            UMAP parameter: the effective scale of embedded points. In combination with `min_dist`
            this determines how clustered/clumped the embedded points are.
        random_state
            random seed
        """

        if "UMAP1" in self.metadata.columns and "UMAP2" in self.metadata.columns:
            self.metadata = self.metadata.drop(["UMAP1", "UMAP2"], axis=1)

        umap = mfmf.mofax_support.run_umap(
            model=self,
            groups=groups,
            factors=factors,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            random_state=random_state,
            **kwargs,
        )

        # merge with samples metadata
        self.metadata = pd.merge(
            left=self.metadata,
            right=umap,
            left_index=True,
            right_index=True,
        )

        print("UMAP coordinates added to the samples_metadata")


class FactorModelGroup:
    def __init__(self, models: Optional[Dict[str, mfmf.core.FactorModel]] = None):
        """
        A group of FactorModel objects

        Parameters
        ----------
        models (Dict[str, mfmf.core.FactorModel]): A dictionary of models
        """

        self._models = None

        if models is not None:
            if not isinstance(models, dict):
                raise TypeError("Parameter 'models' must be a dictionary.")

            for id, model in models.items():
                self.add_model(id, model)

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, *args):

        raise AttributeError(
            "Please use 'FactorModelGroup.add_model()' to add a model."
        )

    def add_model(self, id: str, model: mfmf.core.FactorModel):
        """
        Add a model to the group

        Parameters
        ----------
        id (str): An identifier for the specific model in the group
        model (mfmf.core.FactorModel): The model
        """

        if not isinstance(id, str):
            raise TypeError("Parameter 'id' must be a string.")

        if not isinstance(model, mfmf.core.FactorModel):
            raise TypeError(
                "Parameter 'model' must be of type 'mfmf.core.FactorModel'."
            )

        if self._models is None:
            self._models = {}

        self.models[id] = model

    def fit(self, logging: bool = True):
        """Train all models in the group

        Args:
            logging (bool, optional): _description_. Defaults to True.
        """

        for model in self.models.values():
            model.fit(sample_reg="horseshoe", feature_reg="horseshoe", logging=logging)

    def compare_models(self):

        comparison = pd.DataFrame()

        for id, model in self.models.items():

            first = model.loss_during_training.loss.head(1).values[0]
            last = model.loss_during_training.loss.tail(1).values[0]
            sum_var_explained = sum(model.calculate_variance_explained().R2)

            comparison = pd.concat(
                [
                    comparison,
                    pd.DataFrame(
                        {
                            "model": [id],
                            "before_training": [first],
                            "after_training": [last],
                            "decrease": [first - last],
                            "decrease_pct": [(last / first) * 100],
                            "sum_var_explained": [sum_var_explained],
                        },
                    ),
                ]
            )

        comparison["scaled_var_explained"] = (
            comparison["sum_var_explained"] - comparison["sum_var_explained"].min()
        ) / (
            comparison["sum_var_explained"].max()
            - comparison["sum_var_explained"].min()
        )

        comparison["scaled_ELBO_loss"] = (
            comparison["decrease"] - comparison["decrease"].miFn()
        ) / (comparison["decrease"].max() - comparison["decrease"].min())

        return comparison
