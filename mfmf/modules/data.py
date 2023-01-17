from __future__ import annotations

from typing import Union, List

import anndata
import muon
import numpy as np
import torch
import logging

import mfmf


class View:
    """Holds one data set and extracts the relevant information."""

    def __init__(
        self,
        name: str,
        data: Union[np.ndarray, torch.Tensor, anndata.AnnData],
    ) -> None:
        """Holds a single named 2D torch.tensor.

        Args:
            name (str):
                A name given to the view.
            data (Union[np.ndarray, torch.Tensor, anndata.AnnData]):
                A 2D data container with the shape (n_obs, n_features).

        Example:
            $ import numpy
            $ import torch
            $ import mfmf
            $
            $ my_numpy = numpy.random.normal(size=[5, 10])
            $ my_tensor = torch.normal(mean=0, std=1, size=[3, 20])
            $
            $ imp = mfmf.importer.Importer()
            $ cll_data = imp.load_CLL()
            $ my_anndata = cll_data["mrna"]
            $
            $ my_numpy_view = View(name="numpy", data=my_numpy)
            $ my_tensor_view = View(name="torch", data=my_tensor)
            $ my_anndata_view = View(name="anndata", data=my_anndata)

        """

        if isinstance(data, muon.MuData):
            raise TypeError(
                "Please use function 'FactorModel.add_MultiView()' for MuData objects."
            )
        elif not isinstance(data, (np.ndarray, torch.Tensor, anndata.AnnData)):
            raise TypeError(
                "Parameter 'data' must be one of ['numpy.ndarray', 'torch.Tensor', 'anndata.AnnData']."
            )

        # Parse data and extract relevant information
        if isinstance(data, (np.ndarray, torch.Tensor)):

            if not len(data.squeeze().shape) == 2:
                raise ValueError("Paramter 'data' must be a 2D matrix.")

            self._n_obs, self._n_features = data.squeeze().shape
            self.obs_names = [f"obs{i}" for i in range(self.n_obs)]
            self.feature_names = [f"feature{i}" for i in range(self.n_features)]
            self._data = torch.tensor(data) if isinstance(data, np.ndarray) else data

        elif isinstance(data, anndata.AnnData):

            # Can never be triggered because anndata checks for that
            # if not len(data.X.squeeze().shape) == 2:
            #     raise ValueError("Paramter 'data' must be a 2D matrix.")

            self._n_obs, self._n_features = data.X.squeeze().shape
            self.obs_names = list(data.obs_names)
            self.feature_names = list(data.var_names)
            self._data = torch.tensor(data.X)

        self.name = name

    def __repr__(self) -> str:

        return f"{self.name} ({self.n_obs} obs x {self.n_features} features)"

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):

        if not isinstance(name, str):
            raise TypeError("Parameter 'name' must be a string.")

        self._name = name

    @property
    def n_obs(self):
        return self._n_obs

    @n_obs.setter
    def n_obs(self, *args):

        raise AttributeError("Parameter 'n_obs' is derived from data and can't be set.")

    @property
    def obs_names(self):
        return self._obs_names

    @obs_names.setter
    def obs_names(self, obs_names: List[str]):

        if not isinstance(obs_names, list):
            raise TypeError("Parameter 'obs_names' must be a list.")
        if not all(isinstance(obs_name, str) for obs_name in obs_names):
            raise TypeError("All elements of 'obs_names' must be strings.")

        if len(obs_names) != self.n_obs:
            raise ValueError(
                "Parameter 'obs_names' must have the same length as the number of observations."
            )

        self._obs_names = obs_names

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, *args):

        raise AttributeError(
            "Parameter 'n_features' is derived from data and can't be set."
        )

    @property
    def feature_names(self):
        return self._feature_names

    @feature_names.setter
    def feature_names(self, feature_names: List[str]):

        if not isinstance(feature_names, list):
            raise TypeError("Parameter 'feature_names' must be a list.")
        if not all(isinstance(obs_name, str) for obs_name in feature_names):
            raise TypeError("All elements of 'feature_names' must be strings.")

        if len(feature_names) != self.n_features:
            raise ValueError(
                "Parameter 'feature_names' must be the same length as n_features."
            )

        self._feature_names = feature_names

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):

        self._data = data


class MergedView:
    """
    Class to merge multiple views into a single view.
    """

    def __init__(
        self,
        data: anndata.AnnData,
        feature_offsets: dict(tuple(int, int)),
    ):
        """
        Initialize MergedView object.
        """

        view_names = []

        if not isinstance(data, anndata.AnnData):
            raise TypeError("Parameter 'data' must be an anndata.AnnData object.")

        if not isinstance(feature_offsets, dict):
            raise TypeError("Parameter 'feature_offsets' must be a dict.")

        for view_name, (start, end) in feature_offsets.items():
            if not isinstance(view_name, str):
                raise TypeError("Keys of 'feature_offsets' must be strings.")
            if not isinstance(start, int):
                raise TypeError("Start indices of 'feature_offsets' must be integers.")
            if not isinstance(end, int):
                raise TypeError("End indices of 'feature_offsets' must be integers.")
            if not start < end:
                raise ValueError("Start indices must be smaller than end indices.")
            view_names.append(view_name)

        self.data = torch.tensor(data.X)
        self.feature_offsets = feature_offsets
        self.view_names = view_names

        self.n_obs = data.X.shape[0]
        self.n_features = data.X.shape[1]
        self.obs_names = data.obs_names
        self.feature_names = data.var_names

    @property
    def feature_offsets(self):
        return self._feature_offsets

    @feature_offsets.setter
    def feature_offsets(self, feature_offsets):

        self._feature_offsets = feature_offsets

    @property
    def view_names(self):
        return self._view_names

    @view_names.setter
    def view_names(self, view_names):

        self._view_names = view_names

    def get_view(self, view_name: str):
        """
        Get data from view with name 'view_name'.
        """
        if view_name not in self.view_names:
            raise ValueError(f"View with name '{view_name}' not found.")

        view_data = self.data[
            :, self.feature_offsets[view_name][0] : self.feature_offsets[view_name][1]
        ]

        if 0 in list(view_data.squeeze().shape):
            raise ValueError(
                f"View with name '{view_name}' and feature offsets [{self.feature_offsets[view_name][0]}, {self.feature_offsets[view_name][1]}] is empty."
            )

        view = mfmf.modules.data.View(name=view_name, data=view_data)

        return view
