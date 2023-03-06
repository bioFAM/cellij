from __future__ import annotations
from typing import Optional, List
import os
import torch
import anndata
import muon as mu
import numpy as np
from functools import reduce
import pandas as pd

from sklearn.impute import KNNImputer

from importlib import resources


class DataContainer:
    """Container to hold all data for a FactorModel.

    Holds a list of anndata objects and provides methods to add, remove, and
    modify data. Also provides methods to merge those anndata and returns them
    as a tensor for training or as a mudata for the user.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._values = torch.Tensor()
        self._views = {}
        self._names = []
        self._obs_names = {}
        self._feature_names = {}
        self._obs_idx = {}
        self._feature_idx = {}
        self._metadata = {}

    @property
    def values(self):
        return self._values
    
    @property
    def views(self):
        return self._views

    @property
    def names(self):
        return self._names

    @property
    def obs_names(self):
        return self._obs_names
    
    @property
    def n_obs(self):
        return len(self._obs_names)
    
    @property
    def n_features(self):
        return len(self._feature_names)

    @property
    def feature_names(self):
        return self._feature_names

    def add_data(
        self,
        data: anndata.AnnData,
        name: str,
        **kwargs,
    ):

        if not isinstance(data, anndata.AnnData):
            raise TypeError("Data must be a anndata.AnnData.")

        if name in self._names:
            raise ValueError(f"Data with name {name} already exists.")

        self._names.append(name)

        has_metadata = data.obs is not None and len(data.obs) > 0

        if has_metadata:
            self._metadata[name] = data.obs

        # TODO(ttreis)
        # delete duplicate metadata entries, might arise when mudata has
        # metadata. Then, the metadata is attached to every resulting
        # anndata object

        self._views[name] = data
        self._obs_names[name] = data.obs_names.to_list()
        self._feature_names[name] = data.var_names.to_list()

    def merge_views(self, **kwargs):

        """Merges all views into a single tensor."""

        views = {}
        for name in self._names:
            views[name] = self._views[name].to_df().sort_index()

        merged_view = reduce(
            lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True, how="outer"
            ),
            views.values(),
        )
        merged_obs_names = merged_view.index.to_list()
        merged_feature_names = merged_view.columns

        na_strategy = kwargs.get("na_strategy", None)

        if na_strategy == "knn_by_obs":

            if "k" not in kwargs:
                k = int(np.round(np.sqrt(merged_view.shape[0])))
            else:
                k = kwargs["k"]
            imputer = KNNImputer(n_neighbors=k)
            merged_df_imputed = pd.DataFrame(
                imputer.fit_transform(merged_view.values), index=merged_obs_names, columns=merged_feature_names
            )
            self._data = torch.Tensor(merged_df_imputed.values)

        elif na_strategy is None:

            self._data = torch.Tensor(merged_view.values)
            
        self._obs_names = merged_obs_names
        self._feature_names = merged_feature_names

        for name in self._names:

            view_obs_names = self._views[name].obs_names.to_list()
            view_feature_names = self._views[name].var_names.to_list()

            self._obs_idx[name] = [
                i for i, val in enumerate(merged_obs_names) if val in view_obs_names
            ]

            self._feature_idx[name] = [
                i for i, val in enumerate(merged_feature_names) if val in view_feature_names
            ]

        

    def to_df(self) -> pd.DataFrame:
        """Returns a pandas dataframe holding the data with the given name."""

        res = pd.DataFrame(
            self._values.numpy(), index=self._obs_names, columns=self._feature_names
        )

        return res

    def to_anndata(self) -> anndata.AnnData:

        """Returns an anndata object holding the data with the given name."""

        res = pd.DataFrame(
            self._values.numpy(), index=self._obs_names, columns=self._feature_names
        )

        return anndata.AnnData(res)


class Importer:
    """Class to facilitate easy import of different data."""

    def __init__(self, encoding="utf_8"):

        self.encoding = encoding

    def load_CLL(self, use_drug_compound_names=True) -> mu.MuData:
        """Loads a published multi-omics data set.

        This function provides a small multi-omics data set to be used for testing.
        It's available at: https://muon-tutorials.readthedocs.io/en/latest/CLL.html.

        The function returns a muon.MuData object with 4 modalities and associated
        metadata for 200 patients each:
          - Drugs: 200 x 310
          - Methylation: 200 x 4248
          - mRNA: 200 x 5000
          - Mutations: 200 x 69

        The associated metadata contains the columns:
          - Gender: m (male), f (female)
          - Age: age in years
          - TTT: time (in years) which passed from taking the sample to the next treatment
          - TTD: time (in years) which passed from taking the sample to patients' death
          - treatedAfter: (TRUE/FALSE)
          - Died: whether the patient died (TRUE/FALSE)
          - IGHV_status
          - trisomy12


        Parameters
        ----------
        use_drug_compound_names : bool, optional
            If True, the drug names are replaced by the compound names, by default True

        Returns
        -------
        mu.MuData
            A muon.MuData object with 4 modalities and associated metadata for 200 patients each

        """

        with resources.path("cellij.data", "cll_metadata.csv") as res_path:

            obs = pd.read_csv(
                filepath_or_buffer=os.fspath(res_path),
                sep=",",
                index_col="Sample",
                encoding=self.encoding,
            )

        modalities = {}

        for ome in ["drugs", "methylation", "mrna", "mutations"]:

            with resources.path("cellij.data", f"cll_{ome}.csv") as res_path:

                modality = pd.read_csv(
                    filepath_or_buffer=os.fspath(res_path),
                    sep=",",
                    index_col=0,
                    encoding=self.encoding,
                ).T

                modalities[ome] = anndata.AnnData(X=modality, dtype="float32")

                if use_drug_compound_names and ome == "drugs":

                    with resources.path(
                        "cellij.data", "id_to_drug_names.csv"
                    ) as compound_path:

                        compound_names = pd.read_csv(
                            filepath_or_buffer=os.fspath(compound_path),
                            sep=";",
                            header=None,
                            encoding=self.encoding,
                        )
                        compound_names = pd.DataFrame(compound_names)
                        compound_names.columns = ["id", "name"]

                    ome_colnames = modalities[ome].var_names.tolist()

                    for i, colname in enumerate(ome_colnames):
                        base_id = colname[0 : len(colname) - 2]
                        drug_name_for_base_id = compound_names.query("id == @base_id")[
                            "name"
                        ].values[0]
                        # print(drug_name_for_base_id)
                        ome_colnames[i] = colname.replace(
                            f"{base_id}", drug_name_for_base_id
                        )

                    modalities[ome].var_names = ome_colnames

        mdata = mu.MuData(modalities)
        mdata.obs = mdata.obs.join(obs)

        return mdata
