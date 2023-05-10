from __future__ import annotations

import os
from functools import reduce
from importlib import resources

import anndata
import muon as mu
import numpy as np
import pandas as pd
import torch

import cellij


class DataContainer:
    """Container to hold all data for a FactorModel.

    Holds a list of anndata objects and provides methods to add, remove, and
    modify data. Also provides methods to merge those anndata and returns them
    as a tensor for training or as a mudata for the user.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._values = np.ndarray
        self._feature_groups = {}
        self._names = []
        self._obs_names = {}
        self._feature_names = {}
        self._merged_obs_names = []
        self._merged_feature_names = []
        self._obs_idx = {}
        self._feature_idx = {}
        self._metadata = {}

    @property
    def values(self):
        return self._values

    @property
    def feature_groups(self):
        return self._feature_groups

    @property
    def names(self):
        return self._names

    @property
    def obs_names(self):
        return self._obs_names

    @property
    def merged_obs_names(self):
        return self._merged_obs_names

    @merged_obs_names.setter
    def merged_obs_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError("Duplicate names found in observations.")

        self._merged_obs_names = names

    @property
    def n_obs(self):
        return len(self._merged_obs_names)

    @property
    def n_features(self):
        return len(self._merged_feature_names)

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def merged_feature_names(self):
        return self._merged_feature_names

    @merged_feature_names.setter
    def merged_feature_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError("Duplicate names found in features.")

        self._merged_feature_names = names

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

        self._feature_groups[name] = data
        self._obs_names[name] = data.obs_names.to_list()
        self._feature_names[name] = data.var_names.to_list()

    def merge_data(self, **kwargs):
        """Merges all feature_groups into a single tensor."""
        feature_groups = {}
        for name in self._names:
            feature_groups[name] = self._feature_groups[name].to_df().sort_index()

        merged_feature_group = reduce(
            lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="outer"),
            feature_groups.values(),
        )
        merged_obs_names = merged_feature_group.index.to_list()
        merged_feature_names = merged_feature_group.columns

        na_strategy = kwargs.get("na_strategy", None)
        if na_strategy is None:
            self._values = merged_feature_group.values

        else:
            self._values = cellij.impute_data(
                data=merged_feature_group,
                strategy=na_strategy,
                kwargs=kwargs,
            )

        self._merged_obs_names = merged_obs_names
        self._merged_feature_names = merged_feature_names

        for name in self._names:
            feature_group_obs_names = self._feature_groups[name].obs_names.to_list()
            feature_group_feature_names = self._feature_groups[name].var_names.to_list()

            self._obs_idx[name] = [i for i, val in enumerate(merged_obs_names) if val in feature_group_obs_names]

            self._feature_idx[name] = [
                i for i, val in enumerate(merged_feature_names) if val in feature_group_feature_names
            ]

    def to_df(self) -> pd.DataFrame:
        """Returns a 'pandas.DataFrame' representation of the contained data with feature and observation names."""
        print(self._values)
        res = pd.DataFrame(
            data=self._values,
            index=self._merged_obs_names,
            columns=self._merged_feature_names,
        )

        return res

    def to_anndata(self) -> anndata.AnnData:
        """Returns a 'anndata.AnnData' representation of the contained data with feature and observation names."""
        return anndata.AnnData(self.to_df())

    def to_tensor(self) -> torch.Tensor:
        """Returns a 'torch.Tensor' representation of the contained data."""
        return torch.Tensor(self._values)


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
                    with resources.path("cellij.data", "id_to_drug_names.csv") as compound_path:
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
                        drug_name_for_base_id = compound_names.query("id == @base_id")["name"].values[0]
                        # print(drug_name_for_base_id)
                        ome_colnames[i] = colname.replace(f"{base_id}", drug_name_for_base_id)

                    modalities[ome].var_names = ome_colnames

        mdata = mu.MuData(modalities)
        mdata.obs = mdata.obs.join(obs)

        return mdata

    def load_MEFISTO(self) -> mu.MuData:
        """Loads a synthetic data set with 4 views and a temporal covariate.

        Data is generated as illustrated in the MEFISTO tutorial:
        https://raw.githack.com/bioFAM/MEFISTO_tutorials/master/MEFISTO_temporal.html

        The function returns a muon.MuData object with 4 modalities with
        200 observations each:
          - view1: 200 x 200
          - view2: 200 x 200
          - view3: 200 x 200
          - view4: 200 x 200
        These data have one covariate defining a timeline. The simulation is
        based on 4 factors, two of which vary smoothly along the covariate (with
        different lengthscales) and two are independent of the covariate.

        Returns
        -------
        mu.MuData
            A muon.MuData object with 4 modalities

        """
        modalities = {}
        for ome in ["view1", "view2", "view3", "view4"]:
            with resources.path("cellij.data", f"gp_{ome}.csv") as res_path:
                modality = pd.read_csv(
                    filepath_or_buffer=os.fspath(res_path),
                    sep=",",
                    index_col=0,
                    encoding=self.encoding,
                ).T
                modalities[ome] = anndata.AnnData(X=modality, dtype="float32")

        return mu.MuData(modalities)

    def load_Guo2010(self) -> mu.MuData:
        """Loads a real data set with qPCR data (48 genes) on cell differentiation.

        Modified from https://pyro.ai/examples/gplvm.html:
        The data consists of single-cell qPCR data for 48 genes obtained from
        mice (Guo et al., 2010 [1]). This data is available at the Open Data
        Science repository. The data contains 48 columns, with each column
        corresponding to (normalized) measurements of each gene. Cells
        differentiate during their development and these data were obtained at
        various stages of development. The various stages are labelled from the
        1-cell stage to the 64-cell stage. For the 32-cell stage, the data is
        further differentiated into 'trophectoderm' (TE) and 'inner cell mass'
        (ICM). ICM further differentiates into 'epiblast' (EPI) and 'primitive
        endoderm' (PE) at the 64-cell stage. Each of the rows in the dataset is
        labelled with one of these stages.

        Returns
        -------
        mu.MuData
            A muon.MuData object with 1 modality and associated metadata
            - 437 observations
            - 48 features
            - metadata:
                - n_cells: number of cells in sample (int)
                - division: division stage, primary differentiation axis (int)
                - division_scaled: 'division' but scaled to [0.0, 1.0]
                - label: tissue label, secondary differentiation axis (str)

        """
        modalities = {}
        with resources.path("cellij.data", "guo2010_qpcr.csv") as res_path:
            modality = pd.read_csv(
                filepath_or_buffer=os.fspath(res_path),
                sep=",",
                index_col=0,
                encoding=self.encoding,
            )
            cell_labels = modality.index.values.tolist()
            modality.index = [i.replace(" ", "_") for i in modality.index.values.tolist()]
            modality.index = [f"obs{idx}_{name}" for name, idx in zip(modality.index, range(len(modality.index)))]
            modalities["qPCR"] = anndata.AnnData(X=modality, dtype="float32")

            mdata = mu.MuData(modalities)
            metadata = pd.DataFrame()
            metadata["n_cells"] = [int(label.split(" ")[0]) for label in cell_labels]
            metadata["label"] = [label.split(" ")[1] if len(label.split(" ")) == 2 else None for label in cell_labels]
            metadata["division"] = np.log2(metadata["n_cells"]).astype(int)
            metadata["division_scaled"] = metadata["division"] / 6
            metadata.index = mdata.obs.index
            mdata.obs = mdata.obs.join(metadata)

        return mdata
