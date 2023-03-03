from __future__ import annotations

import os
import anndata
import muon as mu
import numpy as np
import pandas as pd

from importlib import resources
from collections import UserDict


class DataContainer(UserDict):
    """Container to hold all data for a FactorModel.

    Holds a list of anndata objects and provides methods to add, remove, and
    modify data. Also provides methods to merge those anndata and returns them
    as a tensor for training or as a mudata for the user.

    """

    def to_mudata(self):

        """Returns a mudata object holding all anndata added to the container."""

        pass

    def prepare_for_training(self):

        """Merges all data and converts it to a tensor that is used during training."""

        pass


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

        with resources.path("mfmf.data", "cll_metadata.csv") as res_path:

            obs = pd.read_csv(
                filepath_or_buffer=os.fspath(res_path),
                sep=",",
                index_col="Sample",
                encoding=self.encoding,
            )

        modalities = {}

        for ome in ["drugs", "methylation", "mrna", "mutations"]:

            with resources.path("mfmf.data", f"cll_{ome}.csv") as res_path:

                modalities[ome] = anndata.AnnData(
                    pd.read_csv(
                        filepath_or_buffer=os.fspath(res_path),
                        sep=",",
                        index_col=0,
                        encoding=self.encoding,
                    ).T
                )

                if use_drug_compound_names and ome == "drugs":

                    with resources.path("mfmf.data", "id_to_drug_names.csv") as compound_path:

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
