import unittest

import pyro
import mfmf
import torch
import numpy as np
import mofax as mfx
from pandas.testing import assert_frame_equal
import itertools


torch.manual_seed(8)


class mofax_support_TestClass(unittest.TestCase):
    def setUp(self):
        self.optimizer = pyro.optim.Adam({"lr": 0.0005, "betas": (0.90, 0.999)})
        self.n_epochs = 2
        imp = mfmf.importer.Importer()
        self.cll_data = imp.load_CLL()
        self.loss = mfmf.loss.Loss(
            loss_fn=pyro.infer.Trace_ELBO(),
            epochs=self.n_epochs,
        )
        self.fm = mfmf.core.FactorModel(
            n_factors=2, optimizer=self.optimizer, loss=self.loss
        )
        self.fm.add_view("mutations", self.cll_data["mutations"])
        self.fm.metadata = self.cll_data.obs
        self.fm.fit()
        # Can't use the same model for all tests because they might run in parallel?
        # self.fm.save_to_hdf5("./model.hdf5")
        # self.mfx_model = mfx.mofa_model("./model.hdf5")

    def test_get_shape_produces_same_output(self):

        self.fm.save_to_hdf5("./model_test_get_shape_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model(
            "./model_test_get_shape_produces_same_output.hdf5"
        )

        mfx_output = self.mfx_model.get_shape()
        mfmf_output = self.fm.get_shape()

        assert mfx_output == mfmf_output

    def test_get_samples_produces_same_output(self):

        self.fm.save_to_hdf5("./model_get_samples_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model("./model_get_samples_produces_same_output.hdf5")

        mfx_output = (
            self.mfx_model.get_samples().sort_values("sample").reset_index(drop=True)
        )
        mfmf_output = self.fm.get_samples().sort_values("sample").reset_index(drop=True)

        assert_frame_equal(mfx_output, mfmf_output)

    def test_get_cells_produces_same_output(self):

        self.fm.save_to_hdf5("./model_get_cells_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model("./model_get_cells_produces_same_output.hdf5")

        mfx_output = (
            self.mfx_model.get_cells().sort_values("cell").reset_index(drop=True)
        )
        mfmf_output = self.fm.get_cells().sort_values("cell").reset_index(drop=True)

        assert_frame_equal(mfx_output, mfmf_output)

    def test_get_features_produces_same_output(self):

        self.fm.save_to_hdf5("./model_get_features_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model(
            "./model_get_features_produces_same_output.hdf5"
        )

        mfx_output = (
            self.mfx_model.get_features().sort_values("feature").reset_index(drop=True)
        )
        mfmf_output = (
            self.fm.get_features().sort_values("feature").reset_index(drop=True)
        )

        assert_frame_equal(mfx_output, mfmf_output)

    def test_get_groups_produces_same_output(self):

        self.fm.save_to_hdf5("./model_get_groups_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model("./model_get_groups_produces_same_output.hdf5")

        mfx_output = sorted(self.mfx_model.get_groups())
        mfmf_output = sorted(self.fm.get_groups())

        assert mfx_output == mfmf_output

    def test_get_views_produces_same_output(self):

        self.fm.save_to_hdf5("./model_get_views_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model("./model_get_views_produces_same_output.hdf5")

        mfx_output = sorted(self.mfx_model.get_views())
        mfmf_output = sorted(self.fm.get_views())

        assert mfx_output == mfmf_output

    def test_get_top_features_produces_same_output(self):

        self.fm.save_to_hdf5("./model_get_top_features_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model(
            "./model_get_top_features_produces_same_output.hdf5"
        )

        mfx_output = sorted(self.mfx_model.get_top_features())
        mfmf_output = sorted(self.fm.get_top_features())

        assert mfx_output == mfmf_output

    def test_get_factors_produces_same_output(self):

        self.fm.save_to_hdf5("./model_get_factors_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model("./model_get_factors_produces_same_output.hdf5")

        mfx_output = sorted(
            [x for x in itertools.chain.from_iterable(self.mfx_model.get_factors())]
        )
        mfmf_output = sorted(
            [x for x in itertools.chain.from_iterable(self.fm.get_factors())]
        )

        assert mfx_output == mfmf_output

    def test_get_weights_produces_same_output(self):

        self.fm.save_to_hdf5("./model_get_weights_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model("./model_get_weights_produces_same_output.hdf5")

        mfx_output = sorted(itertools.chain.from_iterable(self.mfx_model.get_weights()))
        mfmf_output = sorted(itertools.chain.from_iterable(self.fm.get_weights()))

        assert mfx_output == mfmf_output

    # def test_run_umap_produces_same_output(self):

    #     fm = mfmf.core.FactorModel(
    #         n_factors=2, optimizer=self.optimizer, loss=self.loss
    #     )
    #     cll_data = self.cll_data  # modify local copy
    #     cll_data["mrna"].obs = cll_data[
    #         cll_data.obs.index.isin(cll_data["mrna"].obs.index)
    #     ].obs
    #     fm.add_view("mrna", cll_data["mrna"])
    #     fm.add_covariates(
    #         level="obs",
    #         flavour="unordered",
    #         covariates={
    #             "female": fm.metadata[fm.metadata.Gender == "f"].index.tolist(),
    #             "male": fm.metadata[fm.metadata.Gender == "m"].index.tolist(),
    #         },
    #     )
    #     fm.fit()

    #     fm.save_to_hdf5("./model_run_umap_produces_same_output.hdf5")
    #     mfx_model = mfx.mofa_model("./model_run_umap_produces_same_output.hdf5")

    #     fm.run_umap()
    #     mfx_model.run_umap()

    #     fm.metadata.sort_index(axis=0, inplace=True)
    #     fm.metadata.sort_index(axis=1, inplace=True)
    #     mfx_model.metadata.sort_index(axis=0, inplace=True)
    #     mfx_model.metadata.sort_index(axis=1, inplace=True)

    #     colwise_comparison = []
    #     for col in fm.metadata.columns:
    #         colwise_comparison.append(
    #             all(
    #                 fm.metadata[col].astype("str")
    #                 == mfx_model.metadata[col].astype("str")
    #             )
    #         )

    #     assert all(colwise_comparison)

    def test_check_views_produces_same_output(self):

        self.fm.save_to_hdf5("./model_check_views_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model("./model_check_views_produces_same_output.hdf5")

        mfx_output = sorted(self.mfx_model._check_views(views=None))
        mfmf_output = sorted(self.fm._check_views(views=None))

        assert mfx_output == mfmf_output

    def test_check_groups_produces_same_output(self):

        self.fm.save_to_hdf5("./model_check_groups_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model(
            "./model_check_groups_produces_same_output.hdf5"
        )

        mfx_output = sorted(self.mfx_model._check_groups(groups=None))
        mfmf_output = sorted(self.fm._check_groups(groups=None))

        assert mfx_output == mfmf_output

    def test_check_factors_produces_same_output(self):

        self.fm.save_to_hdf5("./model_check_factors_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model(
            "./model_check_factors_produces_same_output.hdf5"
        )

        mfx_output = [
            str(x)
            for x in itertools.chain.from_iterable(
                self.mfx_model._check_factors(factors=None)
            )
        ]

        mfmf_output = [
            str(x)
            for x in itertools.chain.from_iterable(self.fm._check_factors(factors=None))
        ]

        assert sorted(mfx_output) == sorted(mfmf_output)

    # def test_calculate_variance_explained_produces_same_output(self):

    #     self.fm.save_to_hdf5(
    #         "./model_calculate_variance_explained_produces_same_output.hdf5"
    #     )
    #     self.mfx_model = mfx.mofa_model(
    #         "./model_calculate_variance_explained_produces_same_output.hdf5"
    #     )

    #     mfmf_output = self.fm.calculate_variance_explained()
    #     mfx_output = self.mfx_model.calculate_variance_explained()

    #     mfmf_output.sort_values("R2", inplace=True)
    #     mfmf_output.sort_index(axis=1, inplace=True)
    #     mfx_output.sort_values("R2", inplace=True)
    #     mfx_output.sort_index(axis=1, inplace=True)

    #     assert_frame_equal(mfmf_output, mfx_output)

    def test_get_variance_explained_produces_same_output(self):

        self.fm.save_to_hdf5("./model_get_variance_explained_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model(
            "./model_get_variance_explained_produces_same_output.hdf5"
        )

        mfmf_output = self.fm.get_variance_explained()
        mfx_output = self.mfx_model.get_variance_explained()

        mfmf_output.sort_values("R2", inplace=True)
        mfmf_output.sort_index(axis=1, inplace=True)
        mfmf_output.reset_index(inplace=True, drop=True)
        mfx_output.sort_values("R2", inplace=True)
        mfx_output.sort_index(axis=1, inplace=True)
        mfx_output.reset_index(inplace=True, drop=True)

        assert_frame_equal(mfmf_output, mfx_output)

    def test_get_sample_r2_produces_same_output(self):

        self.fm.save_to_hdf5("./model_get_sample_r2_produces_same_output.hdf5")
        self.mfx_model = mfx.mofa_model(
            "./model_get_sample_r2_produces_same_output.hdf5"
        )

        mfx_output = sorted(self.mfx_model.get_sample_r2())
        mfmf_output = sorted(self.fm.get_sample_r2())

        assert mfx_output == mfmf_output
