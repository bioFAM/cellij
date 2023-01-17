import unittest

import mfmf
import numpy as np
import torch
import pandas as pd


class utils_TestClass(unittest.TestCase):
    def setUp(self):
        imp = mfmf.importer.Importer()
        self.cll_data = imp.load_CLL()

    def test_can_merge_two_Views(self):
        view_a = mfmf.modules.data.View(name="mrna", data=self.cll_data["mrna"])
        view_b = mfmf.modules.data.View(
            name="mutations", data=self.cll_data["mutations"]
        )

        merged_view = mfmf.utils.merge_views(views=[view_a, view_b])

        assert isinstance(merged_view, mfmf.modules.data.MergedView)

    def test_error_when_parameter_views_in_merge_views_is_not_a_list_of_Views(self):

        with self.assertRaises(TypeError):
            mfmf.utils.merge_views(views=["mrna", "mutations"])

        with self.assertRaises(TypeError):
            mfmf.utils.merge_views(views=1)

    def test_error_when_nan_replacement_in_merge_views_is_not_a_float(self):

        view_a = mfmf.modules.data.View(name="mrna", data=self.cll_data["mrna"])
        view_b = mfmf.modules.data.View(
            name="mutations", data=self.cll_data["mutations"]
        )

        with self.assertRaises(TypeError):
            merged_view = mfmf.utils.merge_views(
                views=[view_a, view_b], nan_replacement="0.0"
            )

    def test_error_when_duplicate_feature_names_in_Views_passed_to_merge_views(self):

        view_a = mfmf.modules.data.View(name="mrna", data=self.cll_data["mrna"])
        view_b = mfmf.modules.data.View(
            name="mutations", data=self.cll_data["mutations"]
        )
        view_c = mfmf.modules.data.View(
            name="mutations", data=self.cll_data["mutations"]
        )

        with self.assertRaises(ValueError):
            merged_view = mfmf.utils.merge_views(views=[view_a, view_b, view_c])

    def test_minmax_scale_throws_no_error_when_input_is_numpy_array(self):

        mfmf.utils.minmax_scale(data=np.random.rand(3, 3))

    def test_minmax_scale_throws_no_error_when_input_is_torch_tensor(self):

        mfmf.utils.minmax_scale(data=torch.tensor(np.random.rand(3, 3)))

    def test_minmax_scale_throws_error_when_input_is_not_numpy_array_or_torch_tensor(
        self,
    ):

        with self.assertRaises(TypeError):
            mfmf.utils.minmax_scale(data="mrna")

    def test_plot_optimal_assignment_throws_no_error_when_input_is_numpy_array(self):

        mfmf.utils.plot_optimal_assignment(
            x1=np.random.rand(3, 3), x2=np.random.rand(3, 3), assignment_dim=1
        )

    def test_plot_optimal_assignment_throws_no_error_when_input_is_torch_tensor(self):

        mfmf.utils.plot_optimal_assignment(
            x1=torch.tensor(np.random.rand(3, 3)),
            x2=np.random.rand(3, 3),
            assignment_dim=1,
        )

    def test_plot_optimal_assignment_throws_no_error_when_input_is_pandas_df(self):

        mfmf.utils.plot_optimal_assignment(
            x1=pd.DataFrame(np.random.rand(3, 3)),
            x2=np.random.rand(3, 3),
            assignment_dim=1,
        )

    def test_plot_optimal_throws_an_error_if_input_data_is_higher_than_2D(self):

        with self.assertRaises(ValueError):
            mfmf.utils.plot_optimal_assignment(
                x1=np.random.rand(3, 3, 3), x2=np.random.rand(3, 3), assignment_dim=1
            )
