from __future__ import annotations
import unittest

import mfmf
import torch
import numpy
import anndata
import muon


class View_TestClass(unittest.TestCase):
    def setUp(self):

        self.my_numpy = numpy.random.normal(size=[5, 10])
        self.my_tensor = torch.normal(mean=0, std=1, size=[3, 20])
        self.my_anndata = anndata.AnnData(self.my_numpy)

        imp = mfmf.importer.Importer()
        self.cll_data = imp.load_CLL()

        assert isinstance(self.cll_data, muon.MuData)

    def test_can_create_View_from_numpy(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        assert isinstance(my_view, mfmf.modules.data.View)

    def test_can_create_View_from_tensor(self):

        my_view = mfmf.modules.data.View(name="tensor", data=self.my_tensor)

        assert isinstance(my_view, mfmf.modules.data.View)

    def test_can_create_View_from_anndata(self):

        my_view = mfmf.modules.data.View(name="anndata", data=self.my_anndata)

        assert isinstance(my_view, mfmf.modules.data.View)

    def test_cannot_create_View_from_mudata(self):

        with self.assertRaises(TypeError) as cm:

            my_view = mfmf.modules.data.View(name="mudata", data=self.cll_data)
            self.assertEqual(
                "Please use function 'FactorModel.add_merged_view()' for MuData objects.",
                str(cm.exception),
            )

    def test_error_when_data_parameter_of_View_is_invalid(self):

        with self.assertRaises(TypeError) as cm:

            my_view = mfmf.modules.data.View(name="invalid", data=1)
            self.assertEqual(
                "Parameter 'data' must be a numpy array, a torch tensor, or an anndata AnnData object.",
                str(cm.exception),
            )

    def test_error_when_name_parameter_of_View_is_not_str(self):

        with self.assertRaises(TypeError) as cm:

            my_view = mfmf.modules.data.View(
                name=1, data=torch.normal(mean=0, std=1, size=[3, 20])
            )
            self.assertEqual(
                "Parameter 'name' must be a string.",
                str(cm.exception),
            )

    def test_error_when_data_parameter_has_more_than_two_dimensions(self):

        with self.assertRaises(ValueError) as cm:

            my_numpy_view = mfmf.modules.data.View(
                name="invalid", data=numpy.random.normal(size=[5, 10, 3])
            )
            self.assertEqual(
                "Parameter 'data' must be a 2D numpy array or a torch tensor.",
                str(cm.exception),
            )

        with self.assertRaises(ValueError) as cm:

            my_tensor_view = mfmf.modules.data.View(
                name="invalid", data=torch.normal(mean=0, std=1, size=[3, 20, 5])
            )
            self.assertEqual(
                "Parameter 'data' must be a 2D numpy array or a torch tensor.",
                str(cm.exception),
            )

    def test_can_get_n_obs_from_View(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        assert my_view.n_obs == 5

    def test_can_get_n_features_from_View(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        assert my_view.n_features == 10

    def test_can_get_obs_names_from_View(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        assert my_view.obs_names == ["obs" + str(i) for i in range(5)]

    def test_can_get_feature_names_from_View(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        assert my_view.feature_names == ["feature" + str(i) for i in range(10)]

    def test_error_when_trying_to_set_n_obs_of_View(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        with self.assertRaises(AttributeError) as cm:

            my_view.n_obs = 10
            self.assertEqual(
                "Parameter 'n_obs' is derived from data and can't be set.",
                str(cm.exception),
            )

    def test_error_when_trying_to_set_n_features_of_View(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        with self.assertRaises(AttributeError) as cm:

            my_view.n_features = 10
            self.assertEqual(
                "Parameter 'n_features' is derived from data and can't be set.",
                str(cm.exception),
            )

    def test_no_error_when_assigning_list_of_strings_to_obs_names_of_View(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        my_view.obs_names = ["obs_" + str(i) for i in range(5)]

        assert my_view.obs_names == ["obs_" + str(i) for i in range(5)]

    def test_error_when_obs_names_is_not_a_list(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        with self.assertRaises(TypeError) as cm:

            my_view.obs_names = ("obs_" + str(i) for i in range(5))
            self.assertEqual(
                "Parameter 'obs_names' must be a list.",
                str(cm.exception),
            )

    def test_error_when_obs_names_is_not_a_list_of_strings(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        with self.assertRaises(TypeError) as cm:

            my_view.obs_names = [1, 2, 3]
            self.assertEqual(
                "All elements of 'obs_names' must be strings.",
                str(cm.exception),
            )

    def test_error_when_feature_names_is_not_a_list(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        with self.assertRaises(TypeError) as cm:

            my_view.feature_names = ("feature_" + str(i) for i in range(10))
            self.assertEqual(
                "Parameter 'feature_names' must be a list.",
                str(cm.exception),
            )

    def test_error_when_feature_names_is_not_a_list_of_strings(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        with self.assertRaises(TypeError) as cm:

            my_view.feature_names = [1, 2, 3]
            self.assertEqual(
                "All elements of 'feature_names' must be strings.",
                str(cm.exception),
            )

    def test_no_error_when_assigning_list_of_strings_to_feature_names_of_View(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)
        my_view.feature_names = ["feature_" + str(i) for i in range(10)]

        assert my_view.feature_names == ["feature_" + str(i) for i in range(10)]

    def test_error_when_the_length_of_obs_names_doesnt_match_n_obs_of_View(self):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        with self.assertRaises(ValueError) as cm:

            my_view.obs_names = ["obs_" + str(i) for i in range(6)]
            self.assertEqual(
                "The length of 'obs_names' must match the number of observations.",
                str(cm.exception),
            )

    def test_error_when_the_length_of_feature_names_doesnt_match_n_features_of_View(
        self,
    ):

        my_view = mfmf.modules.data.View(name="numpy", data=self.my_numpy)

        with self.assertRaises(ValueError) as cm:

            my_view.feature_names = ["feature_" + str(i) for i in range(11)]
            self.assertEqual(
                "The length of 'feature_names' must match the number of features.",
                str(cm.exception),
            )


class MergedView_TestClass(unittest.TestCase):
    def setUp(self):

        self.my_numpy = numpy.random.normal(size=[5, 10])
        self.my_tensor = torch.normal(mean=0, std=1, size=[3, 20])
        self.my_anndata = anndata.AnnData(self.my_numpy)

        self.feature_offsets = {"a": (0, 3), "b": (3, 5)}

        imp = mfmf.importer.Importer()
        self.cll_data = imp.load_CLL()

        assert isinstance(self.cll_data, muon.MuData)

    def test_can_create_MergedView_from_anndata(self):

        my_merged_view = mfmf.modules.data.MergedView(
            data=self.my_anndata, feature_offsets=self.feature_offsets
        )

        assert isinstance(my_merged_view, mfmf.modules.data.MergedView)

    def test_error_when_data_parameter_of_MergedView_is_not_anndata(self):

        with self.assertRaises(TypeError) as cm:

            mfmf.modules.data.MergedView(
                data=torch.normal(mean=0, std=1, size=[3, 20]),
                feature_offsets=self.feature_offsets,
            )
            self.assertEqual(
                "Parameter 'data' must be an anndata AnnData object.",
                str(cm.exception),
            )

    def test_no_error_when_feature_offsets_is_a_dict(self):

        my_numpy = numpy.random.normal(size=[5, 10])
        my_anndata = anndata.AnnData(my_numpy)
        my_merged_view = mfmf.modules.data.MergedView(
            data=my_anndata, feature_offsets=self.feature_offsets
        )

        assert isinstance(my_merged_view, mfmf.modules.data.MergedView)

    def test_error_when_feature_offsets_is_not_a_dict(self):

        my_numpy = numpy.random.normal(size=[5, 10])
        my_anndata = anndata.AnnData(my_numpy)
        with self.assertRaises(TypeError) as cm:

            mfmf.modules.data.MergedView(
                data=my_anndata,
                feature_offsets=1,
            )
            self.assertEqual(
                "Parameter 'feature_offsets' must be a list.",
                str(cm.exception),
            )

    def test_MergedView_can_return_a_single_view(self):

        my_merged_view = mfmf.modules.data.MergedView(
            data=self.my_anndata,
            feature_offsets=self.feature_offsets,
        )

        my_view = my_merged_view.get_view("a")

        assert isinstance(my_view, mfmf.modules.data.View)

    def test_error_when_view_name_is_not_in_view_names(self):

        my_merged_view = mfmf.modules.data.MergedView(
            data=self.my_anndata,
            feature_offsets=self.feature_offsets,
        )

        with self.assertRaises(ValueError) as cm:

            my_merged_view.get_view("c")
            self.assertEqual(
                "View with name 'c' not found.",
                str(cm.exception),
            )

    def test_error_when_get_view_would_result_in_an_empty_View(self):

        my_merged_view = mfmf.modules.data.MergedView(
            data=self.my_anndata, feature_offsets={"a": (0, 3), "b": (11, 15)}
        )

        with self.assertRaises(ValueError) as cm:

            my_merged_view.get_view("b")
            self.assertEqual(
                "View with name 'b' and feature offsets [11, 15] is empty.",
                str(cm.exception),
            )
