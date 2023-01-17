import unittest

import mfmf
import mudata


class Importer_TestClass(unittest.TestCase):
    def test_can_interface_with_CLL_data(self):

        imp = mfmf.modules.importer.Importer()
        cll_data = imp.load_CLL()

        assert isinstance(cll_data, mudata._core.mudata.MuData)

    def test_CLL_data_is_as_expected(self):

        imp = mfmf.modules.importer.Importer()
        cll_data = imp.load_CLL()

        expected_keys = ["drugs", "methylation", "mrna", "mutations"]
        found_keys = list(cll_data.mod.keys())
        assert found_keys == expected_keys

        expected_obs_cols = [
            "Diagnosis",
            "Gender",
            "IGHV",
            "Age4Main",
            "T5",
            "T6",
            "treatedAfter",
            "died",
            "IC50beforeTreatment",
            "ConsClust",
        ]
        found_obs_cols = list(cll_data.obs.columns)
        assert found_obs_cols == expected_obs_cols

        assert cll_data.n_mod == 4
        assert cll_data.n_obs == 200
        assert cll_data.n_var == cll_data.n_vars == 9627
