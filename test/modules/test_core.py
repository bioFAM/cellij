import unittest

import pyro
import mfmf
import torch

torch.manual_seed(8)


class core_TestClass(unittest.TestCase):
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
        self.fm.add_view("mrna", self.cll_data["mrna"])

    def test_can_create_FactorModel_object(self):

        fm = mfmf.core.FactorModel(
            n_factors=2, optimizer=self.optimizer, loss=self.loss
        )

    def test_error_when_optimizer_in_FactorModel_is_not_of_type_PyroOptim(self):

        with self.assertRaises(TypeError):
            fm = mfmf.core.FactorModel(n_factors=2, optimizer="mfmf", loss=self.loss)

    def test_no_error_when_loss_in_FactorModel_is_of_type_PyroElbo(self):

        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=pyro.infer.Trace_ELBO(),
        )

    def test_no_error_when_loss_in_FactorModel_is_of_type_mfmfLoss(self):

        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=mfmf.loss.Loss(
                loss_fn=pyro.infer.Trace_ELBO(),
                epochs=self.n_epochs,
            ),
        )

    def test_no_error_when_loss_in_FactorModel_is_of_type_mfmfEarlyStoppingLoss(self):

        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=mfmf.loss.EarlyStoppingLoss(
                loss_fn=pyro.infer.Trace_ELBO(),
                epochs=10,
                report_after_n_epochs=2,
                min_decrease=0.01,
                max_flat_intervals=3,
            ),
        )

    def test_error_when_loss_in_FactorModel_is_not_of_type_PyroElbo_or_mfmf_Loss(self):

        with self.assertRaises(TypeError):
            fm = mfmf.core.FactorModel(
                n_factors=2, optimizer=self.optimizer, loss="mfmf"
            )

    def test_error_when_dtype_in_FactorModel_is_not_of_type_torch_dtype(self):

        with self.assertRaises(TypeError):
            fm = mfmf.core.FactorModel(
                n_factors=2,
                optimizer=self.optimizer,
                loss=self.loss,
                dtype="mfmf",
            )

    def test_error_when_device_in_FactorModel_is_not_cpu_or_cuda(self):

        with self.assertRaises(ValueError):
            fm = mfmf.core.FactorModel(
                n_factors=2,
                optimizer=self.optimizer,
                loss=self.loss,
                device="mfmf",
            )

    def test_error_when_n_factors_in_FactorModel_is_not_positive_integer(self):

        with self.assertRaises(NotImplementedError):
            fm = mfmf.core.FactorModel(
                n_factors="auto",
                optimizer=self.optimizer,
                loss=self.loss,
                device="cpu",
            )

        with self.assertRaises(ValueError):
            fm = mfmf.core.FactorModel(
                n_factors=0,
                optimizer=self.optimizer,
                loss=self.loss,
                device="cpu",
            )

        with self.assertRaises(ValueError):
            fm = mfmf.core.FactorModel(
                n_factors=-1,
                optimizer=self.optimizer,
                loss=self.loss,
                device="cpu",
            )

        with self.assertRaises(TypeError):
            fm = mfmf.core.FactorModel(
                n_factors="mfmf",
                optimizer=self.optimizer,
                loss=self.loss,
                device="cpu",
            )

    def test_pyroELBO_is_wrapped_in_mfmfLoss_when_passed_to_FactorModel(self):

        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=pyro.infer.Trace_ELBO(),
        )

        assert isinstance(fm.loss, mfmf.modules.loss.Loss)

    def test_can_use_horseshoe_prior_for_feature_reg_in_FactorModel(self):
        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=self.loss,
        )
        fm.add_view("mrna", self.cll_data["mrna"])
        fm.fit(feature_reg="horseshoe")

    def test_can_use_finnish_horseshoe_prior_for_feature_reg_in_FactorModel(self):
        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=self.loss,
        )
        fm.add_view("mrna", self.cll_data["mrna"])
        fm.fit(feature_reg="finnish_horseshoe")

    def test_can_use_ard_prior_for_feature_reg_in_FactorModel(self):
        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=self.loss,
        )
        fm.add_view("mrna", self.cll_data["mrna"])
        fm.fit(feature_reg="ard")

    def test_can_use_spike_and_slab_prior_for_feature_reg_in_FactorModel(self):
        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=self.loss,
        )
        fm.add_view("mrna", self.cll_data["mrna"])
        fm.fit(feature_reg="spike-and-slab")

    def test_can_use_ard_spike_and_slab_prior_for_feature_reg_in_FactorModel(self):
        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=self.loss,
        )
        fm.add_view("mrna", self.cll_data["mrna"])
        fm.fit(feature_reg="ard_spike-and-slab")

    def test_can_use_horseshoe_prior_for_sample_reg_in_FactorModel(self):
        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=self.loss,
        )
        fm.add_view("mrna", self.cll_data["mrna"])
        fm.fit(sample_reg="horseshoe")

    def test_can_use_finnish_horseshoe_prior_for_sample_reg_in_FactorModel(self):
        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=self.loss,
        )
        fm.add_view("mrna", self.cll_data["mrna"])
        fm.fit(sample_reg="finnish_horseshoe")

    def test_can_use_ard_prior_for_sample_reg_in_FactorModel(self):
        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=self.loss,
        )
        fm.add_view("mrna", self.cll_data["mrna"])
        fm.fit(sample_reg="ard")

    def test_can_use_spike_and_slab_prior_for_sample_reg_in_FactorModel(self):
        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=self.loss,
        )
        fm.add_view("mrna", self.cll_data["mrna"])
        fm.fit(sample_reg="spike-and-slab")

    def test_can_use_ard_spike_and_slab_prior_for_sample_reg_in_FactorModel(self):
        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=self.loss,
        )
        fm.add_view("mrna", self.cll_data["mrna"])
        fm.fit(sample_reg="ard_spike-and-slab")

    def test_can_use_AutoNormal_guide_in_FactorModel(self):
        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=self.loss,
            guide="AutoNormal",
        )
        fm.add_view("mrna", self.cll_data["mrna"])
        fm.fit()

    def test_can_use_AutoDelta_guide_in_FactorModel(self):
        fm = mfmf.core.FactorModel(
            n_factors=2,
            optimizer=self.optimizer,
            loss=self.loss,
            guide="AutoDelta",
        )
        fm.add_view("mrna", self.cll_data["mrna"])
        fm.fit()

    def test_error_when_data_in_add_views_is_not_MuData(self):

        my_tensor = torch.rand(1, 2)

        with self.assertRaises(TypeError):
            self.fm.add_views(my_tensor)

    def test_error_when_view_in_get_merged_view_doesnt_exist(self):

        with self.assertRaises(ValueError):
            self.fm.get_merged_view("mfmf")

    def test_error_when_store_params_in_FactorModel_fit_is_not_boolean(self):

        with self.assertRaises(TypeError):
            self.fm.fit(store_params=1)
