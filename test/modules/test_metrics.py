import unittest

import pyro
import mfmf
import torch

torch.manual_seed(8)


class metrics_TestClass(unittest.TestCase):
    def setUp(self):
        optimizer = pyro.optim.Adam({"lr": 0.005, "betas": (0.90, 0.999)})
        imp = mfmf.importer.Importer()
        cll_data = imp.load_CLL()

        loss = mfmf.loss.Loss(
            loss_fn=pyro.infer.Trace_ELBO(),
            epochs=10,
        )
        self.untrained_model = mfmf.core.FactorModel(
            n_factors=2, optimizer=optimizer, loss=loss, device="cpu"
        )
        self.untrained_model.add_views(cll_data)

        self.trained_model = mfmf.core.FactorModel(
            n_factors=2, optimizer=optimizer, loss=loss, device="cpu"
        )
        self.trained_model.add_view("mrna", cll_data["mrna"])
        self.trained_model.fit()

    def test_can_plot_elbo_of_trained_model(self):

        self.trained_model.plot_elbo()
        self.trained_model.plot_elbo(window=10)
        self.trained_model.plot_elbo(method="none")
        self.trained_model.plot_elbo(method="mean")
        self.trained_model.plot_elbo(method="median")

    def test_error_when_passing_untrained_model_to_plot_elbo(self):

        with self.assertRaises(TypeError):
            self.untrained_model.plot_elbo()

    def test_error_when_passing_invalid_window_size_to_plot_elbo(self):

        with self.assertRaises(ValueError):
            self.trained_model.plot_elbo(window=0)

        with self.assertRaises(ValueError):
            self.trained_model.plot_elbo(window=-1)

        with self.assertRaises(TypeError):
            self.trained_model.plot_elbo(window="mfmf")

    def test_error_when_passing_invalid_method_to_plot_elbo(self):

        with self.assertRaises(TypeError):
            self.trained_model.plot_elbo(method=0)

        with self.assertRaises(ValueError):
            self.trained_model.plot_elbo(method="mfmf")

    def test_can_render_model(self):

        self.trained_model.render_model(render_distributions=True, render_params=True)

    def test_error_when_passing_untrained_model_to_render_model(self):

        with self.assertRaises(TypeError):
            self.untrained_model.render_model()

    def test_error_when_render_params_in_render_model_is_not_bool(self):

        with self.assertRaises(TypeError):
            self.trained_model.render_model(render_params="yes")

    def test_error_when_render_distributions_in_render_model_is_not_bool(self):

        with self.assertRaises(TypeError):
            self.trained_model.render_model(render_distributions="yes")

    def test_error_when_ax_is_not_a_matplotlib_axis(self):

        with self.assertRaises(TypeError):
            self.trained_model.render_model(ax="yes")

    # def test_can_plot_norm_of_trained_model(self):

    #     self.trained_model.plot_diagnostics(metrics="norm(AutoNormal.locs.w)")
