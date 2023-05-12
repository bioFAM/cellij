import torch
import gpytorch
import pandas as pd


class PseudotimeGP(gpytorch.models.PyroGP):
    def __init__(
        self,
        inducing_points: pd.DataFrame,
        n_factors: int,
        smoothness: float = 1.5,
        scale_param: bool = False,
    ) -> None:
        """Spatial factors for a single cell using GPs with Matérn kernel

        Args:
            inducing_points: DataFrame with columns 'x', 'y', initial inducing
                point locations
            n_factors: number of spatial factors
            smoothness: smoothness parameter of kernel, can be 0.5, 1.5 or 2.5
            scale_param: if True, use kernel scale parameter
        """
        n_inducing = len(inducing_points)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=n_inducing,
            batch_shape=torch.Size([n_factors]),
        )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            model=self,
            inducing_points=torch.tensor(data=inducing_points.values, dtype=torch.float32),
            variational_distribution=variational_distribution,
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_factors]))
        super(PseudotimeGP, self).__init__(
            variational_strategy=variational_strategy,
            likelihood=likelihood,
            num_data=len(inducing_points),
            name_prefix="PseudotimeGP",
        )
        self.likelihood = likelihood

        # Mean, covar
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )

    def forward(self, x):
        mean = self.mean_module(x)  # Returns an n_data vec
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)