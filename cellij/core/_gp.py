import gpytorch
import torch


class PseudotimeGP(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        n_factors: int,
        init_lengthscale=5.0,
    ) -> None:
        n_inducing = len(inducing_points)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=n_inducing,
            batch_shape=torch.Size([n_factors]),
        )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            model=self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=False,
        )

        super().__init__(variational_strategy=variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size([n_factors]),
        )
        self.kernel = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([n_factors]))
        self.covar_module = gpytorch.kernels.ScaleKernel(self.kernel)
        self.covar_module.base_kernel.lengthscale = torch.tensor(init_lengthscale)

    def forward(self, x) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
