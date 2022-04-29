from aca import adaptive_cross_approximation
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.kernels import GridInterpolationKernel, ProductStructureKernel
from gpytorch.means import ConstantMean
from typing import Optional
import gpytorch
import torch

# Model 1: random inducing points that are learned (OK)
# Model 2: aca inducing points that are learned (OK)
# Model 3: aca inducing points fixed
# Model 4: aca inducing points updated once in a while
# Model 5: structured kernel interpolation


class SOR_RandomInducingPoints(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood,
                 m: int):
        """
        Subset of Regressors approach for the approximation
        of the covariance kernel in a Gaussian Process.
        Initially, the set of m inducing points is randomly chosen.
        The inducing points are learnable parameters of the model.

        :param train_x: training inputs (n, D)
        :param train_y: training outputs (n,)
        :param m: number of inducing points
        :param likelihood: likelihood function
        """
        super(SOR_RandomInducingPoints, self).__init__(
            train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        # Sample of m inducing points
        n = train_x.shape[0]
        permutation = torch.randperm(n)
        inducing_points = train_x[permutation[:m]].detach().clone()

        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(
            self.base_covar_module,
            inducing_points=inducing_points,
            likelihood=likelihood
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SOR_AdaptiveCrossApproximation(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood,
                 m: Optional[int] = None, eps: float = 5e-1,):
        """
        Subset of Regressors approach for the approximation
        of the covariance kernel in a Gaussian Process.
        Initially, the set of m inducing points is chosen according
        to Adapative Cross Approximation for the SPSD matrix.

        :param train_x: training inputs (n, D)
        :param train_y: training outputs (n,)
        :param likelihood: likelihood function
        :param m: maximum number of inducing points
        :param eps: threshold on the residual trace within ACA
        """
        super(SOR_AdaptiveCrossApproximation, self).__init__(
            train_x, train_y, likelihood)

        # Default mean and covariance
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())

        # Eventually select all points
        if m is None:
            m = train_x.shape[0]

        # Precompute covariance matrix
        K = self.base_covar_module(train_x, train_x)

        # Identify inducing points
        if train_x.is_cuda:
            self.base_covar_module = self.base_covar_module.cuda()
            inducing_points = adaptive_cross_approximation(
                K, max_iter=m,
            )
        else:
            inducing_points = adaptive_cross_approximation(
                K.detach().cpu(), max_iter=m,
            )

        # Select inducing points
        inducing_points = train_x[inducing_points, :].detach().clone()
        print(f"Inducing points: {inducing_points.shape}")

        self.covar_module = InducingPointKernel(
            self.base_covar_module,
            inducing_points=inducing_points,
            likelihood=likelihood
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class KISS(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood,
                 m: Optional[int] = None,):
        """
        Subset of Regressors approach for the approximation
        of the covariance kernel in a Gaussian Process.
        Initially, the set of m inducing points is chosen according
        to Adapative Cross Approximation for the SPSD matrix.

        :param train_x: training inputs (n, D)
        :param train_y: training outputs (n,)
        :param likelihood: likelihood function
        :param m: maximum number of inducing points
        """
        super(KISS, self).__init__(
            train_x, train_y, likelihood)

        # Default mean and covariance
        self.mean_module = ConstantMean()

        # Size of the training set
        n, d = train_x.shape
        print(f"Training set size: {n}")
        print(f"Dimension: {d}")

        # Eventually select all points
        if m is None:
            m = int(gpytorch.utils.grid.choose_grid_size(train_x))

        print(f"Grid Size: {m}")

        self.covar_module = ScaleKernel(
            GridInterpolationKernel(
                RBFKernel(), grid_size=m, num_dims=d
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SKIP(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood,
                 m: Optional[int] = None,):
        """
        Subset of Regressors approach for the approximation
        of the covariance kernel in a Gaussian Process.
        Initially, the set of m inducing points is chosen according
        to Adapative Cross Approximation for the SPSD matrix.

        :param train_x: training inputs (n, D)
        :param train_y: training outputs (n,)
        :param likelihood: likelihood function
        :param m: maximum number of inducing points
        """
        super(SKIP, self).__init__(
            train_x, train_y, likelihood)

        # Default mean and covariance
        self.mean_module = ConstantMean()

        # Size of the training set
        n, d = train_x.shape
        print(f"Training set size: {n}")
        print(f"Dimension: {d}")

        # Eventually select all points
        if m is None:
            m = n

        print(f"Grid Size: {m}")

        self.covar_module = ProductStructureKernel(
            ScaleKernel(GridInterpolationKernel(
                RBFKernel(), grid_size=m, num_dims=1
            )), num_dims=d)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
