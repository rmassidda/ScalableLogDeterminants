from math import floor
from matplotlib import pyplot as plt
from models import SOR_AdaptiveCrossApproximation
from models import SOR_RandomInducingPoints
from models import KISS
from models import SKIP
from utils import load_precipitations, load_audio, load_redundant_wave
from utils import plot_model
from scipy.io import loadmat
import gpytorch
import torch

# Read elevators
# data = torch.Tensor(loadmat('data/elevators.mat')['data'])
# X = data[:, :-1]
# y = data[:, -1]

# Read precipitations
# train_x, train_y, test_x, test_y = load_precipitations()

# Read audio
# train_x, train_y, test_x, test_y = load_audio()

# Read reduntant wave
train_x, train_y, test_x, test_y = load_redundant_wave()

# Preprocessing
train_x = torch.Tensor(train_x)
train_y = torch.Tensor(train_y)
test_x = torch.Tensor(test_x)
test_y = torch.Tensor(test_y)
# X, y = torch.Tensor(X), torch.Tensor(y)
# X = X - X.min(0)[0]
# X = 2 * (X / X.max(0)[0]) - 1

# Eventually plot data
dimensionality = train_x.shape[1]

print(
    f"Train shape: {train_x.shape, train_y.shape} | "
    f"Test shape: {test_x.shape, test_y.shape}"
)

likelihood = gpytorch.likelihoods.GaussianLikelihood()

model = SOR_RandomInducingPoints(
    train_x, train_y, m=20, likelihood=likelihood)
# model = SOR_AdaptiveCrossApproximation(
#     train_x, train_y, likelihood=likelihood, m=10)
# model = KISS(
#     train_x, train_y, likelihood=likelihood, m=4)
# model = SKIP(
#     train_x, train_y, likelihood=likelihood, m=4)

# Number of epochs
training_iterations = 100

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


def eval():
    model.eval()
    likelihood.eval()

    # Plot model before training
    if dimensionality == 1:
        # Eventually get inducing points
        if isinstance(model, SOR_RandomInducingPoints) or \
           isinstance(model, SOR_AdaptiveCrossApproximation):
            inducing_points = model.covar_module.inducing_points
        else:
            inducing_points = None
        with gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.max_root_decomposition_size(100):
            plot_model(model, likelihood, train_x, train_y, test_x,
                       inducing_points)

    with gpytorch.settings.fast_pred_var(), \
            gpytorch.settings.max_root_decomposition_size(100):
        mse_train = gpytorch.metrics.mean_squared_error(
            model(train_x), train_y)
        mse_test = gpytorch.metrics.mean_squared_error(model(test_x), test_y)
        print(
            f"MSE train: {mse_train.item():.3f} | "
            f"MSE test: {mse_test.item():.3f}")


def train():
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    for epoch in range(training_iterations):
        # Init optimization step
        optimizer.zero_grad()

        # Posterior distribution
        with gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.max_root_decomposition_size(100):
            output = model(train_x)

        # Extract the negative marginal likelihood
        pos_mll = mll(output, train_y)
        assert isinstance(pos_mll, torch.Tensor)  # Make the typechecker happy
        neg_mll = - pos_mll

        # Optimization step
        neg_mll.backward()
        optimizer.step()

        # Model in evaluation mode
        # model.eval()
        # likelihood.eval()

        # TODO: Lengthscale
        lengthscale = 0.

        # TODO: Noise
        noise = 0.

        # MSE
        # mse_train = gpytorch.metrics.mean_squared_error(output, train_y)
        # mse_test = gpytorch.metrics.mean_squared_error(model(test_x), test_y)

        # Verbose
        print(
            f"Epoch {epoch:03d} | "
            f"-mll: {neg_mll.item():.3f} | "
            # f"MSE train: {mse_train.item():.3f} | "
            # f"MSE test: {mse_test.item():.3f} | "
            f"Lengthscale: {lengthscale:.3f} | "
            f"Noise: {noise:.3f}"
        )

        # Reset training mode
        # model.train()
        # likelihood.train()


eval()
train()
eval()
