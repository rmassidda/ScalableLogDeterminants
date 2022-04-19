from math import floor
from models import SOR_AdaptiveCrossApproximation
from models import SOR_RandomInducingPoints
from models import KISS
from models import SKIP
from scipy.io import loadmat
import gpytorch
import torch

# Read input data
data = torch.Tensor(loadmat('data/elevators.mat')['data'])
X = data[:, :-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:, -1]

# Training set
train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

# Test set
test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

likelihood = gpytorch.likelihoods.GaussianLikelihood()

# model = SOR_RandomInducingPoints(
#     train_x, train_y, m=10, likelihood=likelihood)
# model = SOR_AdaptiveCrossApproximation(
#     train_x, train_y, likelihood=likelihood,
#     eps=1000, m=20)
# model = KISS(
#     train_x, train_y, likelihood=likelihood, m=4)
model = SKIP(
    train_x, train_y, likelihood=likelihood, m=400)

# Number of epochs
training_iterations = 100

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


def train():
    for epoch in range(training_iterations):
        # Init optimization step
        optimizer.zero_grad()

        # Posterior distribution
        output = model(train_x)

        # Extract the negative marginal likelihood
        pos_mll = mll(output, train_y)
        assert isinstance(pos_mll, torch.Tensor)  # Make the typechecker happy
        neg_mll = - pos_mll

        # Optimization step
        neg_mll.backward()
        optimizer.step()

        # Model in evaluation mode
        model.eval()
        likelihood.eval()

        # TODO: Lengthscale
        lengthscale = 0.

        # TODO: Noise
        noise = 0.

        # MSE
        mse_train = gpytorch.metrics.mean_squared_error(output, train_y)
        mse_test = gpytorch.metrics.mean_squared_error(model(test_x), test_y)

        # Verbose
        print(
            f"Epoch {epoch:03d} | "
            f"-mll: {neg_mll.item():.3f} | "
            f"MSE train: {mse_train.item():.3f} | "
            f"MSE test: {mse_test.item():.3f} | "
            f"Lengthscale: {lengthscale:.3f} | "
            f"Noise: {noise:.3f}"
        )

        # Reset training mode
        model.train()
        likelihood.train()


train()
