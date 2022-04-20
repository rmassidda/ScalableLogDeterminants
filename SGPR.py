from models import SOR_AdaptiveCrossApproximation
from models import SOR_RandomInducingPoints
from models import KISS
from models import SKIP
from utils import load_precipitations, load_audio, load_redundant_wave
from utils import load_elevators
from utils import plot_model, plot_data
import gpytorch
import torch


def eval(model, likelihood, mll, train_x, train_y, test_x, test_y):
    # Evaluation mode
    model.eval()
    likelihood.eval()

    # Plotting if 1-D
    if train_x.shape[1] == 1:
        # Eventually get inducing points
        if isinstance(model, SOR_RandomInducingPoints) or \
           isinstance(model, SOR_AdaptiveCrossApproximation):
            inducing_points = model.covar_module.inducing_points
        else:
            inducing_points = None
        plot_model(model, likelihood, train_x, train_y, test_x,
                   inducing_points)

    # Evaluation
    output = model(train_x)
    pos_mll = mll(output, train_y)
    assert isinstance(pos_mll, torch.Tensor)  # Make the typechecker happy
    neg_mll = - pos_mll
    mse_train = gpytorch.metrics.mean_squared_error(
        output, train_y)
    mse_test = gpytorch.metrics.mean_squared_error(model(test_x), test_y)
    # print(
    #     f"-mll: {neg_mll.item():.3f} | "
    #     f"MSE train: {mse_train.item():.3f} | "
    #     f"MSE test: {mse_test.item():.3f}")

    return neg_mll, mse_train, mse_test


def train(model, likelihood, mll, train_x, train_y, test_x, test_y, epochs):
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    for epoch in range(epochs):
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

        # # TODO: Lengthscale
        # lengthscale = 0.

        # # TODO: Noise
        # noise = 0.

        # # Verbose
        # print(
        #     f"Epoch {epoch:03d} | "
        #     f"-mll: {neg_mll.item():.3f} | "
        #     f"Lengthscale: {lengthscale:.3f} | "
        #     f"Noise: {noise:.3f}"
        # )


def experimental_setting(
    model_generator, train_x, train_y, test_x, test_y,
    m, training_iterations=100
):
    # Preprocessing
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y)

    print(
        f"Train shape: {train_x.shape, train_y.shape} | "
        f"Test shape: {test_x.shape, test_y.shape}"
    )

    # Init model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = model_generator(
        train_x, train_y, likelihood=likelihood, m=m)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    eval(
        model, likelihood, mll, train_x, train_y, test_x, test_y
    )

    train(
        model, likelihood, mll, train_x, train_y, test_x, test_y,
        training_iterations
    )

    return eval(
        model, likelihood, mll, train_x, train_y, test_x, test_y
    )


# Read elevators
# train_x, train_y, test_x, test_y = load_elevators()

# Read precipitations
# train_x, train_y, test_x, test_y = load_precipitations()

# Read audio
# train_x, train_y, test_x, test_y = load_audio()

# Read redundant wave
# train_x, train_y, test_x, test_y = load_redundant_wave()

# Experiment 1: Increasing m on the redundant wave dataset
data = {
    'redundant_wave': {
        'random': {
            'mll': [],
            'mse_train': [],
            'mse_test': [],
            'range': [1, 2, 4, 8, 16, 32, 64, 128]
        },
        'adaptive': {
            'mll': [],
            'mse_train': [],
            'mse_test': [],
            'range': [1, 2, 4, 8, 16, 32, 64, 128]
        },
        'KISS': {
            'mll': [],
            'mse_train': [],
            'mse_test': [],
            'range': [4, 8, 16, 32, 64, 128]
        },
    }
}

# Experiment 1a: Random inducing points
for m in data['redundant_wave']['random']['range']:
    print(f"m = {m}")
    train_x, train_y, test_x, test_y = load_redundant_wave()
    neg_mll, mse_train, mse_test = experimental_setting(
        model_generator=SOR_RandomInducingPoints, train_x=train_x,
        train_y=train_y, test_x=test_x, test_y=test_y, m=m
    )
    print(
        f"m = {m} | "
        f"-mll: {neg_mll.item():.3f} | "
        f"mse_train = {mse_train.item()} |"
        f"mse_test = {mse_test.item()}")

    data['redundant_wave']['random']['mll'].append(neg_mll.item())
    data['redundant_wave']['random']['mse_train'].append(mse_train.item())
    data['redundant_wave']['random']['mse_test'].append(mse_test.item())

# Experiment 1b: Adaptive inducing points
for m in data['redundant_wave']['adaptive']['range']:
    print(f"m = {m}")
    train_x, train_y, test_x, test_y = load_redundant_wave()
    neg_mll, mse_train, mse_test = experimental_setting(
        model_generator=SOR_AdaptiveCrossApproximation, train_x=train_x,
        train_y=train_y, test_x=test_x, test_y=test_y, m=m
    )
    print(
        f"m = {m} | "
        f"-mll: {neg_mll.item():.3f} | "
        f"mse_train = {mse_train.item()} |"
        f"mse_test = {mse_test.item()}")

    data['redundant_wave']['adaptive']['mll'].append(neg_mll.item())
    data['redundant_wave']['adaptive']['mse_train'].append(mse_train.item())
    data['redundant_wave']['adaptive']['mse_test'].append(mse_test.item())

# Experiment 1c: KISS
for m in data['redundant_wave']['KISS']['range']:
    print(f"m = {m}")
    train_x, train_y, test_x, test_y = load_redundant_wave()
    neg_mll, mse_train, mse_test = experimental_setting(
        model_generator=KISS, train_x=train_x,
        train_y=train_y, test_x=test_x, test_y=test_y, m=m
    )
    print(
        f"m = {m} | "
        f"-mll: {neg_mll.item():.3f} | "
        f"mse_train = {mse_train.item()} |"
        f"mse_test = {mse_test.item()}")

    data['redundant_wave']['KISS']['mll'].append(neg_mll.item())
    data['redundant_wave']['KISS']['mse_train'].append(mse_train.item())
    data['redundant_wave']['KISS']['mse_test'].append(mse_test.item())

# Plot the results
plot_data(data, 'redundant_wave', 'mll')
plot_data(data, 'redundant_wave', 'mse_train')
plot_data(data, 'redundant_wave', 'mse_test')
