from models import SOR_AdaptiveCrossApproximation
from models import SOR_RandomInducingPoints
from models import KISS
from utils import load_redundant_wave
from utils import plot_model, plot_data
import gpytorch
import numpy as np
import time
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

    # eval(
    #     model, likelihood, mll, train_x, train_y, test_x, test_y
    # )

    train(
        model, likelihood, mll, train_x, train_y, test_x, test_y,
        training_iterations
    )

    return eval(
        model, likelihood, mll, train_x, train_y, test_x, test_y
    )


def repeat_experiment(dataset_getter, model_generator, m_range, runs=1):
    # Experimental results
    res_mll = np.zeros((runs, len(m_range)))
    res_mse_train = np.zeros((runs, len(m_range)))
    res_mse_test = np.zeros((runs, len(m_range)))
    res_time = np.zeros((runs, len(m_range)))

    for run in range(runs):
        for m_idx, m in enumerate(m_range):
            print(f"m = {m}")
            train_x, train_y, test_x, test_y = dataset_getter()
            tic = time.time()
            neg_mll, mse_train, mse_test = experimental_setting(
                model_generator=model_generator, train_x=train_x,
                train_y=train_y, test_x=test_x, test_y=test_y, m=m
            )
            toc = time.time()
            print(
                f"m = {m} | "
                f"-mll: {neg_mll.item():.3f} | "
                f"mse_train = {mse_train.item()} |"
                f"mse_test = {mse_test.item()}")

            res_mll[run, m_idx] = neg_mll.item()
            res_mse_train[run, m_idx] = mse_train.item()
            res_mse_test[run, m_idx] = mse_test.item()
            res_time[run, m_idx] = toc - tic

    # Average the results over the runs
    res_mll = np.mean(res_mll, axis=0)
    res_mse_train = np.mean(res_mse_train, axis=0)
    res_mse_test = np.mean(res_mse_test, axis=0)
    res_time = np.mean(res_time, axis=0)

    return {
        'mll': res_mll,
        'mse_train': res_mse_train,
        'mse_test': res_mse_test,
        'time': res_time,
        'range': m_range
    }


def main(n_runs: int = 5):
    # Experimental results
    data = {
        'redundant_wave': {}
    }

    # Experiment 1a: Random inducing points
    data['redundant_wave']['random'] = repeat_experiment(
        dataset_getter=load_redundant_wave,
        model_generator=SOR_RandomInducingPoints,
        m_range=[1, 2, 4, 8, 16, 32, 64, 128],
        runs=n_runs
    )

    # Experiment 1b: Adaptive inducing points
    data['redundant_wave']['adaptive'] = repeat_experiment(
        dataset_getter=load_redundant_wave,
        model_generator=SOR_AdaptiveCrossApproximation,
        m_range=[1, 2, 4, 8, 16, 32, 64, 128],
        runs=n_runs
    )

    # Experiment 1c: KISS
    data['redundant_wave']['KISS'] = repeat_experiment(
        dataset_getter=load_redundant_wave,
        model_generator=KISS,
        m_range=[4, 8, 16, 32, 64, 128],
        runs=n_runs
    )

    # Plot the results
    plot_data(data, 'redundant_wave', 'mll')
    plot_data(data, 'redundant_wave', 'mse_train')
    plot_data(data, 'redundant_wave', 'mse_test')
    plot_data(data, 'redundant_wave', 'time')


if __name__ == "__main__":
    main()
