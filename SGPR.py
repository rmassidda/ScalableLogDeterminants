from models import KISS
# from models import SKIP
from models import SOR_AdaptiveCrossApproximation
from models import SOR_RandomInducingPoints
from utils import load_precipitations
from utils import load_redundant_wave
from utils import plot_model, plot_data
import gpytorch
import numpy as np
import pickle
import time
import torch


# Set cuda device
USE_CUDA = True
if USE_CUDA:
    # Set device
    torch.cuda.set_device(0)
    # Empty cache
    torch.cuda.empty_cache()


def plot_init(
    model_generator, train_x, train_y, test_x, test_y, m
):
    # Preprocessing
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y)

    if USE_CUDA:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()

    print(
        f"Train shape: {train_x.shape, train_y.shape} | "
        f"Test shape: {test_x.shape, test_y.shape}"
    )

    # Init model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = model_generator(
        train_x, train_y, likelihood=likelihood, m=m)
    model.base_covar_module.base_kernel.lengthscale = 0.5
    if USE_CUDA:
        likelihood = likelihood.cuda()
        model = model.cuda()
    model.eval()
    likelihood.eval()
    # Plot model with the inducing points
    plot_model(model, likelihood, train_x, train_y, test_x,
               model.covar_module.inducing_points)


def eval(model, likelihood, mll, train_x, train_y, test_x, test_y):
    # Evaluation mode
    model.eval()
    likelihood.eval()

    # Handle extremely ill-conditioned matrices
    with gpytorch.settings.cholesky_jitter(1e-1):
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

    # Empty cuda cache
    if USE_CUDA:
        torch.cuda.empty_cache()

    return neg_mll.item(), mse_train.item(), mse_test.item()


def train(model, likelihood, mll, train_x, train_y, test_x, test_y, epochs):
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    for epoch in range(epochs):
        # Init optimization step
        optimizer.zero_grad()

        # Handle extremely ill-conditioned matrices
        with gpytorch.settings.cholesky_jitter(1e-1):
            # Posterior distribution
            output = model(train_x)

            # Extract the negative marginal likelihood
            pos_mll = mll(output, train_y)
        assert isinstance(pos_mll, torch.Tensor)  # Make the typechecker happy
        neg_mll = - pos_mll

        # Optimization step
        neg_mll.backward()
        optimizer.step()

    # Empty cuda cache
    if USE_CUDA:
        torch.cuda.empty_cache()


def experimental_setting(
    model_generator, train_x, train_y, test_x, test_y,
    m, training_iterations=100
):
    # Preprocessing
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y)
    if USE_CUDA:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()

    print(
        f"Train shape: {train_x.shape, train_y.shape} | "
        f"Test shape: {test_x.shape, test_y.shape}"
    )

    # Init model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = model_generator(
        train_x, train_y, likelihood=likelihood, m=m)

    if USE_CUDA:
        likelihood = likelihood.cuda()
        model = model.cuda()

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    train(
        model, likelihood, mll, train_x, train_y, test_x, test_y,
        training_iterations
    )

    return eval(
        model, likelihood, mll, train_x, train_y, test_x, test_y
    )


def repeat_experiment(dataset_getter, model_generator, m_range, runs=1,
                      training_iterations=100):
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
                train_y=train_y, test_x=test_x, test_y=test_y, m=m,
                training_iterations=training_iterations
            )
            toc = time.time()
            print(
                f"m = {m} | "
                f"-mll: {neg_mll:.3f} | "
                f"mse_train = {mse_train} |"
                f"mse_test = {mse_test}")

            res_mll[run, m_idx] = neg_mll
            res_mse_train[run, m_idx] = mse_train
            res_mse_test[run, m_idx] = mse_test
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
        'redundant_wave': {},
        'precipitations': {},
        'prec_init': {}
    }

    # Range
    range_inducing = [1, 2, 4, 8, 16, 32, 64, 128]
    small_range_inducing = [1, 2, 4, 8, 16, 32, 64]
    range_kiss = [4, 8, 16, 32, 64, 128]
    small_range_kiss = [4, 8, 16, 32, 48, 64]

    # Experiment 1: Redundant Waves
    # Experiment 1a: Random inducing points
    data['redundant_wave']['random'] = repeat_experiment(
        dataset_getter=load_redundant_wave,
        model_generator=SOR_RandomInducingPoints,
        m_range=range_inducing,
        runs=n_runs
    )

    # Experiment 1b: Adaptive inducing points
    data['redundant_wave']['adaptive'] = repeat_experiment(
        dataset_getter=load_redundant_wave,
        model_generator=SOR_AdaptiveCrossApproximation,
        m_range=range_inducing,
        runs=n_runs
    )

    # Experiment 1c: KISS
    data['redundant_wave']['KISS'] = repeat_experiment(
        dataset_getter=load_redundant_wave,
        model_generator=KISS,
        m_range=range_kiss,
        runs=n_runs
    )

    # Experiment 1: Plot the results
    plot_data(data, 'redundant_wave', 'mll')
    plot_data(data, 'redundant_wave', 'mse_train')
    plot_data(data, 'redundant_wave', 'mse_test')
    plot_data(data, 'redundant_wave', 'time')

    # Experiment 2a: Show the different inizializations on redundant waves
    tx, ty, vx, vy = load_redundant_wave()
    plot_init(
        SOR_RandomInducingPoints, tx, ty, vx, vy, 10
    )
    plot_init(
        SOR_AdaptiveCrossApproximation, tx, ty, vx, vy, 10
    )

    # Experiment 2b: Show the different inizializations on precipitations
    data['prec_init']['random'] = repeat_experiment(
        dataset_getter=load_precipitations,
        model_generator=SOR_RandomInducingPoints,
        m_range=small_range_inducing,
        runs=n_runs, training_iterations=1
    )
    data['prec_init']['adaptive'] = repeat_experiment(
        dataset_getter=load_precipitations,
        model_generator=SOR_AdaptiveCrossApproximation,
        m_range=small_range_inducing,
        runs=n_runs, training_iterations=1
    )
    plot_data(data, 'prec_init', 'mll', log=False)
    plot_data(data, 'prec_init', 'mse_train', log=False)
    plot_data(data, 'prec_init', 'mse_test', log=False)
    plot_data(data, 'prec_init', 'time')

    # Experiment 3a: Random inducing points
    data['precipitations']['random'] = repeat_experiment(
        dataset_getter=load_precipitations,
        model_generator=SOR_RandomInducingPoints,
        m_range=range_inducing,
        runs=n_runs
    )

    # Experiment 3b: Adaptive inducing points
    data['precipitations']['adaptive'] = repeat_experiment(
        dataset_getter=load_precipitations,
        model_generator=SOR_AdaptiveCrossApproximation,
        m_range=range_inducing,
        runs=n_runs
    )

    # Experiment 3c: KISS
    data['precipitations']['KISS'] = repeat_experiment(
        dataset_getter=load_precipitations,
        model_generator=KISS,
        m_range=small_range_kiss,
        runs=n_runs
    )

    # Experiment 3: Plot the results
    plot_data(data, 'precipitations', 'mll')
    plot_data(data, 'precipitations', 'mse_train')
    plot_data(data, 'precipitations', 'mse_test')
    plot_data(data, 'precipitations', 'time')

    # Store results
    with open('results.pkl', 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main(5)
