from matplotlib import pyplot as plt
from scipy.io import loadmat
from typing import Optional
import gpytorch
import numpy as np
import time
import torch


SMALL_SIZE = 12
MEDIUM_SIZE = 17
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def load_elevators() -> tuple[np.ndarray, ...]:
    data = torch.Tensor(loadmat('data/elevators.mat')['data'])
    X = data[:, :-1].detach().numpy()
    # Preprocess as in the original notebook
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1].detach().numpy()

    # Fix random seed
    np.random.seed(42)

    # Permute the data
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]

    # Split in training and test
    train_x = X[:int(0.8*X.shape[0])]
    train_y = y[:int(0.8*X.shape[0])]
    test_x = X[int(0.8*X.shape[0]):]
    test_y = y[int(0.8*X.shape[0]):]

    return train_x, train_y, test_x, test_y


def load_precipitations(
        max_points: Optional[int] = 2000,
        seed: int = 42) -> tuple[np.ndarray, ...]:

    # List to accumulate the examples
    X = []
    y = []

    with open("data/processed-data-2010-jan.csv", "r") as f:
        for line in f.readlines()[1:]:
            # Each line is composed of 7 and 1 target
            tokens = line.split(',')
            assert len(tokens) == 8

            # As in the original implementation, we only
            # require latitude, longitude, time and
            # precipitation
            tokens = tokens[4:]

            # Input features
            X.append(np.array([float(tok) for tok in tokens[:-1]]))
            # Target features
            y.append(np.array([float(tokens[-1])]))

    # Aggregate the examples
    X = np.stack(X)
    y = np.stack(y)

    # Make the typechecker happy
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)

    # Fix random seed
    np.random.seed(seed)

    # Permutate samples
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]

    # Keep only max_points
    if max_points is not None:
        X = X[:max_points]
        y = y[:max_points]

    # Split into training and test sets
    train_x = X[:int(0.8*X.shape[0])]
    train_y = y[:int(0.8*X.shape[0])].reshape(-1)
    test_x = X[int(0.8*X.shape[0]):]
    test_y = y[int(0.8*X.shape[0]):].reshape(-1)

    return train_x, train_y, test_x, test_y


def load_audio():

    # Load the MATLAB matrix
    m = loadmat("data/audio_data.mat")

    # Train set
    X_train, y_train = m["xtrain"], m["ytrain"]
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32).reshape(-1)

    # Test set
    X_test, y_test = m["xtest"], m["ytest"]
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32).reshape(-1)

    return X_train, y_train, X_test, y_test


def load_redundant_wave(a: float = 1.3, b: float = 0.5) \
        -> tuple[np.ndarray, ...]:
    # Sample n = 100 in normal distribution
    # with mean = 0 and std = 1
    n = 100
    x = np.random.normal(0, 1, np.ceil(n/4).astype(np.int32))
    x = np.concatenate(
        (x, np.random.normal(2, 0.2, np.ceil(3*n/4).astype(np.int32)))
    )
    x = np.concatenate(
        (x, np.random.normal(4, 0.2, np.ceil(3*n/4).astype(np.int32)))
    )
    x = np.concatenate(
        (x, np.random.normal(6, 1, np.ceil(n/4).astype(np.int32)))
    )
    y = np.sin(a * x) + b + np.random.normal(0, 5e-2, x.shape)

    # Training set
    x = x.reshape(-1, 1)
    y = y.reshape(-1)

    # Permute the data
    perm = np.random.permutation(x.shape[0])
    train_x = x[perm]
    train_y = y[perm]

    # Test set
    test_x = np.linspace(-3, 9, 100).reshape(-1, 1)
    test_y = np.sin(a * test_x) + b
    test_y = test_y.reshape(-1)

    return train_x, train_y, test_x, test_y


def plot_model(model: gpytorch.models.ExactGP, likelihood,
               train_x: torch.Tensor, train_y: torch.Tensor,
               test_x: torch.Tensor,
               inducing_points: Optional[torch.Tensor] = None) -> None:
    with torch.no_grad():
        # Forward pass
        with gpytorch.settings.cholesky_jitter(1e-1):
            observed_pred = likelihood(model(test_x))

        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(16, 10))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.cpu().numpy(), observed_pred.mean.cpu().numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(
            test_x.cpu().numpy().flatten(), lower.cpu().numpy(),
            upper.cpu().numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

        # Eventually plot vertical lines at the inducing points
        if inducing_points is not None:
            ax.vlines(
              inducing_points.cpu().numpy(), -3, 3, linestyles='dashed')

        # Get timestamp
        timestamp = f"{time.time():.2f}"
        # Save plot
        plt.savefig(f"{timestamp}.png")


def plot_data(data, dataset, metric, log=True):
    plt.figure(figsize=(12, 10))
    plt.xlabel("m")

    # Set label
    if metric == 'mll':
        # latex label
        ylabel = "$\\log P(y \\mid X, \\theta)$"
    elif metric == 'mse_train':
        # latex label
        ylabel = "MSE on Training Set"
    elif metric == 'mse_test':
        # latex label
        ylabel = "MSE on Test Set"
    elif metric == 'time':
        ylabel = "Time (s)"
    else:
        ylabel = metric
    plt.ylabel(ylabel)

    # Log scale y
    if log:
        plt.yscale('log')

    if 'random' in data[dataset]:
        plt.plot(
            data[dataset]['random']['range'],
            data[dataset]['random'][metric],
            label='Random inducing points',
            linewidth=3
        )

    if 'adaptive' in data[dataset]:
        plt.plot(
            data[dataset]['adaptive']['range'],
            data[dataset]['adaptive'][metric],
            label='Adaptive inducing points',
            linewidth=3
        )

    if 'KISS' in data[dataset]:
        plt.plot(
            data[dataset]['KISS']['range'],
            data[dataset]['KISS'][metric],
            label='KISS-GP',
            linewidth=3
        )

    # Show legend
    plt.legend()
    plt.savefig(f"{dataset}_{metric}.png")
