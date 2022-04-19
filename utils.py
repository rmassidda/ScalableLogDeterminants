from scipy.io import loadmat
import numpy as np


def load_precipitations() -> tuple[np.ndarray, np.ndarray]:

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

    y = y.reshape(-1)

    return X, y


def load_audio():

    # Load the MATLAB matrix
    m = loadmat("data/audio_data.mat")

    # Load the full dataset
    X, y = m["xfull"], m["yfull"]

    X = X.astype(np.float32)
    y = y.astype(np.float32).reshape(-1)

    return X, y

    # # Load the train and test sets
    # X_train, y_train = m["xtrain"], m["ytrain"]
    # X_test, y_test = m["xtest"], m["ytest"]

    # return X_train, y_train, X_test, y_test
