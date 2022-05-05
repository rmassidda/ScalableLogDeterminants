from utils import plot_data
import numpy as np
import pickle
import sys


def main(path):
    # Open results
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Experiment 1: Plot the results
    plot_data(data, 'redundant_wave', 'mll')
    plot_data(data, 'redundant_wave', 'mse_train')
    plot_data(data, 'redundant_wave', 'mse_test')
    plot_data(data, 'redundant_wave', 'time')

    # Ugly fix: get the range of the precipitations dataset
    #           for the KISS-GP experiment
    print(data['precipitations']['KISS']['range'])
    data['precipitations']['KISS']['range'] = \
        np.power(data['precipitations']['KISS']['range'], 3)
    print(data['precipitations']['KISS']['range'])
    # cut to the first 2 values
    data['precipitations']['KISS']['range'] = \
        data['precipitations']['KISS']['range'][:2]
    data['precipitations']['KISS']['mll'] = \
        data['precipitations']['KISS']['mll'][:2]
    data['precipitations']['KISS']['mse_train'] = \
        data['precipitations']['KISS']['mse_train'][:2]
    data['precipitations']['KISS']['mse_test'] = \
        data['precipitations']['KISS']['mse_test'][:2]
    data['precipitations']['KISS']['time'] = \
        data['precipitations']['KISS']['time'][:2]
    print(data['precipitations']['KISS']['range'])

    # Experiment 3: Plot the results
    plot_data(data, 'precipitations', 'mll')
    plot_data(data, 'precipitations', 'mse_train')
    plot_data(data, 'precipitations', 'mse_test')
    plot_data(data, 'precipitations', 'time')


if __name__ == "__main__":
    main(sys.argv[1])
