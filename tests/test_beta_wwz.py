"""
This module compares the beta with the current wwz
There is a difference as of June 1st, 2020.
"""

import time

# noinspection Mypy
import matplotlib.pyplot as plt
# noinspection Mypy
import numpy as np
import libwwz
import beta_wwz.beta_wwz as beta_wwz


# Code to remove data points at random

def remove_fraction_with_seed(data, fraction, seed=0):
    """
    removes fraction of data at random with given seed.
    :param data: data to remove
    :param fraction: fraction to remove
    :param seed: seed for randomness
    :return: data with fraction removed
    """
    n_to_remove = int(len(data) * fraction)
    np.random.seed(seed)

    return np.delete(data, np.random.choice(np.arange(len(data)), n_to_remove, replace=False))


def run_comparison_examples() -> None:
    """
    An example of a sine function time series with missing data will be shown.
    """

    # Set timestamps
    sample_freq = 80
    timestamp = np.arange(0, 10, 1 / sample_freq)

    # Create simple signal (2hz) and complex signal (2hz to 3hz)
    simple_signal = np.sin(timestamp * 2 * (2 * np.pi))

    complex_signal = (1 - timestamp / 10) ** 2 * np.sin(timestamp * 2 * (2 * np.pi)) + \
                     (timestamp / 10) ** 2 * np.sin(timestamp * 3 * (2 * np.pi))

    # Remove 40% of the signal at random
    simple_removed = remove_fraction_with_seed(simple_signal, 0.4)
    complex_removed = remove_fraction_with_seed(complex_signal, 0.4)
    timestamp_removed = remove_fraction_with_seed(timestamp, 0.4)

    # Get the WWZ of the signals
    print('starting default')
    starttime = time.time()
    WWZ_simple = libwwz.wwt_para(timestamp, simple_signal, 1, 5, 0.5, 0.01, 800)
    print(round(time.time() - starttime, 2), 'seconds has passed')

    WWZ_simple_removed = libwwz.wwt_para(timestamp_removed, simple_removed, 1, 5, 0.5, 0.01, 400)
    print(round(time.time() - starttime, 2), 'seconds has passed')
    WWZ_complex = libwwz.wwt_para(timestamp, complex_signal, 1, 5, 0.5, 0.01, 800)
    print(round(time.time() - starttime, 2), 'seconds has passed')
    WWZ_complex_removed = libwwz.wwt_para(timestamp_removed, complex_removed, 1, 5, 0.5, 0.01, 400)
    print(round(time.time() - starttime, 2), 'seconds has passed')

    # Get the WWZ of the signals using beta
    print('starting beta')
    starttime = time.time()
    WWZ_simple_beta = beta_wwz.wwt(timestamp, simple_signal, 1, 5, 0.5, 0.01, 800, False)
    print(round(time.time() - starttime, 2), 'seconds has passed')

    WWZ_simple_removed_beta = beta_wwz.wwt(timestamp_removed, simple_removed, 1, 5, 0.5, 0.01, 400, False)
    print(round(time.time() - starttime, 2), 'seconds has passed')
    WWZ_complex_beta= beta_wwz.wwt(timestamp, complex_signal, 1, 5, 0.5, 0.01, 800, False)
    print(round(time.time() - starttime, 2), 'seconds has passed')
    WWZ_complex_removed_beta = beta_wwz.wwt(timestamp_removed, complex_removed, 1, 5, 0.5, 0.01, 400, False)
    print(round(time.time() - starttime, 2), 'seconds has passed')

    # Plot

    plt.rcParams["figure.figsize"] = [14, 6]
    plt.rcParams.update({'font.size': 18})

    plt.figure(0)
    plt.subplot(211)
    plt.plot(timestamp, simple_signal)
    plt.plot(timestamp_removed, simple_removed, 'o', alpha=0.7)
    plt.title('The simple signal and complex signal')

    plt.subplot(212)
    plt.plot(timestamp, complex_signal)
    plt.plot(timestamp_removed, complex_removed, 'o', alpha=0.7)

    plt.figure(1)
    plt.subplot(211)
    plt.contourf(WWZ_simple[0],
                 WWZ_simple[1],
                 WWZ_simple[2])
    plt.title('WWZ plot of the simple and simple removed')

    plt.subplot(212)
    plt.contourf(WWZ_simple_removed[0],
                 WWZ_simple_removed[1],
                 WWZ_simple_removed[2])

    plt.figure(2)

    plt.subplot(211)
    plt.contourf(WWZ_complex[0],
                 WWZ_complex[1],
                 WWZ_complex[2])
    plt.title('WWZ plot of the complex and complex removed')

    plt.subplot(212)
    plt.contourf(WWZ_complex_removed[0],
                 WWZ_complex_removed[1],
                 WWZ_complex_removed[2])

    plt.figure(3)
    plt.subplot(211)
    plt.contourf(WWZ_simple[0] - WWZ_simple_beta[0],
                 WWZ_simple[1] - WWZ_simple_beta[1],
                 WWZ_simple[2] - WWZ_simple_beta[2])
    plt.title('Plot of the difference between beta for simple and simple removed')

    plt.subplot(212)
    plt.contourf(WWZ_simple_removed[0] - WWZ_simple_removed_beta[0],
                 WWZ_simple_removed[1] - WWZ_simple_removed_beta[1],
                 WWZ_simple_removed[2] - WWZ_simple_removed_beta[2])

    plt.figure(4)

    plt.subplot(211)
    plt.contourf(WWZ_complex[0] - WWZ_complex_beta[0],
                 WWZ_complex[1] - WWZ_complex_beta[0],
                 WWZ_complex[2] - WWZ_complex_beta[0])
    plt.title('Plot of the difference between beta for complex and complex removed')

    plt.subplot(212)
    plt.contourf(WWZ_complex_removed[0] - WWZ_complex_removed_beta[0],
                 WWZ_complex_removed[1] - WWZ_complex_removed_beta[1],
                 WWZ_complex_removed[2] - WWZ_complex_removed_beta[2])

    plt.show()


if __name__ == "__main__":
    run_comparison_examples()
