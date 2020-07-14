"""
This module provides examples demonstrating the WWZ by looking at a simple 2 Hz signal and a mixed signal
Please select whether to run in parallel or not.
Can change time divisions (ntau) and scale factor (freq_steps).
"""

import time

# noinspection Mypy
import matplotlib.pyplot as plt
# noinspection Mypy
import numpy as np
import libwwz

# Select Mode...
parallel = True

# number of time
ntau = 400  # Creates new time with this many divisions.
freq_steps = 0.5   # Resolution of frequency steps


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


def run_examples() -> None:
    """
    An example of a sine function time series with missing data will be shown.
    """

    # Set timestamps
    sample_freq = 80
    timestamp = np.arange(0, 10, 1 / sample_freq)

    # Create simple signal (2hz) and complex signal (2hz + 4hz)
    sine_2hz = np.sin(timestamp * 2 * (2 * np.pi))
    sine_4hz = np.sin(timestamp * 4 * (2 * np.pi))

    simple_signal = sine_2hz
    complex_signal = sine_2hz + 0.8*sine_4hz

    # Remove 40% of the signal at random
    simple_removed = remove_fraction_with_seed(simple_signal, 0.4)
    complex_removed = remove_fraction_with_seed(complex_signal, 0.4)
    timestamp_removed = remove_fraction_with_seed(timestamp, 0.4)

    # Get the WWZ of the signals
    starttime = time.time()
    WWZ_simple = libwwz.wwt(timestamp, simple_signal, 1, 5, freq_steps, 0.01, ntau, parallel)
    print(round(time.time() - starttime, 2), 'seconds has passed')
    WWZ_simple_removed = libwwz.wwt(timestamp_removed, simple_removed, 1, 5, freq_steps, 0.01, ntau, parallel)
    print(round(time.time() - starttime, 2), 'seconds has passed')
    WWZ_complex = libwwz.wwt(timestamp, complex_signal, 1, 5, freq_steps, 0.01, ntau, parallel)
    print(round(time.time() - starttime, 2), 'seconds has passed')
    WWZ_complex_removed = libwwz.wwt(timestamp_removed, complex_removed, 1, 5, freq_steps, 0.01, ntau, parallel)
    print(round(time.time() - starttime, 2), 'seconds has passed')

    # Plot
    plt.rcParams["figure.figsize"] = [10, 6]
    plt.rcParams.update({'font.size': 18})

    plt.figure(0)
    plt.subplot(211)
    plt.plot(timestamp, simple_signal)
    plt.plot(timestamp_removed, simple_removed, 'o', alpha=0.7)
    plt.suptitle('The simple signal and complex signal')
    plt.xlabel("time (s)")
    plt.ylabel("amplitude simple")

    plt.subplot(212)
    plt.plot(timestamp, complex_signal)
    plt.plot(timestamp_removed, complex_removed, 'o', alpha=0.7)
    plt.ylabel("amplitude complex")

    plt.figure(1)
    plt.subplot(211)
    plt.contourf(WWZ_simple[0],
                 WWZ_simple[1],
                 WWZ_simple[2])

    plt.subplot(212)
    plt.contourf(WWZ_simple_removed[0],
                 WWZ_simple_removed[1],
                 WWZ_simple_removed[2])

    plt.suptitle('WWZ of simple and simple removed')

    plt.figure(2)
    plt.subplot(211)
    plt.contourf(WWZ_simple[0],
                 WWZ_simple[1],
                 WWZ_simple[3])

    plt.subplot(212)
    plt.contourf(WWZ_simple_removed[0],
                 WWZ_simple_removed[1],
                 WWZ_simple_removed[3])

    plt.suptitle('WWA of simple and simple removed')

    plt.figure(3)
    plt.subplot(211)
    plt.contourf(WWZ_complex[0],
                 WWZ_complex[1],
                 WWZ_complex[2])
    plt.colorbar()


    plt.subplot(212)
    plt.contourf(WWZ_complex_removed[0],
                 WWZ_complex_removed[1],
                 WWZ_complex_removed[2])
    plt.colorbar()


    plt.suptitle('WWZ of complex and complex removed')

    plt.figure(4)
    plt.subplot(211)
    plt.contourf(WWZ_complex[0],
                 WWZ_complex[1],
                 WWZ_complex[3])
    plt.colorbar()


    plt.subplot(212)
    plt.contourf(WWZ_complex_removed[0],
                 WWZ_complex_removed[1],
                 WWZ_complex_removed[3])

    plt.suptitle('WWA of complex and complex removed')
    plt.colorbar()

    plt.show()


if __name__ == "__main__":
    run_examples()
