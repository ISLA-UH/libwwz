"""
This module provides examples demonstrating the WWZ by looking at a simple signal (3 Hz) and a mixed signal (2 & 4 Hz).
The amplitude for 2 and 3 Hz are 1, where the 4 Hz is 1.3.

Please select whether to run in parallel or not.
You can change time divisions (ntau), scale factor (freq_steps), and decay_constant (c).

NOTE: The WWZ shows better information on frequency and WWA shows better information on amplitude.
"""

import time

# noinspection Mypy
import matplotlib.pyplot as plt
# noinspection Mypy
import numpy as np
import tests.beta_wwz as beta
import plotting.plot_methods as wwz_plot

# Select Mode...
parallel = True

# number of time
ntau = 20  # Creates new time with this many divisions.

# linear
freq_low = 1
freq_high = 5
freq_steps = 0.1  # Resolution of frequency steps
freq_lin = [freq_low, freq_high, freq_steps]

# octave
freq_target = 2
freq_low = 0.5
freq_high = 8
band_order = 3
log_scale_base = 10**(3/10)
override = False
freq_oct = [freq_target, freq_low, freq_high, band_order, log_scale_base, override]

# other
c = 0.0125    # Decay constant for analyzing wavelet (negligible at c < 0.02)


# Code to remove data points at random

def remove_fraction_with_seed(data, fraction, seed=np.random.randint(1)):
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
    An example of WWZ/WWA using a sine function time series with missing data will be shown.
    """

    # Set timestamps
    sample_freq = 80
    timestamp = np.arange(0, 60, 1 / sample_freq)

    # Create simple signal (2hz)
    sine_2hz = np.sin(timestamp * 2 * (2 * np.pi))

    simple_signal = sine_2hz

    # Remove 80% of the signal at random
    simple_removed = remove_fraction_with_seed(simple_signal, 0.8)
    timestamp_removed = remove_fraction_with_seed(timestamp, 0.8)

    # Get the WWZ/WWA of the signals (linear)
    starttime = time.time()
    WWZ_simple = beta.wwt(timestamp, simple_signal, ntau, freq_oct, c, 'octave')
    print(round(time.time() - starttime, 2), 'seconds has passed (finished WWZ_simple)')
    WWZ_simple_removed = beta.wwt(timestamp_removed, simple_removed, ntau, freq_oct, c, 'octave')
    print(round(time.time() - starttime, 2), 'seconds has passed (finished WWZ_simple_removed)')

    # Plot
    plt.rcParams["figure.figsize"] = [9, 4]
    plt.rcParams.update({'font.size': 14})

    # Plot of base functions
    plt.figure(0)
    plt.plot(timestamp, simple_signal, '-')
    plt.plot(timestamp_removed, simple_removed, 'o')
    plt.ylabel("simple (count)")
    plt.legend(['full', 'removed'], loc=1, fontsize=10)
    plt.xlabel("time (s)")
    plt.suptitle('The simple signal (2 Hz)')

    # Plot of WWZ for simple and simple removed
    wwz_plot.octave_plotter(WWZ_simple[0],
                            WWZ_simple[1],
                            WWZ_simple[2],
                            band_order=band_order,
                            log_scale_base=log_scale_base)
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time')
    plt.title('WWZ of the full signal (2 Hz)')

    wwz_plot.octave_plotter(WWZ_simple_removed[0],
                            WWZ_simple_removed[1],
                            WWZ_simple_removed[2],
                            band_order=band_order,
                            log_scale_base=log_scale_base)
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time')
    plt.title('WWZ of the removed signal (2 Hz)')


    # Plot of WWA for the same signal
    wwz_plot.octave_plotter(WWZ_simple[0],
                            WWZ_simple[1],
                            WWZ_simple[3],
                            band_order=band_order,
                            log_scale_base=log_scale_base)
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time')
    plt.title('WWA of the full signal (2 Hz)')

    wwz_plot.octave_plotter(WWZ_simple_removed[0],
                            WWZ_simple_removed[1],
                            WWZ_simple_removed[3],
                            band_order=band_order,
                            log_scale_base=log_scale_base)
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time')
    plt.title('WWA of the removed signal (2 Hz)')

    plt.show()


if __name__ == "__main__":
    run_examples()
