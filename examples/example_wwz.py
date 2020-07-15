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
import libwwz

# Select Mode...
parallel = True

# number of time
ntau = 400  # Creates new time with this many divisions.
freq_steps = 0.5   # Resolution of frequency steps
c = 0.0125    # Decay constant for analyzing wavelet (negligible at c < 0.02)


# Code to remove data points at random

def remove_fraction_with_seed(data, fraction, seed=np.random.randint(0, 100, 1)):
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
    timestamp = np.arange(0, 10, 1 / sample_freq)

    # Create simple signal (2hz) and mixed signal (2hz + 4hz)
    sine_2hz = np.sin(timestamp * 2 * (2 * np.pi))
    sine_3hz = np.sin(timestamp * 3 * (2 * np.pi))
    sine_4hz = np.sin(timestamp * 4 * (2 * np.pi))

    simple_signal = sine_3hz
    mixed_signal = sine_2hz + 1.3*sine_4hz

    # Remove 40% of the signal at random
    simple_removed = remove_fraction_with_seed(simple_signal, 0.4)
    mixed_removed = remove_fraction_with_seed(mixed_signal, 0.4)
    timestamp_removed = remove_fraction_with_seed(timestamp, 0.4)

    # Get the WWZ/WWA of the signals
    starttime = time.time()
    WWZ_simple = libwwz.wwt(timestamp, simple_signal, 1, 5, freq_steps, c, ntau, parallel)
    print(round(time.time() - starttime, 2), 'seconds has passed (finished WWZ_simple)')
    WWZ_simple_removed = libwwz.wwt(timestamp_removed, simple_removed, 1, 5, freq_steps, c, ntau, parallel)
    print(round(time.time() - starttime, 2), 'seconds has passed (finished WWZ_simple_removed)')
    WWZ_mixed = libwwz.wwt(timestamp, mixed_signal, 1, 5, freq_steps, c, ntau, parallel)
    print(round(time.time() - starttime, 2), 'seconds has passed (finished WWZ_mixed)')
    WWZ_mixed_removed = libwwz.wwt(timestamp_removed, mixed_removed, 1, 5, freq_steps, c, ntau, parallel)
    print(round(time.time() - starttime, 2), 'seconds has passed (finished WWZ_mixed_removed)')

    # Normalize WWZ and multiply by WWA...
    def get_WWZA(wwt: np.ndarray) -> np.ndarray:
        """
        Retruns a normalized WWZ multiplied WWA given a wwt output.
        :param wwt: output of libwwz.wwt
        :return: normalized wwz multiplied by wwa
        """

        WWZ_normalized = (wwt[2] - wwt[2].min()) / (wwt[2].max() - wwt[2].min())
        WWZA = WWZ_normalized * wwt[3]

        return WWZA

    # Plot
    plt.rcParams["figure.figsize"] = [10, 6]
    plt.rcParams.update({'font.size': 16})

    # Plot of base functions
    plt.figure(0)
    plt.subplot(211)
    plt.plot(timestamp, simple_signal, '-')
    plt.plot(timestamp_removed, simple_removed, 'o')
    plt.suptitle('The simple signal and mixed signal')
    plt.ylabel("simple (count)")
    plt.legend(['full', 'removed'], loc=1, fontsize=10)
    plt.xticks([])

    plt.subplot(212)
    plt.plot(timestamp, mixed_signal, '-')
    plt.plot(timestamp_removed, mixed_removed, 'o')
    plt.ylabel("mixed (count)")
    plt.xlabel("time (s)")
    plt.legend(['full', 'removed'], loc=1, fontsize=10)

    # Plot of WWZ for simple and simple removed
    plt.figure(1)
    plt.subplot(211)
    plt.contourf(WWZ_simple[0],
                 WWZ_simple[1],
                 WWZ_simple[2])
    plt.colorbar()
    plt.xticks([])
    plt.ylabel('full (Hz)')

    plt.subplot(212)
    plt.contourf(WWZ_simple_removed[0],
                 WWZ_simple_removed[1],
                 WWZ_simple_removed[2])
    plt.ylabel('removed (Hz)')
    plt.xlabel('time (s)')
    plt.colorbar()
    plt.suptitle('WWZ of simple and simple removed')

    # Plot of WWZ for mixed and mixed removed
    plt.figure(2)
    plt.subplot(211)
    plt.contourf(WWZ_simple[0],
                 WWZ_simple[1],
                 WWZ_simple[3])
    plt.ylabel('full (Hz)')
    plt.colorbar()
    plt.xticks([])

    plt.subplot(212)
    plt.contourf(WWZ_simple_removed[0],
                 WWZ_simple_removed[1],
                 WWZ_simple_removed[3])
    plt.ylabel('removed (Hz)')
    plt.xlabel('time (s)')
    plt.colorbar()
    plt.suptitle('WWA of simple and simple removed')

    # Plot of WWA for simple and simple removed
    plt.figure(3)
    plt.subplot(211)
    plt.contourf(WWZ_mixed[0],
                 WWZ_mixed[1],
                 WWZ_mixed[2])
    plt.ylabel('full (Hz)')
    plt.xticks([])
    plt.colorbar()

    plt.subplot(212)
    plt.contourf(WWZ_mixed_removed[0],
                 WWZ_mixed_removed[1],
                 WWZ_mixed_removed[2])
    plt.ylabel('removed (Hz)')
    plt.xlabel('time (s)')
    plt.colorbar()
    plt.suptitle('WWZ of mixed and mixed removed')

    # Plot of WWA for mixed and mixed removed
    plt.figure(4)
    plt.subplot(211)
    plt.contourf(WWZ_mixed[0],
                 WWZ_mixed[1],
                 WWZ_mixed[3])
    plt.ylabel('full (Hz)')
    plt.xticks([])
    plt.colorbar()

    plt.subplot(212)
    plt.contourf(WWZ_mixed_removed[0],
                 WWZ_mixed_removed[1],
                 WWZ_mixed_removed[3])
    plt.ylabel('removed (Hz)')
    plt.xlabel('time (s)')
    plt.colorbar()
    plt.suptitle('WWA of mixed and mixed removed')

    # Plot of WWZA for simple and simple removed
    plt.figure(5)
    plt.subplot(211)
    plt.contourf(WWZ_simple[0],
                 WWZ_simple[1],
                 get_WWZA(WWZ_simple))
    plt.colorbar()
    plt.xticks([])
    plt.ylabel('full (Hz)')

    plt.subplot(212)
    plt.contourf(WWZ_simple_removed[0],
                 WWZ_simple_removed[1],
                 get_WWZA(WWZ_simple_removed))
    plt.ylabel('removed (Hz)')
    plt.xlabel('time (s)')
    plt.colorbar()
    plt.suptitle('WWA of simple and simple removed')

    # Plot of WWZA for mixed and mixed removed
    plt.figure(6)
    plt.subplot(211)
    plt.contourf(WWZ_mixed[0],
                 WWZ_mixed[1],
                 get_WWZA(WWZ_mixed))
    plt.ylabel('full (Hz)')
    plt.xticks([])
    plt.colorbar()

    plt.subplot(212)
    plt.contourf(WWZ_mixed_removed[0],
                 WWZ_mixed_removed[1],
                 get_WWZA(WWZ_mixed_removed))
    plt.ylabel('removed (Hz)')
    plt.xlabel('time (s)')
    plt.colorbar()
    plt.suptitle('WWA of mixed and mixed removed')

    plt.show()


if __name__ == "__main__":
    run_examples()
