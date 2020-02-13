"""
This module provides functions for computing the weighted wavelet z transform over input values.
"""

import numpy as np


def wwt(timestamp: np.ndarray,
        magnitude: np.ndarray,
        f_lo: int,
        f_hi: int,
        df: float,
        dcon: float,
        time_divisions: int) -> np.ndarray:
    """
    This function takes timestamps (floats not datetime obj), magnitude (payload values), f_lo, f_hi, df (the desired \
    ranges of frequencies (f_lo to f_hi) and the steps (df), dcon (the window steps for WWZ), and time_divisions ( \
    number of points to make tau (the new timestamp that is evenly spaced). The code is based on G. Foster's FORTRAN
    code as well as eaydin's python 2.7 code. The code is updated to use numpy methods and allow for float value tau.
    It returns an array with matrix of new evenly spaced timestamps, frequencies, wwz-power, amplitude, coefficients,
    and effective number. Specific equations can be found on Grant Foster's "WAVELETS FOR PERIOD ANALYSIS OF UNEVENLY
    SAMPLED TIME SERIES". Some of the equations are labeled in the code with corresponding numbers.

    :param timestamp: An array with corresponding times for the magnitude (payload).
    :param magnitude: An array with payload values
    :param f_lo: the low end of frequency to cast WWZ
    :param f_hi: the high end of frequency to cast WWZ
    :param df: frequency steps for casting WWZ
    :param dcon: decay constant for the Morlet wavelet (negligible <0.02)
    :param time_divisions:
    :return: Tau, Freq, WWZ, AMP, COEF, NEFF in a pandas DataFrame
    """

    # Initialize before the for loops
    dvarw = 0.0  # The wavelet variance

    # Frequencies to compute WWZ
    freq = np.linspace(f_lo, f_hi, round((f_hi - f_lo) / df) + 1)
    nfreq = len(freq)

    # Insure time_divisions are smaller than the time stamp array
    if time_divisions > len(timestamp):
        time_divisions = len(timestamp)
        print('adjusted time_divisions to: ', time_divisions)

    # Time Shifts (tau) to compute WWZ
    tau = np.linspace(timestamp[0], timestamp[-1], time_divisions)
    ntau = len(tau)

    # Creating array for output
    output = np.empty((ntau * nfreq, 6))
    numdat = len(timestamp)
    index = 0

    # WWT Stars Here

    for dtau in tau:
        # Initialize the outputs for each iteration
        nstart = 1
        dmfre = 0.0
        dmamp = 0.0
        dmcon = 0.0
        dmneff = 0.0  # N_eff is the effective number (x)^2/(x^2)
        dmz = -1.0  # less than the smallest WWZ

        # loop over each interested frequency over the taus
        for dfreq in freq:
            # Initialize a vector (3) and matrix (3,3) and dweight2 and set domega
            dvec = np.zeros(3)
            dmat = np.zeros([3, 3])
            dweight2 = 0
            domega = 2 * np.pi * dfreq

            # Get weights
            for idat in range(nstart, numdat):
                # initialize dz and dweight
                dz = domega * (timestamp[idat] - dtau)
                dweight = np.exp(-1 * dcon * dz ** 2)
                # get upper triangular matrix of the weights and vector
                if dweight > 10 ** -9:
                    cos_dz = np.cos(dz)
                    sin_dz = np.sin(dz)
                    dweight2 += dweight ** 2
                    dvarw += dweight * magnitude[idat] ** 2

                    dmat[0, 0] += dweight
                    dmat[0, 1] += dweight * cos_dz
                    dmat[0, 2] += dweight * sin_dz
                    dmat[1, 1] += dweight * cos_dz ** 2
                    dmat[1, 2] += dweight * cos_dz * sin_dz
                    dmat[2, 2] += dweight * sin_dz ** 2

                    dvec[0] += dweight * magnitude[idat]
                    dvec[1] += dweight * magnitude[idat] * cos_dz
                    dvec[2] += dweight * magnitude[idat] * sin_dz

                elif dz > 0:
                    break
                else:
                    nstart = idat + 1

            # Get dneff
            if dweight2 > 0:
                dneff = (dmat[0, 0] ** 2) / dweight2
            else:
                dneff = 0

            # Get damp, dpower, dpowz
            dcoef = [0, 0, 0]
            damp = 0
            dpower = 0

            if dneff > 3:
                dvec = dvec / dmat[0, 0]
                # avoid for loops
                dmat[..., 1:] /= dmat[0, 0]

                # set dvarw
                if dmat[0, 0] > 0.005:
                    dvarw = dvarw / dmat[0, 0]
                else:
                    dvarw = 0.0

                # some initialize
                dmat[0, 0] = 1.0
                davew = dvec[0]
                dvarw = dvarw - (davew ** 2)

                if dvarw <= 0.0:
                    dvarw = 10 ** -12

                # avoid for loops
                dmat[1, 0] = dmat[0, 1]
                dmat[2, 0] = dmat[0, 2]
                dmat[2, 1] = dmat[1, 2]

                dmat = np.linalg.inv(dmat)

                # set dcoef and dpower
                dcoef = dmat.dot(dvec)
                dpower = np.dot(dcoef, dvec) - (davew ** 2)

                dpowz = (dneff - 3.0) * dpower / (2.0 * (dvarw - dpower))
                dpower = (dneff - 1.0) * dpower / (2.0 * dvarw)
                damp = np.sqrt(dcoef[1] ** 2 + dcoef[2] ** 2)

            else:
                dpowz = 0.0
                dpower = 0.0
                damp = 0.0

            if dneff < (10 ** (-9)):
                dneff = 0.0

            if damp < (10 ** (-9)):
                damp = 0.0

            if dpower < (10 ** (-9)):
                dpower = 0.0

            if dpowz < (10 ** (-9)):
                dpowz = 0.0

            # Let's write everything out.
            output[index] = [dtau, dfreq, dpowz, damp, dcoef[0], dneff]

            index = index + 1

    # Format the output to be in len(tau) by len(freq) matrix for each value with correct labels
    # NEEDS WORK

    tau_mat = output[:, 0].reshape([ntau, nfreq])
    freq_mat = output[:, 1].reshape([ntau, nfreq])
    wwz_mat = output[:, 2].reshape([ntau, nfreq])
    amp_mat = output[:, 3].reshape([ntau, nfreq])
    dcoef_mat = output[:, 4].reshape([ntau, nfreq])
    dneff_mat = output[:, 5].reshape([ntau, nfreq])

    return output
