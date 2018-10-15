import numpy as np
from scipy import signal
from math import sqrt


# noinspection PyTupleAssignmentBalance
def get_filter_coefficients(fs, bands, filter_order):
    """Get filter coefficients.

    Parameters
    ----------
    fs: int
    bands: ndarray
        Shape: (nbands, )
    filter_order: int

    Returns
    -------
    b: ndarray
        Shape: (nbands, nbands)
    a: ndarray
        Shape: (nbands, nbands)

    """

    # Finds the upper and lower octave band frequencies using f_d
    f_d = sqrt(2)
    f_up = bands * f_d
    f_low = bands / f_d

    # Normalises these frequencies
    w_n = np.array([f_low, f_up]) / (0.5 * fs)

    # Creates empty arrays for the a and b coefficients for each channel
    n_bands = len(bands)
    b = np.zeros([n_bands, 2 * filter_order + 1])
    a = np.zeros([n_bands, 2 * filter_order + 1])

    # If only one band is given then band pass filter it
    if n_bands == 1:
        b, a = signal.butter(filter_order, w_n, btype='bandpass', output='ba')
    # If multiple bands are given:
    # TODO: check slices
    else:
        for i in range(n_bands):
            # For the lowest band low pass filter instead
            if i == 0:
                b[i, :], a[i, :] = signal.butter(2 * filter_order, w_n[0, 0],
                                                 btype='low', output='ba')
            # For the highest band high pass filter instead
            elif i == (n_bands - 1):
                b[i, :], a[i, :] = signal.butter(2 * filter_order, w_n[-1, 1],
                                                 btype='high', output='ba')
            # For all other bands band pass filter
            else:
                b[i, :], a[i, :] = signal.butter(filter_order, w_n[:, i],
                                                 btype='bandpass', output='ba')

    # NOTE: to ensure that the a and b coefficients are a consistent length,
    # the order of the low and high pass filters is double that of the band
    # pass filters. This is because a band pass filter performs both low and
    # high pass filtering so is double the length.

    return b, a


def apply_filter_coefficients(x, b, a):
    """Effective octave band filtering

    Apply the coefficients get by get_filter_coefficients() to filter the input buffer.
    Important: the previous filter states must not reset

    TODO: to understand and correct the docstrings / the code

    Parameters
    ----------
    x: ndarray
        Shape: (buffer_length, )
    b: ndarray
        Shape: (nbands, nbands)
    a: ndarray
        Shape: (nbands, nbands)

    Returns
    -------
    y: ndarray
        Shape: (nbands, buffer_length)

    """

    n_bands = a.shape[0]

    # Use a function attribute as a buffer
    bw_filter = apply_filter_coefficients
    if not hasattr(bw_filter, 'previous_state'):
        bw_filter.previous_state = np.zeros((b.shape[0], 4))

    y = np.zeros((a.shape[0], x.shape[0]))

    for i in range(n_bands):  # apply the coefficients to each band in order
        y[i, :], zi = signal.lfilter(b[i, :], a[i, :], x, axis=-1, zi=bw_filter.previous_state[i, :])
        bw_filter.previous_state[i, :] = zi

    return y
