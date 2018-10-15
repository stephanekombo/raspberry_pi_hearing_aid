import numpy as np


def freq_shift(x, fs, f_shift, feed_through=False):
    """Performs a shift in the frequency domain and converts back to the time domain

    TODO: test it

    Parameters
    ----------
    x: ndarray
        The data to be shifted
    fs: int
        Sampling frequency of the data
    f_shift: int
        the frequency to shift the up by, this will be approximate as the exact
        frequency shift is limited to sampling limitations
    feed_through: bool
        If True, feed through audio only

    Returns
    -------
    x_shift: ndarray
        The time domain, frequency shifted data

    """

    if feed_through:
        return x

    x.shape = (-1, x.size)
    block_length = x.shape[1]

    # Compute the Fourier transform of the data
    fft_x = np.fft.rfft(x, n=block_length)

    # Work out the number of samples to shift the data by
    sample_shift = f_shift * block_length / (fs/2)

    # If the number of samples is less than one then shift by one sample instead
    if sample_shift < 1:
        sample_shift = 1

    sample_shift = int(sample_shift)

    # Shifted signal
    fft_x_shift_f = np.zeros_like(fft_x)

    # Use indexing to shift the signal, the frequencies below f_shift will be 0
    fft_x_shift_f[:, sample_shift:] = fft_x[:, :-sample_shift]

    # Inverse Fourier transform the signal to get it back into the time domain
    x_shift = np.fft.irfft(fft_x_shift_f, n=block_length)

    return x_shift
