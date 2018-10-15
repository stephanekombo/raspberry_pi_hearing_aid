import numpy as np


def noise_reduction(x, env, feed_through=False):
    """Reduces the noise in the signal by applying a gain between 0 and 1.

    This gain's value is:
        closer to 1 if the Signal-to-Noise Ratio (SNR) is high
        closer to 0 if the SNR is low.
    This means that the buffers with a poor SNR are reduced in volume.

    Parameters
    ----------
    x: ndarray
        the input data
    env: ndarray
        envelope of the input signal
    feed_through: bool
        if True, feed through audio only

    Returns
    -------
    y: ndarray
        noise reduced signal

    See also
    -------
    Wiener filtering

    """

    if feed_through:
        return x

    # Find the largest and the smallest amplitudes in the envelope for each channel
    largest = np.max(env, axis=1)
    smallest = np.min(env, axis=1)

    # Estimate the Signal-to-Noise Ratio
    snr = 10 * np.log10(largest/smallest)

    # Define and apply a gain (between 0 and 1, 1 for a perfect SNR)
    w = snr / (1 + snr)
    y = x * np.tile(w, (x.shape[1], 1)).T

    return y
