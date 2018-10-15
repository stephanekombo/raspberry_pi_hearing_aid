import numpy as np


def envelope(x, fs, tau_attack, tau_release, step_size, feed_through=False):
    """Find the envelope of a signal.

    Parameters
    ----------
    x: ndarray
        The signal that will be compressed
        Shape: (n_bands, block_length)
    fs: int
        Sampling frequency of the signal
    Hearing aid parameters:
        The following parameters are arrays of floats to be applied to each octave
        band in two channels (left and right ears),
        they have length up to 10 for 2 channels and 5 octave bands
        tau_attack: ndarray, float
            Attack times
        tau_release: ndarray, float
            Release times
        step_size: int
            How often the envelope is sampled. 1: = every time, like the original
        feed_through: bool
            If True, feed through audio only

    Return
    ------
    env: ndarray
        The envelope of the signal
        Shape: (n_bands, block_length)

    Reference
    ---------
    J.M. Kates, Digital Hearing Aids, Plural Publishing Inc., Abingdon,
    UK, 2008, pp. 220‐247

    """

    if feed_through:
        return x

    e0 = envelope   # Shortcut to use a function attribute as a buffer

    x_abs = np.abs(x)
    n_bands, block_length = x.shape
    env = np.zeros((n_bands, block_length))
    alpha_a = np.exp(-1 / (tau_attack * fs))
    alpha_r = np.exp(-1 / (tau_release * fs))

    # Use a function attribute as a buffer
    if not hasattr(e0, 'buffer'):
        e0.buffer = np.zeros(n_bands)

    for j in np.arange(block_length, step=step_size):
        bool_a = (x_abs[:, j] > e0.buffer[:])
        bool_r = (x_abs[:, j] <= e0.buffer[:])
        env[:, j] = bool_a * (alpha_a * e0.buffer[:] + (1 - alpha_a) * x_abs[:, j]) + \
                    bool_r * (alpha_r * e0.buffer[:] + (1 - alpha_r) * x_abs[:, j])
        # if x_abs[:, j] > e0.buffer[:]:
        #     env[:, j] = alpha_a * e0.buffer[:] + (1 - alpha_a) * x_abs[:, j]
        # else:
        #     env[:, j] = alpha_r * e0.buffer[:] + (1 - alpha_r) * x_abs[:, j]

        e0.buffer[:] = env[:, j]

        if j + step_size < block_length:
            env[:, j + 1:j + step_size] = np.tile(env[:, j], (step_size - 1, 1)).T
        else:
            env[:, j + 1:] = np.tile(env[:, j], (block_length - j - 1, 1)).T

    return env
