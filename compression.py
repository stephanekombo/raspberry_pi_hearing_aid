import numpy as np
from scipy import interpolate


def compression(x, env, knee_x, knee_y, gain0, max_a, max_sig=110, feed_through=False):
    """Compresses a signal given the envelope and other compression parameters

    TODO: write docstring, test it without feed_through

    Parameters
    ----------
    x: ndarray
        The signal that will be compressed
    env: ndarray
        The envelope of x (must have same size as x)
    Hearing aid parameters:
        The following parameters are arrays of floats to be applied to each octave
        band in two channels (left and right ears), they have length up to 10 for 2
        channels and 5 octave bands
        knee_x:
        knee_y:
        gain0:
        max_a:
        max_sig: float
            An estimate for the maximum decibel value you would ever expect as the input
        feed_through: bool
            if True, feed through audio only

    Returns
    -------
    y: ndarray
        the compressed signal

    """

    if feed_through:
        return x

    n_bands, block_length = env.shape

    # Take the dB value of envelope. May raise a warning because of zero values
    # TODO: understand it and correct it
    env_db = np.zeros_like(env)
    nonzero = np.where(env > 0)
    env_db[nonzero] = 20 * np.log10(env[nonzero])
    # env_db = 20 * np.log10(env)

    # Clear up
    env_db[env_db < 1] = 1
    env_db[env_db > max_sig] = max_sig
    env_db = np.round(env_db)

    # Defines the x axis on the output / input plot for compression, defined by the knees
    in_plot = np.vstack((np.zeros(n_bands), knee_x, max_sig * np.ones(n_bands)))

    # Defines the y axis of the output / input plot defined by the lower and upper knees
    out_plot = np.vstack((gain0, knee_y, max_a))

    # Creates an interpolation class for the output / input curve which allows you
    # to give an envelope as an input and receive the desired output signal as an output
    out_env = np.zeros_like(env)
    for i in range(n_bands):
        look_up = interpolate.interp1d(in_plot[:, i], out_plot[:, i])
        out_env[i] = look_up(env_db[i])

    # by subtracting the output and input an array of dB gains to apply to the
    # signal to achieve the correct compression is found
    gain_db = out_env - env_db
    gain = np.power(10, (gain_db / 20))

    y = x * gain

    return y
