import numpy as np


def get_current_val(h, bands, feed_through=False):
    """Get the values from the open GUI.

    ...In theory. The GUI is still TODO.

    Parameters
    ----------
    h: tuple
        Parameters gotten from the GUI
        knee_in: ndarray
            Shape: (nbands, )
        knee_out:ndarray
            Shape: (nbands, )
        gain_0: ndarray
            Shape: (nbands, )
        tau_attack: float
        tau_release: float
        max_amplitude: float
        overall_attenuation: float
    bands: ndarray
        Shape: (nbands, )
    feed_through: bool
        If True, feed through audio only

    Returns
    -------
    knee_x_h: ndarray
        Shape: (nbands, )
    knee_y_h:ndarray
        Shape: (nbands, )
    gain_0_h: ndarray
        Shape: (nbands, )
    t_attack_h: ndarray
        Shape: (nbands, )
    t_release_h: ndarray
        Shape: (nbands, )
    max_a_h: ndarray
        Shape: (nbands, )
    overall_attenuation: float

    """

    # Read the input tuple h
    knee_in, knee_out, gain_0, tau_attack, tau_release, max_amplitude, overall_attenuation = h

    if feed_through:
        # Values are passed without the GUI.
        knee_x_h = knee_in
        knee_y_h = knee_out
        gain0_h = gain_0
        t_attack_h = tau_attack * np.ones_like(bands)
        t_release_h = tau_release * np.ones_like(bands)
        max_a_h = max_amplitude * np.ones_like(bands)
        overall_att_h = overall_attenuation

    # TODO: call the right GUI functions here if GUI used
    pass

    return knee_x_h, knee_y_h, gain0_h, t_attack_h, t_release_h, max_a_h, overall_att_h
