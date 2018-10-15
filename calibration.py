import numpy as np


def input_calibration(x, in_calibration_db, buffer_length, feed_through=False):
    """Multiply every channel with the appropriate value.

    Parameters
    ----------
    x: ndarray
        Shape: (nbands, buffer_length)
    in_calibration_db: ndarray
        Shape: (nbands, )
    buffer_length: int
    feed_through: bool
        If True, feed through audio only

    Returns
    -------
    y: ndarray
        Shape: (nbands, buffer_length)

    """

    if feed_through:
        return x

    return x * np.tile(np.power(10, in_calibration_db / 20), (buffer_length, 1)).T


def output_calibration(x, out_calibration_db, overall_att_h, feed_through=False):
    """Output calibration: bring it to the right level as calibrated.

    An amplitude of 1 is the max of the output

    Parameters
    ----------
    x
    out_calibration_db
    overall_att_h
    feed_through: bool
        If True, feed through audio only

    """

    if feed_through:
        return x

    m = np.max(np.abs(x), axis=1)  # Max values in points
    m_db = 20 * np.log10(m)
    new_m_db = out_calibration_db - m_db
    new_m = np.power(10, new_m_db / 20)
    out_c = x / np.tile(m * new_m, (x.shape[1], 1)).T
    # Reduction by a further 20 dB, as this is WAY too loud
    y = out_c * np.power(10, -overall_att_h / 20)

    return y
