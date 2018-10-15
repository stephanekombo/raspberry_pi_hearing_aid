import numpy as np


def overlap_add(x, prev_input_buffer, prev_output_buffer, hamming_window, buffer_length, feed_through=False):
    """Overlap and add

    Parameters
    ----------
    x
    prev_input_buffer
    prev_output_buffer
    hamming_window
    buffer_length
    feed_through: bool
        If True, feed through audio only

    """

    if feed_through:
        return x

    # Overlap and add (OAA)
    data_2 = np.concatenate((prev_input_buffer, x))  # OAA
    prev_input_buffer[:] = x
    out_h = hamming_window * data_2  # Hamming

    # Reverse the overlap and add by splitting
    out_2 = out_h[:buffer_length] + prev_output_buffer
    prev_output_buffer[:] = out_h[buffer_length:]
    y = out_2

    return y
