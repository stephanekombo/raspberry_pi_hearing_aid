import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
from octave_band_filt import get_filter_coefficients, apply_filter_coefficients
from calibration import input_calibration, output_calibration
from envelope import envelope
from noise_reduction import noise_reduction
from compression import compression
from overlap_add import overlap_add
from freq_shift import freq_shift
from plot_buffers import plot_buffers
from get_current_val import get_current_val
from full_fitting import full_fitting


def full_process(input_buffer, prev_input_buffer, prev_output_buffer,
                 octave_bands, f_sample, sample_factor, f_shift, buffer_length, step_length, h,
                 b_coefficients, a_coefficients, hamming_window, in_calibration_db, out_calibration_db,
                 feed_through=False, ft_in_cal=False, ft_gui=False, ft_env=False, ft_nr=False,
                 ft_com=False, ft_out_cal=False, ft_oa=False, ft_fs=False, ft_plot=False):
    """Offline prototype of the future Raspberry Pi Hearing Aid

    This main function calls all the signal processors needed to enhance each buffer.
    Supposed to be call in the process callback in the online version.
    TODO: implement a GUI, write docstring

    Written by Stéphane Kombo (stephane.kombo.pro@gmail.com)
    Largely based on the work of Stefan Bleeck and the 2018 GDP Group 16 - University of Southampton

    Parameters
    ----------
    input_buffer: ndarray
        Shape: (sample_factor * buffer_length, )
    prev_input_buffer: ndarray
        Shape: (buffer_length, )
    prev_output_buffer: ndarray
        Shape: (buffer_length, )
    octave_bands: ndarray
        Shape: (nbands, )
    f_sample: int
        For now, ALSA seems to impose 48000 Hz. We want 32000 Hz.
    sample_factor: int
        Factor to use in order to have a 16 KHz in the signal processors.
        sample_factor is 3 if f_sample is 48 KHz, sample_factor is 2 if f_sample is 32 KHz
        Should correspond to the nperiods ALSA parameter. TODO: confirm
    f_shift: int
    buffer_length: int
        Need to be a power of 2 for JACK.
    step_length: int
        How often the signal is sampled for the envelope
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
    b_coefficients: ndarray
        Shape: (nbands, nbands)
    a_coefficients: ndarray
        Shape: (nbands, nbands)
    hamming_window: ndarray
        Shape: (2 * buffer_length, )
    in_calibration_db: ndarray
        Shape: (nbands, )
    out_calibration_db: ndarray
        Shape: (nbands, )
    feed_through: bool
        If True, feed through audio only in the full process (the full process is bypassed)
        If False, the regular process is called
    ft_in_cal: bool
        If True, feed through audio only in the input calibration (it is bypassed)
    ft_gui: bool
        If True, GUI is bypassed
    ft_env: bool
        If True, feed through audio only in the envelope computation (it is bypassed)
    ft_nr: bool
        If True, feed through audio only in the noise reduction (it is bypassed)
    ft_com: bool
        If True, feed through audio only in the compression (it is bypassed)
    ft_out_cal: bool
        If True, feed through audio only in the output calibration (it is bypassed)
    ft_oa: bool
        If True, feed through audio only in the overlap and add (it is bypassed)
    ft_fs: bool
        If True, feed through audio only in the frequency shift (it is bypassed)
    ft_plot: bool
        If False, input and output buffers are plotted a each step (it is bypassed)

    Returns
    -------
    output_buffer: ndarray
        Shape: (sample_factor * buffer_length, )
        Only needed for the online version

    """

    if feed_through:
        # In the online version, we have to copy the buffers as array because memory views are read only
        in_buf = np.array(input_buffer, copy=True)
        if not ft_plot:
            plot_buffers(in_buf)
        # Needed for thee online version
        return in_buf

    # Downsample
    in_data = input_buffer[::sample_factor]

    # Octave band filtering
    out_f = apply_filter_coefficients(in_data, b_coefficients, a_coefficients)

    # Input calibration: multiply every channel with the appropriate value
    out_f_c = input_calibration(out_f, in_calibration_db, buffer_length, feed_through=ft_in_cal)

    # Get the values from the open GUI
    knee_x_h, knee_y_h, gain0_h, t_attack_h, t_release_h, max_a_h, overall_att_h = get_current_val(h, octave_bands,
                                                                                                   feed_through=ft_gui)

    # Envelope
    env = envelope(out_f_c, f_sample//sample_factor, t_attack_h, t_release_h, step_length, feed_through=ft_env)

    # Noise reduction
    out_nr = noise_reduction(out_f_c, env, feed_through=ft_nr)

    # Compression
    out_com = compression(out_nr, env, knee_x_h, knee_y_h, gain0_h, max_a_h, feed_through=ft_com)

    # Output calibration
    out_cal = output_calibration(out_com, out_calibration_db, overall_att_h, feed_through=ft_out_cal)

    # Sum channels into the right format
    out_sum = np.sum(out_cal, axis=0)

    # Overlap and add (OAA)
    out_oa = overlap_add(out_sum, prev_input_buffer, prev_output_buffer,
                         hamming_window, buffer_length, feed_through=ft_oa)

    # Feedback cancellation using frequency shift
    out_fs = freq_shift(out_oa, f_sample // sample_factor, f_shift, feed_through=ft_fs)

    # Upsample
    out_up = np.reshape(sample_factor * [out_fs], newshape=(sample_factor * out_fs.size), order='F')

    if not ft_plot:
        plot_buffers(input_buffer)

    # Needed for thee online version
    return out_up


# Script in order to test the function
if __name__ == '__main__':
    # %% Select the bypassed signal processors (if feed_through is True, the function is bypassed)

    feed_through_full_process = not True
    ft_in_cal_ = not True
    ft_gui_ = True      # Has to be True until the GUI is implemented
    ft_env_ = not True
    ft_nr_ = not True
    ft_com_ = not True
    ft_out_cal_ = not True
    ft_oa_ = not True
    ft_fs_ = not True
    ft_plot_ = not True      # Seems to imply the exit code 139 (interrupted by signal 11: SIGSEGV) in the online code

    if not ft_plot_:
        plt.interactive(False)
        plt.figure(num='plot_buffers')

    # %% Configure

    test_filename = 'BKBE0209_32K.WAV'  # Choose a WAV file to test

    # Test without the control on JACK. Not implemented yet.
    # The problem is that JACK can only read ALSA parameters and not set it up.
    # THe best way seems to coonfigure ALSA with qjackctl before lanching the hearing aid.
    fs = 48000
    sample_factor_ = 3
    fs_calibration = 44100
    n_bands = 5
    block_length = 256
    f_shift_ = 15
    filter_order = 2  # Used in the octave filter
    bands = 250 * np.power(2, range(n_bands))  # Octave band center frequencies
    step_size = 7  # How often the envelope is sampled. 1: = every time
    max_sig = 110    # An estimate for the maximum dB value you would ever expect as the input

    # # Default test
    # fs = 32000
    # sample_factor_ = 2
    # fs_calibration = 44100
    # n_bands = 5
    # block_length = 256
    # f_shift_ = 15
    # filter_order = 2  # Used in the octave filter
    # bands = 250 * np.power(2, range(n_bands))  # Octave band center frequencies
    # step_size = 7  # How often the envelope is sampled. 1: = every time
    # max_sig = 110  # An estimate for the maximum dB value you would ever expect as the input

    # %% Open the compression GUI

    # Initial values
    knee_x = 55 * np.ones(n_bands)
    knee_y = 55 * np.ones(n_bands)
    gain0 = 0 * np.ones(n_bands)
    t_attack = 5 * 1e-3  # In seconds
    t_release = 20 * 1e-3  # In seconds
    overall_att = 30
    max_a = 110

    x1 = np.array([np.zeros(n_bands), gain0])
    x2 = np.array([knee_x, knee_y])
    x3 = np.array([max_sig * np.ones(n_bands), max_a * np.ones(n_bands)])

    h_ = full_fitting(x1, x2, x3, t_attack, t_release, overall_att)

    # %% Start

    fs_file, all_data = wavfile.read(test_filename)  # Read from file
    all_data = (all_data / 32768)   # For wavfile
    all_data = signal.resample_poly(all_data, fs_file, 32000)  # You might use fs instead of 32000

    prev_in_buffer = np.zeros(block_length)
    prev_out_buffer = np.zeros(block_length)

    # Take filter coefficients out of the loop, they won't change!
    b, a = get_filter_coefficients(fs // sample_factor_, bands, filter_order)
    # Save more time, prepare hamming window:
    hamming_prepare = signal.windows.hamming(2 * block_length)

    # %% Setup

    nr_loop = all_data.size // (2 * block_length) - 1  # Number of sample for the loop
    full_out = np.zeros(all_data.size)
    in_cali_db = np.array([90, 90, 90, 90, 90])  # In a file they are all the same loud
    out_cali_db = np.array([106, 111.1, 111.3, 110.3, 99.7])  # Same headphone

    c_count = 0

    # %% Run in a loop

    for ind in range(nr_loop):
        in_buffer = all_data[c_count:c_count + sample_factor_ * block_length]  # Read from file
        out_buffer = full_out[c_count:c_count + sample_factor_ * block_length]

        if ind >= len(all_data) // (sample_factor_ * block_length):
            break

        out_buffer[:] = full_process(in_buffer, prev_in_buffer, prev_out_buffer,
                                     bands, fs, sample_factor_, f_shift_, block_length, step_size, h_,
                                     b, a, hamming_prepare, in_cali_db, out_cali_db,
                                     feed_through_full_process, ft_in_cal_, ft_gui_, ft_env_,
                                     ft_nr_, ft_com_, ft_out_cal_, ft_fs_, ft_plot_)

        c_count = c_count + sample_factor_ * block_length

        # Allows the GUI to catch up with reading the parameters value
        # % draw now

    full_out = full_out * 1000
    wavfile.write('full_out.wav', 32000, full_out)
    # wavfile.write('full_out.wav', fs, full_out)

    print('\nDone\n')
