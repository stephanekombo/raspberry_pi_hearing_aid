"""Online prototype of the future Raspberry Pi Hearing Aid

TODO: implement a GUI, write a proper docstring

Written by Stéphane Kombo (stephane.kombo.pro@gmail.com) on the base of this file:
https://jackclient-python.readthedocs.io/en/0.4.5/_downloads/thru_client.py

Largely based on the work of Stefan Bleeck and the 2018 GDP Group 16 - University of Southampton

"""

import os
import sys
import jack
import signal
import threading
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from full_process import full_process
from full_fitting import full_fitting
from octave_band_filt import get_filter_coefficients

# %% Select the bypassed signal processors  (if feed_through is True, the function is bypassed)

feed_through_client = not True
feed_through_full_process = not True    # Not understandable error

ft_in_cal = not True
ft_gui = True      # Has to be True until the GUI is implemented
ft_env = not True
ft_nr = not True
ft_com = not True
ft_out_cal = not True
ft_oa = not True
ft_fs = not True
ft_plot = True      # Seems to imply the exit code 139 (interrupted by signal 11: SIGSEGV)

# %% Configure

if not ft_plot:
    plt.interactive(False)
    plt.figure(num='plot_buffers')

# fs = 48000
# sample_factor = 3
# fs_calibration = 44100
# n_bands = 5
# block_length = 512    # Not good
# f_shift = 15
# filter_order = 2  # Used in the octave filter
# bands = 250 * np.power(2, range(n_bands))  # Octave band center frequencies
# step_size = 7  # How often the envelope is sampled. 1: = every time
# max_sig = 110   # An estimate for the maximum dB value you would ever expect as the input

# Default values
fs = 32000
sample_factor = 2
fs_calibration = 44100
n_bands = 5
block_length = 128
f_shift = 15
filter_order = 2  # Used in the octave filter
bands = 250 * np.power(2, range(n_bands))  # Octave band center frequencies
step_size = 7  # How often the envelope is sampled. 1: = every time
max_sig = 110   # An estimate for the maximum dB value you would ever expect as the input

# Activate jack daemon with the desired parameters.
# Beware, --period <-> 2 * block_length
# --nperiods <-?-> sample_factor, --rate <-> f_sample
subprocess.run(["jackd -d alsa --rate 32000 --period 256 -nperiods 2"], shell=True)

# %% Open the compression GUI

knee_x = 55 * np.ones(n_bands)
knee_y = 55 * np.ones(n_bands)
gain0 = 0 * np.ones(n_bands)
t_attack = 5 * 1e-3    # In seconds
t_release = 20 * 1e-3    # In seconds
overall_att = 0# In dB
max_a = 110

x1 = np.array([np.zeros(n_bands), gain0])
x2 = np.array([knee_x, knee_y])
x3 = np.array([max_sig * np.ones(n_bands), max_a * np.ones(n_bands)])

h = full_fitting(x1, x2, x3, t_attack, t_release, overall_att)

# %% Start

prev_in_buffer = np.zeros(block_length)
prev_out_buffer = np.zeros(block_length)

# Take filter coefficients out of the loop, they won't change!
b, a = get_filter_coefficients(fs // sample_factor, bands, filter_order)
# Save more time, prepare hamming window:
hamming_prepare = windows.hamming(2 * block_length)

# %% Setup

in_cali_db = np.array([70, 82.5, 81.5, 78, 74])  # With Stefan's microphone
# in_cali_db = np.array([0, 0, 0, 0, 0])  # Try and error
# in_cali_db = np.array([90, 90, 90, 90, 90])  # For WAV files
out_cali_db = np.array([106, 111.1, 111.3, 110.3, 99.7])  # Same headphone
# out_cali_db = np.array([90, 90, 90, 90, 90])  # Try and error

# %% Client definition

if sys.version_info < (3, 0):
    # In Python 2.x, event.wait() cannot be interrupted with Ctrl+C.
    # Therefore, we disable the whole KeyboardInterrupt mechanism.
    # This will not close the JACK client properly, but at least we can
    # use Ctrl+C.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
else:
    # If you use Python 3.x, everything should be fine.
    pass

argv = iter(sys.argv)
# By default, use script name without extension as client name:
defaultclientname = os.path.splitext(os.path.basename(next(argv)))[0]
clientname = next(argv, defaultclientname)
servername = next(argv, None)

client = jack.Client(clientname, servername=servername)

if client.status.server_started:
    print('JACK server started')
if client.status.name_not_unique:
    print('unique name {0!r} assigned'.format(client.name))

event = threading.Event()

# %% Callback definition

if feed_through_client:

    @client.set_process_callback
    def process(frames):
        assert len(client.inports) == len(client.outports)
        assert frames == client.blocksize
        for in_buffer, out_buffer in zip(client.inports, client.outports):
            in_buffer_view = memoryview(in_buffer.get_buffer()).cast('f')
            out_buffer_view = memoryview(out_buffer.get_buffer()).cast('f')
            for i in range(len(out_buffer_view)):
                out_buffer_view[i] = in_buffer_view[i]

else:

    @client.set_process_callback
    def process(frames):
        assert len(client.inports) == len(client.outports)
        assert frames == client.blocksize
        for in_buffer, out_buffer in zip(client.inports, client.outports):
            in_buffer_view = memoryview(in_buffer.get_buffer()).cast('f')
            out_buffer_view = memoryview(out_buffer.get_buffer()).cast('f')
            out_tmp = full_process(in_buffer_view, prev_in_buffer, prev_out_buffer,
                                   bands, fs, sample_factor, f_shift, block_length, step_size, h,
                                   b, a, hamming_prepare, in_cali_db, out_cali_db,
                                   feed_through_full_process, ft_in_cal, ft_gui, ft_env, ft_nr,
                                   ft_com, ft_out_cal, ft_oa, ft_fs, ft_plot)
            for i in range(len(out_buffer_view)):
                out_buffer_view[i] = out_tmp[i] * 100   # better to control overall_att


@client.set_shutdown_callback
def shutdown(status, reason):
    print('JACK shutdown!')
    print('status:', status)
    print('reason:', reason)
    event.set()


# create two port pairs
for number in 1, 2:
    client.inports.register('input_{0}'.format(number))
    client.outports.register('output_{0}'.format(number))

with client:
    # When entering this with-statement, client.activate() is called.
    # This tells the JACK server that we are ready to roll.
    # Our process() callback will start running now.

    # Connect the ports.  You can't do this before the client is activated,
    # because we can't make connections to clients that aren't running.
    # Note the confusing (but necessary) orientation of the driver backend
    # ports: playback ports are "input" to the backend, and capture ports
    # are "output" from it.

    capture = client.get_ports(is_physical=True, is_output=True)
    if not capture:
        raise RuntimeError('No physical capture ports')

    for src, dest in zip(capture, client.inports):
        client.connect(src, dest)

    playback = client.get_ports(is_physical=True, is_input=True)
    if not playback:
        raise RuntimeError('No physical playback ports')

    for src, dest in zip(client.outports, playback):
        client.connect(src, dest)
        # client.samplerate=44100;

    print('Press Ctrl+C to stop')
    try:
        event.wait()
    except KeyboardInterrupt:
        print('\nInterrupted by user')

    # When the above with-statement is left (either because the end of the
    # code block is reached, or because an exception was raised inside),
    # client.deactivate() and client.close() are called automatically.
