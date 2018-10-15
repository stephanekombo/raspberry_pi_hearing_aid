import matplotlib.pyplot as plt


def plot_buffers(input_buffer):
    """Instant plot of buffers"""

    # Clear previous axes in the current figure
    plt.clf()

    # # Adjust subplots display
    # plt.subplots_adjust(left=0.125, right=0.9, bottom=0.05, top=0.85, wspace=0.1, hspace=0.3)

    # # Input subplot
    # plt.subplot(211)
    plt.title("Input buffer", loc='left')
    plt.plot(input_buffer, color='b')

    # Display: not necessary in interactive mode
    # plt.draw()
    # plt.show(block=True)
    # plt.show(block=False)
    plt.pause(0.001)
