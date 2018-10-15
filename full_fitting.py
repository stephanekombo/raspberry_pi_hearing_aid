def full_fitting(x1, x2, x3, tau_attack, tau_release, overall_att):
    """Open the compression GUI.

    For now (default):

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

    """

    gain_0 = x1[1]
    knee_in = x2[0]
    knee_out = x2[1]
    max_amplitude = x3[0, 0]
    overall_attenuation = overall_att

    return knee_in, knee_out, gain_0, tau_attack, tau_release, max_amplitude, overall_attenuation
