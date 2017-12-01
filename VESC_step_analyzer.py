import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LogLocator
from matplotlib.colors import LogNorm


# Parse the maximum current, p gain, d gain from the file name
# The filename is the full local path, ie 'data/time_angle_I15d100p8.npy'
# The file prefix is the prefix before the params, ie 'data/time_angle_'
def get_Imax_P_D_from_file(file_prefix, file):
    config_substring = file[len(file_prefix):]

    current = float(config_substring[config_substring.find("I") + 1:config_substring.find("p")])
    p_gain = float(config_substring[config_substring.find("p") + 1:config_substring.find("d")]) / 1000
    d_gain = float(config_substring[config_substring.find("d") + 1:config_substring.find(".")]) / 1000
    return (current, p_gain, d_gain)


# Return tuple of (positions, times) from given data file
def load_position_time_vectors(filename):
    data = np.load(filename)
    return (data[:, 1], data[:, 0])


# Get position vector from data collected from the V1 1.5s step test
def v1_get_step_subset(position_vector):
    step2_index = [500, 600]
    step2_data = position_vector[slice(*step2_index)]
    return step2_data


# Create a dictionary of specifications from each step test
def load_spec_dict(filenames_prefix):
    # Configure which parameters to log
    param_names = ['step_data', 'I_max', 'Kp', 'Kd', 'Mp', 'Tr', 'Ts', 'Damping']
    step_array_size = 100

    filenames = glob.glob(filenames_prefix + '*')
    num_files = len(filenames)

    # independent variables are I_max, Kp, and Kd, so
    # organize a dictionary of 3d arrays with imax, kp, kd
    # as indices and dictionaries of specs as values
    param_values = [np.zeros((num_files, 1)) for i in param_names]
    param_values[0] = np.zeros((num_files, step_array_size))

    # TODO: change the Kd multiplier for when it saves the file so that 0.0005 Kd is not 0.0
    kd_test_vals = [0.0000, 0.001, 0.002, 0.004, 0.008]
    kp_test_vals = [0.025, 0.05, 0.1, 0.2, 0.3]
    I_max_test_vals = [15, 30]
    kdn = len(kd_test_vals)
    kpn = len(kp_test_vals)
    I_maxn = len(I_max_test_vals)

    # make sure our files match up with our matrix dimensions
    assert kdn * kpn * I_maxn == num_files

    param_values = [np.zeros((I_maxn, kpn, kdn)) for i in param_names]

    # Use a 4D array to store the step test encoder readings
    param_values[0] = np.zeros((I_maxn, kpn, kdn, step_array_size))

    param_dict_matrix = dict(zip(param_names, param_values));

    index = 0
    for filename in filenames:
        # Get max current, p, d gains from filename
        (current, p_gain, d_gain) = get_Imax_P_D_from_file(filenames_prefix, filename)

        # Load encoder angle history and teensy timestamps
        (position, times) = load_position_time_vectors(filename)

        # Indicies specifying which phases happen when
        step2_data = v1_get_step_subset(position)

        dt = 0.002

        Mp = overshoot(step2_data)
        tr = rise_time(step2_data, dt)
        ts = settling_time(step2_data, dt)
        damp2 = damping_from_tstr(step2_data, dt)
        # Can also find damping coeff from the overshoot,
        # but doesn't apply for damp > 1
        # damp = damping_from_Mp(step2_data)

        kp_index = kp_test_vals.index(p_gain)
        kd_index = kd_test_vals.index(d_gain)
        I_max_index = I_max_test_vals.index(current)

        param_dict_matrix['step_data'][I_max_index][kp_index][kd_index][0:len(step2_data)] = step2_data
        param_dict_matrix['Mp'][I_max_index][kp_index][kd_index] = Mp
        param_dict_matrix['Tr'][I_max_index][kp_index][kd_index] = tr
        param_dict_matrix['Ts'][I_max_index][kp_index][kd_index] = ts
        param_dict_matrix['Damping'][I_max_index][kp_index][kd_index] = damp2
    return param_dict_matrix


## Analyze overshoot during first step ##
def overshoot(step_position):
    # maximum angle (overshoot over 180deg)
    end_val = step_position[-1]
    start_val = step_position[0]
    if end_val > start_val:
        max_val = np.max(step_position)
    else:
        max_val = np.min(step_position)

    # difference btn maximum and steady state value
    overshoot = abs(max_val - end_val)

    # magnitude in degrees of the step upwards
    step_size = abs(end_val - start_val)

    # maximum overshoot parameter
    return overshoot / step_size


## Analyze rise time
def rise_time(step_position, dt):
    # should robably find the dt from the time variable instead

    # 10% mark
    # signed size of the step
    initial_val = step_position[0]
    ss_val = step_position[-1]
    step_size = ss_val - initial_val

    ten_percent = initial_val + 0.1 * step_size
    ninety_percent = initial_val + 0.9 * step_size

    if ss_val > initial_val:
        rise_time = (np.argmax(step_position > ninety_percent) -
                     np.argmax(step_position > ten_percent)) * dt
    else:
        rise_time = (np.argmax(step_position < ninety_percent) -
                     np.argmax(step_position < ten_percent)) * dt

    return rise_time


## Find the settling time (oscillation < 1% of ss value)
def settling_time(step_position, dt):
    # take the last entry as the steady state value
    initial_val = step_position[0]
    ss_val = step_position[-1]
    step_size = abs(ss_val - initial_val)

    last_steady_index = 0

    logical_greater_than_ss = (abs(step_position - ss_val) > step_size * 0.01)

    # reverse the logical array to find the last index of TRUE, which indicates
    # the first datum within 1% of the ss value
    index_last_out_of_bounds = ((step_position.size - 1) -
                                np.argmax(logical_greater_than_ss[::-1]))

    return index_last_out_of_bounds * dt


## Estimates the damping coefficient from the amount of overshoot
def damping_from_Mp(step_position):
    Mp = overshoot(step_position)
    return damping_from_Mp_given_Mp(Mp)


def damping_from_Mp_given_Mp(Mp):
    if Mp < 0.001:
        return 1.0
    return (np.log(Mp) ** 2 / (np.pi ** 2 + np.log(Mp) ** 2)) ** 0.5


def damping_from_tstr(step_position, dt):
    wn = 1.8 / rise_time(step_position, dt);
    ts = settling_time(step_position, dt)
    damp_est = -np.log(0.01) / (ts * wn)
    return damp_est


# Estimate the m coefficient of the 2nd order PID
# LTI using the rise time and k coefficient
def m_estimate(step_position, dt, k):
    wn = 1.8 / rise_time(step_position, dt)
    return k / (wn ** 2)


# Given the rise time matrix, estimate m for each i, kd, kp
def estimate_m_matrix(Tr_matrix):
    kps = [0.025, 0.05, 0.1, 0.2, 0.3]
    m_div_k = (Tr_matrix / 1.8) ** 2
    m_matrix = m_div_k.copy()
    for kdkp_matrix in m_matrix:
        for i in np.arange(0, kdkp_matrix.shape[0]):
            kdkp_matrix[i, :] = kdkp_matrix[i, :] * kps[i]

    return m_matrix


# Given the settling time matrix, estimate m
def estimate_m_matrix_from_Ts(Ts_matrix):
    kds = [0.0005, 0.001, 0.002, 0.004, 0.008]
    m_div_b = Ts_matrix / (2 * np.log(100))
    m_matrix = m_div_b.copy()
    for kdkp_matrix in m_matrix:
        for i in np.arange(0, kdkp_matrix.shape[1]):
            kdkp_matrix[:, i] = kdkp_matrix[:, i] * kds[i]

    return m_matrix


# Estimate m coefficient from the exp. damping ratio. Verified
def estimate_m_from_damping(D_matrix):
    kps = [0.025, 0.05, 0.1, 0.2, 0.3]
    kds = [0.0005, 0.001, 0.002, 0.004, 0.008]
    [kds_mat, kps_mat] = np.meshgrid(kds, kps)
    # D = np.zeros_like(kdkps)
    D_times_root_m = kds_mat / (2 * np.sqrt(kps_mat))

    return (D_times_root_m / D_matrix) ** 2


def custom_contour_levels(data, n=8):
    return np.logspace(np.log10(data.min()), np.log10(data.max()), n)


def custom_contour_linear_levels(data, n=8):
    return np.linspace(np.log10(data.min()), np.log10(data.max()), n)


# Plot two contour plots of the m estimates. Verified
def plot_m_estimates(param_dict):
    kps = [0.025, 0.05, 0.1, 0.2, 0.3]
    kds = [0, 0.001, 0.002, 0.004, 0.008]
    m_matrix = estimate_m_matrix(param_dict['Tr'])[0]
    m_matrix2 = estimate_m_matrix(param_dict['Ts'])[0]

    m_matrix3 = estimate_m_from_damping(param_dict['Damping'])[0]

    multiplier = 10e6

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    m_plot = ax1.contourf(kds, kps, np.log10(m_matrix * multiplier))
    cbar = f.colorbar(m_plot, ax=ax1, ticks=m_plot.levels)
    cbar.ax.set_yticklabels(np.round((10 ** m_plot.levels), 2))
    ax1.set_title('m estimate for 15A (1e-6) from Tr')
    ax1.set_ylabel('Kp')
    ax1.set_xlabel('Kd')

    m_plot2 = ax2.contourf(kds, kps, np.log10(m_matrix2 * multiplier))
    cbar = f.colorbar(m_plot2, ax=ax2, ticks=m_plot2.levels)
    cbar.ax.set_yticklabels(np.round((10 ** m_plot2.levels), 2))
    ax2.set_title('m estimate for 15A (1e-6) from Ts')
    ax2.set_ylabel('Kp')
    ax2.set_xlabel('Kd')

    m_plot3 = ax3.contourf(kds, kps, np.log10(m_matrix3 * multiplier))
    cbar = f.colorbar(m_plot3, ax=ax3, ticks=m_plot3.levels)
    cbar.ax.set_yticklabels(np.round((10 ** m_plot3.levels), 2))
    ax3.set_title('m estimate for 15A (1e-6) from Damping')
    ax3.set_ylabel('Kp')
    ax3.set_xlabel('Kd')

    f.savefig('plots/Mass_Estimations.png',dpi=400)


# Plot actual damping coeff vs theoretical damping*sqrt(m). Verified
def plot_exp_damping_vs_theoretical_damping(param_dict):
    kps = [0.025, 0.05, 0.1, 0.2, 0.3]
    kds = [0.0005, 0.001, 0.002, 0.004, 0.008]
    [kds_mat, kps_mat] = np.meshgrid(kds, kps)
    # D = np.zeros_like(kdkps)
    D_times_root_m = kds_mat / (2 * np.sqrt(kps_mat))

    f, (ax1, ax2) = plt.subplots(1, 2)

    D = param_dict['Damping'][0]
    D_theory = D_times_root_m

    D_plot = ax1.contourf(kds, kps, np.log10(D))
    cbar = f.colorbar(D_plot, ax=ax1, ticks=D_plot.levels)
    cbar.ax.set_yticklabels(np.round((10 ** D_plot.levels), 2))
    ax1.set_title('Exp Damping')
    ax1.set_ylabel('Kp')
    ax1.set_xlabel('Kd')

    D_plot2 = ax2.contourf(kds, kps, np.log10(D_theory))
    cbar = f.colorbar(D_plot2, ax=ax2, ticks=D_plot2.levels)
    cbar.ax.set_yticklabels(np.round((10 ** D_plot2.levels), 2))
    ax2.set_title('Theoretical Damping * sqrt(m)')
    ax2.set_ylabel('Kp')
    ax2.set_xlabel('Kd')

    f.savefig('plots/Damping_Estimation.png',dpi=400)


# Plot the experimentally derived characteristics
# of the different step function as contour map
# Verified
def plot_step_characteristics(param_dict):
    kps = [0.025, 0.05, 0.1, 0.2, 0.3]
    kds = [0, 0.001, 0.002, 0.004, 0.008]
    I_max_index = 0

    Tr = param_dict['Tr'][I_max_index]
    Ts = param_dict['Ts'][I_max_index]
    Mp = param_dict['Mp'][I_max_index]
    D = param_dict['Damping'][I_max_index]

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)

    # plot Tr
    tr_plot = ax1.contourf(kds, kps, np.log10(Tr))
    cbar = f.colorbar(tr_plot, ax=ax1, ticks=tr_plot.levels)
    cbar.ax.set_yticklabels(np.round(10 ** tr_plot.levels, 3))
    ax1.set_title('Rise Time')

    # plot Ts
    ts_plot = ax2.contourf(kds, kps, np.log10(Ts))
    cbar = f.colorbar(ts_plot, ax=ax2, ticks=ts_plot.levels)
    cbar.ax.set_yticklabels(np.round(10 ** ts_plot.levels, 3))  # vertically oriented colorbar
    ax2.set_title('Settling Time')

    # plot Mp
    mp_plot = ax3.contourf(kds, kps, np.log10(Mp + 1e-5))
    cbar = f.colorbar(mp_plot, ax=ax3, ticks=mp_plot.levels)
    cbar.ax.set_yticklabels(np.round(10 ** mp_plot.levels, 3))  # vertically oriented colorbar
    ax3.set_title('Max Overshoot')

    # plot damping
    d_plot = ax4.contourf(kds, kps, np.log10(D))
    cbar = f.colorbar(d_plot, ax=ax4, ticks=d_plot.levels)
    cbar.ax.set_yticklabels(np.round(10 ** d_plot.levels, 3))  # vertically oriented colorbar
    ax4.set_title('Damping Coeff.')

    ax3.set_xlabel('Kd')
    ax4.set_xlabel('Kd')
    ax1.set_ylabel('Kp')
    ax3.set_ylabel('Kp')

    f.savefig('plots/Step_Characteristics.png',dpi=400)

    # OLD STYLE OF PLOTTING
    # levels = custom_contour_levels(Mp+0.0001)
    # mp_plot = ax3.contourf(kds, kps, Mp,levels=levels)
    # f.colorbar(mp_plot, ax=ax3,ticks = levels)
    # ax3.set_title('Max Overshoot')


def plot_step_responses(param_dict,kd):
    step_data_15A = param_dict['step_data'][0]
    step_data_30A = param_dict['step_data'][1]

    kps = [0.025, 0.05, 0.1, 0.2, 0.3]
    kds = [0.0005, 0.001, 0.002, 0.004, 0.008]

    kd_index = kds.index(kd)

    f,(ax1,ax2) = plt.subplots(1,2)

    for kp_index in np.arange(step_data_30A.shape[0]):
        ax1.plot(step_data_30A[kp_index, kd_index], linewidth=0.5, label='Kp = {}'.format(kps[kp_index]))
    ax1.set_title('I_max = 30A Kd = {}'.format(kd))
    ax1.legend()

    for kp_index in np.arange(step_data_30A.shape[0]):
        ax2.plot(step_data_15A[kp_index, kd_index], linewidth=0.5, label='Kp = {}'.format(kps[kp_index]))
    ax2.set_title('I_max = 15A Kd = {}'.format(kd))
    ax2.legend()

    f.savefig('plots/Step_Responses.png',dpi=400)


def plot_pid_response(encoder, kp, kd, dt, I_max=1, input='V1_step_routine', plot_output=True):
    position_sub = encoder
    vel_sub = np.diff(encoder) / dt
    vel_sub = np.insert(vel_sub, 0, 0)

    if input == 'V1_step_routine':
        set_point = np.zeros_like(position_sub)
        delay = 4  # number of samples the input is delayed by
        set_point[0:250 + delay] = 180.0
        set_point[250 + delay:500 + delay] = 90.0
        set_point[500 + delay:] = 180.0
    else:
        set_point = 180.0

    p_term = (position_sub - set_point) * kp
    d_term = vel_sub * kd
    output = np.clip(p_term + d_term, -1, 1)

    # Multiply by max current, default is one, which plots just PID output
    p_term *= I_max
    d_term *= I_max
    output *= I_max

    plt.figure()
    plt.plot(p_term, label='p_term')
    plt.plot(d_term, label='d_term')
    if plot_output:
        plt.plot(output, label='output')
    plt.legend()
    plt.show()


def plot_fft_of_ss(data):
    ### Analyze vibrations at a set point ###


    pos_slice = data
    # plt.figure()
    # plt.plot(pos_slice)
    fft_pos = np.abs(fft(pos_slice))
    fft_pos = fft_pos[1:int(fft_pos.size / 2)]

    plt.figure()
    plt.plot(np.arange(fft_pos.size), fft_pos)


def theoretical_specs(m, b, k):
    print('m: {} b: {} k: {}'.format(m, b, k))

    w_natural = np.sqrt(k / m)
    damping_k = b / (2 * np.sqrt(k * m))
    print('w_natural: {}'.format(w_natural))
    print('damping: {}'.format(damping_k))

    tr = 1.8 / w_natural
    print('Tr: {}'.format(tr))

    ts = -np.log(0.01) / (w_natural * damping_k)
    print('Ts: {}'.format(ts))

    Mp = np.exp(-damping_k * np.pi / (np.sqrt(1 - damping_k ** 2)))
    print('Mp: {}'.format(Mp))


# TODO: load kds and kps into param dictionary when it goes through the files
# TODO: remove kps and kds references in the plot functions

param_dict = load_spec_dict('data/time_angle_')
plot_step_characteristics(param_dict)
plot_m_estimates(param_dict)
plot_exp_damping_vs_theoretical_damping(param_dict)
plot_step_responses(param_dict,kd=0.004)

# The plot functions don't call show() themselves
# so call show() at the end of the program manually
plt.show()


# (position, times) = load_position_time_vectors('data/time_angle_I15p300d2.npy')
# plot_pid_response(position, 0.2, 0.002, 0.002, I_max=15, plot_output=True)

# Iz = 2.1e-5;
# Kt = 0.028;
# I_max = 15
# m = Iz / (Kt * I_max);
# b = 0.001;
# k = 0.1

# theoretical_specs(m, b, k)
