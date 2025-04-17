
import numpy as np
import os
import sys
import datetime
import re
from matplotlib import pyplot as plt
import h5py
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.cluster import KMeans


def exponential(x, a, b, c, d):
    return a * np.exp(-(x - b) / c) + d

def optimal_bins(data):
    n = len(data)
    if n == 0:
        return {}
    # Sturges' Rule
    sturges_bins = int(np.ceil(np.log2(n) + 1))
    return sturges_bins

def t1_fit(I, Q, delay_times):
    signal = 'None'
    if 'I' in signal:
        signal = I
        plot_sig = 'I'
    elif 'Q' in signal:
        signal = Q
        plot_sig = 'Q'
    else:
        if abs(I[-1] - I[0]) > abs(Q[-1] - Q[0]):
            signal = I
            plot_sig = 'I'
        else:
            signal = Q
            plot_sig = 'Q'

        # Initial guess for parameters
    q1_a_guess = np.max(signal) - np.min(signal)  # Initial guess for amplitude (a)
    q1_b_guess = 0  # Initial guess for time shift (b)
    q1_c_guess = (delay_times[-1] - delay_times[0]) / 5  # Initial guess for decay constant (T1)
    q1_d_guess = np.min(signal)  # Initial guess for baseline (d)

        # Form the guess array
    q1_guess = [q1_a_guess, q1_b_guess, q1_c_guess, q1_d_guess]

        # Define bounds to constrain T1 (c) to be positive, but allow amplitude (a) to be negative
    lower_bounds = [-np.inf, -np.inf, 0, -np.inf]  # Amplitude (a) can be negative/positive, but T1 (c) > 0
    upper_bounds = [np.inf, np.inf, np.inf, np.inf]  # No upper bound on parameters

        # Perform the fit using the 'trf' method with bounds
    q1_popt, q1_pcov = curve_fit(exponential, delay_times, signal,
                                     p0=q1_guess, bounds=(lower_bounds, upper_bounds),
                                     method='trf', maxfev=10000)

        # Generate the fitted exponential curve
    q1_fit_exponential = exponential(delay_times, *q1_popt)

        # Extract T1 and its error
    T1_est = q1_popt[2]  # Decay constant T1
    T1_err = np.sqrt(q1_pcov[2][2]) if q1_pcov[2][2] >= 0 else float('inf')  # Ensure error is valid

    return q1_fit_exponential, T1_err, T1_est, plot_sig

def qspec_plot_results(I, Q, freqs, plot_fit=False, config=None):
    freqs = np.array(freqs)
    freq_q = freqs[np.argmax(I)]

    mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err = fit_lorenzian(I, Q,
                                                                                                               freqs,
                                                                                                              freq_q)
    # Check if the returned values are all None
    # if (mean_I is None and mean_Q is None and I_fit is None and Q_fit is None
    #         and largest_amp_curve_mean is None and largest_amp_curve_fwhm is None):
    #     # If so, return None for the values in this definition as well
    #     return None, None, None

    # If we get here, the fit was successful and we can proceed with plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.rcParams.update({'font.size': 18})

    # I subplot
    ax1.plot(freqs, I, label='I', linewidth=2)
    ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend()

    # Q subplot
    ax2.plot(freqs, Q, label='Q', linewidth=2)
    ax2.set_xlabel("Qubit Frequency (MHz)", fontsize=20)
    ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.legend()

    # Plot the fits
    if plot_fit:
        ax1.plot(freqs, I_fit, 'r--', label='Lorentzian Fit')
        ax1.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

        ax2.plot(freqs, Q_fit, 'r--', label='Lorentzian Fit')
        ax2.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

    # Calculate the middle of the plot area
    plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

    if plot_fit:
        # Add title, centered on the plot area
        if config is not None:  # then its been passed to this definition, so use that
            fig.text(plot_middle, 0.98,
                     f"Qubit Spectroscopy Q{QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                     f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                     f", {config['reps']}*{config['rounds']} avgs",
                     fontsize=24, ha='center', va='top')
        else:
            fig.text(plot_middle, 0.98,
                     f"Qubit Spectroscopy Q{QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                     f" FWHM: {round(largest_amp_curve_fwhm, 1)}",
                     fontsize=24, ha='center', va='top')
    else:
        # Add title, centered on the plot area
        if config is not None:  # then its been passed to this definition, so use that
            fig.text(plot_middle, 0.98,
                     f"Qubit Spectroscopy Q{QubitIndex + 1}",
                     fontsize=24, ha='center', va='top')
        else:
            fig.text(plot_middle, 0.98,
                     f"Qubit Spectroscopy Q{QubitIndex + 1}",
                     fontsize=24, ha='center', va='top')

            # Adjust spacing
    plt.tight_layout()

    # Adjust the top margin to make room for the title
    plt.subplots_adjust(top=0.93)
    plt.show()

    return largest_amp_curve_mean, I_fit, Q_fit

def get_results(I, Q, freqs):
    freqs = np.array(freqs)
    freq_q = freqs[np.argmax(I)]

    mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err = fit_lorenzian(I,
                                                                                                                     Q,
                                                                                                                     freqs,
                                                                                                                     freq_q)
    return largest_amp_curve_mean, I_fit, Q_fit, qspec_fit_err

def lorentzian(f, f0, gamma, A, B):
    return A * gamma ** 2 / ((f - f0) ** 2 + gamma ** 2) + B

def max_offset_difference_with_x(x_values, y_values, offset):
    max_average_difference = -1
    corresponding_x = None

    # average all 3 to avoid noise spikes
    for i in range(len(y_values) - 2):
        # group 3 vals
        y_triplet = y_values[i:i + 3]

        # avg differences for these 3 vals
        average_difference = sum(abs(y - offset) for y in y_triplet) / 3

        # see if this is the highest difference yet
        if average_difference > max_average_difference:
            max_average_difference = average_difference
            # x value for the middle y value in the 3 vals
            corresponding_x = x_values[i + 1]

    return corresponding_x, max_average_difference

def fit_lorenzian(I, Q, freqs, freq_q):
    signal = 'None'
    try:
        # Initial guesses for I and Q
        initial_guess_I = [freq_q, 1, np.max(I), np.min(I)]
        initial_guess_Q = [freq_q, 1, np.max(Q), np.min(Q)]


        # First round of fits (to get rough estimates)
        params_I, _ = curve_fit(lorentzian, freqs, I, p0=initial_guess_I)
        params_Q, _ = curve_fit(lorentzian, freqs, Q, p0=initial_guess_Q)

        # Use these fits to refine guesses
        x_max_diff_I, max_diff_I = max_offset_difference_with_x(freqs, I, params_I[3])
        x_max_diff_Q, max_diff_Q = max_offset_difference_with_x(freqs, Q, params_Q[3])
        initial_guess_I = [x_max_diff_I, 1, np.max(I), np.min(I)]
        initial_guess_Q = [x_max_diff_Q, 1, np.max(Q), np.min(Q)]

        # Second (refined) round of fits, this time capturing the covariance matrices
        params_I, cov_I = curve_fit(lorentzian, freqs, I, p0=initial_guess_I)
        params_Q, cov_Q = curve_fit(lorentzian, freqs, Q, p0=initial_guess_Q)

        # Create the fitted curves
        I_fit = lorentzian(freqs, *params_I)
        Q_fit = lorentzian(freqs, *params_Q)

        # Calculate errors from the covariance matrices
        fit_err_I = np.sqrt(np.diag(cov_I))
        fit_err_Q = np.sqrt(np.diag(cov_Q))

        # Extract fitted means and FWHM (assuming params[0] is the mean and params[1] relates to the width)
        mean_I = params_I[0]
        mean_Q = params_Q[0]
        fwhm_I = 2 * params_I[1]
        fwhm_Q = 2 * params_Q[1]


        # Calculate the amplitude differences from the fitted curves
        amp_I_fit = abs(np.max(I_fit) - np.min(I_fit))
        amp_Q_fit = abs(np.max(Q_fit) - np.min(Q_fit))

        # Choose which curve to use based on the input signal indicator
        if 'None' in signal:
            if amp_I_fit > amp_Q_fit:
                largest_amp_curve_mean = mean_I
                largest_amp_curve_fwhm = fwhm_I
                # error on the Q fit's center frequency (first parameter):
                qspec_fit_err = fit_err_I[0]
            else:
                largest_amp_curve_mean = mean_Q
                largest_amp_curve_fwhm = fwhm_Q
                qspec_fit_err = fit_err_Q[0]
        elif 'I' in signal:
            largest_amp_curve_mean = mean_I
            largest_amp_curve_fwhm = fwhm_I
            qspec_fit_err = fit_err_I[0]
        elif 'Q' in signal:
            largest_amp_curve_mean = mean_Q
            largest_amp_curve_fwhm = fwhm_Q
            qspec_fit_err = fit_err_Q[0]
        else:
            print('Invalid signal passed, please choose "I", "Q", or "None".')
            return None

        # Return all desired results including the error on the Q fit
        return mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err

    except Exception as e:
        return None, None, None, None, None, None, None

def create_folder_if_not_exists(folder_path):
    import os
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def load_from_h5(filename, data_type, save_r=1):  # Added save_r as parameter.

    data = {data_type: {}}  # Initialize the main dictionary with the data_type.

    with h5py.File(filename, 'r') as f:
        for qubit_group in f.keys():
            qubit_index = int(qubit_group[1:]) - 1
            qubit_data = {}
            group = f[qubit_group]

            for dataset_name in group.keys():
                # Attempt to map HDF5 keys to the target dictionaries' keys.
                if data_type == 'Res':
                    target_keys = {'Dates': 'Dates', 'freq_pts': 'freq_pts', 'freq_center': 'freq_center',
                                       'Amps': 'Amps', 'Found Freqs': 'Found Freqs', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                elif data_type == 'QSpec':
                    target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Frequencies': 'Frequencies',
                                       'I Fit': 'I Fit', 'Q Fit': 'Q Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                elif data_type == 'Rabi':
                    target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Gains': 'Gains', 'Fit': 'Fit',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num'}
                elif data_type == 'SS':
                    target_keys = {'Fidelity': 'Fidelity', 'Angle': 'Angle', 'Dates': 'Dates', 'I_g': 'I_g',
                                       'Q_g': 'Q_g', 'I_e': 'I_e', 'Q_e': 'Q_e',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num'}
                elif data_type == 'T1':
                    target_keys = {'T1': 'T1', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                elif data_type == 'T2':
                    target_keys = {'T2': 'T2', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                elif data_type == 'T2E':
                    target_keys = {'T2E': 'T2E', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                elif data_type == 'stark2D':
                    target_keys = {'Dates': 'Dates', 'I':'I', 'Q': 'Q', 'Qu Frequency Sweep':'Qu Frequency Sweep',
                                   'Res Gain Sweep':'Res Gain Sweep','Round Num':'Round Num', 'Batch Num': 'Batch Num',
                                   'Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type =='starkSpec':
                    target_keys = {'Dates': 'Dates', 'I':'I', 'Q': 'Q','P': 'P', 'shots':'shots','Gain Sweep':'Gain Sweep','Round Num':'Round Num', 'Batch Num': 'Batch Num',
                                   'Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}

                else:
                        raise ValueError(f"Unsupported data_type: {data_type}")

                try:
                    mapped_key = target_keys[dataset_name]  # Map HDF5 key to target key.
                    qubit_data[mapped_key] = [group[dataset_name][()]] * save_r  # Expand to match the desired length.

                except KeyError:
                    print(f"Warning: Key '{dataset_name}' not found in target dictionary for data_type '{data_type}'. Skipping.")
                    pass

            data[data_type][qubit_index] = qubit_data

    return data

def process_string_of_nested_lists(data):
    # Remove extra whitespace and non-numeric characters.
    data = re.sub(r'\s*\[(\s*.*?\s*)\]\s*', r'[\1]', data)
    data = data.replace('[ ', '[')
    data = data.replace('[ ', '[')
    data = data.replace('[ ', '[')
    cleaned_data = ''.join(c for c in data if c.isdigit() or c in ['-', '.', ' ', 'e', '[', ']'])
    pattern = r'\[(.*?)\]'  # Regular expression to match data within brackets
    matches = re.findall(pattern, cleaned_data)
    result = []
    for match in matches:
        numbers = [float(x.strip('[').strip(']').replace("'", "").replace(" ", "").replace("  ", "")) for x in
                    match.split()]  # Convert strings to integers
    result.append(numbers)

    return result

def process_h5_data(data):
    # Check if the data is a byte string; decode if necessary.
    if isinstance(data, bytes):
        data_str = data.decode()
    elif isinstance(data, str):
        data_str = data
    else:
        raise ValueError("Unsupported data type. Data should be bytes or string.")

    # Remove extra whitespace and non-numeric characters.
    cleaned_data = ''.join(c for c in data_str if c.isdigit() or c in ['-', '.', ' ', 'e'])

    # Split into individual numbers, removing empty strings.
    numbers = [float(x) for x in cleaned_data.split() if x]
    return numbers

data_dir = "/data/QICK_data/run6/6transmon/Round_Robin_Benchmark_with_stark/Data/2025-04-10/Data_h5/"
QubitIndex = 4
h5_index = 86
fig, ax = plt.subplots(3,1)
power2shift = -25
idx_start = 57

#### qspec data #####
qspec_times = []
qspec_I = []
data_path = os.path.join(data_dir, "QSpec_ge")
h5_files = os.listdir(data_path)
h5_files.sort()


load_data = load_from_h5(os.path.join(data_path, h5_files[idx_start]), 'QSpec', save_r=1)
print(load_data)
start_time = datetime.datetime.fromtimestamp(load_data['QSpec'][QubitIndex].get('Dates', [])[0][0])
qspec_freqs = process_h5_data(load_data['QSpec'][QubitIndex].get('Frequencies', [])[0][0].decode())
qu_freqs = []
qu_freq_errs = []

idx = 0
for h5_file in h5_files[idx_start:h5_index]:
    load_data = load_from_h5(os.path.join(data_path, h5_file), 'QSpec', save_r=1)
    date = datetime.datetime.fromtimestamp(load_data['QSpec'][QubitIndex].get('Dates', [])[0][0])
    #qspec_times.append(date)
    qspec_times.append((date - start_time).total_seconds())
    I = process_h5_data(load_data['QSpec'][QubitIndex].get('I', [])[0][0].decode())
    Q = process_h5_data(load_data['QSpec'][QubitIndex].get('Q', [])[0][0].decode())
    qspec_I.append(I)
    #qspec_plot_results(I, Q, qspec_freqs, plot_fit=True)
    largest_amp_curve_mean, I_fit, Q_fit, qspec_fit_err = get_results(I, Q, qspec_freqs)
    qu_freqs.append(largest_amp_curve_mean)
    qu_freq_errs.append(qspec_fit_err)
    idx += 1

ax[0].scatter(qspec_times, qu_freqs)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Qu Freq (MHz)')
ax[0].set_title('Low Gain Qubit Spec')

#### high gain qspec data #####
hgqspec_times = []
hgqspec_I = []
data_path = os.path.join(data_dir, "high_gain_QSpec_ge")
h5_files = os.listdir(data_path)
h5_files.sort()

idx = 0

load_data = load_from_h5(os.path.join(data_path, h5_files[idx_start]), 'QSpec', save_r=1)
hgqspec_freqs = process_h5_data(load_data['QSpec'][QubitIndex].get('Frequencies', [])[0][0].decode())

for h5_file in h5_files[idx_start:h5_index]:
    load_data = load_from_h5(os.path.join(data_path, h5_file), 'QSpec', save_r=1)
    date = datetime.datetime.fromtimestamp(load_data['QSpec'][QubitIndex].get('Dates', [])[0][0])
    hgqspec_times.append((date - start_time).total_seconds())
    I = process_h5_data(load_data['QSpec'][QubitIndex].get('I', [])[0][0].decode())
    Q = process_h5_data(load_data['QSpec'][QubitIndex].get('Q', [])[0][0].decode())
    hgqspec_I.append(I)
    idx += 1

fig2,ax2 = plt.subplots(2,1)
plot = ax2[0]
cbar = plt.colorbar(plot.pcolormesh(hgqspec_times, hgqspec_freqs, np.transpose(hgqspec_I), shading="nearest", cmap="viridis"), ax=plot)
cbar.set_label('I (a.u.)')
plot.set_xlabel('Time (s)')
plot.set_ylabel('Qu probe freq (MHz)')
plot.set_title('High Gain Qubit Spec')

#### medium gain qspec data #####
mgqspec_times = []
mgqspec_I = []
data_path = os.path.join(data_dir, "med_gain_QSpec_ge")
h5_files = os.listdir(data_path)
h5_files.sort()

idx = 0

load_data = load_from_h5(os.path.join(data_path, h5_files[idx_start]), 'QSpec', save_r=1)
mgqspec_freqs = process_h5_data(load_data['QSpec'][QubitIndex].get('Frequencies', [])[0][0].decode())

for h5_file in h5_files[idx_start:h5_index]:
    load_data = load_from_h5(os.path.join(data_path, h5_file), 'QSpec', save_r=1)
    date = datetime.datetime.fromtimestamp(load_data['QSpec'][QubitIndex].get('Dates', [])[0][0])
    mgqspec_times.append((date - start_time).total_seconds())
    I = process_h5_data(load_data['QSpec'][QubitIndex].get('I', [])[0][0].decode())
    Q = process_h5_data(load_data['QSpec'][QubitIndex].get('Q', [])[0][0].decode())
    mgqspec_I.append(I)
    idx += 1

plot = ax2[1]
cbar = plt.colorbar(plot.pcolormesh(mgqspec_times, mgqspec_freqs, np.transpose(mgqspec_I), shading="nearest", cmap="viridis"), ax=plot)
cbar.set_label('I (a.u.)')
plot.set_xlabel('Time (s)')
plot.set_ylabel('Qu probe freq (MHz)')
plot.set_title('Medium Gain Qubit Spec')
plt.tight_layout()

#### SSF data #####
ss_times = []
ss_fid = []
data_path = os.path.join(data_dir, "SS_ge")
h5_files = os.listdir(data_path)
h5_files.sort()

idx = 0

load_data = load_from_h5(os.path.join(data_path, h5_files[idx_start]), 'SS', save_r=1)

for h5_file in h5_files[idx_start:h5_index]:
    load_data = load_from_h5(os.path.join(data_path, h5_file), 'SS', save_r=1)
    date = datetime.datetime.fromtimestamp(load_data['SS'][QubitIndex].get('Dates', [])[0][0])
    ss_times.append((date - start_time).total_seconds())
    ss_fid.append(load_data['SS'][QubitIndex].get('Fidelity', [])[0][0])
    idx += 1


plot = ax[1]
plot.scatter(ss_times, np.array(ss_fid) * 100)
plot.set_xlabel('Time (s)')
plot.set_ylabel('SSF (%)')
plot.set_title('Single Shot')

#### T1 #########
t1_times = []
data_path = os.path.join(data_dir, "T1_ge")
h5_files = os.listdir(data_path)
h5_files.sort()

load_data = load_from_h5(os.path.join(data_path, h5_files[idx_start]), 'T1', save_r=1)
print(load_data)
delay_times = process_h5_data(load_data['T1'][QubitIndex].get('Delay Times', [])[0][0].decode())
t1s = []
t1_errs = []

idx = 0
for h5_file in h5_files[idx_start:h5_index]:
    print(h5_file)
    load_data = load_from_h5(os.path.join(data_path, h5_file), 'T1', save_r=1)
    date = datetime.datetime.fromtimestamp(load_data['T1'][QubitIndex].get('Dates', [])[0][0])
    t1_times.append((date - start_time).total_seconds())

    I = process_h5_data(load_data['T1'][QubitIndex].get('I', [])[0][0].decode())
    Q = process_h5_data(load_data['T1'][QubitIndex].get('Q', [])[0][0].decode())

    q1_fit_exponential, T1_err, T1_est, plot_sig = t1_fit(I, Q, delay_times)
    t1s.append(T1_est)
    t1_errs.append(T1_err)
    idx += 1

plot = ax[2]
plot.scatter(t1_times, t1s)
plot.set_xlabel('Time (s)')
plot.set_ylabel('T1 (us)')
plot.set_ylim((30,65))
plt.tight_layout()
plt.show()

##### stark shift #########
thresh = -200
theta = 0/180 * np.pi
starkspec_times = []
qspec_freqs = []
data_path = os.path.join(data_dir, "starkSpec")
h5_files = os.listdir(data_path)
h5_files.sort()

I = []
Q = []
P = []

idx = 0

load_data = load_from_h5(os.path.join(data_path, h5_files[idx_start]), 'starkSpec', save_r=1)
gain_sweep = process_h5_data(load_data['starkSpec'][QubitIndex].get('Gain Sweep', [])[0][0].decode())
gain_pts = len(gain_sweep)
freq_sweep = np.concatenate((np.square(np.linspace(-1.0,0, num=125)) * -11, np.square(np.linspace(0,1.0, num=125)) * 8))
#print(freq_sweep)
reps = int(np.shape(process_h5_data(load_data['starkSpec'][QubitIndex].get('I', [])[0][0].decode()))[0]/gain_pts)

print(np.shape(h5_files))
for h5_file in h5_files[idx_start:h5_index]:
    load_data = load_from_h5(os.path.join(data_path, h5_file), 'starkSpec', save_r=1)
    date = datetime.datetime.fromtimestamp(load_data['starkSpec'][QubitIndex].get('Dates', [])[0][0])
    starkspec_times.append((date - start_time).total_seconds())

    I.append(np.array(process_h5_data(load_data['starkSpec'][QubitIndex].get('I', [])[0][0].decode())).reshape([gain_pts, reps]))
    Q.append(np.array(process_h5_data(load_data['starkSpec'][QubitIndex].get('Q', [])[0][0].decode())).reshape([gain_pts, reps]))
    P.append(process_h5_data(load_data['starkSpec'][QubitIndex].get('P', [])[0][0]))

I = np.array(I)
Q = np.array(Q)
P = np.array(P)
print(np.shape(I))
i_new = I[0][0][:] * np.cos(theta) - Q[0][0][:] * np.sin(theta)
q_new = I[0][0][:] * np.sin(theta) + Q[0][0][:] * np.cos(theta)
#kmeans = KMeans(n_clusters=3).fit(np.transpose([i_new, q_new]))

fig2, axes = plt.subplots(2,3)
time_idx = 22
j = 0
m = 0
for i in (np.arange(0,6) * round((gain_pts-5)/5)):
    plot = axes[j][m]
    i_new = I[time_idx][i][:] * np.cos(theta) - Q[time_idx][i][:] * np.sin(theta)
    q_new = I[time_idx][i][:] * np.sin(theta) + Q[time_idx][i][:] * np.cos(theta)
        #kmeans = KMeans(n_clusters=3).fit(np.transpose([i_new, q_new]))
#    plot.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], c='k')
        #idx = kmeans.predict(np.transpose([i_new, q_new]))
    idx = (i_new > thresh)
    plot.scatter(i_new, q_new, c=idx)
    plot.set_xlabel("I")
    plot.set_ylabel("Q")
    plot.set_title(f"freq = {np.round(freq_sweep[i],3)} MHz, t= {np.round(starkspec_times[time_idx])} s")
    m+=1
    if m == 3:
        m=0
        j=1


p_new = np.zeros([h5_index - idx_start, gain_pts])
print(np.shape(I))
print(np.shape(Q))
for j in np.arange(idx_start, h5_index):
    for i in np.arange(0, gain_pts):
        i_new = I[j-idx_start][i][:] * np.cos(theta) - Q[j-idx_start][i][:] * np.sin(theta)
        q_new = I[j-idx_start][i][:] * np.sin(theta) + Q[j-idx_start][i][:] * np.cos(theta)
        idx_post_process = (i_new > thresh)
        p_new[j-idx_start][i] = np.sum(np.array(idx_post_process) == 1)/len(idx_post_process)

fig5, ax5 = plt.subplots(2,1)
plot = ax5[0]
cbar = plt.colorbar(plot.pcolormesh(starkspec_times, freq_sweep, np.transpose(p_new), shading="nearest", cmap="viridis"), ax=plot)
cbar.set_label("P(MS=1)")
plot.set_xlabel('time (s)')
plot.set_ylabel('AC Stark shift [MHz]')
plot.set_title(f'Qubit {QubitIndex + 1}')

##### res stark shift #########

resstarkspec_times = []
resqspec_freqs = []
data_path = os.path.join(data_dir, "res_starkSpec")
h5_files = os.listdir(data_path)
h5_files.sort()

resI = []
resQ = []
resP = []

idx = 0

load_data = load_from_h5(os.path.join(data_path, h5_files[idx_start]), 'starkSpec', save_r=1)
res_gain_sweep = process_h5_data(load_data['starkSpec'][QubitIndex].get('Gain Sweep', [])[0][0].decode())
res_gain_pts = len(res_gain_sweep)
reps = int(np.shape(process_h5_data(load_data['starkSpec'][QubitIndex].get('I', [])[0][0].decode()))[0]/gain_pts)

for h5_file in h5_files[idx_start:h5_index]:
    load_data = load_from_h5(os.path.join(data_path, h5_file), 'starkSpec', save_r=1)
    date = datetime.datetime.fromtimestamp(load_data['starkSpec'][QubitIndex].get('Dates', [])[0][0])
    resstarkspec_times.append((date - start_time).total_seconds())

    resI.append(np.array(process_h5_data(load_data['starkSpec'][QubitIndex].get('I', [])[0][0].decode())).reshape([gain_pts, reps]))
    resQ.append(np.array(process_h5_data(load_data['starkSpec'][QubitIndex].get('Q', [])[0][0].decode())).reshape([gain_pts, reps]))
    resP.append(process_h5_data(load_data['starkSpec'][QubitIndex].get('P', [])[0][0]))

resI = np.array(resI)
resQ = np.array(resQ)
resP = np.array(resP)
print(np.shape(I))
i_new = resI[0][0][:] * np.cos(theta) - resQ[0][0][:] * np.sin(theta)
q_new = resI[0][0][:] * np.sin(theta) + resQ[0][0][:] * np.cos(theta)
#kmeans = KMeans(n_clusters=3).fit(np.transpose([i_new, q_new]))

fig4, axes = plt.subplots(2,3)
time_idx = 0
j = 0
m = 0
for i in (np.arange(0,6) * round((gain_pts-5)/5)):
    plot = axes[j][m]
    plot.set_box_aspect(1)
    i_new = resI[time_idx][i][:] * np.cos(theta) - resQ[time_idx][i][:] * np.sin(theta)
    q_new = resI[time_idx][i][:] * np.sin(theta) + resQ[time_idx][i][:] * np.cos(theta)
        #kmeans = KMeans(n_clusters=3).fit(np.transpose([i_new, q_new]))
#    plot.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], c='k')
        #idx = kmeans.predict(np.transpose([i_new, q_new]))
    idx = (i_new > thresh)
    plot.scatter(i_new, q_new, c=idx)
    plot.set_xlabel("I")
    plot.set_ylabel("Q")
    plot.set_title(f"freq = {np.round(gain_sweep[i]**2 * power2shift,3)} MHz, t= {np.round(starkspec_times[time_idx])} s")
    m+=1
    if m == 3:
        m=0
        j=1

plt.tight_layout()

resp_new = np.zeros([h5_index - idx_start, res_gain_pts])
for j in np.arange(idx_start, h5_index):
    for i in np.arange(0, res_gain_pts):
        i_new = resI[j-idx_start][i][:] * np.cos(theta) - resQ[j-idx_start][i][:] * np.sin(theta)
        q_new = resI[j-idx_start][i][:] * np.sin(theta) + resQ[j-idx_start][i][:] * np.cos(theta)
        idx_post_process = (i_new > thresh)
        resp_new[j-idx_start][i] = np.sum(np.array(idx_post_process) == 1)/len(idx_post_process)


plot = ax5[1]
cbar = plt.colorbar(plot.pcolormesh(resstarkspec_times, (np.array(res_gain_sweep)**2) * power2shift , np.transpose(resp_new), shading="nearest", cmap="viridis"), ax=plot)
cbar.set_label("P(MS=1)")
plot.set_xlabel('time (s)')
plot.set_ylabel('AC Stark shift [MHz]')
plot.set_title(f'Resonator Stark: Qubit {QubitIndex + 1}')
plt.tight_layout()
plt.show()






check = [21, 22, 23]
fig3, ax3 = plt.subplots(2,len(check))
for j in np.arange(0, len(check)):
    check_idx = check[j]

    plot = ax3[0][j]
    plot.plot(hgqspec_freqs, hgqspec_I[:][check_idx], label="gain=1.0")
    plot.plot(mgqspec_freqs, mgqspec_I[:][check_idx], label="gain=0.5")
    plot.plot(mgqspec_freqs, qspec_I[:][check_idx], label="gain=0.15")
    plot.set_xlabel('qubit probe frequency (MHz)')
    plot.set_ylabel('I (a.u.)')
    plot.set_title(f'Qspec at t={np.round(hgqspec_times[check_idx])} s')
    plot.legend()
    #plot.set_xlim((4471-25, 4471))

    plot = ax3[1][j]
    plot.plot(np.array(freq_sweep), p_new[check_idx][:])
    plot.plot(np.array(res_gain_sweep)**2 * power2shift, resp_new[check_idx][:])
    plot.set_xlabel('AC Stark Shift (MHz)')
    plot.set_ylabel('P(MS=1)')
    plot.set_title(f'Stark Shift Spec at t={np.round(starkspec_times[check_idx])} s')

plt.tight_layout()
plt.show()

