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

################################ functions for reading in h5 files ###################################

def create_folder_if_not_exists(folder_path):
    import os
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def datetime_to_unix(self, dt):
    # Convert to Unix timestamp
    unix_timestamp = int(dt.timestamp())
    return unix_timestamp

def unix_to_datetime(self, unix_timestamp):
    # Convert the Unix timestamp to a datetime object
    dt = datetime.fromtimestamp(unix_timestamp)
    return dt

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
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'QSpec':
                    target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Frequencies': 'Frequencies',
                                       'I Fit': 'I Fit', 'Q Fit': 'Q Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'Rabi':
                    target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Gains': 'Gains', 'Fit': 'Fit',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'SS':
                    target_keys = {'Fidelity': 'Fidelity', 'Angle': 'Angle', 'Dates': 'Dates', 'I_g': 'I_g',
                                       'Q_g': 'Q_g', 'I_e': 'I_e', 'Q_e': 'Q_e',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'offset':
                    target_keys = {'Res Frequency':'Res Frequency','Fidelity': 'Fidelity', 'Angle': 'Angle', 'Dates': 'Dates', 'I_g': 'I_g',
                                       'Q_g': 'Q_g', 'I_e': 'I_e', 'Q_e': 'Q_e',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}

                elif data_type == 'T1':
                    target_keys = {'T1': 'T1', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'T2':
                    target_keys = {'T2': 'T2', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'T2E':
                    target_keys = {'T2E': 'T2E', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'stark2D':
                    target_keys = {'Dates': 'Dates', 'I':'I', 'Q': 'Q', 'Qu Frequency Sweep':'Qu Frequency Sweep',
                                   'Res Gain Sweep':'Res Gain Sweep','Round Num':'Round Num', 'Batch Num': 'Batch Num',
                                   'Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type =='starkSpec':
                    target_keys = {'Dates': 'Dates', 'I':'I', 'Q': 'Q','P': 'P', 'shots':'shots','Gain Sweep':'Gain Sweep','Round Num':'Round Num', 'Batch Num': 'Batch Num',
                                   'Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}

                elif data_type == 'Qtemps':
                    target_keys = {'Dates': 'Dates', 'Qfreq_ge': 'Qfreq_ge',
                                   'I1': 'I1', 'Q1': 'Q1', 'Gains1': 'Gains1', 'Fit1': 'Fit1',
                                   'I2': 'I2', 'Q2': 'Q2', 'Gains2': 'Gains2', 'Fit2': 'Fit2',
                                   'Round Num': 'Round Num', 'Batch Num': 'Batch Num', 'Exp Config': 'Exp Config',
                                   'Syst Config': 'Syst Config'}

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

################################## functions for res spec data #######################################

def get_rspec_data(data_dir, QubitIndex, round_num = 0, expt_name = "res_ge"):
    rspec_freqs = []
    rspec_dates = []

    data_path = os.path.join(data_dir, expt_name)
    h5_files = os.listdir(data_path)
    h5_files.sort()
    h5_files_qubit_index = []

    # select only h5 files which have data for the chosen qubit
    for h5_file in h5_files:
        load_data = load_from_h5(os.path.join(data_path, h5_file), 'Res', save_r=1)
        if not np.isnan(load_data['Res'][QubitIndex].get('Dates', [])[0]).any():
            h5_files_qubit_index.append(h5_file)

    num_rounds = len(h5_files_qubit_index)
    if expt_name == 'res_ef':
        round_num = num_rounds-1 #always plot last round for res ef data

    load_data = load_from_h5(os.path.join(data_path, h5_files_qubit_index[0]), 'Res', save_r=1)
    rspec_probe_freqs = process_h5_data(load_data['Res'][QubitIndex].get('freq_pts', [])[0][0].decode())

    idx = 0
    for h5_file in h5_files_qubit_index:
        load_data = load_from_h5(os.path.join(data_path, h5_file), 'Res', save_r=1)
        rspec_dates.append(datetime.datetime.fromtimestamp(load_data['Res'][QubitIndex].get('Dates', [])[0][0]))
        rspec_freqs.append(process_h5_data(load_data['Res'][QubitIndex].get('Found Freqs', [])[0][0].decode())[QubitIndex])
        if round_num == idx:
            rspec_amp0 = process_h5_data(load_data['Res'][QubitIndex].get('Amps', [])[0][0].decode())
            rspec_fit0 = process_h5_data(load_data['Res'][QubitIndex].get('Found Freqs', [])[0][0].decode())
            rspec_freq0 = process_h5_data(load_data['Res'][QubitIndex].get('freq_center', [])[0][0].decode())
        idx +=1

    return rspec_dates, rspec_freqs, rspec_probe_freqs, rspec_amp0, rspec_freq0, rspec_fit0, num_rounds

########################################## functions for QSpec Data ##############################################

def qspec_get_results(I, Q, freqs):
    freqs = np.array(freqs)
    freq_q = freqs[np.argmax(I)]
    max_signal, largest_amp_curve_mean, largest_amp_curve_fwhm, largest_amp_curve_fit, qspec_fit_err = fit_lorenzian(I,Q, freqs, freq_q)
    return max_signal, largest_amp_curve_mean, largest_amp_curve_fwhm, largest_amp_curve_fit, qspec_fit_err

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
                max_signal = 'I'
                largest_amp_curve_mean = mean_I
                largest_amp_curve_fwhm = fwhm_I
                largest_amp_curve_fit = I_fit
                # error on the Q fit's center frequency (first parameter):
                qspec_fit_err = fit_err_I[0]
            else:
                max_signal = 'Q'
                largest_amp_curve_mean = mean_Q
                largest_amp_curve_fwhm = fwhm_Q
                largest_amp_curve_fit = Q_fit
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
        return max_signal, largest_amp_curve_mean, largest_amp_curve_fwhm, largest_amp_curve_fit, qspec_fit_err

    except Exception as e:
        return None, None, None, None, None, None, None

def get_opt_qspec_data(data_dir, QubitIndex, expt_name='qspec_ge'):
    data_path = os.path.join(data_dir, expt_name)
    h5_files = os.listdir(data_path)
    h5_files.sort()
    h5_files_qubit_index = []

    # select only h5 files which have data for the chosen qubit
    for h5_file in h5_files:
        load_data = load_from_h5(os.path.join(data_path, h5_file), 'QSpec', save_r=1)
        if not np.isnan(load_data['QSpec'][QubitIndex].get('Dates', [])[0][0]).any():
            h5_files_qubit_index.append(h5_file)

    load_data = load_from_h5(os.path.join(data_path, h5_files_qubit_index[0]), 'QSpec', save_r=1)
    qspec_probe_freqs = process_h5_data(load_data['QSpec'][QubitIndex].get('Frequencies', [])[0][0].decode())
    qspec_date = datetime.datetime.fromtimestamp(load_data['QSpec'][QubitIndex].get('Dates', [])[0][0])
    I = process_h5_data(load_data['QSpec'][QubitIndex].get('I', [])[0][0].decode())
    Q = process_h5_data(load_data['QSpec'][QubitIndex].get('Q', [])[0][0].decode())
    max_signal, qspec_freq, qspec_fwhm, qspec_fit, qspec_fit_err = qspec_get_results(I, Q, qspec_probe_freqs)


    if max_signal == 'I':
        I_or_Q = I
    else:
        I_or_Q = Q

    return max_signal, I_or_Q, qspec_probe_freqs, qspec_freq, qspec_fwhm, qspec_fit, qspec_fit_err

def get_ext_qspec_data(data_dir, QubitIndex, expt_name="extended_qspec_ge"):
    data_path = os.path.join(data_dir, expt_name)
    h5_files = os.listdir(data_path)
    h5_files.sort()
    h5_files_qubit_index = []

    # select only h5 files which have data for the chosen qubit
    for h5_file in h5_files:
        load_data = load_from_h5(os.path.join(data_path, h5_file), 'QSpec', save_r=1)
        if not np.isnan(load_data['QSpec'][QubitIndex].get('Dates', [])[0][0]).any():
            h5_files_qubit_index.append(h5_file)

    ## get data consistent over all rounds from first file
    load_data = load_from_h5(os.path.join(data_path, h5_files_qubit_index[0]), 'QSpec', save_r=1)
    qspec_probe_freqs = process_h5_data(load_data['QSpec'][QubitIndex].get('Frequencies', [])[0][0].decode())
    qspec_date = datetime.datetime.fromtimestamp(load_data['QSpec'][QubitIndex].get('Dates', [])[0][0])

    I = process_h5_data(load_data['QSpec'][QubitIndex].get('I', [])[0][0].decode())
    Q = process_h5_data(load_data['QSpec'][QubitIndex].get('Q', [])[0][0].decode())
    mag = np.sqrt(np.square(I) + np.square(Q))

    return qspec_probe_freqs, mag, qspec_date

def get_qspec_data(data_dir, QubitIndex, expt_name = 'qspec_ge'):
    qspec_dates = []
    qspec_freqs = []
    qspec_freq_errs = []
    data_path = os.path.join(data_dir, expt_name)
    h5_files = os.listdir(data_path)
    h5_files.sort()

    ## get data consistent over all rounds from first file
    load_data = load_from_h5(os.path.join(data_path, h5_files[0]), 'QSpec', save_r=1)
    qspec_probe_freqs = process_h5_data(load_data['QSpec'][QubitIndex].get('Frequencies', [])[0][0].decode())

    for h5_file in h5_files:
        load_data = load_from_h5(os.path.join(data_path, h5_file), 'QSpec', save_r=1)
        date = datetime.datetime.fromtimestamp(load_data['QSpec'][QubitIndex].get('Dates', [])[0][0])
        qspec_dates.append(date)

        I = process_h5_data(load_data['QSpec'][QubitIndex].get('I', [])[0][0].decode())
        Q = process_h5_data(load_data['QSpec'][QubitIndex].get('Q', [])[0][0].decode())

        max_signal, largest_amp_curve_mean, I_fit, Q_fit, qspec_fit_err = qspec_get_results(I, Q, qspec_probe_freqs)
        qspec_freqs.append(largest_amp_curve_mean)
        qspec_freq_errs.append(qspec_fit_err)

    return qspec_dates, qspec_freqs, qspec_freq_errs

def get_qspec_data_at_time_t(data_dir, QubitIndex, time_idx = 0, signal='I', expt_name='qspec_ge'):
    data_path = os.path.join(data_dir, expt_name)
    h5_files = os.listdir(data_path)
    h5_files.sort()

    ## get data consistent over all rounds from first file
    load_data = load_from_h5(os.path.join(data_path, h5_files[time_idx]), 'QSpec', save_r=1)
    qspec_probe_freqs = process_h5_data(load_data['QSpec'][QubitIndex].get('Frequencies', [])[0][0].decode())
    date = datetime.datetime.fromtimestamp(load_data['QSpec'][QubitIndex].get('Dates', [])[0][0])

    I = process_h5_data(load_data['QSpec'][QubitIndex].get('I', [])[0][0].decode())
    Q = process_h5_data(load_data['QSpec'][QubitIndex].get('Q', [])[0][0].decode())

    if signal == 'I':
        I_or_Q = I
    else:
        I_or_Q = Q

    return date, qspec_probe_freqs, I_or_Q

########################################### functions for Amplitude Rabi Data #####################################

def cosine( x, a, b, c, d):
    return a * np.cos(2. * np.pi * b * x - c * 2 * np.pi) + d

def rabi_get_results(I, Q, gains, grab_depths = False, rolling_avg=False):
    signal = 'None'
    gains = np.array(gains)
    q1_a_guess_I = (np.max(I) - np.min(I)) / 2
    q1_d_guess_I = np.mean(I)
    q1_a_guess_Q = (np.max(Q) - np.min(Q)) / 2
    q1_d_guess_Q = np.mean(Q)
    q1_b_guess = 1 / gains[-1]
    q1_c_guess = 0

    q1_guess_I = [q1_a_guess_I, q1_b_guess, q1_c_guess, q1_d_guess_I]
    q1_popt_I, q1_pcov_I = curve_fit(cosine, gains, I, maxfev=100000, p0=q1_guess_I)
    q1_fit_cosine_I = cosine(gains, *q1_popt_I)

    q1_guess_Q = [q1_a_guess_Q, q1_b_guess, q1_c_guess, q1_d_guess_Q]
    q1_popt_Q, q1_pcov_Q = curve_fit(cosine, gains, Q, maxfev=100000, p0=q1_guess_Q)
    q1_fit_cosine_Q = cosine(gains, *q1_popt_Q)

    first_three_avg_I = np.mean(q1_fit_cosine_I[:3])
    last_three_avg_I = np.mean(q1_fit_cosine_I[-3:])
    first_three_avg_Q = np.mean(q1_fit_cosine_Q[:3])
    last_three_avg_Q = np.mean(q1_fit_cosine_Q[-3:])

    best_signal_fit = None
    pi_amp = None
    if 'Q' in signal:
        best_signal_fit = q1_fit_cosine_Q
                # figure out if you should take the min or the max value of the fit to say where pi_amp should be
        if last_three_avg_Q > first_three_avg_Q:
            pi_amp = gains[np.argmax(best_signal_fit)]
        else:
            pi_amp = gains[np.argmin(best_signal_fit)]
    if 'I' in signal:
        best_signal_fit = q1_fit_cosine_I
                # figure out if you should take the min or the max value of the fit to say where pi_amp should be
        if last_three_avg_I > first_three_avg_I:
            pi_amp = gains[np.argmax(best_signal_fit)]
        else:
            pi_amp = gains[np.argmin(best_signal_fit)]
    if 'None' in signal:
                # choose the best signal depending on which has a larger magnitude
        if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
            max_signal = 'Q'
            best_signal_fit = q1_fit_cosine_Q
                    # figure out if you should take the min or the max value of the fit to say where pi_amp should be
            if last_three_avg_Q > first_three_avg_Q:
                pi_amp = gains[np.argmax(best_signal_fit)]
            else:
                pi_amp = gains[np.argmin(best_signal_fit)]
        else:
            max_signal = 'I'
            best_signal_fit = q1_fit_cosine_I
                    # figure out if you should take the min or the max value of the fit to say where pi_amp should be
            if last_three_avg_I > first_three_avg_I:
                pi_amp = gains[np.argmax(best_signal_fit)]
            else:
                pi_amp = gains[np.argmin(best_signal_fit)]
        tot_amp = [np.sqrt((ifit)**2 + (qfit)**2) for ifit,qfit in zip(q1_fit_cosine_I, q1_fit_cosine_Q)]
        depth = abs(tot_amp[np.argmin(tot_amp)] - tot_amp[np.argmax(tot_amp)])
    else:
        print('Invalid signal passed, please do I Q or None')
    if grab_depths:
        return best_signal_fit, pi_amp, depth
    else:
        return max_signal, best_signal_fit, pi_amp

def get_opt_rabi_data(data_dir, QubitIndex, expt_name='rabi_ge'):
    data_path = os.path.join(data_dir, expt_name)
    h5_files = os.listdir(data_path)
    h5_files.sort()
    h5_files_qubit_index = []

    # select only h5 files which have data for the chosen qubit
    for h5_file in h5_files:
        load_data = load_from_h5(os.path.join(data_path, h5_file), 'Rabi', save_r=1)
        if not np.isnan(load_data['Rabi'][QubitIndex].get('Dates', [])[0][0]).any():
            h5_files_qubit_index.append(h5_file)

    ## get data consistent over all rounds from first file
    load_data = load_from_h5(os.path.join(data_path, h5_files_qubit_index[0]), 'Rabi', save_r=1)
    rabi_gains = process_h5_data(load_data['Rabi'][QubitIndex].get('Gains', [])[0][0].decode())
    rabi_date = datetime.datetime.fromtimestamp(load_data['Rabi'][QubitIndex].get('Dates', [])[0][0])
    I = process_h5_data(load_data['Rabi'][QubitIndex].get('I', [])[0][0].decode())
    Q = process_h5_data(load_data['Rabi'][QubitIndex].get('Q', [])[0][0].decode())

    max_signal, fit, pi_amp = rabi_get_results(I, Q, rabi_gains)

    if max_signal == 'I':
        I_or_Q = I
    elif max_signal == 'Q':
        I_or_Q = Q

    return max_signal, I_or_Q, rabi_gains, pi_amp, fit

########################################### functions for offset freq data #####################################

def get_opt_offset_data(data_dir, QubitIndex, expt_name ='offset'):
    data_path = os.path.join(data_dir, expt_name)
    h5_files = os.listdir(data_path)
    h5_files.sort()
    h5_files_qubit_index = []

    # select only h5 files which have data for the chosen qubit
    for h5_file in h5_files:
        load_data = load_from_h5(os.path.join(data_path, h5_file), 'offset', save_r=1)
        if not np.isnan(load_data['offset'][QubitIndex].get('Dates', [])[0][0]).any():
            h5_files_qubit_index.append(h5_file)


    offset_res_freqs = []
    offset_fids = []

    idx = 0
    for h5_file in h5_files_qubit_index:
        load_data = load_from_h5(os.path.join(data_path, h5_file), 'offset', save_r=1)
        #ss_dates.append(datetime.datetime.fromtimestamp(load_data['SS'][QubitIndex].get('Dates', [])[0][0]))
        offset_fids.append(load_data['offset'][QubitIndex].get('Fidelity', [])[0])
        offset_res_freqs.append(load_data['offset'][QubitIndex].get('Res Frequency', [])[0])

    res_freq_steps = np.unique(offset_res_freqs)
    num_steps = len(res_freq_steps)
    num_ss = int(len(offset_res_freqs)/num_steps)

    mean_fids = np.mean(np.array(offset_fids).reshape([num_steps, num_ss]),axis=1)

    return res_freq_steps, mean_fids


########################################## functions for SSF data #####################################

def hist_ssf(data, axs, row, col, numbins = 100):

    ig = np.array(data[0])
    qg = np.array(data[1])
    ie = np.array(data[2])
    qe = np.array(data[3])

    xg, yg = np.median(ig), np.median(qg)
    xe, ye = np.median(ie), np.median(qe)

    """Compute the rotation angle"""
    theta = -np.arctan2((ye - yg), (xe - xg))
    """Rotate the IQ data"""
    ig_new = ig * np.cos(theta) - qg * np.sin(theta)
    qg_new = ig * np.sin(theta) + qg * np.cos(theta)
    ie_new = ie * np.cos(theta) - qe * np.sin(theta)
    qe_new = ie * np.sin(theta) + qe * np.cos(theta)

    """New means of each blob"""
    xg, yg = np.median(ig_new), np.median(qg_new)
    xe, ye = np.median(ie_new), np.median(qe_new)

    xlims = [np.min(ig_new), np.max(ie_new)]

    plot = axs[row][col]
    plot.scatter(ig_new, qg_new, label='g', color='b', marker='*')
    plot.scatter(ie_new, qe_new, label='e', color='r', marker='*')
    plot.scatter(xg, yg, color='k', marker='o')
    plot.scatter(xe, ye, color='k', marker='o')
    plot.set_xlabel('I [a.u.]')
    plot.set_ylabel('Q [a.u.]')
    plot.legend(loc='lower right')
    plot.set_title(f'rotated theta:{round(theta, 5)}',fontsize=10)
    plot.axis('equal')

    """X and Y ranges for histogram"""
    plot = axs[row][col + 1]
    ng, binsg, pg = plot.hist(ig_new, bins=numbins, range=xlims, color='b', label='g', alpha=0.5)
    ne, binse, pe = plot.hist(ie_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.5)
    plot.set_xlabel('I [a.u.]')

    """Compute the fidelity using overlap of the histograms"""
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
    tind = contrast.argmax()
    threshold = binsg[tind]
    fid = contrast[tind]
    plot.set_title(f"fidelity: {fid * 100:.2f}%, threshold: {threshold: .2f}",fontsize=10)

def get_opt_ssf_data(data_dir, QubitIndex, plot_idx = 0, expt_name = 'ss_ge'):
    data_path = os.path.join(data_dir, expt_name)
    h5_files = os.listdir(data_path)
    h5_files.sort()
    h5_files_qubit_index = []

    # select only h5 files which have data for the chosen qubit
    for h5_file in h5_files:
        load_data = load_from_h5(os.path.join(data_path, h5_file), 'SS', save_r=1)
        if not np.isnan(load_data['SS'][QubitIndex].get('Dates', [])[0][0]).any():
            h5_files_qubit_index.append(h5_file)

    num_rounds = len(h5_files_qubit_index)

    ss_dates = []
    ss_fid = []
    ss_angle = []

    idx = 0
    for h5_file in h5_files_qubit_index:
        load_data = load_from_h5(os.path.join(data_path, h5_file), 'SS', save_r=1)
        ss_dates.append(datetime.datetime.fromtimestamp(load_data['SS'][QubitIndex].get('Dates', [])[0][0]))
        ss_fid.append(load_data['SS'][QubitIndex].get('Fidelity', [])[0])
        ss_angle.append(load_data['SS'][QubitIndex].get('Angle', [])[0])

        if idx == plot_idx:
            I_g = process_h5_data(load_data['SS'][QubitIndex].get('I_g', [])[0][0].decode())
            Q_g = process_h5_data(load_data['SS'][QubitIndex].get('Q_g', [])[0][0].decode())
            I_e = process_h5_data(load_data['SS'][QubitIndex].get('I_e', [])[0][0].decode())
            Q_e = process_h5_data(load_data['SS'][QubitIndex].get('Q_e', [])[0][0].decode())

    return ss_dates, ss_fid, ss_angle, I_g, Q_g, I_e, Q_e, num_rounds

def get_ssf_data(data_dir, QubitIndex, expt_name = 'ss_ge'):
    data_path = os.path.join(data_dir, expt_name)
    h5_files = os.listdir(data_path)
    h5_files.sort()

    ss_dates = []
    ss_fid = []
    ss_angle = []

    for h5_file in h5_files:
        load_data = load_from_h5(os.path.join(data_path, h5_file), 'SS', save_r=1)
        ss_dates.append(datetime.datetime.fromtimestamp(load_data['SS'][QubitIndex].get('Dates', [])[0][0]))
        ss_fid.append(load_data['SS'][QubitIndex].get('Fidelity', [])[0])
        ss_angle.append(load_data['SS'][QubitIndex].get('Angle', [])[0])

    return ss_dates, ss_fid, ss_angle

########################## functions for T1 data ############################################

def exponential(x, a, b, c, d):
    return a * np.exp(-(x - b) / c) + d

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

def get_t1_data(data_dir, QubitIndex, expt_name="t1_ge"):
    data_path = os.path.join(data_dir, expt_name)
    h5_files = os.listdir(data_path)
    h5_files.sort()

    t1_dates = []
    t1s = []
    t1_errs = []

    load_data = load_from_h5(os.path.join(data_path, h5_files[0]), 'T1', save_r=1)
    delay_times = process_h5_data(load_data['T1'][QubitIndex].get('Delay Times', [])[0][0].decode())
    delay_pts = len(delay_times)
    reps = int(len(np.array(process_h5_data(load_data['T1'][QubitIndex].get('I', [])[0][0].decode())))/delay_pts)

    for h5_file in h5_files:
        load_data = load_from_h5(os.path.join(data_path, h5_file), 'T1', save_r=1)
        date = datetime.datetime.fromtimestamp(load_data['T1'][QubitIndex].get('Dates', [])[0][0])
        t1_dates.append(date)

        Ishots = np.array(process_h5_data(load_data['T1'][QubitIndex].get('I', [])[0][0].decode())).reshape([reps,delay_pts])
        Qshots = np.array(process_h5_data(load_data['T1'][QubitIndex].get('Q', [])[0][0].decode())).reshape([reps, delay_pts])

        I = np.mean(Ishots,axis=0)
        Q = np.mean(Qshots,axis=0)

        q1_fit_exponential, T1_err, T1_est, plot_sig = t1_fit(I, Q, delay_times)
        t1s.append(T1_est)
        t1_errs.append(T1_err)


    return t1_dates, t1s, t1_errs

def get_t1_data_at_time_t(data_dir, QubitIndex, time_idx = 0, expt_name='t1_ge'):
    data_path = os.path.join(data_dir, expt_name)
    h5_files = os.listdir(data_path)
    h5_files.sort()

    load_data = load_from_h5(os.path.join(data_path, h5_files[time_idx]), 'T1', save_r=1)
    date = datetime.datetime.fromtimestamp(load_data['T1'][QubitIndex].get('Dates', [])[0][0])
    delay_times = process_h5_data(load_data['T1'][QubitIndex].get('Delay Times', [])[0][0].decode())
    delay_pts = len(delay_times)
    reps = int(len(np.array(process_h5_data(load_data['T1'][QubitIndex].get('I', [])[0][0].decode()))) / delay_pts)

    Ishots = np.array(process_h5_data(load_data['T1'][QubitIndex].get('I', [])[0][0].decode())).reshape([reps, delay_pts])
    Qshots = np.array(process_h5_data(load_data['T1'][QubitIndex].get('Q', [])[0][0].decode())).reshape([reps, delay_pts])


    I = np.mean(Ishots, axis=0)
    Q = np.mean(Qshots, axis=0)

    q1_fit_exponential, T1_err, T1_est, plot_sig = t1_fit(I, Q, delay_times)

    if plot_sig == 'I':
        I_or_Q = I
    else:
        I_or_Q = Q

    return delay_times, date, q1_fit_exponential, T1_err, T1_est, plot_sig, I_or_Q

########################## functions for stark shift data ####################################






########################### functions for rabi population data ##########################


def get_ef_rabi_data(data_dir, QubitIndex, expt_name="q_temperatures"):
    data_path = os.path.join(data_dir, expt_name)
    h5_files = os.listdir(data_path)
    h5_files.sort()
    h5_files_qubit_index = []

    for h5_file in h5_files:
        load_data = load_from_h5(os.path.join(data_path, h5_file), 'Qtemps', save_r=1)
        if not np.isnan(load_data['Qtemps'][QubitIndex].get('Dates', [])[0]).any():
            h5_files_qubit_index.append(h5_file)

    load_data = load_from_h5(os.path.join(data_path, h5_files_qubit_index[0]), 'Qtemps', save_r=1)

    I1 = process_h5_data(load_data['Qtemps'][QubitIndex].get('I1', [])[0][0].decode())
    Q1 = process_h5_data(load_data['Qtemps'][QubitIndex].get('Q1', [])[0][0].decode())
    gains1 = process_h5_data(load_data['Qtemps'][QubitIndex].get('Gains1', [])[0][0].decode())
    mag1 = np.sqrt(np.square(I1) + np.square(Q1))

    I2 = process_h5_data(load_data['Qtemps'][QubitIndex].get('I2', [])[0][0].decode())
    Q2 = process_h5_data(load_data['Qtemps'][QubitIndex].get('Q2', [])[0][0].decode())
    gains2 = process_h5_data(load_data['Qtemps'][QubitIndex].get('Gains2', [])[0][0].decode())
    mag2 = np.sqrt(np.square(I2) + np.square(Q2))

    max_signal, best_signal_fit1, pi_amp1 = rabi_get_results(mag1, mag1, gains1)
    max_signal, best_signal_fit2, pi_amp2 = rabi_get_results(mag2, mag2, gains2)

    return gains1, mag1, best_signal_fit1, pi_amp1, gains2, mag2, best_signal_fit2, pi_amp2


