import numpy as np
import os
import sys
import re
from matplotlib import pyplot as plt
import h5py
from scipy.optimize import curve_fit
from scipy.stats import linregress

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def lorentzian( f, f0, gamma, A, B):
    return A * gamma ** 2 / ((f - f0) ** 2 + gamma ** 2) + B

def linewidth(gain, m, b):
    return m*(gain)**2 + b

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

##  Load data
path = "/data/QICK_data/run6/6transmon/StarkShift/optimization_tests_1/2025-03-12_18-49-21/Q3/stark2D/Data_h5/StarkShift2D_ge"
file = "2025-03-12_19-21-00_StarkShift2D_results_batch_0_Num_per_batch1.h5"
filename = os.path.join(path, file)

QubitIndex = 0
g = [18.29625291]
delta = [2.001569 *1000]
chi = (np.array(g)**2)/np.array(delta)
print(chi)
#chi = [0.02779979982] #[MHz]
#chi_lamb = [0.135] #[MHz]
chi_lamb = chi

kappa = [0.9] #[MHz]
interval = 2
flags = {'gauss': True, 'lorentzian': True, 'slice': True}

load_data = load_from_h5(filename, 'stark2D', save_r=1)
gain_sweep = np.array(process_h5_data(load_data['stark2D'][QubitIndex].get('Res Gain Sweep', [])[0][0].decode()))
qfreq_sweep = np.array(process_h5_data(load_data['stark2D'][QubitIndex].get('Qu Frequency Sweep', [])[0][0].decode()))
gain_pts = len(gain_sweep)
qfreq_pts = len(qfreq_sweep)
I = np.array(process_h5_data(load_data['stark2D'][QubitIndex].get('I', [])[0][0].decode())).reshape((gain_pts, qfreq_pts))
Q = np.array(process_h5_data(load_data['stark2D'][QubitIndex].get('Q', [])[0][0].decode())).reshape((gain_pts, qfreq_pts))
magnitude = np.sqrt(np.square(I) + np.square(Q))

fit_err_list = []
width_list_g = []
freqshift_list_g = []
width_list_l = []
freqshift_list_l = []

gain_interval = np.arange(np.argmin(gain_sweep), np.argmax(gain_sweep), interval)
fig, axes = plt.subplots(3, 1, figsize=(10, 20))
s = 0
for i in np.arange(0, len(gain_interval)):
    idx = i * interval
    freq_q = qfreq_sweep[np.argmin(magnitude[idx,:])]

    if flags['gauss']:
        p0 = [13, -8, freq_q, 5]
        params_g, cov = curve_fit(gauss, qfreq_sweep, magnitude[idx,:], p0=p0)
        width_list_g.append(params_g[3])
        freqshift_list_g.append(params_g[2])

    if flags['lorentzian']:
        p0 = [freq_q, 5, 13, -8 ]
        params_l, cov = curve_fit(lorentzian, qfreq_sweep, magnitude[idx,:], p0=p0)
        width_list_l.append(np.abs(params_l[1])*2)
        freqshift_list_l.append(params_l[0])

    if flags['slice']:
         if i in [0, round(len(gain_interval)/2), len(gain_interval)-1]:

            plot = axes[s]
            plot.set_box_aspect(1)
            plot.plot(qfreq_sweep, magnitude[idx,:])
            if flags['gauss']:
                plot.plot(qfreq_sweep, gauss(qfreq_sweep, *params_g),label='gaussian')
            if flags['lorentzian']:
                plot.plot(qfreq_sweep, lorentzian(qfreq_sweep, *params_l), label='lorentzian')
            plot.set_title(f"1D slice at gain = {round(gain_sweep[idx],2)}")
            plot.set_ylabel("magnitude of I,Q signal")
            plot.set_xlabel("qubit pulse frequency [MHz]")
            plot.legend()

            s=s+1

plt.show()

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
plot = axes[0]
plot.set_box_aspect(1)
plt.colorbar(plot.pcolormesh(qfreq_sweep, gain_sweep**2, magnitude, cmap="viridis"), ax=plot, shrink=0.7)
plot.set_title("magnitude of I,Q signal")
plot.set_ylabel("resonator probe power [a.u.]")
plot.set_xlabel("qubit pulse frequency [MHz]")

idx = np.arange(0, len(gain_interval)) * interval
plot.scatter(freqshift_list_l, gain_sweep[idx]**2, s=1, c='k', label='fit freq')

if flags['gauss']:
    params = linregress(gain_sweep[idx]**2, freqshift_list_g)
    q0 = params.intercept
    wq = (gain_sweep[idx]**2 *params.slope) + q0
    n_g = (wq - q0)/(chi_lamb[QubitIndex] * -2)

if flags['lorentzian']:
    params = linregress(gain_sweep[idx]**2, freqshift_list_l)
    q0 = params.intercept
    wq = (gain_sweep[idx]**2 *params.slope) + q0
    n_l = (wq - q0)/(chi_lamb[QubitIndex] * -2)

plot.plot(wq, gain_sweep[idx]**2, 'r:', label='linear fit')
plot.legend(loc=3)
print(params.slope)

# plot = axes[1]
# plot.set_box_aspect(1)
# plot.plot(n, -(wq - q0), c='k')
# plot.set_xlabel('n [# of photons]')
# plot.set_ylabel('AC Stark shift [MHz]')

plot = axes[2]
plot.set_box_aspect(1)

if flags['gauss']:
    plot.plot(n_g, np.array(width_list_g) * 2.355, 'b', label="gaussian")
if flags['lorentzian']:
    plot.plot(n_l, np.array(width_list_l), 'r', label="lorentzian")

plot.set_ylabel("linewidth [MHz]")

plot.set_xlabel("n")
#params, cov = curve_fit(linewidth, gain_sweep[idx], width_list)
#plot.plot(gain_sweep, linewidth(gain_sweep, *params), 'r--', label='quadratic fit')
plot.legend()

# plot = axes[2]
# plot.plot(n, width_list,'k')
# plot.set_xlabel("n")
# plot.set_ylabel("fwhm")

plot = axes[1]
plot.plot(gain_sweep[idx], n_g, 'b', label='gaussian')
plot.plot(gain_sweep[idx], n_l, 'r', label='lorentzian')
plot.set_xlabel("resonator gain [a.u.]")
plot.set_ylabel("n [# of photons]")
plot.legend()


theta0 = ((2*chi[QubitIndex])/kappa[QubitIndex])
plot = axes[3]
plot.set_box_aspect(1)
if flags['gauss']:
     params, cov = curve_fit(linewidth, gain_sweep[idx], width_list_g)
     n_standev = (params[0]*(gain_sweep**2) / (kappa[QubitIndex] * theta0)) ** 2
     plot.plot(gain_sweep, n_standev, 'b', label='n from gaussian sigma')
if flags['lorentzian']:
    params, cov = curve_fit(linewidth, gain_sweep[idx], width_list_l)
    n_fwhm = (params[0]*(gain_sweep**2))/(2*kappa[QubitIndex]*(theta0**2))
    plot.plot(gain_sweep, n_fwhm, 'r', label='n from lorentzian fwhm')

plot.set_xlabel("resonator gain [a.u.]")
plot.set_ylabel("n [# of photons]")
plot.legend()


plt.show()
