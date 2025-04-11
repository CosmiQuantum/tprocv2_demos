import re
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime
import matplotlib.pyplot as plt
# from datetime import datetime
import datetime
import pytz
import os
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from system_config import QICK_experiment
import numpy as np
import json
import h5py
from qualang_tools.plot import Fit
import visdom
###################################################### Set These #######################################################
save_figs = True
fit_saved = False
show_legends = False
signal = 'None'
run_number = 2 #starting from first run with qubits. Run 1 = run4a at quiet, run 2 = run5a at quiet, etc
figure_quality = 100 #ramp this up to like 500 for presentation plots
final_figure_quality = 200
run_name = '6transmon_run6'
FRIDGE = "QUIET"
run_notes = ('Added more eccosorb filters and a lpf on mxc before and after the device. Added thermometry '
             'next to the device') #please make it brief for the plot
top_folder_dates = ['2025-02-24/Stats']

# ################################################ 01: Get all data ######################################################

q_spec_vs_time = QubitFreqsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
                                  save_figs, fit_saved, signal, run_name, FRIDGE)
date_times_q_spec, q_freqs, qspec_fit_err = q_spec_vs_time.run()

#---------------------------------plot-----------------------------------------------------
def datetime_to_unix(dt):
    # Convert to Unix timestamp
    unix_timestamp = int(dt.timestamp())
    return unix_timestamp


def unix_to_datetime(unix_timestamp):
    # Convert the Unix timestamp to a datetime object
    dt = datetime.fromtimestamp(unix_timestamp)
    return dt


def create_folder_if_not_exists( folder):
    """Creates a folder at the given path if it doesn't already exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)


def exponential( x, a, b, c, d):
    return a * np.exp(-(x - b) / c) + d


def optimal_bins(data):
    n = len(data)
    if n == 0:
        return {}
    # Sturges' Rule
    sturges_bins = int(np.ceil(np.log2(n) + 1))
    return sturges_bins


def process_string_of_nested_lists( data):
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

analysis_folder = f"/data/QICK_data/{run_name}/benchmark_analysis_plots/"
create_folder_if_not_exists(analysis_folder)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()
font = 14
titles = [f"Qubit {i+1}" for i in range(tot_num_of_qubits)]
gaussian_xvals =  {i: [] for i in range(0, tot_num_of_qubits)}
gaussian_yvals =  {i: [] for i in range(0, tot_num_of_qubits)}
gaussian_colors = {i: [] for i in range(0, tot_num_of_qubits)}
gaussian_dates = {i: [] for i in range(0, tot_num_of_qubits)}
mean_values = {}
std_values = {}
from scipy.stats import norm
colors = ['orange','blue','purple','green','brown','pink']
for i, ax in enumerate(axes):

    # Skip processing if data is completely missing (key does not exist)
    if i not in date_times_q_spec or i not in q_freqs:
        print(f"  Skipping Qubit {i}: No data found.")
        ax.set_visible(False)  # Hide subplot
        continue

    # Skip plotting if the lists are completely empty
    if not q_freqs[i] :
        print(f"  Skipping Qubit {i}: Empty data lists.")
        ax.set_visible(False)  # Hide subplot
        continue

    if len( date_times_q_spec[i])>1:
        date_label = date_times_q_spec[i][0]
    else:
        date_label = ''


    if len(q_freqs[i]) >1:
        optimal_bin_num = 30 #optimal_bins(q_freqs[i])

        # Fit a Gaussian to the raw data instead of the histogram
        # get the mean and standard deviation of the data

        mu_1, std_1 = norm.fit(q_freqs[i])
        mean_values[f"Qubit {i + 1}"] = mu_1  # Store the mean value for each qubit
        std_values[f"Qubit {i + 1}"] = std_1

        # Generate x values for plotting a gaussian based on this mean and standard deviation
        x_1 = np.linspace(min(q_freqs[i]), max(q_freqs[i]), optimal_bin_num)
        p_1 = norm.pdf(x_1, mu_1, std_1)

        # Calculate histogram data for t1_vals[i]
        hist_data_1, bins_1 = np.histogram(q_freqs[i], bins=optimal_bin_num)
        bin_centers_1 = (bins_1[:-1] + bins_1[1:]) / 2

        # Scale the Gaussian curve to match the histogram
        # the gaussian height natrually doesnt match the bin heights in the histograms
        # np.diff(bins_1)  calculates the width of each bin by taking the difference between bin edges
        # the total counts are in hist_data_1.sum()
        # to scale, multiply data gaussian by bin width to convert the probability density to probability within each bin
        # then multiply by the total count to scale the probability to match the overall number of datapoints
        # https://mathematica.stackexchange.com/questions/262314/fit-function-to-histogram
        # https://stackoverflow.com/questions/23447262/fitting-a-gaussian-to-a-histogram-with-matplotlib-and-numpy-wrong-y-scaling
        ax.plot(x_1, p_1 * (np.diff(bins_1) * hist_data_1.sum()), 'b--', linewidth=2, color=colors[i])

        # Plot histogram and Gaussian fit for t1_vals[i]
        ax.hist(q_freqs[i], bins=optimal_bin_num, alpha=0.7, color=colors[i], edgecolor='black', label=date_label)

        #make a fuller gaussian to make smoother lotting for cumulative plot
        x_1_full = np.linspace(min(q_freqs[i]), max(q_freqs[i]), 2000)
        p_1_full = norm.pdf(x_1_full, mu_1, std_1)

        gaussian_xvals[i].append(x_1_full)
        gaussian_yvals[i].append(p_1_full )
        gaussian_colors[i].append(colors[i])
        gaussian_dates[i].append(date_label)

        #rough start at errors:
        #counts, bin_edges, _ = ax.hist(t1_vals[i], bins=20, alpha=0.7, color='blue', edgecolor='black')
        #bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        #bin_errors = [np.sqrt(np.sum(t1_errs[i])) for _ in range(len(bin_centers))]  #using error propogation through the sum of valuse in each bin
        #ax.errorbar(bin_centers, counts, yerr=bin_errors, fmt='o', color='red', ecolor='black', capsize=3, linestyle='None')
        if show_legends:
            ax.legend()
        ax.set_title(titles[i] + f" $\mu$: {mu_1:.2f} $\sigma$:{std_1:.2f}",fontsize = font)
        ax.set_xlabel('QFreq (MHz)',fontsize = font)
        ax.set_ylabel('Frequency',fontsize = font)
        ax.tick_params(axis='both', which='major', labelsize=font)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig( analysis_folder + 'QFreqhists.png', transparent=False, dpi=100)
