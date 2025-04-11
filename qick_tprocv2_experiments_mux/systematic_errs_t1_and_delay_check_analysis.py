import re
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
from analysis_006_T1_vs_time_plots import T1VsTime
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

analysis_folder = f"/data/QICK_data/{run_name}/benchmark_analysis_plots/"
create_folder_if_not_exists(analysis_folder)

# Define stop times (as numbers)
stop_times = np.linspace(150, 600, 10)
stop_times = [round(n, 0) for n in stop_times]

# Initialize dictionaries to store data for each qubit.
qubit_stop_times = {q: [] for q in range(tot_num_of_qubits)}
qubit_t1_vals = {q: [] for q in range(tot_num_of_qubits)}
qubit_t1_err = {q: [] for q in range(tot_num_of_qubits)}

# Loop over each stop time and run the measurement routine.
for stop in stop_times:
    # Build the folder name based on the current stop time.
    top_folder_dates = [f'2025-02-24/Stats/{str(stop)}']

    # Run the T1VsTime measurement routine.
    t1_vs_time = T1VsTime(figure_quality, final_figure_quality, tot_num_of_qubits,
                          top_folder_dates, save_figs, fit_saved, signal, run_name, FRIDGE)
    date_times_t1, t1_vals, t1_fit_err = t1_vs_time.run(return_errs=True)

    # Store the current stop time and the T1 data for each qubit.
    for q in range(tot_num_of_qubits):
        if len(t1_vals[q])==1:
            qubit_stop_times[q].append(stop)
            qubit_t1_vals[q].append(t1_vals[q][0])
            qubit_t1_err[q].append(t1_fit_err[q])

# Plotting setup similar to your T2R code
font = 14
titles = [f"Qubit {i + 1}" for i in range(tot_num_of_qubits)]
colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('T1 Values vs Stop Time', fontsize=font)
axes = axes.flatten()
qubit_t1_err=[10,6,10,10,1,3]
# For each qubit, plot its stop times vs T1 values.
for i, ax in enumerate(axes):
    ax.set_title(titles[i], fontsize=font)

    # Retrieve the data from the dictionaries.
    x = qubit_stop_times[i]
    y = qubit_t1_vals[i]
    yerr = qubit_t1_err[i]
    print(x,y)

    # Plot using error bars (no sorting is done)
    ax.errorbar(x, y, yerr=yerr, fmt='o-', color=colors[i], capsize=3)

    # Set x-axis ticks to be the stop times.
    ax.set_xticks(x)
    ax.set_xlabel('Stop Time', fontsize=font - 2)
    ax.set_ylabel('T1 (us)', fontsize=font - 2)
    ax.tick_params(axis='both', which='major', labelsize=8)

    if show_legends:
        ax.legend([titles[i]], edgecolor='black')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(analysis_folder + 'T1_vals_vs_stop_time.pdf', transparent=True, dpi=final_figure_quality)
plt.show()