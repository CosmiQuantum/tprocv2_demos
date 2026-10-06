import numpy as np
import os
import sys
from matplotlib.ticker import LogLocator, LogFormatterMathtext
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))

from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
#from expt_config import *
import glob
import re
import datetime
import ast
import os
import matplotlib.pyplot as plt
import allantools
from scipy.signal import welch
from scipy.stats import norm
from scipy.optimize import curve_fit

class AllanWelchStats:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name):
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.figure_quality = figure_quality
        self.run_name = run_name
        self.number_of_qubits = number_of_qubits
        self.final_figure_quality = final_figure_quality
        self.top_folder_dates = top_folder_dates

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def plot_allan_deviation(self, date_times, vals, show_legends, label="T1"):

        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/allan_stats/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)
        fig.suptitle(f'Overlapping Allan Deviation of {label} Fluctuations', fontsize=font)
        axes = axes.flatten()

        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]

        # -----------------------------------------------------------------------
        #  For each qubit, sort data by timestamp, compute Oadev, and plot
        # -----------------------------------------------------------------------
        for i, ax in enumerate(axes):
            # Hide extra subplots if you have fewer than 6 qubits
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            # Extract this qubit's data
            datetime_strings = date_times[i]  # list of "YYYY-MM-DD HH:MM:SS"
            data = vals[i]

            # Convert to datetime objects
            from datetime import datetime
            dt_objs = [datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s in datetime_strings]

            # Sort (ascending) by time
            combined = list(zip(dt_objs, data))
            combined.sort(key=lambda x: x[0])  # sort by datetime
            sorted_times, sorted_vals = zip(*combined)

            # Convert times -> seconds since first measurement
            t0 = sorted_times[0]
            time_sec = np.array([(t - t0).total_seconds() for t in sorted_times])
            vals_array = np.array(sorted_vals, dtype=float)

            # If you only have a single point, skip
            if len(time_sec) <= 1:
                ax.text(0.5, 0.5, "Not enough points", ha='center', va='center', transform=ax.transAxes)
                continue

            # Approx. average sample rate for Oadev
            avg_dt = np.mean(np.diff(time_sec))
            if avg_dt <= 0:
                avg_dt = 1.0
            rate = 1.0 / avg_dt

            # overlapping Allan deviation
            # Use 'freq' data_type since label is not a phase measure.
            #  auto-select tau points with taus='decade' or you could supply np.logspace(...).
            taus_out, ad, ade, ns = allantools.oadev(
                vals_array,
                rate=rate,
                data_type='freq',
                taus='decade'
            )

            ax.set_xscale('log')

            ax.scatter(taus_out, ad, marker='o', color=colors[i], label=f"Qubit {i + 1}")
            ax.errorbar(taus_out, ad, yerr=ade, fmt='o', color=colors[i])

            if show_legends:
                ax.legend(loc='best', edgecolor='black')

            ax.set_xlim(min(taus_out[taus_out > 0])* 0.8, max(taus_out) * 1.5)
            ax.set_xlabel(r"$\tau$ (s)", fontsize=font - 2)
            ax.set_ylabel(rf"$\sigma_{{{label}}}(\tau)$ (µs)", fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + f'{label}_allan_deviation.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close(fig)

    def plot_welch_spectral_density(self, date_times, vals, show_legends, label="T1"):

        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/welch_stats/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)
        fig.suptitle(f'Welch-method Spectral Density of {label} Fluctuations', fontsize=font)
        axes = axes.flatten()

        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]

        for i, ax in enumerate(axes):
            ax.set_title(titles[i], fontsize=font)

            datetime_strings = date_times[i]  # list of "YYYY-MM-DD HH:MM:SS"
            data = vals[i]

            from datetime import datetime
            dt_objs = [datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s in datetime_strings]

            combined = list(zip(dt_objs, data))
            combined.sort(key=lambda x: x[0])  # sort by datetime
            sorted_times, sorted_vals = zip(*combined)

            # Convert times -> seconds since first measurement
            t0 = sorted_times[0]
            time_sec = np.array([(t - t0).total_seconds() for t in sorted_times])
            vals_array = np.array(sorted_vals, dtype=float)

            #average sampling interval (seconds)
            avg_dt = np.mean(np.diff(time_sec))
            if avg_dt <= 0:
                avg_dt = 1.0

            # Compute Welch PSD
            # Note: fs = 1 / avg_dt  (Hz)
            fs = 1.0 / avg_dt

            freq, psd = welch(vals_array, fs=fs, nperseg=None, scaling='density')

            ax.set_xscale('log')
            ax.set_yscale('log')

            # "freq" in Hz, "psd" has units of (amplitude^2)/Hz
            ax.plot(freq, psd, marker='o', color=colors[i],
                    linestyle='none', label=f"Qubit {i + 1}")

            if show_legends:
                ax.legend(loc='best', edgecolor='black')

            ax.set_xlabel(r"Frequency (Hz)", fontsize=font - 2)
            ax.set_ylabel(rf"$S_{{{label}}}$ ($\mu s^2$/Hz)", fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)


        plt.tight_layout()
        plt.savefig(analysis_folder + f'{label}_welch_spectral_density.pdf',
                    transparent=True, dpi=self.final_figure_quality)
        plt.close(fig)