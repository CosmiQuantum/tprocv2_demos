#!/usr/bin/env python3
"""
Multi-peak Qubit Spectroscopy Data Processing and Plotting Script
Adjusted to:
 - Smooth the chosen data channel (I or Q) before peak detection,
 - Detect peaks/dips on the smoothed data using adaptive thresholds and width filtering,
 - Then perform Lorentzian fits on the original data near those detected peaks.
"""

import os
import sys
import re
import glob
import ast
import json
import copy
import logging
import datetime
import pytz

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter, peak_widths  # <-- Added peak_widths
import h5py
import visdom

# Local imports
from syspurpose.files import three_way_merge
from section_008_save_data_to_h5 import Data_H5
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from analysis_000_load_configs import LoadConfigs
from analysis_001_plot_all_RR_h5 import PlotAllRR
from analysis_002_res_centers_vs_time_plots import ResonatorFreqVsTime
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime
from analysis_004_pi_amp_vs_time_plots import PiAmpsVsTime
from analysis_005_Qtemp_vs_time_plots import QTempsVsTime
from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_007_T2R_vs_time_plots import T2rVsTime
from analysis_008_T2E_vs_time_plots import T2eVsTime
from analysis_009_T1_hist_cumul_err_plots import T1HistCumulErrPlots
from analysis_010_T2R_hist_cumul_err_plots import T2rHistCumulErrPlots
from analysis_011_T2E_hist_cumul_err_plots import T2eHistCumulErrPlots
from analysis_012_save_run_data import SaveRunData
from analysis_013_update_saved_run_data_notes import UpdateNote
from analysis_014_temperature_calcsandplots import TempCalcAndPlots
from analysis_015_plot_all_run_stats import CompareRuns
from analysis_016_metrics_vs_temp import (ResonatorFreqVsTemp, GetThermData, QubitFreqsVsTemp,
                                          PiAmpsVsTemp, T1VsTemp, T2rVsTemp, T2eVsTemp)
from analysis_017_plot_metric_dependencies import PlotMetricDependencies
from analysis_018_box_whisker import PlotBoxWhisker
from analysis_019_allan_welch_stats_plots import AllanWelchStats
from section_011_qubit_temperatures_efRabi import QubitTemperatureProgram, QubitTemperatureRefProgram
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from system_config import QICK_experiment
from qualang_tools.plot import Fit

###################################################### Constants #######################################################
save_figs = True
fit_saved = False
show_legends = False
# signal can be 'I', 'Q', or 'None' (to pick whichever has larger amplitude).
signal = 'None'
run_number = 2  # starting from first run with qubits. Run 1 = run4a, run 2 = run5a, etc.
figure_quality = 100      # Increase for presentation plots
final_figure_quality = 200
run_name = '6transmon_run6'
FRIDGE = "QUIET"          # Overwrites FRIDGE from expt_config if needed
run_notes = (
    'Added more eccosorb filters and a lpf on mxc before and after the device. '
    'Added thermometry next to the device'
)
top_folder_dates = ['2025-02-26']
number_of_qubits = 6

###################################################### Helper Functions #######################################################
def process_h5_data(data):
    """
    Convert h5 data (bytes or string) to a list of floats.
    Removes extra whitespace and non-numeric characters.
    """
    if isinstance(data, bytes):
        data_str = data.decode()
    elif isinstance(data, str):
        data_str = data
    else:
        raise ValueError("Unsupported data type. Data should be bytes or string.")
    cleaned_data = ''.join(c for c in data_str if c.isdigit() or c in ['-', '.', ' ', 'e'])
    numbers = [float(x) for x in cleaned_data.split() if x]
    return numbers

def create_folder_if_not_exists(folder_path):
    """Create folder if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def lorentzian(f, f0, gamma, A, B):
    """
    Single Lorentzian function:
        L(f) = A * gamma^2 / ((f - f0)**2 + gamma**2) + B
    """
    return A * gamma**2 / ((f - f0)**2 + gamma**2) + B

def fit_single_peak(freqs, data, peak_index, window=5, peak_direction=1):
    """
    Fit a single Lorentzian peak around the detected `peak_index`.
    The parameter `peak_direction` (1 or -1) is used to set the initial guess
    for the amplitude A (positive for upward peaks, negative for dips).
    Returns (popt, f_fit, d_fit) if successful, else (None, None, None).
      popt = [f0, gamma, A, B]
      f_fit, d_fit = arrays for plotting the fitted Lorentzian.
    """
    # Slice a region around the peak
    start = max(0, peak_index - window)
    end = min(len(data), peak_index + window)
    f_slice = freqs[start:end]
    d_slice = data[start:end]

    # Initial guess for [f0, gamma, A, B]
    f0_guess = freqs[peak_index]
    gamma_guess = 1.0
    # Use the peak_direction to adjust A_guess:
    A_guess = peak_direction * (np.max(d_slice) - np.min(d_slice))
    B_guess = np.mean(d_slice)
    p0 = [f0_guess, gamma_guess, A_guess, B_guess]

    try:
        popt, pcov = curve_fit(lorentzian, f_slice, d_slice, p0=p0)
        # Generate a high-res fit curve for plotting
        f_fit = np.linspace(f_slice[0], f_slice[-1], 200)
        d_fit = lorentzian(f_fit, *popt)
    except RuntimeError:
        # Fit failed
        return None, None, None

    return popt, f_fit, d_fit

###################################################### Main Execution #######################################################
def main():
    """
    1) Load all data using QubitFreqsVsTime (not shown here; user code).
    2) For each H5 file and each qubit dataset, read out I, Q, Frequencies.
    3) Choose the channel (I or Q) with the largest amplitude,
       then smooth that data for peak detection.
    4) Compute the baseline (mode) of the smoothed values, determine the peak direction,
       detect peaks on the smoothed data using a two-pass method, but fit Lorentzian on the original data.
    5) Plot raw data, each local Lorentzian fit, and mark the peak centers.
    6) Save plots if save_figs = True.
    """

    # 01: Get all data using QubitFreqsVsTime (this calls your existing aggregator)
    q_spec_vs_time = QubitFreqsVsTime(
        figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
        save_figs, fit_saved, signal, run_name, FRIDGE
    )
    date_times_q_spec, q_freqs, qspec_fit_err = q_spec_vs_time.run()

    # 02: Plotting for a specific date
    date_str = '2025-02-26'  # Only plot for one date at a time
    outerFolder = f"/data/QICK_data/{run_name}/{date_str}/"
    outerFolder_save_plots = f"/data/QICK_data/{run_name}/{date_str}_plots/"
    outerFolder_expt = os.path.join(outerFolder, "Data_h5", "QSpec_ge")
    h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

    for h5_file in h5_files:
        # Extract an integer 'save_round' from the filename, if applicable
        save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
        H5_class_instance = Data_H5(h5_file)
        load_data = H5_class_instance.load_from_h5(data_type='QSpec', save_r=int(save_round))

        # Determine which qubit keys have valid dates
        populated_keys = []
        for q_key in load_data['QSpec']:
            dates_list = load_data['QSpec'][q_key].get('Dates', [[]])
            if any(not np.isnan(date) for date in dates_list[0]):
                populated_keys.append(q_key)

        for q_key in populated_keys:
            # Loop over all data sets for this qubit key
            for dataset in range(len(load_data['QSpec'][q_key].get('Dates', [])[0])):
                timestamp = load_data['QSpec'][q_key].get('Dates', [])[0][dataset]
                date_obj = datetime.datetime.fromtimestamp(timestamp)

                # Extract I, Q, Frequencies
                I_raw = load_data['QSpec'][q_key].get('I', [])[0][dataset].decode()
                Q_raw = load_data['QSpec'][q_key].get('Q', [])[0][dataset].decode()
                freqs_raw = load_data['QSpec'][q_key].get('Frequencies', [])[0][dataset].decode()

                I = process_h5_data(I_raw)
                Q = process_h5_data(Q_raw)
                freqs = process_h5_data(freqs_raw)

                round_num = load_data['QSpec'][q_key].get('Round Num', [])[0][dataset]
                batch_num = load_data['QSpec'][q_key].get('Batch Num', [])[0][dataset]

                exp_config_str = load_data['QSpec'][q_key].get('Exp Config', [])[0][dataset].decode()
                safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                exp_config = eval(exp_config_str, safe_globals)

                if len(I) == 0 or len(Q) == 0:
                    continue  # skip if no data

                # Prepare the QubitSpectroscopyGE instance (if you need it for something else)
                qspec_instance = QubitSpectroscopy(
                    q_key, number_of_qubits, outerFolder_save_plots,
                    round_num, signal, save_figs
                )

                # Convert to numpy arrays
                freqs = np.array(freqs)
                I = np.array(I)
                Q = np.array(Q)

                # ---- Choose the channel with the largest amplitude ----
                amplitude_I = np.ptp(I)  # peak-to-peak range
                amplitude_Q = np.ptp(Q)
                if amplitude_I >= amplitude_Q:
                    chosen_signal = 'I'
                    chosen_data = I
                else:
                    chosen_signal = 'Q'
                    chosen_data = Q

                # -------------------- Smoothing Step --------------------
                # A typical Savitzky–Golay setup for preserving overall peak shape:
                savgol_window_length = 7  # Must be odd; tweak as needed
                savgol_polyorder = 2       # Usually 2 or 3 works well
                # If the dataset is shorter than the window, just skip smoothing:
                if len(chosen_data) >= savgol_window_length:
                    smoothed_data = savgol_filter(chosen_data, savgol_window_length, savgol_polyorder)
                else:
                    smoothed_data = chosen_data.copy()
                # -------------------------------------------------------

                # ---- Compute baseline on the SMOOTHED data ----
                hist, bin_edges = np.histogram(smoothed_data, bins=50)
                idx_max = np.argmax(hist)
                baseline = (bin_edges[idx_max] + bin_edges[idx_max+1]) / 2

                # Determine the relative position of the baseline within the span
                span = np.ptp(smoothed_data)
                relative_position = (baseline - np.min(smoothed_data)) / span
                # If baseline is in the lower half, expect upward peaks; otherwise, dips.
                if relative_position < 0.5:
                    peak_direction = 1   # peaks above baseline
                else:
                    peak_direction = -1  # dips below baseline

                # ---- Adaptive Peak Detection ----
                # Base detection parameter
                base_prominence = 0.30

                # Compute a noise measure (median absolute deviation)
                noise_mad = np.median(np.abs(smoothed_data - baseline))
                # Define a dynamic prominence threshold (e.g. at least 3 times the noise level)
                dynamic_prominence = max(base_prominence, 4 * noise_mad)

                # Prepare the data for detection by centering and flipping if needed
                data_for_detection = (smoothed_data - baseline) * peak_direction

                # Two-pass detection:
                # First pass: lenient detection (lower threshold) to capture wide, low peaks
                candidate_peaks, candidate_props = find_peaks(data_for_detection,
                                                              prominence=base_prominence / 2,
                                                              distance=5)
                # Compute peak widths (at half prominence)
                widths, _, _, _ = peak_widths(data_for_detection, candidate_peaks, rel_height=0.5)
                # Second pass: filter candidates.
                # Keep peaks that either have a prominence above our dynamic threshold or are wide enough.
                width_threshold = 10000  # adjust as needed based on expected peak width
                final_peaks = []
                for i, peak in enumerate(candidate_peaks):
                    prom = candidate_props['prominences'][i]
                    if prom >= dynamic_prominence or widths[i] >= width_threshold:
                        final_peaks.append(peak)
                final_peaks = np.array(final_peaks)

                # Use 'window' for Lorentzian fitting on the ORIGINAL data:
                fit_window = 300

                # Fit on the ORIGINAL data near each detected peak
                peak_fits = []
                for idx in final_peaks:
                    popt, f_fit, d_fit = fit_single_peak(
                        freqs, chosen_data, idx,
                        window=fit_window,
                        peak_direction=peak_direction
                    )
                    if popt is not None:
                        peak_fits.append((popt, f_fit, d_fit))

                # ---- Plot the data and the fits ----
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                plt.rcParams.update({'font.size': 18})

                # Plot raw data on both axes
                ax1.plot(freqs, I, label='I', linewidth=2)
                ax2.plot(freqs, Q, label='Q', linewidth=2)

                # Overlay Lorentzian fits on the chosen signal's axis
                peak_centers = []
                if chosen_signal == 'I':
                    for (popt, f_fit, d_fit) in peak_fits:
                        f0 = popt[0]
                        peak_centers.append(f0)
                        ax1.plot(f_fit, d_fit, 'r--', label='Lorentzian Fit')
                        ax1.axvline(f0, color='orange', linestyle='--', linewidth=2)
                else:
                    for (popt, f_fit, d_fit) in peak_fits:
                        f0 = popt[0]
                        peak_centers.append(f0)
                        ax2.plot(f_fit, d_fit, 'r--', label='Lorentzian Fit')
                        ax2.axvline(f0, color='orange', linestyle='--', linewidth=2)

                ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
                ax2.set_xlabel("Qubit Frequency (MHz)", fontsize=20)
                ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)

                if show_legends:
                    ax1.legend()
                    ax2.legend()

                # Combine and sort peak frequencies for the title
                peak_str = ", ".join(f"{pk:.2f}" for pk in sorted(peak_centers))
                fig_title = (
                    f"Qubit Spectroscopy Q{q_key + 1}, peaks from {chosen_signal} \n"
                    f"Peaks at: {peak_str} MHz\n"
                    f"Baseline: {baseline:.2f}\n"
                    f"{exp_config['qubit_spec_ge']['reps']}*{exp_config['qubit_spec_ge']['rounds']} avgs"
                )
                plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2
                fig.text(
                    plot_middle, 0.98, fig_title,
                    fontsize=14, ha='center', va='top'
                )

                plt.tight_layout()
                plt.subplots_adjust(top=0.85)

                # Save figure if enabled
                if save_figs:
                    save_folder = os.path.join(outerFolder, 'QSpec')
                    create_folder_if_not_exists(save_folder)
                    now = datetime.datetime.now()
                    formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                    file_name = os.path.join(
                        save_folder,
                        f"R_{round_num}_Q_{q_key + 1}_{formatted_datetime}_QSpec_q{q_key + 1}.png"
                    )
                    fig.savefig(file_name, dpi=100, bbox_inches='tight')

                plt.close(fig)

if __name__ == "__main__":
    main()
