from section_008_save_data_to_h5 import Data_H5
import matplotlib.dates as mdates
from typing import List
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import glob
import sys
# from section_011_qubit_temperatures_efRabipt3 import Temps_EFAmpRabiExperiment #uses qick modoule
from section_011_qubit_temperatures_efRabipt3_noqick_analysis import Temps_EFAmpRabiExperiment
import math
from collections import defaultdict
from bisect import bisect_left
from scipy.stats import norm
import pytz
# from build_task import *
# from build_state_noqick import *
from expt_config import *
import matplotlib.pyplot as plt
import numpy as np
import ast
from scipy.optimize import curve_fit
import datetime
import re
import logging
import os

sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))

class QubitSpectroscopy:
    def __init__(self, QubitIndex, number_of_qubits,  outerFolder,  round_num, signal, save_figs, experiment = None,
                 live_plot = None, verbose = False, logger = None, qick_verbose=True, increase_reps = False,
                 increase_reps_to = 500, plot_fit=True, zeno_stark=False, zeno_stark_pulse_gain=None,
                 ext_q_spec=False, high_gain_q_spec=False, fit_data=True):

        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.plot_fit=plot_fit
        self.zeno_stark = zeno_stark
        self.zeno_stark_pulse_gain = zeno_stark_pulse_gain
        self.ext_q_spec = ext_q_spec
        self.fit_data = fit_data
        self.high_gain_q_spec = high_gain_q_spec
        if self.zeno_stark:
            self.expt_name = "qubit_spec_ge_zeno_stark"
        elif self.ext_q_spec:
            self.expt_name = "qubit_spec_ge_extended"
        elif self.high_gain_q_spec:
            self.expt_name = "qubit_spec_ge_high_gain"
        else:
            self.expt_name = "qubit_spec_ge"
        self.signal = signal
        self.save_figs = save_figs
        self.experiment = experiment
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.number_of_qubits = number_of_qubits
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")
        self.increase_reps = increase_reps
        self.increase_reps_to = increase_reps_to

        if experiment is not None:
            if self.zeno_stark:
                qze_mask = np.arange(0, self.number_of_qubits + 1)
                qze_mask = np.delete(qze_mask, QubitIndex)
                self.exp_cfg['qze_mask'] = qze_mask
                self.experiment.readout_cfg['res_gain_qze'] = [self.experiment.readout_cfg['res_gain_ge'][QubitIndex],
                                                               0, 0, 0, 0, 0, self.zeno_stark_pulse_gain]
                self.experiment.readout_cfg['res_freq_qze'] = self.experiment.readout_cfg['res_freq_ge']
                self.experiment.readout_cfg['res_phase_qze'] = self.experiment.readout_cfg['res_phase']
                if len(self.experiment.readout_cfg['res_freq_qze']) < 7:  # otherise it keeps appending
                    self.experiment.readout_cfg['res_freq_qze'].append(
                        experiment.readout_cfg['res_freq_qze'][self.QubitIndex])
                    self.experiment.readout_cfg['res_phase_qze'].append(
                        experiment.readout_cfg['res_phase_qze'][self.QubitIndex])

            # self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.live_plot = live_plot
            # self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            # self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            # if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} Qubit Spec configuration: ', self.config)
            # self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} Qubit Spec configuration: {self.config}')


    def plot_results(self, I, Q, freqs, config=None, fig_quality=100, sigma_guess=1, return_fwhm=False):
        freqs = np.array(freqs)
        freq_q = freqs[np.argmax(I)]

        mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err = self.fit_lorenzian(I, Q, freqs,
                                                                                                          freq_q,sigma_guess)

        # Check if the returned values are all None
        if (mean_I is None and mean_Q is None and I_fit is None and Q_fit is None
                and largest_amp_curve_mean is None and largest_amp_curve_fwhm is None):
            # If so, return None for the values in this definition as well
            return None, None, None

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
        if self.plot_fit:
            ax1.plot(freqs, I_fit, 'r--', label='Lorentzian Fit')
            ax1.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

            ax2.plot(freqs, Q_fit, 'r--', label='Lorentzian Fit')
            ax2.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

        # Calculate the middle of the plot area
        plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

        if self.plot_fit:
            # Add title, centered on the plot area
            if config is not None:  # then its been passed to this definition, so use that
                fig.text(plot_middle, 0.98,
                         f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                         f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                         f", {config['reps']}*{config['rounds']} avgs",
                         fontsize=24, ha='center', va='top')
            else:
                fig.text(plot_middle, 0.98,
                         f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                         f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                         f", {self.config['reps']}*{self.config['rounds']} avgs",
                         fontsize=24, ha='center', va='top')
        else:
            # Add title, centered on the plot area
            if config is not None:  # then its been passed to this definition, so use that
                fig.text(plot_middle, 0.98,
                         f"Qubit Spectroscopy Q{self.QubitIndex + 1}" +
                         f", {config['reps']}*{config['rounds']} avgs",
                         fontsize=24, ha='center', va='top')
            else:
                fig.text(plot_middle, 0.98,
                         f"Qubit Spectroscopy Q{self.QubitIndex + 1}",
                         fontsize=24, ha='center', va='top')


                # Adjust spacing
        plt.tight_layout()

        # Adjust the top margin to make room for the title
        plt.subplots_adjust(top=0.93)

        ### Save figure
        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" +
                                     f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)
        if return_fwhm:
            return largest_amp_curve_mean, I_fit, Q_fit, largest_amp_curve_fwhm
        else:
            return largest_amp_curve_mean, I_fit, Q_fit

    def get_results(self, I, Q, freqs):
        freqs = np.array(freqs)
        freq_q = freqs[np.argmax(I)]

        mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err = self.fit_lorenzian(I, Q, freqs, freq_q)

        return largest_amp_curve_mean, I_fit, Q_fit, qspec_fit_err


    def lorentzian(self, f, f0, gamma, A, B):

        return A * gamma ** 2 / ((f - f0) ** 2 + gamma ** 2) + B

    def max_offset_difference_with_x(self, x_values, y_values, offset):
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

    def fit_lorenzian(self, I, Q, freqs, freq_q, sigma_guess = 1):
        try:
            # Initial guesses for I and Q
            initial_guess_I = [freq_q, sigma_guess, np.max(I), np.min(I)]
            initial_guess_Q = [freq_q, sigma_guess, np.max(Q), np.min(Q)]

            # First round of fits (to get rough estimates)
            params_I, _ = curve_fit(self.lorentzian, freqs, I, p0=initial_guess_I)
            params_Q, _ = curve_fit(self.lorentzian, freqs, Q, p0=initial_guess_Q)

            # Use these fits to refine guesses
            x_max_diff_I, max_diff_I = self.max_offset_difference_with_x(freqs, I, params_I[3])
            x_max_diff_Q, max_diff_Q = self.max_offset_difference_with_x(freqs, Q, params_Q[3])
            initial_guess_I = [x_max_diff_I, sigma_guess, np.max(I), np.min(I)]
            initial_guess_Q = [x_max_diff_Q, sigma_guess, np.max(Q), np.min(Q)]

            # Second (refined) round of fits, this time capturing the covariance matrices
            params_I, cov_I = curve_fit(self.lorentzian, freqs, I, p0=initial_guess_I)
            params_Q, cov_Q = curve_fit(self.lorentzian, freqs, Q, p0=initial_guess_Q)

            # Create the fitted curves
            I_fit = self.lorentzian(freqs, *params_I)
            Q_fit = self.lorentzian(freqs, *params_Q)

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
            if 'None' in self.signal or self.signal is None:
                if amp_I_fit > amp_Q_fit:
                    largest_amp_curve_mean = mean_I
                    largest_amp_curve_fwhm = fwhm_I
                    # error on the Q fit's center frequency (first parameter):
                    qspec_fit_err = fit_err_I[0]
                else:
                    largest_amp_curve_mean = mean_Q
                    largest_amp_curve_fwhm = fwhm_Q
                    qspec_fit_err = fit_err_Q[0]
            elif 'I' in self.signal:
                largest_amp_curve_mean = mean_I
                largest_amp_curve_fwhm = fwhm_I
                qspec_fit_err = fit_err_I[0]
            elif 'Q' in self.signal:
                largest_amp_curve_mean = mean_Q
                largest_amp_curve_fwhm = fwhm_Q
                qspec_fit_err = fit_err_Q[0]
            else:
                print('Invalid signal passed, please choose "I", "Q", or "None".')
                return None

            # Return all desired results including the error on the Q fit
            return mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err

        except Exception as e:
            if self.verbose: print("Error during Lorentzian fit:", e)
            self.logger.info(f'Error during Lorentzian fit: {e}')
            return None, None,None,None,None,None,None

    def create_folder_if_not_exists(self, folder_path):
        import os
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


class PlotRR_noQick:
    def __init__(self,  date, figure_quality, save_figs, fit_saved, signal, run_name, number_of_qubits, outerFolder,
                 outerFolder_save_plots, unique_folder_path):
        self.date = date
        self.figure_quality = figure_quality
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.run_name = run_name
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.outerFolder_save_plots = outerFolder_save_plots
        self.unique_folder_path = unique_folder_path # use this when you need to use a different path for anything

    def process_string_of_nested_lists(self, data):
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

    def process_h5_data(self, data):
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

    def string_to_float_list(self, input_string):
        try:
            # Remove 'np.float64()' parts
            cleaned_string = input_string.replace('np.float64(', '').replace(')', '')

            # Use ast.literal_eval for safe evaluation
            float_list = ast.literal_eval(cleaned_string)

            # Check if all elements are floats (or can be converted to floats)
            return [float(x) for x in float_list]
        except (ValueError, SyntaxError, TypeError):
            print("Error: Invalid input string format.  It should be a string representation of a list of numbers.")
            return None

    def run(self, plot_res_spec=True, plot_q_spec=True, plot_rabi=True, rabi_rolling_avg=False, plot_ss=True,
            plot_ss_hist_only=False, ss_plot_title=None, ss_plot_gef=True, plot_t1=True,
            plot_t2r=True, plot_t2e=True, plot_rabis_Qtemps=False):

        # if plot_res_spec:
        #     self.load_plot_save_res_spec()
        # if plot_q_spec:
        #     self.load_plot_save_q_spec()
        if plot_rabis_Qtemps:
            list_of_all_qubits = [i for i in range(self.number_of_qubits + 1)]
            self.load_plot_save_rabis_Qtemps(list_of_all_qubits)
        # if plot_rabi:
        #     if rabi_rolling_avg:
        #         self.load_plot_save_rabi(rabi_rolling_avg=True)
        #     else:
        #         self.load_plot_save_rabi()
        # if plot_ss:
        #     self.load_plot_save_ss(plot_ss_hist_only=plot_ss_hist_only, plot_title=ss_plot_title)
        # if ss_plot_gef:
        #     self.load_plot_save_ss_gef(plot_ssf_gef=ss_plot_gef)
        # if plot_t1:
        #     self.load_plot_save_t1()
        # if plot_t2r:
        #     self.load_plot_save_t2r()
        # if plot_t2e:
        #     self.load_plot_save_t2e()

    def load_plot_save_q_spec(self):
        # ----------------------------------------------Load/Plot/Save QSpec------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/QSpec_ge/"
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
        extracted_freqs = []
        for h5_file in h5_files:
            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='QSpec', save_r=int(save_round))

            populated_keys = []
            for q_key in load_data['QSpec']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['QSpec'][q_key].get('Dates', [[]])

                # Check if any entry in 'Dates' is not NaN
                if any(
                        not np.isnan(date)
                        for date in dates_list[0]  # Iterate over the first batch of dates
                ):
                    populated_keys.append(q_key)

            for q_key in populated_keys:
                for dataset in range(len(load_data['QSpec'][q_key].get('Dates', [])[0])):
                    date = datetime.datetime.fromtimestamp(load_data['QSpec'][q_key].get('Dates', [])[0][dataset])
                    I = self.process_h5_data(load_data['QSpec'][q_key].get('I', [])[0][dataset].decode())
                    Q = self.process_h5_data(load_data['QSpec'][q_key].get('Q', [])[0][dataset].decode())
                    # I_fit = load_data['QSpec'][q_key].get('I Fit', [])[0][dataset]
                    # Q_fit = load_data['QSpec'][q_key].get('Q Fit', [])[0][dataset]
                    freqs = self.process_h5_data(load_data['QSpec'][q_key].get('Frequencies', [])[0][dataset].decode())
                    round_num = load_data['QSpec'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['QSpec'][q_key].get('Batch Num', [])[0][dataset]

                    exp_config = load_data['QSpec'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}

                    exp_config = eval(exp_config, safe_globals)

                    if len(I) > 0:
                        qspec_class_instance = QubitSpectroscopy(q_key, self.number_of_qubits,
                                                                 self.outerFolder_save_plots, round_num, self.signal,
                                                                 self.save_figs)
                        q_spec_cfg = exp_config['qubit_spec_ge']
                        # print('q_spec_cfg: ', q_spec_cfg)
                        qubit_freq, _, _ = qspec_class_instance.plot_results(I, Q, freqs, q_spec_cfg,
                                                                             self.figure_quality)
                        del qspec_class_instance

                        extracted_freqs.append({
                            "filename": os.path.basename(h5_file),
                            "q_key": int(q_key),
                            "dataset": dataset,
                            "round_num": round_num,
                            "batch_num": batch_num,
                            "freq_MHz": qubit_freq,
                            "timestamp": date.timestamp()
                        })

            del H5_class_instance

        return extracted_freqs

    def load_plot_save_rabis_Qtemps(self, list_of_all_qubits):
        # ------------------------------------------------Load/Plot/Save Rabi---------------------------------------
        outerFolder_expt_qtemps = self.unique_folder_path+ "/Data_h5/q_temperatures/"
        h5_files = glob.glob(os.path.join(outerFolder_expt_qtemps, "*.h5"))
        all_files_Qtemp_results = [] #to store qubit temperature results
        cutoff_timestamp = datetime.datetime(2025, 4, 11, 19, 0).timestamp()  # when I started saving qubit freqs in the same files

        extracted_qspec_results = self.load_plot_save_q_spec()
        qspec_grouped_by_qkey = defaultdict(list)
        #sort each list by qubit
        for item in extracted_qspec_results:
            qspec_grouped_by_qkey[item['q_key']].append(item)
        # Sort each list by timestamp
        for qkey in qspec_grouped_by_qkey:
            qspec_grouped_by_qkey[qkey].sort(key=lambda x: x['timestamp'])

        for h5_file in h5_files:

            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='q_temperatures', save_r=int(save_round))

            file_result = {'filename': os.path.basename(h5_file), 'qubits': {}}

            populated_keys = []
            for q_key in load_data['q_temperatures']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['q_temperatures'][q_key].get('Dates', [[]])

                # Check if any entry in 'Dates' is not NaN
                if any(
                        not np.isnan(date)
                        for date in dates_list[0]  # Iterate over the first batch of dates
                ):
                    populated_keys.append(q_key)

            A_amplitude1 = None
            A_amplitude2 = None

            for q_key in populated_keys:
                # print(f"Extracting data for QubitIndex: {q_key}")
                for dataset in range(len(load_data['q_temperatures'][q_key].get('Dates', [])[0])):
                    date = datetime.datetime.fromtimestamp(load_data['q_temperatures'][q_key].get('Dates', [])[0][dataset])
                    round_num = load_data['q_temperatures'][q_key].get('Round Num', [])[0][dataset]
                    # batch_num = load_data['q_temperatures'][q_key].get('Batch Num', [])[0][dataset]
                    #-------------------------------------Grabbing matching qubit frequency for this qubit-------------------------------------
                    if date.timestamp() > cutoff_timestamp: #files after this date contain the matching g-e qubit frequency already
                        qubit_freq_MHz = load_data['q_temperatures'][q_key].get('Qfreq_ge', [])[0][dataset]
                        # print(f"QSpec Q{q_key}: {qubit_freq_MHz:.3f} MHz")
                    else: #look through matching qspec file
                        qtemp_timestamp = date.timestamp()  # Timestamp of this q_temperatures entry

                        # Get all QSpec entries for this qubit
                        qspec_entries = qspec_grouped_by_qkey.get(int(q_key), [])

                        if not qspec_entries:
                            print(f"No QSpec entries found for Q{q_key}")
                            continue

                        # Extract sorted timestamps to use with bisect
                        qspec_timestamps = [entry['timestamp'] for entry in qspec_entries]

                        # Use bisect to find the insertion index
                        idx = bisect_left(qspec_timestamps, qtemp_timestamp)

                        # Search nearby indices (at most 3 comparisons)
                        closest_match = None
                        min_time_diff = float("inf")
                        for i in [idx - 1, idx, idx + 1]:
                            if 0 <= i < len(qspec_entries):
                                time_diff = abs(qspec_entries[i]['timestamp'] - qtemp_timestamp)
                                if time_diff < 60 and time_diff < min_time_diff:
                                    closest_match = qspec_entries[i]
                                    min_time_diff = time_diff

                        if closest_match is not None:
                            qubit_freq_MHz = closest_match['freq_MHz']
                            # print(f"Matched QSpec Q{q_key}: {qubit_freq_MHz:.3f} MHz")
                        else:
                            print(f"No timestamp match in QSpec for Q{q_key} near {qtemp_timestamp}")
                            continue
                    #---------------------------------------------------------------------------------------------

                    I1 = self.process_h5_data(load_data['q_temperatures'][q_key].get('I1', [])[0][dataset].decode())
                    Q1 = self.process_h5_data(load_data['q_temperatures'][q_key].get('Q1', [])[0][dataset].decode())
                    gains1 = self.process_h5_data(load_data['q_temperatures'][q_key].get('Gains1', [])[0][dataset].decode())

                    I2 = self.process_h5_data(load_data['q_temperatures'][q_key].get('I2', [])[0][dataset].decode())
                    Q2 = self.process_h5_data(load_data['q_temperatures'][q_key].get('Q2', [])[0][dataset].decode())
                    gains2 = self.process_h5_data(load_data['q_temperatures'][q_key].get('Gains2', [])[0][dataset].decode())

                    # syst_config = load_data['q_temperatures'][q_key].get('Syst Config', [])[0][dataset].decode()
                    exp_config = load_data['q_temperatures'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                    exp_config = eval(exp_config, safe_globals)
                    rabi_cfg = exp_config['power_rabi_ef']
                    if len(I1) > 0:
                        save_figs = False
                        rabi_class_instance = Temps_EFAmpRabiExperiment(q_key, self.number_of_qubits, list_of_all_qubits,
                                                                      self.outerFolder_save_plots, round_num,
                                                                      self.signal, save_figs)
                        I1 = np.asarray(I1)
                        Q1 = np.asarray(Q1)
                        gains1 = np.asarray(gains1)
                        best_signal_fit1, pi_amp1, A_amplitude1, amp_fit1 = rabi_class_instance.plot_results(I1, Q1, gains1, rabi_cfg, self.figure_quality)
                        del rabi_class_instance

                    if len(I2) > 0:
                        save_figs = False
                        rabi_class_instance = Temps_EFAmpRabiExperiment(q_key, self.number_of_qubits,
                                                                        list_of_all_qubits,
                                                                        self.outerFolder_save_plots, round_num,
                                                                        self.signal, save_figs)
                        I2 = np.asarray(I2)
                        Q2 = np.asarray(Q2)
                        gains2 = np.asarray(gains2)
                        best_signal_fit2, pi_amp2, A_amplitude2, amp_fit2 = rabi_class_instance.plot_results(I2, Q2, gains2, rabi_cfg, self.figure_quality)
                        del rabi_class_instance

                    if A_amplitude1 is not None and A_amplitude2 is not None:
                        A_e = A_amplitude1
                        A_g = A_amplitude2

                        results = self.Qubit_Temperature_Convert(A_e, A_g, qubit_freq_MHz)
                        if results is None:
                            continue  # Skip this dataset
                        T_K, T_mK, P_e, qubit_freq = results
                        print(f"Q{q_key} calculated Temperature:{T_mK}, with P_e = {P_e}, and Qfreq {qubit_freq_MHz} MHz")
                        file_result['qubits'][int(q_key)] = {
                            'A1': A_amplitude1,
                            'A2': A_amplitude2,
                            'T_mK': T_mK,
                            'P_e': P_e,
                            'qubit_freq_MHz': qubit_freq,
                            'date': date.timestamp()}

            all_files_Qtemp_results.append(file_result)
            del H5_class_instance
        return all_files_Qtemp_results

    def Qubit_Temperature_Convert(self, A_e, A_g, qubit_freq_MHz):
        P_e = np.abs(A_e / (A_e + A_g))  # Excited state population (leakage, thermal population)
        P_g = (1 - P_e)
        if P_e <= 0 or P_g <= 0: #if one of them is zero can't calculate the temp
            print("Warning: Invalid population values encountered (<= 0). Skipping this dataset.")
            return None

        ratio = P_g / P_e
        if ratio <= 1: #denominator would become zero at Pg=Pe
            print(f"Warning: Non-physical ratio (P_g/P_e = {ratio:.3f} <= 1) encountered. Skipping this dataset.")
            return None

        qubit_freq_Hz = qubit_freq_MHz * 2 * np.pi * 1e6  # Omega_q in the unit Hz
        k_B = 1.38 * 10 ** -23
        hbar = 1.05 * 10 ** -34
        T_K = hbar * qubit_freq_Hz / (k_B * np.log(P_g/ P_e))  # Temperature in the unit Kelvin
        T_mK = T_K * 1000  # Convert to millikelvin
        return T_K, T_mK, P_e, qubit_freq_MHz

    def plot_qubit_temperatures_vs_time(self, all_files_Qtemp_results, num_qubits=6):
        """
        Plots qubit temperatures vs. time for each qubit in a separate subplot (max 3 columns).

        Parameters:
        - all_files_Qtemp_results: list of dicts returned by `load_plot_save_rabis_Qtemps`
        - num_qubits: total number of qubits to plot (default is 6)
        """

        # Define the colors you want for each qubit
        colors = ["orange", "blue", "purple", "green", "brown", "pink"]

        ncols = min(num_qubits, 3)
        nrows = math.ceil(num_qubits / 3)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(4 * ncols, 4 * nrows),
                                 sharex=False, constrained_layout=True)  # set sharex=False if you want each subplot to manage ticks independently
        axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

        fig.suptitle("Qubit Temperatures vs. Time", fontsize=16)

        # cdt = pytz.timezone('America/Chicago')

        # def localize_cdt(dt):
        #     return dt if dt.tzinfo else cdt.localize(dt)

        # radiation source timestamps
        co60_time = datetime.datetime(2025, 4, 21, 12, 35)
        cs137_time = datetime.datetime(2025, 4, 23, 12, 53)

        events_0418 = [
            ("11:50", "Daniel Entry"),
            ("13:30", "Daniel Exit"),
            ("14:53", "Daniel Entry"),
            ("15:00", "Door Intermission"),
            ("15:06", "Exit/Re-entry Daniel"),
            ("15:12", "Ryan"),
            ("15:40", "Door Intermission"),
            ("16:11", "Daniel Exit"),
            ("16:12", "Daniel Re-entry"),
            ("16:16", "Daniel Final Exit")]

        events_0423 = [
            ("12:50", "Dan-Joyce Entry"),
            ("12:54", "Dan-Joyce Exit"),
            ("13:40", "Grace Entry"),
            ("13:47", "Grace Exit"),
            ("16:40", "Kester-Grace Entry"),
            ("16:48", "Kester-Grace Exit")]



        for q in range(num_qubits):
            times = []
            temps = []

            for file_result in all_files_Qtemp_results:
                qubit_data = file_result['qubits'].get(q)
                if qubit_data:
                    timestamp = qubit_data['date']
                    T_mK = qubit_data['T_mK']
                    times.append(datetime.datetime.fromtimestamp(timestamp))
                    temps.append(T_mK)

            ax = axes[q]

            if not times:
                ax.set_visible(False)
                continue

            # --- Optional: Restrict plot to specific date and time window ---
            restrict_time_xaxis = True  # Set to False to show full range
            date_to_plot = datetime.date(2025, 4, 23)
            time_start = datetime.time(0, 0)  # Start of the window
            time_end = datetime.time(23, 59)  # End of the window

            if restrict_time_xaxis:
                start_time = datetime.datetime.combine(date_to_plot, time_start)
                end_time = datetime.datetime.combine(date_to_plot, time_end)
                #Use finer ticks with hour detail
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
            else:
                #Use coarse ticks with just date
                ax.xaxis.set_major_locator(mdates.DayLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

            # Use scatter instead of plot to avoid connecting lines
            ax.scatter(times, temps, marker='o', color=colors[q % len(colors)], label=f"Q{q + 1}")

            ax.set_title(f"Q{q + 1}", fontsize=14)
            ax.set_ylabel("Temp (mK)", fontsize=12)
            ax.grid(False)

            # Format the x-axis to show dates in a nice format
            # ax.set_ylim(25, 300)
            ax.set_yticks(np.linspace(25, 500, 12))

            # start_time = datetime.datetime(2025, 4, 11, 12, 30)
            # ax.set_xlim(left=start_time)

            ax.tick_params(axis='x', labelrotation=90, labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

            # --- Add vertical lines for known radiation events ---
            for vtime, label in [(co60_time, "Co-60"), (cs137_time, "Cs-137")]:
                ax.axvline(vtime, color='black', linestyle='--', linewidth=1)
                ax.text(vtime, ax.get_ylim()[1] * 0.95, label, rotation=90, verticalalignment='top',
                        horizontalalignment='right', fontsize=10)

            #----------------------Now for other events------------------------
            event_date_0418 = datetime.date(2025, 4, 18)
            event_date_0423 = datetime.date(2025, 4, 23)
            extra_events = [
                *((datetime.datetime.combine(event_date_0418, datetime.time.fromisoformat(t)), label)
                  for t, label in events_0418),
                *((datetime.datetime.combine(event_date_0423, datetime.time.fromisoformat(t)), label)
                  for t, label in events_0423)
            ]

            # Get all extra event labels (exclude Co-60 and Cs-137 from color mapping)
            extra_event_labels = [label for _, label in extra_events]
            unique_labels = list(dict.fromkeys(extra_event_labels))  # maintain order, remove duplicates

            # Create a colormap for the extra events only
            cmap = cm.get_cmap('tab20', len(unique_labels))
            label_to_color = {label: mcolors.to_hex(cmap(i)) for i, label in enumerate(unique_labels)}

            event_lines = []
            for vtime, label in extra_events:
                color = label_to_color[label]
                ax.axvline(vtime, color=color, linestyle='--', linewidth=1)
                event_lines.append(Line2D([0], [0], color=color, linestyle='--', label=label))

            if restrict_time_xaxis:
                ax.set_xlim(start_time, end_time)
                ax.set_autoscale_on(False)

            # Add a combined legend (only once)
            if q == 0:
                ax.legend(
                    handles=event_lines,
                    loc='upper right',
                    bbox_to_anchor=(1.05, 1),
                    fontsize=9,
                    frameon=True,
                    borderaxespad=0.
                )

        # Add a shared X label
        for ax in axes:
            ax.set_xlabel("Time")

        # plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Optionally, auto-format the x-axis date labels
        # fig.autofmt_xdate()

        # Save the figure
        timestp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(self.outerFolder_save_plots, f"QubitTemps_vs_Time_{timestp}.png")
        print("Plot saved to: ", save_path)
        plt.savefig(save_path, dpi=self.figure_quality)
        plt.close(fig)

    def plot_qubit_temperature_histograms(self, all_files_Qtemp_results, num_qubits=6):
        """
        Plots histograms for the temperature (T_mK) data of each qubit.

        Parameters:
        - all_files_Qtemp_results: list of dicts returned by load_plot_save_rabis_Qtemps
        - num_qubits: total number of qubits to plot (default is 6)

        # Note: All datetime objects are naive and assumed to be in Central Time (local system time).
        """
        # Set up the subplots grid (2 rows x 3 columns for 6 qubits)
        ncols = min(num_qubits, 3)
        nrows = math.ceil(num_qubits / 3)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]
        axes: List[Axes] = axes  # Explicitly tell the IDE that these are Axes objects

        # Define font size and colors (same order as in your temperature-vs-time plots)
        font = 14
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        plt.suptitle("Qubit Temperature Histograms", fontsize=font)

        # Titles for each subplot
        titles = [f"Qubit {i + 1}" for i in range(num_qubits)]
        mean_values = {}
        std_values = {}

        # Loop over each qubit / subplot
        for i, ax in enumerate(axes):
            # Gather all temperature data for qubit i across all files.
            temp_vals = []
            for file_result in all_files_Qtemp_results:
                qubit_data = file_result['qubits'].get(i)
                if qubit_data and 'T_mK' in qubit_data:
                    temp_vals.append(qubit_data['T_mK'])

            # If no data is present, hide the subplot.
            if len(temp_vals) == 0:
                plt.setp(ax, visible=False)
                continue

            # Choose a fixed number of bins (you can adjust this number)
            optimal_bin_num = 20

            # Fit a Gaussian to the temperature data
            mu, std = norm.fit(temp_vals)
            mean_values[f"Qubit {i + 1}"] = mu
            std_values[f"Qubit {i + 1}"] = std

            # Generate x values for plotting the Gaussian curve
            x_vals = np.linspace(min(temp_vals), max(temp_vals), optimal_bin_num)
            # Compute the probability density function for the fitted Gaussian
            pdf_vals = norm.pdf(x_vals, mu, std)

            # Compute histogram data to determine scaling (so the Gaussian curve overlays properly)
            hist_data, bins = np.histogram(temp_vals, bins=optimal_bin_num)
            bin_width = np.diff(bins)[0]
            scale_factor = hist_data.sum() * bin_width
            # Scale the PDF accordingly
            scaled_pdf = pdf_vals * scale_factor

            # Plot the Gaussian fit (dashed line) and the histogram
            ax.plot(x_vals, scaled_pdf, linestyle='--', linewidth=2, color=colors[i % len(colors)])
            ax.hist(temp_vals, bins=optimal_bin_num, alpha=0.7, color=colors[i % len(colors)],
                    edgecolor='black')

            # Set subplot title and labels including the Gaussian parameters
            ax.set_title(f"{titles[i]}  $\mu$: {mu:.2f} mK,  $\sigma$: {std:.2f} mK", fontsize=font)
            ax.set_xlabel("Temperature (mK)", fontsize=font)
            ax.set_ylabel("Frequency", fontsize=font)
            ax.tick_params(axis='both', which='major', labelsize=font)

        plt.tight_layout()
        # Save the figure with a timestamp in the filename
        timestp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(self.outerFolder_save_plots, f"QubitTemps_Histograms_{timestp}.png")
        print("Histogram plot saved to:", save_path)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

    def plot_qubit_pe_vs_time(self, all_files_Qtemp_results, num_qubits=6):
        """
        Plots qubit excited state populations (P_e) vs. time in a separate figure.

        Parameters:
        - all_files_Qtemp_results: list of dicts returned by `load_plot_save_rabis_Qtemps`
        - num_qubits: number of qubits to include in the plot (default is 6)
        """

        colors = ["orange", "blue", "purple", "green", "brown", "pink"]
        font = 14

        ncols = min(num_qubits, 3)
        nrows = math.ceil(num_qubits / 3)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(4 * ncols, 4 * nrows),
                                 sharex=False, constrained_layout=True)

        axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

        fig.suptitle("Qubit P_e vs. Time", fontsize=font + 2)

        for q in range(num_qubits):
            times = []
            pe_values = []

            for file_result in all_files_Qtemp_results:
                qubit_data = file_result['qubits'].get(q)
                if qubit_data:
                    timestamp = qubit_data['date']
                    P_e = qubit_data.get('P_e', None)
                    if P_e is not None:
                        times.append(datetime.datetime.fromtimestamp(timestamp))
                        pe_values.append(P_e)

            ax = axes[q]
            ax.scatter(times, pe_values, marker='o', color=colors[q % len(colors)], label=f"Q{q + 1}")
            ax.set_title(f"Q{q + 1}", fontsize=font)
            ax.set_ylabel("$P_e$", fontsize=font)
            ax.set_ylim(0, 0.6)

            start_time = datetime.datetime(2025, 4, 11, 12, 30)
            ax.set_xlim(left=start_time)

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.tick_params(axis='x', labelrotation=45, labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

        for ax in axes:
            ax.set_xlabel("Time", fontsize=font)

        timestp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(self.outerFolder_save_plots, f"QubitPe_vs_Time_{timestp}.png")
        print("Plot saved to:", save_path)
        plt.savefig(save_path, dpi=self.figure_quality)
        plt.close(fig)
        # plt.show()

    def plot_qubit_temp_and_pe_vs_time(self, all_files_Qtemp_results, num_qubits=6):
        """
        Plots qubit temperature (T_mK) and P_e vs. time using scatter points for each qubit (dual y-axes).
        """
        colors = ["orange", "blue", "purple", "green", "brown", "pink"]
        font = 14

        ncols = min(num_qubits, 3)
        nrows = math.ceil(num_qubits / 3)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(4 * ncols, 4 * nrows),
                                 constrained_layout=True)

        axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]
        fig.suptitle("Qubit Temperature and $P_e$ vs. Time", fontsize=font + 2)

        for q in range(num_qubits):
            times = []
            temps = []
            pe_values = []

            for file_result in all_files_Qtemp_results:
                qubit_data = file_result['qubits'].get(q)
                if qubit_data:
                    timestamp = qubit_data['date']
                    times.append(datetime.datetime.fromtimestamp(timestamp))
                    temps.append(qubit_data['T_mK'])
                    pe_values.append(qubit_data.get('P_e', None))

            if not times:
                axes[q].set_visible(False)
                continue


            ax1 = axes[q]
            ax2 = ax1.twinx()

            ax1.set_title(f"Q{q + 1}", fontsize=font)
            ax1.set_xlabel("Time", fontsize=font)

            # Temperature (left axis)
            ax1.set_ylabel("Temp (mK)", color=colors[q % len(colors)], fontsize=font)
            ax1.scatter(times, temps, color=colors[q % len(colors)], marker='o')
            ax1.tick_params(axis='y', labelcolor=colors[q % len(colors)])
            ax1.set_ylim(80, 400)

            # Pe (right axis)
            start_time = datetime.datetime(2025, 4, 11, 12, 30)
            ax1.set_xlim(left=start_time)

            ax2.set_ylabel("$P_e$", color="black", fontsize=font)
            ax2.scatter(times, pe_values, color="black", marker='x')
            ax2.tick_params(axis='y', labelcolor="black")
            ax2.set_ylim(0, 0.6)

            # Format x-axis
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax1.tick_params(axis='x', rotation=45, labelsize=10)

        timestp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(self.outerFolder_save_plots, f"QubitTemps_and_Pe_vs_Time_{timestp}.png")
        print("Combined plot saved to:", save_path)
        plt.savefig(save_path, dpi=self.figure_quality)
        plt.close(fig)
        # plt.show()

    def plot_qubit_temp_pe_freq_vs_time(self, all_files_Qtemp_results, num_qubits=6):
        """
        Plots qubit temperature (T_mK), P_e, and qubit frequency vs. time using triple y-axes.
        """
        colors = ["orange", "blue", "purple", "green", "brown", "pink"]
        font = 14

        ncols = min(num_qubits, 3)
        nrows = math.ceil(num_qubits / 3)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(5 * ncols, 4.5 * nrows),
                                 constrained_layout=True)

        axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]
        fig.suptitle("Qubit Temp, $P_e$, and Freq vs. Time", fontsize=font + 2)

        for q in range(num_qubits):
            times, temps, pe_values, freqs = [], [], [], []
            yaxis_limit = 700

            for file_result in all_files_Qtemp_results:
                qubit_data = file_result['qubits'].get(q)
                if qubit_data:
                    T_mK = qubit_data['T_mK']
                    if T_mK <= yaxis_limit:
                        timestamp = qubit_data['date']
                        times.append(datetime.datetime.fromtimestamp(timestamp))
                        temps.append(T_mK)
                        pe_values.append(qubit_data['P_e'])
                        freqs.append(qubit_data.get('qubit_freq_MHz'))

                    # P_e = pe_values[-1]
                    # if 0.4 <= P_e <= 0.55:
                    #     print(
                    #         f"Q{q}  |  P_e = {P_e:.3f}  |  T_mK = {qubit_data['T_mK']:.2f}  |  Freq = {qubit_data['qubit_freq_MHz']:.3f} MHz  |  Timestamp = {datetime.datetime.fromtimestamp(qubit_data['date'])}")

            if not times:
                axes[q].set_visible(False)
                continue

            ax1 = axes[q]
            ax2 = ax1.twinx()  # Right y-axis for P_e
            ax3 = ax1.twinx()  # New outer-right axis for qubit frequency
            ax3.spines.right.set_position(("outward", 60))  # offset third axis

            # Temp (left axis)
            ax1.set_ylabel("Temp (mK)", color=colors[q % len(colors)], fontsize=font)
            ax1.scatter(times, temps, color=colors[q % len(colors)], marker='o')
            ax1.tick_params(axis='y', labelcolor=colors[q % len(colors)])
            # ax1.set_ylim(100, 400)

            # Pe (middle right axis)
            ax2.set_ylabel("$P_e$", color="black", fontsize=font)
            ax2.scatter(times, pe_values, color="black", marker='x')
            ax2.tick_params(axis='y', labelcolor="black")
            ax2.set_ylim(0, 0.6)

            # Freq (outer right axis)
            ax3.set_ylabel("Qubit Freq (MHz)", color="gray", fontsize=font)
            ax3.scatter(times, freqs, color="gray", marker='^')
            ax3.tick_params(axis='y', labelcolor="gray")
            ax3.set_ylim(min(freqs) * 0.998, max(freqs) * 1.002)  # dynamic range

            # Time axis (x)
            start_time = datetime.datetime(2025, 4, 11, 12, 30)
            ax1.set_xlim(left=start_time)
            ax1.set_xlabel("Time", fontsize=font)
            ax1.set_title(f"Q{q + 1} Temp, Qfreq & P_e vs. Time", fontsize=font)

            # Add vertical dashed lines for experiment events
            experiment_date = datetime.date(2025, 4, 11)
            event_info = [
                ("13:11", "DC Bias Sweep", "red"),
                ("14:51", "Pump Freq Sweep (early)", "blue"),
                ("15:20", "Pump Freq Sweep", "blue"),
                ("15:43", "Pump Freq Sweep", "blue"),
                ("16:19", "Pump Power Sweep", "green"),
                ("16:53", "Pump Power Sweep", "green"),
                ("17:10", "DC Bias Sweep", "red")
            ]

            plotted_labels = set()

            for time_str, label, color in event_info:
                dt = datetime.datetime.strptime(f"{experiment_date} {time_str}", "%Y-%m-%d %H:%M")
                label_to_use = label if label not in plotted_labels else None
                ax1.axvline(x=dt, color=color, linestyle='--', linewidth=1.5, label=label_to_use)
                if label_to_use:
                    plotted_labels.add(label)

            # Only show legend on first subplot (optional)
            if q == 0:
                ax1.legend(loc='upper left', fontsize=10)

            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax1.tick_params(axis='x', rotation=45, labelsize=10)

        timestp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        save_path = os.path.join(self.outerFolder_save_plots, f"QubitTemps_Pe_Freq_vs_Time_{timestp}.png")
        print("Combined plot saved to:", save_path)
        plt.savefig(save_path, dpi=self.figure_quality)
        plt.close(fig)