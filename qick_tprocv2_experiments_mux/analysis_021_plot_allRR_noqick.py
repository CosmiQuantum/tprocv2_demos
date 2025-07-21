from section_008_save_data_to_h5 import Data_H5
import matplotlib.dates as mdates
from typing import List
from matplotlib.axes import Axes
import glob
from matplotlib.lines import Line2D
from matplotlib import cm, colors as mcolors
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
from matplotlib.ticker import StrMethodFormatter
import logging
import os

# -----------------This script currently has the capacity to plot T1, Qfreqs, and RPMs (ef Rabi) data--------=====
# Can also do T1 vs time and Q1 vs time

sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
class T1Measurement:
    def __init__(self, QubitIndex, number_of_qubits,  outerFolder, round_num, signal, save_figs, experiment = None,
                 live_plot = None, fit_data = None, increase_qubit_reps = False, qubit_to_increase_reps_for = None,
                 multiply_qubit_reps_by = 0, verbose = False, logger = None, qick_verbose=True, save_shots=False,
                 set_relax_delay=False, relax_delay=1000):

        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.expt_name = "T1_ge"
        self.fit_data = fit_data
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.live_plot = live_plot
        self.signal = signal
        self.save_figs = save_figs
        self.verbose = verbose
        self.save_shots = save_shots
        self.set_relax_delay = set_relax_delay
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        # if experiment is not None:
            # self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            # self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            # self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            # if increase_qubit_reps:
            #         if self.QubitIndex==qubit_to_increase_reps_for:
            #             self.logger.info(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
            #             if self.verbose: print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
            #             self.config["reps"] *= multiply_qubit_reps_by
            # if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} T1 configuration: {self.config}')
            # self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} T1 configuration: {self.config}')
            # if self.set_relax_delay:
            #     self.config['relax_delay'] = relax_delay
            #     print(f'set t1 relax delay to {relax_delay} us')

    def exponential(self, x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def exponential(self, x, a, b, c, d):
        return a * np.exp(- (x - b) / c) + d

    def t1_fit(self, I, Q, delay_times):
        if 'I' in self.signal:
            signal = I
            plot_sig = 'I'
        elif 'Q' in self.signal:
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
        q1_popt, q1_pcov = curve_fit(self.exponential, delay_times, signal,
                                     p0=q1_guess, bounds=(lower_bounds, upper_bounds),
                                     method='trf', maxfev=10000)

        # Generate the fitted exponential curve
        q1_fit_exponential = self.exponential(delay_times, *q1_popt)

        # Extract T1 and its error
        T1_est = q1_popt[2]  # Decay constant T1
        T1_err = np.sqrt(q1_pcov[2][2]) if q1_pcov[2][2] >= 0 else float('inf')  # Ensure error is valid

        return q1_fit_exponential, T1_err, T1_est, plot_sig

    def plot_results(self, I, Q, delay_times, now, config = None, fig_quality =100):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})

        # Calculate the middle of the plot area
        plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2


        if self.fit_data:
            q1_fit_exponential, T1_err, T1_est, plot_sig = self.t1_fit(I, Q, delay_times)

            if 'I' in plot_sig:
                ax1.plot(delay_times, q1_fit_exponential, '-', color='red', linewidth=3, label="Fit")
            else:
                ax2.plot(delay_times, q1_fit_exponential, '-', color='red', linewidth=3, label="Fit")

            # Add title, centered on the plot area
            if config is not None:
                fig.text(plot_middle, 0.98,
                         f"Q{self.QubitIndex + 1} " + f"T1={T1_est:.2f} us" + f", {float(config['reps'])}*{float(config['rounds'])} avgs,",
                         fontsize=24, ha='center',
                         va='top')  # , pi gain %.2f" % float(config['pi_amp']) + f", {float(config['sigma']) * 1000} ns sigma
            else:
                fig.text(plot_middle, 0.98,
                         f"T1 Q{self.QubitIndex + 1}, T1 %.2f us" % T1_est + f", {self.config['reps']}*{self.config['rounds']} avgs,",
                         fontsize=24, ha='center', va='top')

        else:
            if config is not None:
                fig.text(plot_middle, 0.98,
                         f"T1 Q{self.QubitIndex + 1}" + f", {float(config['reps'])}*{float(config['rounds'])} avgs,",
                         fontsize=24, ha='center',
                         va='top')  # , pi gain %.2f" % float(config['pi_amp']) + f", {float(config['sigma']) * 1000} ns sigma"   you can put this back once you save configs properly for when replotting
            else:
                fig.text(plot_middle, 0.98,
                         f"T1 Q{self.QubitIndex + 1}",
                         fontsize=24, ha='center', va='top')
            q1_fit_exponential = None
            T1_est = None
            T1_err = None

        # I subplot
        ax1.plot(delay_times, I, label="Gain (a.u.)", linewidth=2)
        ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        # ax1.axvline(freq_q, color='orange', linestyle='--', linewidth=2)

        # Q subplot
        ax2.plot(delay_times, Q, label="Q", linewidth=2)
        ax2.set_xlabel("Delay time (us)", fontsize=20)
        ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        # ax2.axvline(freq_q, color='orange', linestyle='--', linewidth=2)

        # Adjust spacing
        plt.tight_layout()

        # Adjust the top margin to make room for the title
        plt.subplots_adjust(top=0.93)
        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')  # , facecolor='white'
        plt.close(fig)


class T1VsTime:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, fridge, exp_name = 'ge'):
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.figure_quality = figure_quality
        self.run_name = run_name
        self.number_of_qubits = number_of_qubits
        self.final_figure_quality = final_figure_quality
        self.top_folder_dates = top_folder_dates
        self.fridge = fridge
        self.exp_name = exp_name

    def datetime_to_unix(self, dt):
        # Convert to Unix timestamp
        unix_timestamp = int(dt.timestamp())
        return unix_timestamp

    def unix_to_datetime(self, unix_timestamp):
        # Convert the Unix timestamp to a datetime object
        dt = datetime.fromtimestamp(unix_timestamp)
        return dt

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def exponential(self, x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def optimal_bins(self, data):
        n = len(data)
        if n == 0:
            return {}
        # Sturges' Rule
        sturges_bins = int(np.ceil(np.log2(n) + 1))
        return sturges_bins

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
            numbers = [float(x.strip('[').strip(']').replace("'", "").replace(" ", "").replace("  ", "")) for x in match.split()] # Convert strings to integers
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

    def run(self, return_errs = False, exp_extension=''):
        import datetime

        # ----------Load/get data------------------------
        t1_vals = {i: [] for i in range(self.number_of_qubits)}
        t1_errs = {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        #print(self.top_folder_dates)
        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/" + folder_date + "/"
                outerFolder_save_plots = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/RR_metrics/Plots/T1_ge"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # ------------------------------------------------Load/Plot/Save T1----------------------------------------------
            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/t1{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/t1_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            #print(outerFolder_expt)
            for h5_file in h5_files:

                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=f't1{exp_extension}', save_r=int(save_round))
                # if '01-27' in outerFolder_expt:
                #     print(load_data)
                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f't1{exp_extension}']:
                    for dataset in range(len(load_data[f't1{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f't1{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T1 = load_data['t1'][q_key].get('t1', [])[0][dataset]
                        # errors = load_data['t1'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data[f't1{exp_extension}'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        I = self.process_h5_data(load_data[f't1{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data[f't1{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                        delay_times = self.process_h5_data(load_data[f't1{exp_extension}'][q_key].get('Delay Times', [])[0][dataset].decode())
                        # fit = load_data['t1'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data[f't1{exp_extension}'][q_key].get('Round Num', [])[0][dataset]
                        try:
                            batch_num = load_data[f't1{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]
                            syst_config = load_data[f't1{exp_extension}'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f't1{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I) > 0:

                            T1_class_instance = T1Measurement(q_key, self.number_of_qubits, outerFolder_save_plots, round_num, self.signal, self.save_figs,
                                                              fit_data=True)
                            #T1_spec_cfg = exp_config['T1_ge']
                            q1_fit_exponential, T1_err, T1_est, plot_sig = T1_class_instance.t1_fit(I, Q, delay_times)
                            if T1_est < 0:
                                print("The value is negative, continuing...")
                                continue
                            if T1_est > 1000:
                                print("The value is above 1000 us, this is a bad fit, continuing...")
                                continue
                            if T1_err >= 0.8 * T1_est:
                                print(
                                    f"Skipping T1 = {T1_est:.3f} µs because its error {T1_err:.3f} µs is >= 80% of its value.")
                                continue

                            t1_vals[q_key].extend([T1_est])
                            t1_errs[q_key].extend([T1_err])
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del T1_class_instance

                del H5_class_instance
        if return_errs:
            return date_times, t1_vals, t1_errs
        else:
            return date_times, t1_vals

    def plot_without_errs(self, date_times, t1_vals, show_legends):
        #---------------------------------plot-----------------------------------------------------
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        #----------------To Plot a specific timeframe------------------
        from datetime import datetime
        year = 2025
        month = 1
        day1 = 22  # Start date
        day2 = 23  # End date
        hour_start = 0  # Start hour
        hour_end = 23  # End hour
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 59)
        #-----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i+1}" for i in range(self.number_of_qubits)]
        colors = ['orange','blue','purple','green','brown','pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('T1 Values vs Time',fontsize = font)
        axes = axes.flatten()

        from datetime import datetime
        for i, ax in enumerate(axes):

            if i >= self.number_of_qubits: # If we have fewer qubits than subplots, stop plotting and hide the rest
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize = font)

            x = date_times[i]
            y = t1_vals[i]

            # Convert strings to datetime objects.
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            # Combine datetime objects and y values into a list of tuples and sort by datetime.
            combined = list(zip(datetime_objects, y))
            combined.sort(reverse=True, key=lambda x: x[0])

            if len(combined) == 0:
                # If this qubit has no data, just skip
                ax.set_visible(False)
                continue

            # Unpack them back into separate lists, in order from latest to most recent.
            sorted_x, sorted_y = zip(*combined)
            ax.scatter(sorted_x, sorted_y, color=colors[i])

            # Set x-axis limits for the specific timeframe
            ax.set_xlim(start_time, end_time)

            sorted_x = np.asarray(sorted(x))
            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically choose good tick locations
            # ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))  # Format as month-day
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))  # Show day and time
            ax.tick_params(axis='x', rotation=45)  # Rotate ticks for better readability

            # Disable scientific notation and format y-ticks
            ax.ticklabel_format(style="plain", axis="y")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))  #decimal places


            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font-2)
            ax.set_ylabel('T1 (us)', fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'T1_vals.pdf', transparent=True, dpi=self.final_figure_quality)
        print('Plot saved to: ', analysis_folder)
        plt.close()

    def plot_with_errs(self, date_times, t1_vals, t1_fit_err, show_legends,exp_extension=''):
        # ---------------------------------plot-----------------------------------------------------
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        # ----------------To Plot a specific timeframe------------------
        from datetime import datetime
        year = 2025
        month = 1
        day1 = 22  # Start date
        day2 = 23  # End date
        hour_start = 0  # Start hour
        hour_end = 23  # End hour
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 59)
        # -----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        ext = exp_extension.replace('_', '')
        plt.suptitle(f'T1 Values vs Time {ext}', fontsize=font)
        axes = axes.flatten()

        import matplotlib.dates as mdates
        from matplotlib.ticker import StrMethodFormatter

        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]
            y = t1_vals[i]
            err = t1_fit_err[i]

            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])
            if len(combined) == 0:
                ax.set_visible(False)
                continue
            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)

            #ax.set_xlim(start_time, end_time)

            ax.errorbar(
                sorted_x, sorted_y, yerr=sorted_err,
                fmt='none',
                ecolor=colors[i],
                elinewidth=1,
                capsize=0
            )

            ax.scatter(
                sorted_x, sorted_y,
                s=10,
                color=colors[i],
                alpha=0.5
            )

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.tick_params(axis='x', rotation=45)

            ax.ticklabel_format(style="plain", axis="y")

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel('T1 (us)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + f'T1_vals{exp_extension}.pdf', transparent=True, dpi=self.final_figure_quality)
        print('Plot saved to:', analysis_folder)
        plt.close()

    def plot_with_errs_single_plot(self, date_times, t1_vals, t1_fit_err, show_legends):
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")
        from datetime import datetime
        year = 2025
        month = 1
        day1 = 22
        day2 = 23
        hour_start = 0
        hour_end = 23
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 59)
        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('T1 Values vs Time', fontsize=font)
        import matplotlib.dates as mdates
        from matplotlib.ticker import StrMethodFormatter
        for i in range(self.number_of_qubits):
            x = date_times[i]
            y = t1_vals[i]
            err = t1_fit_err[i]
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]
            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])
            if len(combined) == 0:
                continue
            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)
            ax.errorbar(sorted_x, sorted_y, yerr=sorted_err, fmt='none', ecolor=colors[i], elinewidth=1, capsize=0,
                        label=titles[i] if show_legends else None)
            ax.scatter(sorted_x, sorted_y, s=10, color=colors[i], alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis='x', rotation=45)
        ax.ticklabel_format(style="plain", axis="y")
        if show_legends:
            ax.legend(edgecolor='black')
        ax.set_xlabel('Time', fontsize=font - 2)
        ax.set_ylabel('T1 (us)', fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()
        plt.savefig(analysis_folder + 'T1_vals_single_plot.pdf', transparent=True, dpi=self.final_figure_quality)
        print('Plot saved to:', analysis_folder)
        plt.close()


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


    def plot_results(self, I, Q, freqs, config=None, fig_quality=100, sigma_guess=1, return_fwhm=False, return_fit_err=False):
        freqs = np.array(freqs)
        freq_q = freqs[np.argmax(I)]

        mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err = self.fit_lorenzian(I, Q, freqs,
                                                                                                          freq_q,sigma_guess)

        # Check if the returned values are all None
        if (mean_I is None and mean_Q is None and I_fit is None and Q_fit is None
                and largest_amp_curve_mean is None and largest_amp_curve_fwhm is None):
            # If so, return None for the values in this definition as well
            empties = [None, None, None]
            if return_fwhm:
                empties.append(None)
            if return_fit_err:
                empties.append(None)
            return tuple(empties)

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
        plt.subplots_adjust(top=0.93, right=0.78)

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
        if return_fwhm and return_fit_err: #both set to True
            return largest_amp_curve_mean, I_fit, Q_fit, largest_amp_curve_fwhm, fit_err
        elif return_fwhm:
            return largest_amp_curve_mean, I_fit, Q_fit, largest_amp_curve_fwhm
        elif return_fit_err:
            return largest_amp_curve_mean, I_fit, Q_fit, fit_err
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

class QubitFreqsVsTime:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name,  fridge):
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.figure_quality = figure_quality
        self.run_name = run_name
        self.number_of_qubits = number_of_qubits
        self.final_figure_quality = final_figure_quality
        self.top_folder_dates = top_folder_dates
        self.fridge = fridge

    def datetime_to_unix(self, dt):
        # Convert to Unix timestamp
        unix_timestamp = int(dt.timestamp())
        return unix_timestamp

    def unix_to_datetime(self, unix_timestamp):
        # Convert the Unix timestamp to a datetime object
        dt = datetime.fromtimestamp(unix_timestamp)
        return dt

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def exponential(self, x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def optimal_bins(self, data):
        n = len(data)
        if n == 0:
            return {}
        # Sturges' Rule
        sturges_bins = int(np.ceil(np.log2(n) + 1))
        return sturges_bins

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
            numbers = [float(x.strip('[').strip(']').replace("'", "").replace(" ", "").replace("  ", "")) for x in match.split()] # Convert strings to integers
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

    def run(self,exp_extension=''):
        import datetime

        qubit_frequencies = {i: [] for i in range(self.number_of_qubits)}
        qspec_fit_errs= {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/" + folder_date + "/"
                outerFolder_save_plots = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/RR_metrics/Plots/QSpec_ge"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # ------------------------------------------Load/Plot/Save Q Spec------------------------------------
            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/optimization/Data_h5/qspec{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/optimization/Data_h5/QSpec/"


            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]

                H5_class_instance = Data_H5(h5_file)
                #H5_class_instance.print_h5_contents(h5_file)
                #sometimes you get '1(1)' when redownloading the h5 files for some reason
                load_data = H5_class_instance.load_from_h5(data_type=f'qspec{exp_extension}', save_r=int(save_round.split('(')[0]))

                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'qspec{exp_extension}']:
                    for dataset in range(len(load_data[f'qspec{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'qspec{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        date = datetime.datetime.fromtimestamp(load_data[f'qspec{exp_extension}'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        I = self.process_h5_data(load_data[f'qspec{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data[f'qspec{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                        # I_fit = load_data['qspec'][q_key].get('I Fit', [])[0][dataset]
                        # Q_fit = load_data['qspec'][q_key].get('Q Fit', [])[0][dataset]
                        freqs = self.process_h5_data(load_data[f'qspec{exp_extension}'][q_key].get('Frequencies', [])[0][dataset].decode())
                        round_num = load_data[f'qspec{exp_extension}'][q_key].get('Round Num', [])[0][dataset]
                        batch_num = load_data[f'qspec{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]
                        try:
                            syst_config = load_data[f'qspec{exp_extension}'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'qspec{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I) > 0:
                            qspec_class_instance = QubitSpectroscopy(q_key, self.number_of_qubits, outerFolder_save_plots, round_num, self.signal,
                                                                     self.save_figs)
                            # if '_' in exp_extension:
                            #     q_spec_cfg = exp_config[f'qubit_spec{exp_extension}']
                            # else:
                            #     q_spec_cfg = exp_config['qubit_spec_ge']
                            largest_amp_curve_mean, I_fit, Q_fit, qspec_fit_err = qspec_class_instance.get_results(I, Q, freqs)
                            if qspec_fit_err is not None and qspec_fit_err < 1: #above 1 MHz fit err is probably not a good fit
                                qubit_frequencies[q_key].extend([largest_amp_curve_mean])
                                qspec_fit_errs[q_key].extend([qspec_fit_err])
                                date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del qspec_class_instance

                del H5_class_instance
        return date_times, qubit_frequencies, qspec_fit_errs

    def plot_without_errs(self, date_times, qubit_frequencies, show_legends):
        # ---------------------------------plot-----------------------------------------------------
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        # ----------------To Plot a specific timeframe------------------
        from datetime import datetime
        year = 2025
        month = 1
        day1 = 24  # Start date
        day2 = 25  # End date
        hour_start = 0  # Start hour
        hour_end = 12  # End hour
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 0)
        # -----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('Qubit Frequencies vs Time', fontsize=font)
        axes = axes.flatten()

        from datetime import datetime
        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:  # If we have fewer qubits than subplots, stop plotting and hide the rest
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]
            y = qubit_frequencies[i]

            # Convert strings to datetime objects.
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            # Combine datetime objects and y values into a list of tuples and sort by datetime.
            combined = list(zip(datetime_objects, y))
            combined.sort(reverse=True, key=lambda x: x[0])

            if len(combined) == 0:
                # If this qubit has no data, just skip
                ax.set_visible(False)
                continue

            # Unpack them back into separate lists, in order from latest to most recent.
            sorted_x, sorted_y = zip(*combined)
            ax.scatter(sorted_x, sorted_y, color=colors[i])

            # Set x-axis limits for the specific timeframe
            #ax.set_xlim(start_time, end_time)

            #ax.set_ylim(sorted_y[0] - 2.0, sorted_y[0] + 2.0)

            sorted_x = np.asarray(sorted(x))

            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically choose good tick locations
            # ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))  # Format as month-day
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))  # Show day and time
            ax.tick_params(axis='x', rotation=45)  # Rotate ticks for better readability

            # Disable scientific notation and format y-ticks
            ax.ticklabel_format(style="plain", axis="y")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))  # 2 decimal places

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel('Qubit Frequency (MHz)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Q_Freqs_no_errs.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close()

    def plot_hist(self,  qubit_frequencies, show_legends):
        # ---------------------------------Setup Analysis Folder-----------------------------------------------------
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        # ----------------Histogram Plotting of Qubit Frequencies------------------
        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.suptitle('Histogram of Qubit Frequencies', fontsize=font)
        axes = axes.flatten()

        means = []
        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            # Ignore the date_times; only use qubit_frequencies.
            y = qubit_frequencies[i]

            if len(y) == 0:
                # If this qubit has no data, hide the subplot.
                ax.set_visible(False)
                continue
            y = self.remove_none_values_1D(y)
            # Plot histogram of the frequency data.
            ax.hist(y, bins=50, color=colors[i], edgecolor='black', alpha=0.7)
            means.append(np.mean(y))
            if show_legends:
                ax.legend([f"Freq Data Qubit {i + 1}"], edgecolor='black')
            ax.set_xlabel('Qubit Frequency (MHz)', fontsize=font - 2)
            ax.set_ylabel('Count', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Q_Freqs_no_errs.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close()
        return means

    def remove_none_values(self,list1, list2, list3):
        """Removes None values from list1 and their corresponding indices in list2 and list3."""
        if not (len(list1) == len(list2) == len(list3)):
            raise ValueError("All lists must have the same length")

        # Filter out None values and their corresponding elements in list2 and list3
        filtered_data = [(x, y, z) for x, y, z in zip(list1, list2, list3) if x is not None]

        # Unzip to separate the lists
        filtered_list1, filtered_list2, filtered_list3 = zip(*filtered_data) if filtered_data else ([], [], [])

        return list(filtered_list1), list(filtered_list2), list(filtered_list3)

    def remove_none_values_1D(self,list1):
        """Removes None values from list1 and their corresponding indices in list2 and list3."""

        # Filter out None values and their corresponding elements in list2 and list3
        filtered_data = [x for x in list1 if x is not None]

        return filtered_data
    def plot_with_errs(self, date_times, qubit_frequencies, qspec_fit_err, show_legends, exp_extension=''):
        #---------------------------------plot-----------------------------------------------------
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        # ----------------To Plot a specific timeframe------------------
        from datetime import datetime
        year = 2025
        month = 1
        day1 = 24  # Start date
        day2 = 25  # End date
        hour_start = 0  # Start hour
        hour_end = 12  # End hour
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 0)
        # -----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i+1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        ext = exp_extension.split('_')[0]
        plt.suptitle(f'Qubit Frequencies vs Time {ext}', fontsize=font)
        axes = axes.flatten()

        from datetime import datetime  # (if not already imported)
        # Loop over each qubit’s data.
        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:  # Hide extra subplots.
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]       # list of date strings
            y = qubit_frequencies[i]
            err = qspec_fit_err[i]  # corresponding error bars

            # Convert date strings to datetime objects.
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            # Combine datetime objects, frequencies, and error values, then sort in ascending order.
            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])  # sort by time (oldest first)

            if len(combined) == 0:
                # Skip if there is no data for this qubit.
                ax.set_visible(False)
                continue

            # Unpack the sorted data.
            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)
            sorted_y, sorted_x,sorted_err = self.remove_none_values(sorted_y,sorted_x,sorted_err)
            #try:
            ax.errorbar(
                sorted_x, sorted_y, yerr=sorted_err,
                fmt='none',
                ecolor=colors[i],
                elinewidth=1,
                capsize=0
            )
            #except:
            #    print(sorted_x,sorted_y)

            ax.scatter(
                sorted_x, sorted_y,
                s=10,
                color=colors[i],
                alpha=0.5
            )

            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.tick_params(axis='x', rotation=45)

            ax.ticklabel_format(style="plain", axis="y")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font-2)
            ax.set_ylabel('Qubit Frequency (MHz)', fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + f'Q_Freqs{exp_extension}.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close()

    def plot_with_errs_single_plot(self, date_times, qubit_frequencies, qspec_fit_err, show_legends):
        # ---------------------------------folder setup-----------------------------------------------------
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        from datetime import datetime
        year = 2025
        month = 1
        day1 = 24  # Start date
        day2 = 25  # End date
        hour_start = 0  # Start hour
        hour_end = 12  # End hour
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 0)

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Qubit Frequencies vs Time', fontsize=font)

        for i in range(self.number_of_qubits):
            x = date_times[i]
            y = qubit_frequencies[i]
            err = qspec_fit_err[i]

            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])

            if len(combined) == 0:
                continue

            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)

            ax.errorbar(
                sorted_x, sorted_y, yerr=sorted_err,
                fmt='none',
                ecolor=colors[i],
                elinewidth=1,
                capsize=0,
                label=titles[i] if show_legends else None
            )
            ax.scatter(
                sorted_x, sorted_y,
                s=10,
                color=colors[i],
                alpha=0.5
            )

        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis='x', rotation=45)

        ax.ticklabel_format(style="plain", axis="y")
        from matplotlib.ticker import StrMethodFormatter
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

        if show_legends:
            ax.legend(edgecolor='black')

        ax.set_xlabel('Time', fontsize=font - 2)
        ax.set_ylabel('Qubit Frequency (MHz)', fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Q_Freqs_single_plot.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close()


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

    def run(self, plot_res_spec=False, plot_q_spec=False, plot_rabi=False, rabi_rolling_avg=False, plot_ss=False,
            plot_ss_hist_only=False, ss_plot_title=None, ss_plot_gef=False, plot_t1=False,
            plot_t2r=False, plot_t2e=False, plot_rabis_Qtemps=False):

        # if plot_res_spec:
        #     self.load_plot_save_res_spec()
        # if plot_q_spec:
        #     self.load_plot_save_q_spec()
        if plot_rabis_Qtemps:
            list_of_all_qubits = [i for i in range(self.number_of_qubits + 1)]
            self.load_plot_save_rabis_Qtemps(list_of_all_qubits, save_figs = True, get_qtemp_data = False)
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

    def load_plot_save_t1(self):
        # ------------------------------------------------Load/Plot/Save T1----------------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/T1_ge/"
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

        for h5_file in h5_files:

            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='T1', save_r=int(save_round))

            populated_keys = []
            for q_key in load_data['T1']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['T1'][q_key].get('Dates', [[]])

                # Check if any entry in 'Dates' is not NaN
                if any(
                        not np.isnan(date)
                        for date in dates_list[0]  # Iterate over the first batch of dates
                ):
                    populated_keys.append(q_key)

            for q_key in populated_keys:
                for dataset in range(len(load_data['T1'][q_key].get('Dates', [])[0])):
                    # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                    # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                    date = datetime.datetime.fromtimestamp(load_data['T1'][q_key].get('Dates', [])[0][dataset])
                    I = self.process_h5_data(load_data['T1'][q_key].get('I', [])[0][dataset].decode())
                    Q = self.process_h5_data(load_data['T1'][q_key].get('Q', [])[0][dataset].decode())
                    delay_times = self.process_h5_data(
                        load_data['T1'][q_key].get('Delay Times', [])[0][dataset].decode())
                    # fit = load_data['T1'][q_key].get('Fit', [])[0][dataset]
                    round_num = load_data['T1'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['T1'][q_key].get('Batch Num', [])[0][dataset]

                    exp_config = load_data['T1'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}

                    exp_config = eval(exp_config, safe_globals)

                    if len(I) > 0:
                        T1_class_instance = T1Measurement(q_key, self.number_of_qubits, self.outerFolder_save_plots,
                                                          round_num, self.signal, self.save_figs, fit_data=True)
                        T1_spec_cfg = exp_config['T1_ge']
                        T1_class_instance.plot_results(I, Q, delay_times, date, T1_spec_cfg, self.figure_quality)
                        del T1_class_instance

            del H5_class_instance

    def load_plot_save_q_spec(self):
        # ----------------------------------------------Load/Plot/Save QSpec------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/qspec_ge/"
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
        extracted_freqs = []
        for h5_file in h5_files:
            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='qspec_ge', save_r=int(save_round))

            populated_keys = []
            for q_key in load_data['qspec_ge']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['qspec_ge'][q_key].get('Dates', [[]])

                # Check if any entry in 'Dates' is not NaN
                if any(
                        not np.isnan(date)
                        for date in dates_list[0]  # Iterate over the first batch of dates
                ):
                    populated_keys.append(q_key)

            for q_key in populated_keys:
                for dataset in range(len(load_data['qspec_ge'][q_key].get('Dates', [])[0])):
                    date = datetime.datetime.fromtimestamp(load_data['qspec_ge'][q_key].get('Dates', [])[0][dataset])
                    I = self.process_h5_data(load_data['qspec_ge'][q_key].get('I', [])[0][dataset].decode())
                    Q = self.process_h5_data(load_data['qspec_ge'][q_key].get('Q', [])[0][dataset].decode())
                    # I_fit = load_data['QSpec'][q_key].get('I Fit', [])[0][dataset]
                    # Q_fit = load_data['QSpec'][q_key].get('Q Fit', [])[0][dataset]
                    freqs = self.process_h5_data(load_data['qspec_ge'][q_key].get('Frequencies', [])[0][dataset].decode())
                    round_num = load_data['qspec_ge'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['qspec_ge'][q_key].get('Batch Num', [])[0][dataset]

                    exp_config = load_data['qspec_ge'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}

                    exp_config = eval(exp_config, safe_globals)

                    if len(I) > 0:
                        qspec_class_instance = QubitSpectroscopy(q_key, self.number_of_qubits,
                                                                 self.outerFolder_save_plots, round_num, self.signal,
                                                                 self.save_figs)
                        q_spec_cfg = exp_config['qubit_spec_ge']
                        # print('q_spec_cfg: ', q_spec_cfg)
                        qubit_freq, _, _, qspec_fit_err = qspec_class_instance.plot_results(I, Q, freqs, q_spec_cfg,
                                                        self.figure_quality, return_fit_err = True) # You don’t need to mention every parameter in the call
                        del qspec_class_instance

                        extracted_freqs.append({
                            "filename": os.path.basename(h5_file),
                            "q_key": int(q_key),
                            "dataset": dataset,
                            "round_num": round_num,
                            "batch_num": batch_num,
                            "freq_MHz": qubit_freq,
                            "Qfreq_fit_err": qspec_fit_err,
                            "timestamp": date.timestamp()
                        })

            del H5_class_instance

        return extracted_freqs

    def load_plot_save_rabis_Qtemps(self, list_of_all_qubits, save_figs = False, get_qtemp_data = False):
        # ------------------------------------------------Load/Plot/Save Rabi---------------------------------------
        outerFolder_expt_qtemps = self.unique_folder_path+ "/Data_h5/q_temperatures/"
        h5_files_qtemps = glob.glob(os.path.join(outerFolder_expt_qtemps, "*.h5"))
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

        for h5_file in h5_files_qtemps:

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
            A_amplitude_err1 = None
            A_amplitude_err2 = None

            for q_key in populated_keys:
                # print(f"Extracting data for QubitIndex: {q_key}")
                for dataset in range(len(load_data['q_temperatures'][q_key].get('Dates', [])[0])):
                    date = datetime.datetime.fromtimestamp(load_data['q_temperatures'][q_key].get('Dates', [])[0][dataset])
                    round_num = load_data['q_temperatures'][q_key].get('Round Num', [])[0][dataset]
                    # batch_num = load_data['q_temperatures'][q_key].get('Batch Num', [])[0][dataset]

                    #-------------------------------------Grabbing matching qubit frequency for this qubit-------------------------------------
                    if date.timestamp() > cutoff_timestamp and get_qtemp_data:
                        # Files after this date contain the matching g-e qubit frequency already BUT the files do not contain the corresponding qspec fit errors.

                        # The line below extracts the qfreq saved in each rabi pop. meas. file, but it does not extract the error of the qspec fit because that was not saved in the h5 files.
                        qubit_freq_MHz_rpmfile = load_data['q_temperatures'][q_key].get('Qfreq_ge', [])[0][dataset] #extract to compare with the 'matching' method
                        print(f"QSpec from RPM file, Q{q_key}: {qubit_freq_MHz_rpmfile} MHz") # print to compare

                        # To find the correct qspec fit error from the ge qspec files, we have to match the qspec files to the RPM files via time stamps.

                        # Build the QTemp timestamp:
                        qtemp_timestamp = date.timestamp()

                        # Grab all QSpec entries for this qubit:
                        qspec_entries = qspec_grouped_by_qkey.get(int(q_key), [])
                        if not qspec_entries:
                            print(f"No QSpec entries found for Q{q_key}")
                            continue

                        # Find the QSpec dict whose timestamp is closest to qtemp_timestamp:
                        # Computes the absolute time difference between that entry’s timestamp and your current qubit‐temperature timestamp
                        closest_match = min(
                            qspec_entries,
                            key=lambda entry: abs(entry['timestamp'] - qtemp_timestamp) ) # tells Python to pick the element for which the key function returns the smallest value

                        # Extract frequency and its fit error from that match:
                        qubit_freq_MHz = closest_match['freq_MHz']
                        qfreq_err = closest_match['Qfreq_fit_err']

                        print(
                            f"Matched QSpec for Q{q_key}: {qubit_freq_MHz:.3f} MHz  "
                            f"(QSpec t={datetime.datetime.fromtimestamp(closest_match['timestamp'])})" )

                    elif date.timestamp() <= cutoff_timestamp and get_qtemp_data: #-----this look through matching qspec file ONLY, does not extract qfreq from RPM h5 file----
                        # Build the QTemp timestamp:
                        qtemp_timestamp = date.timestamp()

                        # Grab all QSpec entries for this qubit:
                        qspec_entries = qspec_grouped_by_qkey.get(int(q_key), [])
                        if not qspec_entries:
                            print(f"No QSpec entries found for Q{q_key}")
                            continue

                        # Find the QSpec dict whose timestamp is closest to qtemp_timestamp:
                        # Computes the absolute time difference between that entry’s timestamp and your current qubit‐temperature timestamp
                        closest_match = min(
                            qspec_entries,
                            key=lambda entry: abs(entry['timestamp'] - qtemp_timestamp) ) # tells Python to pick the element for which the key function returns the smallest value

                        # Extract frequency and its fit error from that match:
                        qubit_freq_MHz = closest_match['freq_MHz']
                        qfreq_err = closest_match['Qfreq_fit_err']

                        print(
                            f"Matched QSpec for Q{q_key}: {qubit_freq_MHz:.3f} MHz  "
                            f"(QSpec t={datetime.datetime.fromtimestamp(closest_match['timestamp'])})")
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
                        rabi_class_instance = Temps_EFAmpRabiExperiment(q_key, self.number_of_qubits, list_of_all_qubits,
                                                                      self.outerFolder_save_plots, round_num,
                                                                      self.signal, save_figs)
                        I1 = np.asarray(I1)
                        Q1 = np.asarray(Q1)
                        gains1 = np.asarray(gains1)
                        best_signal_fit1, pi_amp1, A_amplitude1, A_amplitude_err1, amp_fit1 = rabi_class_instance.plot_results(I1, Q1, gains1, rabi_cfg, self.figure_quality)
                        del rabi_class_instance

                    if len(I2) > 0:
                        rabi_class_instance = Temps_EFAmpRabiExperiment(q_key, self.number_of_qubits,
                                                                        list_of_all_qubits,
                                                                        self.outerFolder_save_plots, round_num,
                                                                        self.signal, save_figs)
                        I2 = np.asarray(I2)
                        Q2 = np.asarray(Q2)
                        gains2 = np.asarray(gains2)
                        best_signal_fit2, pi_amp2, A_amplitude2, A_amplitude_err2, amp_fit2 = rabi_class_instance.plot_results(I2, Q2, gains2, rabi_cfg, self.figure_quality)
                        del rabi_class_instance

                    if not get_qtemp_data:
                        continue  # Skip the rest of this block if not returning data

                    if (A_amplitude1 is not None and A_amplitude2 is not None and
                        A_amplitude_err1 is not None and A_amplitude_err2 is not None):
                        A_e = A_amplitude1
                        A_g = A_amplitude2

                        results = self.Qubit_Temperature_Convert(A_e, A_g, qubit_freq_MHz)
                        if results is None:
                            continue  # Skip this dataset
                        T_K, T_mK, P_e, qubit_freq = results
                        print(f"Q{q_key + 1} calculated Temperature:{T_mK}, with P_e = {P_e}, and Qfreq {qubit_freq_MHz} MHz")

                        # Compute propagated 1-sigma error (std) on T_mK
                        try:
                            T_err = self.compute_temperature_error_RPM(
                                A1=A_amplitude1,
                                A2=A_amplitude2,
                                Pe=P_e,
                                T_mK=T_mK,
                                qubit_freq_MHz=qubit_freq,
                                sigma_A1=A_amplitude_err1,
                                sigma_A2=A_amplitude_err2,
                                sigma_qfreq_MHz=qfreq_err
                            )
                        except Exception as e:
                            print(f"Error computing T_err for Q{q_key}: {e}")
                            continue

                        if T_err is not None:
                            file_result['qubits'][int(q_key)] = {
                                'A1': A_amplitude1,
                                'A1_err': A_amplitude_err1,
                                'A2_err': A_amplitude_err2,
                                'A2': A_amplitude2,
                                'T_mK': T_mK,
                                'T_mK_err': T_err,
                                'P_e': P_e,
                                'qubit_freq_MHz': qubit_freq,
                                "Qfreq_fit_err" : qfreq_err,
                                'date': date.timestamp(),
                                'filepath': h5_file}
                        else:
                            print(f"Skipping Q{q_key} entry because T_err was not calculated successfully.")

            if get_qtemp_data:
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

    def compute_temperature_error_RPM(self, A1, A2, Pe, T_mK, qubit_freq_MHz, sigma_A1, sigma_A2, sigma_qfreq_MHz):
        """
        Propagate the 1-sigma uncertainties in A1, A2 and f_ge
        into a 1-sigma uncertainty on T_mK, given you already know
        Pe and T_mK.

        Inputs:
          A1, A2               – fitted amplitudes
          Pe                   – thermal population associated with T_mK
          T_mK                 – temperature via rabi pop. meas. in mK
          qubit_freq_MHz       – fitted g-e qubit frequency (MHz)
          sigma_A1, sigma_A2   – 1-sigma errors on A1 and A2 (standard deviations)
          sigma_qfreq_MHz      – 1-sigma error on qubit_freq_MHz (standard deviation)

        Returns:
          sigma_T_mK           – propagated 1-sigma error on T_mK
        """
        # get sigma_Pe from A1,A2 errors
        sum_A = A1 + A2
        # ∂Pe/∂A1 =  A2 / (A1+A2)^2
        # ∂Pe/∂A2 = -A1 / (A1+A2)^2
        dPe_dA1 = A2 / sum_A ** 2
        dPe_dA2 = -A1 / sum_A ** 2

        sigma_Pe = np.sqrt(
            (dPe_dA1 * sigma_A1) ** 2 +
            (dPe_dA2 * sigma_A2) ** 2
        )

        # convert MHz → Hz for the qubit frequency and its error
        f0_Hz = qubit_freq_MHz * 1e6
        sigma_f0_Hz = sigma_qfreq_MHz * 1e6

        # build the log term (we already know Pe)
        ln_arg = np.log((1 - Pe) / Pe)

        # partial derivatives of T_mK
        # ∂T/∂f0  = T_mK / f0_Hz
        dT_df0 = T_mK / f0_Hz

        # ∂T/∂Pe  = T_mK / [ ln_arg * Pe * (1-Pe) ]
        dT_dPe = T_mK / (ln_arg * Pe * (1 - Pe))

        # combine in quadrature
        sigma_T_mK = np.sqrt(
            (dT_df0 * sigma_f0_Hz) ** 2 +
            (dT_dPe * sigma_Pe) ** 2
        )

        return sigma_T_mK # Temperature calculation error via rabi population measurements

    # Helper for fitting & plotting a line on `ax`
    def do_linear_fit_and_plot_qtemps_RPM(self, ax, times_arr, temps_arr, initial_time, final_time, mask, color, label_prefix):
        """
        Perform a linear regression on the subset of (times_arr, temps_arr)
        indicated by `mask` (which itself should already restrict times_arr
        to be between initial_time and final_time).  Then plot the best‐fit
        line onto `ax`, extending from initial_time to final_time.

        The line is parameterized as
            temperature (mK) = m * (hours since initial_time) + b,
        and we compute an R² to indicate goodness of fit.  The x‐axis on the
        plot is in actual datetime.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to draw the fit‐line.
        times_arr : 1D np.array of floats
            An array of POSIX timestamps (in seconds).  You still pass every
            timestamp in here, even those outside [initial_time, final_time].
        temps_arr : 1D np.array of floats
            The matching temperatures (mK) at each timestamp.
        initial_time : float
            POSIX timestamp (seconds) marking the “start” of the fit window.
        final_time : float
            POSIX timestamp (seconds) marking the “end” of the fit window.
        mask : 1D boolean np.array
            A boolean mask the same length as times_arr, True for any index i
            such that initial_time <= times_arr[i] <= final_time.  Only those
            points will be used for the regression.
        color : str
            Color used for drawing the line.
        label_prefix : str
            A short label (e.g. “Full ramp” or “Up to 120 mK”) that will be
            prepended to the slope and R² in the legend.
        """
        # Restrict to exactly the points the user passed (the mask should
        # already enforce initial_time <= times_arr <= final_time)
        x_sel = times_arr[mask]
        y_sel = temps_arr[mask]
        if len(x_sel) < 2:
            # Not enough points to do a proper fit. skip drawing anything.
            return

        # Convert those timestamps into “hours since initial_time”
        x_hours = (x_sel - initial_time) / 3600.0  # hrs

        # Linear regression (y = m * x + b) on (x_hours, y_sel)
        m, b = np.polyfit(x_hours, y_sel, 1)

        # Compute R-squared for these selected points
        y_fit_at_points = m * x_hours + b
        residuals = y_sel - y_fit_at_points
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_sel - np.mean(y_sel)) ** 2)
        r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        # Build a “fine” x‐grid that spans exactly from initial_time → final_time.
        # In units of hours since initial_time, that means from t = 0 → t = (final_time - initial_time)/3600.
        hours_start = 0.0
        hours_end = (final_time - initial_time) / 3600.0

        x_fit_line = np.linspace(hours_start, hours_end, 100)
        y_fit_line = m * x_fit_line + b

        # Convert those “hours since initial_time” back into real datetimes, so we can plot on ax.
        dt_fit = [datetime.datetime.fromtimestamp(initial_time + (h * 3600.0)) for h in x_fit_line]

        # draw the line on ax, with high zorder so it sits on top of the scatter.
        ax.plot(
            dt_fit,
            y_fit_line,
            linestyle='-',
            linewidth=2,
            color=color,
            label=f"{label_prefix}: slope={m:.1f} mK/h, R²={r2:.2f}",
            zorder=10)

    def plot_qubit_temperatures_vs_time_RPMs(self, all_files_Qtemp_results, num_qubits=6, yaxis_min = 10, yaxis_max = 950, restrict_time_xaxis = False,
                                             plot_extra_event_lines = False, rad_events_plot_lines = True, plot_error_bars=False, fit_to_line=False, average_per_heater_step=False):
        """
        Plots qubit temperatures vs. time for each qubit in a separate subplot (max 3 columns).

        Parameters:
        - all_files_Qtemp_results: list of dicts returned by `load_plot_save_rabis_Qtemps`
        - num_qubits: total number of qubits to plot (default is 6)
        - restrict_time_xaxis : do you want to plot only a certain region of time?
        - plot_extra_event_lines: do you want to plot vertical dashed lines to mark extra events that happened (besides source instalation)?
        - plot_error_bars: do you want to plot error bars?
        - fit_to_line : do you want to perform linar fits? Right now it is set up to fit two linear fits: (1)full heater ramp up 2)and up to 120 mK)
        """

        # Define the colors you want for each qubit
        colors = ["orange", "blue", "purple", "green", "brown", "pink"]
        legend_handles = []
        ncols = min(num_qubits, 3)
        nrows = math.ceil(num_qubits / 3)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(4 * ncols, 4 * nrows),
                                 sharex=False, constrained_layout=True)  # set sharex=False if you want each subplot to manage ticks independently
        axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

        fig.suptitle("Qubit Temperatures vs. Time", fontsize=16)

        # Optional: to plot radiation source events
        events_radiation = [
            (datetime.datetime(2025, 4, 21, 12, 35), "Co-60"),
            (datetime.datetime(2025, 4, 23, 12, 53), "Cs-137"),
            (datetime.datetime(2025, 4, 28, 9, 40), "Cs-137 Closer"),
            (datetime.datetime(2025, 5, 4, 18, 20), "Cs-137 Removed"),
            (datetime.datetime(2025, 5, 5, 11, 51), "Cs-137 Hot"),
            (datetime.datetime(2025, 5, 5, 14, 40), "Cs-137 removed"),
            (datetime.datetime(2025, 5, 6, 15, 28), "Cs-137 Hot"),
            (datetime.datetime(2025, 5, 6, 16, 0), "Cs-137 removed"),
            (datetime.datetime(2025, 5, 7, 10, 36), "Cs-137 Hot"),
            (datetime.datetime(2025, 5, 7, 16, 20), "Cs-137 removed")
        ]

        # Optional: Now for other events. Only relevant if plot_extra_event_lines is set to True!!!
        events_0418 = [
            (datetime.datetime(2025, 4, 18, 11, 50), "Daniel Entry"),
            (datetime.datetime(2025, 4, 18, 13, 30), "Daniel Exit"),
            (datetime.datetime(2025, 4, 18, 14, 53), "Daniel Entry"),
            (datetime.datetime(2025, 4, 18, 15, 0), "Door Intermission"),
            (datetime.datetime(2025, 4, 18, 15, 6), "Exit/Re-entry Daniel"),
            (datetime.datetime(2025, 4, 18, 15, 12), "Ryan Entry"),
            (datetime.datetime(2025, 4, 18, 15, 40), "Door Intermission"),
            (datetime.datetime(2025, 4, 18, 16, 11), "Daniel Exit"),
            (datetime.datetime(2025, 4, 18, 16, 12), "Daniel Entry"),
            (datetime.datetime(2025, 4, 18, 16, 16), "Daniel Exit")]

        events_0423 = [
            (datetime.datetime(2025, 4, 23, 12, 50), "Dan-Joyce Entry"),
            (datetime.datetime(2025, 4, 23, 12, 54), "Dan-Joyce Exit"),
            (datetime.datetime(2025, 4, 23, 13, 40), "Grace Entry"),
            (datetime.datetime(2025, 4, 23, 13, 47), "Grace Exit"),
            (datetime.datetime(2025, 4, 23, 16, 40), "Kester-Grace Entry"),
            (datetime.datetime(2025, 4, 23, 16, 48), "Kester-Grace Exit")]

        heater_events = [
            (datetime.datetime(2025, 5, 8, 18, 50), "20mK step"),
            (datetime.datetime(2025, 5, 9, 10, 24), "40mK step"),
            (datetime.datetime(2025, 5, 10, 1, 39), "60mK step"),
            (datetime.datetime(2025, 5, 10, 18, 31), "80mK step"),
            (datetime.datetime(2025, 5, 11, 14, 46), "100mK step"),
            (datetime.datetime(2025, 5, 12, 15, 11), "120mK step"),
            (datetime.datetime(2025, 5, 13, 12, 2), "140mK step"),
            (datetime.datetime(2025, 5, 14, 12, 31), "160mK step"),
            (datetime.datetime(2025, 5, 14, 22, 50), "Heater Off")]

        # Optional: Restrict plot to specific date and time window. Will only go into effect if restrict_time_xaxis = True
        # date_to_plot = datetime.date(2025, 4, 17)
        # start_datetime = datetime.time(0, 0)  # Start of the window
        # end_datetime = datetime.time(23, 59)
        start_datetime = datetime.datetime(2025, 5, 7, 16, 20)
        end_datetime = datetime.datetime(2025, 5, 16, 23, 59)

        for q in range(num_qubits):
            times = []
            temps = []
            errs = []

            for file_result in all_files_Qtemp_results:
                qubit_data = file_result['qubits'].get(q)
                if qubit_data:
                    T_err = qubit_data['T_mK_err']
                    T_mK = qubit_data['T_mK']

                    # Skip if relative error is ≥ 80%
                    if T_err / T_mK >= 0.80:
                        continue

                    # if T_err > 150:  # skip if error is too large (for example, larger than 300mK)
                    #     continue
                    errs.append(T_err)
                    temps.append(T_mK)

                    timestamp = qubit_data['date']
                    times.append(datetime.datetime.fromtimestamp(timestamp))

                    # if T_mK > 800:
                    #     print(f"High Temperature ({T_mK:.1f} mK) in file {qubit_data['filepath']} for Q{q + 1}. A1={qubit_data['A1']}, A2={qubit_data['A2']}, Qfreq={qubit_data['qubit_freq_MHz']}.")

            ax = axes[q]

            if not times:
                ax.set_visible(False)
                continue

            # ------------To average all of the points during each heater step to facilitate fitting the data to a line----------------------------------
            if average_per_heater_step:
                binned_times, binned_temps, binned_errs = [], [], []

                # Extracting heater step times
                step_events = [(dt, label) for dt, label in heater_events if "step" in label.lower()]
                step_events.sort()
                step_times = [dt for dt, _ in step_events]

                # Adding start and end boundaries
                pre_step_time = datetime.datetime.min
                post_step_time = datetime.datetime(2025, 5, 14, 22, 50)  # Heater was turned Off

                # Creating list of bin edges: [[start to 20mK], [20mK to 40mK], ..., [160mK to heater off]]
                bin_edges = [pre_step_time] + step_times + [post_step_time, datetime.datetime.max]

                times_np = np.array(times)
                temps_np = np.array(temps)
                errs_np = np.array(errs)

                # Bin and average data
                for i in range(len(bin_edges) - 1):
                    start, end = bin_edges[i], bin_edges[i + 1]
                    mask = (times_np >= start) & (times_np < end)

                    if np.sum(mask) < 2:
                        continue  # skip bins with too few points

                    avg_time = np.mean([t.timestamp() for t in times_np[mask]])
                    avg_time_dt = datetime.datetime.fromtimestamp(avg_time)
                    avg_temp = np.mean(temps_np[mask])
                    avg_err = np.sqrt(np.sum(errs_np[mask] ** 2)) / np.sum(mask) # propagating independent, uncorrelated, Gaussian uncertainties (standard deviations).

                    binned_times.append(avg_time_dt)
                    binned_temps.append(avg_temp)
                    binned_errs.append(avg_err)

                times, temps, errs = binned_times, binned_temps, binned_errs
            # -----------------------------------------------------------------------------------------------------

            if restrict_time_xaxis: # tweak format as needed for the x axis ticks
                #for a single day
                # start_time = datetime.datetime.combine(date_to_plot, start_datetime)
                # end_time = datetime.datetime.combine(date_to_plot, end_datetime)
                # #Use finer ticks with hour detail
                # ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%H'))

                # For multiple Days
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%H'))
            else:
                #Use coarse ticks with just date
                # ax.xaxis.set_major_locator(mdates.DayLocator())
                # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%H'))

            # plot with or without error bars
            if plot_error_bars:
                ax.errorbar(
                    times,
                    temps,
                    yerr=errs,
                    fmt='o',
                    capsize=4,
                    markersize=6,
                    color=colors[q % len(colors)],
                    label=f"Q{q + 1}"
                )
            else:
                ax.scatter(
                    times,
                    temps,
                    marker='o',
                    color=colors[q % len(colors)],
                    label=f"Q{q + 1}"
                )

            ax.set_title(f"Q{q + 1}", fontsize=14)
            ax.set_ylabel("Temp (mK)", fontsize=12)
            ax.grid(False)

            # Format the x-axis to show dates in a nice format
            ax.set_ylim(yaxis_min, yaxis_max)
            ax.set_yticks(np.linspace(yaxis_min, yaxis_max, 10))

            # start_time = datetime.datetime(2025, 4, 11, 12, 30)
            # ax.set_xlim(left=start_time)

            ax.tick_params(axis='x', labelrotation=45, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)

            if rad_events_plot_lines:
                for vtime, label in events_radiation:
                    if not restrict_time_xaxis or (restrict_time_xaxis and start_datetime <= vtime <= end_datetime):
                        ax.axvline(vtime, color='black', linestyle='--', linewidth=1)
                        ax.text(vtime, ax.get_ylim()[1] * 0.95, label, rotation=90, verticalalignment='top',
                                horizontalalignment='right', fontsize=10)


            # Combine (conditionally) the extra events you want to plot
            plot_0418_events = False
            plot_0423_events = False
            plot_heater_events = True

            extra_events = []
            if plot_0418_events:
                extra_events += events_0418
            if plot_0423_events:
                extra_events += events_0423
            if plot_heater_events:
                extra_events += heater_events

            if restrict_time_xaxis:
                ax.set_xlim(start_datetime, end_datetime)
                ax.set_autoscale_on(False)

            if plot_extra_event_lines:
                # Only keep events within the plot window if restrict_time_xaxis is True
                if restrict_time_xaxis:
                    extra_events = [(vtime, label) for vtime, label in extra_events if start_datetime <= vtime <= end_datetime]

                # Map each unique label to a unique color
                unique_labels = list(dict.fromkeys(label for _, label in extra_events))
                # cmap = cm.get_cmap('tab20', len(unique_labels)) pastels
                cmap = cm.get_cmap('Set1', len(unique_labels)) #dark colors
                label_to_color = {label: mcolors.to_hex(cmap(i)) for i, label in enumerate(unique_labels)}

                # Track which labels were already used in the legend
                used_labels = set()

                # Plot vertical lines for each event, reusing colors
                for vtime, label in extra_events:
                    color = label_to_color[label]
                    ax.axvline(vtime, color=color, linestyle='--', linewidth=1)
                    if label not in used_labels:
                        legend_handles.append(Line2D([0], [0], color=color, linestyle='--', label=label, alpha=1.0))
                        used_labels.add(label)

            if fit_to_line:  # fit data to a line, choosing where to start and stop based on event time stamps
                # pull out all three relevant heater events
                for dt, label in heater_events:
                    if label == "20mK step":
                        t20_ts = dt.timestamp()
                    elif label == "60mK step":
                        t60_ts = dt.timestamp()
                    elif label == "120mK step":
                        t120_ts = dt.timestamp()
                    elif label == "160mK step":
                        t160_ts = dt.timestamp()
                    elif label == "100mK step":
                        t100_ts = dt.timestamp()

                # turn existing lists of datetimes/temps into arrays of POSIX seconds
                times_arr = np.array([t.timestamp() for t in times])
                temps_arr = np.array(temps)

                # for Q5, start at the first time stamp plotted and go all the way to 160 mK for the “full” fit,
                # but only to 120 mK for the “up to 120 mK” fit

                drop_some_pts = False # set this to true if you want to disregard points above/under a certain temperature.
                # This introduces biases though because you are essentially selecting the data. Use only for tests.
                if q == 4:
                    start_ts = times_arr.min() # alternatively, you could start at t20_ts
                    final_full_ts = t160_ts
                    final_120_ts = t120_ts

                    if drop_some_pts:
                        drop_region = ((times_arr >= start_ts) & (times_arr <= t60_ts) & (temps_arr > 122.0))

                        # basic time masks
                        base_mask_full = (times_arr >= start_ts) & (times_arr <= final_full_ts)
                        base_mask_to120 = (times_arr >= start_ts) & (times_arr <= final_120_ts)
                        # now remove any points in drop_region
                        mask_full = base_mask_full & (~drop_region)
                        mask_to120 = base_mask_to120 & (~drop_region)
                    else:
                        mask_full = (times_arr >= start_ts) & (times_arr <= final_full_ts)
                        mask_to120 = (times_arr >= start_ts) & (times_arr <= final_120_ts)

                    color_full = "black"
                    color_to120 = "green"
                    prefix_full = "Full ramp"
                    prefix_120 = "Up to 120 mK"

                # for Q1, start at the first time stamp plotted and again go to 160 mK for the “full” fit,
                # but only to 120 mK for the “up to 120 mK” fit
                elif q == 0:
                    start_ts = times_arr.min() # alternatively, you could start at t20_ts
                    final_full_ts = t160_ts
                    final_120_ts = t120_ts

                    if drop_some_pts:
                        drop_region1 = ((times_arr >= start_ts) & (times_arr <= t60_ts) & (temps_arr > 122.0))
                        drop_region2 = (times_arr >= t60_ts) & (times_arr <= t100_ts) & (temps_arr > 176.0)

                        drop_region = drop_region1 | drop_region2

                        # basic time masks
                        base_mask_full = (times_arr >= start_ts) & (times_arr <= final_full_ts)
                        base_mask_to120 = (times_arr >= start_ts) & (times_arr <= final_120_ts)
                        # remove any points in drop_region
                        mask_full = base_mask_full & (~drop_region)
                        mask_to120 = base_mask_to120 & (~drop_region)
                    else:
                        mask_full = (times_arr >= start_ts) & (times_arr <= final_full_ts)
                        mask_to120 = (times_arr >= start_ts) & (times_arr <= final_120_ts)

                    color_full = "black"
                    color_to120 = "green"
                    prefix_full = "Full ramp"
                    prefix_120 = "Up to 120 mK"

                else:
                    # skipping other q’s:
                    continue

                # plot the full‐ramp line (20→160 mK or 60→160 mK)
                self.do_linear_fit_and_plot_qtemps_RPM(
                    ax=ax,
                    times_arr=times_arr,
                    temps_arr=temps_arr,
                    initial_time=start_ts,
                    final_time=final_full_ts,
                    mask=mask_full,
                    color=color_full,
                    label_prefix=prefix_full
                )

                # plot the “up to 120 mK” line
                self.do_linear_fit_and_plot_qtemps_RPM(
                    ax=ax,
                    times_arr=times_arr,
                    temps_arr=temps_arr,
                    initial_time=start_ts,
                    final_time=final_120_ts,
                    mask=mask_to120,
                    color=color_to120,
                    label_prefix=prefix_120
                )

                ax.legend(fontsize=9, loc="upper left")


            # Add a combined legend (only once)
            if q == 0 and legend_handles:
                fig.legend(handles= legend_handles)

        # Add a shared X label
        for ax in axes:
            ax.set_xlabel("Time")

        # Save the figure
        paramvstime_dir = os.path.join(self.outerFolder_save_plots, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)

        timestp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(paramvstime_dir, f"QubitTemps_vs_Time_{timestp}.png")
        print("Plot saved to: ", save_path)
        plt.savefig(save_path, dpi=self.figure_quality)
        plt.close(fig)

    def plot_qubit_temperature_histograms_RPMs(self, all_files_Qtemp_results, num_qubits=6):
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

        # From Gaussian fit
        mean_values = {}
        std_values = {}

        # From weighted average method
        w_mean_values = {}
        w_std_values = {}

        # Loop over each qubit / subplot
        for i, ax in enumerate(axes):
            # Gather all temperature data for qubit i across all files.
            temp_vals = []
            temp_errs = []
            for file_result in all_files_Qtemp_results:
                qubit_data = file_result['qubits'].get(i)
                if not qubit_data:
                    continue

                T_mK = qubit_data.get('T_mK')
                T_err = qubit_data.get('T_mK_err')

                # skip if either is missing or relative error is larger than threshold
                if T_mK is None or T_err is None:
                    continue
                # if T_err / T_mK >= 0.80: #80%
                #     continue

                temp_vals.append(T_mK)
                temp_errs.append(T_err)

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

            # Inverse-variance weighted average ---------------------------------------------
            temps = np.asarray(temp_vals)
            errs = np.asarray(temp_errs)
            weights = 1.0 / errs ** 2  # w_i = 1/σ_i²

            w_mean = np.sum(weights * temps) / np.sum(weights) # inverse-variance weighted average
            w_err = np.sqrt(1.0 / np.sum(weights))  # 1-σ error (uncertainty) on the mean; this tells you how precisely you've determined the weighted average itself.

            w_mean_values[f"Qubit {i + 1}"] = w_mean  # weighted mean
            w_std_values[f"Qubit {i + 1}"] = w_err  # its uncertainty
            # -------------------------------------------------------------------------------

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
            ax.plot(x_vals, scaled_pdf, linestyle='--', linewidth=2, color=colors[i % len(colors)], label='Gaussian fit')
            ax.hist(temp_vals, bins=optimal_bin_num, alpha=0.7, color=colors[i % len(colors)], edgecolor='black')
            ax.axvline(w_mean, color='k', lw=2, label=f'Weighted μ = {w_mean:.2f}±{w_err:.2f} mK')
            print(f"Qubit {i + 1} | Weighted Mean: {w_mean:.2f}, Gaussian μ: {mu:.2f}")
            print(f"Weights: {weights}")
            print(f"Sum(weights): {np.sum(weights):.2f}, Max weight: {np.max(weights):.2f}")

            # Set subplot title and labels including the Gaussian parameters
            ax.set_title(f"{titles[i]}  $\mu$: {mu:.2f} mK,  $\sigma$: {std:.2f} mK", fontsize=font)
            ax.set_xlabel("Temperature (mK)", fontsize=font)
            ax.set_ylabel("Frequency", fontsize=font)
            ax.tick_params(axis='both', which='major', labelsize=font)

        plt.tight_layout()

        hist_dir = os.path.join(self.outerFolder_save_plots, "qtemps_hists")
        os.makedirs(hist_dir, exist_ok=True)

        timestp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(hist_dir, f"QubitTemps_Histograms_{timestp}.png")
        print("Histogram plot saved to:", save_path)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

    def plot_qubit_pe_vs_time_RPMs(self, all_files_Qtemp_results, num_qubits=6):
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

        paramvstime_dir = os.path.join(self.outerFolder_save_plots, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)

        timestp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(paramvstime_dir, f"QubitPe_vs_Time_{timestp}.png")
        print("Plot saved to:", save_path)
        plt.savefig(save_path, dpi=self.figure_quality)
        plt.close(fig)
        # plt.show()

    def plot_qubit_temp_and_pe_vs_time_RPMs(self, all_files_Qtemp_results, num_qubits=6):
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

        paramvstime_dir = os.path.join(self.outerFolder_save_plots, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)

        timestp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(paramvstime_dir, f"QubitTemps_and_Pe_vs_Time_{timestp}.png")
        print("Combined plot saved to:", save_path)
        plt.savefig(save_path, dpi=self.figure_quality)
        plt.close(fig)
        # plt.show()

    def plot_qubit_temp_pe_freq_vs_time_RPMs(self, all_files_Qtemp_results, num_qubits=6):
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

        paramvstime_dir = os.path.join(self.outerFolder_save_plots, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)

        timestp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        save_path = os.path.join(paramvstime_dir, f"QubitTemps_Pe_Freq_vs_Time_{timestp}.png")
        print("Combined plot saved to:", save_path)
        plt.savefig(save_path, dpi=self.figure_quality)
        plt.close(fig)