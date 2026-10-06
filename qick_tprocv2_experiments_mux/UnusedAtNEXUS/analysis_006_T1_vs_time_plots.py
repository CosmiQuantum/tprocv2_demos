import numpy as np
import os
import sys
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
import math
import ast
import os
import matplotlib.pyplot as plt
import allantools
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter

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
                outerFolder = f"/data/QICK_data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/data/QICK_data/{self.run_name}/" + folder_date + "_plots/"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # ------------------------------------------------Load/Plot/Save T1----------------------------------------------
            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/T1{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/T1_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            #print(outerFolder_expt)
            for h5_file in h5_files:

                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=f'T1{exp_extension}', save_r=int(save_round))
                # if '01-27' in outerFolder_expt:
                #     print(load_data)
                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'T1{exp_extension}']:
                    for dataset in range(len(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                        # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        I = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                        delay_times = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Delay Times', [])[0][dataset].decode())
                        # fit = load_data['T1'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data[f'T1{exp_extension}'][q_key].get('Round Num', [])[0][dataset]
                        try:
                            batch_num = load_data[f'T1{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]
                            syst_config = load_data[f'T1{exp_extension}'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'T1{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
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

    def run_IBM_qze(self, exp_extension=''):
        import datetime

        # ----------Load/get data------------------------
        Is = {i: [] for i in range(self.number_of_qubits)}
        Qs = {i: [] for i in range(self.number_of_qubits)}
        amps = {i: [] for i in range(self.number_of_qubits)}
        gains = {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        #print(self.top_folder_dates)
        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"/data/QICK_data/{self.run_name}/" + folder_date + "/study_data"
                outerFolder_save_plots = f"/data/QICK_data/{self.run_name}/" + folder_date + "_plots/"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # ------------------------------------------------Load/Plot/Save T1----------------------------------------------
            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/T1{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/T1_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            #print(outerFolder_expt)
            for h5_file in h5_files:

                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=f'T1{exp_extension}', save_r=int(save_round))
                # if '01-27' in outerFolder_expt:
                #     print(load_data)
                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'T1{exp_extension}']:
                    for dataset in range(len(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                        # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        I = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                        #delay_times = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Delay Times', [])[0][dataset])
                        # fit = load_data['T1'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data[f'T1{exp_extension}'][q_key].get('Round Num', [])[0][dataset]
                        try:
                            batch_num = load_data[f'T1{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]
                            syst_config = load_data[f'T1{exp_extension}'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'T1{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
                            #print(exp_config)
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I) > 0:
                            Is[q_key].extend(I)
                            Qs[q_key].extend(Q)
                            gain = round(
                                float(syst_config.split('res_gain_qze\': [')[-1].split(']')[0].split(',')[-1].split('(')[-1].replace(')','')), 6)

                            gains[q_key].append(gain)
                            amp=np.hypot(I, Q)
                            amps[q_key].extend(amp)
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])


                del H5_class_instance
        return Is,Qs,amps, gains

    def run_IBM_qze_rounds(self, exp_extension=''):
        import datetime

        # ----------Load/get data------------------------
        Is = {i: [] for i in range(self.number_of_qubits)}
        Qs = {i: [] for i in range(self.number_of_qubits)}
        amps = {i: [] for i in range(self.number_of_qubits)}
        gains = {i: [] for i in range(self.number_of_qubits)}
        rounds_completed = {i: [] for i in range(self.number_of_qubits)}
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        #print(self.top_folder_dates)
        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"/data/QICK_data/{self.run_name}/" + folder_date + "/study_data"
                outerFolder_save_plots = f"/data/QICK_data/{self.run_name}/" + folder_date + "_plots/"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # ------------------------------------------------Load/Plot/Save T1----------------------------------------------
            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/T1{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/T1_ge/"
            round_we_are_on=outerFolder_expt.split('qubit_0round')[-1].split('/study_data')[0]
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            #print(outerFolder_expt)
            for h5_file in h5_files:

                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=f'T1{exp_extension}', save_r=int(save_round))
                # if '01-27' in outerFolder_expt:
                #     print(load_data)
                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'T1{exp_extension}']:
                    for dataset in range(len(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                        # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        I = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                        #delay_times = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Delay Times', [])[0][dataset])
                        # fit = load_data['T1'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data[f'T1{exp_extension}'][q_key].get('Round Num', [])[0][dataset]
                        try:
                            batch_num = load_data[f'T1{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]
                            syst_config = load_data[f'T1{exp_extension}'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'T1{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
                            #print(exp_config)
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I) > 0:
                            Is[q_key].extend(I)
                            Qs[q_key].extend(Q)

                            gain = round(
                                float(syst_config.split('res_gain_qze\': [')[-1].split(']')[0].split(',')[-1].split('(')[-1].replace(')','')), 6)

                            gains[q_key].append(gain)
                            rounds_completed[q_key].append(round_we_are_on)
                            amp=np.hypot(I, Q)
                            amps[q_key].extend(amp)
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])


                del H5_class_instance
        return Is,Qs,amps, gains, rounds_completed
    def plot_IBM_qze(self,amps,gains, save_path):

        qubit_ids = sorted(amps.keys())  # → [0, 1, 2, 3, 4, 5]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=False, sharey=False)
        axes = axes.flatten()

        for idx, q in enumerate(qubit_ids):
            ax = axes[idx]
            #print(len(gains[q]), len(amps[q]))
            ax.plot(gains[q], [1/n for n in amps[q]], marker='o', linewidth=1)
            ax.set_title(f"Qubit {q}")
            ax.set_xlabel("Pulse gain (a.u.)")
            ax.set_ylabel("1/ T1 Signal amplitude (a.u.)")

        for j in range(len(qubit_ids), len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        self.create_folder_if_not_exists(save_path)
        fig.savefig(save_path + 'gamma.png', transparent=False, dpi=self.final_figure_quality)

        print('Plot saved to: ', save_path)
    def plot_IBM_qze_normal(self,amps,gains, save_path):

        qubit_ids = sorted(amps.keys())  # → [0, 1, 2, 3, 4, 5]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=False, sharey=False)
        axes = axes.flatten()

        for idx, q in enumerate(qubit_ids):
            ax = axes[idx]
            #print(len(gains[q]), len(amps[q]))
            ax.plot(gains[q], amps[q], marker='o', linewidth=1)
            ax.set_title(f"Qubit {q}")
            ax.set_xlabel("Pulse gain (a.u.)")
            ax.set_ylabel("T1 Signal amplitude (a.u.)")

        for j in range(len(qubit_ids), len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        self.create_folder_if_not_exists(save_path)
        fig.savefig(save_path + 't1.png', transparent=False, dpi=self.final_figure_quality)

        print('Plot saved to: ', save_path)
    def plot_IBM_qze_compare(self,amps,gains, rounds,save_path):

        q = 0
        gains_q = np.asarray(gains[q])
        amps_q = np.asarray(amps[q])
        rounds_q = np.asarray(rounds[q])
        fig, ax = plt.subplots(figsize=(6, 4))
        from itertools import cycle
        colour_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for r_id in np.unique(rounds_q):
            mask = rounds_q == r_id
            ax.plot(
                gains_q[mask],
                1.0 / amps_q[mask],
                linestyle="-",
                label=f"Round {r_id}",
                color=next(colour_cycle),
                linewidth=0.8,

            )

        ax.set_title("Qubit 0")
        ax.set_xlabel("Pulse gain (a.u.)")
        ax.set_ylabel("1 / T1 signal amplitude (a.u.)")
        ax.legend(
            loc="center left",  # anchor on the left‐centre of the legend box
            bbox_to_anchor=(1.02, 0.5),  # shift legend just outside the axes
            frameon=False,  # remove legend border
            fontsize="small"  # or an explicit int, e.g. 8
        )
        fig.tight_layout(rect=[0, 0, 0.85, 1])

        self.create_folder_if_not_exists(save_path)
        fig.savefig(save_path + "gamma_q0.png",
                    transparent=False,
                    dpi=self.final_figure_quality)

        print("Plot saved to:", save_path)

    def plot_IBM_qze_normal_compare(self,amps,gains,rounds, save_path):

        q = 0
        gains_q = np.asarray(gains[q])
        amps_q = np.asarray(amps[q])
        rounds_q = np.asarray(rounds[q])
        fig, ax = plt.subplots(figsize=(6, 4))
        from itertools import cycle
        colour_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for r_id in np.unique(rounds_q):
            mask = rounds_q == r_id
            ax.plot(
                gains_q[mask],
                1.0 / amps_q[mask],
                linestyle="-",
                label=f"Round {r_id}",
                color=next(colour_cycle),
                linewidth=0.8,
            )

        ax.set_title("Qubit 0")
        ax.set_xlabel("Pulse gain (a.u.)")
        ax.set_ylabel("T1 signal amplitude (a.u.)")
        ax.legend(
            loc="center left",  # anchor on the left‐centre of the legend box
            bbox_to_anchor=(1.02, 0.5),  # shift legend just outside the axes
            frameon=False,  # remove legend border
            fontsize="small"  # or an explicit int, e.g. 8
        )
        fig.tight_layout(rect=[0, 0, 0.85, 1])

        self.create_folder_if_not_exists(save_path)
        fig.savefig(save_path + "t1_q0.png",
                    transparent=False,
                    dpi=self.final_figure_quality)

        print("Plot saved to:", save_path)

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
        # 2) For each qubit, sort data by timestamp, compute Oadev, and plot
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

            # Compute overlapping Allan deviation
            # Use 'freq' data_type since label is not a phase measure.
            # We'll auto-select tau points with taus='decade' or you could supply np.logspace(...).
            taus_out, ad, ade, ns = allantools.oadev(
                vals_array,
                rate=rate,
                data_type='freq',
                taus='decade'
            )

            # Plot on log axes to mimic a standard Allan plot
            ax.set_xscale('log')
            ax.set_yscale('log')

            ax.plot(taus_out, ad, marker='o', color=colors[i], label=f"Qubit {i + 1}")

            # Optional: plot error bars
            ax.errorbar(taus_out, ad, yerr=ade, fmt='o', color=colors[i])

            if show_legends:
                ax.legend(loc='best', edgecolor='black')

            ax.set_xlabel(r"$\tau$ (s)", fontsize=font - 2)
            ax.set_ylabel(rf"$\sigma_{{{label}}}(\tau)$ (µs)", fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + f'{label}_allan_deviation.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close(fig)
