import numpy as np
import os
import sys
sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from tprocv2_demos.qick_tprocv2_experiments_mux.socProxy import makeProxy
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
#from expt_config import *
from collections import OrderedDict
import glob
import re
import datetime
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
                 signal, run_name, fridge, run_number, exp_name = 'ge'):
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.run_number = run_number
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

    # --- helpers: robust flatten + per-delay collapse ---
    def flatten_numeric(self, x): # this is for when you save T1 shots
        """Flatten arbitrarily nested array-likes into a 1D float array."""
        arr = np.asarray(x, dtype=object).ravel()
        if arr.dtype == object:
            parts = []
            for v in arr:
                parts.append(np.asarray(v).ravel())
            arr = np.concatenate(parts) if len(parts) else np.array([], dtype=float)
        return np.asarray(arr, dtype=float).ravel()

    def collapse_per_delay(self, y, N, reducer="mean"): # this is for when you save T1 shots
        """
        Accepts 1D length N (already per-delay), 1D length N*R (flattened shots),
        or 2D shaped (N, R) or (R, N). Returns 1D length N.
        """
        y = np.asarray(y)
        # allow 2D directly
        if y.ndim == 2:
            if y.shape[0] == N:
                Y = y
            elif y.shape[1] == N:
                Y = y.T
            else:
                raise ValueError(f"Unexpected 2D shape {y.shape} for N={N}")
            return np.median(Y, axis=1) if reducer == "median" else np.mean(Y, axis=1)

        # otherwise coerce to flat and handle N or N*R
        y = self.flatten_numeric(y)
        total = y.size
        if total == N:
            return y
        if total % N != 0:
            raise ValueError(f"signal.size={total} not divisible by N_delays={N}")
        R = total // N
        Y = y.reshape(N, R)
        return np.median(Y, axis=1) if reducer == "median" else np.mean(Y, axis=1)

    def run(self, return_errs = False, exp_extension='', saved_shots = False):
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
                outerFolder = f"/data/QICK_data/{self.run_name}/" + folder_date + "/study_data/"
                outerFolder_save_plots = f"/data/QICK_data/{self.run_name}/" + folder_date + "/documentation/"
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
            soc, soccfg = makeProxy()

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
                        # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                        # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data[f't1{exp_extension}'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        # --- NEW: make per-shot data compatible with per-delay fitting --------------------------------
                        if saved_shots:
                            exp_config_str = load_data['t1_ge'][q_key]['Exp Config'][0][dataset].decode()
                            syst_config_str = load_data['t1_ge'][q_key]['Syst Config'][0][dataset].decode()

                            Ishots_raw = self.process_h5_data(load_data['t1_ge'][q_key]['I'][0][dataset].decode())
                            Qshots_raw = self.process_h5_data(load_data['t1_ge'][q_key]['Q'][0][dataset].decode())

                            replica = OfflineAcquireReplica(remove_offset=True, length_norm=True)
                            replica.setup_offline_from_strings(exp_config_str, syst_config_str, soccfg,
                                                               qubit_index=int(q_key))

                            # Authoritative dims from EXP config
                            exp_cfg = replica._safe_eval_cfg(exp_config_str)
                            steps = int(exp_cfg['T1_ge']['steps'])
                            reps = int(exp_cfg['T1_ge']['reps'])
                            # rounds  = int(exp_cfg['T1_ge']['rounds'])   # not used here; H5 holds one round

                            Ishots = replica.coerce_to_rounds_N_reps(Ishots_raw, steps, reps)
                            Qshots = replica.coerce_to_rounds_N_reps(Qshots_raw, steps, reps)

                            # We only provided ONE rounds worth of raw shots from H5 -> tell the replica that
                            I, Q = replica.acquire_offline(Ishots, Qshots,soft_avgs=1)  # we have only done 1 round and a bunch of reps
                        # -----------------------------------------------------------------------------------------------
                        else:
                            I = self.process_h5_data(load_data['t1_ge'][q_key].get('I', [])[0][dataset].decode())
                            Q = self.process_h5_data(load_data['t1_ge'][q_key].get('Q', [])[0][dataset].decode())
                        delay_times = self.process_h5_data(load_data[f't1{exp_extension}'][q_key].get('Delay Times', [])[0][dataset].decode())
                        # fit = load_data['T1'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data[f't1{exp_extension}'][q_key].get('Round Num', [])[0][dataset]

                        # --- NEW: makes per-shot data compatible with per-delay fitting --------------------------------
                        delay_times = self.flatten_numeric(delay_times)
                        N = int(delay_times.size)
                        if N == 0:
                            print("Skipping dataset: empty delay_times")
                            continue

                        # Choose reducer: "mean" (default) or set self.per_delay_reducer="median" upstream
                        reducer = getattr(self, "per_delay_reducer", "mean")

                        try:
                            # If I/Q are (N,R) matrices from save_shots=True, this handles them directly.
                            # If they are flattened (N*R,), it reshapes and reduces to length N.
                            I = self.collapse_per_delay(I, N, reducer=reducer)
                            Q = self.collapse_per_delay(Q, N, reducer=reducer)
                        except ValueError as e:
                            print(f"Skipping dataset due to shape mismatch: {e}")
                            continue
                        #-----------------------------------------------------------------------------------------------

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

                            if (self.run_number == 8) and (q_key != 5) and (T1_est <= 22):  # # QUIET run 8 patch while fitting is fixed
                                print(
                                    f"Skipping T1 = {T1_est:.3f} µs for Q{q_key + 1} because it is presumed to be a bad fit (Run 8 patch).")
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
        # year = 2025
        # month = 1
        # day1 = 22  # Start date
        # day2 = 23  # End date
        # hour_start = 0  # Start hour
        # hour_end = 23  # End hour
        # start_time = datetime(year, month, day1, hour_start, 0)
        # end_time = datetime(year, month, day2, hour_end, 59)
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
            # ax.set_ylim(25, 150)

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


class OfflineAcquireReplica:
    """
    Offline mirror of QICK Averager acquire() math WITHOUT thresholding:
      avg over reps -> divide by ro length -> subtract IQ offset -> sum over rounds -> /soft_avgs
    Returns (I_final, Q_final) each shape (N_steps,).
    """

    def __init__(self, remove_offset=True, length_norm=True, progress=True):
        self.remove_offset = bool(remove_offset)
        self.length_norm   = bool(length_norm)
        self.progress      = bool(progress)

        # QICK-like fields
        self.soccfg = None
        self.ro_chs = OrderedDict()
        self.gen_chs = OrderedDict()
        self.loop_dims = None
        self.avg_level = None
        self.reads_per_shot = None
        self.counter_addr = None

        # cached dims/params
        self._N_steps   = None
        self._reps      = None
        self._rounds    = None
        self._ro_cycles = None
        self._iq_offset = None
        self._ro_index  = None  # readout channel index for this qubit

    # ---------- helpers ----------
    def _safe_eval_cfg(self, cfg_str):
        s = re.sub(r"<qick\.asm_v2\.QickParam object at 0x[0-9a-fA-F]+>", "None", cfg_str)
        s = re.sub(r"np\.float64\(\s*([^)]+)\s*\)", r"float(\1)", s)
        safe_globals = {"np": np, "array": np.array, "float": float, "__builtins__": {}}
        return eval(s, safe_globals)

    def _ensure_rounds_axis(self, A, *, N, rounds, reps):
        """
        Return A shaped as (rounds, N, reps). Accepts 1D/2D/3D.
        """
        A = np.asarray(A)
        if A.ndim == 3:
            return A
        if A.ndim == 2:
            N2, R2 = A.shape
            if N2 != N:
                raise ValueError(f"2D shots first dim {N2} != provided N {N}")
            if R2 % rounds != 0:
                raise ValueError(f"cannot split {R2} into rounds={rounds}")
            if (R2 // rounds) != reps:
                raise ValueError(f"2D shots second dim implies reps={R2//rounds}, expected {reps}")
            return A.reshape(N, rounds, reps).swapaxes(0, 1)
        if A.ndim == 1:
            total = A.size
            if total != N*rounds*reps:
                raise ValueError(f"1D shots size {total} != N*rounds*reps={N*rounds*reps}")
            return A.reshape(rounds, N, reps)
        raise ValueError(f"Expected 1D/2D/3D, got {A.ndim}D {A.shape}")

    def _extract_t1_dims(self, exp_cfg):
        def _as_int(x):
            try: return int(x)
            except Exception: return int(float(str(x)))
        T1 = exp_cfg['T1_ge']
        steps  = (T1.get('steps') or T1.get('n_steps') or T1.get('n_expts'))
        reps   = (T1.get('reps')  or T1.get('nreps'))
        rounds = (T1.get('rounds') or T1.get('soft_avgs'))
        if steps is None or reps is None or rounds is None:
            raise ValueError("Experiment config missing steps/reps/rounds for T1_ge.")
        return _as_int(steps), _as_int(reps), _as_int(rounds)

    def _compute_ro_norm_and_offset(self, *, soccfg, syst_cfg, qubit_index):
        ro_ch_list = syst_cfg['ro_ch']
        res_ch     = syst_cfg['res_ch']
        mixer_freq = float(syst_cfg['mixer_freq'])
        nqz_res    = int(syst_cfg['nqz_res'])
        res_len_us = float(syst_cfg['res_length'])
        res_freqs  = syst_cfg['res_freq_ge']
        ro_phases  = syst_cfg.get('ro_phase', [0]*len(res_freqs))

        ro_ch_for_q = int(ro_ch_list[int(qubit_index)])
        f_q         = float(res_freqs[int(qubit_index)])
        ph_q        = float(ro_phases[int(qubit_index)])

        ro_cycles = int(soccfg.us2cycles(us=res_len_us, ro_ch=ro_ch_for_q))  # length for normalization

        # build ro regs like QICK does to evaluate the PFB doubling condition
        rocfg   = soccfg['readouts'][ro_ch_for_q]
        ro_regs = soccfg.calc_ro_regs(rocfg, phase=ph_q, sel='product')

        ro_ch0_for_rounding = int(ro_ch_list[0])
        mixer_info    = soccfg.calc_mixer_freq(res_ch, mixer_freq, nqz_res, ro_ch0_for_rounding)
        mixer_rounded = mixer_info['rounded']

        ro_pars = {'gen_ch': res_ch, 'freq': f_q}
        soccfg.calc_ro_freq(rocfg, ro_pars, ro_regs,
                            absolute_freqs=False,
                            mixer_freq=mixer_rounded,
                            flip_freq=False)

        offset = float(rocfg['iq_offset'])
        if 'pfb_ch' in ro_regs:
            fs_int = 2**rocfg['b_dds']
            if ro_regs['f_int'] == (ro_regs['pfb_ch'] % 2) * (fs_int//2):
                offset *= 2.0

        return ro_cycles, offset, ro_ch_for_q, ro_regs

    # ---------- setup ----------
    def setup_offline_from_strings(self, exp_config_str, syst_config_str, soccfg, qubit_index):
        exp_cfg  = self._safe_eval_cfg(exp_config_str)
        syst_cfg = self._safe_eval_cfg(syst_config_str)
        self.soccfg = soccfg

        steps, reps, rounds = self._extract_t1_dims(exp_cfg)
        self._N_steps = steps
        self._reps    = reps
        self._rounds  = rounds

        ro_cycles, iq_offset, ro_ch_for_q, _ = self._compute_ro_norm_and_offset(
            soccfg=soccfg, syst_cfg=syst_cfg, qubit_index=qubit_index
        )
        self._ro_cycles = ro_cycles
        self._iq_offset = iq_offset
        self._ro_index  = ro_ch_for_q

        # Mirror QICK bookkeeping: loop_dims = [reps, steps], avg_level = 0 (avg over reps)
        self.ro_chs = OrderedDict({
            ro_ch_for_q: {
                'length': int(ro_cycles),
                'trigs': 1,
                'edge_counting': False,
                'ro_config': self.soccfg['readouts'][ro_ch_for_q]
            }
        })
        self.loop_dims      = [self._reps, self._N_steps]
        self.avg_level      = 0
        self.reads_per_shot = [1]

    def coerce_to_rounds_N_reps(self, A, steps, reps):
        A = np.asarray(A)
        if A.ndim == 3:
            # assume already (rounds, N, reps)
            return A
        if A.ndim == 2:
            r0, r1 = A.shape
            # common cases: (reps, N) or (N, reps)
            if r0 == reps and r1 == steps:
                return A.T[None, ...]  # -> (1, N, reps)
            if r0 == steps and r1 == reps:
                return A[None, ...]  # -> (1, N, reps)
            raise ValueError(
                f"Unexpected 2D shape {A.shape}; expected (reps,N)=({reps},{steps}) or (N,reps)=({steps},{reps}).")
        if A.ndim == 1:
            total = A.size
            if total == steps * reps:
                return A.reshape(steps, reps)[None, ...]
            if total == steps:
                return A.reshape(1, steps, 1)
            raise ValueError(f"Unexpected 1D length {total}; cannot infer (N,reps) from steps={steps}, reps={reps}.")
        raise ValueError(f"Shots must be 1D/2D/3D, got {A.ndim}D {A.shape}")


    # ---------- QICK-faithful averaging ----------
    def _ro_offset_qick(self, ro_ch, chcfg):
        rocfg = self.soccfg['readouts'][ro_ch]
        offset = rocfg['iq_offset']
        if chcfg is not None and 'pfb_ch' in chcfg:
            fs_int = 2**rocfg['b_dds']
            if chcfg['f_int'] == (chcfg['pfb_ch'] % 2) * (fs_int//2):
                offset *= 2
        return float(offset)

    def _average_buf_qick(self, d_reps, reads_per_shot, *, length_norm=True, remove_offset=True):
        avg_d = []
        for i_ch, (ch, ro) in enumerate(self.ro_chs.items()):
            # average over avg_level (==0) i.e. over reps
            avg = d_reps[i_ch].sum(axis=self.avg_level) / self.loop_dims[self.avg_level]
            if length_norm and not ro['edge_counting']:
                avg = avg / ro['length']
                if remove_offset:
                    avg -= self._ro_offset_qick(ch, ro.get('ro_config'))
            # move reads_per_shot axis to front (we have 1 read)
            avg_d.append(np.moveaxis(avg, -2, 0))  # -> (1, steps, 2)
        return avg_d

    # ---------- public: acquire offline (no thresholding) ----------
    def acquire_offline(self, Ishots, Qshots, *, soft_avgs=None):
        """
        Inputs:
          Ishots, Qshots shaped like (rounds, N, reps) or flattenable to that.
        Output:
          (I_final, Q_final) each shape (N,)
        """
        if any(x is None for x in (self._N_steps, self._reps, self._rounds, self._ro_cycles)):
            raise RuntimeError("Call setup_offline_from_strings(...) first.")

        if soft_avgs is None:
            soft_avgs = self._rounds
        if int(soft_avgs) != self._rounds:
            # we sum 'soft_avgs' rounds; this mirrors QICK's software averaging
            pass

        I3 = self._ensure_rounds_axis(Ishots, N=self._N_steps, rounds=self._rounds, reps=self._reps)
        Q3 = self._ensure_rounds_axis(Qshots, N=self._N_steps, rounds=self._rounds, reps=self._reps)

        # accumulate per-round like QICK does, using the same averaging kernel
        summed = None
        for r in range(self._rounds):
            # pack this round like acc_buf: (reps, steps, 1, 2)
            I_round = I3[r]  # (N, reps)
            Q_round = Q3[r]
            packed = np.zeros((self._reps, self._N_steps, 1, 2), dtype=np.int64)
            packed[..., 0, 0] = I_round.T  # (reps, steps)
            packed[..., 0, 1] = Q_round.T

            round_avg_list = self._average_buf_qick([packed], self.reads_per_shot,
                                                    length_norm=self.length_norm,
                                                    remove_offset=self.remove_offset)
            # round_avg_list[0] has shape (1, steps, 2)
            round_avg = round_avg_list[0][0]  # (steps, 2)

            if summed is None:
                summed = round_avg.copy()
            else:
                summed += round_avg

        summed /= float(soft_avgs)  # software average, like QICK

        I_final = summed[:, 0]  # (N,)
        Q_final = summed[:, 1]
        return I_final, Q_final

