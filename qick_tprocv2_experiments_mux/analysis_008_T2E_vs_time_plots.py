import numpy as np
import os
import sys
sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))

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
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import norm
from scipy.optimize import curve_fit

class T2eVsTime:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, exp_config, fridge, list_of_all_qubits):
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.figure_quality = figure_quality
        self.run_name = run_name
        self.number_of_qubits = number_of_qubits
        self.final_figure_quality = final_figure_quality
        self.top_folder_dates = top_folder_dates
        self.exp_config = exp_config
        self.fridge = fridge
        self.list_of_all_qubits = list_of_all_qubits

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

    def exponential(x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def optimal_bins(data):
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

    def run(self):
        import datetime

        # ----------Load/get data------------------------
        t2e_vals = {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}

        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"/data/QICK_data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/data/QICK_data/{self.run_name}/" + folder_date + "_plots/"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # -------------------------------------------------------Load/Plot/Save T2E------------------------------------------
            outerFolder_expt = outerFolder + "/Data_h5/T2E_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type='T2E', save_r=int(save_round))

                for q_key in load_data['T2E']:
                    for dataset in range(len(load_data['T2E'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data['T2E'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T2 = load_data['T2E'][q_key].get('T2', [])[0][dataset]
                        # errors = load_data['T2E'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data['T2E'][q_key].get('Dates', [])[0][dataset])
                        I = self.process_h5_data(load_data['T2E'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data['T2E'][q_key].get('Q', [])[0][dataset].decode())
                        delay_times = self.process_h5_data(load_data['T2E'][q_key].get('Delay Times', [])[0][dataset].decode())
                        # fit = load_data['T2E'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data['T2E'][q_key].get('Round Num', [])[0][dataset]
                        batch_num = load_data['T2E'][q_key].get('Batch Num', [])[0][dataset]

                        if len(I) > 0:
                            T2E_class_instance = T2EMeasurement(q_key, self.list_of_all_qubits, outerFolder_save_plots, round_num, self.signal, self.save_figs,
                                                               fit_data=True)
                            try:
                                fitted, t2e_est, t2e_err, plot_sig = T2E_class_instance.t2_fit(delay_times, I, Q)
                            except Exception as e:
                                print(f"good fit not found, error: {e}")
                                continue
                            T2E_cfg = ast.literal_eval(self.exp_config['SpinEcho_ge'].decode())
                            if t2e_est < 0:
                                print("The value is negative, continuing...")
                                continue
                            if t2e_est > 50:
                                print("The value is above 50 us, this is a bad fit, continuing...")
                                continue
                            t2e_vals[q_key].extend([t2e_est])
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del T2E_class_instance
                del H5_class_instance
        return date_times, t2e_vals

    def plot(self, date_times, t2e_vals, show_legends):
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

        font = 14
        titles = [f"Qubit {i+1}" for i in range(self.number_of_qubits)]
        colors = ['orange','blue','purple','green','brown','pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('T2E Values vs Time',fontsize = font)
        axes = axes.flatten()

        from datetime import datetime
        for i, ax in enumerate(axes):

            if i >= self.number_of_qubits: # If we have fewer qubits than subplots, stop plotting and hide the rest
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize = font)

            x = date_times[i]
            y = t2e_vals[i]

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

            sorted_x = np.asarray(sorted(x))

            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically choose good tick locations
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))  # Format as month-day
            ax.tick_params(axis='x', rotation=45)  # Rotate ticks for better readability

            # Disable scientific notation and format y-ticks
            ax.ticklabel_format(style="plain", axis="y")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))  # 2 decimal places

            ax.scatter(x, y, color=colors[i])
            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time (Days)', fontsize=font-2)
            ax.set_ylabel('T2E (us)', fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2E_vals.png', transparent=False, dpi=self.final_figure_quality)

        #plt.show()
