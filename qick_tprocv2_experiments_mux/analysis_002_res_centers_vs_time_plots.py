import numpy as np
import os
import sys
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import norm
from scipy.optimize import curve_fit

class ResonatorFreqVsTime:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, exp_config, fridge, list_of_all_qubits, outerFolder):
        self.figure_quality = figure_quality
        self.number_of_qubits = number_of_qubits
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.run_name = run_name
        self.outerFolder = outerFolder
        self.top_folder_dates = top_folder_dates
        self.final_figure_quality = final_figure_quality
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

    def run(self):
        import datetime
        # ----------Load/get data------------------------
        resonator_centers = {i: [] for i in range(self.number_of_qubits)}
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

            # ------------------------------------------Load/Plot/Save Res Spec------------------------------------
            outerFolder_expt = self.outerFolder + "/Data_h5/Res_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                # H5_class_instance.print_h5_contents(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type='Res', save_r=int(save_round))

                # just look at this resonator data, should have batch_num of arrays in each one
                # right now the data writes the same thing batch_num of times, so it will do the same 5 datasets 5 times, until you fix this just grab the first one (All 5)

                populated_keys = []
                for q_key in load_data['Res']:
                    # Access 'Dates' for the current q_key
                    dates_list = load_data['Res'][q_key].get('Dates', [[]])

                    # Check if any entry in 'Dates' is not NaN
                    if any(
                            not np.isnan(date)
                            for date in dates_list[0]  # Iterate over the first batch of dates
                    ):
                        populated_keys.append(q_key)

                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                print(populated_keys)
                for q_key in populated_keys:
                    # go through each dataset in the batch and plot
                    for dataset in range(len(load_data['Res'][q_key].get('Dates', [])[0])):
                        date = datetime.datetime.fromtimestamp(
                            load_data['Res'][q_key].get('Dates', [])[0][dataset])  # single date per dataset

                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        freq_pts = self.process_h5_data(load_data['Res'][q_key].get('freq_pts', [])[0][
                                                            dataset].decode())  # comes in as an array but put into a byte string, need to convert to list
                        # print(freq_pts)
                        print(load_data['Res'][q_key].get('freq_center', [])[0][dataset].decode())

                        freq_center = self.process_h5_data(load_data['Res'][q_key].get('freq_center', [])[0][dataset].decode()) # comes in as an array but put into a string, need to convert to list
                        freqs_found = self.string_to_float_list(load_data['Res'][q_key].get('Found Freqs', [])[0][dataset].decode())  # comes in as a list of floats in string format, need to convert
                        amps = self.process_string_of_nested_lists(
                            load_data['Res'][q_key].get('Amps', [])[0][dataset].decode())  # list of lists
                        print('here: ', amps)
                        round_num = load_data['Res'][q_key].get('Round Num', [])[0][dataset]  # already a float
                        batch_num = load_data['Res'][q_key].get('Batch Num', [])[0][dataset]
                        freq_pts_data = load_data['Res'][q_key].get('freq_pts', [])[0][dataset].decode()

                        # Replace whitespace between numbers with commas to make it a valid list
                        formatted_str = freq_pts_data.replace('  ', ',').replace('\n', '')
                        formatted_str = formatted_str.replace(' ', ',').replace('\n', '')
                        formatted_str = formatted_str.replace(',]', ']').replace('\n', '')
                        formatted_str = formatted_str.replace('],[', '],[')
                        formatted_str = re.sub(r",,", ",", formatted_str)
                        formatted_str = re.sub(r",\s*([\]])", r"\1", formatted_str)
                        formatted_str = re.sub(r"(\d+)\.,", r"\1.0,",
                                               formatted_str)  # Fix malformed floating-point numbers (e.g., '5829.,' -> '5829.0')
                        # Convert to NumPy array
                        freq_points = np.array(eval(formatted_str))
                        # print('here: ', freq_points)
                        if len(freq_pts) > 0:
                            res_class_instance = ResonanceSpectroscopy(q_key, self.number_of_qubits, self.list_of_all_qubits, outerFolder_save_plots, round_num,
                                                                       self.save_figs)

                            res_spec_cfg = ast.literal_eval(self.exp_config['res_spec'].decode())
                            res_freqs = res_class_instance.get_results(freq_pts, freq_center, amps)

                            resonator_centers[q_key].extend([res_freqs[q_key]])
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del res_class_instance

                del H5_class_instance
        return date_times, resonator_centers

    def plot(self, date_times, resonator_centers, show_legends):
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
        colors = ['orange','blue','purple','green','brown','pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        titles = [f"Res {i + 1}" for i in range(self.number_of_qubits)]
        from datetime import datetime
        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:  # If we have fewer qubits than subplots, stop plotting and hide the rest
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize = font)

            x = date_times[i]
            y = resonator_centers[i]

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
            # ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))  # Show day and time
            ax.tick_params(axis='x', rotation=45)  # Rotate ticks for better readability

            # Disable scientific notation and format y-ticks
            ax.ticklabel_format(style="plain", axis="y")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))  # 2 decimal places

            # ax.scatter(x, y, color=colors[i])
            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time (Days)', fontsize=font-2)
            ax.set_ylabel('Resonator Center (MHz)', fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Res_Centers.pdf', transparent=True, dpi=self.final_figure_quality)
        print('Saved fig to: ', analysis_folder)
        #plt.show()
