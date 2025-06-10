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

class QubitLineWidths:
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
        fwhms = {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"/data/QICK_data/{self.run_name}/" + folder_date + "/study_data/"
                outerFolder_save_plots = f"/data/QICK_data/{self.run_name}/" + folder_date + "/documentation/"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # ------------------------------------------Load/Plot/Save Q Spec------------------------------------
            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/QSpec{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/QSpec_ge/"


            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            print(outerFolder_expt)
            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]

                H5_class_instance = Data_H5(h5_file)
                #H5_class_instance.print_h5_contents(h5_file)
                #sometimes you get '1(1)' when redownloading the h5 files for some reason
                load_data = H5_class_instance.load_from_h5(data_type=f'QSpec{exp_extension}', save_r=int(save_round.split('(')[0]))

                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'QSpec{exp_extension}']:
                    for dataset in range(len(load_data[f'QSpec{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'QSpec{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        date = datetime.datetime.fromtimestamp(load_data[f'QSpec{exp_extension}'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        I = self.process_h5_data(load_data[f'QSpec{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data[f'QSpec{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                        # I_fit = load_data['QSpec'][q_key].get('I Fit', [])[0][dataset]
                        # Q_fit = load_data['QSpec'][q_key].get('Q Fit', [])[0][dataset]
                        freqs = self.process_h5_data(load_data[f'QSpec{exp_extension}'][q_key].get('Frequencies', [])[0][dataset].decode())
                        round_num = load_data[f'QSpec{exp_extension}'][q_key].get('Round Num', [])[0][dataset]
                        batch_num = load_data[f'QSpec{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]
                        try:
                            syst_config = load_data[f'QSpec{exp_extension}'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'QSpec{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
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
                            largest_amp_curve_mean, I_fit, Q_fit, qspec_fit_err, fwhm = qspec_class_instance.get_results(I, Q, freqs, return_fwhm=True)
                            if qspec_fit_err is not None and qspec_fit_err < 1 and fwhm>0: #above 1 MHz fit err is probably not a good fit
                                qubit_frequencies[q_key].extend([largest_amp_curve_mean])
                                qspec_fit_errs[q_key].extend([qspec_fit_err])
                                fwhms[q_key].extend([fwhm])
                                date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del qspec_class_instance

                del H5_class_instance
        return date_times, qubit_frequencies, qspec_fit_errs, fwhms

    def plot_without_errs(self, date_times, qubit_frequencies, show_legends, ext=''):
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
            ax.set_ylabel(f'{ext} transition linewiths (MHz)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + f'Qubit_linewidths_{ext}.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close()

    def plot_hist(self,  qubit_frequencies, show_legends, ext=''):
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



            # Ignore the date_times; only use qubit_frequencies.
            y = qubit_frequencies[i]

            if len(y) == 0:
                # If this qubit has no data, hide the subplot.
                ax.set_visible(False)
                continue
            y = self.remove_none_values_1D(y)

            optimal_bin_num = 45  # self.optimal_bins(t1_vals[i])

            # Fit a Gaussian to the raw data instead of the histogram
            # get the mean and standard deviation of the data
            mu_1, std_1 = norm.fit(y)

            # Generate x values for plotting a gaussian based on this mean and standard deviation
            x_1 = np.linspace(np.min(y), np.max(y), optimal_bin_num)
            p_1 = norm.pdf(x_1, mu_1, std_1)

            # Calculate histogram data for t1_vals[i]
            hist_data_1, bins_1 = np.histogram(y, bins=optimal_bin_num)
            bin_centers_1 = (bins_1[:-1] + bins_1[1:]) / 2
            ax.set_title(titles[i] + r' $\mu$' +f' {round(mu_1,2)},' + r'$\sigma$' + f' {round(std_1,2)} MHz', fontsize=font)
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
            ax.hist(y, bins=optimal_bin_num, alpha=0.7, color=colors[i], edgecolor='black')

            # Plot histogram of the frequency data.

            means.append(np.mean(y))
            if show_legends:
                ax.legend([f"Freq Data Qubit {i + 1}"], edgecolor='black')
            ax.set_xlabel(f'{ext} transition linewiths (MHz)', fontsize=font - 2)
            ax.set_ylabel('Count', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + f'Q_linewidths_no_errs_{ext}.pdf', transparent=True, dpi=self.final_figure_quality)
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