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
from match_h5files_to_pngs_get_timestamps import load_h5_png_map, create_h5_png_map
import glob
import re
from pathlib import Path
import datetime
import ast
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import norm
from scipy.optimize import curve_fit

class QubitFreqsVsTime:
    def __init__(self, base_data_path, plots_path, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name,  fridge):
        self.save_figs = save_figs
        self.plots_path = plots_path
        self.fit_saved = fit_saved
        self.signal = signal
        self.base_data_path = base_data_path
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

    def run(self,exp_extension='', use_png_timestamps = False):
        import datetime

        if use_png_timestamps:
            # This is a setting used to extract the timestamps in the png file names instead of using the ones
            # stored inside the h5 files (which mark the time that the file was saved, not when the meas was done).
            # --- loader for the h5–png map -----------------
            map_loader = load_h5_png_map()

        qubit_frequencies = {i: [] for i in range(self.number_of_qubits)}
        qspec_fit_errs= {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                timestamp_dir = os.path.join(self.base_data_path, folder_date)
                if "run6" in self.run_name: # For QUIET, science run data was saved differently
                    if "ge_round_robin_presciencerun_data" in folder_date:
                        outerFolder = timestamp_dir + "/study_data/" # where data is stored
                    else:
                        outerFolder = timestamp_dir + "/optimization/" # where data is stored
                else:
                    outerFolder = timestamp_dir + "/study_data/"  # where data is stored

                #outerFolder_save_plots = timestamp_dir + "/documentation/" # where plots will be stored
                print("Looking inside: ", outerFolder)

            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                #outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            if use_png_timestamps:
                # --- load the mapping HDF5 for this timestamp_dir, if it exists ---
                map_path = os.path.join(timestamp_dir, "documentation/h5_png_timestamp_map.h5")
                if os.path.exists(map_path):
                    mapping_data = map_loader.load_map(map_path)

                else:
                    print(f"[INFO] Mapping file not found at {map_path}.")
                    print(f"[INFO] Attempting to create a new mapping...")

                    # Instantiate mapping creator
                    mapper = create_h5_png_map()

                    try:
                        # Run mapping creation for this timestamp_dir
                        records = mapper.collect_matches(Path(timestamp_dir))

                        # Save mapping to the expected path
                        mapper.save_to_h5(Path(map_path), Path(timestamp_dir), records)

                        # Load the newly created mapping
                        mapping_data = map_loader.load_map(map_path)

                        print(f"[INFO] Successfully created mapping at {map_path}.")

                    except Exception as e:
                        print(f"[WARN] Failed to create mapping: {e}")
                        # print("[WARN] Falling back to HDF5 timestamps instead.")
                        mapping_data = None

            # ------------------------------------------Load/Plot/Save Q Spec------------------------------------
            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/qspec{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/qspec_ge/"


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
                    # Run 9 patch, accidentally took punched out data for Q4, bad.
                    if ("AB_paper_data_batch1_25dB_DACatten_noQ5/2026-04-17_00-34-47" in folder_date
                        and int(q_key) == 3):
                        print(f"Skipping Q4 data due to punchout in {self.run_name}/{folder_date}")
                        continue
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
                        # I_fit = load_data['QSpec'][q_key].get('I Fit', [])[0][dataset]
                        # Q_fit = load_data['QSpec'][q_key].get('Q Fit', [])[0][dataset]
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
                            qspec_class_instance = QubitSpectroscopy(q_key, self.number_of_qubits, self.plots_path, round_num, self.signal,
                                                                     self.save_figs)
                            # if '_' in exp_extension:
                            #     q_spec_cfg = exp_config[f'qubit_spec{exp_extension}']
                            # else:
                            #     q_spec_cfg = exp_config['qubit_spec_ge']
                            largest_amp_curve_mean, I_fit, Q_fit, qspec_fit_err, largest_amp_curve_fwhm, signal = qspec_class_instance.get_results(I, Q, freqs)

                            # -- Quality cuts --
                            # Require the fitted frequency to be near a peak or dip in the raw data that was used to find ge qfreq
                            # Vertical line should not be farther than 0.5MHz from found peak/dip
                            good_center = (
                                    largest_amp_curve_mean is not None
                                    and largest_amp_curve_fwhm is not None
                                    and signal is not None
                                    and np.isfinite(largest_amp_curve_mean)
                                    and np.isfinite(largest_amp_curve_fwhm)
                                    and (
                                        min(abs(largest_amp_curve_mean - freqs[np.argmin(I)]),
                                            abs(largest_amp_curve_mean - freqs[np.argmax(I)]) ) < 0.5 if signal == "I"
                                        else
                                        min(abs(largest_amp_curve_mean - freqs[np.argmin(Q)]),
                                            abs(largest_amp_curve_mean - freqs[np.argmax(Q)]) ) < 0.5 )
                                        )

                            good_fit = (
                                    qspec_fit_err is not None
                                    and np.isfinite(qspec_fit_err)
                                    and qspec_fit_err < 1.0 # above 1 MHz fit err is probably not a good fit
                                    and 0.01 < largest_amp_curve_fwhm < 10.0 # width of peak
                                    and good_center
                            )

                            if good_fit:
                                qubit_frequencies[q_key].extend([largest_amp_curve_mean])
                                qspec_fit_errs[q_key].extend([qspec_fit_err])

                                # # If you want to look at scans that made it through, uncomment this:
                                # qspec_class_instance.plot_results(I, Q, freqs)

                                if use_png_timestamps:
                                    # --- use PNG filename timestamp from mapping if available ------
                                    # the reason for this is bc the png timestamp is more accurate than the h5 file ones
                                    if mapping_data is not None:
                                        # mapping uses experiment='t1_ge', qubit as 1-indexed
                                        qubit_in_map = q_key + 1
                                        subset = map_loader.filter_by(
                                            mapping_data,
                                            experiment=f"qspec{exp_extension}",
                                            qubit=qubit_in_map,
                                            round=round_num)

                                        if len(subset) > 0:
                                            png_ts = subset[0]["png_timestamp"].decode()
                                            try:
                                                png_dt = datetime.datetime.strptime(png_ts, "%Y-%m-%d_%H-%M-%S")
                                                date_str = png_dt.strftime("%Y-%m-%d %H:%M:%S")  # from png file
                                            except Exception:
                                                # in case of weird format, fall back
                                                # date_str = date.strftime("%Y-%m-%d %H:%M:%S") # from h5 file
                                                continue  # skip
                                        else:
                                            # no mapping match for this qubit/round, fall back
                                            # date_str = date.strftime("%Y-%m-%d %H:%M:%S") # from h5 file
                                            continue  # skip
                                    else:
                                        # no mapping file for this timestamp_dir, fall back
                                        # date_str = date.strftime("%Y-%m-%d %H:%M:%S") # from h5 file
                                        continue  # skip
                                    date_times[q_key].append(date_str)

                                else:
                                    date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])  # og way, from h5 file

                            ## If you want to look at scans that failed quality cuts, uncomment this:
                            # else:
                            #     if (q_key == 1 or q_key == 4) and "run6" in self.run_name:
                            #         qspec_class_instance.plot_results(I, Q, freqs)
                            del qspec_class_instance

                del H5_class_instance
        return date_times, qubit_frequencies, qspec_fit_errs

    def plot_without_errs(self, date_times, qubit_frequencies, show_legends):
        # ---------------------------------plot-----------------------------------------------------
        self.create_folder_if_not_exists(self.plots_path)
        analysis_folder = os.path.join(self.plots_path, "features_vs_time/")
        self.create_folder_if_not_exists(analysis_folder)
        
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
        analysis_folder = f"{self.plots_path}/histograms/"
        self.create_folder_if_not_exists(analysis_folder)

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
        # ---------------------------------plot path-----------------------------------------------------
        self.create_folder_if_not_exists(self.plots_path)
        analysis_folder = os.path.join(self.plots_path, "features_vs_time/")
        self.create_folder_if_not_exists(analysis_folder)

        font = 18
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        ext = exp_extension.split('_')[0]
        plt.suptitle(f'g-e Qubit Frequencies (MHz) vs Time {ext}', fontsize=font)
        axes = axes.flatten()

        from datetime import datetime

        # -------- preprocess once: sort/clean/store data and find common width --------
        processed_data = []
        global_width = 0

        for i in range(self.number_of_qubits):
            x = date_times[i]
            y = qubit_frequencies[i]
            err = qspec_fit_err[i]

            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]
            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])

            if len(combined) == 0:
                processed_data.append(None)
                continue

            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)
            sorted_y, sorted_x, sorted_err = self.remove_none_values(sorted_y, sorted_x, sorted_err)

            if len(sorted_y) == 0:
                processed_data.append(None)
                continue

            sorted_y = np.array(sorted_y, dtype=float)
            sorted_err = np.array(sorted_err, dtype=float)

            local_min = np.min(sorted_y - sorted_err)
            local_max = np.max(sorted_y + sorted_err)
            global_width = max(global_width, local_max - local_min)

            processed_data.append((sorted_x, sorted_y, sorted_err))

        # add padding so points/error bars are not pressed against the borders
        padding_fraction = 0.15  # try 0.20 if you want even more room
        global_width *= (1 + 2 * padding_fraction)

        # -------- plotting loop --------
        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits or processed_data[i] is None:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            sorted_x, sorted_y, sorted_err = processed_data[i]

            center = np.mean(sorted_y)
            ax.set_ylim(center - global_width / 2, center + global_width / 2)

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

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.tick_params(axis='x', rotation=45)

            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

            if show_legends:
                ax.legend(edgecolor='black')

            ax.set_xlabel('Time', fontsize=16)
            ax.set_ylabel('Freq (MHz)', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=10)

        plt.tight_layout()
        plt.savefig(
            analysis_folder + f'Q_Freqs{exp_extension}.pdf',
            transparent=True,
            dpi=self.final_figure_quality
        )
        print('Plot saved to:', analysis_folder)
        plt.close()

    def plot_with_errs_single_plot(self, date_times, qubit_frequencies, qspec_fit_err, show_legends):
        # ---------------------------------folder setup-----------------------------------------------------
        self.create_folder_if_not_exists(self.plots_path)
        analysis_folder = os.path.join(self.plots_path, "features_vs_time/")
        self.create_folder_if_not_exists(analysis_folder)

        from datetime import datetime
        year = 2025
        month = 1
        day1 = 24  # Start date
        day2 = 25  # End date
        hour_start = 0  # Start hour
        hour_end = 12  # End hour
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 0)

        font = 18
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
        ax.tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Q_Freqs_single_plot.pdf', dpi=self.final_figure_quality)
        plt.close()

