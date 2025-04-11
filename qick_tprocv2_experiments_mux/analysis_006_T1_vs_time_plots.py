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
import matplotlib.patches as mpatches
# import allantools #commented this out for now since it was giving the error ModuleNotFoundError: No module named 'allantools'
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter

class T1VsTime:
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
        self.list_of_all_qubits = list_of_all_qubits
        self.fridge = fridge

        t1_ge_str = self.exp_config['T1_ge'].decode('utf-8')
        t1_ge_dict = ast.literal_eval(t1_ge_str)
        self.reps = t1_ge_dict['reps']
        self.rounds = t1_ge_dict['rounds']

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

    def run(self, return_errs = False):
        import datetime

        # New list to collect Q2 data with the desired T1 values
        q2_combined_fit_data = []

        # ----------Load/get data------------------------
        t1_vals = {i: [] for i in range(self.number_of_qubits)}
        t1_errs = {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        #print(self.top_folder_dates)
        t1_errors = {i: [] for i in range(self.number_of_qubits)}

        cutoff_date = datetime.datetime.strptime('2025-02-28', '%Y-%m-%d')
        for folder_date in self.top_folder_dates:
            current_date = datetime.datetime.strptime(folder_date, '%Y-%m-%d')
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"/data/QICK_data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/data/QICK_data/{self.run_name}/" + folder_date + "_plots/"
            elif self.fridge.upper() == 'NEXUS':
                if current_date < cutoff_date:
                    #For Regular RR at NEXUS
                    outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                    outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"

                else:
                    # #For Fast RR at NEXUS
                    outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/Fast_RR/" + folder_date + "/"
                    outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/Fast_RR/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # ------------------------------------------------Load/Plot/Save T1----------------------------------------------
            outerFolder_expt = outerFolder + "/Data_h5/T1_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            #print(outerFolder_expt)
            for h5_file in h5_files:

                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type='T1', save_r=int(save_round))
                # if '01-27' in outerFolder_expt:
                #     print(load_data)
                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31),  # Optimization Issues and non RR work in progress
                    datetime.date(2025, 2, 11),  # TWPA optimization work, fridge pressure issues, touch tests at nexus
                }

                for q_key in load_data['T1']:
                    for dataset in range(len(load_data['T1'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data['T1'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                        # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data['T1'][q_key].get('Dates', [])[0][dataset])

                        # Disregard data for qubit 4 (key 3) in the specific timeframe.
                        if q_key == 3 and datetime.datetime(2025, 3, 11, 21, 44, 8) <= date <= datetime.datetime(2025,
                                                            3, 11,22, 45,36):
                            print(f"Skipping data for qubit 4 at {date} (specific timeframe), due to failed Pi Amp detected by Arianna")
                            continue

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        I = self.process_h5_data(load_data['T1'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data['T1'][q_key].get('Q', [])[0][dataset].decode())
                        delay_times = self.process_h5_data(load_data['T1'][q_key].get('Delay Times', [])[0][dataset].decode())
                        # fit = load_data['T1'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data['T1'][q_key].get('Round Num', [])[0][dataset]
                        batch_num = load_data['T1'][q_key].get('Batch Num', [])[0][dataset]

                        if len(I) > 0:
                            T1_class_instance = T1Measurement(q_key, self.number_of_qubits, self.list_of_all_qubits, outerFolder_save_plots, round_num, self.signal, self.save_figs,
                                                              fit_data=True)
                            T1_spec_cfg = ast.literal_eval(self.exp_config['T1_ge'].decode())
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
                            t1_errors[q_key].append(T1_err)

                            # --- NEW BLOCK: Collect data of two specified T1 fits for one qubit ---
                            # if q_key == 3 and (round(T1_est, 2) == 20.74 or round(T1_est, 2) == 25.38):
                            #     q2_combined_fit_data.append({
                            #         'delay_times': delay_times,
                            #         'fit': q1_fit_exponential,
                            #         'T1_est': T1_est,
                            #         'plot_sig': plot_sig,
                            #         'I': I
                            #     })
                            # -------------------------------------------------------

                            del T1_class_instance

                del H5_class_instance

        # --------------------------------------New, plotting two T1 fits in a single plot-----------------------------
        # After processing all files, if we collected any Q2 datasets with the desired T1 values,
        # plot their fit curves on a combined figure.
        # print(q2_combined_fit_data)
        # print(len(q2_combined_fit_data))
        # if len(q2_combined_fit_data) == 2:  # Ensure exactly two fits are available
        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     plt.rcParams.update({'font.size': 18})
        #
        #     # Center title above the subplot
        #     plot_middle = (ax.get_position().x0 + ax.get_position().x1) / 2
        #     fig.text(plot_middle, 0.98, "T1 Fits for Q4", fontsize=24, ha='center', va='top')
        #
        #     # Plot both fits
        #     for idx, data in enumerate(q2_combined_fit_data):
        #         # color = f"C{idx}"  # Use different colors for each dataset
        #         color = 'grey'
        #
        #         #data
        #         # ax.plot(data['delay_times'], data['Q'], color=color, linestyle='-', linewidth=1.5, label=f"Raw Q Data (T1={data['T1_est']:.2f} µs)")
        #         ax.plot(data['delay_times'], data['I'], color=color, linestyle='-', linewidth=1.5, label="_nolegend_")
        #
        #         #fits
        #         ax.plot(data['delay_times'], data['fit'], linewidth=3, label=f"T1={data['T1_est']:.2f} µs")
        #
        #     # Formatting
        #     ax.set_xlabel("Delay time (us)", fontsize=20)
        #     ax.set_ylabel("I Amplitude (a.u.)", fontsize=20)
        #     ax.tick_params(axis='both', which='major', labelsize=16)
        #
        #     # Add legend
        #     ax.legend()
        #
        #     # Save the plot
        #     analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
        #     combined_folder = os.path.join(analysis_folder, "Combined_Fits")
        #     self.create_folder_if_not_exists(combined_folder)
        #     now = datetime.datetime.now()
        #     formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        #     file_name = os.path.join(combined_folder, f"Combined_T1_Q4_fits_{formatted_datetime}.png")
        #     fig.savefig(file_name, dpi=self.figure_quality, bbox_inches='tight')
        #     plt.close(fig)
        # --------------------------------------------------------------------------------------------------

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
            #For regular RR at NEXUS
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)

            #For fast RR at NEXUS
            # analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/Fast_RR/benchmark_analysis_plots/"
            # self.create_folder_if_not_exists(analysis_folder)
            # analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/Fast_RR/benchmark_analysis_plots/features_vs_time/"
            # self.create_folder_if_not_exists(analysis_folder)

        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        #----------------To Plot a specific timeframe------------------
        # from datetime import datetime
        # year = 2025
        # month = 2
        # day1 = 6  # Start date
        # day2 = 13  # End date
        # hour_start = 12  # Start hour
        # hour_end = 16  # End hour
        # start_time = datetime(year, month, day1, hour_start, 0)
        # end_time = datetime(year, month, day2, hour_end, 59)
        #-----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i+1}" for i in range(self.number_of_qubits)]
        colors = ['orange','blue','purple','green','brown','pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('T1 Values vs Time',fontsize = font)
        axes = axes.flatten()

        #---------- fixed y axis ticks ---------------
        fixed_y_min = 8 # Set the fixed y-axis lower limit
        fixed_y_max = 30  # Set the fixed y-axis upper limit
        fixed_step_size = 2  # Set y-axis step size

        y_ticks = np.arange(fixed_y_min, fixed_y_max + fixed_step_size, fixed_step_size)
        #-----------------------------------------------

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

            #no error bars
            ax.scatter(sorted_x, sorted_y, color=colors[i])

            # Set x-axis limits for the specific timeframe
            # ax.set_xlim(start_time, end_time)

            sorted_x = np.asarray(sorted(x))
            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically choose good tick locations
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))  # Format as month-day
            # ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))  # Show day and time
            ax.tick_params(axis='x', rotation=45)  # Rotate ticks for better readability

            ax.set_yticks(y_ticks)  # Apply uniform y-ticks
            ax.set_ylim(fixed_y_min, fixed_y_max)  # Set fixed y-axis range

            # Disable scientific notation and format y-ticks
            ax.ticklabel_format(style="plain", axis="y")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))  #decimal places


            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time (Days)', fontsize=font-2)
            ax.set_ylabel('T1 (us)', fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()

        plt.savefig(analysis_folder + 'T1_vals.png', transparent=False, dpi=self.final_figure_quality)
        print('Plot saved at: ', analysis_folder)
        # plt.show()
        plt.close()

    def plot_with_errs(self, date_times, t1_vals, t1_fit_err, show_legends):
        # ---------------------------------plot-----------------------------------------------------
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            # For regular RR at NEXUS
            # analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            # self.create_folder_if_not_exists(analysis_folder)
            # analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            # self.create_folder_if_not_exists(analysis_folder)

            # For fast RR at NEXUS
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/Fast_RR/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/Fast_RR/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        # ----------------To Plot a specific timeframe------------------
        # from datetime import datetime
        # year = 2025
        # month1 = 3
        # month2 = 3
        # day1 = 7  # Start date
        # day2 = 8  # End date
        # hour_start = 14  # Start hour
        # hour_end = 1  # End hour
        # start_time = datetime(year, month1, day1, hour_start, 0)
        # end_time = datetime(year, month2, day2, hour_end, 0)
        # -----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        # colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        colors = ['black', 'black', 'black', 'black', 'black', 'black']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.suptitle('T1 Values vs Time', fontsize=font)
        axes = axes.flatten()

        # ---------- fixed y axis ticks ---------------
        fixed_y_min = 8  # Set the fixed y-axis lower limit
        fixed_y_max = 30  # Set the fixed y-axis upper limit
        fixed_step_size = 2  # Set y-axis step size

        y_ticks = np.arange(fixed_y_min, fixed_y_max + fixed_step_size, fixed_step_size)

        import matplotlib.dates as mdates
        from matplotlib.ticker import StrMethodFormatter
        from datetime import datetime

        #-------------------------------------------------------------------------------------
        # Define background intervals and corresponding named colors.
        bg_intervals = [
            # (datetime(2025, 1, 21), datetime(2025, 2, 3)),  # Jan 21-Feb 2
            # (datetime(2025, 2, 4), datetime(2025, 2, 18)),  # Feb 4-17
            # (datetime(2025, 2, 18), datetime(2025, 2, 28)),  # Feb 18-27

            (datetime(2025, 2, 28), datetime(2025, 3, 1)),  # Feb 28
            (datetime(2025, 3, 6), datetime(2025, 3, 8)),  # March 6-7
            (datetime(2025, 3, 11), datetime(2025, 3, 13)),  # March 11-12
            (datetime(2025, 3, 13), datetime(2025, 3, 14)),  # March 13
            (datetime(2025, 3, 14), datetime(2025, 3, 15)),  # March 14
            (datetime(2025, 3, 15), datetime(2025, 3, 18)),  # March 15-16
            (datetime(2025, 3, 25), datetime(2025, 3, 27))  # March 25 and a little bit of the 26th after 12am
        ]
        bg_colors = plt.get_cmap("tab10").colors[:10]
        bg_labels = [
            # "Jan 21 - Feb 2: No sources, regular RR",
            # "Feb 4-17: Ba source, regular RR",
            # "Feb 18-27: Ba source (Close, regular RR)",

            "Feb 28: Ba source (Close, fast RR)",
            "Mar 6-7: Cs source, 9 sheets, fast RR",
            "Mar 11-12: No sources, fast RR",
            "Mar 13: Cs source, 6 sheets, fast RR",
            "Mar 14: Cs source, 3 sheets, fast RR",
            "Mar 15-16: Cs source, no sheets, fast RR",
            "Mar 25-26: No sources, fast RR"
        ]
        # Create Patch objects for the intervals so we can make a single figure-level legend.
        # interval_patches = [
        #     mpatches.Patch(facecolor=c, alpha=0.6, label=lbl)
        #     for c, lbl in zip(bg_colors, bg_labels)
        # ]
        #-----------------------------------------------------------------------------------------

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

            #limit x axis
            # ax.set_xlim(start_time, end_time)

            # -----------------Add background shading--------------------
            # for (start, end), shade_color in zip(bg_intervals, bg_colors):
            #     ax.axvspan(start, end, facecolor=shade_color, alpha=0.6)
            # -----------------------------------------------------------

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

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically choose good tick locations

            # start_ordinal = mdates.date2num(start_time)
            # end_ordinal = mdates.date2num(end_time)
            # num_ticks = 9
            # tick_positions = np.linspace(start_ordinal, end_ordinal, num_ticks)
            # ax.set_xticks(tick_positions)

            # ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))  # Format as month-day
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))  # Show day and time
            ax.tick_params(axis='x', rotation=45)  # Rotate ticks for better readability

            ax.set_yticks(y_ticks)  # Apply uniform y-ticks
            ax.set_ylim(fixed_y_min, fixed_y_max)  # Set fixed y-axis range

            # Disable scientific notation and format y-ticks
            ax.ticklabel_format(style="plain", axis="y")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))  # decimal places

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time (Days)', fontsize=font - 2)
            ax.set_ylabel('T1 (us)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        # ---------------------Add a figure-level legend showing what each shaded area corresponds to-------------------------------
        # fig.legend(
        #     handles=interval_patches,
        #     loc='lower right',  # Choose where you want it placed
        #     title='Shaded Intervals',
        #     fancybox=False)
        # ------------------------------------------------------------------------------------------------

        plt.tight_layout()
        # plt.savefig(analysis_folder + 'T1_vals.png', transparent=False, dpi=self.final_figure_quality)
        # print('Plot saved to:', analysis_folder)
        plt.show()
        # plt.close()


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
