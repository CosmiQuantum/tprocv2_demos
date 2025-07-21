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
import ast
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

class PiAmpsVsTime:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name):
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.figure_quality = figure_quality
        self.run_name = run_name
        self.number_of_qubits = number_of_qubits
        self.final_figure_quality = final_figure_quality
        self.top_folder_dates = top_folder_dates

    def datetime_to_unix(self, dt):
        # Convert to Unix timestamp
        unix_timestamp = int(dt.timestamp())
        return unix_timestamp

    def unix_to_datetime(self, unix_timestamp):
        # Convert the Unix timestamp to a datetime object
        dt = datetime.fromtimestamp(self, unix_timestamp)
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

    def run(self, plot_depths = False, rolling_avg = False, exp_extension=''):

        import datetime
        # ----------Load/get data------------------------
        pi_amps = {i: [] for i in range(self.number_of_qubits)}
        depths = {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        for folder_date in self.top_folder_dates:
            outerFolder = f"/data/QICK_data/{self.run_name}/" + folder_date + "/"
            outerFolder_save_plots = f"/data/QICK_data/{self.run_name}/" + folder_date + "_plots/"

            # ------------------------------------------------Load/Plot/Save Rabi---------------------------------------
            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/Rabi{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/Rabi_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=f'Rabi{exp_extension}', save_r=int(save_round))

                for q_key in load_data[f'Rabi{exp_extension}']:
                    for dataset in range(len(load_data[f'Rabi{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'Rabi{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        date = datetime.datetime.fromtimestamp(load_data[f'Rabi{exp_extension}'][q_key].get('Dates', [])[0][dataset])
                        I = self.process_h5_data(load_data[f'Rabi{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data[f'Rabi{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                        gains = self.process_h5_data(load_data[f'Rabi{exp_extension}'][q_key].get('Gains', [])[0][dataset].decode())
                        # fit = load_data['Rabi'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data[f'Rabi{exp_extension}'][q_key].get('Round Num', [])[0][dataset]
                        batch_num = load_data[f'Rabi{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]
                        try:
                            syst_config = load_data[f'Rabi{exp_extension}'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'Rabi{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I) > 0:
                            rabi_class_instance = AmplitudeRabiExperiment(q_key, self.number_of_qubits, outerFolder_save_plots, round_num,
                                                                          self.signal, self.save_figs)
                            #rabi_cfg = exp_config['power_rabi_ge']
                            I = np.asarray(I)
                            Q = np.asarray(Q)
                            gains = np.asarray(gains)
                            if plot_depths:
                                best_signal_fit, pi_amp, depth = rabi_class_instance.get_results(I, Q, gains,
                                                                                                 grab_depths=True)
                                depths[q_key].extend([depth])
                            elif rolling_avg:
                                best_signal_fit, pi_amp = rabi_class_instance.get_results(I, Q, gains, rolling_avg=True)
                            else:
                                best_signal_fit, pi_amp = rabi_class_instance.get_results(I, Q, gains)

                            pi_amps[q_key].extend([pi_amp])

                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del rabi_class_instance
                del H5_class_instance
        if plot_depths:
            return date_times, pi_amps, depths
        else:
            return date_times, pi_amps

    def runQZE(self, outerFolder, outerFolder_save_plots, fit=False, expt_name = "len_rabi_ge",
               old_format=False, filter_amp_above=None,mark_w01s=False, plot_detuned_amps=False,plot_detuning=False,
               pi_line_label_left=None,pi_line_label_right=None):
        # ------------------------------------------------Load/Plot/Save Rabi---------------------------------------
        outerFolder_expt = os.path.join(outerFolder, "Data_h5", "Rabi_QZE")

        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
        all_I = []
        all_Q = []
        all_mag = []
        all_gains = []
        proj_pulse_gains = []
        starked_frequency = []
        baseline_frequency = []
        fwhm_w01 = []
        fwhm_w01_starked = []
        for h5_file in h5_files:
            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='Rabi', save_r=int(save_round))

            populated_keys = []
            for q_key in load_data['Rabi']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['Rabi'][q_key].get('Dates', [[]])
                # Check if any entry in 'Dates' is not NaN
                if any(not np.isnan(date) for date in dates_list[0]):
                    populated_keys.append(q_key)

            # Loop over each populated q_key
            #for q_key in range(0): #only looking at first qubit for now
                # Create lists to hold all datasets for this q_key

            # Loop over each dataset for the current q_key
            num_datasets = len(load_data['Rabi'][0].get('Dates', [])[0])
            for dataset in range(num_datasets):
                date = datetime.datetime.fromtimestamp(
                    load_data['Rabi'][0].get('Dates', [])[0][dataset])
                I_data = self.process_h5_data(
                    load_data['Rabi'][0].get('I', [])[0][dataset].decode())
                Q_data = self.process_h5_data(
                    load_data['Rabi'][0].get('Q', [])[0][dataset].decode())
                # mag = self.process_h5_data(
                #     load_data['Rabi'][0].get('Mag', [])[0][dataset].decode())
                gains_data = self.process_h5_data(
                    load_data['Rabi'][0].get('Gains', [])[0][dataset].decode())
                round_num = load_data['Rabi'][0].get('Round Num', [])[0][dataset]
                batch_num = load_data['Rabi'][0].get('Batch Num', [])[0][dataset]
                syst_config = load_data['Rabi'][0].get('Syst Config', [])[0][dataset].decode()
                exp_config = load_data['Rabi'][0].get('Exp Config', [])[0][dataset].decode()
                safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                exp_config = eval(exp_config, safe_globals)
                all_I.append(I_data)
                all_Q.append(Q_data)
                #all_mag.append(mag)
                all_gains.append(gains_data)
                proj_pulse_gains.append(round(float(syst_config.split('res_gain_qze\': [')[-1].split(']')[0].split('0,')[-1]),4))
                starked_frequency.append(round(float(syst_config.split('qubit_freq_ge_starked\': [')[-1].split(']')[0].split(',')[0]),4))
                fwhm_w01_starked.append(round(float(syst_config.split('fwhm_w01_starked\':')[-1].split(',')[0]), 4))
                fwhm_w01.append(round(float(syst_config.split('fwhm_w01\':')[-1].split(',')[0]), 4))
                baseline_frequency.append(round(float(syst_config.split('qubit_freq_ge\': ')[-1].split(',')[0]), 4))

            del H5_class_instance

        frequency_difference=[w01-w01st for w01, w01st in zip(baseline_frequency,starked_frequency)]
        # If we have any valid data, plot all the measurements together
        if all_I:
            rabi_class_instance = AmplitudeRabiExperiment(
                0, self.number_of_qubits,
                outerFolder_save_plots, 0,
                self.signal, self.save_figs, expt_name =expt_name )
            # Now call the updated plot_QZE method with lists of datasets
            if fit:
                rabi_class_instance.plot_QZE_fit(all_I, all_Q, all_gains, proj_pulse_gains, self.figure_quality)
            else:
                if old_format:
                    rabi_class_instance.plot_QZE_old_format(all_I, all_Q, all_gains, proj_pulse_gains, self.figure_quality)
                else:
                    rabi_class_instance.plot_QZE(all_I, all_Q, all_gains, proj_pulse_gains, self.figure_quality,
                                                 filter_amp_above=filter_amp_above, mark_w01s=mark_w01s,pi_line_label_left=pi_line_label_left,pi_line_label_right=pi_line_label_right)
                    rabi_class_instance.plot_QZE_basic(all_I, all_Q, all_gains, proj_pulse_gains, self.figure_quality,
                                                 filter_amp_above=filter_amp_above, mark_w01s=mark_w01s,
                                                 pi_line_label_left=pi_line_label_left,
                                                 pi_line_label_right=pi_line_label_right)

                    if plot_detuned_amps:
                        try:
                            rabi_class_instance.plot_QZE_detuned_amp(all_I, all_Q, all_gains, proj_pulse_gains, self.figure_quality,
                                                        filter_amp_above=filter_amp_above, mark_w01s=mark_w01s,
                                                                     frequency_difference=frequency_difference, fwhm_w01=fwhm_w01,
                                                                     fwhm_w01_starked=fwhm_w01_starked, pi_line_label_left=pi_line_label_left,pi_line_label_right=pi_line_label_right)
                        except:
                            return
                    if plot_detuning:
                        try:
                            rabi_class_instance.plot_QZE_detuning(all_I, all_Q, all_gains, proj_pulse_gains,
                                                                     self.figure_quality,
                                                                     filter_amp_above=filter_amp_above, mark_w01s=mark_w01s,
                                                                     frequency_difference=frequency_difference,
                                                                     fwhm_w01=fwhm_w01,
                                                                     fwhm_w01_starked=fwhm_w01_starked)
                        except:
                            return
            del rabi_class_instance

    def process_and_fit_QZE(self, I, Q, gains, proj_pulse_gains, fig_quality=100):
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import datetime
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        # Sort the data based on proj_pulse_gains from least to greatest
        proj_gains_array = np.array(proj_pulse_gains)
        sorted_indices = np.argsort(proj_gains_array)
        sorted_I = [I[i] for i in sorted_indices]
        sorted_Q = [Q[i] for i in sorted_indices]
        sorted_gains = [gains[i] for i in sorted_indices]
        sorted_proj_pulse_gains = proj_gains_array[sorted_indices]

        # Set up the figure and colormap (lower gains = warm, higher gains = cool)
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.rcParams.update({'font.size': 18})
        cmap = plt.get_cmap('coolwarm_r')
        min_gain = np.min(sorted_proj_pulse_gains)
        max_gain = np.max(sorted_proj_pulse_gains)
        if max_gain - min_gain == 0:
            norm_gains = np.zeros_like(sorted_proj_pulse_gains)
        else:
            norm_gains = (sorted_proj_pulse_gains - min_gain) / (max_gain - min_gain)

        # Process each dataset corresponding to a different projection pulse gain
        for idx in range(len(sorted_I)):
            # x-axis: drive pulse widths and y-axis: measured magnitude from I and Q
            x = np.array(sorted_gains[idx])
            magnitudes = np.abs(np.array(sorted_I[idx]) + 1j * np.array(sorted_Q[idx]))

            # Filter the data: only keep points where 0.11 < x < 0.4
            mask = (x > 0.11) & (x < 0.4)
            if np.sum(mask) < 4:  # skip if there are not enough points to fit a curve
                continue
            x_filtered = x[mask]
            y_filtered = magnitudes[mask]

            # Fit a third-degree polynomial to the filtered data.
            poly_coeffs = np.polyfit(x_filtered, y_filtered, deg=3)
            fit_poly = np.poly1d(poly_coeffs)
            # Create a smooth set of x values over the filtered range for the fit curve
            x_fit = np.linspace(np.min(x_filtered), np.max(x_filtered), 200)
            y_fit = fit_poly(x_fit)

            # Determine the color based on the normalized projection gain for this dataset
            color = cmap(norm_gains[idx])
            # Plot the filtered data as scatter points
            ax.scatter(x_filtered, y_filtered, color=color, s=50, label=f'Proj Gain {sorted_proj_pulse_gains[idx]:.2f}')
            # Plot the fitted curve as a line
            ax.plot(x_fit, y_fit, color=color, linewidth=2)

        # Labeling and styling the plot
        ax.set_xlabel("Qubit drive pulse width (us)", fontsize=14)
        ax.set_ylabel("Magnitude (a.u.)", fontsize=14)
        ax.set_title("Processed QZE Data with Curve Fitting", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend()

        plt.tight_layout()

        # Save the figure in both PNG and PDF formats
        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt,
                                 f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}_processed")
        fig.savefig(file_name + '.png', dpi=fig_quality, bbox_inches='tight')
        fig.savefig(file_name + '.pdf', dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)
        return

    def runQZE_systematics_subtraction(self, outerFolder, outerFolder_systematics, outerFolder_save_plots, fit=False, expt_name = "len_rabi_ge"):
        # ------------------------------------------------Load/Plot/Save Rabi---------------------------------------
        outerFolder_expt = os.path.join(outerFolder, "Data_h5", "Rabi_QZE")
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
        all_I = []
        all_Q = []
        all_mag = []
        all_gains = []
        proj_pulse_gains = []
        for h5_file in h5_files:
            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='Rabi', save_r=int(save_round))

            populated_keys = []
            for q_key in load_data['Rabi']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['Rabi'][q_key].get('Dates', [[]])
                # Check if any entry in 'Dates' is not NaN
                if any(not np.isnan(date) for date in dates_list[0]):
                    populated_keys.append(q_key)

            # Loop over each populated q_key
            #for q_key in range(0): #only looking at first qubit for now
                # Create lists to hold all datasets for this q_key

            # Loop over each dataset for the current q_key
            num_datasets = len(load_data['Rabi'][0].get('Dates', [])[0])
            for dataset in range(num_datasets):
                date = datetime.datetime.fromtimestamp(
                    load_data['Rabi'][0].get('Dates', [])[0][dataset])
                I_data = self.process_h5_data(
                    load_data['Rabi'][0].get('I', [])[0][dataset].decode())
                Q_data = self.process_h5_data(
                    load_data['Rabi'][0].get('Q', [])[0][dataset].decode())
                # mag = self.process_h5_data(
                #     load_data['Rabi'][0].get('Mag', [])[0][dataset].decode())
                gains_data = self.process_h5_data(
                    load_data['Rabi'][0].get('Gains', [])[0][dataset].decode())
                round_num = load_data['Rabi'][0].get('Round Num', [])[0][dataset]
                batch_num = load_data['Rabi'][0].get('Batch Num', [])[0][dataset]
                syst_config = load_data['Rabi'][0].get('Syst Config', [])[0][dataset].decode()
                exp_config = load_data['Rabi'][0].get('Exp Config', [])[0][dataset].decode()
                safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                exp_config = eval(exp_config, safe_globals)
                all_I.append(I_data)
                all_Q.append(Q_data)
                #all_mag.append(mag)
                all_gains.append(gains_data)
                proj_pulse_gains.append(round(float(syst_config.split('res_gain_qze\': [')[-1].split(']')[0].split('0,')[-1]),4))


            del H5_class_instance

        outerFolder_expt_systematics = os.path.join(outerFolder_systematics, "Data_h5", "Rabi_QZE")
        h5_files_systematics = glob.glob(os.path.join(outerFolder_expt_systematics, "*.h5"))
        all_I_systematics = []
        all_Q_systematics = []
        all_mag_systematics = []
        all_gains_systematics = []
        proj_pulse_gains_systematics = []
        for h5_file in h5_files_systematics:
            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='Rabi', save_r=int(save_round))

            populated_keys = []
            for q_key in load_data['Rabi']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['Rabi'][q_key].get('Dates', [[]])
                # Check if any entry in 'Dates' is not NaN
                if any(not np.isnan(date) for date in dates_list[0]):
                    populated_keys.append(q_key)

            # Loop over each populated q_key
            # for q_key in range(0): #only looking at first qubit for now
            # Create lists to hold all datasets for this q_key

            # Loop over each dataset for the current q_key
            num_datasets = len(load_data['Rabi'][0].get('Dates', [])[0])
            for dataset in range(num_datasets):
                date = datetime.datetime.fromtimestamp(
                    load_data['Rabi'][0].get('Dates', [])[0][dataset])
                I_data = self.process_h5_data(
                    load_data['Rabi'][0].get('I', [])[0][dataset].decode())
                Q_data = self.process_h5_data(
                    load_data['Rabi'][0].get('Q', [])[0][dataset].decode())
                # mag = self.process_h5_data(
                #     load_data['Rabi'][0].get('Mag', [])[0][dataset].decode())
                gains_data = self.process_h5_data(
                    load_data['Rabi'][0].get('Gains', [])[0][dataset].decode())
                round_num = load_data['Rabi'][0].get('Round Num', [])[0][dataset]
                batch_num = load_data['Rabi'][0].get('Batch Num', [])[0][dataset]
                syst_config = load_data['Rabi'][0].get('Syst Config', [])[0][dataset].decode()
                exp_config = load_data['Rabi'][0].get('Exp Config', [])[0][dataset].decode()
                safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                exp_config = eval(exp_config, safe_globals)
                all_I_systematics.append(I_data)
                all_Q_systematics.append(Q_data)
                # all_mag.append(mag)
                all_gains_systematics.append(gains_data)
                proj_pulse_gains_systematics.append(
                    round(float(syst_config.split('res_gain_qze\': [')[-1].split(']')[0].split('0,')[-1]), 4))

            del H5_class_instance

        # If we have any valid data, plot all the measurements together
        if all_I:
            rabi_class_instance = AmplitudeRabiExperiment(
                0, self.number_of_qubits,
                outerFolder_save_plots, 0,
                self.signal, self.save_figs, expt_name =expt_name )
            # Now call the updated plot_QZE method with lists of datasets
            if fit:

                rabi_class_instance.plot_QZE_w_systematic_subtraction_fit(all_I, all_Q, all_gains, proj_pulse_gains,
                                                                          all_I_systematics, all_Q_systematics,
                                                                          all_gains_systematics, proj_pulse_gains_systematics,
                                                                          self.figure_quality)
            else:
                rabi_class_instance.plot_QZE_w_systematic_subtraction(all_I, all_Q, all_gains, proj_pulse_gains,
                                                                      all_I_systematics, all_Q_systematics,
                                                                      all_gains_systematics, proj_pulse_gains_systematics,
                                                                      self.figure_quality)
            del rabi_class_instance


    def plot(self, date_times, pi_amps, show_legends,exp_extension=''):
        #---------------------------------plot-----------------------------------------------------
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i+1}" for i in range(self.number_of_qubits)]
        colors = ['orange','blue','purple','green','brown','pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        ext = exp_extension.split('_')[0]
        plt.title(f'Pi Amplitudes vs Time {ext}',fontsize = font)
        axes = axes.flatten()
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        from datetime import datetime

        for i, ax in enumerate(axes):

            ax.set_title(titles[i], fontsize = font)

            x = date_times[i]
            y = pi_amps[i]

            # Convert strings to datetime objects.
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            # Combine datetime objects and y values into a list of tuples and sort by datetime.
            combined = list(zip(datetime_objects, y))
            combined.sort(reverse=True, key=lambda x: x[0])
            if combined:

                # Unpack them back into separate lists, in order from latest to most recent.
                sorted_x, sorted_y = zip(*combined)
                ax.scatter(sorted_x, sorted_y, color=colors[i])

                sorted_x = np.asarray(sorted(x))

                num_points = 5
                indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

                # Set new x-ticks using the datetime objects at the selected indices
                ax.set_xticks(sorted_x[indices])
                ax.set_xticklabels([dt for dt in sorted_x[indices]], rotation=45)

                ax.scatter(x, y, color=colors[i])
                if show_legends:
                    ax.legend(edgecolor='black')
                ax.set_xlabel('Time (Days)', fontsize=font-2)
                ax.set_ylabel('Pi Amp (a.u.)', fontsize=font-2)
                ax.tick_params(axis='both', which='major', labelsize=8)
            else:
                continue

        plt.tight_layout()
        plt.savefig(analysis_folder + f'Pi_Amps{exp_extension}.pdf', transparent=True, dpi=self.final_figure_quality)

        #plt.show()

    def plot_hist(self, date_times, pi_amps, show_legends):
        # Create analysis folders if they do not exist.
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

        # Create a 2x3 grid of subplots.
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.suptitle('Histogram of Pi Amplitudes with Gaussian Fit', fontsize=font)
        axes = axes.flatten()

        optimal_bin_num = 50
        for i, ax in enumerate(axes):
            # Check that there is data for this qubit
            if i not in pi_amps or len(pi_amps[i]) == 0:
                ax.set_visible(False)
                continue

            # If there is enough data, fit a Gaussian
            if len(pi_amps[i]) > 1:
                # Fit a Gaussian to the pi_amp data
                mu, sigma = norm.fit(pi_amps[i])
                # Create x values for plotting the Gaussian curve
                x = np.linspace(min(pi_amps[i]), max(pi_amps[i]), optimal_bin_num)
                p = norm.pdf(x, mu, sigma)
                # Get histogram data for proper scaling of the Gaussian curve
                hist_data, bins = np.histogram(pi_amps[i], bins=optimal_bin_num)
                scale = np.diff(bins) * hist_data.sum()
                # Plot the scaled Gaussian curve
                ax.plot(x, p * scale, 'b--', linewidth=2, color=colors[i])
                # Plot the histogram of pi_amp data
                ax.hist(pi_amps[i], bins=optimal_bin_num, alpha=0.7, color=colors[i], edgecolor='black')
                # Add the Gaussian parameters in the title
                ax.set_title(titles[i] + f" $\mu$: {mu:.2f} $\sigma$: {sigma:.2f}", fontsize=font)
            else:
                # If not enough data for a fit, simply plot the histogram
                ax.hist(pi_amps[i], bins=optimal_bin_num, alpha=0.7, color=colors[i], edgecolor='black')
                ax.set_title(titles[i], fontsize=font)

            if show_legends:
                ax.legend([titles[i]], edgecolor='black')
            ax.set_xlabel('Pi Amp (a.u.)', fontsize=font - 2)
            ax.set_ylabel('Count', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Pi_Amps_Hist_Gaussian.pdf', transparent=True, dpi=self.final_figure_quality)
        # plt.show()

    def plot_scatter_rolling_avg(self, pi_amps, pi_amps_from_rolling_avg, show_legends):
        # Create analysis folders if they do not exist.
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

        # Create a 2x3 grid of subplots.
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.suptitle('Scatter Plot: Pi Amp vs. Difference (Rolling Avg - Pi Amp)', fontsize=font)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            ax.set_title(titles[i], fontsize=font)
            # Convert lists to numpy arrays for element-wise operations.
            x = np.array(pi_amps[i])
            y = np.array(pi_amps_from_rolling_avg[i]) - x

            # Create a scatter plot.
            ax.scatter(x, y, color=colors[i], edgecolor='black')

            if show_legends:
                ax.legend([titles[i]], edgecolor='black')

            ax.set_xlabel('Pi Amp (a.u.)', fontsize=font - 2)
            ax.set_ylabel('Rolling Avg - Pi Amp (a.u.)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Pi_Amps_Scatter.pdf', transparent=True, dpi=self.final_figure_quality)
        # plt.show()

    def plot_histogram_heatmap(self, pi_amps, pi_amps_from_rolling_avg, show_legends):
        # Create analysis folders if they do not exist.
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]

        # Create a 2x3 grid of subplots.
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.suptitle('2D Histogram Heatmap: Pi Amp vs. Difference (Rolling Avg - Pi Amp)', fontsize=font)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            ax.set_title(titles[i], fontsize=font)
            # Convert lists to numpy arrays for element-wise operations.
            x = np.array(pi_amps[i])
            y = np.array(pi_amps_from_rolling_avg[i]) - x

            # Compute full range for both x and y.
            if len(x) > 1:
                min_val = min(np.min(x), np.min(y))
                max_val = max(np.max(x), np.max(y))
                full_range = [[min_val, max_val], [min_val, max_val]]
            else:
                full_range = None  # Use default if there is not enough data.

            # Create a 2D histogram over the full range.
            h = ax.hist2d(x, y, bins=49, range=full_range, cmap='viridis')

            # Add a colorbar to show the count scale.
            plt.colorbar(h[3], ax=ax)

            if show_legends:
                ax.text(0.95, 0.95, titles[i],
                        transform=ax.transAxes, verticalalignment='top',
                        horizontalalignment='right', color='white', fontsize=12,
                        bbox={'facecolor': 'black', 'alpha': 0.5})

            ax.set_xlabel('Pi Amp (a.u.)', fontsize=font - 2)
            ax.set_ylabel('Rolling Avg - Pi Amp (a.u.)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

            # Set the x and y limits to match the full range.
            if full_range is not None:
                ax.set_xlim(full_range[0])
                ax.set_ylim(full_range[1])
                ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Pi_Amps_Histogram_Heatmap.pdf', transparent=True, dpi=self.final_figure_quality)
        # plt.show()

    def plot_vs_signal_depth(self, date_times, pi_amps, depths, show_legends):
        #---------------------------------plot-----------------------------------------------------
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/other/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i+1}" for i in range(self.number_of_qubits)]
        colors = ['orange','blue','purple','green','brown','pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('Pi Amplitudes vs Signal Depths',fontsize = font)
        axes = axes.flatten()
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        from datetime import datetime

        for i, ax in enumerate(axes):

            ax.set_title(titles[i], fontsize = font)

            x = depths[i]
            y = pi_amps[i]

            ax.scatter(x, y, color=colors[i])
            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Signal Depth', fontsize=font-2)
            ax.set_ylabel('Pi Amp (a.u.)', fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Pi_Amps_vs_depth.pdf', transparent=True, dpi=self.final_figure_quality)

        #plt.show()

    def plot_signal_depth_vs_time(self, date_times, pi_amps, depths, show_legends):
        #---------------------------------plot-----------------------------------------------------
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/other/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i+1}" for i in range(self.number_of_qubits)]
        colors = ['orange','blue','purple','green','brown','pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('Signal Depths vs Time',fontsize = font)
        axes = axes.flatten()
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        from datetime import datetime

        for i, ax in enumerate(axes):

            ax.set_title(titles[i], fontsize = font)

            x = date_times[i]
            y = depths[i]

            # Convert strings to datetime objects.
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            # Combine datetime objects and y values into a list of tuples and sort by datetime.
            combined = list(zip(datetime_objects, y))
            combined.sort(reverse=True, key=lambda x: x[0])

            # Unpack them back into separate lists, in order from latest to most recent.
            sorted_x, sorted_y = zip(*combined)
            ax.scatter(sorted_x, sorted_y, color=colors[i])

            sorted_x = np.asarray(sorted(x))

            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

            # Set new x-ticks using the datetime objects at the selected indices
            ax.set_xticks(sorted_x[indices])
            ax.set_xticklabels([dt for dt in sorted_x[indices]], rotation=45)

            ax.scatter(x, y, color=colors[i])
            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time (Days)', fontsize=font-2)
            ax.set_ylabel('Signal Depth (a.u.)', fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'depths_vs_time.pdf', transparent=True, dpi=self.final_figure_quality)

        #plt.show()

    def plot_vs_temps(self, date_times, pi_amps, temps, show_legends):
        #---------------------------------plot-----------------------------------------------------
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/other/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i+1}" for i in range(self.number_of_qubits)]
        colors = ['orange','blue','purple','green','brown','pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('Pi Amplitudes vs Qubit temps',fontsize = font)
        axes = axes.flatten()
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        from datetime import datetime

        for i, ax in enumerate(axes):

            ax.set_title(titles[i], fontsize = font)
            x = temps[i]
            y = pi_amps[i]

            ax.scatter(x, y, color=colors[i])
            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Qubit temp (mK)', fontsize=font-2)
            ax.set_ylabel('Pi Amp (a.u)', fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Pi_Amps_vs_qtemp.pdf', transparent=True, dpi=self.final_figure_quality)

        #plt.show()

    def plot_vs_ssf(self, date_times, pi_amps, ssf, show_legends):
        #---------------------------------plot-----------------------------------------------------
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/other/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i+1}" for i in range(self.number_of_qubits)]
        colors = ['orange','blue','purple','green','brown','pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('Pi Amplitudes vs SSF',fontsize = font)
        axes = axes.flatten()
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        from datetime import datetime

        for i, ax in enumerate(axes):

            ax.set_title(titles[i], fontsize = font)
            x = ssf[i]
            y = pi_amps[i]

            ax.scatter(x, y, color=colors[i])
            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('SSF', fontsize=font-2)
            ax.set_ylabel('Pi Amp (a.u)', fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Pi_Amps_vs_ssf.pdf', transparent=True, dpi=self.final_figure_quality)

        #plt.show()

    def qtemp_vs_time(self, date_times, temps, show_legends):
        #---------------------------------plot-----------------------------------------------------
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        from datetime import datetime
        # Convert strings to datetime objects.
        date_times = {
            i: [
                datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S") if isinstance(date_str, str) else date_str
                for date_str in dates
            ]
            for i, dates in date_times.items()
        }

        font = 14
        titles = [f"Qubit {i+1}" for i in range(self.number_of_qubits)]
        colors = ['orange','blue','purple','green','brown','pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('Qubit Temp vs Time',fontsize = font)
        axes = axes.flatten()
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        from datetime import datetime

        for i, ax in enumerate(axes):

            ax.set_title(titles[i], fontsize = font)

            x = date_times[i]
            y = temps[i]



            # Combine datetime objects and y values into a list of tuples and sort by datetime.
            combined = list(zip(x, y))
            combined.sort(reverse=True, key=lambda x: x[0])

            # Unpack them back into separate lists, in order from latest to most recent.
            sorted_x, sorted_y = zip(*combined)
            ax.scatter(sorted_x, sorted_y, color=colors[i])

            sorted_x = np.asarray(sorted(x))

            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

            # Set new x-ticks using the datetime objects at the selected indices
            ax.set_xticks(sorted_x[indices])
            ax.set_xticklabels([dt for dt in sorted_x[indices]], rotation=45)

            ax.scatter(x, y, color=colors[i])
            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time (Days)', fontsize=font-2)
            ax.set_ylabel('Qubit Temp (mK)', fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'qubit_temp.pdf', transparent=True, dpi=self.final_figure_quality)

        #plt.show()