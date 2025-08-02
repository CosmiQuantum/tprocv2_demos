import numpy as np
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_008_save_data_to_h5 import Data_H5
from section_005_single_shot_ge import SingleShot
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from section_005_single_shot_gef import SingleShot_ef
from section_011_qubit_temperatures_efRabipt3 import Temps_EFAmpRabiExperiment
import matplotlib.dates as mdates
from typing import List
from matplotlib.axes import Axes
# from expt_config import *
import glob
import re
import datetime
import ast
import os
import sys
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from bisect import bisect_left
from scipy.stats import norm

sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))


class PlotAllRR:
    def __init__(self, date, figure_quality, save_figs, fit_saved, signal, run_name, number_of_qubits, outerFolder,
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
        self.unique_folder_path = unique_folder_path  # use this when you need to use a different path for anything

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

        if plot_res_spec:
            self.load_plot_save_res_spec()
        if plot_q_spec:
            self.load_plot_save_q_spec()
        if plot_rabis_Qtemps:
            list_of_all_qubits = [i for i in range(self.number_of_qubits + 1)]
            self.load_plot_save_rabis_Qtemps(list_of_all_qubits)
        if plot_rabi:
            if rabi_rolling_avg:
                self.load_plot_save_rabi(rabi_rolling_avg=True)
            else:
                self.load_plot_save_rabi()
        if plot_ss:
            self.load_plot_save_ss(plot_ss_hist_only=plot_ss_hist_only, plot_title=ss_plot_title)
        if ss_plot_gef:
            self.load_plot_save_ss_gef(plot_ssf_gef=ss_plot_gef)
        if plot_t1:
            self.load_plot_save_t1()
        if plot_t2r:
            self.load_plot_save_t2r()
        if plot_t2e:
            self.load_plot_save_t2e()

    def load_plot_save_res_spec(self):
        # ------------------------------------------Load/Plot/Save Res Spec------------------------------------
        outerFolder_expt = os.path.join(self.outerFolder, "Data_h5")
        h5_files = glob.glob(os.path.join(outerFolder_expt, "Res_ge", "*.h5"))
        h5_files += glob.glob(os.path.join(outerFolder_expt, "Res", "*.h5"))
        print(outerFolder_expt)
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

            for q_key in populated_keys:
                # go through each dataset in the batch and plot
                for dataset in range(len(load_data['Res'][q_key].get('Dates', [])[0])):
                    date = datetime.datetime.fromtimestamp(
                        load_data['Res'][q_key].get('Dates', [])[0][dataset])  # single date per dataset
                    freq_pts = self.process_h5_data(load_data['Res'][q_key].get('freq_pts', [])[0][
                                                        dataset].decode())  # comes in as an array but put into a byte string, need to convert to list

                    freq_center = self.process_h5_data(load_data['Res'][q_key].get('freq_center', [])[0][
                                                           dataset].decode())  # comes in as an array but put into a string, need to convert to list
                    freqs_found = self.string_to_float_list(load_data['Res'][q_key].get('Found Freqs', [])[0][
                                                                dataset].decode())  # comes in as a list of floats in string format, need to convert
                    amps = self.process_string_of_nested_lists(
                        load_data['Res'][q_key].get('Amps', [])[0][dataset].decode())  # list of lists
                    syst_config = load_data['Res'][q_key].get('Syst Config', [])[0][dataset].decode()
                    exp_config = load_data['Res'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                    syst_config = eval(syst_config, safe_globals)
                    exp_config = eval(exp_config, safe_globals)

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
                        res_class_instance = ResonanceSpectroscopy(q_key, self.number_of_qubits,
                                                                   self.outerFolder_save_plots, round_num,
                                                                   self.save_figs)
                        res_spec_cfg = exp_config['res_spec']
                        res_class_instance.plot_results(freq_points, freq_center, amps, res_spec_cfg,
                                                        self.figure_quality)
                        del res_class_instance

            del H5_class_instance

    def load_plot_save_q_spec(self):
        # ----------------------------------------------Load/Plot/Save QSpec------------------------------------
        # outerFolder_expt = self.outerFolder + "/Data_h5/QSpec_ge/"
        outerFolder_expt = self.outerFolder + "/Data_h5/QSpec_ef/"
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
            I_s=[]
            Q_s=[]
            for q_key in populated_keys:
                for dataset in range(len(load_data['QSpec'][q_key].get('Dates', [])[0])):
                    date = datetime.datetime.fromtimestamp(load_data['QSpec'][q_key].get('Dates', [])[0][dataset])
                    I = self.process_h5_data(load_data['QSpec'][q_key].get('I', [])[0][dataset].decode())
                    Q = self.process_h5_data(load_data['QSpec'][q_key].get('Q', [])[0][dataset].decode())
                    I_s.append(I)
                    Q_s.append(Q)
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

    def roll(self, data: np.ndarray) -> np.ndarray:

        kernel = np.ones(5) / 5
        smoothed = np.convolve(data, kernel, mode='valid')

        # Preserve the original array's shape by padding the edges
        pad_size = (len(data) - len(smoothed)) // 2
        return np.concatenate((data[:pad_size], smoothed, data[-pad_size:]))

    def load_plot_save_rabi(self, rabi_rolling_avg=False):
        # ------------------------------------------------Load/Plot/Save Rabi---------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/Rabi_ge/"
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

        for h5_file in h5_files:

            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='Rabi', save_r=int(save_round))

            populated_keys = []
            for q_key in load_data['Rabi']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['Rabi'][q_key].get('Dates', [[]])

                # Check if any entry in 'Dates' is not NaN
                if any(
                        not np.isnan(date)
                        for date in dates_list[0]  # Iterate over the first batch of dates
                ):
                    populated_keys.append(q_key)

            for q_key in populated_keys:
                for dataset in range(len(load_data['Rabi'][q_key].get('Dates', [])[0])):
                    date = datetime.datetime.fromtimestamp(load_data['Rabi'][q_key].get('Dates', [])[0][dataset])
                    I = self.process_h5_data(load_data['Rabi'][q_key].get('I', [])[0][dataset].decode())
                    Q = self.process_h5_data(load_data['Rabi'][q_key].get('Q', [])[0][dataset].decode())
                    gains = self.process_h5_data(load_data['Rabi'][q_key].get('Gains', [])[0][dataset].decode())
                    # fit = load_data['Rabi'][q_key].get('Fit', [])[0][dataset]
                    round_num = load_data['Rabi'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['Rabi'][q_key].get('Batch Num', [])[0][dataset]
                    syst_config = load_data['Rabi'][q_key].get('Syst Config', [])[0][dataset].decode()
                    exp_config = load_data['Rabi'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                    exp_config = eval(exp_config, safe_globals)

                    if len(I) > 0:

                        rabi_class_instance = AmplitudeRabiExperiment(q_key, self.number_of_qubits,
                                                                      self.outerFolder_save_plots, round_num,
                                                                      self.signal, self.save_figs)
                        rabi_cfg = exp_config['power_rabi_ge']
                        I = np.asarray(I)
                        Q = np.asarray(Q)

                        if rabi_rolling_avg:
                            I = self.roll(I)
                            Q = self.roll(Q)

                        gains = np.asarray(gains)
                        rabi_class_instance.plot_results(I, Q, gains, rabi_cfg, self.figure_quality)
                        del rabi_class_instance

            del H5_class_instance

    def load_plot_save_ss(self, plot_ss_hist_only, plot_title):

        # ------------------------------------------------Load/Plot/Save SS---------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/SS_ge/"
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

        for h5_file in h5_files:

            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='SS', save_r=int(save_round))

            populated_keys = []
            for q_key in load_data['SS']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['SS'][q_key].get('Dates', [[]])

                # Check if any entry in 'Dates' is not NaN
                if any(
                        not np.isnan(date)
                        for date in dates_list[0]  # Iterate over the first batch of dates
                ):
                    populated_keys.append(q_key)

            for q_key in populated_keys:
                for dataset in range(len(load_data['SS'][q_key].get('Dates', [])[0])):
                    date = datetime.datetime.fromtimestamp(load_data['SS'][q_key].get('Dates', [])[0][dataset])
                    angle = load_data['SS'][q_key].get('Angle', [])[0][dataset]
                    fidelity = load_data['SS'][q_key].get('Fidelity', [])[0][dataset]
                    I_g = self.process_h5_data(load_data['SS'][q_key].get('I_g', [])[0][dataset].decode())
                    Q_g = self.process_h5_data(load_data['SS'][q_key].get('Q_g', [])[0][dataset].decode())
                    I_e = self.process_h5_data(load_data['SS'][q_key].get('I_e', [])[0][dataset].decode())
                    Q_e = self.process_h5_data(load_data['SS'][q_key].get('Q_e', [])[0][dataset].decode())
                    round_num = load_data['SS'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['SS'][q_key].get('Batch Num', [])[0][dataset]
                    # syst_config = load_data['SS'][q_key].get('Syst Config', [])[0][dataset].decode()
                    # exp_config = load_data['SS'][q_key].get('Exp Config', [])[0][dataset].decode()
                    # safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                    # syst_config = eval(syst_config, safe_globals)
                    # exp_config = eval(exp_config, safe_globals)
                    from expt_config import expt_cfg as exp_config
                    I_g = np.array(I_g)
                    Q_g = np.array(Q_g)
                    I_e = np.array(I_e)
                    Q_e = np.array(Q_e)

                    if len(Q_g) > 0:
                        ss_class_instance = SingleShot(q_key, self.number_of_qubits, self.outerFolder_save_plots,
                                                       round_num, self.save_figs)

                        if type(exp_config) is dict:
                            readout_opt = exp_config['Readout_Optimization']
                            if isinstance(readout_opt, str):
                                ss_cfg = ast.literal_eval(readout_opt)
                            else:
                                ss_cfg = readout_opt
                        else:
                            ss_cfg = ast.literal_eval(exp_config['Readout_Optimization'].decode())
                        if plot_ss_hist_only:
                            ss_class_instance.only_hist_ssf(data=[I_g, Q_g, I_e, Q_e], cfg=ss_cfg, plot=True,
                                                            plot_title=plot_title)
                        else:
                            ss_class_instance.hist_ssf(data=[I_g, Q_g, I_e, Q_e], cfg=ss_cfg, plot=True)
                        del ss_class_instance

            del H5_class_instance

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

    def load_plot_save_t2r(self):
        # -------------------------------------------------------Load/Plot/Save T2R------------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/T2_ge/"
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

        for h5_file in h5_files:
            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='T2', save_r=int(save_round))

            populated_keys = []
            for q_key in load_data['T2']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['T2'][q_key].get('Dates', [[]])

                # Check if any entry in 'Dates' is not NaN
                if any(
                        not np.isnan(date)
                        for date in dates_list[0]  # Iterate over the first batch of dates
                ):
                    populated_keys.append(q_key)

            for q_key in populated_keys:
                for dataset in range(len(load_data['T2'][q_key].get('Dates', [])[0])):
                    # T2 = load_data['T2'][q_key].get('T2', [])[0][dataset]
                    # errors = load_data['T2'][q_key].get('Errors', [])[0][dataset]
                    date = datetime.datetime.fromtimestamp(load_data['T2'][q_key].get('Dates', [])[0][dataset])
                    I = self.process_h5_data(load_data['T2'][q_key].get('I', [])[0][dataset].decode())
                    Q = self.process_h5_data(load_data['T2'][q_key].get('Q', [])[0][dataset].decode())
                    delay_times = self.process_h5_data(
                        load_data['T2'][q_key].get('Delay Times', [])[0][dataset].decode())
                    # fit = load_data['T2'][q_key].get('Fit', [])[0][dataset]
                    round_num = load_data['T2'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['T2'][q_key].get('Batch Num', [])[0][dataset]

                    exp_config = load_data['T2'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}

                    exp_config = eval(exp_config, safe_globals)

                    if len(I) > 0:
                        T2_class_instance = T2RMeasurement(q_key, self.number_of_qubits, self.outerFolder_save_plots,
                                                           round_num, self.signal, self.save_figs, fit_data=True)
                        try:
                            fitted, t2r_est, t2r_err, plot_sig = T2_class_instance.t2_fit(delay_times, I, Q)
                        except Exception as e:
                            print('Fit didnt work due to error: ', e)
                            continue
                        T2_cfg = exp_config['Ramsey_ge']
                        T2_class_instance.plot_results(I, Q, delay_times, date, fitted, t2r_est, t2r_err, plot_sig,
                                                       config=T2_cfg, fig_quality=self.figure_quality)
                        del T2_class_instance

            del H5_class_instance

    def load_plot_save_t2e(self):
        # -------------------------------------------------------Load/Plot/Save T2E------------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/T2E_ge/"
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

        for h5_file in h5_files:
            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='T2E', save_r=int(save_round))
            populated_keys = []
            for q_key in load_data['T2E']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['T2E'][q_key].get('Dates', [[]])

                # Check if any entry in 'Dates' is not NaN
                if any(
                        not np.isnan(date)
                        for date in dates_list[0]  # Iterate over the first batch of dates
                ):
                    populated_keys.append(q_key)

            for q_key in populated_keys:
                for dataset in range(len(load_data['T2E'][q_key].get('Dates', [])[0])):
                    # T2 = load_data['T2E'][q_key].get('T2', [])[0][dataset]
                    # errors = load_data['T2E'][q_key].get('Errors', [])[0][dataset]
                    date = datetime.datetime.fromtimestamp(load_data['T2E'][q_key].get('Dates', [])[0][dataset])
                    I = self.process_h5_data(load_data['T2E'][q_key].get('I', [])[0][dataset].decode())
                    Q = self.process_h5_data(load_data['T2E'][q_key].get('Q', [])[0][dataset].decode())
                    delay_times = self.process_h5_data(
                        load_data['T2E'][q_key].get('Delay Times', [])[0][dataset].decode())
                    # fit = load_data['T2E'][q_key].get('Fit', [])[0][dataset]
                    round_num = load_data['T2E'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['T2E'][q_key].get('Batch Num', [])[0][dataset]

                    exp_config = load_data['T2E'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}

                    exp_config = eval(exp_config, safe_globals)

                    if len(I) > 0:
                        T2E_class_instance = T2EMeasurement(q_key, self.number_of_qubits, self.outerFolder_save_plots,
                                                            round_num, self.signal, self.save_figs, fit_data=True)
                        try:
                            fitted, t2e_est, t2e_err, plot_sig = T2E_class_instance.t2_fit(delay_times, I, Q)
                        except Exception as e:
                            print('Fit didnt work due to error: ', e)
                            continue
                        T2E_cfg = exp_config['SpinEcho_ge']
                        T2E_class_instance.plot_results(I, Q, delay_times, date, fitted, t2e_est, t2e_err, plot_sig,
                                                        config=T2E_cfg, fig_quality=self.figure_quality)
                        del T2E_class_instance

            del H5_class_instance

    def load_plot_save_ss_gef(self, plot_ssf_gef, process_one_file=False, file_to_process=None, qubit_index=None):

        # ------------------------------------------------Load/Plot/Save g-e-f SS---------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/SS_gef/"  # checks folder for a single date

        if process_one_file == False:
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
        if process_one_file == True:
            h5_files = [file_to_process]

        for h5_file in h5_files:

            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='SS_gef', save_r=int(save_round))

            # If a specific qubit is specified, filter the loaded data.
            if qubit_index is not None:
                if qubit_index in load_data['SS_gef']:
                    # Keep only the data for the selected qubit.
                    load_data['SS_gef'] = {qubit_index: load_data['SS_gef'][qubit_index]}
                else:
                    print(f"No data for qubit with index {qubit_index} found in file {h5_file}.")
                    continue  # move to next file

            populated_keys = []
            for q_key in load_data['SS_gef']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['SS_gef'][q_key].get('Dates', [[]])

                # Check if any entry in 'Dates' is not NaN
                if any(
                        not np.isnan(date)
                        for date in dates_list[0]  # Iterate over the first batch of dates
                ):
                    populated_keys.append(q_key)

            for q_key in populated_keys:
                for dataset in range(len(load_data['SS_gef'][q_key].get('Dates', [])[0])):
                    date = datetime.datetime.fromtimestamp(load_data['SS_gef'][q_key].get('Dates', [])[0][dataset])
                    angle = load_data['SS_gef'][q_key].get('Angle_ge', [])[0][dataset]
                    # fidelity = load_data['SS_gef'][q_key].get('Fidelity', [])[0][dataset]
                    I_g = self.process_h5_data(load_data['SS_gef'][q_key].get('I_g', [])[0][dataset].decode())
                    Q_g = self.process_h5_data(load_data['SS_gef'][q_key].get('Q_g', [])[0][dataset].decode())
                    I_e = self.process_h5_data(load_data['SS_gef'][q_key].get('I_e', [])[0][dataset].decode())
                    Q_e = self.process_h5_data(load_data['SS_gef'][q_key].get('Q_e', [])[0][dataset].decode())
                    I_f = self.process_h5_data(load_data['SS_gef'][q_key].get('I_f', [])[0][dataset].decode())
                    Q_f = self.process_h5_data(load_data['SS_gef'][q_key].get('Q_f', [])[0][dataset].decode())
                    round_num = load_data['SS_gef'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['SS_gef'][q_key].get('Batch Num', [])[0][dataset]
                    # syst_config = load_data['SS_gef'][q_key].get('Syst Config', [])[0][dataset].decode()
                    # exp_config = load_data['SS_gef'][q_key].get('Exp Config', [])[0][dataset].decode()
                    # safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                    # syst_config = eval(syst_config, safe_globals)
                    # exp_config = eval(exp_config, safe_globals)
                    from expt_config import expt_cfg as exp_config
                    I_g = np.array(I_g)
                    Q_g = np.array(Q_g)
                    I_e = np.array(I_e)
                    Q_e = np.array(Q_e)
                    I_f = np.array(I_f)
                    Q_f = np.array(Q_f)

                    if len(Q_g) > 0:
                        ss_class_instance = SingleShot_ef(q_key, self.number_of_qubits, self.outerFolder_save_plots,
                                                          round_num, self.save_figs)

                        if type(exp_config) is dict:
                            readout_opt = exp_config['Readout_Optimization']
                            if isinstance(readout_opt, str):
                                ss_cfg = ast.literal_eval(readout_opt)
                            else:
                                ss_cfg = readout_opt
                        else:
                            ss_cfg = ast.literal_eval(exp_config['Readout_Optimization'].decode())
                        if plot_ssf_gef:
                            ig_new, qg_new, ie_new, qe_new, if_new, qf_new, theta_ge, threshold_ge = ss_class_instance.hist_ssf(
                                data=[I_g, Q_g, I_e, Q_e, I_f, Q_f], cfg=ss_cfg, plot=True, fig_quality=200)
                        else:
                            ig_new, qg_new, ie_new, qe_new, if_new, qf_new, theta_ge, threshold_ge = ss_class_instance.hist_ssf(
                                data=[I_g, Q_g, I_e, Q_e, I_f, Q_f], cfg=ss_cfg, plot=False, fig_quality=200)
                        del ss_class_instance
                        del H5_class_instance
            return I_g, Q_g, I_e, Q_e, I_f, Q_f, ig_new, qg_new, ie_new, qe_new, if_new, qf_new, theta_ge, threshold_ge  # new arrays are the rotated data

    def load_plot_save_rabis_Qtemps(self, list_of_all_qubits):
        # ------------------------------------------------Load/Plot/Save Rabi---------------------------------------
        outerFolder_expt_qtemps = self.unique_folder_path + "/Data_h5/q_temperatures/"
        h5_files = glob.glob(os.path.join(outerFolder_expt_qtemps, "*.h5"))
        all_files_Qtemp_results = []  # to store qubit temperature results
        cutoff_timestamp = datetime.datetime(2025, 4, 11, 19,
                                             0).timestamp()  # when I started saving qubit freqs in the same files

        extracted_qspec_results = self.load_plot_save_q_spec()
        qspec_grouped_by_qkey = defaultdict(list)
        # sort each list by qubit
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
                    date = datetime.datetime.fromtimestamp(
                        load_data['q_temperatures'][q_key].get('Dates', [])[0][dataset])
                    round_num = load_data['q_temperatures'][q_key].get('Round Num', [])[0][dataset]
                    # batch_num = load_data['q_temperatures'][q_key].get('Batch Num', [])[0][dataset]
                    # -------------------------------------Grabbing matching qubit frequency for this qubit-------------------------------------
                    if date.timestamp() > cutoff_timestamp:  # files after this date contain the matching g-e qubit frequency already
                        qubit_freq_MHz = load_data['q_temperatures'][q_key].get('Qfreq_ge', [])[0][dataset]
                        # print(f"QSpec Q{q_key}: {qubit_freq_MHz:.3f} MHz")
                    else:  # look through matching qspec file
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
                    # ---------------------------------------------------------------------------------------------

                    I1 = self.process_h5_data(load_data['q_temperatures'][q_key].get('I1', [])[0][dataset].decode())
                    Q1 = self.process_h5_data(load_data['q_temperatures'][q_key].get('Q1', [])[0][dataset].decode())
                    gains1 = self.process_h5_data(
                        load_data['q_temperatures'][q_key].get('Gains1', [])[0][dataset].decode())

                    I2 = self.process_h5_data(load_data['q_temperatures'][q_key].get('I2', [])[0][dataset].decode())
                    Q2 = self.process_h5_data(load_data['q_temperatures'][q_key].get('Q2', [])[0][dataset].decode())
                    gains2 = self.process_h5_data(
                        load_data['q_temperatures'][q_key].get('Gains2', [])[0][dataset].decode())

                    # syst_config = load_data['q_temperatures'][q_key].get('Syst Config', [])[0][dataset].decode()
                    exp_config = load_data['q_temperatures'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                    exp_config = eval(exp_config, safe_globals)
                    rabi_cfg = exp_config['power_rabi_ef']
                    if len(I1) > 0:
                        save_figs = False
                        rabi_class_instance = Temps_EFAmpRabiExperiment(q_key, self.number_of_qubits,
                                                                        list_of_all_qubits,
                                                                        self.outerFolder_save_plots, round_num,
                                                                        self.signal, save_figs)
                        I1 = np.asarray(I1)
                        Q1 = np.asarray(Q1)
                        gains1 = np.asarray(gains1)
                        best_signal_fit1, pi_amp1, A_amplitude1, amp_fit1 = rabi_class_instance.plot_results(I1, Q1,
                                                                                                             gains1,
                                                                                                             rabi_cfg,
                                                                                                             self.figure_quality)
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
                        best_signal_fit2, pi_amp2, A_amplitude2, amp_fit2 = rabi_class_instance.plot_results(I2, Q2,
                                                                                                             gains2,
                                                                                                             rabi_cfg,
                                                                                                             self.figure_quality)
                        del rabi_class_instance

                    if A_amplitude1 is not None and A_amplitude2 is not None:
                        A_e = A_amplitude1
                        A_g = A_amplitude2

                        results = self.Qubit_Temperature_Convert(A_e, A_g, qubit_freq_MHz)
                        if results is None:
                            continue  # Skip this dataset
                        T_K, T_mK, P_e, qubit_freq = results
                        print(
                            f"Q{q_key} calculated Temperature:{T_mK}, with P_e = {P_e}, and Qfreq {qubit_freq_MHz} MHz")
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
        if P_e <= 0 or P_g <= 0:  # if one of them is zero can't calculate the temp
            print("Warning: Invalid population values encountered (<= 0). Skipping this dataset.")
            return None

        ratio = P_g / P_e
        if ratio <= 1:  # denominator would become zero at Pg=Pe
            print(f"Warning: Non-physical ratio (P_g/P_e = {ratio:.3f} <= 1) encountered. Skipping this dataset.")
            return None

        qubit_freq_Hz = qubit_freq_MHz * 2 * np.pi * 1e6  # Omega_q in the unit Hz
        k_B = 1.38 * 10 ** -23
        hbar = 1.05 * 10 ** -34
        T_K = hbar * qubit_freq_Hz / (k_B * np.log(P_g / P_e))  # Temperature in the unit Kelvin
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
                                 sharex=False,
                                 constrained_layout=True)  # set sharex=False if you want each subplot to manage ticks independently
        axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

        fig.suptitle("Qubit Temperatures vs. Time", fontsize=16)

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
            # Use scatter instead of plot to avoid connecting lines
            ax.scatter(times, temps, marker='o', color=colors[q % len(colors)], label=f"Q{q + 1}")

            ax.set_title(f"Q{q + 1}", fontsize=14)
            ax.set_ylabel("Temp (mK)", fontsize=12)
            ax.grid(False)

            # Format the x-axis to show dates in a nice format
            ax.set_ylim(100, 400)
            # ax.set_yticks(np.linspace(25, 300, 12))

            start_time = datetime.datetime(2025, 4, 11, 12, 30)
            ax.set_xlim(left=start_time)

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.tick_params(axis='x', labelrotation=45, labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

        # Add a shared X label if desired (comment out if not needed)
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
