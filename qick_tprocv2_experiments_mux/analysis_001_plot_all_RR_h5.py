from tprocv2_demos.qick_tprocv2_experiments_mux.socProxy import makeProxy
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
#from expt_config import *
import glob
from collections import OrderedDict
from tqdm import tqdm
import re
import datetime
from scipy.signal import find_peaks
import ast
import os
import sys
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from bisect import bisect_left
from scipy.stats import norm
import h5py

sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))

class PlotAllRR:
    def __init__(self,  date, figure_quality, save_figs, fit_saved, signal, run_name, run_num, number_of_qubits, outerFolder,
                 outerFolder_save_plots, unique_folder_path, saved_shots):
        self.date = date
        self.figure_quality = figure_quality
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.run_name = run_name
        self.run_num = run_num
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.outerFolder_save_plots = outerFolder_save_plots
        self.unique_folder_path = unique_folder_path # use this when you need to use a different path for anything
        self.saved_shots = saved_shots

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
    
    def run(self, plot_res_spec = True, plot_q_spec = True, plot_rabi = True, rabi_rolling_avg=False, plot_ss = True,
            plot_ss_hist_only=False,ss_plot_title = None, ss_plot_gef = True, plot_t1 = True,
            plot_t2r = True, plot_t2e = True, plot_rabis_Qtemps = False, plot_t1_shots_analysis = False):

        if plot_res_spec:
            self.load_plot_save_res_spec()
        if plot_q_spec:
            self.load_plot_save_q_spec()
        if plot_rabis_Qtemps:
            list_of_all_qubits= [i for i in range(self.number_of_qubits + 1)]
            self.load_plot_save_rabis_Qtemps(list_of_all_qubits)
        if plot_rabi:
            if rabi_rolling_avg:
                self.load_plot_save_rabi(rabi_rolling_avg=True)
            else:
                self.load_plot_save_rabi()
        if plot_ss:
            if self.run_num == 4:
                self.run4_load_plot_save_ss(plot_ss_hist_only = plot_ss_hist_only, plot_title = ss_plot_title)
            else:
                self.load_plot_save_ss(plot_ss_hist_only = plot_ss_hist_only, plot_title = ss_plot_title)
        if ss_plot_gef:
            self.load_plot_save_ss_gef(plot_ssf_gef = ss_plot_gef)
        if plot_t1:
            self.load_plot_save_t1(saved_shots = self.saved_shots)
        if plot_t1_shots_analysis:
            self.load_t1_shots_vs_avgIQ_arrays()
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
            #H5_class_instance.print_h5_contents(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type=  'Res', save_r = int(save_round))
        
            #just look at this resonator data, should have batch_num of arrays in each one
            #right now the data writes the same thing batch_num of times, so it will do the same 5 datasets 5 times, until you fix this just grab the first one (All 5)
        
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
                #go through each dataset in the batch and plot
                for dataset in range(len(load_data['Res'][q_key].get('Dates', [])[0])):
                    date = datetime.datetime.fromtimestamp(load_data['Res'][q_key].get('Dates', [])[0][dataset])   #single date per dataset
                    freq_pts = self.process_h5_data(load_data['Res'][q_key].get('freq_pts', [])[0][dataset].decode())   # comes in as an array but put into a byte string, need to convert to list

                    freq_center = self.process_h5_data(load_data['Res'][q_key].get('freq_center', [])[0][dataset].decode()) # comes in as an array but put into a string, need to convert to list
                    freqs_found = self.string_to_float_list(load_data['Res'][q_key].get('Found Freqs', [])[0][dataset].decode()) #comes in as a list of floats in string format, need to convert
                    amps =  self.process_string_of_nested_lists(load_data['Res'][q_key].get('Amps', [])[0][dataset].decode())  #list of lists
                    syst_config = load_data['Res'][q_key].get('Syst Config', [])[0][dataset].decode()
                    exp_config = load_data['Res'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                    syst_config = eval(syst_config, safe_globals)
                    exp_config = eval(exp_config, safe_globals)

                    round_num = load_data['Res'][q_key].get('Round Num', [])[0][dataset] #already a float
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
                    #print('here: ', freq_points)
                    if len(freq_pts) > 0:
                        res_class_instance = ResonanceSpectroscopy(q_key, self.number_of_qubits, self.outerFolder_save_plots, round_num, self.save_figs)
                        res_spec_cfg = exp_config['res_spec']
                        res_class_instance.plot_results(freq_points, freq_center, amps, res_spec_cfg, self.figure_quality)
                        del res_class_instance
        
            del H5_class_instance

    def load_plot_save_q_spec(self):
        # ----------------------------------------------Load/Plot/Save QSpec------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/QSpec_ge/"
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
        extracted_freqs = []
        for h5_file in h5_files:
            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type=  'QSpec', save_r = int(save_round))
        
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
        
            for q_key in populated_keys:
                for dataset in range(len(load_data['QSpec'][q_key].get('Dates', [])[0])):
                    date = datetime.datetime.fromtimestamp(load_data['QSpec'][q_key].get('Dates', [])[0][dataset])
                    I = self.process_h5_data(load_data['QSpec'][q_key].get('I', [])[0][dataset].decode())
                    Q = self.process_h5_data(load_data['QSpec'][q_key].get('Q', [])[0][dataset].decode())
                    #I_fit = load_data['QSpec'][q_key].get('I Fit', [])[0][dataset]
                    #Q_fit = load_data['QSpec'][q_key].get('Q Fit', [])[0][dataset]
                    freqs = self.process_h5_data(load_data['QSpec'][q_key].get('Frequencies', [])[0][dataset].decode())
                    round_num = load_data['QSpec'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['QSpec'][q_key].get('Batch Num', [])[0][dataset]

                    exp_config = load_data['QSpec'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}

                    exp_config = eval(exp_config, safe_globals)
        
                    if len(I)>0:
        
                        qspec_class_instance = QubitSpectroscopy(q_key, self.number_of_qubits, self.outerFolder_save_plots, round_num, self.signal, self.save_figs)
                        q_spec_cfg = exp_config['qubit_spec_ge']
                        #print('q_spec_cfg: ', q_spec_cfg)
                        qubit_freq, _, _ = qspec_class_instance.plot_results(I, Q, freqs, q_spec_cfg, self.figure_quality)
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
            load_data = H5_class_instance.load_from_h5(data_type=  'Rabi', save_r = int(save_round))
        
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
                    date= datetime.datetime.fromtimestamp(load_data['Rabi'][q_key].get('Dates', [])[0][dataset])
                    I = self.process_h5_data(load_data['Rabi'][q_key].get('I', [])[0][dataset].decode())
                    Q = self.process_h5_data(load_data['Rabi'][q_key].get('Q', [])[0][dataset].decode())
                    gains = self.process_h5_data(load_data['Rabi'][q_key].get('Gains', [])[0][dataset].decode())
                    #fit = load_data['Rabi'][q_key].get('Fit', [])[0][dataset]
                    round_num = load_data['Rabi'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['Rabi'][q_key].get('Batch Num', [])[0][dataset]
                    syst_config = load_data['Rabi'][q_key].get('Syst Config', [])[0][dataset].decode()
                    exp_config = load_data['Rabi'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                    exp_config = eval(exp_config, safe_globals)
        
                    if len(I)>0:
        
                        rabi_class_instance = AmplitudeRabiExperiment(q_key, self.number_of_qubits, self.outerFolder_save_plots, round_num, self.signal, self.save_figs)
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
        print('Running load_plot_save_ss function')
        # ------------------------------------------------Load/Plot/Save SS---------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/ss_ge/"

        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

        for h5_file in h5_files:
        
            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]

            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type=  'SS', save_r = int(save_round))
        
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
                    date= datetime.datetime.fromtimestamp(load_data['SS'][q_key].get('Dates', [])[0][dataset])
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
        
                    if len(Q_g)>0:
                        ss_class_instance = SingleShot(q_key, self.number_of_qubits, self.outerFolder_save_plots, round_num, self.save_figs)

                        if type(exp_config) is dict:
                            readout_opt = exp_config['Readout_Optimization']
                            if isinstance(readout_opt, str):
                                ss_cfg = ast.literal_eval(readout_opt)
                            else:
                                ss_cfg = readout_opt
                        else:
                            ss_cfg = ast.literal_eval(exp_config['Readout_Optimization'].decode())
                        if plot_ss_hist_only:
                            ss_class_instance.only_hist_ssf(data=[I_g, Q_g, I_e, Q_e], cfg=ss_cfg, plot=True, plot_title=plot_title)
                        else:
                            ss_class_instance.hist_ssf(data=[I_g, Q_g, I_e, Q_e], cfg=ss_cfg, plot=True)
                        del ss_class_instance
        
            del H5_class_instance

    def run4_load_plot_save_ss(self, plot_ss_hist_only, plot_title, print_contents_ofH5 = False):
        print('Running RUN 4 version of load_plot_save_ss function')
        # ------------------------------------------------Load/Plot/Save SS---------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/ss_ge/"

        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

        for h5_file in h5_files:
            filename = os.path.basename(h5_file)
            QubitIndex = None
            if 'Qubit_3' in filename:
                QubitIndex = 2
            elif 'Qubit_4' in filename:
                QubitIndex = 3

            with h5py.File(h5_file, 'r') as f:
                if print_contents_ofH5:
                    print(f"\n--- Contents of {filename} ---")
                    for name in f:
                        print(name)
                    for name in f.keys():
                        obj = f[name]
                        if isinstance(obj, h5py.Group):
                            print(f"Group: {name}/")
                            for subname in obj:
                                print(f"  {name}/{subname}")
                        elif isinstance(obj, h5py.Dataset):
                            print(f"Dataset available for: {name}")
                    print("--- End of contents ---\n")

                try:
                    iq_list_e = f['excited_iq_data'][:]
                    iq_list_g = f['ground_iq_data'][:]
                    qubit_frequency = f['qubit_frequency'][()]
                    print('qubit freq:',qubit_frequency)
                    frequency = f['frequency'][()]
                    print('resonator freq:', frequency)

                    if "2024-11-12_10-00-32" in self.outerFolder:
                        iq_list_e = f['excited_iq_data'][QubitIndex, 0, :, :]  # shape (3000, 2)
                        iq_list_g = f['ground_iq_data'][QubitIndex, 0, :, :]  # shape (3000, 2)

                except Exception as e:
                    print(f"Failed to extract IQ data from {filename}: {e}")
                    continue

            ss_class = SingleShot(QubitIndex, self.number_of_qubits,  self.outerFolder, self.outerFolder_save_plots, self.run_num, save_figs=self.save_figs, experiment = None,
                                    verbose = False, logger = None, qick_verbose=False)
            ss_class.plot_results(iq_list_g, iq_list_e, QubitIndex,  fig_quality=200)


    def load_plot_save_t1(self, saved_shots = False):
        # ------------------------------------------------Load/Plot/Save T1----------------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/t1_ge/"
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
        soc, soccfg = makeProxy()

        for h5_file in h5_files:
        
            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type=  't1_ge', save_r = int(save_round))
        
            populated_keys = []
            for q_key in load_data['t1_ge']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['t1_ge'][q_key].get('Dates', [[]])
        
                # Check if any entry in 'Dates' is not NaN
                if any(
                        not np.isnan(date)
                        for date in dates_list[0]  # Iterate over the first batch of dates
                ):
                    populated_keys.append(q_key)
        
            for q_key in populated_keys:
                for dataset in range(len(load_data['t1_ge'][q_key].get('Dates', [])[0])):
                    #T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                    #errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                    date= datetime.datetime.fromtimestamp(load_data['t1_ge'][q_key].get('Dates', [])[0][dataset])

                    # cutoff when we switched to saving both averaged arrays *and* shots under Ishots/Qshots
                    cutoff_dt = datetime.datetime(2025, 10, 24, 13, 58, 37)

                    # --- NEW: make per-shot data compatible with per-delay fitting --------------------------------
                    if saved_shots:
                        # --- process IQ shots and turn them into IQ arrays (using Arianna's func, not QICK) --------------------------------
                        print("Processing shots...")

                        # --- load cfg strings from H5 ---
                        exp_config_str = load_data['t1_ge'][q_key]['Exp Config'][0][dataset].decode()
                        syst_config_str = load_data['t1_ge'][q_key]['Syst Config'][0][dataset].decode()

                        # --- choose which datasets hold the *shots* based on date ---
                        if date < cutoff_dt:
                            # before 2025-10-24_13-58-37: shots were saved under 'I' and 'Q'
                            I_key, Q_key = 'I', 'Q'
                        else:
                            # on/after the cutoff: shots were saved under 'Ishots' and 'Qshots'
                            I_key, Q_key = 'Ishots', 'Qshots'

                        # --- raw shots from H5 ---
                        Ishots_raw = self.process_h5_data(load_data['t1_ge'][q_key][I_key][0][dataset].decode())
                        Qshots_raw = self.process_h5_data(load_data['t1_ge'][q_key][Q_key][0][dataset].decode())

                        # --- path to the soccfg dump (txt file made with save_run_soccfg_params.py) ---
                        if self.run_num == 8:  # this does work
                            soccfg_dump_path = "/data/QICK_data/run8/6transmon/run8_soccfg_params/soccfg_full_dump_2025-11-10_15-14-35_firmware_during_run8_updated.txt"
                        elif self.run_num == 6:  # this doesn't work yet (shots need to be processed diff for run 6) but the skeleton is set up
                            soccfg_dump_path = "/data/QICK_data/run6/6transmon/loud2_soccfg_params/soccfg_full_dump_2025-11-04_16-30-54_firmware_during_run6.txt"

                        # --- init offline replica (no live soccfg) and set it up from strings + dump ---
                        replica = OfflineAcquireReplica(remove_offset=True, length_norm=True, edge_counting=False)
                        replica.setup_offline_from_strings(
                            exp_config_str,
                            syst_config_str,
                            soccfg_dump_path,
                            qubit_index=int(q_key))

                        exp_cfg = replica._safe_eval_cfg(exp_config_str)
                        syst_cfg = replica._safe_eval_cfg(syst_config_str)

                        # Pull steps/reps from Syst Config first; fall back to Exp Config only if missing
                        steps = int(syst_cfg.get('steps', exp_cfg['T1_ge']['steps']))
                        reps = int(syst_cfg.get('reps', exp_cfg['T1_ge']['reps']))
                        # rounds not needed here; H5 holds one round

                        # --- coerce raw shots to (rounds, N, reps) before averaging ---
                        Ishots = replica.coerce_to_rounds_N_reps(Ishots_raw, steps, reps)
                        Qshots = replica.coerce_to_rounds_N_reps(Qshots_raw, steps, reps)

                        # --- acquire (software average over a single round) ---
                        I, Q = replica.acquire_offline(Ishots, Qshots, soft_avgs=1)
                    # -----------------------------------------------------------------------------------------------
                    else:
                        I = self.process_h5_data(load_data['t1_ge'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data['t1_ge'][q_key].get('Q', [])[0][dataset].decode())

                    delay_times = self.process_h5_data(load_data['t1_ge'][q_key].get('Delay Times', [])[0][dataset].decode())
                    #fit = load_data['T1'][q_key].get('Fit', [])[0][dataset]
                    round_num = load_data['t1_ge'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['t1_ge'][q_key].get('Batch Num', [])[0][dataset]

                    # exp_config = load_data['t1_ge'][q_key].get('Exp Config', [])[0][dataset].decode()
                    # safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                    #
                    # exp_config = eval(exp_config, safe_globals)
        
                    if len(I)>0:
                        T1_class_instance = T1Measurement(q_key, self.number_of_qubits, self.outerFolder_save_plots, round_num, self.signal, self.save_figs, fit_data = True)
                        # T1_spec_cfg = exp_config['T1_ge'] # not using it for now, found out the one that should be used is the syst config one. that one gets updated during meas but expt doesn't
                        T1_class_instance.plot_results(I, Q, delay_times, date, self.figure_quality)
                        del T1_class_instance
        
            del H5_class_instance

    def load_t1_shots_vs_avgIQ_arrays(self, plot_both_methods_tog = False, plot_both_methods_diff = False, plot_T1res_method_comp = True):
        # ------------------------------------------------Load/Plot/Save T1----------------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/t1_ge/"
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

        # store post-processed T1 results from both methods, per qubit
        t1_results_by_qubit = {}  # q_index -> {"qick_avg": [...], "shots_avg": [...]}

        for h5_file in h5_files:

            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='t1_ge', save_r=int(save_round))

            populated_keys = []
            for q_key in load_data['t1_ge']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['t1_ge'][q_key].get('Dates', [[]])

                # Check if any entry in 'Dates' is not NaN
                if any(
                        not np.isnan(date)
                        for date in dates_list[0]  # Iterate over the first batch of dates
                ):
                    populated_keys.append(q_key)

            for q_key in populated_keys:
                for dataset in range(len(load_data['t1_ge'][q_key].get('Dates', [])[0])):

                    # -------- load t1 vals and errs saved into the h5 files during RR -------------------------------
                    # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                    # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                    date = datetime.datetime.fromtimestamp(load_data['t1_ge'][q_key].get('Dates', [])[0][dataset])

                    # --- process IQ shots and turn them into IQ arrays (using Arianna's func, not QICK) --------------------------------
                    print("Processing shots...")

                    # --- load cfg strings from H5 ---
                    exp_config_str = load_data['t1_ge'][q_key]['Exp Config'][0][dataset].decode()
                    syst_config_str = load_data['t1_ge'][q_key]['Syst Config'][0][dataset].decode()

                    # --- raw shots from H5  ---
                    I_key, Q_key = 'Ishots', 'Qshots'
                    if I_key not in load_data['t1_ge'][q_key] or Q_key not in load_data['t1_ge'][q_key]:
                        raise KeyError(f"{q_key}: HDF5 missing '{I_key}'/'{Q_key}'. "
                                       f"Found keys: {list(load_data['t1_ge'][q_key].keys())}")

                    # --- raw shots from H5 ---
                    Ishots_raw = self.process_h5_data(load_data['t1_ge'][q_key][I_key][0][dataset].decode())
                    Qshots_raw = self.process_h5_data(load_data['t1_ge'][q_key][Q_key][0][dataset].decode())

                    # --- path to the soccfg dump (txt file made with save_run_soccfg_params.py) ---
                    if self.run_num == 8:  # this does work
                        soccfg_dump_path = "/data/QICK_data/run8/6transmon/run8_soccfg_params/soccfg_full_dump_2025-11-10_15-14-35_firmware_during_run8_updated.txt"
                    elif self.run_num == 6:  # this doesn't work yet (shots need to be processed diff for run 6) but the skeleton is set up
                        soccfg_dump_path = "/data/QICK_data/run6/6transmon/loud2_soccfg_params/soccfg_full_dump_2025-11-04_16-30-54_firmware_during_run6.txt"

                    # --- init offline replica (no live soccfg) and set it up from strings + dump ---
                    replica = OfflineAcquireReplica(remove_offset=True, length_norm=True, edge_counting=False)
                    replica.setup_offline_from_strings(
                        exp_config_str,
                        syst_config_str,
                        soccfg_dump_path,
                        qubit_index=int(q_key))

                    exp_cfg = replica._safe_eval_cfg(exp_config_str)
                    syst_cfg = replica._safe_eval_cfg(syst_config_str)

                    # Pull steps/reps from Syst Config first; fall back to Exp Config only if missing. Sys config is the updated one in each measurement during RR
                    steps = int(syst_cfg.get('steps', exp_cfg['T1_ge']['steps']))
                    reps = int(syst_cfg.get('reps', exp_cfg['T1_ge']['reps']))
                    # rounds not needed here; H5 holds one round

                    # --- coerce raw shots to (rounds, N, reps) before averaging ---
                    Ishots = replica.coerce_to_rounds_N_reps(Ishots_raw, steps, reps)
                    Qshots = replica.coerce_to_rounds_N_reps(Qshots_raw, steps, reps)

                    # --- acquire (software average over a single round) ---
                    I_viashots, Q_viashots = replica.acquire_offline(Ishots, Qshots, soft_avgs=1)

                    # -------------------------- Extract avg IQ arrays from h5 files (these were made by QICK) --------------------------------------------
                    I = self.process_h5_data(load_data['t1_ge'][q_key].get('I', [])[0][dataset].decode())
                    Q = self.process_h5_data(load_data['t1_ge'][q_key].get('Q', [])[0][dataset].decode())

                    delay_times = self.process_h5_data(load_data['t1_ge'][q_key].get('Delay Times', [])[0][dataset].decode())
                    # fit = load_data['T1'][q_key].get('Fit', [])[0][dataset]
                    round_num = load_data['t1_ge'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['t1_ge'][q_key].get('Batch Num', [])[0][dataset]
                    exp_config = load_data['t1_ge'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                    exp_config = eval(exp_config, safe_globals)

                    if len(I) > 0:
                        T1_spec_cfg = exp_config['T1_ge']
                        # ------------------------------- plot a la avg IQ arrays ------------------------------
                        T1_class_instance = T1Measurement(q_key, self.number_of_qubits, self.outerFolder_save_plots,
                                                          round_num, self.signal, self.save_figs, fit_data=True)
                        I_avg, Q_avg, t, fit_avg, T1_err_avg, T1_est_avg, plot_sig_avg = T1_class_instance.plot_results(I, Q, delay_times, date, T1_spec_cfg,
                                                                                           self.figure_quality, iminuit_fit_instead = True)
                        del T1_class_instance

                        # ----------------------------------- plot a la shots ------------------------------
                        T1_class_instance = T1Measurement(q_key, self.number_of_qubits, self.unique_folder_path,
                                                          round_num, self.signal, self.save_figs, fit_data=True)
                        I_sh,  Q_sh,  t, fit_sh,  T1_err_sh,  T1_est_sh,  plot_sig_sh = T1_class_instance.plot_results(I_viashots, Q_viashots, delay_times, date, T1_spec_cfg,
                                                                                            self.figure_quality, iminuit_fit_instead = True)
                        del T1_class_instance

                        avg_tuple = (I_avg, Q_avg, t, fit_avg, T1_err_avg, T1_est_avg, plot_sig_avg)
                        shots_tuple = (I_sh, Q_sh, t, fit_sh, T1_err_sh, T1_est_sh, plot_sig_sh)

                        # NEW: store post-processed T1 results for this dataset
                        q_int = int(q_key)
                        if q_int not in t1_results_by_qubit:
                            t1_results_by_qubit[q_int] = {"qick_avg": [], "shots_avg": []}
                        t1_results_by_qubit[q_int]["qick_avg"].append(T1_est_avg)
                        t1_results_by_qubit[q_int]["shots_avg"].append(T1_est_sh)

                        # --- plot both methods in the same plot (offline shots averaged and QICK-averaged IQ arrays) ---
                        if plot_both_methods_tog:
                            self.plot_t1_overlay(
                                avg_tuple, shots_tuple,
                                labels=("Avg-IQ", "Shots-Offline"),
                                title_prefix="T1 Overlay",
                                qubit_index=q_key,
                                out_dir=f"/data/QICK_data/run8/6transmon/replotted_RR_data/{self.date}/avgIQ_andshots_plotted_tog",
                                dpi=140)

                        # --- difference plot (offline shots-averaged minus QICK-averaged IQ arrays) ---
                        if plot_both_methods_diff:
                            self.plot_t1_array_difference(
                                I_qick=I,  # QICK-averaged IQ from H5
                                Q_qick=Q,
                                I_fromshots=I_viashots,  # Your offline-averaged-from-shots IQ
                                Q_fromshots=Q_viashots,
                                delay_times=delay_times,
                                title_prefix="T1 AvgIQ vs Shots Diff",
                                qubit_index=q_key,
                                out_dir=f"/data/QICK_data/run8/6transmon/replotted_RR_data/{self.date}/avgIQ_minus_shots_diff",
                                dpi=140
                            )

            del H5_class_instance

        # after processing all datasets, plot T1 from both methods for all qubits
        if plot_T1res_method_comp:
            self.plot_t1_methods_comparison_all_qubits(
                t1_results_by_qubit,
                out_dir=f"/data/QICK_data/run8/6transmon/replotted_RR_data/{self.date}/T1_results_method_comparison_all_qubits"
            )

    def plot_t1_methods_comparison_all_qubits(
            self,
            t1_results_by_qubit,
            out_dir,
            title_prefix="T1: Qick Avg IQ vs Offline Avg IQ Shots",
            dpi=140,
    ):
        """
        Plot T1 estimates from both averaging methods for all qubits. Default qick way vs offline way.

        t1_results_by_qubit:
            dict[q_index] = {
                "qick_avg":   [T1_est_avg_0,   T1_est_avg_1,   ...],
                "shots_avg": [T1_est_shots_0, T1_est_shots_1, ...],
            }
            (Lists are in the order datasets were processed.)
        """
        os.makedirs(out_dir, exist_ok=True)

        qubit_indices = sorted(t1_results_by_qubit.keys())
        n_qubits = len(qubit_indices)
        if n_qubits == 0:
            print("[T1 summary] No T1 results to plot.")
            return

        # simple grid layout: up to 3 columns
        ncols = min(3, n_qubits)
        nrows = int(np.ceil(n_qubits / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharey=True)
        axes = np.atleast_1d(axes).ravel()

        for ax, qidx in zip(axes, qubit_indices):
            data = t1_results_by_qubit[qidx]
            t1_avg = np.asarray(data["qick_avg"], dtype=float)
            t1_sh = np.asarray(data["shots_avg"], dtype=float)

            # --- max |delta T1| on overlapping points (ignore any extra failed entries) ---
            min_len = min(len(t1_avg), len(t1_sh))
            if min_len > 0:
                diffs = np.abs(t1_avg[:min_len] - t1_sh[:min_len])
                max_diff = float(np.nanmax(diffs))
            else:
                max_diff = np.nan

            n_pts = max(len(t1_avg), len(t1_sh))
            x = np.arange(n_pts)

            # protect against unequal lengths (shouldn't normally happen)
            if len(t1_avg) != n_pts:
                t1_avg = np.pad(
                    t1_avg,
                    (0, n_pts - len(t1_avg)),
                    mode="constant",
                    constant_values=np.nan
                )
            if len(t1_sh) != n_pts:
                t1_sh = np.pad(
                    t1_sh,
                    (0, n_pts - len(t1_sh)),
                    mode="constant",
                    constant_values=np.nan
                )

            ax.plot(x, t1_avg, "o-", label="Qick T1", linewidth=1)
            ax.plot(x, t1_sh, "s--", label="Offline Shots T1", linewidth=1)

            # per-qubit title
            if np.isfinite(max_diff):
                ax.set_title(f"Q{qidx + 1}, max T1 diff = {max_diff:.3g} us", fontsize=9 )
            else:
                ax.set_title(f"Q{qidx + 1}, max delta T1 = n/a", fontsize=9 )

            ax.set_xlabel("Dataset index")
            ax.grid(alpha=0.3)

        # Common y-label and global title
        for ax in axes:
            ax.set_ylabel(r"$T_1$ (µs)")

        # Remove any unused axes if n_qubits < nrows*ncols
        for j in range(len(qubit_indices), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(title_prefix, fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])

        fname = os.path.join(out_dir, "T1_Comparison_All_Qubits.png")
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)

        print(f"[T1 summary] Saved comparison plot to: {fname}")

    def plot_t1_array_difference(self,
                                 I_qick, Q_qick,
                                 I_fromshots, Q_fromshots,
                                 delay_times,
                                 title_prefix,
                                 qubit_index,
                                 out_dir,
                                 dpi=140):
        """
        Plot the point-by-point difference between the QICK-averaged IQ arrays
        and the arrays obtained by averaging shots offline.

        delta I = I_fromshots - I_qick
        delta Q = Q_fromshots - Q_qick
        """
        os.makedirs(out_dir, exist_ok=True)

        I_qick = np.asarray(I_qick, dtype=float)
        Q_qick = np.asarray(Q_qick, dtype=float)
        I_fromshots = np.asarray(I_fromshots, dtype=float)
        Q_fromshots = np.asarray(Q_fromshots, dtype=float)
        delay_times = np.asarray(delay_times, dtype=float)

        # Basic shape sanity checks
        if I_qick.shape != I_fromshots.shape:
            raise ValueError(
                f"I array shape mismatch: QICK {I_qick.shape} vs shots {I_fromshots.shape}"
            )
        if Q_qick.shape != Q_fromshots.shape:
            raise ValueError(
                f"Q array shape mismatch: QICK {Q_qick.shape} vs shots {Q_fromshots.shape}"
            )
        if delay_times.shape[0] != I_qick.shape[0]:
            raise ValueError(
                f"delay_times length {delay_times.shape[0]} does not match IQ length {I_qick.shape[0]}"
            )

        dI = I_fromshots - I_qick
        dQ = Q_fromshots - Q_qick
        dIQ_mag = np.sqrt(dI ** 2 + dQ ** 2)

        max_abs_dI = np.max(np.abs(dI))
        max_abs_dQ = np.max(np.abs(dQ))
        max_abs_dIQ = np.max(dIQ_mag)

        try:
            qb_str = f"Q{int(qubit_index) + 1}"
        except (TypeError, ValueError):
            qb_str = f"Q{qubit_index}"

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 8))

        # diff I
        axes[0].plot(delay_times, dI, marker='o')
        axes[0].axhline(0.0, linestyle='--', linewidth=0.8)
        axes[0].set_ylabel(r'$\Delta I$ (shots - QICK)')
        axes[0].set_title(
            f"{title_prefix} {qb_str}\n"
            f"max |delta I| = {max_abs_dI:.3g}, |delta Q| = {max_abs_dQ:.3g}")

        # diff Q
        axes[1].plot(delay_times, dQ, marker='o')
        axes[1].axhline(0.0, linestyle='--', linewidth=0.8)
        axes[1].set_ylabel(r'$\Delta Q$ (shots - QICK)')

        fig.tight_layout()

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = os.path.join(
            out_dir,
            f"{title_prefix.replace(' ', '_')}_{qb_str}_diff_{now}.png"
        )
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)

        # Optional: print summary to log / stdout
        msg = (f"[T1 diff] {qb_str}: max |delta I| = {max_abs_dI:.3g}, "
               f"max |delta Q| = {max_abs_dQ:.3g}")
        if hasattr(self, "logger"):
            self.logger.info(msg)
        else:
            print(msg)


    def plot_t1_overlay(self,
            avg_res,  # (I, Q, delay_times, fit, T1_err, T1_est, plot_sig) from method A
            shots_res,  # (I, Q, delay_times, fit, T1_err, T1_est, plot_sig) from method B
            *,
            labels=("QICK Avg-IQ", "Shots-Offline"),
            title_prefix="T1 Overlay",
            qubit_index=None,
            out_dir=None,
            dpi=120,
            fig_size=(10, 8)
    ):
        """
        Overlay T1 data & fits from two pipelines on the same figure.

        Parameters
        ----------
        avg_res : tuple
            (I, Q, delay_times, fit, T1_err, T1_est, plot_sig) for pipeline A.
        shots_res : tuple
            (I, Q, delay_times, fit, T1_err, T1_est, plot_sig) for pipeline B.
        labels : (str, str)
            Legend labels for A and B.
        title_prefix : str
            Prefix for the figure title.
        qubit_index : int or None
            If provided, included in the title.
        out_dir : str or None
            If provided, the plot is saved there (PNG). Folder is created if needed.
        dpi : int
            Figure DPI when saving.
        fig_size : (float, float)
            Matplotlib figure size.
        """
        (I_a, Q_a, t_a, fit_a, err_a, T1_a, sig_a) = avg_res
        (I_b, Q_b, t_b, fit_b, err_b, T1_b, sig_b) = shots_res

        # Sanity: time axes must match to overlay meaningfully
        if not np.allclose(np.asarray(t_a), np.asarray(t_b)):
            # If different, we still plot both, but warn in the title.
            time_mismatch = True
            t = t_a  # use A's axis for x labels
        else:
            time_mismatch = False
            t = t_a

        fig, (axI, axQ) = plt.subplots(2, 1, figsize=fig_size, sharex=True)
        plt.rcParams.update({'font.size': 16})

        # Title
        qtxt = f" Q{qubit_index + 1}" if qubit_index is not None else ""
        warn = " [time axes differ]" if time_mismatch else ""
        fig.suptitle(f"{title_prefix}{qtxt}{warn}", y=0.98, fontsize=20)

        # --- I panel ---
        axI.plot(t, I_a, marker = "o", label=f"{labels[0]}: I") # lw=2
        axI.plot(t, I_b, marker = "o", linestyle="--", label=f"{labels[1]}: I") # lw=2

        # if fits exist and were done on I, overlay them
        if fit_a is not None and (sig_a == 'I'):
            lab = f"{labels[0]} fit (T1={T1_a:.2f} µs±{(err_a or np.nan):.2g})"
            axI.plot(t, fit_a, lw=3, alpha=0.9, label=lab)
        if fit_b is not None and (sig_b == 'I'):
            lab = f"{labels[1]} fit (T1={T1_b:.2f} µs±{(err_b or np.nan):.2g})"
            axI.plot(t, fit_b, lw=3, alpha=0.9, linestyle="--", label=lab)

        axI.set_ylabel("I amplitude (a.u.)")
        axI.legend(loc="best")
        axI.grid(True, alpha=0.25)

        # --- Q panel ---
        axQ.plot(t, Q_a, marker = "o" , label=f"{labels[0]}: Q") # lw=2
        axQ.plot(t, Q_b, marker = "o", linestyle="--", label=f"{labels[1]}: Q") # lw=2

        # if fits exist and were done on Q, overlay them
        if fit_a is not None and (sig_a == 'Q'):
            lab = f"{labels[0]} fit (T1={T1_a:.2f} µs±{(err_a or np.nan):.2g})"
            axQ.plot(t, fit_a, lw=3, alpha=0.9, label=lab)
        if fit_b is not None and (sig_b == 'Q'):
            lab = f"{labels[1]} fit (T1={T1_b:.2f} µs±{(err_b or np.nan):.2g})"
            axQ.plot(t, fit_b, lw=3, alpha=0.9, linestyle="--", label=lab)

        axQ.set_xlabel("Delay time (µs)")
        axQ.set_ylabel("Q amplitude (a.u.)")
        axQ.legend(loc="best")
        axQ.grid(True, alpha=0.25)

        plt.tight_layout(rect=(0, 0, 1, 0.96))

        # Save if requested
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            qb = f"_Q{qubit_index + 1}" if qubit_index is not None else ""
            fname = f"T1_overlay{qb}_{now}.png"
            path = os.path.join(out_dir, fname)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            # Return fig, axes, and saved path for logging
            return fig, (axI, axQ), path

        return fig, (axI, axQ), None

    def load_plot_save_t2r(self):
        # -------------------------------------------------------Load/Plot/Save T2R------------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/t2_ge/"
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
        
        for h5_file in h5_files:
            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type=  't2_ge', save_r = int(save_round))
        
            populated_keys = []
            for q_key in load_data['t2_ge']:
                # Access 'Dates' for the current q_key
                dates_list = load_data['t2_ge'][q_key].get('Dates', [[]])
        
                # Check if any entry in 'Dates' is not NaN
                if any(
                        not np.isnan(date)
                        for date in dates_list[0]  # Iterate over the first batch of dates
                ):
                    populated_keys.append(q_key)
        
            for q_key in populated_keys:
                for dataset in range(len(load_data['t2_ge'][q_key].get('Dates', [])[0])):
                    #T2 = load_data['T2'][q_key].get('T2', [])[0][dataset]
                    #errors = load_data['T2'][q_key].get('Errors', [])[0][dataset]
                    date = datetime.datetime.fromtimestamp(load_data['t2_ge'][q_key].get('Dates', [])[0][dataset])
                    I = self.process_h5_data(load_data['t2_ge'][q_key].get('I', [])[0][dataset].decode())
                    Q = self.process_h5_data(load_data['t2_ge'][q_key].get('Q', [])[0][dataset].decode())
                    delay_times = self.process_h5_data(load_data['t2_ge'][q_key].get('Delay Times', [])[0][dataset].decode())
                    #fit = load_data['T2'][q_key].get('Fit', [])[0][dataset]
                    round_num = load_data['t2_ge'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['t2_ge'][q_key].get('Batch Num', [])[0][dataset]

                    exp_config = load_data['t2_ge'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}

                    exp_config = eval(exp_config, safe_globals)
        
                    if len(I) > 0:
                        T2_class_instance = T2RMeasurement(q_key, self.number_of_qubits, self.outerFolder_save_plots, round_num, self.signal, self.save_figs, fit_data = True)
                        try:
                            fitted, t2r_est, t2r_err, plot_sig = T2_class_instance.t2_fit(delay_times, I, Q)
                        except Exception as e:
                            print('Fit didnt work due to error: ', e)
                            continue
                        # T2_cfg = exp_config['Ramsey_ge']

                        # --------- simple peak-count gate on the fitted curve ----------
                        try:
                            min_peaks = 3
                            y_fit = np.asarray(fitted, float)
                            if y_fit.size < 3:
                                continue

                            # ignore micro-wiggles: require peaks be at least ~10% of the record apart
                            min_dist = max(3, y_fit.size // 10)

                            pks, _ = find_peaks(y_fit, distance=min_dist)
                            trs, _ = find_peaks(-y_fit, distance=min_dist)
                            n_osc = min(len(pks), len(trs))  # need alternating ups/downs

                            if n_osc < min_peaks:
                                print('Rejected a T2R scan. Failed ramsey shape, less than 3 oscillations.')
                                continue
                        except Exception:
                            # if peak counting fails for any reason, be conservative and skip
                            continue
                        # ----------------------------------------------------------------
                        # goodness of fit check---------------------------------------------
                        y = I if plot_sig == "I" else Q

                        # Compute R-squared goodness-of-fit
                        ss_res = np.sum((y - fitted) ** 2)
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        r2 = 1 - ss_res / ss_tot

                        if r2 < 0.15:  # adjust threshold if needed
                            print(f"Bad T2R fit for Q{int(q_key) + 1}, R² = {r2:.2f}")
                            continue
                        #--------------------------------------------------------------------------------

                        if t2r_est < 0:
                            print("The value is negative, continuing...")
                            continue
                        max_t1 = 150 # max(t1_vals[q_key]) # our T1s are below this rn
                        if t2r_est > 2 * max_t1:
                            print(f"The value is above 2*{max_t1} us, this is a bad fit, continuing...")
                            continue

                        T2_class_instance.plot_results(I, Q, delay_times, date, fitted, t2r_est, t2r_err, plot_sig, fig_quality=self.figure_quality)
                        del T2_class_instance
        
            del H5_class_instance

    def load_plot_save_t2e(self):
        # -------------------------------------------------------Load/Plot/Save T2E------------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/T2E_ge/"
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
        
        for h5_file in h5_files:
            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type=  'T2E', save_r = int(save_round))
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
                    #T2 = load_data['T2E'][q_key].get('T2', [])[0][dataset]
                    #errors = load_data['T2E'][q_key].get('Errors', [])[0][dataset]
                    date = datetime.datetime.fromtimestamp(load_data['T2E'][q_key].get('Dates', [])[0][dataset])
                    I = self.process_h5_data(load_data['T2E'][q_key].get('I', [])[0][dataset].decode())
                    Q = self.process_h5_data(load_data['T2E'][q_key].get('Q', [])[0][dataset].decode())
                    delay_times = self.process_h5_data(load_data['T2E'][q_key].get('Delay Times', [])[0][dataset].decode())
                    #fit = load_data['T2E'][q_key].get('Fit', [])[0][dataset]
                    round_num = load_data['T2E'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['T2E'][q_key].get('Batch Num', [])[0][dataset]

                    exp_config = load_data['T2E'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}

                    exp_config = eval(exp_config, safe_globals)
        
                    if len(I) > 0:
                        T2E_class_instance = T2EMeasurement(q_key, self.number_of_qubits, self.outerFolder_save_plots, round_num, self.signal, self.save_figs, fit_data = True)
                        try:
                            fitted, t2e_est, t2e_err, plot_sig = T2E_class_instance.t2_fit(delay_times, I, Q)
                        except Exception as e:
                            print('Fit didnt work due to error: ', e)
                            continue
                        T2E_cfg = exp_config['SpinEcho_ge']
                        T2E_class_instance.plot_results(I, Q, delay_times, date, fitted, t2e_est, t2e_err, plot_sig, config = T2E_cfg, fig_quality=self.figure_quality)
                        del T2E_class_instance
        
            del H5_class_instance

    def load_plot_save_ss_gef(self, plot_ssf_gef, process_one_file = False, file_to_process = None, qubit_index = None):

        # ------------------------------------------------Load/Plot/Save g-e-f SS---------------------------------------
        outerFolder_expt = self.outerFolder + "/Data_h5/SS_gef/" #checks folder for a single date

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
                            ig_new, qg_new, ie_new, qe_new, if_new, qf_new, theta_ge, threshold_ge = ss_class_instance.hist_ssf(data=[I_g, Q_g, I_e, Q_e, I_f, Q_f], cfg=ss_cfg, plot=True, fig_quality = 200)
                        else:
                            ig_new, qg_new, ie_new, qe_new, if_new, qf_new, theta_ge, threshold_ge = ss_class_instance.hist_ssf(
                                data=[I_g, Q_g, I_e, Q_e, I_f, Q_f], cfg=ss_cfg, plot=False, fig_quality=200)
                        del ss_class_instance
                        del H5_class_instance
            return I_g, Q_g, I_e, Q_e, I_f, Q_f, ig_new, qg_new, ie_new, qe_new, if_new, qf_new, theta_ge, threshold_ge # new arrays are the rotated data


    def load_plot_save_rabis_Qtemps(self, list_of_all_qubits):
        # ------------------------------------------------Load/Plot/Save Rabi---------------------------------------
        outerFolder_expt_qtemps = self.unique_folder_path+ "/Data_h5/q_temperatures/"
        h5_files = glob.glob(os.path.join(outerFolder_expt_qtemps, "*.h5"))
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
                    date = datetime.datetime.fromtimestamp(load_data['q_temperatures'][q_key].get('Dates', [])[0][dataset])
                    round_num = load_data['q_temperatures'][q_key].get('Round Num', [])[0][dataset]
                    # batch_num = load_data['q_temperatures'][q_key].get('Batch Num', [])[0][dataset]
                    #-------------------------------------Grabbing matching qubit frequency for this qubit-------------------------------------
                    if date.timestamp() > cutoff_timestamp: #files after this date contain the matching g-e qubit frequency already
                        qubit_freq_MHz = load_data['q_temperatures'][q_key].get('Qfreq_ge', [])[0][dataset]
                        # print(f"QSpec Q{q_key}: {qubit_freq_MHz:.3f} MHz")
                    else: #look through matching qspec file
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
                        save_figs = False
                        rabi_class_instance = Temps_EFAmpRabiExperiment(q_key, self.number_of_qubits, list_of_all_qubits,
                                                                      self.outerFolder_save_plots, round_num,
                                                                      self.signal, save_figs)
                        I1 = np.asarray(I1)
                        Q1 = np.asarray(Q1)
                        gains1 = np.asarray(gains1)
                        best_signal_fit1, pi_amp1, A_amplitude1, amp_fit1 = rabi_class_instance.plot_results(I1, Q1, gains1, rabi_cfg, self.figure_quality)
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
                        best_signal_fit2, pi_amp2, A_amplitude2, amp_fit2 = rabi_class_instance.plot_results(I2, Q2, gains2, rabi_cfg, self.figure_quality)
                        del rabi_class_instance

                    if A_amplitude1 is not None and A_amplitude2 is not None:
                        A_e = A_amplitude1
                        A_g = A_amplitude2

                        results = self.Qubit_Temperature_Convert(A_e, A_g, qubit_freq_MHz)
                        if results is None:
                            continue  # Skip this dataset
                        T_K, T_mK, P_e, qubit_freq = results
                        print(f"Q{q_key} calculated Temperature:{T_mK}, with P_e = {P_e}, and Qfreq {qubit_freq_MHz} MHz")
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
                                 sharex=False, constrained_layout=True)  # set sharex=False if you want each subplot to manage ticks independently
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


import re
import numpy as np
from collections import OrderedDict

class OfflineAcquireReplica:
    """
    Offline mirror of QICK Averager acquire() math WITHOUT thresholding:
      avg over reps -> divide by ro length -> subtract IQ offset -> sum over rounds -> /soft_avgs
    Returns (I_final, Q_final) each shape (N_steps,).

    This version reads necessary per-readout info (decimated MHz, iq_offset_effective)
    from a saved soccfg text dump created by your helper script.
    """

    # -------------------- init --------------------
    def __init__(self, remove_offset=True, length_norm=True, progress=True, edge_counting = False):
        self.remove_offset = bool(remove_offset)
        self.length_norm   = bool(length_norm)
        self.progress      = bool(progress)
        self.edge_counting = bool(edge_counting)

        # No live soccfg used anymore; store a parsed dump instead
        self._dump_readouts = {}   # ro_ch -> {"decimated_MHz": float, "iq_offset_effective": float}

        # QICK-like fields (kept for downstream logic)
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
        self._iq_offset = None     # effective offset
        self._ro_index  = None     # readout channel index for this qubit

    # -------------------- helpers --------------------
    def _safe_eval_cfg(self, cfg_str):
        s = re.sub(r"<qick\.asm_v2\.QickParam object at 0x[0-9a-fA-F]+>", "None", cfg_str)
        s = re.sub(r"np\.float64\(\s*([^)]+)\s*\)", r"float(\1)", s)
        safe_globals = {"np": np, "array": np.array, "float": float, "__builtins__": {}}
        return eval(s, safe_globals)

    def _ensure_rounds_axis(self, A, *, N, rounds, reps):
        """
        Return A shaped as (rounds, N, reps). Accepts many 1D/2D/3D variants.
        """
        A = np.asarray(A)

        if A.ndim == 3:
            r, n, rr = A.shape
            if n != N:
                raise ValueError(f"3D shots second dim {n} != N {N}")
            if rr != reps:
                print(f"[warn] 3D shots reps={rr} != cfg reps={reps}; using shots value.")
                self._reps = rr
            if r != rounds:
                print(f"[warn] 3D shots rounds={r} != cfg rounds={rounds}; using shots value.")
                self._rounds = r
            return A

        if A.ndim == 2:
            r0, r1 = A.shape
            # (N, reps)
            if r0 == N and r1 == reps:
                return A[None, ...]  # (1, N, reps)
            # (reps, N)
            if r0 == reps and r1 == N:
                return A.T[None, ...]  # (1, N, reps)
            # (rounds*N, reps)
            if (r0 % N) == 0 and r1 == reps:
                rcalc = r0 // N
                return A.reshape(rcalc, N, reps)
            # (N, rounds*reps)
            if (r1 % reps) == 0 and r0 == N:
                rcalc = r1 // reps
                return A.reshape(1, N, reps) if rcalc == 1 else A.reshape(rcalc, N, reps)

            raise ValueError(f"Unexpected 2D shape {A.shape} for N={N}, reps={reps}, rounds={rounds}")

        if A.ndim == 1:
            total = A.size
            if total == N * reps:
                return A.reshape(1, N, reps)
            if total == N:
                # already per-step averaged; treat as reps=1, rounds=1
                return A.reshape(1, N, 1)
            raise ValueError(f"1D shots length {total} not compatible with N={N}, reps={reps}")

        raise ValueError(f"Shots must be 1D/2D/3D, got {A.ndim}D {A.shape}")

    def _extract_t1_dims_from_syst(self, syst_cfg, qubit_index):
        """
        Pull steps, reps, and rounds for T1 from the top-level system config.

        This syst_cfg is already experiment- and qubit-specific,
        so all three are just scalars.
        """

        def pick_required(name, *aliases):
            for key in (name,) + aliases:
                if key in syst_cfg:
                    return syst_cfg[key]
            raise ValueError(f"System config missing required key '{name}' (or aliases {aliases}).")

        steps = pick_required('steps', 'n_steps', 'n_expts')
        reps = pick_required('reps', 'nreps')
        rounds = pick_required('rounds', 'soft_avgs')

        return int(steps), int(reps), int(rounds)

    # -------------------- dump parsing --------------------
    def _parse_soccfg_dump(self, text):
        """
        Parse the soccfg dump text to fill self._dump_readouts with:
          ro_ch -> {"decimated_MHz": float, "iq_offset_effective": float}
        We parse from two places:
          1) [FULL PRINT OUTPUT OF SOCCFG] block: lines like
             "<ch>: ... decimated=38.400 MHz"
          2) [READOUT CHANNEL DIAGNOSTICS] block: lines like
             "--- Readout Channel <ch> ---" + "iq_offset_effective: <val>"
        """
        self._dump_readouts = {}

        # 1) decimated MHz lines from full print
        #    e.g.: "2:  axis_pfb_readout_v4 - ... decimated=38.400 MHz"
        for m in re.finditer(r"^\s*(\d+):\s*axis_.*readout.*?decimated\s*=\s*([0-9.]+)\s*MHz",
                             text, flags=re.MULTILINE):
            ch = int(m.group(1))
            dec = float(m.group(2))
            self._dump_readouts.setdefault(ch, {})["decimated_MHz"] = dec

        # 2) diagnostics block for effective offset per channel
        diag_blocks = re.split(r"\n\s*\[READOUT CHANNEL DIAGNOSTICS\]\s*\n", text, maxsplit=1)
        if len(diag_blocks) == 2:
            diagnostics_text = diag_blocks[1]

            # iterate per-channel block
            for m in re.finditer(r"---\s*Readout Channel\s+(\d+)\s*---\s*([\s\S]*?)(?=(?:---\s*Readout Channel|\Z))",
                                 diagnostics_text):
                ch = int(m.group(1))
                body = m.group(2)

                # decimated_MHz_exact
                mm = re.search(r"decimated_MHz_exact:\s*([0-9.]+)", body)
                if mm:
                    self._dump_readouts.setdefault(ch, {})["decimated_MHz_exact"] = float(mm.group(1))

                # iq_offset_effective
                mm = re.search(r"iq_offset_effective:\s*([\-0-9.]+)", body)
                if mm:
                    self._dump_readouts.setdefault(ch, {})["iq_offset_effective"] = float(mm.group(1))

        # Defaults if fields missing
        for ch, d in list(self._dump_readouts.items()):
            if "decimated_MHz_exact" not in d and "decimated_MHz" not in d:
                d["decimated_MHz"] = 307.2  # fallback for old dumps
                print(f"[warn] no decimated_MHz(_exact) found for ro_ch {ch}; defaulting to 307.2")
            if "iq_offset_effective" not in d:
                d["iq_offset_effective"] = 0.0

    def _to_float(self,val):
        # handles numbers or strings like "38.4", "38.4 MHz", etc.
        s = str(val)
        import re
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not m:
            raise ValueError(f"Cannot parse float from {val!r}")
        return float(m.group(0))
    # -------------------- compute norm & offset (from dump only) --------------------
    def _compute_ro_norm_and_offset_from_dump(self, *, syst_cfg, qubit_index):
        """
        Use parsed dump + per-qubit res_length from syst_cfg to reproduce QICK math:
          ro_length_cycles = trunc(res_length_us * decimated_MHz)
          iq_offset_effective from diagnostics
        """
        qidx = int(qubit_index)

        # ro_ch can be scalar or list
        ro_ch_field = syst_cfg['ro_ch']
        ro_ch_for_q = int(ro_ch_field[qidx]) if isinstance(ro_ch_field, (list, tuple)) else int(ro_ch_field)

        # res_length can be scalar or list (some setups vary per qubit)
        res_len_field = (syst_cfg.get('res_length') or syst_cfg.get('res_len') or syst_cfg.get('read_length'))
        if res_len_field is None:
            raise KeyError("System config missing 'res_length' (or alias).")
        res_len_us = float(res_len_field[qidx] if isinstance(res_len_field, (list, tuple)) else res_len_field)

        # decimated rate & effective IQ offset from the soccfg dump you already parse
        info = self._dump_readouts.get(ro_ch_for_q, {})

        # Prefer exact if present; otherwise fall back to decimated_MHz.
        if "decimated_MHz_exact" in info:
            dec_mhz = self._to_float(info["decimated_MHz_exact"])
        else:
            if "decimated_MHz" not in info:
                raise KeyError(
                    f"Neither 'decimated_MHz_exact' nor 'decimated_MHz' found for ro_ch={ro_ch_for_q}. "
                    f"Available keys: {list(info.keys())}"
                )
            dec_mhz = self._to_float(info["decimated_MHz"])

        offset_eff = float(info.get("iq_offset_effective", 0.0))

        # EXACTLY how QICK derives buffer length: trunc(us * MHz)
        ro_cycles = int(np.trunc(res_len_us * dec_mhz))
        return ro_cycles, offset_eff, ro_ch_for_q

    def to_int(self, val, scale, quantize=1, parname=None, trunc=False):
        """Convert a parameter value from user units to ASM units.
        Normally this means converting from float to int.
        For the v2 tProcessor this can also convert QickParam to QickRawParam.
        To avoid overflow, values are rounded towards zero using np.trunc().

        Parameters
        ----------
        val : float or QickParam
            parameter value or sweep range
        scale : float
            conversion factor
        quantize : int
            rounding step for ASM value
        parname : str
            parameter type - only for sweeps
        trunc : bool
            round towards zero using np.trunc(), instead of to closest integer using np.round()

        Returns
        -------
        int or QickRawParam
            ASM value
        """
        if hasattr(val, 'to_int'):
            return val.to_int(scale, quantize=quantize, parname=parname, trunc=trunc)
        else:
            if trunc:
                return int(quantize * np.trunc(val * scale / quantize))
            else:
                return int(quantize * np.round(val * scale / quantize))

    # -------------------- setup --------------------
    def setup_offline_from_strings(self, exp_config_str, syst_config_str, soccfg_dump_path, qubit_index):
        """
        exp_config_str: repr(string) of exp cfg
        syst_config_str: repr(string) of system cfg
        soccfg_dump_path: path to the saved dump .txt
        qubit_index: int
        """
        exp_cfg  = self._safe_eval_cfg(exp_config_str)
        syst_cfg = self._safe_eval_cfg(syst_config_str)

        # Parse dump once
        with open(soccfg_dump_path, "r") as f:
            dump_text = f.read()
        self._parse_soccfg_dump(dump_text)

        # Dimensions: now from system config, not experiment config
        steps, reps, rounds = self._extract_t1_dims_from_syst(syst_cfg, qubit_index)
        self._N_steps = steps
        self._reps = reps
        self._rounds = rounds

        # Norm + offset from dump only
        ro_cycles, iq_offset, ro_ch_for_q = self._compute_ro_norm_and_offset_from_dump(
            syst_cfg=syst_cfg, qubit_index=qubit_index
        )
        self._ro_cycles = ro_cycles
        self._iq_offset = iq_offset
        self._ro_index  = ro_ch_for_q

        # self.debug_print_params("offline_setup", qubit_index) # optional, print params

        # Mirror QICK bookkeeping (edge_counting=False like your run-8 baseline)
        self.ro_chs = OrderedDict({
            ro_ch_for_q: {
                'length': int(ro_cycles),
                'trigs': 1,
                'edge_counting': self.edge_counting,
                # minimal "ro_config" that carries effective offset (so _ro_offset_qick can use it)
                'ro_config': {'iq_offset_effective': float(iq_offset)}
            }
        })
        self.loop_dims      = [self._reps, self._N_steps]
        self.avg_level      = 0
        self.reads_per_shot = [1]

    def _choose_steps_reps(self, flat, steps, reps):
        # Candidate A: assume saved as (steps, reps)
        A = flat.reshape(steps, reps)
        # Candidate B: assume saved as (reps, steps) -> transpose to (steps, reps)
        B = flat.reshape(reps, steps).T
        # Heuristic: the correct orientation should have larger variation across steps
        sA = A.mean(axis=1).std()
        sB = B.mean(axis=1).std()
        return A if sA >= sB else B

    def debug_print_params(self, label, qubit_index):
        print(
            f"[{label}] Q{int(qubit_index) + 1}: "
            f"steps={self._N_steps}, reps={self._reps}, rounds={self._rounds}, "
            f"ro_cycles={self._ro_cycles}, iq_offset={self._iq_offset}"
        )

    def coerce_to_rounds_N_reps(self, A, steps, reps):
        A = np.asarray(A)

        if A.ndim == 3:
            return A

        if A.ndim == 2:
            r0, r1 = A.shape
            # exact matches
            if r0 == steps and r1 == reps:
                return A[None, ...]  # (1, steps, reps)
            if r0 == reps and r1 == steps:
                return A.T[None, ...]  # (1, steps, reps)
            # ambiguous: try to infer (flatten and choose)
            if r0 * r1 == steps * reps:
                C = self._choose_steps_reps(A.ravel(), steps, reps)
                return C[None, ...]
            raise ValueError(f"Unexpected 2D shape {A.shape} for steps={steps}, reps={reps}")

        if A.ndim == 1:
            total = A.size
            if total == steps * reps:
                C = self._choose_steps_reps(A, steps, reps)
                return C[None, ...]  # (1, steps, reps)
            if total == steps:
                return A.reshape(1, steps, 1)  # already per-step mean
            raise ValueError(f"1D shots length {total} not compatible with steps={steps}, reps={reps}")

        raise ValueError(f"Shots must be 1D/2D/3D, got {A.ndim}D {A.shape}")

    # -------------------- averaging kernel --------------------
    def _ro_offset_qick(self, ro_ch, chcfg):
        """
        Use the effective offset we stored in ro_config.
        No doubling logic here (already done when generating the dump).
        """
        try:
            if chcfg and 'iq_offset_effective' in chcfg:
                return float(chcfg['iq_offset_effective'])
        except Exception:
            pass
        return 0.0

    def _average_buf_qick(self, d_reps, reads_per_shot, *, length_norm=True, remove_offset=True):
        avg_d = []
        for i_ch, (ch, ro) in enumerate(self.ro_chs.items()):
            summed_int = np.add.reduce(d_reps[i_ch], axis=self.avg_level, dtype=np.int64)
            avg = summed_int.astype(np.float64) / float(self.loop_dims[self.avg_level])

            if length_norm and not ro['edge_counting']:
                avg = avg / float(ro['length'])

            if remove_offset:
                off = self._ro_offset_qick(ch, ro.get('ro_config'))
                avg[..., 0] -= off  # I
                avg[..., 1] -= off  # Q

            avg_d.append(np.moveaxis(avg, -2, 0))  # -> (1, steps, 2)
        return avg_d

    # -------------------- public: acquire offline --------------------
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

        I3 = self._ensure_rounds_axis(Ishots, N=self._N_steps, rounds=self._rounds, reps=self._reps)
        Q3 = self._ensure_rounds_axis(Qshots, N=self._N_steps, rounds=self._rounds, reps=self._reps)

        # accumulate per-round like QICK does, using the same averaging kernel
        summed = None
        for r in range(self._rounds):
            # pack this round like acc_buf: (reps, steps, 1, 2)# Normalize raw ADC counts to QICK's floating-point "a.u." scale
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
