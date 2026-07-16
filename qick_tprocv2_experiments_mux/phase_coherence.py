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
import glob
import matplotlib.dates as mdates
import re
import datetime
import ast
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

class PhaseCoherencePlots:
    def __init__(self, outerFolder_save_plots, run_number, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs,
                 fit_saved, signal, run_name, fridge):
        self.outerFolder_save_plots = outerFolder_save_plots
        self.run_number = run_number
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

    def fit_flat_model(self, t, signal, sigma=None):
        """
        Fits the flat model y(t) = d to the data.

        Automatically handles weighted (chi^2) or unweighted (RSS) cases.

        Parameters
        ----------
        t : array-like
            Time values (not used directly but kept for symmetry with other fits)
        signal : array-like
            Measured signal values
        sigma : array-like or None
            Uncertainties. If provided, a weighted chi^2 fit is used.

        Returns
        -------
        flat_obj_val : float
            Minimization objective value (chi^2 if weighted, RSS if unweighted)

        d_fit : float
            Best-fit constant value
        """

        t = np.asarray(t, float)
        signal = np.asarray(signal, float)

        if sigma is not None:
            sigma = np.asarray(sigma, float)

            def flat_obj(d):
                model = np.full_like(signal, d)
                r = (signal - model) / sigma
                return np.sum(r * r)

        else:

            def flat_obj(d):
                model = np.full_like(signal, d)
                r = signal - model
                return np.sum(r * r)

        m = Minuit(flat_obj, d=np.median(signal))
        m.errordef = Minuit.LEAST_SQUARES
        m.migrad()

        flat_obj_val = float(m.fval)
        d_fit = float(m.values["d"])

        return flat_obj_val, d_fit

        
    def exp_vs_ramsey_bic(self, delay_times, y, fitted, k_fit=6, k_exp=3, threshold=15):
        """
        Compare a Ramsey fit against a non-oscillatory exponential baseline using BIC.

        Parameters
        ----------
        delay_times : array-like
            Time axis of the measurement.
        y : array-like
            Raw data (I or Q trace).
        fitted : array-like
            Ramsey model fit to the same data.
        k_fit : int
            Number of parameters in the Ramsey model (default = 6).
        k_exp : int
            Number of parameters in the exponential baseline (default = 3).
        threshold : float
            Required ΔBIC(exp−Ramsey) to accept Ramsey as truly oscillatory.

        Returns
        -------
        keep_ramsey : bool
            True if Ramsey model is strongly favored over exponential baseline.
        delta_bic_exp : float
            ΔBIC = BIC_exp − BIC_ramsey (positive favors Ramsey).
        """

        y = np.asarray(y, float)
        fitted = np.asarray(fitted, float)
        t = np.asarray(delay_times, float)

        n = len(y)
        if n < 8:
            # not enough points to judge
            return False, np.nan

        # SSE of Ramsey model
        sse_fit = np.sum((y - fitted) ** 2)

        # ---------------- exponential baseline model ----------------
        def exp_baseline(t, c, A, tau):
            return c + A * (1.0 - np.exp(-t / tau))

        # simple initial guesses
        m = max(3, n // 10)
        c0 = np.mean(y[-m:])  # late-time plateau
        A0 = np.mean(y[:m]) - c0  # early - late
        tau0 = 0.2 * (t[-1] - t[0]) if t[-1] > t[0] else 1.0

        try:
            popt_exp, _ = curve_fit(
                exp_baseline, t, y, p0=[c0, A0, tau0], maxfev=10000
            )
            y_exp = exp_baseline(t, *popt_exp)
            sse_exp = np.sum((y - y_exp) ** 2)
        except Exception:
            # if exponential fit fails, don't reject Ramsey on this basis
            return True, np.nan
        # ------------------------------------------------------------

        # Guard against log(0)
        eps = 1e-12
        sse_fit = max(sse_fit, eps)
        sse_exp = max(sse_exp, eps)

        # BIC values
        bic_ramsey = k_fit * np.log(n) + n * np.log(sse_fit / n)
        bic_exp = k_exp * np.log(n) + n * np.log(sse_exp / n)

        delta_bic_exp = bic_exp - bic_ramsey  # positive => Ramsey better than exponential

        keep_ramsey = (delta_bic_exp >= threshold)
        return keep_ramsey, delta_bic_exp

    def flat_vs_ramsey_bic(self, y, fitted, k_fit=6, k0=1, threshold=2):#12
        """
        BIC goodness-of-fit test: flat constant baseline vs Ramsey shape.

        Returns
        -------
        keep_ramsey : bool
            True if Ramsey is favored over flat baseline by at least `threshold`.
        delta_bic : float
            ΔBIC = BIC_flat − BIC_ramsey (positive means Ramsey is better).
        """
        y = np.asarray(y, float)
        fitted = np.asarray(fitted, float)

        n = len(y)  # number of datapoints
        if n < 3:
            return False, np.nan

        # SSE of fitted model (sum of squared errors). We want the residuals to be small (so SSE small)
        sse_fit = np.sum((y - fitted) ** 2)

        # SSE of flat constant baseline model
        # So, we do the same but now considering a “no oscillation” baseline model
        y0 = np.mean(y)
        sse0 = np.sum((y - y0) ** 2)

        # Guard against log(0), since BIC contains log(SSE/n).
        eps = 1e-12
        sse_fit = max(sse_fit, eps)
        sse0 = max(sse0, eps)

        # BIC values (using BIC formula)
        bic_fit = k_fit * np.log(n) + n * np.log(sse_fit / n)  # BIC of ramsey model
        bic0 = k0 * np.log(n) + n * np.log(sse0 / n)  # BIC of a constant baseline model

        delta_bic = bic0 - bic_fit  # positive means oscillatory model is better. Smaller SSE = better fit = smaller BIC for that model
        # we want a big baseline BIC (so bad BIC) - a small oscillatory BIC (so a good BIC)

        # Decision threshold, change as needed
        keep_ramsey = (delta_bic >= threshold)
        return keep_ramsey, delta_bic

    def run(self,return_errs=False, t1_vals = None):
        import datetime
        # ----------Load/get data------------------------

        t1_vals = {i: [] for i in range(self.number_of_qubits)}
        t1_errs = {i: [] for i in range(self.number_of_qubits)}
        I_per_pt_errs = {i: [] for i in range(self.number_of_qubits)} # to store the errors of each point in the I-T1 curve
        Q_per_pt_errs = {i: [] for i in range(self.number_of_qubits)} # to store the errors of each point in the Q-T1 curve
        date_times = {i: [] for i in range(self.number_of_qubits)}

        

        t2r_vals = {i: [] for i in range(self.number_of_qubits)}
        t2r_errs = {i: [] for i in range(self.number_of_qubits)}
        rounds_r = []
        reps_r = []
        file_names = []
        date_times_r = {i: [] for i in range(self.number_of_qubits)}
        mean_values_r = {}

        t2e_vals = {i: [] for i in range(self.number_of_qubits)}
        t2e_errs = {i: [] for i in range(self.number_of_qubits)}
        rounds_e = []
        reps_e = []
        date_times_e = {i: [] for i in range(self.number_of_qubits)}
        mean_values_e= {}
        

        
        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"/exp/cosmiq/data/QUIET/QICK_data/{self.run_name}/" + folder_date + "/study_data/"
                        # f"/exp/cosmiq/data/QUIET/QICK_data/{self.run_name}/" + folder_date + "/study_data/" # CEPH
                        # f"/data/QICK_data/{self.run_name}/" + folder_date + "/study_data/" #daq01
                #print('Looking inside: ', outerFolder)
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # -------------------------------------------------------Load/Plot/Save T2------------------------------------------
            outerFolder_expt = outerFolder + "/Data_h5/t1_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt_r, "*.h5"))
            outerFolder_expt_r = outerFolder + "/Data_h5/t2_ge/"
            h5_files_r = glob.glob(os.path.join(outerFolder_expt_r, "*.h5"))
            outerFolder_expt_e = outerFolder + "/Data_h5/t2e_ge/"
            h5_files_e = glob.glob(os.path.join(outerFolder_expt_e, "*.h5"))
            

            # for h5_file in h5_files:
            #     save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            #     H5_class_instance = Data_H5(h5_file)
            #     load_data = H5_class_instance.load_from_h5(data_type=f't1{exp_extension}', save_r=int(save_round))
            #     # if '01-27' in outerFolder_expt:
            #     #     print(load_data)
            #     # Define specific days to exclude
            #     exclude_dates = {}
            #         #datetime.date(2025, 1, 26),  # power outage
            #         #datetime.date(2025, 1, 29),  # HEMT Issues
            #         #datetime.date(2025, 1, 30),  # HEMT Issues
            #         #datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress

            #     for q_key in load_data[f't1{exp_extension}']:
            #         # Run 9 patch, accidentally took punched out data for Q4, bad.
            #         if ("AB_paper_data_batch1_25dB_DACatten_noQ5/2026-04-17_00-34-47" in folder_date
            #             and int(q_key) == 3):
            #             print(f"Skipping Q4 data due to punchout in {self.run_name}/{folder_date}")
            #             continue
            #         for dataset in range(len(load_data[f't1{exp_extension}'][q_key].get('Dates', [])[0])):
            #             if 'nan' in str(load_data[f't1{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
            #                 continue
            #             # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
            #             # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
            #             date = datetime.datetime.fromtimestamp(load_data[f't1{exp_extension}'][q_key].get('Dates', [])[0][dataset])

            #             # cutoff when we switched to saving both averaged arrays *and* shots under Ishots/Qshots
            #             cutoff_dt = datetime.datetime(2025, 10, 24, 13, 58, 37)

            #             # Skip processing if the date (as a date object) is in the excluded set
            #             if date.date() in exclude_dates:
            #                 print(f"Skipping data for {date} (excluded date)")
            #                 continue

            #             # run 8 patch to include a dataset with no saved shots
            #             #if folder_date == "2025-10-24_01-41-30":  # don't change for QUIET analysis, make more general in the future though
            #             #    process_shots = False
                            
            #             # --- make per-shot data compatible with per-delay fitting --------------------------------
            #             if process_shots:
            #                 # --- process IQ shots and turn them into IQ arrays (using Arianna's func, not QICK) --------------------------------
            #                 print("Processing shots...")

            #                 # --- load cfg strings from H5 ---
            #                 exp_config_str = load_data[f't1{exp_extension}'][q_key]['Exp Config'][0][dataset].decode()
            #                 syst_config_str = load_data[f't1{exp_extension}'][q_key]['Syst Config'][0][dataset].decode()

            #                 # --- choose which datasets hold the *shots* based on date ---
            #                 if date < cutoff_dt:
            #                     # before 2025-10-24_13-58-37: shots were saved under 'I' and 'Q'
            #                     I_key, Q_key = 'I', 'Q'
            #                 else:
            #                     # on/after the cutoff: shots were saved under 'Ishots' and 'Qshots'
            #                     I_key, Q_key = 'Ishots', 'Qshots'

            #                 # --- raw shots from H5 ---
            #                 Ishots_raw = self.process_h5_data(load_data[f't1{exp_extension}'][q_key][I_key][0][dataset].decode())
            #                 Qshots_raw = self.process_h5_data(load_data[f't1{exp_extension}'][q_key][Q_key][0][dataset].decode())

            #                 # --- path to the soccfg dump (txt file made with save_run_soccfg_params.py) ---
            #                 #if self.run_number == 9: #This works
            #                     #soccfg_dump_path = "/exp/cosmiq/data/QUIET/QICK_data/run9/6transmon/run9_soccfg_params/soccfg_full_dump_2026-04-20_21-30-15_firmware_during_run9.txt"
            #                         #"/data/QICK_data/run9/6transmon/run9_soccfg_params/soccfg_full_dump_2026-04-20_21-30-15_firmware_during_run9.txt" #daq01
            #                 #if self.run_number == 8:  # This works
            #                     #soccfg_dump_path = "/exp/cosmiq/data/QUIET/QICK_data/run8/6transmon/run8_soccfg_params/soccfg_full_dump_2025-11-10_15-14-35_firmware_during_run8_updated.txt"
            #                         # r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\soccfg_full_dump_2025-11-10_15-14-35_firmware_during_run8_updated.txt"
            #                         # "/data/QICK_data/run8/6transmon/run8_soccfg_params/soccfg_full_dump_2025-11-10_15-14-35_firmware_during_run8_updated.txt"
            #                 #elif self.run_number == 6:  # This doesn't work yet (shots need to be processed diff for run 6) but the skeleton is set up
            #                     #soccfg_dump_path = "/data/QICK_data/run6/6transmon/loud2_soccfg_params/soccfg_full_dump_2025-11-04_16-30-54_firmware_during_run6.txt"

            #                 # --- init offline replica (no live soccfg) and set it up from strings + dump ---
            #                 replica = OfflineAcquireReplica(remove_offset=True, length_norm=True, edge_counting=False)
            #                 replica.setup_offline_from_strings(
            #                     exp_config_str,
            #                     syst_config_str,
            #                     soccfg_dump_path,
            #                     qubit_index=int(q_key))

            #                 exp_cfg = replica._safe_eval_cfg(exp_config_str)
            #                 syst_cfg = replica._safe_eval_cfg(syst_config_str)

            #                 # Pull steps/reps from Syst Config first; fall back to Exp Config only if missing. Sys config is the updated one in each measurement during RR
            #                 steps = int(syst_cfg.get('steps', exp_cfg[f'T1{exp_extension}']['steps']))
            #                 reps = int(syst_cfg.get('reps', exp_cfg[f'T1{exp_extension}']['reps']))
            #                 # rounds not needed here; H5 holds one round

            #                 # --- coerce raw shots to (rounds, N, reps) before averaging ---
            #                 Ishots = replica.coerce_to_rounds_N_reps(Ishots_raw, steps, reps)
            #                 Qshots = replica.coerce_to_rounds_N_reps(Qshots_raw, steps, reps)

            #                 # --- acquire (software average over a single round) ---
            #                 I, Q, I_errs, Q_errs = replica.acquire_offline(Ishots, Qshots, soft_avgs=1, per_pt_errs = self.per_pt_errs)
            #             # -----------------------------------------------------------------------------------------------
            #             else:
            #                 I = self.process_h5_data(load_data[f't1{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
            #                 Q = self.process_h5_data(load_data[f't1{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
            #                 I_errs = None
            #                 Q_errs = None

            #             delay_times = self.process_h5_data(load_data[f't1{exp_extension}'][q_key].get('Delay Times', [])[0][dataset].decode())
            #             # fit = load_data['T1'][q_key].get('Fit', [])[0][dataset]
            #             round_num = load_data[f't1{exp_extension}'][q_key].get('Round Num', [])[0][dataset]

            #             # try:
            #             #     batch_num = load_data[f't1{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]
            #             #     syst_config = load_data[f't1{exp_extension}'][q_key].get('Syst Config', [])[0][dataset].decode()
            #             #     exp_config = load_data[f't1{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
            #             #     safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
            #             #     exp_config = eval(exp_config, safe_globals)
            #             # except:
            #             #     exp_config =None

            #             if len(I) > 0:
            #                 T1_class_instance = T1Measurement(q_key, self.number_of_qubits, self.outerFolder_save_plots, round_num, self.signal, self.save_figs,
            #                                                   fit_data=True)
            #                 #T1_spec_cfg = exp_config['T1_ge']
            #                 q1_fit_exponential, T1_err, T1_est, fit_info = T1_class_instance.t1_fit_iminuit(I, Q, delay_times)
            #                 if T1_est < 0:
            #                     print("The value is negative, continuing...")
            #                     continue

            #                 if T1_est > 600:
            #                     print("The value is above 600 us, this is a bad fit, continuing...")
            #                     continue

            #                 # ------------------------ Quality cut: flat BIC vs exponential BIC test---------------------
            #                 t = fit_info["t"]
            #                 signal = fit_info["signal"]
            #                 sigma = fit_info["sigma"]

            #                 flat_obj_val, d_flat = self.fit_flat_model(t, signal, sigma)

            #                 k_flat = 1
            #                 n = len(t)

            #                 # determine correct BIC formula
            #                 if sigma is not None:
            #                     # weighted case (objective = chi^2)
            #                     bic_flat = flat_obj_val + k_flat * np.log(n)
            #                 else:
            #                     # unweighted case (objective = RSS)
            #                     # Smaller RSS = better fit = smaller BIC for that model
            #                     bic_flat = n * np.log(flat_obj_val / n) + k_flat * np.log(n)

            #                 bic_exp = fit_info["bic_score"]

            #                 # if BIC score is larger than zero -> exponential is better! (good T1 curve)
            #                 # if BIC score is less than zero -> data looks flat
            #                 delta_bic = bic_flat - bic_exp

            #                 if delta_bic < 35:
            #                     # I verified this BIC score for runs 4-9 and it worked well for ALL of them! No bad fits left.
            #                     continue
            #                 #---------------------------------------------------------------------------------------------

            #                 ## To look at T1 plots of data that made it through you can uncomment this:
            #                 ## and remember to set save figs to true in analysis master script
            #                 #T1_class_instance.plot_results(I, Q, delay_times, folder_date, iminuit_fit_instead=True)

            #                 # if T1_err >= 0.8 * T1_est:
            #                 #     print(
            #                 #         f"Skipping T1 = {T1_est:.3f} µs because its error {T1_err:.3f} µs is >= 80% of its value.")
            #                 #     continue

            #                 t1_vals[q_key].extend([T1_est])
            #                 t1_errs[q_key].extend([T1_err])

            #                 # --- store per-point errors too, only if we had process_shots ---
            #                 if process_shots:
            #                     I_per_pt_errs[int(q_key)].append(I_errs)
            #                     Q_per_pt_errs[int(q_key)].append(Q_errs)

            #                 if use_png_timestamps:
            #                     # --- use PNG filename timestamp from mapping if available ------
            #                     # the reason for this is bc the png timestamp is more accurate than the h5 file ones
            #                     if mapping_data is not None:
            #                         # mapping uses experiment='t1_ge', qubit as 1-indexed
            #                         qubit_in_map = q_key + 1
            #                         subset = map_loader.filter_by(
            #                             mapping_data,
            #                             experiment=f"t1{exp_extension}",
            #                             qubit=qubit_in_map,
            #                             round=round_num)

            #                         if len(subset) > 0:
            #                             png_ts = subset[0]["png_timestamp"].decode()
            #                             try:
            #                                 png_dt = datetime.datetime.strptime(png_ts, "%Y-%m-%d_%H-%M-%S")
            #                                 date_str = png_dt.strftime("%Y-%m-%d %H:%M:%S") # from png file
            #                             except Exception:
            #                                 # in case of weird format, fall back
            #                                 # date_str = date.strftime("%Y-%m-%d %H:%M:%S") # from h5 file
            #                                 continue # skip
            #                         else:
            #                             # no mapping match for this qubit/round, fall back
            #                             # date_str = date.strftime("%Y-%m-%d %H:%M:%S") # from h5 file
            #                             continue # skip
            #                     else:
            #                         # no mapping file for this timestamp_dir, fall back
            #                         # date_str = date.strftime("%Y-%m-%d %H:%M:%S") # from h5 file
            #                         continue # skip
            #                     date_times[q_key].append(date_str)

            #                 else:
            #                     date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])  # og way, from h5 file

            #                 del T1_class_instance

            #     del H5_class_instance
            
            
            for h5_file in h5_files_r:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type='t2_ge', save_r=int(save_round))

                populated_keys = []
                for q_key in load_data['t2_ge']:
                    # Run 9 patch, accidentally took punched out data for Q4, bad.
                    if ("AB_paper_data_batch1_25dB_DACatten_noQ5/2026-04-17_00-34-47" in folder_date
                            and int(q_key) == 3):
                        print(f"Skipping Q4 data due to punchout in {self.run_name}/{folder_date}")
                        continue
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
                        # T2 = load_data['T2'][q_key].get('T2', [])[0][dataset]
                        # errors = load_data['T2'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data['t2_ge'][q_key].get('Dates', [])[0][dataset])
                        I = self.process_h5_data(load_data['t2_ge'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data['t2_ge'][q_key].get('Q', [])[0][dataset].decode())
                        delay_times = self.process_h5_data(
                            load_data['t2_ge'][q_key].get('Delay Times', [])[0][dataset].decode())
                        # fit = load_data['T2'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data['t2_ge'][q_key].get('Round Num', [])[0][dataset]
                        batch_num = load_data['t2_ge'][q_key].get('Batch Num', [])[0][dataset]

                        #exp_config = load_data['t2_ge'][q_key].get('Exp Config', [])[0][dataset].decode()
                        safe_globals = {"np": np, "array": np.array, "__builtins__": {}}

                        #exp_config = eval(exp_config, safe_globals)

                        if len(I) > 0:
                            T2_class_instance = T2RMeasurement(q_key, self.number_of_qubits,
                                                               self.outerFolder_save_plots, round_num_r, self.signal,
                                                               self.save_figs, fit_data=True)
                            try:
                                fitted, t2r_est, t2r_err, plot_sig, out = T2_class_instance.t2_fit_iminuit(delay_times, I, Q, make_plots = False)
                            except Exception as e:
                                print('Fit didnt work due to error: ', e)
                                continue
                            # T2_cfg = exp_config['Ramsey_ge']

                            # --------- simple peak-count gate on the fitted curve ----------
                            try:
                                min_peaks = 2
                                y_fit = np.asarray(fitted, float)
                                t = np.asarray(delay_times, float)

                                dt = np.median(np.diff(t))
                                f_fit = abs(out["f"][0])  # cycles per microsecond if t is in us

                                # If frequency is tiny, you can't reliably peak-count anyway
                                if f_fit < 1e-6:
                                    n_osc = 0
                                else:
                                    period_samp = max(3, int(round(1.0 / (f_fit * dt))))
                                    min_dist = max(3, period_samp // 2)  # peaks at least half-period apart

                                    pks, _ = find_peaks(y_fit, distance=min_dist)
                                    trs, _ = find_peaks(-y_fit, distance=min_dist)
                                    n_osc = min(len(pks), len(trs))

                                if n_osc < min_peaks:
                                    print(f'Rejected a T2R scan. Failed ramsey shape, less than {min_peaks} oscillations.')
                                    continue
                            except Exception:
                                # if peak counting fails for any reason, be conservative and skip
                                continue

                            # -------------------- flat baseline vs Ramsey shape BIC test -------------------------------
                            # Works like this: keep_ramsey = (delta_bic >= threshold)
                            y = I if plot_sig == "I" else Q

                            if self.run_number == 9:
                                LIN_Rams_BIC_thresh = 100
                            else: #runs 4-8
                                LIN_Rams_BIC_thresh = 3 # 35 threshold is good for runs 4-8

                            keep_ramsey, delta_bic = self.flat_vs_ramsey_bic(y, fitted, k_fit=6, k0=1, threshold=LIN_Rams_BIC_thresh)
                            if not keep_ramsey:
                                print(f"Rejected by BIC: ΔBIC(line-Ramsey) = {delta_bic:.2f}")
                                #T2_class_instance.t2_fit_iminuit(delay_times, I, Q, make_plots=True,title_ext=f"ΔBIC(line-Ramsey) = {delta_bic:.2f}")
                                continue

                            # ---------------- Exponential vs Ramsey BIC test ----------------
                            # It works by checking: keep_ramsey = (delta_bic_exp >= threshold)
                            y = I if plot_sig == "I" else Q

                            if self.run_number == 9:
                                EXP_Rams_BIC_thresh = 100
                            else: #runs 4-8
                                EXP_Rams_BIC_thresh = 1 # 10 threshold is good for runs 4-8

                            keep_ramsey, delta_bic_exp = self.exp_vs_ramsey_bic(delay_times, y, fitted, k_fit=6, k_exp=3, threshold=EXP_Rams_BIC_thresh)
                            if not keep_ramsey:
                                print(f"Rejected by exp-BIC: ΔBIC(exp−Ramsey) = {delta_bic_exp:.2f}")
                                #T2_class_instance.t2_fit_iminuit(delay_times, I, Q, make_plots=True,title_ext=f"ΔBIC(exp−Ramsey) = {delta_bic_exp:.2f}")
                                continue
                            # -------------------- Other cuts-----------------------
                            if t2r_est < 0:
                                print("The value is negative, continuing...")
                                continue

                            if t1_vals is not None:
                                max_t1 = max(t1_vals[q_key])  # theoretical value
                                if t2r_est > 2 * max_t1:
                                    print(f"The value is above 2*{max_t1} us, this is a bad fit, continuing...")
                                    continue

                            # -------------------- Normalized Root Mean Square Error cut ---------------------
                            # You want to keep data below the threshold. We expect good fits to have small residuals.
                            # small NRMSE = good fit = keep, and large NRMSE = poor fit = reject

                            if self.run_number == 8 or self.run_number == 7: # Specific to QUIET
                                nrmse_threshold = 3.224 #0.1224
                            elif self.run_number == 6:
                                nrmse_threshold = 0.1065
                            elif self.run_number == 5:
                                nrmse_threshold = 0.1319
                            elif self.run_number == 4:
                                nrmse_threshold = 0.15 # all run 4 plots were good so threshold here is just a dummy that doesn't filter anything
                            elif self.run_number == 9:
                                nrmse_threshold = 0.079
                            else: # default for runs with great plots / no major issues
                                nrmse_threshold = 0.15

                            nrmse_score = out["nrmse"]
                            if nrmse_score> nrmse_threshold:
                                print(f"Rejected due to NRMSE cut. Value was above threshold of {nrmse_threshold}")
                                #T2_class_instance.t2_fit_iminuit(delay_times, I, Q, make_plots=True,title_ext=f"NRMSE:{nrmse_score:.4f}")
                                continue

                            # If you want to plot the data that made it through, uncomment this:
                            #T2_class_instance.t2_fit_iminuit(delay_times, I, Q, make_plots = True, title_ext = f"NRMSE:{nrmse_score:.4f}")

                            t2r_vals[q_key].extend([t2r_est])
                            t2r_errs[q_key].extend([t2r_err])
                            date_times_r[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del T2_class_instance

                del H5_class_instance
        if return_errs:
            return date_times_r, t2r_vals, t2r_errs
        else:
            return date_times_r, t2r_vals





        for h5_file in h5_files_e:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type='T2E', save_r=int(save_round))
                populated_keys = []
                for q_key in load_data['T2E']:
                    # Run 9 patch, accidentally took punched out data for Q4, bad.
                    if ("AB_paper_data_batch1_25dB_DACatten_noQ5/2026-04-17_00-34-47" in folder_date
                            and int(q_key) == 3):
                        print(f"Skipping Q4 data due to punchout in {self.run_name}/{folder_date}")
                        continue
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

                        #exp_config = load_data['T2E'][q_key].get('Exp Config', [])[0][dataset].decode()
                        safe_globals = {"np": np, "array": np.array, "__builtins__": {}}

                        # exp_config = eval(exp_config, safe_globals)

                        if len(I) > 0:
                            T2E_class_instance = T2EMeasurement(q_key, self.number_of_qubits,
                                                                self.outerFolder_save_plots, round_num, self.signal,
                                                                self.save_figs, fit_data=True)
                            try:
                                fitted, t2e_est, t2e_err, plot_sig, out = T2E_class_instance.t2_fit_iminuit(delay_times,I, Q, make_plots = False)
                            except Exception as e:
                                print('Fit didnt work due to error: ', e)
                                continue
                            # T2E_cfg = exp_config['SpinEcho_ge']

                            # --------- simple peak-count gate on the fitted curve ----------
                            try:
                                min_peaks = 2
                                y_fit = np.asarray(fitted, float)
                                t = np.asarray(delay_times, float)

                                dt = np.median(np.diff(t))
                                f_fit = abs(out["f"][0])  # cycles per microsecond if t is in us

                                # If frequency is tiny, you can't reliably peak-count anyway
                                if f_fit < 1e-6:
                                    n_osc = 0
                                else:
                                    period_samp = max(3, int(round(1.0 / (f_fit * dt))))
                                    min_dist = max(3, period_samp // 2)  # peaks at least half-period apart

                                    pks, _ = find_peaks(y_fit, distance=min_dist)
                                    trs, _ = find_peaks(-y_fit, distance=min_dist)
                                    n_osc = min(len(pks), len(trs))

                                if n_osc < min_peaks:
                                    print(f'Rejected a T2E scan. Failed T2E shape, less than {min_peaks} oscillations.')
                                    continue
                            except Exception:
                                # if peak counting fails for any reason, be conservative and skip
                                continue

                            # -------------------- flat baseline vs Ramsey shape BIC test -------------------------------
                            y = I if plot_sig == "I" else Q

                            keep_ramsey, delta_bic = self.flat_vs_ramsey_bic(y, fitted, k_fit=6, k0=1, threshold=2) # 35 threshold is good for runs 4-9

                            if not keep_ramsey:
                                print(f"Rejected by flat BIC test: ΔBIC(line-Ramsey) = {delta_bic:.2f}")
                                continue

                            # ---------------- Exponential vs T2E shape BIC test ----------------
                            y = I if plot_sig == "I" else Q

                            keep_ramsey, delta_bic_exp = self.exp_vs_ramsey_bic(
                                delay_times, y, fitted, k_fit=6, k_exp=3, threshold=2) # 10 threshold is good for runs 4-9

                            if not keep_ramsey:
                                print(f"Rejected by exponential BIC test: ΔBIC(exp-Ramsey) = {delta_bic_exp:.2f}")
                                continue

                            # ---------------------- Other cuts ----------------------------
                            if t2e_est < 0:
                                print("The value is negative, continuing...")
                                continue
                            if t1_vals is not None:
                                max_t1 = max(t1_vals[q_key])  # theoretical value
                                if t2e_est > 2 * max_t1:
                                    print(f"The value is above 2*{max_t1} us, this is a bad fit, continuing...")
                                    continue

                            # -------------------- Normalized Root Mean Square Error cut ---------------------
                            # You want to keep data below the threshold. We expect good fits to have small residuals.
                            # small NRMSE = good fit = keep, and large NRMSE = poor fit = reject

                            nrmse_threshold = 5 #0.14 # tested for QUIET runs 4-9 and it worked well for all!
                            nrmse_score = out["nrmse"]
                            if nrmse_score > nrmse_threshold:
                                print(f"Rejected due to NRMSE cut. Value was above threshold of {nrmse_threshold}")
                                continue

                            # If you want to plot the data that made it through, uncomment this:
                            # T2E_class_instance.t2_fit_iminuit(delay_times, I, Q, make_plots = True)

                            t2e_vals[q_key].extend([t2e_est])
                            t2e_errs[q_key].extend([t2e_err])
                            date_times_e[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del T2E_class_instance
                del H5_class_instance
        if return_errs:
            return date_times_e, t2e_vals, t2e_errs
        else:
            return date_times_e, t2e_vals






        
        
    def plot_without_errs(self, date_times, t2_vals, show_legends):
        # ---------------------------------plot-----------------------------------------------------
        analysis_folder = f"{self.outerFolder_save_plots}/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('T2R Values vs Time', fontsize=font)
        axes = axes.flatten()
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        from datetime import datetime
        for i, ax in enumerate(axes):

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]
            y = t2_vals[i]

            # Convert strings to datetime objects.
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            # Combine datetime objects and y values into a list of tuples and sort by datetime.
            combined = list(zip(datetime_objects, y))
            combined.sort(reverse=True, key=lambda x: x[0])

            # Unpack them back into separate lists, in order from latest to most recent.
            sorted_x, sorted_y = zip(*combined)
            ax.scatter(sorted_x, sorted_y, color=colors[i])
            # print(len(sorted_y))
            # print(len(sorted_x))

            sorted_x = np.asarray(sorted(x))

            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

            # Set new x-ticks using the datetime objects at the selected indices
            ax.set_xticks(sorted_x[indices])
            ax.set_xticklabels([dt for dt in sorted_x[indices]], rotation=45)

            ax.scatter(x, y, color=colors[i])
            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel('T2R (us)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2R_vals.pdf', transparent=True, dpi=self.final_figure_quality)

        # plt.close()

    def plot_with_errs(self, date_times_r, t2r_vals, t2r_fit_err, date_times_e, t2e_vals, t2e_fit_err, show_legends):
        # ---------------------------------plot-----------------------------------------------------
        analysis_folder = f"{self.outerFolder_save_plots}/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        colors2 = ['pink', 'orange', 'blue', 'purple', 'green', 'brown']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True, sharex=True)
        plt.suptitle('T2R Values vs Time', fontsize=font)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            x = date_times_r[i]
            y = t2r_vals[i]
            err = t2r_fit_err[i]

            x2 = date_times_e[i]
            y2 = t2e_vals[i]
            err2 = t2e_fit_err[i]

            datetime_objects = [datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])
            if len(combined) == 0:
                ax.set_visible(False)
                continue
            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)

            combined2 = list(zip(datetime_objects, y2, err2))
            combined2.sort(key=lambda tup: tup[0])
            if len(combined2) == 0:
                ax.set_visible(False)
                continue
            sorted_x2, sorted_y2, sorted_err2 = zip(*combined2)
            sorted_x2 = np.array(sorted_x2)
            
            # ax.set_ylim(7, 95)

            ax.errorbar(
                sorted_x, sorted_y, yerr=sorted_err,
                fmt='o',
                markersize=3,
                label='Ramsey',
                color=colors[i],
                ecolor=colors[i],
                elinewidth=1,
                capsize=0,
                alpha=0.5
            )

            ax.errorbar(
                sorted_x2, sorted_y2, yerr=sorted_err2,
                fmt='x',
                markersize=3,
                label='Echo',
                color=colors2[i],
                ecolor=colors2[i],
                elinewidth=1,
                capsize=0,
                alpha=0.5
            )
            

            #ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.tick_params(axis='x', rotation=45)

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel('T2 (R & E) (us)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2_vals.pdf', transparent=True, dpi=self.final_figure_quality)
        print('Plot saved to:', analysis_folder)
        plt.show()
        plt.close()

    def plot_with_errs2(self, date_times_r, t1_vals, t1_errs, date_times, t2r_vals, t2r_fit_err, date_times_e, t2e_vals, t2e_fit_err, show_legends):
        # ---------------------------------plot-----------------------------------------------------                                                                                              
        analysis_folder = f"{self.outerFolder_save_plots}/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True, sharex=True)
        plt.suptitle('T2R Values vs Time', fontsize=font)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            xt = date_times[i]
            yt = t1_vals[i]
            terr = t1_errs[i]
            
            x = date_times_r[i]
            y = t2r_vals[i]
            err = t2r_fit_err[i]

            x2 = date_times_e[i]
            y2 = t2e_vals[i]
            err2 = t2e_fit_err[i]


            datetime_objects_t = [datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in xt]
            datetime_objects = [datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]
            datetime_objects2 = [datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x2]

            
            combined_t = list(zip(datetime_objects_t, yt, terr))
            combined_t.sort(key=lambda tup: tup[0])
            if len(combined_t) == 0:
                ax.set_visible(False)
                continue
            sorted_xt, sorted_yt, sorted_terr = zip(*combined_t)
            sorted_xt = np.array(sorted_xt)
            
            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])
            if len(combined) == 0:
                ax.set_visible(False)
                continue
            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)

            combined2 = list(zip(datetime_objects2, y2, err2))
            combined2.sort(key=lambda tup: tup[0])
            if len(combined2) == 0:
                ax.set_visible(False)
                continue
            sorted_x2, sorted_y2, sorted_err2 = zip(*combined2)
            sorted_x2 = np.array(sorted_x2)

            ##Caluclating Pure Dephasing

            t2_prop = np.array(sorted_y) / np.array(sorted_y2[:252])
            pure_dephase = 1 - t2_prop

            ## Propagate Error into pure Dephasing
            t2r_rel_uncer = np.array(sorted_err) / np.array(sorted_y)
            t2e_rel_uncer = np.array(sorted_err2) / np.array(sorted_y2)

            pure_dephase_uncert = np.abs(pure_dephase * np.sqrt((t2r_rel_uncer)**2 + (t2e_rel_uncer[:252])**2))
            
            ## We'll ignore the logistics of "time" for now


            ax.set_ylim(-0.1, 1.1)                                                                                                                                                                  
            
            
            ax.errorbar(
                sorted_yt, pure_dephase, xerr=sorted_terr, yerr=pure_dephase_uncert,
                fmt='o',
                markersize=3,
                label='Dephase',
                color=colors[i],
                ecolor=colors[i],
                elinewidth=1,
                capsize=0,
                alpha=0.5
            )

            ##Run 8
            #if i==1: 
                #ax.axvline(x=datetime.datetime(2025, 10, 27, 22, 0, 0), color='r', linestyle='--', label='Reference Line')
            #elif i==2:
                #ax.axvline(x=datetime.datetime(2025,10, 23, 00, 0, 0), color='r', linestyle='--', label='Reference Line')
                #ax.axvline(x=datetime.datetime(2025,10, 27, 22, 0, 0), color='r', linestyle='--', label='Reference Line')
                #ax.axvline(x=datetime.datetime(2025,10, 29, 00, 0, 0), color='r', linestyle='--', label='Reference Line')
            #elif i==3:
                #ax.axvline(x=datetime.datetime(2025, 10, 23, 1, 0, 0), color='r', linestyle='--', label='Reference Line')
            #elif i==5:
                #ax.axvline(x=datetime.datetime(2025, 10, 27, 14, 0, 0), color='r', linestyle='--', label='Reference Line')
                #ax.axvline(x=datetime.datetime(2025, 10, 31, 12, 0, 0), color='r', linestyle='--', label='Reference Line')
                #ax.axvline(x=datetime.datetime(2025, 11, 1, 00, 0, 0), color='r', linestyle='--', label='Reference Line')
                
            #ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            #ax.tick_params(axis='x', rotation=45)

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('T1', fontsize=font - 2)
            ax.set_ylabel('Pure Dephasing Value', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Dephase_Values.pdf', transparent=True, dpi=self.final_figure_quality)
        print('Plot saved to:', analysis_folder)
        plt.show()
        plt.close()

            

    def plot_with_errs_single_plot(self, date_times, t2_vals, t2_fit_err, show_legends):
        analysis_folder = f"{self.outerFolder_save_plots}/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)
        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('T2R Values vs Time', fontsize=font)
        from datetime import datetime
        import matplotlib.dates as mdates
        for i in range(self.number_of_qubits):
            x = date_times[i]
            y = t2_vals[i]
            err = t2_fit_err[i]
            datetime_objects = [datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]
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
        if show_legends:
            ax.legend(edgecolor='black')
        ax.set_xlabel('Time', fontsize=font - 2)
        ax.set_ylabel('T2R (us)', fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2R_vals_single_plot.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close()




