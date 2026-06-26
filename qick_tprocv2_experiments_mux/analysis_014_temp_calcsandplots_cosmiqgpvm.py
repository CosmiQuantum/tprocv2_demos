from bisect import bisect_left
import re
import ast
import numpy as np
import sys
import h5py
import pickle
from mpl_toolkits.mplot3d import proj3d
from matplotlib.ticker import FormatStrFormatter
from matplotlib.axes import Axes
from sklearn.mixture import GaussianMixture
from scipy.optimize import least_squares
import matplotlib.ticker as mticker
from scipy.stats import norm
import os
sys.path.insert(0, os.path.abspath("/home/quietuser/Documents/GitHub/QICK_Qubit_LabSuite/src"))
from qicklab.analysis.qspec import AnaQSpec
from qicklab.analysis.ssf import AnaSSF
from Arianna_non_prebuilt_SSF_doublegauss_funcs import non_prebuilt_ssf_analysis_class
from analysis_002_res_centers_vs_time_plots import ResonatorFreqVsTime
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime
from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_007_T2R_vs_time_plots import T2rVsTime
from analysis_008_T2E_vs_time_plots import T2eVsTime
from matplotlib.ticker import MaxNLocator
from qicklab.datahandling.datafile_tools import find_h5_files
from analysis_021_plot_allRR_noqick import PlotRR_noQick
import math
from matplotlib.lines import Line2D
import datetime
import pandas as pd
from pathlib import Path
from iminuit import Minuit
import matplotlib.dates as mdates
from bisect import bisect_left
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import matplotlib.animation as animation

save_figs = True
figure_quality = 100 #ramp this up to like 500 for presentation plots

class SSFTempCalcAndPlots:
    def __init__(self, figure_quality, number_of_qubits, run_num, save_figs):
        self.save_figs = save_figs
        self.figure_quality = figure_quality
        self.number_of_qubits = number_of_qubits
        self.run_num = run_num

    def calculate_qubit_temperature(self, frequency_mhz, ground_state_population, excited_state_population):
        k_B = 1.380649e-23  # Boltzmann constant in J/K
        h = 6.62607015e-34  # Planck's constant in J·s
        frequency_hz = frequency_mhz * 1e6
        #T = (h * frequency_hz) / (k_B * np.log(ground_state_population / excited_state_population))
        # Check for invalid populations
        if excited_state_population <= 0 or ground_state_population <= 0: #if one of them is zero can't calculate the temp
            print("Warning: Invalid population values encountered (<= 0). Skipping this dataset.")
            return None

        ratio = ground_state_population / excited_state_population
        if ratio <= 1: #denominator would become zero at Pg=Pe
            print(f"Warning: Non-physical ratio (P_g/P_e = {ratio:.3f} <= 1) encountered. Skipping this dataset.")
            return None

        # If valid, calculate the temperature
        T = (h * frequency_hz) / (k_B * np.log(ratio))
        return T


    def fit_double_gaussian_with_full_coverage(self, iq_data): #iq_data is ig_new or ie_new (IQ data post-rotation)
        gmm = GaussianMixture(n_components=2)
        gmm.fit(iq_data.reshape(-1, 1))

        means = gmm.means_.flatten()
        sigmas = np.sqrt(gmm.covariances_).flatten()
        weights = gmm.weights_

        ground_gaussian = np.argmin(means)
        excited_gaussian = 1 - ground_gaussian

        # Generate x values to approximate the crossing point
        x_vals = np.linspace(means[ground_gaussian] - 3 * sigmas[ground_gaussian],
                             means[excited_gaussian] + 3 * sigmas[excited_gaussian], 1000)

        # Calculate Gaussian fits for each x value
        ground_gaussian_fit = weights[ground_gaussian] * (1 / (np.sqrt(2 * np.pi) * sigmas[ground_gaussian])) * np.exp(
            -0.5 * ((x_vals - means[ground_gaussian]) / sigmas[ground_gaussian]) ** 2)
        excited_gaussian_fit = weights[excited_gaussian] * (
                    1 / (np.sqrt(2 * np.pi) * sigmas[excited_gaussian])) * np.exp(
            -0.5 * ((x_vals - means[excited_gaussian]) / sigmas[excited_gaussian]) ** 2)

        # Find the x value where the two Gaussian functions are closest
        crossing_point = x_vals[np.argmin(np.abs(ground_gaussian_fit - excited_gaussian_fit))]

        labels = gmm.predict(iq_data.reshape(-1, 1))

        ground_data = iq_data[(labels == ground_gaussian) & (iq_data < crossing_point)]
        excited_data = iq_data[(labels == excited_gaussian) & (iq_data > crossing_point)]

        ground_state_population = len(ground_data) / len(iq_data)
        excited_state_population_leakage = len(excited_data) / len(iq_data)

        return ground_state_population, excited_state_population_leakage, gmm, means, sigmas, weights, crossing_point, ground_gaussian, excited_gaussian, ground_data, excited_data, iq_data


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

    def run_ssf_qtemps(self, pairs_info, limit_temp_k=0.8, use_gessf_thresh_only: bool = False, fallback_to_threshold: bool = False):
        """
        Parameters
        ----------
        pairs_info : dict
            {qubit: [ {"qspec":..., "ssf":..., "qfreq_MHz":<MHz>,
                        "ig_new":<np.ndarray>, "ie_new":<np.ndarray>, "data_timestamp":<unix-time> }, … ]}
        limit_temp_k : float
        Discard temperatures above this value (default 0.8 K → 800 mK).
        use_gessf_thresh_only : bool
        If True, *only* use each file’s g-e SSF threshold to split P_g/P_e. This threshold is calculated using
        the function ssf_fit_two_gaussians_midpoint.
        fallback_to_threshold : bool
        If True, first attempt the g-state double‐Gaussian fit; on failure or bad‐fit
        fall back to each file’s g-e SSF threshold.

        Returns
        -------
        all_qubit_temperatures : dict {qubit: [temp_mK, …]}
        all_qubit_timestamps   : dict {qubit: [datetime, …]}
        all_qubit_temperatures_errs : dict {qubit: [temp_mK_error, …]}

        fit_results : dict
        { qubit_index: [
          {
            "dataset": <int>,
            "timestamp": <datetime>,
            "temperature_mK": <float>,
            "ig_new": np.ndarray,
            "ground_data": np.ndarray,
            "excited_data": np.ndarray,
            "ground_gaussian": <int>,
            "excited_gaussian": <int>,
            "pop_threshold": <float>,
            "weights": np.ndarray(shape=(2,)),
            "sigmas": np.ndarray(shape=(2,)), #sigma of each gaussian in the double gaussian fit
            "total_sigma_Pe": sigma_Pe, # total 1‐σ uncertainty on Pe
            "means": np.ndarray(shape=(2,)),
            "Pg": Pg,
            "Pe": Pe,
            "qfreq_mhz": freq_mhz, # qubit frequency
            "qfreq_mhz_err": freq_mhz_err,
          }]}
        """
        # initialise output arrays
        all_qubit_temperatures = {i: [] for i in range(self.number_of_qubits)}
        all_qubit_temperatures_errs = {i: [] for i in range(self.number_of_qubits)}
        all_qubit_timestamps = {i: [] for i in range(self.number_of_qubits)}
        fit_results = {qid: [] for qid in range(self.number_of_qubits)}

        for qid, records in pairs_info.items():  # loop over qubits
            for idx, rec in enumerate(records):  # …and every pair
                # Tracker for what happens
                used_fallback = False

                freq_mhz = rec["qfreq_MHz"]
                freq_mhz_err = rec["qfreq_MHz_err"]
                ig_new = rec["ig_new"]
                ie_new = rec["ie_new"]
                ts_unix = rec["data_timestamp"]
                ssf_fid = rec["ssf_fid"]

                # Decide which threshold approach to use
                if use_gessf_thresh_only:
                    # ----------Calculate g-e threshold for each ssf file ---------------------
                    ge_thresh, ge_thresh_err, means, sigmas, weights, ground_idx, excited_idx = self.ssf_fit_two_gaussians_midpoint(ig_new, ie_new) #fits two states' data to a double gaussian

                    #--------------- use g-e SSF threshold to calculate Pg and Pe ---------------
                    mask = (ig_new <= ge_thresh)
                    Pg = mask.mean()
                    Pe = 1.0 - Pg
                    pop_threshold = ge_thresh

                    # -- 1-σ contribution to Pe from the threshold uncertainty --
                    mask_plus = (ig_new <= ge_thresh + ge_thresh_err)
                    Pe_plus = 1.0 - mask_plus.mean()

                    mask_minus = (ig_new <= ge_thresh - ge_thresh_err)
                    Pe_minus = 1.0 - mask_minus.mean()

                    sigma_Pe_from_thresh = 0.5 * abs(Pe_plus - Pe_minus)

                    # statistical err of Pe
                    Nshots = ig_new.size
                    sigma_Pe_stat = np.sqrt(Pe * (1 - Pe) / Nshots)

                    # -- total 1‐σ uncertainty on Pe--
                    sigma_Pe = np.sqrt(sigma_Pe_from_thresh ** 2 + sigma_Pe_stat ** 2)

                    # We don't care about these for this method, the user can check plots using function plot_ssf_ge_thresh if needed
                    ground_gaussian = ground_idx
                    excited_gaussian = excited_idx
                    ground_data = excited_data = None

                elif fallback_to_threshold:
                    # -------- double-Gaussian fit on ground state data, with fallback method --------------------------
                    try:
                        (Pg, Pe, gmm, means, sigmas, weights, threshold_mid, threshold_mid_err, ground_gaussian, excited_gaussian,
                         ground_data, excited_data, _) = self.fit_double_gaussian_midpoint(ig_new) #fits only 1 state's data to a double gaussian

                        pop_threshold = threshold_mid

                        # -- 1-σ contribution to Pe from the threshold uncertainty --
                        mask_plus = (ig_new <= threshold_mid + threshold_mid_err)
                        Pe_plus = 1.0 - mask_plus.mean()

                        mask_minus = (ig_new <= threshold_mid - threshold_mid_err)
                        Pe_minus = 1.0 - mask_minus.mean()

                        sigma_Pe_from_thresh = 0.5 * abs(Pe_plus - Pe_minus)

                        # statistical err of Pe
                        Nshots = ig_new.size
                        sigma_Pe_stat = np.sqrt(Pe * (1 - Pe) / Nshots)

                        # -- total 1‐σ uncertainty on Pe--
                        sigma_Pe = np.sqrt(sigma_Pe_from_thresh ** 2 + sigma_Pe_stat ** 2)

                        # Ensure crossing point (where threshold is set) isn’t too close to the ground histogram mean
                        mu_g = means[ground_gaussian]
                        sigma_g = np.sqrt(sigmas[ground_gaussian])
                        n_sigma = 1.5
                        if (pop_threshold - mu_g) <= n_sigma * sigma_g:
                            raise ValueError("Crossing point too close to ground mean. Probably incorrect fitting, switching to fallback method.")

                    except Exception: # Use fallback method: using g-e SSF threshold to calculate Pg and Pe
                        # ----------Calculate g-e threshold for each ssf file ---------------------
                        ge_thresh, ge_thresh_err, means, sigmas, weights, ground_idx, excited_idx = self.ssf_fit_two_gaussians_midpoint(ig_new, ie_new)
                        print(f"[run] Q{qid + 1} dataset {idx}: GMM fit failed or too close crossing. Falling back to g-e SSF threshold")
                        pop_threshold = ge_thresh
                        mask = (ig_new <= ge_thresh)
                        Pg = mask.mean()
                        Pe = 1.0 - Pg

                        # -- 1-σ contribution to Pe from the threshold uncertainty --
                        mask_plus = (ig_new <= ge_thresh + ge_thresh_err)
                        Pe_plus = 1.0 - mask_plus.mean()

                        mask_minus = (ig_new <= ge_thresh - ge_thresh_err)
                        Pe_minus = 1.0 - mask_minus.mean()

                        sigma_Pe_from_thresh = 0.5 * abs(Pe_plus - Pe_minus)

                        # statistical err of Pe
                        Nshots = ig_new.size
                        sigma_Pe_stat = np.sqrt(Pe * (1 - Pe) / Nshots)

                        # -- total 1‐σ uncertainty on Pe--
                        sigma_Pe = np.sqrt(sigma_Pe_from_thresh ** 2 + sigma_Pe_stat ** 2)

                        # We don't care about these for this method, the user can check plots using function plot_ssf_ge_thresh if needed
                        ground_gaussian = ground_idx
                        excited_gaussian = excited_idx
                        ground_data = excited_data = None
                        used_fallback = True

                else:
                    # -------- Only using double-Gaussian fit on ground state data, without fallback method --------------------------
                    (Pg, Pe, gmm, means, sigmas, weights, threshold_mid, threshold_mid_err, ground_gaussian, excited_gaussian,
                     ground_data, excited_data, _) = self.fit_double_gaussian_midpoint(ig_new)

                    pop_threshold = threshold_mid

                    # -- 1-σ contribution to Pe from the threshold uncertainty --
                    mask_plus = (ig_new <= threshold_mid + threshold_mid_err)
                    Pe_plus = 1.0 - mask_plus.mean()

                    mask_minus = (ig_new <= threshold_mid - threshold_mid_err)
                    Pe_minus = 1.0 - mask_minus.mean()

                    sigma_Pe_from_thresh = 0.5 * abs(Pe_plus - Pe_minus)

                    # statistical err of Pe
                    Nshots = ig_new.size
                    sigma_Pe_stat = np.sqrt(Pe * (1 - Pe) / Nshots)

                    # -- total 1‐σ uncertainty on Pe--
                    sigma_Pe = np.sqrt(sigma_Pe_from_thresh ** 2 + sigma_Pe_stat ** 2)

                pop_threshold = float(pop_threshold)

                #Calculate qubit temps using Pg and Pe
                temp_k = self.calculate_qubit_temperature(freq_mhz, Pg, Pe)


                # -------- screening -----------------------------------------
                if temp_k is None:
                    # un-physical, skip
                    continue
                if temp_k > limit_temp_k:
                    print(f"[run]  Q{qid + 1}: {temp_k * 1e3:.1f} mK  > {limit_temp_k * 1e3:.0f} mK  → dropped")
                    continue

                # Now call on the function compute_temperature_error_SSF to calculate the errs of the qubit temps
                T_mK = temp_k * 1e3
                sigma_TmK, sigma_Pe_total = self.compute_temperature_error_SSF(Pe, sigma_Pe, T_mK, freq_mhz, freq_mhz_err)

                # -------- save qubit temps and timestamps ----------------------------------------------
                all_qubit_temperatures[qid].append(T_mK)  # temperatures in mK
                all_qubit_temperatures_errs[qid].append(sigma_TmK) #temperature errors
                all_qubit_timestamps[qid].append(datetime.datetime.fromtimestamp(ts_unix)) # time stamps

                fit_results[qid].append({
                    "dataset": idx,
                    "timestamp": datetime.datetime.fromtimestamp(ts_unix),
                    "temperature_mK": T_mK,
                    "ig_new": ig_new,
                    "ground_data": ground_data,
                    "excited_data": excited_data,
                    "ground_gaussian": ground_gaussian,
                    "excited_gaussian": excited_gaussian,
                    "pop_threshold": pop_threshold,
                    "weights": weights,
                    "sigmas": sigmas, # of each gaussian in the double gaussian fit
                    "total_sigma_Pe": sigma_Pe, # total 1‐σ uncertainty on Pe
                    "means": means,
                    "Pg": Pg,
                    "Pe": Pe,
                    "qfreq_mhz": freq_mhz,
                    "qfreq_mhz_err": freq_mhz_err,
                    "ssf_fid": ssf_fid, # single shot fidelity
                    "used_gessf_thresh_only": use_gessf_thresh_only, #True when the user decides to use this method
                    "used_fallback_method": used_fallback, #only True if it goes into effect, regardless of user decision
                })

        return all_qubit_temperatures, all_qubit_timestamps, all_qubit_temperatures_errs, fit_results

    def run_ssf_qtemps_notprebuilt(self, pairs_info, limit_temp_k=0.8, do_plots = False, save_figs_path = ""):
        """
        Use Arianna's from-scratch double-Gaussian fitter (SciPy+KMeans)
        and a chi^2 cut as the SSF method.

        Parameters
        ----------
        pairs_info : dict
            {qubit: [ {"qspec":..., "ssf":..., "qfreq_MHz":<MHz>, "qfreq_MHz_err":<MHz_err>,
                       "ig_new":<np.ndarray>, "ie_new":<np.ndarray>, "data_timestamp":<unix-time> }, … ]}
        limit_temp_k : float
            Drop temperatures above this (default 0.8 K -> 800 mK).
        max_chi2_red : float
            Maximum reduced chi^2 allowed for the double-Gaussian fit.
            chi2_red ~ 1 is ideal; 1–2 is usually fine.
        """

        # initialise output arrays
        all_qubit_temperatures      = {i: [] for i in range(self.number_of_qubits)}
        all_qubit_temperatures_errs = {i: [] for i in range(self.number_of_qubits)}
        all_qubit_timestamps        = {i: [] for i in range(self.number_of_qubits)}
        fit_results                 = {qid: [] for qid in range(self.number_of_qubits)}

        for qid, records in pairs_info.items():  # loop over qubits
            for idx, rec in enumerate(records):  # …and every SSF+qspec pair
                freq_mhz     = rec["qfreq_MHz"]
                freq_mhz_err = rec["qfreq_MHz_err"]
                ig_new       = rec["ig_new"]
                ie_new       = rec["ie_new"]
                ts_unix      = rec["data_timestamp"]

                # ----------------- fit double Gaussian on ground-state SSF -----------------
                notprebuiltclass = non_prebuilt_ssf_analysis_class()
                (params, xvals,
                 ground_gaussian, excited_gaussian, sum_gaussians,
                 Pg, Pe, lo, hi) = notprebuiltclass.fit_double_gaussian_on_ground_Arianna(ig_new, numbins=55)

                # ----------------- unpack fit parameters -----------------
                mu1, mu2 = params["mu"]
                sig1, sig2 = params["sigma"]
                w1, w2 = params["weight"]
                pop_threshold = params["threshold"]  # midpoint between means

                # Grab chi^2 from your params dict (2 * NLL_2G)
                chi2 = params["chisq"]
                x_finite = ig_new[np.isfinite(ig_new)]
                nll2 = chi2 / 2.0  # since chisq = 2 * NLL_2G

                # ----------------- fit single Gaussian (finds NLL1 for Likelihood ratio test) -----------------
                params_1g = notprebuiltclass.single_gaussian_nll1(ig_new)

                nll1 = params_1g["nll"]

                # ----------------- Likelihood-ratio test -----------------
                # lr_stat = 2 * (NLL_1G - NLL_2G)
                lr_stat = 2.0 * (nll1 - nll2)

                lrt_limit = 765 # 35.0  # higher = stricter

                if lr_stat < lrt_limit:
                    # 2-Gaussian is NOT strongly favored over single Gaussian
                    # -> treat as essentially single blob, reject this dataset
                    print(f'Rejected a fit with Likelihood ratio test score < {lrt_limit}')

                    if do_plots:
                        bad_plots_path = os.path.join(save_figs_path, "bad_fits_LRT_failed")
                        os.makedirs(bad_plots_path, exist_ok=True)

                        notprebuiltclass.plot_Ariannas_doublegauss_func(
                            ig_new,  # ground-state rotated I shots
                            ie_new,  # excited-state rotated I shots
                            params,
                            numbins=55,
                            save_figs_path=bad_plots_path,
                            filename_ext=f"Q{qid + 1}_Dataset{idx}",
                            title_ext=f"Q{qid + 1}, chi2={chi2:.2f}, LRT value={lr_stat:.2f}"
                        )

                        # self.plot_gaussians_qtemps(qid, bad_plots_path, ig_new, ground_data,
                        #                            excited_data, params["ground_gaussian_idx"],
                        #                            params["excited_gaussian_idx"], pop_threshold,
                        #                            idx, params["weight"],
                        #                            params["sigma"], params["mu"], temperature_mk=None)

                    continue


                # ---------------------- Ground vs excited data split ------------------------------------------
                ground_data  = ig_new[ig_new <= pop_threshold]
                excited_data = ig_new[ig_new > pop_threshold]

                # ----------------- population error: simple binomial -----------------
                # fitter already computed Pg, Pe for ig_new
                Nshots = ig_new.size
                sigma_Pe = np.sqrt(Pe * (1.0 - Pe) / Nshots) if Nshots > 0 else 0.0

                # ----------------- convert Pg, Pe to temperature -----------------
                temp_k = self.calculate_qubit_temperature(freq_mhz, Pg, Pe)

                # discard unphysical or too-hot temps
                if (temp_k is None): # or (temp_k > limit_temp_k)
                    # if self.verbose: print(f"Q{qid+1} idx {idx}: T={temp_k*1e3:.1f} mK > {limit_temp_k*1e3:.0f} mK")
                    continue

                T_mK = temp_k * 1e3
                sigma_TmK, sigma_Pe_total = self.compute_temperature_error_SSF(
                    Pe, sigma_Pe, T_mK, freq_mhz, freq_mhz_err)

                if do_plots:
                    os.makedirs(save_figs_path, exist_ok=True)
                    notprebuiltclass.plot_Ariannas_doublegauss_func(
                        ig_new,  # ground-state rotated I shots
                        ie_new,  # excited-state rotated I shots
                        params,
                        numbins=55,
                        save_figs_path = save_figs_path,
                        filename_ext = f"Q{qid + 1}_Dataset{idx}",
                        title_ext = f"Q{qid + 1}, chi2={chi2:.2f}, LRT value={lr_stat:.2f}"
                    )

                    # self.plot_gaussians_qtemps(qid, save_figs_path, ig_new, ground_data,
                    #                            excited_data, params["ground_gaussian_idx"],
                    #                            params["excited_gaussian_idx"], pop_threshold,
                    #                            idx, params["weight"],
                    #                            params["sigma"], params["mu"], temperature_mk=None)

                # ----------------- save qubit temps + timestamps -----------------
                dt = datetime.datetime.fromtimestamp(ts_unix)

                all_qubit_temperatures[qid].append(T_mK)
                all_qubit_temperatures_errs[qid].append(sigma_TmK)
                all_qubit_timestamps[qid].append(dt)

                # ----------------- save detailed fit info -----------------
                fit_results[qid].append({
                    "dataset": idx,
                    "timestamp": dt,
                    "temperature_mK": T_mK,
                    "ig_new": ig_new,
                    "ground_data": ground_data,
                    "excited_data": excited_data,
                    "ground_gaussian": ground_gaussian,
                    "excited_gaussian": excited_gaussian,
                    "pop_threshold": pop_threshold,
                    "weights": np.array([w1, w2]),
                    "sigmas":  np.array([sig1, sig2]),
                    "means":   np.array([mu1, mu2]),
                    "Pg": Pg,
                    "Pe": Pe,
                    "qfreq_mhz": freq_mhz,
                    "qfreq_mhz_err": freq_mhz_err,
                    "chi2": chi2
                })

        return all_qubit_temperatures, all_qubit_timestamps, all_qubit_temperatures_errs, fit_results

    def run_ssf_qtemps_iminuit(self, pairs_info, run_num, limit_temp_k=0.8, do_plots = False, save_figs_path = "", dontuse_midpt_thresh = False, low_leakage_mode = False, ssf_hist_ylim = None,
                               apply_quality_cuts=True, calc_SNR=False, calc_e_state_decay = True):
        """
        Uses iminuit instead of GMM for double gaussian fitting and minimization.

        Parameters
        ----------
        pairs_info : dict of matched up SSF and Qspec files based on timestamps.
            {qubit: [ {"qspec":..., "ssf":..., "qfreq_MHz":<MHz>,
                        "ig_new":<np.ndarray>, "ie_new":<np.ndarray>, "data_timestamp":<unix-time> }, … ]}
        limit_temp_k : float
        Discard temperatures above this value (default 0.8 K → 800 mK).
        low_leakage_mode: used when fitting SSF data with very smalll thermal populations (example QUIET run 9)
        dontuse_midpt_thresh: when set to False, uses threshold method. When set to True, uses weights of gaussian mixture as Pg and Pe.
        apply_quality_cuts: if True, implements quality cuts that are used for qubit temps

        Returns
        -------
        all_qubit_temperatures : dict {qubit: [temp_mK, …]}
        all_qubit_timestamps   : dict {qubit: [datetime, …]}
        all_qubit_temperatures_errs : dict {qubit: [temp_mK_error, …]}

        fit_results : dict
        { qubit_index: [
          {
            "dataset": <int>,
            "timestamp": <datetime>,
            "temperature_mK": <float>,
            "ig_new": np.ndarray,
            "ground_data": np.ndarray,
            "excited_data": np.ndarray,
            "ground_gaussian": <int>,
            "excited_gaussian": <int>,
            "pop_threshold": <float>,
            "weights": np.ndarray(shape=(2,)),
            "sigmas": np.ndarray(shape=(2,)), #sigma of each gaussian in the double gaussian fit
            "total_sigma_Pe": sigma_Pe, # total 1‐σ uncertainty on Pe
            "means": np.ndarray(shape=(2,)),
            "Pg": Pg,
            "Pe": Pe,
            "qfreq_mhz": freq_mhz, # qubit frequency
            "qfreq_mhz_err": freq_mhz_err,
            ...,
          }]}
        """
        # initialise output arrays
        all_qubit_temperatures = {i: [] for i in range(self.number_of_qubits)}
        all_qubit_temperatures_errs = {i: [] for i in range(self.number_of_qubits)}
        all_qubit_timestamps = {i: [] for i in range(self.number_of_qubits)}
        fit_results = {qid: [] for qid in range(self.number_of_qubits)}

        for qid, records in pairs_info.items():  # loop over qubits
            for idx, rec in enumerate(records):  # …and every pair
                freq_mhz = rec["qfreq_MHz"]
                freq_mhz_err = rec["qfreq_MHz_err"]
                ig_new = rec["ig_new"]
                ie_new = rec["ie_new"]
                ssf_fid = rec["ssf_fid"]
                ts_unix = rec["data_timestamp"]

                # Extracting SSF errors
                ssf_err_shot = rec["ssf_err_shot"]
                ssf_err_bins = rec["ssf_err_bins"]
                ssf_err_fit = rec["ssf_err_fit"]
                ssf_err_total = rec["ssf_err_total"]
                ssf_vs_bins = rec["ssf_vs_bins"]

                # -------- Only using double-Gaussian fit on ground state data, without fallback method --------------------------
                (Pg, Pe, sigma_Pe, m2, means, sigmas, weights, pop_threshold, pop_threshold_err,
                 ground_gaussian, excited_gaussian,
                 ground_data, excited_data, x,
                 lr_stat, nll1, nll2) = self.fit_double_gaussian_midpoint_iminuit(ig_new, dontuse_midpt_thresh, low_leakage_mode)

                # ---------- quality cut limits (do 2 gaussians fit the data better than a single one?) ---------------------

                LR_STAT_LIMITS = {
                    5: {0: 706,
                        1: 619,
                        2: 622,
                        3: 645,
                        4: 460,
                        5: 645,
                    },
                    6: {0: 1071.96, #need to improve fitting to lower this asap bc good ones are also getting cut
                        1: 730,
                        2: 684,
                        3: 604,
                        4: 717,
                        5: 655,
                    },
                    7: {0: 645,
                        1: 645,
                        2: 645,
                        3: 645, # No good data for this qubit in this run, all LRT scores below 280
                        4: 645,
                        5: 645, # No good data for this qubit in this run
                    },
                    8: {0: 645,
                        1: 645,
                        2: 669,
                        3: 645,  # No good data for this qubit in this run
                        4: 645,
                        5: 645,  # No good data for this qubit in this run
                        },
                    9: {0: 250, # not optimized yet for any of the Qs
                        1: 200,
                        2: 200,
                        3: 200,
                        4: 0, # no data for Q5 in this run
                        5: 100,
                        }
                }

                DEFAULT_LR_STAT_LIMIT = 645.0
                if run_num in LR_STAT_LIMITS: # extract the limit for this qubit in this run, or use default
                    lr_stat_limit = LR_STAT_LIMITS[run_num].get(qid, DEFAULT_LR_STAT_LIMIT)
                else:
                    lr_stat_limit = DEFAULT_LR_STAT_LIMIT

                if apply_quality_cuts and lr_stat < lr_stat_limit: # higher = stricter
                    print(f'Rejected a fit with Likelihood ratio test score < {lr_stat_limit}')
                    # not convincingly bimodal --> skip this dataset, it is better described by a single gaussian

                    if do_plots:
                        bad_plots_path = os.path.join(save_figs_path, f"bad_fits_LRT_failed/Q{qid+1}")
                        os.makedirs(bad_plots_path, exist_ok=True)
                        self.plot_gaussians_qtemps(qid, bad_plots_path, ig_new, ground_data,
                                                            excited_data, ground_gaussian,
                                                            excited_gaussian, pop_threshold,
                                                            idx, weights, sigmas, means, temperature_mk = None,
                                                            title_ext = f"LRT val:{lr_stat:.2f}",
                                                            dontuse_midpt_thresh = dontuse_midpt_thresh, ylim = ssf_hist_ylim)

                    continue

                # ---- Run 9 (low_leakage_mode) small-thermal-pop Gaussian spread cut and/or gaussians overlap cut ----------------
                sigma_g, sigma_e = sigmas[0], sigmas[1]
                sigma_ratio = sigma_e / sigma_g if sigma_g > 0 else np.inf
                sig_ratio_thresh = 3.0 # much larger than 1 = excited Gaussian is very broad -> suspicious. For now leaving loose cut
                bad_leakage_spread = (sigma_e <= 0 or sigma_g <= 0 or sigma_ratio > sig_ratio_thresh)

                mu_g, mu_e = means[0], means[1]
                overlap_metric = abs(mu_e - mu_g) / (sigma_g + sigma_e)
                if qid == 0:
                    overlap_min = 2.0 # Large value -> good separation -> LOW overlap -> good fit
                elif qid == 1:
                    overlap_min = 2.12
                elif qid == 2:
                    overlap_min = 2.06
                elif qid == 3:
                    overlap_min = 2.08
                elif qid == 5:
                    overlap_min = 2.49
                else:
                    overlap_min = 2.0
                bad_overlap = overlap_metric < overlap_min

                if apply_quality_cuts and low_leakage_mode and (bad_leakage_spread or bad_overlap): # only for run 9 so far
                    print(f"Rejected fit | sigma_ratio={sigma_ratio:.2f} (>{sig_ratio_thresh}) or overlap={overlap_metric:.2f} (<{overlap_min})")
                    if do_plots:
                        bad_plots_path = os.path.join(save_figs_path, f"bad_fits_LRT_failed/Q{qid+1}")
                        os.makedirs(bad_plots_path, exist_ok=True)
                        self.plot_gaussians_qtemps(qid, bad_plots_path, ig_new, ground_data,
                                                            excited_data, ground_gaussian,
                                                            excited_gaussian, pop_threshold,
                                                            idx, weights, sigmas, means, temperature_mk = None,
                                                            title_ext = f"sigma ratio:{sigma_ratio:.2f}, overlap val:{overlap_metric:.2f}",
                                                            dontuse_midpt_thresh = dontuse_midpt_thresh, ylim = ssf_hist_ylim)
                    continue

                if pop_threshold is not None:
                    pop_threshold = float(pop_threshold)

                #Calculate qubit temps using Pg and Pe
                temp_k = self.calculate_qubit_temperature(freq_mhz, Pg, Pe)

                # -------- screening -----------------------------------------
                if temp_k is None:
                    # un-physical, skip
                    continue
                if temp_k > limit_temp_k:
                    print(f"[run]  Q{qid + 1}: {temp_k * 1e3:.1f} mK  > {limit_temp_k * 1e3:.0f} mK  -> dropped")
                    continue

                # Now call on the function compute_temperature_error_SSF to calculate the errs of the qubit temps
                T_mK = temp_k * 1e3
                sigma_TmK, sigma_Pe_total = self.compute_temperature_error_SSF(Pe, sigma_Pe, T_mK, freq_mhz, freq_mhz_err)

                # Plotting
                if do_plots:
                    save_figs_path_clean = os.path.join(save_figs_path, f"Q{qid + 1}") # to separate plots by qubit
                    self.plot_gaussians_qtemps(qid, save_figs_path_clean, ig_new, ground_data,
                                               excited_data, ground_gaussian,
                                               excited_gaussian, pop_threshold,
                                               idx, weights,
                                               sigmas, means, T_mK, title_ext=f"{datetime.datetime.fromtimestamp(ts_unix)}, Qfreq:{freq_mhz:.2f}MHz, LRT val:{lr_stat:.2f}, sigma ratio:{sigma_ratio:.2f}, overlap val:{overlap_metric:.2f}",
                                               dontuse_midpt_thresh = dontuse_midpt_thresh, ylim = ssf_hist_ylim)

                # -------------------------- optionally calulate SNR of the scan -------------------------
                snr = np.nan
                if calc_SNR:
                    save_figs_path_SNR = os.path.join(save_figs_path, "SNR")
                    snr, snr_info = self.fit_ssf_ge_double_gaussian_SNR_iminuit(ig_new,ie_new, qid, plot = False, qubit_folder = save_figs_path_SNR, dataset = idx)
                    
                # -------- optionally estimate e-state population on the left due to T1 decay, failed pi pulses, etc. --------
                # ---------------- fit excited-prepared SSF data ----------------
                ie_new_ground_frac = np.nan
                ie_new_ground_frac_err = np.nan
                ie_new_excited_frac = np.nan
                ie_new_excited_frac_err = np.nan
                ie_new_lr_stat = np.nan
                ie_new_fit_results = np.nan

                if calc_e_state_decay:
                    ie_new_fit_results = self.fit_e_state_double_gaussian_iminuit(ie_new, qid=qid, dataset=idx)
                    if ie_new_fit_results is not None:
                        ie_new_ground_frac = ie_new_fit_results["ie_new_ground_frac"]
                        ie_new_ground_frac_err = ie_new_fit_results["ie_new_ground_frac_err"]
                        ie_new_excited_frac = ie_new_fit_results["ie_new_excited_frac"]
                        ie_new_excited_frac_err = ie_new_fit_results["ie_new_excited_frac_err"]
                        ie_new_lr_stat = ie_new_fit_results["ie_new_lr_stat"]

                # -------- save qubit temps and timestamps ----------------------------------------------
                all_qubit_temperatures[qid].append(T_mK)  # temperatures in mK
                all_qubit_temperatures_errs[qid].append(sigma_TmK) #temperature errors
                all_qubit_timestamps[qid].append(datetime.datetime.fromtimestamp(ts_unix)) # time stamps

                fit_results[qid].append({
                    "dataset": idx,
                    "timestamp": datetime.datetime.fromtimestamp(ts_unix),
                    "temperature_mK": T_mK,
                    "temperature_err_mK": sigma_TmK,
                    "ig_new": ig_new,
                    "ie_new": ie_new,
                    "ground_data": ground_data,
                    "excited_data": excited_data,
                    "ground_gaussian": ground_gaussian, # index
                    "excited_gaussian": excited_gaussian, # index
                    "pop_threshold": pop_threshold,
                    "weights": weights,
                    "sigmas": sigmas, # of each gaussian in the double gaussian fit
                    "total_sigma_Pe": sigma_Pe, # total 1‐σ uncertainty on Pe
                    "means": means,
                    "Pg": Pg,
                    "Pe": Pe,
                    "ssf_fid": ssf_fid,
                    "ssf_err_shot": ssf_err_shot,
                    "ssf_err_bins": ssf_err_bins,
                    "ssf_err_fit": ssf_err_fit,
                    "ssf_err_total": ssf_err_total, # Total SSF error
                    "qfreq_mhz": freq_mhz,
                    "qfreq_mhz_err": freq_mhz_err,
                    "ssf_SNR": snr,

                    # Excited-prepared ie_new double-Gaussian fit.
                    # ie_new_ground_frac = observed ground-like fraction in intended e-state data.
                    # If calc_e_state_decay is set to False, these are np.nan
                    "ie_new_fit_results": ie_new_fit_results,
                    "ie_new_ground_frac": ie_new_ground_frac,
                    "ie_new_ground_frac_err": ie_new_ground_frac_err,
                    "ie_new_excited_frac": ie_new_excited_frac,
                    "ie_new_excited_frac_err": ie_new_excited_frac_err,
                    "ie_new_lr_stat": ie_new_lr_stat,
                })

        return all_qubit_temperatures, all_qubit_timestamps, all_qubit_temperatures_errs, fit_results

    def compute_temperature_error_SSF(
            self,
            Pe,
            sigma_Pe,
            T_mK,
            qubit_freq_MHz,
            sigma_qfreq_MHz):
        """
        Propagate 1-sigma uncertainties in Pe and f_ge into a 1-sigma uncertainty on T_mK.

        Inputs:
          Pe                    : excited-state population (SSF)
          sigma_Pe              : 1-sigma "fit/stat/threshold" uncertainty on Pe (baseline)
          Pe_dist_err           : optional extra 1-sigma scatter on Pe across the run (e.g., histogram sigma_w)
          T_mK                  : computed temperature (mK)
          qubit_freq_MHz        : fitted g-e qubit frequency (MHz)
          sigma_qfreq_MHz       : 1-sigma error on qubit_freq_MHz (MHz)

        Returns:
          sigma_T_mK            : propagated 1-sigma error on T_mK (mK)
          sigma_Pe              : Pe uncertainty from double gauss fitting function
        """
        # --- sanitize inputs ---
        Pe = float(Pe)
        sigma_Pe = float(sigma_Pe)
        T_mK = float(T_mK)
        qubit_freq_MHz = float(qubit_freq_MHz)
        sigma_qfreq_MHz = float(sigma_qfreq_MHz)

        # Convert MHz -> Hz
        f0_Hz = qubit_freq_MHz * 1e6
        sigma_f0_Hz = sigma_qfreq_MHz * 1e6

        # Build the logarithmic term
        ln_arg = np.log((1.0 - Pe) / Pe)

        # Partial derivatives:
        # T_mK = (h f0)/(kB ln_arg) * 1e3  => dT/df0 = T_mK/f0
        dT_df0 = T_mK / f0_Hz

        # dT/dPe = T_mK / [ ln_arg * Pe * (1-Pe) ]
        dT_dPe = T_mK / (ln_arg * Pe * (1.0 - Pe))

        sigma_T_mK = np.sqrt(
            (dT_df0 * sigma_f0_Hz) ** 2 +
            (dT_dPe * sigma_Pe) ** 2
        )

        return sigma_T_mK, sigma_Pe

    def plot_2D_ssf_thermal_pop(
            self,
            fit_results,
            qid,
            save_figs_path=None,
            bins=np.linspace(-0.75, 1.75, 180),
            normalize_each_row=True,
            sort_by_time=True,
            cmap="viridis",
            fig_quality=300,
            title=None,
            filename=None,
            show=True,
    ):
        """
        Make a 2D heatmap of SSF ground-state data over time.

        Each row is one SSF dataset. The x-axis is normalized so that:
            main ground Gaussian mean  -> 0
            thermal/excited Gaussian mean -> 1

        This makes it easier to visually compare the thermal excitation population
        across many SSF datasets in one plot.

        Parameters
        ----------
        fit_results : dict
            Output from run_ssf_qtemps_iminuit().
            Expected format:
                fit_results[qid] = [
                    {
                        "timestamp": datetime,
                        "ig_new": np.ndarray,
                        "means": np.ndarray(shape=(2,)),
                        "ground_gaussian": int,
                        "excited_gaussian": int,
                        "Pe": float,
                        "temperature_mK": float,
                        ...
                    },
                    ...
                ]

        qid : int
            Zero-indexed qubit index.

        save_figs_path : str or None
            Folder where figure should be saved. If None, figure is not saved.

        bins : np.ndarray
            Bins for the normalized readout axis.

        normalize_each_row : bool
            If True, divide each histogram by its own max count.
            This helps compare shape/thermal shoulder instead of total counts.

        sort_by_time : bool
            If True, sort datasets by timestamp before plotting.

        cmap : str
            Matplotlib colormap.

        fig_quality : int
            DPI for saved figure.

        title : str or None
            Optional plot title.

        filename : str or None
            Optional filename.

        show : bool
            Whether to call plt.show().

        Returns
        -------
        fig, ax, heatmap, centers, records_used
        """
        if qid not in fit_results:
            raise KeyError(f"Qubit index {qid} not found in fit_results.")

        records = fit_results[qid]

        if len(records) == 0:
            raise ValueError(f"No fit_results found for Q{qid + 1}.")

        # Remove records missing needed information
        records_used = []
        for rec in records:
            needed_keys = ["ig_new", "means", "ground_gaussian", "excited_gaussian"]
            if not all(k in rec for k in needed_keys):
                continue

            if rec["ig_new"] is None or rec["means"] is None:
                continue

            records_used.append(rec)

        if len(records_used) == 0:
            raise ValueError(f"No valid records found for Q{qid + 1}.")

        if sort_by_time and "timestamp" in records_used[0]:
            records_used = sorted(records_used, key=lambda r: r["timestamp"])

        heatmap = []
        Pe_vals = []
        T_vals = []
        timestamps = []

        for rec in records_used:
            ig_new = np.asarray(rec["ig_new"]).ravel()

            means = np.asarray(rec["means"])
            ground_idx = int(rec["ground_gaussian"])
            excited_idx = int(rec["excited_gaussian"])

            ground_mean = means[ground_idx]
            excited_mean = means[excited_idx]

            separation = excited_mean - ground_mean

            if np.isclose(separation, 0):
                print(f"Skipping dataset {rec.get('dataset', 'unknown')} because means overlap.")
                continue

            # Normalize so ground peak -> 0 and excited/thermal peak -> 1
            ig_norm = (ig_new - ground_mean) / separation

            # If the excited peak lands at x = -1, flip the axis
            # so thermal population is always near x = +1.
            if excited_mean < ground_mean:
                ig_norm = -ig_norm

            counts, edges = np.histogram(ig_norm, bins=bins)

            if normalize_each_row:
                row_max = np.max(counts)
                if row_max > 0:
                    counts = counts / row_max

            heatmap.append(counts)
            Pe_vals.append(rec.get("Pe", np.nan))
            T_vals.append(rec.get("temperature_mK", np.nan))
            timestamps.append(rec.get("timestamp", None))

        heatmap = np.asarray(heatmap)

        if heatmap.size == 0:
            raise ValueError(f"No histograms were generated for Q{qid + 1}.")

        centers = 0.5 * (bins[:-1] + bins[1:])

        fig, ax = plt.subplots(figsize=(10, 7))

        im = ax.imshow(
            heatmap,
            aspect="auto",
            origin="lower",
            extent=[centers[0], centers[-1], 0, len(heatmap) - 1],
            cmap=cmap,
            interpolation="nearest",
        )

        ax.axvline(0, linestyle="--", linewidth=1.5, color="white", alpha=0.85)
        ax.axvline(1, linestyle="--", linewidth=1.5, color="red", alpha=0.85)

        ax.text(
            0,
            len(heatmap) - 1,
            "  |g> peak",
            color="white",
            fontsize=11,
            va="top",
            ha="left",
        )

        ax.text(
            1,
            len(heatmap) - 1,
            "  |e>/thermal peak",
            color="red",
            fontsize=11,
            va="top",
            ha="left",
        )

        ax.set_xlabel(
            "Normalized rotated SSF axis\nmain |g> peak = 0, thermal/excited peak = 1",
            fontsize=14,
        )
        ax.set_ylabel("SSF dataset / time index", fontsize=14)

        if title is None:
            title = f"Run 9 Q{qid + 1} SSF thermal population trend"

        ax.set_title(title, fontsize=16)

        cbar = fig.colorbar(im, ax=ax)
        if normalize_each_row:
            cbar.set_label("Counts normalized to each row maximum", fontsize=12)
        else:
            cbar.set_label("Counts", fontsize=12)

        # Add sparse timestamp labels
        if all(ts is not None for ts in timestamps):
            n = len(timestamps)
            tick_idx = np.linspace(0, n - 1, min(6, n), dtype=int)
            tick_labels = [timestamps[i].strftime("%m-%d %H:%M") for i in tick_idx]

            ax.set_yticks(tick_idx)
            ax.set_yticklabels(tick_labels)

        # Optional second y-axis showing Pe or temperature at a few points
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())

        n = len(Pe_vals)
        tick_idx = np.linspace(0, n - 1, min(6, n), dtype=int)

        pe_labels = []
        for i in tick_idx:
            pe = Pe_vals[i]
            temp = T_vals[i]

            if np.isfinite(pe) and np.isfinite(temp):
                pe_labels.append(f"Pe={pe:.3f}, T={temp:.0f} mK")
            elif np.isfinite(pe):
                pe_labels.append(f"Pe={pe:.3f}")
            elif np.isfinite(temp):
                pe_labels.append(f"T={temp:.0f} mK")
            else:
                pe_labels.append("")

        ax2.set_yticks(tick_idx)
        ax2.set_yticklabels(pe_labels, fontsize=9)
        ax2.set_ylabel("Extracted Pe / T", fontsize=12)

        plt.tight_layout()

        if save_figs_path is not None:
            os.makedirs(save_figs_path, exist_ok=True)

            if filename is None:
                filename = f"Q{qid + 1}_SSF_thermal_population_2D_heatmap.png"

            full_path = os.path.join(save_figs_path, filename)
            fig.savefig(full_path, dpi=fig_quality, bbox_inches="tight")
            print(f"Saved: {full_path}")

        if show:
            plt.show()

        return fig, ax, heatmap, centers, records_used

    def update_ssf_errors_with_pe_scatter_inplace(self, ssf_fit_results, Pe_dist_err_dict=None, ssf_pe_scatter_min_n=5,
            verbose=True, preserve_base_sigma=True):
        """
        In-place update of SSF Pe + temperature error bars using an optional Pe_dist_err_dict
        (from your SSF Pe histogram function with only_return_mu_and_sigma=True).

        This version assumes compute_temperature_error_SSF has been upgraded to accept Pe_dist_err
        and returns (sigma_T_mK, sigma_Pe_total).

        Parameters
        ----------
        ssf_fit_results : dict
            fit_results from run_ssf_qtemps:
              { qid: [ { "Pe":..., "total_sigma_Pe":..., "temperature_mK":..., "qfreq_mhz":..., "qfreq_mhz_err":..., ... }, ... ], ... }

        Pe_dist_err_dict : dict or None
            Slim dict from plot_all_Qs_Pe_hists_ssf(... only_return_mu_and_sigma=True):
              { q: {"mu_w":..., "sigma_w":..., "n_kept":..., "n_raw":...}, ... }

        ssf_pe_scatter_min_n : int
            Require at least this many kept points before using sigma_w as a 1s scatter.

        preserve_base_sigma : bool
            If True, stores the original (pre-inflation) sigma in "total_sigma_Pe_base" once,
            so rerunning this updater doesn't repeatedly inflate.

        Returns
        -------
        ssf_fit_results : same object (mutated)
        stats : dict with counts of updated/skipped
        """
        import numpy as np

        stats = {
            "updated_points": 0,
            "skipped_missing_fields": 0,
            "skipped_bad_numbers": 0,
            "skipped_bad_qubit_key": 0,
            "skipped_compute_fail": 0,
            "used_pe_dist_err": 0,
        }

        if Pe_dist_err_dict is None:
            Pe_dist_err_dict = {}

        if ssf_fit_results is None or not isinstance(ssf_fit_results, dict):
            raise ValueError("ssf_fit_results must be a dict: {qid: [dict, dict, ...], ...}")

        for q_key, rec_list in ssf_fit_results.items():

            # normalize q key
            try:
                q_int = int(q_key)
            except Exception:
                stats["skipped_bad_qubit_key"] += 1
                continue

            if not isinstance(rec_list, list):
                continue

            for d in rec_list:
                if not isinstance(d, dict):
                    stats["skipped_missing_fields"] += 1
                    continue

                # required fields
                Pe = d.get("Pe", None)
                sigma_Pe = d.get("total_sigma_Pe", None)
                T_mK = d.get("temperature_mK", None)
                qfreq_mhz = d.get("qfreq_mhz", None)
                qfreq_mhz_err = d.get("qfreq_mhz_err", None)

                req = [Pe, sigma_Pe, T_mK, qfreq_mhz, qfreq_mhz_err]
                if any(v is None for v in req):
                    stats["skipped_missing_fields"] += 1
                    continue

                # numeric sanity
                try:
                    Pe = float(Pe)
                    sigma_Pe = float(sigma_Pe)
                    T_mK = float(T_mK)
                    qfreq_mhz = float(qfreq_mhz)
                    qfreq_mhz_err = float(qfreq_mhz_err)
                except Exception:
                    stats["skipped_bad_numbers"] += 1
                    continue

                if not np.all(np.isfinite([Pe, sigma_Pe, T_mK, qfreq_mhz, qfreq_mhz_err])):
                    stats["skipped_bad_numbers"] += 1
                    continue

                # sanity bounds
                if (Pe <= 0.0) or (Pe >= 1.0) or (sigma_Pe <= 0.0) or (T_mK <= 0.0) or (qfreq_mhz <= 0.0) or (
                        qfreq_mhz_err < 0.0):
                    stats["skipped_bad_numbers"] += 1
                    continue

                # preserve base sigma once (so reruns don't double-inflate)
                if preserve_base_sigma and ("total_sigma_Pe_base" not in d):
                    d["total_sigma_Pe_base"] = sigma_Pe

                base_sigma = float(d.get("total_sigma_Pe_base", sigma_Pe))

                # --- Optional histogram-based Pe scatter for this qubit ---
                Pe_dist_err = None
                h = Pe_dist_err_dict.get(q_int, None)
                if h is not None:
                    sigma_w = h.get("sigma_w", np.nan)
                    n_kept = int(h.get("n_kept", 0) or 0)
                    if np.isfinite(sigma_w) and (sigma_w > 0.0) and (n_kept >= ssf_pe_scatter_min_n):
                        Pe_dist_err = float(sigma_w)
                        stats["used_pe_dist_err"] += 1

                # --- Recompute error bars using upgraded compute_temperature_error_SSF ---
                try:
                    sigma_TmK_new, sigma_Pe_total = self.compute_temperature_error_SSF(
                        Pe=Pe,
                        sigma_Pe=base_sigma,  # baseline (fit/stat)
                        Pe_dist_err=Pe_dist_err,  # optional distribution scatter
                        T_mK=T_mK,
                        qubit_freq_MHz=qfreq_mhz,
                        sigma_qfreq_MHz=qfreq_mhz_err,
                    )
                except Exception:
                    stats["skipped_compute_fail"] += 1
                    continue

                # store back in-place (match your SSF schema)
                d["total_sigma_Pe"] = float(sigma_Pe_total) if np.isfinite(sigma_Pe_total) else np.nan
                d["Pe_dist_err"] = Pe_dist_err  # None or float
                d["temperature_mK_err"] = float(sigma_TmK_new) if np.isfinite(sigma_TmK_new) else np.nan

                stats["updated_points"] += 1

        if verbose:
            print("[update_ssf_errors_with_pe_scatter_inplace] stats:", stats)

        return ssf_fit_results, stats

    # -------------------- “g-e threshold only” runner -----------------
    def plot_ssf_ge_thresh(self, pairs_info: dict, plotting_path: str, numbins: int = 64):
        """
        For every (qubit,dataset) in `pairs_info`:
        •fit a two–Gaussian GMM to ig_new + ie_new
        •use the midpoint of the component means as threshold
        •save a diagnostic plot
        •collect numerical results in a return‑dict

        Parameters
        ----------
        pairs_info  : { qubit_index : [record,…] } – must contain
                      ig_new  and  ie_new  per record.
        out_root    : top‑level directory where plots will be written.
        numbins     : histogram bins for the diagnostic plot.

        Returns
        -------
        thresh_results : { qubit_index : [ {dataset,threshold,means,sigmas,
                                            weights,ground_idx,excited_idx}, … ] }
        """

        thresh_results = {q: [] for q in pairs_info}

        for qid, records in pairs_info.items():
            # one folder per qubit
            q_folder = os.path.join(plotting_path, f"Q{qid + 1}")
            os.makedirs(q_folder, exist_ok=True)

            # Make a date‐stamped subfolder
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            made_on_folder = os.path.join(q_folder, f"made_on_{date_str}")
            os.makedirs(made_on_folder, exist_ok=True)

            for rec in records:
                ig_new = rec["ig_new"] #prepared ground state data (rotated I values)
                ie_new = rec["ie_new"] #prepared first excited state data (rotated I values)
                ds = rec.get("dataset", "NA")

                # ---------- fit & extract numbers ----------
                thresh, means, sigmas, weights, ground_idx, excited_idx = self.ssf_fit_two_gaussians_midpoint(ig_new, ie_new)

                # ---------- plot to check things fitted correctly ----------
                fig, ax = plt.subplots(figsize=(7, 4))
                all_i = np.concatenate([ig_new, ie_new])

                # histogram of *all* shots (does not show overlaps)
                # n, edges, _ = ax.hist(all_i, bins=numbins, alpha=0.35, color="grey", label="all shots")
                # counts, edges = np.histogram(all_i, bins=numbins) # just extracting edges

                # Plot g and e histograms separately (shows populations that overlap)
                edges = np.linspace(all_i.min(), all_i.max(), numbins + 1)
                ax.hist(ig_new, bins=edges, alpha=0.55, color="royalblue", label="g-state")
                ax.hist(ie_new, bins=edges, alpha=0.55, color="crimson", label="e-state")

                x_grid = np.linspace(all_i.min(), all_i.max(), 400)
                g_pdf = (weights[ground_idx] /
                         (np.sqrt(2 * np.pi) * sigmas[ground_idx]) *
                         np.exp(-0.5 * ((x_grid - means[ground_idx]) /
                                        sigmas[ground_idx]) ** 2))
                e_pdf = (weights[excited_idx] /
                         (np.sqrt(2 * np.pi) * sigmas[excited_idx]) *
                         np.exp(-0.5 * ((x_grid - means[excited_idx]) /
                                        sigmas[excited_idx]) ** 2))

                # Component‑specific scaling. We scale since we want to plot y-axis in counts instead of PDFs to match original SSF plots
                counts_g, _ = np.histogram(ig_new, bins=edges)
                counts_e, _ = np.histogram(ie_new, bins=edges)

                peak_g = counts_g.max()
                peak_e = counts_e.max()

                # factor that makes the PDF peak equal the tallest bar
                scale_g = peak_g / g_pdf.max()
                scale_e = peak_e / e_pdf.max()

                ax.plot(x_grid, g_pdf * scale_g, color="blue", lw=2,
                        label="ground Gaussian")
                ax.plot(x_grid, e_pdf * scale_e, color="red", lw=2,
                        label="excited Gaussian")

                # vertical markers
                ax.axvline(means[ground_idx], color="blue", ls="--")
                ax.axvline(means[excited_idx], color="red", ls="--")
                ax.axvline(thresh, color="black", ls=":",
                           label=f"g-e threshold = {thresh:.2f}")

                ax.set_title(f"Q{qid + 1}")
                ax.set_xlabel("I'  (rotated)")
                ax.set_ylabel("Counts")
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend(frameon=False)
                fig.tight_layout()

                fname = os.path.join(made_on_folder, f"Q{qid + 1}_midpoint_fit_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
                fig.savefig(fname, dpi=self.figure_quality)
                plt.close(fig)

                # ---------- store numbers ----------
                thresh_results[qid].append(dict(dataset=ds, threshold=thresh, means=means, sigmas=sigmas, weights=weights, ground_idx=int(ground_idx), excited_idx=int(excited_idx)))
        print('Plots saved to:', plotting_path)
        return thresh_results

    def plot_gaussians_qtemps(
            self,
            q_key,
            qubit_folder,
            ig_new,
            ground_data,
            excited_data,
            ground_gaussian,
            excited_gaussian,
            pop_threshold,
            dataset,
            weights,
            sigmas,
            means,
            temperature_mk=None,
            title_ext="",
            dontuse_midpt_thresh=False,
            numbins=64,
            ylim = None
    ):
        """
        Clean SSF double-Gaussian visualization.

        - If dontuse_midpt_thresh=True:
            Uses mixture weights as Pg/Pe. No hard threshold or shaded regions.
        - If dontuse_midpt_thresh=False:
            Shows midpoint threshold and shaded split regions.
        """
        os.makedirs(qubit_folder, exist_ok=True)

        xdata = np.asarray(ig_new, dtype=float).ravel()
        xdata = xdata[np.isfinite(xdata)]
        if xdata.size == 0:
            return

        weights = np.asarray(weights, dtype=float).ravel()
        means = np.asarray(means, dtype=float).ravel()
        sigmas = np.asarray(sigmas, dtype=float).ravel()

        # Normalize weights for safety
        weights = weights / np.sum(weights)

        xlims = [float(np.min(xdata)), float(np.max(xdata))]

        # Histogram
        counts, edges = np.histogram(xdata, bins=numbins, range=xlims)
        bin_w = edges[1] - edges[0]
        N = xdata.size

        plt.figure(figsize=(10, 6))

        plt.hist(
            xdata,
            bins=numbins,
            range=xlims,
            density=False,
            alpha=0.5,
            color="gray",
            edgecolor="black",
            label="Histogram of $I_g$",
        )

        # Generate smooth curves
        xplot = np.linspace(xlims[0], xlims[1], 1000)

        comp0 = N * bin_w * weights[0] * norm.pdf(xplot, loc=means[0], scale=sigmas[0])
        comp1 = N * bin_w * weights[1] * norm.pdf(xplot, loc=means[1], scale=sigmas[1])
        mixture = comp0 + comp1

        plt.plot(xplot, comp0, color="blue", linewidth=2,
                 label=f"Component 0 (w={weights[0]:.3f})")

        plt.plot(xplot, comp1, color="red", linewidth=2,
                 label=f"Component 1 (w={weights[1]:.3f})")

        plt.plot(xplot, mixture, color="black", linestyle="--", linewidth=2,
                 label="Mixture")

        # Only draw threshold if using midpoint method
        if not dontuse_midpt_thresh and pop_threshold is not None:
            plt.axvline(
                float(pop_threshold),
                color="black",
                linestyle=":",
                linewidth=2,
                label=f"Threshold ({float(pop_threshold):.3f})",
            )

            if ground_data is not None and excited_data is not None:
                gd = np.asarray(ground_data, dtype=float)
                ed = np.asarray(excited_data, dtype=float)

                plt.hist(
                    gd,
                    bins=numbins,
                    range=xlims,
                    alpha=0.25,
                    color="blue",
                    label="Ground side (cut)",
                )

                plt.hist(
                    ed,
                    bins=numbins,
                    range=xlims,
                    alpha=0.25,
                    color="red",
                    label="Excited side (cut)",
                )

        # Title
        title = f"SSF g-data fit, Q{q_key + 1}"
        if temperature_mk is not None:
            title += f" , T={temperature_mk:.2f}mK"
        if title_ext:
            title += f" {title_ext}"

        plt.title(title)
        plt.xlabel("$I_g$", fontsize=14)
        plt.ylabel("Counts", fontsize=14)

        if ylim is not None:
            plt.ylim(0, ylim)

        plt.legend()

        plot_filename = os.path.join(
            qubit_folder,
            f"Q{q_key + 1}_SSF_gaussfit_Dataset{dataset}_{datetime.datetime.now():%Y%m%d%H%M%S}.png")
        #print('saved: ', plot_filename)
        plt.savefig(plot_filename, dpi = 300, bbox_inches="tight")
        plt.close()

    def plot_gaussians_SNR(
            self,
            q_key,
            qubit_folder,
            ig_new,
            ie_new,
            weights,
            sigmas,
            means,
            snr,
            dataset=None,
            title_ext="",
            numbins=64,
            ylim=None,
    ):
        """
        Plot the prepared |g> and prepared |e> SSF data together with the
        fitted two-Gaussian mixture used to calculate readout SNR.

        This is intended only as a diagnostic plot for checking the SNR fit:

            SNR = |mu_e - mu_g| / sqrt((sigma_g^2 + sigma_e^2) / 2)

        After ordering, this assumes:
            means[0], sigmas[0], weights[0] -> prepared |g> component
            means[1], sigmas[1], weights[1] -> prepared |e> component
        """

        os.makedirs(qubit_folder, exist_ok=True)

        ig = np.asarray(ig_new, dtype=float).ravel()
        ie = np.asarray(ie_new, dtype=float).ravel()

        ig = ig[np.isfinite(ig)]
        ie = ie[np.isfinite(ie)]

        if ig.size == 0 or ie.size == 0:
            return

        weights = np.asarray(weights, dtype=float).ravel()
        means = np.asarray(means, dtype=float).ravel()
        sigmas = np.asarray(sigmas, dtype=float).ravel()

        if weights.size != 2 or means.size != 2 or sigmas.size != 2:
            return

        if np.sum(weights) <= 0 or not np.isfinite(np.sum(weights)):
            return

        # Normalize weights for safety
        weights = weights / np.sum(weights)

        xdata = np.concatenate([ig, ie])

        x_min = float(np.min(xdata))
        x_max = float(np.max(xdata))

        if x_min == x_max:
            return

        xlims = [x_min, x_max]

        # Histogram scaling
        counts, edges = np.histogram(xdata, bins=numbins, range=xlims)
        bin_w = edges[1] - edges[0]
        N = xdata.size

        plt.figure(figsize=(10, 6))

        # Plot prepared-state histograms separately
        plt.hist(
            ig,
            bins=numbins,
            range=xlims,
            density=False,
            alpha=0.45,
            color="blue",
            edgecolor="black",
            label="Prepared $|g\\rangle$ data",
        )

        plt.hist(
            ie,
            bins=numbins,
            range=xlims,
            density=False,
            alpha=0.45,
            color="red",
            edgecolor="black",
            label="Prepared $|e\\rangle$ data",
        )

        # Smooth fitted curves
        xplot = np.linspace(xlims[0], xlims[1], 1000)

        comp_g = N * bin_w * weights[0] * norm.pdf(
            xplot,
            loc=means[0],
            scale=sigmas[0],
        )

        comp_e = N * bin_w * weights[1] * norm.pdf(
            xplot,
            loc=means[1],
            scale=sigmas[1],
        )

        mixture = comp_g + comp_e

        plt.plot(
            xplot,
            comp_g,
            color="blue",
            linewidth=2,
            label=(
                f"$|g\\rangle$ fit: "
                f"$\\mu_g$={means[0]:.3f}, "
                f"$\\sigma_g$={sigmas[0]:.3f}"
            ),
        )

        plt.plot(
            xplot,
            comp_e,
            color="red",
            linewidth=2,
            label=(
                f"$|e\\rangle$ fit: "
                f"$\\mu_e$={means[1]:.3f}, "
                f"$\\sigma_e$={sigmas[1]:.3f}"
            ),
        )

        plt.plot(
            xplot,
            mixture,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Two-Gaussian mixture",
        )

        # Mark fitted means
        plt.axvline(
            means[0],
            color="blue",
            linestyle=":",
            linewidth=2,
            label="$\\mu_g$",
        )

        plt.axvline(
            means[1],
            color="red",
            linestyle=":",
            linewidth=2,
            label="$\\mu_e$",
        )

        title = f"SSF SNR double-Gaussian fit, Q{q_key + 1}"
        if dataset is not None:
            title += f", Dataset {dataset}"
        title += f", SNR={snr:.2f}"

        if title_ext:
            title += f" {title_ext}"

        plt.title(title)
        plt.xlabel("Rotated/projected SSF signal", fontsize=14)
        plt.ylabel("Counts", fontsize=14)

        if ylim is not None:
            plt.ylim(0, ylim)

        plt.legend(fontsize=10)
        plt.tight_layout()

        if dataset is None:
            dataset_str = ""
        else:
            dataset_str = f"_Dataset{dataset}"

        plot_filename = os.path.join(
            qubit_folder,
            f"Q{q_key + 1}_SSF_SNR_gaussfits{dataset_str}_{datetime.datetime.now():%Y%m%d%H%M%S}.png",
        )

        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()

    # -------------------------------OLD WAY: MADE FOR MIDPOINT THRESHOLD METHOD ONLY------------
    # def plot_gaussians_qtemps(self, q_key, qubit_folder, ig_new, ground_data, excited_data, ground_gaussian, excited_gaussian, pop_threshold, dataset, weights, sigmas, means,
    #                           temperature_mk = None, title_ext = "", dontuse_midpt_thresh = False):
    #     # -----------------PLOTS TO CHECK g-state double gaussian FITS AND THRESHOLDS---------------
    #     # Plotting double gaussian distributions and fitting
    #     xlims = [np.min(ig_new), np.max(ig_new)]
    #     plt.figure(figsize=(10, 6))
    #
    #     # Plot histogram for `ig_new`
    #     steps = 3000
    #     # numbins = round(math.sqrt(steps))
    #     numbins = 64
    #     n, bins, _ = plt.hist(ig_new, bins=numbins, range=xlims, density=False, alpha=0.5,
    #                           label='Histogram of $I_g$',
    #                           color='gray')
    #     # print(numbins)
    #     # Use the midpoints of bins to create boolean masks
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #     ground_region = (bin_centers <= pop_threshold)
    #     excited_region = (bin_centers > pop_threshold)
    #
    #     # Calculate scaling factors for each region
    #     scaling_factor_ground = max(n[ground_region]) / max(
    #         (weights[ground_gaussian] / (np.sqrt(2 * np.pi) * sigmas[ground_gaussian])) * np.exp(
    #             -0.5 * ((bin_centers[ground_region] - means[ground_gaussian]) / sigmas[
    #                 ground_gaussian]) ** 2))
    #
    #     scaling_factor_excited = max(n[excited_region]) / max(
    #         (weights[excited_gaussian] / (np.sqrt(2 * np.pi) * sigmas[excited_gaussian])) * np.exp(
    #             -0.5 * ((bin_centers[excited_region] - means[excited_gaussian]) / sigmas[
    #                 excited_gaussian]) ** 2))
    #
    #     # Generate x values for plotting Gaussian components
    #     x = np.linspace(xlims[0], xlims[1], 1000)
    #     ground_gaussian_fit = scaling_factor_ground * (
    #             weights[ground_gaussian] / (np.sqrt(2 * np.pi) * sigmas[ground_gaussian])) * np.exp(
    #         -0.5 * ((x - means[ground_gaussian]) / sigmas[ground_gaussian]) ** 2)
    #     excited_gaussian_fit = scaling_factor_excited * (
    #             weights[excited_gaussian] / (np.sqrt(2 * np.pi) * sigmas[excited_gaussian])) * np.exp(
    #         -0.5 * ((x - means[excited_gaussian]) / sigmas[excited_gaussian]) ** 2)
    #
    #     plt.plot(x, ground_gaussian_fit, label='Ground Gaussian Fit', color='blue', linewidth=2)
    #     plt.plot(x, excited_gaussian_fit, label='Excited (leakage) Gaussian Fit', color='red', linewidth=2)
    #
    #     if not dontuse_midpt_thresh:
    #         plt.axvline(pop_threshold, color='black', linestyle='--', linewidth=1,
    #                 label=f'Threshold ({pop_threshold:.2f})')
    #
    #     # Add shading for ground and excited state regions
    #     x_vals = np.linspace(np.min(ig_new), np.max(ig_new), 1000)
    #
    #     # Add shading for ground_data points
    #     plt.hist(
    #         ground_data, bins=numbins, range=[np.min(ig_new), np.max(ig_new)], density=False,
    #         alpha=0.5, color="blue", label="Ground Data Region", zorder=2
    #     )
    #
    #     # Add shading for excited_data points
    #     plt.hist(
    #         excited_data, bins=numbins, range=[np.min(ig_new), np.max(ig_new)], density=False,
    #         alpha=0.5, color="red", label="Excited Data Region", zorder=3
    #     )
    #
    #     # plt.hist(
    #     #     iq_data, bins=numbins, range=[np.min(ig_new), np.max(ig_new)], density=False,
    #     #     alpha=0.2, color="green", label="All IQ Data Region", zorder=1
    #     # )
    #     if temperature_mk is not None:
    #         plt.title(
    #             f"G-state double gaussian fit ; Qubit {q_key + 1} ; Temp= {temperature_mk:2f} mK {title_ext}")
    #     else:
    #         plt.title(
    #             f"G-state double gaussian fit ; Qubit {q_key + 1} {title_ext}")
    #
    #     plt.xlabel("$I_g$' " , fontsize=14)
    #     plt.ylabel('Counts', fontsize=14)
    #     plt.legend()
    #     # plt.show()
    #
    #     # Save the plot to the Temperatures folder
    #     plot_filename = os.path.join(qubit_folder, f"Q{q_key + 1}_SSF_gstate_gaussfit_Dataset{dataset}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    #     plt.savefig(plot_filename)
    #     # print(f"Plot saved to: {qubit_folder}")
    #     plt.close()

    def timestamp(self, fname):
        """
        Extract YYYY-MM-DD_HH-MM-SS from `fname` and return a datetime object.

        A single-line regex is compiled inside the function, so nothing sits
        at module scope.
        """
        ts_re = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
        m = ts_re.search(os.path.basename(fname))
        if m is None:
            raise ValueError("No timestamp found in: {}".format(fname))
        return datetime.datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")

    # -------------------------------- helper to extract qubit from h5 file. For qubit 1 index is zero, qubit 2 is index 1, etc.
    def qubit_of(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                # match "Q1", "Q2", ...
                if key.startswith("Q") and key[1:].isdigit():
                    return int(key[1:]) - 1
        raise ValueError(f"No Q<digit> group in {h5_path}")

    def pair_qspec_and_ssf(self, qspec_files, ssf_files, tolerance_seconds=10):
        """
        Params
        ------
        qspec_files: dict[int, list[str]] (for multiple qubits. Note: int=qubit index) OR list[str] (for a single qubit).
            If dict, keys are qubit indices and values are lists of full‐path .h5 files.
        ssf_files:  same shape as qspec_files
        tolerance_seconds: maximum allowed pairing offset in seconds
        """

        # Build per‐qubit buckets
        if isinstance(qspec_files, dict):
            qspec_by_q = {qi: list(lst) for qi, lst in qspec_files.items()}
        else:
            qspec_by_q = {}
            for f in qspec_files:
                qi = self.qubit_of(f)  # calls your existing helper
                qspec_by_q.setdefault(qi, []).append(f)

        if isinstance(ssf_files, dict):
            ssf_by_q = {qi: list(lst) for qi, lst in ssf_files.items()}
        else:
            ssf_by_q = {}
            for f in ssf_files:
                qi = self.qubit_of(f)
                ssf_by_q.setdefault(qi, []).append(f)

        pairs_by_qubit = {}
        unmatched_qspec = {}
        unmatched_ssf = {}

        # For each qubit, match QSpec → SSF by nearest‐timestamp
        for qi in set(qspec_by_q) | set(ssf_by_q):
            spec_list = sorted(qspec_by_q.get(qi, []), key=lambda p: self.timestamp(p))
            ssf_list = sorted(ssf_by_q.get(qi, []), key=lambda p: self.timestamp(p))

            spec_times = [self.timestamp(p) for p in spec_list]
            ssf_times = [self.timestamp(p) for p in ssf_list]

            matches, lonely_ssf = [], []

            # loop over SSF (anchor)
            for t_ssf, f_ssf in zip(ssf_times, ssf_list):
                idx = bisect_left(spec_times, t_ssf)
                candidates = []
                if idx < len(spec_list):
                    candidates.append((spec_times[idx], spec_list[idx]))
                if idx > 0:
                    candidates.append((spec_times[idx - 1], spec_list[idx - 1]))

                best = None
                for t_spec, f_spec in candidates:
                    delta = abs((t_spec - t_ssf).total_seconds())
                    if delta <= tolerance_seconds and (best is None or delta < best[0]):
                        best = (delta, f_spec)

                if best is not None:
                    matches.append((best[1], f_ssf))
                else:
                    lonely_ssf.append(f_ssf)

            pairs_by_qubit[qi] = matches
            unmatched_ssf[qi] = lonely_ssf
            used_spec = {qspec_path for (qspec_path, _) in matches}
            unmatched_qspec[qi] = [p for p in spec_list if p not in used_spec]

        return pairs_by_qubit, unmatched_qspec, unmatched_ssf

    def plot_ssf_SNR_vs_time(self, fit_results, plot_path, n_qubits=6):
        """
        Plot SSF readout SNR vs time for each qubit.

        Expects fit_results[q] to contain records with:
            rec["timestamp"]
            rec["ssf_SNR"]
        """

        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']
        markers = ['o', 's', '^', 'D', 'v', 'P']  # Q1-Q6
        os.makedirs(plot_path, exist_ok=True)

        fig, ax = plt.subplots(figsize=(15, 10))
        date_fmt = DateFormatter('%m-%d-%H')
        #date_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M:%S")

        if not isinstance(fit_results, dict):
            raise TypeError("fit_results must be a dictionary keyed by qubit index.")

        for q in range(n_qubits):
            records = fit_results.get(q, []) or []

            times_vals = []
            snr_vals = []

            for rec in records:
                if not isinstance(rec, dict):
                    continue

                timestamp = rec.get("timestamp", None)
                snr = rec.get("ssf_SNR", None)

                if timestamp is None or snr is None:
                    continue

                try:
                    snr = float(snr)
                except Exception:
                    continue

                if not np.isfinite(snr):
                    continue

                times_vals.append(timestamp)
                snr_vals.append(snr)

            if len(snr_vals) == 0:
                print(f"No valid SSF SNR values found for Q{q + 1}")
                continue

            ax.plot(
                times_vals,
                snr_vals,
                marker=markers[q % len(markers)],
                color=colors[q % len(colors)],
                alpha=0.7,
                markersize=5,
                linestyle="-",
                label=f"Q{q + 1}"
            )

        ax.set_title("SSF Readout SNR vs Time", fontsize=18)
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("SSF SNR", fontsize=16)

        ax.xaxis.set_major_formatter(date_fmt)
        plt.setp(ax.get_xticklabels(), rotation=45, fontsize=16)
        plt.setp(ax.get_yticklabels(), fontsize=16)

        ax.legend(fontsize=14)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        fname = os.path.join(
            plot_path,
            f"AllQubits_SSF_SNR_vs_Time_{datetime.datetime.now():%Y%m%d%H%M%S}.pdf"
        )

        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("Saved SSF SNR vs time plot to ->", fname)

    def plot_ssf_vs_time(self, fit_results, plot_path, n_qubits):
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']
        os.makedirs(plot_path, exist_ok=True)

        fig, ax = plt.subplots(figsize=(15, 10))
        date_fmt = DateFormatter('%m-%d-%H')

        if not isinstance(fit_results, dict):
            raise TypeError("fit_results must be a dictionary keyed by qubit index.")

        for q in range(n_qubits):
            records = fit_results.get(q, []) or []

            times_vals = []
            ssf_vals = []
            ssf_errs = []

            for rec in records:
                if not isinstance(rec, dict):
                    continue

                timestamp = rec.get("timestamp", None)
                ssf = rec.get("ssf_fid", None)
                ssf_err = rec.get("ssf_err_total", None)

                if timestamp is None or ssf is None:
                    continue

                if not np.isfinite(ssf):
                    continue

                if ssf_err is None or not np.isfinite(ssf_err):
                    continue

                times_vals.append(timestamp)
                ssf_vals.append(ssf)
                ssf_errs.append(ssf_err)

            if len(ssf_vals) == 0:
                print(f"No valid SSF values found for Q{q + 1}")
                continue

            ax.errorbar(
                times_vals,
                ssf_vals,
                yerr=ssf_errs,
                fmt="o",
                color=colors[q % len(colors)],
                alpha=0.7,
                capsize=3,
                markersize=5,
                linestyle="None",
                label=f"Q{q + 1}")

        ax.set_title("Single-Shot Fidelity vs Time", fontsize=18)
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("Single-Shot Fidelity", fontsize=16)

        ax.xaxis.set_major_formatter(date_fmt)
        plt.setp(ax.get_xticklabels(), rotation=45, fontsize=16)
        plt.setp(ax.get_yticklabels(), fontsize=16)

        ax.legend(fontsize=14)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        fname = os.path.join(plot_path,f"AllQubits_SSF_vs_Time_{datetime.datetime.now():%Y%m%d%H%M%S}.pdf")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved SSF vs time plot to ->", fname)

    #  Scatter plot – qubit temperatures vs. time  (all dates, each qubit its own subplot)
    def plot_qubit_temperatures_vs_time_ssf(self, all_qubit_temperatures, all_qubit_timestamps, all_qubit_temperatures_errs,
                                            out_dir, rel_err_cutoff = None, plot_error_bars = False, yaxis_min = None, yaxis_max = None):
        """Scatter plot of qubit temperatures vs. time for each qubit, optionally with error bars."""

        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(15, 10))
        date_fmt = DateFormatter('%m-%d-%H')

        for q in all_qubit_temperatures.keys():
            temps = all_qubit_temperatures[q]
            times = all_qubit_timestamps[q]
            errs = all_qubit_temperatures_errs[q]

            if not temps:
                continue

            # Filter out temperature data with error > 300 mK
            filtered = [(t, T, e)
                for t, T, e in zip(times, temps, errs)
                if T > 0 and T < 600] # and e / T < rel_err_cutoff. rel_err_cutoff is the relative error, it should be a decimal (aka 0.4 = 40% relative error and so forth)
            if not filtered:
                continue

            times_filtered, temps_filtered, errs_filtered = zip(*filtered)

            ax = plt.subplot(2, 3, q + 1)
            if plot_error_bars:
                ax.errorbar(
                    times_filtered,
                    temps_filtered,
                    yerr=errs_filtered,
                    fmt='o',
                    capsize=4,
                    markersize=5,
                    color=colors[q % len(colors)],
                    ecolor=colors[q % len(colors)],
                    label=f"Q{q + 1}"
                )
            else:
                ax.scatter(
                    times_filtered,
                    temps_filtered,
                    color=colors[q % len(colors)],
                    alpha=0.7,
                    label=f"Q{q + 1}"
                )

            ax.set_title(f"Qubit {q + 1}, SSF Method", fontsize=18)
            ax.set_xlabel("Time", fontsize=16)
            ax.set_ylabel("Effective Temperature (mK)", fontsize=16)
            # ax.grid(alpha=0.3)
            # ax.legend()
            ax.xaxis.set_major_formatter(date_fmt)
            plt.setp(ax.get_xticklabels(), rotation=45, fontsize=16)

            if yaxis_min is not None and yaxis_max is not None:
                ax.set_ylim(yaxis_min, yaxis_max)
                ax.set_yticks(np.linspace(yaxis_min, yaxis_max, 10))

            plt.setp(ax.get_yticklabels(), fontsize=16)

        plt.tight_layout()
        #plt.suptitle("SSF Effective Qubit Temps vs. Time", fontsize=16)
        fname = os.path.join(
            out_dir,
            f"AllQubits_SSF_Temps_vs_Time_{datetime.datetime.now():%Y%m%d%H%M%S}.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Saved all-dates scatter to: ", fname)

    # Histograms – temperature distributions  (all dates, each qubit subplot)
    def plot_all_Qs_qtemps_hists_ssf(self, all_qubit_temperatures, all_qubit_temperatures_errs, out_dir, bins=20, rel_err_cutoff = None):
        """
            Make per-qubit temperature histograms (SSF), with an overlaid
            inverse-variance-weighted Gaussian (same approach as your RPMs plot).

            all_qubit_temperatures: dict {qindex: [T_mK, ...]}
            all_qubit_temperatures_errs: dict {qindex: [T_err_mK, ...]}
            """
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        os.makedirs(out_dir, exist_ok=True)

        # make a 2x3 grid like before (assumes up to 6 qubits)
        fig = plt.figure(figsize=(15, 10))

        # iterate in sorted order so subplot indices are stable
        for q in sorted(all_qubit_temperatures.keys()):
            temps_list = all_qubit_temperatures.get(q, [])
            errs_list = all_qubit_temperatures_errs.get(q, [])

            if not temps_list or not errs_list:
                continue

            # --- continue-style filtering (skip on any bad condition) ---
            temp_vals = []
            temp_errs = []
            for T, e in zip(temps_list, errs_list):
                # try coercion
                try:
                    T = float(T)
                    e = float(e)
                except (TypeError, ValueError):
                    continue
                # hard bounds / invalids
                # if T <= 0 or T > 600:
                #     continue
                if T <= 0:
                    continue
                # optional relative error cutoff
                if rel_err_cutoff is not None and (e / T) > rel_err_cutoff:
                    continue
                temp_vals.append(T)
                temp_errs.append(e)

            if len(temp_vals) == 0:
                continue

            temps = np.asarray(temp_vals, dtype=float)
            errs = np.asarray(temp_errs, dtype=float)

            # # --- Weighted mean/std (same recipe as RPMs) ---
            # err_floor = 1e-12
            # safe_errs = np.clip(errs, err_floor, np.inf)
            # # clip tiny errors (robustness)
            # low_clip_percentile = 1.0
            # clip_threshold = np.nanpercentile(safe_errs, low_clip_percentile)
            # safe_errs = np.maximum(safe_errs, clip_threshold)
            # weights = 1.0 / (safe_errs ** 2)
            #
            # mu = np.sum(weights * temps) / np.sum(weights)
            # var = np.sum(weights * (temps - mu) ** 2) / np.sum(weights)
            # std = np.sqrt(var)

            # ---------------------------Weighted mean with robust median-MAD clipping---------------------------
            n_counts = len(temps)

            # keep only finite pairs
            finite = np.isfinite(temps) & np.isfinite(errs)
            temps, errs = temps[finite], errs[finite]
            if temps.size == 0:
                mu_1, std_1 = np.nan, np.nan
            else:
                # robust outlier clip around the median
                k = 2.0  # 2-4  is typical; lower = stricter
                med = np.median(temps)
                mad = np.median(np.abs(temps - med))
                if mad == 0:
                    mad = max(np.std(temps), 1e-12)
                keep = np.abs(temps - med) < k * mad
                temps, errs = temps[keep], errs[keep]

                if temps.size == 0:
                    mu_1, std_1 = np.nan, np.nan
                else:
                    # compute weights and weighted mean/std (using 1/err)
                    err_floor = 1e-12
                    safe_errs = np.clip(errs, err_floor, np.inf)
                    weights = 1.0 / safe_errs

                    w_sum = np.nansum(weights)
                    mu_1 = float(np.nansum(weights * temps) / w_sum)

                    var = float(np.nansum(weights * (temps - mu_1) ** 2) / w_sum)
                    std_1 = float(np.sqrt(max(var, 0.0)))

            # --- Histogram (raw counts) ---
            ax = plt.subplot(2, 3, q + 1)
            hist_data, edges = np.histogram(temps, bins=bins)
            bin_width = np.diff(edges)[0]

            ax.hist(temps,
                    bins=bins,
                    alpha=0.7,
                    color=colors[q % len(colors)],
                    edgecolor='black',
                    label="Counts")

            # --- Weighted Gaussian overlay, area-matched to histogram ---
            x_vals = np.linspace(temps.min(), temps.max(), 400)
            pdf_vals = norm.pdf(x_vals, mu_1, std_1)
            scale_factor = len(temps) * bin_width  # area-match
            scaled_pdf = pdf_vals * scale_factor
            ax.plot(x_vals, scaled_pdf, linestyle='--', linewidth=2,
                    color='black', label='Weighted Gaussian fit')

            ax.set_title(f"Q{q + 1}  µ={mu_1:.2f} mK,  s={std_1:.2f} mK, c: {n_counts}", fontsize=14)
            ax.set_xlabel("Temperature (mK)")
            ax.set_ylabel("Count")
            ax.grid(alpha=0.3)
            # ax.legend()

        plt.tight_layout()
        fname = os.path.join(out_dir, f"AllQubits_SSFTemps_Hist_{datetime.datetime.now():%Y%m%d%H%M%S}.png")
        plt.savefig(fname, dpi=300)
        plt.close(fig)
        print("Saved all-dates histogram to:", fname)

    def plot_SSF_fid_vs_Pe_viaSSF(self, fit_results, plot_path, n_qubits=6, plot_together=True, sharex=False, sharey=True):
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']
        markers = ['o', 's', '^', 'D', 'v', 'P']
        os.makedirs(plot_path, exist_ok=True)

        if not isinstance(fit_results, dict):
            raise TypeError("fit_results must be a dictionary keyed by qubit index.")

        # ---------------- Helper to extract valid data ----------------
        def extract_qubit_data(q):
            records = fit_results.get(q, []) or []

            pe_vals = []
            pe_errs = []
            ssf_vals = []
            ssf_errs = []

            for rec in records:
                if not isinstance(rec, dict):
                    continue

                pe = rec.get("Pe", None)
                pe_err = rec.get("total_sigma_Pe", None)

                ssf = rec.get("ssf_fid", None)
                ssf_err = rec.get("ssf_err_total", None)

                if pe is None or ssf is None:
                    continue

                if not np.isfinite(pe) or not np.isfinite(ssf):
                    continue

                if pe_err is None or not np.isfinite(pe_err):
                    continue

                if ssf_err is None or not np.isfinite(ssf_err):
                    continue

                pe_vals.append(pe)
                pe_errs.append(pe_err)
                ssf_vals.append(ssf)
                ssf_errs.append(ssf_err)

            return pe_vals, pe_errs, ssf_vals, ssf_errs

        # =====================================================================
        # Option 1: plot all qubits together
        # =====================================================================
        if plot_together:
            fig, ax = plt.subplots(figsize=(11, 8), sharex=sharex, sharey=sharey)

            for q in range(n_qubits):
                pe_vals, pe_errs, ssf_vals, ssf_errs = extract_qubit_data(q)

                if len(ssf_vals) == 0:
                    print(f"No valid SSF/Pe values found for Q{q + 1}")
                    continue

                ax.errorbar(
                    pe_vals,
                    ssf_vals,
                    xerr=pe_errs,
                    yerr=ssf_errs,
                    fmt=markers[q % len(markers)],
                    color=colors[q % len(colors)],
                    alpha=0.7,
                    capsize=3,
                    markersize=5,
                    markeredgecolor="k",
                    linestyle="None",
                    label=f"Q{q + 1}"
                )

            ax.set_title("Single-Shot Fidelity vs $P_e$ (via SSF)", fontsize=18)
            ax.set_xlabel("$P_e$", fontsize=16)
            ax.set_ylabel("Single-Shot Fidelity", fontsize=16)
            ax.set_ylim(0.6, 1.0)
            ax.set_xlim(0.0, 0.07)

            plt.setp(ax.get_xticklabels(), fontsize=16)
            plt.setp(ax.get_yticklabels(), fontsize=16)

            ax.legend(fontsize=14)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            fname = os.path.join(plot_path,f"AllQubits_SSF_vs_Pe_{datetime.datetime.now():%Y%m%d%H%M%S}.pdf")
            plt.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print("Saved combined SSF vs Pe plot to ->", fname)
            return fname

        # =====================================================================
        # Option 2: plot each qubit separately as subplots in one figure
        # =====================================================================
        else:
            nrows = 2
            ncols = 3

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(15, 10),
                sharex=sharex,
                sharey=sharey,
                constrained_layout=True
            )

            axes = np.atleast_1d(axes).ravel()

            for q in range(n_qubits):
                ax = axes[q]

                pe_vals, pe_errs, ssf_vals, ssf_errs = extract_qubit_data(q)

                if len(ssf_vals) == 0:
                    print(f"No valid SSF/Pe values found for Q{q + 1}")
                    ax.set_title(f"Q{q + 1}", loc="left", fontsize=18, fontweight="bold")
                    ax.grid(alpha=0.3)
                    continue

                ax.errorbar(
                    pe_vals,
                    ssf_vals,
                    xerr=pe_errs,
                    yerr=ssf_errs,
                    fmt=markers[q % len(markers)],
                    markersize=5,
                    elinewidth=1,
                    capsize=3,
                    alpha=0.85,
                    color=colors[q % len(colors)],
                    ecolor=colors[q % len(colors)],
                    markeredgecolor="k",
                    linestyle="None",
                    label=f"Q{q + 1}"
                )

                ax.set_title(f"Q{q + 1}", loc="left", fontsize=18, fontweight="bold")
                ax.set_xlabel("$P_e$")
                ax.set_ylabel("Single-Shot Fidelity")
                ax.set_ylim(0.6, 1.0)
                ax.set_xlim(0.0, 0.07)
                ax.grid(alpha=0.3)
                ax.legend(loc="best", fontsize=14, frameon=False)
                ax.tick_params(axis="both", labelsize=16)

            # Hide unused subplot panels, if any
            for k in range(n_qubits, len(axes)):
                axes[k].set_visible(False)

            # Hide inner axis labels/tick labels for shared axes
            # for ax in axes:
            #     ax.label_outer()

            fig.suptitle("Single-Shot Fidelity vs $P_e$ (via SSF)", fontsize=16)
            fname = os.path.join(plot_path,f"Subplots_SSF_vs_Pe_{datetime.datetime.now():%Y%m%d%H%M%S}.pdf")
            plt.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("Saved subplot SSF vs Pe plot to ->", fname)

    def plot_all_Qs_Pe_hists_ssf(
            self,
            SSF_double_gauss_fit_results,
            out_dir,
            bins=20,
            rel_err_cutoff=None,
            only_return_mu_and_sigma=True,
            make_plot=True,
            save_plot=True,
            robust_clip=True,
            clip_k=2.0,
    ):
        """
        Per-qubit Pe histograms (SSF) + inverse-variance-weighted Gaussian overlay.

        Returns (when only_return_mu_and_sigma=True):
          Pe_dist_err_dict = { q: {"mu_w":..., "sigma_w":..., "n_kept":..., "n_raw":...}, ... }

        SSF input format:
          SSF_double_gauss_fit_results[q] = [ {"Pe": float, "total_sigma_Pe": float, "timestamp": datetime, ...}, ... ]
        """
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        os.makedirs(out_dir, exist_ok=True)

        Pe_dist_err_dict = {}

        # Only allocate a figure if we actually need to plot
        fig = None
        if make_plot:
            fig = plt.figure(figsize=(15, 10))

        for q in sorted(SSF_double_gauss_fit_results.keys()):
            entries = SSF_double_gauss_fit_results.get(q, [])
            if not entries:
                continue

            # raw counts = number of entries with dict type (before filtering)
            n_raw = sum(1 for d in entries if isinstance(d, dict))

            # extract + filter
            Pe_vals = []
            Pe_errs = []

            for d in entries:
                if not isinstance(d, dict):
                    continue

                P = d.get("Pe", None)
                e = d.get("total_sigma_Pe", None)

                try:
                    P = float(P)
                    e = float(e)
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(P) or not np.isfinite(e):
                    continue

                # physical-ish bounds for population
                if P <= 0.0 or P >= 1.0:
                    continue

                if rel_err_cutoff is not None and (e / max(P, 1e-300)) > rel_err_cutoff:
                    continue

                Pe_vals.append(P)
                Pe_errs.append(e)

            if len(Pe_vals) == 0:
                continue

            Pe_vals = np.asarray(Pe_vals, dtype=float)
            Pe_errs = np.asarray(Pe_errs, dtype=float)

            # keep only finite pairs (redundant but safe)
            finite = np.isfinite(Pe_vals) & np.isfinite(Pe_errs)
            Pe_vals, Pe_errs = Pe_vals[finite], Pe_errs[finite]
            if Pe_vals.size == 0:
                continue

            # robust median-MAD clipping (optional)
            if robust_clip:
                med = np.median(Pe_vals)
                mad = np.median(np.abs(Pe_vals - med))
                if mad == 0:
                    mad = max(np.std(Pe_vals), 1e-12)
                keep = np.abs(Pe_vals - med) < clip_k * mad
                Pe_vals, Pe_errs = Pe_vals[keep], Pe_errs[keep]

            n_kept = int(Pe_vals.size)
            if n_kept == 0:
                continue

            # inverse-variance weights
            err_floor = 1e-12
            safe_errs = np.clip(Pe_errs, err_floor, np.inf)
            weights = 1.0 / (safe_errs ** 2)

            # for 1/sigma weights choice (less sensitive to outliers with small errs)
            # safe_errs = np.clip(Pe_errs, err_floor, np.inf)
            # weights = 1.0 / safe_errs

            w_sum = np.nansum(weights)
            if not np.isfinite(w_sum) or w_sum <= 0:
                continue

            mu_w = float(np.nansum(weights * Pe_vals) / w_sum)
            var_w = float(np.nansum(weights * (Pe_vals - mu_w) ** 2) / w_sum)
            sigma_w = float(np.sqrt(max(var_w, 0.0)))

            Pe_dist_err_dict[int(q)] = {
                "mu_w": mu_w,
                "sigma_w": sigma_w,
                "n_kept": n_kept,
                "n_raw": int(n_raw),
            }

            # ---- Plot if requested ----
            if make_plot:
                ax = plt.subplot(2, 3, int(q) + 1)

                hist_data, edges = np.histogram(Pe_vals, bins=bins)
                bin_width = np.diff(edges)[0] if len(edges) > 1 else 1.0

                ax.hist(
                    Pe_vals,
                    bins=bins,
                    alpha=0.7,
                    color=colors[int(q) % len(colors)],
                    edgecolor='black',
                    label="Counts"
                )

                # weighted Gaussian overlay, scaled to histogram area
                if np.isfinite(mu_w) and np.isfinite(sigma_w) and sigma_w > 0:
                    x_vals = np.linspace(Pe_vals.min(), Pe_vals.max(), 400)
                    pdf_vals = norm.pdf(x_vals, mu_w, sigma_w)
                    scale_factor = len(Pe_vals) * bin_width
                    ax.plot(x_vals, pdf_vals * scale_factor, linestyle='--', linewidth=2,
                            color='black', label='Weighted Gaussian fit')

                ax.set_title(f"Q{int(q) + 1}  µ={mu_w:.4f},  s={sigma_w:.4f}, n={n_kept}", fontsize=14)
                ax.set_xlabel("Thermal Population ($P_e$)")
                ax.set_ylabel("Count")
                ax.grid(alpha=0.3)

        # Save plot only if requested
        if make_plot and save_plot:
            fname = os.path.join(out_dir, f"AllQubits_SSF_Pe_Hists_{datetime.datetime.now():%Y%m%d%H%M%S}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=300)
            plt.close(fig)
            print("Saved all-dates SSF histogram to:", fname)
        elif make_plot:
            plt.close(fig)

        # Return stats (slim dict)  this is what you need for inflation/updating later
        if only_return_mu_and_sigma:
            return Pe_dist_err_dict

        # If you later want to return more, you can expand this
        return Pe_dist_err_dict

    # def plot_temp_histograms(self, qubit_temperatures, out_dir, bins=20):
    #     """
    #     Parameters
    #     ----------
    #     qubit_temperatures : dict {qubit: [(temp_mK, unix_ts), …]}
    #     out_dir            : str   folder that will receive the PNG
    #     colors             : list  colour per qubit (defaults if None)
    #     bins               : int   histogram bins
    #     """
    #     colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
    #
    #     os.makedirs(out_dir, exist_ok=True)
    #
    #     plt.figure(figsize=(15, 10))
    #     for q, data in qubit_temperatures.items():
    #         temps = [t for t, _ in data]
    #         plt.subplot(2, 3, q + 1)
    #         plt.hist(temps, bins=bins, color=colors[q], alpha=0.7,
    #                  edgecolor='black')
    #         plt.title(f"Qubit {q + 1} Temperature Distribution")
    #         plt.xlabel("Temperature (mK)")
    #         plt.ylabel("Count")
    #         plt.grid(alpha=0.3)
    #
    #     plt.tight_layout()
    #     fname = os.path.join(
    #         out_dir,
    #         f"Temperature_Histograms_{datetime.datetime.now():%Y%m%d%H%M%S}.png")
    #     plt.savefig(fname, dpi=300)
    #     plt.close()
    #     print("Saved histogram →", fname)

    # def plot_temp_scatter(self, qubit_temperatures, out_dir):
    #     """
    #     Parameters
    #     ----------
    #     qubit_temperatures : dict {qubit: [(temp_mK, unix_ts), …]}
    #     out_dir            : str   folder that will receive the PNG
    #     colors             : list  color per qubit (defaults if None)
    #     """
    #     colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
    #
    #     os.makedirs(out_dir, exist_ok=True)
    #
    #     plt.figure(figsize=(15, 10))
    #     date_fmt = DateFormatter('%m-%d\n%H:%M')
    #
    #     for q, data in qubit_temperatures.items():
    #         if not data:
    #             continue
    #         temps, ts = zip(*data)
    #         times = [datetime.datetime.fromtimestamp(t) for t in ts]
    #
    #         ax = plt.subplot(2, 3, q + 1)
    #         ax.scatter(times, temps, color=colors[q], alpha=0.7, edgecolor='black')
    #         ax.set_title(f"Qubit {q + 1} Temperature vs Time")
    #         ax.set_xlabel("Time")
    #         ax.set_ylabel("Temperature (mK)")
    #         ax.grid(alpha=0.3)
    #         ax.xaxis.set_major_formatter(date_fmt)
    #         plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    #
    #     plt.tight_layout()
    #     fname = os.path.join(
    #         out_dir,
    #         f"Temperature_Scatter_{datetime.datetime.now():%Y%m%d%H%M%S}.png")
    #     plt.savefig(fname, dpi=300)
    #     plt.close()
    #     print("Saved scatter →", fname)

    def plot_ssf_ge_thresh_split_gstate(self, q_key: int,rec: dict, out_folder: str):
        """
        Plot a simple ig_new histogram split at the g-e SSF threshold.

        Parameters
        ----------
        q_key : int
            Zero-based qubit index (so Q1→0, Q5→4, etc).
        rec : dict
            One entry from fit_results, must contain
            "ig_new", "crossing_point", and "dataset".
        out_folder : str
            Directory where the .png should be saved.
        """
        print('Processing plots...')
        ig = rec["ig_new"]  # rotated SSF I values for prepared ground state
        thresh = rec["pop_threshold"]  # data_threshold
        temp_mk = rec["temperature_mK"]
        dataset = rec["dataset"]

        steps = 3000
        # numbins = round(math.sqrt(steps))
        numbins = 64

        fig, ax = plt.subplots()

        bin_edges = np.histogram_bin_edges(ig, bins=numbins)

        ax.hist(ig, bins=bin_edges, alpha=0.3, color="grey", label="all g-state data", zorder=1)
        ax.hist(ig[ig <= thresh], bins=bin_edges, alpha=0.7, label="|g⟩ region", color="blue", zorder=2)
        ax.hist(ig[ig > thresh], bins=bin_edges, alpha=0.7, label="|e⟩ leakage region", color="red", zorder=3)

        ax.axvline(thresh, linestyle="--", color="black", label=f"ssf g-e threshold={thresh:.2f}")
        ax.set_title(f"Method: g-e double gaussian fit ; Q{q_key + 1}; Temp= {temp_mk:2f} mK")
        ax.set_xlabel("$I_g$'")
        ax.set_ylabel("Counts")
        ax.legend()

        os.makedirs(out_folder, exist_ok=True)
        fname = os.path.join( out_folder, f"Q{q_key + 1}_SSF_ge_threshold_split_{dataset}.png" )
        fig.savefig(fname, dpi=self.figure_quality)
        plt.close(fig)

    def single_gaussian_wthresh(self, iq_data: np.ndarray, k_sigma: float = 3.0, n_points: int = 500):
        """
        Fit a single Gaussian to iq_data (ig_new), choose threshold = μ + k_sigma·σ,
        and also return x & y arrays for the fitted Gaussian curve.

        Returns
        -------
        Pg : float
          P(|g⟩) = fraction of points ≤ thresh
        Pe : float
          P(|e⟩) = 1 − Pg
        thresh : float
          μ + k_sigma·σ
        mu : float
          mean of iq_data
        sigma : float
          std­dev of iq_data
        ground_data : np.ndarray
        excited_data : np.ndarray
        x_gauss : np.ndarray
          abscissa for Gaussian curve
        y_gauss : np.ndarray
          ordinate (pdf) of Gaussian at x_gauss
        """
        #fit mean & std
        mu = np.mean(iq_data)
        sigma = np.std(iq_data, ddof=1)

        #define threshold
        thresh = mu + k_sigma * sigma

        #calculate populations
        Pg = np.mean(iq_data <= thresh)
        Pe = 1.0 - Pg

        #split data
        ground_data = iq_data[iq_data <= thresh]
        excited_data = iq_data[iq_data > thresh]

        #build Gaussian curve
        x_gauss = np.linspace(iq_data.min(), iq_data.max(), n_points)
        y_gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_gauss - mu) / sigma) ** 2)

        return Pg, Pe, thresh, mu, sigma, ground_data, excited_data, x_gauss, y_gauss

    def fit_double_gaussian_midpoint(self, iq_data): #iq_data is either ig_new or ie_new (IQ data post-rotation)
        """
        This function can fit the SSF ground state data to a double gaussian, and calculate the population threshold by
        finding the midpoint between the means of the two gaussians. Can also be used on the First excited state SSF data but
        for qubit temperature calculations the user should only provide ig_new.

        Serves the same purpose as fit_double_gaussian_with_full_coverage(), but sets the population threshold
        as the midpoint between the two Gaussian means instead of the crossing point of the two gaussians.

        Returns:
          Pg, Pe, gmm, means, sigmas, weights,
          threshold_mid, threshold_mid_err, ground_gaussian, excited_gaussian,
          ground_data, excited_data, iq_data

          Note: threshold_mid_err is the 1-σ uncertainty on `threshold_mid`, estimated via GMM responsibilities.
          On the other hand, 'sigmas' contains the sigma value of each gaussian in the double gaussian fit.
        """
        # fit GMM
        gmm = GaussianMixture(n_components=2)
        gmm.fit(iq_data.reshape(-1, 1))

        means = gmm.means_.flatten()
        sigmas = np.sqrt(gmm.covariances_).flatten()
        weights = gmm.weights_

        # identify which component is "ground" (lower mean)
        ground_gaussian = np.argmin(means)
        excited_gaussian = 1 - ground_gaussian

        #compute midpoint threshold
        threshold_mid = 0.5 * (means[ground_gaussian] + means[excited_gaussian])

        # ----------------------------- Estimate each gaussian mean’s uncertainty using responsibilities ------------------------
        all_i = iq_data.reshape(-1, 1)
        resp = gmm.predict_proba(all_i)  # shape = (Nshots, 2)
        rg = resp[:, ground_gaussian]  # “ground” responsibility per shot
        re = resp[:, excited_gaussian]  # “excited” responsibility per shot

        N_g = rg.sum()  # effective number of points in ground cluster
        N_e = re.sum()  # effective number of points in excited cluster

        sigma_g = sigmas[ground_gaussian]
        sigma_e = sigmas[excited_gaussian]

        # σ_{μ_g} ≈ σ_g / sqrt(N_g), σ_{μ_e} ≈ σ_e / sqrt(N_e)
        sigma_mu_g = sigma_g / np.sqrt(N_g) if N_g > 0 else 0.0
        sigma_mu_e = sigma_e / np.sqrt(N_e) if N_e > 0 else 0.0

        # Propagate into σ_threshold = ½ * sqrt(σ_{μ_g}² + σ_{μ_e}²)
        threshold_mid_err = 0.5 * np.sqrt(sigma_mu_g ** 2 + sigma_mu_e ** 2)
        #------------------------------------------------------------------------

        # Split using threshold
        ground_data = iq_data[iq_data <= threshold_mid]
        excited_data = iq_data[iq_data > threshold_mid]

        # Compute populations
        Pg = len(ground_data) / len(iq_data)
        Pe = len(excited_data) / len(iq_data)

        return Pg, Pe, gmm, means, sigmas, weights, threshold_mid, threshold_mid_err, ground_gaussian, excited_gaussian, ground_data, excited_data, iq_data

    def ssf_fit_two_gaussians_midpoint(self, ig_new: np.ndarray, ie_new: np.ndarray):
        """
        Fits a two component GMM (double gaussian) to all shots (ig_new + ie_new) and chooses the
        threshold as the midpoint between the two component means.
        Also returns a 1-σ error on that midpoint.

        Returns
        -------
        thresh           : (μ_g + μ_e) / 2
        thresh_err       : The 1-sigma uncertainty on that midpoint threshold, estimated by
                           propagating the GMM-responsibility-based errors of each Gaussian mean.
        means, sigmas    : np.ndarray shape (2,)
        weights          : np.ndarray shape (2,)
        ground_idx       : component index for ground cluster
        excited_idx      : component index for excited cluster
        """

        # Fit a 2‑component Gaussian mixture
        all_i = np.concatenate([ig_new, ie_new]).reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, covariance_type="full")
        gmm.fit(all_i)

        means = gmm.means_.flatten()
        sigmas = np.sqrt(gmm.covariances_).flatten()
        weights = gmm.weights_

        ground_idx, excited_idx = np.argsort(means)  # smaller mean = ground
        mu_g, mu_e = means[ground_idx], means[excited_idx]
        sigma_g = sigmas[ground_idx]
        sigma_e = sigmas[excited_idx]

        # Mid‑point threshold
        threshold = 0.5 * (mu_g + mu_e)

        #-----------Estimate σ_mean for each Gaussian via responsibilities (the uncertaintiy of each mean)----
        resp = gmm.predict_proba(all_i)  # shape = (Nshots, 2)
        rg = resp[:, ground_idx]  # “ground” responsibility per shot
        re = resp[:, excited_idx]  # “excited” responsibility per shot

        N_g = rg.sum()  # effective number of points in ground cluster
        N_e = re.sum()  # effective number of points in excited cluster

        # σ_{μ_g} ≈ σ_g / sqrt(N_g), σ_{μ_e} ≈ σ_e / sqrt(N_e)
        sigma_mu_g = sigma_g / np.sqrt(N_g) if N_g > 0 else 0.0
        sigma_mu_e = sigma_e / np.sqrt(N_e) if N_e > 0 else 0.0

        # Propagate into σ_threshold = ½ * sqrt(σ_{μ_g}² + σ_{μ_e}²)
        thresh_err = 0.5 * np.sqrt(sigma_mu_g ** 2 + sigma_mu_e ** 2)
        # ---------------------------------------------------------------------------------------------------

        return threshold, thresh_err, means, sigmas, weights, ground_idx, excited_idx

    def process_ssf_and_qfreq_data_qtemps(self, Science_Qubits, paths):
        """
        This function processes the ssf and g-e quit spec data for each qubit found inside the files in CEPH and returns the dictionary:

        pairs_info[q].append({
            "qspec_path": qspec_path,
            "ssf_path"  : ssf_path,
            "qfreq_MHz" : freq_cache[fq_key],     # MHz
            "qfreq_MHz_err": (1-sigma error on that freq),
            "ig_new"   : ig_new_cache[ss_key],
            "ie_new": ie_new_cache[ss_key],
            "data_timestamp" : timestamp_ssf_cache[ss_key].timestamp(), # unix-timestamps
        })

        The dictionary contains matched up SSF and g-e qubit spec h5 files that are within a specified number of seconds (tolerance_seconds). That way the user can
        use the returned dictionary to calculate qubit temperatures using SSF data and the qubit freq that was measured at around the same time that the SSF data was taken.
        """
        print('Processing SSF and g-e quit spec data for temperature analysis...')
        freq_cache = {}  # for qubit freqs (MHz)
        freq_err_cache = {}  # 1-? error (std) on that freq
        ig_new_cache = {}  # for ground state roated I data (SSF)
        ie_new_cache = {}  # for first excited state roated I data (SSF)
        timestamp_ssf_cache = {}  # for ssf data time stamps (qubit temperature time stamps)
        ssf_fid_cache = {}  # for single shot fidelity values

        #-- To store SSF errors --
        ssf_err_shot_cache = {}
        ssf_err_bins_cache = {}
        ssf_err_fit_cache = {}
        ssf_err_total_cache = {}
        ssf_vs_bins_cache = {}

        if self.run_num == 4 or self.run_num ==5:
            folder_qspec = "study_data"
            expt_name_qspec = "qspec_ge"
            datagroup_qspec = 'QSpec'

            expt_name_ssf = "ss_ge"
            datagroup_ssf = 'SS'
            folder_ssf = "study_data"

            tolerance_seconds = 10

        elif self.run_num == 6:
            # Identify whether we're processing pre-science-run RR data by path substring
            presr_tag = "ge_round_robin_presciencerun_data"
            is_preSR = any(presr_tag in str(p) for p in (paths or []))

            if is_preSR:
                folder_qspec = "study_data"
            else:
                folder_qspec = "optimization"
            expt_name_qspec = "qspec_ge"
            datagroup_qspec = 'QSpec'

            if is_preSR:
                folder_ssf = "study_data"
            else:
                folder_ssf = "optimization"
            expt_name_ssf = "ss_ge"
            datagroup_ssf = 'SS'

            if is_preSR:
                tolerance_seconds = 10
            else:
                tolerance_seconds = 600

        elif self.run_num == 7:
            folder_qspec = "study_data"
            expt_name_qspec = "qspec_ge"
            datagroup_qspec = 'QSpec'

            expt_name_ssf = "ss_ge"
            datagroup_ssf = 'SS'
            folder_ssf = "study_data"

            tolerance_seconds = 10

        elif self.run_num == 8:
            folder_qspec = "study_data"
            expt_name_qspec = "qspec_ge"
            datagroup_qspec = 'QSpec'

            expt_name_ssf = "ss_ge"
            datagroup_ssf = 'SS'
            folder_ssf = "study_data"

            tolerance_seconds = 10

        elif self.run_num == 9:
            folder_qspec = "study_data"
            expt_name_qspec = "qspec_ge"
            datagroup_qspec = 'QSpec'

            expt_name_ssf = "ss_ge"
            datagroup_ssf = 'SS'
            folder_ssf = "study_data"

            tolerance_seconds = 10

        elif self.run_num == 9.2: # run 9c
            folder_qspec = "study_data"
            expt_name_qspec = "qspec_ge"
            datagroup_qspec = 'QSpec'

            expt_name_ssf = "ss_ge"
            datagroup_ssf = 'SS'
            folder_ssf = "study_data"

            tolerance_seconds = 10

        else:
            raise ValueError("You must choose run_num = 4,5,6,7, 8, 9 or 9.2 (run 9c). Otherwise, define a section for your run of interest inside process_ssf_and_qfreq_data_qtemps().")

        for full_path in paths:

            # Check existence of Data_h5 folder before doing anything else
            ssf_data_h5_path = os.path.join(full_path, folder_ssf, "Data_h5", expt_name_ssf)
            qspec_data_h5_path = os.path.join(full_path, folder_qspec, "Data_h5", expt_name_qspec)

            if not os.path.isdir(ssf_data_h5_path):
                print(f"Skipping {full_path}; SSF Data_h5 folder missing: {ssf_data_h5_path}")
                continue

            if not os.path.isdir(qspec_data_h5_path):
                print(f"Skipping {full_path}; QSpec Data_h5 folder missing: {qspec_data_h5_path}")
                continue

            path = os.path.dirname(full_path)  # one level up from the dataset
            dataset = os.path.basename(full_path)  # just the '2025-04-16_11-47-09' part

            for QubitIndex in Science_Qubits:  # We are only taking science data for some qubits
                # Run 9 patch: accidentally took punched-out data for Q4 in this dataset.
                # QubitIndex == 3 corresponds to Q4.
                if (QubitIndex == 3 and "AB_paper_data_batch1_25dB_DACatten_noQ5/2026-04-17_00-34-47" in str(full_path).replace("\\", "/")):
                    print(f"Skipping Q4 data due to punchout in {full_path}", flush=True)
                    continue
                try:
                    # --- Load QSpec ---
                    qspec_obj = AnaQSpec(path, dataset, QubitIndex, folder_qspec, expt_name_qspec, datagroup_qspec)
                    qspec_data = qspec_obj.load_all()

                    qspec_dates = qspec_data["dates"]
                    qspec_n = int(qspec_data["n"])
                    qspec_probe_freqs = qspec_data["probe_freqs"]
                    qspec_I = qspec_data["I"]
                    qspec_Q = qspec_data["Q"]

                    qspec_freqs, qspec_errs, qspec_fwhms = qspec_obj.get_all_qspec_freq(qspec_probe_freqs, qspec_I, qspec_Q, qspec_n)

                    # recreate the list of file–paths in the SAME order the helper used
                    h5_files, data_path, n = find_h5_files(
                        path, dataset, expt_name_qspec,
                        folder=folder_qspec,
                        qubit_index=QubitIndex)
                    h5_paths = [os.path.join(data_path, f) for f in h5_files]

                    for i in range(qspec_n):
                        freq_cache[(h5_paths[i], QubitIndex)] = qspec_freqs[i]
                        freq_err_cache[(h5_paths[i], QubitIndex)] = qspec_errs[i]
                except Exception as e:
                    print(f"Error extracting Qspec data in {dataset} for Q{QubitIndex + 1}: {e}")
                    #raise ValueError(f"Skipped QSpec scan in {dataset} for Q{QubitIndex}: {e}") # for debugging

                try:
                    # --- Load SSF ---
                    ssf_ge = AnaSSF(path, dataset, QubitIndex, folder_ssf, expt_name_ssf, datagroup_ssf)
                    ssf_data= ssf_ge.load_all()

                    ssf_dates = ssf_data["dates"]
                    ssf_n = int(ssf_data["n"])  # convert here if needed
                    I_g = ssf_data["I_g"]
                    Q_g = ssf_data["Q_g"]
                    I_e = ssf_data["I_e"]
                    Q_e = ssf_data["Q_e"]

                    # recreate the list of SSF-file paths in the SAME order the helper used
                    ssf_files, ssf_data_path, ssf_n2 = find_h5_files(
                        path, dataset, expt_name_ssf,
                        folder=folder_ssf,
                        qubit_index=QubitIndex)
                    ssf_paths = [os.path.join(ssf_data_path, f) for f in ssf_files]

                    # iterate through every round (file)
                    for i in range(ssf_n):
                        try:
                            ssf_results = ssf_ge.get_ssf_in_round(I_g, Q_g, I_e, Q_e, i)
                        except Exception as e:
                            print(f"rotate-Ig failed ({ssf_paths[i]}): {e}")
                            continue

                        ig_new = ssf_results["ig_new"]
                        ie_new = ssf_results["ie_new"]
                        fid = ssf_results["ssf"]
                        fid_err_shot = ssf_results["ssf_err_shot"]
                        fid_err_bins = ssf_results["ssf_err_bins"]
                        fid_err_fit = ssf_results["ssf_err_fit"]
                        fid_err_total = ssf_results["ssf_err_total"]
                        fid_bins = ssf_results["ssf_vs_bins"]

                        key = (ssf_paths[i], QubitIndex)
                        ig_new_cache[key] = ig_new
                        ie_new_cache[key] = ie_new
                        ssf_fid_cache[key] = fid
                        ssf_err_shot_cache[key] = fid_err_shot
                        ssf_err_bins_cache[key] = fid_err_bins
                        ssf_err_fit_cache[key] = fid_err_fit
                        ssf_err_total_cache[key] = fid_err_total
                        ssf_vs_bins_cache[key] = fid_bins
                        timestamp_ssf_cache[key] = ssf_dates[i]

                except Exception as e:
                    print(f"Error extracting SSF data in {dataset} for Q{QubitIndex + 1}: {e}")
                    #raise ValueError(f"Failed loading SSF for qubit {QubitIndex} from {full_path}: {e}") # for debugging

        # Organize files by type and qubit index after loading all the data
        qspec_h5s = {q: [] for q in Science_Qubits}
        for (path, qidx) in freq_cache.keys():
            qspec_h5s[qidx].append(path)

        ssf_h5s = {q: [] for q in Science_Qubits}
        for (path, qidx) in ig_new_cache.keys():
            ssf_h5s[qidx].append(path)

        ########################################## Pair up Qspec_ge data and ssf_ge h5 files ###########################################
        pairs_by_qubit, lonely_qspec, lonely_ssf = self.pair_qspec_and_ssf(qspec_h5s, ssf_h5s, tolerance_seconds=tolerance_seconds)

        # Store relevant info for these pairs in a dictionary
        pairs_info = {q: [] for q in Science_Qubits}
        for q in Science_Qubits:
            for qspec_path, ssf_path in pairs_by_qubit.get(q, []):
                fq_key = (qspec_path, q)
                ss_key = (ssf_path, q)
                if fq_key not in freq_cache or ss_key not in ig_new_cache:
                    continue  # skip incomplete pair

                pairs_info[q].append({
                    "qspec_path": qspec_path,
                    "ssf_path": ssf_path,
                    "qfreq_MHz": freq_cache[fq_key],  # MHz
                    "qfreq_MHz_err": freq_err_cache[fq_key],  # 1-sigma (standard deviation) fit error on "qfreq_MHz_err"
                    "ig_new": ig_new_cache[ss_key],
                    "ie_new": ie_new_cache[ss_key],
                    "ssf_fid": ssf_fid_cache[ss_key], # SSF value
                    "ssf_err_shot": ssf_err_shot_cache[ss_key], # finite-shot/statistical uncertainty on SSF from counting shots relative to the chosen threshold
                    "ssf_err_bins": ssf_err_bins_cache[ss_key], # SSF uncertainty from sensitivity to the histogram binning choice
                    "ssf_err_fit": ssf_err_fit_cache[ss_key], # additional SSF uncertainty from the fitted Gaussian threshold; zero for max_contrast
                    "ssf_err_total": ssf_err_total_cache[ss_key], # total SSF uncertainty, combining shot, binning, and fit contributions in quadrature
                    "ssf_vs_bins": ssf_vs_bins_cache[ss_key], # diagnostic array of SSF values obtained when recalculating SSF with different numbins values
                    "data_timestamp": timestamp_ssf_cache[ss_key].timestamp(),  # unix-timestamps
                })
        return pairs_info

    def gaussian_pdf(self, x, mu, sigma):
        """Normalized 1D Gaussian pdf."""
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

    def fit_ssf_ge_double_gaussian_SNR_iminuit(self, ig_new, ie_new, q_key, plot = False, qubit_folder = "SSF_SNR_fit_plots",
                                                dataset = None, ssf_hist_ylim = None):
        """
        Fit prepared |g> and prepared |e> SSF data together as a two-Gaussian
        mixture using iminuit, then calculate readout SNR.

        This is intended for SNR/readout-separation analysis, not thermal
        population extraction.

        Parameters
        ----------
        ig_new : array-like
            Prepared |g> rotated/projected SSF data.
        ie_new : array-like
            Prepared |e> rotated/projected SSF data.

        Returns
        -------
        snr : float
            Readout SNR,

                SNR = |mu_e - mu_g| / sqrt((sigma_g^2 + sigma_e^2) / 2)

        snr_info : dict
            Dictionary containing fitted means, sigmas, weights, Minuit object,
            and fit quality information.
        """

        # -------------------- Prepare data --------------------
        ig = np.asarray(ig_new, dtype=float).ravel()
        ie = np.asarray(ie_new, dtype=float).ravel()

        ig = ig[np.isfinite(ig)]
        ie = ie[np.isfinite(ie)]

        if ig.size == 0 or ie.size == 0:
            raise ValueError("ig_new and ie_new must both contain finite data.")

        x = np.concatenate([ig, ie])

        if x.size < 10:
            raise ValueError("Not enough total points to fit prepared g/e distributions.")

        eps = 1e-300

        # -------------------- Initial guesses --------------------
        # Use the prepared-state medians as the starting means.
        mu_g_init = np.median(ig)
        mu_e_init = np.median(ie)

        sigma_g_init = np.std(ig)
        sigma_e_init = np.std(ie)

        std_all = np.std(x)
        if not np.isfinite(std_all) or std_all <= 0:
            std_all = 1.0

        if not np.isfinite(sigma_g_init) or sigma_g_init <= 0:
            sigma_g_init = std_all / 2.0

        if not np.isfinite(sigma_e_init) or sigma_e_init <= 0:
            sigma_e_init = std_all / 2.0

        # Starting mixture weight based on number of prepared ground shots.
        w_g_init = ig.size / x.size
        w_g_init = np.clip(w_g_init, 1e-6, 1.0 - 1e-6)

        # -------------------- Two-Gaussian mixture NLL --------------------
        def nll_2g(mu1, sigma1, mu2, sigma2, w1):
            w1_clipped = np.clip(w1, 1e-6, 1.0 - 1e-6)
            w2 = 1.0 - w1_clipped

            g1 = self.gaussian_pdf(x, mu1, sigma1)
            g2 = self.gaussian_pdf(x, mu2, sigma2)

            p = w1_clipped * g1 + w2 * g2
            p = np.clip(p, eps, None)

            return -np.sum(np.log(p))

        m2 = Minuit(
            nll_2g,
            mu1=mu_g_init,
            sigma1=sigma_g_init,
            mu2=mu_e_init,
            sigma2=sigma_e_init,
            w1=w_g_init,
        )

        m2.limits["sigma1"] = (1e-6, None)
        m2.limits["sigma2"] = (1e-6, None)
        m2.limits["w1"] = (1e-6, 1.0 - 1e-6)
        m2.errordef = Minuit.LIKELIHOOD

        m2.simplex()
        m2.migrad()
        m2.hesse()

        nll_2g_min = m2.fval

        # -------------------- Extract parameters --------------------
        mu1 = float(m2.values["mu1"])
        sigma1 = float(m2.values["sigma1"])
        mu2 = float(m2.values["mu2"])
        sigma2 = float(m2.values["sigma2"])
        w1 = float(m2.values["w1"])
        w2 = 1.0 - w1

        means_raw = np.array([mu1, mu2], dtype=float)
        sigmas_raw = np.array([sigma1, sigma2], dtype=float)
        weights_raw = np.array([w1, w2], dtype=float)

        # -------------------- Identify prepared |g> and |e> components --------------------
        # Do not assume ground is always the lower-mean Gaussian. Instead, assign
        # components based on closeness to the prepared |g> and |e> medians.
        dist_to_g = np.abs(means_raw - mu_g_init)
        dist_to_e = np.abs(means_raw - mu_e_init)

        ground_idx_raw = int(np.argmin(dist_to_g))
        excited_idx_raw = int(np.argmin(dist_to_e))

        # If both medians picked the same Gaussian, fall back to sorting by mean.
        # This keeps the function from crashing on poor/degenerate fits.
        if ground_idx_raw == excited_idx_raw:
            order = np.argsort(means_raw)
        else:
            order = np.array([ground_idx_raw, excited_idx_raw], dtype=int)

        means = means_raw[order]
        sigmas = sigmas_raw[order]
        weights = weights_raw[order]

        # After ordering, index 0 = prepared |g>, index 1 = prepared |e>
        ground_idx = 0
        excited_idx = 1

        mu_g = means[ground_idx]
        mu_e = means[excited_idx]
        sigma_g = sigmas[ground_idx]
        sigma_e = sigmas[excited_idx]

        # -------------------- Calculate SNR --------------------
        denom = np.sqrt(0.5 * (sigma_g ** 2 + sigma_e ** 2))

        if denom <= 0 or not np.isfinite(denom):
            snr = np.nan
        else:
            snr = np.abs(mu_e - mu_g) / denom

        # ------------------ Optional fits plotting --------------
        if plot:
            self.plot_gaussians_SNR(
                q_key=q_key,
                qubit_folder=qubit_folder,
                ig_new=ig_new,
                ie_new=ie_new,
                weights=weights,
                sigmas=sigmas,
                means=means,
                snr=snr,
                dataset=dataset,
                title_ext=f"valid={bool(m2.valid)}",
                numbins=64,
                ylim=ssf_hist_ylim,
            )

        snr_info = {
            "snr": snr,
            "means": means,
            "sigmas": sigmas,
            "weights": weights,
            "mu_g_prep": mu_g,
            "mu_e_prep": mu_e,
            "sigma_g_prep": sigma_g,
            "sigma_e_prep": sigma_e,
            "weight_g_prep": weights[ground_idx],
            "weight_e_prep": weights[excited_idx],
            "ground_idx": ground_idx,
            "excited_idx": excited_idx,
            "means_raw": means_raw,
            "sigmas_raw": sigmas_raw,
            "weights_raw": weights_raw,
            "n_g_prep": ig.size,
            "n_e_prep": ie.size,
            "nll_2g_min": nll_2g_min,
            "minuit": m2,
            "valid": bool(m2.valid),
        }

        return snr, snr_info

    def fit_e_state_double_gaussian_iminuit(
            self,
            iq_data,
            qid=None,
            dataset=None,
            right_weight_init=0.8,
            verbose=False):
        """
        Fit SSF excited-prepared data, ie_new, to a two-Gaussian mixture.

        This uses the same fitting logic as fit_double_gaussian_midpoint_iminuit(),
        but with initial conditions appropriate for excited-prepared data.

        For ie_new:
            lower-mean Gaussian  -> decay during readout, failed pi pulses, etc component
            higher-mean Gaussian -> excited-state component

        Main quantity of interest:
            ie_new_ground_frac

        This is the observed ground-like fraction in the intended excited-state data.
        It is not automatically pure T1 decay.
        """

        try:
            # -------------------- Prepare data --------------------
            x = np.asarray(iq_data, dtype=float).ravel()
            x = x[np.isfinite(x)]

            if x.size == 0:
                raise ValueError("iq_data is empty after removing non-finite values.")

            eps = 1e-300  # to avoid log(0)

            # -------------------- Initial guesses --------------------
            # Same logic as the usual fitter, but tailored to excited-prepared data.
            # For ie_new, the dominant blob should usually be the excited-like
            # component on the right, with a smaller ground-like component on the left.
            q25, q75 = np.percentile(x, [25, 75])

            std_all = np.std(x)
            if std_all <= 0:
                std_all = 1.0

            mu1_init = q75  # right / excited-like component
            mu2_init = q25  # left / ground-like component
            w1_init = right_weight_init  # dominant right component

            sigma1_init = std_all / 2.0
            sigma2_init = std_all / 2.0

            # -------------------- 1-Gaussian null model --------------------
            def nll_1g(mu, sigma):
                g = self.gaussian_pdf(x, mu, sigma)
                p = np.clip(g, eps, None)
                return -np.sum(np.log(p))

            mu0_init = np.mean(x)
            sigma0_init = std_all

            m1 = Minuit(nll_1g, mu=mu0_init, sigma=sigma0_init)
            m1.limits["sigma"] = (1e-6, None)
            m1.errordef = Minuit.LIKELIHOOD
            m1.simplex()
            m1.migrad()
            m1.hesse()

            nll_1g_min = m1.fval

            # -------------------- 2-Gaussian mixture --------------------
            def nll_2g(mu1, sigma1, mu2, sigma2, w1):
                w1_clipped = np.clip(w1, 1e-6, 1.0 - 1e-6)
                w2 = 1.0 - w1_clipped

                g1 = self.gaussian_pdf(x, mu1, sigma1)
                g2 = self.gaussian_pdf(x, mu2, sigma2)

                p = w1_clipped * g1 + w2 * g2
                p = np.clip(p, eps, None)
                return -np.sum(np.log(p))

            m2 = Minuit(
                nll_2g,
                mu1=mu1_init,
                sigma1=sigma1_init,
                mu2=mu2_init,
                sigma2=sigma2_init,
                w1=w1_init)

            m2.limits["sigma1"] = (1e-6, None)
            m2.limits["sigma2"] = (1e-6, None)
            m2.limits["w1"] = (1e-6, 1.0 - 1e-6)
            m2.errordef = Minuit.LIKELIHOOD
            m2.simplex()
            m2.migrad()
            m2.hesse()

            nll_2g_min = m2.fval

            # -------------------- Likelihood-ratio statistic --------------------
            ie_new_lr_stat = 2.0 * (nll_1g_min - nll_2g_min)

            # -------------------- Extract and sort components --------------------
            mu1, sigma1, mu2, sigma2, w1 = m2.values
            w2 = 1.0 - w1

            means = np.array([mu1, mu2], dtype=float)
            sigmas = np.array([sigma1, sigma2], dtype=float)
            weights = np.array([w1, w2], dtype=float)

            # Same sorting logic as your ig_new fitter:
            # lower mean first, higher mean second.
            order = np.argsort(means)
            means = means[order]
            sigmas = sigmas[order]
            weights = weights[order]

            # For ie_new:
            # lower-mean component = ground-like
            # higher-mean component = excited-like
            ie_new_ground_frac = float(weights[0])
            ie_new_excited_frac = float(weights[1])

            # -------------------- Uncertainty from Minuit on w1 --------------------
            sigma_w1 = None

            if m2.covariance is not None:
                try:
                    sigma_w1 = float(np.sqrt(m2.covariance["w1", "w1"]))
                except Exception:
                    sigma_w1 = None

            if sigma_w1 is None:
                sigma_w1 = float(m2.errors["w1"])

            # Since w2 = 1 - w1, use same uncertainty for both mixture weights.
            ie_new_ground_frac_err = sigma_w1
            ie_new_excited_frac_err = sigma_w1

            # -------------------- Build output dictionary only after success --------------------
            ie_fit_results = {
                "ie_new_ground_frac": ie_new_ground_frac,
                "ie_new_ground_frac_err": ie_new_ground_frac_err,
                "ie_new_excited_frac": ie_new_excited_frac,
                "ie_new_excited_frac_err": ie_new_excited_frac_err,
                "ie_new_means": means,
                "ie_new_sigmas": sigmas,
                "ie_new_weights": weights,
                "ie_new_ground_mean": float(means[0]),
                "ie_new_excited_mean": float(means[1]),
                "ie_new_ground_sigma": float(sigmas[0]),
                "ie_new_excited_sigma": float(sigmas[1]),
                "ie_new_lr_stat": float(ie_new_lr_stat),
                "ie_new_nll_1g": float(nll_1g_min),
                "ie_new_nll_2g": float(nll_2g_min),
                "ie_new_Nshots": int(x.size)}
            
            if verbose:
                qlabel = f"Q{qid + 1}" if qid is not None else "Q?"
                dlabel = f"dataset {dataset}" if dataset is not None else "dataset ?"
                print(f"\n{qlabel}, {dlabel} excited-prepared ie_new fit:")
                print(
                    f"  ground-like fraction = "
                    f"{ie_new_ground_frac:.4f} ± {ie_new_ground_frac_err:.4f}")
                print(
                    f"  excited-like fraction = "
                    f"{ie_new_excited_frac:.4f} ± {ie_new_excited_frac_err:.4f}")
                print(f"  LRT = {ie_new_lr_stat:.2f}")

            return ie_fit_results

        except Exception as err:
            qlabel = f"Q{qid + 1}" if qid is not None else "Q?"
            dlabel = f"dataset {dataset}" if dataset is not None else "dataset ?"
            print(f"{qlabel}, {dlabel}: excited-prepared ie_new fit failed: {err}")
            return None
    
    def fit_double_gaussian_midpoint_iminuit(self, iq_data, dontuse_midpt_thresh = False, low_leakage_mode = False):
        """
        Iminuit-based version of fit_double_gaussian_midpoint().

        Default: calculates the Pe threshold by finding the midpoint of the two gaussian means.
        If dontuse_midpt_thresh is set to True, it instead uses the underlying probabilities (weights) found during
        the iminuit likelihood minimization process.

        low_leakage_mode is for runs where the thermal population is estimated to be < 2%

        Returns:
          Pg, Pe, minuit_2g, means, sigmas, weights,
          threshold_mid, threshold_mid_err, ground_gaussian, excited_gaussian,
          ground_data, excited_data, iq_data,
          lr_stat, nll_1g, nll_2g
        """

        # -------------------- Prepare data --------------------
        x = np.asarray(iq_data, dtype=float).ravel()
        if x.size == 0:
            raise ValueError("iq_data is empty; cannot fit double Gaussian.")

        eps = 1e-300  # to avoid log(0)

        # -------------------- Initial guesses (shared) --------------------
        # Usually: dominant left cluster + smaller excited/leakage Gaussian shifted right.
        q25, q50, q75, q95, q99 = np.percentile(x, [25, 50, 75, 95, 99])

        std_all = np.std(x)
        if std_all <= 0:
            std_all = 1.0  # fallback

        # ---- initial guesses for 2-Gaussian mixture ----
        if low_leakage_mode: # when thermal populations are really low
            mu1_init = np.percentile(x, 50)
            mu2_init = np.percentile(x, 99)
            w1_init = 0.99
        else:
            mu1_init = q25
            mu2_init = q75
            w1_init = 0.5

        sigma1_init = std_all / 2.0
        sigma2_init = std_all / 2.0

        # -------------------- 1-Gaussian (null model) --------------------
        def nll_1g(mu, sigma):
            g = self.gaussian_pdf(x, mu, sigma)
            p = np.clip(g, eps, None)
            return -np.sum(np.log(p))

        mu0_init = np.mean(x)
        sigma0_init = std_all

        m1 = Minuit(nll_1g, mu=mu0_init, sigma=sigma0_init)
        m1.limits["sigma"] = (1e-6, None)
        m1.errordef = Minuit.LIKELIHOOD
        m1.simplex()
        m1.migrad()
        m1.hesse()

        nll_1g_min = m1.fval  # NLL for best 1-Gaussian fit

        # -------------------- 2-Gaussian mixture --------------------
        def nll_2g(mu1, sigma1, mu2, sigma2, w1):
            # enforce w1 between 0 and 1 via limits (Minuit) but still be safe numerically
            w1_clipped = np.clip(w1, 1e-6, 1.0 - 1e-6)
            w2 = 1.0 - w1_clipped

            g1 = self.gaussian_pdf(x, mu1, sigma1)
            g2 = self.gaussian_pdf(x, mu2, sigma2)

            p = w1_clipped * g1 + w2 * g2
            p = np.clip(p, eps, None)
            return -np.sum(np.log(p))

        m2 = Minuit(
            nll_2g,
            mu1=mu1_init,
            sigma1=sigma1_init,
            mu2=mu2_init,
            sigma2=sigma2_init,
            w1=w1_init,
        )
        m2.limits["sigma1"] = (1e-6, None)
        m2.limits["sigma2"] = (1e-6, None)
        m2.limits["w1"] = (1e-6, 1.0 - 1e-6)
        m2.errordef = Minuit.LIKELIHOOD
        m2.simplex()
        m2.migrad()
        m2.hesse()

        nll_2g_min = m2.fval  # NLL for best 2-Gaussian mixture

        # -------------------- Likelihood-ratio statistic --------------------
        # Λ = 2 (NLL_1g - NLL_2g); Δk = 3 extra params (mu2, sigma2, w1)
        lr_stat = 2.0 * (nll_1g_min - nll_2g_min)

        # -------------------- Extract and sort components --------------------
        mu1, sigma1, mu2, sigma2, w1 = m2.values
        w2 = 1.0 - w1 # weights = the fraction of the data that belongs to each Gaussian

        means = np.array([mu1, mu2], dtype=float)
        sigmas = np.array([sigma1, sigma2], dtype=float)
        weights = np.array([w1, w2], dtype=float)

        # identify which component is "ground" (lower mean)
        ground_gaussian = int(np.argmin(means))
        excited_gaussian = 1 - ground_gaussian

        order = np.array([ground_gaussian, excited_gaussian])
        ground_gaussian = 0 # based on how it was sorted inside order array
        excited_gaussian = 1 # based on how it was sorted inside order array
        means = means[order]
        sigmas = sigmas[order]
        weights = weights[order]

        if dontuse_midpt_thresh:
            Pg = weights[0] #ground gaussian comes first
            Pe = weights[1]
            threshold_mid = None
            threshold_mid_err = None

            # --- uncertainty from Minuit on w1 (and thus also on w2=1-w1) ---
            # Prefer covariance if available; fall back to errors.
            sigma_w1 = None
            if m2.covariance is not None:
                try:
                    sigma_w1 = float(np.sqrt(m2.covariance["w1", "w1"]))
                except Exception:
                    sigma_w1 = None
            if sigma_w1 is None:
                sigma_w1 = float(m2.errors["w1"])

            # Pe is either w1 or w2; in both cases sigma is sigma_w1
            sigma_Pe = sigma_w1
            sigma_Pg = sigma_w1

            # In weights-mode, "ground_data/excited_data" via a hard cut is not defined.
            ground_data = None
            excited_data = None

        else:
            # -------------------- Midpoint threshold --------------------
            threshold_mid = 0.5 * (means[ground_gaussian] + means[excited_gaussian])

            g_ground = self.gaussian_pdf(x, means[ground_gaussian], sigmas[ground_gaussian])
            g_excited = self.gaussian_pdf(x, means[excited_gaussian], sigmas[excited_gaussian])

            pg = weights[ground_gaussian] * g_ground
            pe = weights[excited_gaussian] * g_excited
            denom = np.clip(pg + pe, eps, None)  # total

            rg = pg / denom  # responsibility for ground
            re = pe / denom  # responsibility for excited

            N_g = rg.sum()
            N_e = re.sum()

            sigma_g = sigmas[ground_gaussian]
            sigma_e = sigmas[excited_gaussian]

            sigma_mu_g = sigma_g / np.sqrt(N_g) if N_g > 0 else 0.0
            sigma_mu_e = sigma_e / np.sqrt(N_e) if N_e > 0 else 0.0

            # -------------------- Midpoint threshold error -------------------
            threshold_mid_err = 0.5 * np.sqrt(sigma_mu_g ** 2 + sigma_mu_e ** 2)

            # -------------------- Split data and compute populations --------------------
            ground_data = x[x <= threshold_mid]
            excited_data = x[x > threshold_mid]
            Pg = len(ground_data) / len(x)
            Pe = len(excited_data) / len(x)

            # -------------------- Pe uncertainty -----------------------------
            # Total 1-sigma uncertainty on Pe
            mask_plus = (x <= threshold_mid + threshold_mid_err)
            Pe_plus = 1.0 - mask_plus.mean()
            mask_minus = (x <= threshold_mid - threshold_mid_err)
            Pe_minus = 1.0 - mask_minus.mean()
            sigma_Pe_from_thresh = 0.5 * abs(Pe_plus - Pe_minus)

            Nshots = x.size
            sigma_Pe_stat = np.sqrt(Pe * (1.0 - Pe) / Nshots)

            sigma_Pe = np.sqrt(sigma_Pe_from_thresh ** 2 + sigma_Pe_stat ** 2)

        return (
            Pg,
            Pe,
            sigma_Pe,
            m2,  # 2-Gaussian Minuit object
            means, # of the two gaussians, list
            sigmas, # of the two gaussians, list
            weights, # of the two gaussians, list
            threshold_mid,
            threshold_mid_err,
            ground_gaussian, # index
            excited_gaussian, # index
            ground_data,
            excited_data,
            x, # iq_data
            lr_stat, # likelihood ratio test score
            nll_1g_min, # negative-log likelihood, 1 gaussian
            nll_2g_min, # negative-log likelihood, 2 gaussians
        )

class RPMTempCalcAndPlots:
    def __init__(self, figure_quality, number_of_qubits):
        self.figure_quality = figure_quality
        self.number_of_qubits = number_of_qubits

    def run_RPMqtemps(self, base_dir, target_dates, filter_keywords, fit_saved, signal, run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                     outerFolder_RR_plots, replot_RPMs = False, get_qtemp_data = False, get_london_data = False, figure_quality = 200, save_figsRR = False,
                      exclude_temp_sweeps = False, passing_pre_sciencerun_data = False, filter_out_bad_RPM_fits = False, use_png_timestamps = False,
                      combine_IQ_signal = False):

        combined_qtemp_data = []  # list of results from different .h5 files

        os.makedirs(outerFolder_RR_plots, exist_ok=True)

        #------------------------------------------ Looping through data folders and files -------------------------------------------------------
        # Note: this is tailored for how files are organized by ryan for QUIET
        for root, dirs, _ in os.walk(base_dir): # root = substudy, dirs = date folder
            dirs.sort()  # alphabetical to chronological for YYYY-MM-DD_HH-MM-SS
            for d in dirs:
                full_path = os.path.join(root, d)
                # Match folders like '2025-04-16_11-47-09' based on prefix date
                if (
                        any(d.startswith(date) for date in target_dates)
                        and len(d) >= 19
                        and any(keyword in full_path for keyword in filter_keywords)
                        and (not exclude_temp_sweeps or "temperature_sweep" not in full_path.lower())
                    ):  # checks if path includes each keyword (source_off or source_on) and whether you set the temp sweep data to be excluded or not

                    if run_num == 7: # depending on the run, the q_temperatures data is stored in a different place
                        data_path = os.path.join(full_path, "study_data") # run 7 qubit temperature and RR data was stored in study_data folder
                    elif run_num == 6:
                        if passing_pre_sciencerun_data:
                            data_path = os.path.join(full_path, "study_data") # pre-science-run data was stored in study_data folder
                        else:
                            data_path = os.path.join(full_path, "optimization") # science-run data was stored in optimization folder
                    elif run_num == 8:
                        data_path = os.path.join(full_path, "study_data")
                    elif run_num == 9:
                        data_path = os.path.join(full_path, "study_data")
                    elif run_num == 9.2:
                        data_path = os.path.join(full_path, "study_data")
                    else:
                        raise ValueError("run_num must be 6, 7, 8, 9, 9.2 (run 9c) OR you must add an 'if statement' for the run number you want. Specify if RPM data is in optimization OR study_data folder.")

                    if os.path.isdir(data_path):
                        date_string = d[:10]  # Extract 'YYYY-MM-DD'
                        #print(f"Analyzing: {data_path}")

                        # If both are set equal to data_path, it is assumed that q_temperatures and qspec_ge data are share the same path
                        outerFolder = data_path  # RR data (g-e Qspec) folder path before Data_h5
                        outerFolder_qtemps_data = data_path  # Qubit temps data folder path before Data_h5

                        if not os.path.exists(outerFolder): os.makedirs(outerFolder)
                        if not os.path.exists(outerFolder_qtemps_data): os.makedirs(outerFolder_qtemps_data)

                        # ---------------------------------------- Initialize the PlotRR_noQick class ------------------------------------------------
                        plotter = PlotRR_noQick(date_string, figure_quality, save_figsRR, fit_saved, signal, run_name,
                                                tot_num_of_qubits, outerFolder, outerFolder_RR_plots, outerFolder_qtemps_data, run_num, filter_out_bad_RPM_fits)

                        if replot_RPMs:
                            # You still have to set save_figsRR to True if you want to save plots
                            # set filter_out_bad_RPM_fits to True to save filtered ones, otherwise no quality cuts will be applied
                            # ------------------------------------To re-plot the RPM plots (the rest have been internally commented out)-----------------------------------------------------
                            plotter.run(plot_res_spec = False, plot_q_spec = False, plot_rabi = False, plot_ss = False,  ss_plot_gef = False, plot_t1 = False,
                                        plot_t2r = False, plot_t2e = False, plot_rabis_Qtemps = True, combine_rpm_IQ_signal = combine_IQ_signal)

                        if get_qtemp_data: # returns RPM qubit temperature data (and qfreqs that were used for the calculations)
                            # ---------------------------------------- Load data and append to list spanning multiple dates --------------------------------------------------
                            qtemp_data = plotter.load_plot_save_rabis_Qtemps(list_of_all_qubits, run_num, save_figs = save_figsRR, get_qtemp_data = get_qtemp_data, filter_out_bad_RPM_fits = filter_out_bad_RPM_fits,
                                                                             use_png_timestamps = use_png_timestamps, combine_IQ_signal = combine_IQ_signal)
                            combined_qtemp_data.extend(qtemp_data)

                        if get_london_data: # returns RPM qubit temperature data, qfreqs that were used to calculate the temps, and resonator freqs
                            # ---------------------------------------- Load data and append to list spanning multiple dates --------------------------------------------------
                            london_data = plotter.load_qfreqs_resfreqs_qtemps(list_of_all_qubits, run_num, save_figs = False, get_data = get_london_data)
                            combined_qtemp_data.extend(london_data)

        return combined_qtemp_data # Will be empty if get_qtemp_data or get_london_data are set to False


class combined_Qtemp_studies:
    def __init__(self, figure_quality, number_of_qubits):
        self.figure_quality = figure_quality
        self.number_of_qubits = number_of_qubits

    def create_processed_coherence_inputs(
            self,
            coherence_cache_dir,
            run_name,
            date_times_res_spec,
            res_freqs,
            date_times_q_spec,
            q_freqs,
            qspec_fit_err,
            date_times_t1,
            t1_vals,
            t1_fit_err,
            date_times_t2r,
            t2r_vals,
            t2r_fit_err,
            date_times_t2e,
            t2e_vals,
            t2e_fit_err,
            I_per_pt_errs=None,
            Q_per_pt_errs=None,
    ):
        """
        Save processed coherence inputs for one run.

        Each coherence metric is saved to a separate pickle file:
            - resonator spectroscopy
            - qubit spectroscopy
            - T1
            - T2 Ramsey
            - T2 Echo

        Parameters
        ----------
        coherence_cache_dir : str
            Directory where processed coherence pickle files should be saved.

        run_name : str
            Name of the run. Used in the filename so each run gets separate cache files.

        date_times_res_spec : list
            Timestamps for resonator spectroscopy measurements.

        res_freqs : list or dict
            Resonator frequency values.

        date_times_q_spec : list
            Timestamps for qubit spectroscopy measurements.

        q_freqs : list or dict
            Qubit frequency values.

        qspec_fit_err : list or dict
            Qubit spectroscopy fit errors.

        date_times_t1 : list
            Timestamps for T1 measurements.

        t1_vals : list or dict
            T1 values.

        t1_fit_err : list or dict
            T1 fit errors.

        date_times_t2r : list
            Timestamps for T2 Ramsey measurements.

        t2r_vals : list or dict
            T2 Ramsey values.

        t2r_fit_err : list or dict
            T2 Ramsey fit errors.

        date_times_t2e : list
            Timestamps for T2 Echo measurements.

        t2e_vals : list or dict
            T2 Echo values.

        t2e_fit_err : list or dict
            T2 Echo fit errors.

        I_per_pt_errs : list or dict, optional
            Optional per-point I errors from T1 shot processing.

        Q_per_pt_errs : list or dict, optional
            Optional per-point Q errors from T1 shot processing.

        Returns
        -------
        saved_paths : dict
            Dictionary containing the saved pickle paths for each metric.
        """

        os.makedirs(coherence_cache_dir, exist_ok=True)

        saved_paths = {
            "res_spec": os.path.join(
                coherence_cache_dir,
                f"{run_name}_processed_res_spec.pkl"
            ),
            "qspec": os.path.join(
                coherence_cache_dir,
                f"{run_name}_processed_qspec.pkl"
            ),
            "t1": os.path.join(
                coherence_cache_dir,
                f"{run_name}_processed_t1.pkl"
            ),
            "t2r": os.path.join(
                coherence_cache_dir,
                f"{run_name}_processed_t2r.pkl"
            ),
            "t2e": os.path.join(
                coherence_cache_dir,
                f"{run_name}_processed_t2e.pkl"
            ),
        }

        res_spec_data = {
            "date_times_res_spec": date_times_res_spec,
            "res_freqs": res_freqs,
        }

        qspec_data = {
            "date_times_q_spec": date_times_q_spec,
            "q_freqs": q_freqs,
            "qspec_fit_err": qspec_fit_err,
        }

        t1_data = {
            "date_times_t1": date_times_t1,
            "t1_vals": t1_vals,
            "t1_fit_err": t1_fit_err,
            "I_per_pt_errs": I_per_pt_errs,
            "Q_per_pt_errs": Q_per_pt_errs,
        }

        t2r_data = {
            "date_times_t2r": date_times_t2r,
            "t2r_vals": t2r_vals,
            "t2r_fit_err": t2r_fit_err,
        }

        t2e_data = {
            "date_times_t2e": date_times_t2e,
            "t2e_vals": t2e_vals,
            "t2e_fit_err": t2e_fit_err,
        }

        data_to_save = {
            "res_spec": res_spec_data,
            "qspec": qspec_data,
            "t1": t1_data,
            "t2r": t2r_data,
            "t2e": t2e_data,
        }

        for metric, data in data_to_save.items():
            with open(saved_paths[metric], "wb") as f:
                pickle.dump(data, f)

            print(f"Saved processed {metric} results to:")
            print(saved_paths[metric])

        return saved_paths
    
    def load_processed_coherence_inputs(
            self,
            res_spec_path,
            qspec_path,
            t1_path,
            t2r_path,
            t2e_path,
    ):
        """
        Load previously saved processed coherence inputs.

        Parameters
        ----------
        res_spec_path : str
            Path to the saved resonator spectroscopy pickle file.

        qspec_path : str
            Path to the saved qubit spectroscopy pickle file.

        t1_path : str
            Path to the saved T1 pickle file.

        t2r_path : str
            Path to the saved T2 Ramsey pickle file.

        t2e_path : str
            Path to the saved T2 Echo pickle file.

        Returns
        -------
        date_times_res_spec : list
            Timestamps for resonator spectroscopy measurements.

        res_freqs : list or dict
            Resonator frequency values.

        date_times_q_spec : list
            Timestamps for qubit spectroscopy measurements.

        q_freqs : list or dict
            Qubit frequency values.

        qspec_fit_err : list or dict
            Qubit spectroscopy fit errors.

        date_times_t1 : list
            Timestamps for T1 measurements.

        t1_vals : list or dict
            T1 values.

        t1_fit_err : list or dict
            T1 fit errors.

        date_times_t2r : list
            Timestamps for T2 Ramsey measurements.

        t2r_vals : list or dict
            T2 Ramsey values.

        t2r_fit_err : list or dict
            T2 Ramsey fit errors.

        date_times_t2e : list
            Timestamps for T2 Echo measurements.

        t2e_vals : list or dict
            T2 Echo values.

        t2e_fit_err : list or dict
            T2 Echo fit errors.

        I_per_pt_errs : list or dict or None
            Optional per-point I errors from T1 shot processing.

        Q_per_pt_errs : list or dict or None
            Optional per-point Q errors from T1 shot processing.
        """

        with open(res_spec_path, "rb") as f:
            res_spec_data = pickle.load(f)

        with open(qspec_path, "rb") as f:
            qspec_data = pickle.load(f)

        with open(t1_path, "rb") as f:
            t1_data = pickle.load(f)

        with open(t2r_path, "rb") as f:
            t2r_data = pickle.load(f)

        with open(t2e_path, "rb") as f:
            t2e_data = pickle.load(f)

        print("Loaded processed resonator spectroscopy results from:")
        print(res_spec_path)

        print("Loaded processed qubit spectroscopy results from:")
        print(qspec_path)

        print("Loaded processed T1 results from:")
        print(t1_path)

        print("Loaded processed T2 Ramsey results from:")
        print(t2r_path)

        print("Loaded processed T2 Echo results from:")
        print(t2e_path)

        date_times_res_spec = res_spec_data["date_times_res_spec"]
        res_freqs = res_spec_data["res_freqs"]

        date_times_q_spec = qspec_data["date_times_q_spec"]
        q_freqs = qspec_data["q_freqs"]
        qspec_fit_err = qspec_data["qspec_fit_err"]

        date_times_t1 = t1_data["date_times_t1"]
        t1_vals = t1_data["t1_vals"]
        t1_fit_err = t1_data["t1_fit_err"]

        I_per_pt_errs = t1_data.get("I_per_pt_errs", None)
        Q_per_pt_errs = t1_data.get("Q_per_pt_errs", None)

        date_times_t2r = t2r_data["date_times_t2r"]
        t2r_vals = t2r_data["t2r_vals"]
        t2r_fit_err = t2r_data["t2r_fit_err"]

        date_times_t2e = t2e_data["date_times_t2e"]
        t2e_vals = t2e_data["t2e_vals"]
        t2e_fit_err = t2e_data["t2e_fit_err"]

        return (
            date_times_res_spec,
            res_freqs,
            date_times_q_spec,
            q_freqs,
            qspec_fit_err,
            date_times_t1,
            t1_vals,
            t1_fit_err,
            date_times_t2r,
            t2r_vals,
            t2r_fit_err,
            date_times_t2e,
            t2e_vals,
            t2e_fit_err,
            I_per_pt_errs, # T1
            Q_per_pt_errs, # T1
        )

    def get_or_create_processed_coherence_inputs(
            self,
            run_number,
            run_name,
            coherence_cache_dir,
            coh_qtemp_ana_flags,
            figure_quality,
            final_figure_quality,
            tot_num_of_qubits,
            top_folder_dates,
            save_figs,
            fit_saved,
            signal,
            FRIDGE,
            data_path,
            plots_path,
            per_pt_errs_t1=False,
            process_shots_t1ge=False,
    ):
        """
        Either load cached processed coherence inputs or process the coherence data
        normally and optionally save the results.

        This is intended to be used for one run at a time.

        Returns
        -------
        date_times_res_spec, res_freqs,
        date_times_q_spec, q_freqs, qspec_fit_err,
        date_times_t1, t1_vals, t1_fit_err,
        date_times_t2r, t2r_vals, t2r_fit_err,
        date_times_t2e, t2e_vals, t2e_fit_err,
        I_per_pt_errs, Q_per_pt_errs
        """
        #coherence_cache_dir = os.path.join(coherence_cache_dir,f"run{run_number}")
        os.makedirs(coherence_cache_dir, exist_ok=True)

        cache_prefix = f"run{run_number}"

        res_spec_cache_path = os.path.join(
            coherence_cache_dir,
            f"{cache_prefix}_processed_res_spec.pkl"
        )

        qspec_cache_path = os.path.join(
            coherence_cache_dir,
            f"{cache_prefix}_processed_qspec.pkl"
        )

        t1_cache_path = os.path.join(
            coherence_cache_dir,
            f"{cache_prefix}_processed_t1.pkl"
        )

        t2r_cache_path = os.path.join(
            coherence_cache_dir,
            f"{cache_prefix}_processed_t2r.pkl"
        )

        t2e_cache_path = os.path.join(
            coherence_cache_dir,
            f"{cache_prefix}_processed_t2e.pkl"
        )

        # ------------------------------------------------------------
        # Option 1: Load cached processed coherence files
        # ------------------------------------------------------------
        if coh_qtemp_ana_flags["use_cached_coherence_files"]:
            return self.load_processed_coherence_inputs(
                res_spec_path=res_spec_cache_path,
                qspec_path=qspec_cache_path,
                t1_path=t1_cache_path,
                t2r_path=t2r_cache_path,
                t2e_path=t2e_cache_path,
            )
        else:
            # ------------------------------------------------------------
            # Option 2: Process coherence data normally
            # ------------------------------------------------------------
            res_spec_vs_time = ResonatorFreqVsTime(
                data_path,
                plots_path,
                figure_quality,
                final_figure_quality,
                tot_num_of_qubits,
                top_folder_dates,
                save_figs,
                fit_saved,
                signal,
                run_name,
                FRIDGE
            )

            date_times_res_spec, res_freqs = res_spec_vs_time.run()

            q_spec_vs_time = QubitFreqsVsTime(
                data_path,
                plots_path,
                figure_quality,
                final_figure_quality,
                tot_num_of_qubits,
                top_folder_dates,
                save_figs,
                fit_saved,
                signal,
                run_name,
                FRIDGE
            )

            date_times_q_spec, q_freqs, qspec_fit_err = q_spec_vs_time.run(
                exp_extension="_ge",
                use_png_timestamps=False
            )

            t1_vs_time = T1VsTime(
                plots_path,
                figure_quality,
                final_figure_quality,
                tot_num_of_qubits,
                top_folder_dates,
                save_figs,
                fit_saved,
                signal,
                run_name,
                FRIDGE,
                run_number,
                per_pt_errs=per_pt_errs_t1
            )

            I_per_pt_errs = None
            Q_per_pt_errs = None

            if per_pt_errs_t1 and process_shots_t1ge:
                (
                    date_times_t1,
                    t1_vals,
                    t1_fit_err,
                    I_per_pt_errs,
                    Q_per_pt_errs,
                ) = t1_vs_time.run(
                    return_errs=True,
                    exp_extension="_ge",
                    process_shots=process_shots_t1ge
                )
            else:
                date_times_t1, t1_vals, t1_fit_err = t1_vs_time.run(
                    return_errs=True,
                    exp_extension="_ge"
                )

            t2r_vs_time = T2rVsTime(
                plots_path,
                run_number,
                figure_quality,
                final_figure_quality,
                tot_num_of_qubits,
                top_folder_dates,
                save_figs,
                fit_saved,
                signal,
                run_name,
                FRIDGE
            )

            date_times_t2r, t2r_vals, t2r_fit_err = t2r_vs_time.run(
                return_errs=True,
                t1_vals=t1_vals
            )

            t2e_vs_time = T2eVsTime(
                plots_path,
                run_number,
                figure_quality,
                final_figure_quality,
                tot_num_of_qubits,
                top_folder_dates,
                save_figs,
                fit_saved,
                signal,
                run_name,
                FRIDGE
            )

            date_times_t2e, t2e_vals, t2e_fit_err = t2e_vs_time.run(
                return_errs=True,
                t1_vals=t1_vals
            )

            # ------------------------------------------------------------
            # Optionally save processed coherence files
            # ------------------------------------------------------------
            if coh_qtemp_ana_flags["create_cached_coherence_files"]:
                self.create_processed_coherence_inputs(
                    coherence_cache_dir=coherence_cache_dir,
                    run_name=cache_prefix,
                    date_times_res_spec=date_times_res_spec,
                    res_freqs=res_freqs,
                    date_times_q_spec=date_times_q_spec,
                    q_freqs=q_freqs,
                    qspec_fit_err=qspec_fit_err,
                    date_times_t1=date_times_t1,
                    t1_vals=t1_vals,
                    t1_fit_err=t1_fit_err,
                    date_times_t2r=date_times_t2r,
                    t2r_vals=t2r_vals,
                    t2r_fit_err=t2r_fit_err,
                    date_times_t2e=date_times_t2e,
                    t2e_vals=t2e_vals,
                    t2e_fit_err=t2e_fit_err,
                    I_per_pt_errs=I_per_pt_errs,
                    Q_per_pt_errs=Q_per_pt_errs,
                )

        return (
            date_times_res_spec,
            res_freqs,
            date_times_q_spec,
            q_freqs,
            qspec_fit_err,
            date_times_t1,
            t1_vals,
            t1_fit_err,
            date_times_t2r,
            t2r_vals,
            t2r_fit_err,
            date_times_t2e,
            t2e_vals,
            t2e_fit_err,
            I_per_pt_errs,
            Q_per_pt_errs,
        )

    def save_processed_ssf_rpm_inputs(self,
            fit_results_g,
            all_files_Qtemp_results_RPMs,
            save_dir,
            tag="ssf_rpm_processed"):
        """
        Save the processed SSF and RPM inputs into separate pickle files.

        This lets you avoid rerunning the slow processing step before calling
        SSF_fid_vs_RRPM_Pe_3D.

        Parameters
        ----------
        fit_results_g : dict
            Processed SSF fit results.

        all_files_Qtemp_results_RPMs : list
            Processed RPM qubit temperature / P_e results.

        save_dir : str
            Folder where the cached files should be saved.

        tag : str
            Name prefix to identify this cache set.

        Returns
        -------
        paths : dict
            Dictionary containing the saved file paths.
        """

        os.makedirs(save_dir, exist_ok=True)

        #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        ssf_path = os.path.join(
            save_dir,
            f"{tag}_SSF_fit_results_g.pkl"
        )

        rpm_path = os.path.join(
            save_dir,
            f"{tag}_all_files_Qtemp_results_RPMs.pkl"
        )

        with open(ssf_path, "wb") as f:
            pickle.dump(fit_results_g, f)

        with open(rpm_path, "wb") as f:
            pickle.dump(all_files_Qtemp_results_RPMs, f)

        print("Saved processed SSF fit results to:")
        print(ssf_path)

        print("Saved processed RPM results to:")
        print(rpm_path)

        return {
            "fit_results_g": ssf_path,
            "all_files_Qtemp_results_RPMs": rpm_path,
        }

    def load_processed_ssf_rpm_inputs(
            self,
            fit_results_g_path=None,
            all_files_Qtemp_results_RPMs_path=None,
            start_time=None,
            time_key="date",
    ):
        """
        Load previously saved processed SSF and RPM inputs.

        This version allows either file to be missing. If a path is None,
        empty, or does not exist, that dataset is returned as an empty dict.

        Optional RPM timestamp filtering:
        If start_time is provided, only RPM file_results with
        file_result[time_key] >= start_time are kept.

        Parameters
        ----------
        fit_results_g_path : str or None
            Path to the saved fit_results_g pickle file.

        all_files_Qtemp_results_RPMs_path : str or None
            Path to the saved RPM results pickle file.

        Returns
        -------
        fit_results_g : dict
            Loaded SSF fit results, or {} if unavailable.

        all_files_Qtemp_results_RPMs : dict or list
            Loaded RPM results, or {} if unavailable.
        """

        fit_results_g = {}
        all_files_Qtemp_results_RPMs = {}

        if fit_results_g_path and os.path.isfile(fit_results_g_path):
            with open(fit_results_g_path, "rb") as f:
                fit_results_g = pickle.load(f)

            print("Loaded processed SSF fit results from:")
            print(fit_results_g_path)
        else:
            print("No processed SSF fit results loaded.")
            print(f"SSF path was: {fit_results_g_path}")

        if all_files_Qtemp_results_RPMs_path and os.path.isfile(all_files_Qtemp_results_RPMs_path):
            with open(all_files_Qtemp_results_RPMs_path, "rb") as f:
                all_files_Qtemp_results_RPMs = pickle.load(f)

            print("Loaded processed RPM results from:")
            print(all_files_Qtemp_results_RPMs_path)
        else:
            print("No processed RPM results loaded.")
            print(f"RPM path was: {all_files_Qtemp_results_RPMs_path}")

        return fit_results_g, all_files_Qtemp_results_RPMs

    def extract_pe_from_fit_results(self, fit_results_g, n_qubits):
        """
        Converts SSF fit_results_g into per-qubit Pe arrays.

        Input
        -----
        fit_results_g[qid] = [
            {"Pe": ..., "total_sigma_Pe": ...},
            ...
        ]

        Output
        ------
        pe_vals[qid] = [Pe, Pe, ...]
        pe_errs[qid] = [sigma_Pe, sigma_Pe, ...]
        """

        pe_vals = [[] for _ in range(n_qubits)]
        pe_errs = [[] for _ in range(n_qubits)]

        for qid in range(n_qubits):

            entries = fit_results_g.get(qid, []) if isinstance(fit_results_g, dict) else []

            for entry in entries:

                if not isinstance(entry, dict):
                    continue

                pe = entry.get("Pe", None)
                pe_err = entry.get("total_sigma_Pe", None)

                try:
                    pe = float(pe)
                except Exception:
                    pe = np.nan

                try:
                    pe_err = float(pe_err) if pe_err is not None else np.nan
                except Exception:
                    pe_err = np.nan

                if np.isfinite(pe):
                    pe_vals[qid].append(pe)
                    pe_errs[qid].append(pe_err)

        return pe_vals, pe_errs

    def ssf_fit_results_to_per_qubit_lists(
            self,
            fit_results,
            n_qubits=6,
            temp_key="temperature_mK",
            err_key="temperature_err_mK",
            ssf_key="ssf_fid",
            ssf_err_key="ssf_err_total",
            snr_key="ssf_SNR",
            ie_new_ground_key="ie_new_ground_frac",
            ie_new_ground_err_key="ie_new_ground_frac_err",
            alt_err_keys=("temperature_mK_err", "T_mK_err", "T_err_mK", "T_err"),
            qubit_keys=("qid", "qubit", "qubit_index", "QubitIndex"),
            keep_nans=False):
        """
        Convert SSF fit_results into per-qubit lists.

        Returns
        -------
        temps : list of lists
            temps[qid] = [temperature_mK, ...]

        errs : list of lists
            errs[qid] = [temperature_error_mK, ...]

        ssf_vals : list of lists
            ssf_vals[qid] = [ssf_fid, ...]

        ssf_errs : list of lists
            ssf_errs[qid] = [ssf_err_total, ...]

        snr_vals : list of lists
            snr_vals[qid] = [ssf_SNR, ...]

        ie_ground_vals : list of lists
            ie_ground_vals[qid] = [ie_new_ground_frac, ...]

        ie_ground_errs : list of lists
            ie_ground_errs[qid] = [ie_new_ground_frac_err, ...]

        Supported input formats
        -----------------------
        1. Dictionary form:
            fit_results[qid] = [record_dict, record_dict, ...]

        2. Flat list form:
            fit_results = [record_dict, record_dict, ...]

           In this case, each record must contain a qubit identifier using one
           of the names in qubit_keys, e.g. "qid" or "qubit_index".

        Notes
        -----
        - If an error key is missing, np.nan is appended so the arrays stay aligned.
        - If the SSF key is missing, np.nan is appended so the arrays stay aligned.
        - If the SNR key is missing, np.nan is appended so the arrays stay aligned.
        - If the ie_new ground-fraction key is missing, np.nan is appended.
        - If keep_nans=False, records with non-finite temperatures are skipped.
        """

        temps = [[] for _ in range(n_qubits)]
        errs = [[] for _ in range(n_qubits)]
        ssf_vals = [[] for _ in range(n_qubits)]
        ssf_errs = [[] for _ in range(n_qubits)]
        snr_vals = [[] for _ in range(n_qubits)]
        ie_ground_vals = [[] for _ in range(n_qubits)]
        ie_ground_errs = [[] for _ in range(n_qubits)]

        # ---------------- Helpers ----------------
        def _to_float_or_nan(val):
            """Safely convert scalar-like values to float; otherwise return np.nan."""
            try:
                if val is None:
                    return np.nan

                val = float(val)

                if np.isfinite(val):
                    return val

                return np.nan

            except (TypeError, ValueError):
                return np.nan

        def _get_first_available(rec, keys):
            """Return the first available record value from a list/tuple of keys."""
            for key in keys:
                if key in rec:
                    return rec.get(key)
            return None

        def _get_temp_err(rec):
            """Get temperature error, allowing fallback key names."""
            if err_key in rec:
                return rec.get(err_key)

            for key in alt_err_keys:
                if key in rec:
                    return rec.get(key)

            return None

        def _append_record(qid, rec):
            """Append one record into the per-qubit containers."""
            if not isinstance(rec, dict):
                return

            if qid is None or qid < 0 or qid >= n_qubits:
                return

            T = _to_float_or_nan(rec.get(temp_key, None))

            if not keep_nans and not np.isfinite(T):
                return

            T_err = _to_float_or_nan(_get_temp_err(rec))
            ssf = _to_float_or_nan(rec.get(ssf_key, None))
            ssf_err = _to_float_or_nan(rec.get(ssf_err_key, None))
            snr = _to_float_or_nan(rec.get(snr_key, None))

            ie_ground = _to_float_or_nan(rec.get(ie_new_ground_key, None))
            ie_ground_err = _to_float_or_nan(rec.get(ie_new_ground_err_key, None))

            temps[qid].append(T)
            errs[qid].append(T_err)
            ssf_vals[qid].append(ssf)
            ssf_errs[qid].append(ssf_err)
            snr_vals[qid].append(snr)
            ie_ground_vals[qid].append(ie_ground)
            ie_ground_errs[qid].append(ie_ground_err)

        # ---------------- Dictionary form: {qid: [records]} ----------------
        if isinstance(fit_results, dict):
            for qid in range(n_qubits):
                records = fit_results.get(qid, []) or []

                for rec in records:
                    _append_record(qid, rec)

            return (
                temps,
                errs,
                ssf_vals,
                ssf_errs,
                snr_vals,
                ie_ground_vals,
                ie_ground_errs
            )

        # ---------------- Flat list form: [records] ----------------
        if isinstance(fit_results, (list, tuple)):
            for rec in fit_results:
                if not isinstance(rec, dict):
                    continue

                qid = _get_first_available(rec, qubit_keys)

                try:
                    qid = int(qid)
                except (TypeError, ValueError):
                    continue

                _append_record(qid, rec)

            return (
                temps,
                errs,
                ssf_vals,
                ssf_errs,
                snr_vals,
                ie_ground_vals,
                ie_ground_errs)

        raise TypeError(
            "fit_results must be either a dict keyed by qubit index "
            "or a flat list of record dictionaries.")

    def rpm_results_to_per_qubit_lists(
            self,
            all_files_Qtemp_results_RPMs,
            n_qubits=6,
            temp_key="T_mK",
            err_key="T_mK_err",
            pe_key="P_e",
            pe_err_key="P_e_err_total",
            keep_nans=False,
    ):
        """
        Converts RPM per-file results into:
          times[qid]   = [datetime, ...]
          temps[qid]   = [T_mK, ...]
          errs[qid]    = [T_err_mK (or nan), ...]
          pe_vals[qid] = [P_e, ...]
          pe_errs[qid] = [P_e_err_total (or nan), ...]

        Robustness:
          * file_result["qubits"] may use int keys OR string keys.
          * If an error is missing, np.nan is appended so arrays stay aligned.
          * Non-finite values are skipped unless keep_nans=True.
        """

        times = [[] for _ in range(n_qubits)]
        temps = [[] for _ in range(n_qubits)]
        errs = [[] for _ in range(n_qubits)]
        pe_vals = [[] for _ in range(n_qubits)]
        pe_errs = [[] for _ in range(n_qubits)]

        file_list = all_files_Qtemp_results_RPMs or []
        for file_result in file_list:
            if not isinstance(file_result, dict):
                continue

            qubits_dict = file_result.get("qubits", {}) or {}
            if not isinstance(qubits_dict, dict):
                continue

            for qid in range(n_qubits):
                # accept both int and str keys
                qrec = qubits_dict.get(qid, None)
                if qrec is None:
                    qrec = qubits_dict.get(str(qid), None)

                if not isinstance(qrec, dict):
                    continue

                # ---------------- timestamp ----------------
                timestamp = qrec.get("date", None)
                time_val = datetime.datetime.fromtimestamp(timestamp)

                # ---------------- temperature ----------------
                T = qrec.get(temp_key, None)
                Te = qrec.get(err_key, None)

                try:
                    T = float(T) if T is not None else np.nan
                except Exception:
                    T = np.nan

                try:
                    Te = float(Te) if Te is not None else np.nan
                except Exception:
                    Te = np.nan

                # ---------------- Pe ----------------
                Pe = qrec.get(pe_key, None)
                Pee = qrec.get(pe_err_key, None)

                try:
                    Pe = float(Pe) if Pe is not None else np.nan
                except Exception:
                    Pe = np.nan

                try:
                    Pee = float(Pee) if Pee is not None else np.nan
                except Exception:
                    Pee = np.nan

                # ---------------- append ----------------
                if keep_nans or np.isfinite(T):
                    times[qid].append(time_val)
                    temps[qid].append(T)
                    errs[qid].append(Te if np.isfinite(Te) else np.nan)
                    pe_vals[qid].append(Pe)
                    pe_errs[qid].append(Pee if np.isfinite(Pee) else np.nan)

        return times, temps, errs, pe_vals, pe_errs

    def plot_rpm_pe_vs_shifted_time_by_run(self,
            run_num_list,
            rpm_times_by_run,
            rpm_Pe_by_run,
            rpm_Pe_errs_by_run,
            qubits_to_plot=None,
            num_qubits=6,
            time_units="hours",
            plot_percent=True,
            ylim=None,
            save_plt_path=None,
            fig_title=r"RPM $P_e$ vs shifted time across runs"
    ):
        """
        Plot RPM P_e vs shifted time for multiple runs.

        Each subplot is one qubit. Each run is shifted so that the first RPM
        timestamp for that qubit/run starts at t = 0.

        Parameters
        ----------
        run_num_list : list
            Runs to include, e.g. [6, 7, 8, 9].

        rpm_times_by_run : dict
            rpm_times_by_run[run_num][qid] = [datetime, ...]

        rpm_Pe_by_run : dict
            rpm_Pe_by_run[run_num][qid] = [P_e, ...]

        rpm_Pe_errs_by_run : dict
            rpm_Pe_errs_by_run[run_num][qid] = [P_e_err, ...]

        qubits_to_plot : list or None
            Qubit indices to plot. If None, plots all qubits.

        num_qubits : int
            Total number of qubits.

        time_units : str
            "seconds", "hours", or "days".

        plot_percent : bool
            If True, plots P_e in percent. If False, plots raw P_e.

        ylim : tuple, float, or None
            If tuple, uses ax.set_ylim(*ylim).
            If float, uses ax.set_ylim(0, ylim).

        save_plt_path : str or None
            Folder to save plot. If None, shows plot instead.

        fig_title : str
            Figure title.
        """

        if qubits_to_plot is None:
            qubits_to_plot = list(range(num_qubits))

        colors = ["orange", "blue", "purple", "green", "brown", "pink", "gray", "red"]
        font = 14

        nplot = len(qubits_to_plot)
        ncols = min(nplot, 3)
        nrows = math.ceil(nplot / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4.5 * ncols, 4 * nrows),
            sharex=False,
            constrained_layout=True
        )

        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        fig.suptitle(fig_title, fontsize=font + 2)

        for ax_i, qid in enumerate(qubits_to_plot):
            ax = axes[ax_i]

            for run_i, run_num in enumerate(run_num_list):
                if run_num not in rpm_times_by_run:
                    continue

                times = rpm_times_by_run[run_num][qid]
                pe_values = rpm_Pe_by_run[run_num][qid]
                pe_err_values = rpm_Pe_errs_by_run[run_num][qid]

                if len(times) == 0:
                    continue

                # Convert to arrays
                times_arr = np.asarray(times)
                pe_arr = np.asarray(pe_values, dtype=float)
                pe_err_arr = np.asarray(pe_err_values, dtype=float)

                # Shift each run so it starts at t = 0
                start_time = times_arr[0]
                shifted_seconds = np.asarray([
                    (t - start_time).total_seconds() for t in times_arr
                ])

                if time_units == "seconds":
                    shifted_time = shifted_seconds
                    xlabel = "Shifted time (s)"
                elif time_units == "hours":
                    shifted_time = shifted_seconds / 3600.0
                    xlabel = "Shifted time (hours)"
                elif time_units == "days":
                    shifted_time = shifted_seconds / (3600.0 * 24.0)
                    xlabel = "Shifted time (days)"
                else:
                    raise ValueError("time_units must be 'seconds', 'hours', or 'days'.")

                if plot_percent:
                    yvals = 100 * pe_arr
                    yerrs = 100 * pe_err_arr
                    ylabel = r"$P_e$ (%)"
                else:
                    yvals = pe_arr
                    yerrs = pe_err_arr
                    ylabel = r"$P_e$"

                ax.errorbar(
                    shifted_time,
                    yvals,
                    yerr=yerrs,
                    fmt=".",
                    color=colors[run_i % len(colors)],
                    ecolor=colors[run_i % len(colors)],
                    elinewidth=1,
                    capsize=3,
                    alpha=0.8,
                    label=f"Run {run_num}"
                )

                if len(yvals) > 0:
                    print(
                        f"Q{qid + 1}, Run {run_num}: "
                        f"first Pe = {yvals[0]:.4f}, "
                        f"last Pe = {yvals[-1]:.4f}, "
                        f"duration = {shifted_time[-1]:.2f} {time_units}"
                    )

            ax.set_title(f"Q{qid + 1}", fontsize=font)
            ax.set_xlabel(xlabel, fontsize=font)
            ax.set_ylabel(ylabel, fontsize=font)

            if ylim is not None:
                if isinstance(ylim, tuple):
                    ax.set_ylim(*ylim)
                else:
                    ax.set_ylim(0, ylim)

            ax.tick_params(axis="x", labelrotation=45, labelsize=12)
            ax.tick_params(axis="y", labelsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        # Hide unused subplot axes
        for j in range(len(qubits_to_plot), len(axes)):
            axes[j].axis("off")

        if save_plt_path is not None:
            os.makedirs(save_plt_path, exist_ok=True)

            timestp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_file = os.path.join(
                save_plt_path,
                f"RPM_Pe_vs_shifted_time_by_run_{timestp}.pdf"
            )

            print("Plot saved to:", save_file)
            plt.savefig(save_file)
            plt.close(fig)
        else:
            plt.show()

    def plot_ssf_log_overlay_by_run(
            self,
            fit_results_by_run,
            qid,
            save_figs_path=None,
            bins=np.linspace(-0.75, 1.75, 220),
            run_order=None,
            cmap_name="Blues",
            plot_individual=False,
            plot_run_median=True,
            individual_alpha=0.22,
            individual_lw=1.0,
            median_lw=2.8,
            smooth_window=1,
            ymin=1e-4,
            ymax=1.5,
            title=None,
            filename=None,
            fig_quality=300,
            show=True,
    ):
        """
        Overlay SSF ground-state histograms from multiple runs on one log-y plot.

        Each dataset is normalized so that:
            main |g> / 0-state peak -> x = 0
            thermal/excited / 1-state peak -> x = 1
            histogram maximum -> y = 1

        This is meant to show whether the thermal-excitation shoulder near x=1
        decreases from run to run.

        Parameters
        ----------
        fit_results_by_run : dict
            Example:
                {
                    5: fit_results_run5,
                    6: fit_results_run6,
                    7: fit_results_run7,
                    8: fit_results_run8,
                    9: fit_results_run9,
                }

            where each fit_results_runX has the format returned by
            run_ssf_qtemps_iminuit():
                fit_results[qid] = [record1, record2, ...]

        qid : int
            Zero-indexed qubit index.

        bins : np.ndarray
            Bins for normalized SSF axis.

        cmap_name : str
            Matplotlib colormap name. Good options:
                "Blues", "Reds", "Purples", "viridis", "plasma"

        plot_individual : bool
            If True, plot every valid SSF histogram faintly.

        plot_run_median : bool
            If True, plot the median histogram for each run as a thicker line.

        smooth_window : int
            Optional moving-average smoothing window.
            Use 1 for no smoothing.

        Returns
        -------
        fig, ax, run_histograms
            run_histograms[run_num] contains normalized histograms and metadata.
        """

        def _smooth(y, window):
            if window is None or window <= 1:
                return y

            kernel = np.ones(window) / window
            return np.convolve(y, kernel, mode="same")

        if run_order is None:
            run_order = sorted(fit_results_by_run.keys())

        centers = 0.5 * (bins[:-1] + bins[1:])

        cmap = plt.get_cmap(cmap_name)

        # Use lighter colors for earlier runs and darker colors for later runs
        color_vals = np.linspace(0.35, 0.95, len(run_order))
        run_colors = {
            run: cmap(color_vals[i])
            for i, run in enumerate(run_order)
        }

        fig, ax = plt.subplots(figsize=(9.5, 6.5))

        run_histograms = {}

        for run_num in run_order:
            fit_results = fit_results_by_run[run_num]

            if qid not in fit_results or len(fit_results[qid]) == 0:
                print(f"Skipping Run {run_num}, Q{qid + 1}: no fit results.")
                continue

            records = fit_results[qid]

            hists_this_run = []
            records_used = []

            for rec in records:
                needed_keys = ["ig_new", "means", "ground_gaussian", "excited_gaussian"]

                if not all(k in rec for k in needed_keys):
                    continue

                if rec["ig_new"] is None or rec["means"] is None:
                    continue

                ig_new = np.asarray(rec["ig_new"]).ravel()
                means = np.asarray(rec["means"])

                ground_idx = int(rec["ground_gaussian"])
                excited_idx = int(rec["excited_gaussian"])

                ground_mean = means[ground_idx]
                excited_mean = means[excited_idx]

                separation = excited_mean - ground_mean

                if np.isclose(separation, 0):
                    print(
                        f"Skipping Run {run_num}, Q{qid + 1}, "
                        f"dataset {rec.get('dataset', 'unknown')}: means overlap."
                    )
                    continue

                # Important:
                # This maps ground_mean -> 0 and excited_mean -> 1.
                # Even if separation is negative, this still correctly maps the excited mean to +1.
                ig_norm = (ig_new - ground_mean) / separation

                counts, _ = np.histogram(ig_norm, bins=bins)

                counts = counts.astype(float)

                # Normalize main histogram maximum to 1
                max_count = np.max(counts)
                if max_count <= 0:
                    continue

                counts = counts / max_count

                # Optional smoothing
                counts = _smooth(counts, smooth_window)

                # Avoid plotting zeros on log scale
                counts[counts <= 0] = np.nan

                hists_this_run.append(counts)
                records_used.append(rec)

                if plot_individual:
                    ax.plot(
                        centers,
                        counts,
                        color=run_colors[run_num],
                        alpha=individual_alpha,
                        linewidth=individual_lw,
                    )

            if len(hists_this_run) == 0:
                print(f"Skipping Run {run_num}, Q{qid + 1}: no valid histograms.")
                continue

            hists_this_run = np.asarray(hists_this_run)

            run_histograms[run_num] = {
                "histograms": hists_this_run,
                "records": records_used,
                "color": run_colors[run_num],
            }

            if plot_run_median:
                median_hist = np.nanmedian(hists_this_run, axis=0)
                median_hist = _smooth(median_hist, smooth_window)
                median_hist[median_hist <= 0] = np.nan

                # Helpful summary for legend
                pe_vals = np.array([r.get("Pe", np.nan) for r in records_used], dtype=float)
                temp_vals = np.array([r.get("temperature_mK", np.nan) for r in records_used], dtype=float)

                med_pe = np.nanmedian(pe_vals)
                med_temp = np.nanmedian(temp_vals)

                label = f"Run {run_num}"

                if np.isfinite(med_pe):
                    label += f", med Pe={med_pe:.3f}"

                if np.isfinite(med_temp):
                    label += f", med T={med_temp:.0f} mK"

                ax.plot(
                    centers,
                    median_hist,
                    color=run_colors[run_num],
                    alpha=1.0,
                    linewidth=median_lw,
                    label=label,
                )

        ax.axvline(0, linestyle="--", linewidth=1.4, color="gray", alpha=0.8)
        ax.axvline(1, linestyle="--", linewidth=1.4, color="red", alpha=0.8)

        ax.text(
            0.02,
            0.94,
            r"$|g\rangle$ peak aligned",
            transform=ax.transAxes,
            fontsize=11,
            color="gray",
        )

        ax.text(
            0.58,
            0.94,
            r"thermal / $|e\rangle$ peak aligned",
            transform=ax.transAxes,
            fontsize=11,
            color="red",
        )

        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(bins[0], bins[-1])

        ax.set_xlabel(
            "Normalized rotated SSF axis\n"
            r"main $|g\rangle$ peak = 0, thermal/$|e\rangle$ peak = 1",
            fontsize=13,
        )

        ax.set_ylabel("Normalized counts", fontsize=13)

        if title is None:
            title = f"Q{qid + 1} SSF thermal population comparison across runs"

        ax.set_title(title, fontsize=16)

        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, which="both", alpha=0.25)

        plt.tight_layout()

        if save_figs_path is not None:
            os.makedirs(save_figs_path, exist_ok=True)

            if filename is None:
                filename = f"Q{qid + 1}_SSF_log_overlay_by_run.png"

            full_path = os.path.join(save_figs_path, filename)
            fig.savefig(full_path, dpi=fig_quality, bbox_inches="tight")
            print(f"Saved: {full_path}")

        if show:
            plt.show()

        return fig, ax, run_histograms

    def plot_t1t2_vs_qtemps(self, out_dir, all_qubit_temperatures_ssf_g = None, all_qubit_timestamps_ssf_g = None,
                          all_files_Qtemp_results_RPMs = None, t1_vals = None, t1_dates = None, t2r_vals = None, t2r_dates = None,
                            t2e_vals=None, t2e_dates=None, restrict_time_xaxis=False, start_time = None, end_time = None,
                            plot_extra_event_lines=False, qbt_to_plt = [0,1,2,3,4,5], max_match_dt_s=10, save_name="T1_T2_vs_Qtemp.png"):

        """
        Plots T1/T2R/T2E vs effective qubit temperature.

        Layout:
          - rows = len(qbt_to_plt) (each row is a qubit)
          - cols = 2
              col 0: RPM temperatures
              col 1: SSF temperatures

        Matching strategy:
          - For each coherence datapoint timestamp, match to the nearest temperature timestamp
            within max_match_delta_s. Otherwise skip (left as NaN and not plotted).

        Expected input shapes (flexible, but these are the “happy path”):
          - t1_vals[q], t1_dates[q]  are arrays for qubit q (same length)
          - t2r_vals[q], t2r_dates[q]
          - t2e_vals[q], t2e_dates[q]
          - all_qubit_temperatures_ssf_g[q], all_qubit_timestamps_ssf_g[q]

          - all_files_Qtemp_results_RPMs:
              list of per-file dicts. Each dict should contain (any one of these patterns works):
                A) dict["timestamps"][q] and dict["temps_mK"][q]
                B) dict["qubits"][q]["timestamps"] and dict["qubits"][q]["temps_mK"]
              (If your keys differ, tweak _extract_rpm_for_qubit below.)
        """
        os.makedirs(out_dir, exist_ok=True)

        qbt_to_plt = list(qbt_to_plt)
        nrows = len(qbt_to_plt)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=2,
            figsize=(13, max(3.0, 2.6 * nrows)),
            constrained_layout=True,
        )
        if nrows == 1:
            axes = np.array([axes])

        metrics = [
            ("T1", t1_vals, t1_dates, "o"),
            ("T2R", t2r_vals, t2r_dates, "s"),
            ("T2E", t2e_vals, t2e_dates, "^"),
        ]

        for row, q in enumerate(qbt_to_plt):
            ax_rpm: Axes = axes[row, 0]
            ax_ssf: Axes = axes[row, 1]

            # --------------------- Gather RPM temps for this qubit ---------------------
            rpm_times_list = []
            rpm_temps_list = []

            if all_files_Qtemp_results_RPMs is not None:
                for f in all_files_Qtemp_results_RPMs:
                    # Expected: f["qubits"][q]["timestamps"], f["qubits"][q]["temps_mK"]
                    if isinstance(f, dict) and ("qubits" in f) and (q in f["qubits"]):
                        qb = f["qubits"][q]
                        if ("timestamps" in qb) and ("temps_mK" in qb):
                            rpm_times_list.extend(list(qb["timestamps"]))
                            rpm_temps_list.extend(list(qb["temps_mK"]))

            if len(rpm_times_list) > 0:
                rpm_times = np.array(rpm_times_list, dtype="datetime64[ns]")
                rpm_temps = np.array(rpm_temps_list, dtype=float)
            else:
                rpm_times = np.array([], dtype="datetime64[ns]")
                rpm_temps = np.array([], dtype=float)

            # sort RPM by time (required for before/after matching)
            if len(rpm_times) > 0:
                o = np.argsort(rpm_times)
                rpm_times = rpm_times[o]
                rpm_temps = rpm_temps[o]

            if restrict_time_xaxis and len(rpm_times) > 0:
                m = np.ones(len(rpm_times), dtype=bool)
                if start_time is not None:
                    m &= (rpm_times >= np.datetime64(start_time))
                if end_time is not None:
                    m &= (rpm_times <= np.datetime64(end_time))
                rpm_times = rpm_times[m]
                rpm_temps = rpm_temps[m]

            # --------------------- Gather SSF temps for this qubit ---------------------
            if (
                    all_qubit_temperatures_ssf_g is not None
                    and all_qubit_timestamps_ssf_g is not None
                    and q < len(all_qubit_timestamps_ssf_g)
                    and q < len(all_qubit_temperatures_ssf_g)
                    and all_qubit_timestamps_ssf_g[q] is not None
                    and all_qubit_temperatures_ssf_g[q] is not None
            ):
                ssf_times = np.array(all_qubit_timestamps_ssf_g[q], dtype="datetime64[ns]")
                ssf_temps = np.array(all_qubit_temperatures_ssf_g[q], dtype=float)
            else:
                ssf_times = np.array([], dtype="datetime64[ns]")
                ssf_temps = np.array([], dtype=float)

            # sort SSF by time (required for before/after matching)
            if len(ssf_times) > 0:
                o = np.argsort(ssf_times)
                ssf_times = ssf_times[o]
                ssf_temps = ssf_temps[o]

            if restrict_time_xaxis and len(ssf_times) > 0:
                m = np.ones(len(ssf_times), dtype=bool) # Start by assuming every SSF temperature point is valid
                if start_time is not None:
                    m &= (ssf_times >= np.datetime64(start_time)) # “Keep only those after start_time”
                if end_time is not None:
                    m &= (ssf_times <= np.datetime64(end_time))  # “Keep only those before end_time”

                # Apply mask
                ssf_times = ssf_times[m]
                ssf_temps = ssf_temps[m]

            # --------------------- Plot coherence metrics ---------------------
            for name, vals_by_q, dates_by_q, marker in metrics:
                if vals_by_q is None or dates_by_q is None:
                    continue
                if q >= len(vals_by_q) or q >= len(dates_by_q):
                    continue
                if vals_by_q[q] is None or dates_by_q[q] is None:
                    continue

                y = np.array(vals_by_q[q], dtype=float)
                t = np.array(dates_by_q[q], dtype="datetime64[ns]")

                if len(y) == 0 or len(t) == 0:
                    continue

                if restrict_time_xaxis:
                    m = np.ones(len(t), dtype=bool) # Start by assuming every SSF temperature point is valid
                    if start_time is not None:
                        m &= (t >= np.datetime64(start_time)) # “Keep only those after start_time”
                    if end_time is not None:
                        m &= (t <= np.datetime64(end_time)) # “Keep only those before end_time”

                    # Apply mask
                    t = t[m]
                    y = y[m]
                    if len(t) == 0:
                        continue

                # -------- RPM match using before/after --------
                x_rpm = np.full(len(t), np.nan, dtype=float)
                if len(rpm_times) > 0:
                    idx = np.searchsorted(rpm_times, t, side="left")  # insertion points
                    for i in range(len(t)):
                        j = int(idx[i])

                        best_temp = np.nan
                        best_dt = np.inf

                        # candidate before
                        if j - 1 >= 0:
                            dt_before = abs((t[i] - rpm_times[j - 1]).astype("timedelta64[s]").astype(float))
                            if dt_before < best_dt:
                                best_dt = dt_before
                                best_temp = float(rpm_temps[j - 1])

                        # candidate after
                        if j < len(rpm_times):
                            dt_after = abs((t[i] - rpm_times[j]).astype("timedelta64[s]").astype(float))
                            if dt_after < best_dt:
                                best_dt = dt_after
                                best_temp = float(rpm_temps[j])

                        if best_dt <= max_match_dt_s:
                            x_rpm[i] = best_temp

                # -------- SSF match using before/after --------
                x_ssf = np.full(len(t), np.nan, dtype=float)
                if len(ssf_times) > 0:
                    idx = np.searchsorted(ssf_times, t, side="left")
                    for i in range(len(t)):
                        j = int(idx[i])

                        best_temp = np.nan
                        best_dt = np.inf

                        if j - 1 >= 0:
                            dt_before = abs((t[i] - ssf_times[j - 1]).astype("timedelta64[s]").astype(float))
                            if dt_before < best_dt:
                                best_dt = dt_before
                                best_temp = float(ssf_temps[j - 1])

                        if j < len(ssf_times):
                            dt_after = abs((t[i] - ssf_times[j]).astype("timedelta64[s]").astype(float))
                            if dt_after < best_dt:
                                best_dt = dt_after
                                best_temp = float(ssf_temps[j])

                        if best_dt <= max_match_dt_s:
                            x_ssf[i] = best_temp

                ok_y = np.isfinite(y)

                ok_rpm = ok_y & np.isfinite(x_rpm)
                if np.any(ok_rpm):
                    ax_rpm.scatter(x_rpm[ok_rpm], y[ok_rpm], marker=marker, label=name, alpha=0.85)

                ok_ssf = ok_y & np.isfinite(x_ssf)
                if np.any(ok_ssf):
                    ax_ssf.scatter(x_ssf[ok_ssf], y[ok_ssf], marker=marker, label=name, alpha=0.85)

            # --------------------- Cosmetics ---------------------
            ax_rpm.set_title(f"Q{q + 1} — RPM temps")
            ax_ssf.set_title(f"Q{q + 1} — SSF temps")
            ax_rpm.set_xlabel("Effective qubit temperature (mK)")
            ax_ssf.set_xlabel("Effective qubit temperature (mK)")
            ax_rpm.set_ylabel("Coherence (µs)")

            ax_rpm.grid(alpha=0.3)
            ax_ssf.grid(alpha=0.3)
            ax_rpm.legend(fontsize=9, loc="best")
            ax_ssf.legend(fontsize=9, loc="best")

            if plot_extra_event_lines:
                ax_rpm.text(
                    0.02, 0.95,
                    "plot_extra_event_lines=True\n(no-op for temp-x plots)",
                    transform=ax_rpm.transAxes,
                    va="top",
                    fontsize=8,
                    alpha=0.7,
                )

        savepath = os.path.join(out_dir, save_name)
        fig.savefig(savepath)
        plt.close(fig)

        return savepath


    def Qtemps_vs_time_comb_methods_3col(self, all_qubit_temperatures_ssf_g, all_qubit_timestamps_ssf_g, all_qubit_temps_ssf_errs_g, all_qubit_temperatures_ssf_ge, all_qubit_timestamps_ssf_ge,
                                    all_qubit_temps_ssf_errs_ge, out_dir, all_files_Qtemp_results_RPMs, restrict_time_xaxis = False, plot_extra_event_lines = False, rad_events_plot_lines = True,
                                    plot_error_bars = False):
        """
        Plots qubit temperatures vs time for two qubits, using temperature data obtained using these three methods:
        1. Rabi population measurements
        2. Fitting the ssf prepared ground state data to a double gaussian and using the means of the two gaussians to calculate
            the midpoint and use that as the population threshold.
        3. Fitting the ssf prepared ground state data AND the prepared excited state data to a double gaussian and using the means of
            the two gaussians to calculate the midpoint and use that as the population threshold (this represents the ssf g-e threshold).

        This function returns a plot that contains two rows (one for each qubit) showcasing the results for each method in a
        separate subplot (column).
        """

        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        os.makedirs(out_dir, exist_ok=True)
        err_filter = 200 # used to filter out datapoints with an error above this value

        # Processing RPMs Qubit Temperature Results and putting it into dicts:
        times_RPM = {q: [] for q in range(self.number_of_qubits)}
        temps_RPM = {q: [] for q in range(self.number_of_qubits)}
        errs_RPM = {q: [] for q in range(self.number_of_qubits)}

        for rec in all_files_Qtemp_results_RPMs:
            for q in range(self.number_of_qubits):
                d = rec['qubits'].get(q)
                if not d: # no data
                    continue

                if d['T_mK_err'] <= err_filter: # Only keep if T_mK_err ≤ 300 mK
                    err_rpm = d['T_mK_err']
                    errs_RPM[q].append(err_rpm)
                    t = datetime.datetime.fromtimestamp(d['date'])
                    times_RPM[q].append(t)
                    temps_RPM[q].append(d['T_mK'])

        # SSF Qubit Temperature data for method that uses g-state double gauss threshold as population threshold
        times_ssf_g = {
            q: [t for t, e in zip(all_qubit_timestamps_ssf_g[q], all_qubit_temps_ssf_errs_g[q]) if e <= err_filter]
            for q in all_qubit_temperatures_ssf_g
        }
        temps_ssf_g = {
            q: [y for y, e in zip(all_qubit_temperatures_ssf_g[q], all_qubit_temps_ssf_errs_g[q]) if e <= err_filter]
            for q in all_qubit_temperatures_ssf_g
        }
        errs_ssf_g = {
            q: [e for e in all_qubit_temps_ssf_errs_g[q] if e <= err_filter]
            for q in all_qubit_temperatures_ssf_g
        }

        # SSF Qubit Temperature data for method that uses g-e ssf threshold as population threshold
        # This data is filtered (has temperature errors below err_filter)
        times_ssf_ge = {
            q: [t for t, e in zip(all_qubit_timestamps_ssf_ge[q], all_qubit_temps_ssf_errs_ge[q]) if e <= err_filter]
            for q in all_qubit_temperatures_ssf_ge
        }
        temps_ssf_ge = {
            q: [y for y, e in zip(all_qubit_temperatures_ssf_ge[q], all_qubit_temps_ssf_errs_ge[q]) if e <= err_filter]
            for q in all_qubit_temperatures_ssf_ge
        }
        errs_ssf_ge = {
            q: [e for e in all_qubit_temps_ssf_errs_ge[q] if e <= err_filter]
            for q in all_qubit_temperatures_ssf_ge
        }

        #----------- Plot only a certain range of dates/time (only goes into effect if restrict_time_xaxis is set to true)
        if restrict_time_xaxis:
            window_start = datetime.datetime(2025, 4, 18, 0, 0)
            window_end = datetime.datetime(2025, 5, 4, 23, 59)

        # ---------- Radiation source events
        rad_events = []
        if rad_events_plot_lines:
            rad_events = [
                (datetime.datetime(2025, 4, 21, 12, 35), "Co-60"),
                (datetime.datetime(2025, 4, 23, 12, 53), "Cs-137"),
                (datetime.datetime(2025, 4, 28, 9, 40), "Cs-137 closer"),
                (datetime.datetime(2025, 5, 4, 18, 20), "Cs-137 removed"),
            ]

        #-------------- Plotting
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True, sharex=True, constrained_layout=True)
        col_titles = ["Method #1: Rabi Pop. Meas.", "Method #2: g-state double gaussian", "Method #3: g & e-state double gaussian"]

        for c, title in enumerate(col_titles):
            axes[0, c].set_title(title, fontsize=16, pad=12)

        # date formatter
        date_fmt = DateFormatter('%m-%d-%H')

        # Plot each qubit (rows) × method (cols)
        for row, q in enumerate([0, 4]):  # Q1 and Q5
            for col in range(3):
                ax = axes[row, col]

                if col == 0:
                    ts, ys, es = times_RPM[q], temps_RPM[q], errs_RPM[q]
                elif col == 1:
                    ts, ys, es = times_ssf_g.get(q, []), temps_ssf_g.get(q, []), errs_ssf_g.get(q, [])
                else:
                    ts, ys, es = times_ssf_ge.get(q, []), temps_ssf_ge.get(q, []), errs_ssf_ge.get(q, [])

                if not ts or not ys:
                    ax.set_visible(False)
                    continue

                # scatter
                color = colors[q % len(colors)]
                if plot_error_bars:
                    ax.errorbar(
                        ts,
                        ys,
                        yerr=es,
                        fmt='o',
                        capsize=4,
                        markersize=5,
                        color=color,
                        ecolor=color,
                        label=f"Q{q + 1}"
                    )
                else:
                    ax.scatter(
                        ts,
                        ys,
                        s=40,
                        alpha=0.7,
                        color=color,
                        label=f"Q{q + 1}"
                    )

                # qubit label
                ax.text(0.02, 0.95, f"Q{q + 1}", transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

                # # individual x-axis formatting
                # ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                # ax.xaxis.set_major_formatter(date_fmt)
                # plt.setp(ax.get_xticklabels(), rotation=45, fontsize=10)

                # y-axis on leftmost col
                if col == 0:
                    ax.set_ylabel("Temp (mK)", fontsize=12)
                # x-axis on bottom row
                if row == 1:
                    ax.set_xlabel("Time", fontsize=12)

                ax.set_ylim(50, 950)
                ax.grid(False)

                # apply time window
                if restrict_time_xaxis:
                    ax.set_xlim(window_start, window_end)

                # add radiation lines
                for t_evt, lbl in rad_events:
                    ax.axvline(t_evt, color='black', linestyle='--', linewidth=1)
                    ax.text(t_evt, ax.get_ylim()[1] * 0.9, lbl, rotation=90, va='top', ha='right', fontsize=9)

        locator = mdates.AutoDateLocator()
        for ax in axes.flatten():
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(date_fmt)
            ax.tick_params(axis='x', labelrotation=45, labelsize=10)

        fig.suptitle("Qubit Temperatures vs Time", fontsize=18)
        paramvstime_dir = os.path.join(out_dir, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(paramvstime_dir, f"Qtemps_vs_Time_methods_comparisons_{stamp}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print("Plot saved to →", out_path)

    def Qtemps_vs_time_comb_2subplts(self, all_qubit_temperatures_ssf_g, all_qubit_timestamps_ssf_g, all_qubit_temperatures_ssf_ge,
                                    all_qubit_timestamps_ssf_ge, out_dir: str, all_files_Qtemp_results_RPMs, restrict_time_xaxis = False,
                                    plot_extra_event_lines = False, rad_events_plot_lines = False):
        """
        Plots qubit temperatures vs time for two qubits (set up for Q1 and Q5), using temperature data obtained using these three methods:
        1. Rabi population measurements
        2. Fitting the ssf prepared ground state data to a double gaussian and using the means of the two gaussians to calculate
            the midpoint and use that as the population threshold.
        3. Fitting the ssf prepared ground state data AND the prepared excited state data to a double gaussian and using the means of
            the two gaussians to calculate the midpoint and use that as the population threshold (this represents the ssf g-e threshold).

        This function returns a plot that contains two rows (one for each qubit) showcasing the results for each method in a
        SINGLE subplot for each qubit (so just 1 column).
        """

        os.makedirs(out_dir, exist_ok=True)

        # Build rabi population measurement (RPM) data dicts
        num_qubits = self.number_of_qubits
        times_RPM = {q: [] for q in range(num_qubits)}
        temps_RPM = {q: [] for q in range(num_qubits)}
        for rec in all_files_Qtemp_results_RPMs:
            for q, lst in times_RPM.items():
                d = rec["qubits"].get(q)
                if d:
                    t = datetime.datetime.fromtimestamp(d["date"])
                    times_RPM[q].append(t)
                    temps_RPM[q].append(d["T_mK"])

        # SSF data (methods #2 and #3)
        times_g = all_qubit_timestamps_ssf_g
        temps_g = all_qubit_temperatures_ssf_g
        times_ge = all_qubit_timestamps_ssf_ge
        temps_ge = all_qubit_temperatures_ssf_ge

        # Optional time window (only goes into effect if restrict_time_xaxis = True)
        if restrict_time_xaxis:
            window_start = datetime.datetime(2025, 4, 18, 0, 0)
            window_end = datetime.datetime(2025, 5, 4, 23, 59)

        # Radiation‐event lines, optional too.
        rad_events = []
        if rad_events_plot_lines:
            rad_events = [
                (datetime.datetime(2025, 4, 21, 12, 35), "Co-60"),
                (datetime.datetime(2025, 4, 23, 12, 53), "Cs-137"),
                (datetime.datetime(2025, 4, 28, 9, 40), "Cs-137 closer"),
                (datetime.datetime(2025, 5, 4, 18, 20), "Cs-137 removed"),
            ]

        # Two subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        date_fmt = DateFormatter('%m-%d-%H')

        methods = [
            ("RPM Pop. Meas.", times_RPM, temps_RPM, "orange"),
            ("SSF g-only", times_g, temps_g, "blue"),
            ("SSF g+e", times_ge, temps_ge, "red")
        ]

        # Plot per qubit
        for ax, q in zip(axes, [0, 4]):  # Q1 (0) and Q5 (4)
            for label, tdict, ydict, color in methods:
                ts = tdict.get(q, [])
                ys = ydict.get(q, [])
                if ts and ys:
                    ax.scatter(ts, ys,
                               label=label,
                               s=30,
                               alpha=0.8,
                               edgecolors='k',
                               color=color)

            ax.set_title(f"Q{q + 1}", loc="left", fontsize=14, fontweight="bold")
            ax.set_ylabel("Temp (mK)")
            ax.grid(False)

            # common x‐formatter
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(date_fmt)
            ax.tick_params(axis='x', rotation=45, labelsize=10)

            if restrict_time_xaxis:
                ax.set_xlim(window_start, window_end)

            # radiation events
            for t_evt, lbl in rad_events:
                ax.axvline(t_evt, color='gray', linestyle='--', linewidth=1)
                ax.text(t_evt, ax.get_ylim()[1] * 0.9,
                        lbl, rotation=90,
                        va='top', ha='right', fontsize=9)

            ax.legend(loc="upper left", fontsize=10)

        axes[-1].set_xlabel("Time")
        fig.suptitle("Qubit Temperatures vs Time", fontsize=16)

        # Save
        paramvstime_dir = os.path.join(out_dir, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(paramvstime_dir, f"Qtemps_TwoMethodsCompare_{stamp}.png")
        fig.savefig(out_path, dpi=self.figure_quality)
        plt.close(fig)
        print("Saved combined methods plot →", out_path)

    def Pe_vs_time_comb_methods(self, all_files_Qtemp_results_RPMs, fit_results_g, fit_results_ge, out_dir,
                                restrict_time_xaxis = False, plot_extra_event_lines = False, rad_events_plot_lines = False):
        """
        Plots thermal population (Pe) vs time for two qubits, using data obtained using these three methods:
        1. Rabi population measurements
        2. Fitting the ssf prepared ground state data to a double gaussian and using the means of the two gaussians to calculate
            the midpoint and use that as the population threshold.
        3. Fitting the ssf prepared ground state data AND the prepared excited state data to a double gaussian and using the means of
            the two gaussians to calculate the midpoint and use that as the population threshold (this represents the ssf g-e threshold).

        This function returns a plot that contains two rows (one for each qubit) showcasing the results for each method in a
        separate subplot (column).
        """
        os.makedirs(out_dir, exist_ok=True)
        date_fmt = DateFormatter('%m-%d-%H')

        # Extract RPM Pe data into dicts
        num_qubits = self.number_of_qubits
        times_RPM = {q: [] for q in range(num_qubits)}
        pops_RPM = {q: [] for q in range(num_qubits)}
        for rec in all_files_Qtemp_results_RPMs:
            for q, d in rec['qubits'].items():
                # d contains 'date' and 'P_e'
                t = datetime.datetime.fromtimestamp(d['date'])
                times_RPM[q].append(t)
                pops_RPM[q].append(d['P_e'])

        # Gather SSF‐g and SSF‐g+e Pe data from fit_results
        times_g = {q: [r['timestamp'] for r in lst] for q, lst in fit_results_g.items()}
        pops_g = {q: [r['Pe'] for r in lst] for q, lst in fit_results_g.items()}
        times_ge = {q: [r['timestamp'] for r in lst] for q, lst in fit_results_ge.items()}
        pops_ge = {q: [r['Pe'] for r in lst] for q, lst in fit_results_ge.items()}

        # Optional time window
        if restrict_time_xaxis:
            window_start = datetime.datetime(2025, 4, 18, 0, 0)
            window_end = datetime.datetime(2025, 5, 4, 23, 59)

        # Also optional, Radiation events
        rad_events = []
        if rad_events_plot_lines:
            rad_events = [
                (datetime.datetime(2025, 4, 21, 12, 35), "Co-60"),
                (datetime.datetime(2025, 4, 23, 12, 53), "Cs-137"),
                (datetime.datetime(2025, 4, 28, 9, 40), "Cs-137 closer"),
                (datetime.datetime(2025, 5, 4, 18, 20), "Cs-137 removed"),
            ]

        # Create 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True, sharex=True, constrained_layout=True)
        col_titles = ["Method #1: Rabi Pop. Meas.",
                      "Method #2: SSF g‐only",
                      "Method #3: SSF g+e"]
        for c, title in enumerate(col_titles):
            axes[0, c].set_title(title, fontsize=16, pad=12)

        # Plot each qubit (rows) × method (cols)
        method_data = [
            (times_RPM, pops_RPM, 'orange'),
            (times_g, pops_g, 'blue'),
            (times_ge, pops_ge, 'red')]

        for row, q in enumerate([0, 4]):  # Q1 and Q5
            for col, (t_dict, p_dict, color) in enumerate(method_data):
                ax = axes[row, col]
                ts = t_dict.get(q, [])
                ps = p_dict.get(q, [])
                if ts and ps:
                    ax.scatter(ts, ps,
                               s=40, alpha=0.8,
                               edgecolors='k', color=color)

                ax.text(0.02, 0.95, f"Q{q + 1}",
                        transform=ax.transAxes,
                        fontsize=14, fontweight='bold', va='top')
                ax.set_ylabel("Pe" if col == 0 else "")
                if row == 1:
                    ax.set_xlabel("Time")
                ax.set_ylim(0, 1)
                ax.grid(False)

                # apply shared x‐axis ticks & labels
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(date_fmt)
                ax.tick_params(axis='x', rotation=45, labelsize=10)

                if restrict_time_xaxis:
                    ax.set_xlim(window_start, window_end)

                if rad_events_plot_lines:
                    for t_evt, lbl in rad_events:
                        ax.axvline(t_evt, color='gray', linestyle='--', linewidth=1)
                        ax.text(t_evt, ax.get_ylim()[1] * 0.9,
                                lbl, rotation=90, va='top', ha='right', fontsize=9)

        fig.suptitle("Thermal Population vs Time", fontsize=18)

        # Save
        paramvstime_dir = os.path.join(out_dir, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(paramvstime_dir, f"Pe_vs_Time_comb_methods_{stamp}.png")
        fig.savefig(out_path, dpi=self.figure_quality)
        plt.close(fig)
        print("Plot saved to →", out_path)

    def Pe_vs_time_comb_2subplts(self, all_files_Qtemp_results_RPMs: list, fit_results_g: dict, fit_results_ge: dict, out_dir: str,
                                 restrict_time_xaxis: bool = False, plot_extra_event_lines: bool = False, rad_events_plot_lines: bool = False):
        """
        Plots thermal populations vs time for two qubits (set up for Q1 and Q5), using data obtained via these three methods:
        1. Rabi population measurements
        2. Fitting the ssf prepared ground state data to a double gaussian and using the means of the two gaussians to calculate
            the midpoint and use that as the population threshold.
        3. Fitting the ssf prepared ground state data AND the prepared excited state data to a double gaussian and using the means of
            the two gaussians to calculate the midpoint and use that as the population threshold (this represents the ssf g-e threshold).

        This function returns a plot that contains two rows (one for each qubit) showcasing the results for each method in a
        SINGLE subplot for each qubit (so just 1 column).
        """

        os.makedirs(out_dir, exist_ok=True)
        # transforms rabi population measurement (RPM) P_e data into dictionaries
        num_qubits = self.number_of_qubits
        times_RPM = {q: [] for q in range(num_qubits)}
        pops_RPM = {q: [] for q in range(num_qubits)}
        for rec in all_files_Qtemp_results_RPMs:
            for q in range(num_qubits):
                d = rec["qubits"].get(q)
                if not d:
                    continue
                t = datetime.datetime.fromtimestamp(d["date"])
                times_RPM[q].append(t)
                pops_RPM[q].append(d["P_e"])

        # SSF‐g P_e data: fit_results_g[q] is a list of dicts with keys "timestamp" and "Pe"
        times_g = {}
        pops_g = {}
        for q, lst in fit_results_g.items():
            times_g[q] = [item["timestamp"] for item in lst]
            pops_g[q] = [item["Pe"] for item in lst]

        # SSF‐g+e P_e data:
        times_ge = {}
        pops_ge = {}
        for q, lst in fit_results_ge.items():
            times_ge[q] = [item["timestamp"] for item in lst]
            pops_ge[q] = [item["Pe"] for item in lst]

        # Optional time window
        if restrict_time_xaxis:
            window_start = datetime.datetime(2025, 4, 18, 0, 0)
            window_end = datetime.datetime(2025, 5, 4, 23, 59)

        # Optional Radiation events
        rad_events = []
        if rad_events_plot_lines:
            rad_events = [
                (datetime.datetime(2025, 4, 21, 12, 35), "Co-60"),
                (datetime.datetime(2025, 4, 23, 12, 53), "Cs-137"),
                (datetime.datetime(2025, 4, 28, 9, 40), "Cs-137 closer"),
                (datetime.datetime(2025, 5, 4, 18, 20), "Cs-137 removed"),
            ]

        # 2‐row subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        date_fmt = DateFormatter('%m-%d-%H')

        methods = [
            ("RPM Pop. Meas.", times_RPM, pops_RPM, "orange"),
            ("SSF g-only", times_g, pops_g, "blue"),
            ("SSF g+e", times_ge, pops_ge, "red")
        ]

        # Plot for Q1 & Q5
        for ax, q in zip(axes, [0, 4]):
            for label, tdict, pdict, color in methods:
                ts = tdict.get(q, [])
                ps = pdict.get(q, [])
                if ts and ps:
                    ax.scatter(
                        ts, ps,
                        label=label,
                        s=30, alpha=0.8,
                        edgecolors='k',
                        color=color
                    )

            ax.set_title(f"Q{q + 1}", loc="left", fontsize=14, fontweight="bold")
            ax.set_ylabel("Thermal Population ($P_e$)", fontsize=12)
            ax.set_ylim(0, 1)
            ax.grid(False)

            # x‐axis formatting
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(date_fmt)
            ax.tick_params(axis='x', rotation=45, labelsize=10)

            if restrict_time_xaxis:
                ax.set_xlim(window_start, window_end)

            # radiation lines
            for t_evt, lbl in rad_events:
                ax.axvline(t_evt, color='gray', linestyle='--', linewidth=1)
                ax.text(t_evt,
                        ax.get_ylim()[1] * 0.9,
                        lbl,
                        rotation=90,
                        va='top',
                        ha='right',
                        fontsize=9)

            ax.legend(loc="upper left", fontsize=10)

        axes[-1].set_xlabel("Time", fontsize=12)
        fig.suptitle("Thermal Population vs Time (Q1 & Q5)", fontsize=16)

        # Save
        paramvstime_dir = os.path.join(out_dir, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(paramvstime_dir, f"Pe_vsTime_comb_methods_singleplot_2qubits_{stamp}.png")
        fig.savefig(fname, dpi=self.figure_quality)
        plt.close(fig)
        print("Saved P_e plot →", fname)

    def Pe_vs_time_comb_allQs_1col(
            self,
            ssf_fit_results,
            out_dir,
            all_files_Qtemp_results_RPMs,
            restrict_time_xaxis=False,
            restrict_time_yaxis=False,
            ylims=None,
            rad_events_plot_lines=False,
            qubits_to_plot=None,
            sort_by_time=True,
    ):
        """
        Plot P_e vs time for RPM and SSF.

        SSF input format (your fit_results):
          ssf_fit_results[qid] = list of dicts with keys including:
            - "timestamp" (datetime)
            - "Pe" (float)
            - "total_sigma_Pe" (float)   # optional but recommended

        RPM input format (your existing):
          all_files_Qtemp_results_RPMs: list of rec dicts where
            rec["qubits"][q]["P_e"], ["P_e_err_total"], ["date"] exist.

        Saves a PNG and returns its path.
        """
        os.makedirs(out_dir, exist_ok=True)

        num_qubits = self.number_of_qubits

        # -------------------- Build RPM dicts --------------------
        times_RPM = {q: [] for q in range(num_qubits)}
        Pe_RPM = {q: [] for q in range(num_qubits)}
        PeErr_RPM = {q: [] for q in range(num_qubits)}

        for rec in all_files_Qtemp_results_RPMs:
            qubits_dict = rec.get("qubits", {})
            if not isinstance(qubits_dict, dict):
                continue

            for q in range(num_qubits):
                d = qubits_dict.get(q)
                if not d:
                    continue

                pe = d.get("P_e", None)
                ts = d.get("date", None)  # epoch seconds
                if pe is None or ts is None:
                    continue

                try:
                    pe = float(pe)
                    ts = float(ts)
                except Exception:
                    continue

                if not np.isfinite(pe) or not np.isfinite(ts):
                    continue

                t = datetime.datetime.fromtimestamp(ts)
                times_RPM[q].append(t)
                Pe_RPM[q].append(pe)

                pe_err = d.get("P_e_err_total", None)
                try:
                    pe_err = float(pe_err) if pe_err is not None else np.nan
                except Exception:
                    pe_err = np.nan
                PeErr_RPM[q].append(pe_err)

        # -------------------- Build SSF dicts from fit_results --------------------
        times_SSF = {q: [] for q in range(num_qubits)}
        Pe_SSF = {q: [] for q in range(num_qubits)}
        PeErr_SSF = {q: [] for q in range(num_qubits)}

        if ssf_fit_results is None or not isinstance(ssf_fit_results, dict):
            raise ValueError("ssf_fit_results must be a dict like fit_results[qid] = [ {...}, ... ]")

        for q in range(num_qubits):
            entries = ssf_fit_results.get(q, [])
            if not entries:
                continue

            for r in entries:
                if not isinstance(r, dict):
                    continue

                t = r.get("timestamp", None)
                pe = r.get("Pe", None)  # note: your key is "Pe" not "P_e"
                pe_err = r.get("total_sigma_Pe", None)

                if t is None or pe is None:
                    continue
                if not isinstance(t, datetime.datetime):
                    # in case something serialized weirdly
                    continue

                try:
                    pe = float(pe)
                except Exception:
                    continue
                if not np.isfinite(pe):
                    continue

                if pe_err is None:
                    pe_err = np.nan
                else:
                    try:
                        pe_err = float(pe_err)
                    except Exception:
                        pe_err = np.nan

                times_SSF[q].append(t)
                Pe_SSF[q].append(pe)
                PeErr_SSF[q].append(pe_err)

        # -------------------- Optional: sort each series by time --------------------
        def _sort_series(tlist, ylist, elist):
            if not tlist or not ylist:
                return tlist, ylist, elist
            order = np.argsort([tt.timestamp() for tt in tlist])
            t_sorted = [tlist[i] for i in order]
            y_sorted = [ylist[i] for i in order]
            e_sorted = [elist[i] for i in order] if elist is not None and len(elist) == len(ylist) else elist
            return t_sorted, y_sorted, e_sorted

        if sort_by_time:
            for q in range(num_qubits):
                times_RPM[q], Pe_RPM[q], PeErr_RPM[q] = _sort_series(times_RPM[q], Pe_RPM[q], PeErr_RPM[q])
                times_SSF[q], Pe_SSF[q], PeErr_SSF[q] = _sort_series(times_SSF[q], Pe_SSF[q], PeErr_SSF[q])

        # -------------------- Optional time window --------------------
        if restrict_time_xaxis:
            window_start = datetime.datetime(2025, 4, 18, 0, 0)
            window_end = datetime.datetime(2025, 5, 4, 23, 59)

        # -------------------- Optional radiation events --------------------
        rad_events = []
        if rad_events_plot_lines:
            rad_events = [
                (datetime.datetime(2025, 4, 21, 12, 35), "Co-60"),
                (datetime.datetime(2025, 4, 23, 12, 53), "Cs-137"),
                (datetime.datetime(2025, 4, 28, 9, 40), "Cs-137 closer"),
                (datetime.datetime(2025, 5, 4, 18, 20), "Cs-137 removed"),
            ]

        # -------------------- Decide which qubits to plot --------------------
        if qubits_to_plot is None:
            qubits_to_plot = list(range(num_qubits))
        else:
            qubits_to_plot = sorted(
                q for q in qubits_to_plot
                if isinstance(q, int) and 0 <= q < num_qubits
            )
        if not qubits_to_plot:
            raise ValueError("qubits_to_plot is empty after filtering valid indices.")

        # -------------------- Plot --------------------
        nrows = len(qubits_to_plot)
        fig, axes = plt.subplots(
            nrows, 1,
            figsize=(12, 3.2 * nrows),
            sharex=True,
            constrained_layout=True
        )
        if nrows == 1:
            axes = [axes]

        date_fmt = DateFormatter("%m-%d-%H")

        for ax, q in zip(axes, qubits_to_plot):
            # RPM
            if times_RPM[q] and Pe_RPM[q]:
                use_yerr = len(PeErr_RPM[q]) == len(Pe_RPM[q])
                ax.errorbar(
                    times_RPM[q], Pe_RPM[q],
                    yerr=PeErr_RPM[q] if use_yerr else None,
                    fmt="o", markersize=4, elinewidth=1, capsize=3,
                    alpha=0.85, color="orange", ecolor="orange",
                    markeredgecolor="k", label="RPM $P_e$"
                )

            # SSF
            if times_SSF[q] and Pe_SSF[q]:
                use_yerr = len(PeErr_SSF[q]) == len(Pe_SSF[q])
                ax.errorbar(
                    times_SSF[q], Pe_SSF[q],
                    yerr=PeErr_SSF[q] if use_yerr else None,
                    fmt="o", markersize=4, elinewidth=1, capsize=3,
                    alpha=0.85, color="blue", ecolor="blue",
                    markeredgecolor="k", label="SSF $P_e$"
                )

            ax.set_title(f"Q{q + 1}", loc="left", fontsize=13, fontweight="bold")
            ax.set_ylabel("$P_e$")
            ax.grid(False)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
            ax.xaxis.set_major_formatter(date_fmt)
            ax.tick_params(axis="x", rotation=45, labelsize=9)

            # y limits
            if restrict_time_yaxis and ylims is not None and len(ylims) == 2:
                ax.set_ylim(float(ylims[0]), float(ylims[1]))
            else:
                ax.set_ylim(-0.02, 1.02)

            # x limits
            if restrict_time_xaxis:
                ax.set_xlim(window_start, window_end)

            for t_evt, lbl in rad_events:
                ax.axvline(t_evt, color="gray", linestyle="--", linewidth=1)
                ax.text(t_evt, ax.get_ylim()[1] * 0.9, lbl, rotation=90, va="top", ha="right", fontsize=8)

            # mean difference line in legend
            mean_diff_str = None
            if Pe_RPM[q] and Pe_SSF[q]:
                mean_diff = np.nanmean(Pe_RPM[q]) - np.nanmean(Pe_SSF[q])
                mean_diff_str = f"<RPM> - <SSF> = {mean_diff:.4f}"

            handles, labels = ax.get_legend_handles_labels()
            if mean_diff_str is not None:
                handles.append(plt.Line2D([], [], color="none"))
                labels.append(mean_diff_str)
            ax.legend(handles, labels, loc="upper left", fontsize=9, frameon=False)

        axes[-1].set_xlabel("Time")
        fig.suptitle("Excited-State Population $P_e$ vs Time (RPM vs SSF)", fontsize=15)

        # -------------------- Save --------------------
        paramvstime_dir = os.path.join(out_dir, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(paramvstime_dir, f"Pe_Comparison_RPM_vs_SSF_AllQs_{stamp}.png")
        fig.savefig(out_path, dpi=self.figure_quality)
        plt.close(fig)
        print("Saved Pe comparison plot:", out_path)
        return out_path

    def get_time_alphas(self, times, alpha_min=0.20, alpha_max=0.95, gamma=1.25):
        """
        Convert timestamps into smoothly varying alpha values.

        Earlier times get alpha_min.
        Later times get alpha_max.

        gamma controls how the gradient is stretched:
            gamma = 1.0  -> linear gradient
            gamma < 1.0  -> spreads out early-time differences more
            gamma > 1.0  -> spreads out late-time differences more
        """

        if times is None or len(times) == 0:
            return np.array([])

        t_seconds = np.array([t.timestamp() for t in times], dtype=float)

        if len(t_seconds) == 1 or np.nanmax(t_seconds) == np.nanmin(t_seconds):
            return np.full(len(t_seconds), alpha_max)

        # Normalize time from 0 to 1
        t_norm = (t_seconds - np.nanmin(t_seconds)) / (
                np.nanmax(t_seconds) - np.nanmin(t_seconds)
        )

        # Optional nonlinear stretch
        t_norm = t_norm ** gamma

        alphas = alpha_min + t_norm * (alpha_max - alpha_min)

        return alphas

    def add_alpha_gradient_bars(
            self,
            fig,
            colors,
            labels,
            alpha_min=0.20,
            alpha_max=0.95,
            box_pos=(0.75, 0.15, 0.12, 0.20),
            title="Time\nEarlier $\\rightarrow$ Later"
    ):
        """
        Add small alpha-gradient bars, one per qubit color.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to add the inset legend to.

        colors : list
            List of colors, one per qubit.

        labels : list
            List of labels, e.g. ["Q1", "Q2", ...].

        box_pos : tuple
            Position of inset axes in figure coordinates:
            (left, bottom, width, height).
        """

        import matplotlib.colors as mcolors

        ax_grad = fig.add_axes(box_pos)
        # ax_grad.set_title(title, fontsize=9)

        n = len(colors)
        n_alpha = 100

        for i, (color, label) in enumerate(zip(colors, labels)):
            rgb = mcolors.to_rgb(color)

            grad = np.ones((1, n_alpha, 4))
            grad[:, :, 0] = rgb[0]
            grad[:, :, 1] = rgb[1]
            grad[:, :, 2] = rgb[2]
            grad[:, :, 3] = np.linspace(alpha_min, alpha_max, n_alpha)

            y0 = n - i - 1

            ax_grad.imshow(
                grad,
                extent=[0, 1, y0, y0 + 0.6],
                aspect="auto"
            )

            ax_grad.text(
                -0.08,
                y0 + 0.3,
                label,
                ha="right",
                va="center",
                fontsize=14
            )

        ax_grad.set_xlim(-0.25, 1.0)
        ax_grad.set_ylim(0, n)
        ax_grad.set_xticks([0, 1])
        ax_grad.set_xticklabels(["early", "late"], fontsize=14)
        ax_grad.set_yticks([])

        for spine in ax_grad.spines.values():
            spine.set_visible(False)

    def get_time_bins(self, times):
        """
        Split timestamps into early/middle/late bins.

        Returns
        -------
        time_bins : np.ndarray of str
            Values are "early", "middle", or "late".
        """

        if times is None or len(times) == 0:
            return np.array([])

        time_seconds = np.array([t.timestamp() for t in times], dtype=float)

        if len(time_seconds) < 3:
            return np.array(["middle"] * len(time_seconds))

        q1, q2 = np.percentile(time_seconds, [33.33, 66.67])

        time_bins = []

        for t in time_seconds:
            if t <= q1:
                time_bins.append("early")
            elif t <= q2:
                time_bins.append("middle")
            else:
                time_bins.append("late")

        return np.array(time_bins)

    def add_left_duplicate_z_axis_projected_auto(self, ax):
        """
        Duplicate the 3D z-axis on the left side, tilted with the 3D plot.

        This automatically finds the leftmost projected vertical edge of the 3D box
        and draws a duplicate z-axis there using the same z ticks and labels as
        the real z-axis.
        """

        fig = ax.figure
        fig.canvas.draw()

        def project_point(x, y, z):
            # Project 3D data point to 2D axes coordinates
            x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
            x_display, y_display = ax.transData.transform((x2, y2))
            x_axes, y_axes = ax.transAxes.inverted().transform((x_display, y_display))
            return np.array([x_axes, y_axes])

        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()

        x_candidates = [xlim[0], xlim[1]]
        y_candidates = [ylim[0], ylim[1]]

        z0, z1 = zlim[0], zlim[1]

        # Build the four possible vertical z-edges of the 3D box
        edges = []
        for x in x_candidates:
            for y in y_candidates:
                p0 = project_point(x, y, z0)
                p1 = project_point(x, y, z1)
                x_mean = 0.5 * (p0[0] + p1[0])
                edges.append((x_mean, x, y, p0, p1))

        # Pick the leftmost projected vertical edge
        edges = sorted(edges, key=lambda item: item[0])
        _, x_edge, y_edge, p0, p1 = edges[0]

        # Copy real z ticks/labels
        zticks = ax.get_zticks()
        zticklabels = [lab.get_text() for lab in ax.get_zticklabels()]
        zlabel = ax.get_zlabel()

        zlo, zhi = min(zlim), max(zlim)

        # Direction of the tilted z-edge in axes coordinates
        v = p1 - p0
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return

        v = v / v_norm

        # Perpendicular direction for ticks/labels.
        # Force it to point left.
        n = np.array([-v[1], v[0]])
        if n[0] > 0:
            n = -n

        tick_len = 0.018
        tick_label_pad = 0.040
        axis_label_pad = 0.120

        # Draw the duplicated tilted z-axis line
        ax.add_line(Line2D(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            transform=ax.transAxes,
            color="black",
            linewidth=1.0,
            clip_on=False
        ))

        # Draw ticks and labels
        for z, tick_label in zip(zticks, zticklabels):
            if z < zlo or z > zhi:
                continue

            p = project_point(x_edge, y_edge, z)

            tick_end = p + tick_len * n
            label_pos = p + tick_label_pad * n

            ax.add_line(Line2D(
                [p[0], tick_end[0]],
                [p[1], tick_end[1]],
                transform=ax.transAxes,
                color="black",
                linewidth=1.0,
                clip_on=False
            ))

            ax.text2D(
                label_pos[0],
                label_pos[1],
                tick_label,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=ax.zaxis.get_ticklabels()[0].get_size()
                if len(ax.zaxis.get_ticklabels()) > 0 else 10,
                color="black"
            )

        # Axis label tilted along the copied z-axis
        mid = 0.5 * (p0 + p1)
        label_pos = mid + axis_label_pad * n

        angle = np.degrees(np.arctan2(v[1], v[0]))

        ax.text2D(
            label_pos[0],
            label_pos[1],
            zlabel,
            transform=ax.transAxes,
            rotation=angle,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=ax.zaxis.label.get_size(),
            color=ax.zaxis.label.get_color()
        )

    def SSF_fid_vs_RRPM_Pe_3D(
            self,
            ssf_fit_results,
            all_files_Qtemp_results_RPMs,
            out_dir,
            qubits_to_plot=None,
            colors=['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred'],
            tolerance_seconds=10,
            sort_by_time=True,
            xlims=None,
            ylims=None,
            zlims=None,
            RPM_Pe_rel_err_cut=None,
            axis_order="time_pe_ssf",
            elev=25,
            azim=-60,
            plot_qubits_separately = False,
            show_bottom_shadow=True):
        """
        Make a 3D plot showing how SSF vs RPM Pe evolves over time.

        axis_order options
        ------------------
        "time_pe_ssf":
            x = time since start [hours]
            y = RPM Pe
            z = SSF fidelity

        "pe_time_ssf":
            x = RPM Pe
            y = time since start [hours]
            z = SSF fidelity

        "pe_ssf_time":
            x = RPM Pe
            y = SSF fidelity
            z = time since start [hours]
        """

        from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection

        os.makedirs(out_dir, exist_ok=True)

        num_qubits = self.number_of_qubits
        markers = ['o', 's', '^', 'D', 'v', 'P']

        allowed_axis_orders = ["time_pe_ssf", "pe_time_ssf", "pe_ssf_time"]
        if axis_order not in allowed_axis_orders:
            raise ValueError(f"axis_order must be one of {allowed_axis_orders}")

        if ssf_fit_results is None or not isinstance(ssf_fit_results, dict):
            raise ValueError("ssf_fit_results must be a dict like fit_results[qid] = [ {...}, ... ]")

        # -------------------- Decide which qubits to plot --------------------
        if qubits_to_plot is None:
            qubits_to_plot = list(range(num_qubits))
        else:
            qubits_to_plot = sorted(
                q for q in qubits_to_plot
                if isinstance(q, int) and 0 <= q < num_qubits
            )

        if not qubits_to_plot:
            raise ValueError("qubits_to_plot is empty after filtering valid indices.")

        # ================================================================
        # 1. Build RPM dictionaries
        # ================================================================
        times_RPM = {q: [] for q in range(num_qubits)}
        Pe_RPM = {q: [] for q in range(num_qubits)}
        PeErr_RPM = {q: [] for q in range(num_qubits)}

        for rec in all_files_Qtemp_results_RPMs:
            if not isinstance(rec, dict):
                continue

            qubits_dict = rec.get("qubits", {})
            if not isinstance(qubits_dict, dict):
                continue

            for q in range(num_qubits):
                d = qubits_dict.get(q)
                if not d:
                    continue

                pe = d.get("P_e", None)
                pe_err = d.get("P_e_err_total", None)
                ts = d.get("date", None)

                if pe is None or ts is None:
                    continue

                try:
                    pe = float(pe)
                    ts = float(ts)
                except Exception:
                    continue

                if not np.isfinite(pe) or not np.isfinite(ts):
                    continue

                try:
                    pe_err = float(pe_err) if pe_err is not None else np.nan
                except Exception:
                    pe_err = np.nan

                if RPM_Pe_rel_err_cut is not None:
                    if pe <= 0 or not np.isfinite(pe_err):
                        continue
                    if pe_err / pe >= RPM_Pe_rel_err_cut:
                        continue

                times_RPM[q].append(datetime.datetime.fromtimestamp(ts))
                Pe_RPM[q].append(pe)
                PeErr_RPM[q].append(pe_err)

        # ================================================================
        # 2. Build SSF dictionaries
        # ================================================================
        times_SSF = {q: [] for q in range(num_qubits)}
        SSF_vals = {q: [] for q in range(num_qubits)}
        SSF_errs = {q: [] for q in range(num_qubits)}

        for q in range(num_qubits):
            entries = ssf_fit_results.get(q, []) or []

            for r in entries:
                if not isinstance(r, dict):
                    continue

                t = r.get("timestamp", None)
                ssf = r.get("ssf_fid", None)
                ssf_err = r.get("ssf_err_total", None)

                if t is None or ssf is None:
                    continue

                if not isinstance(t, datetime.datetime):
                    continue

                try:
                    ssf = float(ssf)
                except Exception:
                    continue

                if not np.isfinite(ssf):
                    continue

                try:
                    ssf_err = float(ssf_err) if ssf_err is not None else np.nan
                except Exception:
                    ssf_err = np.nan

                times_SSF[q].append(t)
                SSF_vals[q].append(ssf)
                SSF_errs[q].append(ssf_err)

        # ================================================================
        # 3. Optional sorting
        # ================================================================
        def _sort_series(tlist, ylist, elist):
            if not tlist or not ylist:
                return tlist, ylist, elist

            order = np.argsort([tt.timestamp() for tt in tlist])
            t_sorted = [tlist[i] for i in order]
            y_sorted = [ylist[i] for i in order]
            e_sorted = [elist[i] for i in order] if elist is not None and len(elist) == len(ylist) else elist

            return t_sorted, y_sorted, e_sorted

        if sort_by_time:
            for q in range(num_qubits):
                times_RPM[q], Pe_RPM[q], PeErr_RPM[q] = _sort_series(
                    times_RPM[q], Pe_RPM[q], PeErr_RPM[q]
                )
                times_SSF[q], SSF_vals[q], SSF_errs[q] = _sort_series(
                    times_SSF[q], SSF_vals[q], SSF_errs[q]
                )

        # ================================================================
        # 4. Match SSF to nearest RPM by timestamp
        # ================================================================
        matched = {
            q: {
                "Pe_RPM": [],
                "PeErr_RPM": [],
                "SSF": [],
                "SSF_err": [],
                "dt_seconds": [],
                "t_SSF": [],
                "t_RPM": [],
            }
            for q in range(num_qubits)
        }

        for q in qubits_to_plot:
            if len(times_RPM[q]) == 0 or len(times_SSF[q]) == 0:
                print(f"Q{q + 1}: missing RPM or SSF data, skipping.")
                continue

            rpm_ts = np.array([t.timestamp() for t in times_RPM[q]])

            for t_ssf, ssf, ssf_err in zip(times_SSF[q], SSF_vals[q], SSF_errs[q]):
                ssf_ts = t_ssf.timestamp()

                dt = np.abs(rpm_ts - ssf_ts)
                nearest_idx = int(np.argmin(dt))
                dt_min = float(dt[nearest_idx])

                if dt_min > tolerance_seconds:
                    continue

                pe = Pe_RPM[q][nearest_idx]
                pe_err = PeErr_RPM[q][nearest_idx]
                t_rpm = times_RPM[q][nearest_idx]

                if not np.isfinite(pe) or not np.isfinite(ssf):
                    continue

                if not np.isfinite(pe_err) or not np.isfinite(ssf_err):
                    continue

                matched[q]["Pe_RPM"].append(pe)
                matched[q]["PeErr_RPM"].append(pe_err)
                matched[q]["SSF"].append(ssf)
                matched[q]["SSF_err"].append(ssf_err)
                matched[q]["dt_seconds"].append(dt_min)
                matched[q]["t_SSF"].append(t_ssf)
                matched[q]["t_RPM"].append(t_rpm)

            print(
                f"Q{q + 1}: matched {len(matched[q]['SSF'])} SSF/RPM points "
                f"within {tolerance_seconds}s each."
            )

        # ================================================================
        # 5. Convert time to hours since first matched RPM point
        # ================================================================
        all_times = []
        for q in qubits_to_plot:
            all_times.extend(matched[q]["t_RPM"])

        if len(all_times) == 0:
            print("No matched SSF/RPM points found. No 3D plot made.")
            return None, matched

        t0 = min(all_times)

        # ================================================================
        # 6. Helper for axis ordering
        # ================================================================
        def _axis_values(pe_vals, ssf_vals, t_hours):
            if axis_order == "time_pe_ssf":
                return (
                    t_hours,
                    pe_vals,
                    ssf_vals,
                    "Time since start [hours]",
                    "RPM $P_e$",
                    "Single-Shot Fidelity",
                )

            elif axis_order == "pe_time_ssf":
                return (
                    pe_vals,
                    t_hours,
                    ssf_vals,
                    "RPM $P_e$",
                    "Time since start [hours]",
                    "Single-Shot Fidelity",
                )

            elif axis_order == "pe_ssf_time":
                return (
                    pe_vals,
                    ssf_vals,
                    t_hours,
                    "RPM $P_e$",
                    "Single-Shot Fidelity",
                    "Time since start [hours]",
                )

        # ================================================================
        # 7. Plot
        # ================================================================
        paramvstime_dir = os.path.join(out_dir, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if not plot_qubits_separately: # a single 3D figure, all qubits in it
            # Bigger figure helps a lot
            fig = plt.figure(figsize=(13.5, 10.5))
            ax = fig.add_subplot(111, projection="3d")

            xlabel = None
            ylabel = None
            zlabel = None

            max_ssf_wall_lines = []
            for q in qubits_to_plot:
                if len(matched[q]["SSF"]) == 0:
                    continue

                q_color = colors[q % len(colors)]
                q_marker = markers[q % len(markers)]

                pe_vals = np.array(matched[q]["Pe_RPM"], dtype=float)
                ssf_vals = np.array(matched[q]["SSF"], dtype=float)
                pe_errs = np.array(matched[q]["PeErr_RPM"], dtype=float)
                ssf_errs = np.array(matched[q]["SSF_err"], dtype=float)

                t_hours = np.array([
                    (t - t0).total_seconds() / 3600.0
                    for t in matched[q]["t_RPM"]])

                # Sort by time
                order = np.argsort(t_hours)
                pe_vals = pe_vals[order]
                ssf_vals = ssf_vals[order]
                pe_errs = pe_errs[order]
                ssf_errs = ssf_errs[order]
                t_hours = t_hours[order]

                xvals, yvals, zvals, xlabel, ylabel, zlabel = _axis_values(
                    pe_vals,
                    ssf_vals,
                    t_hours)
                # --------------------------------------------------
                # Optional shadow/projection on the bottom plane
                # --------------------------------------------------
                if show_bottom_shadow:

                    # Choose the bottom of the visible z-axis
                    if zlims is not None:
                        z_shadow = zlims[0]
                    else:
                        z_shadow = np.nanmin(zvals)

                    ax.scatter(
                        xvals,
                        yvals,
                        np.full_like(zvals, z_shadow),
                        marker=q_marker,
                        s=28,
                        color=q_color,
                        alpha=0.15,
                        edgecolor="none",
                        zorder=0
                    )

                # Put the Pe and SSF errors on the correct 3D axes
                if axis_order == "time_pe_ssf":
                    xerr_3d = None  # x = time
                    yerr_3d = pe_errs  # y = RPM Pe
                    zerr_3d = ssf_errs  # z = SSF

                elif axis_order == "pe_time_ssf":
                    xerr_3d = pe_errs  # x = RPM Pe
                    yerr_3d = None  # y = time
                    zerr_3d = ssf_errs  # z = SSF

                elif axis_order == "pe_ssf_time":
                    xerr_3d = pe_errs  # x = RPM Pe
                    yerr_3d = ssf_errs  # y = SSF
                    zerr_3d = None  # z = time

                # Plot error bars first so markers sit on top
                ax.errorbar(
                    xvals,
                    yvals,
                    zvals,
                    xerr=xerr_3d,
                    yerr=yerr_3d,
                    zerr=zerr_3d,
                    fmt="none",
                    ecolor=q_color,
                    elinewidth=1.2,
                    capsize=3,
                    alpha=0.5,
                    zorder=1)

                ax.scatter(
                    xvals,
                    yvals,
                    zvals,
                    marker=q_marker,
                    s=35,
                    color=q_color,
                    edgecolor="k",
                    linewidth=0.5,
                    alpha=0.85,
                    label=f"Q{q + 1}",
                    zorder=3)

                imax_ssf = int(np.nanargmax(ssf_vals))
                max_ssf_wall_lines.append({
                    "q": q,
                    "color": q_color,
                    "z_max": zvals[imax_ssf],
                    "ssf_max": ssf_vals[imax_ssf]})

            # Bigger labelpad
            ax.set_xlabel(xlabel, fontsize=14, labelpad=30)
            ax.set_ylabel(ylabel, fontsize=14, labelpad=30)
            ax.set_zlabel(zlabel, fontsize=14, labelpad=30)

            # Bigger tick label padding too
            ax.tick_params(axis='x', pad=10, labelsize=11)
            ax.tick_params(axis='y', pad=10, labelsize=11)
            ax.tick_params(axis='z', pad=10, labelsize=11)

            ax.set_title(
                f"SSF Fidelity vs RPM $P_e$ vs Time\n"
                f"(nearest-time match, tolerance={tolerance_seconds}s)",
                fontsize=16,
                pad=24)

            if xlims is not None:
                ax.set_xlim(*xlims)

            if ylims is not None:
                ax.set_ylim(*ylims)

            if zlims is not None:
                ax.set_zlim(*zlims)

                ztick_vals = np.round(np.arange(zlims[0], zlims[1] + 0.0001, 0.05), 2)
                ax.set_zticks(ztick_vals)
                ax.set_zticklabels([f"{z:.2f}" for z in ztick_vals])

            ax.view_init(elev=elev, azim=azim)

            self.add_left_duplicate_z_axis_projected_auto(ax)

            # --------------------------------------------------
            # Draw max-SSF wall guide lines AFTER final limits
            # --------------------------------------------------
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            y_wall = y1

            for item in max_ssf_wall_lines:
                ax.plot(
                    [x0, x1],
                    [y_wall, y_wall],
                    [item["z_max"], item["z_max"]],
                    color=item["color"],
                    linewidth=2.5,
                    linestyle="None",
                    alpha=0.60,
                    zorder=100)

                print(
                    f"Q{item['q'] + 1}: max SSF wall line at "
                    f"SSF={item['ssf_max']:.4f}, z={item['z_max']:.4f}, "
                    f"Pe wall={y_wall:.4f}")

            ax.legend(fontsize=10, frameon=True)
            out_path = os.path.join(paramvstime_dir,f"SSF_fid_vs_RPM_Pe_vs_Time_3D_{axis_order}_{stamp}.pdf")

            # Much more generous margins
            fig.subplots_adjust(
                left=0.15,
                right=0.92,
                bottom=0.10,
                top=0.88)

            fig.savefig(out_path,dpi=self.figure_quality,bbox_inches="tight", pad_inches=0.65)
        else:
            # ------------------------------------------------
            # Separate qubit plots in a 2 x 3 grid
            # ------------------------------------------------
            nrows = 2
            ncols = 3

            fig = plt.figure(figsize=(18, 11))
            axes = []
            for i in range(nrows * ncols):
                ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
                axes.append(ax)

            for ax, q in zip(axes, qubits_to_plot):

                if len(matched[q]["SSF"]) > 0:

                    q_color = colors[q % len(colors)]
                    q_marker = markers[q % len(markers)]

                    pe_vals = np.array(matched[q]["Pe_RPM"], dtype=float)
                    ssf_vals = np.array(matched[q]["SSF"], dtype=float)
                    pe_errs = np.array(matched[q]["PeErr_RPM"], dtype=float)
                    ssf_errs = np.array(matched[q]["SSF_err"], dtype=float)

                    t_hours = np.array([(t - t0).total_seconds() / 3600.0 for t in matched[q]["t_RPM"]])

                    order = np.argsort(t_hours)
                    pe_vals = pe_vals[order]
                    ssf_vals = ssf_vals[order]
                    pe_errs = pe_errs[order]
                    ssf_errs = ssf_errs[order]
                    t_hours = t_hours[order]

                    xvals, yvals, zvals, xlabel, ylabel, zlabel = _axis_values(pe_vals, ssf_vals, t_hours)

                    # --------------------------------------------------
                    # Optional shadow/projection on the bottom plane
                    # --------------------------------------------------
                    if show_bottom_shadow:

                        # Choose the bottom of the visible z-axis
                        if zlims is not None:
                            z_shadow = zlims[0]
                        else:
                            z_shadow = np.nanmin(zvals)

                        ax.scatter(
                            xvals,
                            yvals,
                            np.full_like(zvals, z_shadow),
                            marker=q_marker,
                            s=28,
                            color=q_color,
                            alpha=0.15,
                            edgecolor="none",
                            zorder=0
                        )

                    if axis_order == "time_pe_ssf":
                        xerr_3d = None
                        yerr_3d = pe_errs
                        zerr_3d = ssf_errs

                    elif axis_order == "pe_time_ssf":
                        xerr_3d = pe_errs
                        yerr_3d = None
                        zerr_3d = ssf_errs

                    elif axis_order == "pe_ssf_time":
                        xerr_3d = pe_errs
                        yerr_3d = ssf_errs
                        zerr_3d = None

                    ax.errorbar(
                        xvals,
                        yvals,
                        zvals,
                        xerr=xerr_3d,
                        yerr=yerr_3d,
                        zerr=zerr_3d,
                        fmt="none",
                        ecolor=q_color,
                        elinewidth=1.0,
                        capsize=2,
                        alpha=0.5,
                        zorder=1)

                    ax.scatter(
                        xvals,
                        yvals,
                        zvals,
                        marker=q_marker,
                        s=28,
                        color=q_color,
                        edgecolor="k",
                        linewidth=0.4,
                        alpha=0.85,
                        label=f"Q{q + 1}",
                        zorder=3)

                    # Optional max-SSF wall line for each separate qubit
                    imax_ssf = int(np.nanargmax(ssf_vals))
                    z_max = zvals[imax_ssf]
                    ssf_max = ssf_vals[imax_ssf]

                    ax.set_xlabel(xlabel, fontsize=9, labelpad=10)
                    ax.set_ylabel(ylabel, fontsize=9, labelpad=10)
                    ax.set_zlabel(zlabel, fontsize=9, labelpad=10)

                    ax.tick_params(axis='x', pad=3, labelsize=8)
                    ax.tick_params(axis='y', pad=3, labelsize=8)
                    ax.tick_params(axis='z', pad=3, labelsize=8)

                    if xlims is not None:
                        ax.set_xlim(*xlims)

                    if ylims is not None:
                        ax.set_ylim(*ylims)

                    if zlims is not None:
                        ax.set_zlim(*zlims)

                        ztick_vals = np.round(np.arange(zlims[0], zlims[1] + 0.0001, 0.05),2)
                        ax.set_zticks(ztick_vals)
                        ax.set_zticklabels([f"{z:.2f}" for z in ztick_vals])

                    ax.view_init(elev=elev, azim=azim)

                    x0, x1 = ax.get_xlim()
                    y0, y1 = ax.get_ylim()
                    y_wall = y1

                    ax.plot(
                        [x0, x1],
                        [y_wall, y_wall],
                        [z_max, z_max],
                        color=q_color,
                        linewidth=2.0,
                        linestyle="None",
                        alpha=0.60,
                        zorder=100)

                    print(
                        f"Q{q + 1}: max SSF wall line at "
                        f"SSF={ssf_max:.4f}, z={z_max:.4f}, "
                        f"Pe wall={y_wall:.4f}")

                    #ax.legend(loc="best", fontsize=8, frameon=False)

                ax.set_title(
                    f"Q{q + 1}",
                    loc="left",
                    fontsize=13,
                    fontweight="bold")

            for k in range(len(qubits_to_plot), len(axes)):
                axes[k].set_visible(False)

            fig.subplots_adjust(
                left=0.04,
                right=0.96,
                bottom=0.06,
                top=0.90,
                wspace=0.20,
                hspace=0.25)

            fig.suptitle(
                f"SSF Fidelity vs RPM $P_e$ vs Time\n"
                f"(nearest-time match, tolerance={tolerance_seconds}s)",
                fontsize=16)

            out_path = os.path.join(paramvstime_dir, f"SSF_fid_vs_RPM_Pe_vs_Time_3D_Subplots_{axis_order}_{stamp}.pdf")

            fig.savefig(out_path, dpi=self.figure_quality)

        plt.close(fig)
        print("Saved 3D SSF fidelity vs RPM Pe subplot plot:", out_path)
        return matched

    def animate_SSF_fid_vs_RRPM_Pe_2D(
            self,
            matched,
            out_dir,
            qubits_to_plot=None,
            colors=['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred'],
            markers=['o', 's', '^', 'D', 'v', 'P'],
            xlims=None,
            ylims=None,
            tolerance_seconds=10,
            fps=10,
            frame_step=5,
            show_errorbars=True,
            plot_ideal_line=False,
            save_as="mp4"):
        """
        Make a small animation showing SSF vs RPM Pe points appearing over time.

        Input
        -----
        matched : dict
            Dictionary produced by SSF_fid_vs_RRPM_Pe or SSF_fid_vs_RRPM_Pe_3D.

            Expected structure:
                matched[q]["Pe_RPM"]
                matched[q]["PeErr_RPM"]
                matched[q]["SSF"]
                matched[q]["SSF_err"]
                matched[q]["t_RPM"]

        Axes
        ----
        x = RPM Pe
        y = SSF fidelity
        """
        os.makedirs(out_dir, exist_ok=True)

        paramvstime_dir = os.path.join(out_dir, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)

        if qubits_to_plot is None:
            qubits_to_plot = sorted(list(matched.keys()))
        else:
            qubits_to_plot = [
                q for q in qubits_to_plot
                if q in matched
            ]

        if not qubits_to_plot:
            raise ValueError("No valid qubits_to_plot found in matched dictionary.")

        # --------------------------------------------------
        # Collect all matched points into one time-ordered list
        # --------------------------------------------------
        all_points = []

        for q in qubits_to_plot:
            npts = len(matched[q]["SSF"])

            for i in range(npts):
                pe = matched[q]["Pe_RPM"][i]
                ssf = matched[q]["SSF"][i]
                pe_err = matched[q]["PeErr_RPM"][i]
                ssf_err = matched[q]["SSF_err"][i]
                t = matched[q]["t_RPM"][i]

                if t is None:
                    continue

                try:
                    pe = float(pe)
                    ssf = float(ssf)
                    pe_err = float(pe_err)
                    ssf_err = float(ssf_err)
                except Exception:
                    continue

                if not np.isfinite(pe) or not np.isfinite(ssf):
                    continue

                if not np.isfinite(pe_err):
                    pe_err = 0.0

                if not np.isfinite(ssf_err):
                    ssf_err = 0.0

                all_points.append({
                    "q": q,
                    "pe": pe,
                    "ssf": ssf,
                    "pe_err": pe_err,
                    "ssf_err": ssf_err,
                    "time": t,
                })

        if len(all_points) == 0:
            print("No valid matched points found. No animation made.")
            return None

        all_points = sorted(all_points, key=lambda d: d["time"])

        t0 = all_points[0]["time"]
        t_final = all_points[-1]["time"]
        total_hours = (t_final - t0).total_seconds() / 3600.0

        # Frame indices. Use frame_step to keep the movie small.
        frame_indices = list(range(1, len(all_points) + 1, frame_step))

        if frame_indices[-1] != len(all_points):
            frame_indices.append(len(all_points))

        # --------------------------------------------------
        # Auto axis limits if not supplied
        # --------------------------------------------------
        if xlims is None:
            all_pe = np.array([p["pe"] for p in all_points])
            xmin = np.nanmin(all_pe)
            xmax = np.nanmax(all_pe)
            xpad = 0.05 * (xmax - xmin) if xmax > xmin else 0.01
            xlims = (max(0, xmin - xpad), xmax + xpad)

        if ylims is None:
            all_ssf = np.array([p["ssf"] for p in all_points])
            ymin = np.nanmin(all_ssf)
            ymax = np.nanmax(all_ssf)
            ypad = 0.05 * (ymax - ymin) if ymax > ymin else 0.02
            ylims = (ymin - ypad, ymax + ypad)

        # --------------------------------------------------
        # Set up figure
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 8))

        def draw_frame(frame_num):
            ax.clear()

            points_now = all_points[:frame_num]
            current_time = points_now[-1]["time"]

            elapsed_hours = (current_time - t0).total_seconds() / 3600.0
            elapsed_days = elapsed_hours / 24.0
            progress = 100.0 * frame_num / len(all_points)

            # Optional ideal line
            if plot_ideal_line:
                pe_line = np.linspace(max(0, xlims[0]), xlims[1], 300)
                ssf_ideal = 1 - 2 * pe_line

                ax.plot(
                    pe_line,
                    ssf_ideal,
                    color="black",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.9,
                    label=r"Ideal: SSF $= 1 - 2P_e$",
                    zorder=5
                )

            # Plot points by qubit
            for q in qubits_to_plot:
                q_points = [p for p in points_now if p["q"] == q]

                if len(q_points) == 0:
                    continue

                q_color = colors[q % len(colors)]
                q_marker = markers[q % len(markers)]

                pe_vals = np.array([p["pe"] for p in q_points])
                ssf_vals = np.array([p["ssf"] for p in q_points])
                pe_errs = np.array([p["pe_err"] for p in q_points])
                ssf_errs = np.array([p["ssf_err"] for p in q_points])

                # Put Q4 in the back if present
                zorder_val = 1 if q == 3 else 3

                if show_errorbars:
                    ax.errorbar(
                        pe_vals,
                        ssf_vals,
                        xerr=pe_errs,
                        yerr=ssf_errs,
                        fmt="none",
                        ecolor=q_color,
                        elinewidth=0.8,
                        capsize=2,
                        alpha=0.35,
                        zorder=zorder_val
                    )

                ax.scatter(
                    pe_vals,
                    ssf_vals,
                    marker=q_marker,
                    s=35,
                    facecolors=q_color,
                    edgecolors="k",
                    linewidths=0.6,
                    alpha=0.85,
                    label=f"Q{q + 1}",
                    zorder=zorder_val + 1
                )

            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)

            ax.set_xlabel("RPM $P_e$", fontsize=14)
            ax.set_ylabel("Single-Shot Fidelity", fontsize=14)

            ax.set_title(
                f"SSF Fidelity vs RPM $P_e$ over Time\n"
                f"(nearest-time match, tolerance={tolerance_seconds}s)",
                fontsize=15
            )

            # Live-updating time metric text box
            ax.text(
                0.03,
                0.97,
                (
                    f"Current time: {current_time:%m-%d %H:%M:%S}\n"
                    f"Elapsed: {elapsed_hours:.1f} h ({elapsed_days:.2f} d)\n"
                    f"Total span: {total_hours:.1f} h\n"
                    f"Points shown: {frame_num}/{len(all_points)} ({progress:.0f}%)"
                ),
                transform=ax.transAxes,
                fontsize=11,
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="black")
            )

            ax.grid(alpha=0.3)
            ax.legend(fontsize=10, frameon=True, loc="best")

            return ax,

        ani = animation.FuncAnimation(
            fig,
            draw_frame,
            frames=frame_indices,
            interval=1000 / fps,
            blit=False
        )

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if save_as.lower() == "gif":
            out_path = os.path.join(
                paramvstime_dir,
                f"SSF_fid_vs_RPM_Pe_animation_{stamp}.gif"
            )
            writer = animation.PillowWriter(fps=fps)
            ani.save(out_path, writer=writer, dpi=self.figure_quality)

        else:
            out_path = os.path.join(
                paramvstime_dir,
                f"SSF_fid_vs_RPM_Pe_animation_{stamp}.mp4"
            )

            if animation.writers.is_available("ffmpeg"):
                writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
                ani.save(out_path, writer=writer, dpi=self.figure_quality)
            else:
                # Fallback to GIF if ffmpeg is unavailable
                out_path = out_path.replace(".mp4", ".gif")
                writer = animation.PillowWriter(fps=fps)
                ani.save(out_path, writer=writer, dpi=self.figure_quality)

        plt.close(fig)

        print("Saved SSF vs RPM Pe animation:", out_path)
        return out_path

    def mean_datetime(self, dt_list):
        """
        Average a list of datetime objects.
        """
        if not dt_list:
            return None

        ts = [
            t.timestamp()
            for t in dt_list
            if isinstance(t, datetime.datetime)
        ]

        if len(ts) == 0:
            return None

        return datetime.datetime.fromtimestamp(float(np.mean(ts)))

    def nearest_neighbor_average_plot_data(self, q_match, n_neighbors=10, min_neighbors=3):
        """
        Reduce matched SSF/RPM points by averaging nearest neighbors in RPM Pe.

        Important:
        - Neighbors are chosen only by Pe, not by SSF.
        - Points are sorted by Pe and averaged in non-overlapping groups.
        - Each original point is used once.
        - This reduces visual crowding without using the y-axis value to define groups.

        Error bars:
        - Pe error: propagated measurement error on the mean, sqrt(sum(err_i^2)) / N
        - SSF error: propagated measurement error on the mean, sqrt(sum(err_i^2)) / N
        """

        pe = np.array(q_match["Pe_RPM"], dtype=float)
        pe_err = np.array(q_match["PeErr_RPM"], dtype=float)
        ssf = np.array(q_match["SSF"], dtype=float)
        ssf_err = np.array(q_match["SSF_err"], dtype=float)
        dt_seconds = np.array(q_match["dt_seconds"], dtype=float)

        t_ssf = list(q_match["t_SSF"])
        t_rpm = list(q_match["t_RPM"])

        good = (
                np.isfinite(pe)
                & np.isfinite(pe_err)
                & np.isfinite(ssf)
                & np.isfinite(ssf_err)
                & np.isfinite(dt_seconds)
        )

        if np.sum(good) < min_neighbors:
            return q_match

        pe = pe[good]
        pe_err = pe_err[good]
        ssf = ssf[good]
        ssf_err = ssf_err[good]
        dt_seconds = dt_seconds[good]

        good_indices = np.where(good)[0]
        t_ssf = [t_ssf[i] for i in good_indices]
        t_rpm = [t_rpm[i] for i in good_indices]

        # Sort only by Pe so the averaging is independent of SSF.
        order = np.argsort(pe)

        pe = pe[order]
        pe_err = pe_err[order]
        ssf = ssf[order]
        ssf_err = ssf_err[order]
        dt_seconds = dt_seconds[order]
        t_ssf = [t_ssf[i] for i in order]
        t_rpm = [t_rpm[i] for i in order]

        # Build non-overlapping nearest-neighbor groups.
        groups = [
            list(range(i, min(i + n_neighbors, len(pe))))
            for i in range(0, len(pe), n_neighbors)
        ]

        # Avoid a tiny final group by merging it into the previous group.
        if len(groups) > 1 and len(groups[-1]) < min_neighbors:
            groups[-2].extend(groups[-1])
            groups = groups[:-1]

        avg = {
            "Pe_RPM": [],
            "PeErr_RPM": [],
            "SSF": [],
            "SSF_err": [],
            "dt_seconds": [],
            "t_SSF": [],
            "t_RPM": [],
        }

        for g in groups:
            g = np.array(g, dtype=int)
            n = len(g)

            if n < min_neighbors:
                continue

            pe_g = pe[g]
            pe_err_g = pe_err[g]
            ssf_g = ssf[g]
            ssf_err_g = ssf_err[g]
            dt_g = dt_seconds[g]

            # Arithmetic means keep the averaging simple and transparent.
            pe_mean = float(np.mean(pe_g))
            ssf_mean = float(np.mean(ssf_g))
            dt_mean = float(np.mean(dt_g))

            # Propagate measurement uncertainty on the mean.
            # This keeps the averaged x and y error bars consistent:
            # both show the measurement uncertainty of the averaged value.
            pe_total_err = float(np.sqrt(np.sum(pe_err_g ** 2)) / n)
            ssf_total_err = float(np.sqrt(np.sum(ssf_err_g ** 2)) / n)

            avg["Pe_RPM"].append(pe_mean)
            avg["PeErr_RPM"].append(pe_total_err)
            avg["SSF"].append(ssf_mean)
            avg["SSF_err"].append(ssf_total_err)
            avg["dt_seconds"].append(dt_mean)
            avg["t_SSF"].append(self.mean_datetime([t_ssf[i] for i in g]))
            avg["t_RPM"].append(self.mean_datetime([t_rpm[i] for i in g]))

        return avg

    def SSF_fid_vs_RRPM_Pe(
            self,
            ssf_fit_results,
            all_files_Qtemp_results_RPMs,
            out_dir,
            qubits_to_plot=None,
            colors=['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred'],
            tolerance_seconds=10, # 10s for all runs except QUIET run 6 SCIENCE RUN data (600s)
            plot_together=False,
            sort_by_time=True,
            xlims=None,
            ylims=None,
            RPM_Pe_rel_err_cut = None,
            plot_with_t_color_gradient = False,
            plot_with_t_markers = False, # only implemented for plot_together case currently
            plot_ideal_line = False,
            nearest_neighbor_average=False,
            nn_average_neighbors=10,
            nn_average_min_neighbors=3): # This just prevents the last averaged point from being made from only 1 or 2 leftover points.
        """
        Plot SSF fidelity vs RPM-extracted thermal population Pe.

        This function matches SSF points to RPM Pe points by nearest timestamp
        for the same qubit.

        SSF input format
        ----------------
        ssf_fit_results[qid] = list of dicts with keys including:
            - "timestamp"      : datetime.datetime
            - "ssf_fid"        : float
            - "ssf_err_total"  : float

        RPM input format
        ----------------
        all_files_Qtemp_results_RPMs = list of record dicts where:
            rec["qubits"][q]["P_e"]
            rec["qubits"][q]["P_e_err_total"]
            rec["qubits"][q]["date"]    # epoch seconds

        Plot meaning
        ------------
        x    = RPM Pe
        xerr = RPM Pe error
        y    = SSF fidelity
        yerr = SSF fidelity error

        Matching
        --------
        For each SSF point, the nearest RPM Pe point in time is selected.
        The pair is kept only if |t_SSF - t_RPM| <= tolerance_seconds.
        """

        os.makedirs(out_dir, exist_ok=True)

        # Do not allow both time-encoding styles at once.
        # If marker time-bins are requested, turn off the alpha gradient.
        if plot_with_t_markers:
            plot_with_t_color_gradient = False

        num_qubits = self.number_of_qubits
        markers = ['o', 's', '^', 'D', 'v', 'P']

        if ssf_fit_results is None or not isinstance(ssf_fit_results, dict):
            raise ValueError("ssf_fit_results must be a dict like fit_results[qid] = [ {...}, ... ]")

        # -------------------- Decide which qubits to plot --------------------
        if qubits_to_plot is None:
            qubits_to_plot = list(range(num_qubits))
        else:
            qubits_to_plot = sorted(
                q for q in qubits_to_plot
                if isinstance(q, int) and 0 <= q < num_qubits
            )

        if not qubits_to_plot:
            raise ValueError("qubits_to_plot is empty after filtering valid indices.")

        # ================================================================
        # 1. Build RPM dictionaries
        # ================================================================
        times_RPM = {q: [] for q in range(num_qubits)}
        Pe_RPM = {q: [] for q in range(num_qubits)}
        PeErr_RPM = {q: [] for q in range(num_qubits)}

        for rec in all_files_Qtemp_results_RPMs:
            if not isinstance(rec, dict):
                continue

            qubits_dict = rec.get("qubits", {})
            if not isinstance(qubits_dict, dict):
                continue

            for q in range(num_qubits):
                d = qubits_dict.get(q)
                if not d:
                    continue

                pe = d.get("P_e", None)
                pe_err = d.get("P_e_err_total", None)
                ts = d.get("date", None)  # epoch seconds

                if pe is None or ts is None:
                    continue

                try:
                    pe = float(pe)
                    ts = float(ts)
                except Exception:
                    continue

                if not np.isfinite(pe) or not np.isfinite(ts):
                    continue

                try:
                    pe_err = float(pe_err) if pe_err is not None else np.nan
                except Exception:
                    pe_err = np.nan

                if RPM_Pe_rel_err_cut is not None:
                    rel_err = pe_err / pe
                    if rel_err >= RPM_Pe_rel_err_cut: # optional relative err cut
                        print(f"Removed datapoint for Q{q} due to Pe relative err of {rel_err*100:.2f}%")
                        continue

                times_RPM[q].append(datetime.datetime.fromtimestamp(ts))
                Pe_RPM[q].append(pe)
                PeErr_RPM[q].append(pe_err)

        # ================================================================
        # 2. Build SSF dictionaries
        # ================================================================
        times_SSF = {q: [] for q in range(num_qubits)}
        SSF_vals = {q: [] for q in range(num_qubits)}
        SSF_errs = {q: [] for q in range(num_qubits)}

        for q in range(num_qubits):
            entries = ssf_fit_results.get(q, []) or []

            for r in entries:
                if not isinstance(r, dict):
                    continue

                t = r.get("timestamp", None)
                ssf = r.get("ssf_fid", None)
                ssf_err = r.get("ssf_err_total", None)

                if t is None or ssf is None:
                    continue

                if not isinstance(t, datetime.datetime):
                    continue

                try:
                    ssf = float(ssf)
                except Exception:
                    continue

                if not np.isfinite(ssf):
                    continue

                try:
                    ssf_err = float(ssf_err) if ssf_err is not None else np.nan
                except Exception:
                    ssf_err = np.nan

                times_SSF[q].append(t)
                SSF_vals[q].append(ssf)
                SSF_errs[q].append(ssf_err)

        # ================================================================
        # 3. Optional sorting
        # ================================================================
        def _sort_series(tlist, ylist, elist):
            if not tlist or not ylist:
                return tlist, ylist, elist

            order = np.argsort([tt.timestamp() for tt in tlist])
            t_sorted = [tlist[i] for i in order]
            y_sorted = [ylist[i] for i in order]
            e_sorted = [elist[i] for i in order] if elist is not None and len(elist) == len(ylist) else elist

            return t_sorted, y_sorted, e_sorted

        if sort_by_time:
            for q in range(num_qubits):
                times_RPM[q], Pe_RPM[q], PeErr_RPM[q] = _sort_series(times_RPM[q], Pe_RPM[q], PeErr_RPM[q])
                times_SSF[q], SSF_vals[q], SSF_errs[q] = _sort_series(times_SSF[q], SSF_vals[q], SSF_errs[q])

        # ================================================================
        # 4. Match SSF to nearest RPM by timestamp
        # ================================================================
        matched = {
            q: {
                "Pe_RPM": [],
                "PeErr_RPM": [],
                "SSF": [],
                "SSF_err": [],
                "dt_seconds": [],
                "t_SSF": [],
                "t_RPM": [],
            }
            for q in range(num_qubits)
        }

        for q in qubits_to_plot:
            if len(times_RPM[q]) == 0 or len(times_SSF[q]) == 0:
                print(f"Q{q + 1}: missing RPM or SSF data, skipping.")
                continue

            rpm_ts = np.array([t.timestamp() for t in times_RPM[q]])

            for t_ssf, ssf, ssf_err in zip(times_SSF[q], SSF_vals[q], SSF_errs[q]):
                ssf_ts = t_ssf.timestamp()

                dt = np.abs(rpm_ts - ssf_ts)
                nearest_idx = int(np.argmin(dt))
                dt_min = float(dt[nearest_idx])

                if dt_min > tolerance_seconds:
                    continue

                pe = Pe_RPM[q][nearest_idx]
                pe_err = PeErr_RPM[q][nearest_idx]
                t_rpm = times_RPM[q][nearest_idx]

                if not np.isfinite(pe) or not np.isfinite(ssf):
                    continue

                if not np.isfinite(pe_err) or not np.isfinite(ssf_err):
                    continue

                matched[q]["Pe_RPM"].append(pe)
                matched[q]["PeErr_RPM"].append(pe_err)
                matched[q]["SSF"].append(ssf)
                matched[q]["SSF_err"].append(ssf_err)
                matched[q]["dt_seconds"].append(dt_min)
                matched[q]["t_SSF"].append(t_ssf)
                matched[q]["t_RPM"].append(t_rpm)

            print(
                f"Q{q + 1}: matched {len(matched[q]['SSF'])} SSF/RPM points "
                f"within {tolerance_seconds}s each.")

        # ================================================================
        # Print maximum SSF for each qubit
        # ================================================================
        print("\nMaximum matched SSF fidelity per qubit:")

        for q in qubits_to_plot:
            ssf_arr = np.array(matched[q]["SSF"], dtype=float)

            if len(ssf_arr) == 0:
                print(f"Q{q + 1}: no matched points")
                continue

            max_idx = int(np.nanargmax(ssf_arr))

            print(
                f"Q{q + 1}: max SSF = {matched[q]['SSF'][max_idx]:.4f} "
                f"at Pe = {matched[q]['Pe_RPM'][max_idx]:.4f}"
            )
        # ================================================================
        # 5. Plot
        # ================================================================
        paramvstime_dir = os.path.join(out_dir, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if plot_together:
            fig, ax = plt.subplots(figsize=(11, 8))

            for q in qubits_to_plot:
                if len(matched[q]["SSF"]) == 0:
                    continue

                # Keep a copy of the original matched data for this qubit.
                # This lets us temporarily replace matched[q] only for plotting.
                original_matched_q = matched[q]
                if nearest_neighbor_average:
                    avg_data = self.nearest_neighbor_average_plot_data(
                        matched[q],
                        n_neighbors=nn_average_neighbors,
                        min_neighbors=nn_average_min_neighbors)

                    print(
                        f"Q{q + 1}: nearest-neighbor averaged "
                        f"{len(matched[q]['SSF'])} points into {len(avg_data['SSF'])} points "
                        f"using {nn_average_neighbors} neighbors per averaged point.")

                    # Temporarily use the averaged data for the rest of this plotting loop.
                    matched[q] = avg_data

                q_color = colors[q % len(colors)]
                q_marker = markers[q % len(markers)]

                # --------------------------------------------------
                # Time styling
                # Priority:
                #   1. plot_with_t_markers: early/middle/late marker fill
                #   2. plot_with_t_color_gradient: alpha gradient
                #   3. default constant alpha
                # --------------------------------------------------
                if plot_with_t_markers:
                    time_bins = self.get_time_bins(matched[q]["t_RPM"])
                    alphas = np.full(len(matched[q]["SSF"]), 0.8)

                elif plot_with_t_color_gradient:
                    time_bins = None
                    alphas = self.get_time_alphas(
                        matched[q]["t_RPM"],
                        alpha_min=0.35,
                        alpha_max=1.0,
                        gamma=1.6,
                    )

                else:
                    time_bins = None
                    alphas = np.full(len(matched[q]["SSF"]), 0.8)

                # Plot point-by-point so each point can have its own alpha/fill
                for idx, (pe, ssf, pe_err, ssf_err, a) in enumerate(zip(
                        matched[q]["Pe_RPM"],
                        matched[q]["SSF"],
                        matched[q]["PeErr_RPM"],
                        matched[q]["SSF_err"],
                        alphas)):

                    if plot_with_t_markers:
                        tbin = time_bins[idx]

                        if tbin == "early":
                            markerfacecolor = "none"  # open marker
                            alpha = 0.95
                        elif tbin == "middle":
                            markerfacecolor = q_color  # half-transparent filled marker
                            alpha = 0.45
                        else:  # late
                            markerfacecolor = q_color  # filled marker
                            alpha = 0.95

                    else:
                        markerfacecolor = q_color
                        alpha = a

                    # lower zorder = plotted first
                    zorder_val = 100 if q == 0 else 3

                    # Plot error bars first, behind marker
                    ax.errorbar(
                        pe,
                        ssf,
                        xerr=pe_err,
                        yerr=ssf_err,
                        fmt="none",
                        elinewidth=1,
                        capsize=3,
                        alpha=alpha - 0.2,
                        ecolor=q_color,
                        zorder=zorder_val
                    )

                    # Plot marker on top so error bars do not show through it
                    ax.scatter(
                        pe,
                        ssf,
                        marker=q_marker,
                        s=45,
                        facecolors=markerfacecolor,
                        edgecolors="k",
                        linewidths=0.8,
                        alpha=alpha,
                        color=q_color if markerfacecolor != "none" else None,
                        zorder=zorder_val + 1
                    )

                # Dummy point only for qubit legend since we are plotting point-by-point above
                if not plot_with_t_color_gradient:
                    ax.errorbar(
                        [],
                        [],
                        fmt=q_marker,
                        markersize=5,
                        color=q_color,
                        markerfacecolor=q_color,
                        markeredgecolor="k",
                        linestyle="None",
                        label=f"Q{q + 1}"
                    )

                # --------------------------------------------------
                # Optional linear fit: SSF = intercept + slope * Pe
                # This falls under the same flag as the ideal/reference line.
                # --------------------------------------------------
                if plot_ideal_line:
                    # Default robust fit for all qubits.
                    # Q1 has several clear outliers, so we use a stronger Cauchy down-weighting
                    # for Q1 only. This is checked against the default f_scale=3.0 fit.
                    if q == 0:
                        fit_f_scale = 2.0 # has pretty bad outliers in run 9
                    # elif q == 3:
                    #     fit_f_scale = 2.5
                    else:
                        fit_f_scale = 3.0
                    slope, intercept, r2 = self.fit_line_for_qubit(
                        matched[q]["Pe_RPM"],
                        matched[q]["SSF"],
                        pe_errs=matched[q]["PeErr_RPM"],
                        ssf_errs=matched[q]["SSF_err"],
                        loss_method="cauchy",
                        f_scale=fit_f_scale)

                    if slope is not None:
                        x_fit = np.array(matched[q]["Pe_RPM"], dtype=float)
                        x_fit = x_fit[np.isfinite(x_fit)]

                        if len(x_fit) >= 2:
                            pe_fit_line = np.linspace(np.min(x_fit), np.max(x_fit), 200)
                            ssf_fit_line = intercept + slope * pe_fit_line

                            ax.plot(
                                pe_fit_line,
                                ssf_fit_line,
                                color=q_color,
                                linestyle="None",
                                linewidth=2.0,
                                alpha=0.9,
                                label=(
                                    rf"Q{q + 1} fit: "
                                    rf"$m={slope:.2f}$, "
                                    rf"$b={intercept:.3f}$, "
                                    rf"$R^2={r2:.2f}$"),
                                zorder=6)
                # Restore the original full matched data before moving to the next qubit.
                matched[q] = original_matched_q

            if nearest_neighbor_average:
                title_extra = (
                    # f"nearest-time match, tol={tolerance_seconds}s; "
                    f"nearest-neighbor avg, N={nn_average_neighbors}")
            else:
                title_extra = "" #f"nearest-time match, tol={tolerance_seconds}s"

            ax.set_title(
                f"SSF Fidelity vs RPM $P_e$ "
                f"({title_extra})",
                fontsize=16)
            ax.set_xlabel("RPM $P_e$", fontsize=14)
            ax.set_ylabel("Single-Shot Fidelity", fontsize=14)
            ax.tick_params(axis="both", which="major", labelsize=14)

            if xlims is not None:
                ax.set_xlim(*xlims)
            if ylims is not None:
                ax.set_ylim(*ylims)

            if plot_ideal_line:
                cur_xlim = ax.get_xlim()

                pe_line = np.linspace(max(0, cur_xlim[0]), cur_xlim[1], 300)
                ssf_ideal = 1 - 2 * pe_line

                ax.plot(
                    pe_line,
                    ssf_ideal,
                    color="black",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.9,
                    label=r"Ideal: SSF $= 1 - 2P_e$",
                    zorder=5
                )

                ax.set_xlim(cur_xlim)

            ax.grid(alpha=0.3)

            # Main qubit/ideal-line legend
            if plot_ideal_line and not plot_with_t_color_gradient:
                qubit_legend = ax.legend(
                    fontsize=11,
                    frameon=True,
                    loc="best"
                )
                ax.add_artist(qubit_legend)

            # Only show alpha-gradient legend if using continuous alpha gradient
            if plot_with_t_color_gradient:
                qubit_labels = [f"Q{q + 1}" for q in qubits_to_plot]
                qubit_colors = [colors[q % len(colors)] for q in qubits_to_plot]

                self.add_alpha_gradient_bars(
                    fig,
                    qubit_colors,
                    qubit_labels,
                    alpha_min=0.20,
                    alpha_max=0.95,
                    box_pos=(0.75, 0.65, 0.12, 0.22) # box_pos=(left, bottom, width, height)
                )

            # Only show marker-fill legend if using early/middle/late marker bins
            if plot_with_t_markers:
                time_marker_handles = [
                    Line2D(
                        [0], [0],
                        marker="o",
                        color="gray",
                        markerfacecolor="none",
                        markeredgecolor="k",
                        linestyle="None",
                        markersize=7,
                        label="Early"
                    ),
                    Line2D(
                        [0], [0],
                        marker="o",
                        color="gray",
                        markerfacecolor="gray",
                        markeredgecolor="k",
                        linestyle="None",
                        markersize=7,
                        alpha=0.45,
                        label="Middle"
                    ),
                    Line2D(
                        [0], [0],
                        marker="o",
                        color="gray",
                        markerfacecolor="gray",
                        markeredgecolor="k",
                        linestyle="None",
                        markersize=7,
                        alpha=0.95,
                        label="Late"
                    ),
                ]

                time_legend = ax.legend(
                    handles=time_marker_handles,
                    title="Time bin",
                    fontsize=14,
                    title_fontsize=14,
                    frameon=True,
                    loc="lower right")
                ax.add_artist(time_legend)

            out_path = os.path.join(paramvstime_dir,f"SSF_fid_vs_RPM_Pe_AllQs_{stamp}.pdf")
            fig.savefig(out_path, dpi=self.figure_quality)
            plt.close(fig)
            print("Saved SSF fidelity vs RPM Pe plot: ", out_path)
        else:
            nrows = 2
            ncols = 3
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(15, 10),
                sharex=False,
                sharey=True,
                constrained_layout=True)

            axes = np.atleast_1d(axes).ravel()

            for ax, q in zip(axes, qubits_to_plot):

                # Keep a copy of the original matched data for this qubit.
                # This lets us temporarily replace matched[q] only for plotting.
                original_matched_q = matched[q]

                if len(matched[q]["SSF"]) > 0:

                    if nearest_neighbor_average:
                        avg_data = self.nearest_neighbor_average_plot_data(
                            matched[q],
                            n_neighbors=nn_average_neighbors,
                            min_neighbors=nn_average_min_neighbors
                        )

                        print(
                            f"Q{q + 1}: nearest-neighbor averaged "
                            f"{len(matched[q]['SSF'])} points into {len(avg_data['SSF'])} points "
                            f"using {nn_average_neighbors} neighbors per averaged point."
                        )

                        # Temporarily use the averaged data for the rest of this subplot.
                        matched[q] = avg_data

                    if plot_with_t_color_gradient:
                        alphas = self.get_time_alphas(
                            matched[q]["t_RPM"],
                            alpha_min=0.35,
                            alpha_max=1.0,
                            gamma=1.6,
                        )
                    else:
                        alphas = np.full(len(matched[q]["SSF"]), 0.8)

                    for pe, ssf, pe_err, ssf_err, a in zip(
                            matched[q]["Pe_RPM"],
                            matched[q]["SSF"],
                            matched[q]["PeErr_RPM"],
                            matched[q]["SSF_err"],
                            alphas):
                        ax.errorbar(
                            pe,
                            ssf,
                            xerr=pe_err,
                            yerr=ssf_err,
                            fmt=markers[q % len(markers)],
                            markersize=5,
                            elinewidth=1,
                            capsize=3,
                            alpha=a,
                            color=colors[q % len(colors)],
                            ecolor=colors[q % len(colors)],
                            markeredgecolor="k",
                            linestyle="None"
                        )

                    if not plot_with_t_color_gradient:
                        ax.errorbar( # Dummy, just to plot legend (since above we are point-by-point plotting)
                            [],
                            [],
                            fmt=markers[q % len(markers)],
                            markersize=5,
                            color=colors[q % len(colors)],
                            markeredgecolor="k",
                            linestyle="None",
                            label=f"Q{q + 1}"
                        )
                    if plot_ideal_line:
                        # Default robust fit for all qubits.
                        # Q1 has several clear outliers, so we use a stronger Cauchy down-weighting
                        # for Q1 only. This is checked against the default f_scale=3.0 fit.
                        if q == 0:
                            fit_f_scale = 2.0 # has pretty bad outliers
                        # elif q == 3:
                        #     fit_f_scale = 2.5
                        else:
                            fit_f_scale = 3.0
                        slope, intercept, r2 = self.fit_line_for_qubit(
                            matched[q]["Pe_RPM"],
                            matched[q]["SSF"],
                            pe_errs=matched[q]["PeErr_RPM"],
                            ssf_errs=matched[q]["SSF_err"],
                            loss_method="cauchy",
                            f_scale=fit_f_scale)

                        if slope is not None:
                            x_fit = np.array(matched[q]["Pe_RPM"], dtype=float)
                            x_fit = x_fit[np.isfinite(x_fit)]

                            if len(x_fit) >= 2:
                                pe_fit_line = np.linspace(np.min(x_fit), np.max(x_fit), 200)
                                ssf_fit_line = intercept + slope * pe_fit_line

                                ax.plot(
                                    pe_fit_line,
                                    ssf_fit_line,
                                    color="cyan",
                                    linestyle="None",
                                    linewidth=4.0,
                                    alpha=1.0,
                                    zorder = 1000,
                                    label=(
                                        rf"Fit: "
                                        rf"$m={slope:.2f}$, "
                                        rf"$b={intercept:.3f}$, "
                                        rf"$R^2={r2:.2f}$"
                                    )
                                )

                #ax.label_outer()
                ax.set_title(f"Q{q + 1}", loc="left", fontsize=14, fontweight="bold")
                ax.set_ylabel("SSF")
                ax.grid(alpha=0.3)
                ax.tick_params(axis="both", which="major", labelsize=14)
                #ax.legend(loc="best", fontsize=9, frameon=False)
                ax.set_box_aspect(1)

                if xlims is not None:
                    ax.set_xlim(*xlims)
                else: # for a test
                    xlims_per_q = {
                        0: [0.0, 0.035],
                        1: [0.01, 0.045],
                        2: [0.01, 0.065],
                        3: [0.01, 0.05],
                        5: [0.0025, 0.018]}
                    ax.set_xlim(xlims_per_q[q])

                if ylims is not None:
                    ax.set_ylim(*ylims)

                # --------------------------------------------------
                # Ideal reference line: draw after xlims are finalized
                # so every subplot uses the same visible Pe range.
                # --------------------------------------------------
                if plot_ideal_line:
                    cur_xlim = ax.get_xlim()
                    pe_line = np.linspace(max(0, cur_xlim[0]), cur_xlim[1], 300)
                    ssf_ideal = 1 - 2 * pe_line

                    ax.plot(
                        pe_line,
                        ssf_ideal,
                        color="black",
                        linestyle="--",
                        linewidth=2,
                        alpha=0.9,
                        label=r"Ideal: SSF $= 1 - 2P_e$",
                        zorder=5)
                ax.legend(loc="best", fontsize=10, frameon=True)

                # Restore the original full matched data before moving to the next qubit.
                matched[q] = original_matched_q

            for k in range(len(qubits_to_plot), len(axes)):
                axes[k].set_visible(False)

            fig.supxlabel("RPM $P_e$")

            fig.supylabel("SSF Fidelity")

            if nearest_neighbor_average:
                title_extra = f"nearest-neighbor avg, N={nn_average_neighbors}"
            else:
                title_extra = f"nearest-time match, tolerance={tolerance_seconds}s"

            fig.suptitle(
                f"SSF Fidelity vs RPM $P_e$ "
                f"({title_extra})",
                fontsize=16)

            if plot_with_t_color_gradient:
                qubit_labels = [f"Q{q + 1}" for q in qubits_to_plot]
                qubit_colors = [colors[q % len(colors)] for q in qubits_to_plot]

                self.add_alpha_gradient_bars(
                    fig,
                    qubit_colors,
                    qubit_labels,
                    alpha_min=0.35,
                    alpha_max=1.0,
                    box_pos=(0.83, 0.15, 0.12, 0.22) # box_pos=(left, bottom, width, height)
                )

            out_path = os.path.join(paramvstime_dir, f"SSF_fid_vs_RPM_Pe_Subplots_{stamp}.pdf")
            fig.savefig(out_path, dpi=self.figure_quality)
            plt.close(fig)

            print("Saved SSF fidelity vs RPM Pe subplot plot: ", out_path)

    def plot_ssf_SNR_vs_pe(
            self,
            fit_results,
            all_files_Qtemp_results_RPMs,
            plot_path,
            n_qubits=6,
            qubits_to_plot=None,
            colors=None,
            tolerance_seconds=10,
            sort_by_time=True,
            xlims=None,
            ylims=None,
            RPM_Pe_rel_err_cut=None,
            plot_together=True,
            plot_with_t_color_gradient=False,
    ):
        """
        Plot SSF readout SNR vs RPM-extracted thermal population Pe.

        This function matches SSF SNR points to RPM Pe points by nearest timestamp
        for the same qubit.

        SSF/SNR input format
        --------------------
        fit_results[qid] = list of dicts with keys including:
            - "timestamp" : datetime.datetime
            - "ssf_SNR"   : float

        RPM input format
        ----------------
        all_files_Qtemp_results_RPMs = list of record dicts where:
            rec["qubits"][q]["P_e"]
            rec["qubits"][q]["P_e_err_total"]
            rec["qubits"][q]["date"]    # epoch seconds

        Plot meaning
        ------------
        x    = RPM Pe
        xerr = RPM Pe error
        y    = SSF readout SNR

        Matching
        --------
        For each SSF SNR point, the nearest RPM Pe point in time is selected.
        The pair is kept only if |t_SSF - t_RPM| <= tolerance_seconds.
        """

        if colors is None:
            colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']

        markers = ['o', 's', '^', 'D', 'v', 'P']  # Q1-Q6

        os.makedirs(plot_path, exist_ok=True)

        if fit_results is None or not isinstance(fit_results, dict):
            raise ValueError("fit_results must be a dict like fit_results[qid] = [ {...}, ... ]")

        # -------------------- Decide which qubits to plot --------------------
        if qubits_to_plot is None:
            qubits_to_plot = list(range(n_qubits))
        else:
            qubits_to_plot = sorted(
                q for q in qubits_to_plot
                if isinstance(q, int) and 0 <= q < n_qubits
            )

        if not qubits_to_plot:
            raise ValueError("qubits_to_plot is empty after filtering valid indices.")

        # ================================================================
        # 1. Build RPM dictionaries
        # ================================================================
        times_RPM = {q: [] for q in range(n_qubits)}
        Pe_RPM = {q: [] for q in range(n_qubits)}
        PeErr_RPM = {q: [] for q in range(n_qubits)}

        for rec in all_files_Qtemp_results_RPMs:
            if not isinstance(rec, dict):
                continue

            qubits_dict = rec.get("qubits", {})
            if not isinstance(qubits_dict, dict):
                continue

            for q in range(n_qubits):
                d = qubits_dict.get(q)
                if not d:
                    continue

                pe = d.get("P_e", None)
                pe_err = d.get("P_e_err_total", None)
                ts = d.get("date", None)  # epoch seconds

                if pe is None or ts is None:
                    continue

                try:
                    pe = float(pe)
                    ts = float(ts)
                except Exception:
                    continue

                if not np.isfinite(pe) or not np.isfinite(ts):
                    continue

                try:
                    pe_err = float(pe_err) if pe_err is not None else np.nan
                except Exception:
                    pe_err = np.nan

                if RPM_Pe_rel_err_cut is not None:
                    if not np.isfinite(pe_err) or pe <= 0:
                        continue
                    if pe_err / pe >= RPM_Pe_rel_err_cut:
                        continue

                times_RPM[q].append(datetime.datetime.fromtimestamp(ts))
                Pe_RPM[q].append(pe)
                PeErr_RPM[q].append(pe_err)

        # ================================================================
        # 2. Build SSF SNR dictionaries
        # ================================================================
        times_SSF = {q: [] for q in range(n_qubits)}
        SNR_vals = {q: [] for q in range(n_qubits)}

        for q in range(n_qubits):
            entries = fit_results.get(q, []) or []

            for r in entries:
                if not isinstance(r, dict):
                    continue

                t = r.get("timestamp", None)
                snr = r.get("ssf_SNR", None)

                if t is None or snr is None:
                    continue

                if not isinstance(t, datetime.datetime):
                    continue

                try:
                    snr = float(snr)
                except Exception:
                    continue

                if not np.isfinite(snr):
                    continue

                times_SSF[q].append(t)
                SNR_vals[q].append(snr)

        # ================================================================
        # 3. Optional sorting
        # ================================================================
        def _sort_series(tlist, ylist, elist=None):
            if not tlist or not ylist:
                return tlist, ylist, elist

            order = np.argsort([tt.timestamp() for tt in tlist])
            t_sorted = [tlist[i] for i in order]
            y_sorted = [ylist[i] for i in order]

            if elist is not None and len(elist) == len(ylist):
                e_sorted = [elist[i] for i in order]
            else:
                e_sorted = elist

            return t_sorted, y_sorted, e_sorted

        if sort_by_time:
            for q in range(n_qubits):
                times_RPM[q], Pe_RPM[q], PeErr_RPM[q] = _sort_series(
                    times_RPM[q],
                    Pe_RPM[q],
                    PeErr_RPM[q]
                )

                times_SSF[q], SNR_vals[q], _ = _sort_series(
                    times_SSF[q],
                    SNR_vals[q],
                    None
                )

        # ================================================================
        # 4. Match SSF SNR to nearest RPM by timestamp
        # ================================================================
        matched = {
            q: {
                "Pe_RPM": [],
                "PeErr_RPM": [],
                "SNR": [],
                "dt_seconds": [],
                "t_SSF": [],
                "t_RPM": [],
            }
            for q in range(n_qubits)
        }

        for q in qubits_to_plot:
            if len(times_RPM[q]) == 0 or len(times_SSF[q]) == 0:
                print(f"Q{q + 1}: missing RPM or SSF SNR data, skipping.")
                continue

            rpm_ts = np.array([t.timestamp() for t in times_RPM[q]])

            for t_ssf, snr in zip(times_SSF[q], SNR_vals[q]):
                ssf_ts = t_ssf.timestamp()

                dt = np.abs(rpm_ts - ssf_ts)
                nearest_idx = int(np.argmin(dt))
                dt_min = float(dt[nearest_idx])

                if dt_min > tolerance_seconds:
                    continue

                pe = Pe_RPM[q][nearest_idx]
                pe_err = PeErr_RPM[q][nearest_idx]
                t_rpm = times_RPM[q][nearest_idx]

                if not np.isfinite(pe) or not np.isfinite(snr):
                    continue

                if not np.isfinite(pe_err):
                    continue

                matched[q]["Pe_RPM"].append(pe)
                matched[q]["PeErr_RPM"].append(pe_err)
                matched[q]["SNR"].append(snr)
                matched[q]["dt_seconds"].append(dt_min)
                matched[q]["t_SSF"].append(t_ssf)
                matched[q]["t_RPM"].append(t_rpm)

            print(
                f"Q{q + 1}: matched {len(matched[q]['SNR'])} SSF SNR/RPM points "
                f"within {tolerance_seconds}s each."
            )

        # ================================================================
        # 5. Print basic summary
        # ================================================================
        print("\nMaximum matched SSF SNR per qubit:")

        for q in qubits_to_plot:
            snr_arr = np.array(matched[q]["SNR"], dtype=float)

            if len(snr_arr) == 0:
                print(f"Q{q + 1}: no matched points")
                continue

            max_idx = int(np.nanargmax(snr_arr))

            print(
                f"Q{q + 1}: max SNR = {matched[q]['SNR'][max_idx]:.3f} "
                f"at Pe = {matched[q]['Pe_RPM'][max_idx]:.5f}"
            )

        # ================================================================
        # 6. Plot
        # ================================================================
        paramvstime_dir = os.path.join(plot_path, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if plot_together:
            fig, ax = plt.subplots(figsize=(11, 8))

            for q in qubits_to_plot:
                if len(matched[q]["SNR"]) == 0:
                    continue

                q_color = colors[q % len(colors)]
                q_marker = markers[q % len(markers)]

                if plot_with_t_color_gradient:
                    alphas = self.get_time_alphas(
                        matched[q]["t_RPM"],
                        alpha_min=0.20,
                        alpha_max=0.95
                    )
                else:
                    alphas = np.full(len(matched[q]["SNR"]), 0.8)

                for pe, snr, pe_err, a in zip(
                        matched[q]["Pe_RPM"],
                        matched[q]["SNR"],
                        matched[q]["PeErr_RPM"],
                        alphas):
                    # x-error only, because we do not currently have SNR error
                    ax.errorbar(
                        pe,
                        snr,
                        xerr=pe_err,
                        fmt="none",
                        elinewidth=1,
                        capsize=3,
                        alpha=max(a - 0.2, 0.1),
                        ecolor=q_color,
                    )

                    ax.scatter(
                        pe,
                        snr,
                        marker=q_marker,
                        s=50,
                        facecolors=q_color,
                        edgecolors="k",
                        linewidths=0.8,
                        alpha=a,
                    )

                # Dummy handle for legend
                ax.errorbar(
                    [],
                    [],
                    fmt=q_marker,
                    markersize=6,
                    color=q_color,
                    markerfacecolor=q_color,
                    markeredgecolor="k",
                    linestyle="None",
                    label=f"Q{q + 1}"
                )

            ax.set_title(
                f"SSF Readout SNR vs RPM $P_e$ "
                f"(nearest-time match, tolerance={tolerance_seconds}s)",
                fontsize=16
            )
            ax.set_xlabel("RPM $P_e$", fontsize=14)
            ax.set_ylabel("SSF Readout SNR", fontsize=14)

            if xlims is not None:
                ax.set_xlim(*xlims)
            if ylims is not None:
                ax.set_ylim(*ylims)

            ax.grid(alpha=0.3)
            ax.legend(fontsize=11, frameon=True, loc="best")

            if plot_with_t_color_gradient:
                qubit_labels = [f"Q{q + 1}" for q in qubits_to_plot]
                qubit_colors = [colors[q % len(colors)] for q in qubits_to_plot]

                self.add_alpha_gradient_bars(
                    fig,
                    qubit_colors,
                    qubit_labels,
                    alpha_min=0.20,
                    alpha_max=0.95,
                    box_pos=(0.75, 0.15, 0.12, 0.22)
                )

            plt.tight_layout()

            out_path = os.path.join(
                paramvstime_dir,
                f"SSF_SNR_vs_RPM_Pe_AllQs_{stamp}.pdf"
            )

            fig.savefig(out_path, dpi=self.figure_quality, bbox_inches="tight")
            plt.close(fig)

            print("Saved SSF SNR vs RPM Pe plot: ", out_path)

        else:
            nrows = 2
            ncols = 3

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(15, 10),
                sharex=False,
                sharey=True,
                constrained_layout=True
            )

            axes = np.atleast_1d(axes).ravel()

            for ax, q in zip(axes, qubits_to_plot):
                q_color = colors[q % len(colors)]
                q_marker = markers[q % len(markers)]

                if len(matched[q]["SNR"]) > 0:
                    if plot_with_t_color_gradient:
                        alphas = self.get_time_alphas(
                            matched[q]["t_RPM"],
                            alpha_min=0.20,
                            alpha_max=0.95
                        )
                    else:
                        alphas = np.full(len(matched[q]["SNR"]), 0.8)

                    for pe, snr, pe_err, a in zip(
                            matched[q]["Pe_RPM"],
                            matched[q]["SNR"],
                            matched[q]["PeErr_RPM"],
                            alphas):
                        ax.errorbar(
                            pe,
                            snr,
                            xerr=pe_err,
                            fmt=q_marker,
                            markersize=5,
                            elinewidth=1,
                            capsize=3,
                            alpha=a,
                            color=q_color,
                            ecolor=q_color,
                            markeredgecolor="k",
                            linestyle="None"
                        )

                    ax.errorbar(
                        [],
                        [],
                        fmt=q_marker,
                        markersize=5,
                        color=q_color,
                        markeredgecolor="k",
                        linestyle="None",
                        label=f"Q{q + 1}"
                    )

                ax.set_title(f"Q{q + 1}", loc="left", fontsize=14, fontweight="bold")
                ax.set_ylabel("SSF SNR")
                ax.grid(alpha=0.3)
                ax.set_box_aspect(1)

                if xlims is not None:
                    ax.set_xlim(*xlims)

                if ylims is not None:
                    ax.set_ylim(*ylims)

                ax.legend(loc="best", fontsize=10, frameon=True)

            for k in range(len(qubits_to_plot), len(axes)):
                axes[k].set_visible(False)

            fig.supxlabel("RPM $P_e$")
            fig.supylabel("SSF Readout SNR")

            fig.suptitle(
                f"SSF Readout SNR vs RPM $P_e$ "
                f"(nearest-time match, tolerance={tolerance_seconds}s)",
                fontsize=16
            )

            if plot_with_t_color_gradient:
                qubit_labels = [f"Q{q + 1}" for q in qubits_to_plot]
                qubit_colors = [colors[q % len(colors)] for q in qubits_to_plot]

                self.add_alpha_gradient_bars(
                    fig,
                    qubit_colors,
                    qubit_labels,
                    alpha_min=0.20,
                    alpha_max=0.95,
                    box_pos=(0.83, 0.15, 0.12, 0.22)
                )

            out_path = os.path.join(
                paramvstime_dir,
                f"SSF_SNR_vs_RPM_Pe_Subplots_{stamp}.pdf"
            )

            fig.savefig(out_path, dpi=self.figure_quality, bbox_inches="tight")
            plt.close(fig)

            print("Saved SSF SNR vs RPM Pe subplot plot: ", out_path)

    def fit_line_for_qubit(self, pe_vals, ssf_vals, pe_errs=None, ssf_errs=None, loss_method = "cauchy", f_scale = 3.0):
        """
        Fits SSF = intercept + slope * Pe.

        Uses all finite points. If both Pe and SSF uncertainties are provided,
        Pe uncertainty is propagated into an effective SSF uncertainty using:

            sigma_eff^2 = sigma_SSF^2 + (slope * sigma_Pe)^2

        Then a robust weighted least-squares fit is used, making the result less
        sensitive to outliers/readout-parameter artifacts.

        Returns:
            slope, intercept, r2

        Additional context on loss methods:
        f_scale is the cutoff scale for the residuals in the robust fit. The residuals are r = (SSF_data - SSF_fit) / sigma_eff.
        If you choose, say, f_scale = 3, it means that points within about 3 effective sigma of the fit are treated fairly normally.
        Points farther away than that start getting strongly down-weighted by the robust loss.

        smaller f_scale  -> more aggressive outlier rejection
        larger f_scale   -> less aggressive, closer to ordinary least squares

        linear   = ordinary weighted least squares
        soft_l1  = gentle robust fitting
        huber    = moderate robust fitting with a clearer cutoff
        cauchy   = strong outlier down-weighting
        arctan   = very strong outlier down-weighting
        """
        valid_losses = ["linear", "soft_l1", "huber", "cauchy", "arctan"]
        if loss_method not in valid_losses:
            raise ValueError(
                f"loss_method must be one of {valid_losses}, got {loss_method}")

        x = np.array(pe_vals, dtype=float)
        y = np.array(ssf_vals, dtype=float)

        good = np.isfinite(x) & np.isfinite(y)

        if pe_errs is not None:
            xerr = np.array(pe_errs, dtype=float)
            good &= np.isfinite(xerr) & (xerr >= 0)
        else:
            xerr = None

        if ssf_errs is not None:
            yerr = np.array(ssf_errs, dtype=float)
            good &= np.isfinite(yerr) & (yerr > 0)
        else:
            yerr = None

        x = x[good]
        y = y[good]

        if xerr is not None:
            xerr = xerr[good]

        if yerr is not None:
            yerr = yerr[good]

        if len(x) < 2:
            return None, None, None

        # Initial unweighted fit
        slope0, intercept0 = np.polyfit(x, y, 1)

        # If no uncertainty info is available, use a robust unweighted fit
        if yerr is None:
            sigma_eff = np.ones_like(y)
        else:
            # Initial effective uncertainty using the first slope estimate
            if xerr is not None:
                sigma_eff = np.sqrt(yerr ** 2 + (slope0 * xerr) ** 2)
            else:
                sigma_eff = yerr.copy()

            # Prevent tiny error bars from dominating
            sigma_floor = max(0.25 * np.nanmedian(sigma_eff), 1e-6)
            sigma_eff = np.maximum(sigma_eff, sigma_floor)

        try:
            from scipy.optimize import least_squares

            def residuals(params):
                slope, intercept = params
                y_model = slope * x + intercept
                return (y - y_model) / sigma_eff

            result = least_squares(
                residuals,
                x0=[slope0, intercept0],
                loss=loss_method,
                f_scale=f_scale
            )

            slope, intercept = result.x

            # Update sigma_eff once using the robust-fit slope
            if yerr is not None and xerr is not None:
                sigma_eff = np.sqrt(yerr ** 2 + (slope * xerr) ** 2)
                sigma_floor = max(0.25 * np.nanmedian(sigma_eff), 1e-6)
                sigma_eff = np.maximum(sigma_eff, sigma_floor)

                result = least_squares(
                    residuals,
                    x0=[slope, intercept],
                    loss=loss_method,
                    f_scale=f_scale
                )

                slope, intercept = result.x

        except Exception:
            # Fallback to weighted polyfit if scipy robust fit fails
            weights = 1.0 / sigma_eff
            slope, intercept = np.polyfit(x, y, 1, w=weights)

        y_fit = slope * x + intercept

        # Weighted R^2
        weights_r2 = 1.0 / sigma_eff ** 2
        y_mean_weighted = np.average(y, weights=weights_r2)

        ss_res = np.sum(weights_r2 * (y - y_fit) ** 2)
        ss_tot = np.sum(weights_r2 * (y - y_mean_weighted) ** 2)

        if ss_tot > 0:
            r2 = 1 - ss_res / ss_tot
        else:
            r2 = np.nan

        return slope, intercept, r2

    def Qtemps_vs_time_comb_allQs_1col(self, all_qubit_temperatures_ssf_g, all_qubit_timestamps_ssf_g,
                                              out_dir, all_files_Qtemp_results_RPMs, all_qubit_temps_errs_g,
                                            restrict_time_xaxis=False, restrict_time_yaxis = False, ylims = [],
                                            plot_extra_event_lines=False, rad_events_plot_lines=False, qubits_to_plot = None,
                                            plot_rpm_I_only=False, plot_rpm_Q_only=False):
        """Works for more than 2 qubits and does not plot g-e ssf method, only rpm and regular ground state ssf method
            Always plots err bars.
        """
        os.makedirs(out_dir, exist_ok=True)

        # --- Build RPM dicts ---
        num_qubits = self.number_of_qubits  # expect 6 in your setup
        times_RPM = {q: [] for q in range(num_qubits)}
        temps_RPM = {q: [] for q in range(num_qubits)}
        errs_RPM = {q: [] for q in range(num_qubits)}

        if plot_rpm_I_only:
            times_RPM_I = {q: [] for q in range(num_qubits)}
            temps_RPM_I = {q: [] for q in range(num_qubits)}
            errs_RPM_I = {q: [] for q in range(num_qubits)}

        if plot_rpm_Q_only:
            times_RPM_Q = {q: [] for q in range(num_qubits)}
            temps_RPM_Q = {q: [] for q in range(num_qubits)}
            errs_RPM_Q = {q: [] for q in range(num_qubits)}

        for rec in all_files_Qtemp_results_RPMs:
            for q in range(num_qubits):
                d = rec.get("qubits", {}).get(q)
                if not d:
                    continue

                # if d["T_mK_err"]/d["T_mK"] > 0.5:
                #     continue

                cutoff_temp = 600
                if d["T_mK"] > cutoff_temp:
                    continue

                t = datetime.datetime.fromtimestamp(d["date"])
                times_RPM[q].append(t)
                temps_RPM[q].append(d["T_mK"])
                errs_RPM[q].append(d["T_mK_err"])

                qf = d.get("qubit_freq_MHz", None)
                ef = d.get("Qfreq_fit_err", None)  # MHz

                rr = None
                if plot_rpm_Q_only or plot_rpm_I_only:
                    date = ""
                    figure_quality = ""
                    save_figs = ""
                    fit_saved = ""
                    signal = ""
                    run_name = ""
                    number_of_qubits = ""
                    outerFolder = ""
                    outerFolder_save_plots = ""
                    unique_folder_path = ""
                    run_num = ""
                    filter_out_bad_RPM_fits = ""
                    rr = PlotRR_noQick(date, figure_quality, save_figs, fit_saved, signal, run_name, number_of_qubits, outerFolder,
                                        outerFolder_save_plots, unique_folder_path, run_num, filter_out_bad_RPM_fits)

                # ------------------ Optional RPM from I-only amplitudes ------------------
                # Pe distribution error from I-only amplitude case has not been implemented yet!!!
                # Here we just include fit err into Pe_I
                if plot_rpm_I_only:
                    A1I = d.get("A_I_1", None)
                    A2I = d.get("A_I_2", None)
                    sA1I = d.get("sigma_A_I_1", None)
                    sA2I = d.get("sigma_A_I_2", None)

                    if (A1I is not None and A2I is not None and qf is not None
                            and sA1I is not None and sA2I is not None and ef is not None):

                        out = rr.Qubit_Temperature_Convert(A1I, A2I, qf)
                        if out is not None:
                            _, TmK_I, Pe_I, _ = out

                            Terr_I,_ = rr.compute_temperature_error_RPM(
                                A1=A1I, A2=A2I, Pe=Pe_I, T_mK=TmK_I, qubit_freq_MHz=qf,
                                sigma_A1=sA1I, sigma_A2=sA2I, sigma_qfreq_MHz=ef
                            )

                            if np.isfinite(TmK_I) and np.isfinite(Terr_I) and TmK_I <= 600:
                                times_RPM_I[q].append(t)
                                temps_RPM_I[q].append(TmK_I)
                                errs_RPM_I[q].append(Terr_I)

                # ------------------ Optional RPM from Q-only amplitudes ------------------
                # Pe distribution error from Q-only amplitude case has not been implemented yet!!!
                # Here we just include fit err into Pe_Q
                if plot_rpm_Q_only:
                    A1Q = d.get("A_Q_1", None)
                    A2Q = d.get("A_Q_2", None)
                    sA1Q = d.get("sigma_A_Q_1", None)
                    sA2Q = d.get("sigma_A_Q_2", None)

                    if (A1Q is not None and A2Q is not None and qf is not None
                            and sA1Q is not None and sA2Q is not None and ef is not None):

                        out = rr.Qubit_Temperature_Convert(A1Q, A2Q, qf)
                        if out is not None:
                            _, TmK_Q, Pe_Q, _ = out

                            Terr_Q, _ = rr.compute_temperature_error_RPM(
                                A1=A1Q, A2=A2Q, Pe=Pe_Q, T_mK=TmK_Q, qubit_freq_MHz=qf,
                                sigma_A1=sA1Q, sigma_A2=sA2Q, sigma_qfreq_MHz=ef
                            )

                            if np.isfinite(TmK_Q) and np.isfinite(Terr_Q) and TmK_Q <= cutoff_temp:
                                times_RPM_Q[q].append(t)
                                temps_RPM_Q[q].append(TmK_Q)
                                errs_RPM_Q[q].append(Terr_Q)

        # --- SSF (g-only) dicts ---
        times_g = all_qubit_timestamps_ssf_g  # {q: [datetime...]}
        temps_g = all_qubit_temperatures_ssf_g  # {q: [float...]}
        errs_g = all_qubit_temps_errs_g # {q: [float...]}

        # --- Optional time window ---
        if restrict_time_xaxis:
            window_start = datetime.datetime(2025, 4, 18, 0, 0)
            window_end = datetime.datetime(2025, 5, 4, 23, 59)

        # --- Optional radiation events ---
        rad_events = []
        if rad_events_plot_lines:
            rad_events = [
                (datetime.datetime(2025, 4, 21, 12, 35), "Co-60"),
                (datetime.datetime(2025, 4, 23, 12, 53), "Cs-137"),
                (datetime.datetime(2025, 4, 28, 9, 40), "Cs-137 closer"),
                (datetime.datetime(2025, 5, 4, 18, 20), "Cs-137 removed"),
            ]

        # --- Decide which qubits to plot ---
        if qubits_to_plot is None:
            qubits_to_plot = list(range(num_qubits))  # default: all qubits
        else:
            # clean + clamp to valid range
            qubits_to_plot = sorted(
                q for q in qubits_to_plot
                if isinstance(q, int) and 0 <= q < num_qubits
            )
        if not qubits_to_plot:
            raise ValueError("qubits_to_plot is empty after filtering valid indices.")

        # --- Make N rows (one per qubit), 1 column ---
        nrows = len(qubits_to_plot)
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 3.2 * nrows), sharex=True, constrained_layout=True)
        if nrows == 1:
            axes = [axes]

        date_fmt = DateFormatter('%m-%d-%H')

        methods = [
            ("RPM Qtemps (IQ amp)", times_RPM, temps_RPM, errs_RPM, "orange"),
            ("SSF Qtemps", times_g, temps_g, errs_g, "blue")]

        if plot_rpm_I_only:
            methods.insert(1, ("RPM Qtemps (I-only)", times_RPM_I, temps_RPM_I, errs_RPM_I, "green"))

        if plot_rpm_Q_only:
            methods.insert(1, ("RPM Qtemps (Q-only)", times_RPM_Q, temps_RPM_Q, errs_RPM_Q, "purple"))

        for ax, q in zip(axes, qubits_to_plot):
            for label, tdict, ydict, edict, color in methods:
                ts = tdict.get(q, [])
                ys = ydict.get(q, [])
                es = None if edict is None else edict.get(q, [])

                if ts and ys:
                    # Use errors only if they're valid and the same length
                    use_yerr = es is not None and len(es) == len(ys)

                    ax.errorbar(
                        ts,
                        ys,
                        yerr=es if use_yerr else None,
                        fmt='o',
                        markersize=4,
                        elinewidth=1,
                        capsize=3,
                        alpha=0.85,
                        color=color,
                        ecolor=color,
                        markeredgecolor='k',
                        label=label
                    )

            ax.set_title(f"Q{q + 1}", loc="left", fontsize=13, fontweight="bold")
            ax.set_ylabel("Temp (mK)")
            ax.grid(False)

            # ax.xaxis.set_major_locator(mdates.AutoDateLocator()) # automatic
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))

            if restrict_time_yaxis:
                ax.set_ylim(ylims[0], ylims[1])
            ax.xaxis.set_major_formatter(date_fmt)
            ax.tick_params(axis='x', rotation=45, labelsize=9)

            if restrict_time_xaxis:
                ax.set_xlim(window_start, window_end)

            for t_evt, lbl in rad_events:
                ax.axvline(t_evt, color='gray', linestyle='--', linewidth=1)
                ax.text(t_evt, ax.get_ylim()[1] * 0.9, lbl, rotation=90, va='top', ha='right', fontsize=8)

            # ------------------ Distribution-level mean difference (RPM - SSF) ------------------
            mean_diff_str = None

            if q in temps_RPM and q in temps_g and temps_RPM[q] and temps_g[q]:
                mean_diff = np.nanmean(temps_RPM[q]) - np.nanmean(temps_g[q])
                mean_diff_str = f"<RPM> - <SSF> = {mean_diff:.4f} mK"

            handles, labels = ax.get_legend_handles_labels()

            if mean_diff_str is not None:
                handles.append(plt.Line2D([], [], color='none'))
                labels.append(mean_diff_str)

            ax.legend(handles, labels, loc="upper left", fontsize=9, frameon=False)

        axes[-1].set_xlabel("Time")
        fig.suptitle("Effective Qubit Temperatures vs Time", fontsize=15)

        # Save
        paramvstime_dir = os.path.join(out_dir, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(paramvstime_dir, f"Qtemps_Comparison_TwoMethods_AllQs_{stamp}.pdf")
        fig.savefig(out_path) # dpi=self.figure_quality
        plt.close(fig)
        print("Saved combined methods plot: ", out_path)
        return out_path

    def load_mixing_chamber_csv(self, csv_path, restrict_time=False,
                                start_time=None, end_time=None):
        """
        Load mixing-chamber CSV and return:
            - times: list[str] formatted '%Y-%m-%d %H:%M:%S'
            - mix_s: list[float]   (DRI-MIX-S, mK)
            - mix_h: list[float]   (DRI-MIX-H, µW)

        If restrict_time=True:
            Only rows within [start_time, end_time] are kept.
            Both start_time and end_time must be in '%Y-%m-%d %H:%M:%S' format.
            If either is None, that bound is ignored.
        """

        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        # --- Parse timestamps into datetime ---
        df["Time"] = pd.to_datetime(df["Time"])

        # --- Apply time restriction if requested ---
        if restrict_time:
            # Convert to datetime if provided
            if start_time is not None:
                start_time = pd.to_datetime(start_time)
                df = df[df["Time"] >= start_time]

            if end_time is not None:
                end_time = pd.to_datetime(end_time)
                df = df[df["Time"] <= end_time]

        # --- Format timestamps to match your T1 format ---
        times = df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

        # --- Convert values (remove units, convert to float) ---
        mix_s = (
            df["DRI-MIX-S"]
            .astype(str)
            .str.replace("mK", "", regex=False)
            .str.strip()
            .astype(float)
            .tolist()
        )

        mix_h = (
            df["DRI-MIX-H"]
            .astype(str)
            .str.replace("µW", "", regex=False)
            .str.strip()
            .astype(float)
            .tolist()
        )

        return times, mix_s, mix_h


    def plot_qtemps_and_coherence_res(self, out_dir, all_qubit_temperatures_ssf_g = None, all_qubit_timestamps_ssf_g = None,
                                      all_files_Qtemp_results_RPMs = None, fridge_temps = None, fridge_dates = None,
                                      t1_vals = None, t1_dates = None, qfreqs_vals = None, qfreqs_dates = None, resfreqs_vals=None,
                                        resfreqs_dates=None, t2r_vals=None, t2r_dates=None, t2e_vals=None, t2e_dates=None,
                                      restrict_time_xaxis=False, start_time = None, end_time = None, plot_extra_event_lines=False,
                                      rad_events_plot_lines=False, run_num =""):
        """
        One subplot per qubit.
        Plots (only if provided):
          - RPM Qtemps
          - SSF g-only Qtemps
          - Fridge mixing-chamber temp
          - T1 (µs)
          - T2 Ramsey (µs)
          - T2 Echo (µs)
          - Qubit freq (MHz)
          - Resonator freq (MHz)
        """
        os.makedirs(out_dir, exist_ok=True)
        num_qubits = self.number_of_qubits
        qubits_to_plot = list(range(num_qubits))
        nrows = len(qubits_to_plot)
        small_fs = 14  # font size for y axes labels and ticks

        # ------------------------------------------------------------------
        # RPM Qtemps (per-qubit dicts)
        # ------------------------------------------------------------------
        times_RPM = {q: [] for q in range(num_qubits)}
        temps_RPM = {q: [] for q in range(num_qubits)}

        if all_files_Qtemp_results_RPMs is not None:
            for rec in all_files_Qtemp_results_RPMs:
                for q in range(num_qubits):
                    d = rec.get("qubits", {}).get(q)
                    if not d:
                        continue
                    if d["T_mK"] < 150: # mK, just filtering out bad data for run 9
                        t = datetime.datetime.fromtimestamp(d["date"])
                        times_RPM[q].append(t)
                        temps_RPM[q].append(d["T_mK"])

        # ------------------------------------------------------------------
        # SSF g-only Qtemps (expect dicts {q: [datetimes]} and {q: [floats]})
        # ------------------------------------------------------------------
        if all_qubit_timestamps_ssf_g is not None:
            times_g = all_qubit_timestamps_ssf_g
        else:
            times_g = {q: [] for q in range(num_qubits)}

        if all_qubit_temperatures_ssf_g is not None:
            temps_g = all_qubit_temperatures_ssf_g
        else:
            temps_g = {q: [] for q in range(num_qubits)}

        # ------------------------------------------------------------------
        # Fridge temperatures (global series, same for all qubits)
        # fridge_dates: list of strings "%Y-%m-%d %H:%M:%S"
        # fridge_temps: list of floats (mK)
        # ------------------------------------------------------------------
        fridge_times = None
        if fridge_temps is not None and fridge_dates is not None:
            time_fmt = "%Y-%m-%d %H:%M:%S"
            fridge_times = [
                datetime.datetime.strptime(d, time_fmt) for d in fridge_dates]

        # ------------------------------------------------------------------
        # T1 per qubit (2D lists: t1_dates[q] -> list[str], t1_vals[q] -> list[float])
        # ------------------------------------------------------------------
        t1_times = {q: [] for q in range(num_qubits)}
        t1_values = {q: [] for q in range(num_qubits)}

        if t1_vals is not None and t1_dates is not None:
            time_fmt = "%Y-%m-%d %H:%M:%S"
            for q in qubits_to_plot:
                if q >= len(t1_dates) or q >= len(t1_vals):
                    continue
                q_dates = t1_dates[q]
                q_vals = t1_vals[q]
                if not q_dates or not q_vals:
                    continue
                q_times = [datetime.datetime.strptime(d, time_fmt) for d in q_dates]
                order = np.argsort(q_times)
                t1_times[q] = list(np.array(q_times)[order])
                t1_values[q] = list(np.array(q_vals)[order])

        # ------------------------------------------------------------------
        # Qubit frequencies per qubit (2D lists, MHz)
        # ------------------------------------------------------------------
        qfreq_times = {q: [] for q in range(num_qubits)}
        qfreq_values = {q: [] for q in range(num_qubits)}

        if qfreqs_vals is not None and qfreqs_dates is not None:
            time_fmt = "%Y-%m-%d %H:%M:%S"

            for q in qubits_to_plot:
                if q >= len(qfreqs_dates) or q >= len(qfreqs_vals):
                    continue
                q_dates = qfreqs_dates[q]
                q_vals = qfreqs_vals[q]
                if not q_dates or not q_vals:
                    continue
                q_times = [datetime.datetime.strptime(d, time_fmt) for d in q_dates]
                order = np.argsort(q_times)
                qfreq_times[q] = list(np.array(q_times)[order])
                qfreq_values[q] = list(np.array(q_vals)[order])

        # ------------------------------------------------------------------
        # Resonator frequencies per qubit (2D lists, MHz)
        # ------------------------------------------------------------------
        resfreq_times = {q: [] for q in range(num_qubits)}
        resfreq_values = {q: [] for q in range(num_qubits)}

        if resfreqs_vals is not None and resfreqs_dates is not None:
            time_fmt = "%Y-%m-%d %H:%M:%S"
            for q in qubits_to_plot:
                if q >= len(resfreqs_dates) or q >= len(resfreqs_vals):
                    continue
                q_dates = resfreqs_dates[q]
                q_vals = resfreqs_vals[q]
                if not q_dates or not q_vals:
                    continue
                q_times = [datetime.datetime.strptime(d, time_fmt) for d in q_dates]
                order = np.argsort(q_times)
                resfreq_times[q] = list(np.array(q_times)[order])
                resfreq_values[q] = list(np.array(q_vals)[order])

        # ------------------------------------------------------------------
        # T2 Ramsey per qubit (2D lists: t2r_dates[q] -> list[str], t2r_vals[q] -> list[float])
        # ------------------------------------------------------------------
        t2r_times = {q: [] for q in range(num_qubits)}
        t2r_values = {q: [] for q in range(num_qubits)}

        if t2r_vals is not None and t2r_dates is not None:
            time_fmt = "%Y-%m-%d %H:%M:%S"
            for q in qubits_to_plot:
                if q >= len(t2r_dates) or q >= len(t2r_vals):
                    continue
                q_dates = t2r_dates[q]
                q_vals = t2r_vals[q]
                if not q_dates or not q_vals:
                    continue
                q_times = [datetime.datetime.strptime(d, time_fmt) for d in q_dates]
                order = np.argsort(q_times)
                t2r_times[q] = list(np.array(q_times)[order])
                t2r_values[q] = list(np.array(q_vals)[order])

        # ------------------------------------------------------------------
        # T2 Echo per qubit (2D lists: t2e_dates[q] -> list[str], t2e_vals[q] -> list[float])
        # ------------------------------------------------------------------
        t2e_times = {q: [] for q in range(num_qubits)}
        t2e_values = {q: [] for q in range(num_qubits)}

        if t2e_vals is not None and t2e_dates is not None:
            time_fmt = "%Y-%m-%d %H:%M:%S"

            for q in qubits_to_plot:
                if q >= len(t2e_dates) or q >= len(t2e_vals):
                    continue
                q_dates = t2e_dates[q]
                q_vals = t2e_vals[q]
                if not q_dates or not q_vals:
                    continue
                q_times = [datetime.datetime.strptime(d, time_fmt) for d in q_dates]
                order = np.argsort(q_times)
                t2e_times[q] = list(np.array(q_times)[order])
                t2e_values[q] = list(np.array(q_vals)[order])

        # ------------------------------------------------------------------
        # Decide which qubits actually have any data
        # ------------------------------------------------------------------
        qubits_to_plot = []
        for q in range(num_qubits):
            has_rpm = bool(times_RPM[q])
            has_ssf = bool(times_g.get(q, []))
            has_t1 = bool(t1_times[q])
            has_t2r = bool(t2r_times[q])
            has_t2e = bool(t2e_times[q])
            has_qf = bool(qfreq_times[q])
            has_rf = bool(resfreq_times[q])

            if has_rpm or has_ssf or has_t1 or has_t2r or has_t2e or has_qf or has_rf:
                qubits_to_plot.append(q)

        if not qubits_to_plot:
            print("[WARN] No data found for any qubit. Nothing to plot.")
            return None

        nrows = len(qubits_to_plot)

        # ------------------------------------------------------------------
        # Optional time window
        # ------------------------------------------------------------------
        if restrict_time_xaxis:
            window_start = start_time
            window_end = end_time

        # ------------------------------------------------------------------
        # Optional radiation events
        # ------------------------------------------------------------------
        rad_events = []
        if rad_events_plot_lines:
            rad_events = [
                (datetime.datetime(2025, 4, 21, 12, 35), "Co-60"),
                (datetime.datetime(2025, 4, 23, 12, 53), "Cs-137"),
                (datetime.datetime(2025, 4, 28, 9, 40), "Cs-137 closer"),
                (datetime.datetime(2025, 5, 4, 18, 20), "Cs-137 removed"),
            ]

        # ------------------------------------------------------------------
        # Make figure: 1 row per qubit
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(nrows, 1, figsize=(27, 8 * nrows), sharex=True, constrained_layout=True)
        ln_style = "-"

        if nrows == 1:
            axes = [axes]

        #date_fmt = DateFormatter("%m-%d-%H")
        date_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M:%S")

        # Qtemp methods
        methods = [
            ("RPM Qtemps", times_RPM, temps_RPM, "black"),
            ("SSF g-only Qtemps", times_g, temps_g, "blue"),
        ]

        for ax, q in zip(axes, qubits_to_plot):
            # ------------------------------
            # Left axis: Qtemp (mK)
            # ------------------------------
            for label, tdict, ydict, color in methods:
                ts = tdict.get(q, []) if isinstance(tdict, dict) else []
                ys = ydict.get(q, []) if isinstance(ydict, dict) else []

                # Sort by timestamp so the connecting line follows time order
                order = np.argsort(ts)

                ts = np.array(ts)[order]
                ys = np.array(ys)[order]

                if len(ts) > 0 and len(ys) > 0:
                    ax.plot(
                        ts,
                        ys,
                        linestyle=ln_style,
                        linewidth=1.2,
                        marker="o",  # default scatter-like marker
                        markersize=5,
                        markeredgecolor="k",
                        markerfacecolor=color,
                        color=color,
                        alpha=0.85,
                        label=label,
                    )

            ax.set_title(f"Q{q + 1}", loc="left", fontsize=14, fontweight="bold")
            ax.set_ylabel("Qubit Effective Temperature (mK)", fontsize=small_fs)
            ax.grid(False)

            #ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            locator = mdates.AutoDateLocator(minticks=8, maxticks=12)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(date_fmt)
            ax.tick_params(axis="x", rotation=45, labelsize=small_fs)
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_xlabel("Time", fontsize=small_fs)
            ax.tick_params(axis="y", labelsize=small_fs)
            # left_ymin, left_ymax = ax.get_ylim() # get y limits to use them for fridge y axis too
            if restrict_time_xaxis:
                ax.set_xlim(window_start, window_end)

            # ------------------------------
            # Extra y-axes
            # ------------------------------
            right_axes_offset = 0
            handles, labels = ax.get_legend_handles_labels()

            # Fridge temp (global) on first right axis
            fridge_ax = None
            if fridge_times is not None and fridge_temps is not None:
                # ax.plot(
                #     fridge_times,
                #     fridge_temps,
                #     color="green",
                #     alpha=0.5,
                #     linewidth=1.5,
                #     label="Fridge MCP1 (mK)",
                # )
                ## To make the fridge temp have its own y axis:
                fridge_ax = ax.twinx()
                fridge_ax.plot(
                    fridge_times,
                    fridge_temps,
                    color="green",
                    alpha=0.5,
                    linewidth=1.5,
                    label="Fridge MCP1 (mK)",
                )
                fridge_ax.set_ylabel("MCP1 Temp (mK)", color="green", fontsize=small_fs)
                fridge_ax.tick_params(axis="y", labelcolor="green", labelsize=small_fs)

                # # Match Qtemps axis limits
                # fridge_ax.set_ylim(left_ymin, left_ymax)
                #
                # # Match Qtemps ticks
                # fridge_ax.set_yticks(ax.get_yticks())

                h2, l2 = fridge_ax.get_legend_handles_labels()
                handles += h2
                labels += l2
                right_axes_offset = 1.0

            # T1 (µs) on second right axis (per qubit)
            if t1_times[q] and t1_values[q]:
                t1_ax = ax.twinx()
                t1_ax.spines["right"].set_position(("axes", 1.0 + 0.08 * right_axes_offset))
                t1_ax.plot(
                    t1_times[q],
                    t1_values[q],
                    marker="^",
                    markersize=5,
                    linestyle=ln_style,
                    color="red",
                    alpha=0.8,
                    label="T1 (µs)",
                )
                t1_ax.set_ylabel("T1 (µs)", color="red", fontsize=small_fs)
                t1_ax.tick_params(axis="y", labelcolor="red", labelsize=small_fs)
                h3, l3 = t1_ax.get_legend_handles_labels()
                handles += h3
                labels += l3
                right_axes_offset += 1

            # T2 Ramsey (µs) on next right axis (per qubit)
            if t2r_times[q] and t2r_values[q]:
                t2r_ax = ax.twinx()
                t2r_ax.spines["right"].set_position(("axes", 1.0 + 0.05 * right_axes_offset))
                t2r_ax.plot(
                    t2r_times[q],
                    t2r_values[q],
                    marker="v",
                    markersize=5,
                    linestyle=ln_style,
                    color="orange",
                    alpha=0.8,
                    label="T2R (µs)",
                )
                t2r_ax.set_ylabel("T2R (µs)", color="orange", fontsize=small_fs)
                t2r_ax.tick_params(axis="y", labelcolor="orange", labelsize=small_fs)
                h_t2r, l_t2r = t2r_ax.get_legend_handles_labels()
                handles += h_t2r
                labels += l_t2r
                right_axes_offset += 1

            # T2 Echo (µs) on next right axis (per qubit)
            if t2e_times[q] and t2e_values[q]:
                t2e_ax = ax.twinx()
                t2e_ax.spines["right"].set_position(("axes", 1.0 + 0.05 * right_axes_offset))
                t2e_ax.plot(
                    t2e_times[q],
                    t2e_values[q],
                    marker="D",
                    markersize=5,
                    linestyle=ln_style,
                    color="green",
                    alpha=0.8,
                    label="T2E (µs)",
                )
                t2e_ax.set_ylabel("T2E (µs)", color="green", fontsize=small_fs)
                t2e_ax.tick_params(axis="y", labelcolor="green", labelsize=small_fs)
                h_t2e, l_t2e = t2e_ax.get_legend_handles_labels()
                handles += h_t2e
                labels += l_t2e
                right_axes_offset += 1

            # Qfreq (MHz) on third right axis (per qubit)
            if qfreq_times[q] and qfreq_values[q]:
                qf_ax = ax.twinx()
                qf_ax.spines["right"].set_position(("axes", 1.0 + 0.05 * right_axes_offset))
                qf_ax.plot(
                    qfreq_times[q],
                    qfreq_values[q],
                    marker="s",
                    markersize=5,
                    linestyle=ln_style,
                    color="purple",
                    alpha=0.8,
                    label="Qfreq (MHz)",
                )
                qf_ax.set_ylabel("Qfreq (MHz)", color="purple", fontsize=small_fs)
                qf_ax.tick_params(axis="y", labelcolor="purple", labelsize=small_fs)
                qf_ax.yaxis.get_offset_text().set_visible(False)
                qf_ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f")) # Force 2-decimal formatting
                h4, l4 = qf_ax.get_legend_handles_labels()
                handles += h4
                labels += l4
                right_axes_offset += 1

            # Resonator freq (MHz) on next right axis (per qubit)
            if resfreq_times[q] and resfreq_values[q]:
                rf_ax = ax.twinx()

                # Keep resonator data behind the legend
                rf_ax.set_zorder(0)
                ax.set_zorder(1)
                ax.patch.set_visible(False)

                rf_ax.spines["right"].set_position(("axes", 1.0 + 0.053 * right_axes_offset))
                rf_ax.plot(
                    resfreq_times[q],
                    resfreq_values[q],
                    marker="o",
                    markersize=5,
                    linestyle=ln_style,
                    color="blue",
                    alpha=0.8,
                    label="Res freq (MHz)"
                )
                rf_ax.set_ylabel("Res freq (MHz)", color="blue", fontsize=small_fs)
                rf_ax.tick_params(axis="y", labelcolor="blue", labelsize=small_fs)
                rf_ax.yaxis.get_offset_text().set_visible(False)
                rf_ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
                h_rf, l_rf = rf_ax.get_legend_handles_labels()
                handles += h_rf
                labels += l_rf
                right_axes_offset += 1

            # Radiation event lines
            for t_evt, lbl in rad_events:
                ax.axvline(t_evt, color="gray", linestyle="--", linewidth=1)
                ax.text(
                    t_evt,
                    ax.get_ylim()[1] * 0.9,
                    lbl,
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=8,
                )

            if handles:
                leg = ax.legend(
                    handles,
                    labels,
                    loc="upper left",
                    fontsize=10,
                    frameon=True,
                    fancybox=False
                )
                # Force legend to be drawn on top of the data
                leg.set_zorder(10000000)

                frame = leg.get_frame()
                frame.set_facecolor("white")
                frame.set_alpha(1.0)
                frame.set_edgecolor("black")
                frame.set_linewidth(0.8)

        #axes[-1].set_xlabel("Time", fontsize = small_fs) if u only want it under the last subplot
        fig.suptitle("Qtemps, Fridge, T1, T2R, T2E, Qfreq, and Res Freq vs Time", fontsize=18)

        # Save
        paramvstime_dir = os.path.join(out_dir, "params_vs_time")
        os.makedirs(paramvstime_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(paramvstime_dir, f"run{run_num}_Qtemps_Coherence_allQs_{stamp}.pdf")
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
        print("Saved combined methods plot: ", out_path)
        return out_path