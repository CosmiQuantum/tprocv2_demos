from bisect import bisect_left
import re
import ast
import numpy as np
import h5py
from sklearn.mixture import GaussianMixture
from qicklab.analysis import qspec, t1, ssf
from matplotlib.ticker import MaxNLocator
from analysis_021_plot_allRR_noqick import PlotRR_noQick
import math
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from bisect import bisect_left
from matplotlib.dates import DateFormatter

save_figs = True
figure_quality = 100 #ramp this up to like 500 for presentation plots


class SSFTempCalcAndPlots:
    def __init__(self, figure_quality, number_of_qubits, save_figs):
        self.save_figs = save_figs
        self.figure_quality = figure_quality
        self.number_of_qubits = number_of_qubits

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
            "crossing_point": <float>,
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
                sigma_TmK = self.compute_temperature_error_SSF(Pe, sigma_Pe, T_mK, freq_mhz, freq_mhz_err)

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
                    "used_gessf_thresh_only": use_gessf_thresh_only, #True when the user decides to use this method
                    "used_fallback_method": used_fallback, #only True if it goes into effect, regardless of user decision
                })

        return all_qubit_temperatures, all_qubit_timestamps, all_qubit_temperatures_errs, fit_results


    def compute_temperature_error_SSF(self, Pe, sigma_Pe, T_mK, qubit_freq_MHz, sigma_qfreq_MHz):
        """
        Propagate the 1-σ uncertainties in Pe and f_ge
        into a 1-σ uncertainty on T_mK, given you already know
        Pe, σ_Pe, and T_mK.

        Inputs:
          Pe                    – excited‐state population (from SSF)
          sigma_Pe              – 1-σ uncertainty on Pe
          T_mK                  – computed temperature (mK)
          qubit_freq_MHz        – fitted g–e qubit frequency (MHz)
          sigma_qfreq_MHz       – 1-σ error on qubit_freq_MHz (standard deviation)

        Returns:
          sigma_T_mK            – propagated 1-σ error on T_mK (mK)
        """
        # Convert qubit frequency and its error from MHz → Hz
        f0_Hz = qubit_freq_MHz * 1e6
        sigma_f0_Hz = sigma_qfreq_MHz * 1e6

        # Build the logarithmic term (given Pe)
        ln_arg = np.log((1.0 - Pe) / Pe)

        # Partial derivatives of T_mK
        #    T_mK = (h * f0_Hz) / (kB * ln_arg) * 1e3
        #    ∂T/∂f0  = T_mK / f0_Hz
        dT_df0 = T_mK / f0_Hz

        # ∂T/∂Pe = T_mK / [ ln_arg * Pe * (1 - Pe) ]
        dT_dPe = T_mK / (ln_arg * Pe * (1.0 - Pe))

        # Combine in quadrature to get total σ_T_mK
        sigma_T_mK = np.sqrt(
            (dT_df0 * sigma_f0_Hz) ** 2 +
            (dT_dPe * sigma_Pe) ** 2
        )

        return sigma_T_mK

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

    def plot_gaussians_qtemps(self, q_key, qubit_folder, ig_new, ground_data, excited_data, ground_gaussian, excited_gaussian, pop_threshold, temperature_mk, dataset, weights, sigmas, means):
        # -----------------PLOTS TO CHECK g-state double gaussian FITS AND THRESHOLDS---------------
        # Plotting double gaussian distributions and fitting
        xlims = [np.min(ig_new), np.max(ig_new)]
        plt.figure(figsize=(10, 6))

        # Plot histogram for `ig_new`
        steps = 3000
        # numbins = round(math.sqrt(steps))
        numbins = 64
        n, bins, _ = plt.hist(ig_new, bins=numbins, range=xlims, density=False, alpha=0.5,
                              label='Histogram of $I_g$',
                              color='gray')
        # print(numbins)
        # Use the midpoints of bins to create boolean masks
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ground_region = (bin_centers <= pop_threshold)
        excited_region = (bin_centers > pop_threshold)

        # Calculate scaling factors for each region
        scaling_factor_ground = max(n[ground_region]) / max(
            (weights[ground_gaussian] / (np.sqrt(2 * np.pi) * sigmas[ground_gaussian])) * np.exp(
                -0.5 * ((bin_centers[ground_region] - means[ground_gaussian]) / sigmas[
                    ground_gaussian]) ** 2))

        scaling_factor_excited = max(n[excited_region]) / max(
            (weights[excited_gaussian] / (np.sqrt(2 * np.pi) * sigmas[excited_gaussian])) * np.exp(
                -0.5 * ((bin_centers[excited_region] - means[excited_gaussian]) / sigmas[
                    excited_gaussian]) ** 2))

        # Generate x values for plotting Gaussian components
        x = np.linspace(xlims[0], xlims[1], 1000)
        ground_gaussian_fit = scaling_factor_ground * (
                weights[ground_gaussian] / (np.sqrt(2 * np.pi) * sigmas[ground_gaussian])) * np.exp(
            -0.5 * ((x - means[ground_gaussian]) / sigmas[ground_gaussian]) ** 2)
        excited_gaussian_fit = scaling_factor_excited * (
                weights[excited_gaussian] / (np.sqrt(2 * np.pi) * sigmas[excited_gaussian])) * np.exp(
            -0.5 * ((x - means[excited_gaussian]) / sigmas[excited_gaussian]) ** 2)

        plt.plot(x, ground_gaussian_fit, label='Ground Gaussian Fit', color='blue', linewidth=2)
        plt.plot(x, excited_gaussian_fit, label='Excited (leakage) Gaussian Fit', color='red', linewidth=2)
        plt.axvline(pop_threshold, color='black', linestyle='--', linewidth=1,
                    label=f'Threshold ({pop_threshold:.2f})')

        # Add shading for ground and excited state regions
        x_vals = np.linspace(np.min(ig_new), np.max(ig_new), 1000)

        # Add shading for ground_data points
        plt.hist(
            ground_data, bins=numbins, range=[np.min(ig_new), np.max(ig_new)], density=False,
            alpha=0.5, color="blue", label="Ground Data Region", zorder=2
        )

        # Add shading for excited_data points
        plt.hist(
            excited_data, bins=numbins, range=[np.min(ig_new), np.max(ig_new)], density=False,
            alpha=0.5, color="red", label="Excited Data Region", zorder=3
        )

        # plt.hist(
        #     iq_data, bins=numbins, range=[np.min(ig_new), np.max(ig_new)], density=False,
        #     alpha=0.2, color="green", label="All IQ Data Region", zorder=1
        # )

        plt.title(
            f"Method: Ground-state double gaussian fit ; Qubit {q_key + 1} ; Temp= {temperature_mk:2f} mK")
        plt.xlabel("$I_g$' " , fontsize=14)
        plt.ylabel('Counts', fontsize=14)
        plt.legend()
        # plt.show()

        # Save the plot to the Temperatures folder
        plot_filename = os.path.join(qubit_folder, f"Q{q_key + 1}_SSF_ground_gaussianfit_Dataset{dataset}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        plt.savefig(plot_filename)
        # print(f"Plot saved to: {qubit_folder}")
        plt.close()

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
            free_ssf = set(ssf_list)

            matches, lonely_spec = [], []
            for t_spec, f_spec in zip(spec_times, spec_list):
                idx = bisect_left(ssf_times, t_spec)
                candidates = []
                if idx < len(ssf_list):
                    candidates.append((ssf_times[idx], ssf_list[idx]))
                if idx > 0:
                    candidates.append((ssf_times[idx - 1], ssf_list[idx - 1]))

                best = None
                for t_ssf, f_ssf in candidates:
                    delta = abs((t_ssf - t_spec).total_seconds())
                    if delta <= tolerance_seconds and (best is None or delta < abs((best[0] - t_spec).total_seconds())):
                        best = (t_ssf, f_ssf)

                if best and best[1] in free_ssf:
                    matches.append((f_spec, best[1]))
                    free_ssf.remove(best[1])
                else:
                    lonely_spec.append(f_spec)

            pairs_by_qubit[qi] = matches
            unmatched_qspec[qi] = lonely_spec
            unmatched_ssf[qi] = list(free_ssf)

        return pairs_by_qubit, unmatched_qspec, unmatched_ssf

    #  Scatter plot – qubit temperatures vs. time  (all dates, each qubit its own subplot)
    def plot_qubit_temperatures_vs_time_ssf(self, all_qubit_temperatures, all_qubit_timestamps, all_qubit_temperatures_errs,
                                            out_dir, plot_error_bars = False):
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
                if e <= 300]
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

            ax.set_title(f"Qubit {q + 1} Temperature vs Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Temperature (mK)")
            # ax.grid(alpha=0.3)
            ax.legend()
            ax.xaxis.set_major_formatter(date_fmt)
            plt.setp(ax.get_xticklabels(), rotation=45, fontsize=10)
            ax.set_yticks(np.linspace(50, 950, 10))
            plt.setp(ax.get_yticklabels(), fontsize=10)

        plt.tight_layout()
        fname = os.path.join(
            out_dir,
            f"AllQubits_Temps_vs_Time_{datetime.datetime.now():%Y%m%d%H%M%S}.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Saved all-dates scatter →", fname)

    # Histograms – temperature distributions  (all dates, each qubit subplot)
    def plot_all_qubits_hist_ssf(self, all_qubit_temperatures, out_dir, bins=20):
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(15, 10))

        for q in all_qubit_temperatures.keys():
            temps = all_qubit_temperatures[q]
            if not temps:
                continue

            ax = plt.subplot(2, 3, q + 1)
            ax.hist(temps, bins=bins,
                    color=colors[q], alpha=0.7, edgecolor='black')
            ax.set_title(f"Qubit {q + 1} Temperature Distribution")
            ax.set_xlabel("Temperature (mK)")
            ax.set_ylabel("Count")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(
            out_dir,
            f"AllQubits_Temp_Hist_{datetime.datetime.now():%Y%m%d%H%M%S}.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Saved all-dates histogram →", fname)

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
        as the midpoint between the two Gaussian means instead of the crossing point of the two gaussians. Also calculates
        the populations properly, since fit_double_gaussian_with_full_coverage used labels and was less accurate.

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
        This function processes the ssf and g-e quit spec data for each qubit found inside the files in cosmiqgpvm02 and returns the dictionary:

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
        freq_cache = {}  # for qubit freqs (MHz)
        freq_err_cache = {}  # 1-σ error (std) on that freq
        ig_new_cache = {}  # for ground state roated I data (SSF)
        ie_new_cache = {}  # for first excited state roated I data (SSF)
        timestamp_ssf_cache = {}  # for ssf data time stamps (qubit temperature time stamps)

        for full_path in paths:
            path = os.path.dirname(full_path)  # one level up from the dataset
            dataset = os.path.basename(full_path)  # just the '2025-04-16_11-47-09' part

            for QubitIndex in Science_Qubits:  # We are only taking science data for some qubits
                try:
                    # --- Load QSpec ---
                    qspec_obj = qspec(path, dataset, QubitIndex)
                    qspec_dates, qspec_n, qspec_probe_freqs, qspec_I, qspec_Q = qspec_obj.load_all()
                    qspec_freqs, qspec_errs, qspec_fwhms = qspec_obj.get_all_qspec_freq(qspec_probe_freqs, qspec_I,
                                                                                        qspec_Q, qspec_n)

                    # recreate the list of file–paths in the SAME order the helper used
                    qspec_dir = os.path.join(path, dataset, qspec_obj.folder, "Data_h5", qspec_obj.expt_name)
                    h5_files = sorted(os.listdir(qspec_dir))
                    h5_paths = [os.path.join(qspec_dir, f) for f in h5_files]

                    for i in range(qspec_n):
                        freq_cache[(h5_paths[i], QubitIndex)] = qspec_freqs[i]
                        freq_err_cache[(h5_paths[i], QubitIndex)] = qspec_errs[i]
                except Exception as e:
                    print(f"Skipped QSpec scan in {dataset} for Q{QubitIndex}: {e}")

                try:
                    # --- Load SSF ---
                    ssf_ge = ssf(path, dataset, QubitIndex)
                    ssf_dates, ssf_n, I_g, Q_g, I_e, Q_e = ssf_ge.load_all()

                    # recreate the list of SSF-file paths in the SAME order the helper used
                    ssf_dir = os.path.join(path, dataset, ssf_ge.folder, "Data_h5", ssf_ge.expt_name)
                    ssf_paths = [os.path.join(ssf_dir, f) for f in sorted(os.listdir(ssf_dir))]  # length==ssf_n

                    # iterate through every round (file)
                    for i in range(ssf_n):
                        try:
                            _, _, _, ig_new, _, ie_new, _, _, _, _, _ = ssf_ge.get_ssf_in_round(I_g, Q_g, I_e, Q_e, i)
                        except Exception as e:
                            print(f"rotate-Ig failed ({ssf_paths[i]}): {e}")
                            continue

                        key = (ssf_paths[i], QubitIndex)
                        ig_new_cache[key] = ig_new
                        ie_new_cache[key] = ie_new
                        timestamp_ssf_cache[key] = ssf_dates[i]

                except Exception as e:
                    print(f"Failed loading SSF for qubit {QubitIndex} from {full_path}: {e}")

        # Organize files by type and qubit index after loading all the data
        qspec_h5s = {q: [] for q in Science_Qubits}
        for (path, qidx) in freq_cache.keys():
            qspec_h5s[qidx].append(path)

        ssf_h5s = {q: [] for q in Science_Qubits}
        for (path, qidx) in ig_new_cache.keys():
            ssf_h5s[qidx].append(path)

        ########################################## Pair up Qspec_ge data and ssf_ge h5 files ###########################################
        pairs_by_qubit, lonely_qspec, lonely_ssf = self.pair_qspec_and_ssf(qspec_h5s, ssf_h5s, tolerance_seconds=10)

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
                    "qfreq_MHz_err": freq_err_cache[fq_key],  # 1-σ (standard deviation) fit error on "qfreq_MHz_err"
                    "ig_new": ig_new_cache[ss_key],
                    "ie_new": ie_new_cache[ss_key],
                    "data_timestamp": timestamp_ssf_cache[ss_key].timestamp(),  # unix-timestamps
                })
        return pairs_info

class RPMTempCalcAndPlots:
    def __init__(self, figure_quality, number_of_qubits, save_figs):
        self.save_figs = save_figs
        self.figure_quality = figure_quality
        self.number_of_qubits = number_of_qubits

    def run_RPMqtemps(self, base_dir, target_dates, filter_keywords, fit_saved, signal, run_name, list_of_all_qubits, tot_num_of_qubits,
                     outerFolder_RR_plots, replot_RPMs = False, get_qtemp_data = False, figure_quality = 200, save_figsRR = False, exclude_temp_sweeps = False):

        combined_qtemp_data = []  # list of results from different .h5 files

        os.makedirs(outerFolder_RR_plots, exist_ok=True)

        #------------------------------------------ Looping through data folders and files -------------------------------------------------------
        # Note: this is tailored for how things are organized in cosmiqgpvm02
        for root, dirs, files in os.walk(base_dir):
            dirs.sort()  # alphabetical → chronological for YYYY-MM-DD_HH-MM-SS
            for d in dirs:
                full_path = os.path.join(root, d)
                # Match folders like '2025-04-16_11-47-09' based on prefix date
                if (
                        any(d.startswith(date) for date in target_dates)
                        and len(d) >= 19
                        and any(keyword in full_path for keyword in filter_keywords)
                        and (not exclude_temp_sweeps or "temperature_sweep" not in full_path.lower())
                    ):  # checks if path includes each keyword (source_off or source_on) and whether you set the temp sweep data to be excluded or not
                    optimization_path = os.path.join(full_path, "optimization")
                    if os.path.isdir(optimization_path):
                        date_string = d[:10]  # Extract 'YYYY-MM-DD'
                        print(f"Analyzing: {optimization_path}")

                        outerFolder = optimization_path  # RR data (g-e Qspec) folder path before Data_h5
                        outerFolder_qtemps_data = optimization_path  # Qubit temps data folder path before Data_h5

                        if not os.path.exists(outerFolder): os.makedirs(outerFolder)
                        if not os.path.exists(outerFolder_qtemps_data): os.makedirs(outerFolder_qtemps_data)

                        # ---------------------------------------- Initialize the PlotRR_noQick class ------------------------------------------------
                        plotter = PlotRR_noQick(date_string, figure_quality, save_figsRR, fit_saved, signal, run_name,
                                                tot_num_of_qubits, outerFolder, outerFolder_RR_plots, outerFolder_qtemps_data)

                        if replot_RPMs:
                            # ------------------------------------To re-plot the RPM plots, or any other data from the selected date-----------------------------------------------------
                            plotter.run(plot_res_spec = False, plot_q_spec = False, plot_rabi = False, plot_ss = False,  ss_plot_gef = False, plot_t1 = False,
                                        plot_t2r = False, plot_t2e = False, plot_rabis_Qtemps = True)

                        if get_qtemp_data:
                            # ---------------------------------------- Load data and append to list spanning multiple dates --------------------------------------------------
                            qtemp_data = plotter.load_plot_save_rabis_Qtemps(list_of_all_qubits, save_figs = False, get_qtemp_data = get_qtemp_data)
                            combined_qtemp_data.extend(qtemp_data)

        return combined_qtemp_data # Will be empty if get_qtemp_data is set to False


class combined_Qtemp_studies:
    def __init__(self, figure_quality, number_of_qubits):
        self.figure_quality = figure_quality
        self.number_of_qubits = number_of_qubits

    def Qtemps_vs_time_comb_methods(self, all_qubit_temperatures_ssf_g, all_qubit_timestamps_ssf_g, all_qubit_temps_ssf_errs_g, all_qubit_temperatures_ssf_ge, all_qubit_timestamps_ssf_ge,
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
