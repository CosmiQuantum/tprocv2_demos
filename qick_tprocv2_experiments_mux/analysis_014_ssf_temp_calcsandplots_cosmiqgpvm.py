from bisect import bisect_left
import re
import ast
import numpy as np
import h5py
from sklearn.mixture import GaussianMixture
from qicklab.analysis import qspec, t1, ssf
from matplotlib.ticker import MaxNLocator
import math
import os
import datetime
import matplotlib.pyplot as plt
from bisect import bisect_left
from matplotlib.dates import DateFormatter

save_figs = True
figure_quality = 100 #ramp this up to like 500 for presentation plots


class TempCalcAndPlots:
    def __init__(self, figure_quality, number_of_qubits, save_figs, outerFolder):
        self.save_figs = save_figs
        self.figure_quality = figure_quality
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder

        # Create the folder if it doesn't exist
        if not os.path.exists(self.outerFolder):
            os.makedirs(self.outerFolder)

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

    def run(self, pairs_info, limit_temp_k=0.8, use_gessf_thresh_only: bool = False, fallback_to_threshold: bool = False):
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
            "sigmas": np.ndarray(shape=(2,)),
            "means": np.ndarray(shape=(2,)),
            "Pg": Pg,
            "Pe": Pe,
          }]}
        """
        # initialise output arrays
        all_qubit_temperatures = {i: [] for i in range(self.number_of_qubits)}
        all_qubit_timestamps = {i: [] for i in range(self.number_of_qubits)}
        fit_results = {qid: [] for qid in range(self.number_of_qubits)}

        for qid, records in pairs_info.items():  # loop over qubits
            for idx, rec in enumerate(records):  # …and every pair
                # Tracker for what happens
                used_fallback = False

                freq_mhz = rec["qfreq_MHz"]
                ig_new = rec["ig_new"]
                ie_new = rec["ie_new"]
                ts_unix = rec["data_timestamp"]

                # Decide which threshold approach to use
                if use_gessf_thresh_only:
                    # ----------Calculate g-e threshold for each ssf file ---------------------
                    ge_thresh, means, sigmas, weights, ground_idx, excited_idx = self.ssf_fit_two_gaussians_midpoint(ig_new, ie_new)

                    #--------------- use g-e SSF threshold to calculate Pg and Pe ---------------
                    mask = (ig_new <= ge_thresh)
                    Pg = mask.mean()
                    Pe = 1.0 - Pg
                    pop_threshold = ge_thresh

                    ground_gaussian = ground_idx
                    excited_gaussian = excited_idx
                    ground_data = excited_data = None

                elif fallback_to_threshold:
                    # -------- double-Gaussian fit on ground state data, with fallback method --------------------------
                    try:
                        (Pg, Pe, gmm, means, sigmas, weights, threshold_mid, ground_gaussian, excited_gaussian,
                         ground_data, excited_data, _) = self.fit_double_gaussian_midpoint(ig_new)

                        pop_threshold = threshold_mid

                        # Ensure crossing point (where threshold is set) isn’t too close to the ground histogram mean
                        mu_g = means[ground_gaussian]
                        sigma_g = np.sqrt(sigmas[ground_gaussian])
                        n_sigma = 1.5
                        if (pop_threshold - mu_g) <= n_sigma * sigma_g:
                            raise ValueError("Crossing point too close to ground mean. Probably incorrect fitting, switching to fallback method.")

                    except Exception: # Use fallback method: using g-e SSF threshold to calculate Pg and Pe
                        # ----------Calculate g-e threshold for each ssf file ---------------------
                        ge_thresh, means, sigmas, weights, ground_idx, excited_idx = self.ssf_fit_two_gaussians_midpoint(ig_new, ie_new)
                        print(f"[run] Q{qid + 1} dataset {idx}: GMM fit failed or too close crossing. Falling back to g-e SSF threshold")
                        pop_threshold = ge_thresh
                        mask = (ig_new <= ge_thresh)
                        Pg = mask.mean()
                        Pe = 1.0 - Pg

                        # We don't care about these for this method, the user can check plots using function plot_ssf_ge_thresh if needed
                        ground_gaussian = ground_idx
                        excited_gaussian = excited_idx
                        ground_data = excited_data = None
                        used_fallback = True

                else:
                    # -------- Only using double-Gaussian fit on ground state data, without fallback method --------------------------
                    (Pg, Pe, gmm, means, sigmas, weights, threshold_mid, ground_gaussian, excited_gaussian,
                     ground_data, excited_data, _) = self.fit_double_gaussian_midpoint(ig_new)

                    pop_threshold = threshold_mid

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

                # -------- save qubit temps and timestamps ----------------------------------------------
                all_qubit_temperatures[qid].append(temp_k * 1e3)  # mK
                all_qubit_timestamps[qid].append(
                    datetime.datetime.fromtimestamp(ts_unix))
                fit_results[qid].append({
                    "dataset": idx,
                    "timestamp": datetime.datetime.fromtimestamp(ts_unix),
                    "temperature_mK": temp_k * 1e3,
                    "ig_new": ig_new,
                    "ground_data": ground_data,
                    "excited_data": excited_data,
                    "ground_gaussian": ground_gaussian,
                    "excited_gaussian": excited_gaussian,
                    "pop_threshold": pop_threshold,
                    "weights": weights,
                    "sigmas": sigmas,
                    "means": means,
                    "Pg": Pg,
                    "Pe": Pe,
                    "used_gessf_thresh_only": use_gessf_thresh_only, #True when the user decides to use this method
                    "used_fallback_method": used_fallback, #only True if it goes into effect, regardless of user decision
                })

        return all_qubit_temperatures, all_qubit_timestamps, fit_results

    # -------------------- NEW “g-e threshold only” runner -----------------
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
                           label=f"threshold = {thresh:.2f}")

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

    def plot_gaussians_qtemps(self, q_key, qubit_folder, ig_new, ground_data, excited_data, ground_gaussian, excited_gaussian, crossing_point, temperature_mk, dataset, weights, sigmas, means):
        # Note: crossing point is the threshold that is used to determine Pg and Pe.
        # Originally it was the crossing point between the two gaussians.
        # You can provide something else to be used as the threshold tho (such as the midpoint between the two gaussian means)

        # -----------------PLOTS TO CHECK FITS AND THRESHOLDS---------------
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
        ground_region = (bin_centers < crossing_point)
        excited_region = (bin_centers >= crossing_point)

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
        plt.axvline(crossing_point, color='black', linestyle='--', linewidth=1,
                    label=f'Threshold ({crossing_point:.2f})')

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
            f"SSF Histogram and Double Gaussian Fit ; Qubit {q_key + 1} ; Temp= {temperature_mk:2f} mK")
        plt.xlabel('Rot $I_g$' , fontsize=14)
        plt.ylabel('Counts', fontsize=14)
        plt.legend()
        # plt.show()

        # Save the plot to the Temperatures folder
        plot_filename = os.path.join(qubit_folder, f"Q{q_key + 1}_SSFhist_gaussianfit_Dataset{dataset}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
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
        qspec_files: dict[int, list[str]] OR list[str]
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

    #  Scatter – temperatures vs. time  (all dates, each qubit its own subplot)
    def plot_all_qubits_scatter(self, all_qubit_temperatures, all_qubit_timestamps, out_dir):
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(15, 10))
        date_fmt = DateFormatter('%m-%d-%H')

        for q in all_qubit_temperatures.keys():
            temps = all_qubit_temperatures[q]
            times = all_qubit_timestamps[q]
            if not temps:
                continue

            ax = plt.subplot(2, 3, q + 1)
            ax.scatter(times, temps,
                       color=colors[q], alpha=0.7, edgecolor='black',
                       label=f"Q{q + 1}")
            ax.set_title(f"Qubit {q + 1} Temperature vs Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Temperature (mK)")
            # ax.grid(alpha=0.3)
            ax.legend()
            ax.xaxis.set_major_formatter(date_fmt)
            plt.setp(ax.get_xticklabels(), rotation=45, fontsize=10)
            ax.set_ylim(50, 950)
            ax.set_yticks(np.linspace(50, 950, 10), fontsize=10)

        plt.tight_layout()
        fname = os.path.join(
            out_dir,
            f"AllQubits_Temps_vs_Time_{datetime.datetime.now():%Y%m%d%H%M%S}.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Saved all-dates scatter →", fname)

    # Histograms – temperature distributions  (all dates, each qubit subplot)
    def plot_all_qubits_hist(self, all_qubit_temperatures, out_dir, bins=20):
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

    # Temperature histograms   --------------------------------------------------
    def plot_temp_histograms(self, qubit_temperatures, out_dir, bins=20):
        """
        Parameters
        ----------
        qubit_temperatures : dict {qubit: [(temp_mK, unix_ts), …]}
        out_dir            : str   folder that will receive the PNG
        colors             : list  colour per qubit (defaults if None)
        bins               : int   histogram bins
        """
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(15, 10))
        for q, data in qubit_temperatures.items():
            temps = [t for t, _ in data]
            plt.subplot(2, 3, q + 1)
            plt.hist(temps, bins=bins, color=colors[q], alpha=0.7,
                     edgecolor='black')
            plt.title(f"Qubit {q + 1} Temperature Distribution")
            plt.xlabel("Temperature (mK)")
            plt.ylabel("Count")
            plt.grid(alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(
            out_dir,
            f"Temperature_Histograms_{datetime.datetime.now():%Y%m%d%H%M%S}.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Saved histogram →", fname)

    # Temperature-vs-time scatter --------------------------------------------
    def plot_temp_scatter(self, qubit_temperatures, out_dir):
        """
        Parameters
        ----------
        qubit_temperatures : dict {qubit: [(temp_mK, unix_ts), …]}
        out_dir            : str   folder that will receive the PNG
        colors             : list  color per qubit (defaults if None)
        """
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(15, 10))
        date_fmt = DateFormatter('%m-%d\n%H:%M')

        for q, data in qubit_temperatures.items():
            if not data:
                continue
            temps, ts = zip(*data)
            times = [datetime.datetime.fromtimestamp(t) for t in ts]

            ax = plt.subplot(2, 3, q + 1)
            ax.scatter(times, temps, color=colors[q], alpha=0.7, edgecolor='black')
            ax.set_title(f"Qubit {q + 1} Temperature vs Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Temperature (mK)")
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_formatter(date_fmt)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        fname = os.path.join(
            out_dir,
            f"Temperature_Scatter_{datetime.datetime.now():%Y%m%d%H%M%S}.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Saved scatter →", fname)

    def plot_threshold_split(self, q_key: int,rec: dict, out_folder: str):
        """
        Plot a simple histogram split at the SSF threshold.

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
        ig = rec["ig_new"]  # rotated SSF I values
        thresh = rec["pop_threshold"]  # data_threshold

        steps = 3000
        # numbins = round(math.sqrt(steps))
        numbins = 64

        fig, ax = plt.subplots()
        ax.hist(ig, bins=numbins, alpha=0.3, color="grey", label="all shots")
        ax.hist(ig[ig <= thresh], bins=numbins, alpha=0.7, label="|g⟩ data", color="blue")
        ax.hist(ig[ig > thresh], bins=numbins, alpha=0.7, label="|e⟩ leakage", color="red")
        ax.axvline(thresh, linestyle="--", color="black", label=f"threshold={thresh:.2f}")
        ax.set_title(f"Q{q_key + 1} SSF Threshold Split")
        ax.set_xlabel("I'")
        ax.set_ylabel("Counts")
        ax.legend()

        os.makedirs(out_folder, exist_ok=True)
        fname = os.path.join( out_folder, f"Q{q_key + 1}_threshold_split.png" )
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
          threshold_mid, ground_gaussian, excited_gaussian,
          ground_data, excited_data, iq_data
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

        labels = gmm.predict(iq_data.reshape(-1, 1))

        # split into ground vs excited (using midpoint of gaussian means as a threshold)
        ground_data = iq_data[(labels == ground_gaussian) & (iq_data <= threshold_mid)]
        excited_data = iq_data[(labels == excited_gaussian) & (iq_data > threshold_mid)]

        # calculate populations
        Pg = len(ground_data) / len(iq_data)
        Pe = len(excited_data) / len(iq_data)

        return Pg, Pe, gmm, means, sigmas, weights, threshold_mid, ground_gaussian, excited_gaussian, ground_data, excited_data, iq_data

    def ssf_fit_two_gaussians_midpoint(self, ig_new: np.ndarray, ie_new: np.ndarray):
        """
        Fits a two component GMM (double gaussian) to all shots (ig_new + ie_new) and chooses the
        threshold as the midpoint between the two component means.

        Returns
        -------
        thresh           : (μ_g + μ_e) / 2
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

        # Mid‑point threshold
        threshold = 0.5 * (mu_g + mu_e)

        return threshold, means, sigmas, weights, ground_idx, excited_idx

