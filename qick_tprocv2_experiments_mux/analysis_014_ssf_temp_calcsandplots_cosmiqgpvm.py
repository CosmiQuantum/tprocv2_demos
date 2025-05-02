from bisect import bisect_left
import re
import ast
import numpy as np
import h5py
from sklearn.mixture import GaussianMixture
from qicklab.analysis import qspec, t1, ssf
import math
import os
import datetime
import matplotlib.pyplot as plt
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


    def fit_double_gaussian_with_full_coverage(self, iq_data):
        gmm = GaussianMixture(n_components=2)
        gmm.fit(iq_data.reshape(-1, 1))

        means = gmm.means_.flatten()
        covariances = np.sqrt(gmm.covariances_).flatten()
        weights = gmm.weights_

        ground_gaussian = np.argmin(means)
        excited_gaussian = 1 - ground_gaussian

        # Generate x values to approximate the crossing point
        x_vals = np.linspace(means[ground_gaussian] - 3 * covariances[ground_gaussian],
                             means[excited_gaussian] + 3 * covariances[excited_gaussian], 1000)

        # Calculate Gaussian fits for each x value
        ground_gaussian_fit = weights[ground_gaussian] * (1 / (np.sqrt(2 * np.pi) * covariances[ground_gaussian])) * np.exp(
            -0.5 * ((x_vals - means[ground_gaussian]) / covariances[ground_gaussian]) ** 2)
        excited_gaussian_fit = weights[excited_gaussian] * (
                    1 / (np.sqrt(2 * np.pi) * covariances[excited_gaussian])) * np.exp(
            -0.5 * ((x_vals - means[excited_gaussian]) / covariances[excited_gaussian]) ** 2)

        # Find the x value where the two Gaussian functions are closest
        crossing_point = x_vals[np.argmin(np.abs(ground_gaussian_fit - excited_gaussian_fit))]

        labels = gmm.predict(iq_data.reshape(-1, 1))

        ground_data = iq_data[(labels == ground_gaussian) & (iq_data < crossing_point)]
        excited_data = iq_data[(labels == excited_gaussian) & (iq_data > crossing_point)]

        ground_state_population = len(ground_data) / len(iq_data)
        excited_state_population_overlap = len(excited_data) / len(iq_data)

        return ground_state_population, excited_state_population_overlap, gmm, means, covariances, weights, crossing_point, ground_gaussian, excited_gaussian, ground_data, excited_data, iq_data


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

    def run(self, pairs_info, limit_temp_k=0.8):
        """
        Parameters
        ----------
        pairs_info : dict
            {qubit: [ {"qspec":..., "ssf":..., "freq":<MHz>,
                        "Ig":<np.ndarray>, "ts":<unix-time> }, … ]}
        limit_temp_k : float
            Discard temperatures above this value (default 0.8 K → 800 mK).

        Returns
        -------
        all_qubit_temperatures : dict {qubit: [temp_mK, …]}
        all_qubit_timestamps   : dict {qubit: [datetime, …]}
        """
        # initialise output arrays
        all_qubit_temperatures = {i: [] for i in range(self.number_of_qubits)}
        all_qubit_timestamps = {i: [] for i in range(self.number_of_qubits)}

        for qid, records in pairs_info.items():  # loop over qubits
            for rec in records:  # …and every pair
                freq_mhz = rec["qfreq_MHz"]
                ig_new = rec["ig_new"]
                ts_unix = rec["data_timestamp"]

                # -------- double-Gaussian fit on ground state data --------------------------
                Pg, Pe, *_ = self.fit_double_gaussian_with_full_coverage(ig_new)
                temp_k = self.calculate_qubit_temperature(freq_mhz, Pg, Pe)

                # -------- screening -----------------------------------------
                if temp_k is None:
                    # un-physical (Pg/Pe ≤ 1) – skip
                    continue
                if temp_k > limit_temp_k:
                    print(f"[run]  Q{qid}: {temp_k * 1e3:.1f} mK  > {limit_temp_k * 1e3:.0f} mK  → dropped")
                    continue

                # -------- save qubit temps and timestamps ----------------------------------------------
                all_qubit_temperatures[qid].append(temp_k * 1e3)  # mK
                all_qubit_timestamps[qid].append(
                    datetime.datetime.fromtimestamp(ts_unix))

        return all_qubit_temperatures, all_qubit_timestamps

    def plot_gaussians_qtemps(self, q_key, qubit_folder, fidelity, ig_new, ground_data, excited_data, ground_gaussian, excited_gaussian, crossing_point, temperature_mk, dataset, weights, covariances, means):
        # -----------------PLOTS TO CHECK FITS AND THRESHOLDS---------------
        # Plotting double gaussian distributions and fitting
        xlims = [np.min(ig_new), np.max(ig_new)]
        plt.figure(figsize=(10, 6))

        # Plot histogram for `ig_new`
        steps = 3000
        numbins = round(math.sqrt(steps))
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
            (weights[ground_gaussian] / (np.sqrt(2 * np.pi) * covariances[ground_gaussian])) * np.exp(
                -0.5 * ((bin_centers[ground_region] - means[ground_gaussian]) / covariances[
                    ground_gaussian]) ** 2))

        scaling_factor_excited = max(n[excited_region]) / max(
            (weights[excited_gaussian] / (np.sqrt(2 * np.pi) * covariances[excited_gaussian])) * np.exp(
                -0.5 * ((bin_centers[excited_region] - means[excited_gaussian]) / covariances[
                    excited_gaussian]) ** 2))

        # Generate x values for plotting Gaussian components
        x = np.linspace(xlims[0], xlims[1], 1000)
        ground_gaussian_fit = scaling_factor_ground * (
                weights[ground_gaussian] / (np.sqrt(2 * np.pi) * covariances[ground_gaussian])) * np.exp(
            -0.5 * ((x - means[ground_gaussian]) / covariances[ground_gaussian]) ** 2)
        excited_gaussian_fit = scaling_factor_excited * (
                weights[excited_gaussian] / (np.sqrt(2 * np.pi) * covariances[excited_gaussian])) * np.exp(
            -0.5 * ((x - means[excited_gaussian]) / covariances[excited_gaussian]) ** 2)

        plt.plot(x, ground_gaussian_fit, label='Ground Gaussian Fit', color='blue', linewidth=2)
        plt.plot(x, excited_gaussian_fit, label='Excited (leakage) Gaussian Fit', color='red', linewidth=2)
        plt.axvline(crossing_point, color='black', linestyle='--', linewidth=1,
                    label=f'Crossing Point ({crossing_point:.2f})')

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
            f"Fidelity Histogram and Double Gaussian Fit ; Qubit {q_key + 1}; Fidelity = {fidelity * 100:.2f}% ; Temp= {temperature_mk:2f} mK")
        plt.xlabel('$I_g$', fontsize=14)
        # plt.ylabel('Probability Density', fontsize=14) or is it counts? i think it might just be counts
        plt.legend()
        # plt.show()

        # Save the plot to the Temperatures folder
        plot_filename = os.path.join(qubit_folder,
                                     f"Qubit{q_key + 1}_Fidelityhist_gaussianfit_Dataset{dataset}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        plt.savefig(plot_filename)
        # print(f"Plot saved to {plot_filename}")
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

    # -------------------------------- helper to discover qubit index
    def qubit_of(self, h5_path):
        with h5py.File(h5_path, "r") as h5:
            k = next(g for g in h5.keys() if g.isdigit())  # first top-level group
            return int(k)

    def pair_qspec_and_ssf(self, qspec_files, ssf_files, tolerance_seconds=10):
        """
        Return
        -------
        pairs_by_qubit   dict {qubit_index: [(qspec_path, ssf_path), …]}
        unmatched_qspec  dict {qubit_index: [qspec_path, …]}
        unmatched_ssf    dict {qubit_index: [ssf_path,   …]}
        """
        #This is currently the case
        if isinstance(qspec_files, dict):
            qspec_files = [f for lst in qspec_files.values() for f in lst]
        if isinstance(ssf_files, dict):
            ssf_files = [f for lst in ssf_files.values() for f in lst]

        # -------------------------------- helper to discover qubit index
        def qubit_of(h5_path):
            with h5py.File(h5_path, "r") as h5:
                k = next(g for g in h5.keys() if g.isdigit())  # first top-level group
                return int(k)

        # -------------------------------- bucket files by qubit index
        qspec_by_q = {}
        for f in qspec_files:
            qi = qubit_of(f)
            qspec_by_q.setdefault(qi, []).append(f)

        ssf_by_q = {}
        for f in ssf_files:
            qi = qubit_of(f)
            ssf_by_q.setdefault(qi, []).append(f)

        # -------------------------------- pair inside each bucket
        pairs_by_qubit = {}
        unmatched_qspec = {}
        unmatched_ssf = {}

        for qi in qspec_by_q.keys() | ssf_by_q.keys():  # union of keys
            qspec = sorted((self.timestamp(f), f) for f in qspec_by_q.get(qi, []))
            ssf = sorted((self.timestamp(f), f) for f in ssf_by_q.get(qi, []))

            ssf_times = [t for t, _ in ssf]
            free_ssf = {f for _, f in ssf}

            pairs, lonely_q = [], []

            for tq, fq in qspec:
                i = bisect_left(ssf_times, tq)
                cands = []
                if i < len(ssf): cands.append(ssf[i])
                if i:            cands.append(ssf[i - 1])

                best = None
                for ts, fs in cands:
                    if abs((ts - tq).total_seconds()) <= tolerance_seconds:
                        if best is None or abs(ts - tq) < abs(best[0] - tq):
                            best = (ts, fs)

                if best and best[1] in free_ssf:
                    pairs.append((fq, best[1]))
                    free_ssf.remove(best[1])
                else:
                    lonely_q.append(fq)

            pairs_by_qubit[qi] = pairs
            unmatched_qspec[qi] = lonely_q
            unmatched_ssf[qi] = list(free_ssf)

        return pairs_by_qubit, unmatched_qspec, unmatched_ssf

    #  Scatter – temperatures vs. time  (all dates, each qubit its own subplot)
    def plot_all_qubits_scatter(self, all_qubit_temperatures, all_qubit_timestamps, out_dir):
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(15, 10))
        date_fmt = DateFormatter('%m-%d\n%H:%M')

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
            ax.grid(alpha=0.3)
            ax.legend()
            ax.xaxis.set_major_formatter(date_fmt)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

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
