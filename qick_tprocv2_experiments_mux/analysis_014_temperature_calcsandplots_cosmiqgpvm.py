from section_008_save_data_to_h5 import Data_H5
import re
import datetime as dt
from bisect import bisect_left
import glob
import re
import datetime
import ast
import os
import sys
from matplotlib.dates import DateFormatter
import numpy as np
import h5py
from sklearn.mixture import GaussianMixture
from qicklab.analysis import qspec, t1, ssf
import matplotlib.pyplot as plt
import math

save_figs = True
fit_saved = False
signal = 'None'
figure_quality = 100 #ramp this up to like 500 for presentation plots


class TempCalcAndPlots:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, outerFolder, fridge):

        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.figure_quality = figure_quality
        self.run_name = run_name
        self.number_of_qubits = number_of_qubits
        self.final_figure_quality = final_figure_quality
        self.top_folder_dates = top_folder_dates
        self.outerFolder = outerFolder
        self.temperature_folder = os.path.join(self.outerFolder, "Temperatures")
        self.fridge = fridge

        # Create the folder if it doesn't exist
        if not os.path.exists(self.temperature_folder):
            os.makedirs(self.temperature_folder)

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

    def run(self, pairs_by_qubit, science_qubits=(0, 4), tolerance_seconds=10):
        """
        Parameters
        ----------
        pairs_by_qubit : dict {QubitIndex : [(qspec_h5, ssf_h5), …]}
        science_qubits : iterable[int]      which qubits to analyse
        tolerance_seconds : int             (kept only for completeness)

        Returns
        -------
        all_qubit_temps      dict {QubitIndex : [temp_mK,   …]}
        all_qubit_timestamps dict {QubitIndex : [unix_time, …]}
        """

        all_qubit_temps = {q: [] for q in science_qubits}
        all_qubit_timestamps = {q: [] for q in science_qubits}

        # helper to split a full file path → (parent_path, dataset)
        def split_paths(h5_path):
            ts_dir = os.path.dirname(os.path.dirname(os.path.dirname(h5_path)))
            return os.path.dirname(ts_dir), os.path.basename(ts_dir)

        for QubitIndex in science_qubits:

            for qspec_h5, ssf_h5 in pairs_by_qubit.get(QubitIndex, []):

                parent_path, dataset = split_paths(qspec_h5)  # same for SSF

                # --------------------- QSpec : get qubit_frequency --------------------
                try:
                    qspec_obj = qspec(parent_path, dataset, QubitIndex)
                    qs_dates, qs_n, qs_probe_f, qs_I, qs_Q = qspec_obj.load_all()
                    qs_freqs, *_ = qspec_obj.get_all_qspec_freq(qs_probe_f, qs_I, qs_Q, qs_n)

                    file_ts_qspec = self.timestamp(qspec_h5).timestamp() # extracts the time stamp embedded in the file name
                    idx_qspec = int(np.argmin(np.abs(np.array(qs_dates) -
                                                     file_ts_qspec)))
                    qubit_frequency = float(qs_freqs[idx_qspec])  # [MHz]
                except Exception as e:
                    print(f"[run] QSpec load/fit failed (Q{QubitIndex}) → {e}")
                    continue

                # --------------------- SSF : get Ig slice -----------------------------
                try:
                    ssf_obj = ssf(parent_path, dataset, QubitIndex)
                    ss_dates, ss_n, I_g, Q_g, I_e, Q_e, fid, angles = ssf_obj.load_all()

                    file_ts_ssf = self.timestamp(ssf_h5).timestamp()
                    idx_ssf = int(np.argmin(np.abs(np.array(ss_dates) -
                                                   file_ts_ssf)))
                    ig_new = np.array(I_g[idx_ssf])  # 1-D Ig samples
                except Exception as e:
                    print(f"[run] SSF load failed (Q{QubitIndex}) → {e}")
                    continue

                # --------------------- Double-Gaussian fit -----------------------------
                try:
                    (ground_state_population,
                     excited_state_population_overlap,
                     gmm, means, covariances, weights,
                     crossing_point,
                     ground_gaussian, excited_gaussian,
                     ground_data, excited_data,
                     iq_data) = self.fit_double_gaussian_with_full_coverage(ig_new)
                except Exception as e:
                    print(f"[run] Gaussian fit failed (Q{QubitIndex}) → {e}")
                    continue

                # --------------------- Temperature calculation -------------------------
                temperature_k = self.calculate_qubit_temperature(
                    qubit_frequency,
                    ground_state_population,
                    excited_state_population_overlap)

                if temperature_k is None:
                    continue  # skip unphysical or bad dataset

                all_qubit_temps[QubitIndex].append(temperature_k * 1e3)  # K→mK
                all_qubit_timestamps[QubitIndex].append(ss_dates[idx_ssf])  # unix

        return all_qubit_temps, all_qubit_timestamps

    def ran(self, QubitIndex):
        q_key = QubitIndex
        all_qubit_temperatures = {i: [] for i in range(self.number_of_qubits)}
        all_qubit_timestamps = {i: [] for i in range(self.number_of_qubits)}
        # ----------------------------------------------Load/Plot/Save QSpec------------------------------------
        for date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"/data/QICK_data/{self.run_name}/" + date + "/"
                outerFolder_save_plots = f"/data/QICK_data/{self.run_name}/" + date + "_plots/"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # Update self.temperature_folder for the current date
            self.temperature_folder = os.path.join(outerFolder, "Temperatures")
            if not os.path.exists(self.temperature_folder):
                os.makedirs(self.temperature_folder)

            outerFolder_expt = outerFolder + "/Data_h5/QSpec_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

            qubit_frequencies = []

            for h5_file in h5_files:
                #print(h5_file)
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=  'QSpec', save_r = int(save_round))

                # Define the time frame to exclude
                # exclude_start = datetime.datetime(2025, 1, 26)  # Start date (inclusive)
                # exclude_end = datetime.datetime(2025, 1, 27)  # End date (inclusive)

                # Extract the relevant portion of the file name for matching
                base_h5_file = "_".join(h5_file.split('/')[-1].split('_')[:2])  # Extract up to 2024-12-11_11-45-27
                #print(f"Base H5 File for Matching: {base_h5_file}")

                # Convert current file's timestamp to a datetime object
                current_timestamp = datetime.datetime.strptime(base_h5_file, "%Y-%m-%d_%H-%M-%S")

                #new way (matches time stamps in the file names within 10 seconds of eachother)
                qubit_frequency = [
                    entry['largest_amp_curve_mean']
                    for entry in qubit_frequencies
                    if entry['q_key'] == q_key and abs(
                        (datetime.datetime.strptime(
                            "_".join(entry['h5_file'].split('/')[-1].split('_')[:2]), "%Y-%m-%d_%H-%M-%S") - current_timestamp).total_seconds()) <= 10]

                if len(qubit_frequency) == 0:
                    print(f"No match found for h5_file: {base_h5_file}, q_key: {q_key}. Skipping.")
                    continue

                qubit_frequency = qubit_frequency[0] #there should only be one value inside this list
                #print('qubit_frequency is: ', qubit_frequency)
                ground_state_population, excited_state_population_overlap, gmm, means, covariances, weights, crossing_point, ground_gaussian, excited_gaussian, ground_data, excited_data, iq_data = self.fit_double_gaussian_with_full_coverage(ig_new)
                temperature_k = self.calculate_qubit_temperature(qubit_frequency, ground_state_population,
                                                            excited_state_population_overlap)

                limit_temp = 0.5 #kelvin, equivalent to 500mK
                if temperature_k is not None and temperature_k <= limit_temp:
                    temperature_mk = temperature_k * 1e3
                    # print(f"Ground state population: {ground_state_population}")
                    # print(f"Excited state (leakage) population: {excited_state_population_overlap}")
                    # print(f"Qubit {q_key + 1} Temperature: {temperature_mk:.2f} mK", "\n")
                    qubit_temperatures[q_key].append((temperature_mk, timestamp))

                    # temperature_mk = temperature_k * 1e3
                    # print(f"Ground state population: {ground_state_population}")
                    # print(f"Excited state (leakage) population: {excited_state_population_overlap}")
                    # print(f"Qubit {q_key + 1} Temperature: {temperature_mk:.2f} mK", "\n")
                    # qubit_temperatures[q_key].append((temperature_mk, timestamp))#save temps for each qubit

                else:
                    # Distinguish between unphysical value or out-of-range value
                    if temperature_k is None:
                        print(f"Warning: Unphysical temperature for Qubit {q_key + 1}. Skipping.")
                        pass # Skip this dataset
                    else:
                        temperature_mk = temperature_k * 1e3
                        limit_temp_mk = limit_temp * 1e3  # Convert K to mK
                        print(f"Warning: Temperature {temperature_mk:.2f} mK exceeds {limit_temp_mk} mK for Qubit {q_key + 1}. Skipping this dataset.")
                        pass  # Skip this dataset

                del H5_class_instance

            #Saving data for all days
            for q_id in range(self.number_of_qubits):
                for (temp_mk, ts) in qubit_temperatures[q_id]:
                    all_qubit_temperatures[q_id].append(temp_mk)
                    all_qubit_timestamps[q_id].append(datetime.datetime.fromtimestamp(ts)) # Convert timestamp to datetime for easier plotting later

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
        return dt.datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")

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