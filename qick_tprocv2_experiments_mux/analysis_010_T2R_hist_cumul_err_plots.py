import numpy as np
import os
import sys
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
import glob
import re
from scipy.signal import find_peaks
import datetime
import ast
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import json
import h5py
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

class T2rHistCumulErrPlots:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, base_data_path, plots_path, run_name, fridge):
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.figure_quality = figure_quality
        self.base_data_path = base_data_path
        self.run_name = run_name
        self.plots_path = plots_path
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

    def flat_vs_ramsey_bic(self, y, fitted, k_fit=6, k0=1, threshold=12):
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

    def run(self, t1_vals = None):
        import datetime
        # ----------Load/get data from T2R------------------------
        t2r_vals = {i: [] for i in range(self.number_of_qubits)}
        t2r_errs = {i: [] for i in range(self.number_of_qubits)}
        qubit_for_this_index = []
        rounds = []
        reps = []
        file_names = []
        dates = {i: [] for i in range(self.number_of_qubits)}

        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                # Build full paths using the provided base_path and folder_date
                outerFolder = os.path.join(self.base_data_path, folder_date, "study_data")
                self.create_folder_if_not_exists(outerFolder)

                outerFolder_save_plots = os.path.join(self.base_data_path, "benchmark_analysis_plots",
                                                      f"{folder_date}_RRplots")
                self.create_folder_if_not_exists(outerFolder_save_plots)
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            outerFolder_expt = outerFolder + "/Data_h5/t2_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type='t2_ge', save_r=int(save_round))

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

                        exp_config = load_data['t2_ge'][q_key].get('Exp Config', [])[0][dataset].decode()
                        safe_globals = {"np": np, "array": np.array, "__builtins__": {}}

                        exp_config = eval(exp_config, safe_globals)

                        if len(I) > 0:
                            T2_class_instance = T2RMeasurement(q_key, self.number_of_qubits,
                                                               self.plots_path, round_num, self.signal,
                                                               self.save_figs, fit_data=True)
                            try:
                                fitted, t2r_est, t2r_err, plot_sig, out = T2_class_instance.t2_fit_iminuit(delay_times,
                                                                                                           I, Q)
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
                                    print(
                                        f'Rejected a T2R scan. Failed ramsey shape, less than {min_peaks} oscillations.')
                                    continue
                            except Exception:
                                # if peak counting fails for any reason, be conservative and skip
                                continue

                            # -------------------- flat baseline vs Ramsey shape BIC test -------------------------------
                            y = I if plot_sig == "I" else Q

                            keep_ramsey, delta_bic = self.flat_vs_ramsey_bic(y, fitted, k_fit=6, k0=1, threshold=35)

                            if not keep_ramsey:
                                print(f"Rejected by BIC: ΔBIC = {delta_bic:.2f}")
                                continue

                            # ---------------- Exponential vs Ramsey BIC test ----------------
                            y = I if plot_sig == "I" else Q

                            keep_ramsey, delta_bic_exp = self.exp_vs_ramsey_bic(
                                delay_times, y, fitted, k_fit=6, k_exp=3, threshold=10)

                            if not keep_ramsey:
                                print(f"Rejected by exp-BIC: ΔBIC(exp−Ramsey) = {delta_bic_exp:.2f}")
                                continue
                            # ---------------------------------------------------------------

                            if t2r_est < 0:
                                print("The value is negative, continuing...")
                                continue
                            if t1_vals is not None:
                                max_t1 = max(t1_vals[q_key])  # theoretical value
                                if t2r_est > 2 * max_t1:
                                    print(f"The value is above 2*{max_t1} us, this is a bad fit, continuing...")
                                    continue
                            t2r_vals[q_key].extend([t2r_est])  # Store T2 values
                            t2r_errs[q_key].extend([t2r_err])  # Store T2 error values
                            dates[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])  # Decode bytes to string

                            del T2_class_instance

                del H5_class_instance
        return dates, t2r_vals, t2r_errs

    def plot(self, dates, t2r_vals, t2r_errs, show_legends):
        #---------------------------------plot-----------------------------------------------------
        analysis_folder = os.path.join(self.plots_path, "benchmark_analysis_plots")
        self.create_folder_if_not_exists(analysis_folder)

        analysis_folder = os.path.join(self.plots_path, "benchmark_analysis_plots", "t2_ge")
        self.create_folder_if_not_exists(analysis_folder)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        font = 14
        titles = [f"Q{i+1}" for i in range(self.number_of_qubits)]
        gaussian_xvals =  {i: [] for i in range(0, self.number_of_qubits)}
        gaussian_yvals =  {i: [] for i in range(0, self.number_of_qubits)}
        gaussian_colors = {i: [] for i in range(0, self.number_of_qubits)}
        gaussian_dates = {i: [] for i in range(0, self.number_of_qubits)}
        mean_values = {}
        std_values = {}
        colors = ['orange','blue','purple','green','brown','pink']
        for i, ax in enumerate(axes):
            if len(dates[i])>1:
                date_label = dates[i][0]
            else:
                date_label = ''


            if len(t2r_vals[i]) >1:
                optimal_bin_num = 45 #self.optimal_bins(t2r_vals[i])

                # Fit a Gaussian to the raw data instead of the histogram
                # get the mean and standard deviation of the data
                # mu_1, std_1 = norm.fit(t2r_vals[i])

                # # NEW: WEIGHTED MEANS -------------------------------------------------
                # # Weighted Gaussian fit (using inverse-variance weights), per-qubit i
                # t2s = np.asarray(t2r_vals[i], dtype=float)
                # errs = np.asarray(t2r_errs[i], dtype=float)
                #
                # # avoiding infinite weights and NaN pollution
                # err_floor = 1e-12
                # safe_errs = np.clip(errs, err_floor, np.inf)
                # # weights = 1.0 / (safe_errs ** 2)
                # weights = 1.0 / (safe_errs)
                #
                # w_sum = np.nansum(weights)
                # mu_1 = float(np.nansum(weights * t2s) / w_sum)
                # var = float(np.nansum(weights * (t2s - mu_1) ** 2) / w_sum)
                # std_1 = float(np.sqrt(max(var, 0.0)))
                # # ---------------------------------------------------------------------
                # --- Weighted mean with robust median-MAD clipping ------------------------
                t2rs = np.asarray(t2r_vals[i], dtype=float)
                errs = np.asarray(t2r_errs[i], dtype=float)
                n_counts = len(t2rs)

                # keep only finite pairs
                finite = np.isfinite(t2rs) & np.isfinite(errs)
                t2rs, errs = t2rs[finite], errs[finite]
                if t2rs.size == 0:
                    mu_1, std_1 = np.nan, np.nan
                else:
                    # robust outlier clip around the median (tune k if you like)
                    k = 2.0  # 2-4 is typical. 2 is stricter
                    med = np.median(t2rs)
                    mad = np.median(np.abs(t2rs - med))
                    # fallback if MAD is zero (all equal or super-tight); use small epsilon
                    if mad == 0:
                        mad = max(np.std(t2rs), 1e-12)
                    keep = np.abs(t2rs - med) < k * mad

                    t2rs, errs = t2rs[keep], errs[keep]

                    if t2rs.size == 0:
                        mu_1, std_1 = np.nan, np.nan
                    else:
                        # compute weights and weighted mean/std
                        err_floor = 1e-12
                        safe_errs = np.clip(errs, err_floor, np.inf)

                        # can also do 1/sigma^2
                        weights = 1.0 / safe_errs

                        w_sum = np.nansum(weights)
                        mu_1 = float(np.nansum(weights * t2rs) / w_sum)

                        # weighted variance (with chosen weights convention)
                        var = float(np.nansum(weights * (t2rs - mu_1) ** 2) / w_sum)
                        std_1 = float(np.sqrt(max(var, 0.0)))
                # --------------------------------------------------------------------------

                mean_values[f"Qubit {i + 1}"] = mu_1  # Store the mean value for each qubit
                std_values[f"Qubit {i + 1}"] = std_1

                # Generate x values for plotting a gaussian based on this mean and standard deviation
                x_1 = np.linspace(min(t2r_vals[i]), max(t2r_vals[i]), optimal_bin_num)
                p_1 = norm.pdf(x_1, mu_1, std_1)

                # Calculate histogram data for t1_vals[i]
                hist_data_1, bins_1 = np.histogram(t2r_vals[i], bins=optimal_bin_num)
                bin_centers_1 = (bins_1[:-1] + bins_1[1:]) / 2

                # Scale the Gaussian curve to match the histogram
                # the gaussian height natrually doesnt match the bin heights in the histograms
                # np.diff(bins_1)  calculates the width of each bin by taking the difference between bin edges
                # the total counts are in hist_data_1.sum()
                # to scale, multiply data gaussian by bin width to convert the probability density to probability within each bin
                # then multiply by the total count to scale the probability to match the overall number of datapoints
                # https://mathematica.stackexchange.com/questions/262314/fit-function-to-histogram
                # https://stackoverflow.com/questions/23447262/fitting-a-gaussian-to-a-histogram-with-matplotlib-and-numpy-wrong-y-scaling
                #ax.plot(x_1, p_1 * (np.diff(bins_1) * hist_data_1.sum()), 'b--', linewidth=2, color='black') # old way

                bin_width_1 = np.diff(bins_1)[0]  # scalar: average width of each bin
                N_counts = hist_data_1.sum()  # total number of points
                scaled_pdf = p_1 * (bin_width_1 * N_counts)

                ax.plot(x_1, scaled_pdf, 'b--', linewidth=2, color='black')

                # additional scaling option----------------------------------------------------
                # Curve is peak-matched to the histogram's tallest bin:
                # peak_hist = np.max(hist_data_1) if hist_data_1.size else 0.0
                # peak_pdf = np.max(p_1) if p_1.size else 0.0
                # scale_factor = (peak_hist / peak_pdf) if peak_pdf > 0 else 1.0
                # scaled_pdf = p_1 * scale_factor
                #--------------------------------------------------------------------------------------------

                # Plot histogram and Gaussian fit for t1_vals[i]
                ax.hist(t2r_vals[i], bins=optimal_bin_num, alpha=0.7, color=colors[i], edgecolor='black', label=date_label)

                #make a fuller gaussian to make smoother lotting for cumulative plot
                x_1_full = np.linspace(min(t2r_vals[i]), max(t2r_vals[i]), 2000)
                p_1_full = norm.pdf(x_1_full, mu_1, std_1)

                gaussian_xvals[i].append(x_1_full)
                gaussian_yvals[i].append(p_1_full )
                gaussian_colors[i].append(colors[i])
                gaussian_dates[i].append(date_label)

                #rough start at errors:
                #counts, bin_edges, _ = ax.hist(t1_vals[i], bins=20, alpha=0.7, color='blue', edgecolor='black')
                #bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                #bin_errors = [np.sqrt(np.sum(t1_errs[i])) for _ in range(len(bin_centers))]  #using error propogation through the sum of valuse in each bin
                #ax.errorbar(bin_centers, counts, yerr=bin_errors, fmt='o', color='red', ecolor='black', capsize=3, linestyle='None')
                if show_legends:
                    ax.legend()
                ax.set_title(titles[i] + f"Weighted $\mu$: {mu_1:.2f} $\sigma$:{std_1:.2f}, c: {n_counts}",fontsize = font)
                ax.set_xlabel('T2R (µs)',fontsize = font)
                ax.set_ylabel('Frequency',fontsize = font)
                ax.tick_params(axis='both', which='major', labelsize=font)

        plt.tight_layout()
        plt.savefig( analysis_folder + 'hists.pdf', transparent=True, dpi=self.final_figure_quality)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        plt.title('Cumulative Distribution',fontsize = font)
        for i in range(0, len(t2r_vals)):
            if len(dates[i])>1:
                date_label = dates[i][0]
            else:
                date_label = ''

            if len(t2r_vals[i]) > 1:
                t1_vals_sorted = np.sort(t2r_vals[i])
                len_samples = len(t1_vals_sorted)
                var = np.linspace(1,len_samples,len_samples)/ len_samples

                cumulative_gaussian = np.cumsum(gaussian_yvals[i][0]) / np.sum(gaussian_yvals[i][0])
                ax.scatter(t1_vals_sorted,var,color = colors[i], label = f'Q{i+1}', s = 5)
                ax.plot(gaussian_xvals[i][0], cumulative_gaussian, color=colors[i], label='Gauss Fit ' + f'Q{i + 1}',
                        linestyle='--')
                ax.tick_params(axis='both', which='major', labelsize=font)
        #ax.set_title('')
        ax.set_xlabel('T2R (us)',fontsize = font)
        ax.set_ylabel('Cumulative Distribution',fontsize = font)
        ax.loglog()
        ax.legend(edgecolor='black')
        #ax.set_xlim(10**0, 10**3)
        #ax.set_ylim(10 ** -7, 10 ** 0) #to compare to johns plot, need to adjust a little
        plt.tight_layout()
        plt.savefig(analysis_folder + 'cumulative.pdf', transparent=True, dpi=self.final_figure_quality)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('Fit Error vs T2R Time',fontsize = font)
        axes = axes.flatten()
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        for i, ax in enumerate(axes):

            if len(dates[i])>1:
                date_label = dates[i][0]
            else:
                date_label = ''
            ax.set_title(titles[i], fontsize = font)
            ax.scatter(t2r_vals[i], t2r_errs[i], label = date_label, color = colors[i])
            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('T2R (us)', fontsize = font)
            ax.set_ylabel('Fit error (us)', fontsize = font)
            ax.tick_params(axis='both', which='major', labelsize=font)
        plt.tight_layout()
        plt.savefig(analysis_folder + 'errs.pdf', transparent=True, dpi=self.final_figure_quality)
        #plt.show()
        print('Plots saved to: ', analysis_folder)

        return std_values, mean_values

