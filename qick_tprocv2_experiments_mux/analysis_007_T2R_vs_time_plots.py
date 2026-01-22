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
#from expt_config import *
import glob
import re
import datetime
import ast
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

class T2rVsTime:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, fridge):
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

    def run(self,return_errs=False, t1_vals = None):
        import datetime
        # ----------Load/get data------------------------
        t2_vals = {i: [] for i in range(self.number_of_qubits)}
        t2_errs = {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}

        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"/data/QICK_data/{self.run_name}/" + folder_date + "/study_data/"
                outerFolder_save_plots = f"/data/QICK_data/{self.run_name}/" + folder_date + "/documentation/"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # -------------------------------------------------------Load/Plot/Save T2------------------------------------------
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
                                                               self.outerFolder_save_plots, round_num, self.signal,
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
                            t2_vals[q_key].extend([t2r_est])
                            t2_errs[q_key].extend([t2r_err])
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del T2_class_instance

                del H5_class_instance
        if return_errs:
            return date_times, t2_vals, t2_errs
        else:
            return date_times, t2_vals

    def plot_without_errs(self, date_times, t2_vals, show_legends):
        # ---------------------------------plot-----------------------------------------------------
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
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

    def plot_with_errs(self, date_times, t2_vals, t2_fit_err, show_legends):
        # ---------------------------------plot-----------------------------------------------------
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.suptitle('T2R Values vs Time', fontsize=font)
        axes = axes.flatten()

        from datetime import datetime
        import matplotlib.dates as mdates

        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]
            y = t2_vals[i]
            err = t2_fit_err[i]

            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])
            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)

            # ax.set_ylim(7, 95)

            ax.errorbar(
                sorted_x, sorted_y, yerr=sorted_err,
                fmt='none',  #no marker
                ecolor=colors[i],
                elinewidth=1,
                capsize=0
            )
            ax.scatter(
                sorted_x, sorted_y,
                s=10,
                color=colors[i],
                alpha=0.5
            )

            #ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.tick_params(axis='x', rotation=45)

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel('T2R (us)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2R_vals.pdf', transparent=True, dpi=self.final_figure_quality)
        print('Plot saved to:', analysis_folder)
        plt.close()

    def plot_with_errs_single_plot(self, date_times, t2_vals, t2_fit_err, show_legends):
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
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
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]
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
