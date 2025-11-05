import numpy as np
import os
import sys
from tprocv2_demos.qick_tprocv2_experiments_mux.socProxy import makeProxy
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
import glob
from collections import OrderedDict
import re
import datetime
import ast
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

class T1HistCumulErrPlots:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, base_data_path, plots_path, run_name, run_notes, run_number, fridge):
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.figure_quality = figure_quality
        self.base_data_path = base_data_path
        self.number_of_qubits = number_of_qubits
        self.run_name = run_name
        self.plots_path = plots_path
        self.final_figure_quality = final_figure_quality
        self.top_folder_dates = top_folder_dates
        self.run_notes = run_notes
        self.run_number = run_number
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

    def run(self,exp_extension='', saved_shots = False):
        # ----------Load/get data from T1------------------------
        t1_vals = {i: [] for i in range(self.number_of_qubits)}
        t1_errs = {i: [] for i in range(self.number_of_qubits)}
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

                outerFolder_save_plots = os.path.join(self.base_data_path, "benchmark_analysis_plots", f"{folder_date}_RRplots")
                self.create_folder_if_not_exists(outerFolder_save_plots)
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/t1{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/t1_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            soc, soccfg = makeProxy()
            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=  f't1{exp_extension}', save_r = int(save_round))

                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26), #power outage
                    datetime.date(2025, 1, 29), #HEMT Issues
                    datetime.date(2025, 1, 30), #HEMT Issues
                    datetime.date(2025, 1, 31)  #Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f't1{exp_extension}']:
                    for dataset in range(len(load_data[f't1{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f't1{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T1_est_from_h5s= load_data['T1'][q_key].get('T1', [])[0][dataset]
                        # errors_from_h5s = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                        date= datetime.datetime.fromtimestamp(load_data[f't1{exp_extension}'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        # --- NEW: make per-shot data compatible with per-delay fitting --------------------------------
                        if saved_shots:
                            exp_config_str = load_data['t1_ge'][q_key]['Exp Config'][0][dataset].decode()
                            syst_config_str = load_data['t1_ge'][q_key]['Syst Config'][0][dataset].decode()

                            Ishots_raw = self.process_h5_data(load_data['t1_ge'][q_key]['I'][0][dataset].decode())
                            Qshots_raw = self.process_h5_data(load_data['t1_ge'][q_key]['Q'][0][dataset].decode())

                            replica = OfflineAcquireReplica(remove_offset=True, length_norm=True)
                            replica.setup_offline_from_strings(exp_config_str, syst_config_str, soccfg,
                                                               qubit_index=int(q_key))

                            # Authoritative dims from EXP config
                            exp_cfg = replica._safe_eval_cfg(exp_config_str)
                            steps = int(exp_cfg['T1_ge']['steps'])
                            reps = int(exp_cfg['T1_ge']['reps'])
                            # rounds  = int(exp_cfg['T1_ge']['rounds'])   # not used here; H5 holds one round

                            Ishots = replica.coerce_to_rounds_N_reps(Ishots_raw, steps, reps)
                            Qshots = replica.coerce_to_rounds_N_reps(Qshots_raw, steps, reps)

                            # We only provided ONE rounds worth of raw shots from H5 -> tell the replica that
                            I, Q = replica.acquire_offline(Ishots, Qshots, soft_avgs=1) # we have only done 1 round and a bunch of reps
                        # -----------------------------------------------------------------------------------------------
                        else:
                            I = self.process_h5_data(load_data['t1_ge'][q_key].get('I', [])[0][dataset].decode())
                            Q = self.process_h5_data(load_data['t1_ge'][q_key].get('Q', [])[0][dataset].decode())

                        delay_times = self.process_h5_data(load_data[f't1{exp_extension}'][q_key].get('Delay Times', [])[0][dataset].decode())
                        #fit = load_data['T1'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data[f't1{exp_extension}'][q_key].get('Round Num', [])[0][dataset]
                        batch_num = load_data[f't1{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]
                        try:
                            exp_config = load_data[f't1{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I)>0:
                            T1_class_instance = T1Measurement(q_key, self.number_of_qubits, outerFolder_save_plots, round_num, self.signal,
                                                              self.save_figs, fit_data = True)
                            #T1_spec_cfg = exp_config['T1_ge']
                            try:
                                q1_fit_exponential, T1_err, T1, plot_sig = T1_class_instance.t1_fit(I, Q, delay_times)
                            except Exception as e:
                                print('Fit didnt work due to error: ', e)
                                continue
                            if T1 < 0:
                                print("The value is negative, continuing...")
                                continue
                            if T1 > 1000:
                                print("The value is above 750us us, this is a bad fit, continuing...")
                                continue
                            # if T1_err >= 0.8 * T1:
                            #     print(
                            #         f"Skipping T1 = {T1:.3f} µs because its error {T1_err:.3f} µs is >= 80% of its value.")
                            #     continue

                            if (self.run_number == 8) and (q_key != 5) and (T1 <= 22): # # QUIET run 8 patch while fitting is fixed
                                print(
                                    f"Skipping T1 = {T1:.3f} µs for Q{q_key + 1} because it is presumed to be a bad fit (Run 8 patch).")
                                continue

                            t1_vals[q_key].extend([T1])  # Store T1 values
                            t1_errs[q_key].extend([T1_err])  # Store T1 error values
                            dates[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])  # Decode bytes to string

                            # You can also append qubit indices if needed
                            qubit_for_this_index.extend([q_key])
                            del T1_class_instance

                del H5_class_instance
        return dates, t1_vals, t1_errs

    def plot(self, dates, t1_vals, t1_errs, show_legends,exp_extension=''):
        #---------------------------------plot-----------------------------------------------------
        if self.fridge.upper() == 'QUIET':
            analysis_folder = os.path.join(self.plots_path, "benchmark_analysis_plots")
            self.create_folder_if_not_exists(analysis_folder)

            analysis_folder = os.path.join(self.plots_path, "benchmark_analysis_plots", "t1_ge")
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/T1/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        ext = exp_extension.replace('_', '')
        font = 14
        plt.suptitle(f'{ext} T1 Values Binned', fontsize=font)
        titles = [f"Q{i+1}" for i in range(self.number_of_qubits)]
        gaussian_xvals =  {i: [] for i in range(self.number_of_qubits)}
        gaussian_yvals =  {i: [] for i in range(self.number_of_qubits)}
        gaussian_colors = {i: [] for i in range(self.number_of_qubits)}
        gaussian_dates = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        std_values = {}
        colors = ['orange','blue','purple','green','brown','pink']
        for i, ax in enumerate(axes):
            # Check if the key exists in all dictionaries before accessing
            if i not in dates or i not in t1_vals or i not in t1_errs:
                ax.set_visible(False)  # Hide subplot if qubit data is missing
                continue

            # Ensure that the data lists are not empty
            if len(dates[i]) == 0 or len(t1_vals[i]) == 0 or len(t1_errs[i]) == 0:
                ax.set_visible(False)  # Hide subplot if no valid data
                continue

            if len(dates[i])>1:
                date_label = dates[i][0]
            else:
                date_label = ''

            if len(t1_vals[i]) >1:
                optimal_bin_num = 45#self.optimal_bins(t1_vals[i])

                # Fit a Gaussian to the raw data instead of the histogram
                # get the mean and standard deviation of the data
                # mu_1, std_1 = norm.fit(t1_vals[i])

                # # NEW: WEIGHTED MEANS -------------------------------------------------
                # # Weighted Gaussian fit (using inverse-variance weights), per-qubit i
                # t1s = np.asarray(t1_vals[i], dtype=float)
                # errs = np.asarray(t1_errs[i], dtype=float)
                #
                # # avoiding infinite weights and NaN pollution
                # err_floor = 1e-12
                # safe_errs = np.clip(errs, err_floor, np.inf)
                # # weights = 1.0 / (safe_errs ** 2)
                # weights = 1.0 / (safe_errs)
                #
                # w_sum = np.nansum(weights)
                # mu_1 = float(np.nansum(weights * t1s) / w_sum)
                # var = float(np.nansum(weights * (t1s - mu_1) ** 2) / w_sum)
                # std_1 = float(np.sqrt(max(var, 0.0)))
                # # ---------------------------------------------------------------------
                # --- Weighted mean with robust median-MAD clipping ------------------------
                t1s = np.asarray(t1_vals[i], dtype=float)
                errs = np.asarray(t1_errs[i], dtype=float)

                n_counts = len(t1s)

                # 0) keep only finite pairs
                finite = np.isfinite(t1s) & np.isfinite(errs)
                t1s, errs = t1s[finite], errs[finite]
                if t1s.size == 0:
                    mu_1, std_1 = np.nan, np.nan
                else:
                    # 1) robust outlier clip around the median (tune k if you like)
                    k = 2.0  # 2-4 is typical. 2 is stricter
                    med = np.median(t1s)
                    mad = np.median(np.abs(t1s - med))
                    # fallback if MAD is zero (all equal or super-tight); use small epsilon
                    if mad == 0:
                        mad = max(np.std(t1s), 1e-12)
                    keep = np.abs(t1s - med) < k * mad

                    t1s, errs = t1s[keep], errs[keep]

                    if t1s.size == 0:
                        mu_1, std_1 = np.nan, np.nan
                    else:
                        # 2) compute weights and weighted mean/std
                        err_floor = 1e-12
                        safe_errs = np.clip(errs, err_floor, np.inf)

                        # your choice: 1/s
                        weights = 1.0 / safe_errs

                        w_sum = np.nansum(weights)
                        mu_1 = float(np.nansum(weights * t1s) / w_sum)

                        # weighted variance (with your weights convention)
                        var = float(np.nansum(weights * (t1s - mu_1) ** 2) / w_sum)
                        std_1 = float(np.sqrt(max(var, 0.0)))
                # --------------------------------------------------------------------------

                mean_values[f"Qubit {i + 1}"] = mu_1  # Store the mean value for each qubit
                std_values[f"Qubit {i + 1}"] = std_1  # Store the standard deviation value for each qubit

                # Generate x values for plotting a gaussian based on this mean and standard deviation
                x_1 = np.linspace(min(t1_vals[i]), max(t1_vals[i]), optimal_bin_num)
                p_1 = norm.pdf(x_1, mu_1, std_1)

                # Calculate histogram data for t1_vals[i]
                hist_data_1, bins_1 = np.histogram(t1_vals[i], bins=optimal_bin_num)
                bin_centers_1 = (bins_1[:-1] + bins_1[1:]) / 2

                # Scale the Gaussian curve to match the histogram
                # the gaussian height natrually doesnt match the bin heights in the histograms
                # np.diff(bins_1)  calculates the width of each bin by taking the difference between bin edges
                # the total counts are in hist_data_1.sum()
                # to scale, multiply data gaussian by bin width to convert the probability density to probability within each bin
                # then multiply by the total count to scale the probability to match the overall number of datapoints
                # https://mathematica.stackexchange.com/questions/262314/fit-function-to-histogram
                # https://stackoverflow.com/questions/23447262/fitting-a-gaussian-to-a-histogram-with-matplotlib-and-numpy-wrong-y-scaling
                # ax.plot(x_1, p_1 * (np.diff(bins_1) * hist_data_1.sum()), 'b--', linewidth=2, color='black') old way

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
                # --------------------------------------------------------------------------------------------

                # Plot histogram and Gaussian fit for t1_vals[i]
                ax.hist(t1_vals[i], bins=optimal_bin_num, alpha=0.7,color=colors[i], edgecolor='black', label=date_label)

                #make a fuller gaussian to make smoother plotting for cumulative plot
                x_1_full = np.linspace(min(t1_vals[i]), max(t1_vals[i]), 2000)
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
                ax.set_title(titles[i] + f" Weighted $\mu$: {mu_1:.2f} $\sigma$:{std_1:.2f}, c: {n_counts}",fontsize = font)
                ax.set_xlabel('T1 (µs)',fontsize = font)
                ax.set_ylabel('Frequency',fontsize = font)
                ax.tick_params(axis='both', which='major', labelsize=font)

        plt.tight_layout()
        plt.savefig( analysis_folder + f'hists{exp_extension}.pdf', transparent=False, dpi=self.final_figure_quality)


        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        plt.title(f'{ext} Cumulative Distribution',fontsize = font)
        for i in range(0, len(t1_vals)):
            if len(dates[i])>1:
                date_label = dates[i][0]
            else:
                date_label = ''

            if len(t1_vals[i]) > 1:
                t1_vals_sorted = np.sort(t1_vals[i])
                len_samples = len(t1_vals_sorted)
                var = np.linspace(1,len_samples,len_samples)/ len_samples

                cumulative_gaussian = np.cumsum(gaussian_yvals[i][0]) / np.sum(gaussian_yvals[i][0])
                ax.scatter(t1_vals_sorted,var,color = colors[i], label = f'Q{i+1}', s = 5)
                ax.plot(gaussian_xvals[i][0], cumulative_gaussian, color=colors[i], label='Gauss Fit ' + f'Q{i + 1}',
                        linestyle='--')
                ax.tick_params(axis='both', which='major', labelsize=font)
        #ax.set_title('')
        ax.set_xlabel('T1 (us)',fontsize = font)
        ax.set_ylabel('Cumulative Distribution',fontsize = font)
        ax.loglog()
        ax.legend(edgecolor='black')
        #ax.set_xlim(10**0, 10**3)
        #ax.set_ylim(10 ** -7, 10 ** 0) #to compare to johns plot, need to adjust a little
        plt.tight_layout()
        plt.savefig(analysis_folder + f'cumulative{exp_extension}.pdf', transparent=False, dpi=self.final_figure_quality)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title(f'{ext} Fit Error vs T1 Time',fontsize = font)
        axes = axes.flatten()
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        for i, ax in enumerate(axes):

            # Check if the key exists in all dictionaries before accessing
            if i not in dates or i not in t1_vals or i not in t1_errs:
                ax.set_visible(False)  # Hide subplot if qubit data is missing
                continue

            if len(dates[i])>1:
                date_label = dates[i][0]
            else:
                date_label = ''
            ax.set_title(titles[i], fontsize = font)
            ax.scatter(t1_vals[i],t1_errs[i], label = date_label, color = colors[i])
            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('T1 (us)', fontsize = font)
            ax.set_ylabel('Fit error (us)', fontsize = font)
            ax.tick_params(axis='both', which='major', labelsize=font)
        plt.tight_layout()
        plt.savefig(analysis_folder + f'errs{exp_extension}.pdf', transparent=False, dpi=self.final_figure_quality)

        #plt.show()
        print('Plots saved to: ', analysis_folder)
        return std_values, mean_values


class OfflineAcquireReplica:
    """
    Offline mirror of QICK Averager acquire() math WITHOUT thresholding:
      avg over reps -> divide by ro length -> subtract IQ offset -> sum over rounds -> /soft_avgs
    Returns (I_final, Q_final) each shape (N_steps,).
    """

    def __init__(self, remove_offset=True, length_norm=True, progress=True):
        self.remove_offset = bool(remove_offset)
        self.length_norm   = bool(length_norm)
        self.progress      = bool(progress)

        # QICK-like fields
        self.soccfg = None
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
        self._iq_offset = None
        self._ro_index  = None  # readout channel index for this qubit

    # ---------- helpers ----------
    def _safe_eval_cfg(self, cfg_str):
        s = re.sub(r"<qick\.asm_v2\.QickParam object at 0x[0-9a-fA-F]+>", "None", cfg_str)
        s = re.sub(r"np\.float64\(\s*([^)]+)\s*\)", r"float(\1)", s)
        safe_globals = {"np": np, "array": np.array, "float": float, "__builtins__": {}}
        return eval(s, safe_globals)

    def _ensure_rounds_axis(self, A, *, N, rounds, reps):
        """
        Return A shaped as (rounds, N, reps). Accepts 1D/2D/3D.
        """
        A = np.asarray(A)
        if A.ndim == 3:
            return A
        if A.ndim == 2:
            N2, R2 = A.shape
            if N2 != N:
                raise ValueError(f"2D shots first dim {N2} != provided N {N}")
            if R2 % rounds != 0:
                raise ValueError(f"cannot split {R2} into rounds={rounds}")
            if (R2 // rounds) != reps:
                raise ValueError(f"2D shots second dim implies reps={R2//rounds}, expected {reps}")
            return A.reshape(N, rounds, reps).swapaxes(0, 1)
        if A.ndim == 1:
            total = A.size
            if total != N*rounds*reps:
                raise ValueError(f"1D shots size {total} != N*rounds*reps={N*rounds*reps}")
            return A.reshape(rounds, N, reps)
        raise ValueError(f"Expected 1D/2D/3D, got {A.ndim}D {A.shape}")

    def _extract_t1_dims(self, exp_cfg):
        def _as_int(x):
            try: return int(x)
            except Exception: return int(float(str(x)))
        T1 = exp_cfg['T1_ge']
        steps  = (T1.get('steps') or T1.get('n_steps') or T1.get('n_expts'))
        reps   = (T1.get('reps')  or T1.get('nreps'))
        rounds = (T1.get('rounds') or T1.get('soft_avgs'))
        if steps is None or reps is None or rounds is None:
            raise ValueError("Experiment config missing steps/reps/rounds for T1_ge.")
        return _as_int(steps), _as_int(reps), _as_int(rounds)

    def _compute_ro_norm_and_offset(self, *, soccfg, syst_cfg, qubit_index):
        ro_ch_list = syst_cfg['ro_ch']
        res_ch     = syst_cfg['res_ch']
        mixer_freq = float(syst_cfg['mixer_freq'])
        nqz_res    = int(syst_cfg['nqz_res'])
        res_len_us = float(syst_cfg['res_length'])
        res_freqs  = syst_cfg['res_freq_ge']
        ro_phases  = syst_cfg.get('ro_phase', [0]*len(res_freqs))

        ro_ch_for_q = int(ro_ch_list[int(qubit_index)])
        f_q         = float(res_freqs[int(qubit_index)])
        ph_q        = float(ro_phases[int(qubit_index)])

        ro_cycles = int(soccfg.us2cycles(us=res_len_us, ro_ch=ro_ch_for_q))  # length for normalization

        # build ro regs like QICK does to evaluate the PFB doubling condition
        rocfg   = soccfg['readouts'][ro_ch_for_q]
        ro_regs = soccfg.calc_ro_regs(rocfg, phase=ph_q, sel='product')

        ro_ch0_for_rounding = int(ro_ch_list[0])
        mixer_info    = soccfg.calc_mixer_freq(res_ch, mixer_freq, nqz_res, ro_ch0_for_rounding)
        mixer_rounded = mixer_info['rounded']

        ro_pars = {'gen_ch': res_ch, 'freq': f_q}
        soccfg.calc_ro_freq(rocfg, ro_pars, ro_regs,
                            absolute_freqs=False,
                            mixer_freq=mixer_rounded,
                            flip_freq=False)

        offset = float(rocfg['iq_offset'])
        if 'pfb_ch' in ro_regs:
            fs_int = 2**rocfg['b_dds']
            if ro_regs['f_int'] == (ro_regs['pfb_ch'] % 2) * (fs_int//2):
                offset *= 2.0

        return ro_cycles, offset, ro_ch_for_q, ro_regs

    # ---------- setup ----------
    def setup_offline_from_strings(self, exp_config_str, syst_config_str, soccfg, qubit_index):
        exp_cfg  = self._safe_eval_cfg(exp_config_str)
        syst_cfg = self._safe_eval_cfg(syst_config_str)
        self.soccfg = soccfg

        steps, reps, rounds = self._extract_t1_dims(exp_cfg)
        self._N_steps = steps
        self._reps    = reps
        self._rounds  = rounds

        ro_cycles, iq_offset, ro_ch_for_q, _ = self._compute_ro_norm_and_offset(
            soccfg=soccfg, syst_cfg=syst_cfg, qubit_index=qubit_index
        )
        self._ro_cycles = ro_cycles
        self._iq_offset = iq_offset
        self._ro_index  = ro_ch_for_q

        # Mirror QICK bookkeeping: loop_dims = [reps, steps], avg_level = 0 (avg over reps)
        self.ro_chs = OrderedDict({
            ro_ch_for_q: {
                'length': int(ro_cycles),
                'trigs': 1,
                'edge_counting': False,
                'ro_config': self.soccfg['readouts'][ro_ch_for_q]
            }
        })
        self.loop_dims      = [self._reps, self._N_steps]
        self.avg_level      = 0
        self.reads_per_shot = [1]

    def coerce_to_rounds_N_reps(self, A, steps, reps):
        A = np.asarray(A)
        if A.ndim == 3:
            # assume already (rounds, N, reps)
            return A
        if A.ndim == 2:
            r0, r1 = A.shape
            # common cases: (reps, N) or (N, reps)
            if r0 == reps and r1 == steps:
                return A.T[None, ...]  # -> (1, N, reps)
            if r0 == steps and r1 == reps:
                return A[None, ...]  # -> (1, N, reps)
            raise ValueError(
                f"Unexpected 2D shape {A.shape}; expected (reps,N)=({reps},{steps}) or (N,reps)=({steps},{reps}).")
        if A.ndim == 1:
            total = A.size
            if total == steps * reps:
                return A.reshape(steps, reps)[None, ...]
            if total == steps:
                return A.reshape(1, steps, 1)
            raise ValueError(f"Unexpected 1D length {total}; cannot infer (N,reps) from steps={steps}, reps={reps}.")
        raise ValueError(f"Shots must be 1D/2D/3D, got {A.ndim}D {A.shape}")


    # ---------- QICK-faithful averaging ----------
    def _ro_offset_qick(self, ro_ch, chcfg):
        rocfg = self.soccfg['readouts'][ro_ch]
        offset = rocfg['iq_offset']
        if chcfg is not None and 'pfb_ch' in chcfg:
            fs_int = 2**rocfg['b_dds']
            if chcfg['f_int'] == (chcfg['pfb_ch'] % 2) * (fs_int//2):
                offset *= 2
        return float(offset)

    def _average_buf_qick(self, d_reps, reads_per_shot, *, length_norm=True, remove_offset=True):
        avg_d = []
        for i_ch, (ch, ro) in enumerate(self.ro_chs.items()):
            # average over avg_level (==0) i.e. over reps
            avg = d_reps[i_ch].sum(axis=self.avg_level) / self.loop_dims[self.avg_level]
            if length_norm and not ro['edge_counting']:
                avg = avg / ro['length']
                if remove_offset:
                    avg -= self._ro_offset_qick(ch, ro.get('ro_config'))
            # move reads_per_shot axis to front (we have 1 read)
            avg_d.append(np.moveaxis(avg, -2, 0))  # -> (1, steps, 2)
        return avg_d

    # ---------- public: acquire offline (no thresholding) ----------
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
        if int(soft_avgs) != self._rounds:
            # we sum 'soft_avgs' rounds; this mirrors QICK's software averaging
            pass

        I3 = self._ensure_rounds_axis(Ishots, N=self._N_steps, rounds=self._rounds, reps=self._reps)
        Q3 = self._ensure_rounds_axis(Qshots, N=self._N_steps, rounds=self._rounds, reps=self._reps)

        # accumulate per-round like QICK does, using the same averaging kernel
        summed = None
        for r in range(self._rounds):
            # pack this round like acc_buf: (reps, steps, 1, 2)
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

