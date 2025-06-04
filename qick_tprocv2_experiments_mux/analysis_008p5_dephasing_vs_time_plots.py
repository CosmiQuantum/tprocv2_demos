import numpy as np
import os
import sys

sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
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
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import norm
from scipy.optimize import curve_fit

class DephasingVsTime:
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
        self.name=None

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

    def exponential(x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def optimal_bins(data):
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

    def run_T1(self,return_errs=False,return_noise_gain=False,return_freq_offset=False,name='Dephased_ge', savefigs=False):
        import datetime
        self.name=name
        # ----------Load/get data------------------------
        t1_vals = {i: [] for i in range(self.number_of_qubits)}
        t1_errs = {i: [] for i in range(self.number_of_qubits)}
        Is = {i: [] for i in range(self.number_of_qubits)}
        Qs = {i: [] for i in range(self.number_of_qubits)}
        noise_gains = {i: [] for i in range(self.number_of_qubits)}
        Delay_Times = {i: [] for i in range(self.number_of_qubits)}
        freq_offsets = {i: [] for i in range(self.number_of_qubits)}
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

            # -------------------------------------------------------Load/Plot/Save T2E------------------------------------------
            outerFolder_expt = outerFolder + f"/Data_h5/{name}/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type='T1', save_r=int(save_round))

                #H5_class_instance.print_h5_contents(h5_file)
                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data['T1']:
                    for dataset in range(len(load_data['T1'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data['T1'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T2 = load_data['T2E'][q_key].get('T2', [])[0][dataset]
                        # errors = load_data['T2E'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data['T1'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        I = self.process_h5_data(load_data['T1'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data['T1'][q_key].get('Q', [])[0][dataset].decode())
                        delay_times = self.process_h5_data(load_data['T1'][q_key].get('Delay Times', [])[0][dataset].decode())
                        # fit = load_data['T2E'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data['T1'][q_key].get('Round Num', [])[0][dataset]
                        batch_num = load_data['T1'][q_key].get('Batch Num', [])[0][dataset]

                        syst_config = load_data['T1'][q_key].get('Syst Config', [])[0][dataset].decode()
                        exp_config = load_data['T1'][q_key].get('Exp Config', [])[0][dataset].decode()
                        safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                        exp_config = eval(exp_config, safe_globals)



                        if len(I) > 0:

                            noise_gains[q_key].extend([
                                round(float(syst_config.split('noise_pulse_gain\': ')[-1].split(',')[0]), 4)])
                            if return_freq_offset:
                                freq_offsets[q_key].extend([
                                    round(float(syst_config.split('noise_offset_freq_from_ef\': ')[-1].split('}')[0]), 4)])
                            T2E_class_instance = T2EMeasurement(q_key, self.number_of_qubits, outerFolder_save_plots, round_num, self.signal, self.save_figs,
                                                               fit_data=True)
                            #try:
                            fitted, t2e_est, t2e_err, plot_sig = T2E_class_instance.t2_fit(delay_times, I, Q, mag=True)
                            if savefigs:
                                self.plot_results(I, Q, delay_times,  fitted, t2e_est,
                                                 t2e_err, config=None, fig_quality=100,outerFolder=outerFolder, expt_name=name)
                            # except Exception as e:
                            #     print(f"good fit not found, error: {e}")
                            #     continue
                            #T2E_cfg = exp_config['SpinEcho_ge']
                            # if t2e_est < 0:
                            #     print("The value is negative, continuing...")
                            #     continue
                            # if t2e_est > 300:
                            #     print("The value is above 300 us, this is a bad fit, continuing...")
                            #     continue
                            # if t2e_err >= 0.8 * t2e_est:
                            #     print(
                            #         f"Skipping T2R = {t2e_est:.3f} µs because its error {t2e_err:.3f} µs is >= 80% of its value.")
                            #     continue
                            t1_vals[q_key].extend([t2e_est])
                            t1_errs[q_key].extend([t2e_err])

                            Is[q_key].extend([I])
                            Qs[q_key].extend([Q])
                            Delay_Times[q_key].extend([delay_times])
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del T2E_class_instance
                del H5_class_instance
        if return_errs:
            return date_times, t1_vals, t1_errs
        elif return_noise_gain:
            return date_times, t1_vals, noise_gains, Is, Qs, Delay_Times
        elif return_freq_offset:
            return date_times, t1_vals, noise_gains, Is, Qs, Delay_Times, freq_offsets
        else:
            return date_times, t1_vals

    def run(self,return_errs=False,return_noise_gain=False,return_freq_offset=False,name='Dephased_ge', savefigs=False):
        import datetime
        self.name=name
        # ----------Load/get data------------------------
        t2e_vals = {i: [] for i in range(self.number_of_qubits)}
        t2e_errs = {i: [] for i in range(self.number_of_qubits)}
        Is = {i: [] for i in range(self.number_of_qubits)}
        Qs = {i: [] for i in range(self.number_of_qubits)}
        noise_gains = {i: [] for i in range(self.number_of_qubits)}
        Delay_Times = {i: [] for i in range(self.number_of_qubits)}
        freq_offsets = {i: [] for i in range(self.number_of_qubits)}
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

            # -------------------------------------------------------Load/Plot/Save T2E------------------------------------------
            outerFolder_expt = outerFolder + f"/Data_h5/{name}/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type='T2E', save_r=int(save_round))

                #H5_class_instance.print_h5_contents(h5_file)
                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data['T2E']:
                    for dataset in range(len(load_data['T2E'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data['T2E'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T2 = load_data['T2E'][q_key].get('T2', [])[0][dataset]
                        # errors = load_data['T2E'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data['T2E'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        I = self.process_h5_data(load_data['T2E'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data['T2E'][q_key].get('Q', [])[0][dataset].decode())
                        delay_times = self.process_h5_data(load_data['T2E'][q_key].get('Delay Times', [])[0][dataset].decode())
                        # fit = load_data['T2E'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data['T2E'][q_key].get('Round Num', [])[0][dataset]
                        batch_num = load_data['T2E'][q_key].get('Batch Num', [])[0][dataset]

                        syst_config = load_data['T2E'][q_key].get('Syst Config', [])[0][dataset].decode()
                        exp_config = load_data['T2E'][q_key].get('Exp Config', [])[0][dataset].decode()
                        safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                        exp_config = eval(exp_config, safe_globals)



                        if len(I) > 0:


                            T2E_class_instance = T2EMeasurement(q_key, self.number_of_qubits, outerFolder_save_plots, round_num, self.signal, self.save_figs,
                                                               fit_data=True)
                            #try:
                            fitted, t2e_est, t2e_err, plot_sig = T2E_class_instance.t2_fit(delay_times, I, Q, mag=True)
                            if savefigs:
                                self.plot_results(I, Q, delay_times,  fitted, t2e_est,
                                                 t2e_err, config=None, fig_quality=100,outerFolder=outerFolder, expt_name=name)
                            # except Exception as e:
                            #     print(f"good fit not found, error: {e}")
                            #     continue
                            T2E_cfg = exp_config['SpinEcho_ge']
                            if t2e_est < 0:
                                print("The value is negative, continuing...")
                                continue
                            if t2e_est > 300:
                                print("The value is above 300 us, this is a bad fit, continuing...")
                                continue
                            if t2e_err >= 0.2 * t2e_est:
                                print(
                                    f"Skipping T2R = {t2e_est:.3f} µs because its error {t2e_err:.3f} µs is >= 80% of its value.")
                                continue
                            t2e_vals[q_key].extend([t2e_est])
                            t2e_errs[q_key].extend([t2e_err])

                            Is[q_key].extend([I])
                            Qs[q_key].extend([Q])
                            Delay_Times[q_key].extend([delay_times])
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])
                            noise_gains[q_key].extend([
                                round(float(syst_config.split('noise_pulse_gain\': ')[-1].split(',')[0]), 4)])
                            if return_freq_offset:
                                freq_offsets[q_key].extend([
                                    round(float(syst_config.split('noise_offset_freq_from_ef\': ')[-1].split('}')[0]),
                                          4)])
                            del T2E_class_instance
                del H5_class_instance
        if return_errs:
            return date_times, t2e_vals, t2e_errs
        elif return_noise_gain:
            return date_times, t2e_vals, noise_gains, Is, Qs, Delay_Times
        elif return_freq_offset:
            return date_times, t2e_vals, noise_gains, Is, Qs, Delay_Times, freq_offsets
        else:
            return date_times, t2e_vals
    def plot_results(self, I, Q, delay_times,  fit, t2e_est,
                     t2e_err, config=None, fig_quality=100, outerFolder='', expt_name=None):
        mag = np.hypot(I, Q)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plt.rcParams.update({'font.size': 18})

        plot_middle = (ax.get_position().x0 + ax.get_position().x1) / 2
        title_str = (f"  "
                     f"T2 = {t2e_est:0.2f} µs  "
                     f"({int(config['reps'])} × {int(config['rounds'])} avgs)"
                     if config is not None else
                     f"T2 = {t2e_est:0.2f} µs")
        fig.text(plot_middle, 0.98, title_str, fontsize=24, ha='center', va='top')

        ax.plot(delay_times, mag, "-", label="Magnitude", linewidth=2)
        ax.plot(delay_times, fit, "-", color='red', linewidth=3, label="Fit")

        ax.set_xlabel("Delay time (µs)", fontsize=20)
        ax.set_ylabel("Magnitude (a.u.)", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        if self.save_figs:
            outerFolder_expt = os.path.join(outerFolder, expt_name)
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            fmt_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            fname = f"R_{fmt_time}_{expt_name}.png"
            fig.savefig(os.path.join(outerFolder_expt, fname),
                        dpi=fig_quality, bbox_inches='tight')

        plt.close(fig)
    def plot_noise_vs_offset(
            self,
            date_times: dict[int, list[str]],
            Is: dict[int, list[list[float]]],
            Qs: dict[int, list[list[float]]],
            delay_times: dict[int, list[list[float]]],  # <-- now list-of-lists
            noise_gains: dict[int, list[float]],
            freq_offsets: dict[int, list[float]],
            cmap: str = "viridis",
            show_colorbar: bool = True,
    ):
        """
        Heat-map of |I + iQ| versus delay and noise gain for every qubit that
        actually has data.  Missing qubits are silently skipped.

        Parameters
        ----------
        date_times   : {qubit: [timestamp str, …]}   (kept for symmetry / future filtering)
        Is, Qs       : {qubit: 2-D array-like}       shape → (n_gains, n_delays_for_that_gain)
        delay_times  : {qubit: 2-D array-like}       same shape as Is / Qs
        noise_gains  : {qubit: 1-D array-like}       length == n_gains
        """

        # ---------- where to save ----------
        if self.fridge.upper() == "QUIET":
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/noise_vs_delay/"
        elif self.fridge.upper() == "NEXUS":
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/noise_vs_delay/"
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")
        self.create_folder_if_not_exists(analysis_folder)
        print(freq_offsets)
        # ---------- which qubits have data ----------
        valid_qubits = [
            q for q in range(self.number_of_qubits)
            if q in Is and q in Qs and q in delay_times and q in freq_offsets
               and len(Is[q]) and len(Qs[q]) and len(delay_times[q]) and len(freq_offsets[q])
        ]
        if not valid_qubits:
            raise ValueError("No data found for any qubit")

        n_plots = len(valid_qubits)
        n_cols = min(3, n_plots)  # up to 3 columns looks nice
        n_rows = ceil(n_plots / n_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3.2 * n_rows),
            sharex=False, sharey=False,
        )
        axes = np.asarray(axes).flatten()
        plt.rcParams.update({"font.size": 14})

        fig.suptitle(f"T2 vs Noise Pulse Freq offset for noise pulse gain 0.02", fontsize=16)

        # ---------- draw each qubit ----------
        import matplotlib as mpl
        norm = mpl.colors.Normalize(vmin=2.75, vmax=5.1)
        for ax, q in zip(axes, valid_qubits):
            # -- to NumPy ------------------------------------------------------
            I_mat = np.asarray(Is[q], dtype=float)
            Q_mat = np.asarray(Qs[q], dtype=float)
            delays_mat = np.asarray(delay_times[q], dtype=float)
            offsets_vec = np.asarray(freq_offsets[q], dtype=float)

            # -- shape checks --------------------------------------------------
            if I_mat.shape != Q_mat.shape:
                raise ValueError(f"Shape mismatch I vs Q for qubit {q}")
            if delays_mat.shape != I_mat.shape:
                raise ValueError(
                    f"Delay-time shape mismatch for qubit {q}: "
                    f"{delays_mat.shape} vs {I_mat.shape}"
                )
            if len(offsets_vec) != I_mat.shape[0]:
                raise ValueError(
                    f"#offset rows ≠ data rows for qubit {q}: "
                    f"{len(offsets_vec)} vs {I_mat.shape[0]}"
                )

            # -- sort rows by offset so Y is monotonic -------------------------
            sort_idx = np.argsort(offsets_vec)
            offsets_sorted = offsets_vec[sort_idx]
            magnitude_sorted = np.hypot(I_mat, Q_mat)[sort_idx]
            delays_sorted = delays_mat[sort_idx]

            # -- build *regular* offset grid, inserting NaN rows --------------
            diffs = np.diff(offsets_sorted)
            step = np.min(diffs[diffs > 0])  # smallest positive gap ⇒ Δf
            full_y = np.arange(offsets_sorted[0],
                               offsets_sorted[-1] + 0.5 * step,
                               step)

            # prepare empty grid full of NaNs
            mag_grid = np.full((len(full_y), magnitude_sorted.shape[1]), np.nan)
            delay_grid = np.full_like(mag_grid, np.nan)

            full_y = np.arange(offsets_sorted[0],
                               offsets_sorted[-1] + 0.5 * step,
                               step)

            # 1)  Z-values: start filled with NaN so they can be masked later
            mag_grid = np.full((len(full_y), magnitude_sorted.shape[1]), np.nan)

            # 2)  X-coordinates: *never* NaN → use the delay vector from the first
            #     real row for every row in the grid (most experiments reuse the
            #     same delay sweep for every offset).
            x_template = delays_sorted[0]  # 1-D vector
            delay_grid = np.repeat(x_template[None, :], len(full_y), axis=0)

            # Copy real data rows into their correct places
            for row, f in enumerate(offsets_sorted):
                idx = int(round((f - full_y[0]) / step))
                mag_grid[idx] = magnitude_sorted[row]
                # (delay_grid already OK)

            # Mask NaNs so they plot as "bad"
            mag_masked = np.ma.masked_invalid(mag_grid)

            # mask NaNs so they get “bad” color
            mag_masked = np.ma.masked_invalid(mag_grid)

            # -- colormap with gray 'bad' color -------------------------------
            cmap_use = plt.get_cmap(cmap).copy()
            cmap_use.set_bad(color="lightgray")

            # -- coordinate meshes for pcolormesh -----------------------------
            X = delay_grid
            Y = np.repeat(full_y[:, None], X.shape[1], axis=1)

            pcm = ax.pcolormesh(X, Y, mag_masked,
                                shading="nearest", cmap=cmap_use, vmin=2.7, vmax=5.1  )

            # -- cosmetics -----------------------------------------------------
            ax.set_title(f"Qubit {q + 1}")
            ax.set_xlabel("Delay (us)")
            ax.set_ylabel("Freq offset (MHz)")
            ax.ticklabel_format(style="plain", axis="x")
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

            if show_colorbar:
                cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
                cbar.set_label("Magnitude")
        # ---------- hide any leftover empty axes ----------
        for ax in axes[n_plots:]:
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        outfile = os.path.join(analysis_folder, "noise_vs_freq_offset.png")
        plt.savefig(outfile, dpi=self.final_figure_quality)
        print("Plot saved at:", outfile)
        plt.close()
    def plot_noise_vs_gain_v2(
            self,
            date_times: dict[int, list[str]],
            Is: dict[int, list[list[float]]],
            Qs: dict[int, list[list[float]]],
            delay_times: dict[int, list[list[float]]],  # <-- now list-of-lists
            noise_gains: dict[int, list[float]],

            cmap: str = "viridis",
            show_colorbar: bool = True,
            zlim=None
    ):
        """
        Heat-map of |I + iQ| versus delay and noise gain for every qubit that
        actually has data.  Missing qubits are silently skipped.

        Parameters
        ----------
        date_times   : {qubit: [timestamp str, …]}   (kept for symmetry / future filtering)
        Is, Qs       : {qubit: 2-D array-like}       shape → (n_gains, n_delays_for_that_gain)
        delay_times  : {qubit: 2-D array-like}       same shape as Is / Qs
        noise_gains  : {qubit: 1-D array-like}       length == n_gains
        """
        from matplotlib.ticker import StrMethodFormatter
        import matplotlib as mpl
        # ---------- where to save ----------
        if self.fridge.upper() == "QUIET":
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/noise_vs_delay/"
        elif self.fridge.upper() == "NEXUS":
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/noise_vs_delay/"
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")
        self.create_folder_if_not_exists(analysis_folder)

        # ---------- which qubits have data ----------
        valid_qubits = [
            q for q in range(self.number_of_qubits)
            if q in Is and q in Qs and q in delay_times and q in noise_gains
               and len(Is[q]) and len(Qs[q]) and len(delay_times[q]) and len(noise_gains[q])
        ]
        if not valid_qubits:
            raise ValueError("No data found for any qubit")

        n_plots = len(valid_qubits)
        n_cols = min(3, n_plots)  # up to 3 columns looks nice
        n_rows = ceil(n_plots / n_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3.2 * n_rows),
            sharex=False, sharey=False,
        )
        axes = np.asarray(axes).flatten()
        plt.rcParams.update({"font.size": 14})

        fig.suptitle(f"T2 vs Noise Pulse Freq offset for noise pulse gain 0.02", fontsize=16)
        if zlim is not None:
            vmin, vmax = zlim
            z_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        # ---------- draw each qubit ----------
        for ax, q in zip(axes, valid_qubits):
            # -- to NumPy ------------------------------------------------------
            I_mat = np.asarray(Is[q], dtype=float)
            Q_mat = np.asarray(Qs[q], dtype=float)
            delays_mat = np.asarray(delay_times[q], dtype=float)
            offsets_vec = np.asarray(noise_gains[q], dtype=float)

            # -- shape checks --------------------------------------------------
            if I_mat.shape != Q_mat.shape:
                raise ValueError(f"Shape mismatch I vs Q for qubit {q}")
            if delays_mat.shape != I_mat.shape:
                raise ValueError(
                    f"Delay-time shape mismatch for qubit {q}: "
                    f"{delays_mat.shape} vs {I_mat.shape}"
                )
            if len(offsets_vec) != I_mat.shape[0]:
                raise ValueError(
                    f"#offset rows ≠ data rows for qubit {q}: "
                    f"{len(offsets_vec)} vs {I_mat.shape[0]}"
                )

            # -- sort rows by offset so Y is monotonic -------------------------
            sort_idx = np.argsort(offsets_vec)
            offsets_sorted = offsets_vec[sort_idx]
            magnitude_sorted = np.hypot(I_mat, Q_mat)[sort_idx]
            delays_sorted = delays_mat[sort_idx]

            # -- build *regular* offset grid, inserting NaN rows --------------
            diffs = np.diff(offsets_sorted)
            step = np.min(diffs[diffs > 0])  # smallest positive gap ⇒ Δf
            full_y = np.arange(offsets_sorted[0],
                               offsets_sorted[-1] + 0.5 * step,
                               step)

            # prepare empty grid full of NaNs
            mag_grid = np.full((len(full_y), magnitude_sorted.shape[1]), np.nan)
            delay_grid = np.full_like(mag_grid, np.nan)

            full_y = np.arange(offsets_sorted[0],
                               offsets_sorted[-1] + 0.5 * step,
                               step)

            # 1)  Z-values: start filled with NaN so they can be masked later
            mag_grid = np.full((len(full_y), magnitude_sorted.shape[1]), np.nan)

            # 2)  X-coordinates: *never* NaN → use the delay vector from the first
            #     real row for every row in the grid (most experiments reuse the
            #     same delay sweep for every offset).
            x_template = delays_sorted[0]  # 1-D vector
            delay_grid = np.repeat(x_template[None, :], len(full_y), axis=0)

            # Copy real data rows into their correct places
            for row, f in enumerate(offsets_sorted):
                idx = int(round((f - full_y[0]) / step))
                mag_grid[idx] = magnitude_sorted[row]
                # (delay_grid already OK)

            # Mask NaNs so they plot as "bad"
            mag_masked = np.ma.masked_invalid(mag_grid)

            # mask NaNs so they get “bad” color
            mag_masked = np.ma.masked_invalid(mag_grid)

            # -- colormap with gray 'bad' color -------------------------------
            cmap_use = plt.get_cmap(cmap).copy()
            cmap_use.set_bad(color="lightgray")

            # -- coordinate meshes for pcolormesh -----------------------------
            X = delay_grid
            Y = np.repeat(full_y[:, None], X.shape[1], axis=1)
            if zlim is None:
                pcm = ax.pcolormesh(X, Y, mag_masked,
                                    shading="nearest", cmap=cmap_use)
            else:
                pcm = ax.pcolormesh(
                    X, Y, mag_masked,
                    shading="nearest",
                    cmap=cmap_use,
                    norm=z_norm,  # ← use the shared norm (or None)
                )
            # -- cosmetics -----------------------------------------------------
            ax.set_title(f"Qubit {q + 1}")
            ax.set_xlabel("Delay time (us)")
            ax.set_ylabel("Noise gain (a.u.)")
            ax.ticklabel_format(style="plain", axis="x")
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))



            if show_colorbar:
                cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
                cbar.set_label("Signal magnitude")
        # ---------- hide any leftover empty axes ----------
        for ax in axes[n_plots:]:
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        outfile = os.path.join(analysis_folder, f"noise_vs_gains{self.name}.png")
        plt.savefig(outfile, dpi=self.final_figure_quality)
        print("Plot saved at:", outfile)
        plt.close()

    def plot_noise_vs_delay(
            self,
            date_times: dict[int, list[str]],
            Is: dict[int, list[list[float]]],
            Qs: dict[int, list[list[float]]],
            delay_times: dict[int, list[list[float]]],  # <-- now list-of-lists
            noise_gains: dict[int, list[float]],
            cmap: str = "viridis",
            show_colorbar: bool = True,
    ):
        """
        Heat-map of |I + iQ| versus delay and noise gain for every qubit that
        actually has data.  Missing qubits are silently skipped.

        Parameters
        ----------
        date_times   : {qubit: [timestamp str, …]}   (kept for symmetry / future filtering)
        Is, Qs       : {qubit: 2-D array-like}       shape → (n_gains, n_delays_for_that_gain)
        delay_times  : {qubit: 2-D array-like}       same shape as Is / Qs
        noise_gains  : {qubit: 1-D array-like}       length == n_gains
        """

        # ---------- where to save ----------
        if self.fridge.upper() == "QUIET":
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/noise_vs_delay/"
        elif self.fridge.upper() == "NEXUS":
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/noise_vs_delay/"
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")
        self.create_folder_if_not_exists(analysis_folder)

        # ---------- which qubits have data ----------
        valid_qubits = [
            q for q in range(self.number_of_qubits)
            if q in Is and q in Qs and q in delay_times and q in noise_gains
               and len(Is[q]) and len(Qs[q]) and len(delay_times[q]) and len(noise_gains[q])
        ]
        if not valid_qubits:
            raise ValueError("No data found for any qubit")

        n_plots = len(valid_qubits)
        n_cols = min(3, n_plots)  # up to 3 columns looks nice
        n_rows = ceil(n_plots / n_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3.2 * n_rows),
            sharex=False, sharey=False,
        )
        axes = np.asarray(axes).flatten()
        plt.rcParams.update({"font.size": 14})
        fig.suptitle("T2 vs Noise Pulse Gain", fontsize=16)

        # ---------- draw each qubit ----------
        for ax, q in zip(axes, valid_qubits):
            I_mat = np.asarray(Is[q], dtype=float)
            Q_mat = np.asarray(Qs[q], dtype=float)

            delays_mat = np.asarray(delay_times[q], dtype=float)
            gains_vec = np.asarray(noise_gains[q], dtype=float)

            # sanity checks -------------------------------------------------------
            if I_mat.shape != Q_mat.shape:
                raise ValueError(f"Shape mismatch I vs Q for qubit {q}")
            if delays_mat.shape != I_mat.shape:
                raise ValueError(
                    f"Delay-time shape mismatch for qubit {q}: "
                    f"{delays_mat.shape} vs {I_mat.shape}"
                )

            if len(gains_vec) != I_mat.shape[0]:
                raise ValueError(
                    f"Number of gain rows does not match data for qubit {q}: "
                    f"{len(gains_vec)} vs {I_mat.shape[0]}"
                )

            # magnitude + coordinate grids ---------------------------------------
            magnitude = np.sqrt(I_mat ** 2 + Q_mat ** 2)

            # Broadcast gain values along the delay axis to build a full Y grid
            Y = np.repeat(gains_vec[:, None], delays_mat.shape[1], axis=1)
            X = delays_mat
            #print(X)
            #print(Y)
            pcm = ax.pcolormesh(X, Y, magnitude, shading="auto", cmap=cmap)

            # cosmetics -----------------------------------------------------------
            ax.set_title(f"Qubit {q + 1}")
            ax.set_xlabel("Delay time (us)")
            ax.set_ylabel("Noise gain (a.u.)")
            ax.ticklabel_format(style="plain", axis="x")
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

            if show_colorbar:
                cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
                cbar.set_label("Signal Magnitude")

        # ---------- hide any leftover empty axes ----------
        for ax in axes[n_plots:]:
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        outfile = os.path.join(analysis_folder, "noise_vs_delay.png")
        plt.savefig(outfile, dpi=self.final_figure_quality)
        print("Plot saved at:", outfile)
        plt.close()

    def plot_without_errs(self, date_times, t2e_vals, show_legends):
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

        # ----------------To Plot a specific timeframe------------------
        from datetime import datetime
        year = 2025
        month = 1
        day1 = 24  # Start date
        day2 = 25  # End date
        hour_start = 0  # Start hour
        hour_end = 12  # End hour
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 0)
        # -----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('T2E Values vs Time', fontsize=font)
        axes = axes.flatten()

        from datetime import datetime
        for i, ax in enumerate(axes):

            if i >= self.number_of_qubits:  # If we have fewer qubits than subplots, stop plotting and hide the rest
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]
            y = t2e_vals[i]

            # Convert strings to datetime objects.
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            # Combine datetime objects and y values into a list of tuples and sort by datetime.
            combined = list(zip(datetime_objects, y))
            combined.sort(reverse=True, key=lambda x: x[0])

            if len(combined) == 0:
                # If this qubit has no data, just skip
                ax.set_visible(False)
                continue

            # Unpack them back into separate lists, in order from latest to most recent.
            sorted_x, sorted_y = zip(*combined)
            ax.scatter(sorted_x, sorted_y, color=colors[i])

            # Set x-axis limits for the specific timeframe
            ax.set_xlim(start_time, end_time)

            sorted_x = np.asarray(sorted(x))

            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically choose good tick locations
            # ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))  # Format as month-day
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))  # Show day and time
            ax.tick_params(axis='x', rotation=45)  # Rotate ticks for better readability

            # Disable scientific notation and format y-ticks
            ax.ticklabel_format(style="plain", axis="y")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))  # 2 decimal places

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel('T2E (us)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2E_vals.png', transparent=False, dpi=self.final_figure_quality)
        print('Plot saved at: ', analysis_folder)
        plt.close()

    def plot_with_errs(self, date_times, t2e_vals, t2e_fit_err, show_legends):
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

        # ----------------To Plot a specific timeframe------------------
        from datetime import datetime
        year = 2025
        month = 1
        day1 = 24
        day2 = 25
        hour_start = 0
        hour_end = 12
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 0)
        # -----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.suptitle('T2E Values vs Time', fontsize=font)
        axes = axes.flatten()

        import matplotlib.dates as mdates
        from matplotlib.ticker import StrMethodFormatter

        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]
            y = t2e_vals[i]
            err = t2e_fit_err[i]

            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])
            if len(combined) == 0:
                ax.set_visible(False)
                continue
            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)

            #ax.set_xlim(start_time, end_time)

            ax.errorbar(
                sorted_x, sorted_y, yerr=sorted_err,
                fmt='none',
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

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.tick_params(axis='x', rotation=45)

            ax.ticklabel_format(style="plain", axis="y")

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel('T2E (us)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2E_vals.png', transparent=False, dpi=self.final_figure_quality)
        print('Plot saved at: ', analysis_folder)
        plt.close()

    def plot_with_errs_vs_echo(self, date_times, t2e_vals, t2e_fit_err,date_times_echo, t2e_vals_echo, t2e_fit_err_echo, show_legends):
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

        # ----------------To Plot a specific timeframe------------------
        from datetime import datetime
        year = 2025
        month = 1
        day1 = 24
        day2 = 25
        hour_start = 0
        hour_end = 12
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 0)
        # -----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.suptitle('Dynamically Decoupled and Echo Values vs Time', fontsize=font)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            # ----------- MAIN DATA (T2E) -----------
            dt_main = [datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s in date_times[i]]
            combo = list(zip(dt_main, t2e_vals[i], t2e_fit_err[i]))
            combo.sort(key=lambda tup: tup[0])  # chronological
            if not combo:  # empty → hide axis
                ax.set_visible(False)
                continue
            x_main, y_main, err_main = zip(*combo)
            err_main = np.asarray(err_main, dtype=float)
            err_main = np.nan_to_num(np.abs(err_main), nan=0.0, posinf=0.0, neginf=0.0)

            ax.errorbar(x_main, y_main,
                        yerr=err_main,
                        fmt='none',
                        ecolor='green',
                        elinewidth=1,
                        capsize=0)
            ax.scatter(x_main, y_main,
                       s=10,
                       color='green',
                       alpha=0.7,
                       label="Dynamically decoupled" if show_legends else None)

            # ----------- ECHO DATA -----------
            dt_echo = [datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s in date_times_echo[i]]
            combo_e = list(zip(dt_echo, t2e_vals_echo[i], t2e_fit_err_echo[i]))
            combo_e.sort(key=lambda tup: tup[0])
            if combo_e:  # only plot if data exists
                x_echo, y_echo, err_echo = zip(*combo_e)
                err_echo = np.asarray(err_echo, dtype=float)
                err_echo = np.nan_to_num(np.abs(err_echo), nan=0.0, posinf=0.0, neginf=0.0)
                ax.errorbar(x_echo, y_echo,
                            yerr=err_echo,
                            fmt='none',
                            ecolor='blue',
                            linestyle='--',
                            alpha=0.6,
                            elinewidth=1,
                            capsize=0)
                ax.scatter(x_echo, y_echo,
                           s=12,
                           marker='x',
                           color='blue',
                           alpha=0.9,
                           label="Echo" if show_legends else None)

            # ----------- AXIS FORMATTING -----------
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.tick_params(axis='x', rotation=45)
            ax.ticklabel_format(style="plain", axis="y")
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel('T2E (µs)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)
            if show_legends:
                ax.legend(edgecolor='black', fontsize=8)

        plt.tight_layout()
        plt.savefig(
            analysis_folder + 'dynamical_decoupling_and_echo_vals.png',
            transparent=False,
            dpi=self.final_figure_quality
        )
        print('Plot saved at: ', analysis_folder)
        plt.close()

    def plot_with_errs_vs_everything(
            self,
            date_times, t2e_vals, t2e_fit_err,
            date_times_echo, t2e_vals_echo, t2e_fit_err_echo,
            date_times_dephasing_ef, dephasing_vals_ef, dephasing_fit_err_ef,
            date_times_dephasing_fh, dephasing_vals_fh, dephasing_fit_err_fh,
            show_legends=True):
        # ─────────────────────────────────────────────────────────────── paths ──
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder += "features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = (f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}"
                               "/benchmark_analysis_plots/")
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder += "features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        # ──────────────────────────────────────────────────────── figure setup ──
        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.subplots_adjust(right=0.83)
        plt.suptitle('T₂ Values vs Time', fontsize=font)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            # DD
            dt_main = [datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s in date_times[i]]
            combo = sorted(zip(dt_main, t2e_vals[i], t2e_fit_err[i]), key=lambda t: t[0])
            if not combo:
                ax.set_visible(False)
                continue

            x_main, y_main, err_main = zip(*combo)
            err_main = np.nan_to_num(np.abs(err_main), nan=0.0, posinf=0.0, neginf=0.0)
            ax.errorbar(x_main, y_main, yerr=err_main,
                        fmt='none', ecolor='green', elinewidth=1, capsize=0)
            ax.scatter(x_main, y_main, s=10, color='green', alpha=0.7,
                       label="Dynamically decoupled" if show_legends else None)

            # echo
            dt_echo = [datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s in date_times_echo[i]]
            combo_e = sorted(zip(dt_echo, t2e_vals_echo[i], t2e_fit_err_echo[i]), key=lambda t: t[0])
            if combo_e:
                x_e, y_e, err_e = zip(*combo_e)
                err_e = np.nan_to_num(np.abs(err_e), nan=0.0, posinf=0.0, neginf=0.0)
                ax.errorbar(x_e, y_e, yerr=err_e,
                            fmt='none', ecolor='blue', linestyle='--', alpha=0.6,
                            elinewidth=1, capsize=0)
                ax.scatter(x_e, y_e, s=12, marker='x', color='blue', alpha=0.9,
                           label="Echo" if show_legends else None)

            #  dephasing EF data
            dt_ef = [datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s
                     in date_times_dephasing_ef[i]]
            combo_ef = sorted(zip(dt_ef, dephasing_vals_ef[i], dephasing_fit_err_ef[i]),
                              key=lambda t: t[0])
            if combo_ef:
                x_ef, y_ef, err_ef = zip(*combo_ef)
                err_ef = np.nan_to_num(np.abs(err_ef), nan=0.0, posinf=0.0, neginf=0.0)
                ax.errorbar(x_ef, y_ef, yerr=err_ef,
                            fmt='none', ecolor='orange', linestyle='--', alpha=0.6,
                            elinewidth=1, capsize=0)
                ax.scatter(x_ef, y_ef, s=14, marker='^', color='orange', alpha=0.85,
                           label="Dephasing EF" if show_legends else None)

            # dephasing FH data
            dt_fh = [datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s
                     in date_times_dephasing_fh[i]]
            combo_fh = sorted(zip(dt_fh, dephasing_vals_fh[i], dephasing_fit_err_fh[i]),
                              key=lambda t: t[0])
            if combo_fh:
                x_fh, y_fh, err_fh = zip(*combo_fh)
                err_fh = np.nan_to_num(np.abs(err_fh), nan=0.0, posinf=0.0, neginf=0.0)
                ax.errorbar(x_fh, y_fh, yerr=err_fh,
                            fmt='none', ecolor='purple', linestyle='--', alpha=0.6,
                            elinewidth=1, capsize=0)
                ax.scatter(x_fh, y_fh, s=14, marker='D', color='purple', alpha=0.85,
                           label="Dephasing FH" if show_legends else None)

            # ─────────────────────────── axes formatting ──────────────────────
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel(r'$T_2$ ($\mu$s)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)
            if show_legends:
                ax.legend(loc="center left",  # middle-right of the axes
                          bbox_to_anchor=(1.02, 0.5),  # 1.00 = axes edge; 1.02 gives a small gap
                          borderaxespad=0.0,
                          frameon=False, fontsize=8)

        plt.tight_layout()
        out_file = analysis_folder + "t2_vs_time_all_modes.pdf"
        plt.savefig(out_file, dpi=self.final_figure_quality)
        print(f"Plot saved at: {out_file}")
        plt.close()

    def plot_with_errs_single_plot(self, date_times, t2e_vals, t2e_fit_err, show_legends):
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
        from datetime import datetime
        year = 2025
        month = 1
        day1 = 24
        day2 = 25
        hour_start = 0
        hour_end = 12
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 0)
        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('T2E Values vs Time', fontsize=font)
        import matplotlib.dates as mdates
        from matplotlib.ticker import StrMethodFormatter
        for i in range(self.number_of_qubits):
            x = date_times[i]
            y = t2e_vals[i]
            err = t2e_fit_err[i]
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
        ax.ticklabel_format(style="plain", axis="y")
        if show_legends:
            ax.legend(edgecolor='black')
        ax.set_xlabel('Time', fontsize=font - 2)
        ax.set_ylabel('T2E (us)', fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2E_vals_single_plot.png', transparent=False, dpi=self.final_figure_quality)
        print('Plot saved at: ', analysis_folder)
        plt.close()

