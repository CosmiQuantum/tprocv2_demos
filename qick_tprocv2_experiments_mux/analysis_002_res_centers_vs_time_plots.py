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
#from expt_config import *
import glob
import re
import datetime
import ast
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.stats import norm
from scipy.optimize import curve_fit

class ResonatorFreqVsTime:
    def __init__(self, base_data_path, plots_path, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, fridge):
        self.figure_quality = figure_quality
        self.base_data_path = base_data_path
        self.plots_path = plots_path
        self.number_of_qubits = number_of_qubits
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.run_name = run_name
        self.top_folder_dates = top_folder_dates
        self.final_figure_quality = final_figure_quality
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

    def plot_resonator_fit(self, freqs_MHz, amps, fit_result, q_idx, date, round_num, batch_num):
        freqs_MHz = np.asarray(freqs_MHz, dtype=float)
        amps = np.asarray(amps, dtype=float)

        good = np.isfinite(freqs_MHz) & np.isfinite(amps)
        freqs_MHz = freqs_MHz[good]
        amps = amps[good]

        f_dense = np.linspace(np.min(freqs_MHz), np.max(freqs_MHz), 1000)

        f0 = fit_result["f0_MHz"]
        kappa = fit_result["kappa_MHz"]
        depth = fit_result["depth"]
        baseline = fit_result["baseline"]
        slope = fit_result["slope"]

        fit_dense = baseline + slope * (f_dense - f0) - depth / (1.0 + 4.0 * ((f_dense - f0) / kappa) ** 2)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(freqs_MHz, amps, "o", markersize=4, label="Data")
        ax.plot(f_dense, fit_dense, linewidth=2, label="Lorentzian fit")
        ax.axvline(f0, linestyle="--", linewidth=1.5, label=f"$f_r$ = {f0:.4f} MHz")

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.set_title(f"Resonator {q_idx + 1} Fit")

        fit_text = f"$f_r$ = {f0:.4f} MHz\n$\\kappa/2\\pi$ = {kappa:.4f} MHz\n$Q_L$ = {fit_result['Q_loaded']:.0f}\n$\\tau_r$ = {fit_result['tau_us']:.4f} $\\mu$s"
        ax.text(0.03, 0.97, fit_text, transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.legend()
        ax.grid(alpha=0.25)
        plt.tight_layout()

        fit_plot_folder = os.path.join(self.plots_path, "resonator_fits")
        os.makedirs(fit_plot_folder, exist_ok=True)

        date_string = date.strftime("%Y-%m-%d_%H-%M-%S")
        save_name = f"Q{q_idx + 1}_res_fit_{date_string}_round{int(round_num)}_batch{int(batch_num)}.png"
        save_path = os.path.join(fit_plot_folder, save_name)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved resonator fit plot: {save_path}")

    def fit_resonator_lorentzian(self, freqs_MHz, amps, amp_errs=None):
        freqs_MHz = np.asarray(freqs_MHz, dtype=float)
        amps = np.asarray(amps, dtype=float)

        if amp_errs is not None:
            amp_errs = np.asarray(amp_errs, dtype=float)
            good = np.isfinite(freqs_MHz) & np.isfinite(amps) & np.isfinite(amp_errs) & (amp_errs > 0)
            amp_errs = amp_errs[good]
        else:
            good = np.isfinite(freqs_MHz) & np.isfinite(amps)

        freqs_MHz = freqs_MHz[good]
        amps = amps[good]

        if len(freqs_MHz) < 6:
            return None

        def lorentzian_dip(f, f0, kappa_MHz, depth, baseline, slope):
            return baseline + slope * (f - f0) - depth / (1.0 + 4.0 * ((f - f0) / kappa_MHz) ** 2)

        min_idx = np.argmin(amps)
        f0_guess = freqs_MHz[min_idx]

        n_edge = max(2, len(amps) // 10)
        edge_amps = np.concatenate([amps[:n_edge], amps[-n_edge:]])
        baseline_guess = np.median(edge_amps)
        depth_guess = baseline_guess - amps[min_idx]

        if depth_guess <= 0:
            return None

        half_level = baseline_guess - depth_guess / 2.0
        below_half = np.where(amps < half_level)[0]

        if len(below_half) >= 2:
            kappa_guess = freqs_MHz[below_half[-1]] - freqs_MHz[below_half[0]]
        else:
            kappa_guess = 0.1 * (np.max(freqs_MHz) - np.min(freqs_MHz))

        freq_step = np.median(np.diff(np.sort(freqs_MHz)))
        sweep_width = np.max(freqs_MHz) - np.min(freqs_MHz)

        if kappa_guess <= 0:
            kappa_guess = 5 * freq_step

        if amp_errs is None:
            def cost(f0, kappa_MHz, depth, baseline, slope):
                model = lorentzian_dip(freqs_MHz, f0, kappa_MHz, depth, baseline, slope)
                return np.sum((amps - model) ** 2)
        else:
            def cost(f0, kappa_MHz, depth, baseline, slope):
                model = lorentzian_dip(freqs_MHz, f0, kappa_MHz, depth, baseline, slope)
                return np.sum(((amps - model) / amp_errs) ** 2)

        m = Minuit(cost, f0=f0_guess, kappa_MHz=kappa_guess, depth=depth_guess, baseline=baseline_guess, slope=0.0)

        m.limits["f0"] = (np.min(freqs_MHz), np.max(freqs_MHz))
        m.limits["kappa_MHz"] = (freq_step / 2.0, sweep_width)
        m.limits["depth"] = (0.0, None)

        m.errordef = Minuit.LEAST_SQUARES
        m.migrad()
        m.hesse()

        if not m.valid:
            return None

        f0_fit = m.values["f0"]
        kappa_MHz = m.values["kappa_MHz"]
        depth = m.values["depth"]
        baseline = m.values["baseline"]
        slope = m.values["slope"]

        f0_err = m.errors["f0"]
        kappa_err = m.errors["kappa_MHz"]

        fit_amps = lorentzian_dip(freqs_MHz, f0_fit, kappa_MHz, depth, baseline, slope)
        residuals = amps - fit_amps
        rms_residual = np.sqrt(np.mean(residuals ** 2))
        dip_snr = depth / rms_residual if rms_residual > 0 else np.inf

        Q_loaded = f0_fit / kappa_MHz
        tau_us = 1.0 / (2.0 * np.pi * kappa_MHz)

        if amp_errs is None:
            chi2 = np.nan
            reduced_chi2 = np.nan
        else:
            chi2 = np.sum((residuals / amp_errs) ** 2)
            ndof = len(amps) - len(m.parameters)
            reduced_chi2 = chi2 / ndof if ndof > 0 else np.nan

        return {"f0_MHz": f0_fit, "f0_err_MHz": f0_err, "kappa_MHz": kappa_MHz, "kappa_err_MHz": kappa_err,
                "Q_loaded": Q_loaded, "tau_us": tau_us, "depth": depth, "baseline": baseline, "slope": slope,
                "dip_snr": dip_snr, "rms_residual": rms_residual, "chi2": chi2, "reduced_chi2": reduced_chi2,
                "fit_amps": fit_amps}

    def run(self,exp_extension='_ge', fit_resonators=False):
        # fit_resonators is to get kappa. No need to fit if you just want the resonator frequencies.
        import datetime
        # ----------Load/get data------------------------
        resonator_centers = {i: [] for i in range(self.number_of_qubits)}
        date_times = {i: [] for i in range(self.number_of_qubits)}
        resonator_fit_results = {i: [] for i in range(self.number_of_qubits)} # only used when fit_resonators = True

        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                timestamp_dir = os.path.join(self.base_data_path, folder_date)
                if "run6" in self.run_name:  # For QUIET, science run data was saved differently
                    if "ge_round_robin_presciencerun_data" in folder_date:
                        outerFolder = timestamp_dir + "/study_data/"  # where data is stored
                    else:
                        outerFolder = timestamp_dir + "/optimization/"  # where data is stored
                else:
                    outerFolder = timestamp_dir + "/study_data/"  # where data is stored
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # ------------------------------------------Load/Plot/Save Res Spec------------------------------------
            outerFolder_expt = outerFolder + f"/Data_h5/res{exp_extension}/"

            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                #H5_class_instance.print_h5_contents(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=f'res{exp_extension}', save_r=int(save_round))


                # just look at this resonator data, should have batch_num of arrays in each one
                # right now the data writes the same thing batch_num of times, so it will do the same 5 datasets 5 times, until you fix this just grab the first one (All 5)
                for q_key in load_data[f'res{exp_extension}']:
                    # print("all batch_num datasets------------------------", load_data['Res'][q_key].get('Amps', [])[0])
                    # print("one dataset------------------------",load_data['Res'][q_key].get('Amps', [])[0][0].decode())
                    # go through each dataset in the batch and plot

                    # Run 9 patch, accidentally took punched out data for Q4, bad.
                    if ("AB_paper_data_batch1_25dB_DACatten_noQ5/2026-04-17_00-34-47" in folder_date
                            and int(q_key) == 3):
                        print(f"Skipping Q4 data due to punchout in {self.run_name}/{folder_date}")
                        continue
                    # Run 9c patch: Skip Q5 for run 9c, data is not usable or trustworthy due to strong TLS effects
                    if (int(q_key) == 4 and "run9c" in str(self.run_name).replace("\\", "/")):
                        print(
                            f"Skipping Q5 for run 9c in {self.run_name}. It is not usable/trustworthy due to TLS effects.",
                            flush=True)
                        continue
                    for dataset in range(len(load_data[f'res{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'res{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue

                        date = datetime.datetime.fromtimestamp(
                            load_data[f'res{exp_extension}'][q_key].get('Dates', [])[0][dataset])  # single date per dataset

                        freq_pts = self.process_h5_data(load_data[f'res{exp_extension}'][q_key].get('freq_pts', [])[0][
                                                       dataset].decode())  # comes in as an array but put into a byte string, need to convert to list
                        freq_center = self.process_h5_data(load_data[f'res{exp_extension}'][q_key].get('freq_center', [])[0][
                                                          dataset].decode())  # comes in as an array but put into a string, need to convert to list
                        freqs_found = self.string_to_float_list(load_data[f'res{exp_extension}'][q_key].get('Found Freqs', [])[0][
                                                               dataset].decode())  # comes in as a list of floats in string format, need to convert
                        amps = self.process_string_of_nested_lists(load_data[f'res{exp_extension}'][q_key].get('Amps', [])[0][dataset].decode())  # list of lists
                        round_num = load_data[f'res{exp_extension}'][q_key].get('Round Num', [])[0][dataset]  # already a float
                        batch_num = load_data[f'res{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]

                        try:
                            exp_config = load_data[f'res{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config = None

                        if len(freq_pts) > 0:
                            res_class_instance = ResonanceSpectroscopy(q_key, self.number_of_qubits, self.plots_path, round_num, self.save_figs)
                            #res_spec_cfg = exp_config['res_spec']
                            res_freqs = res_class_instance.get_results(freq_pts, freq_center, amps)

                            resonator_centers[q_key].extend([res_freqs[q_key]])
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            if fit_resonators:
                                q_idx = int(q_key)
                                absolute_freqs = np.asarray(freq_pts, dtype=float) + float(freq_center[q_idx])
                                fit_result = self.fit_resonator_lorentzian(absolute_freqs, amps[q_idx])

                                if fit_result is not None:
                                    fit_result["date"] = date.strftime("%Y-%m-%d %H:%M:%S")
                                    fit_result["round_num"] = round_num
                                    fit_result["batch_num"] = batch_num
                                    resonator_fit_results[q_idx].append(fit_result)

                                    print(f"Q{q_idx + 1}: f0 = {fit_result['f0_MHz']:.6f} MHz, kappa/2pi = {fit_result['kappa_MHz']:.6f} MHz, QL = {fit_result['Q_loaded']:.0f}, tau = {fit_result['tau_us']:.6f} us")

                                    self.plot_resonator_fit(absolute_freqs, amps[q_idx], fit_result, q_idx, date, round_num, batch_num)
                                else:
                                    print(f"Q{q_idx + 1}: resonator fit failed.")

                            del res_class_instance

                del H5_class_instance
        if fit_resonators:
            return date_times, resonator_centers, resonator_fit_results

        return date_times, resonator_centers

    def plot(self, date_times, resonator_centers, show_legends, exp_extension = ''):
        #---------------------------------plot-----------------------------------------------------
        self.create_folder_if_not_exists(self.plots_path)
        analysis_folder = os.path.join(self.plots_path, "features_vs_time/")
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        colors = ['orange','blue','purple','green','brown','pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        ext = exp_extension.split('_')[0]
        plt.title(f'Resonator centers vs Time {ext}',fontsize = font)
        axes = axes.flatten()
        titles = [f"Res {i + 1}" for i in range(self.number_of_qubits)]
        from datetime import datetime
        for i, ax in enumerate(axes):

            ax.set_title(titles[i], fontsize = font)

            x = date_times[i]
            y = resonator_centers[i]

            # Convert strings to datetime objects.
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            # Combine datetime objects and y values into a list of tuples and sort by datetime.
            combined = list(zip(datetime_objects, y))

            if len(combined) == 0:
                # If this qubit has no data, just skip
                ax.set_visible(False)
                continue

            combined.sort(reverse=True, key=lambda x: x[0])

            # Unpack them back into separate lists, in order from latest to most recent.
            sorted_x, sorted_y = zip(*combined)
            ax.scatter(sorted_x, sorted_y, color=colors[i])

            sorted_x = np.asarray(sorted(x))

            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

            # Set new x-ticks using the datetime objects at the selected indices
            ax.set_xticks(sorted_x[indices])
            ax.set_xticklabels([dt for dt in sorted_x[indices]], rotation=45)

            ax.scatter(x, y, color=colors[i])
            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time (Days)', fontsize=font-2)
            ax.set_ylabel('Resonator Center (MHz)', fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + f'Res_Centers{exp_extension}.pdf', transparent=True, dpi=self.final_figure_quality)
        print('Plot saved to:', analysis_folder)

        #plt.show()
