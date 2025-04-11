import h5py
import json
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
import numpy as np

class CompareRuns:
    def __init__(self, run_number_list, run_name_folder):
        self.run_number_list = run_number_list
        self.run_name_folder = run_name_folder

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def load_from_h5(self, filename):
        with h5py.File(filename, 'r') as hf:
            # Load attributes

            date_times_res_spec = json.loads(hf.attrs['date_times_res_spec'])
            res_freqs = json.loads(hf.attrs['res_freqs'])

            date_times_q_spec = json.loads(hf.attrs['date_times_q_spec'])
            q_freqs = json.loads(hf.attrs['q_freqs'])

            date_times_pi_amps = json.loads(hf.attrs['date_times_pi_amps'])
            pi_amps = json.loads(hf.attrs['pi_amp'])

            date_times_t1 = json.loads(hf.attrs['date_times_t1'])
            t1_vals = json.loads(hf.attrs['t1_vals'])
            t1_errs = json.loads(hf.attrs['t1_errs'])
            t1_std_values = json.loads(hf.attrs['t1_std_values'])
            t1_mean_values = json.loads(hf.attrs['t1_mean_values'])

            date_times_t2r = json.loads(hf.attrs['date_times_t2r'])
            t2r_vals = json.loads(hf.attrs['t2r_vals'])
            t2r_errs = json.loads(hf.attrs['t2r_errs'])
            t2r_std_values = json.loads(hf.attrs['t2r_std_values'])
            t2r_mean_values = json.loads(hf.attrs['t2r_mean_values'])

            date_times_t2e = json.loads(hf.attrs['date_times_t2e'])
            t2e_vals = json.loads(hf.attrs['t2e_vals'])
            t2e_errs = json.loads(hf.attrs['t2e_errs'])
            t2e_std_values = json.loads(hf.attrs['t2e_std_values'])
            t2e_mean_values = json.loads(hf.attrs['t2e_mean_values'])

            run_number = hf.attrs['run_number']
            run_notes = hf.attrs['run_notes']
            last_date = hf.attrs['last_date']

        return {
            'date_times_res_spec':date_times_res_spec,
            'res_freqs':res_freqs,
            'date_times_q_spec':date_times_q_spec,
            'q_freqs':q_freqs,
            'date_times_pi_amps':date_times_pi_amps,
            'pi_amps':pi_amps,
            'date_times_t1':date_times_t1,
            't1_vals': t1_vals,
            't1_errs': t1_errs,
            't1_std_values': t1_std_values,
            't1_mean_values': t1_mean_values,
            'run_number': run_number,
            'run_notes': run_notes,
            'last_date': last_date,
            'date_times_t2r': date_times_t2r,
            't2r_vals': t2r_vals,
            't2r_errs': t2r_errs,
            't2r_std_values': t2r_std_values,
            't2r_mean_values': t2r_mean_values,
            'date_times_t2e': date_times_t2e,
            't2e_vals': t2e_vals,
            't2e_errs': t2e_errs,
            't2e_std_values': t2e_std_values,
            't2e_mean_values': t2e_mean_values,
        }

    import math
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    def plot_decoherence_vs_run(self, skip_qubit_t2e=False, qubit_to_skip_t2e=None):
        import math
        if len(self.run_number_list) > 1:
            t1_data = {}
            t1_err = {}
            t2r_data = {}
            t2r_err = {}
            t2e_data = {}
            t2e_err = {}
            run_notes_all = []

            qubit_list = None

            # load data for each run
            for r in self.run_number_list:
                run_stats_folder = f"run_stats/QUIET/run{r}/"
                filename = run_stats_folder + 'experiment_data.h5'
                loaded_data = self.load_from_h5(filename)

                t1_means = loaded_data['t1_mean_values']
                t1_stds = loaded_data['t1_std_values']
                t2r_means = loaded_data['t2r_mean_values']
                t2r_stds = loaded_data['t2r_std_values']
                t2e_means = loaded_data['t2e_mean_values']
                t2e_stds = loaded_data['t2e_std_values']
                run_notes = loaded_data['run_notes']

                # on the first run, initialize lists for each qubit
                if qubit_list is None:
                    qubit_list = list(t1_means.keys())
                    for qb in qubit_list:
                        t1_data[qb] = []
                        t1_err[qb] = []
                        t2r_data[qb] = []
                        t2r_err[qb] = []
                        t2e_data[qb] = []
                        t2e_err[qb] = []

                # get the data for this run and append to list for saving and plotting
                for qb in qubit_list:
                    t1_data[qb].append(t1_means[qb])
                    t1_err[qb].append(t1_stds[qb])
                    t2r_data[qb].append(t2r_means[qb])
                    t2r_err[qb].append(t2r_stds[qb])
                    t2e_data[qb].append(t2e_means[qb])
                    t2e_err[qb].append(t2e_stds[qb])
                run_notes_all.append(run_notes)

            # Create a grid of subplots: 2 rows and ceil(qubit_count/2) columns
            n_qubits = len(qubit_list)
            if n_qubits == 1:
                fig, axes = plt.subplots(1, 1, sharex=True, figsize=(6, 4))
                axes = [axes]
                ncols = 1
                nrows = 1
            else:
                ncols = math.ceil(n_qubits / 2)
                nrows = 2
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(6 * ncols, 4 * nrows))
                axes = axes.flatten()

            x = self.run_number_list
            for i, qb in enumerate(qubit_list):
                ax = axes[i]
                ax.errorbar(x, t1_data[qb], yerr=t1_err[qb], fmt='o-', label='T1', capsize=3)
                if skip_qubit_t2e and i == qubit_to_skip_t2e:
                    print(f"Skipping t2r for Qubit {qubit_to_skip_t2e + 1}")
                    highest_points = []
                    for idx in range(len(x)):
                        highest_point = max(
                            t1_data[qb][idx] + t1_err[qb][idx],
                            t2e_data[qb][idx] + t2e_err[qb][idx]
                        )
                        highest_points.append(highest_point)
                else:
                    ax.errorbar(x, t2r_data[qb], yerr=t2r_err[qb], fmt='o-', label='T2R', capsize=3)
                    highest_points = []
                    for idx in range(len(x)):
                        highest_point = max(
                            t1_data[qb][idx] + t1_err[qb][idx],
                            t2r_data[qb][idx] + t2r_err[qb][idx],
                            t2e_data[qb][idx] + t2e_err[qb][idx]
                        )
                        highest_points.append(highest_point)
                ax.errorbar(x, t2e_data[qb], yerr=t2e_err[qb], fmt='o-', label='T2E', capsize=3)

                ax.set_ylabel('Time (µs)')
                ax.set_title(qb)
                ax.legend()
                ax.set_ylim(0, 85)

                n = 0
                # Annotate for each x-value with the corresponding note
                for idx, x_val in enumerate(x):
                    words = run_notes_all[idx].split()
                    plotting_note = '\n'.join(
                        ' '.join(words[i:i + 3]) for i in range(0, len(words), 3))
                    text_x_offset = x_val + 0.1 if n < 1 else x_val
                    n += 1
                    ax.annotate(
                        plotting_note,
                        xy=(text_x_offset, highest_points[idx]),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', alpha=1)
                    )

            # Turn off any unused subplots and set xlabel only for the bottom row
            total_axes = nrows * ncols
            for j in range(n_qubits, total_axes):
                axes[j].axis('off')

            for i, ax in enumerate(axes[:n_qubits]):
                # Determine the row index based on grid placement
                row_index = i // ncols
                if row_index == nrows - 1:
                    ax.set_xlabel('Run Number')
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                else:
                    ax.set_xlabel('')

            plt.tight_layout()
            analysis_folder = f"/data/QICK_data/{self.run_name_folder}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            plt.savefig(analysis_folder + 'compare_runs.pdf', dpi=500)

        elif len(self.run_number_list) == 1:
            # Only 1 data point per qubit
            r = self.run_number_list[0]
            run_stats_folder = f"run_stats/run{r}/"
            filename = run_stats_folder + 'experiment_data.h5'
            loaded_data = self.load_from_h5(filename)

            t1_means = loaded_data['t1_mean_values']
            t1_stds = loaded_data['t1_std_values']
            t2r_means = loaded_data['t2r_mean_values']
            t2r_stds = loaded_data['t2r_std_values']
            t2e_means = loaded_data['t2e_mean_values']
            t2e_stds = loaded_data['t2e_std_values']
            run_notes = loaded_data['run_notes']

            qubit_list = list(t1_means.keys())
            x = [r]

            n_qubits = len(qubit_list)
            if n_qubits == 1:
                fig, axes = plt.subplots(1, 1, sharex=True, figsize=(6, 4))
                axes = [axes]
                ncols = 1
                nrows = 1
            else:
                ncols = math.ceil(n_qubits / 2)
                nrows = 2
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(6 * ncols, 4 * nrows))
                axes = axes.flatten()

            for i, qb in enumerate(qubit_list):
                ax = axes[i]
                ax.errorbar(x, [t1_means[qb]], yerr=[t1_stds[qb]], fmt='o-', label='T1', capsize=3)
                ax.errorbar(x, [t2r_means[qb]], yerr=[t2r_stds[qb]], fmt='o-', label='T2R', capsize=3)
                ax.errorbar(x, [t2e_means[qb]], yerr=[t2e_stds[qb]], fmt='o-', label='T2E', capsize=3)

                ax.set_ylabel('Time (µs)')
                ax.set_title(qb)
                ax.legend()

                # Annotate the single point with the run note
                ax.annotate(
                    run_notes,
                    xy=(x[-1], t1_means[qb]),
                    xytext=(0, 20),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', alpha=0.7)
                )

            # Turn off any unused subplots and set xlabel only for the bottom row
            total_axes = nrows * ncols
            for j in range(n_qubits, total_axes):
                axes[j].axis('off')

            for i, ax in enumerate(axes[:n_qubits]):
                row_index = i // ncols
                if row_index == nrows - 1:
                    ax.set_xlabel('Run Number')
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                else:
                    ax.set_xlabel('')

            plt.tight_layout()
            analysis_folder = f"/data/QICK_data/{self.run_name_folder}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            plt.savefig(analysis_folder + 'compare_runs.pdf', dpi=500)

    def plot_decoherence_vs_qfreq(self):
        import math
        # If no runs are specified, do nothing
        if not self.run_number_list:
            print("No runs provided!")
            return

        n_plots = len(self.run_number_list)
        if n_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=False)
            axes = [axes]
            ncols = 1
            nrows = 1
        else:
            ncols = math.ceil(n_plots / 2)
            nrows = 2
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 6 * nrows), sharex=False)
            axes = axes.flatten()

        metric_markers = {
            'T1': 'o',
            'T2R': 's',
            'T2E': '^'
        }

        # Loop over each run
        for i, run_number in enumerate(self.run_number_list):
            ax = axes[i]

            # Load the data for this run
            run_stats_folder = f"run_stats/QUIET/run{run_number}/"
            filename = run_stats_folder + 'experiment_data.h5'
            loaded_data = self.load_from_h5(filename)

            # Extract the per-qubit arrays
            t1_vals_all = loaded_data['t1_vals']
            t2r_vals_all = loaded_data['t2r_vals']
            t2e_vals_all = loaded_data['t2e_vals']
            freq_vals_all = loaded_data['q_freqs']

            # Sort qubits for consistent ordering and build a colormap for qubits
            qubit_list = sorted(t1_vals_all.keys())
            cmap = plt.cm.get_cmap('tab10')
            qubit_colors = {qb: cmap(idx % 10) for idx, qb in enumerate(qubit_list)}

            # Loop over each qubit, compute medians, and scatter the three metrics
            for qb in qubit_list:
                freq_median = np.median(freq_vals_all[qb])
                t1_median = np.median(t1_vals_all[qb])
                t2r_median = np.median(t2r_vals_all[qb])
                t2e_median = np.median(t2e_vals_all[qb])

                ax.scatter(freq_median, t1_median, color=qubit_colors[qb], marker=metric_markers['T1'])
                ax.scatter(freq_median, t2r_median, color=qubit_colors[qb], marker=metric_markers['T2R'])
                ax.scatter(freq_median, t2e_median, color=qubit_colors[qb], marker=metric_markers['T2E'])

            ax.set_ylabel('Median Time (µs)')
            ax.set_title(f'Decoherence vs. Qubit Frequency (Run {run_number})')
            ax.xaxis.set_major_locator(MaxNLocator(integer=False))

            # Create dual legends
            qubit_legend_handles = [ax.scatter([], [], color=qubit_colors[qb], marker='o', label=qb)
                                    for qb in qubit_list]
            metric_legend_handles = [ax.scatter([], [], color='black', marker=metric_markers[metric], label=metric)
                                     for metric in metric_markers]

            qubit_legend = ax.legend(handles=qubit_legend_handles, title="Qubits",
                                     loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
            metric_legend = ax.legend(handles=metric_legend_handles, title="Metrics",
                                      loc="upper left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
            ax.add_artist(qubit_legend)

        # Turn off any unused subplots and set xlabel only for bottom row subplots
        total_axes = nrows * ncols
        for j in range(n_plots, total_axes):
            axes[j].axis('off')

        for i, ax in enumerate(axes[:n_plots]):
            row_index = i // ncols
            if row_index == nrows - 1:
                ax.set_xlabel('Median Qubit Frequency (MHz)')
            else:
                ax.set_xlabel('')

        plt.tight_layout()
        analysis_folder = f"/data/QICK_data/{self.run_name_folder}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        plt.savefig(analysis_folder + 'compare_runs_qfreq_vs_decoherence.pdf', dpi=500)
