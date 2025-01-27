import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import allantools


class AllanDevDemo:
    def __init__(self):
        # For illustration purposes only
        self.run_name = "6transmon_run5"
        self.final_figure_quality = 100
        self.number_of_qubits = 6

    def create_folder_if_not_exists(self, folder):
        """Dummy function: In real usage, ensure 'folder' is created if not present."""
        pass

    def plot_allan_deviation(self, date_times, t1_vals, show_legends=True):
        """
        Plot the Overlapping Allan Deviation of T1 fluctuations for each qubit
        in a 2x3 grid of subplots.
        """
        #-----------------------------------------------------------------------
        # 1) Create directories (dummy here) and set up figure
        #-----------------------------------------------------------------------
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)
        fig.suptitle('Overlapping Allan Deviation of T1 Fluctuations', fontsize=font)
        axes = axes.flatten()

        colors = ['orange','blue','purple','green','brown','pink']
        titles = [f"Qubit {i+1}" for i in range(self.number_of_qubits)]

        #-----------------------------------------------------------------------
        # 2) For each qubit, sort data by timestamp, compute Oadev, and plot
        #-----------------------------------------------------------------------
        for i, ax in enumerate(axes):
            # Hide extra subplots if you have fewer than 6 qubits
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            # Extract this qubit's data
            datetime_strings = date_times[i]  # list of "YYYY-MM-DD HH:MM:SS"
            t1_data = t1_vals[i]

            # Convert to datetime objects
            dt_objs = [datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s in datetime_strings]

            # Sort (ascending) by time
            combined = list(zip(dt_objs, t1_data))
            combined.sort(key=lambda x: x[0])  # sort by datetime
            sorted_times, sorted_t1 = zip(*combined)

            # Convert times -> seconds since first measurement
            t0 = sorted_times[0]
            time_sec = np.array([(t - t0).total_seconds() for t in sorted_times])
            t1_array = np.array(sorted_t1, dtype=float)

            # If you only have a single point, skip
            if len(time_sec) <= 1:
                ax.text(0.5, 0.5, "Not enough points", ha='center', va='center', transform=ax.transAxes)
                continue

            # Approx. average sample rate for Oadev
            avg_dt = np.mean(np.diff(time_sec))
            if avg_dt <= 0:
                avg_dt = 1.0
            rate = 1.0 / avg_dt

            # Compute overlapping Allan deviation
            # Use 'freq' data_type since T1 is not a phase measure.
            # We'll auto-select tau points with taus='decade' or you could supply np.logspace(...).
            taus_out, ad, ade, ns = allantools.oadev(
                t1_array,
                rate=rate,
                data_type='freq',
                taus='decade'
            )

            # Plot on log axes to mimic a standard Allan plot
            ax.set_xscale('log')
            ax.set_yscale('log')
            print(len(taus_out), len(ad))
            ax.plot(taus_out, ad, marker='o', color=colors[i], label=f"Qubit {i+1}")

            # Optional: plot error bars
            # ax.errorbar(taus_out, ad, yerr=ade, fmt='o', color=colors[i])

            if show_legends:
                ax.legend(loc='best', edgecolor='black')

            ax.set_xlabel(r"$\tau$ (s)", fontsize=font-2)
            ax.set_ylabel(r"$\sigma_{T1}(\tau)$ (µs)", fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        #-----------------------------------------------------------------------
        # 3) Finalize and save
        #-----------------------------------------------------------------------
        plt.tight_layout()
        plt.savefig(analysis_folder + 'T1_allan_deviation.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.show()  # or plt.close(fig) in real usage

    def plot_allan_deviation_uniform(self, t1_vals, uniform_dt=1.0, show_legends=True):
        """
        Treat each qubit's T1 data as if sampled at a constant interval 'uniform_dt'.
        No timestamps are used here. 't1_vals' is a dict {0: [...], 1: [...], ...}.
        """
        # -----------------------------------------------------------
        # Create directories (dummy in this example)
        # -----------------------------------------------------------
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        # -----------------------------------------------------------
        # Set up figure: 2x3 subplots for up to 6 qubits
        # -----------------------------------------------------------
        font = 14
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle('Allan Deviation of T1 (Assuming Uniform Time Steps)', fontsize=font)
        axes = axes.flatten()

        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]

        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:
                # Hide extra subplots if fewer than 6 qubits
                ax.set_visible(False)
                continue

            # Grab this qubit's data
            t1_data = t1_vals.get(i, [])
            # Make sure it is a normal float list (not np.float64 objects)
            t1_data = [float(x) for x in t1_data]

            ax.set_title(titles[i], fontsize=font)

            # If we have <2 data points, we can't do much
            if len(t1_data) < 2:
                ax.text(0.5, 0.5, "Not enough data", ha='center', va='center', transform=ax.transAxes)
                continue

            # Allantools 'oadev' - treat T1 as 'freq' data at a uniform rate = 1/dt
            rate = 1.0 / uniform_dt

            # Use either 'decade' or a custom tau array
            taus_out, ad, ade, ns = allantools.oadev(
                data=t1_data,
                rate=rate,
                data_type='freq',
                taus='decade'
            )

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.plot(taus_out, ad, marker='o', color=colors[i], label=f"Qubit {i + 1}")

            if show_legends:
                ax.legend(loc='best', edgecolor='black')

            ax.set_xlabel(r"$\tau$ (s)", fontsize=font - 2)
            ax.set_ylabel(r"$\sigma_{T1}(\tau)$ (µs)", fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'T1_allan_deviation_uniform.pdf',
                    transparent=True, dpi=self.final_figure_quality)
        plt.show()


def generate_dummy_data(num_qubits=6, num_points=100):
    """
    Create a dictionary of date_times and T1 values that are non-uniformly spaced
    for demonstration.  Each qubit has 'num_points' samples, spaced ~5 minutes apart
    with random jitters.
    """
    base_time = datetime(2025,1,1,12,0,0)  # arbitrary start
    date_times = {}
    t1_vals    = {}

    for q in range(num_qubits):
        dt_list = []
        t1_list = []
        current_time = base_time
        for _ in range(num_points):
            # Convert current_time to string
            dt_list.append(current_time.strftime("%Y-%m-%d %H:%M:%S"))
            # Make up a T1 value (µs) that fluctuates around ~ 200 µs
            fluct = np.random.normal(loc=0.0, scale=10.0)  # random noise
            t1_value = 200.0 + fluct
            t1_list.append(t1_value)

            # Advance current_time by ~5 minutes + random offset
            delta_minutes = 5 + np.random.rand()  # e.g. 5–6 minutes
            current_time = current_time + timedelta(minutes=delta_minutes)

        date_times[q] = dt_list
        t1_vals[q]    = t1_list

    return date_times, t1_vals



