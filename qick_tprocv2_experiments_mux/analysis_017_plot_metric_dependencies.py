import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pytz

class PlotMetricDependencies:
    def __init__(self,run_name, number_of_qubits, final_figure_quality, fridge):
        self.run_name = run_name
        self.number_of_qubits = number_of_qubits
        self.final_figure_quality = final_figure_quality
        self.fridge = fridge

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime

    def plot(self, date_times_1, metric_1, date_times_2, metric_2, metric_1_label, metric_2_label):
        analysis_folder = f"/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/RR_metrics/Plots/metrics_vs_eachother"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f'{metric_1_label} vs {metric_2_label}', fontsize=font)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            ax.set_title(titles[i], fontsize=font)

            # Parse the timestamp strings into datetime objects for both metrics.
            datetime_objects_1 = [datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") for dt in date_times_1[i]]
            datetime_objects_2 = [datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") for dt in date_times_2[i]]

            # Combine and sort the timestamp-metric pairs by time (most recent first)
            combined_1 = list(zip(datetime_objects_1, metric_1[i]))
            combined_2 = list(zip(datetime_objects_2, metric_2[i]))
            combined_1.sort(reverse=True, key=lambda item: item[0])
            combined_2.sort(reverse=True, key=lambda item: item[0])
            sorted_date_times_1, sorted_metric_1 = zip(*combined_1)
            sorted_date_times_2, sorted_metric_2 = zip(*combined_2)

            # Convert lists to numpy arrays (this helps with numerical operations)
            sorted_date_times_1 = np.array(sorted_date_times_1)
            sorted_date_times_2 = np.array(sorted_date_times_2)
            sorted_metric_1 = np.array(sorted_metric_1)
            sorted_metric_2 = np.array(sorted_metric_2)

            # Use the first metrics timestamps as the reference.
            ref_times = sorted_date_times_1
            ref_metrics = sorted_metric_1
            other_times = sorted_date_times_2
            other_metrics = sorted_metric_2

            # For each timestamp in the reference metric, find the closest timestamp in the other metric.
            matched_ref_metrics = []
            matched_other_metrics = []

            used_indices = set()

            for t_ref, m_ref in zip(ref_times, ref_metrics):
                # Compute time differences only with unused indices
                valid_indices = [j for j in range(len(other_times)) if j not in used_indices]
                if not valid_indices:
                    break  # No more unmatched points in other_times

                diffs = np.abs(np.array([(t_ref - other_times[j]).total_seconds() for j in valid_indices]))
                idx_closest = valid_indices[np.argmin(diffs)]

                matched_ref_metrics.append(m_ref)
                matched_other_metrics.append(other_metrics[idx_closest])
                used_indices.add(idx_closest)

            matched_ref_metrics = np.array(matched_ref_metrics)
            matched_other_metrics = np.array(matched_other_metrics)

            ax.scatter(matched_ref_metrics, matched_other_metrics, color='blue')
            ax.set_xlabel(metric_1_label, fontsize=font - 2)
            ax.set_ylabel(metric_2_label, fontsize=font - 2)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(analysis_folder, f'{metric_1_label}_vs_{metric_2_label}_correlation.png')
        plt.savefig(save_path, transparent=True, dpi=self.final_figure_quality)
        print('Plot saved to: ', analysis_folder)
        # plt.show()

    def plot_shared_datetimes(self, date_times, metric_1, metric_2, metric_1_label, metric_2_label):
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime

        # Create folders if they do not exist.
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/metric_interdependencies/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        # Assuming self.number_of_qubits defines the number of subplots (here, 6)
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f'{metric_1_label} vs {metric_2_label}', fontsize=font)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            ax.set_title(titles[i], fontsize=font)

            # Parse the timestamp strings into datetime objects
            datetime_objects = [datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") for dt in date_times[i]]

            # Since both metrics share the same timestamps, zip them together.
            # Sorting here is optional depending on whether you want the points in a certain order.
            combined = list(zip(datetime_objects, metric_1[i], metric_2[i]))
            # Sort by time (most recent first); change reverse to False if you prefer ascending order.
            combined.sort(reverse=True, key=lambda item: item[0])
            sorted_date_times, sorted_metric_1, sorted_metric_2 = zip(*combined)

            # Convert lists to numpy arrays (optional, but useful for numerical work)
            sorted_metric_1 = np.array(sorted_metric_1)
            sorted_metric_2 = np.array(sorted_metric_2)

            # Scatter plot of the two metrics (timestamps are used for sorting only)
            ax.scatter(sorted_metric_1, sorted_metric_2, color='blue')
            ax.set_xlabel(metric_1_label, fontsize=font - 2)
            ax.set_ylabel(metric_2_label, fontsize=font - 2)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = analysis_folder + f'{metric_1_label}_vs_{metric_2_label}_correlation.pdf'
        plt.savefig(save_path, transparent=True, dpi=self.final_figure_quality)
        # plt.show()

    def scatter_plot_two_y_axis(self, date_times_1, metric_1, date_times_2, metric_2, metric_1_label, metric_2_label):

        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/metric_interdependencies/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f'{metric_1_label} and {metric_2_label} over time', fontsize=font)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            ax.set_title(titles[i], fontsize=font)
            print(f"Qubit {i + 1} date_times_1: {date_times_1[i]}")
            print(f"Qubit {i + 1} date_times_2: {date_times_2[i]}")
            datetime_objects_1 = [datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") for dt in date_times_1[i]]
            datetime_objects_2 = [datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") for dt in date_times_2[i]]

            sorted_date_times_1, sorted_metric_1 = zip(
                *sorted(zip(datetime_objects_1, metric_1[i]), key=lambda x: x[0]))
            sorted_date_times_2, sorted_metric_2 = zip(
                *sorted(zip(datetime_objects_2, metric_2[i]), key=lambda x: x[0]))

            ax.scatter(sorted_date_times_1, sorted_metric_1, label=metric_1_label, color='blue')
            ax.set_xlabel("Datetime", fontsize=font - 2)
            ax.set_ylabel(metric_1_label, fontsize=font - 2, color='blue')
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
            ax.tick_params(axis='y', labelcolor='blue')

            ax2 = ax.twinx()
            ax2.scatter(sorted_date_times_2, sorted_metric_2, label=metric_2_label, color='red')
            ax2.set_ylabel(metric_2_label, fontsize=font - 2, color='red')
            ax2.tick_params(axis='y', labelcolor='red')

        plt.tight_layout()
        save_path = analysis_folder + f'{metric_1_label}_and_{metric_2_label}_timeseries.pdf'
        plt.savefig(save_path, transparent=True, dpi=self.final_figure_quality)
        # plt.show()

    # -------------------------------- Single scatter plot for two metrics ---------------------------------------------
    def plot_single_pair(self, date_times_1, metric_1, date_times_2, metric_2,
                         metric_1_label="Qubit 1 T1", metric_2_label="Qubit 3 T1"):
        """
        Creates a SINGLE scatter plot of metric_1 vs metric_2,
        matching data points by the same 'closest timestamp' logic (no repeated matches),
        picking the shorter list as reference.
        """
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/RR_metrics/Plots/metrics_vs_eachother"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/correlations_singleplots/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        # Convert timestamps to datetime objects
        datetime_objects_1 = [datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S") for dt_str in date_times_1]
        datetime_objects_2 = [datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S") for dt_str in date_times_2]

        # Zip and sort by time descending
        combined_1 = list(zip(datetime_objects_1, metric_1))
        combined_2 = list(zip(datetime_objects_2, metric_2))
        combined_1.sort(reverse=True, key=lambda item: item[0])
        combined_2.sort(reverse=True, key=lambda item: item[0])

        # Unzip
        sorted_date_times_1, sorted_metric_1 = zip(*combined_1) if combined_1 else ([], [])
        sorted_date_times_2, sorted_metric_2 = zip(*combined_2) if combined_2 else ([], [])

        sorted_date_times_1 = np.array(sorted_date_times_1)
        sorted_metric_1 = np.array(sorted_metric_1)
        sorted_date_times_2 = np.array(sorted_date_times_2)
        sorted_metric_2 = np.array(sorted_metric_2)

        # Pick the shorter array as reference
        if len(sorted_date_times_1) <= len(sorted_date_times_2):
            ref_times = sorted_date_times_1
            ref_metrics = sorted_metric_1
            other_times = sorted_date_times_2
            other_metrics = sorted_metric_2
            x_label = metric_1_label
            y_label = metric_2_label
        else:
            ref_times = sorted_date_times_2
            ref_metrics = sorted_metric_2
            other_times = sorted_date_times_1
            other_metrics = sorted_metric_1
            x_label = metric_2_label
            y_label = metric_1_label

        # Match with no repeated points
        matched_ref = []
        matched_other = []
        used_indices = set()

        for t_ref, m_ref in zip(ref_times, ref_metrics):
            valid_indices = [j for j in range(len(other_times)) if j not in used_indices]
            if not valid_indices:
                break

            diffs = np.abs(np.array([(t_ref - other_times[j]).total_seconds() for j in valid_indices]))
            idx_closest = valid_indices[np.argmin(diffs)]

            matched_ref.append(m_ref)
            matched_other.append(other_metrics[idx_closest])
            used_indices.add(idx_closest)

        matched_ref = np.array(matched_ref)
        matched_other = np.array(matched_other)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(matched_ref, matched_other, color='blue')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{metric_1_label} vs {metric_2_label}")

        plt.tight_layout()

        filename = f"{metric_1_label}_vs_{metric_2_label}_correlation.png".replace(' ', '_').replace('(', '').replace(
            ')', '')
        save_path = os.path.join(analysis_folder, filename)
        plt.savefig(save_path, transparent=False, dpi=self.final_figure_quality)
        print('Plot saved at:', save_path)

    #----------------------------------Plots T1 vs. Time  and other metrics vs time if data is provided --------------------------------------------
    def plot_q1_temp_and_t1( #works for any qubit, just provide the data corresponding to the qubit you want
            self,
            q1_t1_times, #T1 data
            q1_t1_vals,
            q1_temp_times = None,
            q1_temps = None, #Qubit temperature data
            temp_label="mK",
            t1_label="T1 (µs)",
            magcan_dates=None,
            magcan_temps=None, #Magnetic shield can temperature data
            magcan_label="mK",
            mcp2_dates=None, #MCP2 temperature data
            mcp2_temps=None,
            mcp2_label="mK",
            Q1_freqs=None,
            Q1_dates_spec=None, #Qubit frequency data
            qspec_label="Q1 Frequency (MHz)",
            date_times_pi_amps_Q1 = None,
            pi_amps_Q1 = None, #Pi Amp data
            pi_amps_label = "Pi Amp (a.u.)"):
        """
        Plots Qubit 1's T1 vs.time and other metrics vs time in the same plot,
        Also plots qubit temp data, fridge thermometry data, Pi Amp, and qubit frequency data during this time frame IF provided.
        """

        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        print(self.run_name)
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/correlations_singleplots/"
        self.create_folder_if_not_exists(analysis_folder)

        # If timestamps are strings, convert them to datetime objects
        def ensure_datetime(ts_list):
            if not ts_list:
                return []
            central = pytz.timezone("America/Chicago")
            if isinstance(ts_list[0], str):
                return [central.localize(datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")) for t_str in ts_list]
            elif isinstance(ts_list[0], datetime):
                # Convert existing datetime objects to Central Time
                return [dt.astimezone(central) for dt in ts_list]
            return ts_list

        q1_t1_times_dt = ensure_datetime(q1_t1_times)

        if q1_temp_times is not None:
            q1_temp_times_dt = ensure_datetime(q1_temp_times)
        else:
            q1_temp_times_dt = []

        if mcp2_dates is not None:
            mcp2_dates_dt = ensure_datetime(mcp2_dates)
        else:
            mcp2_dates_dt = []

        if magcan_dates is not None:
            magcan_dates_dt = ensure_datetime(magcan_dates)
        else:
            magcan_dates_dt = []

        if Q1_dates_spec is not None:
            Q1_dates_spec_dt = ensure_datetime(Q1_dates_spec)
        else:
            Q1_dates_spec_dt = []

        if date_times_pi_amps_Q1 is not None:
            date_times_pi_amps_Q1_dt = ensure_datetime(date_times_pi_amps_Q1)
        else:
            date_times_pi_amps_Q1_dt = []

        # figure
        fig, ax1 = plt.subplots(figsize=(14, 6))

        #Plots T1 data on ax1 (left y-axis)
        ax1.scatter(q1_t1_times_dt, q1_t1_vals, color='blue', alpha=0.8, label=t1_label, edgecolor='black')
        ax1.set_xlabel("Time")
        ax1.set_ylabel(t1_label, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        ############################# This sets a limit on the x axis if you only want to look at specific dates
        year = 2024
        month = 12
        day1 = 14
        day2 = 15
        start_time = datetime(year, month, day1, 20, 0, 0, tzinfo=pytz.timezone("America/Chicago"))
        end_time = datetime(year, month, day2, 14, 0, 0, tzinfo=pytz.timezone("America/Chicago"))
        ax1.set_xlim(start_time, end_time)  # Set the x-axis limits
        #############################

        ############################# use these limits to see cooldown data only, (without this it shows quiet thermometry data for the entire run)
        # year = 2024
        # month = 12
        # day1 = 9
        # day2 = 20
        # start_time = datetime(year, month, day1, 8, 0, 0, tzinfo=pytz.timezone("America/Chicago"))
        # end_time = datetime(year, month, day2, 12, 0, 0, tzinfo=pytz.timezone("America/Chicago"))
        # ax1.set_xlim(start_time, end_time)  # Set the x-axis limits
        #############################

        ############################# use these limits to see warmup data only
        # year = 2024
        # month = 12
        # day1 = 20
        # day2 = 20
        # start_time = datetime(year, month, day1, 10, 0, 0, tzinfo=pytz.timezone("America/Chicago"))
        # end_time = datetime(year, month, day2, 19, 0, 0, tzinfo=pytz.timezone("America/Chicago"))
        # ax1.set_xlim(start_time, end_time)  # Set the x-axis limits
        #############################

        # Check if we have ANY temperature data to plot on ax2
        has_qubit_temp = bool(q1_temp_times_dt and q1_temps)
        has_mcp2_temp = bool(mcp2_dates_dt and mcp2_temps)
        has_magcan_temp = bool(magcan_dates_dt and magcan_temps)
        ax2 = None
        if has_qubit_temp or has_mcp2_temp or has_magcan_temp:
            # Create ax2 only if we actually have data for it
            ax2 = ax1.twinx()

        #Plots qubit temperature data on ax2 (right y-axis)
        if q1_temp_times_dt and q1_temps:
            ax2.scatter(q1_temp_times_dt, q1_temps, color='red', alpha=0.8, label=temp_label, edgecolor='black')
            ax2.set_ylabel("Temperature (mK)", color='black')
            ax2.tick_params(axis='y', labelcolor='black')

        # plot thermometry data if available on ax2
        if mcp2_dates_dt and mcp2_temps:
            ax2.scatter(
                mcp2_dates_dt, mcp2_temps,
                color='orange', alpha=0.8, label=mcp2_label
            )
        if magcan_dates_dt and magcan_temps:
            ax2.scatter(
                magcan_dates_dt, magcan_temps,
                color='green', alpha=0.8, label=magcan_label
            )

        # Plots frequency data on ax3
        ax3 = None
        if Q1_dates_spec_dt and Q1_freqs:
            ax3 = ax1.twinx()  # third y-axis (right side) for frequency
            ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis
            ax3.scatter(Q1_dates_spec_dt, Q1_freqs, color='orchid', alpha=0.8, label=qspec_label, edgecolor='black')
            ax3.set_ylabel("Frequency (MHz)", color='purple')
            ax3.tick_params(axis='y', labelcolor='purple')

        # Plots frequency data on ax4
        ax4 = None
        if date_times_pi_amps_Q1_dt and pi_amps_Q1:
            ax4 = ax1.twinx()  # fourth y-axis (right side) for Pi Amp (a.u.)
            ax4.spines["right"].set_position(("outward", 120))  # Offset the third axis
            ax4.scatter(date_times_pi_amps_Q1_dt, pi_amps_Q1, color='gold', alpha=0.8, label=pi_amps_label, edgecolor='black')
            ax4.set_ylabel("Pi Amp (a.u.)", color='goldenrod')
            ax4.tick_params(axis='y', labelcolor='goldenrod')

        plt.title(" Qubit 1 run5a: T1 , Freq, Temperature, and Pi Amp vs. Time")

        # Collect legend info from ax1 and ax2
        handles1, labels1 = ax1.get_legend_handles_labels() #always provided
        handles2, labels2 = ax2.get_legend_handles_labels() if ax2 is not None else ([], []) #if provided
        handles3, labels3 = ax3.get_legend_handles_labels() if ax3 is not None else ([], []) #if provided
        handles4, labels4 = ax4.get_legend_handles_labels() if ax4 is not None else ([], []) #if provided

        # Make one combined legend on ax1 (or plt)
        ax1.legend(handles1 + handles2 + handles3 + handles4, labels1 + labels2 + labels3 + labels4, loc='upper left', fontsize=10, markerscale=0.8, handletextpad=0.6, borderpad=0.7, labelspacing=0.4)

        fig.tight_layout()

        unique_str = datetime.now().strftime('%Y%m%d%H%M%S')
        plot_filename = os.path.join(analysis_folder, f"Q1_Temp_and_T1_vs_Time_updatedcooldown_closeup_{unique_str}.png")
        plt.savefig(plot_filename, transparent=False, dpi=self.final_figure_quality)
        #plt.show()
        plt.close()

    def plot_autocorrelation(self, times, metric_values, label, qubit_index):
        """
        Plots the autocorrelation of a metric (like T1 or Qubit Frequency) for a given qubit.
        Parameters:
        - times: list of datetime objects (or strings that can be parsed)
        - values: list or np.array of metric values (floats)
        - label: string, label for what you're autocorrelating (e.g., "T1 (us)" or "Qubit Frequency (MHz)")
        - qubit_index: int, which qubit it is (for title/saving)
        """
        analysis_folder = f"/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/RR_metrics/Plots/Autocorrelations"
        os.makedirs(analysis_folder, exist_ok=True)

        values = np.array(metric_values) #input T1 values or Qfreq values, etc

        # Manual method
        # # subtract mean
        # values_centered = values - np.mean(values) #since we are just interested in the fluctuations in the data over time
        # # autocorrelation
        # autocorr = np.correlate(values_centered, values_centered, mode='full')
        # autocorr = autocorr[autocorr.size // 2:]  # Take only positive lags
        # autocorr /= autocorr[0]  # Normalize to initial val
        # # create lags
        # lags = np.arange(len(autocorr))
        # # plot
        # plt.figure(figsize=(8, 5))
        # plt.plot(lags, autocorr, marker='o')
        # plt.title(f"Autocorrelation of {label} (Qubit {qubit_index + 1})", fontsize=14)
        # plt.xlabel("Lag (number of points)", fontsize=12)
        # plt.ylabel("Autocorrelation", fontsize=12)
        # plt.grid(True)
        # plt.tight_layout()

        # Matplotlib method
        # Plot autocorrelation
        plt.figure(figsize=(8, 5))
        plt.acorr(values, maxlags=9)
        plt.title(f"Autocorrelation of {label} (Qubit {qubit_index + 1})", fontsize=14)
        plt.xlabel("Lag (number of points)", fontsize=12) #check label
        plt.ylabel("Autocorrelation", fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(analysis_folder,
                                 f"Autocorrelation_Qubit{qubit_index + 1}_{label.replace(' ', '_').replace('(', '').replace(')', '')}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Plot for Qubit {qubit_index + 1} saved to: {save_path}")