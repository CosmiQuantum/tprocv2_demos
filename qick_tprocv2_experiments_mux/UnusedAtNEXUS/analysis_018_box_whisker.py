import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pytz
import seaborn as sns
import pandas as pd

class PlotBoxWhisker:
    def __init__(self,run_name, number_of_qubits, final_figure_quality):
        self.run_name = run_name
        self.number_of_qubits = number_of_qubits
        self.final_figure_quality = final_figure_quality

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def plot(self, metric_1, metric_label="T1 (µs)"):
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)

        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/metric_boxplots/"
        self.create_folder_if_not_exists(analysis_folder)

        fig, ax = plt.subplots(figsize=(8, 6))

        boxplot_data = []
        for qubit_index in range(self.number_of_qubits):
            data_for_qubit = metric_1[qubit_index]
            boxplot_data.append(data_for_qubit)

        ax.boxplot(boxplot_data, showfliers=True)

        ax.set_xticks(range(1, self.number_of_qubits + 1))
        ax.set_xticklabels([str(i+1) for i in range(self.number_of_qubits)], fontsize=12)
        ax.set_xlabel("Qubit Number", fontsize=14)
        ax.set_ylabel(metric_label, fontsize=14)
        ax.set_title(f"{metric_label} by Qubit", fontsize=16)
        ax.set_ylim(bottom=0)
        plt.tight_layout()

        outfile = f"{analysis_folder}boxplot_{metric_label.replace(' ', '_')}non_zero_suppressed.png"
        plt.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
        plt.close(fig)


    def plot_three_metrics(self, metric_1, metric_2, metric_3,
                           metric_labels=["T1 (µs)", "T2R (µs)", "T2E (µs)"], metric_colors=["skyblue", "lightgreen", "lightcoral"]):

        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/metric_boxplots/"
        self.create_folder_if_not_exists(analysis_folder)

        fig, ax = plt.subplots(figsize=(12, 6))

        all_boxplot_data = []
        positions = []
        group_centers = []
        for qubit_index in range(self.number_of_qubits):
            start_pos = qubit_index * 4 + 1
            all_boxplot_data.append(metric_1[qubit_index])
            all_boxplot_data.append(metric_2[qubit_index])
            all_boxplot_data.append(metric_3[qubit_index])
            positions.extend([start_pos, start_pos + 1, start_pos + 2])
            group_centers.append(start_pos + 1)

        bp = ax.boxplot(all_boxplot_data, positions=positions, widths=0.6,
                        showfliers=True, patch_artist=True)

        for i, box in enumerate(bp['boxes']):
            metric_index = i % 3
            box.set_facecolor(metric_colors[metric_index])
            box.set_edgecolor("black")

        for qubit_index in range(self.number_of_qubits - 1):
            separator = qubit_index * 4 + 3.5
            ax.axvline(x=separator, color='gray', linestyle='dotted')

        ax.set_xticks(group_centers)
        ax.set_xticklabels([str(i + 1) for i in range(self.number_of_qubits)], fontsize=12)
        ax.set_xlabel("Qubit Number", fontsize=14)

        ax.set_ylabel("Metric Values", fontsize=14)
        ax.set_title("Metrics by Qubit", fontsize=16)
        ax.set_ylim(bottom=0)

        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='s', color='black',
                                  markerfacecolor=metric_colors[i], linestyle='None',
                                  markersize=10, label=metric_labels[i])
                           for i in range(3)]
        ax.legend(handles=legend_elements, loc='best')

        plt.tight_layout()

        file_label = "_".join([label.replace(" ", "_") for label in metric_labels])
        outfile = f"{analysis_folder}boxplot_{file_label}_non_zero_suppressed.png"
        plt.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
        plt.close(fig)

    def plot_three_metrics_by_freq(self, q_freq_means, metric_1, metric_2, metric_3,
                                  metric_labels=["T1 (µs)", "T2R (µs)", "T2E (µs)"],
                                  metric_colors=["skyblue", "lightgreen", "lightcoral"]):
        import matplotlib.patches as mpatches

        # Create analysis folder if it doesn't exist
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/metric_boxplots/"
        self.create_folder_if_not_exists(analysis_folder)

        # Reorganize the data into a long-form DataFrame
        data = []
        for qubit_index in range(self.number_of_qubits):
            q_freq = q_freq_means[qubit_index]  # corresponding frequency mean for this qubit
            for metric, label in zip([metric_1, metric_2, metric_3], metric_labels):
                for value in metric[qubit_index]:
                    data.append({
                        "q_freq": q_freq,
                        "Metric": label,
                        "Value": value
                    })
        df = pd.DataFrame(data)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Group the data by q_freq and Metric
        grouped = df.groupby(['q_freq', 'Metric'])['Value'].apply(list).reset_index()

        # Prepare lists to hold box data, their positions, and colors.
        box_data = []
        positions = []
        colors = []

        # For each qubit frequency group, calculate an offset for each metric.
        n_metrics = len(metric_labels)
        # Offsets: e.g., for 3 metrics, offsets will be [-0.2, 0, +0.2]
        offsets = [(i - (n_metrics - 1) / 2) * 0.2 for i in range(n_metrics)]

        # Get unique frequency groups (sorted)
        unique_freqs = sorted(df['q_freq'].unique())
        for q in unique_freqs:
            for i, label in enumerate(metric_labels):
                # Get the list of values for the current q_freq and metric.
                values = grouped[(grouped['q_freq'] == q) & (grouped['Metric'] == label)]['Value']
                if not values.empty:
                    # Use the offset to place box plots within the group.
                    box_data.append(values.values[0])
                    positions.append(q + offsets[i])
                    colors.append(metric_colors[i])

        # Create the box plots at specified positions.
        bp = ax.boxplot(box_data, positions=positions, widths=15, patch_artist=True)

        # Color each box according to the metric.
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Set x-axis ticks at the centers (i.e. the original q_freq values)
        ax.set_xticks(unique_freqs)
        ax.set_xticklabels([f"{freq:.2f}" for freq in unique_freqs], fontsize=14)

        # Set axis labels and title with font size 14
        ax.set_xlabel("Qubit Frequency Mean (MHz)", fontsize=14)
        ax.set_ylabel("Coherence time (µs)", fontsize=14)
        ax.set_title("Coherence by Qubit Frequency", fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_ylim(bottom=0)

        # Create custom legend handles for the metric colors.
        handles = [mpatches.Patch(color=metric_colors[i], label=metric_labels[i]) for i in range(n_metrics)]
        ax.legend(handles=handles, fontsize=14)

        plt.tight_layout()

        # Save the figure
        file_label = "_".join([label.replace(" ", "_") for label in metric_labels])
        outfile = f"{analysis_folder}boxplot_{file_label}_qfreq.png"
        plt.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
        plt.close(fig)

    def plot_three_metrics_by_freq_x_break(self, q_freq_means, metric_1, metric_2, metric_3,
                                           metric_labels=["T1 (µs)", "T2R (µs)", "T2E (µs)"],
                                           metric_colors=["skyblue", "lightgreen", "lightcoral"],
                                           freq_gap_threshold=5):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import pandas as pd
        import numpy as np

        # Create analysis folder if it doesn't exist
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/metric_boxplots/"
        self.create_folder_if_not_exists(analysis_folder)

        # Reorganize the data into a long-form DataFrame
        data = []
        for qubit_index in range(self.number_of_qubits):
            q_freq = q_freq_means[qubit_index]  # corresponding frequency mean for this qubit
            for metric, label in zip([metric_1, metric_2, metric_3], metric_labels):
                for value in metric[qubit_index]:
                    data.append({
                        "q_freq": q_freq,
                        "Metric": label,
                        "Value": value
                    })
        df = pd.DataFrame(data)

        # Group the data by q_freq and Metric
        grouped = df.groupby(['q_freq', 'Metric'])['Value'].apply(list).reset_index()

        # Compute sorted unique frequencies
        unique_freqs = sorted(df['q_freq'].unique())

        # Partition frequencies into segments based on gap threshold
        segments = []
        current_segment = []
        for freq in unique_freqs:
            if not current_segment:
                current_segment.append(freq)
            else:
                if freq - current_segment[-1] > freq_gap_threshold:
                    segments.append(current_segment)
                    current_segment = [freq]
                else:
                    current_segment.append(freq)
        if current_segment:
            segments.append(current_segment)

        n_segments = len(segments)

        # Create subplots for each segment; share y-axis so the coherence times align.
        fig, axes = plt.subplots(1, n_segments, sharey=True, figsize=(12, 6))
        # If only one segment, wrap axes in a list for uniform processing.
        if n_segments == 1:
            axes = [axes]

        n_metrics = len(metric_labels)
        # Offsets to separate the box plots for each metric (e.g., for three metrics: -0.2, 0, +0.2)
        offsets = [(i - (n_metrics - 1) / 2) * 0.2 for i in range(n_metrics)]

        # Loop over each segment and plot the box plots on its corresponding axis.
        for ax, segment in zip(axes, segments):
            seg_box_data = []
            seg_positions = []
            seg_colors = []
            # For the x-axis ticks, we'll keep only the original frequencies (without offsets)
            for q in segment:
                for i, label in enumerate(metric_labels):
                    # Get values for the current q_freq and metric.
                    values = grouped[(grouped['q_freq'] == q) & (grouped['Metric'] == label)]['Value']
                    if not values.empty:
                        seg_box_data.append(values.values[0])
                        seg_positions.append(q + offsets[i])
                        seg_colors.append(metric_colors[i])
            bp = ax.boxplot(seg_box_data, positions=seg_positions, widths=0.15, patch_artist=True)
            # Color the boxes for each metric.
            for patch, color in zip(bp['boxes'], seg_colors):
                patch.set_facecolor(color)
            # Set x-axis ticks to the actual qubit frequencies for this segment.
            ax.set_xticks(segment)
            ax.set_xticklabels([f"{freq:.2f}" for freq in segment], fontsize=14)
            ax.set_ylim(bottom=0)
            ax.tick_params(axis='x', labelsize=14)
            # Note: individual x-axis labels have been removed

        # Add a single overall x-axis label for the entire figure
        fig.supxlabel("Qubit Frequency Mean (MHz)", fontsize=14)

        # Add overall y label and title
        axes[0].set_ylabel("Coherence time (µs)", fontsize=14)
        fig.suptitle("Coherence by Qubit Frequency", fontsize=14)

        # Create a custom legend (added on the first axis)
        handles = [mpatches.Patch(color=metric_colors[i], label=metric_labels[i]) for i in range(n_metrics)]
        axes[0].legend(handles=handles, fontsize=14)

        # If more than one segment, add diagonal break marks between the axes.
        if n_segments > 1:
            d = 0.015  # size of break marks in axes coordinates
            for i in range(n_segments - 1):
                # For the right end of the left axis:
                kwargs = dict(transform=axes[i].transAxes, color='k', clip_on=False)
                axes[i].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
                axes[i].plot((1 - d, 1 + d), (-d, d), **kwargs)
                # For the left end of the right axis:
                kwargs.update(transform=axes[i + 1].transAxes)
                axes[i + 1].plot((-d, d), (1 - d, 1 + d), **kwargs)
                axes[i + 1].plot((-d, d), (-d, d), **kwargs)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        file_label = "_".join([label.replace(" ", "_") for label in metric_labels])
        outfile = f"{analysis_folder}boxplot_{file_label}_qfreq_x_break.png"
        plt.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
        plt.close(fig)

    def plot_three_metrics_by_freq_comp_run_x_break(
            self,
            q_freq_means,
            metric_1, metric_2, metric_3,
            metric_1_r2, metric_2_r2, metric_3_r2,
            metric_labels=["T1 (µs)", "T2R (µs)", "T2E (µs)"],
            metric_colors=["skyblue", "lightgreen", "lightcoral"], #"lightblue", "cornflowerblue", "royalblue"
            freq_gap_threshold=5,
            plot_outliers = True
    ):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import pandas as pd
        import numpy as np

        # -------------------------------------------------------------------------
        # 1. Use a modern style for a cleaner look
        # -------------------------------------------------------------------------
        import seaborn as sns
        sns.set_style("whitegrid")

        # Create analysis folder if it doesn't exist
        analysis_folder = f"/data/QICK_data/{self.run_name}/benchmark_analysis_plots/metric_boxplots/"
        self.create_folder_if_not_exists(analysis_folder)

        # Reorganize the data into a long-form DataFrame
        data = []
        for qubit_index in range(self.number_of_qubits):
            q_freq = q_freq_means[qubit_index]  # corresponding frequency mean for this qubit

            # Run 3 data
            for metric, label in zip([metric_1, metric_2, metric_3], metric_labels):
                for value in metric[qubit_index]:
                    data.append({
                        "q_freq": q_freq,
                        "Metric": label,
                        "Value": value,
                        "Run": "Run 3"
                    })

            # Run 2 data
            for metric, label in zip([metric_1_r2, metric_2_r2, metric_3_r2], metric_labels):
                for value in metric[str(qubit_index)]:
                    data.append({
                        "q_freq": q_freq,
                        "Metric": label,
                        "Value": value,
                        "Run": "Run 2"
                    })

        df = pd.DataFrame(data)

        # Group the data by q_freq, Metric, and Run
        grouped = df.groupby(['q_freq', 'Metric', 'Run'])['Value'].apply(list).reset_index()

        # Compute sorted unique frequencies
        unique_freqs = sorted(df['q_freq'].unique())

        # Partition frequencies into segments based on gap threshold
        segments = []
        current_segment = []
        for freq in unique_freqs:
            if not current_segment:
                current_segment.append(freq)
            else:
                if freq - current_segment[-1] > freq_gap_threshold:
                    segments.append(current_segment)
                    current_segment = [freq]
                else:
                    current_segment.append(freq)
        if current_segment:
            segments.append(current_segment)

        n_segments = len(segments)

        # -------------------------------------------------------------------------
        # 2. Create subplots for each frequency segment; share y-axis
        # -------------------------------------------------------------------------
        fig, axes = plt.subplots(1, n_segments, sharey=True, figsize=(12, 6))
        if n_segments == 1:
            axes = [axes]

        n_metrics = len(metric_labels)

        # -------------------------------------------------------------------------
        # 3. Define smaller offsets and use alpha for Run 2
        #    - We’ll place T1, T2R, T2E in small steps around each frequency
        #    - Then shift them slightly for Run 2 vs Run 3
        # -------------------------------------------------------------------------
        # Example: T1, T2R, T2E for Run 2 are around [-0.2, -0.1, 0.0]
        #          T1, T2R, T2E for Run 3 are around [ 0.1,  0.2, 0.3]
        run2_offsets = [-1.5 + i * 0.4 for i in range(n_metrics)]
        run3_offsets = [0.5 + i * 0.4 for i in range(n_metrics)]

        # Helper to pick the right alpha given "Run"
        def alpha_for_run(run_label):
            return 0.5 if run_label == "Run 2" else 1.0

        for ax, segment in zip(axes, segments):

            # Remove all grid lines
            ax.grid(False)
            # Enable vertical grid lines only
            ax.xaxis.grid(True, linestyle='--', alpha=0.5)

            seg_box_data = []
            seg_positions = []
            seg_colors = []
            seg_alphas = []  # store alpha for each box

            # For each frequency in the segment, plot metrics for Run 2 and Run 3
            for q in segment:
                for i, label in enumerate(metric_labels):
                    # Plot Run 2
                    values_run2 = grouped[
                        (grouped['q_freq'] == q) &
                        (grouped['Metric'] == label) &
                        (grouped['Run'] == "Run 2")
                        ]['Value']
                    if not values_run2.empty:
                        seg_box_data.append(values_run2.values[0])
                        seg_positions.append(q + run2_offsets[i])
                        seg_colors.append(metric_colors[i])
                        seg_alphas.append(alpha_for_run("Run 2"))

                    # Plot Run 3
                    values_run3 = grouped[
                        (grouped['q_freq'] == q) &
                        (grouped['Metric'] == label) &
                        (grouped['Run'] == "Run 3")
                        ]['Value']
                    if not values_run3.empty:
                        seg_box_data.append(values_run3.values[0])
                        seg_positions.append(q + run3_offsets[i])
                        seg_colors.append(metric_colors[i])
                        seg_alphas.append(alpha_for_run("Run 3"))
            if plot_outliers:
                bp = ax.boxplot(seg_box_data, positions=seg_positions, widths=0.3, patch_artist=True, showfliers=True)
            else:
                bp = ax.boxplot(seg_box_data, positions=seg_positions, widths=0.3, patch_artist=True, showfliers=False)

            # Apply facecolors and alpha to each box
            for patch, color, alpha_val in zip(bp['boxes'], seg_colors, seg_alphas):
                patch.set_facecolor(color)
                patch.set_alpha(alpha_val)
                patch.set_edgecolor("black")
                patch.set_linewidth(1.0)

            # Style the outliers
            for flier, color, alpha_val in zip(bp['fliers'], seg_colors, seg_alphas):
                flier.set_marker('o')
                flier.set_markerfacecolor(color)
                flier.set_markeredgecolor("black")
                flier.set_alpha(alpha_val)
                flier.set_markersize(4)

            ax.set_xticks(segment)
            ax.set_xticklabels([f"{freq:.2f}" for freq in segment], fontsize=12)
            ax.tick_params(axis='x', labelsize=12)

            ax.set_ylim(0, 90)

            ax.text(0.25, 0.95, "Run 2", transform=ax.transAxes,
                    fontsize=12,  ha='center', va='top')
            ax.text(0.75, 0.95, "Run 3", transform=ax.transAxes,
                    fontsize=12,  ha='center', va='top')

        # Overall labels
        fig.supxlabel("Qubit Frequency Mean (MHz)", fontsize=14)
        axes[0].set_ylabel("Coherence time (µs)", fontsize=14)

        # -------------------------------------------------------------------------
        # 4. Create simpler legends
        # -------------------------------------------------------------------------
        # Legend for the metrics (colors)
        import matplotlib.lines as mlines
        metric_handles = [
            mpatches.Patch(color=metric_colors[i], label=metric_labels[i], alpha=1.0)
            for i in range(n_metrics)
        ]
        #metric_legend = axes[-1].legend(handles=metric_handles, fontsize=12,
        #                                loc='upper left', bbox_to_anchor=(1.05, 1))
        #axes[-1].add_artist(metric_legend)

        # Legend for run (alpha difference)
        run2_handle = mpatches.Patch(facecolor="gray", alpha=0.5, edgecolor="black", label="Run 2")
        run3_handle = mpatches.Patch(facecolor="gray", alpha=1.0, edgecolor="black", label="Run 3")
        #axes[0].legend(handles=[run2_handle, run3_handle],  fontsize=12, loc='upper left')

        # Diagonal break marks if more than one segment
        if n_segments > 1:
            d = 0.015
            for i in range(n_segments - 1):
                kwargs = dict(transform=axes[i].transAxes, color='k', clip_on=False)
                axes[i].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
                axes[i].plot((1 - d, 1 + d), (-d, d), **kwargs)
                kwargs.update(transform=axes[i + 1].transAxes)
                axes[i + 1].plot((-d, d), (1 - d, 1 + d), **kwargs)
                axes[i + 1].plot((-d, d), (-d, d), **kwargs)

        fig.suptitle("Coherence by Qubit Frequency", fontsize=16)

        # Place a global legend centered at the top, just below the title
        fig.legend(handles=metric_handles, fontsize=12,
                   loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=len(metric_handles))

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        file_label = "_".join([label.replace(" ", "_") for label in metric_labels])
        if plot_outliers:
            outfile = f"{analysis_folder}boxplot_{file_label}_qfreq_x_break_comp_runs_with_outliers.png"
        else:
            outfile = f"{analysis_folder}boxplot_{file_label}_qfreq_x_break_comp_runs_no_outliers.png"
        plt.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
        plt.close(fig)



