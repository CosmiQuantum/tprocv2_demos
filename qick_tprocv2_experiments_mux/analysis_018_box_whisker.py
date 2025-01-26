import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pytz

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
        plt.tight_layout()

        outfile = f"{analysis_folder}boxplot_{metric_label.replace(' ', '_')}.pdf"
        plt.savefig(outfile, transparent=True, dpi=self.final_figure_quality)
        plt.close(fig)
