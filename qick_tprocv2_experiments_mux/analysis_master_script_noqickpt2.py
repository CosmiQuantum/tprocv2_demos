# from analysis_021_plot_allRR_noqick import QubitFreqsVsTime
# from analysis_021_plot_allRR_noqick import T1VsTime
from analysis_017_plot_metric_dependencies import PlotMetricDependencies
from expt_config import expt_cfg, list_of_all_qubits, FRIDGE
from qicklab.analysis import qspec, t1
import os
import numpy as np
###################################################### Set These #######################################################
# Setup
QubitIndex = 0  # or whatever qubit you are analyzing
theta = 0
threshold = 0

# Initialize outputs
tot_num_of_qubits = 6 # Total number of qubits currently at QUIET
all_qspec_dates = [[] for _ in range(tot_num_of_qubits)]
all_qspec_freqs = [[] for _ in range(tot_num_of_qubits)]
all_t1_dates = [[] for _ in range(tot_num_of_qubits)]
all_t1_vals = [[] for _ in range(tot_num_of_qubits)]

# Other params
save_figs = False
fit_saved = False
show_legends = False
signal = 'None'
run_number = 3 #starting from first run with qubits. Run 1 = run4a at quiet, run 2 = run5a at quiet, etc
figure_quality = 100 #ramp this up to like 500 for presentation plots
final_figure_quality = 200
run_name = 'run6/6transmon'
FRIDGE = "QUIET"

################################################## File Paths #################################################################
paths = [
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_11-47-09",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_12-51-09",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_17-50-00",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_22-47-49",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-17_03-42-36",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-17_08-42-24",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy3/2025-04-17_12-28-37",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy3/2025-04-17_17-22-46",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy3/2025-04-17_22-16-39",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy3/2025-04-18_01-45-53",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy3/2025-04-18_06-40-55",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-18_11-59-33",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-18_16-56-58",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-18_21-51-13",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_02-45-41",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_07-39-57",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_12-34-26",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_17-48-44",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_22-43-02",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_03-37-50",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_08-32-36",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_13-26-47",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_18-25-13",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_23-25-04",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-21_04-23-31",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy1/2025-04-21_10-17-14",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy1/2025-04-21_15-12-01",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy1/2025-04-21_20-09-53",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy1/2025-04-22_01-07-08",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy1/2025-04-22_06-04-35",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy2/2025-04-22_21-52-40",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy3/2025-04-23_08-50-38",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-23_11-20-17",
    #"/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-23_14-46-57",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-23_18-13-57",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-23_21-40-34",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_01-06-04",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_04-33-07",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_07-56-53",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_11-18-03",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_14-41-03",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_18-04-49",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_21-31-17",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_00-54-28",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_04-20-54",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_07-46-50",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_11-10-03",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_14-34-10",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_17-57-10",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_21-26-15",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_00-54-11",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_04-21-21",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_07-55-49",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_11-24-18",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_14-49-55",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_18-15-55"
] # data folders
################################################# Get all data ######################################################
for full_path in paths:
    path = os.path.dirname(full_path)  # one level up from the dataset
    dataset = os.path.basename(full_path)  # just the '2025-04-16_11-47-09' part

    for QubitIndex in [0,4]: # We are only taking science data for qubits 1 and 5
        try:
            # --- Load QSpec ---
            qspec_obj = qspec(path, dataset, QubitIndex)
            qspec_dates, qspec_n, qspec_probe_freqs, qspec_I, qspec_Q = qspec_obj.load_all()
            qspec_freqs, qspec_errs, qspec_fwhms = qspec_obj.get_all_qspec_freq(qspec_probe_freqs, qspec_I, qspec_Q, qspec_n)

            all_qspec_dates[QubitIndex].extend(qspec_dates)
            all_qspec_freqs[QubitIndex].extend(qspec_freqs)

        except Exception as e:
            print(f"Failed loading Qspec for qubit {QubitIndex} from {path}: {e}")

        try:
            # --- Load T1 ---
            t1_obj = t1(path, dataset, QubitIndex, theta, threshold)
            t1_dates, t1_n, delay_times, t1_steps, t1_reps, t1_I_shots, t1_Q_shots = t1_obj.load_all()
            t1_p_excited = t1_obj.process_shots(t1_I_shots, t1_Q_shots, t1_n, t1_steps)
            t1s, t1_errs = t1_obj.get_all_t1(delay_times, t1_p_excited, t1_n)

            all_t1_dates[QubitIndex].extend(t1_dates)
            all_t1_vals[QubitIndex].extend(t1s)

        except Exception as e:
            print(f"Failed loading T1 for qubit {QubitIndex} from {path}: {e}")
########################################### Metrics Vs Each Other ############################################
# Qubit spec data for Q1 and Q5 only
Q1_freqs = all_qspec_freqs[0]
Q1_dates_spec = all_qspec_dates[0]

Q5_freqs = all_qspec_freqs[4]
Q5_dates_spec = all_qspec_dates[4]

# This collects T1 data for Q1 and Q5 only
qubit1_t1times = all_t1_dates[0]
qubit1_t1 = all_t1_vals[0]

qubit5_t1times = all_t1_dates[4]
qubit5_t1 = all_t1_vals[4]

#re-format date times before passing them through plotting function
Q1_dates_spec_str = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in Q1_dates_spec]
qubit1_t1times_str = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in qubit1_t1times]

# Initialize class
plotter = PlotMetricDependencies(run_name, tot_num_of_qubits, final_figure_quality, fridge=FRIDGE)

# Q1 Freq vs Q1 T1
plotter.plot(Q1_dates_spec_str, Q1_freqs, qubit1_t1times_str, qubit1_t1, metric_1_label = 'Q1 Freq (MHz)',
             metric_2_label = 'T1 (us)')

# # Q5 Freq vs Q5 T1
# plotter.plot(Q5_dates_spec, Q5_freqs, qubit5_t1times, qubit5_t1, metric_1_label = 'Q5 Freq (MHz)',
#              metric_2_label = 'T1 (us)')
########################################### Qubits Vs Each Other ############################################
# # Q1 T1 vs Q5 T1
# plotter.plot_single_pair(date_times_1=qubit1_t1times, metric_1=qubit1_t1, date_times_2=qubit5_t1times, metric_2=qubit5_t1,
#                          metric_1_label="T1_Qubit_1", metric_2_label="T1_Qubit_3")
#
# # Q1 freq vs Q5 freq
# plotter.plot_single_pair(date_times_1=Q1_dates_spec, metric_1=Q1_freqs, date_times_2=Q5_dates_spec, metric_2=Q5_freqs,
#                         metric_1_label="Qubit_1_Freq", metric_2_label="Qubit_3_Freq")

########################################## Autocorrelation Plots ###############################################
# # Plot autocorrelation for Q1 frequency
# plotter.plot_autocorrelation(Q1_dates_spec, Q1_freqs, label="Qubit 1 Frequency (MHz)", qubit_index=0)
#
# # Plot autocorrelation for Q1 T1
# plotter.plot_autocorrelation(qubit1_t1times, qubit1_t1, label="T1 1 (us)", qubit_index=0)
#
# # Plot autocorrelation for Q5 frequency
# plotter.plot_autocorrelation(Q1_dates_spec, Q1_freqs, label="Qubit 5 Frequency (MHz)", qubit_index=5)
#
# # Plot autocorrelation for Q5 T1
# plotter.plot_autocorrelation(qubit1_t1times, qubit1_t1, label="T1 5 (us)", qubit_index=5)