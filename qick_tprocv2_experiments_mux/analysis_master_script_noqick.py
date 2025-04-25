from analysis_021_plot_allRR_noqick import QubitFreqsVsTime
from analysis_021_plot_allRR_noqick import T1VsTime
from analysis_017_plot_metric_dependencies import PlotMetricDependencies
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE

###################################################### Set These #######################################################
save_figs = False
fit_saved = False
show_legends = False
signal = 'None'
run_number = 3 #starting from first run with qubits. Run 1 = run4a at quiet, run 2 = run5a at quiet, etc
figure_quality = 100 #ramp this up to like 500 for presentation plots
final_figure_quality = 200
run_name = 'run6/6transmon/Round_Robin_Benchmark/Data'
#run_name = 'run6/6transmon/QZE/QZE_measurement/Optimization/'
FRIDGE = "QUIET"

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
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_01-06-04"
] # data folders
top_folder_dates = ["/".join(path.strip("/").split("/")[-2:]) + "/" for path in paths] #extracts last two folders from each path

# ################################################ 01: Get all data ######################################################
q_spec_vs_time = QubitFreqsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
                                  save_figs, fit_saved, signal, run_name, FRIDGE)
date_times_q_spec, q_freqs, qspec_fit_err = q_spec_vs_time.run('_ge')


t1_vs_time = T1VsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, FRIDGE)
date_times_t1, t1_vals, t1_fit_err = t1_vs_time.run(return_errs=True, exp_extension = '_ge')

######################################### 01: Qubit Freqs vs Time Plots #############################################
# q_spec_vs_time.plot_without_errs(date_times_q_spec, q_freqs,show_legends)
# q_spec_vs_time.plot_with_errs(date_times_q_spec, q_freqs, qspec_fit_err, show_legends)
# q_spec_vs_time.plot_with_errs_single_plot(date_times_q_spec, q_freqs, qspec_fit_err, show_legends=True)

################################################ 02: T1 vs Time Plots #################################################
# t1_vs_time.plot_without_errs(date_times_t1, t1_vals, show_legends)
# t1_vs_time.plot_with_errs(date_times_t1, t1_vals, t1_fit_err, show_legends)
# t1_vs_time.plot_with_errs_single_plot(date_times_t1, t1_vals, t1_fit_err, show_legends=True)

# ########################################## 03: Metrics Vs Each Other ############################################
# Qubit spec data for Q1 and Q5 only
Q1_freqs = q_freqs[0]
Q1_dates_spec = date_times_q_spec[0]

#This collects qubit spec data for Q5 only
Q5_freqs = q_freqs[4]
Q5_dates_spec = date_times_q_spec[4]

# This collects T1 data for Q1 and Q5 only
qubit1_t1times = date_times_t1[0]
qubit1_t1 = t1_vals[0]

qubit5_t1times = date_times_t1[4]
qubit5_t1 = t1_vals[4]

plotter = PlotMetricDependencies(run_name, tot_num_of_qubits, final_figure_quality, fridge=FRIDGE)

plotter.plot(Q1_dates_spec, Q1_freqs, qubit1_t1times, qubit1_t1, metric_1_label = 'Q1 Freq (MHz)',
             metric_2_label = 'T1 (us)')
# plotter.plot(Q5_dates_spec, Q5_freqs, qubit5_t1times, qubit5_t1, metric_1_label = 'Q5 Freq (MHz)',
#              metric_2_label = 'T1 (us)')
########################################### 04: Qubits Vs Each Other ############################################
# #Q1 T1 vs Q3 T1
# # plotter.plot_single_pair(date_times_1=qubit1_times, metric_1=qubit1_t1, date_times_2=qubit3_times, metric_2=qubit3_t1,
# #                          metric_1_label="T1_Qubit_1", metric_2_label="T1_Qubit_3")
#
# #Q1 temperatures and other metrics vs time, for 1 qubit
# plotter.plot_q1_temp_and_t1(q1_temps=q1_temps, q1_t1_times=qubit1_times, q1_temp_times=q1_temp_times,
#                             q1_t1_vals=qubit1_t1, temp_label="Qubit Temp (mK)", t1_label="T1 (µs)",
#                             magcan_dates = mcp2_dates, magcan_temps = magcan_temps, magcan_label = "Mag Can Temp (mK)",
#                             mcp2_dates = mcp2_dates, mcp2_temps = mcp2_temps, mcp2_label = "MCP2 Temp (mK)",
#                             Q1_freqs = Q1_freqs, Q1_dates_spec = Q1_dates_spec, qspec_label = "Q1 Frequency (MHz)",
#                             date_times_pi_amps_Q1 = date_times_pi_amps_Q1, pi_amps_Q1 = pi_amps_Q1,
#                             pi_amps_label = "Pi Amp (a.u.)")
#