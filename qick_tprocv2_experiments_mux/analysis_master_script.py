# from syspurpose.files import three_way_merge
from section_008_save_data_to_h5 import Data_H5
from analysis_000_load_configs import LoadConfigs
from analysis_001_plot_all_RR_h5 import PlotAllRR
from analysis_002_res_centers_vs_time_plots import ResonatorFreqVsTime
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime
from analysis_004_pi_amp_vs_time_plots import PiAmpsVsTime
from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_005_Qtemp_vs_time_plots import QTempsVsTime
from analysis_007_T2R_vs_time_plots import T2rVsTime
from analysis_008_T2E_vs_time_plots import T2eVsTime
from analysis_009_T1_hist_cumul_err_plots import T1HistCumulErrPlots
from analysis_010_T2R_hist_cumul_err_plots import T2rHistCumulErrPlots
from analysis_011_T2E_hist_cumul_err_plots import T2eHistCumulErrPlots
from analysis_012_save_run_data import SaveRunData
from analysis_013_update_saved_run_data_notes import UpdateNote
from analysis_014_temperature_calcsandplots import TempCalcAndPlots
from analysis_015_plot_all_run_stats import CompareRuns
from analysis_016_metrics_vs_temp import (ResonatorFreqVsTemp, GetThermData, QubitFreqsVsTemp,
                                          PiAmpsVsTemp, T1VsTemp, T2rVsTemp, T2eVsTemp)
from analysis_017_plot_metric_dependencies import PlotMetricDependencies
from analysis_018_box_whisker import PlotBoxWhisker
from analysis_019_allan_welch_stats_plots import AllanWelchStats
from section_011_qubit_temperatures_efRabi import QubitTemperatureProgram, QubitTemperatureRefProgram
import matplotlib.pyplot as plt
# from datetime import datetime
import datetime
import pytz
import os
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from system_config import QICK_experiment
import numpy as np
import json
import h5py
# from qualang_tools.plot import Fit
import visdom
###################################################### Set These #######################################################
save_figs = True
fit_saved = False
show_legends = False
signal = 'None'
run_number = 3 #starting from first run with qubits. Run 1 = run4a at quiet, run 2 = run5a at quiet, etc
figure_quality = 100 #ramp this up to like 500 for presentation plots
final_figure_quality = 200
run_name = 'run6/6transmon/Round_Robin_Benchmark/Data'
#run_name = 'run6/6transmon/QZE/QZE_measurement/Optimization/'
FRIDGE = "QUIET"
run_notes = ('Added IR shielding, better cryo terminators, thermalizing with 0dB attenuator ') #please make it brief for the plot
top_folder_dates = ['2025-02-21','2025-02-23','2025-03-11','2025-03-12','2025-03-13', '2025-03-14','2025-03-15','2025-03-16','2025-03-17','2025-03-18',
                    '2025-03-19','2025-03-20','2025-03-21','2025-03-22','2025-03-23','2025-03-24','2025-03-25','2025-03-26',
                    '2025-03-27','2025-03-28','2025-03-29','2025-03-31','2025-04-01','2025-04-02','2025-04-09']

#
#top_folder_dates = ['2025-04-02']

#
date = '2025-03-28'
outerFolder = f"/data/QICK_data/{run_name}/" + date + "/"
# ################################################ 01: Get all data ######################################################
res_spec_vs_time = ResonatorFreqVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
                                       save_figs, fit_saved, signal, run_name)
date_times_res_spec, res_freqs = res_spec_vs_time.run()
#
q_spec_vs_time = QubitFreqsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
                                  save_figs, fit_saved, signal, run_name, FRIDGE)
date_times_q_spec, q_freqs, qspec_fit_err = q_spec_vs_time.run()

pi_amps_vs_time = PiAmpsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
                              fit_saved,signal, run_name)
date_times_pi_amps, pi_amps = pi_amps_vs_time.run(plot_depths=False)

t1_vs_time = T1VsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, FRIDGE)
date_times_t1, t1_vals, t1_fit_err = t1_vs_time.run(return_errs=True)

t2r_vs_time = T2rVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name)
date_times_t2r, t2r_vals, t2r_fit_err = t2r_vs_time.run(return_errs=True)

t2e_vs_time = T2eVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, FRIDGE)
date_times_t2e, t2e_vals, t2e_fit_err = t2e_vs_time.run(return_errs=True)
#
# # # ####################################### 02: Plot All Individual Data Plots ###########################################
#date = '2025-04-02'  #only plot all of the data for one date at a time because there is a lot
#outerFolder = f"/data/QICK_data/{run_name}/" + date + "/"
# outerFolder_save_plots = f"/data/QICK_data/{run_name}/" + date + "_plots/"
# plotter = PlotAllRR(date, figure_quality, save_figs, fit_saved, signal, run_name, tot_num_of_qubits, outerFolder,
#                   outerFolder_save_plots)
# plotter.run(plot_res_spec = False, plot_q_spec = False, plot_rabi = False, plot_ss = True, plot_t1 = False,
#             plot_t2r = True, plot_t2e = True)

########################################## 03: Resonator Freqs vs Time Plots ###########################################
res_spec_vs_time.plot(date_times_res_spec, res_freqs, show_legends)

######################################### 04: Qubit Freqs vs Time Plots #############################################
q_spec_vs_time.plot_without_errs(date_times_q_spec, q_freqs,show_legends)
q_spec_vs_time.plot_with_errs(date_times_q_spec, q_freqs, qspec_fit_err, show_legends)
q_spec_vs_time.plot_with_errs_single_plot(date_times_q_spec, q_freqs, qspec_fit_err, show_legends=True)

############################################## 05: Pi Amp vs Time Plots ###############################################
pi_amps_vs_time.plot(date_times_pi_amps, pi_amps, show_legends)

# #----------------------------------------------Extra pi amp analysis----------------------------------------------------
# #can only have the 'depths' argument returned here if plot_depths=True, otherwise delete it
# date_times, pi_amps, depths = pi_amps_vs_time.run(plot_depths=True)
#
# pi_amps_vs_time.plot_vs_signal_depth(date_times, pi_amps, depths, show_legends)
# pi_amps_vs_time.plot_signal_depth_vs_time(date_times, pi_amps, depths, show_legends)

# temps_class_obj = TempCalcAndPlots(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
#                                    save_figs, fit_saved, signal, run_name, fridge = FRIDGE)
#
# temps, qubit_temp_dates = temps_class_obj.get_temps()
# filtered_pi_amps = temps_class_obj.get_filtered_pi_amps(qubit_temp_dates, date_times, pi_amps)
# pi_amps_vs_time.plot_vs_temps(date_times, filtered_pi_amps, temps, show_legends)
#
# ssf, qubit_ssf_dates = temps_class_obj.get_ssf()
# filtered_pi_amps = temps_class_obj.get_filtered_pi_amps(qubit_ssf_dates, date_times, pi_amps)
# pi_amps_vs_time.plot_vs_ssf(date_times, filtered_pi_amps, ssf, show_legends)

############# 05: Qubit Temp vs time (not working currently, use arianna code at bottom) #############################
# qtemp_vs_time = QTempsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
#                                fit_saved,signal, run_name, exp_config)
#
# qubit_temp_dates, qubit_temperatures = qtemp_vs_time.run()
# qtemp_vs_time.plot(qubit_temp_dates, qubit_temperatures, show_legends)
#
################################################ 06: T1 vs Time Plots #################################################
t1_vs_time.plot_without_errs(date_times_t1, t1_vals, show_legends)
t1_vs_time.plot_with_errs(date_times_t1, t1_vals, t1_fit_err, show_legends)
t1_vs_time.plot_with_errs_single_plot(date_times_t1, t1_vals, t1_fit_err, show_legends=True)

################################################# 07: T2R vs Time Plots ################################################
#t2r_vs_time.plot_without_errs(date_times_t2r, t2r_vals, t2r_fit_err, show_legends)
t2r_vs_time.plot_with_errs(date_times_t2r, t2r_vals, t2r_fit_err, show_legends)
t2r_vs_time.plot_with_errs_single_plot(date_times_t2r, t2r_vals, t2r_fit_err, show_legends=True)

################################################# 08: T2E vs Time Plots ################################################
#t2e_vs_time.plot_without_errs(date_times_t2e, t2e_vals, t2e_fit_err, show_legends)
t2e_vs_time.plot_with_errs(date_times_t2e, t2e_vals, t2e_fit_err, show_legends)
t2e_vs_time.plot_with_errs_single_plot(date_times_t2e, t2e_vals, t2e_fit_err, show_legends=True)

############################################## 09: T1 hist/cumul/err Plots #############################################
t1_distribution_plots = T1HistCumulErrPlots(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
                                            save_figs, fit_saved, signal, run_name, run_notes, run_number, fridge=FRIDGE)
dates, t1_vals, t1_errs = t1_distribution_plots.run()
t1_std_values, t1_mean_values = t1_distribution_plots.plot(dates, t1_vals, t1_errs, show_legends)

############################################## 10: T2R hist/cumul/err Plots ############################################
t2r_distribution_plots = T2rHistCumulErrPlots(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
                                            save_figs, fit_saved, signal, run_name, fridge=FRIDGE)
dates, t2r_vals, t2r_errs = t2r_distribution_plots.run(t1_vals)
t2r_std_values, t2r_mean_values = t2r_distribution_plots.plot(dates, t2r_vals, t2r_errs, show_legends)

############################################## 11: T2E hist/cumul/err Plots ############################################
t2e_distribution_plots = T2eHistCumulErrPlots(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
                                            save_figs, fit_saved, signal, run_name, fridge=FRIDGE)
dates, t2e_vals, t2e_errs = t2e_distribution_plots.run(t1_vals)
t2e_std_values, t2e_mean_values = t2e_distribution_plots.plot(dates, t2e_vals, t2e_errs, show_legends)

# # ############################ 12: Save the Key Statistics for This Run to Compare Later #################################
# #need to run 00,01, and 08-10 before this to get all of the variables
# saver = SaveRunData(run_number,FRIDGE, run_notes)
# saver.run(date_times_res_spec, date_times_q_spec, date_times_pi_amps, date_times_t1, date_times_t2r, date_times_t2e,
#           res_freqs, q_freqs, pi_amps, t1_vals, t1_errs, t1_std_values, t1_mean_values, t2r_vals, t2r_errs,
#           t2r_mean_values, t2r_std_values, t2e_vals, t2e_errs, t2e_mean_values, t2e_std_values)

################################## 13: Update Saved Run Notes For Comparison Plot ######################################
# run_number_to_update = 2
# new_run_notes = ("Added more eccosorb filters and a lpf on mxc before and after the device. Added thermometry "
#                  "next to the device")
# updater = UpdateNote(run_number_to_update, new_run_notes)
# updater.run(FRIDGE)
#
############################################### 14: Run Comparison Plots ##############################################
run_number_list = [1,2,3]
comparing_runs = CompareRuns(run_number_list, run_name)

run_stats_folder = f"run_stats/QUIET/run{2}/"
filename = run_stats_folder + 'experiment_data.h5'
loaded_data = comparing_runs.load_from_h5(filename)

t1_vals_r2 = loaded_data['t1_vals']
t2r_vals_r2 = loaded_data['t2r_vals']
t2e_vals_r2 = loaded_data['t2e_vals']

# comparing_runs.plot_decoherence_vs_run(skip_qubit_t2e=False, qubit_to_skip_t2e=0)
# #compare median qubit freq to median decoherence by run number
# comparing_runs.plot_decoherence_vs_qfreq()

# # ############################################### 15: Qubit Temperature Plots ############################################
# temps_class_obj = TempCalcAndPlots(list_of_all_qubits,figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
#               fit_saved, signal, run_name, outerFolder)
#
# all_qubit_temps, all_qubit_times = temps_class_obj.run()
# #
# # #Grabbing only Q1 temperature data
# q1_temp_times = all_qubit_times[0]
# q1_temps = all_qubit_temps[0]
#
# # # ########################################## 16: Metrics Vs Temperature Plots ############################################
# therm = GetThermData(f'/data/QICK_data/{run_name}/Thermometer_Data/')
# #mcp2_dates are just the dates over which thermometry data was taken, works for both datasets
# mcp2_dates, mcp2_temps, magcan_temps = therm.run()
#
# res_spec_vs_temp = ResonatorFreqVsTemp(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
#                                        save_figs, fit_saved, signal, run_name)
# date_times, res_freqs = res_spec_vs_temp.run()
# res_spec_vs_temp.plot(date_times, res_freqs, mcp2_dates, magcan_temps, show_legends)
#
# q_spec_vs_temp = QubitFreqsVsTemp(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
#                                   save_figs, fit_saved, signal, run_name)
# date_times, q_freqs = q_spec_vs_temp.run()
# q_spec_vs_temp.plot(date_times, q_freqs, mcp2_dates, mcp2_temps, show_legends)
#
# pi_amps_vs_temp = PiAmpsVsTemp(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
#                                fit_saved,signal, run_name)
# date_times, pi_amps = pi_amps_vs_temp.run()
#
# pi_amps_vs_temp.plot(date_times, pi_amps, mcp2_dates, mcp2_temps, show_legends)
#
# t1_vs_temp = T1VsTemp(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
#                  signal, run_name)
# date_times, t1_vals = t1_vs_temp.run()
# t1_vs_temp.plot(date_times, t1_vals, mcp2_dates, mcp2_temps, show_legends)
#
# t2r_vs_temp = T2rVsTemp(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
#                  signal, run_name)
# date_times, t2r_vals = t2r_vs_temp.run()
# t2r_vs_temp.plot(date_times, t2r_vals, mcp2_dates, mcp2_temps, show_legends)
#
# t2e_vs_temp = T2eVsTemp(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
#                  signal, run_name)
# date_times, t2e_vals = t2e_vs_temp.run()
# t2e_vs_temp.plot(date_times, t2e_vals, mcp2_dates, mcp2_temps, show_legends)
#
# ########################################## 16: Metrics Vs Each Other ############################################
#This collects qubit spec data for Q1 only
# date_times_pi_amps_Q1 = date_times_pi_amps[0]
# pi_amps_Q1 = pi_amps[0]
#
# #This collects qubit spec data for Q1 only
# Q1_freqs = q_freqs[0]
# Q1_dates_spec = date_times_q_spec[0]
#
# # This collects T1 data for Q1 and Q3 only
# qubit1_times = date_times_t1[0]  # timestamps for qubit 1
# qubit1_t1 = t1_vals[0]        # T1 values for qubit 1
# # #
# # # qubit3_times = date_times_t1[2]  # timestamps for qubit 3
# # # qubit3_t1 = t1_vals[2]        # T1 values for qubit 3
#
#now plot them vs eachother
plotter = PlotMetricDependencies(run_name, tot_num_of_qubits, final_figure_quality, fridge=FRIDGE)

plotter.plot(date_times_q_spec, q_freqs, date_times_t1, t1_vals, metric_1_label = 'Q Freq (MHz)',
             metric_2_label = 'T1 (us)')
plotter.plot(date_times_pi_amps, pi_amps, date_times_t1, t1_vals, metric_1_label = 'Pi Amp (a.u.)',
             metric_2_label = 'T1 (us)')

plotter.plot(date_times_q_spec, q_freqs, date_times_t2r, t2r_vals, metric_1_label = 'Q Freq (MHz)',
             metric_2_label = 'T2R (us)')
plotter.plot(date_times_pi_amps, pi_amps, date_times_t2r, t2r_vals, metric_1_label = 'Pi Amp (a.u.)',
             metric_2_label = 'T2R (us)')

plotter.plot(date_times_q_spec, q_freqs, date_times_t2e, t2e_vals, metric_1_label = 'Q Freq (MHz)',
             metric_2_label = 'T2E (us)')
plotter.plot(date_times_pi_amps, pi_amps, date_times_t2e, t2e_vals, metric_1_label = 'Pi Amp (a.u.)',
             metric_2_label = 'T2E (us)')

plotter.plot(date_times_q_spec, q_freqs, date_times_pi_amps, pi_amps, metric_1_label = 'Q Freq (MHz)',
             metric_2_label = 'Pi Amp (a.u.)')

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
# ##################################### 17: Box And Whisker Qubit Comparison ############################################
boxwhisker = PlotBoxWhisker(run_name, tot_num_of_qubits, final_figure_quality)
# # # boxwhisker.plot(res_freqs, metric_label="Resonator Frequencies (MHz)")
# # # boxwhisker.plot(q_freqs, metric_label="Qubit Frequencies (MHz)")
# # # boxwhisker.plot(pi_amps, metric_label="Pi Amplitude (a.u.)")
# # # boxwhisker.plot(t1_vals, metric_label="T1 (µs)")
# # # boxwhisker.plot(t2r_vals, metric_label="T2R (µs)")
# # # boxwhisker.plot(t2e_vals, metric_label="T2E (µs)")
# # # boxwhisker.plot_three_metrics(t1_vals, t2r_vals, t2e_vals)
means = q_spec_vs_time.plot_hist(q_freqs, show_legends)
#boxwhisker.plot_three_metrics_by_freq(means, t1_vals, t2r_vals, t2e_vals)
#boxwhisker.plot_three_metrics_by_freq_x_break(means, t1_vals, t2r_vals, t2e_vals)
boxwhisker.plot_three_metrics_by_freq_comp_run_x_break(means, t1_vals, t2r_vals, t2e_vals,t1_vals_r2, t2r_vals_r2, t2e_vals_r2, plot_outliers=False)

# ################################## 18: Allan Deviation/ Welch Spectral Density #########################################
stats = AllanWelchStats(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name)
stats.plot_allan_deviation(date_times_q_spec, q_freqs, show_legends, label='QFreq')
stats.plot_allan_deviation(date_times_t1, t1_vals, show_legends, label='T1')
stats.plot_allan_deviation(date_times_t2r, t2r_vals, show_legends, label='T2R')
stats.plot_allan_deviation(date_times_t2e, t2e_vals, show_legends, label='T2E')

stats.plot_welch_spectral_density(date_times_q_spec, q_freqs, show_legends, label='QFreq')
stats.plot_welch_spectral_density(date_times_t1, t1_vals, show_legends, label='T1')
stats.plot_welch_spectral_density(date_times_t2r, t2r_vals, show_legends, label='T2R')
stats.plot_welch_spectral_density(date_times_t2e, t2e_vals, show_legends, label='T2E')

# ################################################### 19: Extra #########################################################
plotter = PlotMetricDependencies(run_name, tot_num_of_qubits, final_figure_quality, FRIDGE)
plotter.plot_shared_datetimes(date_times_q_spec, q_freqs, qspec_fit_err, metric_1_label = 'Q Freq (MHz)',
             metric_2_label = 'Q Freq Fit Err (MHz)')
plotter.scatter_plot_two_y_axis(date_times_q_spec, q_freqs, date_times_q_spec, qspec_fit_err, metric_1_label = 'Q Freq (MHz)',
             metric_2_label = 'Q Freq Fit Err (MHz)')

# plot SSF historgrams nicely
#
# outerFolder = '/data/QICK_data/run6/6transmon/Round_Robin_Benchmark/Data/2025-03-15/'
# outerFolder_save_plots = f"/data/QICK_data/{run_name}/Round_Robin_Benchmark/Data/run3_ss_hist_plots/"
# plotter = PlotAllRR('2025-03-15', figure_quality, save_figs, fit_saved, signal, run_name, tot_num_of_qubits, outerFolder,
#                   outerFolder_save_plots)
# plotter.run(plot_res_spec = False, plot_q_spec = False, plot_rabi = False, plot_ss = True, plot_ss_hist_only=True,
#             ss_plot_title='Run 3', plot_t1 = False,
#             plot_t2r = False, plot_t2e = False)
# outerFolder = '/home/quietuser/Downloads/2024-12-17-20250316T015414Z-001/2024-12-17/'
# outerFolder_save_plots = f"/data/QICK_data/{run_name}/Round_Robin_Benchmark/Data/run2_ss_hist_plots/"
# plotter = PlotAllRR('2024-12-17', figure_quality, save_figs, fit_saved, signal, run_name, tot_num_of_qubits, outerFolder,
#                   outerFolder_save_plots)
# plotter.run(plot_res_spec = False, plot_q_spec = False, plot_rabi = False, plot_ss = True, plot_ss_hist_only=True,
#             ss_plot_title='Run 2', plot_t1 = False,
#             plot_t2r = False, plot_t2e = False)