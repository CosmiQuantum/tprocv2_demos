from syspurpose.files import three_way_merge
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
from qualang_tools.plot import Fit
import visdom
###################################################### Set These #######################################################
save_figs = True
fit_saved = False
show_legends = False
signal = 'None'
run_number = 3 #starting from first run with qubits. Run 1 = run4a at quiet, run 2 = run5a at quiet, etc
figure_quality = 100 #ramp this up to like 500 for presentation plots
final_figure_quality = 200
# run_name = 'run6/6transmon/Round_Robin_Benchmark/Data'
run_name = 'run6/6transmon/ef_studies_t1_tests/'
FRIDGE = "QUIET"
run_notes = ('Added IR shielding, better cryo terminators, thermalizing with 0dB attenuator ') #please make it brief for the plot

top_folder_dates = ['2025-03-30']
exp_extension='_ge' #'_ge' or '_fe' or '_fg'

#
# date = '2025-02-23'
# outerFolder = f"/data/QICK_data/{run_name}/" + date + "/"
################################################ 01: Get all data ######################################################
res_spec_vs_time = ResonatorFreqVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
                                       save_figs, fit_saved, signal, run_name)
date_times_res_spec, res_freqs = res_spec_vs_time.run()

q_spec_vs_time = QubitFreqsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
                                  save_figs, fit_saved, signal, run_name, FRIDGE)
date_times_q_spec, q_freqs, qspec_fit_err = q_spec_vs_time.run(exp_extension=exp_extension)

pi_amps_vs_time = PiAmpsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
                              fit_saved,signal, run_name)
date_times_pi_amps, pi_amps = pi_amps_vs_time.run(plot_depths=False, exp_extension=exp_extension)

t1_vs_time = T1VsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, FRIDGE)
date_times_t1, t1_vals, t1_fit_err = t1_vs_time.run(return_errs=True,exp_extension=exp_extension)

########################################## 03: Resonator Freqs vs Time Plots ###########################################
res_spec_vs_time.plot(date_times_res_spec, res_freqs, show_legends, exp_extension=exp_extension)

######################################### 04: Qubit Freqs vs Time Plots #############################################
q_spec_vs_time.plot_with_errs(date_times_q_spec, q_freqs, qspec_fit_err, show_legends,exp_extension=exp_extension)

############################################## 05: Pi Amp vs Time Plots ###############################################
pi_amps_vs_time.plot(date_times_pi_amps, pi_amps, show_legends, exp_extension=exp_extension)

################################################# 06: T1 vs Time Plots #################################################
#t1_vs_time.plot_without_errs(date_times_t1, t1_vals, show_legends)
t1_vs_time.plot_with_errs(date_times_t1, t1_vals, t1_fit_err, show_legends,exp_extension=exp_extension)

############################################## 09: T1 hist/cumul/err Plots #############################################
t1_distribution_plots = T1HistCumulErrPlots(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
                                            save_figs, fit_saved, signal, run_name, run_notes, run_number, fridge=FRIDGE)
dates, t1_vals, t1_errs = t1_distribution_plots.run(exp_extension=exp_extension)
t1_std_values, t1_mean_values = t1_distribution_plots.plot(dates, t1_vals, t1_errs, show_legends,exp_extension=exp_extension)

