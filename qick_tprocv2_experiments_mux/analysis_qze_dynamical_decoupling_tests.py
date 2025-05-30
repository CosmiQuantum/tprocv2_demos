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
from analysis_008p5_dephasing_vs_time_plots import DephasingVsTime
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
run_name = 'run6b/6transmon/QZE/dephasing_from_higher_energy_levels'
#run_name = 'run6/6transmon/QZE/QZE_measurement/Optimization/'
FRIDGE = "QUIET"
run_notes = ('Added IR shielding, better cryo terminators, thermalizing with 0dB attenuator ') #please make it brief for the plot
#top_folder_dates = ['2025-05-30_08-00-00']
top_folder_dates = ['2025-05-30_10-09-37']
#



#date = '2025-03-28'
#outerFolder = f"/data/QICK_data/{run_name}/" + date + "/study_data/"
############################################### 01: Get all data ######################################################
#
# t2e_vs_time = T2eVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
#                  signal, run_name, FRIDGE)
# date_times_t2e, t2e_vals, t2e_fit_err = t2e_vs_time.run(return_errs=True)

dephasing_vs_time = DephasingVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, FRIDGE)
date_times, t2e_vals, noise_gains, Is, Qs,Delay_Times = dephasing_vs_time.run(return_noise_gain=True,name='T2E_ge')


################################################# 08: T2E vs Time Plots ################################################
dephasing_vs_time.plot_noise_vs_gain_v2(date_times=date_times,Is=Is, Qs=Qs,delay_times=Delay_Times,noise_gains=noise_gains)

#top_folder_dates = ['2025-05-29_21-15-09','2025-05-29_22-38-13','2025-05-29_22-56-40']
#top_folder_dates = ['2025-05-30_08-34-32']
#top_folder_dates = ['2025-05-30_09-15-42']
#top_folder_dates = ['2025-05-30_11-38-01']
top_folder_dates = ['2025-05-30_12-27-10']
dephasing_vs_time = DephasingVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, FRIDGE)
date_times, t2e_vals, noise_gains, Is, Qs,Delay_Times, freq_offset = dephasing_vs_time.run(return_freq_offset=True,name='T2E_ge')


################################################# 08: T2E vs Time Plots ################################################
dephasing_vs_time.plot_noise_vs_offset(date_times=date_times,Is=Is, Qs=Qs,delay_times=Delay_Times,noise_gains=noise_gains, freq_offsets=freq_offset)
