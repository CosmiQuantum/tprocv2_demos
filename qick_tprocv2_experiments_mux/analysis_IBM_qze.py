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
#run_name = 'run7/6transmon/QZE_IBM/300_t1_points_500_avgs_relax_delay1ms_2tests_slice30ms_round0/'
#run_name = 'run7/6transmon/QZE_IBM/300_t1_points_500_avgs_relax_delay1ms_2tests_slice30ms_round1/'
#run_name = 'run7/6transmon/QZE_IBM/300_t1_points_500_avgs_relax_delay1ms_2tests_slice20ms_round1/'
#run_name = 'run7/6transmon/QZE_IBM/300_t1_points_500_avgs_relax_delay1ms_2tests_slice20ms_round0/'
# run_name = 'run7/6transmon/QZE_IBM/300_t1_points_500_avgs_relax_delay1ms_2tests_slice10ms_round0/'
# run_name = 'run7/6transmon/QZE_IBM/300_t1_points_500_avgs_relax_delay1ms_2tests_slice10ms_round1/'
run_name = 'run7/6transmon/QZE_IBM/50_t1_points_1500_avgs_relax_delay1ms_2tests_slice20ms_round1/'

FRIDGE = "QUIET"
run_notes = ('Added IR shielding, better cryo terminators, thermalizing with 0dB attenuator ') #please make it brief for the plot
# top_folder_dates = ['qubit_0_2025-08-03_17-06-20','qubit_1_2025-08-03_17-12-03','qubit_2_2025-08-03_17-17-13',
#                     'qubit_3_2025-08-03_17-22-21', 'qubit_4_2025-08-03_17-27-33',
#                     'qubit_5_2025-08-03_17-32-42']
# top_folder_dates = ['qubit_0_2025-08-03_17-38-00','qubit_1_2025-08-03_17-44-11','qubit_2_2025-08-03_17-49-23',
#                     'qubit_3_2025-08-03_17-54-37', 'qubit_4_2025-08-03_17-59-56',
#                     'qubit_5_2025-08-03_18-05-16']
# top_folder_dates = ['qubit_0_2025-08-03_16-34-31','qubit_1_2025-08-03_16-39-41','qubit_2_2025-08-03_16-44-55',
#                     'qubit_3_2025-08-03_16-50-36', 'qubit_4_2025-08-03_16-55-50',
#                     'qubit_5_2025-08-03_17-01-02']
# top_folder_dates = ['qubit_0_2025-08-03_16-02-17','qubit_1_2025-08-03_16-07-28','qubit_2_2025-08-03_16-12-42',
#                     'qubit_3_2025-08-03_16-18-15', 'qubit_4_2025-08-03_16-23-32',
#                     'qubit_5_2025-08-03_16-29-12']
# top_folder_dates = ['qubit_0_2025-08-03_14-59-50','qubit_1_2025-08-03_15-05-01','qubit_2_2025-08-03_15-10-12',
#                     'qubit_3_2025-08-03_15-15-21', 'qubit_4_2025-08-03_15-20-26',
#                     'qubit_5_2025-08-03_15-25-35']
# top_folder_dates = ['qubit_0_2025-08-03_15-30-53','qubit_1_2025-08-03_15-36-13','qubit_2_2025-08-03_15-41-29',
#                     'qubit_3_2025-08-03_15-46-27', 'qubit_4_2025-08-03_15-51-45',
#                     'qubit_5_2025-08-03_15-56-51']
top_folder_dates = ['qubit_0','qubit_1','qubit_2','qubit_3', 'qubit_4','qubit_5']

# ################################################ 01: Get all data ######################################################

t1_vs_time = T1VsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, FRIDGE)
Is,Qs,amps,gains = t1_vs_time.run_IBM_qze()
t1_vs_time.plot_IBM_qze(amps,gains, f'/data/QICK_data/{run_name}/analysis/')
t1_vs_time.plot_IBM_qze_normal(amps,gains, f'/data/QICK_data/{run_name}/analysis/')


