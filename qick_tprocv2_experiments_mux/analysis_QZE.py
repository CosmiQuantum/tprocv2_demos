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
final_figure_quality = 100

FRIDGE = "QUIET"
run_notes = ('Added IR shielding, better cryo terminators, thermalizing with 0dB attenuator ') #please make it brief for the plot

date='no_qubit_rabi_drive2025-05-20_06-55-16'

outerFolder = '/data/QICK_data/run6/6transmon/QZE/rabi/study_data'
outerFolder_systematics = f'/data/QICK_data/run6/6transmon/QZE/rabi/{date}/study_data'

outerFolder_save_plots = os.path.join(f'/data/QICK_data/run6/6transmon/QZE/rabi/{date}/study_data/', "QZE_plot")
if not os.path.exists(outerFolder_save_plots):
    os.makedirs(outerFolder_save_plots)

################################################ 01: Get all data ######################################################

pi_amps_vs_time = PiAmpsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, [date], save_figs,
                              fit_saved,signal,'/data/QICK_data/run6/6transmon/QZE/')

pi_amps_vs_time.runQZE(outerFolder_systematics, outerFolder_save_plots, fit=False, expt_name = "length_rabi_ge_qze",
                       old_format=False, filter_amp_above=0, mark_w01s=True, plot_detuned_amps=True,plot_detuning=True,
                       pi_line_label_left='No\nZeno\nPulse',pi_line_label_right='Zeno Pulse', qubit_index=1,
                       z_limit=None,z_limit_lower=None, gain_max=0.075)#5.5
# pi_amps_vs_time.runQZE_systematics_subtraction(outerFolder, outerFolder_systematics, outerFolder_save_plots,
#                                                fit=False, expt_name = "length_rabi_ge_qze")
#
# pi_amps_vs_time.plot_chevron_qze(outerFolder_systematics, outerFolder_save_plots, fit=False, expt_name = "length_rabi_ge_qze",
#                        old_format=False, filter_amp_above=0, mark_w01s=True, plot_detuned_amps=True,plot_detuning=True,
#                        pi_line_label_left='Actual\nQfreq',pi_line_label_right='Shifted Qfreq', qubit_index=1,z_limit=5.5)
