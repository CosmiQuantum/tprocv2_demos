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
save_figs = False
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
top_folder_dates = ['2025-05-29_07-22-46', '2025-05-29_09-00-02']

top_folder_dates = ['Statistics_ef_noise_gain_0p02_100nsNoisePulse_2025-06-04_21-37-45']#,'Statistics_ef_noise_gain_0p02_100nsNoisePulse_2025-06-05_09-28-35'

dephasing_vs_time = DephasingVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, FRIDGE)
date_times_t2r, t2r_vals, t2r_fit_err, t2r_batch_nums = dephasing_vs_time.run(return_errs=True,name='T2_ge',savefigs=True, filters=True, batch_nums=True,save_plot_folder='documentation')
date_times_t2e, t2e_vals, t2e_fit_err, t2e_batch_nums = dephasing_vs_time.run(return_errs=True,name='T2E_ge',savefigs=True, filters=True, batch_nums=True,save_plot_folder='documentation')
date_times_dephasing, dephasing_vals, dephasing_fit_err, dephasing_batch_nums = dephasing_vs_time.run(return_errs=True,name='DD_ge',savefigs=True, filters=True, batch_nums=True,save_plot_folder='documentation')

date_times_t2r_w_noise, t2r_vals_w_noise, t2r_fit_err_w_noise, t2r_batch_nums_w_noise = dephasing_vs_time.run(return_errs=True,name='T2_ge_w_noise',savefigs=True, filters=True, batch_nums=True,save_plot_folder='documentation')
date_times_t2e_w_noise, t2e_vals_w_noise, t2e_fit_err_w_noise, t2e_batch_nums_w_noise = dephasing_vs_time.run(return_errs=True,name='T2E_ge_w_noise',savefigs=True, filters=True, batch_nums=True,save_plot_folder='documentation')
date_times_dephasing_ef, dephasing_vals_ef, dephasing_fit_err_ef, dephasing_batch_nums_w_noise = dephasing_vs_time.run(return_errs=True,name='DD_ge_ef_noise',savefigs=True, filters=True, batch_nums=True,save_plot_folder='documentation')



#################### T1
date_times_t1_ge, t1_ge_vals, t1_ge_fit_err, t1_ge_batch_nums = dephasing_vs_time.run_T1(return_errs=True,name='T1_ge',savefigs=True, fit=True, batch_nums=True)
date_times_t1_fe, t1_fe_vals, t1_fe_fit_err, t1_fe_batch_nums = dephasing_vs_time.run_T1(return_errs=True,name='T1_fe',savefigs=True, fit=True, batch_nums=True)

date_times_t1_ge_w_noise, t1_ge_vals_w_noise, t1_ge_fit_err_w_noise, t1_ge_batch_nums_w_noise = dephasing_vs_time.run_T1(return_errs=True,name='T1_ge_w_noise',savefigs=True, fit=True, batch_nums=True)
date_times_t1_fe_w_ef, t1_fe_vals_ef, t1_fe_fit_err_ef, t1_fe_batch_nums_w_noise = dephasing_vs_time.run_T1(return_errs=True,name='T1_fe_w_noise',savefigs=True, fit=True, batch_nums=True)



dephasing_vs_time.plot_with_errs_subtract_T1(
    date_times_t1=date_times_t1_ge, t1_vals=t1_ge_vals, t1_fit_err=t1_ge_fit_err,t1_batch_num=t1_ge_batch_nums,
    date_times_t2r=date_times_t2r, t2r_vals=t2r_vals, t2r_fit_err=t2r_fit_err,t2r_batch_num=t2r_batch_nums,
    date_times_dephasing=date_times_dephasing, dephasing_vals=dephasing_vals, dephasing_fit_err=dephasing_fit_err,dephasing_batch_num=dephasing_batch_nums,
    date_times_t2e=date_times_t2e, t2e_vals=t2e_vals, t2e_fit_err=t2e_fit_err,t2e_batch_num=t2e_batch_nums,
    date_times_t2r_w_noise=date_times_t2r_w_noise, t2r_vals_w_noise=t2r_vals_w_noise, t2r_fit_err_w_noise=t2r_fit_err_w_noise,t2r_batch_num_w_noise=t2r_batch_nums_w_noise,
    date_times_t2e_w_noise=date_times_t2e_w_noise, t2e_vals_w_noise=t2e_vals_w_noise, t2e_fit_err_w_noise=t2e_fit_err_w_noise,t2e_batch_num_w_noise=t2e_batch_nums_w_noise,
    date_times_dephasing_ef=date_times_dephasing_ef, dephasing_vals_ef=dephasing_vals_ef, dephasing_fit_err_ef=dephasing_fit_err_ef,dephasing_batch_num_w_noise=dephasing_batch_nums_w_noise,
    show_legends=True, extra_label=' - T1_ge',save_name='t1_ge_subtracted_')
dephasing_vs_time.plot_with_errs_subtract_T1(
    date_times_t1=date_times_t1_ge_w_noise, t1_vals=t1_ge_vals_w_noise, t1_fit_err=t1_ge_fit_err_w_noise,t1_batch_num=t1_ge_batch_nums_w_noise,
    date_times_t2r=date_times_t2r, t2r_vals=t2r_vals, t2r_fit_err=t2r_fit_err,t2r_batch_num=t2r_batch_nums,
    date_times_dephasing=date_times_dephasing, dephasing_vals=dephasing_vals, dephasing_fit_err=dephasing_fit_err,dephasing_batch_num=dephasing_batch_nums,
    date_times_t2e=date_times_t2e, t2e_vals=t2e_vals, t2e_fit_err=t2e_fit_err,t2e_batch_num=t2e_batch_nums,
    date_times_t2r_w_noise=date_times_t2r_w_noise, t2r_vals_w_noise=t2r_vals_w_noise, t2r_fit_err_w_noise=t2r_fit_err_w_noise,t2r_batch_num_w_noise=t2r_batch_nums_w_noise,
    date_times_t2e_w_noise=date_times_t2e_w_noise, t2e_vals_w_noise=t2e_vals_w_noise, t2e_fit_err_w_noise=t2e_fit_err_w_noise,t2e_batch_num_w_noise=t2e_batch_nums_w_noise,
    date_times_dephasing_ef=date_times_dephasing_ef, dephasing_vals_ef=dephasing_vals_ef, dephasing_fit_err_ef=dephasing_fit_err_ef,dephasing_batch_num_w_noise=dephasing_batch_nums_w_noise,
    show_legends=True, extra_label=' - T1_ge_w_noise',save_name='t1_ge_w_noise_subtracted_')

####### do individual comparisons
dephasing_vs_time.plot_with_errs_subtract_T1_individual(
    date_times_t1=date_times_t1_ge, t1_vals=t1_ge_vals, t1_fit_err=t1_ge_fit_err,t1_batch_num=t1_ge_batch_nums,
    date_times_t1_w_noise=date_times_t1_ge_w_noise, t1_vals_w_noise=t1_ge_vals_w_noise, t1_fit_err_w_noise=t1_ge_fit_err_w_noise,t1_batch_num_w_noise=t1_ge_batch_nums_w_noise,
    date_times_dephasing=date_times_dephasing, dephasing_vals=dephasing_vals, dephasing_fit_err=dephasing_fit_err,dephasing_batch_num=dephasing_batch_nums,
    date_times_dephasing_ef=date_times_dephasing_ef, dephasing_vals_ef=dephasing_vals_ef, dephasing_fit_err_ef=dephasing_fit_err_ef,dephasing_batch_num_w_noise=dephasing_batch_nums_w_noise,
    show_legends=True, extra_label=' - $T1_{ge}$',save_name='t1_ge_subtracted_just_dd_')

dephasing_vs_time.plot_with_errs_subtract_T1_individual(
    date_times_t1=date_times_t1_ge, t1_vals=t1_ge_vals, t1_fit_err=t1_ge_fit_err, t1_batch_num=t1_ge_batch_nums,
    date_times_t1_w_noise=date_times_t1_ge_w_noise, t1_vals_w_noise=t1_ge_vals_w_noise,
    t1_fit_err_w_noise=t1_ge_fit_err_w_noise, t1_batch_num_w_noise=t1_ge_batch_nums_w_noise,
    date_times_t2e=date_times_t2e, t2e_vals=t2e_vals, t2e_fit_err=t2e_fit_err,t2e_batch_num=t2e_batch_nums,
    date_times_t2e_w_noise=date_times_t2e_w_noise, t2e_vals_w_noise=t2e_vals_w_noise, t2e_fit_err_w_noise=t2e_fit_err_w_noise,t2e_batch_num_w_noise=t2e_batch_nums_w_noise,
    show_legends=True, extra_label=' - $T1_{ge}$',save_name='t1_ge_subtracted_just_t2e_')

dephasing_vs_time.plot_with_errs_subtract_T1_individual(
    date_times_t1=date_times_t1_ge, t1_vals=t1_ge_vals, t1_fit_err=t1_ge_fit_err, t1_batch_num=t1_ge_batch_nums,
    date_times_t1_w_noise=date_times_t1_ge_w_noise, t1_vals_w_noise=t1_ge_vals_w_noise,
    t1_fit_err_w_noise=t1_ge_fit_err_w_noise, t1_batch_num_w_noise=t1_ge_batch_nums_w_noise,
    date_times_t2r=date_times_t2r, t2r_vals=t2r_vals, t2r_fit_err=t2r_fit_err,t2r_batch_num=t2r_batch_nums,
    date_times_t2r_w_noise=date_times_t2r_w_noise, t2r_vals_w_noise=t2r_vals_w_noise, t2r_fit_err_w_noise=t2r_fit_err_w_noise,t2r_batch_num_w_noise=t2r_batch_nums_w_noise,
    show_legends=True, extra_label=' - $T1_{ge}$',save_name='t1_ge_subtracted_just_t2r_')


dephasing_vs_time.plot_with_errs_vs_everything(date_times_t2r=date_times_t2r, t2r_vals=t2r_vals, t2r_fit_err=t2r_fit_err,
    date_times_dephasing=date_times_dephasing, dephasing_vals=dephasing_vals, dephasing_fit_err=dephasing_fit_err,
    date_times_t2e=date_times_t2e, t2e_vals=t2e_vals, t2e_fit_err=t2e_fit_err,
    date_times_t2r_w_noise=date_times_t2r_w_noise, t2r_vals_w_noise=t2r_vals_w_noise, t2r_fit_err_w_noise=t2r_fit_err_w_noise,
    date_times_t2e_w_noise=date_times_t2e_w_noise, t2e_vals_w_noise=t2e_vals_w_noise, t2e_fit_err_w_noise=t2e_fit_err_w_noise,
    date_times_dephasing_ef=date_times_dephasing_ef, dephasing_vals_ef=dephasing_vals_ef, dephasing_fit_err_ef=dephasing_fit_err_ef,
    show_legends=True)

dephasing_vs_time.plot_with_errs_vs_everything_t1(date_times_t1_ge=date_times_t1_ge, t1_ge_vals=t1_ge_vals, t1_ge_fit_err=t1_ge_fit_err,
    date_times_t1_fe=date_times_t1_fe, t1_fe_vals=t1_fe_vals, t1_fe_fit_err=t1_fe_fit_err,
    date_times_t1_ge_w_noise=date_times_t1_ge_w_noise, t1_ge_vals_w_noise=t1_ge_vals_w_noise, t1_ge_fit_err_w_noise=t1_ge_fit_err_w_noise,
    date_times_t1_fe_w_ef=date_times_t1_fe_w_ef, t1_fe_vals_ef=t1_fe_vals_ef, t1_fe_fit_err_ef=t1_fe_fit_err_ef,
    show_legends=True)