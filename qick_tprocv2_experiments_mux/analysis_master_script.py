# from syspurpose.files import three_way_merge
import sys
import os
#sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
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
#from analysis_018_box_whisker import PlotBoxWhisker
from AB_Paper_Analysis_Plots import boxwhisker_t1t2_per_qubit_vs_run, boxwhisker_qfreq_per_qubit_vs_run
from analysis_019_allan_welch_stats_plots import AllanWelchStats
from analysis_022_Qfreq_hist_plots import QfreqHistPlots
from section_011_qubit_temperatures_efRabi import QubitTemperatureProgram, QubitTemperatureRefProgram
import matplotlib.pyplot as plt
# from datetime import datetime
import datetime
import pytz
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from system_config import QICK_experiment
import numpy as np
import json
import h5py
# from qualang_tools.plot import Fit
# import visdom
###################################################### Set These #######################################################
save_figs = False
fit_saved = False
show_legends = False
signal = 'None'

figure_quality = 100 #ramp this up to like 500 for presentation plots
final_figure_quality = 200

run_num_list = [9] # options: 4,5,6,7,8,9
t1_vals_by_run  = {}
t2r_vals_by_run = {}
t2e_vals_by_run = {}
qfreq_vals_by_run = {}

t1_errs_by_run  = {}
t2r_errs_by_run = {}
t2e_errs_by_run = {}
qfreq_errs_by_run = {}

for run_number in run_num_list:
    print(f'Processing run {run_number} data.')
    if run_number == 9:
        process_shots_t1ge = False
        per_pt_errs_t1 = False
        run_name = "run9/6transmon/round_robin_benchmark"
        data_path = f"/exp/cosmiq/data/QUIET/QICK_data/{run_name}" #CEPH
                    # f'/data/QICK_data/{run_name}' #daq01
        plots_path = "/home/acolonce/Documents/analysis/coherence" #cosmiqserver01
                    # "/data/QICK_data/run9/6transmon/analysis" #daq01

        top_folder_dates = [
                            "AB_paper_data_batch1_25dB_DACatten_noQ5/2026-04-17_00-34-47", # ignore Q4 in this data, punched out too much!!
                            
                            "AB_paper_data_batch2_25dB_DACatten_onlyQ4/2026-04-17_16-47-30",

                            "AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-17_20-54-40",
                            "AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-17_21-53-52",
                            "AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-17_22-51-33",
                            "AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-17_23-47-15",
                            "AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-18_00-42-05",
                            "AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-18_11-24-10",
                            "AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-18_13-25-28",

                            "AB_paper_data_batch4_25dB_DACatten_noQ5/2026-04-18_23-06-45",
                            "AB_paper_data_batch4_25dB_DACatten_noQ5/2026-04-19_00-16-08",
                            "AB_paper_data_batch4_25dB_DACatten_noQ5/2026-04-19_01-18-51",
                            "AB_paper_data_batch4_25dB_DACatten_noQ5/2026-04-19_17-54-48",

                            "AB_paper_does_no_rpm_fromRR_affect_coh_25dBDAC/2026-04-18_20-58-30",

                            "AB_paper_data_batch5_25dBDAC_onlyQ1_onlySSF/2026-04-20_18-43-20",

                            "AB_paper_data_batch6_25dB_DACatten_onlyQ1/2026-04-20_18-50-54",

                            "AB_paper_data_batch7_25dB_DACatten_noQ5/2026-04-21_02-55-23",

                            "AB_paper_data_batch8_25dB_DACatten_noQ5/2026-04-21_11-34-57",

                            "AB_paper_data_batch9_25dB_DACatten_noQ5Q4/2026-04-22_03-09-46",

                            "AB_paper_data_batch10_25dB_DACatten_onlyQ4/2026-04-22_11-48-15",

                            "AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_16-37-14",
                            "AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_16-54-54",
                            "AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_17-06-49",
                            "AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_17-22-15",
                            "AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_17-35-15",
                            "AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_18-11-40",
                            "AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_19-55-31",

                            "AB_paper_data_batch12_25dB_DACatten_noQ5/2026-04-22_20-53-08",

                            "AB_paper_data_batch13_25dB_DACatten_noQ5noQ6/2026-04-23_06-22-14",

                            "AB_paper_data_batch14_25dB_DACatten_noQ5noQ6/2026-04-23_12-20-26",

                            "AB_paper_data_batch15_25dB_DACatten_onlyQ6/2026-04-23_16-59-52",
                            "AB_paper_data_batch15_25dB_DACatten_onlyQ6/2026-04-23_17-25-33",
                            "AB_paper_data_batch15_25dB_DACatten_onlyQ6/2026-04-23_18-49-54",
                            "AB_paper_data_batch15_25dB_DACatten_onlyQ6/2026-04-23_19-06-16",
                            "AB_paper_data_batch15_25dB_DACatten_onlyQ6/2026-04-23_19-24-51",

                            "AB_paper_data_batch16_25dB_DACatten_noQ5/2026-04-23_20-14-15",
                            "AB_paper_data_batch16_25dB_DACatten_noQ5/2026-04-23_20-17-04",
                            "AB_paper_data_batch16_25dB_DACatten_noQ5/2026-04-23_21-22-18",
                            "AB_paper_data_batch16_25dB_DACatten_noQ5/2026-04-23_22-29-25",
                            "AB_paper_data_batch16_25dB_DACatten_noQ5/2026-04-23_23-40-11",

                            "AB_paper_data_batch17_25dB_DACatten_noQ5/2026-04-24_00-51-38",

                            "AB_paper_data_batch18_25dB_DACatten_noQ5noQ1/2026-04-24_14-37-30",

                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-24_18-51-32",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-24_21-13-58",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-24_23-27-19",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-21-18",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-22-27",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-30-29",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-33-26",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-34-54",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-36-30",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-48-04",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-57-26",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_00-04-04",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_00-35-21",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_12-49-11",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_12-55-00",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_12-57-51",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_12-59-36",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_13-02-45",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_13-04-33",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_13-25-03",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_13-27-10",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_13-39-30",
                            "AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_13-41-18",

                            "AB_paper_data_batch20_25dB_DACatten_noQ5/2026-04-26_13-45-45",
                            "AB_paper_data_batch20_25dB_DACatten_noQ5/2026-04-26_20-58-39",

                            "AB_paper_data_batch21_25dB_DACatten_noQ5/2026-04-27_08-07-32",
                            "AB_paper_data_batch21_25dB_DACatten_noQ5/2026-04-27_11-41-29",
                            "AB_paper_data_batch21_25dB_DACatten_noQ5/2026-04-27_13-07-15"]

    if run_number == 8:
        process_shots_t1ge = True
        per_pt_errs_t1 = True
        run_name = "run8/6transmon/round_robin"
        # 'run8/6transmon/round_robin/temperature_sweep_qubit_data'
        # 'run8/6transmon/round_robin/AB_paper_datadump_for_analysis'
        data_path = f"/exp/cosmiq/data/QUIET/QICK_data/{run_name}" #CEPH
            # f'/data/QICK_data/{run_name}' # daq01
        plots_path = "/home/acolonce/Documents/analysis/coherence" #cosmiqserver01
            #"/data/QICK_data/run8/6transmon/analysis" #daq01

        # all of run 8 data
        top_folder_dates = [
            "AB_Paper_Data_24hrs/2025-10-19_11-09-32",  # only T1 shots, no T1 QICK-averaged IQ data
            "AB_Paper_Data_24hrs/2025-10-19_12-05-25",  # only T1 shots, no T1 QICK-averaged IQ data
            "AB_Paper_Data_24hrs/2025-10-19_19-43-00",  # only T1 shots, no T1 QICK-averaged IQ data
            "AB_Paper_Data_24hrs/2025-10-19_20-25-18",  # only T1 shots, no T1 QICK-averaged IQ data
            "AB_Paper_Data_24hrs/2025-10-20_12-10-19",  # only T1 shots, no T1 QICK-averaged IQ data

            "ABpaperdata2ndbatch_21dB_DACatten_Q1to5/2025-10-23_00-49-28",  # only T1 shots, no T1 QICK-averaged IQ data

            "ABpaperdata3rdbatch_21dB_DACatten_Q1to5/2025-10-23_14-47-22",  # only T1 shots, no T1 QICK-averaged IQ data
            # "ABpaperdata3rdbatch_21dB_DACatten_Q1to5_not1shots/2025-10-24_01-41-30",  # no T1 shots saved, only QICK averaged IQ data. Leave commented out. Need to debug script to incorporate this
            "ABpaperdata3rdbatch_21dB_DACatten_Q1to5_t1shots_optional/2025-10-24_13-58-37", # From this point forward, both T1 shots and averaged IQ arrays were saved
            "ABpaperdata3rdbatch_21dB_DACatten_Q6_t1shots_optional/2025-10-27_14-15-40",
            "ABpaperdata3rdbatch_21dB_DACatten_Q6_t1shots_optional/2025-10-27_14-24-29",
            "ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional/2025-10-27_22-04-57",

            "ABpaperdata_21dB_DACatten_Q1to6_t1shots_optional_newopt/2025-10-28_21-57-47",
            "ABpaperdata_21dB_DACatten_Q1to6_t1shots_optional_newopt/2025-10-29_18-38-25",
            "ABpaperdata_21dB_DACatten_Q1to6_t1shots_optional_newopt/2025-10-29_23-48-45",

            "18dB_DAC_testdata_allQs_exceptQ4/2025-10-31_01-54-57",

            "19dB_DAC_testdata_allQs/2025-10-31_20-40-11",
            "19dB_DAC_testdata_allQs/2025-11-01_12-54-55",
        ]

        # All run 8 qubit temperature sweep data except the 200mK dataset bc no qubits visible
        # top_folder_dates = [
        #     "temperature_sweep_run8_25dBDAC_onechan_day1/2025-11-18_08-39-37",
        #     "temperature_sweep_run8_25dBDAC_onechan_day1/2025-11-18_09-02-01",
        #     "temperature_sweep_run8_25dBDAC_onechan_day1/2025-11-18_12-40-59",
        #     "temperature_sweep_run8_25dBDAC_onechan_day1/2025-11-18_14-26-01",
        #     "temperature_sweep_run8_25dBDAC_onechan_day1/2025-11-18_14-48-11",
        #
        #     "temperature_sweep_run8_25dBDAC_onechan_day2/2025-11-19_08-04-25",
        #     "temperature_sweep_run8_25dBDAC_onechan_day2/2025-11-19_11-00-00",
        #     "temperature_sweep_run8_25dBDAC_onechan_day2/2025-11-19_11-27-04",
        #
        #     "temperature_sweep_run8_25dBDAC_onechan_day3/2025-11-20_07-31-49",
        #
        #     "temp_sweep_run8_25dBDAC_onechan_day4_175mK/2025-11-21_08-01-57",
        #     "temp_sweep_run8_25dBDAC_onechan_day4_175mK/2025-11-21_08-33-09",
        #     "temp_sweep_run8_25dBDAC_onechan_day4_175mK/2025-11-21_08-45-17",
        #     "temp_sweep_run8_25dBDAC_onechan_day4_175mK/2025-11-21_08-54-05"]

    elif run_number == 7:
        process_shots_t1ge = False
        per_pt_errs_t1 = False
        run_name = 'run7/6transmon/round_robin_benchmark/AB_paper_data'
        data_path = f"/exp/cosmiq/data/QUIET/QICK_data/{run_name}" #CEPH
            # f'/data/QICK_data/{run_name}' #daq01
        plots_path = "/home/acolonce/Documents/analysis/coherence" #cosmiqserver01
            #"/data/QICK_data/run7/6transmon/analysis" #daq01

        # all dates:
        top_folder_dates = ["2025-07-19_08-34-39",
                            "2025-07-19_16-16-14",
                            "2025-07-19_16-56-45",
                            "2025-07-19_23-11-39",
                            "2025-07-20_06-33-03" ]

    elif run_number == 6:
        process_shots_t1ge = False
        per_pt_errs_t1 = False
        run_name = 'run6/6transmon'
        data_path = f"/exp/cosmiq/data/QUIET/QICK_data/{run_name}" #CEPH
            #f'/data/QICK_data/{run_name}' #daq01
        plots_path = "/home/acolonce/Documents/analysis/coherence" #cosmiqserver01
            #"/data/QICK_data/run6/6transmon/analysis" #daq01

        # all pre-science run data (AB paper data):
        # Can be found oth locally in daq01 or on CEPH
        top_folder_dates = [
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-21",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-22",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-23",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-24",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-26",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-28",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-03-01",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-03-02"]

        # Science run data: can ONLY be found on CEPH!!
        # If you want to process all "science run" data
        # "TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/2025-05-15_14-47-38",
        # "TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/2025-05-15_18-08-15",
        # "TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/2025-05-15_22-02-12",
        # "TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/2025-05-16_01-28-20",
        # "TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/2025-05-16_04-49-59",
        # "TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/2025-05-16_08-13-47",
        #
        # "TLS_Comprehensive_Study/source_off_detuning_24MHz_Q1_substudy1/2025-05-15_11-19-50",
        # "TLS_Comprehensive_Study/source_off_detuning_24MHz_Q1_substudy1/2025-05-15_18-35-56",
        #
        # "TLS_Comprehensive_Study/source_off_post_temperature_sweep_substudy1/2025-05-14_19-25-55",
        # "TLS_Comprehensive_Study/source_off_post_temperature_sweep_substudy1/2025-05-14_22-50-51",
        # "TLS_Comprehensive_Study/source_off_post_temperature_sweep_substudy1/2025-05-15_02-29-34",
        # "TLS_Comprehensive_Study/source_off_post_temperature_sweep_substudy1/2025-05-15_05-50-12",
        # "TLS_Comprehensive_Study/source_off_post_temperature_sweep_substudy1/2025-05-15_09-13-30",
        #
        # "TLS_Comprehensive_Study/source_off_substudy1/2025-04-15_21-24-46",
        #
        # "TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_11-47-09",
        # "TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_12-51-09",
        # "TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_17-50-00",
        # "TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_22-47-49",
        # "TLS_Comprehensive_Study/source_off_substudy2/2025-04-17_03-42-36",
        # "TLS_Comprehensive_Study/source_off_substudy2/2025-04-17_08-42-24",
        #
        # "TLS_Comprehensive_Study/source_off_substudy3/2025-04-17_12-28-37",
        # "TLS_Comprehensive_Study/source_off_substudy3/2025-04-17_17-22-46",
        # "TLS_Comprehensive_Study/source_off_substudy3/2025-04-17_22-16-39",
        # "TLS_Comprehensive_Study/source_off_substudy3/2025-04-18_01-45-53",
        # "TLS_Comprehensive_Study/source_off_substudy3/2025-04-18_06-40-55",
        #
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-18_11-59-33",
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-18_16-56-58",
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-18_21-51-13",
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_02-45-41",
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_07-39-57",
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_12-34-26",
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_17-48-44",
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_22-43-02",
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_03-37-50",
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_08-32-36",
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_13-26-47",
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_18-25-13",
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_23-25-04",
        # "TLS_Comprehensive_Study/source_off_substudy4/2025-04-21_04-23-31",
        #
        # "TLS_Comprehensive_Study/source_off_substudy5/2025-05-04_20-56-05",
        # "TLS_Comprehensive_Study/source_off_substudy5/2025-05-04_23-28-05",
        # "TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_03-03-40",
        # "TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_06-40-15",
        # "TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_10-18-53",
        # "TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_13-57-22",
        # "TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_17-34-21",
        # "TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_21-18-14",
        # "TLS_Comprehensive_Study/source_off_substudy5/2025-05-06_02-18-57",
        #
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-06_11-30-17",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-06_14-50-55",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-06_18-14-29",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-06_21-35-26",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_01-00-14",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_04-23-45",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_07-46-44",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_11-09-17",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_14-30-29",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_17-50-59",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_21-13-50",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-08_00-36-15",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-08_03-56-41",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-08_07-19-10",
        # "TLS_Comprehensive_Study/source_off_substudy6/2025-05-08_11-53-46"]

    elif run_number == 5:
        process_shots_t1ge = False
        per_pt_errs_t1 = False
        run_name = 'run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20'
        data_path = f"/exp/cosmiq/data/QUIET/QICK_data/{run_name}" #CEPH
            #f'/data/QICK_data/{run_name}' #daq01
        plots_path = "/home/acolonce/Documents/analysis/coherence" #cosmiqserver01
            #"/data/QICK_data/run5/6transmon/analysis" #daq01

        # all dates:
        top_folder_dates = [ # Condensing started 12/8/2024
                            "2024-12-09",
                            "2024-12-10",
                            "2024-12-11",
                            "2024-12-12",
                            "2024-12-13",
                            "2024-12-14",
                            "2024-12-15",
                            "2024-12-16",
                            "2024-12-17",
                            "2024-12-18",
                            "2024-12-19",
                            "2024-12-20"]
    elif run_number == 4:
        process_shots_t1ge = False
        per_pt_errs_t1 = False
        run_name = 'run4/6transmon/Official_run4_RR_Data_which_started_Nov21'
        data_path = f"/exp/cosmiq/data/QUIET/QICK_data/{run_name}" #CEPH
            #f'/data/QICK_data/{run_name}' #daq01
        plots_path = "/home/acolonce/Documents/analysis/coherence" #cosmiqserver01
            #"/data/QICK_data/run4/6transmon/analysis" #daq01

        # all dates:
        top_folder_dates = [
                            "2024-11-21",
                            "2024-11-23",
                            "2024-11-24",
                            "2024-11-25",
                            "2024-12-09",
                            "2024-12-10"]

    FRIDGE = "QUIET"
    run_notes = ('Added IR shielding, better cryo terminators, thermalizing with 0dB attenuator ') #please make it brief for the plot

    ################################################ 01: Get all data ######################################################
    # res_spec_vs_time = ResonatorFreqVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
    #                                        save_figs, fit_saved, signal, run_name, FRIDGE)
    # date_times_res_spec, res_freqs = res_spec_vs_time.run()
    # #
    # q_spec_vs_time = QubitFreqsVsTime(data_path, plots_path, figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
    #                                   save_figs, fit_saved, signal, run_name, FRIDGE)
    # date_times_q_spec, q_freqs, qspec_fit_err = q_spec_vs_time.run(exp_extension='_ge', use_png_timestamps = False)

    #print("qspec fit errs Q1: ", qspec_fit_err[0])
    #print("mean qspec fit err Q1: ", np.mean(qspec_fit_err[0]))

    # pi_amps_vs_time = PiAmpsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
    #                               fit_saved,signal, run_name)
    # date_times_pi_amps, pi_amps = pi_amps_vs_time.run(plot_depths=False)

    t1_vs_time = T1VsTime(plots_path, figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                     signal, run_name, FRIDGE, run_number, per_pt_errs = per_pt_errs_t1)

    if per_pt_errs_t1 and process_shots_t1ge: # this will only work if process_shots_t1ge is set to True too
        date_times_t1, t1_vals, t1_fit_err, I_per_pt_errs, Q_per_pt_errs = t1_vs_time.run(return_errs=True, exp_extension = '_ge', process_shots = process_shots_t1ge)
    else:
        date_times_t1, t1_vals, t1_fit_err = t1_vs_time.run(return_errs=True, exp_extension = '_ge')

    t2r_vs_time = T2rVsTime(plots_path, run_number, figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
                            fit_saved, signal, run_name, FRIDGE)
    date_times_t2r, t2r_vals, t2r_fit_err = t2r_vs_time.run(return_errs=True, t1_vals = t1_vals)

    t2e_vs_time = T2eVsTime(plots_path, run_number, figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
                            fit_saved, signal, run_name, FRIDGE)
    date_times_t2e, t2e_vals, t2e_fit_err = t2e_vs_time.run(return_errs=True, t1_vals = t1_vals)

    # ---------------- Store results ----------------
    ## stores data like t1_vals_by_run[6][3], where 6=run number and 3=qubit index (0 based)
    t1_vals_by_run[run_number] = t1_vals
    t1_errs_by_run[run_number] = t1_fit_err

    t2r_vals_by_run[run_number] = t2r_vals
    t2r_errs_by_run[run_number] = t2r_fit_err

    t2e_vals_by_run[run_number] = t2e_vals
    t2e_errs_by_run[run_number] = t2e_fit_err

    # qfreq_vals_by_run[run_number] = q_freqs
    # qfreq_errs_by_run[run_number] = qspec_fit_err

######################################## Print QICK soccfg live ###########################################
# If you want to print out the soccfg QICK output, uncomment this:
# from tprocv2_demos.qick_tprocv2_experiments_mux.socProxy import makeProxy
# soc, soccfg = makeProxy()
# print(soccfg)

######################################## 02: Plot All Individual RR Plots ###########################################
# date = "2025-10-31_20-40-11"
# #date = "2025-10-27_22-04-57"  #only plot all of the data for one date at a time because there is a lot
# process_shots_t1 = False # only for analyzing T1 data
# unique_folder_path = "" # only used when plot_rabis_Qtemps = True or for load_t1_shots_vs_avgIQ_arrays()

# # --------------------------------------------- For looping through the data --------------------------------
# outerFolder = f"/data/QICK_data/run8/6transmon/round_robin/AB_Paper_Data_24hrs/{date}/study_data"
# outerFolder = f"/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/{date}/study_data"
# outerFolder = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional/2025-10-27_22-04-57/study_data"
                #f"/data/QICK_data/run8/6transmon/round_robin/ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional/{date}/study_data"
#outerFolder = f"/data/QICK_data/run8/6transmon/round_robin/ABpaperdata3rdbatch_21dB_DACatten_Q1to5_not1shots/{date}/study_data"
# outerFolder = fr"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\19dB_DAC_testdata_allQs\{date}\study_data"

# # ---------------------------------------------- For saving RR plots ---------------------------------------------
# outerFolder_save_plots = f"/data/QICK_data/run8/6transmon/replotted_RR_data/{date}/"
# outerFolder_save_plots = fr"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\19dB_DAC_testdata_allQs/replotted_RR_data/{date}/"
# # outerFolder_save_plots = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional/replotted_RR_data/2025-10-27_22-04-57/t2r_ge/"
# #C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\19dB_DAC_testdata_allQs
#plotter = PlotAllRR(date, figure_quality, save_figs, fit_saved, signal, run_name, run_number, tot_num_of_qubits, outerFolder = outerFolder,
                  #outerFolder_save_plots = outerFolder_save_plots, unique_folder_path = unique_folder_path, process_shots = process_shots_t1)
#plotter.run(plot_res_spec = False, plot_q_spec = False, plot_rabi = False, rabi_rolling_avg=False, plot_ss = False,
            #plot_ss_hist_only=False,ss_plot_title = None, ss_plot_gef = False, plot_t1 = False,
            #plot_t2r = True, plot_t2e = False, plot_rabis_Qtemps = False)

########################################### 03: Resonator Freqs vs Time Plots ###########################################
# res_spec_vs_time.plot(date_times_res_spec, res_freqs, show_legends)
#
# ######################################### 04: Qubit Freqs vs Time Plots #############################################
#q_spec_vs_time.plot_without_errs(date_times_q_spec, q_freqs,show_legends)
#q_spec_vs_time.plot_with_errs(date_times_q_spec, q_freqs, qspec_fit_err, show_legends) # shows error bars, do this one!!
#q_spec_vs_time.plot_with_errs_single_plot(date_times_q_spec, q_freqs, qspec_fit_err, show_legends=True)

# ############################################## 05: Pi Amp vs Time Plots ###############################################
# pi_amps_vs_time.plot(date_times_pi_amps, pi_amps, show_legends)

# #----------------------------------------------Extra pi amp analysis----------------------------------------------------
# #can only have the 'depths' argument returned here if plot_depths=True, otherwise delete it
# date_times, pi_amps, depths = pi_amps_vs_time.run(plot_depths=True)
#
# pi_amps_vs_time.plot_vs_signal_depth(date_times, pi_amps, depths, show_legends)
# pi_amps_vs_time.plot_signal_depth_vs_time(date_times, pi_amps, depths, show_legends)
#
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

# ################################################ 06: T1 vs Time Plots #################################################
# t1_vs_time.plot_without_errs(date_times_t1, t1_vals, show_legends)
#t1_vs_time.plot_with_errs(date_times_t1, t1_vals, t1_fit_err, show_legends) # shows error bars, do thi sone!!
# t1_vs_time.plot_with_errs_single_plot(date_times_t1, t1_vals, t1_fit_err, show_legends=True)
#
# ################################################# 07: T2R vs Time Plots ################################################
# #t2r_vs_time.plot_without_errs(date_times_t2r, t2r_vals, t2r_fit_err, show_legends)
#t2r_vs_time.plot_with_errs(date_times_t2r, t2r_vals, t2r_fit_err, show_legends) # shows error bars, do this one!!
# t2r_vs_time.plot_with_errs_single_plot(date_times_t2r, t2r_vals, t2r_fit_err, show_legends=True)
#
# ################################################# 08: T2E vs Time Plots ################################################
# #t2e_vs_time.plot_without_errs(date_times_t2e, t2e_vals, t2e_fit_err, show_legends)
#t2e_vs_time.plot_with_errs(date_times_t2e, t2e_vals, t2e_fit_err, show_legends) # shows error bars, do this one!!
# t2e_vs_time.plot_with_errs_single_plot(date_times_t2e, t2e_vals, t2e_fit_err, show_legends=True)
############################################### Qubit Frequency hist Plots #############################################
# qfreq_distribution_plots = QfreqHistPlots(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
#                                             save_figs, fit_saved, signal, data_path, plots_path, run_name, fridge=FRIDGE)
# qfreq_distribution_plots.run(q_freqs, qspec_fit_err)
# ######################################### 09: T1 hist/cumul/err Plots (not in use anymore; we use box and whisker plots for medians) #############################################
# t1_distribution_plots = T1HistCumulErrPlots(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
#                                             save_figs, fit_saved, signal, data_path, plots_path, run_name, run_notes, run_number, fridge=FRIDGE)
# dates, t1_vals, t1_errs = t1_distribution_plots.run(exp_extension="_ge", process_shots = process_shots_t1ge)
# t1_std_values, t1_mean_values = t1_distribution_plots.plot(dates, t1_vals, t1_errs, show_legends)

# # # # ###################################### 10: T2R hist/cumul/err Plots (not in use anymore; we use box and whisker plots for medians) ############################################
# t2r_distribution_plots = T2rHistCumulErrPlots(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
#                                             save_figs, fit_saved, signal, data_path, plots_path, run_name, fridge=FRIDGE)
# dates, t2r_vals, t2r_errs = t2r_distribution_plots.run(t1_vals = t1_vals)
# t2r_std_values, t2r_mean_values = t2r_distribution_plots.plot(dates, t2r_vals, t2r_errs, show_legends)
# # # # #
# # # ####################################### 11: T2E hist/cumul/err Plots (not in use anymore; we use box and whisker plots for medians) ############################################
# t2e_distribution_plots = T2eHistCumulErrPlots(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
#                                             save_figs, fit_saved, signal, data_path, plots_path, run_name, fridge=FRIDGE)
# dates, t2e_vals, t2e_errs = t2e_distribution_plots.run(t1_vals = t1_vals)
# t2e_std_values, t2e_mean_values = t2e_distribution_plots.plot(dates, t2e_vals, t2e_errs, show_legends)

# ############################ 12: Save the Key Statistics for This Run to Compare Later #################################
# May or may not still work. THis was added by Olivia and used to work before, but Arianna hasn't used it in a while.
#need to run 00,01, and 08-10 before this to get all of the variables
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
# run_number_list = [1,2,3]
# comparing_runs = CompareRuns(run_number_list, run_name)

# run_stats_folder = f"run_stats/QUIET/run{2}/"
# filename = run_stats_folder + 'experiment_data.h5'
# loaded_data = comparing_runs.load_from_h5(filename)

# t1_vals_r2 = loaded_data['t1_vals']
# t2r_vals_r2 = loaded_data['t2r_vals']
# t2e_vals_r2 = loaded_data['t2e_vals']
# comparing_runs.plot_freqs_vs_run()
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
# plotter = PlotMetricDependencies(run_name, tot_num_of_qubits, final_figure_quality, fridge=FRIDGE)

# plotter.plot(date_times_q_spec, q_freqs, date_times_t1, t1_vals, metric_1_label = 'Q Freq (MHz)',
#              metric_2_label = 'T1 (us)')
# plotter.plot(date_times_pi_amps, pi_amps, date_times_t1, t1_vals, metric_1_label = 'Pi Amp (a.u.)',
#              metric_2_label = 'T1 (us)')
#
# plotter.plot(date_times_q_spec, q_freqs, date_times_t2r, t2r_vals, metric_1_label = 'Q Freq (MHz)',
#              metric_2_label = 'T2R (us)')
# plotter.plot(date_times_pi_amps, pi_amps, date_times_t2r, t2r_vals, metric_1_label = 'Pi Amp (a.u.)',
#              metric_2_label = 'T2R (us)')
#
# plotter.plot(date_times_q_spec, q_freqs, date_times_t2e, t2e_vals, metric_1_label = 'Q Freq (MHz)',
#              metric_2_label = 'T2E (us)')
# plotter.plot(date_times_pi_amps, pi_amps, date_times_t2e, t2e_vals, metric_1_label = 'Pi Amp (a.u.)',
#              metric_2_label = 'T2E (us)')
#
# plotter.plot(date_times_q_spec, q_freqs, date_times_pi_amps, pi_amps, metric_1_label = 'Q Freq (MHz)',
#              metric_2_label = 'Pi Amp (a.u.)')

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
##################################### 17: Box And Whisker Qubit Comparison ############################################
# -------------------------- Old way -------------------------------
# boxwhisker = PlotBoxWhisker(run_name, tot_num_of_qubits, final_figure_quality)
# # # boxwhisker.plot(res_freqs, metric_label="Resonator Frequencies (MHz)")
# # # boxwhisker.plot(q_freqs, metric_label="Qubit Frequencies (MHz)")
# # # boxwhisker.plot(pi_amps, metric_label="Pi Amplitude (a.u.)")
# # # boxwhisker.plot(t1_vals, metric_label="T1 (µs)")
# # # boxwhisker.plot(t2r_vals, metric_label="T2R (µs)")
# # # boxwhisker.plot(t2e_vals, metric_label="T2E (µs)")
# # # boxwhisker.plot_three_metrics(t1_vals, t2r_vals, t2e_vals)
# means = q_spec_vs_time.plot_hist(q_freqs, show_legends)
#boxwhisker.plot_three_metrics_by_freq(means, t1_vals, t2r_vals, t2e_vals)
#boxwhisker.plot_three_metrics_by_freq_x_break(means, t1_vals, t2r_vals, t2e_vals)
# boxwhisker.plot_three_metrics_by_freq_comp_run_x_break(means, t1_vals, t2r_vals, t2e_vals,t1_vals_r2, t2r_vals_r2, t2e_vals_r2, plot_outliers=False)

#------------------------ New way for AB Paper, by Arianna -------------------------------------------
## Coherence box plots
boxwhisker_t1t2_per_qubit_vs_run(
    run_num_list,
    t1_vals_by_run=t1_vals_by_run,
    t2r_vals_by_run=t2r_vals_by_run,
    t2e_vals_by_run=t2e_vals_by_run,
    do_T1=True, do_T2R=True, do_T2E=True,
    ylims=(0, 180),
    yticks=np.arange(0, 181, 20),
    mode="separate",
    save_plt_path = "/home/acolonce/Documents/analysis/multirun/coherence" #cosmiqserver01
                    # '/data/QICK_data/multirun_analysis/coherence_analysis' #daq01
)

## Qubit freq box plots
# ge_qfreq_centers = [4189.8773, 3820.4723, 4161.3726, 4463.15226, 4471.43854, 4997.86] # plots will be centered around these vals
# boxwhisker_qfreq_per_qubit_vs_run(
#     run_num_list,
#     qfreq_vals_by_run=qfreq_vals_by_run,
#     qfreq_errs_by_run=qfreq_errs_by_run,
#     qfreq_centers=ge_qfreq_centers,
#     freq_window=50.0,
#     save_plt_path = "/data/QICK_data/multirun_analysis/coherence_analysis") # set to 'None' to use plt.show()

# # ################################## 18: Allan Deviation/ Welch Spectral Density #########################################
# stats = AllanWelchStats(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
#                  signal, run_name)
# stats.plot_allan_deviation(date_times_q_spec, q_freqs, show_legends, label='QFreq')
# stats.plot_allan_deviation(date_times_t1, t1_vals, show_legends, label='T1')
# stats.plot_allan_deviation(date_times_t2r, t2r_vals, show_legends, label='T2R')
# stats.plot_allan_deviation(date_times_t2e, t2e_vals, show_legends, label='T2E')
#
# stats.plot_welch_spectral_density(date_times_q_spec, q_freqs, show_legends, label='QFreq')
# stats.plot_welch_spectral_density(date_times_t1, t1_vals, show_legends, label='T1')
# stats.plot_welch_spectral_density(date_times_t2r, t2r_vals, show_legends, label='T2R')
# stats.plot_welch_spectral_density(date_times_t2e, t2e_vals, show_legends, label='T2E')

# ################################################### 19: Extra #########################################################
# plotter = PlotMetricDependencies(run_name, tot_num_of_qubits, final_figure_quality, FRIDGE)
# plotter.plot_shared_datetimes(date_times_q_spec, q_freqs, qspec_fit_err, metric_1_label = 'Q Freq (MHz)',
#              metric_2_label = 'Q Freq Fit Err (MHz)')
# plotter.scatter_plot_two_y_axis(date_times_q_spec, q_freqs, date_times_q_spec, qspec_fit_err, metric_1_label = 'Q Freq (MHz)',
#              metric_2_label = 'Q Freq Fit Err (MHz)')

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