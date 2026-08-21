import sys
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import os
#sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from analysis_021_plot_allRR_noqick import QubitSpectroscopy
from qicklab.analysis.qspec import AnaQSpec
from qicklab.analysis.ssf import AnaSSF
from Arianna_non_prebuilt_SSF_doublegauss_funcs import non_prebuilt_ssf_analysis_class
from section_008_save_data_to_h5 import Data_H5
from analysis_014_temp_calcsandplots_cosmiqgpvm import SSFTempCalcAndPlots, combined_Qtemp_studies, RPMTempCalcAndPlots
import glob
import re
import pickle
import datetime
from analysis_002_res_centers_vs_time_plots import ResonatorFreqVsTime
import ast
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import math
import h5py
import pandas as pd
from expt_config import expt_cfg, list_of_all_qubits, FRIDGE
from analysis_021_plot_allRR_noqick import PlotRR_noQick
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime
from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_007_T2R_vs_time_plots import T2rVsTime
from analysis_008_T2E_vs_time_plots import T2eVsTime
from AB_Paper_Analysis_Plots import boxwhisker_qtemps_per_qubit_vs_run_choice, boxwhisker_pe_per_qubit_vs_run_hybrid, boxwhisker_ssf_per_qubit_vs_run, boxwhisker_snr_per_qubit_vs_run, boxwhisker_ie_new_Pg_per_Q_vs_run
from pathlib import Path
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
run_num = 8 #Options for QUIET: 5(for coherence only),6,7,8,9,9.2(this is run 9c)
run_name = f'run{run_num}/6transmon' # this is for temps analysis, for coherence analysis it's defined in its respective section
signal = 'None' # Do not change
final_figure_quality = 200 # plot quality
plot_ssf_gef = False # Do you want to re-plot g-e-f SSF data and save the plots?
replot_RPMs = False # Do you want to re-plot rabi population measurements from RR data but not extract temps? Only make plots
save_figsRR = False # Do you want to save (or not save) the RR RPM plots as you process the data? aka when get_qtemp_data = True
save_figs = False # To be used in general for any function or class to save (or not save) plots.
save_figs_SSF = False # Do you want to save gaussian fit plots while calculating ssf qtemps? iminuit case only
fit_saved = False # Not used here, set to false.
exclude_temp_sweeps = False # Do you want to exclude the folders that contain data taken during the heater temperature sweep?
filter_out_bad_RPM_fits = True # filter out bad rpm fits? this doesn't work perfect but helps a bit
filter_out_bad_SSF_qtemp_fits = True # filter out SSF data that can't be properly fitted for qubit temp calcs?
get_qtemp_data = True # Do you want to calculate RPM qubit temperatures? This returns RPM qubit temperatures and qubit freqs for specified dates.
get_london_data = False # This returns RPM qubit temperatures, resonator freqs, and qubit freqs for specified dates. Designed for London Penetration analysis.

pre_sciencerun6_data = True # Do you also want to incorporate the run 6 pre-science run data? This only applies when run_num = 6

use_iminuit_gdoublegauss_ssf = True # do you want to fit the g-state to a double gaussian using iminuit? The default is GMM instead
#Double gaussian fitting is optimized for lower thermal pops (<2%) if this is set to true:
low_thermal_pops = True if run_num == 9 else False # This run number is specific to QUIET. Used in SSF qtemps
calc_SNR_ssfqtemps = True # calculate SNR of SSF scans?
calc_SSF_e_decay = True # calculate the e-state decay population in SSF scans?

# When re-plotting SSF g-state histograms using iminuit, do you want to limit y-axis to see thermal pop region better?:
ssf_hist_ylim = None

rpm_combine_IQ_signal = False # uses ssf angle to rotate rabi population measurement data into a combined IQ signal lying on the same axis

figure_quality = 200
theta = 0
threshold = 0
tot_num_of_qubits = 6 # Total number of qubits currently at QUIET

# What method or methods do you want to use to calculate qubit temperatures?
qtemp_method_flags = {"Qtemps_viaRPM": False, "Qtemps_viaSSF_ge_thresh": False, "Qtemps_viaSSF_gmeans_thresh": False, "Qtemps_viaSSF_with_fallback": False,
                      "combined_studies_Qtemps": False}

# What analysis plots do you want to make?
analysis_flags = {"Qtemps_vs_time_viaSSF": False,  "Qtemps_vs_time_viaRPM": False, "Threshold_Check_Qtemps_viaSSF": False, "ge_thresh_check_ssf": False,
                  "Qtemps_hists_viaRPM": False, "Pe_hists_viaRPM": False, "Qtemps_hists_viaSSF": False, "Pe_hists_viaSSF": False, "Pe_vs_time_viaRPM": False,
                  "qtemps_Pe_vs_time_viaRPM": False, "qtemps_Pe_gefreq_vs_time_viaRPM": False, "SSF_vs_time": False, "SSF_fid_vs_Pe_viaSSF": False, "ssf_SNR_vs_time": False}

# For combined analysis (SSF qtemps + RPM qtemps analyses OR analyses across multiple runs). To enable these set "combined_studies_Qtemps" to True in qtemp_method_flags
comb_analysis_flags = {"load_rpm": False, "load_ssf": False, "use_cached_qtemp_files": False, "create_cached_qtemp_files": False, "Qtemps_vs_time_comb_separate_plts": False,"Qtemps_vs_time_comb_single_plt": False,
                       "Pe_vs_time_comb_separate_plts": False, "Pe_vs_time_comb_single_plt": False, "qtemp_box_whisker_allruns_allQs": False, "Pe_box_whisker_allruns_allQs": False, "ssf_box_whisker_allruns_allQs": False,
                       "plot_ssf_log_curves": False, "SSF_fid_vs_RRPM_Pe_2D": False, "SSF_fid_vs_RRPM_Pe_3D": False, "SSF_fid_vs_RRPM_Pe_video": False, "SNR_vs_RRPM_Pe": False,
                       "SNR_box_whisker_allruns_allQs": False, "ie_new_Pg_boxwhisk_allruns_allQs": False, "multirun_RPM_Pe_vs_t": False}

# For London Penetration Depth analysis
london_flags = {"get_qfreqs_resfreqs_qtemps": False}

# For double-gaussian SSF analysis using alternative methods (does not require any other flags to be set to True above!)
alt_ssf_analysis_flags = {"jupyter_method_Arianna": False, "iminuit_method": False}

# For coherence-qubit temps combined analysis. These flags act on their own, no need to set anything above to True.
coh_qtemp_ana_flags = {"run_qtemps_section": True, "run_coherence_section": True, "use_cached_qtemp_files": False, "use_cached_coherence_files": False, "create_cached_qtemp_files": False,"create_cached_coherence_files": False, "load_mcp1_temps": True,
                       "plot_qtemps_t1_ftemps_qfreq": True, "plot_RPM_qtemps_qfreq_fridge_only": False, "plot_SSF_qtemps_qfreq_fridge_only": False,
                       "SSF_lims_per_scan_viaSSF": False, "SSF_lims_per_scan_viaRPM": False}

############################################################################## Set up #######################################################################################################################
#----------------------------------------------------- For qubit temperature calculations via rabi population measurements --------------------------------------------------------------------------------------
# Specify which dates you want to loop through. It will process all the files inside all the folders that contain these dates in their title.
# ----------------------------------------------------------------------------- Run 6 --------------------------------------------------------
target_dates_qtemps_RPM_sciencerun = [
    "2025-04-16",
    "2025-04-17",
    "2025-04-18",
    "2025-04-19",
    "2025-04-20",
    "2025-04-21", #starts source on (Co)
    "2025-04-22",
    "2025-04-23", #switched source (to Cs)
    "2025-04-24",
    "2025-04-25",
    "2025-04-26",
    "2025-04-27",
    "2025-04-28", #Cs source moved closer
    "2025-04-29",
    "2025-04-30",
    "2025-05-01",
    "2025-05-02",
    "2025-05-03",
    "2025-05-04", # Cs source removed. No sources in Cleanroom.
    "2025-05-05",
    "2025-05-06",
    "2025-05-07",
    "2025-05-08",
    "2025-05-09",
    "2025-05-10",
    "2025-05-11",
    "2025-05-12",
    "2025-05-13",
    "2025-05-14",
    "2025-05-15",
    "2025-05-16",
    "2025-05-20",
    "2025-05-21",
    "2025-05-28",
    "2025-05-29",
    "2025-05-31",
    "2025-06-01" # Last Science run data
    ]

# For data during Heater temperature steps, run 6 (20mK to 160mK)
# target_dates_qtemps_RPM_sciencerun = ["2025-05-08", "2025-05-09", "2025-05-10", "2025-05-11", "2025-05-12", "2025-05-13", "2025-05-14"]

# For pre-science-run data
target_dates_qtemps_RPM_presciencerun = [
                                        '2025-04-11',
                                        '2025-04-12'
                                        ]

# Base path of where the data is stored up to the Study Name (TLS_Comprehensive_Study or ef_studies_pre_science_run)
base_dir_sciencerun = "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study" # for QUIET run 6 science run data
                    # Options: no other options. Science run data is only on CEPH
base_dir_pre_sciencerun = "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ef_studies_pre_science_run" # for run 6 pre-science run data
                    # Options:
                    # "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ef_studies_pre_science_run" # CEPH
                    # "/data/QICK_data/run6/6transmon/ef_studies_pre_science_run" # daq01
# To save plots
r6_plts_prefix = "/home/acolonce/Documents/analysis" #cosmiqserver01
# Options:
# "/home/acolonce/Documents/analysis" #cosmiqserver01
# "/data/QICK_data/run6/6transmon/analysis" #daq01
# "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/run6" #1hw

# For round robin plots:
outerFolder_qtemps_plots_RR_run6 =  f"{r6_plts_prefix}/rpm_qtemps"
#For analysis:
outerFolder_qtemps_plots_run6 = f"{r6_plts_prefix}/rpm_qtemps"

# For London Penetration Depth analysis, which is done on run 6 temperature sweep data. This is where we save the plots:
outerFolder_london_path = f"{r6_plts_prefix}/London_Penetration_Depth"

# Which RPM qubit temperature data do you want to look at? List here key words in the substudy name
# For the heater temperature sweep:
# filter_keywords_sciencerun = ['source_off_temperature_sweep']

# For source on and source off:
# filter_keywords_sciencerun = ['source_off', 'source_on']

# Specifically to look at source off data for run 6: science-run data as well as pre-science-run data
filter_keywords_sciencerun = ['source_off']
filter_keywords_presciencerun = ['q_temperatures_efRabi'] # no source was present, although not specified in the substudy name.

#-----------------------------------------------------------------------run 7------------------------------------------------------------
# Base path of where the data is stored up to the Study Name (round_robin_benchmark)
base_dir_run7 = "/exp/cosmiq/data/QUIET/QICK_data/run7/6transmon/round_robin_benchmark"
    # Options:
    #"/exp/cosmiq/data/QUIET/QICK_data/run7/6transmon/round_robin_benchmark" # CEPH
    # "/data/QICK_data/run7/6transmon/round_robin_benchmark" # daq01

# For all AB paper data:
target_dates_qtemps_RPM_run7 = ["2025-07-19", "2025-07-20"]

# To save plots
r7_plts_prefix = "/home/acolonce/Documents/analysis" #cosmiqserver01
# Options:
# "/home/acolonce/Documents/analysis" #cosmiqserver01
# "/data/QICK_data/run7/6transmon/analysis" #daq01
# "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/run7" #1hw

# For round robin plots:
outerFolder_qtemps_plots_RR_run7 =  f"{r7_plts_prefix}/rpm_qtemps/replotted_RR_data/"
#For analysis:
outerFolder_qtemps_plots_run7 = f"{r7_plts_prefix}/rpm_qtemps"

# Substudy name on the file path, doesn't have to be exact, it will look for these key terms in the name
# All AB paper data:
filter_keywords_run7 = ['AB_paper_data']

#-----------------------------------------------------------------------run 8------------------------------------------------------------
# Base path of where the data is stored up to the Study Name (round_robin_benchmark)
base_dir_run8 = "/exp/cosmiq/data/QUIET/QICK_data/run8/6transmon/round_robin" # CEPH
    # Options:
    #"/exp/cosmiq/data/QUIET/QICK_data/run8/6transmon/round_robin" # CEPH
    #"/data/QICK_data/run8/6transmon/round_robin" # up to study name. # daq01
    #r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8" # Arianna's pc

# all AB paper data: (specify up to the day only)
# target_dates_qtemps_RPM_run8 = [
#   "2025-10-19",
#   "2025-10-20",
#   "2025-10-23",
#   "2025-10-24",
#   "2025-10-27",
#   "2025-10-28",
#   "2025-10-29",
#   "2025-10-31",
#   "2025-11-01"
# ]

# Temp sweep data:
target_dates_qtemps_RPM_run8 = [
  "2025-11-18",
  "2025-11-19",
  "2025-11-20",
  "2025-11-21"
]

# To save plots
r8_plts_prefix = "/home/acolonce/Documents/analysis" #cosmiqserver01
# Options:
# "/home/acolonce/Documents/analysis" #cosmiqserver01
# "/data/QICK_data/run8/6transmon/analysis" #daq01
# "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/run8" #1hw

# For round robin plots:
outerFolder_qtemps_plots_RR_run8 =  f"{r8_plts_prefix}/rpm_qtemps/replotted_RR_data/"
#For analysis:
outerFolder_qtemps_plots_run8 = f"{r8_plts_prefix}/rpm_qtemps"

# Substudy name on the file path, doesn't have to be exact, it will look for these key terms in the name. THese are substudies.
# For all AB paper data:
# filter_keywords_run8 = ["AB_Paper_Data_24hrs", "ABpaperdata2ndbatch_21dB_DACatten_Q1to5", "ABpaperdata3rdbatch_21dB_DACatten_Q1to5_not1shots",
#                         "ABpaperdata3rdbatch_21dB_DACatten_Q1to5", "ABpaperdata3rdbatch_21dB_DACatten_Q1to5_t1shots_optional",
#                         "ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional", "ABpaperdata3rdbatch_21dB_DACatten_Q6_t1shots_optional",
#                         "ABpaperdata_21dB_DACatten_Q1to6_t1shots_optional_newopt", "19dB_DAC_testdata_allQs"]

# filter_keywords_run8 = ["AB_paper_datadump_for_analysis"] # for debugging and tesing

# # For run 8 temp sweep data:
filter_keywords_run8 = ["temperature_sweep_run8_25dBDAC_onechan_day1", "temperature_sweep_run8_25dBDAC_onechan_day2",
                        "temperature_sweep_run8_25dBDAC_onechan_day3", "temp_sweep_run8_25dBDAC_onechan_day4_175mK"]

#-----------------------------------------------------------------------run 9------------------------------------------------------------
#Base path of where the data is stored up to the Study Name (round_robin_benchmark)
base_dir_run9 = "/exp/cosmiq/data/QUIET/QICK_data/run9/6transmon/round_robin_benchmark"
    # Options:
    #"/exp/cosmiq/data/QUIET/QICK_data/run9/6transmon/round_robin_benchmark" # CEPH
    #"/data/QICK_data/run9/6transmon/round_robin_benchmark" # up to study name. # daq01

# All AB paper data:
target_dates_qtemps_RPM_run9 = [
    "2026-04-17",
    "2026-04-18",
    "2026-04-19",
    "2026-04-20",
    "2026-04-21",
    "2026-04-22",
    "2026-04-23",
    "2026-04-24",
    "2026-04-25",
    "2026-04-26",
    "2026-04-27"]

# To save plots
r9_plts_prefix = "/home/acolonce/Documents/analysis" #cosmiqserver01
# Options:
# "/home/acolonce/Documents/analysis" #cosmiqserver01
# "/data/QICK_data/run9/6transmon/analysis" #daq01
# "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/run9" #1hw

# For round robin plots:
outerFolder_qtemps_plots_RR_run9 =  f"{r9_plts_prefix}/rpm_qtemps/replotted_RR_data/"
#For analysis:
outerFolder_qtemps_plots_run9 = f"{r9_plts_prefix}/rpm_qtemps"

# Substudy name on the file path, doesn't have to be exact, it will look for these key terms in the name. These are substudies.
filter_keywords_run9 = [
    "AB_paper_data_batch1_25dB_DACatten_noQ5",
    "AB_paper_data_batch2_25dB_DACatten_onlyQ4",
    "AB_paper_data_batch3_25dB_DACatten_noQ5",
    "AB_paper_data_batch4_25dB_DACatten_noQ5",
    "AB_paper_data_batch5_25dBDAC_onlyQ1_onlySSF",
    "AB_paper_data_batch6_25dB_DACatten_onlyQ1",
    "AB_paper_does_no_rpm_fromRR_affect_coh_25dBDAC",
    "AB_paper_data_batch7_25dB_DACatten_noQ5",
    "AB_paper_data_batch8_25dB_DACatten_noQ5",
    "AB_paper_data_batch9_25dB_DACatten_noQ5Q4",
    "AB_paper_data_batch10_25dB_DACatten_onlyQ4",
    "AB_paper_data_batch11_25dB_DACatten_noQ5",
    "AB_paper_data_batch12_25dB_DACatten_noQ5",
    "AB_paper_data_batch13_25dB_DACatten_noQ5noQ6",
    "AB_paper_data_batch14_25dB_DACatten_noQ5noQ6",
    "AB_paper_data_batch15_25dB_DACatten_onlyQ6",
    "AB_paper_data_batch16_25dB_DACatten_noQ5",
    "AB_paper_data_batch17_25dB_DACatten_noQ5",
    "AB_paper_data_batch18_25dB_DACatten_noQ5noQ1",
    "AB_paper_data_batch19_25dB_DACatten_noQ5",
    "AB_paper_data_batch20_25dB_DACatten_noQ5",
    "AB_paper_data_batch21_25dB_DACatten_noQ5"]

#-----------------------------------------------------------------------run 9c ------------------------------------------------------------
#Base path of where the data is stored up to the Study Name (round_robin_benchmark)
base_dir_run9c = "/exp/cosmiq/data/QUIET/QICK_data/run9c/6transmon/round_robin_benchmark"

target_dates_qtemps_RPM_run9c = [
    "2026-06-05",
    "2026-06-09",
    "2026-06-14", "2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18", "2026-06-19"]

# To save plots
r9c_plts_prefix = "/home/acolonce/Documents/analysis" #cosmiqserver01

# For round robin plots:
outerFolder_qtemps_plots_RR_run9c =  f"{r9c_plts_prefix}/rpm_qtemps/replotted_RR_data/"
#For analysis:
outerFolder_qtemps_plots_run9c = f"{r9c_plts_prefix}/rpm_qtemps"

# Substudy name on the file path, doesn't have to be exact, it will look for these key terms in the name. These are substudies.
filter_keywords_run9c = [
    "Day2_base_not_fully_opt_yet_25dBDAC","Day4_base_not_fully_opt_yet_25dBDAC",
    "prejul15_outage_no_warm_filt_25dBDAC_noQ5", "postjul15_outage_no_warm_filt_25dBDAC_noQ5",
    "ABpaper_batch1_25dBDAC_ogfilters_noQ5", "ABpaper_batch2_25dBDAC_ogfilters_noQ5", "ABpaper_batch3_25dBDAC_ogfilters_noQ5"]

#-------------------------------------------------------------------------------- Assign func variables depending on run number ---------------------------------------------------------------------------
if run_num == 6: # We have science-run data as well as pre-science-run data available
    Science_Qubits = [0, 4]
    base_dir = base_dir_sciencerun
    filter_keywords = filter_keywords_sciencerun
    outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run6
    outerFolder_qtemps_plots = outerFolder_qtemps_plots_RR_run6
    target_dates_qtemps_RPM = target_dates_qtemps_RPM_sciencerun
    if pre_sciencerun6_data: # if True, it means you also want to analyze or incorporate pre-science-run data from run 6
        base_dir2 = base_dir_pre_sciencerun
        filter_keywords2 = filter_keywords_presciencerun
        target_dates_qtemps_RPM2 = target_dates_qtemps_RPM_presciencerun
elif run_num == 7:
    Science_Qubits = [0, 1, 2, 3, 4, 5]
    base_dir = base_dir_run7
    filter_keywords = filter_keywords_run7
    outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run7
    outerFolder_qtemps_plots = outerFolder_qtemps_plots_run7
    target_dates_qtemps_RPM = target_dates_qtemps_RPM_run7

elif run_num == 8:
    Science_Qubits = [0, 1, 2, 3, 4, 5]
    base_dir = base_dir_run8
    filter_keywords = filter_keywords_run8
    outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run8
    outerFolder_qtemps_plots = outerFolder_qtemps_plots_run8
    target_dates_qtemps_RPM = target_dates_qtemps_RPM_run8

elif run_num == 9:
    Science_Qubits = [0, 1, 2, 3, 4, 5]
    base_dir = base_dir_run9
    filter_keywords = filter_keywords_run9
    outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run9
    outerFolder_qtemps_plots = outerFolder_qtemps_plots_run9
    target_dates_qtemps_RPM = target_dates_qtemps_RPM_run9

elif run_num == 9.2: # this is run 9c
    Science_Qubits = [0, 1, 2, 3, 4, 5]
    base_dir = base_dir_run9c
    filter_keywords = filter_keywords_run9c
    outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run9c
    outerFolder_qtemps_plots = outerFolder_qtemps_plots_run9c
    target_dates_qtemps_RPM = target_dates_qtemps_RPM_run9c

elif run_num == 5: # No RPM data for this run, only ssf analysis can be done
    Science_Qubits = [0, 1, 2, 3, 4, 5]
    base_dir = ""
    filter_keywords = []
    outerFolder_qtemps_plots_RR = ""
    outerFolder_qtemps_plots = ""
    target_dates_qtemps_RPM = ""
    print('There is no RPM data for this run.')
else:
    print("No qubit temps section defined for this run. Qubit temps were only measured in runs 5-9 at QUIET.")

#-------------------------------------- For qubit temperature calculations via SSF methods (double gaussian over g-state data and double gaussian over g and e-state data ---------------------------------------------
# Note: ssf qtemps analysis scripts expect paths in this form: "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_03-03-40"
# If you want to loop through all the data corresponding to 1 day, you must list all the paths for that day. This method does not accept just a single date as a path.
# I have included examples for how the paths are structured for each run

# ------------------------------------------------------------------------------------------------run 4--------------------------------------------------------------------------------------------------------------
# NOT AB paper data: this is the only folder with "usable" data for this run and there is so little we can't doa  proper analysis.
paths_SSFmethods_run4 = ["/exp/cosmiq/data/QUIET/QICK_data/run4/6transmon/folders_with_SSF_data_entire_run4/ssf_data_and_readoutopt/2024-11-13_08-23-41"]

# To save plots
r4_plts_prefix = "/home/acolonce/Documents/analysis" #cosmiqserver01
# Options:
# "/home/acolonce/Documents/analysis" #cosmiqserver01
# "/data/QICK_data/run4/6transmon/analysis" #daq01
# "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/run4" #1hw

path_saveplots_fits_run4 = f"{r4_plts_prefix}/ssf_qtemps/gaussfits"
path_saveplots_ssf_qtemps_vsT_run4 = f"{r4_plts_prefix}/ssf_qtemps"
# ------------------------------------------------------------------------------------------------run 5----------------------------------------------------------------------------------------------------------
# All data
r5_path_prefix = "/exp/cosmiq/data/QUIET/QICK_data/run5"
                # Options:
                #"/data/QICK_data/run5" #daq01
                # "/exp/cosmiq/data/QUIET/QICK_data/run5" # CEPH

paths_SSFmethods_run5 = [
    f"{r5_path_prefix}/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-09",
    f"{r5_path_prefix}/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-10",
    f"{r5_path_prefix}/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-11",
    f"{r5_path_prefix}/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-12",
    f"{r5_path_prefix}/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-13",
    f"{r5_path_prefix}/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-14",
    f"{r5_path_prefix}/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-15",
    f"{r5_path_prefix}/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-16",
    f"{r5_path_prefix}/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-17",
    f"{r5_path_prefix}/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-18",
    f"{r5_path_prefix}/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-19",
    f"{r5_path_prefix}/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-20"
]

# To save plots
r5_plts_prefix = "/home/acolonce/Documents/analysis" #cosmiqserver01
# Options:
# "/home/acolonce/Documents/analysis" #cosmiqserver01
# "/data/QICK_data/run5/6transmon/analysis" #daq01
# "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/run5" #1hw

path_saveplots_fits_run5 = f"{r5_plts_prefix}/ssf_qtemps/gaussfits"
path_saveplots_ssf_qtemps_vsT_run5 = f"{r5_plts_prefix}/ssf_qtemps"
# ------------------------------------------------------------------------------------------------run 6------------------------------------------------------------------------------------------------------------
# Science-Run Data (only exists on CEPH)
paths_SSFmethods_SR = [
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy1/2025-04-15_21-24-46",

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

    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-04_20-56-05",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-04_23-28-05",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_03-03-40",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_06-40-15",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_10-18-53",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_13-57-22",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_17-34-21",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_21-18-14",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-06_02-18-57",

    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-06_11-30-17",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-06_14-50-55",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-06_18-14-29",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-06_21-35-26",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_01-00-14",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_04-23-45",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_07-46-44",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_11-09-17",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_14-30-29",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_17-50-59",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_21-13-50",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-08_00-36-15",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-08_03-56-41",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-08_07-19-10",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-08_11-53-46",

    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_post_temperature_sweep_substudy1/",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_detuning_24MHz_Q1_substudy1/",
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/"
    ]

# All pre-Science-Run Data (also exists on daq01, you can find it here: /data/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data)
paths_SSFmethods_preSR = [
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-21",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-22",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-23",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-24",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-26",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-28",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-03-01",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-03-02"
                        ]

# To save plots
r6_plts_prefix = "/home/acolonce/Documents/analysis" #cosmiqserver01
# Options:
# "/home/acolonce/Documents/analysis" #cosmiqserver01
# "/data/QICK_data/run6/6transmon/analysis" #daq01
# "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/run6" #1hw

path_saveplots_fits_run6 = f"{r6_plts_prefix}/ssf_qtemps/gaussfits"
path_saveplots_ssf_qtemps_vsT_run6 = f"{r6_plts_prefix}/ssf_qtemps"
# ----------------------------------------------------------------------------------------------run 7----------------------------------------------------------------------------------------------------------
r7_path_prefix = "/exp/cosmiq/data/QUIET/QICK_data/run7"
                # Options:
                #"/exp/cosmiq/data/QUIET/QICK_data/run7" # CEPH
                # "/data/QICK_data/run7" # daq01
paths_SSFmethods_run7 = [
  f"{r7_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data/2025-07-19_08-34-39",
  f"{r7_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data/2025-07-19_16-16-14",
  f"{r7_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data/2025-07-19_16-56-45",
  f"{r7_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data/2025-07-19_23-11-39",
  f"{r7_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data/2025-07-20_06-33-03"
]

# To save plots
r7_plts_prefix = "/home/acolonce/Documents/analysis" #cosmiqserver01
# Options:
# "/home/acolonce/Documents/analysis" #cosmiqserver01
# "/data/QICK_data/run7/6transmon/analysis" #daq01
# "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/run7" #1hw

path_saveplots_fits_run7 = f"{r7_plts_prefix}/ssf_qtemps/gaussfits"
path_saveplots_ssf_qtemps_vsT_run7 = f"{r7_plts_prefix}/ssf_qtemps"

# ----------------------------------------------------------------------------------------------run 8----------------------------------------------------------------------------------------------------------
r8_path_prefix = "/exp/cosmiq/data/QUIET/QICK_data/run8"
                #"/exp/cosmiq/data/QUIET/QICK_data/run8" # CEPH
                # "/data/QICK_data/run8" # daq01
                # r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8" # Arianna's local pc

# All AB paper data
paths_SSFmethods_run8 = [ # for daq01 case, need to make CEPH version
  f"{r8_path_prefix}/6transmon/round_robin/AB_Paper_Data_24hrs/2025-10-19_11-09-32",
  f"{r8_path_prefix}6transmon/round_robin/AB_Paper_Data_24hrs/2025-10-19_12-05-25",
  f"{r8_path_prefix}/6transmon/round_robin/AB_Paper_Data_24hrs/2025-10-19_19-43-00",
  f"{r8_path_prefix}/6transmon/round_robin/AB_Paper_Data_24hrs/2025-10-19_20-25-18",
  f"{r8_path_prefix}/6transmon/round_robin/AB_Paper_Data_24hrs/2025-10-20_12-10-19",

  f"{r8_path_prefix}/6transmon/round_robin/ABpaperdata2ndbatch_21dB_DACatten_Q1to5/2025-10-23_00-49-28",

  f"{r8_path_prefix}/6transmon/round_robin/ABpaperdata3rdbatch_21dB_DACatten_Q1to5/2025-10-23_14-47-22",

  f"{r8_path_prefix}/6transmon/round_robin/ABpaperdata3rdbatch_21dB_DACatten_Q1to5_not1shots/2025-10-24_01-41-30",

  f"{r8_path_prefix}/6transmon/round_robin/ABpaperdata3rdbatch_21dB_DACatten_Q1to5_t1shots_optional/2025-10-24_13-58-37",

  f"{r8_path_prefix}/6transmon/round_robin/ABpaperdata3rdbatch_21dB_DACatten_Q6_t1shots_optional/2025-10-27_14-15-40",
  f"{r8_path_prefix}/6transmon/round_robin/ABpaperdata3rdbatch_21dB_DACatten_Q6_t1shots_optional/2025-10-27_14-24-29",

  f"{r8_path_prefix}6transmon/round_robin/ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional/2025-10-27_22-04-57",

  f"{r8_path_prefix}/6transmon/round_robin/ABpaperdata_21dB_DACatten_Q1to6_t1shots_optional_newopt/2025-10-28_21-57-47",
  f"{r8_path_prefix}/6transmon/round_robin/ABpaperdata_21dB_DACatten_Q1to6_t1shots_optional_newopt/2025-10-29_18-38-25",
  f"{r8_path_prefix}/6transmon/round_robin/ABpaperdata_21dB_DACatten_Q1to6_t1shots_optional_newopt/2025-10-29_23-48-45",

  f"{r8_path_prefix}/6transmon/round_robin/18dB_DAC_testdata_allQs_exceptQ4/2025-10-31_01-54-57",

  f"{r8_path_prefix}/6transmon/round_robin/19dB_DAC_testdata_allQs/2025-10-31_20-40-11",
  f"{r8_path_prefix}/6transmon/round_robin/19dB_DAC_testdata_allQs/2025-11-01_12-54-55"
]

# To save plots
r8_plts_prefix = "/home/acolonce/Documents/analysis" #cosmiqserver01
# Options:
# "/home/acolonce/Documents/analysis" #cosmiqserver01
# "/data/QICK_data/run8/6transmon/analysis" #daq01
# "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/run8" #1hw
# r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8" # Arianna's local pc

path_saveplots_fits_run8 = f"{r8_plts_prefix}/ssf_qtemps/gaussfits"
path_saveplots_ssf_qtemps_vsT_run8 = f"{r8_plts_prefix}/ssf_qtemps"

# ----------------------------------------------------------------------------------------------run 9----------------------------------------------------------------------------------------------------------
r9_path_prefix = "/exp/cosmiq/data/QUIET/QICK_data/run9"
                # Options:
                #"/exp/cosmiq/data/QUIET/QICK_data/run9" # CEPH
                # "/data/QICK_data/run9" # daq01
                # r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run9" # Arianna's local pc
# AB paper data
paths_SSFmethods_run9 = [
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch1_25dB_DACatten_noQ5/2026-04-17_00-34-47",  # ignore Q4 in this data, punched out too much!!

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch2_25dB_DACatten_onlyQ4/2026-04-17_16-47-30",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-17_20-54-40",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-17_21-53-52",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-17_22-51-33",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-17_23-47-15",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-18_00-42-05",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-18_11-24-10",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch3_25dB_DACatten_noQ5/2026-04-18_13-25-28",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch4_25dB_DACatten_noQ5/2026-04-18_23-06-45",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch4_25dB_DACatten_noQ5/2026-04-19_00-16-08",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch4_25dB_DACatten_noQ5/2026-04-19_01-18-51",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch4_25dB_DACatten_noQ5/2026-04-19_17-54-48",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_does_no_rpm_fromRR_affect_coh_25dBDAC/2026-04-18_20-58-30",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch5_25dBDAC_onlyQ1_onlySSF/2026-04-20_18-43-20",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch6_25dB_DACatten_onlyQ1/2026-04-20_18-50-54",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch7_25dB_DACatten_noQ5/2026-04-21_02-55-23",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch8_25dB_DACatten_noQ5/2026-04-21_11-34-57",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch9_25dB_DACatten_noQ5Q4/2026-04-22_03-09-46",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch10_25dB_DACatten_onlyQ4/2026-04-22_11-48-15",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_16-37-14",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_16-54-54",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_17-06-49",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_17-22-15",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_17-35-15",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_18-11-40",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch11_25dB_DACatten_noQ5/2026-04-22_19-55-31",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch12_25dB_DACatten_noQ5/2026-04-22_20-53-08",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch13_25dB_DACatten_noQ5noQ6/2026-04-23_06-22-14",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch14_25dB_DACatten_noQ5noQ6/2026-04-23_12-20-26",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch15_25dB_DACatten_onlyQ6/2026-04-23_16-59-52",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch15_25dB_DACatten_onlyQ6/2026-04-23_17-25-33",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch15_25dB_DACatten_onlyQ6/2026-04-23_18-49-54",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch15_25dB_DACatten_onlyQ6/2026-04-23_19-06-16",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch15_25dB_DACatten_onlyQ6/2026-04-23_19-24-51",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch16_25dB_DACatten_noQ5/2026-04-23_20-14-15",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch16_25dB_DACatten_noQ5/2026-04-23_20-17-04",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch16_25dB_DACatten_noQ5/2026-04-23_21-22-18",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch16_25dB_DACatten_noQ5/2026-04-23_22-29-25",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch16_25dB_DACatten_noQ5/2026-04-23_23-40-11",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch17_25dB_DACatten_noQ5/2026-04-24_00-51-38",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch18_25dB_DACatten_noQ5noQ1/2026-04-24_14-37-30",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-24_18-51-32",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-24_21-13-58",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-24_23-27-19",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-21-18",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-22-27",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-30-29",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-33-26",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-34-54",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-36-30",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-48-04",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-25_23-57-26",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_00-04-04",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_00-35-21",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_12-49-11",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_12-55-00",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_12-57-51",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_12-59-36",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_13-02-45",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_13-04-33",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_13-25-03",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_13-27-10",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_13-39-30",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch19_25dB_DACatten_noQ5/2026-04-26_13-41-18",
    
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch20_25dB_DACatten_noQ5/2026-04-26_13-45-45",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch20_25dB_DACatten_noQ5/2026-04-26_20-58-39",

    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch21_25dB_DACatten_noQ5/2026-04-27_08-07-32",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch21_25dB_DACatten_noQ5/2026-04-27_11-41-29",
    f"{r9_path_prefix}/6transmon/round_robin_benchmark/AB_paper_data_batch21_25dB_DACatten_noQ5/2026-04-27_13-07-15"
]

# To save plots
r9_plts_prefix = "/home/acolonce/Documents/analysis" #cosmiqserver01
# Options:
# "/home/acolonce/Documents/analysis" #cosmiqserver01
# "/data/QICK_data/run9/6transmon/analysis" #daq01
# "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/run9" #1hw

path_saveplots_fits_run9 = f"{r9_plts_prefix}/ssf_qtemps/gaussfits"
path_saveplots_ssf_qtemps_vsT_run9 = f"{r9_plts_prefix}/ssf_qtemps"

# ---------------------------------------------------------------------------------------------- run 9c ----------------------------------------------------------------------------------------------------------
r9c_path_prefix = "/exp/cosmiq/data/QUIET/QICK_data/run9c"

paths_SSFmethods_run9c = [
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/Day2_base_not_fully_opt_yet_25dBDAC/2026-06-05_23-17-54",

    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/Day4_base_not_fully_opt_yet_25dBDAC/2026-06-09_09-43-21",

    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/prejul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-14_21-13-10",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/prejul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-14_21-17-55",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/prejul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-14_21-18-14",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/prejul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-14_21-19-26",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/prejul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-14_21-23-30",

    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_10-24-54",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_10-25-34",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_10-26-35",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_10-27-31",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_10-34-02",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_15-46-13",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_15-46-45",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_15-51-05",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_15-52-11",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_15-54-23",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_15-57-31",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-06-05",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-16-41",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-17-12",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-19-09",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-20-54",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-23-24",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-23-35",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-30-06",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-35-48",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_17-39-03",

    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/ABpaper_batch1_25dBDAC_ogfilters_noQ5/2026-06-15_19-38-45",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/ABpaper_batch1_25dBDAC_ogfilters_noQ5/2026-06-15_20-57-52",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/ABpaper_batch1_25dBDAC_ogfilters_noQ5/2026-06-16_00-22-56",

    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/ABpaper_batch2_25dBDAC_ogfilters_noQ5/2026-06-16_12-02-29",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/ABpaper_batch2_25dBDAC_ogfilters_noQ5/2026-06-16_15-16-01",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/ABpaper_batch2_25dBDAC_ogfilters_noQ5/2026-06-17_09-44-00",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/ABpaper_batch2_25dBDAC_ogfilters_noQ5/2026-06-17_11-46-15",

    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/ABpaper_batch3_25dBDAC_ogfilters_noQ5/2026-06-18_11-38-44",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/ABpaper_batch3_25dBDAC_ogfilters_noQ5/2026-06-18_17-18-45",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/ABpaper_batch3_25dBDAC_ogfilters_noQ5/2026-06-19_00-28-27",
    f"{r9c_path_prefix}/6transmon/round_robin_benchmark/ABpaper_batch3_25dBDAC_ogfilters_noQ5/2026-06-19_10-53-54"
    ]

# To save plots
r9c_plts_prefix = "/home/acolonce/Documents/analysis" #cosmiqserver01

path_saveplots_fits_run9c = f"{r9c_plts_prefix}/ssf_qtemps/gaussfitsfits"
path_saveplots_ssf_qtemps_vsT_run9c = f"{r9c_plts_prefix}/ssf_qtemps"

#------------------------------------------------------------------------------ Assign func variables depending on run number ---------------------------------------
if run_num == 6:  # We have science-run data as well as pre-science-run data available. Note: we already defined Science_Qubits for the science run above.
    paths_SSFmethods = paths_SSFmethods_SR.copy()
    if pre_sciencerun6_data:  # If True, include pre-science-run data
        Science_Qubits = [0, 1, 2, 3, 4, 5] # include all qubits, since pre-SR data was taken for all Qs
        paths_SSFmethods += paths_SSFmethods_preSR
    path_saveplots_fits = path_saveplots_fits_run6
    path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run6
elif run_num == 7: # already defined Science_Qubits above
    paths_SSFmethods = paths_SSFmethods_run7
    path_saveplots_fits = path_saveplots_fits_run7
    path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run7
elif run_num == 8: # already defined Science_Qubits above
    paths_SSFmethods = paths_SSFmethods_run8
    path_saveplots_fits = path_saveplots_fits_run8
    path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run8
elif run_num == 9: # already defined Science_Qubits above
    paths_SSFmethods = paths_SSFmethods_run9
    path_saveplots_fits = path_saveplots_fits_run9
    path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run9
elif run_num == 9.2: # this is run 9c # already defined Science_Qubits above
    paths_SSFmethods = paths_SSFmethods_run9c
    path_saveplots_fits = path_saveplots_fits_run9c
    path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run9c
elif run_num == 4:
    Science_Qubits = [2,3] # we only have ssf data for Q3 and Q4 for this run.
    paths_SSFmethods = paths_SSFmethods_run4
    path_saveplots_fits = path_saveplots_fits_run4
    path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run4
elif run_num == 5:
    Science_Qubits = [0, 1, 2, 3, 4, 5] # we have ssf data for all qubits in run 5
    paths_SSFmethods = paths_SSFmethods_run5
    path_saveplots_fits = path_saveplots_fits_run5
    path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run5
else:
    raise ValueError("You must choose run_num = 4, 5, 6, 7, 8 or 9. Otherwise, define a section for your run of interest.")

#-------------------------------------------------------- For coherence data -------------------------------------------
if coh_qtemp_ana_flags["run_coherence_section"]:
    if run_num == 9.2: # run 9c
        print('(this is actually run 9c, we just label it run 9.2)')
        process_shots_t1ge = False # the option exists for this run, but for analysis consistency w initial runs we keep it off unless necessary.
        per_pt_errs_t1 = False
        run_name = 'run9c/6transmon/round_robin_benchmark'
        data_path = f"/exp/cosmiq/data/QUIET/QICK_data/{run_name}" #CEPH
        plots_path = "/home/acolonce/Documents/analysis/coherence" #cosmiqserver01

        top_folder_dates = [
            "Day2_base_not_fully_opt_yet_25dBDAC/2026-06-05_23-17-54",

            "Day4_base_not_fully_opt_yet_25dBDAC/2026-06-09_09-43-21",

            "prejul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-14_21-13-10",
            "prejul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-14_21-17-55",
            "prejul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-14_21-18-14",
            "prejul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-14_21-19-26",
            "prejul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-14_21-23-30",

            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_10-24-54",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_10-25-34",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_10-26-35",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_10-27-31",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_10-34-02",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_15-46-13",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_15-46-45",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_15-51-05",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_15-52-11",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_15-54-23",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_15-57-31",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-06-05",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-16-41",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-17-12",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-19-09",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-20-54",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-23-24",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-23-35",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-30-06",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_16-35-48",
            "postjul15_outage_no_warm_filt_25dBDAC_noQ5/2026-06-15_17-39-03",

            "ABpaper_batch1_25dBDAC_ogfilters_noQ5/2026-06-15_19-38-45",
            "ABpaper_batch1_25dBDAC_ogfilters_noQ5/2026-06-15_20-57-52",
            "ABpaper_batch1_25dBDAC_ogfilters_noQ5/2026-06-16_00-22-56",

            "ABpaper_batch2_25dBDAC_ogfilters_noQ5/2026-06-16_12-02-29",
            "ABpaper_batch2_25dBDAC_ogfilters_noQ5/2026-06-16_15-16-01",
            "ABpaper_batch2_25dBDAC_ogfilters_noQ5/2026-06-17_09-44-00",
            "ABpaper_batch2_25dBDAC_ogfilters_noQ5/2026-06-17_11-46-15",

            "ABpaper_batch3_25dBDAC_ogfilters_noQ5/2026-06-18_11-38-44",
            "ABpaper_batch3_25dBDAC_ogfilters_noQ5/2026-06-18_17-18-45",
            "ABpaper_batch3_25dBDAC_ogfilters_noQ5/2026-06-19_00-28-27",
            "ABpaper_batch3_25dBDAC_ogfilters_noQ5/2026-06-19_10-53-54"
            ]
    elif run_num == 9:
        process_shots_t1ge = False  # the option exists for this run, but for analysis consistency w initial runs we keep it off unless necessary.
        per_pt_errs_t1 = False
        run_name = "run9/6transmon/round_robin_benchmark"
        data_path = f"/exp/cosmiq/data/QUIET/QICK_data/{run_name}"  # CEPH
        # f'/data/QICK_data/{run_name}' #daq01
        plots_path = "/home/acolonce/Documents/analysis/coherence"  # cosmiqserver01
        # "/data/QICK_data/run9/6transmon/analysis" #daq01
    
        top_folder_dates = [
            "AB_paper_data_batch1_25dB_DACatten_noQ5/2026-04-17_00-34-47",  # ignore Q4 in this data, punched out too much!!
    
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
    elif run_num == 8:
        process_shots_t1ge = True
        per_pt_errs_t1 = True
        run_name = "run8/6transmon/round_robin"
        # 'run8/6transmon/round_robin/temperature_sweep_qubit_data'
        # 'run8/6transmon/round_robin/AB_paper_datadump_for_analysis'
        data_path = f"/exp/cosmiq/data/QUIET/QICK_data/{run_name}"  # CEPH
        # f'/data/QICK_data/{run_name}' # daq01
        plots_path = "/home/acolonce/Documents/analysis/coherence"  # cosmiqserver01
        # "/data/QICK_data/run8/6transmon/analysis" #daq01

        # all of run 8 data
        # top_folder_dates = [
        #     "AB_Paper_Data_24hrs/2025-10-19_11-09-32",  # only T1 shots, no T1 QICK-averaged IQ data
        #     "AB_Paper_Data_24hrs/2025-10-19_12-05-25",  # only T1 shots, no T1 QICK-averaged IQ data
        #     "AB_Paper_Data_24hrs/2025-10-19_19-43-00",  # only T1 shots, no T1 QICK-averaged IQ data
        #     "AB_Paper_Data_24hrs/2025-10-19_20-25-18",  # only T1 shots, no T1 QICK-averaged IQ data
        #     "AB_Paper_Data_24hrs/2025-10-20_12-10-19",  # only T1 shots, no T1 QICK-averaged IQ data
        #
        #     "ABpaperdata2ndbatch_21dB_DACatten_Q1to5/2025-10-23_00-49-28",
        #     # only T1 shots, no T1 QICK-averaged IQ data
        #
        #     "ABpaperdata3rdbatch_21dB_DACatten_Q1to5/2025-10-23_14-47-22",
        #     # only T1 shots, no T1 QICK-averaged IQ data
        #     # "ABpaperdata3rdbatch_21dB_DACatten_Q1to5_not1shots/2025-10-24_01-41-30",  # no T1 shots saved, only QICK averaged IQ data. Leave commented out. Need to debug script to incorporate this
        #     "ABpaperdata3rdbatch_21dB_DACatten_Q1to5_t1shots_optional/2025-10-24_13-58-37",
        #     # From this point forward, both T1 shots and averaged IQ arrays were saved
        #     "ABpaperdata3rdbatch_21dB_DACatten_Q6_t1shots_optional/2025-10-27_14-15-40",
        #     "ABpaperdata3rdbatch_21dB_DACatten_Q6_t1shots_optional/2025-10-27_14-24-29",
        #     "ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional/2025-10-27_22-04-57",
        #
        #     "ABpaperdata_21dB_DACatten_Q1to6_t1shots_optional_newopt/2025-10-28_21-57-47",
        #     "ABpaperdata_21dB_DACatten_Q1to6_t1shots_optional_newopt/2025-10-29_18-38-25",
        #     "ABpaperdata_21dB_DACatten_Q1to6_t1shots_optional_newopt/2025-10-29_23-48-45",
        #
        #     "18dB_DAC_testdata_allQs_exceptQ4/2025-10-31_01-54-57",
        #
        #     "19dB_DAC_testdata_allQs/2025-10-31_20-40-11",
        #     "19dB_DAC_testdata_allQs/2025-11-01_12-54-55",
        # ]

        # All run 8 qubit temperature sweep data except the 200mK dataset bc no qubits visible
        top_folder_dates = [
            "temperature_sweep_run8_25dBDAC_onechan_day1/2025-11-18_08-39-37",
            "temperature_sweep_run8_25dBDAC_onechan_day1/2025-11-18_09-02-01",
            "temperature_sweep_run8_25dBDAC_onechan_day1/2025-11-18_12-40-59",
            "temperature_sweep_run8_25dBDAC_onechan_day1/2025-11-18_14-26-01",
            "temperature_sweep_run8_25dBDAC_onechan_day1/2025-11-18_14-48-11",

            "temperature_sweep_run8_25dBDAC_onechan_day2/2025-11-19_08-04-25",
            "temperature_sweep_run8_25dBDAC_onechan_day2/2025-11-19_11-00-00",
            "temperature_sweep_run8_25dBDAC_onechan_day2/2025-11-19_11-27-04",

            "temperature_sweep_run8_25dBDAC_onechan_day3/2025-11-20_07-31-49",

            "temp_sweep_run8_25dBDAC_onechan_day4_175mK/2025-11-21_08-01-57",
            "temp_sweep_run8_25dBDAC_onechan_day4_175mK/2025-11-21_08-33-09",
            "temp_sweep_run8_25dBDAC_onechan_day4_175mK/2025-11-21_08-45-17",
            "temp_sweep_run8_25dBDAC_onechan_day4_175mK/2025-11-21_08-54-05"]

    elif run_num == 7:
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

    elif run_num == 6:
        process_shots_t1ge = False
        per_pt_errs_t1 = False
        run_name = 'run6/6transmon'
        data_path = f"/exp/cosmiq/data/QUIET/QICK_data/{run_name}" #CEPH
            #f'/data/QICK_data/{run_name}' #daq01
        plots_path = "/home/acolonce/Documents/analysis/coherence" #cosmiqserver01
            #"/data/QICK_data/run6/6transmon/analysis" #daq01

        # all pre-science run data (AB paper data):
        # Can be found both locally in daq01 or on CEPH
        top_folder_dates = [
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-21",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-22",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-23",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-24",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-26",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-28",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-03-01",
        "ge_round_robin_presciencerun_data/ge_coherence_data/2025-03-02" #]

        # Science run data: can ONLY be found on CEPH!!
        # If you want to process all "science run" data
        "TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/2025-05-15_14-47-38",
        "TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/2025-05-15_18-08-15",
        "TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/2025-05-15_22-02-12",
        "TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/2025-05-16_01-28-20",
        "TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/2025-05-16_04-49-59",
        "TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/2025-05-16_08-13-47",

        "TLS_Comprehensive_Study/source_off_detuning_24MHz_Q1_substudy1/2025-05-15_11-19-50",
        "TLS_Comprehensive_Study/source_off_detuning_24MHz_Q1_substudy1/2025-05-15_18-35-56",

        "TLS_Comprehensive_Study/source_off_post_temperature_sweep_substudy1/2025-05-14_19-25-55",
        "TLS_Comprehensive_Study/source_off_post_temperature_sweep_substudy1/2025-05-14_22-50-51",
        "TLS_Comprehensive_Study/source_off_post_temperature_sweep_substudy1/2025-05-15_02-29-34",
        "TLS_Comprehensive_Study/source_off_post_temperature_sweep_substudy1/2025-05-15_05-50-12",
        "TLS_Comprehensive_Study/source_off_post_temperature_sweep_substudy1/2025-05-15_09-13-30",

        "TLS_Comprehensive_Study/source_off_substudy1/2025-04-15_21-24-46",

        "TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_11-47-09",
        "TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_12-51-09",
        "TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_17-50-00",
        "TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_22-47-49",
        "TLS_Comprehensive_Study/source_off_substudy2/2025-04-17_03-42-36",
        "TLS_Comprehensive_Study/source_off_substudy2/2025-04-17_08-42-24",

        "TLS_Comprehensive_Study/source_off_substudy3/2025-04-17_12-28-37",
        "TLS_Comprehensive_Study/source_off_substudy3/2025-04-17_17-22-46",
        "TLS_Comprehensive_Study/source_off_substudy3/2025-04-17_22-16-39",
        "TLS_Comprehensive_Study/source_off_substudy3/2025-04-18_01-45-53",
        "TLS_Comprehensive_Study/source_off_substudy3/2025-04-18_06-40-55",

        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-18_11-59-33",
        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-18_16-56-58",
        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-18_21-51-13",
        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_02-45-41",
        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_07-39-57",
        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_12-34-26",
        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_17-48-44",
        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_22-43-02",
        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_03-37-50",
        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_08-32-36",
        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_13-26-47",
        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_18-25-13",
        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_23-25-04",
        "TLS_Comprehensive_Study/source_off_substudy4/2025-04-21_04-23-31",

        "TLS_Comprehensive_Study/source_off_substudy5/2025-05-04_20-56-05",
        "TLS_Comprehensive_Study/source_off_substudy5/2025-05-04_23-28-05",
        "TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_03-03-40",
        "TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_06-40-15",
        "TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_10-18-53",
        "TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_13-57-22",
        "TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_17-34-21",
        "TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_21-18-14",
        "TLS_Comprehensive_Study/source_off_substudy5/2025-05-06_02-18-57",

        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-06_11-30-17",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-06_14-50-55",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-06_18-14-29",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-06_21-35-26",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_01-00-14",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_04-23-45",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_07-46-44",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_11-09-17",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_14-30-29",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_17-50-59",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-07_21-13-50",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-08_00-36-15",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-08_03-56-41",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-08_07-19-10",
        "TLS_Comprehensive_Study/source_off_substudy6/2025-05-08_11-53-46"]

    elif run_num == 5:
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
    elif run_num == 4:
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

############################################################################### Qubit temperature calculations via rabi population measurements #####################################################
if qtemp_method_flags["Qtemps_viaRPM"]:
    RPM_calcs = RPMTempCalcAndPlots(figure_quality, tot_num_of_qubits)
    combined_qtemp_data = RPM_calcs.run_RPMqtemps(base_dir, target_dates_qtemps_RPM, filter_keywords, fit_saved, signal, run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                            outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, get_london_data, figure_quality, save_figsRR, exclude_temp_sweeps, filter_out_bad_RPM_fits = filter_out_bad_RPM_fits,
                                                  passing_pre_sciencerun_data = False, combine_IQ_signal = rpm_combine_IQ_signal)

    if run_num == 6:
        if pre_sciencerun6_data:
            combined_qtemp_data2 = RPM_calcs.run_RPMqtemps(base_dir2, target_dates_qtemps_RPM2, filter_keywords2, fit_saved, signal, run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                                                          outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, get_london_data, figure_quality, save_figsRR, exclude_temp_sweeps, filter_out_bad_RPM_fits = filter_out_bad_RPM_fits,
                                                           passing_pre_sciencerun_data = True, combine_IQ_signal = rpm_combine_IQ_signal)

            combined_qtemp_data += combined_qtemp_data2

    del RPM_calcs # to free up memory
    #----------------------------------------------------------------------- RPM Analysis -------------------------------------------------------------------------
    # These are not used in the definitions that follow, are just needed to re-initialize the class
    outerFolder = ""
    outerFolder_qtemps_data = ""
    date_string = ""
    Pe_dist_err_dict = None
    RPM_plotter = PlotRR_noQick(date_string, figure_quality, save_figs, fit_saved, signal, run_name, tot_num_of_qubits, outerFolder, outerFolder_qtemps_plots, outerFolder_qtemps_data, run_num, filter_out_bad_RPM_fits)

    # ---------------------------------------------- RPM Pe histograms -------------------------------------------
    if analysis_flags["Pe_hists_viaRPM"]:
        RPM_plotter.plot_qubit_Pe_histograms_RPMs(combined_qtemp_data, num_qubits=tot_num_of_qubits, rel_err_cutoff=None,
                                                                    only_return_mu_and_sigma=True, make_plot=True, save_plot=True)

    if analysis_flags["Qtemps_vs_time_viaRPM"]:
        #------------------------------------------------------------------- Qubit temperatures vs time via RPMs ----------------------------------------------------
        RPM_plotter.plot_qubit_temperatures_vs_time_RPMs(combined_qtemp_data, num_qubits=tot_num_of_qubits, yaxis_min = 40, yaxis_max = 140, rel_err_cutoff = 0.8, restrict_time_xaxis = False,
                                                         plot_extra_event_lines = False, rad_events_plot_lines = False, plot_error_bars = True, fit_to_line=False, average_per_heater_step=False,
                                                         fit_to_exp = False)

    if analysis_flags["Qtemps_hists_viaRPM"]:
        #----------------------------------------------------------------- Histograms of Qubit temperatures (via RPMs) -----------------------------------------------
        RPM_plotter.plot_qubit_temperature_histograms_RPMs(combined_qtemp_data, num_qubits=6, rel_err_cutoff = None)

    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    if analysis_flags["Pe_vs_time_viaRPM"]:
        #------------------------------------------------------------ Excited state populations (P_e) vs time (via RPMs) ----------------------------------------
        RPM_plotter.plot_qubit_pe_vs_time_RPMs(combined_qtemp_data, ylim = None)

    if analysis_flags["qtemps_Pe_vs_time_viaRPM"]:
        #---------------------------------------------------------- Qubit temp and P_e vs time in the same plot (via RPMs) ------------------------------------
        RPM_plotter.plot_qubit_temp_and_pe_vs_time_RPMs(combined_qtemp_data)

    if analysis_flags["qtemps_Pe_gefreq_vs_time_viaRPM"]:
        #---------------------------------------------------- Qubit temp, P_e, and g-e qubit freq vs time in the same plot (via RPMs) --------------------------
        RPM_plotter.plot_qubit_temp_pe_freq_vs_time_RPMs(combined_qtemp_data)


############################################################### Qubit temperature calculations via SSF measurements #################################################
#-------------------------------------------------------------- Process and pair up the ssf and g-e qubit spec data -------------------------------------------------
if qtemp_method_flags["Qtemps_viaSSF_ge_thresh"] or qtemp_method_flags["Qtemps_viaSSF_gmeans_thresh"] or qtemp_method_flags["Qtemps_viaSSF_with_fallback"]:

    method_one = qtemp_method_flags["Qtemps_viaSSF_gmeans_thresh"]
    method_two = qtemp_method_flags["Qtemps_viaSSF_ge_thresh"]
    method_three = qtemp_method_flags["Qtemps_viaSSF_with_fallback"]

    if (method_one + method_two + method_three) != 1:  # True==1, False==0
        raise ValueError("You must set *only one* of these to True: Qtemps_viaSSF_gmeans_thresh, Qtemps_viaSSF_ge_thresh or Qtemps_viaSSF_with_fallback. Please pick one and try again.")

    SSF_calcs_obj = SSFTempCalcAndPlots(figure_quality, tot_num_of_qubits, run_num, save_figs)
    pairs_info = SSF_calcs_obj.process_ssf_and_qfreq_data_qtemps(Science_Qubits, paths_SSFmethods)

    # ------------------------------------------------------------------- Calculate Qubit Temperatures ----------------------------------------------------------------------------
    if qtemp_method_flags["Qtemps_viaSSF_gmeans_thresh"]: # Default method of the function - fits only PREPARED GROUND STATE SSF data to a double gaussian ; threshold = midpoint of the two gaussian means
        if use_iminuit_gdoublegauss_ssf:
            # Made a special iminuit-based double gaussian fitting function. For now it is only set up to fit g-state data.
            # optionally saves fitted data and shows which scans were filtered out and which were kept
            all_qubit_temps, all_qubit_times, all_qubit_temps_errs, fit_results = SSF_calcs_obj.run_ssf_qtemps_iminuit(pairs_info, run_num=run_num, limit_temp_k=1.0, do_plots=save_figs_SSF, save_figs_path = path_saveplots_fits, dontuse_midpt_thresh = True,
                                                                                                                        low_leakage_mode = low_thermal_pops, ssf_hist_ylim = ssf_hist_ylim, apply_quality_cuts = filter_out_bad_SSF_qtemp_fits, calc_SNR = calc_SNR_ssfqtemps,
                                                                                                                       calc_e_state_decay = calc_SSF_e_decay)
        else:
            all_qubit_temps, all_qubit_times, all_qubit_temps_errs, fit_results  = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=1.0, use_gessf_thresh_only = False, fallback_to_threshold = False)
    elif qtemp_method_flags["Qtemps_viaSSF_ge_thresh"]: # Fits both GROUND STATE and PREPARED EXCITED STATE SSF data to a double gaussian ; threshold = midpoint of the two gaussian means
        all_qubit_temps, all_qubit_times, all_qubit_temps_errs, fit_results  = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=1.0, use_gessf_thresh_only = True, fallback_to_threshold = False)
    elif qtemp_method_flags["Qtemps_viaSSF_with_fallback"]: # Uses Default method and if the fit fails it falls back to the method that fits both GROUND STATE and PREPARED EXCITED STATE SSF data to a double gaussian
        all_qubit_temps, all_qubit_times, all_qubit_temps_errs, fit_results  = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=1.0, use_gessf_thresh_only = False, fallback_to_threshold = True)

    # --------------------------------------------------------- SSF qubit temps analysis --------------------------------------------------------
    # --------------------------------------------------------- SNR vs time for each qubit --------------------------------------------------------------
    if analysis_flags["ssf_SNR_vs_time"]:
        SSF_calcs_obj.plot_ssf_SNR_vs_time(fit_results,path_saveplots_ssf_qtemps_vsT,n_qubits=6)
    # --------------------------------------------------------- SSF vs time for each qubit --------------------------------------------------------------
    if analysis_flags["SSF_vs_time"]:
        SSF_calcs_obj.plot_ssf_vs_time(fit_results, path_saveplots_ssf_qtemps_vsT, n_qubits =6)
    # --------------------------------------------------------- SSF vs Pe for each qubit -------------------------------------------------------------------------
    if analysis_flags["SSF_fid_vs_Pe_viaSSF"]:
        SSF_calcs_obj.plot_SSF_fid_vs_Pe_viaSSF(fit_results, path_saveplots_ssf_qtemps_vsT, n_qubits =6, plot_together = False, sharex=True, sharey=True)
    # ---------------------------------------- SSF thermal population (Pe) histograms --------------------------------------------------------------
    if analysis_flags["Pe_hists_viaSSF"]:
        SSF_calcs_obj.plot_all_Qs_Pe_hists_ssf(fit_results, out_dir=path_saveplots_ssf_qtemps_vsT, bins=45, rel_err_cutoff=None, only_return_mu_and_sigma=True)
    # --------------------------------------------------------------------------- SSF Temperature Histograms --------------------------------------------------------------------------------------
    if analysis_flags["Qtemps_hists_viaSSF"]:
        SSF_calcs_obj.plot_all_Qs_qtemps_hists_ssf(all_qubit_temps, all_qubit_temps_errs, path_saveplots_ssf_qtemps_vsT, bins=45)
    #------------------------------------------------------------------ Temperatures vs Time Scatter Plot --------------------------------------------------------------------------
    if analysis_flags["Qtemps_vs_time_viaSSF"]:
        SSF_calcs_obj.plot_qubit_temperatures_vs_time_ssf(all_qubit_temps, all_qubit_times, all_qubit_temps_errs, path_saveplots_ssf_qtemps_vsT, plot_error_bars = True,
                                                          yaxis_min = 40, yaxis_max = 200)
    #------------------------------------------------------------ Check General SSF Double Gaussian Fits and g-e threshold ---------------------------------------------------------
    if analysis_flags["ge_thresh_check_ssf"]:
        thresh_results = SSF_calcs_obj.plot_ssf_ge_thresh(pairs_info=pairs_info, plotting_path=path_saveplots_fits)

    #---------------------------------------------------- Check population threshold for Qubit Temperature Calcs via BOTH SSF methods ----------------------------------------------
    if analysis_flags["Threshold_Check_Qtemps_viaSSF"]:
        for q_key, recs in fit_results.items():
            # path_saveplots/Q1, Q2, etc.
            qubit_folder = os.path.join(path_saveplots_fits, f"Q{q_key+1}")
            os.makedirs(qubit_folder, exist_ok=True)
            # Make a date‐stamped subfolder
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            made_on_folder = os.path.join(qubit_folder, f"made_on_{date_str}")
            os.makedirs(made_on_folder, exist_ok=True)

            for rec in recs:
                uses_thr = rec.get("used_gessf_thresh_only", False) #Looks up the key "uses_ssf_data_threshold" in the result dictionary. If it’s missing (or False), the code did not use the SSF threshold.
                used_fb = rec.get("used_fallback_method", False) # similar check for fall back option

                if not uses_thr and not used_fb: #if 'uses_gessf_data_threshold' was False or used_fallback_method was False / not used
                    # plots the double-gaussian fits on the ground state data and shows where the population threshold was set (midpoint of the two means)
                    SSF_calcs_obj.plot_gaussians_qtemps(q_key, made_on_folder, rec["ig_new"], rec["ground_data"], rec["excited_data"], rec["ground_gaussian"],
                                                        rec["excited_gaussian"], rec["pop_threshold"], rec["temperature_mK"], rec["dataset"], rec["weights"],
                                                        rec["sigmas"], rec["means"])
                else:
                    # plots the g-e threshold and only the ground state data to show how the g-e threshold was used to determine Pg and Pe
                    SSF_calcs_obj.plot_threshold_split(q_key, rec, made_on_folder)

####################################### SSF qubit temps analysis WITHOUT pre-built sklearn.mixture.GaussianMixture double gaussian fitting functions ##########################################
if alt_ssf_analysis_flags["jupyter_method_Arianna"]:
    SSF_calcs_obj = SSFTempCalcAndPlots(figure_quality, tot_num_of_qubits, run_num, save_figs)
    pairs_info = SSF_calcs_obj.process_ssf_and_qfreq_data_qtemps(Science_Qubits, paths_SSFmethods)

    qubit_folder = os.path.join(path_saveplots_fits, "Arianna_nonprebuilt")
    os.makedirs(qubit_folder, exist_ok=True)
    # Make a date‐stamped subfolder
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    made_on_folder = os.path.join(qubit_folder, f"made_on_{date_str}")
    os.makedirs(made_on_folder, exist_ok=True)

    # Using ground-state double gaussian fit method
    all_qubit_temps, all_qubit_times, all_qubit_temps_errs, fit_results = SSF_calcs_obj.run_ssf_qtemps_notprebuilt(
        pairs_info, limit_temp_k=1.0, do_plots = True, save_figs_path = made_on_folder)

elif alt_ssf_analysis_flags["iminuit_method"]:

    SSF_calcs_obj = SSFTempCalcAndPlots(figure_quality, tot_num_of_qubits, run_num, save_figs)
    pairs_info = SSF_calcs_obj.process_ssf_and_qfreq_data_qtemps(Science_Qubits, paths_SSFmethods)

    qubit_folder = os.path.join(path_saveplots_fits, "Iminuit_method")
    os.makedirs(qubit_folder, exist_ok=True)
    # Make a date‐stamped subfolder
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    made_on_folder = os.path.join(qubit_folder, f"made_on_{date_str}")
    os.makedirs(made_on_folder, exist_ok=True)

    # Using ground-state double gaussian fit method
    all_qubit_temps, all_qubit_times, all_qubit_temps_errs, fit_results = SSF_calcs_obj.run_ssf_qtemps_iminuit(pairs_info, run_num=run_num, limit_temp_k=1.0,
                                                                                                                do_plots = True, save_figs_path = made_on_folder, dontuse_midpt_thresh = True,
                                                                                                               low_leakage_mode = low_thermal_pops, ssf_hist_ylim = ssf_hist_ylim, apply_quality_cuts = filter_out_bad_SSF_qtemp_fits,
                                                                                                               calc_SNR = calc_SNR_ssfqtemps, calc_e_state_decay = calc_SSF_e_decay)

################################################### Combined Qubit Temperature Analyses ##########################################################
#################################### Analyses combining multiple qubit temp methods AND/OR multiple runs #########################################
run_num_list = [5,6,7,8,9] # for quiet, start at 5. no qtemp data for run 4. use run 9.2 for run 9c
rpm_temps_by_run = {}      # rpm_temps_by_run[run][qid] = [T_mK, ...]
rpm_temps_errs_by_run  = {}      # matching errors
rpm_Pe_by_run = {}      # rpm_Pe_by_run[run][qid] = [P_e, ...]
rpm_Pe_errs_by_run  = {}      # matching Pe errors
rpm_times_by_run = {}  # rpm_times_by_run[run][qid] = [datetime, ...]

ssf_g_temps_by_run  = {}   # ssf ground-double-gauss temps
ssf_g_temp_errs_by_run  = {}      # matching errors
ssf_g_Pe_by_run  = {}   # ssf ground-double-gauss Pe
ssf_g_Pe_errs_by_run = {} # matching Pe errors
fit_results_g_by_run = {}  # fit_results_g_by_run[run][qid] = list of accepted SSF fit records

ssf_ge_temps_by_run = {}   # ssf g-e threshold temps (if you compute them)
ssf_ge_temp_errs_by_run  = {}      # matching errors

ssf_fid_values_by_run = {} # single shot fidelity values
ssf_err_values_by_run = {} # single shot fidelity errors (total errs)
ssf_snr_by_run = {}  # SNR values from each ssf scan
ie_new_Pg_vals_by_run = {} # pop. corresponding to T1 decay, failed pi pulses etc in SSF meas.
ie_new_Pg_errs_by_run = {} # corresponding errors

if qtemp_method_flags["combined_studies_Qtemps"]:
    for run_num in run_num_list:
        # ---- always reset optional pre-SR variables each iteration ----
        base_dir2 = None
        filter_keywords2 = None
        target_dates_qtemps_RPM2 = None

        # -- Reset this param for SSF qubit temps --
        low_thermal_pops = False

        if run_num == 5:
            # ---------------- RPM (none) ----------------
            Science_Qubits = [0, 1, 2, 3, 4, 5]
            base_dir = ""
            filter_keywords = []
            outerFolder_qtemps_plots_RR = ""
            outerFolder_qtemps_plots = ""
            target_dates_qtemps_RPM = ""
            print("There is no RPM data for run 5 (SSF only).")

            # ---------------- SSF ----------------
            paths_SSFmethods = paths_SSFmethods_run5
            path_saveplots_fits = path_saveplots_fits_run5
            path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run5

        elif run_num == 6:
            # ---------------- RPM (science run; optional pre-science add-on) ----------------
            Science_Qubits = [0, 4]
            base_dir = base_dir_sciencerun
            filter_keywords = filter_keywords_sciencerun
            outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run6
            outerFolder_qtemps_plots = outerFolder_qtemps_plots_RR_run6
            target_dates_qtemps_RPM = target_dates_qtemps_RPM_sciencerun

            if pre_sciencerun6_data:
                base_dir2 = base_dir_pre_sciencerun
                filter_keywords2 = filter_keywords_presciencerun
                target_dates_qtemps_RPM2 = target_dates_qtemps_RPM_presciencerun

            # ---------------- SSF (science-run paths; optional pre-science add-on) ----------------
            paths_SSFmethods = paths_SSFmethods_SR.copy()
            if pre_sciencerun6_data:
                # include all qubits since pre-SR SSF was taken for all Qs
                Science_Qubits = [0, 1, 2, 3, 4, 5]
                paths_SSFmethods += paths_SSFmethods_preSR

            path_saveplots_fits = path_saveplots_fits_run6
            path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run6


        elif run_num == 7:
            # ---------------- RPM ----------------
            Science_Qubits = [0, 1, 2, 3, 4, 5]
            base_dir = base_dir_run7
            filter_keywords = filter_keywords_run7
            outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run7
            outerFolder_qtemps_plots = outerFolder_qtemps_plots_run7
            target_dates_qtemps_RPM = target_dates_qtemps_RPM_run7

            # ---------------- SSF ----------------
            paths_SSFmethods = paths_SSFmethods_run7
            path_saveplots_fits = path_saveplots_fits_run7
            path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run7


        elif run_num == 8:
            # ---------------- RPM ----------------
            Science_Qubits = [0, 1, 2, 3, 4, 5]
            base_dir = base_dir_run8
            filter_keywords = filter_keywords_run8
            outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run8
            outerFolder_qtemps_plots = outerFolder_qtemps_plots_run8
            target_dates_qtemps_RPM = target_dates_qtemps_RPM_run8

            # ---------------- SSF ----------------
            paths_SSFmethods = paths_SSFmethods_run8
            path_saveplots_fits = path_saveplots_fits_run8
            path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run8

        elif run_num == 9:
            low_thermal_pops = True # for SSF qubit temps (there were really low thermal pops in this run)

            # ---------------- RPM ----------------
            Science_Qubits = [0, 1, 2, 3, 5]
            base_dir = base_dir_run9
            filter_keywords = filter_keywords_run9
            outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run9
            outerFolder_qtemps_plots = outerFolder_qtemps_plots_run9
            target_dates_qtemps_RPM = target_dates_qtemps_RPM_run9

            # ---------------- SSF ----------------
            paths_SSFmethods = paths_SSFmethods_run9
            path_saveplots_fits = path_saveplots_fits_run9
            path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run9

        elif run_num == 9.2:  # this is run 9c
            # low_thermal_pops = True

            # ---------------- RPM ----------------
            Science_Qubits = [0, 1, 2, 3, 5]
            base_dir = base_dir_run9c
            filter_keywords = filter_keywords_run9c
            outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run9c
            outerFolder_qtemps_plots = outerFolder_qtemps_plots_run9c
            target_dates_qtemps_RPM = target_dates_qtemps_RPM_run9c

            # ---------------- SSF ----------------
            paths_SSFmethods = paths_SSFmethods_run9c
            path_saveplots_fits = path_saveplots_fits_run9c
            path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run9c

        else:
            raise ValueError(f"Unsupported run_num={run_num}")

        # ------------ Initialize class for combined qubit temps analysis ------------------
        combined_studies = combined_Qtemp_studies(figure_quality, tot_num_of_qubits)

        # ============================================================
        # Simple cache options for processed SSF/RPM inputs
        # ============================================================
        if comb_analysis_flags["create_cached_qtemp_files"] and comb_analysis_flags["use_cached_qtemp_files"]:
            raise ValueError("Choose only one or set both to False: create_cached_qtemp_files or use_cached_qtemp_files.")

        cache_dir = f"/home/acolonce/Documents/analysis/cached_processed_data/run{run_num}"

        if comb_analysis_flags["use_cached_qtemp_files"]:
            os.makedirs(cache_dir, exist_ok=True)

            # Default: no cached qtemp data
            fit_results_g = {}
            all_files_Qtemp_results_RPMs = {}

            if filter_out_bad_SSF_qtemp_fits:
                ssf_file_ext = "_filtered"
            else:
                ssf_file_ext = "_unfiltered"

            # Run 5 has SSF data, but no RPM data
            if run_num == 5:
                fit_results_g_cache_path = (f"{cache_dir}/run{run_num}_processed_SSF_fit_results_g{ssf_file_ext}.pkl")
                print(f"Run {run_num}: loading SSF only. No RPM data available.")
                fit_results_g, _ = combined_studies.load_processed_ssf_rpm_inputs(fit_results_g_cache_path,None)

            # Runs 6+ have both SSF and RPM data
            else:
                fit_results_g_cache_path = (f"{cache_dir}/run{run_num}_processed_SSF_fit_results_g{ssf_file_ext}.pkl")
                rpm_results_cache_path = (f"{cache_dir}/run{run_num}_processed_all_files_Qtemp_results_RPMs.pkl")

                print(f"Run {run_num}: loading SSF and RPM data.")
                fit_results_g, all_files_Qtemp_results_RPMs = combined_studies.load_processed_ssf_rpm_inputs(
                    fit_results_g_cache_path,
                    rpm_results_cache_path)

                # #Temporary: to consider only the last chunk of run 9a AB paper data.
                # if run_num == 9 and all_files_Qtemp_results_RPMs:
                #     start_dt = pd.to_datetime("2026-04-25 00:00:00") # last two days only
                #     filtered = []
                #     for rec in all_files_Qtemp_results_RPMs:
                #         new_qubits = {}
                #         for qid, qrec in rec.get("qubits", {}).items():
                #             qdate = pd.to_datetime(qrec.get("date"), unit="s", errors="coerce")
                #             if pd.notna(qdate) and qdate >= start_dt:
                #                 new_qubits[qid] = qrec
                #         if new_qubits:
                #             rec = rec.copy()
                #             rec["qubits"] = new_qubits
                #             filtered.append(rec)
                #     print(f"Run 9 RPM files before filter: {len(all_files_Qtemp_results_RPMs)}")
                #     print(f"Run 9 RPM files after filter: {len(filtered)}")
                #     all_files_Qtemp_results_RPMs = filtered

            # Makes sure processing sections are set to False if the user forgot
            comb_analysis_flags["load_rpm"] = False
            comb_analysis_flags["load_ssf"] = False
        # -----------------------------------------------------------------------

        # ---------------- pre-fill so dict shape is always stable ----------------
        rpm_temps_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        rpm_temps_errs_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        rpm_Pe_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        rpm_Pe_errs_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        rpm_times_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        
        ssf_g_temps_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        ssf_g_temp_errs_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        ssf_g_Pe_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        ssf_g_Pe_errs_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        fit_results_g_by_run[run_num] = {qid: [] for qid in range(tot_num_of_qubits)}
        ssf_ge_temps_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        ssf_ge_temp_errs_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        ssf_fid_values_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        ssf_err_values_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        ssf_snr_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        ie_new_Pg_vals_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        ie_new_Pg_errs_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]

        # ============================================================
        # If using cached qtemp files, adapt cached SSF/RPM data into
        # the same per-run dictionaries used by the plotting functions.
        # ============================================================
        if comb_analysis_flags["use_cached_qtemp_files"]:
            # ---------------- RPM cached data ----------------
            # Run 5 has no RPM data, so leave the pre-filled empty lists.
            if run_num != 5 and all_files_Qtemp_results_RPMs:
                rpm_times, rpm_temps, rpm_temps_errs, rpm_Pe, rpm_Pe_errs = combined_studies.rpm_results_to_per_qubit_lists(all_files_Qtemp_results_RPMs, n_qubits=tot_num_of_qubits)
                rpm_temps_by_run[run_num] = rpm_temps
                rpm_temps_errs_by_run[run_num] = rpm_temps_errs
                rpm_Pe_by_run[run_num] = rpm_Pe
                rpm_Pe_errs_by_run[run_num] = rpm_Pe_errs
                rpm_times_by_run[run_num] = rpm_times

                # ---------------- SSF cached data ----------------
            if fit_results_g:
                fit_results_g_by_run[run_num] = fit_results_g
                ssf_g_temps, ssf_g_temp_errs, ssf_fid_vals, ssf_fid_errs, ssf_snr_vals, ie_new_Pg_vals, ie_new_Pg_errs = (combined_studies.ssf_fit_results_to_per_qubit_lists(fit_results_g,n_qubits=tot_num_of_qubits))
                ssf_g_temps_by_run[run_num] = ssf_g_temps
                ssf_g_temp_errs_by_run[run_num] = ssf_g_temp_errs
                pe_vals, pe_errs = combined_studies.extract_pe_from_fit_results(fit_results_g,tot_num_of_qubits)
                ssf_g_Pe_by_run[run_num] = pe_vals
                ssf_g_Pe_errs_by_run[run_num] = pe_errs
                ssf_fid_values_by_run[run_num] = ssf_fid_vals
                ssf_err_values_by_run[run_num] = ssf_fid_errs
                ssf_snr_by_run[run_num] = ssf_snr_vals
                ie_new_Pg_vals_by_run[run_num] = ie_new_Pg_vals
                ie_new_Pg_errs_by_run[run_num] = ie_new_Pg_errs

            print(f"\nRUN {run_num} cached SSF counts:")
            for q in range(tot_num_of_qubits):
                print(f"  Q{q + 1}: {len(ssf_g_Pe_by_run[run_num][q])}")

        if comb_analysis_flags["load_rpm"]:
            all_files_Qtemp_results_RPMs = {}
            if run_num != 5: # no rpm data for run 5
                # ----------- Get Qubit temperature results via RPMs
                RPM_calcs = RPMTempCalcAndPlots(figure_quality, tot_num_of_qubits)
                all_files_Qtemp_results_RPMs = RPM_calcs.run_RPMqtemps(base_dir, target_dates_qtemps_RPM, filter_keywords, fit_saved, signal,
                                                              run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                                                              outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data,
                                                              get_london_data, figure_quality, save_figsRR, exclude_temp_sweeps,
                                                              passing_pre_sciencerun_data=False, filter_out_bad_RPM_fits = filter_out_bad_RPM_fits,
                                                            combine_IQ_signal = rpm_combine_IQ_signal)
                if run_num == 6:
                    if pre_sciencerun6_data:
                        all_files_Qtemp_results_RPMs2 = RPM_calcs.run_RPMqtemps(base_dir2, target_dates_qtemps_RPM2, filter_keywords2, fit_saved, signal, run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                                                                      outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, get_london_data, figure_quality, save_figsRR, exclude_temp_sweeps,
                                                                                passing_pre_sciencerun_data = True, filter_out_bad_RPM_fits = filter_out_bad_RPM_fits, combine_IQ_signal = rpm_combine_IQ_signal)
                        all_files_Qtemp_results_RPMs += all_files_Qtemp_results_RPMs2

                # ---- ADAPT + STORE (RPM) ----
                rpm_times, rpm_temps, rpm_temps_errs, rpm_Pe, rpm_Pe_errs = combined_studies.rpm_results_to_per_qubit_lists(
                    all_files_Qtemp_results_RPMs,
                    n_qubits=tot_num_of_qubits
                )
                rpm_temps_by_run[run_num] = rpm_temps
                rpm_temps_errs_by_run[run_num] = rpm_temps_errs
                rpm_Pe_by_run[run_num] = rpm_Pe
                rpm_Pe_errs_by_run[run_num] = rpm_Pe_errs
                rpm_times_by_run[run_num] = rpm_times

        if comb_analysis_flags["load_ssf"]:
            # ----------- Get Qubit temperature results via SSF g-e threshold method and SSF g-state double gaussian threshold method
            SSF_calcs_obj = SSFTempCalcAndPlots(figure_quality, tot_num_of_qubits, run_num, save_figs)
            pairs_info = SSF_calcs_obj.process_ssf_and_qfreq_data_qtemps(Science_Qubits, paths_SSFmethods)

            if use_iminuit_gdoublegauss_ssf: # Made a special iminuit-based double gaussian fitting function, but for now it is only set up to fit g-state data.
                all_qubit_temps_g, all_qubit_times_g, all_qubit_temps_errs_g, fit_results_g = SSF_calcs_obj.run_ssf_qtemps_iminuit(pairs_info, run_num=run_num, limit_temp_k=0.6,
                    do_plots=False, dontuse_midpt_thresh = True, low_leakage_mode = low_thermal_pops, ssf_hist_ylim = ssf_hist_ylim, apply_quality_cuts = filter_out_bad_SSF_qtemp_fits,
                                                                                                                    calc_SNR = calc_SNR_ssfqtemps, calc_e_state_decay = calc_SSF_e_decay)
                
                # ---- STORE FULL FIT RESULTS FOR SSF LOG OVERLAY PLOTS ----
                fit_results_g_by_run[run_num] = fit_results_g

                # ---- STORE RESULTS (SSF g) ----
                ssf_g_temps, ssf_g_temp_errs, ssf_fid_vals, ssf_fid_errs, ssf_snr_vals, ie_new_Pg_vals, ie_new_Pg_errs = combined_studies.ssf_fit_results_to_per_qubit_lists(fit_results_g, n_qubits=tot_num_of_qubits)

                print(f"\nRUN {run_num} accepted SSF counts:")
                for q in range(tot_num_of_qubits):
                    print(f"  Q{q + 1}: {len(ssf_g_temps[q])}")

                ssf_g_temps_by_run[run_num] = ssf_g_temps
                ssf_g_temp_errs_by_run[run_num] = ssf_g_temp_errs
                pe_vals, pe_errs = combined_studies.extract_pe_from_fit_results(fit_results_g, tot_num_of_qubits)
                ssf_g_Pe_by_run[run_num] = pe_vals
                ssf_g_Pe_errs_by_run[run_num] = pe_errs
                ssf_fid_values_by_run[run_num] = ssf_fid_vals
                ssf_err_values_by_run[run_num] = ssf_fid_errs
                ssf_snr_by_run[run_num] = ssf_snr_vals
                ie_new_Pg_vals_by_run[run_num] = ie_new_Pg_vals
                ie_new_Pg_errs_by_run[run_num] = ie_new_Pg_errs

            else: # uses sklearn.mixture.GaussianMixture for double gaussian fitting
                all_qubit_temps_g, all_qubit_times_g, all_qubit_temps_errs_g, fit_results_g  = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=1.0, use_gessf_thresh_only = False, fallback_to_threshold = False)
                all_qubit_temps_ge, all_qubit_times_ge, all_qubit_temps_errs_ge, fit_results_ge = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=1.0, use_gessf_thresh_only=True, fallback_to_threshold=False)

                # ---- STORE FULL FIT RESULTS FOR SSF LOG OVERLAY PLOTS ----
                fit_results_g_by_run[run_num] = fit_results_g

                # ---- STORE RESULTS (SSF g) ----
                ssf_g_temps, ssf_g_temp_errs, ssf_fid_vals, ssf_fid_errs, ssf_snr_vals, ie_new_Pg_vals, ie_new_Pg_errs = combined_studies.ssf_fit_results_to_per_qubit_lists(fit_results_g, n_qubits=tot_num_of_qubits)
                print(f"\nRUN {run_num} accepted SSF counts:")
                for q in range(tot_num_of_qubits):
                    print(f"  Q{q + 1}: {len(ssf_g_temps[q])}")

                ssf_g_temps_by_run[run_num] = ssf_g_temps
                ssf_g_temp_errs_by_run[run_num] = ssf_g_temp_errs
                pe_vals, pe_errs = combined_studies.extract_pe_from_fit_results(fit_results_g, tot_num_of_qubits)
                ssf_g_Pe_by_run[run_num] = pe_vals
                ssf_g_Pe_errs_by_run[run_num] = pe_errs
                ssf_fid_values_by_run[run_num] = ssf_fid_vals
                ssf_err_values_by_run[run_num] = ssf_fid_errs
                ssf_snr_by_run[run_num] = ssf_snr_vals
                ie_new_Pg_vals_by_run[run_num] = ie_new_Pg_vals
                ie_new_Pg_errs_by_run[run_num] = ie_new_Pg_errs

                # ---- STORE RESULTS (SSF ge) ----
                # Not tested yet
                ssf_ge_temps, ssf_ge_temp_errs, ssf_fid_vals_ge, ssf_fid_errs_ge, ssf_snr_vals_ge, ie_new_Pg_vals_ge, ie_new_Pg_errs_ge = combined_studies.ssf_fit_results_to_per_qubit_lists(fit_results_ge, n_qubits=tot_num_of_qubits)
                ssf_ge_temps_by_run[run_num] = ssf_ge_temps
                ssf_ge_temp_errs_by_run[run_num] = ssf_ge_temp_errs

        # ============================================================
        # Optionally create cached files after processing
        # ============================================================
        if comb_analysis_flags["create_cached_qtemp_files"] and not comb_analysis_flags["use_cached_qtemp_files"]:
            ssf_cache_path, rpm_cache_path = combined_studies.save_processed_ssf_rpm_inputs(
                fit_results_g, all_files_Qtemp_results_RPMs, save_dir=cache_dir, tag=f"run{run_num}_processed")
        #----------------------------------------------------------------

    if comb_analysis_flags["multirun_RPM_Pe_vs_t"]:
        have_qtemp_inputs = (comb_analysis_flags["use_cached_qtemp_files"] or comb_analysis_flags["load_rpm"])
        if not have_qtemp_inputs:
            raise ValueError('This plot requires qtemp inputs. Either set '
                             'comb_analysis_flags["use_cached_qtemp_files"] = True, or set '
                             'comb_analysis_flags["load_rpm"] to True.')
        combined_studies.plot_rpm_pe_vs_shifted_time_by_run(
            run_num_list=run_num_list,
            rpm_times_by_run=rpm_times_by_run,
            rpm_Pe_by_run=rpm_Pe_by_run,
            rpm_Pe_errs_by_run=rpm_Pe_errs_by_run,
            qubits_to_plot=[0, 1, 2, 3, 5],
            num_qubits=tot_num_of_qubits,
            time_units="hours",
            plot_percent=True,
            ylim=None,
            save_plt_path="/home/acolonce/Documents/analysis/multirun/Pe_vs_t/",
        )

    if comb_analysis_flags["plot_ssf_log_curves"]:
        have_qtemp_inputs = (comb_analysis_flags["use_cached_qtemp_files"] or comb_analysis_flags["load_ssf"])
        if not have_qtemp_inputs:
            raise ValueError('This plot requires qtemp inputs. Either set '
                            'comb_analysis_flags["use_cached_qtemp_files"] = True, or set '
                            'comb_analysis_flags["load_ssf"] to True.')
        ssf_overlay_save_path = "/home/acolonce/Documents/analysis/ssf_qtemps/ssf_log_curves" # cosmiqserver01
            #"/data/QICK_data/run9/6transmon/analysis/ssf_qtemps/ssf_log_curves" #daq01
        for qid in range(tot_num_of_qubits):
            has_any_data = any(
                run_num in fit_results_g_by_run
                and qid in fit_results_g_by_run[run_num]
                and len(fit_results_g_by_run[run_num][qid]) > 0
                for run_num in fit_results_g_by_run)

            if not has_any_data:
                print(f"Skipping Q{qid + 1}: no accepted SSF fit results.")
                continue

            combined_studies.plot_ssf_log_overlay_by_run(
                fit_results_by_run=fit_results_g_by_run,
                qid=qid,
                save_figs_path=ssf_overlay_save_path,
                bins=np.linspace(-0.75, 1.75, 220),
                cmap_name="Blues",
                plot_individual=False,
                plot_run_median=True,
                smooth_window=3,
                ymin=1e-4,
                ymax=1.3,
                title=f"Q{qid + 1} SSF thermal population across runs",
                filename=f"Q{qid + 1}_SSF_log_overlay_by_run.png",
                show=True,
            )
    # --------------------- box and whiskers plots. Per run and per qubit. Separate or together options -------------------
    if comb_analysis_flags["qtemp_box_whisker_allruns_allQs"]:
        have_qtemp_inputs = (comb_analysis_flags["use_cached_qtemp_files"] or (comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"]))
        if not have_qtemp_inputs:
            raise ValueError('This plot requires qtemp inputs. Either set '
                             'comb_analysis_flags["use_cached_qtemp_files"] = True, or set both '
                             'comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"] to True.')
        boxwhisker_qtemps_per_qubit_vs_run_choice(
            run_num_list=run_num_list,
            rpm_temps_by_run=rpm_temps_by_run,
            ssf_g_temps_by_run=ssf_g_temps_by_run,
            ssf_ge_temps_by_run=ssf_ge_temps_by_run,
            qubits_to_plot = [0,1,2],
            plot_mode="compare_methods", # "hybrid" or "all_ssf" or "compare_methods"
            ssf_kind="g",
            layout="separate",
            colors=('darkblue', 'darkblue', 'darkblue', # palevioletred
                    'darkblue', 'darkblue', 'darkblue'),
            ylims=(0, 620),
            yticks=np.arange(0, 601, 100),
            showfliers=True,  # outliers
            save_plt_path = "/home/acolonce/Documents/analysis/multirun/qubit_temps/combined_ssf_rpm", # if set to 'None' uses plt.show()
            # "/home/acolonce/Documents/analysis/multirun/qubit_temps/combined_ssf_rpm" #cosmiqserver01
            # "/data/QICK_data/multirun_analysis/qubit_temps/combined" # daq01
            #"/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/QTemperatures/Plots/combined_analysis_RPM_SSF")
            log_y=True, # only goes into effect for hybrid plot_mode at the moment
            log_yticks_mK= (25, 50, 100, 250, 500))#(10, 20, 50, 100, 200, 500))

    if comb_analysis_flags["Pe_box_whisker_allruns_allQs"]:
        have_qtemp_inputs = (comb_analysis_flags["use_cached_qtemp_files"] or (comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"]))
        if not have_qtemp_inputs:
            raise ValueError('This plot requires qtemp inputs. Either set '
                             'comb_analysis_flags["use_cached_qtemp_files"] = True, or set both '
                             'comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"] to True.')
        # This is only set up in 'hybrid' 'separate' mode
        boxwhisker_pe_per_qubit_vs_run_hybrid(
            run_num_list=run_num_list,
            rpm_pe_by_run=rpm_Pe_by_run,
            ssf_pe_by_run=ssf_g_Pe_by_run,
            qubits_to_plot=[0, 1, 2, 3, 4, 5],
            colors=('darkblue', 'darkblue', 'darkblue', # palevioletred, forestgreen, darkblue
                    'darkblue', 'darkblue', 'darkblue'),
            ylims= (0.001, 0.6), #(0, 0.45),
            yticks=np.arange(0.05, 0.46, 0.1),
            showfliers=True,  # outliers
            fig_title=r"Excited-State Population vs Run Number",
            ylabel=r"Excited-State Population (%)",
            add_last_run_inset=False,
            save_plt_path= "/home/acolonce/Documents/analysis/multirun/qubit_temps/combined_ssf_rpm", # if set to 'None' uses plt.show()
            # "/home/acolonce/Documents/analysis/multirun/qubit_temps/combined_ssf_rpm" #cosmiqserver01
            #"/data/QICK_data/multirun_analysis/qubit_temps/combined") #daq01
            #"/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/QTemperatures/Plots/combined_analysis_RPM_SSF") # CEPH
            log_y=True,
            log_yticks_percent=[0.1, 1, 10, 50])

    if comb_analysis_flags["ssf_box_whisker_allruns_allQs"]:
        have_qtemp_inputs = (comb_analysis_flags["use_cached_qtemp_files"] or comb_analysis_flags["load_ssf"])
        if not have_qtemp_inputs:
            raise ValueError('This plot requires qtemp inputs. Either set '
                             'comb_analysis_flags["use_cached_qtemp_files"] = True, or set '
                             'comb_analysis_flags["load_ssf"] to True.')
        boxwhisker_ssf_per_qubit_vs_run(
                run_num_list,
                ssf_vals_by_run = ssf_fid_values_by_run,
                qubits_to_plot=[0, 1, 2, 3, 4, 5],
                colors="navy", #('orange', 'blue', 'purple', 'green', 'brown', 'palevioletred')
                ylims=None,
                yticks=None,
                showfliers=True,
                save_plt_path= "/home/acolonce/Documents/analysis/multirun/SSF_fid/") # if set to 'None' uses plt.show()
                # "/home/acolonce/Documents/analysis/multirun/SSF_fid" #cosmiqserver01
                #"/data/QICK_data/multirun_analysis/qubit_temps/SSF" ) #daq01
                #"/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/QTemperatures/Plots/SSF_fid") #CEPH

    if comb_analysis_flags["SNR_box_whisker_allruns_allQs"]:
        have_qtemp_inputs = (comb_analysis_flags["use_cached_qtemp_files"] or comb_analysis_flags["load_ssf"])
        if not have_qtemp_inputs:
            raise ValueError(
                'This plot requires SSF inputs. Either set '
                'comb_analysis_flags["use_cached_qtemp_files"] = True, or set '
                'comb_analysis_flags["load_ssf"] = True.')

        boxwhisker_snr_per_qubit_vs_run(
            run_num_list=run_num_list,
            snr_vals_by_run=ssf_snr_by_run,
            n_qubits=tot_num_of_qubits,
            qubits_to_plot=[0, 1, 2, 3, 5],
            colors=('darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue'),
            ylims=None,
            yticks=None,
            showfliers=True, # outliers
            fig_title="Readout SNR vs Run Number",
            ylabel="Readout SNR",
            save_plt_path="/home/acolonce/Documents/analysis/multirun/SNR")

    if comb_analysis_flags["ie_new_Pg_boxwhisk_allruns_allQs"]:
        have_qtemp_inputs = (comb_analysis_flags["use_cached_qtemp_files"]or comb_analysis_flags["load_ssf"])
        if not have_qtemp_inputs:
            raise ValueError(
                'This plot requires SSF inputs. Either set '
                'comb_analysis_flags["use_cached_qtemp_files"] = True, or set '
                'comb_analysis_flags["load_ssf"] = True.')

        boxwhisker_ie_new_Pg_per_Q_vs_run(
            run_num_list=run_num_list,
            ie_new_Pg_vals_by_run=ie_new_Pg_vals_by_run,
            n_qubits=tot_num_of_qubits,
            qubits_to_plot=[0, 1, 2, 3, 5],
            colors=('darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue'),
            ylims=None,
            yticks=None,
            showfliers=False,
            fig_title=r"Excited-State Ground-Like Population vs Run Number",
            ylabel=r"Excited-State Ground-Like Population",
            convert_to_percent=False,
            save_plt_path="/home/acolonce/Documents/analysis/multirun/ie_new_Pg")

    # ------------ Qubit temperatures vs Time using all three methods ------------------------
    if comb_analysis_flags["Qtemps_vs_time_comb_separate_plts"]:
        if len(run_num_list) != 1:
            raise ValueError(f"Expected exactly 1 run in 'run_num_list', but got {len(run_num_list)}. "
                "This section is only set up to process one run at a time.")
        have_qtemp_inputs = (comb_analysis_flags["use_cached_qtemp_files"] or (comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"]))
        if not have_qtemp_inputs:
            raise ValueError('This plot requires qtemp inputs. Either set '
                             'comb_analysis_flags["use_cached_qtemp_files"] = True, or set both '
                             'comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"] to True.')
        # This func has only been set up to work for 2 qubits.
        # Plots two rows (one for each qubit) and 3 columns (one for each method)
        combined_studies.Qtemps_vs_time_comb_methods_3col(all_qubit_temps_g, all_qubit_times_g, all_qubit_temps_errs_g, all_qubit_temps_ge, all_qubit_times_ge, all_qubit_temps_errs_ge,
                                                     outerFolder_qtemps_plots, all_files_Qtemp_results_RPMs, restrict_time_xaxis = False, plot_extra_event_lines = False,
                                                     rad_events_plot_lines = False, plot_error_bars = True)
    if comb_analysis_flags["Qtemps_vs_time_comb_single_plt"]:
        if len(run_num_list) != 1:
            raise ValueError(f"Expected exactly 1 run in 'run_num_list', but got {len(run_num_list)}. "
                "This section is only set up to process one run at a time.")
        have_qtemp_inputs = (comb_analysis_flags["use_cached_qtemp_files"] or (comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"]))
        if not have_qtemp_inputs:
            raise ValueError('This plot requires qtemp inputs. Either set '
                             'comb_analysis_flags["use_cached_qtemp_files"] = True, or set both '
                             'comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"] to True.')
        # This one works for multiple qubits (has been improved)
        # Makes 1 subplot per qubit (and all methods in a single plot). Note: I removed the ge SSF method from being plotted since we haven't been using that one lately.
        # Plots error bars always, unless you pass None instead of all_qubit_temps_errs_g.
        combined_studies.Qtemps_vs_time_comb_allQs_1col(all_qubit_temps_g, all_qubit_times_g, outerFolder_qtemps_plots,
                                                     all_files_Qtemp_results_RPMs, all_qubit_temps_errs_g, restrict_time_yaxis = True, ylims = [70,160],
                                                        rad_events_plot_lines = False, qubits_to_plot = [0,1,2,3,5], # 0-based indexing
                                                        plot_rpm_I_only=False, plot_rpm_Q_only=False)

    if comb_analysis_flags["SSF_fid_vs_RRPM_Pe_2D"]: # only configured to run for one run at a time
        # Plots SSF vs Pe, using Pe values extracted from RPM data, not SSF data
        if len(run_num_list) != 1:
            raise ValueError(f"Expected exactly 1 run in 'run_num_list', but got {len(run_num_list)}. "
                "This section is only set up to process one run at a time.")
        if not comb_analysis_flags["use_cached_qtemp_files"] and not (comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"]):
            raise ValueError(
                'This plot requires either use_cached_files=True, or both '
                'comb_analysis_flags["load_rpm"] and '
                'comb_analysis_flags["load_ssf"] to be True.')

        combined_studies.SSF_fid_vs_RRPM_Pe(
            ssf_fit_results=fit_results_g,
            all_files_Qtemp_results_RPMs=all_files_Qtemp_results_RPMs,
            out_dir=outerFolder_qtemps_plots,
            qubits_to_plot=[0, 1, 2, 3, 5],
            tolerance_seconds=10, # 10 seconds for all runs except Run 6 SCIENCE run data (600s)
            plot_together=True,
            xlims=  (0.0, 0.06),
            ylims= (0.74, 0.925),
            RPM_Pe_rel_err_cut = None, #0.25,
            plot_with_t_color_gradient = True, # to depict time passed
            plot_ideal_line = False,
            plot_with_t_markers=False, # second option to depict time passed. Only for plot_together case
            nearest_neighbor_average=True,
            nn_average_neighbors=2)

    if comb_analysis_flags["SSF_fid_vs_RRPM_Pe_3D"]:  # only configured to run for one run at a time
        # Plots SSF vs Pe, using Pe values extracted from RPM data, not SSF data
        if len(run_num_list) != 1:
            raise ValueError(f"Expected exactly 1 run in 'run_num_list', but got {len(run_num_list)}. "
                             "This section is only set up to process one run at a time.")
        if not comb_analysis_flags["use_cached_qtemp_files"] and not (comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"]):
            raise ValueError(
                'This plot requires either use_cached_files=True, or both '
                'comb_analysis_flags["load_rpm"] and '
                'comb_analysis_flags["load_ssf"] to be True.')

        matched_3d = combined_studies.SSF_fid_vs_RRPM_Pe_3D(
            fit_results_g,
            all_files_Qtemp_results_RPMs,
            out_dir=outerFolder_qtemps_plots,
            qubits_to_plot=[0, 1, 2, 3, 5],
            tolerance_seconds=10,
            RPM_Pe_rel_err_cut=0.5,
            axis_order="time_pe_ssf",
            xlims=(275, 0),  # time
            ylims=(0.0, 0.07),  # RPM Pe
            zlims=(0.6, 0.95),  # SSF
            elev=8, # positive = from above, negative = from below
            azim=-30, #-40
            plot_qubits_separately = True,
            show_bottom_shadow = True
        )

        if comb_analysis_flags["SSF_fid_vs_RRPM_Pe_video"]:  # only configured to run for one run at a time
            anim_path = combined_studies.animate_SSF_fid_vs_RRPM_Pe_2D(
                matched_3d,
                out_dir=outerFolder_qtemps_plots,
                qubits_to_plot=[0, 1, 2, 3, 5],
                xlims=(0, 0.07),
                ylims=(0.6, 1.0),
                fps=10, #Frames per second.
                frame_step=20,
                show_errorbars=True,
                plot_ideal_line=False,
                save_as="gif" # mp4 or gif
            )

    if comb_analysis_flags["SNR_vs_RRPM_Pe"]:  # only configured to run for one run at a time
        # Plots SSF vs Pe, using Pe values extracted from RPM data, not SSF data
        if len(run_num_list) != 1:
            raise ValueError(f"Expected exactly 1 run in 'run_num_list', but got {len(run_num_list)}. "
                             "This section is only set up to process one run at a time.")
        if not comb_analysis_flags["use_cached_qtemp_files"] and not (comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"]):
            raise ValueError(
                'This plot requires either use_cached_files=True, or both '
                'comb_analysis_flags["load_rpm"] and '
                'comb_analysis_flags["load_ssf"] to be True.')
        combined_studies.plot_ssf_SNR_vs_pe(
            fit_results_g,
            all_files_Qtemp_results_RPMs,
            outerFolder_qtemps_plots,
            n_qubits=6,
            tolerance_seconds=10,
            plot_together=True,
            xlims=(0, 0.07),
            ylims=None)

    #----------- Thermal Populations vs Time using all three methods ----
    if comb_analysis_flags["Pe_vs_time_comb_separate_plts"]:
        if len(run_num_list) != 1:
            raise ValueError(f"Expected exactly 1 run in 'run_num_list', but got {len(run_num_list)}. "
                "This section is only set up to process one run at a time.")
        have_qtemp_inputs = (comb_analysis_flags["use_cached_qtemp_files"] or (comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"]))
        if not have_qtemp_inputs:
            raise ValueError('This plot requires qtemp inputs. Either set '
                             'comb_analysis_flags["use_cached_qtemp_files"] = True, or set both '
                             'comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"] to True.')
        # Plots two rows (one for each qubit) and 3 columns (one for each method)
        combined_studies.Pe_vs_time_comb_methods(all_files_Qtemp_results_RPMs, fit_results_g, fit_results_ge, outerFolder_qtemps_plots,
                                                 restrict_time_xaxis = False, plot_extra_event_lines = False, rad_events_plot_lines = False)
    if comb_analysis_flags["Pe_vs_time_comb_single_plt"]:
        if len(run_num_list) != 1:
            raise ValueError(f"Expected exactly 1 run in 'run_num_list', but got {len(run_num_list)}. "
                "This section is only set up to process one run at a time.")
        have_qtemp_inputs = (comb_analysis_flags["use_cached_qtemp_files"] or (comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"]))
        if not have_qtemp_inputs:
            raise ValueError('This plot requires qtemp inputs. Either set '
                             'comb_analysis_flags["use_cached_qtemp_files"] = True, or set both '
                             'comb_analysis_flags["load_rpm"] and comb_analysis_flags["load_ssf"] to True.')
        # THREE METHODS VERSION
        # Plots two rows (one for each qubit) and 1 column (all three methods in a single plot)
        # combined_studies.Pe_vs_time_comb_2subplts(all_files_Qtemp_results_RPMs, fit_results_g, fit_results_ge, outerFolder_qtemps_plots,
        #                              restrict_time_xaxis = False, plot_extra_event_lines = False, rad_events_plot_lines = False)
        # TWO METHODS VERSION (new)
        combined_studies.Pe_vs_time_comb_allQs_1col(fit_results_g, outerFolder_qtemps_plots, all_files_Qtemp_results_RPMs, qubits_to_plot=[0,1,2,3,5], restrict_time_yaxis = True,ylims=[0.0, 0.3])

#################################################### London Penetration Analysis ##########################################################
if london_flags["get_qfreqs_resfreqs_qtemps"]: # There was no "pre-science-run" data in run 6 for this analysis, since the relevant data is the science run heater temperature sweep data
    #--------------------------------------Get RPM qubit temps, qfreqs and res freqs, etc. -----------------------
    RPM_calcs = RPMTempCalcAndPlots(figure_quality, tot_num_of_qubits)
    qfreqs_resfreqs_qtemps_data = RPM_calcs.run_RPMqtemps(base_dir, target_dates_qtemps_RPM, filter_keywords, fit_saved, signal,
                                                  run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                                                  outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, get_london_data,
                                                  figure_quality, save_figsRR, exclude_temp_sweeps, passing_pre_sciencerun_data = False, combine_IQ_signal = rpm_combine_IQ_signal)

    # ----------------------------Dump RPM qubit temps, qfreqs and res freqs, etc in excel spreadhseet -----------------------
    # These are not used in the definitions that follow, are just needed to re-initialize the class
    # this section does NOT include Pe distirbution errs, only Pe errs so far
    outerFolder = ""
    outerFolder_qtemps_data = ""
    date_string = ""
    RPM_plotter = PlotRR_noQick(date_string, figure_quality, save_figs, fit_saved, signal, run_name, tot_num_of_qubits, outerFolder, outerFolder_qtemps_plots, outerFolder_qtemps_data, run_num)
    RPM_plotter.save_RPM_qtemp_data_to_excel(qfreqs_resfreqs_qtemps_data, outerFolder_london_path, run_num, FRIDGE)
    del RPM_calcs

################################################### Coherence + Fridge  + Qtemps Analysis ###########################################
all_files_Qtemp_results_RPMs = None
all_qubit_times_g = None
all_qubit_temps_g = None
mcp1_dates = None
mcp1_temps = None
date_times_t1 = None
t1_vals = None
date_times_q_spec = None
q_freqs = None
date_times_t2r = None
t2r_vals = None
date_times_t2e = None
t2e_vals = None
Pe_dist_err_dict = None
use_png_timestamps = False

restrict_time = True
start_time = datetime.datetime(2025, 11, 18, 6, 0)
end_time = datetime.datetime(2025, 11, 21, 12, 0)
run_num_list = [8]
run6_subfolder="both" # only used for run6 MCP1 csv file data loading. Options: "pre-science-run", "science-run" or "both"
run8_temp_sweep = True

rpm_temps_by_run = {}      # rpm_temps_by_run[run][qid] = [T_mK, ...]
rpm_temps_errs_by_run  = {}      # matching errors
rpm_Pe_by_run = {}      # rpm_Pe_by_run[run][qid] = [P_e, ...]
rpm_Pe_errs_by_run  = {}      # matching Pe errors

ssf_g_temps_by_run  = {}   # ssf ground-double-gauss temps
ssf_g_temp_errs_by_run  = {}      # matching errors
ssf_g_Pe_by_run  = {}   # ssf ground-double-gauss Pe
ssf_g_Pe_errs_by_run = {} # matching Pe errors
fit_results_g_by_run = {}  # fit_results_g_by_run[run][qid] = list of accepted SSF fit records
matched_t1_to_ssf_by_run = {} # used for SSF limitations calcs
matched_rpm_to_ssf_by_run = {} # used for SSF limitations calcs
ssf_limitations_per_scan_by_run = {}
ssf_limitations_median_by_run = {}
ssf_limitations_per_scan_rpmPe_by_run = {}
ssf_limitations_median_rpmPe_by_run = {}

ssf_ge_temps_by_run = {}   # ssf g-e threshold temps (if you compute them)
ssf_ge_temp_errs_by_run  = {}      # matching errors

ssf_fid_values_by_run = {} # single shot fidelity values
ssf_err_values_by_run = {} # single shot fidelity errors (total errs)

t1_vals_by_run  = {}
date_times_t1_by_run = {}
t2r_vals_by_run = {}
t2e_vals_by_run = {}
qfreq_vals_by_run = {}
resfreq_vals_by_run = {}

t1_errs_by_run  = {}
t1_res_lengths_by_run = {}
t2r_errs_by_run = {}
t2e_errs_by_run = {}
qfreq_errs_by_run = {}

all_files_Qtemp_results_RPMs_by_run = {}
# ------------ Initialize class for combined analysis ------------------
combined_studies = combined_Qtemp_studies(figure_quality, tot_num_of_qubits)
        
# ========================================================================================
# Simple cache options for processed SSF/RPM inputs
# ========================================================================================
if coh_qtemp_ana_flags["create_cached_qtemp_files"] and coh_qtemp_ana_flags["use_cached_qtemp_files"]:
    raise ValueError("Choose only one or set both to False: create_cached_qtemp_files or use_cached_qtemp_files.")

cache_dir = f"/home/acolonce/Documents/analysis/cached_processed_data/run{run_num_list[0]}"

if coh_qtemp_ana_flags["use_cached_qtemp_files"]:
    os.makedirs(cache_dir, exist_ok=True)

    for run_number in run_num_list:
        # Default: no cached qtemp data
        fit_results_g = {}
        all_files_Qtemp_results_RPMs = {}

        if filter_out_bad_SSF_qtemp_fits:
            ssf_file_ext = "_filtered"
        else:
            ssf_file_ext = "_unfiltered"

        # Run 4 has no SSF or RPM qubit-temp data
        if run_number == 4:
            print(f"Run {run_number}: no cached SSF or RPM qtemp data. Skipping qtemp loading.")

        # Run 5 has SSF data, but no RPM data
        elif run_number == 5:
            fit_results_g_cache_path = (f"{cache_dir}/run{run_number}_processed_SSF_fit_results_g.pkl")
            print(f"Run {run_number}: loading SSF only. No RPM data available.")
            fit_results_g, _ = combined_studies.load_processed_ssf_rpm_inputs(
                fit_results_g_cache_path,None)

        # Runs 6+ have both SSF and RPM data
        else:
            fit_results_g_cache_path = (f"{cache_dir}/run{run_number}_processed_SSF_fit_results_g.pkl")
            rpm_results_cache_path = (f"{cache_dir}/run{run_number}_processed_all_files_Qtemp_results_RPMs.pkl")
            print(f"Run {run_number}: loading SSF and RPM data.")
            fit_results_g, all_files_Qtemp_results_RPMs = combined_studies.load_processed_ssf_rpm_inputs(
                fit_results_g_cache_path,
                rpm_results_cache_path)

        fit_results_g_by_run[run_number] = fit_results_g
        all_files_Qtemp_results_RPMs_by_run[run_number] = all_files_Qtemp_results_RPMs

    coh_qtemp_ana_flags["run_qtemps_section"] = False
# ---------------------------------------------------------------------------------------
        
if coh_qtemp_ana_flags["run_qtemps_section"]:
    for run_num in run_num_list:
        # ---- always reset optional pre-SR variables each iteration ----
        base_dir2 = None
        filter_keywords2 = None
        target_dates_qtemps_RPM2 = None
        all_files_Qtemp_results_RPMs = {}
        low_thermal_pops = False

        if run_num == 5:
            # ---------------- RPM (none) ----------------
            Science_Qubits = [0, 1, 2, 3, 4, 5]
            base_dir = ""
            filter_keywords = []
            outerFolder_qtemps_plots_RR = ""
            outerFolder_qtemps_plots = ""
            target_dates_qtemps_RPM = ""
            print("There is no RPM data for run 5 (SSF only).")

            # ---------------- SSF ----------------
            paths_SSFmethods = paths_SSFmethods_run5
            path_saveplots_fits = path_saveplots_fits_run5
            path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run5

        elif run_num == 6:
            # ---------------- RPM (science run; optional pre-science add-on) ----------------
            Science_Qubits = [0, 4]
            base_dir = base_dir_sciencerun
            filter_keywords = filter_keywords_sciencerun
            outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run6
            outerFolder_qtemps_plots = outerFolder_qtemps_plots_RR_run6
            target_dates_qtemps_RPM = target_dates_qtemps_RPM_sciencerun

            if pre_sciencerun6_data:
                base_dir2 = base_dir_pre_sciencerun
                filter_keywords2 = filter_keywords_presciencerun
                target_dates_qtemps_RPM2 = target_dates_qtemps_RPM_presciencerun

            # ---------------- SSF (science-run paths; optional pre-science add-on) ----------------
            paths_SSFmethods = paths_SSFmethods_SR.copy()
            if pre_sciencerun6_data:
                # include all qubits since pre-SR SSF was taken for all Qs
                Science_Qubits = [0, 1, 2, 3, 4, 5]
                paths_SSFmethods += paths_SSFmethods_preSR

            path_saveplots_fits = path_saveplots_fits_run6
            path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run6


        elif run_num == 7:
            # ---------------- RPM ----------------
            Science_Qubits = [0, 1, 2, 3, 4, 5]
            base_dir = base_dir_run7
            filter_keywords = filter_keywords_run7
            outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run7
            outerFolder_qtemps_plots = outerFolder_qtemps_plots_run7
            target_dates_qtemps_RPM = target_dates_qtemps_RPM_run7

            # ---------------- SSF ----------------
            paths_SSFmethods = paths_SSFmethods_run7
            path_saveplots_fits = path_saveplots_fits_run7
            path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run7


        elif run_num == 8:
            # ---------------- RPM ----------------
            Science_Qubits = [0, 1, 2, 3, 4, 5]
            base_dir = base_dir_run8
            filter_keywords = filter_keywords_run8
            outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run8
            outerFolder_qtemps_plots = outerFolder_qtemps_plots_run8
            target_dates_qtemps_RPM = target_dates_qtemps_RPM_run8

            # ---------------- SSF ----------------
            paths_SSFmethods = paths_SSFmethods_run8
            path_saveplots_fits = path_saveplots_fits_run8
            path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run8

        elif run_num == 9:
            low_thermal_pops = True
            # ---------------- RPM ----------------
            Science_Qubits = [0, 1, 2, 3, 5]
            base_dir = base_dir_run9
            filter_keywords = filter_keywords_run9
            outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run9
            outerFolder_qtemps_plots = outerFolder_qtemps_plots_run9
            target_dates_qtemps_RPM = target_dates_qtemps_RPM_run9

            # ---------------- SSF ----------------
            paths_SSFmethods = paths_SSFmethods_run9
            path_saveplots_fits = path_saveplots_fits_run9
            path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run9

        elif run_num == 9.2:  # this is run 9c
            # low_thermal_pops = True
            # ---------------- RPM ----------------
            Science_Qubits = [0, 1, 2, 3, 5]
            base_dir = base_dir_run9c
            filter_keywords = filter_keywords_run9c
            outerFolder_qtemps_plots_RR = outerFolder_qtemps_plots_RR_run9c
            outerFolder_qtemps_plots = outerFolder_qtemps_plots_run9c
            target_dates_qtemps_RPM = target_dates_qtemps_RPM_run9c

            # ---------------- SSF ----------------
            paths_SSFmethods = paths_SSFmethods_run9c
            path_saveplots_fits = path_saveplots_fits_run9c
            path_saveplots_ssf_qtemps_vsT = path_saveplots_ssf_qtemps_vsT_run9c

        else:
            raise ValueError(f"Unsupported run_num={run_num}")

        # ---------------- pre-fill so dict shape is always stable ----------------
        rpm_temps_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        rpm_temps_errs_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        rpm_Pe_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        rpm_Pe_errs_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]

        ssf_g_temps_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        ssf_g_temp_errs_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        ssf_g_Pe_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        ssf_g_Pe_errs_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]

        ssf_ge_temps_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]
        ssf_ge_temp_errs_by_run[run_num] = [[] for _ in range(tot_num_of_qubits)]

        if run_num != 5:  # no rpm data for run 5
            # ----------- Get Qubit temperature results via RPMs
            RPM_calcs = RPMTempCalcAndPlots(figure_quality, tot_num_of_qubits)
            all_files_Qtemp_results_RPMs = RPM_calcs.run_RPMqtemps(base_dir, target_dates_qtemps_RPM, filter_keywords,
                                                                   fit_saved, signal,
                                                                   run_name, run_num, list_of_all_qubits,
                                                                   tot_num_of_qubits,
                                                                   outerFolder_qtemps_plots_RR, replot_RPMs,
                                                                   get_qtemp_data,
                                                                   get_london_data, figure_quality, save_figsRR,
                                                                   exclude_temp_sweeps,
                                                                   passing_pre_sciencerun_data=False,
                                                                   filter_out_bad_RPM_fits=filter_out_bad_RPM_fits,
                                                                   combine_IQ_signal=rpm_combine_IQ_signal)
            if run_num == 6:
                if pre_sciencerun6_data:
                    all_files_Qtemp_results_RPMs2 = RPM_calcs.run_RPMqtemps(base_dir2, target_dates_qtemps_RPM2,
                                                                            filter_keywords2, fit_saved, signal,
                                                                            run_name, run_num, list_of_all_qubits,
                                                                            tot_num_of_qubits,
                                                                            outerFolder_qtemps_plots_RR, replot_RPMs,
                                                                            get_qtemp_data, get_london_data,
                                                                            figure_quality, save_figsRR,
                                                                            exclude_temp_sweeps,
                                                                            passing_pre_sciencerun_data=True,
                                                                            filter_out_bad_RPM_fits=filter_out_bad_RPM_fits,
                                                                            combine_IQ_signal=rpm_combine_IQ_signal)
                    all_files_Qtemp_results_RPMs += all_files_Qtemp_results_RPMs2

            # ---- ADAPT + STORE (RPM) ----
            rpm_times, rpm_temps, rpm_temps_errs, rpm_Pe, rpm_Pe_errs = combined_studies.rpm_results_to_per_qubit_lists(
                all_files_Qtemp_results_RPMs,
                n_qubits=tot_num_of_qubits)

            all_files_Qtemp_results_RPMs_by_run[run_num] = all_files_Qtemp_results_RPMs
            rpm_temps_by_run[run_num] = rpm_temps
            rpm_temps_errs_by_run[run_num] = rpm_temps_errs
            rpm_Pe_by_run[run_num] = rpm_Pe
            rpm_Pe_errs_by_run[run_num] = rpm_Pe_errs
            rpm_times_by_run[run_num] = rpm_times

        # ----------- Get Qubit temperature results via SSF g-e threshold method and SSF g-state double gaussian threshold method
        SSF_calcs_obj = SSFTempCalcAndPlots(figure_quality, tot_num_of_qubits, run_num, save_figs)
        pairs_info = SSF_calcs_obj.process_ssf_and_qfreq_data_qtemps(Science_Qubits, paths_SSFmethods)

        if use_iminuit_gdoublegauss_ssf:  # Made a special iminuit-based double gaussian fitting function, but for now it is only set up to fit g-state data.
            all_qubit_temps_g, all_qubit_times_g, all_qubit_temps_errs_g, fit_results_g = SSF_calcs_obj.run_ssf_qtemps_iminuit(
                pairs_info, run_num=run_num, limit_temp_k=0.6, do_plots=False, dontuse_midpt_thresh=True, low_leakage_mode = low_thermal_pops, 
                ssf_hist_ylim = ssf_hist_ylim, apply_quality_cuts = filter_out_bad_SSF_qtemp_fits, calc_SNR = calc_SNR_ssfqtemps, calc_e_state_decay = calc_SSF_e_decay)
            
            # ---- STORE RESULTS (SSF g) ----
            ssf_g_temps, ssf_g_temp_errs, ssf_fid_vals, ssf_fid_errs, ssf_snr_vals, ie_new_Pg_vals, ie_new_Pg_errs = combined_studies.ssf_fit_results_to_per_qubit_lists(fit_results_g, n_qubits=tot_num_of_qubits)

            print(f"\nRUN {run_num} accepted SSF counts:")
            for q in range(tot_num_of_qubits):
                print(f"  Q{q + 1}: {len(ssf_g_temps[q])}")

            fit_results_g_by_run[run_num] = fit_results_g
            ssf_g_temps_by_run[run_num] = ssf_g_temps
            ssf_g_temp_errs_by_run[run_num] = ssf_g_temp_errs
            pe_vals, pe_errs = combined_studies.extract_pe_from_fit_results(fit_results_g, tot_num_of_qubits)
            ssf_g_Pe_by_run[run_num] = pe_vals
            ssf_g_Pe_errs_by_run[run_num] = pe_errs
            ssf_fid_values_by_run[run_num] = ssf_fid_vals
            ssf_err_values_by_run[run_num] = ssf_fid_errs
            ssf_snr_by_run[run_num] = ssf_snr_vals
            ie_new_Pg_vals_by_run[run_num] = ie_new_Pg_vals
            ie_new_Pg_errs_by_run[run_num] = ie_new_Pg_errs

        else:  # uses sklearn.mixture.GaussianMixture for double gaussian fitting
            all_qubit_temps_g, all_qubit_times_g, all_qubit_temps_errs_g, fit_results_g = SSF_calcs_obj.run_ssf_qtemps(
                pairs_info, limit_temp_k=1.0, use_gessf_thresh_only=False, fallback_to_threshold=False)
            all_qubit_temps_ge, all_qubit_times_ge, all_qubit_temps_errs_ge, fit_results_ge = SSF_calcs_obj.run_ssf_qtemps(
                pairs_info, limit_temp_k=1.0, use_gessf_thresh_only=True, fallback_to_threshold=False)

            # ---- STORE RESULTS (SSF g) ----
            ssf_g_temps, ssf_g_temp_errs, ssf_fid_vals, ssf_fid_errs, ssf_snr_vals, ie_new_Pg_vals, ie_new_Pg_errs = combined_studies.ssf_fit_results_to_per_qubit_lists(fit_results_g,n_qubits=tot_num_of_qubits)
            print(f"\nRUN {run_num} accepted SSF counts:")
            for q in range(tot_num_of_qubits):
                print(f"  Q{q + 1}: {len(ssf_g_temps[q])}")

            fit_results_g_by_run[run_num] = fit_results_g
            ssf_g_temps_by_run[run_num] = ssf_g_temps
            ssf_g_temp_errs_by_run[run_num] = ssf_g_temp_errs
            pe_vals, pe_errs = combined_studies.extract_pe_from_fit_results(fit_results_g, tot_num_of_qubits)
            ssf_g_Pe_by_run[run_num] = pe_vals
            ssf_g_Pe_errs_by_run[run_num] = pe_errs
            ssf_fid_values_by_run[run_num] = ssf_fid_vals
            ssf_err_values_by_run[run_num] = ssf_fid_errs
            ssf_snr_by_run[run_num] = ssf_snr_vals
            ie_new_Pg_vals_by_run[run_num] = ie_new_Pg_vals
            ie_new_Pg_errs_by_run[run_num] = ie_new_Pg_errs

            # ---- STORE RESULTS (SSF ge) ----
            # Not tested yet
            ssf_ge_temps, ssf_ge_temp_errs, ssf_fid_vals_ge, ssf_fid_errs_ge, ssf_snr_vals_ge, ie_new_Pg_vals_ge, ie_new_Pg_errs_ge = combined_studies.ssf_fit_results_to_per_qubit_lists(fit_results_ge, n_qubits=tot_num_of_qubits)
            ssf_ge_temps_by_run[run_num] = ssf_ge_temps
            ssf_ge_temp_errs_by_run[run_num] = ssf_ge_temp_errs

        # ============================================================
        # Optionally create cached files after processing
        # ============================================================
        if coh_qtemp_ana_flags["create_cached_qtemp_files"] and not coh_qtemp_ana_flags["use_cached_qtemp_files"]:
            ssf_cache_path, rpm_cache_path = combined_studies.save_processed_ssf_rpm_inputs(
                fit_results_g, all_files_Qtemp_results_RPMs, save_dir=cache_dir, tag=f"run{run_num}_processed")
        # ----------------------------------------------------------------

if coh_qtemp_ana_flags["load_mcp1_temps"]:
    mcp1_base_dir = "/exp/cosmiq/data/QUIET/MCP1_Grafana_Temperatures/During_AB-Paper_Data-Taking/"
    mcp1_csv_path = combined_studies.get_single_mcp1_csv_for_run(run_num_list[0], mcp1_base_dir, run6_subfolder=run6_subfolder, run8_temp_sweep = run8_temp_sweep)
    print("Using MCP1 CSV:", mcp1_csv_path)
    mcp1_dates, mcp1_temps, _ = combined_studies.load_mixing_chamber_csv(mcp1_csv_path, restrict_time=restrict_time,
                                                            start_time=start_time, end_time=end_time)

if coh_qtemp_ana_flags["run_coherence_section"]:
    for run_num in run_num_list:
        coherence_cache_dir = f"/home/acolonce/Documents/analysis/cached_processed_data/run{run_num}/"
        (   date_times_res_spec,
            res_freqs,
            date_times_q_spec,
            q_freqs,
            qspec_fit_err,
            date_times_t1,
            t1_vals,
            t1_fit_err,
            res_lengths_t1,
            date_times_t2r,
            t2r_vals,
            t2r_fit_err,
            date_times_t2e,
            t2e_vals,
            t2e_fit_err,
            I_per_pt_errs, #T1
            Q_per_pt_errs, #T1
        ) = combined_studies.get_or_create_processed_coherence_inputs(
            run_number=run_num,
            run_name=run_name,
            coherence_cache_dir=coherence_cache_dir,
            coh_qtemp_ana_flags=coh_qtemp_ana_flags,
            figure_quality=figure_quality,
            final_figure_quality=final_figure_quality,
            tot_num_of_qubits=tot_num_of_qubits,
            top_folder_dates=top_folder_dates,
            save_figs=save_figs,
            fit_saved=fit_saved,
            signal=signal,
            FRIDGE=FRIDGE,
            data_path=data_path,
            plots_path=plots_path,
            per_pt_errs_t1=per_pt_errs_t1,
            process_shots_t1ge=process_shots_t1ge)
        
        # ---------------- Another option: store results per run for downstream plotting ----------------
        t1_vals_by_run[run_num] = t1_vals
        t1_errs_by_run[run_num] = t1_fit_err
        #t1_res_lengths_by_run[run_num] = res_lengths_t1
        date_times_t1_by_run[run_num] = date_times_t1

        t2r_vals_by_run[run_num] = t2r_vals
        t2r_errs_by_run[run_num] = t2r_fit_err

        t2e_vals_by_run[run_num] = t2e_vals
        t2e_errs_by_run[run_num] = t2e_fit_err

        qfreq_vals_by_run[run_num] = q_freqs
        qfreq_errs_by_run[run_num] = qspec_fit_err

        resfreq_vals_by_run[run_num] = res_freqs

if coh_qtemp_ana_flags["SSF_lims_per_scan_viaSSF"]:

    comb_plots_path = "/home/acolonce/Documents/analysis/combined_qtemps/qtemps_and_coherence/SSF_limitations_calcs"

    ssf_limitations_results_by_run = (
        combined_studies.run_ssf_limitations_per_scan_for_runs(
            run_num_list=run_num_list,
            fit_results_g_by_run=fit_results_g_by_run,
            date_times_t1_by_run=date_times_t1_by_run,
            t1_vals_by_run=t1_vals_by_run,
            t1_errs_by_run=t1_errs_by_run,
            t1_res_lengths_by_run=t1_res_lengths_by_run,
            out_dir=comb_plots_path,
            n_qubits=tot_num_of_qubits,
            sensitive_fraction=0.5,
            t1_match_max_dt_s=10.0, #change to 600 for run 6 science run
            make_ssf_pe_table=True,
            make_rpm_pe_table=coh_qtemp_ana_flags["SSF_lims_per_scan_viaRPM"],
            all_files_Qtemp_results_RPMs_by_run=all_files_Qtemp_results_RPMs_by_run,
            rpm_match_max_dt_s=10, #change to 600 for run 6 science run
        )
    )

if coh_qtemp_ana_flags["plot_qtemps_t1_ftemps_qfreq"]:
    if len(run_num_list) != 1:
        raise ValueError(f"Expected exactly 1 run in 'run_num_list', but got {len(run_num_list)}. "
                         "This plotting section is only set up to process one run at a time at the moment.")
        # This section is set up to process multiple runs, but I have not updated this plotting function to handle more than one.
    comb_plots_path = "/home/acolonce/Documents/analysis/combined_qtemps/qtemps_and_coherence/"

    combined_studies.plot_qtemps_and_coherence_res(
        comb_plots_path,
        # all_qubit_temperatures_ssf_g=all_qubit_temps_g,
        # all_qubit_timestamps_ssf_g=all_qubit_times_g,
        all_files_Qtemp_results_RPMs= all_files_Qtemp_results_RPMs,
        fridge_temps=mcp1_temps,
        fridge_dates=mcp1_dates,
        t1_vals=t1_vals,
        t1_dates=date_times_t1,
        t1_fit_err = t1_fit_err,
        qfreqs_vals=q_freqs,
        qfreqs_dates=date_times_q_spec,
        qfreqs_errs=qspec_fit_err, # new
        # resfreqs_vals= res_freqs,
        # resfreqs_dates= date_times_res_spec,
        # t2r_vals=t2r_vals,
        # t2r_dates=date_times_t2r,
        # t2r_errs=t2r_fit_err, # new
        # t2e_vals=t2e_vals,
        # t2e_dates=date_times_t2e,
        restrict_time_xaxis=restrict_time,
        start_time=start_time,
        end_time=end_time,
        plot_extra_event_lines=False,
        run_num = f"{run_num_list[0]}",
        qubits_to_plot = [0],
        fig_width=30,
        fig_height = 14,
        fnt_sz = 26,
    )
if coh_qtemp_ana_flags["plot_RPM_qtemps_qfreq_fridge_only"]:
    if len(run_num_list) != 1:
        raise ValueError(f"Expected exactly 1 run in 'run_num_list', but got {len(run_num_list)}. "
                         "This plotting section is only set up to process one run at a time at the moment.")
        # This section is set up to process multiple runs, but I have not updated this plotting function to handle more than one.
    comb_plots_path = "/home/acolonce/Documents/analysis/combined_qtemps/qtemps_and_coherence/"

    combined_studies.plot_rpm_qtemp_qfreq_fridge_only(
    out_dir=comb_plots_path,
    all_files_Qtemp_results_RPMs=all_files_Qtemp_results_RPMs,
    fridge_temps=mcp1_temps,
    fridge_dates=mcp1_dates,
    run_num=f"{run_num_list[0]}",
    fridge_label="MCP1 Temp (mK)",
    restrict_time_xaxis=restrict_time,
    start_time=start_time,
    end_time=end_time
)

if coh_qtemp_ana_flags["plot_SSF_qtemps_qfreq_fridge_only"]:
    if len(run_num_list) != 1:
        raise ValueError(f"Expected exactly 1 run in 'run_num_list', but got {len(run_num_list)}. "
                         "This plotting section is only set up to process one run at a time at the moment.")
        # This section is set up to process multiple runs, but I have not updated this plotting function to handle more than one.
    comb_plots_path = "/home/acolonce/Documents/analysis/combined_qtemps/qtemps_and_coherence/"

    combined_studies.plot_ssf_qtemp_qfreq_fridge_only(
        out_dir=comb_plots_path,
        all_qubit_ssf_results=fit_results_g,
        fridge_temps=mcp1_temps,
        fridge_dates=mcp1_dates,
        run_num=run_num_list[0],
        fridge_label="MCP1 Temp (mK)",
        restrict_time_xaxis=restrict_time,
        start_time=start_time,
        end_time=end_time,
    )