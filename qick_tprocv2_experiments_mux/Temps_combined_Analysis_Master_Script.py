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
import datetime
import ast
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import math
import h5py
from expt_config import expt_cfg, list_of_all_qubits, FRIDGE
from analysis_021_plot_allRR_noqick import PlotRR_noQick
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime
from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_007_T2R_vs_time_plots import T2rVsTime
from analysis_008_T2E_vs_time_plots import T2eVsTime
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
run_num = 6
run_name = f'run{run_num}/6transmon' # this is for temps analysis, for coherence analysis it's defined in its respective section
signal = 'None' # Do not change
final_figure_quality = 200 # plot quality
plot_ssf_gef = False # Do you want to re-plot g-e-f SSF data and save the plots?
replot_RPMs = False # Do you want to re-plot rabi population measurements from RR data but not extract temps? Only make plots
save_figsRR = False # Do you want to save (or not save) the RR RPM plots?
save_figs = False # To be used in general for any function or class to save (or not save) plots.
save_figs_SSF = False # Do you want to save gaussian fit plots while calculating ssf qtemps? iminuit case only
fit_saved = False # Not used here, set to false.
exclude_temp_sweeps = True # Do you want to exclude the folders that contain data taken during the heater temperature sweep?
filter_out_bad_amp_fits = True # filter out bad rpm fits? this doesn't work perfect but helps a bit
get_qtemp_data = True # Do you want to calculate RPM qubit temperatures? This returns RPM qubit temperatures and qubit freqs for specified dates.
get_london_data = False # This returns RPM qubit temperatures, resonator freqs, and qubit freqs for specified dates. Designed for London Penetration analysis.

pre_sciencerun6_data = True # Do you also want to incorporate the run 6 pre-science run data? This only applies when run_num = 6

use_iminuit_gdoublegauss_ssf = True # do you want to fit the g-state to a double gaussian using iminuit? The default is GMM instead

rpm_combine_IQ_signal = False # uses ssf angle to rotate rabi population measurement data into a combined IQ signal lying on the same axis

figure_quality = 200
theta = 0
threshold = 0
tot_num_of_qubits = 6 # Total number of qubits currently at QUIET

# What method or methods do you want to use to calculate qubit temperatures?
qtemp_method_flags = {"Qtemps_viaRPM": False, "Qtemps_viaSSF_ge_thresh": False, "Qtemps_viaSSF_gmeans_thresh": False, "Qtemps_viaSSF_with_fallback": False,
                      "combined_studies_qtemps": False}

# What analysis plots do you want to make?
analysis_flags = {"Qtemps_vs_time_viaSSF": False,  "Qtemps_vs_time_viaRPM": False, "Threshold_Check_Qtemps_viaSSF": False, "ge_thresh_check_ssf": False,
                  "Qtemps_hists_viaRPM": False, "Qtemps_hists_viaSSF": False, "Pe_vs_time_viaRPM": False, "qtemps_Pe_vs_time_viaRPM": False, "qtemps_Pe_gefreq_vs_time_viaRPM": False}

# For combined analysis (SSF qtemps + RPM qtemps)
comb_analysis_flags = {"Qtemps_vs_time_comb_separate_plts": False,"Qtemps_vs_time_comb_single_plt": False, "Pe_vs_time_comb_separate_plts": False,
                       "Pe_vs_time_comb_single_plt": False}

# For London Penetration Depth analysis
london_flags = {"get_qfreqs_resfreqs_qtemps": False}

# For double-gaussian SSF analysis using alternative methods
alt_ssf_analysis_flags = {"jupyter_method_Arianna": False, "iminuit_method": True}

# For coherence-qubit temps combined analysis
coh_qtemp_ana_flags = {"load_rpm_qtemps": False, "load_ssf_qtemps": False, "load_mcp1_temps": False, "load_coherence_res": False, "plot_qtemps_t1_ftemps_qfreq": False}
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

# For data before heater temperature steps, run 6
# target_dates_qtemps_RPM_sciencerun = [
#     "2025-04-16",
#     "2025-04-17",
#     "2025-04-18",
#     "2025-04-19",
#     "2025-04-20",
#     "2025-04-21", #starts source on (Co)
#     "2025-04-22",
#     "2025-04-23", #switched source (to Cs)
#     "2025-04-24",
#     "2025-04-25",
#     "2025-04-26",
#     "2025-04-27",
#     "2025-04-28", #Cs source moved closer
#     "2025-04-29",
#     "2025-04-30",
#     "2025-05-01",
#     "2025-05-02",
#     "2025-05-03",
#     "2025-05-04", # Cs source removed. No sources in Cleanroom.
#     "2025-05-05",
#     "2025-05-06"]

# For data during Heater temperature steps, run 6 (20mK to 160mK)
# target_dates_qtemps_RPM_sciencerun = ["2025-05-08", "2025-05-09", "2025-05-10", "2025-05-11", "2025-05-12", "2025-05-13", "2025-05-14"]

# For pre-science-run data
target_dates_qtemps_RPM_presciencerun = ['2025-04-11', '2025-04-12']

# Base path of where the data is stored up to the Study Name (TLS_Comprehensive_Study or ef_studies_pre_science_run)
base_dir_sciencerun = "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study" # for QUIET run 6 science run data
base_dir_pre_sciencerun = "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ef_studies_pre_science_run" # for run 6 pre-science run data

# To re-make and save RPM RR plots
# outerFolder_qtemps_plots_RR_run6 = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/Plots/Plots_RR"
outerFolder_qtemps_plots_RR_run6 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/RPM_QTemps/Plots_RR"
# For RPM Analysis
# outerFolder_qtemps_plots_run6 = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/Plots" # Inside each analysis function, a subfolder will be defined
outerFolder_qtemps_plots_run6 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/RPM_QTemps/hists" # Inside each analysis function, a subfolder will be defined
# For London Penetration Depth analysis, which is done on run 6 temperature sweep data. This is where we save the plots:
outerFolder_london_path = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/London_Penetration_Depth"

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
base_dir_run7 = "/data/QICK_data/run7/6transmon/round_robin_benchmark"

# for 24hr AB data
target_dates_qtemps_RPM_run7 = ["2025-07-19", "2025-07-20"]

# To re-make and save RPM RR plots
outerFolder_qtemps_plots_RR_run7 = "/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_paper_data/benchmark_analysis_plots/RPM_RR_plots"
#
# For RPM Analysis
outerFolder_qtemps_plots_run7 = "/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_paper_data/benchmark_analysis_plots/RPM_analysis" # Inside each analysis function, a subfolder will be defined

# Substudy name on the file path, doesn't have to be exact, it will look for these key terms in the name
filter_keywords_run7 = ['AB_paper_data']

#-----------------------------------------------------------------------run 8------------------------------------------------------------
# Base path of where the data is stored up to the Study Name (round_robin_benchmark)
base_dir_run8 = "/data/QICK_data/run8/6transmon/round_robin" # up to study name

# Arianna's local analysis
# base_dir_run8 = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8" # up to study name

# all AB paper data: (specify up to the day only)
target_dates_qtemps_RPM_run8 = [
  "2025-10-19",
  "2025-10-20",
  "2025-10-23",
  "2025-10-24",
  "2025-10-27",
  "2025-10-28",
  "2025-10-29",
  "2025-10-31",
  "2025-11-01"
]

# for Arianna's local analysis: (specify date folder)
# target_dates_qtemps_RPM_run8 = ["2025-10-19_20-25-18"]

# To re-make and save RPM RR plots
# outerFolder_qtemps_plots_RR_run8 = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\AB_Paper_Data_24hrs\replotted_RR_data\rpm"
    # r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional\replotted_RR_data\rpm"
    #"/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/Plots/run8_analysis/benchmark_analysis_plots/RPM_RR_plots"
outerFolder_qtemps_plots_RR_run8 = "/data/QICK_data/run8/6transmon/replotted_RR_data/rabi_pop_meas"

# For RPM Analysis
# outerFolder_qtemps_plots_run8 = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\AB_Paper_Data_24hrs/rabi_pop_meas_analysis/" # Inside each analysis function, a subfolder will be defined
    # r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional/rabi_pop_meas_analysis/"
    #"/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/Plots/run8_analysis/benchmark_analysis_plots/Qtemps_RPMmethod"
outerFolder_qtemps_plots_run8 = "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/benchmark_analysis_plots/Qtemps_RPMmethod"

# Substudy name on the file path, doesn't have to be exact, it will look for these key terms in the name. THese are substudies.
# For all AB paper data:
filter_keywords_run8 = ["AB_Paper_Data_24hrs", "ABpaperdata2ndbatch_21dB_DACatten_Q1to5", "ABpaperdata3rdbatch_21dB_DACatten_Q1to5_not1shots",
                        "ABpaperdata3rdbatch_21dB_DACatten_Q1to5", "ABpaperdata3rdbatch_21dB_DACatten_Q1to5_t1shots_optional",
                        "ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional", "ABpaperdata3rdbatch_21dB_DACatten_Q6_t1shots_optional",
                        "ABpaperdata_21dB_DACatten_Q1to6_t1shots_optional_newopt", "19dB_DAC_testdata_allQs"]
# filter_keywords_run8 = ["AB_paper_datadump_for_analysis"]

# # For run 8 temp sweep data:
# filter_keywords_run8 = ["temperature_sweep_run8_25dBDAC_onechan_day1", "temperature_sweep_run8_25dBDAC_onechan_day2",
#                         "temperature_sweep_run8_25dBDAC_onechan_day3", "temp_sweep_run8_25dBDAC_onechan_day4_175mK"]

# For Arianna's local analysis
#filter_keywords_run8 = ['ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional']
# filter_keywords_run8 = ["AB_Paper_Data_24hrs"]
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

elif run_num == 5: # No RPM data for this run, only ssf analysis can be done
    Science_Qubits = [0, 1, 2, 3, 4, 5]
    base_dir = ""
    filter_keywords = []
    outerFolder_qtemps_plots_RR = ""
    outerFolder_qtemps_plots = ""
    target_dates_qtemps_RPM = ""
else:
    raise ValueError("You must choose run_num = 6, 7 or 8. Otherwise, define a section for your run of interest.")

#-------------------------------------- For qubit temperature calculations via SSF methods (double gaussian over g-state data and double gaussian over g and e-state data ---------------------------------------------
# Note: you must write paths in this form: "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_03-03-40"
# If you want to loop through all the data corresponding to 1 day, you must list all the paths for that day. This method does not accept just a single date as a path.
# I have included examples for how the paths are structured for each run

# ------------------------------------------------------------------------------------------------run 4--------------------------------------------------------------------------------------------------------------
paths_SSFmethods_run4 = ["/exp/cosmiq/data/QUIET/QICK_data/run4/6transmon/folders_with_SSF_data_entire_run4/ssf_data_and_readoutopt/2024-11-13_08-23-41"] # this is the only folder with "usable" data for this run
path_saveplots_fits_run4 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/QTemperatures/Plots/Qtemps_SSFmethod/Gaussian_Fits_run4" # where to save ssf plots to check gaussian fits
# path_saveplots_fits = f"/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/RR_metrics/Plots/Qtemps_SSFmethod/geSSF_Fits" # to check g-e SSF Double Gaussian Fits and g-e threshold
path_saveplots_ssf_qtemps_vsT_run4 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/QTemperatures/Plots/Qtemps_SSFmethod/Qtemps_vs_Time_run4" # to save qubit temps vs time via ssf methods

# ------------------------------------------------------------------------------------------------run 5----------------------------------------------------------------------------------------------------------
# All data
paths_SSFmethods_run5 = [
    "/exp/cosmiq/data/QUIET/QICK_data/run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-09",
    "/exp/cosmiq/data/QUIET/QICK_data/run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-10",
    "/exp/cosmiq/data/QUIET/QICK_data/run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-11",
    "/exp/cosmiq/data/QUIET/QICK_data/run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-12",
    "/exp/cosmiq/data/QUIET/QICK_data/run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-13",
    "/exp/cosmiq/data/QUIET/QICK_data/run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-14",
    "/exp/cosmiq/data/QUIET/QICK_data/run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-15",
    "/exp/cosmiq/data/QUIET/QICK_data/run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-16",
    "/exp/cosmiq/data/QUIET/QICK_data/run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-17",
    "/exp/cosmiq/data/QUIET/QICK_data/run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-18",
    "/exp/cosmiq/data/QUIET/QICK_data/run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-19",
    "/exp/cosmiq/data/QUIET/QICK_data/run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-20"
]

path_saveplots_fits_run5 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/Qtemps_SSFmethod/GaussFits_r5" # where to save ssf plots to check gaussian fits
path_saveplots_ssf_qtemps_vsT_run5 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/Qtemps_SSFmethod/Qtemps_vs_Time_run5" # to save qubit temps vs time via ssf methods

# ------------------------------------------------------------------------------------------------run 6------------------------------------------------------------------------------------------------------------
# Science-Run Data
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
    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy6/2025-05-08_11-53-46"]

    # "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_post_temperature_sweep_substudy1/",
    # "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_detuning_24MHz_Q1_substudy1/",
    #"/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_detuning_17MHz_Q1_substudy1/"

# All pre-Science-Run Data
paths_SSFmethods_preSR = ["/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-21",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-22",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-23",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-24",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-26",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-28",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-03-01",
                        "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ge_round_robin_presciencerun_data/ge_coherence_data/2025-03-02"]

path_saveplots_fits_run6 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/Qtemps_SSFmethod/GaussFits_r6" # where to save ssf plots to check gaussian fits
path_saveplots_ssf_qtemps_vsT_run6 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/Qtemps_SSFmethod/Qtemps_vs_Time_run6" # to save qubit temps vs time via ssf methods

# ----------------------------------------------------------------------------------------------run 7----------------------------------------------------------------------------------------------------------
paths_SSFmethods_run7 = [
  "/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_paper_data/2025-07-19_08-34-39",
  "/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_paper_data/2025-07-19_16-16-14",
  "/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_paper_data/2025-07-19_16-56-45",
  "/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_paper_data/2025-07-19_23-11-39",
  "/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_paper_data/2025-07-20_06-33-03"
]

path_saveplots_fits_run7 = "/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_paper_data/ssf_qtemps_analysis/GaussFits_r7" # where to save ssf plots to check gaussian fits
path_saveplots_ssf_qtemps_vsT_run7 = "/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_paper_data/ssf_qtemps_analysis/Qtemps_vs_Time_run7" # to save qubit temps vs time via ssf methods

# ----------------------------------------------------------------------------------------------run 8----------------------------------------------------------------------------------------------------------
# Arianna's local analysis:
# paths_SSFmethods_run8 = [r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional\2025-10-27_22-04-57"]
# path_saveplots_fits_run8 = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional\qtemps_ssf_gaussfits" # where to save ssf plots to check gaussian fits
# path_saveplots_ssf_qtemps_vsT_run8 = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional\qtemps_ssf_analysis\Qtemps_vs_Time_run8" # to save qubit temps vs time via ssf methods

# on qubituser-daq01:
# All AB paper data
paths_SSFmethods_run8 = [
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-19_11-09-32",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-19_12-05-25",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-19_19-43-00",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-19_20-25-18",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-20_12-10-19",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-23_00-49-28",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-23_14-47-22",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-24_01-41-30",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-24_13-58-37",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-27_14-15-40",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-27_14-24-29",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-27_22-04-57",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-28_21-57-47",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-29_18-38-25",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-29_23-48-45",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-31_01-54-57",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-10-31_20-40-11",
  "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/2025-11-01_12-54-55"
]

path_saveplots_fits_run8 ="/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/qtemps_ssf_gaussfits" # where to save ssf plots to check gaussian fits
path_saveplots_ssf_qtemps_vsT_run8 = "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_for_analysis/qtemps_ssf_analysis" # to save qubit temps vs time via ssf methods

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
    raise ValueError("You must choose run_num = 4, 5, 6, 7 or 8. Otherwise, define a section for your run of interest.")

#-------------------------------------------------------- For coherence data -------------------------------------------
if run_num == 8:
    # on Arianna's local pc
    run_name_coh = "run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional"
    data_path = fr"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\{run_name}"
    plots_path = data_path
    top_folder_dates = ["2025-10-27_22-04-57"]

    # run_name_coh = 'run8/6transmon/round_robin/temperature_sweep_qubit_data'  # 'run8/6transmon/round_robin/AB_paper_datadump_for_analysis'
    # data_path = f'/data/QICK_data/{run_name}'
    # plots_path = data_path

    # all of run 8 thus far
    # top_folder_dates = [
    #     "2025-10-19_11-09-32",
    #     "2025-10-19_12-05-25",
    #     "2025-10-19_19-43-00",
    #     "2025-10-19_20-25-18",
    #     "2025-10-20_12-10-19",
    #     "2025-10-23_00-49-28",
    #     "2025-10-23_14-47-22",
    #     "2025-10-24_01-41-30",
    #     "2025-10-24_13-58-37",
    #     "2025-10-27_14-15-40",
    #     "2025-10-27_14-24-29",
    #     "2025-10-27_22-04-57",
    #     "2025-10-28_21-57-47",
    #     "2025-10-29_18-38-25",
    #     "2025-10-29_23-48-45",
    #     "2025-10-31_01-54-57",
    #     "2025-10-31_20-40-11",
    #     "2025-11-01_12-54-55"
    # ]

    # # when saving t1 shots + avg IQ data started
    # top_folder_dates = [
    #     "2025-10-24_13-58-37",
    #     "2025-10-27_14-15-40",
    #     "2025-10-27_14-24-29",
    #     "2025-10-27_22-04-57",
    #     "2025-10-28_21-57-47",
    #     "2025-10-29_18-38-25",
    #     "2025-10-29_23-48-45",
    #     "2025-10-31_01-54-57",
    #     "2025-10-31_20-40-11",
    #     "2025-11-01_12-54-55"]

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
elif coh_qtemp_ana_flags["load_coherence_res"] and run_num != 8:
    raise ValueError("You must choose run_num = 8 to load coherence data. Otherwise, define a section for your run of interest.")

############################################################################### Qubit temperature calculations via rabi population measurements #####################################################
if qtemp_method_flags["Qtemps_viaRPM"]:
    RPM_calcs = RPMTempCalcAndPlots(figure_quality, tot_num_of_qubits)
    combined_qtemp_data = RPM_calcs.run_RPMqtemps(base_dir, target_dates_qtemps_RPM, filter_keywords, fit_saved, signal, run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                            outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, get_london_data, figure_quality, save_figsRR, exclude_temp_sweeps, filter_out_bad_amp_fits = filter_out_bad_amp_fits,
                                                  passing_pre_sciencerun_data = False, combine_IQ_signal = rpm_combine_IQ_signal)

    if run_num == 6:
        if pre_sciencerun6_data:
            combined_qtemp_data2 = RPM_calcs.run_RPMqtemps(base_dir2, target_dates_qtemps_RPM2, filter_keywords2, fit_saved, signal, run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                                                          outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, get_london_data, figure_quality, save_figsRR, exclude_temp_sweeps, filter_out_bad_amp_fits = filter_out_bad_amp_fits,
                                                           passing_pre_sciencerun_data = True, combine_IQ_signal = rpm_combine_IQ_signal)

            combined_qtemp_data += combined_qtemp_data2

    del RPM_calcs # to free up memory
    #----------------------------------------------------------------------- RPM Analysis -------------------------------------------------------------------------
    # These are not used in the definitions that follow, are just needed to re-initialize the class
    outerFolder = ""
    outerFolder_qtemps_data = ""
    date_string = ""
    RPM_plotter = PlotRR_noQick(date_string, figure_quality, save_figs, fit_saved, signal, run_name, tot_num_of_qubits, outerFolder, outerFolder_qtemps_plots, outerFolder_qtemps_data, run_num, filter_out_bad_amp_fits)

    if analysis_flags["Qtemps_vs_time_viaRPM"]:
        #------------------------------------------------------------------- Qubit temperatures vs time via RPMs ----------------------------------------------------
        RPM_plotter.plot_qubit_temperatures_vs_time_RPMs(combined_qtemp_data, num_qubits=tot_num_of_qubits, yaxis_min = 10, yaxis_max = 750, rel_err_cutoff = None, restrict_time_xaxis = False,
                                                         plot_extra_event_lines = False, rad_events_plot_lines = False, plot_error_bars = True, fit_to_line=False, average_per_heater_step=False)

    if analysis_flags["Qtemps_hists_viaRPM"]:
        #----------------------------------------------------------------- Histograms of Qubit temperatures (via RPMs) -----------------------------------------------
        RPM_plotter.plot_qubit_temperature_histograms_RPMs(combined_qtemp_data, num_qubits=6, rel_err_cutoff = None)

    if analysis_flags["Pe_vs_time_viaRPM"]:
        #------------------------------------------------------------ Excited state populations (P_e) vs time (via RPMs) ----------------------------------------
        RPM_plotter.plot_qubit_pe_vs_time_RPMs(combined_qtemp_data)

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
            all_qubit_temps, all_qubit_times, all_qubit_temps_errs, fit_results = SSF_calcs_obj.run_ssf_qtemps_iminuit(pairs_info, run_num=run_num, limit_temp_k=1.0, do_plots=save_figs_SSF, save_figs_path = path_saveplots_fits)
        else:
            all_qubit_temps, all_qubit_times, all_qubit_temps_errs, fit_results  = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=1.0, use_gessf_thresh_only = False, fallback_to_threshold = False)
    elif qtemp_method_flags["Qtemps_viaSSF_ge_thresh"]: # Fits both GROUND STATE and PREPARED EXCITED STATE SSF data to a double gaussian ; threshold = midpoint of the two gaussian means
        all_qubit_temps, all_qubit_times, all_qubit_temps_errs, fit_results  = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=1.0, use_gessf_thresh_only = True, fallback_to_threshold = False)
    elif qtemp_method_flags["Qtemps_viaSSF_with_fallback"]: # Uses Default method and if the fit fails it falls back to the method that fits both GROUND STATE and PREPARED EXCITED STATE SSF data to a double gaussian
        all_qubit_temps, all_qubit_times, all_qubit_temps_errs, fit_results  = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=1.0, use_gessf_thresh_only = False, fallback_to_threshold = True)

    #------------------------------------------------------------------ Temperatures vs Time Scatter Plot --------------------------------------------------------------------------
    if analysis_flags["Qtemps_vs_time_viaSSF"]:
        SSF_calcs_obj.plot_qubit_temperatures_vs_time_ssf(all_qubit_temps, all_qubit_times, all_qubit_temps_errs, path_saveplots_ssf_qtemps_vsT, plot_error_bars = True)
    #--------------------------------------------------------------------------- SSF Temperature Histograms --------------------------------------------------------------------------------------
    if analysis_flags["Qtemps_hists_viaSSF"]:
        SSF_calcs_obj.plot_all_qubits_hist_ssf(all_qubit_temps, all_qubit_temps_errs, path_saveplots_ssf_qtemps_vsT, bins=45)
    #------------------------------------------------------------ Check General SSF Double Gaussian Fits and g-e threshold ---------------------------------------------------------
    if analysis_flags["ge_thresh_check_ssf"]:
        thresh_results = SSF_calcs_obj.plot_ssf_ge_thresh(pairs_info=pairs_info, plotting_path=path_saveplots_fits)

    #---------------------------------------------------- Check population threshold for Qubit Temperature Calcs via both SSF methods ----------------------------------------------
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
                                                                                                                do_plots = True, save_figs_path = made_on_folder)

    # This is for plotting outside of run_ssf_qtemps_iminuit(); when do_plots = False instead of True
    # for q_key, recs in fit_results.items():
    #     qubit_folder = os.path.join(path_saveplots_fits, "Iminuit_method")
    #     os.makedirs(qubit_folder, exist_ok=True)
    #     # Make a date‐stamped subfolder
    #     date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    #     made_on_folder = os.path.join(qubit_folder, f"made_on_{date_str}")
    #     os.makedirs(made_on_folder, exist_ok=True)
    #
    #     for rec in recs:
    #         # plots the double-gaussian fits on the ground state data and shows where the population threshold was set (midpoint of the two means)
    #         SSF_calcs_obj.plot_gaussians_qtemps(q_key, made_on_folder, rec["ig_new"], rec["ground_data"],
    #                                             rec["excited_data"], rec["ground_gaussian"],
    #                                             rec["excited_gaussian"], rec["pop_threshold"],
    #                                             rec["temperature_mK"], rec["dataset"], rec["weights"],
    #                                             rec["sigmas"], rec["means"])


################################################### Combined Qubit Temperature Analyses ##########################################################
if qtemp_method_flags["combined_studies_qtemps"]:
    # ----------- Get Qubit temperature results via RPMs
    RPM_calcs = RPMTempCalcAndPlots(figure_quality, tot_num_of_qubits)
    all_files_Qtemp_results_RPMs = RPM_calcs.run_RPMqtemps(base_dir, target_dates_qtemps_RPM, filter_keywords, fit_saved, signal,
                                                  run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                                                  outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data,
                                                  get_london_data, figure_quality, save_figsRR, exclude_temp_sweeps,
                                                  passing_pre_sciencerun_data=False, filter_out_bad_amp_fits = filter_out_bad_amp_fits,
                                                combine_IQ_signal = rpm_combine_IQ_signal)
    if run_num == 6:
        if pre_sciencerun6_data:
            all_files_Qtemp_results_RPMs2 = RPM_calcs.run_RPMqtemps(base_dir2, target_dates_qtemps_RPM2, filter_keywords2, fit_saved, signal, run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                                                          outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, get_london_data, figure_quality, save_figsRR, exclude_temp_sweeps,
                                                                    passing_pre_sciencerun_data = True, filter_out_bad_amp_fits = filter_out_bad_amp_fits, combine_IQ_signal = rpm_combine_IQ_signal)
            all_files_Qtemp_results_RPMs += all_files_Qtemp_results_RPMs2

    # ----------- Get Qubit temperature results via SSF g-e threshold method and SSF g-state double gaussian threshold method
    SSF_calcs_obj = SSFTempCalcAndPlots(figure_quality, tot_num_of_qubits, run_num, save_figs)
    pairs_info = SSF_calcs_obj.process_ssf_and_qfreq_data_qtemps(Science_Qubits, paths_SSFmethods)

    if use_iminuit_gdoublegauss_ssf: # Made a special iminuit-based double gaussian fitting function, but for now it is only set up to fit g-state data.
        all_qubit_temps_g, all_qubit_times_g, all_qubit_temps_errs_g, fit_results_g = SSF_calcs_obj.run_ssf_qtemps_iminuit(pairs_info, run_num=run_num, limit_temp_k=0.6,
            do_plots=False)

    else: # uses sklearn.mixture.GaussianMixture for double gaussian fitting
        all_qubit_temps_g, all_qubit_times_g, all_qubit_temps_errs_g, fit_results_g  = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=1.0, use_gessf_thresh_only = False, fallback_to_threshold = False)
        all_qubit_temps_ge, all_qubit_times_ge, all_qubit_temps_errs_ge, fit_results_ge = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=1.0, use_gessf_thresh_only=True, fallback_to_threshold=False)

    #------------ Initialize class for combined qubit temps analysis ------------------
    combined_studies = combined_Qtemp_studies(figure_quality, tot_num_of_qubits)

    # ------------ Qubit temperatures vs Time using all three methods
    if comb_analysis_flags["Qtemps_vs_time_comb_separate_plts"]:
        # This func has only been set up to work for 2 qubits.
        # Plots two rows (one for each qubit) and 3 columns (one for each method)
        combined_studies.Qtemps_vs_time_comb_methods_3col(all_qubit_temps_g, all_qubit_times_g, all_qubit_temps_errs_g, all_qubit_temps_ge, all_qubit_times_ge, all_qubit_temps_errs_ge,
                                                     outerFolder_qtemps_plots, all_files_Qtemp_results_RPMs, restrict_time_xaxis = False, plot_extra_event_lines = False,
                                                     rad_events_plot_lines = False, plot_error_bars = True)
    if comb_analysis_flags["Qtemps_vs_time_comb_single_plt"]:
        # This one works for multiple qubits (has been improved)
        # Makes 1 subplot per qubit (and all methods in a single plot). Note: I removed the ge SSF method from being plotted since we haven't been using that one lately.
        # Plots error bars always, unless you pass None instead of all_qubit_temps_errs_g.
        combined_studies.Qtemps_vs_time_comb_allQs_1col(all_qubit_temps_g, all_qubit_times_g, outerFolder_qtemps_plots,
                                                     all_files_Qtemp_results_RPMs, all_qubit_temps_errs_g, restrict_time_yaxis = True, ylims = [50,120],
                                                        rad_events_plot_lines = False, qubits_to_plot = [0,1],
                                                        plot_rpm_I_only=True, plot_rpm_Q_only=True)

    #----------- Thermal Populations vs Time using all three methods
    if comb_analysis_flags["Pe_vs_time_comb_separate_plts"]:
        # Plots two rows (one for each qubit) and 3 columns (one for each method)
        combined_studies.Pe_vs_time_comb_methods(all_files_Qtemp_results_RPMs, fit_results_g, fit_results_ge, outerFolder_qtemps_plots,
                                                 restrict_time_xaxis = False, plot_extra_event_lines = False, rad_events_plot_lines = False)
    if comb_analysis_flags["Pe_vs_time_comb_single_plt"]:
        # Plots two rows (one for each qubit) and 1 column (all methods in a single plot)
        combined_studies.Pe_vs_time_comb_2subplts(all_files_Qtemp_results_RPMs, fit_results_g, fit_results_ge, outerFolder_qtemps_plots,
                                     restrict_time_xaxis = False, plot_extra_event_lines = False, rad_events_plot_lines = False)

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
mcp_dates = None
mcp_temps = None
date_times_t1 = None
t1_vals = None
date_times_q_spec = None
q_freqs = None
date_times_t2r = None
t2r_vals = None
date_times_t2e = None
t2e_vals = None

process_shots_t1ge = False
use_png_timestamps = False

restrict_time = False
start_time = datetime.datetime(2025, 11, 18, 0, 0)
end_time = datetime.datetime(2025, 11, 21, 12, 0)

if coh_qtemp_ana_flags["load_rpm_qtemps"]:
    # ----------- Get Qubit temperature results via RPMs
    RPM_calcs = RPMTempCalcAndPlots(figure_quality, tot_num_of_qubits)
    all_files_Qtemp_results_RPMs = RPM_calcs.run_RPMqtemps(base_dir, target_dates_qtemps_RPM, filter_keywords,
                                                           fit_saved, signal,
                                                           run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                                                           outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data,
                                                           get_london_data, figure_quality, save_figsRR,
                                                           exclude_temp_sweeps, passing_pre_sciencerun_data=False,
                                                           filter_out_bad_amp_fits = filter_out_bad_amp_fits,
                                                           use_png_timestamps = use_png_timestamps, combine_IQ_signal = rpm_combine_IQ_signal)
    if run_num == 6:
        if pre_sciencerun6_data:
            all_files_Qtemp_results_RPMs2 = RPM_calcs.run_RPMqtemps(base_dir2, target_dates_qtemps_RPM2,
                                                                    filter_keywords2, fit_saved, signal, run_name,
                                                                    run_num, list_of_all_qubits, tot_num_of_qubits,
                                                                    outerFolder_qtemps_plots_RR, replot_RPMs,
                                                                    get_qtemp_data, get_london_data,
                                                                    figure_quality, save_figsRR, exclude_temp_sweeps,
                                                                    passing_pre_sciencerun_data=True,
                                                                    filter_out_bad_amp_fits = filter_out_bad_amp_fits,
                                                                    use_png_timestamps = use_png_timestamps, combine_IQ_signal = rpm_combine_IQ_signal)
            all_files_Qtemp_results_RPMs += all_files_Qtemp_results_RPMs2

if coh_qtemp_ana_flags["load_ssf_qtemps"]: # IMPORTANT: have not yet implemented use_png_timestamps. TO DO.
    # ----------- Get Qubit temperature results via SSF g-state double gaussian threshold method
    SSF_calcs_obj = SSFTempCalcAndPlots(figure_quality, tot_num_of_qubits, run_num, save_figs)
    pairs_info = SSF_calcs_obj.process_ssf_and_qfreq_data_qtemps(Science_Qubits, paths_SSFmethods)

    all_qubit_temps_g, all_qubit_times_g, all_qubit_temps_errs_g, fit_results_g = SSF_calcs_obj.run_ssf_qtemps(
        pairs_info, limit_temp_k=0.95, use_gessf_thresh_only=False, fallback_to_threshold=False)

if coh_qtemp_ana_flags["load_mcp1_temps"]:
    mcp1_csv_path = "/data/QICK_data/run8/6transmon/round_robin/temperature_sweep_qubit_data/Mixing chamber stage-data-2025-11-25 09_46_33.csv"
    combined_studies = combined_Qtemp_studies(figure_quality, tot_num_of_qubits)
    mcp_dates, mcp_temps, _ = combined_studies.load_mixing_chamber_csv(mcp1_csv_path, restrict_time=True,
                                start_time=start_time, end_time=end_time)
    del combined_studies

if coh_qtemp_ana_flags["load_coherence_res"]:
    # q_spec_vs_time = QubitFreqsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
    #                                   save_figs, fit_saved, signal, run_name_coh, FRIDGE)
    # date_times_q_spec, q_freqs, qspec_fit_err = q_spec_vs_time.run(exp_extension='_ge', use_png_timestamps = use_png_timestamps)

    t1_vs_time = T1VsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,signal,
                          run_name_coh, FRIDGE, run_num)
    date_times_t1, t1_vals, t1_fit_err = t1_vs_time.run(return_errs=True, exp_extension='_ge',process_shots=process_shots_t1ge, use_png_timestamps = use_png_timestamps)

    # IMPORTANT: have not yet implemented use_png_timestamps for t2r and t2e. TO DO.
    t2r_vs_time = T2rVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                     signal, run_name, FRIDGE)
    date_times_t2r, t2r_vals, t2r_fit_err = t2r_vs_time.run(return_errs=True, use_png_timestamps = True)

    t2e_vs_time = T2eVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                     signal, run_name, FRIDGE)
    date_times_t2e, t2e_vals, t2e_fit_err = t2e_vs_time.run(return_errs=True, use_png_timestamps = True)


if coh_qtemp_ana_flags["plot_qtemps_t1_ftemps_qfreq"]:
    comb_plots_path = "/data/QICK_data/run8/6transmon/round_robin/temperature_sweep_qubit_data/analysis_plots/"
    combined_studies = combined_Qtemp_studies(figure_quality, tot_num_of_qubits)
    combined_studies.plot_qtemps_and_coherence_res(comb_plots_path, all_qubit_temperatures_ssf_g=all_qubit_temps_g, all_qubit_timestamps_ssf_g=all_qubit_times_g,
                                  all_files_Qtemp_results_RPMs=all_files_Qtemp_results_RPMs, fridge_temps=mcp_temps, fridge_dates=mcp_dates,
                                  t1_vals=t1_vals, t1_dates=date_times_t1, qfreqs_vals=q_freqs, qfreqs_dates=date_times_q_spec,
                                  restrict_time_xaxis=restrict_time, start_time = start_time, end_time = end_time, plot_extra_event_lines=False)