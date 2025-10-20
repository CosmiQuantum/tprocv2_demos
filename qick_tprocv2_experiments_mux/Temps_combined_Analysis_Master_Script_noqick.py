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

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
run_num = 8 # first run with qubits was QUIET run 3, second run with qubits was QUIET run 4, and so forth
run_name = f'run{run_num}/6transmon'
signal = 'None' # Do not change

plot_ssf_gef = False # Do you want to re-plot g-e-f SSF data and save the plots?
replot_RPMs = False # Do you want to re-plot rabi population measurements from RR data?
save_figsRR = False # Do you want to save (or not save) re-plotted RR measurements plots?
save_figs = False # To be used in general for any function or class to saver (or not save) plots.
fit_saved = False # Not used here, set to false.
exclude_temp_sweeps = True # Do you want to exclude the folders that contain data taken during the heater temperature sweep?

get_qtemp_data = False # Do you want to calculate RPM qubit temperatures? This returns RPM qubit temperatures and qubit freqs for specified dates.
get_london_data = True # This returns RPM qubit temperatures, resonator freqs, and qubit freqs for specified dates. Designed for London Penetration analysis.

pre_sciencerun6_data = True # Do you also want to incorporate the run 6 pre-science run data? THis only applies when run_num = 6

figure_quality = 200
theta = 0
threshold = 0
tot_num_of_qubits = 6 # Total number of qubits currently at QUIET

# What method or methods do you want to use to calculate qubit temperatures?
qtemp_method_flags = {"Qtemps_viaRPM": False, "Qtemps_viaSSF_ge_thresh": False, "Qtemps_viaSSF_gmeans_thresh": True, "Qtemps_viaSSF_with_fallback": False,
                      "combined_studies_qtemps": True}

# What analysis plots do you want to make?
analysis_flags = {"Qtemps_vs_time_viaSSF": False,  "Qtemps_vs_time_viaRPM": False, "Threshold_Check_Qtemps_viaSSF": False, "ge_thresh_check_ssf": False,
                  "Qtemps_hists_viaRPM": False, "Qtemps_hists_viaSSF": True, "Pe_vs_time_viaRPM": False, "qtemps_Pe_vs_time_viaRPM": False, "qtemps_Pe_gefreq_vs_time_viaRPM": False}

# For combined analysis
comb_analysis_flags = {"Qtemps_vs_time_comb_separate_plts": False,"Qtemps_vs_time_comb_single_plt": True, "Pe_vs_time_comb_separate_plts": False,
                       "Pe_vs_time_comb_single_plt": False }

# For London Penetration Depth analysis
london_flags = {"get_qfreqs_resfreqs_qtemps": False}

# For double-gaussian SSF analysis using non-pre-built functions (ft Dan)
non_prebuilt_ana_flags = {"Qtemps_chi2_hists_viaSSF": True}
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
outerFolder_qtemps_plots_RR_run6 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/QTemperatures/Plots/Plots_RR"
# For RPM Analysis
# outerFolder_qtemps_plots_run6 = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/Plots" # Inside each analysis function, a subfolder will be defined
outerFolder_qtemps_plots_run6 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/QTemperatures/Plots" # Inside each analysis function, a subfolder will be defined
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
base_dir_run7 = "/exp/cosmiq/data/QUIET/QICK_data/run7/6transmon/round_robin_benchmark"

# for 24hr AB data
target_dates_qtemps_RPM_run7 = ["2025-07-19", "2025-07-20"]

# To re-make and save RPM RR plots
outerFolder_qtemps_plots_RR_run7 = "/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_tests_data/benchmark_analysis_plots/RPM_RR_plots"
#
# For RPM Analysis
outerFolder_qtemps_plots_run7 = "/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_tests_data/benchmark_analysis_plots/q_temperatures_plots" # Inside each analysis function, a subfolder will be defined

# Substudy name on the file path, doesn't have to be exact, it will look for these key terms in the name
filter_keywords_run7 = ['AB_tests_data']

#-----------------------------------------------------------------------run 8------------------------------------------------------------
# Base path of where the data is stored up to the Study Name (round_robin_benchmark)
base_dir_run8 = "/data/QICK_data/run8/6transmon/round_robin"

# for 24hr AB data
target_dates_qtemps_RPM_run8 = ["2025-10-19"]

# To re-make and save RPM RR plots
outerFolder_qtemps_plots_RR_run8 = "/data/QICK_data/run8/6transmon/round_robin/AB_Paper_Data_24hrs/benchmark_analysis_plots/RPM_RR_plots"
#
# For RPM Analysis
outerFolder_qtemps_plots_run8 = "/data/QICK_data/run8/6transmon/round_robin/AB_Paper_Data_24hrs/benchmark_analysis_plots/Qtemps_RPMmethod" # Inside each analysis function, a subfolder will be defined

# Substudy name on the file path, doesn't have to be exact, it will look for these key terms in the name
filter_keywords_run8 = ['AB_Paper_Data_24hrs']
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
    raise ValueError("You must choose run_num = 6 or run_num = 7. Otherwise, define a section for your run of interest.")

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

path_saveplots_fits_run5 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/QTemperatures/Plots/Qtemps_SSFmethod/Gaussian_Fits_run5" # where to save ssf plots to check gaussian fits
path_saveplots_ssf_qtemps_vsT_run5 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/QTemperatures/Plots/Qtemps_SSFmethod/Qtemps_vs_Time_run5" # to save qubit temps vs time via ssf methods

# ------------------------------------------------------------------------------------------------run 6------------------------------------------------------------------------------------------------------------
# Science-Run Data
paths_SSFmethods_SR = ["/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_03-03-40",
                    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_06-40-15",
                    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_10-18-53",
                    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_13-57-22",
                    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_17-34-21",
                    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_21-18-14"]
# Pre-Science-Run Data
paths_SSFmethods_preSR = ["/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/ef_studies_pre_science_run/q_temperatures_efRabi/2025-04-11_14-05-22"]

path_saveplots_fits_run6 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/QTemperatures/Plots/Qtemps_SSFmethod/Gaussian_Fits_run6" # where to save ssf plots to check gaussian fits
path_saveplots_ssf_qtemps_vsT_run6 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/QTemperatures/Plots/Qtemps_SSFmethod/Qtemps_vs_Time_run6" # to save qubit temps vs time via ssf methods

# ----------------------------------------------------------------------------------------------run 7----------------------------------------------------------------------------------------------------------
paths_SSFmethods_run7 = ["/exp/cosmiq/data/QUIET/QICK_data/run7/6transmon/round_robin_benchmark/AB_tests_data/2025-07-19_08-34-39"]
path_saveplots_fits_run7 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/QTemperatures/Plots/Qtemps_SSFmethod/Gaussian_Fits_run7" # where to save ssf plots to check gaussian fits
path_saveplots_ssf_qtemps_vsT_run7 = "/exp/cosmiq/data/home/cosmiq/Analysis_on1hw_temporary/acolonce/QTemperatures/Plots/Qtemps_SSFmethod/Qtemps_vs_Time_run7" # to save qubit temps vs time via ssf methods

# ----------------------------------------------------------------------------------------------run 8----------------------------------------------------------------------------------------------------------
paths_SSFmethods_run8 = ["/data/QICK_data/run8/6transmon/round_robin/AB_Paper_Data_24hrs/2025-10-19_11-09-32",
                         "/data/QICK_data/run8/6transmon/round_robin/AB_Paper_Data_24hrs/2025-10-19_12-05-25",
                         "/data/QICK_data/run8/6transmon/round_robin/AB_Paper_Data_24hrs/2025-10-19_19-43-00",
                         "/data/QICK_data/run8/6transmon/round_robin/AB_Paper_Data_24hrs/2025-10-19_20-25-18"]

path_saveplots_fits_run8 = "/data/QICK_data/run8/6transmon/round_robin/AB_Paper_Data_24hrs/benchmark_analysis_plots/Qtemps_SSFmethod/Gaussian_Fits_run8" # where to save ssf plots to check gaussian fits
path_saveplots_ssf_qtemps_vsT_run8 = "/data/QICK_data/run8/6transmon/round_robin/AB_Paper_Data_24hrs/benchmark_analysis_plots/Qtemps_SSFmethod/Qtemps_vs_Time_run8" # to save qubit temps vs time via ssf methods

#------------------------------------------------------------------------------ Assign func variables depending on run number ---------------------------------------
if run_num == 6:  # We have science-run data as well as pre-science-run data available. Note: we already defined Science_Qubits above.
    paths_SSFmethods = paths_SSFmethods_SR
    if pre_sciencerun6_data:  # If True, include pre-science-run data
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

############################################################################### Qubit temperature calculations via rabi population measurements #####################################################
if qtemp_method_flags["Qtemps_viaRPM"]:
    RPM_calcs = RPMTempCalcAndPlots(figure_quality, tot_num_of_qubits, save_figs)
    combined_qtemp_data = RPM_calcs.run_RPMqtemps(base_dir, target_dates_qtemps_RPM, filter_keywords, fit_saved, signal, run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                            outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, get_london_data, figure_quality, save_figsRR, exclude_temp_sweeps, passing_pre_sciencerun_data = False)

    if pre_sciencerun6_data:
        combined_qtemp_data2 = RPM_calcs.run_RPMqtemps(base_dir2, target_dates_qtemps_RPM2, filter_keywords2, fit_saved, signal, run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                                                      outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, get_london_data, figure_quality, save_figsRR, exclude_temp_sweeps, passing_pre_sciencerun_data = True)
        combined_qtemp_data += combined_qtemp_data2

    del RPM_calcs # to free up memory
    #----------------------------------------------------------------------- RPM Analysis -------------------------------------------------------------------------
    # These are not used in the definitions that follow, are just needed to re-initialize the class
    outerFolder = ""
    outerFolder_qtemps_data = ""
    date_string = ""
    RPM_plotter = PlotRR_noQick(date_string, figure_quality, save_figs, fit_saved, signal, run_name, tot_num_of_qubits, outerFolder, outerFolder_qtemps_plots, outerFolder_qtemps_data)

    if analysis_flags["Qtemps_vs_time_viaRPM"]:
        #------------------------------------------------------------------- Qubit temperatures vs time via RPMs ----------------------------------------------------
        RPM_plotter.plot_qubit_temperatures_vs_time_RPMs(combined_qtemp_data, num_qubits=tot_num_of_qubits, yaxis_min = 40, yaxis_max = 950, rel_err_cutoff = 0.4, restrict_time_xaxis = False,
                                                         plot_extra_event_lines = False, rad_events_plot_lines = False, plot_error_bars = True, fit_to_line=False, average_per_heater_step=False)

    if analysis_flags["Qtemps_hists_viaRPM"]:
        #----------------------------------------------------------------- Histograms of Qubit temperatures (via RPMs) -----------------------------------------------
        RPM_plotter.plot_qubit_temperature_histograms_RPMs(combined_qtemp_data, num_qubits=6, rel_err_cutoff = 0.4)

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
        all_qubit_temps, all_qubit_times, all_qubit_temps_errs, fit_results  = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=0.8, use_gessf_thresh_only = False, fallback_to_threshold = False)
    elif qtemp_method_flags["Qtemps_viaSSF_ge_thresh"]: # Fits both GROUND STATE and PREPARED EXCITED STATE SSF data to a double gaussian ; threshold = midpoint of the two gaussian means
        all_qubit_temps, all_qubit_times, all_qubit_temps_errs, fit_results  = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=0.8, use_gessf_thresh_only = True, fallback_to_threshold = False)
    elif qtemp_method_flags["Qtemps_viaSSF_with_fallback"]: # Uses Default method and if the fit fails it falls back to the method that fits both GROUND STATE and PREPARED EXCITED STATE SSF data to a double gaussian
        all_qubit_temps, all_qubit_times, all_qubit_temps_errs, fit_results  = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=0.8, use_gessf_thresh_only = False, fallback_to_threshold = True)

    #---------------------------------------------------------------------------- SSF Qubit Temps Analysis -------------------------------------------------------------------------
    #------------------------------------------------------------------ Temperatures vs Time Scatter Plot --------------------------------------------------------------------------
    if analysis_flags["Qtemps_vs_time_viaSSF"]:
        SSF_calcs_obj.plot_qubit_temperatures_vs_time_ssf(all_qubit_temps, all_qubit_times, all_qubit_temps_errs, path_saveplots_ssf_qtemps_vsT, rel_err_cutoff = 0.4, plot_error_bars = True)
    #--------------------------------------------------------------------------- SSF Temperature Histograms --------------------------------------------------------------------------------------
    if analysis_flags["Qtemps_hists_viaSSF"]:
        SSF_calcs_obj.plot_all_qubits_hist_ssf(all_qubit_temps, all_qubit_temps_errs, path_saveplots_ssf_qtemps_vsT, bins=30, rel_err_cutoff = 0.4)
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
if non_prebuilt_ana_flags["Qtemps_chi2_hists_viaSSF"]:
    non_pre_built_ana = non_prebuilt_ssf_analysis_class()
    # To be continued
################################################### Combined Qubit Temperature Analyses ##########################################################
if qtemp_method_flags["combined_studies_qtemps"]:
    # ----------- Get Qubit temperature results via RPMs
    RPM_calcs = RPMTempCalcAndPlots(figure_quality, tot_num_of_qubits, save_figs)
    all_files_Qtemp_results_RPMs = RPM_calcs.run_RPMqtemps(base_dir, target_dates_qtemps_RPM, filter_keywords, fit_saved, signal, run_name, list_of_all_qubits, tot_num_of_qubits,
                                outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, figure_quality, save_figsRR, passing_pre_sciencerun_data = False)
    if pre_sciencerun6_data:
        all_files_Qtemp_results_RPMs2 = RPM_calcs.run_RPMqtemps(base_dir2, target_dates_qtemps_RPM2, filter_keywords2, fit_saved, signal, run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                                                      outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, get_london_data, figure_quality, save_figsRR, exclude_temp_sweeps, passing_pre_sciencerun_data = True)
        all_files_Qtemp_results_RPMs += all_files_Qtemp_results_RPMs2

    # ----------- Get Qubit temperature results via SSF g-e threshold method and SSF g-state double gaussian threshold method
    SSF_calcs_obj = SSFTempCalcAndPlots(figure_quality, tot_num_of_qubits, run_num, save_figs)
    pairs_info = SSF_calcs_obj.process_ssf_and_qfreq_data_qtemps(Science_Qubits, paths_SSFmethods)

    all_qubit_temps_g, all_qubit_times_g, all_qubit_temps_errs_g, fit_results_g  = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=0.95, use_gessf_thresh_only = False, fallback_to_threshold = False)
    all_qubit_temps_ge, all_qubit_times_ge, all_qubit_temps_errs_ge, fit_results_ge = SSF_calcs_obj.run_ssf_qtemps(pairs_info, limit_temp_k=0.95, use_gessf_thresh_only=True, fallback_to_threshold=False)

    #------------ Initialize class for combined qubit temps analysis ------------------
    combined_studies = combined_Qtemp_studies(figure_quality, tot_num_of_qubits)

    # ------------ Qubit temperatures vs Time using all three methods
    if comb_analysis_flags["Qtemps_vs_time_comb_separate_plts"]:
        # Plots two rows (one for each qubit) and 3 columns (one for each method)
        combined_studies.Qtemps_vs_time_comb_methods(all_qubit_temps_g, all_qubit_times_g, all_qubit_temps_errs_g, all_qubit_temps_ge, all_qubit_times_ge, all_qubit_temps_errs_ge,
                                                     outerFolder_qtemps_plots, all_files_Qtemp_results_RPMs, restrict_time_xaxis = False, plot_extra_event_lines = False,
                                                     rad_events_plot_lines = False, plot_error_bars = True)
    if comb_analysis_flags["Qtemps_vs_time_comb_single_plt"]:
        # Plots two rows (one for each qubit) and 1 column (all methods in a single plot)
        combined_studies.Qtemps_vs_time_comb_2subplts(all_qubit_temps_g, all_qubit_times_g, all_qubit_temps_ge, all_qubit_times_ge, outerFolder_qtemps_plots,
                                                     all_files_Qtemp_results_RPMs, restrict_time_xaxis = False, plot_extra_event_lines = False, rad_events_plot_lines = False)

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
if london_flags["get_qfreqs_resfreqs_qtemps"]: # There was no "pre-science-run" data for this analysis, since the relevant data is the science run heater temperature sweep data
    #--------------------------------------Get RPM qubit temps, qfreqs and res freqs, etc. -----------------------
    RPM_calcs = RPMTempCalcAndPlots(figure_quality, tot_num_of_qubits, save_figs)
    qfreqs_resfreqs_qtemps_data = RPM_calcs.run_RPMqtemps(base_dir, target_dates_qtemps_RPM, filter_keywords, fit_saved, signal,
                                                  run_name, run_num, list_of_all_qubits, tot_num_of_qubits,
                                                  outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, get_london_data,
                                                  figure_quality, save_figsRR, exclude_temp_sweeps, passing_pre_sciencerun_data = False)
    # ----------------------------Dump RPM qubit temps, qfreqs and res freqs, etc in excel spreadhseet -----------------------
    # These are not used in the definitions that follow, are just needed to re-initialize the class
    outerFolder = ""
    outerFolder_qtemps_data = ""
    date_string = ""
    RPM_plotter = PlotRR_noQick(date_string, figure_quality, save_figs, fit_saved, signal, run_name, tot_num_of_qubits, outerFolder, outerFolder_qtemps_plots, outerFolder_qtemps_data)
    RPM_plotter.save_RPM_qtemp_data_to_excel(qfreqs_resfreqs_qtemps_data, outerFolder_london_path, run_num, FRIDGE)
    del RPM_calcs