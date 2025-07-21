import sys
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import os
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from analysis_021_plot_allRR_noqick import QubitSpectroscopy
from qicklab.analysis import qspec, ssf
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
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from analysis_021_plot_allRR_noqick import PlotRR_noQick

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
run_name = 'run6/6transmon'
signal = 'None' # Do not change

plot_ssf_gef = False # Do you want to re-plot g-e-f SSF data and save the plots?
replot_RPMs = False # Do you want to re-plot rabi population measurements from RR data?
save_figsRR = False # Do you want to save (or not save) re-plotted RR measurements plots?
save_figs = False # To be used in general for any function or class to saver (or not save) plots.
fit_saved = False # Not used here, set to false.
exclude_temp_sweeps = True # Do you want to exclude the folders that contain data taken during the heater temperature sweep?

get_qtemp_data = True # Do you want to calculate qubit temperatures?

figure_quality = 200
theta = 0
threshold = 0
tot_num_of_qubits = 6 # Total number of qubits currently at QUIET
run_number = 3 # Starting from first run with qubits: Run 1 = run4a at quiet, run 2 = run5a at quiet, etc
figure_quality = 200
Science_Qubits = [0, 4]

# What method or methods do you want to use to calculate qubit temperatures?
qtemp_method_flags = {"Qtemps_viaRPM": True, "Qtemps_viaSSF_ge_thresh": False, "Qtemps_viaSSF_gmeans_thresh": False, "Qtemps_viaSSF_with_fallback": False,
                      "combined_studies_qtemps": False}

# What analysis plots do you want to make?
analysis_flags = {"Qtemps_vs_time_viaSSF": False,  "Qtemps_vs_time_viaRPM": False, "Threshold_Check_Qtemps_viaSSF": False, "ge_thresh_check_ssf": False,
                  "Qtemps_hists_viaRPM": True,  "Pe_vs_time_viaRPM": False, "qtemps_Pe_vs_time_viaRPM": False, "qtemps_Pe_gefreq_vs_time_viaRPM": False}

# For combined analysis
comb_analysis_flags = {"Qtemps_vs_time_comb_separate_plts": False,"Qtemps_vs_time_comb_single_plt": False, "Pe_vs_time_comb_separate_plts": False,
                       "Pe_vs_time_comb_single_plt": False }
############################################################################## Set up ##############################################################################
#-------------------------------------------- For qubit temperature calculations via rabi population measurements ---------------------------------------------------
# Specify which dates you want to loop through. It will process all the files inside all the folders that contain these dates in their title.
target_dates_qtemps_RPM = [
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

# For data before heater temperature steps
# target_dates_qtemps_RPM = [
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

# For data during Heater temperature steps
# target_dates_qtemps_RPM = ["2025-05-07", "2025-05-08", "2025-05-09", "2025-05-10", "2025-05-11", "2025-05-12", "2025-05-13", "2025-05-14", "2025-05-15", "2025-05-16"]

# if you want to look at just one specific date
# target_dates_qtemps_RPM = ["2025-05-05"]

base_dir = "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study"

# To re-make and save RPM RR plots
outerFolder_qtemps_plots_RR = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/Plots/Plots_RR"

# For RPM Analysis
outerFolder_qtemps_plots = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/Plots" # Inside each analysis function, a subfolder will be defined

# filter_keywords = ['source_off', 'source_on'] # set up for RPM measurements. Which data do you want to look at? with source or no source?
filter_keywords = ['source_off']

#------------------- For qubit temperature calculations via SSF methods (double gaussian over g-state data and double gaussian over g and e-state data --------------
# Note: you must write paths in this form: "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_03-03-40"
# If you want to loop through all the data corresponding to 1 day, you must list all the paths for that day. This method does not accept just a single date as a path.
paths_SSFmethods = ["/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_03-03-40",
                    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_06-40-15",
                    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_10-18-53",
                    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_13-57-22",
                    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_17-34-21",
                    "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy5/2025-05-05_21-18-14"]

###################################################### Qubit temperature calculations via rabi population measurements #############################################
if qtemp_method_flags["Qtemps_viaRPM"]:
    RPM_calcs = RPMTempCalcAndPlots(figure_quality, tot_num_of_qubits, save_figs)
    combined_qtemp_data = RPM_calcs.run_RPMqtemps(base_dir, target_dates_qtemps_RPM, filter_keywords, fit_saved, signal, run_name, list_of_all_qubits, tot_num_of_qubits,
                            outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, figure_quality, save_figsRR, exclude_temp_sweeps)
    del RPM_calcs
    #----------------------------------------------------------------------- RPM Analysis -------------------------------------------------------------------------
    # These are not used in the definitions that follow, are just needed to re-initialize the class
    outerFolder = ""
    outerFolder_qtemps_data = ""
    date_string = ""
    RPM_plotter = PlotRR_noQick(date_string, figure_quality, save_figs, fit_saved, signal, run_name, tot_num_of_qubits, outerFolder, outerFolder_qtemps_plots, outerFolder_qtemps_data)

    if analysis_flags["Qtemps_vs_time_viaRPM"]:
        #------------------------------------------------------------------- Qubit temperatures vs time via RPMs ----------------------------------------------------
        RPM_plotter.plot_qubit_temperatures_vs_time_RPMs(combined_qtemp_data, num_qubits=tot_num_of_qubits, yaxis_min = 15, yaxis_max = 700, restrict_time_xaxis = False,
                                                         plot_extra_event_lines = False, rad_events_plot_lines = False, plot_error_bars = True, fit_to_line=False, average_per_heater_step=False)

    if analysis_flags["Qtemps_hists_viaRPM"]:
        #----------------------------------------------------------------- Histograms of Qubit temperatures (via RPMs) -----------------------------------------------
        RPM_plotter.plot_qubit_temperature_histograms_RPMs(combined_qtemp_data)

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

    SSF_calcs_obj = SSFTempCalcAndPlots(figure_quality, tot_num_of_qubits, save_figs)
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
        path_saveplots = f"/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/RR_metrics/Plots/Qtemps_SSFmethod/Qtemps_vs_Time"
        SSF_calcs_obj.plot_qubit_temperatures_vs_time_ssf(all_qubit_temps, all_qubit_times, all_qubit_temps_errs, path_saveplots, plot_error_bars = True)

    #------------------------------------------------------------ Check General SSF Double Gaussian Fits and g-e threshold ---------------------------------------------------------
    if analysis_flags["ge_thresh_check_ssf"]:
        path_saveplots_fits = f"/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/RR_metrics/Plots/Qtemps_SSFmethod/geSSF_Fits"
        thresh_results = SSF_calcs_obj.plot_ssf_ge_thresh(pairs_info=pairs_info, plotting_path=path_saveplots_fits)

    #---------------------------------------------------- Check population threshold for Qubit Temperature Calcs via both SSF methods ----------------------------------------------
    if analysis_flags["Threshold_Check_Qtemps_viaSSF"]:
        path_saveplots_fits = f"/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/RR_metrics/Plots/Qtemps_SSFmethod/Gaussian_Fits"
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

#################################################### Combined Qubit Temperature Analyses ##########################################################
if qtemp_method_flags["combined_studies_qtemps"]:
    # ----------- Get Qubit temperature results via RPMs
    RPM_calcs = RPMTempCalcAndPlots(figure_quality, tot_num_of_qubits, save_figs)
    all_files_Qtemp_results_RPMs = RPM_calcs.run_RPMqtemps(base_dir, target_dates_qtemps_RPM, filter_keywords, fit_saved, signal, run_name, list_of_all_qubits, tot_num_of_qubits,
                                outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, figure_quality, save_figsRR)

    # ----------- Get Qubit temperature results via SSF g-e threshold method and SSF g-state double gaussian threshold method
    SSF_calcs_obj = SSFTempCalcAndPlots(figure_quality, tot_num_of_qubits, save_figs)
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