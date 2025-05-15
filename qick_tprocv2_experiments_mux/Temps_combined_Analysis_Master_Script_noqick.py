import sys
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import os
sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from analysis_021_plot_allRR_noqick import QubitSpectroscopy
from qicklab.analysis import qspec, ssf
from section_008_save_data_to_h5 import Data_H5
from analysis_014_temp_calcsandplots_cosmiqgpvm import SSFTempCalcAndPlots
from analysis_014_temp_calcsandplots_cosmiqgpvm import RPMTempCalcAndPlots
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
save_figsRR = False # Do you want to save (or not save) the re-plotted rabi population measurements plots?
save_figs = False # To be used in general for any function or class to saver (or not save) plots.
fit_saved = False # Not used here, set to false.

get_qtemp_data = True # Do you want to calculate qubit temperatures?

figure_quality = 200
theta = 0
threshold = 0
tot_num_of_qubits = 6 # Total number of qubits currently at QUIET
run_number = 3 # Starting from first run with qubits: Run 1 = run4a at quiet, run 2 = run5a at quiet, etc
figure_quality = 200
Science_Qubits = [0, 4]

# What method or methods do you want to use to calculate qubit temperatures?
qtemp_method_flags = {"Qtemps_viaRPM": True, "Qtemps_viaSSF_ge_thresh": False, "Qtemps_viaSSF_gmeans_thresh": False}

# What analysis plots do you want to make?
analysis_flags = {"Qtemps_vs_time_viaSSF": False,  "Qtemps_vs_time_viaRPM": True, "Threshold_Check_Qtemps_viaSSF": False, "ge_thresh_check_ssf": False,
                  "Qtemps_hists_viaRPM": False,  "Pe_vs_time_viaRPM": False, "qtemps_Pe_vs_time_viaRPM": False, "qtemps_Pe_gefreq_vs_time_viaRPM": False}
#----------------------------------------------------------------------- Set up -----------------------------------------------------------------------------------------
# For qubit temperature calculations via rabi population measurements
target_dates_qtemps_RPM = [
    # "2025-04-16",
    # "2025-04-17",
    # "2025-04-18",
    # "2025-04-19",
    # "2025-04-20",
    # "2025-04-21", #starts source on (Co)
    # "2025-04-22",
    # "2025-04-23", #switched source (to Cs)
    # "2025-04-24",
    # "2025-04-25",
    # "2025-04-26",
    # "2025-04-27",
    # "2025-04-28", #Cs source moved closer
    # "2025-04-29",
    # "2025-04-30",
    # "2025-05-01",
    # "2025-05-02",
    # "2025-05-03",
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
    "2025-05-14"
    ]
base_dir = "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study"

# To re-make and save RPM RR plots
outerFolder_qtemps_plots_RR = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/Plots/PlotRR"

# For Analysis
outerFolder_qtemps_plots = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/Plots"

filter_keywords = ['source_off', 'source_on']

# For qubit temperature calculations via SSF methods (double gaussian over g-state data and double gaussian over g and e-state data

###################################################### Qubit temperature calculations via rabi population measurements #############################################
if qtemp_method_flags["Qtemps_viaRPM"]:
    RPM_calcs = RPMTempCalcAndPlots(figure_quality, tot_num_of_qubits, save_figs)
    combined_qtemp_data = RPM_calcs.run_RPMqtemps(base_dir, target_dates_qtemps_RPM, filter_keywords, fit_saved, signal, run_name, list_of_all_qubits, tot_num_of_qubits,
                            outerFolder_qtemps_plots_RR, replot_RPMs, get_qtemp_data, figure_quality, save_figsRR)

    #----------------------------------------------------------------------- RPM Analysis -------------------------------------------------------------------------
    # These are not used in the definitions that follow, are just needed to initialize the class
    outerFolder = ""
    outerFolder_qtemps_plots = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/Plots"
    outerFolder_qtemps_data = ""
    date_string = ""
    RPM_plotter = PlotRR_noQick(date_string, figure_quality, save_figs, fit_saved, signal, run_name, tot_num_of_qubits, outerFolder, outerFolder_qtemps_plots, outerFolder_qtemps_data)

    if analysis_flags["Qtemps_vs_time_viaRPM"]:
        #------------------------------------------------------------------- Qubit temperatures vs time via RPMs ----------------------------------------------------
        RPM_plotter.plot_qubit_temperatures_vs_time(combined_qtemp_data, restrict_time_xaxis = False, plot_extra_event_lines = False, rad_events_plot_lines = False)

    if analysis_flags["Qtemps_hists_viaRPM"]:
        #----------------------------------------------------------------- Histograms of Qubit temperatures (via RPMs) -----------------------------------------------
        RPM_plotter.plot_qubit_temperature_histograms(combined_qtemp_data)

    if analysis_flags["Pe_vs_time_viaRPM"]:
        #------------------------------------------------------------ Excited state populations (P_e) vs time (via RPMs) ----------------------------------------
        RPM_plotter.plot_qubit_pe_vs_time(combined_qtemp_data)

    if analysis_flags["qtemps_Pe_vs_time_viaRPM"]:
        #---------------------------------------------------------- Qubit temp and P_e vs time in the same plot (via RPMs) ------------------------------------
        RPM_plotter.plot_qubit_temp_and_pe_vs_time(combined_qtemp_data)

    if analysis_flags["qtemps_Pe_gefreq_vs_time_viaRPM"]:
        #---------------------------------------------------- Qubit temp, P_e, and g-e qubit freq vs time in the same plot (via RPMs) --------------------------
        RPM_plotter.plot_qubit_temp_pe_freq_vs_time(combined_qtemp_data)


# ################################################################# Qubit temperature calculations via SSF measurements #################################################
# if qtemp_method_flags["Qtemps_viaSSF_ge_thresh"]:
#
# #----------------------------------------------------------------------- SSF Analysis -------------------------------------------------------------------------
#
# ################################################################# Qubit temperature calculations via SSF measurements #################################################
# if qtemp_method_flags["Qtemps_viaSSF_gmeans_thresh"]:
#
# # ----------------------------------------------------------------------- SSF Analysis -------------------------------------------------------------------------