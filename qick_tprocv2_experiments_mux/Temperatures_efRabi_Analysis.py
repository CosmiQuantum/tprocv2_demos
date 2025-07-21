import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from analysis_001_plot_all_RR_h5 import PlotAllRR
import os
import sys
import h5py
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from analysis_020_gef_ssf_fstate_plots import GEF_SSF_ANALYSIS

#---------------------------------------------------------Folders and Paths-------------------------------------------------------------------
plot_ssf_gef = False # Do you want to re-plot g-e-f SSF data and save the plots?
figure_quality = 200
save_figs = False # If you are running plotter.run, do you want to save all of the plots for the chosen experiment?
fit_saved = False # Not used here, set to false
signal = 'None' # Do not change

run_name = 'run6/6transmon/'
date = '2025-04-11'  # only go through all of the data for one date at a time because there is a lot

# Where plots are saved
# outerFolder = f"/data/QICK_data/run6/6transmon/ef_studies/QubitTemps_efRabi_method/{date}/Optimization/Round_Robin_mode"
# outerFolder_qtemps_data = f"/data/QICK_data/run6/6transmon/ef_studies/QubitTemps_efRabi_method/{date}/Study_Data"
# outerFolder_qtemps_plots = os.path.join(outerFolder_qtemps_data, "analysis_plots")

# For analysis at cosmiqgpvm02
outerFolder = "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_12-51-09/optimization"
outerFolder_qtemps_data = "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_12-51-09/optimization"
outerFolder_qtemps_plots = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/Plots/params_vs_time"


if not os.path.exists(outerFolder): os.makedirs(outerFolder)
if not os.path.exists(outerFolder_qtemps_data): os.makedirs(outerFolder_qtemps_data)
if not os.path.exists(outerFolder_qtemps_plots): os.makedirs(outerFolder_qtemps_plots)
#------------------------------------------------Initialize the Plotting class------------------------------------------------
plotter = PlotAllRR(date, figure_quality, save_figs, fit_saved, signal, run_name, tot_num_of_qubits, outerFolder,
                  outerFolder_qtemps_plots, outerFolder_qtemps_data)

#------------------------------------To re-plot the g-e-f SSF plots, or any other data from the selected date-----------------------------------------------------
# plotter.run(plot_res_spec = False, plot_q_spec = False, plot_rabi = False, plot_ss = False,  ss_plot_gef = True, plot_t1 = False,
#             plot_t2r = False, plot_t2e = False, plot_rabis_Qtemps = True)

# Load data
qtemp_data = plotter.load_plot_save_rabis_Qtemps(list_of_all_qubits)

#Qubit temperatures vs time
# plotter.plot_qubit_temperatures_vs_time(qtemp_data)

#Histograms of Qubit temperatures
# plotter.plot_qubit_temperature_histograms(qtemp_data)

#Excited state populations (P_e) vs time
# plotter.plot_qubit_pe_vs_time(qtemp_data)

#Qubit temp and P_e vs time in the same plot
# plotter.plot_qubit_temp_and_pe_vs_time(qtemp_data)

#Qubit temp, P_e, and g-e qubit freq vs time in the same plot
plotter.plot_qubit_temp_pe_freq_vs_time(qtemp_data)