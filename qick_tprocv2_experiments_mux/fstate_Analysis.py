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
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))

plot_ssf_gef = False # Do you want to re-plot g-e-f SSF data and save the plots?
figure_quality = 200
save_figs = False # If you are running plotter.run, do you want to save all of the plots for the chosen experiment?
fit_saved = False # Not used here, set to false
signal = 'None' # Do not change

run_name = 'run6/6transmon/ef_studies/Optimization/'
date = '2025-03-28'  # only go through all of the data for one date at a time because there is a lot

# Where plots are saved
outerFolder = f"/data/QICK_data/run6/6transmon/ef_studies/Optimization/" + date + "/"
outerFolder_save_plots = f"/data/QICK_data/run6/6transmon/ef_studies/Optimization/" + date + "_plots/"

#Do you want to only process one file inside outerFolder?
process_one_file = True
#file_to_process = r"/data/QICK_data/run6/6transmon/ef_studies_tests/"+ date + "/Data_h5/SS_gef/2025-03-28_19-23-06_SS_gef_results_batch_4_Num_per_batch1.h5"

qubit_index = 1 # 0-5, if you want to select a qubit pick a number, if you don't, set to None
sigma_num = 2.0 # determines how big you want the radius of the circle to be (radius = sigma_num * sigma)

#------------------------------------------------Initialize the Plotting class------------------------------------------------
plotter = PlotAllRR(date, figure_quality, save_figs, fit_saved, signal, run_name, tot_num_of_qubits, outerFolder,
                  outerFolder_save_plots)

#------------------------------------To re-plot the g-e-f SSF plots, or any other data from the selected date-----------------------------------------------------
# plotter.run(plot_res_spec = False, plot_q_spec = False, plot_rabi = False, plot_ss = False,  ss_plot_gef = True, plot_t1 = False,
#             plot_t2r = False, plot_t2e = False)

#-------------------------------------Extract the data from the files or selected file inside outerFolder-------------------------------------------------
I_g, Q_g, I_e, Q_e, I_f, Q_f, ig_new, qg_new, ie_new, qe_new, if_new, qf_new, theta_ge, threshold_ge = plotter.load_plot_save_ss_gef(plot_ssf_gef, process_one_file, file_to_process, qubit_index)

#--------------------------------------Generate the plots that determine the location of the f-state in IQ space------------------------------
outerFolder_fstate = f"/data/QICK_data/run6/6transmon/ef_studies/Data/gef_SSF_fstate_IQspace/" # for identifying the f state in IQ space (ss_gef_fstate experiment)
Analysis = True
RR = False
date_analysis = date
batch_num = None
save_r = None
round_num = None
analysis_gef_SSF = GEF_SSF_ANALYSIS(outerFolder_fstate, qubit_index, Analysis, RR, date_analysis, round_num)
(line_point1, line_point2, center_e, radius_e, T, v, f_outside, line_point1_rot, line_point2_rot, center_e_rot, radius_e_rot, T_rot, v_rot, f_outside_rot
 )= analysis_gef_SSF.fstate_analysis_plot(I_g, Q_g, I_e, Q_e, I_f, Q_f, ig_new, qg_new, ie_new, qe_new, if_new, qf_new, theta_ge, threshold_ge, qubit_index, sigma_num)
