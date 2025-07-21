import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
# from analysis_001_plot_all_RR_h5 import PlotAllRR
from analysis_021_plot_allRR_noqick import PlotRR_noQick
import os

plot_ssf_gef = False # Do you want to re-plot g-e-f SSF data and save the plots?
figure_quality = 200
save_figs = True # If you are running plotter.run, do you want to save all of the plots for the chosen experiment?
fit_saved = False # Not used here, set to false
signal = 'None' # Do not change
run_name = 'run6/6transmon/'

#------------------------------------------------ Load all the data -------------------------------------------------
# For quiet pc
# date = '2025-04-12'  # only go through all of the data for one date at a time because there is a lot
# outerFolder = f"/data/QICK_data/run6/6transmon/ef_studies/QubitTemps_efRabi_method/{date}/Optimization/Round_Robin_mode"
# outerFolder_qtemps_data = f"/data/QICK_data/run6/6transmon/ef_studies/QubitTemps_efRabi_method/{date}/Study_Data"
# outerFolder_qtemps_plots = os.path.join(outerFolder_qtemps_data, "analysis_plots")

# For local pc, Arianna
# date = '2025-04-12'  # only go through all of the data for one date at a time because there is a lot
# outerFolder = f"C:/Users/Arianna/Documents/Grad/Research/CosmicQ/Python/QUIET/Run6/QubitTemps_efRabi_method/{date}/Optimization/Round_Robin_mode"
# outerFolder_qtemps_data = f"C:/Users/Arianna/Documents/Grad/Research/CosmicQ/Python/QUIET/Run6/QubitTemps_efRabi_method/{date}/Study_Data"
# outerFolder_qtemps_plots = os.path.join(outerFolder_qtemps_data, "analysis_plots_localpc")

# For analysis at cosmiqgpvm02
target_dates = [
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

# --- Walking through all the subdirectories ---
filter_keywords = ['source_off', 'source_on']
combined_qtemp_data = [] # list of results from different .h5 files
for root, dirs, files in os.walk(base_dir):
    for d in dirs:
        full_path = os.path.join(root, d)
        # Match folders like '2025-04-16_11-47-09' based on prefix date
        if any(d.startswith(date) for date in target_dates) and len(d) >= 19 and any(keyword in full_path for keyword in filter_keywords): # also checks if path includes each keyword (source_off or source_on)
            optimization_path = os.path.join(full_path, "optimization")
            if os.path.isdir(optimization_path):
                date_string = d[:10]  # Extract 'YYYY-MM-DD'
                print(f"Analyzing: {optimization_path}")

                outerFolder = optimization_path #RR data (g-e Qspec) folder path before Data_h5
                outerFolder_qtemps_data = optimization_path #Qubit temps data folder path before Data_h5

                if not os.path.exists(outerFolder): os.makedirs(outerFolder)
                if not os.path.exists(outerFolder_qtemps_data): os.makedirs(outerFolder_qtemps_data)

                #---------------------------------------- Initialize the PlotRR_noQick class ------------------------------------------------
                # For RR plots
                # outerFolder_qtemps_plots = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/Plots/PlotRR"

                # For Analysis
                outerFolder_qtemps_plots = "/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/QTemperatures/Plots/params_vs_time"

                os.makedirs(outerFolder_qtemps_plots, exist_ok=True)
                plotter = PlotRR_noQick(date_string, figure_quality, save_figs, fit_saved, signal, run_name, tot_num_of_qubits, outerFolder,
                                  outerFolder_qtemps_plots, outerFolder_qtemps_data)

                #------------------------------------To re-plot the RPM plots, or any other data from the selected date-----------------------------------------------------
                # plotter.run(plot_res_spec = False, plot_q_spec = False, plot_rabi = False, plot_ss = False,  ss_plot_gef = False, plot_t1 = False,
                #             plot_t2r = False, plot_t2e = False, plot_rabis_Qtemps = True)

                #---------------------------------------- Load data and append to list spanning multiple dates --------------------------------------------------
                qtemp_data = plotter.load_plot_save_rabis_Qtemps(list_of_all_qubits)
                combined_qtemp_data.extend(qtemp_data)


# ----------------------------------------------- Qubit temperatures vs time ----------------------------------------------------
plotter.plot_qubit_temperatures_vs_time_RPMs(combined_qtemp_data, restrict_time_xaxis = False, plot_extra_event_lines = False, rad_events_plot_lines = False)

# ----------------------------------------------- Histograms of Qubit temperatures -----------------------------------------------
plotter.plot_qubit_temperature_histograms_RPMs(combined_qtemp_data)

# ----------------------------------------------- Excited state populations (P_e) vs time ----------------------------------------
plotter.plot_qubit_pe_vs_time_RPMs(combined_qtemp_data)

# ----------------------------------------------- Qubit temp and P_e vs time in the same plot ------------------------------------
plotter.plot_qubit_temp_and_pe_vs_time_RPMs(combined_qtemp_data)

# ----------------------------------------- Qubit temp, P_e, and g-e qubit freq vs time in the same plot --------------------------
plotter.plot_qubit_temp_pe_freq_vs_time_RPMs(combined_qtemp_data)