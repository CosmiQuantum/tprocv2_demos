from analysis_021_plot_allRR_noqick import QubitSpectroscopy
from qicklab.analysis import qspec, t1, ssf
from section_008_save_data_to_h5 import Data_H5
from analysis_014_ssf_temp_calcsandplots_cosmiqgpvm import TempCalcAndPlots
from expt_config import expt_cfg, list_of_all_qubits, FRIDGE
import glob
import re
import datetime
import ast
import os
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import math
import h5py
###################################################### Set These #######################################################
# Setup
QubitIndex = 0  # or whatever qubit you are analyzing
theta = 0
threshold = 0

# Initialize outputs
tot_num_of_qubits = 6 # Total number of qubits currently at QUIET
all_qspec_dates = [[] for _ in range(tot_num_of_qubits)]
all_qspec_freqs = [[] for _ in range(tot_num_of_qubits)]

all_ssf_qtemp_dates = [[] for _ in range(tot_num_of_qubits)]
all_ssf_qtemps = [[] for _ in range(tot_num_of_qubits)]

# # Other params
save_figs = False
fit_saved = False
signal = 'None'
run_number = 3 #starting from first run with qubits. Run 1 = run4a at quiet, run 2 = run5a at quiet, etc
figure_quality = 100 #ramp this up to like 500 for presentation plots
run_name = 'run6/6transmon'
path_saveplots = f"/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/RR_metrics/Plots/Qtemps_SSFmethod/Qtemps_vs_Time"
################################################## File Paths #################################################################
# paths = [
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_11-47-09",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_12-51-09",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_17-50-00",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_22-47-49",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-17_03-42-36",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-17_08-42-24",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy3/2025-04-17_12-28-37",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy3/2025-04-17_17-22-46",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy3/2025-04-17_22-16-39",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy3/2025-04-18_01-45-53",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy3/2025-04-18_06-40-55",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-18_11-59-33",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-18_16-56-58",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-18_21-51-13",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_02-45-41",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_07-39-57",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_12-34-26",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_17-48-44",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-19_22-43-02",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_03-37-50",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_08-32-36",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_13-26-47",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_18-25-13",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-20_23-25-04",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy4/2025-04-21_04-23-31",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy1/2025-04-21_10-17-14",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy1/2025-04-21_15-12-01",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy1/2025-04-21_20-09-53",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy1/2025-04-22_01-07-08",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy1/2025-04-22_06-04-35",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy2/2025-04-22_21-52-40",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy3/2025-04-23_08-50-38",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-23_11-20-17",
#     #"/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-23_14-46-57",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-23_18-13-57",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-23_21-40-34",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_01-06-04",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_04-33-07",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_07-56-53",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_11-18-03",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_14-41-03",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_18-04-49",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-24_21-31-17",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_00-54-28",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_04-20-54",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_07-46-50",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_11-10-03",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_14-34-10",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_17-57-10",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-25_21-26-15",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_00-54-11",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_04-21-21",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_07-55-49",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_11-24-18",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_14-49-55",
#     "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_18-15-55"
# ] # data folders
paths = ["/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy2/2025-04-16_11-47-09"]
################################################# Get all data ######################################################
Science_Qubits = [0,4]
freq_cache = {}
ig_new_cache = {}
timestamp_ssf_cache= {}
for full_path in paths:
    path = os.path.dirname(full_path)  # one level up from the dataset
    dataset = os.path.basename(full_path)  # just the '2025-04-16_11-47-09' part

    for QubitIndex in Science_Qubits: # We are only taking science data for qubits 1 and 5
        try:
            # --- Load QSpec ---
            qspec_obj = qspec(path, dataset, QubitIndex)
            qspec_dates, qspec_n, qspec_probe_freqs, qspec_I, qspec_Q = qspec_obj.load_all()
            qspec_freqs, qspec_errs, qspec_fwhms = qspec_obj.get_all_qspec_freq(qspec_probe_freqs, qspec_I, qspec_Q, qspec_n)

            # recreate the list of file–paths in the SAME order the helper used
            qspec_dir = os.path.join(path, dataset, qspec_obj.folder, "Data_h5", qspec_obj.expt_name)
            h5_files = sorted(os.listdir(qspec_dir))
            h5_paths = [os.path.join(qspec_dir, f) for f in h5_files]
            print('h5_files: ',h5_files)

            for i in range(qspec_n):
                freq_cache[(h5_paths[i], QubitIndex)] = qspec_freqs[i]
        except Exception as e:
            print(f"Skipped QSpec scan in {dataset} for Q{QubitIndex}: {e}")

        try:
            # --- Load SSF ---
            ssf_ge = ssf(path, dataset, QubitIndex)
            ssf_dates, ssf_n, I_g, Q_g, I_e, Q_e, fid, angles = ssf_ge.load_all()

            # recreate the list of SSF-file paths in the SAME order the helper used
            ssf_dir = os.path.join(path, dataset, ssf_ge.folder, "Data_h5", ssf_ge.expt_name)
            ssf_paths = [os.path.join(ssf_dir, f) for f in sorted(os.listdir(ssf_dir))]  # length==ssf_n
            print('ssf_paths: ', ssf_paths)
            # iterate through every round (file)
            for i in range(ssf_n):
                try:
                    ig_new, *_ = ssf_ge.get_ssf_in_round(I_g, Q_g, I_e, Q_e, i)
                except Exception as e:
                    print(f"rotate-Ig failed ({ssf_paths[i]}): {e}")
                    continue

                key = (ssf_paths[i], QubitIndex)
                ig_new_cache[key] = ig_new
                timestamp_ssf_cache[key] = ssf_dates[i]

        except Exception as e:
            print(f"Failed loading SSF for qubit {QubitIndex} from {full_path}: {e}")

########################################## Pair up Qspec_ge data and ssf_ge h5 files ###########################################
temps_class_obj = TempCalcAndPlots(figure_quality, tot_num_of_qubits, save_figs, path_saveplots)

# Collect every QSpec-GE and SSF-GE HDF5 path (down to each specific file name)
qspec_h5s = {q: [] for q in Science_Qubits}
ssf_h5s   = {q: [] for q in Science_Qubits}
# Match ssf and Qspec files by time stamps
for tdir in paths:
    # ---------- QSpec ----------
    for f in glob.glob(os.path.join(tdir, "Data_h5", "qspec_ge", "*_qspec_ge*_results_*.h5")):
        with h5py.File(f, "r") as h5:
            qi = int(next(k for k in h5.keys() if k.isdigit()))  # 0 or 4
        qspec_h5s[qi].append(f)

    # ---------- SSF ------------
    for f in glob.glob(os.path.join(tdir, "Data_h5", "ss_ge", "*_ss_ge*_results_*.h5")):
        with h5py.File(f, "r") as h5:
            qi = int(next(k for k in h5.keys() if k.isdigit()))
        ssf_h5s[qi].append(f)

pairs_by_qubit, lonely_qspec, lonely_ssf = temps_class_obj.pair_qspec_and_ssf(qspec_h5s, ssf_h5s, tolerance_seconds = 35)
print('pairs_by_qubit: ',pairs_by_qubit)
# Store relevant info for these pairs in a dictionary
pairs_info = {q: [] for q in Science_Qubits}
for q in Science_Qubits:
    for qspec_path, ssf_path in pairs_by_qubit.get(q, []):
        fq_key = (qspec_path, q)
        ss_key = (ssf_path,  q)
        if fq_key not in freq_cache or ss_key not in ig_new_cache:
            continue            # skip incomplete pair

        pairs_info[q].append({
            "qspec_path": qspec_path,
            "ssf_path"  : ssf_path,
            "qfreq_MHz" : freq_cache[fq_key],     # MHz
            "ig_new"   : ig_new_cache[ss_key],
            "data_timestamp" : timestamp_ssf_cache[ss_key] # unix-timestamps
        })

#-------------------------------------------- Calculate Temperatures ---------------------------------------------------
all_qubit_temps, all_qubit_times = temps_class_obj.run(pairs_info, limit_temp_k=0.8)
temps_class_obj.plot_all_qubits_scatter(all_qubit_temps, all_qubit_times, path_saveplots)