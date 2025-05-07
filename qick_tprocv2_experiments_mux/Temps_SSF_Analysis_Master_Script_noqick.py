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
theta = 0
threshold = 0
tot_num_of_qubits = 6 # Total number of qubits currently at QUIET
save_figs = False
fit_saved = False
signal = 'None'
run_number = 3 #starting from first run with qubits. Run 1 = run4a at quiet, run 2 = run5a at quiet, etc
figure_quality = 200 #ramp this up to like 500 for presentation plots
run_name = 'run6/6transmon'
path_saveplots = f"/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/RR_metrics/Plots/Qtemps_SSFmethod/Qtemps_vs_Time"

################################################## File Paths #################################################################
# paths = ["/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_00-54-11",
#         "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_04-21-21",
#         "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_07-55-49",
#         "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_11-24-18",
#         "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_14-49-55",
#         "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_18-15-55",
#         "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_21-48-38",
#         "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-27_01-16-44",
#         "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-27_04-48-04",
#         "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-27_08-17-01",
#         "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-27_11-46-15",
#         "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-27_18-46-03",
#         "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-27_22-13-30"]
paths = ["/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_on_substudy4/2025-04-26_00-54-11"]

################################################# Load all data ##############################################################
Science_Qubits = [4]
analysis_flags = {"Qtemps_vs_time": False, "Gaussian_Fits": True}

all_qspec_dates = [[] for _ in range(tot_num_of_qubits)]
all_qspec_freqs = [[] for _ in range(tot_num_of_qubits)]

all_ssf_qtemp_dates = [[] for _ in range(tot_num_of_qubits)]
all_ssf_qtemps = [[] for _ in range(tot_num_of_qubits)]

freq_cache = {} #for qubit freqs
ig_new_cache = {} #for roated IQ data (SSF)
timestamp_ssf_cache= {} #for ssf data time stamps (qubit temperature time stamps)
fid_cache = {} #for ssf ge fidelities
threshold_cache  = {} #for ssf thresholds

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

            for i in range(qspec_n):
                freq_cache[(h5_paths[i], QubitIndex)] = qspec_freqs[i]
        except Exception as e:
            print(f"Skipped QSpec scan in {dataset} for Q{QubitIndex}: {e}")

        try:
            # --- Load SSF ---
            ssf_ge = ssf(path, dataset, QubitIndex)
            ssf_dates, ssf_n, I_g, Q_g, I_e, Q_e, _ , _ = ssf_ge.load_all()

            # recreate the list of SSF-file paths in the SAME order the helper used
            ssf_dir = os.path.join(path, dataset, ssf_ge.folder, "Data_h5", ssf_ge.expt_name)
            ssf_paths = [os.path.join(ssf_dir, f) for f in sorted(os.listdir(ssf_dir))]  # length==ssf_n

            # iterate through every round (file)
            for i in range(ssf_n):
                try:
                    theta, thresh, fidelity, ig_new, *_= ssf_ge.get_ssf_in_round(I_g, Q_g, I_e, Q_e, i)
                except Exception as e:
                    print(f"rotate-Ig failed ({ssf_paths[i]}): {e}")
                    continue

                key = (ssf_paths[i], QubitIndex)
                ig_new_cache[key] = ig_new
                timestamp_ssf_cache[key] = ssf_dates[i]
                print('fid: ',fidelity)
                # print('thresh ', threshold)
                fid_cache[key] = float(fidelity)
                threshold_cache[key] = float(thresh)


        except Exception as e:
            print(f"Failed loading SSF for qubit {QubitIndex} from {full_path}: {e}")

#Organize files by type and qubit index after loading all the data
qspec_h5s = {q: [] for q in Science_Qubits}
for (path, qidx) in freq_cache.keys():
    qspec_h5s[qidx].append(path)

ssf_h5s = {q: [] for q in Science_Qubits}
for (path, qidx) in ig_new_cache.keys():
    ssf_h5s[qidx].append(path)

########################################## Pair up Qspec_ge data and ssf_ge h5 files ###########################################
temps_class_obj = TempCalcAndPlots(figure_quality, tot_num_of_qubits, save_figs, path_saveplots)
pairs_by_qubit, lonely_qspec, lonely_ssf = temps_class_obj.pair_qspec_and_ssf(qspec_h5s, ssf_h5s, tolerance_seconds = 10)

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
            "data_timestamp" : timestamp_ssf_cache[ss_key].timestamp(), # unix-timestamps
            "fid": fid_cache[ss_key],
            "ssf_threshold": threshold_cache[ss_key]
        })

############################################## Calculate Temperatures ##################################################
all_qubit_temps, all_qubit_times, fit_results  = temps_class_obj.run(pairs_info, limit_temp_k=0.8, use_ssf_thresh_only = False, fallback_to_threshold = True)

######################################### Temperatures vs Time Scatter Plot #############################################
if analysis_flags["Qtemps_vs_time"]:
    temps_class_obj.plot_all_qubits_scatter(all_qubit_temps, all_qubit_times, path_saveplots)

############################################# Check Gaussian Fits (Plots) #############################################
if analysis_flags["Gaussian_Fits"]:
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
            uses_thr = rec.get("used_ssf_thresh_only", False) #Looks up the key "uses_ssf_data_threshold" in the result dictionary. If it’s missing (or False), the code did not use the SSF threshold.
            used_fb = rec.get("used_fallback_method", False) # similar check for fall back option

            if not uses_thr and not used_fb: #if uses_ssf_data_threshold was False or used_fallback_method was False / not used
                # double-gaussian plot
                temps_class_obj.plot_gaussians_qtemps(
                    q_key,
                    made_on_folder,
                    rec["fid"],
                    rec["ig_new"],
                    rec["ground_data"],
                    rec["excited_data"],
                    rec["ground_gaussian"],
                    rec["excited_gaussian"],
                    rec["pop_threshold"],
                    rec["temperature_mK"],
                    rec["dataset"],
                    rec["weights"],
                    rec["covariances"],
                    rec["means"])
            else:
                # fallback/threshold‐only plot
                temps_class_obj.plot_threshold_split(q_key, rec, made_on_folder)