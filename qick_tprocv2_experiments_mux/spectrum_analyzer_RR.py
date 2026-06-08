import copy
import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15))  # need this so it saves absolutely everything returned from the classes
import datetime
import time
import logging
import visdom
import gc, copy
import time
#sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_016_const_res_tone import ResToneSpectrumAnalyzer
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_002_res_spec_ef import ResonanceSpectroscopyEF
from section_015_const_qubit_drive_tone import QubitToneSpectrumAnalyzer
from section_004_qubit_spec_ef import EFQubitSpectroscopy
from section_006_amp_rabi_ef import EF_AmplitudeRabiExperiment
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_011_qubit_temperatures_efRabipt3 import Temps_EFAmpRabiExperiment  # must be pt3 version, do not change
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
# from section_005_single_shot_ef import SingleShot_ef # Kester way
from section_005_single_shot_gef import SingleShot_ef # Arianna way: Fix for example Unmask
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from analysis_021_plot_allRR_noqick import PlotRR_noQick
from analysis_014_temp_calcsandplots_cosmiqgpvm import SSFTempCalcAndPlots

################################################ Run Configurations ####################################################
st = time.time()

n = 1 # number of rounds
use_iminuit_instead = True # for fitting, curve fit when False, iminuit when True
pre_optimize = False # ignore
freq_offset_steps = 10 # ignore
ssf_avgs_per_opt_pt = 5 # ignore
ef_res_sample_number = 1 # keep this as one
save_r = 1  # how many rounds to save after. KEEP THIS AS ONE!!!!!!!!
qubit_freqs_ef = [None] * 6 # don't change this!!! These get updated later on in the code.
number_of_qubits = 6 # total
figure_quality = 200

signal = 'None'  # 'I', or 'Q' depending on where the signal is (after optimization). Keep as None
save_figs = False  # save plots for everything as you go along the RR script?
live_plot = False  # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True  # fit the data here and save or plot the fits?
save_data_h5 = False  # save all of the data to h5 files?

verbose = True  # print everything to the console in real time, good for debugging, bad for memory
qick_verbose = True  # qick verbose prints the progress bar for each qick experiment as it is happening (the red bar that fills out as more experiment rounds/reps are being done)
debug_mode = False  # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script

thresholding = False  # use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false
unmask = True  # Do you want to use the unmasking feature to increase resonator gain? This may not apply to LOUD
save_shots_gerabi = False  # save IQ shots instead of averaged IQ data? for ge rabi
save_shots_efrabi = False  # NOT implemented yet in this experiment. If you want to use this add code block to ef rabi experiment.

Qs_to_look_at = [5] # only list the qubits you want to do the RR for

# Data saving info
run_name = 'run9c'
device_name = '6transmon'
substudy_txt_notes = ('Spectrum anlayzer measurements using the Qick Box. \n')

# set which of the following you'd like to run to 'True'
run_flags = {"long_tof": False, "long_qdrive": True}

# For 25dB DAC, 6/5/2026
res_gain = [0.80, 0.6556, 0.7808, 0.6115, 0.825, 0.8308]  # 0.8125,25dB, [0.8125, 0.836, 0.915, 0.6218, 0.95, 0.97], [0.825, 0.835, 0.915, 0.634, 0.95, 0.97]
freq_offsets = [0.0,0.0,0.0,0.0,0.0,0.0]  # -0.1500, -0.1286, -0.3000, -0.1556, 0.0, -0.0222

#DO NOT CHANGE THESE: They are flags to keep track of what happened in RR along the way
ef_res_any = False # did ef res spec run succesfully for any of the qubits?
ef_qspec_any = False # what about ef qspec?
rpm_any = False # and rabi population measurements?

# To save how long each measurement took for each qubit
meas_time_RR = {}

################################################ Data Saving Setup ##################################################
# Folders
study = 'spectrum_analyzer_meas' #qubit_checkouts
sub_study = 'Qick_Box'
data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if not os.path.exists(f"/data/QICK_data/{run_name}/"):
    os.makedirs(f"/data/QICK_data/{run_name}/")
if not os.path.exists(f"/data/QICK_data/{run_name}/{device_name}/"):
    os.makedirs(f"/data/QICK_data/{run_name}/{device_name}/")
studyFolder = os.path.join(f"/data/QICK_data/{run_name}/{device_name}/", study)
if not os.path.exists(studyFolder):
    os.makedirs(studyFolder)
subStudyFolder = os.path.join(studyFolder, sub_study)
if not os.path.exists(subStudyFolder):
    os.makedirs(subStudyFolder)

dataSetFolder = os.path.join(subStudyFolder, data_set)
optimizationFolder = os.path.join(dataSetFolder, 'optimization')
studyFolder = os.path.join(dataSetFolder, 'study_data')
studyDocumentationFolder = os.path.join(dataSetFolder, 'documentation')
subStudyDataFolder = os.path.join(dataSetFolder, 'study_data')
if not os.path.exists(studyDocumentationFolder):
    os.makedirs(studyDocumentationFolder)
if not os.path.exists(optimizationFolder):
    os.makedirs(optimizationFolder)
if not os.path.exists(subStudyDataFolder):
    os.makedirs(subStudyDataFolder)

file_path = os.path.join(studyDocumentationFolder, 'sub_study_notes.txt')
with open(file_path, "w", encoding="utf-8") as file:
    file.write(substudy_txt_notes)

################################################## Configure logging ###################################################
''' We need to create a custom logger and disable propagation like this
to remove the logs from the underlying qick from saving to the log file for RR'''

log_file = os.path.join(studyDocumentationFolder, "RR_script.log")
rr_logger = logging.getLogger("custom_logger_for_rr_only")
rr_logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

rr_logger.addHandler(file_handler)
rr_logger.propagate = False  # dont propagate logs from underlying qick package

####################################################### RR #############################################################
def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config', 'measurement_timestamp']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num', 'Recycled QFreq',
              'Exp Config', 'Syst Config', 'measurement_timestamp']

# initialize a simple list to store the qspec values in incase a fit fails
stored_qspec_list = [None] * tot_num_of_qubits

if live_plot:
    # Check if visdom is connected right away, otherwise, throw an error
    if not (viz := visdom.Visdom()).check_connection(timeout_seconds=5):
        raise RuntimeError("Visdom server not connected!, Type \'visdom\' into the command line and go to "
                           "http://localhost:8097/ on firefox")

# initialize a dictionary to store the values
res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)

batch_num = 0 # keep as zero
j = 0 # keep as zero
angles = []
while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        meas_time_RR[QubitIndex] = {}

        recycled_qfreq = False # don't change

        # keep these as False
        ef_res_spec_survived = False
        ef_qspec_survived = False

        # Get the config for this qubit
        DAC_attenuator1 = 15
        DAC_attenuator2 = 10
        experiment = QICK_experiment(optimizationFolder, DAC_attenuator1=DAC_attenuator1, DAC_attenuator2=DAC_attenuator2,
                                     qubit_DAC_attenuator1=5,
                                     qubit_DAC_attenuator2=4, ADC_attenuator=17,
                                     fridge=FRIDGE)  # ADC_attenuator MUST be above 16dB
        experiment.create_folder_if_not_exists(optimizationFolder)
        print("DAC atten: ", DAC_attenuator1 + DAC_attenuator2)

        # Mask out all other resonators except this one
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_gain_ef'] = res_gains

        ###################################################### long const tone: res channel #####################################################
        if run_flags["long_tof"]:
            prog_time = 1000000.0  # us; approx run time is prog_time x reps

            # This is part of readout/system config, so set before the experiment program is created.
            # This makes the resonator pulse length long.
            experiment.readout_cfg["res_length"] = prog_time

            long_tof = ResToneSpectrumAnalyzer(
                QubitIndex,
                studyDocumentationFolder,
                experiment,
                j,
                save_figs=save_figs,
                unmasking_resgain=unmask,
                qick_verbose=qick_verbose,
                sa_hold_time=prog_time,
                tof_freq_offset_MHz=1.0,  # old TOF behavior. Use 0.0 for exact res_freq_ge
            )

            long_tof.config["reps"] = 30000
            long_tof.config["soft_avgs"] = 1
            long_tof.config["relax_delay"] = 0.0

            long_tof.run()
            del long_tof
        ################################################## long const tone: qubit channel ##################################################
        if run_flags["long_qdrive"]:
            prog_time = 1000000.0 # approx run time is prog_time x reps
            experiment.qubit_cfg['qubit_length_ge'] = 15 # long constant pulse for SA measurements
            qubit_gains = [0.01, 0.011, 0.033, 0.021, 0.14, 0.08]
            this_gain= qubit_gains[QubitIndex]
            experiment.qubit_cfg['qubit_gain_ge'][QubitIndex] = this_gain

            # Optional: leave the frequency as the stored qubit frequency or override only the selected qubit for a test tone.
            # experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = 4229.89 # MHz

            long_qdrive = QubitToneSpectrumAnalyzer(QubitIndex, tot_num_of_qubits, studyDocumentationFolder,j,
                signal, save_figs, experiment=experiment, live_plot=live_plot, verbose=verbose, logger=rr_logger,
                unmasking_resgain=unmask, qubit_pulse_mode="periodic", qubit_sa_hold_time=prog_time)

            # Optional but recommended for SA measurement
            long_qdrive.config["reps"] = 1
            long_qdrive.config["rounds"] = 1

            long_qdrive.run()
            #experiment.soc.reset_gens() # to stop the pulse, re-run it with this line uncommented.
            del long_qdrive