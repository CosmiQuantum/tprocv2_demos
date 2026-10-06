import sys
import os
from copy import deepcopy
import copy
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import logging
import visdom
sys.path.append(os.path.abspath("/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_002p5_res_centers_vs_time_plots import KappaPunchOutMeasurement
from section_008_save_data_to_h5 import Data_H5
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE

################################################ Run Configurations ####################################################
zero_qubit_drive_gain = False
constant_zeno_pulse = True
adapt_starked_qubit_freq = False
wait_for_res_ring_up = True
n= 1
save_r = 1                           # how many rounds to save after
signal = 'None'                      # 'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization
save_figs = True                     # save plots for everything as you go along the RR script?
live_plot = False                    # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = False                     # fit the data here and save or plot the fits?
save_data_h5 = True                  # save all of the data to h5 files?
verbose = True                       # print everything to the console in real time, good for debugging, bad for memory
debug_mode = True                    # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script
thresholding = False                 # use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false
increase_qubit_reps = False          # if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0       # only has impact if previous line is True
multiply_qubit_reps_by = 2           # only has impact if the line two above is True
Qs_to_look_at = [0,1,2,3,4,5]        # only list the qubits you want to do the RR for
study = 'thomas_punch_out_kappa_data_for_simulation'
sub_study = 'kappa_punch_out'
substudy_txt_notes = ('Taking kappa and punch out data for Thomas Roth group to use for simulation')
# substudy_txt_notes = ('making a qubit frequency vs qubit pulse length sweep for a given zeno pulse gain. This will tell me '
#                       'if i can use a singe value for the starked qubit frequency for every rabi value, or if i need to '
#                       'update it for every point. My prediction is that as long as I wait for the resonator ring up before '
#                       'doing the qubit pulse this should be constant on average across the x axis. ill try for a few different zeno gains.')

# set which of the following you'd like to run to 'True'
run_flags = {"tof": False, "res_spec": True, "q_spec": True, "ss": True, "rabi": True,
             "t1": True, "t2r": True, "t2e": True}

#Folders
if not os.path.exists("/data/QICK_data/run7/"):
    os.makedirs("/data/QICK_data/run7/")
if not os.path.exists("/data/QICK_data/run7/6transmon/"):
    os.makedirs("/data/QICK_data/run7/6transmon/")
studyFolder = os.path.join("/data/QICK_data/run7/6transmon/", study)
if not os.path.exists(studyFolder):
    os.makedirs(studyFolder)
subStudyFolder = os.path.join(studyFolder, sub_study)
if not os.path.exists(subStudyFolder):
    os.makedirs(subStudyFolder)

formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
dataSetFolder = os.path.join(subStudyFolder, formatted_datetime)
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

#Logging
log_file = os.path.join(studyDocumentationFolder, "thomas_punch_out_kappa_data_for_simulation.log")
rr_logger = logging.getLogger("custom_logger_for_rr_only")
rr_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
rr_logger.addHandler(file_handler)
rr_logger.propagate = False

################################################ optimization outputs ##################################################
# Optimization parameters for resonator spectroscopy
res_leng_vals = [5.5, 12, 6.0, 6.5, 5.0, 6.0]
res_gain = [0.9, 0.95, 0.78, 0.58, 0.95, 0.57]
freq_offsets = [-0.1, 0.2, -0.1, -0.4, -0.1, -0.1]

####################################################### RR #############################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num','Recycled QFreq',
              'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Mag', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
#initialize a simple list to store the qspec values in incase a fit fails
stored_qspec_list = [None] * tot_num_of_qubits

if live_plot:
    # Check if visdom is connected right away, otherwise, throw an error
    if not (viz := visdom.Visdom()).check_connection(timeout_seconds=5):
        raise RuntimeError("Visdom server not connected!, Type \'visdom\' into the command line and go to "
                           "http://localhost:8097/ on firefox")



############################## Get first the res, qubit freqs #######################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}


res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)

batch_num=0
j = 0

for QubitIndex in Qs_to_look_at:
    experiment = QICK_experiment(optimizationFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10,
                                 fridge=FRIDGE)
    updated_qubit_gain = 0.05 #experiment.qubit_cfg['qubit_gain_ge'][QubitIndex] * 20
    experiment.qubit_cfg['qubit_gain_ge'][
        QubitIndex] = updated_qubit_gain

    # Mask out all other resonators except this one
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    ################################ Do Res spec punch out test ####################################
    gains = np.linspace(0,0.6,7)
    for gain in gains:
        exp_copy=deepcopy(experiment)
        res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
        res_spec = KappaPunchOutMeasurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, 0,
                                         save_figs=True, experiment=exp_copy, verbose=verbose,
                                         logger=rr_logger, qick_verbose=True)

        res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run(gain)

        exp_copy.readout_cfg['res_freq_ge'] = res_freqs
        rr_logger.info(f"ResSpec for qubit {QubitIndex}: {res_freqs}")

        res_data[QubitIndex]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
        res_data[QubitIndex]['freq_pts'][0] = freq_pts
        res_data[QubitIndex]['freq_center'][0] = freq_center
        res_data[QubitIndex]['Amps'][0] = amps
        res_data[QubitIndex]['Found Freqs'][0] = res_freqs
        res_data[QubitIndex]['Batch Num'][0] = 0
        res_data[QubitIndex]['Exp Config'][0] = expt_cfg
        res_data[QubitIndex]['Syst Config'][0] = sys_config_rspec

        saver_res = Data_H5(studyFolder, res_data, 0, save_r)  # save
        gain_str = str(gain).replace('.','p')
        saver_res.save_to_h5('Res', save_dataset_clean=True, additional_title=f'gain_value_{gain_str}')
        del saver_res
        del res_data
        res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)  # initialize again to a blank for saftey
        del res_spec
        del exp_copy