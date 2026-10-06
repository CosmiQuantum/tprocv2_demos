import sys
import os
from copy import deepcopy

import numpy as np

np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import logging
import visdom
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_006p5_length_rabi_ge import LengthRabiExperiment
from section_005_single_shot_ge import SingleShot
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
Qs_to_look_at = [0]        # only list the qubits you want to do the RR for

# set which of the following you'd like to run to 'True'
run_flags = {"tof": False, "res_spec": True, "q_spec": True, "ss": True, "rabi": True,
             "t1": True, "t2r": True, "t2e": True}

#outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30/", str(datetime.date.today()))


if zero_qubit_drive_gain:
    topFolder_systematics = "/data/QICK_data/run6/6transmon/QZE/qubit_pulse_sent_gain_0"
    if not os.path.exists('/data/QICK_data/run6/6transmon/'): os.makedirs('/data/QICK_data/run6/6transmon/')
    dataFolder_systematics = os.path.join(topFolder_systematics, 'Data')
    optimizationFolder_systematics = os.path.join(topFolder_systematics, 'Optimization')
    documentationFolder_systematics = os.path.join(topFolder_systematics, 'Documentation')
    outerFolder_systematics = os.path.join(dataFolder_systematics, str(datetime.date.today()))

    optDocumentationFolder_systematics = os.path.join(optimizationFolder_systematics, 'opt_documentation')
    opt_notes_path_systematics = os.path.join(optDocumentationFolder_systematics, 'opt_notes.txt')
    studyDocumentationFolder_systematics = os.path.join(outerFolder_systematics, 'study_documentation')
    study_notes_path_systematics = os.path.join(studyDocumentationFolder_systematics, 'study_notes.txt')

    if not os.path.exists(topFolder_systematics): os.makedirs(topFolder_systematics)
    if not os.path.exists(dataFolder_systematics): os.makedirs(dataFolder_systematics)
    if not os.path.exists(optimizationFolder_systematics): os.makedirs(optimizationFolder_systematics)
    if not os.path.exists(documentationFolder_systematics): os.makedirs(documentationFolder_systematics)
    if not os.path.exists(outerFolder_systematics): os.makedirs(outerFolder_systematics)
    if not os.path.exists(studyDocumentationFolder_systematics): os.makedirs(studyDocumentationFolder_systematics)
    if not os.path.exists(optDocumentationFolder_systematics): os.makedirs(optDocumentationFolder_systematics)

topFolder = "/data/QICK_data/run6/6transmon/QZE/QZE_measurement" #+str(datetime.date.today())
if not os.path.exists('/data/QICK_data/run6/6transmon/'): os.makedirs('/data/QICK_data/run6/6transmon/')
dataFolder = os.path.join(topFolder, 'Data')
optimizationFolder = os.path.join(topFolder, 'Optimization')
documentationFolder = os.path.join(topFolder, 'Documentation')
outerFolder = os.path.join(dataFolder, str(datetime.date.today()))

optDocumentationFolder = os.path.join(optimizationFolder, 'opt_documentation')
opt_notes_path = os.path.join(optDocumentationFolder, 'opt_notes.txt')
studyDocumentationFolder = os.path.join(outerFolder, 'study_documentation')
study_notes_path = os.path.join(studyDocumentationFolder, 'study_notes.txt')

if not os.path.exists(topFolder): os.makedirs(topFolder)
if not os.path.exists(dataFolder): os.makedirs(dataFolder)
if not os.path.exists(optimizationFolder): os.makedirs(optimizationFolder)
if not os.path.exists(documentationFolder): os.makedirs(documentationFolder)
if not os.path.exists(outerFolder): os.makedirs(outerFolder)
if not os.path.exists(studyDocumentationFolder): os.makedirs(studyDocumentationFolder)
if not os.path.exists(optDocumentationFolder): os.makedirs(optDocumentationFolder)


################################################ optimization outputs ##################################################
# Optimization parameters for resonator spectroscopy
res_leng_vals = [4, 5, 5, 4, 4.5, 9] # From RR March 22
res_gain = [1, 1, 0.6, 0.6, 1, 0.6]
freq_offsets = [-0.18, -0.18, -0.18, -0.28, -0.18, -0.05]

################################################## Configure logging ###################################################
''' We need to create a custom logger and disable propagation like this 
to remove the logs from the underlying qick from saving to the log file for RR'''

log_file = os.path.join(outerFolder, "RR_script.log")
rr_logger = logging.getLogger("custom_logger_for_rr_only")
rr_logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

rr_logger.addHandler(file_handler)
rr_logger.propagate = False  #dont propagate logs from underlying qick package

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



#Vary the readout pulse height, (number of photons int he resonator) and look at rabi (fig 2 here
# https://iopscience.iop.org/article/10.1088/1367-2630/17/6/063035/pdf)
batch_num=0
j = 0

for QubitIndex in Qs_to_look_at:
    experiment = QICK_experiment(outerFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10,
                                 fridge=FRIDGE)
    updated_qubit_gain = 1 #experiment.qubit_cfg['qubit_gain_ge'][QubitIndex] * 20
    ####################################### Tune up res spec and qspec once #######################################
    # Mask out all other resonators except this one
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    experiment.qubit_cfg['qubit_gain_ge'][
        QubitIndex] = 0.05  # turn it down for this qspec finding to lower err bars and minimize broadening

    ################################ Do Res spec once per qubit and store the value ####################################
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, optimizationFolder, 0,
                                     save_figs=True, experiment=experiment, verbose=verbose,
                                     logger=rr_logger, qick_verbose=True)

    res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()

    experiment.readout_cfg['res_freq_ge'] = res_freqs
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

    saver_res = Data_H5(optimizationFolder, res_data, 0, save_r)  # save
    saver_res.save_to_h5('Res')
    del saver_res
    del res_data
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)  # initialize again to a blank for saftey
    del res_spec

    ############ Qubit Spec ##############
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)

    q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, documentationFolder, 0,
                               signal, save_figs=True, experiment=experiment,
                               live_plot=live_plot, verbose=verbose, logger=rr_logger,
                               qick_verbose=True, increase_reps=True, increase_reps_to=500)
    (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit,
     qubit_freq, sys_config_qspec, fwhm) = q_spec.run(return_fwhm=True)
    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
    experiment.qubit_cfg['qubit_freq_ge_starked'][QubitIndex] = float(
        qubit_freq)  # update this one too incase it isnt updated later
    experiment.qubit_cfg['fwhm_w01'] = fwhm
    stored_qspec = float(qubit_freq)
    rr_logger.info(f"Tune-up: Qubit {QubitIndex + 1} frequency: {stored_qspec}")
    del q_spec

    qspec_data[QubitIndex]['Dates'][0] = (
        time.mktime(datetime.datetime.now().timetuple()))
    qspec_data[QubitIndex]['I'][0] = qspec_I
    qspec_data[QubitIndex]['Q'][0] = qspec_Q
    qspec_data[QubitIndex]['Frequencies'][0] = qspec_freqs
    qspec_data[QubitIndex]['I Fit'][0] = qspec_I_fit
    qspec_data[QubitIndex]['Q Fit'][0] = qspec_Q_fit
    qspec_data[QubitIndex]['Round Num'][0] = 0
    qspec_data[QubitIndex]['Batch Num'][0] = 0
    qspec_data[QubitIndex]['Recycled QFreq'][0] = False  # no rr so no recycling here
    qspec_data[QubitIndex]['Exp Config'][0] = expt_cfg
    qspec_data[QubitIndex]['Syst Config'][0] = sys_config_qspec

    saver_qspec = Data_H5(optimizationFolder, qspec_data, 0, save_r)
    saver_qspec.save_to_h5('QSpec')
    del saver_qspec
    del qspec_data

    # reinitialize
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)

    ################## length rabi test ################
    from section_006p5_length_rabi_ge import LengthRabiExperiment

    experiment.qubit_cfg['qubit_gain_ge'][
        QubitIndex] = 0.05
    len_rabi = LengthRabiExperiment(QubitIndex, tot_num_of_qubits, '/data/QICK_data/run6/6transmon/test/', 0,
                                    signal, save_figs=True, experiment=experiment,
                                    live_plot=live_plot,
                                    increase_qubit_reps=increase_qubit_reps,
                                    qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                    multiply_qubit_reps_by=multiply_qubit_reps_by,
                                    verbose=verbose, logger=rr_logger,
                                    qick_verbose=True)
    (rabi_I, rabi_Q, rabi_gains, rabi_fit, stored_pi_amp, sys_config_rabi) = len_rabi.run()


    #pulse_gains=np.linspace(0.1, 1, 10)
    pulse_gains = [1]

    for gain in pulse_gains:

        j += 1


        ############################################## Start Rabi experiment ###########################################

        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        # tof = TOFExperiment(QubitIndex, outerFolder, experiment, j, save_figs)
        # tof.run()

        if zero_qubit_drive_gain:
            ############################# Do Rabi systematics and store the value #####################################
            experiment.qubit_cfg['qubit_gain_ge'][QubitIndex] = 0

            rabi = LengthRabiExperiment(QubitIndex, tot_num_of_qubits, outerFolder_systematics, j, signal, save_figs,
                                           experiment=experiment, live_plot=live_plot,
                                           increase_qubit_reps=increase_qubit_reps,
                                           qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                           multiply_qubit_reps_by=multiply_qubit_reps_by,
                                           verbose=True, QZE=True, zeno_pulse_gain=gain)

            (rabi_I, rabi_Q, rabi_magnitude, rabi_gains, rabi_fit, pi_amp,
             sys_config_rabi) = rabi.run_QZE(constant_zeno_pulse=constant_zeno_pulse) #thresholding not implemented
            #rabi.run_QZE()

            print('Moving to zeno measurement')
            #rabi.run_QZE(thresholding=thresholding)
            #rabi.run_oscilliscope_zeno(thresholding=thresholding)

            del rabi

            ############################################### Collect Results ################################################

            # ---------------------Collect Rabi Results----------------
            rabi_data[QubitIndex]['Dates'][0] = (
                time.mktime(datetime.datetime.now().timetuple()))
            rabi_data[QubitIndex]['I'][0] = rabi_I
            rabi_data[QubitIndex]['Q'][0] = rabi_Q
            rabi_data[QubitIndex]['Mag'][0] = rabi_magnitude
            rabi_data[QubitIndex]['Gains'][0] = rabi_gains
            rabi_data[QubitIndex]['Fit'][0] = rabi_fit
            rabi_data[QubitIndex]['Round Num'][0] = j
            rabi_data[QubitIndex]['Batch Num'][0] = batch_num
            rabi_data[QubitIndex]['Exp Config'][0] = expt_cfg
            rabi_data[QubitIndex]['Syst Config'][0] = sys_config_rabi

            # --------------------------save Rabi-----------------------
            saver_rabi = Data_H5(outerFolder_systematics, rabi_data, 0, save_r)
            saver_rabi.save_to_h5('Rabi_QZE')
            del saver_rabi
            del rabi_data

            # reset all dictionaries to none for safety
            rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)


        ################################### Do Rabi and store the value #####################################
        experiment.qubit_cfg['qubit_gain_ge'][QubitIndex] = updated_qubit_gain  # ramp it up to see many oscilations

        exp = deepcopy(experiment) #before updating for qze on ch 7, use for qspec
        rabi = LengthRabiExperiment(QubitIndex, tot_num_of_qubits, outerFolder, j, signal, save_figs,
                                    experiment=experiment, live_plot=live_plot,
                                    increase_qubit_reps=increase_qubit_reps,
                                    qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                    multiply_qubit_reps_by=multiply_qubit_reps_by,
                                    verbose=True, QZE=True, zeno_pulse_gain=gain)

        if wait_for_res_ring_up:
            (rabi_I_QZE, rabi_Q_QZE, rabi_magnitude_QZE, rabi_gains_QZE, rabi_fit_QZE, pi_amp_QZE,
             sys_config_rabi_QZE) = rabi.run_QZE_one_starked_qfreq(constant_zeno_pulse=constant_zeno_pulse,
                                                                   adapt_qubit_freq=adapt_starked_qubit_freq,
                                                                   wait_for_res_ring_up=wait_for_res_ring_up,
                                                                   exp=exp, optimizationFolder=optimizationFolder,
                                                                   hold_ground=True, three_pulse_binary=True)  # thres# holding not implemented
        else:
            (rabi_I_QZE, rabi_Q_QZE, rabi_magnitude_QZE, rabi_gains_QZE, rabi_fit_QZE, pi_amp_QZE,
             sys_config_rabi_QZE) = rabi.run_QZE(constant_zeno_pulse=constant_zeno_pulse, adapt_qubit_freq=adapt_starked_qubit_freq,
                                                 wait_for_res_ring_up = wait_for_res_ring_up, exp=exp)  # thresholding not implemented

        # rabi.run_QZE()

        print('Moving to next zeno gain loop')
        # rabi.run_QZE(thresholding=thresholding)
        # rabi.run_oscilliscope_zeno(thresholding=thresholding)

        del rabi

        ############################################### Collect Results ################################################

        # ---------------------Collect Rabi Results----------------
        rabi_data[QubitIndex]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
        rabi_data[QubitIndex]['I'][0] = rabi_I_QZE
        rabi_data[QubitIndex]['Q'][0] = rabi_Q_QZE
        rabi_data[QubitIndex]['Mag'][0] = rabi_magnitude_QZE
        rabi_data[QubitIndex]['Gains'][0] = rabi_gains_QZE
        rabi_data[QubitIndex]['Fit'][0] = rabi_fit_QZE
        rabi_data[QubitIndex]['Round Num'][0] = j
        rabi_data[QubitIndex]['Batch Num'][0] = batch_num
        rabi_data[QubitIndex]['Exp Config'][0] = expt_cfg
        rabi_data[QubitIndex]['Syst Config'][0] = sys_config_rabi_QZE

        # --------------------------save Rabi-----------------------
        saver_rabi = Data_H5(outerFolder, rabi_data, 0, save_r)
        saver_rabi.save_to_h5('Rabi_QZE')
        del saver_rabi
        del rabi_data

        # reset all dictionaries to none for safety
        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)

    del experiment





