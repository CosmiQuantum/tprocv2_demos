import sys
import os
import numpy as np
import datetime
import time
import logging
import gc
import matplotlib.pyplot as plt

np.set_printoptions(threshold=int(1e15))
sys.path.append(os.path.abspath("/home/auxuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
sys.path.append(os.path.abspath("/home/auxuser/Documents/GitHub/")) #need this to run from terminal
from tprocv2_demos.qick_tprocv2_experiments_mux.socProxy import device


# Import experiments and configurations
from res_spec_jcsrun7 import ResonanceSpectroscopy
from resonator_drive_jcrun8 import ResonatorDrive
from system_config import QICK_experiment
from expt_config import expt_cfg, FRIDGE
from section_008_save_data_to_h5 import Data_H5

################################################
# Run Configurations and Optimization Params
################################################
save_r = 1  # how many rounds to save after
save_figs = True  # whether to save plots
fit_data = True  # fit data during the run?
save_data_h5 = True  # save data to h5 files?
verbose = True  # verbose output
debug_mode = True  # if True, errors will stop the run immediately
temperature = 0.013, #0.011 #base temp in kelvin for record-keeping
num_rounds = 1
threading = False

run = "run8"
study = 'debugging'
sub_study = 'drive'
substudy_txt_notes = 'collecting timestream data'
drive = [1]
sensor = [6]
sensor_list = [0,2,3,4,5,6,7]
drive_gain = 0.1
sensor_gain = 1.0
resonator_list = sensor + drive # list of resonators to process
print(resonator_list)

# Set which experiments to run
run_flags = {"rspec": True, "timestream": True}

# Dictionaries
def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in qs}

rspec_keys = ['Dates', 'temperature', 'freq_sweep', 'gain_sweep', 'I', 'Q','fR','Ql','Qi','Qc','Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
drive_keys = ['Dates','drive_idx','sensor_idx','drive_freq','sensor_freq','timestep','Round Num', 'Batch Num','Exp Config','Syst Config']

#Folders
if not os.path.exists(f"/data/QICK_data/{run}/"):
    os.makedirs(f"/data/QICK_data/{run}/")
if not os.path.exists(f"/data/QICK_data/{run}/{device}/"):
    os.makedirs(f"/data/QICK_data/{run}/{device}/")
studyFolder = os.path.join(f"/data/QICK_data/{run}/{device}/", study)
if not os.path.exists(studyFolder):
    os.makedirs(studyFolder)
subStudyFolder = os.path.join(studyFolder, sub_study)
if not os.path.exists(subStudyFolder):
    os.makedirs(subStudyFolder)

file_path = os.path.join(subStudyFolder, 'sub_study_notes.txt')
with open(file_path, "w", encoding="utf-8") as file:
    file.write(substudy_txt_notes)

#Logging
log_file = os.path.join(subStudyFolder, "rspec_timestream_script.log")
rr_logger = logging.getLogger("custom_logger_for_rr_only")
rr_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
rr_logger.addHandler(file_handler)
rr_logger.propagate = False

formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
dataSetFolder = os.path.join(subStudyFolder, formatted_datetime)
optimizationFolder = os.path.join(dataSetFolder, 'optimization')
studyFolder = os.path.join(dataSetFolder, 'study_data')
studyDocumentationFolder = os.path.join(dataSetFolder, 'documentation')
study_notes_path = os.path.join(studyDocumentationFolder, 'study_notes.txt')

experiment = QICK_experiment(
    dataSetFolder,
    DAC_attenuator1=5,
    DAC_attenuator2=10,
    ADC_attenuator=10,
    fridge=FRIDGE
)
experiment.create_folder_if_not_exists(dataSetFolder)
experiment.create_folder_if_not_exists(optimizationFolder)
experiment.create_folder_if_not_exists(studyFolder)
experiment.create_folder_if_not_exists(studyDocumentationFolder)
with open(study_notes_path, "w", encoding="utf-8") as file:
    file.write('Study Notes:')

fig, ax = plt.subplots(4,2)
fig.suptitle(f"drive resonator R{drive}, sensor gain={sensor_gain}")
ax_1 = 0
ax_0 = 0
for s in sensor_list:
    sensor = [s]
    resonator_list = sensor + drive

    for resonator in resonator_list:
        current_resonator = [resonator]

        ## find resonator frequencies
        if run_flags["rspec"]:
            for ResonatorIndex in current_resonator:
                rspec_data = create_data_dict(rspec_keys, save_r, current_resonator)
                timestamp_rspec = time.mktime(datetime.datetime.now().timetuple())

                if ResonatorIndex == sensor[0]:
                    r_spec = ResonanceSpectroscopy(ResonatorIndex, len(current_resonator), studyDocumentationFolder, 0,
                               save_figs=save_figs, experiment=experiment,
                               verbose=verbose, logger=rr_logger, gain=sensor_gain)
                    freq_sweep, I, Q, gain_sweep, sys_config = r_spec.run()
                    #r_spec.plot_raw(freq_sweep,I,Q)
                    if fit_data is True:
                        try:
                            fR, Ql, Qi, Qc = r_spec.DCM_fit(freq_sweep, I, Q, gain_sweep)
                            experiment.readout_cfg["res_freq"][ResonatorIndex] = float(fR[len(gain_sweep)-1])
                            print(f"R{ResonatorIndex} fR = {experiment.readout_cfg['res_freq'][ResonatorIndex]} MHz")
                        except:
                            print("DCM fit failed, continuing")
                            fR = 0
                            Ql = 0
                            Qi = 0
                            Qc = 0
                    else:
                        Ql = 0
                        Qi = 0
                        Qc = 0
                        fR = 0

                if ResonatorIndex == drive[0]:
                    r_spec = ResonanceSpectroscopy(ResonatorIndex, len(current_resonator), studyDocumentationFolder, 0,
                                               save_figs=save_figs, experiment=experiment,
                                               verbose=verbose, logger=rr_logger, dac='fsgen', gain=drive_gain)
                    freq_sweep, I, Q, gain_sweep, sys_config = r_spec.run()
                    amp_fit, fR, fwhm, fit_err = r_spec.plot_fsgen(freq_sweep,I,Q)
                    Ql = 0
                    Qi = 0
                    Qc = 0
                    experiment.readout_cfg["res_freq"][ResonatorIndex] = float(fR)
                    print(f"R{ResonatorIndex} fR = {experiment.readout_cfg['res_freq'][ResonatorIndex]} MHz")


                if save_data_h5:
                    rspec_data[ResonatorIndex]['Dates'][0] = timestamp_rspec
                    rspec_data[ResonatorIndex]['Batch Num'][0] = 0
                    rspec_data[ResonatorIndex]['Round Num'][0] = 0
                    rspec_data[ResonatorIndex]['temperature'][0] = temperature
                    rspec_data[ResonatorIndex]['I'][0] = I
                    rspec_data[ResonatorIndex]['Q'][0] = Q
                    rspec_data[ResonatorIndex]['freq_sweep'][0] = freq_sweep
                    rspec_data[ResonatorIndex]['gain_sweep'][0] = gain_sweep
                    rspec_data[ResonatorIndex]['fR'][0] = fR
                    rspec_data[ResonatorIndex]['Ql'][0] = Ql
                    rspec_data[ResonatorIndex]['Qi'][0] = Qi
                    rspec_data[ResonatorIndex]['Qc'][0] = Qc
                    rspec_data[ResonatorIndex]['Exp Config'][0] = expt_cfg
                    rspec_data[ResonatorIndex]['Syst Config'][0] = sys_config

                    saver_rspec = Data_H5(studyFolder, data=rspec_data, save_r=save_r)
                    saver_rspec.save_to_h5('rspec_jcrun7')
                    del saver_rspec
                    gc.collect()

                del r_spec
                del rspec_data

    ## collect timestream data
    total_counts = []
    for m in np.arange(0,num_rounds):
            print(f'round {m}')
            drive_data = create_data_dict(drive_keys, save_r, resonator_list)
            timestamp_drive = time.mktime(datetime.datetime.now().timetuple())
            stream = ResonatorDrive(drive, sensor, len(sensor), studyDocumentationFolder, dataSetFolder, m,
                               save_figs=save_figs, experiment=experiment,
                               verbose=verbose, logger=rr_logger)
            iq_ddr4, t, timestep, expt_cfg = stream.run(threading=threading)
            block_I, block_Q, block_amps, amp_block_IQ, block_t = stream.average_timestream(iq_ddr4, t)
            # self.plot_timestreamv2(block_I, block_Q, block_amps, amp_block_IQ, block_t)
            I_stack, Q_stack, amp_stack, amp_block_IQ_stack, block_t_stack = stream.stack_repsv2(block_I, block_Q,
                                                                                               block_amps, amp_block_IQ,
                                                                                               block_t)

            stream.plot_timestreamv2(I_stack, Q_stack, amp_stack, amp_block_IQ_stack, block_t_stack, fig, ax[ax_1][ax_0], drive_gain)
            ax_1 = ax_1 + 1
            if ax_1 > 3:
                ax_1 = 0
                ax_0 = 1
            ## save data
            ResonatorIndex = resonator_list[0]
            if save_data_h5:
                drive_data[ResonatorIndex]['Dates'][0] = timestamp_drive
                drive_data[ResonatorIndex]['Batch Num'][0] = 0
                drive_data[ResonatorIndex]['Round Num'][0] = m
                #drive_data[ResonatorIndex]['freq'][0] = fR
                drive_data[ResonatorIndex]['timestep'][0] = timestep
                drive_data[ResonatorIndex]['Exp Config'][0] = expt_cfg
                drive_data[ResonatorIndex]['Syst Config'][0] = sys_config

                saver_drive = Data_H5(studyFolder, data=drive_data, save_r=save_r)
                saver_drive.save_to_h5('drive_jcrun8')
                del saver_drive
                del drive_data
                del stream
                gc.collect()


del experiment
gc.collect()

plt.show()


