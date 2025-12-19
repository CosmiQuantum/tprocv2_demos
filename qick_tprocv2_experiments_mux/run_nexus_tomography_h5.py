import sys
import os
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux"))
from system_config import QICK_experiment
from nexus_tomography_h5 import TomographyMeasurement
from expt_config import tot_num_of_qubits, FRIDGE

import datetime
#outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today()))

run_name = 'run33e'
device_name = '4charge'
study =  'Cs_TimeStudy_Tomography_2'#'Cs_TimeStudy_Tomography' #'SC_Tomography' #'DDon_SC_HoleClosed_noCol' #'Tomography' #
data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

studyFolder = os.path.join(f"/home/nexusadmin/Documents/Data/{run_name}/{device_name}/", study)
if not os.path.exists(studyFolder):
    os.makedirs(studyFolder)


## Unblock one of the follow block to do single qubit tomography or repeated tomography for all qubits

##############################################################

## Single Qubit Tomography

subStudy = 'SingleQ4_Tomography'
day = str(datetime.date.today())
data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

subStudyFolder = os.path.join(studyFolder, subStudy)
if not os.path.exists(subStudyFolder):
    os.makedirs(subStudyFolder)
dayFolder = os.path.join(subStudyFolder, day)
if not os.path.exists(dayFolder):
    os.makedirs(dayFolder)
data_setFolder = os.path.join(dayFolder, data_set)
if not os.path.exists(data_setFolder):
    os.makedirs(data_setFolder)
print(data_setFolder)

qubit = 3  #Qubit to Run [0, 1, 2, 3]

res_len = [3, 5.25, 4.5, 4] #[5, 4.75, 5, 4.25] #[5, 5.6, 5, 3.6]
freq_offset = [-0.25, 0.2, -0.3, 0] #[0.05, -0.225, -0.2, -0.075] #[-0.18, 0.2, -0.3375, 0.09] #MHz

start_voltage = 0 #V
stop_voltage = 0.04 #0.04 #V #1/2 period
voltage_pts = 3
#

## Get num of rounds to use by total time you want, or just set manually below:
run_time = 1 # hrs
round_time = 4.5 #min
round_num = int(run_time*60/round_time)

# round time is ~9-10 sec
rounds = 30000

unmask = True

experiment = QICK_experiment(data_setFolder, fridge=FRIDGE)
print(data_setFolder)
tomography = TomographyMeasurement(qubit, data_setFolder, experiment, tot_num_of_qubits, res_len, freq_offset, unmask, progress = False)
tomography.run_tomography(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts, rounds, plot = False, save = True)

del tomography
del experiment

#####################################################

# ## Repeated All Qubit Tomography
#
# subStudy = 'AllQ_Tomography'
# day = str(datetime.date.today())
# data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#
# subStudyFolder = os.path.join(studyFolder, subStudy)
# if not os.path.exists(subStudyFolder):
#     os.makedirs(subStudyFolder)
# dayFolder = os.path.join(subStudyFolder, day)
# if not os.path.exists(dayFolder):
#     os.makedirs(dayFolder)
# data_setFolder = os.path.join(dayFolder, data_set)
# if not os.path.exists(data_setFolder):
#     os.makedirs(data_setFolder)
#
# qs_to_look_at = [0, 1, 2, 3]
# res_len = [3, 5.25, 4.5, 4] #[5, 4.75, 5, 4.25] #[5, 5.6, 5, 3.6]
# freq_offset = [-0.25, 0.2, -0.3, 0] #[0.05, -0.225, -0.2, -0.075] #[-0.18, 0.2, -0.3375, 0.09] #MHz
#
#
# start_voltage = 0 #V
# stop_voltage = 0.1 #V
# voltage_pts = 30
#
# ## Get num of rounds to use by total time you want, or just set manually below:
# run_time = 1 # hrs
# round_time = 4.5 #min
# round_num = int(run_time*60/round_time)
#
# rounds = 3000
#
# unmask = True
#
# #experiment = QICK_experiment(outerFolder)
# experiment = QICK_experiment(data_setFolder, fridge=FRIDGE)
# print(data_setFolder)
# qs_tomography = AllQubitTomographyMeasurement(data_setFolder, experiment, tot_num_of_qubits, res_len, freq_offset, unmask)
# qs_tomography.allq_run_tomography(experiment.soccfg, experiment.soc, qs_to_look_at,
#                                   start_voltage, stop_voltage, voltage_pts, rounds, plot=True, save=True)
#
# del qs_tomography
# del experiment