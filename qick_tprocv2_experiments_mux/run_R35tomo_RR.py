import sys
import os
import datetime

sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux"))
from system_config import QICK_experiment
from nexus_tomography_R35 import AllQubitTomographyMeasurement
from expt_config import tot_num_of_qubits, FRIDGE

from round_robin_fast_tomo import RR_IntraTomo

run_name = 'run35'
device_name = '4charge'
study = 'Debug Tomography' #'HotCsStudy'
substudy = 'Try4' #'Study1'
data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

studyFolder = os.path.join(f"/home/nexusadmin/Documents/Data/{run_name}/{device_name}/", study)
if not os.path.exists(studyFolder):
    os.makedirs(studyFolder)
subStudyFolder = os.path.join(studyFolder, substudy)
if not os.path.exists(subStudyFolder):
    os.makedirs(subStudyFolder)
datasetFolder = os.path.join(subStudyFolder, data_set)
if not os.path.exists(datasetFolder):
    os.makedirs(datasetFolder)

qs_to_meas = [1, 2, 3, 4] #[1, 2, 3, 4]
res_len = [4, 4, 4, 4]
freq_offset = [0, 0, 0, 0]

start_volt = 0 #V
stop_volt = 0.1 #V #100mV
volt_pts = 30

## Hours of tomography before RR is run
run_time = 0.25 #12 #hrs
tomo_round_time = 3.2 #min #### NEEDS TO BE CHECKED!!!####
tomo_rounds = int(run_time*60/tomo_round_time)

##Rounds of RR to do ### CHECK HOW LONG RR TAKES ON ALL 4 QS ###
RR_rounds = 3

experiment = QICK_experiment(datasetFolder, fridge=FRIDGE)
print(datasetFolder)
tomography = AllQubitTomographyMeasurement(datasetFolder, experiment, tot_num_of_qubits, res_len, freq_offset,
                                           measure_qubits= qs_to_meas, unmasking_resgain=True, progress=False)

#Total # of hours to run
total_hr = 0.5 #48 #Assuming no stops
end_time = datetime.datetime.now() + datetime.timedelta(hours=total_hr)
print(f"Loop started at: {datetime.datetime.now()}")
print(f"Loop scheduled to end at: {end_time}")

while datetime.datetime.now() < end_time:
    tomography.run_tomography(experiment.soccfg, experiment.soc, start_volt, stop_volt, volt_pts,
                                                    tomo_rounds, plot=True, plot_together = False, save=True)
    # remaining_time = (end_time - datetime.datetime.now()).total_seconds()
    # if remaining_time < run_time * 3600:
    #     run_time_left = remaining_time / 3600 #remaining time in hours if less than 12 hrs
    #     tomo_rounds_leftover = int(run_time_left*60/tomo_round_time)
    #     tomography.run_tomography(experiment.soccfg, experiment.soc, start_volt, stop_volt, volt_pts,
    #                               tomo_rounds_leftover, plot=True, plot_together = False, save=True)
    #     RR_IntraTomo(study, substudy, file_timestamp, RR_rounds)
    # else:
    #     tomography.run_tomography(experiment.soccfg, experiment.soc, start_volt, stop_volt, volt_pts,
    #                                                tomo_rounds, plot=True, plot_together = False, save=True)
    #     RR_IntraTomo(study, substudy, file_timestamp, RR_rounds)