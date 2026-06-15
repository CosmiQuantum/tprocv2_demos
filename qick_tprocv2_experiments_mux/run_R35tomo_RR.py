import sys
import os
import datetime

sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux"))
from system_config import QICK_experiment
from nexus_tomography_R35 import AllQubitTomographyMeasurement
from expt_config import tot_num_of_qubits, FRIDGE
from round_robin_fast_tomo import RR_IntraTomo

run_name = 'run37'
device_name = '4charge'
study = 'BackgroundTomography' #'EndOfRunData' #'PostCsTomography'
substudy = 'Dataset1' #'HighgainOpt' #'Dataset2_neg' #'Dataset1' #'Study1'
data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

studyFolder = os.path.join(f"/home/nexusadmin/Documents/Data/{run_name}/{device_name}/", study)
if not os.path.exists(studyFolder):
    os.makedirs(studyFolder)
subStudyFolder = os.path.join(studyFolder, substudy)
if not os.path.exists(subStudyFolder):
    os.makedirs(subStudyFolder)
datasetFolder = os.path.join(subStudyFolder, data_set) #overarching folder for all stuff
if not os.path.exists(datasetFolder):
    os.makedirs(datasetFolder)

qs_to_meas = [1, 2, 3, 4] #[1, 2, 3, 4]
res_len = [4.5, 4.75, 5.5, 4.75] #[4.5, 6, 5, 4.25]
freq_offset = [0, 0, -0.075, 0] #-0.225] #[-0.15, -0.075, -0.2, -0.05]
res_gain = [0.3875, 0.425, 0.3875, 0.425] #[0.3, 0.3, 0.28, 0.3]

start_volt = 0 #V #0
stop_volt = 0.1 #V #0.1 #100mV
volt_pts = 30

## Timing info
total_runhr = 65 #48 #48 #hours
RR_intervalhr = 12 #12 #hours
tomo_round_min = 3.23 #min
RR_rounds = 2 #3 #how many RR rounds to run

total_runsec = total_runhr * 3600
RR_intervalsec = RR_intervalhr * 3600
tomo_round_sec = tomo_round_min * 60

rounds_per_block = int(RR_intervalsec // tomo_round_sec)
if rounds_per_block <= 0:
    raise ValueError("RR interval too short to fit 1 rd of tomography.")

# ## Hours of tomography before RR is run
# run_time = 12 #hrs
# tomo_round_time = 3.23 #min #### NEEDS TO BE CHECKED!!!####
# tomo_rounds = int(run_time*60/tomo_round_time)
#
# ##Rounds of RR to do ### CHECK HOW LONG RR TAKES ON ALL 4 QS ###
# RR_rounds = 3

experiment = QICK_experiment(datasetFolder, fridge=FRIDGE)
print(datasetFolder)
tomography = AllQubitTomographyMeasurement(datasetFolder, experiment, tot_num_of_qubits, res_len, freq_offset,
                                           measure_qubits= qs_to_meas, unmasking_resgain=True, progress=False)

# total_hr = 48 #Assuming no stops
# end_time = datetime.datetime.now() + datetime.timedelta(hours=total_hr)
start_time = datetime.datetime.now()
elapsed_sec = 0.0
print(f"Experiment started at: {datetime.datetime.now()}")
print(f"Scheduled to run for {total_runhr} hours")

# Initial RR
print("Running initial RR")
RR_IntraTomo(datasetFolder, RR_rounds, res_len, res_gain, freq_offset)

# Main Loop
while elapsed_sec + tomo_round_sec <= total_runsec:
    remaining_sec = total_runsec - elapsed_sec

    max_rounds_block = int(remaining_sec // tomo_round_sec)
    rounds_block = min(rounds_per_block, max_rounds_block)

    if rounds_block <= 0:
        print("Not enough time for another tomography round.")
        break

    print(f"Running tomography for {rounds_block} rounds")

    tomography.run_tomography(experiment.soccfg, experiment.soc, start_volt, stop_volt, volt_pts,
                                               rounds_block, plot=False, plot_together=False, save=True)
    elapsed_sec = (datetime.datetime.now() - start_time).total_seconds()

    if elapsed_sec + tomo_round_sec > total_runsec:
        print("Not enough time left for another block - exiting")
        break

    print("Running RR")
    RR_IntraTomo(datasetFolder, RR_rounds, res_len, res_gain, freq_offset)
    elapsed_sec = (datetime.datetime.now() - start_time).total_seconds()

# Final RR after loop
print("Running final RR")
RR_IntraTomo(datasetFolder, RR_rounds, res_len, res_gain, freq_offset)


# while datetime.datetime.now() < end_time:
#     # tomography.run_tomography(experiment.soccfg, experiment.soc, start_volt, stop_volt, volt_pts,
#     #                                                 tomo_rounds, plot=True, plot_together = False, save=True)
#     remaining_time = (end_time - datetime.datetime.now()).total_seconds()
#     if remaining_time < run_time * 3600:
#         run_time_left = remaining_time / 3600 #remaining time in hours if less than 12 hrs
#         tomo_rounds_leftover = int(run_time_left*60/tomo_round_time)
#         if tomo_rounds_leftover <= 0:
#             print("Not enough time for another tomography round. Skipping to final RR.")
#         else:
#             file_timestamp = tomography.run_tomography(experiment.soccfg, experiment.soc, start_volt, stop_volt, volt_pts,
#                                       tomo_rounds_leftover, plot=False, plot_together = False, save=True)
#         RR_IntraTomo(study, substudy, file_timestamp, RR_rounds)
#     else:
#         file_timestamp = tomography.run_tomography(experiment.soccfg, experiment.soc, start_volt, stop_volt, volt_pts,
#                                                    tomo_rounds, plot=False, plot_together = False, save=True)
#         RR_IntraTomo(study, substudy, file_timestamp, RR_rounds)