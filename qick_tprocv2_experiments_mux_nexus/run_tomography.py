import sys
import os
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus"))
from system_config import QICK_experiment
from tomography import TomographyMeasurement
from tomography import AllQubitTomographyMeasurement

import datetime
outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today()))

## Unblock one of the follow block to do single qubit tomography or repeated tomography for all qubits

# ## Single Qubit Tomography
#
# qubit = 3  #Qubit to Run
# start_voltage = 0 #V
# stop_voltage = 0.1 #V
# voltage_pts = 30
#
# experiment = QICK_experiment(outerFolder)
# tomography = TomographyMeasurement(qubit-1, outerFolder, experiment)
# tomography.run_tomography(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts, plot=True, save = False)
#
# del tomography
# del experiment



## Repeated All Qubit Tomography
start_voltage = 0 #V
stop_voltage = 0.1 #V
voltage_pts = 30

## Get num of rounds to use by total time you want, or just set manually below:
run_time = 1 # hrs
round_time = 1.5 #min
round_num = int(run_time*60/round_time)

rounds = 1

experiment = QICK_experiment(outerFolder)
qs_tomography = AllQubitTomographyMeasurement(outerFolder, experiment)
qs_tomography.allq_run_tomography(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts, rounds, plot=False, save=False)

del qs_tomography
del experiment