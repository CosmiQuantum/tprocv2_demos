import sys
import os
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus"))
from system_config import QICK_experiment
from tomography import TomographyMeasurement

import datetime
outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today()))



qubit = 4  #Qubit to Run
start_voltage = 0 #V
stop_voltage = 0.15 #V
voltage_pts = 30


#resGs=np.linspace(0.1,0.5,5)
#resFs=np.linspace(5959.553-1.5, 5959.553+1.5, 4)
#f = 5858.673

#for g in resGs:
#for f in resFs:
experiment = QICK_experiment(outerFolder)
tomography = TomographyMeasurement(qubit-1, outerFolder, experiment)

print(tomography.config)

tomography.run_tomography(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts, plot=True, save=False)


del tomography
del experiment