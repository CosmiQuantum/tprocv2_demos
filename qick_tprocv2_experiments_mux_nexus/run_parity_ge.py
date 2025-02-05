import sys
import os
import time
#from backup.tprocv2_demos.qick_tprocv2_experiments_mux_nexus.round_robin_benchmark import save_figs

sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus"))
from system_config import QICK_experiment
from parity_ge import ParityMeasurement

import datetime
import numpy as np
outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today()))



qubit = 4  #Qubit to Run

voltage = 0.075
timestamp = time.strftime("%H%M%S")
experiment = QICK_experiment(outerFolder)
parity_meas = ParityMeasurement(qubit-1, outerFolder, experiment)

I, Q, timetaken =parity_meas.run(experiment.soccfg, experiment.soc, voltage)

parity_meas.plot_results( I, Q, timetaken, fig_quality=100)
np.savez(os.path.join(outerFolder, f'parity_Q{qubit}_{timestamp}.npz'), I=I, Q=Q, timetaken=timetaken)

del parity_meas