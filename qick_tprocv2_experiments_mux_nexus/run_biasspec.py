import sys
import os
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus"))
from system_config import QICK_experiment
from bias_qubit_spec import BiasQubitSpectroscopy
import datetime

outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today()))

experiment = QICK_experiment(outerFolder)

qubit = 1 #Qubit to Run
<<<<<<< HEAD
start_voltage = 0 #V
stop_voltage = 0.01 #V
=======
start_voltage = -0.1 #V
stop_voltage = 0.1 #V
>>>>>>> 0bc5a5807f81cf04e7b06796a6b9b5114a4a58a9
voltage_pts = 2

bias_spec = BiasQubitSpectroscopy(qubit-1, outerFolder, experiment)

<<<<<<< HEAD
bias_spec.run(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts, plot_sweeps=False, plot_2d=False)
=======
bias_spec.run(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts, plot_sweeps=True, plot_2d=True)
>>>>>>> 0bc5a5807f81cf04e7b06796a6b9b5114a4a58a9

del bias_spec
del experiment