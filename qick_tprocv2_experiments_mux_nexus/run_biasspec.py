import sys
import os
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus"))
from system_config import QICK_experiment
from bias_qubit_spec import BiasQubitSpectroscopy
import datetime
import numpy as np
outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today()))



qubit = 4  #Qubit to Run
start_voltage = 0 #V
stop_voltage = 0.15 #V
voltage_pts = 30

resGs=np.linspace(0.1,0.5,11)
resFs=np.linspace(5958.673-1.5, 5958.673+1.5, 11)

for g in resGs:
    for f in resFs:
        experiment = QICK_experiment(outerFolder)
        bias_spec = BiasQubitSpectroscopy(qubit-1, outerFolder, experiment)
        bias_spec.config['res_freq_ge'][qubit-1]=f
        bias_spec.config['res_gain_ge'][qubit - 1] =g
        print('resG=',g)
        bias_spec.run(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts, plot_sweeps=True, plot_3d=True)

        del bias_spec
        del experiment