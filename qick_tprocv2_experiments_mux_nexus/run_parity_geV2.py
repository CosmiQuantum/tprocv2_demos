import sys
import os


sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))  ## Change for quiet vs nexus
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus"))
#sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus/"))
from system_config import QICK_experiment
from parity_geV2 import ParityMeasurement
import time
import datetime
import numpy as np
from tprocv2_demos.qick_tprocv2_experiments_mux_nexus.parity_geV2 import ParityProgram
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement

from system_config import QICK_experiment  ## Change for quiet vs nexus
# from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits  ## Change for quiet vs nexus
# from expt_config import expt_cfg, list_of_all_qubits
################################################ Run Configurations ####################################################


outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today()))
################################################ optimization outputs ##################################################






qubit = 4  #Qubit to Run
start_voltage = 0 #V
stop_voltage = 0 #V
voltage_pts = 1


#resGs=np.linspace(0.1,0.5,5)
#resFs=np.linspace(5959.553-1.5, 5959.553+1.5, 4)
#f = 5858.673

#for g in resGs:
#for f in resFs:
experiment = QICK_experiment(outerFolder)
parityMeas = ParityMeasurement(qubit-1, outerFolder, experiment)

print(parityMeas.config)

parityMeas.run_Parity(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts, plot=True, save=False)


del parityMeas
del experiment