import sys
import os
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux"))
from system_config import QICK_experiment
from new_nexus_tomography import AllQubitTomographyMeasurement
from expt_config import tot_num_of_qubits, FRIDGE

import datetime