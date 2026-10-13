import time 
from agilent_33220a import Agilent33220A

awg= Agilent33220A("192.168.0.198")

print(awg.idn())
awg.set_pulse_high(1.18)
awg.output_on()

time.sleep(2)
print("HIGH:", awg.inst.query("VOLT:HIGH?"))
awg.output_off()
awg.close()

#-----Testing the MEMS if I can move via python
from mems_controller import MEMSController
import time
mems=MEMSController(
    port="/dev/ttyUSB0",
    Vbias=90,
    HardwareFilterBW=120,
    VdifferenceMax=180,
)



# Loop over X and Y positions
import time
import numpy as np
# Assuming mems object is already initialized: mems = MEMS_device(...)

# ------------------ MEMS settings ------------------
MEMS_MIN = -0.3
MEMS_MAX = 0.3
MEMS_PRECISION = 2  # Number of decimals

# ------------------ Define scan points ------------------
X_MEMS_start = -0.3
X_MEMS_stop  = 0.3
X_MEMS_step  = 0.1

Y_MEMS_start = -0.3
Y_MEMS_stop  = 0.3
Y_MEMS_step  = 0.1

# Create arrays of targets, rounded to 2 decimals
X_MEMS_values = np.round(np.arange(X_MEMS_start, X_MEMS_stop + X_MEMS_step/2, X_MEMS_step), MEMS_PRECISION)
Y_MEMS_values = np.round(np.arange(Y_MEMS_start, Y_MEMS_stop + Y_MEMS_step/2, Y_MEMS_step), MEMS_PRECISION)

# ------------------ Loop over MEMS positions ------------------
for X_MEMS_target in X_MEMS_values:
    # Clamp X to allowed range
    X_MEMS_safe = round(min(max(X_MEMS_target, MEMS_MIN), MEMS_MAX), MEMS_PRECISION)

    for Y_MEMS_target in Y_MEMS_values:
        # Clamp Y to allowed range
        Y_MEMS_safe = round(min(max(Y_MEMS_target, MEMS_MIN), MEMS_MAX), MEMS_PRECISION)

        # Move MEMS safely
        print(f"Moving MEMS to X={X_MEMS_safe}, Y={Y_MEMS_safe}")
        mems.set_position(x=X_MEMS_safe, y=Y_MEMS_safe)
        


#import sys                      
#from pathlib import Path
#MEMS_LIB_PATH= Path("/home/loudadmin/MEMS_ctrl/")
#sys.path.append(str(MEMS_LIB_PATH))
#from MEMS_serial_device import *
#port = "/dev/ttyUSB0"
#Vbias = 90
#HardwareFilterBW = 120
#VdifferenceMax = 180
#mems= MEMS_device(port=port)
#mems.set_mirror_params(Vbias=Vbias,HardwareFilterBW=HardwareFilterBW,VdifferenceMax=VdifferenceMax)
#print(mems)
#mems.HV_on()
#mems.set_mirror_position(x=0.0,y=0.0)
#mems.exit_safely()
