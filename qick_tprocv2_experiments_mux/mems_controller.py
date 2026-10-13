import sys
from pathlib import Path

#Add MEMS vendor library path----
MEMS_LIB_PATH= Path("/home/loudadmin/MEMS_ctrl/")
if str(MEMS_LIB_PATH) not in sys.path:
    sys.path.append(str(MEMS_LIB_PATH))

from MEMS_serial_device import MEMS_device

class MEMSController:
    def __init__(
        self,
        port="/dev/ttyUSB0",
        Vbias=90,
        HardwareFilterBW=120,
        VdifferenceMax=180,
    ):
        self.port = port
        self.Vbias= Vbias
        self.HardwareFilterBW = HardwareFilterBW
        self.VdifferenceMax = VdifferenceMax
        self.mems = MEMS_device(port=self.port)
        self.mems.set_mirror_params(
            Vbias=self.Vbias,
            HardwareFilterBW=self.HardwareFilterBW,
            VdifferenceMax=self.VdifferenceMax,
        )

        self.mems.HV_on()

    def set_position(self, x, y):
        self.mems.set_mirror_position(x=x, y=y)

    def close(self):
        self.mems.exit_safely()
            
            
