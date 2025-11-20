from NetDrivers import E36300
import time

Bias_PS_ip = ['192.168.0.44', '192.168.0.44', '192.168.0.44', '192.168.0.41'] #IP address of bias PS (qubits 1-3 are the same PS)
Bias_ch = [1, 2, 3, 1] #Channel number of qubit 1-4 on associated PS

###################
qubit_index = 2 #0, 1, 2, 3
voltage = 0.0 #0.15 MAX!!! #V
######################

BiasPS = E36300(Bias_PS_ip[qubit_index], server_port = 5025)

if voltage > 0.15:
    print(f"{voltage} V is too high, reset")
else:
    print(f"Setting {qubit_index +1} bias to {voltage}V")
    BiasPS.setVoltage(voltage, Bias_ch[qubit_index])
    #time.sleep(8)