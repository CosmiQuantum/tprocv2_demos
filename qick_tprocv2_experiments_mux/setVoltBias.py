from NetDrivers import E36300, Keithley2400
import time

###################
supply = 'keithley' #'keysight'
qubit_index = 3 #0, 1, 2, 3 (only used for keysight)
voltage = 0.0 #0.15 MAX!! #V
###################

if voltage> 0.15:
    print(f"{voltage} V is too high, reset")
else:
    if supply == 'keithley':
        bias = Keithley2400(server_ip = "192.168.0.45", server_port = 4001)
        bias.clearErrors()
        bias.reset()
        bias.initializeVoltageSource(vrange=0.2, current_limit=1e-6, enable_output=False)
        bias.setSourceVoltage(voltage)
        bias.setOutputState(enable=True)
        print(bias.measureVoltage())

    elif supply == 'keysight':
        Bias_PS_ip = ['192.168.0.44', '192.168.0.44', '192.168.0.44', '192.168.0.41'] #IP address of bias PS (qubits 1-3 are the same PS)
        Bias_ch = [1, 2, 3, 1] #Channel number of qubit 1-4 on associated PS

        BiasPS = E36300(Bias_PS_ip[qubit_index], server_port = 5025)

        if voltage > 0.15:
            print(f"{voltage} V is too high, reset")
        else:
            print(f"Setting qubit {qubit_index +1} bias to {voltage}V")
            set_v = BiasPS.setVoltage(voltage, Bias_ch[qubit_index])
            BiasPS.enable(Bias_ch[qubit_index])
            #time.sleep(8)