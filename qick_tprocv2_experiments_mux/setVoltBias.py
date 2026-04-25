from NetDrivers import E36300, Keithley2400
import time

###################
supply = 'keithley' #'keithley' #'keysight'
qubit_index = 0 #0, 1, 2, 3 (only used for keysight)
voltage = 0.042 #0.15 MAX!! #V
turn_output_off = False ### Don't use for Q4 on keysight! Will turn off HEMT chs too
###################

if voltage> 0.15:
    print(f"{voltage} V is too high, reset")

elif turn_output_off == True:
    if supply == 'keithley':
        bias = Keithley2400(server_ip = "192.168.0.45", server_port = 4001)
        bias.setSourceVoltage(0)
        bias.setOutputState(enable=False)
    elif supply == 'keysight':
        Bias_PS_ip = ['192.168.0.44', '192.168.0.44', '192.168.0.44',
                      '192.168.0.41']  # IP address of bias PS (qubits 1-3 are the same PS)
        Bias_ch = [1, 2, 5, 1]  # Giving Q3 a nonsense ch # bc that channel on keysight is now for warm amp #Channel number of qubit 1-4 on associated PS

        BiasPS = E36300(Bias_PS_ip[qubit_index], server_port=5025)
        set_v = BiasPS.setVoltage(0, Bias_ch[qubit_index])
        BiasPS.disable(Bias_ch[qubit_index])
    print(f'Bias supply {supply} output off')

else:
    if supply == 'keithley':
        bias = Keithley2400(server_ip = "192.168.0.45", server_port = 4001)
        bias.clearErrors()
        bias.reset()
        bias.initializeVoltageSource(vrange=0.2, current_limit=2e-2, enable_output=False)
        bias.setSourceVoltage(voltage)
        bias.setOutputState(enable=True)
        print(bias.measureVoltage())

    elif supply == 'keysight':
        Bias_PS_ip = ['192.168.0.44', '192.168.0.44', '192.168.0.44', '192.168.0.41'] #IP address of bias PS (qubits 1-3 are the same PS)
        Bias_ch = [1, 2, 5, 1] # Giving Q3 a nonsense ch # bc that channel on keysight is now for warm amp #Channel number of qubit 1-4 on associated PS

        BiasPS = E36300(Bias_PS_ip[qubit_index], server_port = 5025)
        print(f"Setting qubit {qubit_index +1} bias to {voltage}V")
        set_v = BiasPS.setVoltage(voltage, Bias_ch[qubit_index])
        BiasPS.enable(Bias_ch[qubit_index])