import numpy as np

FRIDGE = "QUIET"  # change to "NEXUS" as needed

if FRIDGE == "QUIET":
    VNA_res = np.array([6227.187, 6289.175, 6348.55, 6419.665, 6485.360, 6552.35])# ge resonator freqs, updated 10/10 during run 8
    VNA_qubit = np.array([4194.77, 3828.69, 4173.69, 4474.04, 4485.38, 5018.12])  # Qubit freqs g/e Transition, updated during run 8 on 10/10
    ef_freqs = np.array([4020.48, 3650.81, 3999.01, 4302.33, 4313.06, 4848.81]) # Qubit freqs e/f Transition,Arianna 10/13
    fh_freqs = np.array([3820.97, 3450.85, 3798.97, 4110, 4660, 4660.28]) # Qubit freqs f/h Transition
    two_photon_freqs=np.array([4107.61, 3739.36, 4086.36, 4388.34, 4399.3, 4933.52]) # qubit freqs, two photon peak between ge and ef qubit freqs

    # Set this for your experiment
    tot_num_of_qubits = 6

    list_of_all_qubits = list(range(tot_num_of_qubits))

    expt_cfg = {
        "tof": {
            "reps": 1, #reps doesnt make a difference here, leave it at 1
            "soft_avgs": 100,
            "relax_delay": 0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        # "res_spec": { # For thomas roth data
        #     "reps": 300, #3000
        #     "rounds": 1,
        #     "start": -0.5,       # MHz, sweep from -1.0 to +1.0 around center
        #     "step_size": 0.005,  # MHz (5 kHz)
        #     "steps": 201,
        #     "relax_delay": 10,  # [us]
        #     "list_of_all_qubits": list_of_all_qubits,
        # },

        "res_spec": {
            "reps": 300,
            "rounds": 1,
            "start": -1.25,  # [MHz]
            "step_size": 0.05,  # [MHz]
            "steps": 51,
            "relax_delay": 10,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },


        # "res_spec_ef": { # for thomas roth data
        #     "reps": 3000,
        #     "rounds": 1,
        #     "start": -1.0,       # MHz, sweep from -1.0 to +1.0 around center
        #     "step_size": 0.005,  # MHz (5 kHz)
        #     "steps": 401,
        #     "relax_delay": 1000,  # [us]
        #     "list_of_all_qubits": list_of_all_qubits,
        # },

        "res_spec_ef": {
            "reps": 400,
            "rounds": 1,
            "start": -2,  # [MHz]
            "step_size": 0.05,  # [MHz]
            "steps": 100,
            "relax_delay": 1000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        # "qubit_spec_ge": { # broad two-photon peak search
        #     "reps": 4000,  # 300
        #     "rounds": 1,  # 10
        #     "start": list(VNA_qubit - 200),  # [MHz] #-300 #-15
        #     "stop": list(VNA_qubit + 6),  # [MHz] #+15
        #     "steps": 4000,  # 100
        #     "relax_delay": 10,  # 1000 # [us]
        #     "list_of_all_qubits": list_of_all_qubits,
        # },

        # "qubit_spec_ge": { # two-photon peak search
        #     "reps": 500, #300
        #     "rounds": 1, #10
        #     "start": list(two_photon_freqs-6), # [MHz] #-300 #-15
        #     "stop": list(two_photon_freqs+6), # [MHz] #+15
        #     "steps": 300, #100
        #     "relax_delay": 10,  # 1000 # [us]
        #     "list_of_all_qubits": list_of_all_qubits,
        # },

        "qubit_spec_ge": {
            "reps": 600, #300
            "rounds": 1, #10
            "start": list(VNA_qubit-6), # [MHz] #-300 #-15
            "stop": list(VNA_qubit+6), # [MHz] #+15
            "steps": 300, #100
            "relax_delay":10,#1000 # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "qubit_spec_ge_extended": {
            "reps": 500,  # 300
            "rounds": 1,  # 10
            "start": list(VNA_qubit - 250),  # [MHz]
            "stop": list(VNA_qubit + 100),  # [MHz]
            "steps": 1000,  # 100
            "relax_delay": 10,  # 1000, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "qubit_spec_ge_high_gain": {
            "reps": 500,  # 300
            "rounds": 1,  # 10
            "start": list(VNA_qubit - 30),  # [MHz]
            "stop": list(VNA_qubit + 30),  # [MHz]
            "steps": 600,  # 100
            "relax_delay": 10,  # 1000, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "qubit_spec_ge_zeno_stark": {
            "reps": 2500,  # 300
            "rounds": 1,  # 10
            "start": list(VNA_qubit - 25),  # [MHz] #-300 #-15
            "stop": list(VNA_qubit + 5),  # [MHz] #+15
            # "start": list(VNA_qubit - 100),  # [MHz] #-300 #-15
            # "stop": list(VNA_qubit + 100),  # [MHz] #+15
            "steps": 200,  # 100
            "relax_delay": 700,  # 1000, # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "qze_mask": [],
        },

        "qubit_spec_ef": {
            "reps": 3100,  # 300
            "rounds": 1,  # 10
            "start": list(ef_freqs - 1.5),  # [MHz] #-300
            "stop": list(ef_freqs + 1.5),  # [MHz]
            "steps": 300,
            "relax_delay": 1000, #1000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "qubit_spec_fh": {
            "reps": 20000,  # 300
            "rounds": 1,  # 10
            "start": list(fh_freqs - 0.3), # [MHz] #-300 #-6
            "stop": list(fh_freqs +  0.2),  # [MHz] #6
            "steps": 180,#450,  # 1000 #450
            "relax_delay": 1000,  # 1000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "qubit_spec_ftores": {
            "reps": 10000,  # 300
            "rounds": 1,  # 10
            "start": list((VNA_qubit + ef_freqs) - VNA_res - 200),  # [MHz] #-300
            "stop":  list((VNA_qubit + ef_freqs) - VNA_res + 200),  # [MHz]
            "steps": 1000,  # 1000
            "relax_delay": 0.5,  # 1000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "bias_qubit_spec_ge": {
            "reps": 700,  # 100
            "rounds": 1,  # 10
            "start": list(VNA_qubit - 70),  # [MHz]
            "stop": list(VNA_qubit + 70),  # [MHz]
            "steps": 300,
            "relax_delay": 0.5,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "power_rabi_ge": {
            "reps": 200,#500,
            "rounds": 1,  # 5
            "start": [0] * 6,  # [DAC units]
            "stop": [1] * 6,#[1.0] * 6,  # [DAC units]
            "steps": 115, #50,
            "relax_delay": 1000,#1000,#1000,#1000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "length_rabi_ge": {
            "reps": 100,#500,
            "rounds": 1,
            "start": [0.01] * 6,  # [us]
            "stop": [25] * 6,#[0.7] * 6,  # [us]
            "steps": 500,
            "relax_delay": 1000,# [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "length_rabi_Qtemps": {
            "reps": 400,#500,
            "rounds": 1,
            "start": [0.02] * 6,  # [us]
            "stop": [2] * 6,#[0.7] * 6,  # [us]
            "steps": 200,
            "relax_delay": 1000,# [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "length_rabi_ge_pi_len": { #for the zeno tests, bare qubit frequency rabi
            "reps": 300,  # 500,
            "rounds": 1,
            "start": [0.01] * 6,  # [us]
            "stop": [4] * 6,  # [0.7] * 6,  # [us]
            "steps": 60,
            "relax_delay": 1000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "length_rabi_ge_qze": {
            "reps": 400,  # 500,
            "rounds": 1, #600
            "start": [0.01] * 6,  # [us]
            "stop": [10] * 6,   #[3] * 6,   # [us]
            "steps": 150,  # 140,
            "relax_delay": 1000,# [us]
            "list_of_all_qubits": list_of_all_qubits,
            "qze_mask": [],
            "zeno_pulse_width": 0.007,
            "zeno_pulse_period": 0.10,
        },

        "power_rabi_ef": {
            "reps": 550,#400, # for Pg pulse sequence during rpm, or regular ef rabi
            "reps2": 1500, #this is only used for the experiment that uses e-f rabi to calculate qubit temperatures (rpm). Pe pulse sequence
            "rounds": 1,
            "start": [0.0] * 6,  # [DAC units]
            "stop": [1.0] * 6,  # [DAC units]
            "steps": 155,
            "relax_delay": 1000,  # [us]
        },
        "power_rabi_fh": {
            "reps": 200,
            "reps2": 850,  # this is only used for the experiment that uses e-f rabi to calculate qubit temperatures.
            "rounds": 1,
            "start": [0.0] * 6,  # [DAC units]
            "stop": [1.0] * 6,  # [DAC units]
            "steps": 155,
            "relax_delay": 1000,  # [us]
        },
        "T1_ge": {
            "reps": 220, #300
            "rounds": 1, #1
            "start": [0.0] * 6,  # [us]
            "stop": [200]*6, #[250.0] * 6,  # [us]
            "steps": 70,
            "relax_delay": 1000,  # [us] ### Should be >10x T1!
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits
        },

        "T1_IBM_qze": {
            "reps": 1000,  # 300
            "rounds": 1,  # 1
            "start": [30] * 6,  # [us]
            "stop": [31] * 6,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 1,
            "relax_delay": 1000,  # [us] ### Should be >10x T1!
            "wait_time": 30,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "T1_fe": {
            "reps": 200,  # 300
            "rounds": 1,  # 1
            "start": [0.0] * 6,  # [us]
            "stop": [200] * 6,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 150,
            "relax_delay": 1000,  # [us] ### Should be >10x T1!
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "T1_fg": {
            "reps": 200,  # 300
            "rounds": 1,  # 1
            "start": [0.0] * 6,  # [us]
            "stop": [200] * 6,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 150,
            "relax_delay": 1000,  # [us] ### Should be >10x T1!
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "res_stark_shift_2D": {
            "reps": 400,
            "rounds": 1,
            "list_of_all_qubits": list_of_all_qubits,
            "qubit_pulse_delay": 3, #[us] time to wait for qubit pulse after stark tone is sent
            "stark_length": 19,  # [us] stark tone length for 2D scan, overlaps qubit pulse
            "gain_steps": 20,
            "start_gain": 0.00,
            "end_gain": 1.0, #res gain values between -1 and 1
            "qubit_pulse_steps": 500,
            "start_freq": -55,#-35, #[MHz] from qubit freq
            "end_freq":  4,#3, # [MHz] from qubit freq
            "readout_pulse_delay": 2, #2/kappa
            "stark_gain": [],
            "res_freq_stark": [],
            "res_phase_stark": [],
            "stark_mask": [],
        },

        "stark_shift_2D": {
            "reps": 200,
            "rounds": 1,
            "list_of_all_qubits": list_of_all_qubits,
            "qubit_pulse_delay": 0.5,  # [us] time to wait for qubit pulse after stark tone is sent
            "stark_length": 16,  # [us] stark tone length for 2D scan, overlaps qubit pulse
            "gain_steps": 20,
            "start_gain": 0.00,
            "end_gain": 1.0,  # res gain values between -1 and 1
            "qubit_pulse_steps": 400,
            "min_freq": 3,  # [MHz] from qubit freq
            "max_freq": 15,  # [MHz] from qubit freq
            "start_freq": 0,
            "end_freq": 0,
            "readout_pulse_delay": 2,  # 2/kappa
            "detuning": [-20, -10, -10, -10, -15, -10],  # [MHz]
            "stark_gain": [],
        },

        "stark_shift_spec": {
            "reps": 500,
            "rounds": 1,
            "list_of_all_qubits": list_of_all_qubits,
            "stark_length": 25, # [us] stark tone length for TLS spectroscopy, try 10-30% of T1
            "max_shift": 20, # [MHz] specify frequency range of stark shift
            "duffing_constant": [500000,500000,50000,50000,500000,500000], #constant for duffing oscillator model from stark ramsey measurement, update for each qubit
            "gain_steps": 100, #for each branch of pos,neg detuning stark and for entire res stark
            "start_gain": 0.0,
            "end_gain": 1.0, #res gain values between -1 and 1, convert to qubit freq shift w/ stark ramsey
            "readout_pulse_delay": 2, #2/kappa
            "relax_delay": 400, #[us]
            "detuning": [-20, -10, -10, -10, -15, -10], #[MHz] start w/negative detuning, script flips to positive halfway thru scan
            "stark_sigma": 0.01,  # [us] 10 ns
            "stark_gain": [],
            "anharmonicity": [172.34, 176.38, 167.13, 172.57, 172.03, 161.14],
            "res_gain_stark": [],
            "res_freq_stark": [],
            "res_phase_stark": [],
            "stark_mask": [],
        },

        "Ramsey_stark": {
            "reps": 200,  # 300
            "rounds": 1,  # 10
            "start": [0.02] * 6,  # [us]
            "stop": [10] * 6,  # [us]
            "steps": 100,
            "start_gain": 0.0,
            "end_gain": 1.0,
            "gain_steps": 7,
            "ramsey_freq": 0.5, # [MHz]
            "relax_delay": 1000,
            "wait_time": 0.0,  # [us]
            "stark_gain": 0.0,
            "detuning": -20, # [MHz]
            "stark_sigma": 0.01, # [us] 10 ns
            "list_of_all_qubits": list_of_all_qubits,
            "anharmonicity": [172.34, 176.38, 167.13, 172.57, 172.03, 161.14],
        },

        "FastRelEx":{
            "reps": 500000,
            "rounds": 1,
            "list_of_all_qubits": list_of_all_qubits,
            "relax_delay": 10, #[us], keep short to do post-processing
            "meas_wait": 2, #[us], a fast delay after qubit pi pulse
            "readout_pulse_delay": 0.1, #for resonator to ring down, may be ok to set to zero
            "pre_stark_delay": 0.1, #could match to pi pulse length or set to zero
            "stark_sigma": 0.01, #10 ns following Carrol paper
            "stark_length": 2, #try 1-5 us
        },

        "Ramsey_ge": {
            "reps": 350, #300
            "rounds": 1,#10
            "start": [0.0] * 6, # [us]
            "stop":  [60] * 6, # [us]
            "steps": 200,
            "ramsey_freq": 0.12,  # [MHz]
            "relax_delay": 1000, # [us] the time to wait to let the qubit to relax to gnd again after exciting it (make it way above T1)
            "wait_time": 0.0, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "SpinEcho_ge": {
            "reps": 500,
            "rounds": 1,
            "start": [0.0] * 6, # [us]
            "stop":  [80] * 6, # [us]
            "steps": 100,
            "ramsey_freq": 0.12,  # [MHz]
            "relax_delay": 1000, # [us]
            "wait_time": 0.0, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
    #

    #
    #     "res_spec_ef": {
    #         "reps": 100,
    #         "py_avg": 10,
    #         "start": [7148, 0, 7202, 0, 0, 0], # [MHz]
    #         "stop":  [7151, 0, 7207, 0, 0, 0], # [MHz]
    #         "steps": 200,
    #         "relax_delay": 1000, # [us]
    #         "list_of_all_qubits": list_of_all_qubits,
    #     },
    #
    #     "qubit_spec_ef": {
    #         "reps": 100,
    #         "py_avg": 10,
    #         "start": [2750, 0, 0, 0, 0, 0], # [MHz]
    #         "stop":  [2850, 0, 0, 0, 0, 0], # [MHz]
    #         "steps": 500,
    #         "relax_delay": 1000, # [us]
        #         "list_of_all_qubits": list_of_all_qubits,
    #     },
    #
        # "qubit_temp": { #this is for Santi's length rabi qubit temperature script (not working fully)
        #     "reps": 500,
        #     "py_avg": 1, #this is rounds, change after u get script working
        #     "start": [20]*6, # [us]
        #     "expts":  [200] * 6, #points
        #     "step": (22 - 20) / (200 - 1), # [us], step = (stop - start) / (expts - 1)
        #     "relax_delay": 800, # [us]
        #     "list_of_all_qubits": list_of_all_qubits,
        # },


    #
    #     "power_rabi_ef": {
    #         "reps": 1000,
    #         "py_avg": 10,
    #         "start": [0.0] * 6, # [DAC units]
    #         "stop":  [1.0] * 6, # [DAC units]
    #         "steps": 100,
    #         "relax_delay": 1000, # [us]
    #         "list_of_all_qubits": list_of_all_qubits,
    #     },
    #
    #     "Ramsey_ef": {
    #         "reps": 100,
    #         "py_avg": 10,
    #         "start": [0.0] * 6, # [us]
    #         "stop":  [100] * 6, # [us]
    #         "steps": 100,
    #         "ramsey_freq": 0.05,  # [MHz]
    #         "relax_delay": 1000, # [us]
    #         "wait_time": 0.0, # [us]
    #         "list_of_all_qubits": list_of_all_qubits,
    #     },
    #
        "IQ_plot":{
            "steps": 5000, # shots
            "py_avg": 1,
            "reps": 1,
            "relax_delay": 1000, # [us]
            "SS_ONLY": False,
            "list_of_all_qubits": list_of_all_qubits,
        },
    # #

        "Readout_Optimization":{
            "steps": 3000, # shots
            "py_avg": 1,
            "gain_start" : [0, 0, 0, 0],
            "gain_stop" : [1, 0, 0, 0],
            "gain_step" : 0.1,
            "freq_start" : [6176.0, 0, 0, 0],
            "freq_stop" : [6178.0, 0, 0, 0],
            "freq_step" : 0.1,
            "relax_delay": 1000,#600, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

    }

elif FRIDGE == "NEXUS":
    VNA_res = np.array([6187.8, 5828.3, 6074.6, 5959.3])
    VNA_qubit = np.array([4909, 4749.4, 4569, 4759])  # Found on NR25 with the QICK

    tot_num_of_qubits = 4
    list_of_all_qubits = list(range(tot_num_of_qubits))

    expt_cfg = {
        "tof": {
            "reps": 1,  # reps doesnt make a difference here, leave it at 1
            "soft_avgs": 500,
            "relax_delay": 0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "res_spec": {
            "reps": 500,
            "rounds": 1,
            "start": -3.5,  # [MHz]
            "step_size": 0.12,  # [MHz]
            "steps": 101,
            "relax_delay": 20,  # [us]
            "relax_delay_ef": 1000,
            "list_of_all_qubits": list_of_all_qubits,
        },
        "res_spec_ef": {
            "reps": 500,
            "rounds": 1,
            "start": -3.5,  # [MHz]
            "step_size": 0.12,  # [MHz]
            "steps": 101,
            "relax_delay": 1000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        # "res_spec": { # Works for PUNCHOUT only 1/23 to do
        #     "reps": 500,
        #     "rounds": 1,
        #     "start": list(VNA_res - 1),  # [MHz]
        #     "stop": list(VNA_res + 1),
        #     "steps": 101,
        #     "relax_delay": 20,  # [us]
        #     "list_of_all_qubits": list_of_all_qubits,
        # },

        "qubit_spec_ge": {
            "reps": 700,  # 100
            "rounds": 1,  # 10
            "start": list(VNA_qubit - 70),  # [MHz]
            "stop": list(VNA_qubit + 70),  # [MHz]
            "steps": 300,
            "relax_delay": 0.5,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "bias_qubit_spec_ge": {
            "reps": 700,  # 100
            "rounds": 1,  # 10
            "start": list(VNA_qubit - 7),  # [MHz]
            "stop": list(VNA_qubit),  # [MHz]
            "steps": 400,
            "relax_delay": 0.5,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "power_rabi_ge": {
            "reps": 500,  # 100
            "rounds": 1,  # 5
            "start": [0.0] * 6,  # [DAC units]
            "stop": [1.0] * 6,  # [DAC units]
            "steps": 100,
            "relax_delay": 500,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "Readout_Optimization": {
            "steps": 3000,  # shots
            "py_avg": 1,
            "gain_start": [0, 0, 0, 0],
            "gain_stop": [1, 0, 0, 0],
            "gain_step": 0.1,
            "freq_start": [6176.0, 0, 0, 0],
            "freq_stop": [6178.0, 0, 0, 0],
            "freq_step": 0.1,
            "relax_delay": 500,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "T1_ge": {
            "reps": 1000,  # 300
            "rounds": 1,  # 1
            "start": [0.0] * 6,  # [us]
            "stop": [150] * 6,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 80,
            "relax_delay": 500,  # [us] ### Should be >10x T1!
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "Ramsey_ge": {
            "reps": 2000,  # 300
            "rounds": 1,  # 10
            "start": [0.0] * 6,  # [us]
            "stop": [8.0] * 6,  # [us]
            "steps": 100,
            "ramsey_freq": 0.3,  # [MHz]
            "relax_delay": 500,
            # [us] the time to wait to let the qubit to relax to gnd again after exciting it (make it way above T1)
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "SpinEcho_ge": {
            "reps": 2000,
            "rounds": 1,
            "start": [0.0] * 6,  # [us]
            "stop": [15] * 6,  # [us]
            "steps": 100,
            "ramsey_freq": 0.6,  # [MHz]
            "relax_delay": 500,  # [us]
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "parity_ge": {
            "steps": 10000,
            "relax_delay": 500,  # [us]
            "wait_time": (np.pi / 2) / (2 * np.pi * 1.199),  # [us]
        },

        "tomography_ge": {
            "steps": 1,
            "reps": 300,
            "rounds": 1,
            "relax_delay": 500,  # [us]
            "wait_time": 1 / (4 * (2.5)),  # [us]
        }
        #

        #
        #     "res_spec_ef": {
        #         "reps": 100,
        #         "py_avg": 10,
        #         "start": [7148, 0, 7202, 0, 0, 0], # [MHz]
        #         "stop":  [7151, 0, 7207, 0, 0, 0], # [MHz]
        #         "steps": 200,
        #         "relax_delay": 1000, # [us]
        #         "list_of_all_qubits": list_of_all_qubits,
        #     },
        #
        #     "qubit_spec_ef": {
        #         "reps": 100,
        #         "py_avg": 10,
        #         "start": [2750, 0, 0, 0, 0, 0], # [MHz]
        #         "stop":  [2850, 0, 0, 0, 0, 0], # [MHz]
        #         "steps": 500,
        #         "relax_delay": 1000, # [us]
        #         "list_of_all_qubits": list_of_all_qubits,
        #     },
        #
        #     "qubit_temp": {
        #         "reps": 100,
        #         "py_avg": 10,
        #         "start": [0.02] * 6, # [us]
        #         "expts":  [200] * 6,
        #         "step": 0.02, # [us]
        #         "relax_delay": 1000, # [us]
        #         "list_of_all_qubits": list_of_all_qubits,
        #     },
        #
        #     "power_rabi_ef": {
        #         "reps": 1000,
        #         "py_avg": 10,
        #         "start": [0.0] * 6, # [DAC units]
        #         "stop":  [1.0] * 6, # [DAC units]
        #         "steps": 100,
        #         "relax_delay": 1000, # [us]
        #         "list_of_all_qubits": list_of_all_qubits,
        #     },
        #
        #     "Ramsey_ef": {
        #         "reps": 100,
        #         "py_avg": 10,
        #         "start": [0.0] * 6, # [us]
        #         "stop":  [100] * 6, # [us]
        #         "steps": 100,
        #         "ramsey_freq": 0.05,  # [MHz]
        #         "relax_delay": 1000, # [us]
        #         "wait_time": 0.0, # [us]
        #         "list_of_all_qubits": list_of_all_qubits,
        #     },
        #
        #     "IQ_plot":{
        #         "steps": 5000, # shots
        #         "py_avg": 1,
        #         "reps": 1,
        #         "relax_delay": 1000, # [us]
        #         "SS_ONLY": False,
        #         "list_of_all_qubits": list_of_all_qubits,
        #     },
        #

    }

else:
    print('Please set variable FRIDGE to NEXUS, QUIET, or configure for your fridge as needed')
