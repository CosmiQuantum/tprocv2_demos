import numpy as np

FRIDGE = "QUIET"  # change to "NEXUS" as needed

if FRIDGE == "QUIET":
    VNA_res = np.array([6235.097, 6295.745, 6356.840, 6426.675, 6493.5, 6561.2]) # run 9c # run 9a:  6228.487, 6290.125, 6349.800, 6420.615, 6486.0, 6553.600
    VNA_qubit = np.array([4229.89, 3853.24, 4197.02, 4500.9, 4515.34, 5054.02])  # run 9c
    ef_qfreqs = np.array([4055.94, 3675.69, 4022.54, 4329.42, 4343.53, 4886.5]) # Qubit freqs e/f Transition, run 9a
    fh_qfreqs = np.array([3820.97, 3450.85, 3798.97, 4110, 4660, 4660.28]) # Qubit freqs f/h Transition, old run
    two_photon_qfreqs=np.array([4107.61, 3739.36, 4086.36, 4388.34, 4399.3, 4933.52]) # qubit freqs, two photon peak between ge and ef qubit freqs, old run

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

        "res_spec": { # run 9a
            "reps": 270, # 300
            "rounds": 1,
            "start": -1.7,       # MHz
            "step_size": 0.04,  #0.04 # MHz
            "steps": 90, #90,
            "relax_delay": 10,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        # "res_spec": { # zooming out
        #     "reps": 270,
        #     "rounds": 1,
        #     "start": -8.0,  # MHz
        #     "step_size": 0.16,  # MHz
        #     "steps": 101,
        #     "relax_delay": 10,  # us
        #     "list_of_all_qubits": list_of_all_qubits,
        # },


        # "res_spec_ef": { # for thomas roth data
        #     "reps": 3000,
        #     "rounds": 1,
        #     "start": -1.0,       # MHz, sweep from -1.0 to +1.0 around center
        #     "step_size": 0.005,  # MHz (5 kHz)
        #     "steps": 401,
        #     "relax_delay": 900,  # [us]
        #     "list_of_all_qubits": list_of_all_qubits,
        # },

        "res_spec_ef": {
            "reps": 420, # 300
            "rounds": 1,
            "start": -1.7,       # MHz
            "step_size": 0.04,   # MHz
            "steps": 90,
            "relax_delay": 900,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        # "qubit_spec_ge": { # broad search
        #     "reps": 600,  # 300
        #     "rounds": 1,  # 10
        #     "start": list(VNA_qubit - 50),  # [MHz] #-300 #-15
        #     "stop": list(VNA_qubit + 50),  # [MHz] #+15
        #     "steps": 300,  # 100
        #     "relax_delay": 10,  # 1000 # [us]
        #     "list_of_all_qubits": list_of_all_qubits,
        # },

        # "qubit_spec_ge": { # Q5
        #     "reps": 600,  # 300
        #     "rounds": 1,  # 10
        #     "start": list(VNA_qubit - 6),  # [MHz] #-300 #-15
        #     "stop": list(VNA_qubit + 6),  # [MHz] #+15
        #     "steps": 250,  # 100
        #     "relax_delay": 10,  # 1000 # [us]
        #     "list_of_all_qubits": list_of_all_qubits,
        # },

        "qubit_spec_ge": { # RR
            "reps": 600, #300
            "rounds": 1, #10
            "start": list(VNA_qubit-2.0), # [MHz]
            "stop": list(VNA_qubit+2.0), # [MHz]
            "steps": 215, # 220 for -3 +3, 400 for -40, +40, 320 for -6 +6
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
        # "qubit_spec_ef": {
        #     "reps": 2200,  # 300
        #     "rounds": 1,  # 10
        #     "start": list(ef_qfreqs - 195),  # [MHz] #-300
        #     "stop": list(ef_qfreqs + 10),  # [MHz]
        #     "steps": 400,  # 205 for -2 +2
        #     "relax_delay": 900,  # 1000,  # [us]
        #     "list_of_all_qubits": list_of_all_qubits,
        # },
        "qubit_spec_ef": {
            "reps": 1600,  # 2200
            "rounds": 1,  # 10
            "start": list(ef_qfreqs - 4),  # [MHz] #-300
            "stop": list(ef_qfreqs + 4),  # [MHz]
            "steps": 200, # 205 for -2 +2
            "relax_delay": 900, #1000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "qubit_spec_fh": {
            "reps": 20000,  # 300
            "rounds": 1,  # 10
            "start": list(fh_qfreqs - 0.3), # [MHz] #-300 #-6
            "stop": list(fh_qfreqs +  0.2),  # [MHz] #6
            "steps": 180,#450,  # 1000 #450
            "relax_delay": 900,  # 1000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "qubit_spec_ftores": {
            "reps": 10000,  # 300
            "rounds": 1,  # 10
            "start": list((VNA_qubit + ef_qfreqs) - VNA_res - 200),  # [MHz] #-300
            "stop":  list((VNA_qubit + ef_qfreqs) - VNA_res + 200),  # [MHz]
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
            "reps": 120,#500,
            "rounds": 1,  # 5
            "start": [0] * 6,  # [DAC units]
            "stop": [1] * 6,#[1.0] * 6,  # [DAC units]
            "steps": 100, #50,
            "relax_delay": 900,#1000,#1000,#1000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "length_rabi_ge": {
            "reps": 100,#500,
            "rounds": 1,
            "start": [0.01] * 6,  # [us]
            "stop": [25] * 6,#[0.7] * 6,  # [us]
            "steps": 500,
            "relax_delay": 900,# [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "length_rabi_Qtemps": {
            "reps": 400,#500,
            "rounds": 1,
            "start": [0.02] * 6,  # [us]
            "stop": [2] * 6,#[0.7] * 6,  # [us]
            "steps": 200,
            "relax_delay": 900,# [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "length_rabi_ge_pi_len": { #for the zeno tests, bare qubit frequency rabi
            "reps": 300,  # 500,
            "rounds": 1,
            "start": [0.01] * 6,  # [us]
            "stop": [4] * 6,  # [0.7] * 6,  # [us]
            "steps": 60,
            "relax_delay": 900,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "length_rabi_ge_qze": {
            "reps": 400,  # 500,
            "rounds": 1, #600
            "start": [0.01] * 6,  # [us]
            "stop": [10] * 6,   #[3] * 6,   # [us]
            "steps": 150,  # 140,
            "relax_delay": 900,# [us]
            "list_of_all_qubits": list_of_all_qubits,
            "qze_mask": [],
            "zeno_pulse_width": 0.007,
            "zeno_pulse_period": 0.10,
        },

        "power_rabi_ef": {
            "reps": 120,#400, # for Pg pulse sequence during rpm, or regular ef rabi
            "reps2": 5500, #this is only used for the experiment that uses e-f rabi to calculate qubit temperatures (rpm). Pe pulse sequence
            "rounds": 1,
            "start": [0.0] * 6,  # [DAC units]
            "stop": [1.0] * 6,  # [DAC units]
            "steps": 80, #110
            "relax_delay": 900,  # [us]
        },
        "power_rabi_fh": {
            "reps": 200,
            "reps2": 850,  # this is only used for the experiment that uses e-f rabi to calculate qubit temperatures.
            "rounds": 1,
            "start": [0.0] * 6,  # [DAC units]
            "stop": [1.0] * 6,  # [DAC units]
            "steps": 155,
            "relax_delay": 900,  # [us]
        },
        "T1_ge": {
            "reps": 300, #300
            "rounds": 1, #1
            "start": [0.0] * 6,  # [us]
            "stop": [200]*6, #[250.0] * 6,  # [us]
            "steps": 70,
            "relax_delay": 900,  # [us] ### Should be >10x T1!
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits
        },

        "T1_IBM_qze": {
            "reps": 1000,  # 300
            "rounds": 1,  # 1
            "start": [30] * 6,  # [us]
            "stop": [31] * 6,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 1,
            "relax_delay": 900,  # [us] ### Should be >10x T1!
            "wait_time": 30,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "T1_fe": {
            "reps": 200,  # 300
            "rounds": 1,  # 1
            "start": [0.0] * 6,  # [us]
            "stop": [200] * 6,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 150,
            "relax_delay": 900,  # [us] ### Should be >10x T1!
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "T1_fg": {
            "reps": 200,  # 300
            "rounds": 1,  # 1
            "start": [0.0] * 6,  # [us]
            "stop": [200] * 6,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 150,
            "relax_delay": 900,  # [us] ### Should be >10x T1!
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
            "relax_delay": 900,
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
            "reps": 500, #300
            "rounds": 1,#10
            "start": [0.0] * 6, # [us]
            "stop":  [60] * 6, # [us]
            "steps": 215,
            "ramsey_freq": 0.12,  # [MHz]
            "relax_delay": 900, # [us] the time to wait to let the qubit to relax to gnd again after exciting it (make it way above T1)
            "wait_time": 0.0, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "Ramsey_ge_correction": {
            "reps": 20,  # 20,  # 300
            "rounds": 2,  # 10
            "start": 0.0,  # [us]
            "stop": 1,  # [us]
            "steps": 50,
            "ramsey_freq": 4,  # [MHz]
            "relax_delay": 4000,
            # [us] the time to wait to let the qubit to relax to gnd again after exciting it (make it way above T1)
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "SpinEcho_ge": {
            "reps": 370,
            "rounds": 1,
            "start": [0.0] * 6, # [us]
            "stop":  [80] * 6, # [us]
            "steps": 100,
            "ramsey_freq": 0.12,  # [MHz]
            "relax_delay": 900, # [us]
            "wait_time": 0.0, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "IQ_plot":{ # not used currently
            "steps": 5000, # shots
            "py_avg": 1,
            "reps": 1,
            "relax_delay": 900, # [us]
            "SS_ONLY": False,
            "list_of_all_qubits": list_of_all_qubits,
        },

        "Readout_Optimization":{ # used for SSF
            "steps": 15000, # shots
            "py_avg": 1,
            "gain_start" : [0, 0, 0, 0], # used in readout-optimization scripts
            "gain_stop" : [1, 0, 0, 0], # used in readout-optimization scripts
            "gain_step" : 0.1, # used in readout-optimization scripts
            "freq_start" : [6176.0, 0, 0, 0], # used in readout-optimization scripts
            "freq_stop" : [6178.0, 0, 0, 0], # used in readout-optimization scripts
            "freq_step" : 0.1, # used in readout-optimization scripts
            "relax_delay": 900, # 600, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "Pi_Pulse_Fid": {
            "shots_per_pulse_count": 1000,   # reps; number of single-shot measurements for each N to get statistical info
            "py_avg": 1,                      # keep as 1 for single-shot data
            "min_pi_pulses": 0,               # first N value
            "max_pi_pulses": 40,              # last N value
            "pi_pulse_step": 1,               # increment in N
            "relax_delay": 900,               # us, wait time between shots
            "list_of_all_qubits": list_of_all_qubits
        }

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
            "relax_delay": 900,  # [us]
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
            "steps": 15000,  # shots
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
        #         "relax_delay": 900, # [us]
        #         "list_of_all_qubits": list_of_all_qubits,
        #     },
        #
        #     "qubit_spec_ef": {
        #         "reps": 100,
        #         "py_avg": 10,
        #         "start": [2750, 0, 0, 0, 0, 0], # [MHz]
        #         "stop":  [2850, 0, 0, 0, 0, 0], # [MHz]
        #         "steps": 500,
        #         "relax_delay": 900, # [us]
        #         "list_of_all_qubits": list_of_all_qubits,
        #     },
        #
        #     "qubit_temp": {
        #         "reps": 100,
        #         "py_avg": 10,
        #         "start": [0.02] * 6, # [us]
        #         "expts":  [200] * 6,
        #         "step": 0.02, # [us]
        #         "relax_delay": 900, # [us]
        #         "list_of_all_qubits": list_of_all_qubits,
        #     },
        #
        #     "power_rabi_ef": {
        #         "reps": 1000,
        #         "py_avg": 10,
        #         "start": [0.0] * 6, # [DAC units]
        #         "stop":  [1.0] * 6, # [DAC units]
        #         "steps": 100,
        #         "relax_delay": 900, # [us]
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
        #         "relax_delay": 900, # [us]
        #         "wait_time": 0.0, # [us]
        #         "list_of_all_qubits": list_of_all_qubits,
        #     },
        #
        #     "IQ_plot":{
        #         "steps": 5000, # shots
        #         "py_avg": 1,
        #         "reps": 1,
        #         "relax_delay": 900, # [us]
        #         "SS_ONLY": False,
        #         "list_of_all_qubits": list_of_all_qubits,
        #     },
        #

    }

else:
    print('Please set variable FRIDGE to NEXUS, QUIET, or configure for your fridge as needed')
