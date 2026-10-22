from qick import *
import sys
import os
# sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos"))
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from tprocv2_demos.qick_tprocv2_experiments_mux.socProxy import makeProxy
import os
import datetime
import numpy as np


##ENTANGLING CHIP QICK BOARD SETTINGS

# ADC_attenuator MUST be above 16dB
#DAC_attenuator1 and 2 are for the resonators
#qubit_DAC_attenuator1 and 2 are for the qubits
class QICK_experiment:
    def __init__(self, folder, DAC_attenuator1 = 10, DAC_attenuator2 = 15, qubit_DAC_attenuator1 = 5,
                                     qubit_DAC_attenuator2 = 4, ADC_attenuator = 17, fridge = None):
        if fridge == "QUIET":
            # Where do you want to save data
            self.outerFolder = folder
            self.create_folder_if_not_exists(self.outerFolder)

            # attenuation settings
            self.DAC_attenuator1 = DAC_attenuator1
            self.DAC_attenuator2 = DAC_attenuator2
            self.ADC_attenuator = ADC_attenuator
            self.qubit_DAC_attenuator1 = qubit_DAC_attenuator1
            self.qubit_DAC_attenuator2 = qubit_DAC_attenuator2

            # Make proxy to the QICK
            self.soc, self.soccfg = makeProxy()
            print(self.soccfg)

            self.FSGEN_CH      = 0 # 0 for "old QICK", 5 for RF board 7/21/2025
            self.FSGEN_AMPL_CH = 2
            self.MIXMUXGEN_CH  = 4 # Readout resonator DAC channel
            self.MUXRO_CH = [2, 3, 4, 5]	
            ### for the QICK Box,
#           self.qubit_center_freq = 4350
#           self.res_center_freq   = 6330 
#           self.soc.rfb_set_gen_filter(self.MIXMUXGEN_CH, fc=self.res_center_freq / 1000, ftype='bandpass', bw=1.0) # bw is in units of GHz
#           self.soc.rfb_set_gen_filter(self.FSGEN_CH, fc=self.qubit_center_freq / 1000, ftype='bandpass', bw=1.7) # 4350 + 1700/2 = 5200 MHz, so freqs above that get filtered out
#           self.soc.rfb_set_ro_filter(self.MUXRO_CH[0], fc=self.res_center_freq / 1000, ftype='bandpass', bw=1.0) #readout ADC
#           # Set attenuator on DAC.
#           self.soc.rfb_set_gen_rf(self.MIXMUXGEN_CH, self.DAC_attenuator1, self.DAC_attenuator2)  # Verified 30->25 see increased gain in loopback
#           self.soc.rfb_set_gen_rf(self.FSGEN_CH, self.qubit_DAC_attenuator1, self.qubit_DAC_attenuator2)  # Verified 30->25 see increased gain in loopback
#           # Set attenuator on ADC.
#           ### IMPORTANT: set this to 30 and you get 60 dB of warm gain. Set to 0 and you get 90 dB of warm gain
#           self.soc.rfb_set_ro_rf(self.MUXRO_CH[0], self.ADC_attenuator)  # Verified 30->25 see increased gain in loopback

            # Qubit you want to work with
            self.QUBIT_INDEX = 5

            # Hardware Configuration
            self.hw_cfg = {
                # DAC
                "qubit_ch": [self.FSGEN_CH] * 4,  # Qubit Channel Port, Full-speed DAC
                "qubit_ampl_ch": [self.FSGEN_AMPL_CH] * 4,
                "res_ch": [self.MIXMUXGEN_CH] * 4,  # Single Tone Readout Port, MUX DAC
                "qubit_ch_ef": [self.FSGEN_CH] * 4, # Qubit ef Channel, Full-speed DAC
                "nqz_qubit": 1,
                "nqz_res": 2,
                # ADC
                "ro_ch": [self.MUXRO_CH] * 4,  # MUX readout channel
                "list_of_all_qubits": [0, 1, 2, 3]
            }


            #Readout Configuration
            self.readout_cfg = {
                "trig_time": 0.7,  # [Clock ticks] - get this value from TOF experiment
                # Changes related to the resonator output channel
                "mixer_freq": 6000,  # [MHz]
                'res_freq_ge': [6318.8,6276.7,6232.2,6174.7], 
                "res_gain_ge": [0.9, 0.9, 0.9, 0.9],
                "res_freq_ef": [6235.301, 6295.735, 6356.840, 6426.895], 
                "res_gain_ef": [0.95,0.9,0.95,0.55],  # [DAC units]
                "res_freq_fh": [6223.016, 6284.544, 6343.861, 6414.893],  # [MHz]
                "res_gain_fh": [0.95,0.9,0.95,0.55],  # [DAC units]
                "res_length": 3.0,  # [us] (1.0 for res spec)
                "res_phase": [0,0,0,0],
                "ro_phase": [0, 0, 0, 0],# Rotation Angle From QICK Function
                "threshold": [0,0,0,0], # Threshold for Distinguish g/e, from QICK Function
                #"res_ring_up_time": 4,  # Olivia May 17th
                #"qubit_is_in_g_threshold": 100000,
                #"edge_of_e_state_threshold": -26046,
                #"n_resets": 1,  # for active reset
                #"g_center": 0,  # for active reset
                #"e_center": 0,  # for active reset
                #"delay1_act_reset": 10,
                #"delay2_act_reset": 1
            }

            # Qubit Configuration
            self.qubit_cfg = {
                "qubit_freq_ge": [5379.66, 4994.46,5355.04,4948.82],
                "qubit_freq_chevron_detuned_ge": [4189.7582, 3820.4723, 4161.3726, 4463.15226], 
                "qubit_freq_ge_starked": [4189.737678, 3820.4723, 4161.3726, 4463.15226],
                "fwhm_w01_starked": None, #for err bars
                "fwhm_w01": None, #for err bars
                "qubit_gain_ge":  [0.01, 0.011, 0.03, 0.023],
                "qubit_ampl_gain_ge": [0.025] *4,
                "qubit_pi_len": 0.11,
                "qubit_length_ge": 15,
                "qubit_freq_ef": [4059.98, 3674.86, 4027.53, 4347.03],
                "qubit_freq_fh": [4016.3, 3450.8, 3988.44, 4292.73],
                "qubit_freq_ftores": [4016.3, 3644.76, 3988.44, 4292.73],
                "qubit_gain_ef":  [0.01, 0.018, 0.072, 0.028],
                "qubit_gain_fh": [0.001, 0.015, 0.0075, 0.1],
                'qubit_gain_ftores': [1]*4,
                "qubit_length_ef": 22,
                "qubit_length_ftores": [22]*6,
                "qubit_phase": 0,  # [deg]
                "sigma_ampl": [0.03, 0.03, 0.05, 0.04],
                "sigma": [0.1, 0.1, 0.15, 0.1],
                "sigma_ef": [0.17, 0.22, 0.42, 0.28], 
                "sigma_fh": [0.15, 0.21, 0.25, 0.29],  
                "pi_amp": [0.41,0.69,0.76,0.74],
                "pi_amp_ampl": [0.41,0.69,0.76,0.74], 
                "pi_ef_amp": [0.563, 0.673, 0.511, 0.7018],
                "pi_fh_amp": [0.563, 0.8023, 0.511, 0.7018],
                "qubit_mixer_freq": 4300,  # [MHz]

            }
        else:
            print("fridge variable is None or something else, please configure for your fridge or "
                  "change fridge to \"NEXUS\" or \"QUIET\" ")

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def mask_gain_res(self, QUBIT_INDEX, IndexGain = 1, num_qubits=6):
        """Sets the gain for the selected qubit to 1, others to 0."""
        filtered_gain_ge = [0] * num_qubits  # Initialize all gains to 0
        if 0 <= QUBIT_INDEX < num_qubits: #makes sure you are within the range of options
            filtered_gain_ge[QUBIT_INDEX] = IndexGain  # Set the gain for the selected qubit
        return filtered_gain_ge
