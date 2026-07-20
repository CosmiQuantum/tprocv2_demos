from qick import *
import sys
import os
# sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos"))
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from tprocv2_demos.qick_tprocv2_experiments_mux.socProxy import makeProxy
import os
import datetime
import numpy as np

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

            self.FSGEN_CH      = 5 # 0 for "old QICK", 5 for RF board 7/21/2025
            self.FSGEN_AMPL_CH = 2
            self.MIXMUXGEN_CH  = 4 # Readout resonator DAC channel
            self.MUXRO_CH = [2, 3, 4, 5, 6, 7]

            ### NEW for the RF board,
            self.qubit_center_freq = 4250 #Run 9a: 4250 # Value to be in the middle of the qubit freqs.
            self.res_center_freq   = 6330   # Run 9a: 6330 # To be in the middle of the res freqs. 3000-5000 see nothing,6000 and 7000 see something, 8000+ see nothing
            self.soc.rfb_set_gen_filter(self.MIXMUXGEN_CH, fc=self.res_center_freq / 1000, ftype='bandpass', bw=1.0) # bw is in units of GHz
            self.soc.rfb_set_gen_filter(self.FSGEN_CH, fc=self.qubit_center_freq / 1000, ftype='bandpass', bw=1.7) # 4250 + 1700/2 = 5100 MHz, so freqs above that get filtered out
            self.soc.rfb_set_ro_filter(self.MUXRO_CH[0], fc=self.res_center_freq / 1000, ftype='bandpass', bw=1.0) #readout ADC
            # Set attenuator on DAC.
            self.soc.rfb_set_gen_rf(self.MIXMUXGEN_CH, self.DAC_attenuator1, self.DAC_attenuator2)  # Verified 30->25 see increased gain in loopback
            self.soc.rfb_set_gen_rf(self.FSGEN_CH, self.qubit_DAC_attenuator1, self.qubit_DAC_attenuator2)  # Verified 30->25 see increased gain in loopback
            # Set attenuator on ADC.
            ### IMPORTANT: set this to 30 and you get 60 dB of warm gain. Set to 0 and you get 90 dB of warm gain
            self.soc.rfb_set_ro_rf(self.MUXRO_CH[0], self.ADC_attenuator)  # Verified 30->25 see increased gain in loopback


            # Qubit you want to work with
            self.QUBIT_INDEX = 5

            # Hardware Configuration
            self.hw_cfg = {
                # DAC
                "qubit_ch": [self.FSGEN_CH] * 6,  # Qubit Channel Port, Full-speed DAC
                "qubit_ampl_ch": [self.FSGEN_AMPL_CH] * 6,
                "res_ch": [self.MIXMUXGEN_CH] * 6,  # Single Tone Readout Port, MUX DAC
                "qubit_ch_ef": [self.FSGEN_CH]*6, # Qubit ef Channel, Full-speed DAC
                "nqz_qubit": 1,
                "nqz_res": 2,
                # ADC
                "ro_ch": [self.MUXRO_CH] * 6,  # MUX readout channel
                "list_of_all_qubits": [0, 1, 2, 3, 4, 5]
            }

            #Readout Configuration
            self.readout_cfg = {
                "trig_time": 0.75,  # [Clock ticks] - get this value from TOF experiment (updated by Arianna Oct 9 2025 QICK box). Stayed the same for run 9.
                # Changes related to the resonator output channel
                "mixer_freq": 6000,  # [MHz]
                #"res_freq_ge": [6217, 6276, 6335, 6407, 6476, 6538],  # MHz, run 5
                #'res_freq_ge': [6227.237, 6289.175, 6348.550, 6419.665, 6485.315, 6552.300], # run8
                'res_freq_ge': [6236.2014, 6296.685, 6358.0, 6427.895, 6494.4, 6562.500], # run 9c: 6235.097, 6295.745, 6356.840, 6426.675, 6493.5, 6561.2
                #'res_freq_ge': [6223.097, 6284.55, 6343.95, 6414.934, 6481.4, 6547.250],  # run 7
                # "res_freq_ge": [6191.419, 6216.1, 6292.361, 6405.77, 6432.759, 6468.481],  # MHz, run 4a
                # "res_gain_ge": [1] + [0]*5,
                "res_gain_ge": [0.95, 0.9, 0.95, 0.55, 0.55, 0.95], #[1, 1, 1, 1, 1, 1],
                #"res_gain_ge": [0.96, 1, 0.76, 0.58, 0.75, 0.57], # Joyce 04/07 DAC 0
                # set_res_gain_ge(QUBIT_INDEX), #utomatically sets all gains to zero except for the qubit you are observing
                # "res_gain_ge": [1,1,0.7,0.7,0.7,1], #[0.4287450656184295, 0.4903077560386716, 0.4903077560386716, 0.3941941738241592, 0.3941941738241592, 0.4903077560386716],  # DAC units
                # "res_freq_ef": [7149.44, 0, 0, 0, 0, 0], # [MHz]
                # "res_gain_ef": [0.6, 0, 0, 0, 0, 0], # [DAC units]
                "res_freq_ef": [6236.157, 6296.658, 6358.029, 6427.713, 6494.240, 6562.153], # [MHz] Run 9c: 6228.237, 6289.975, 6349.7, 6420.465, 6485.360, 6553.150
                #'res_freq_ef': [6227.076, 6289.092, 6348.423, 6419.612, 6485.195, 6552.256], #Run 8
                "res_gain_ef": [0.95,0.9,0.95,0.55,0.55,0.95],  # [DAC units]
                "res_freq_fh": [6223.016, 6284.544, 6343.861, 6414.893, 6414.893, 6546.754],  # [MHz]
                "res_gain_fh": [0.95,0.9,0.95,0.55,0.55,0.95],  # [DAC units]
                "res_length": 2.0,  # [us] (1.0 for res spec)
                "res_phase": [0,0,0,0,0,0],#[ -180+((1.281174-2.6703) * 180/np.pi), -10, 85,
                            #   0, 150,
                            # -90], #Joyce 3/11
                #"res_phase": [(0.19-0.38) * 180/np.pi, (2.07-3.12-1.16) * 180/np.pi, (-0.35+2.28) * 180/np.pi,
                   #           (-1.36+1.68+1.1) * 180/np.pi, (-2.4-1.5) * 180/np.pi, (-0.56+1.18) * 180/np.pi],
                # [-0.1006 *360/np.pi, -2.412527*360/np.pi, -1.821284*360/np.pi, -1.90962*360/np.pi, -0.566479*360/np.pi, -0.5941687*360/np.pi], # Rotation Angle From QICK Function, is the ang of 10 ss angles per qubit
                # "res_phase": [0]*6,#[-0.1006 *360/np.pi, -2.412527*360/np.pi, -1.821284*360/np.pi, -1.90962*360/np.pi, -0.566479*360/np.pi, -0.5941687*360/np.pi], # Rotation Angle From QICK Function, is the ang of 10 ss angles per qubit
                "ro_phase": [0.00, 0, 0, 0, 0, 0],  # Rotation Angle From QICK Function
                "threshold": [-5.84, 1, -5, -3, -1, 4], #Joyce 3/11
                #"threshold": [7.3961, -12.5812, 4.8613, -7.5323, 7.0689, 4.6805], # Threshold for Distinguish g/e, from QICK Function
                "res_ring_up_time": 4,  # Olivia May 17th
                "qubit_is_in_g_threshold": 100000, #100000,#-8837,
                "edge_of_e_state_threshold": -26046,
                "n_resets": 3,  # for active reset
                "g_center": 0,  # for active reset
                "e_center": 0,  # for active reset
            }

            # Qubit Configuration
            self.qubit_cfg = {
                "qubit_freq_ge":  [4228.49, 3852.44, 4201.92, 4506.32, 4518.33, 5054.72], # run 9c: 4229.89, 3853.24, 4197.02, 4500.9, 4515.34, 5054.02
                "qubit_freq_chevron_detuned_ge": [4189.7582, 3820.4723, 4161.3726, 4463.15226, 4471.43854, 4997.86], # Olivia May 17
                "qubit_freq_ge_starked": [4189.737678, 3820.4723, 4161.3726, 4463.15226, 4471.4469, 4997.86], # Olivia 4/04 for zeno/stark tone
                "fwhm_w01_starked": None, #for err bars
                "fwhm_w01": None, #for err bars
                "qubit_gain_ge":  [0.01, 0.009, 0.0255, 0.022, 0.038, 0.08], # run 9c: [0.01, 0.009, 0.028, 0.021, 0.16, 0.08]
                "qubit_ampl_gain_ge": [0.025] *6,
                "qubit_pi_len": 0.11, # Olivia May 17th
                # [0.4287450656184295, 0.4287450656184295, 0.4903077560386716, 0.6, 0.4903077560386716, 0.4287450656184295], # For spec pulse
                "qubit_length_ge": 15,  # 15 [us] for Qspec Pulse
                "qubit_freq_ef": [4054.38, 3674.86, 4027.53, 4334.48, 4346.29, 4887.39], # run 9c: 4055.94, 3675.69, 4022.54, 4329.42, 4343.53, 4886.5
                # "qubit_freq_ef": [4020.48, 3650.81, 3999.01, 4302.33, 4313.06, 4848.81], # run 8 # [MHz] Freqs of Qubit e/f Transition
                "qubit_freq_fh": [4016.3, 3450.8, 3988.44, 4292.73, 4292.73, 4833.17],
                "qubit_freq_ftores": [4016.3, 3644.76, 3988.44, 4292.73, 4303.18, 4833.17],
                "qubit_gain_ef":  [0.01, 0.018, 0.075, 0.028, 0.087, 0.15], # run 9c: 0.01, 0.016, 0.076, 0.028, 0.1, 0.15
                "qubit_gain_fh": [0.001, 0.015, 0.0075, 0.1, 0.15, 0.005],
                'qubit_gain_ftores': [1]*6,#[0.2, 0.14, 0.04, 0.17, 0.13, 0.08],
                # [0.01, 0.05, 0.05, 0.05, 0.01, 0.5], # [DAC units] Pulse Gain
                "qubit_length_ef": 22, #22.0,
                "qubit_length_ftores": [22]*6,  # 25.0,
                "qubit_phase": 0,  # [deg]
                #"sigma": [0.15]*6,  # [us] for Gaussian Pulse (5+10 DAC atten for qubit)
                "sigma_ampl": [0.03, 0.03, 0.05, 0.04, 0.05, 0.05], #DAC 0 04/07
                "sigma": [0.102, 0.10, 0.185, 0.21, 0.25, 0.98], #  run 9c: [0.11, 0.05, 0.19, 0.21, 0.35, 0.98] # DAC 2
                #"sigma": [0.05, 0.09, 0.07, 0.065, 0.09, 0.3],  # Goal: cut sigma in half [us] for Gaussian Pulse (5+4 DAC atten for qubit)
                # "pi_amp": [0.92, 0.87, 0.75, 0.73, 0.77, 0.78], # old RR values
                "sigma_ef": [0.12, 0.22, 0.42, 0.23, 0.2, 0.84], #run 9: updated all except Q5 # run 8: [0.1, 0.25, 0.2, 0.25, 0.17, 0.28],
                "sigma_fh": [0.15, 0.21, 0.25, 0.29, 0.28, 0.5],  # [us] for Gaussian Pulse, #Arianna 3/27
                "active_reset_test_sigma": 0.102, # just used for testing stuff
                "pi_amp": [0.808643, 0.8, 0.842336, 0.65, 0.66, 0.69], # Arianna 10/10, run 8
                "pi_amp_ampl": [0.5942, 0.634499, 0.76542, 0.7754, 0.55393, 0.9], # Joyce 04/07 DAC 0
                #"pi_amp": [1.0, 0.93, 0.77, 0.8, 0.81, 0.9], # Eyeballed by Sara today (5+10 DAC atten for qubit)
                #"pi_amp": [0.7, 0.95, 0.75, 0.78, 0.77, 0.8],  # With shorter sigma (5+4 DAC instead of 5+5 DAC atten for qubit)
                "pi_ef_amp": [0.563, 0.673, 0.511, 0.7018, 0.6751, 0.589], # Arianna 3/27
                "pi_fh_amp": [0.563, 0.8023, 0.511, 0.7018, 0.6751, 0.589],  # Arianna 3/27
                "qubit_mixer_freq": 4300,  # [MHz]

            }

        elif fridge == "NEXUS":
            # Where do you want to save data
            self.outerFolder = folder
            self.create_folder_if_not_exists(self.outerFolder)

            # attenuation settings
            self.DAC_attenuator1 = DAC_attenuator1
            self.DAC_attenuator2 = DAC_attenuator2
            self.ADC_attenuator = ADC_attenuator

            # Make proxy to the QICK
            self.soc, self.soccfg = makeProxy()
            print(self.soccfg)

            self.FSGEN_CH = 10  # set to 8 for bias spectroscopy, and 10 for everything else (pi pulses, RR)
            self.MIXMUXGEN_CH = 4  # Readout resonator DAC channel
            self.MUXRO_CH = [2, 3, 4, 5]
            # self.MUXRO_CH_RF = 5  # New variable that we need for QICK box

            # self.TESTCH_DAC = 5 # loopback channel for RF board
            # self.TESTCH_ADC = 0  # loopback channel for RF board
            # self.TESTCH_ADC_RF = 4  # New variable that we need for QICK box

            # From mux_simultaneous
            # GEN_CH8 = 8
            # GEN_CH10 = 10
            # GEN_CH12 = 12
            # GEN_CH14 = 14
            # MIXMUXGEN_CH = 4
            # MUXRO_CH = [2, 3, 4, 5]
            # # Qubit you want to work with
            # QUBIT_INDEX = 0

            ### NEW for the RF board
            # self.qubit_center_freq = 4400  # To be in the middle of the qubit freqs.
            # self.res_center_freq = 6330  # To be in the middle of the res freqs. 3000-5000 see nothing,6000 and 7000 see something, 8000+ see nothing
            # self.soc.rfb_set_gen_filter(self.MIXMUXGEN_CH, fc=self.res_center_freq / 1000, ftype='bandpass', bw=1.0)
            # self.soc.rfb_set_gen_filter(self.FSGEN_CH, fc=self.qubit_center_freq / 1000, ftype='bandpass', bw=1.0)
            # self.soc.rfb_set_ro_filter(self.MUXRO_CH_RF, fc=self.res_center_freq / 1000, ftype='bandpass', bw=1.0)
            # # Set attenuator on DAC.
            # self.soc.rfb_set_gen_rf(self.MIXMUXGEN_CH, self.DAC_attenuator1, self.DAC_attenuator2)  # Verified 30->25 see increased gain in loopback
            # self.soc.rfb_set_gen_rf(self.FSGEN_CH, 5, 4)  # Verified 30->25 see increased gain in loopback
            # # Set attenuator on ADC.
            # ### IMPORTANT: set this to 30 and you get 60 dB of warm gain. Set to 0 and you get 90 dB of warm gain
            # self.soc.rfb_set_ro_rf(self.MUXRO_CH_RF, self.ADC_attenuator)  # Verified 30->25 see increased gain in loopback

            # Qubit you want to work with
            self.QUBIT_INDEX = 0

            # Hardware Configuration
            self.hw_cfg = {
                # DAC
                "qubit_ch": [self.FSGEN_CH] * 4,  # Qubit Channel Port, Full-speed DAC
                "res_ch": [self.MIXMUXGEN_CH] * 4,  # Single Tone Readout Port, MUX DAC
                # "qubit_ch_ef": [GEN_CH5]*6, # Qubit ef Channel, Full-speed DAC
                "nqz_qubit": 2,
                "nqz_res": 2,
                # ADC
                "ro_ch": [self.MUXRO_CH] * 4,  # MUX readout channel,
                "list_of_all_qubits": [0, 1, 2, 3]
            }

            # Readout Configuration
            self.readout_cfg = {
                "trig_time": 0.75,  # [Clock ticks] - get this value from TOF experiment
                # Changes related to the resonator output channel
                "mixer_freq": 5500,  # [MHz]
                "res_freq_ge": [6187.191, 5827.678, 6074.095, 5958.453],  # MHz #5958.8 (Grace)
                "res_gain_ge": [0.4, 0.4, 0.4, 0.3875],  # [0.15]*4, #[1, 1, 1, 1],
                "res_length": 4.6,  # 10,  # [us] (1.0 for res spec)
                "res_phase": [0] * 4,
                "ro_phase": [0, 0, 0, 0],  # [0] * 4,  # Rotation Angle From QICK Function
            }

            # Qubit Configuration
            self.qubit_cfg = {
                "qubit_mixer_freq": 4300,  # [MHz]
                "qubit_freq_ge": [4909, 4749.4, 4569, 4756],  # Freqs of Qubit g/e Transition
                "qubit_gain_ge": [0.2] * 4,  # [0.008] * 4,#[0.2, 0.2, 0.2, 0.01], #[0.2] * 4,  #0.07
                "qubit_length_ge": 20,  # [us] for spec Pulse
                "qubit_phase": 0,  # [deg]
                "sigma": [0.04, 0.025, 0.04, 0.03],  # [0.08, 0.15, 0.11, 0.09], # TO DO CHANGE THIS (11/26)
                "pi_amp": [1.0, 0.93, 0.77, 0.846],  # TO DO CHANGE THIS (11/26)

                # "qubit_freqs_ge": [4909, 4749.4, 4569, 4759],  # Freqs of Qubit g/e Transition
                # "qubit_gains_ge": [1] * 4,  # [0.05] * 4
                # "qubit_phases": [0] * 4,  # [deg]
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
