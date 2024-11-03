import sys
import os
import numpy as np
sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_003_qubit_spec_ge import QubitSpectroscopy
from section_004_amp_rabi_ge import AmplitudeRabiExperiment
from section_006_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_007_save_T1 import Save
from system_config import *


# N benchmark
n = 10
Qs = [0,1,2,3,4,5]
save_r = 2 #how many rounds to save after
signal = 'None' # where the signal is (after ss/angle optimization). Put 'None' if no optimization has happened

t1_data = {Q: {'T1': [None]*save_r, 'Errors': [None]*save_r, 'Dates': [None]*save_r,
               'I': [None]*save_r, 'Q':[None]*save_r, 'Delay Times': [None]*save_r, 'Fit': [None]*save_r} for Q in range(6)}
batch_num=0
j = 0
angles=[]
while j < n:
    j += 1
    for QubitIndex in Qs:

        # ---------------------TOF------------------------
        #tof = TOFExperiment(QubitIndex, outerFolder, j)
        #tof.run(soccfg, soc)

        #---------------------Res spec---------------------
        res_spec = ResonanceSpectroscopy(QubitIndex, outerFolder, j)
        res_freqs = res_spec.run(soccfg, soc)

        #-----------------Roll Signal into I---------------
        #get the average theta value, then use that to rotate the signal. Plug that value into system_config res_phase
        # ss = SingleShot(QubitIndex, outerFolder, j, round(4, 3))
        # fid, angle = ss.run(soccfg, soc)
        # angles.append(angle)
        # print(angles)
        # print('avg theta: ', np.average(angles))

        #--------------------Qubit spec--------------------
        # Right now this does not return qubit frequency or update the config with the found values, do we want to change that?
        #only need to update the res_freqs here, it will update the imported config and will change all following classes
        q_spec = QubitSpectroscopy(QubitIndex, outerFolder, res_freqs, j, signal)
        qubit_freq = q_spec.run(soccfg, soc)

        #-----------------------Rabi-----------------------
        # Right now this does not have updated fitting, we need to make sure this fit works every time
        rabi = AmplitudeRabiExperiment(QubitIndex, outerFolder, j, qubit_freq, signal)
        rabi.run(soccfg, soc)

        #------------------------T1-------------------------
        # Also need to update the fit here. Maybe do custom fits for all three of these classes (QSpec/Rabi/T1)
        t1 = T1Measurement(QubitIndex, outerFolder, j, qubit_freq, signal)
        t1_est, t1_err, I, Q, delay_times, q1_fit_exponential = t1.run(soccfg, soc)

        #---------------------Collect Results----------------
        t1_data[QubitIndex]['T1'][j - batch_num*save_r - 1] = t1_est
        t1_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t1_err
        t1_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = datetime.datetime.now()
        t1_data[QubitIndex]['I'][j - batch_num*save_r - 1] = I
        t1_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = Q
        t1_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = delay_times
        t1_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = q1_fit_exponential

    #-----------------------Potentially Save---------------
    # Check if you are at the right round number. If so, then save all of the data and change the round num so you rereplace data starting next round
    if j % save_r == 0:
        batch_num+=1
        saver = Save(outerFolder,t1_data, batch_num, save_r)
        saver.save_to_h5()





