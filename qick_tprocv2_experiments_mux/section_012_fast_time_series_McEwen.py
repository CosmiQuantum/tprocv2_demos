import os
import copy
import datetime
import numpy as np
import logging
np.set_printoptions(threshold=1000000000000000)
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import h5py
# Assuming these are defined elsewhere and importable
from build_task import *
from build_state import *
from expt_config import *
from system_config import QICK_experiment


##----------------------------------------------------------------------------------------------
## Program class definition for the standard McEwen-style relaxation measurement
class FastRelaxationProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        gen_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        
        ##Declare generators: these are the generators needed for readout (resonator). Then do a loop over these for the various readouts needed
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=gen_ch)

        #Now we add a qubit pulse, declare its channel generator
        self.add_pulse(ch=gen_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_chs[0],
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        #         self.add_loop("shotloop", cfg["steps"]) # number of total shots
        self.add_loop("gainloop", cfg["expts"])  # Pulse / no Pulse loop

    #Body: what actually runs
    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play pulse
        self.delay_auto(cfg['meas_wait']) #Wait for some time to let qubit have chance to relax
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])





        
##----------------------------------------------------------------------------------------------
## Program class definition for the fast TLS excitation program
class FastTLSExcitationProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        stark_ch = cfg['qubit_ampl_ch']

        #Declare the generator for the resonator and readout channels
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        #Add a readout pulse
        self.add_pulse(ch=res_ch, name="readout_pulse",
                               style="const",
                               length=cfg['res_length'],
                               mask=cfg["list_of_all_qubits"],
                               )

        #Declare the generator for the stark pulse, which moves us onto resonance with the TLS. Here we need both a detuning and a stark gain        
        self.declare_gen(ch=stark_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_gauss(ch=stark_ch, name="stark_ramp", sigma=cfg['stark_sigma'], length = cfg['stark_sigma'] *2)
        self.add_pulse(ch=stark_ch, name="stark_tone",
                       style="flat_top",
                       envelope="stark_ramp",
                       freq=cfg['qubit_freq_ge'] + cfg['tls_detuning'], #Use the detuning found for this determined TLS (either +/- 15MHz or so)
                       phase=cfg['qubit_phase'], 
                       gain = cfg['tls_gain'], #Use the gain found for this determined TLS (0 to 1 -- should check with Joyce to make sure this clears)
                       length=cfg['stark_length'],
                       )

    #Run the body
    def _body(self, cfg):
        
        self.delay_auto(t=cfg['pre_stark_delay'], tag='pre_stark_delay')  # wait for qubit pi pulse to finish
        self.pulse(ch=self.cfg['qubit_ampl_ch'], name="stark_tone", t=0)  # play stark tone
        self.delay_auto(t=0.01, tag='wait stark')  # wait for stark tone to finish
        self.delay(t=cfg['readout_pulse_delay']) #wait for resonator to return to vacuum
        self.pulse(ch=cfg['res_ch'], name="readout_pulse", t=0)  # play readout pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])  # get readout


##----------------------------------------------------------------------------------------------
# Main class definition for this relaxation-excitation test
class FastRelEx:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, save_figs=False, experiment=None,
                 target_tls=True, tls_gain= -1, tls_detuning=0, verbose = False, logger = None, qick_verbose=True):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "Fast_Relaxation_Excitation"
        self.Qubit = 'Q' + str(self.QubitIndex)

        #Maintining round number -- now it takes on a "which TLS spec ID am I on?" form
        self.round_num = round_num
        self.save_figs = save_figs
        self.experiment = experiment
        self.number_of_qubits = number_of_qubits
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")
        self.target_tls = target_tls
        self.tls_gain = tls_gain
        self.tls_detuning = tls_detuning

        #Now we get the experiment
        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)

            #REL Waypoint: Print q_config
            
            #This seems like it doesn't do anything for single-shot and fastexcitationrelaxation classes except return the config for a particular named measurement
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.tls_cfg = { "tls_gain" : tls_gain, "tls_detuning" : tls_detuning }
            
            #This config is what goes into the averager programs, and it contains the "harder" parameters in the q_config (resonator frequencies, qubit frequencies, etc.)
            #and the softer parameters in the exp_config.
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg, **self.tls_cfg}

            #REL Waypoint: print self.config
            
            
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} FastRelEx configuration: ', self.config)
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} FastRelEx configuration: {self.config}')
        


    #Experiment run function: what we call from the master script
    def run(self):

        #outputs
        raw_I = []
        raw_Q = []
        
        #Use the conditional "target_tls" within this object to determine whether we run our N shots. If we target the TLS, 
        #we don't throw in a pi pulse before measuring.
        if target_tls == True:
            fastExProg = FastTLSExcitationProgram(self.experiment.soccfg,
                                                  reps=self.config['reps'],
                                                  final_delay=self.config['relax_delay'],
                                                  cfg=self.config)
            iq_list_avg = fastExProg.acquire(self.experiment.soc,
                                             soft_avgs=1,
                                             progress=True)

            #REL Waypoint: check dimensionality of data here
            raw_IQ = fastExProg.get_raw()
            raw_I.append(raw_IQ[self.QubitIndex][:,0,0])
            raw_Q.append(raw_IQ[self.QubitIndex][:,0,1])

        
        else:
            #If we're not targeting a tls (for example, if we can't find one), then just do a fast relaxation only program, which does
            #include a pi pulse
            
            fastRelProg = FastRelaxationProgram(self.experiment.soccfg,
                                                reps=self.config['reps'],
                                                final_delay=self.config['relax_delay'],
                                                cfg=self.config)
            iq_list_avg = fastRelProg.acquire(self.experiment.soc,
                                          soft_avgs=1,
                                          progress=True)

            #REL Waypoint: check dimensionality of data here
            raw_IQ = fastRelProg.get_raw()
            raw_I.append(raw_IQ[self.QubitIndex][:,0,0])
            raw_Q.append(raw_IQ[self.QubitIndex][:,0,1])

            
        #Ending: plot results and return I, Q, and config
        self.plot_results(raw_I,raw_Q,self.QubitIndex)
        return raw_I, raw_Q, self.config
        
                          
        

#   def plot_results(self, iq_list_g, iq_list_e, QubitIndex,  fig_quality=100):
#       I_g = iq_list_g[QubitIndex][0].T[0]
#       Q_g = iq_list_g[QubitIndex][0].T[1]
#       I_e = iq_list_e[QubitIndex][0].T[0]
#       Q_e = iq_list_e[QubitIndex][0].T[1]
#
#       fid, threshold, angle, ig_new, ie_new = self.hist_ssf(data=[I_g, Q_g, I_e, Q_e], cfg=self.config, plot=self.save_figs,  fig_quality=fig_quality)
#       if self.verbose: print('Optimal fidelity after rotation = %.3f' % fid)
#       if self.verbose: print('Optimal angle after rotation = %f' % angle)
#       self.logger.info('Optimal fidelity after rotation = %.3f' % fid)
#       self.logger.info('Optimal angle after rotation = %f' % angle)
#       return fid, angle
#
#   def hist_ssf(self, data=None, cfg=None, plot=True,  fig_quality = 100):
#
#       ig = data[0]
#       qg = data[1]
#       ie = data[2]
#       qe = data[3]
#
#       numbins = round(math.sqrt(float(cfg["steps"])))
#
#       xg, yg = np.median(ig), np.median(qg)
#       xe, ye = np.median(ie), np.median(qe)
#
#       if plot == True:
#           fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
#           fig.tight_layout()
#
#           axs[0].scatter(ig, qg, label='g', color='b', marker='*')
#           axs[0].scatter(ie, qe, label='e', color='r', marker='*')
#           axs[0].scatter(xg, yg, color='k', marker='o')
#           axs[0].scatter(xe, ye, color='k', marker='o')
#           axs[0].set_xlabel('I (a.u.)')
#           axs[0].set_ylabel('Q (a.u.)')
#           axs[0].legend(loc='upper right')
#           axs[0].set_title('Unrotated')
#           axs[0].axis('equal')
#       """Compute the rotation angle"""
#       theta = -np.arctan2((ye - yg), (xe - xg))
#       """Rotate the IQ data"""
#       ig_new = ig * np.cos(theta) - qg * np.sin(theta)
#       qg_new = ig * np.sin(theta) + qg * np.cos(theta)
#       ie_new = ie * np.cos(theta) - qe * np.sin(theta)
#       qe_new = ie * np.sin(theta) + qe * np.cos(theta)
#
#       """New means of each blob"""
#       xg, yg = np.median(ig_new), np.median(qg_new)
#       xe, ye = np.median(ie_new), np.median(qe_new)
#
#       # print(xg, xe)
#       #xlims = [xg - ran, xg + ran]
#       xlims = [np.min(ig_new), np.max(ie_new)]
#
#       if plot == True:
#           axs[1].scatter(ig_new, qg_new, label='g', color='b', marker='*')
#           axs[1].scatter(ie_new, qe_new, label='e', color='r', marker='*')
#           axs[1].scatter(xg, yg, color='k', marker='o')
#           axs[1].scatter(xe, ye, color='k', marker='o')
#           axs[1].set_xlabel('I (a.u.)')
#           axs[1].legend(loc='lower right')
#           axs[1].set_title(f'Rotated Theta:{round(theta, 5)}')
#           axs[1].axis('equal')
#
#           """X and Y ranges for histogram"""
#           ng, binsg, pg = axs[2].hist(ig_new, bins=numbins, range=xlims, color='b', label='g', alpha=0.5)
#           ne, binse, pe = axs[2].hist(ie_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.5)
#
#           axs[2].set_xlabel('I(a.u.)')
#       else:
#           ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
#           ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)
#
#       """Compute the fidelity using overlap of the histograms"""
#       contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
#       tind = contrast.argmax()
#       threshold = binsg[tind]
#       fid = contrast[tind]
#       #axs[2].set_title(f"Fidelity = {fid * 100:.2f}%")
#
#
#       if plot == True:
#           self.create_folder_if_not_exists(self.outerFolder)
#           outerFolder_expt = os.path.join(self.outerFolder, "ss_repeat_meas_ge")
#           self.create_folder_if_not_exists(outerFolder_expt)
#           outerFolder_expt = os.path.join(outerFolder_expt, "Q" + str(self.QubitIndex + 1))
#           self.create_folder_if_not_exists(outerFolder_expt)
#           now = datetime.datetime.now()
#           formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
#           file_name = os.path.join(outerFolder_expt,
#                                    f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
#
#           axs[2].set_title(f"Fidelity = {fid * 100:.2f}%")
#           fig.savefig(file_name,  dpi=fig_quality, bbox_inches='tight')
#           plt.close(fig)
#
#       return fid, threshold, theta, ig_new, ie_new
#
#   def only_hist_ssf(self, data=None, cfg=None, plot=True, fig_quality=100, plot_title="Run 3"):
#       import math
#       import numpy as np
#       import matplotlib.pyplot as plt
#       import os
#       import datetime
#
#       # Unpack IQ data
#       ig = data[0]
#       qg = data[1]
#       ie = data[2]
#       qe = data[3]
#
#       # Determine number of bins for the histogram
#       numbins = round(math.sqrt(float(cfg["steps"])))
#
#       # Compute medians (used for rotation angle calculation)
#       xg, yg = np.median(ig), np.median(qg)
#       xe, ye = np.median(ie), np.median(qe)
#
#       # Compute rotation angle
#       theta = -np.arctan2((ye - yg), (xe - xg))
#
#       # Rotate the IQ data
#       ig_new = ig * np.cos(theta) - qg * np.sin(theta)
#       qg_new = ig * np.sin(theta) + qg * np.cos(theta)
#       ie_new = ie * np.cos(theta) - qe * np.sin(theta)
#       qe_new = ie * np.sin(theta) + qe * np.cos(theta)
#
#       # New medians after rotation (not used further in plotting)
#       xg, yg = np.median(ig_new), np.median(qg_new)
#       xe, ye = np.median(ie_new), np.median(qe_new)
#
#       # Define histogram range from the rotated ground state to the excited state
#       xlims = [np.min(ig_new), np.max(ie_new)]
#       ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
#       ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)
#       # Compute the fidelity using the overlap of the histograms
#       contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) /
#                          (0.5 * ng.sum() + 0.5 * ne.sum())))
#       tind = contrast.argmax()
#       threshold = binsg[tind]
#       fid = contrast[tind]
#       if plot:
#           # Create figure and axis for the histogram
#           fig, ax = plt.subplots(figsize=(8, 6))
#
#           # Plot histogram for ground state and first excited state with updated labels
#           ng, binsg, _ = ax.hist(ig_new, bins=numbins, range=xlims, color='b',
#                                  label='Ground', alpha=0.5)
#           ne, binse, _ = ax.hist(ie_new, bins=numbins, range=xlims, color='r',
#                                  label='First Excited State', alpha=0.5)
#
#           # Set axis labels with 12-point font
#           ax.set_xlabel('I (a.u.)', fontsize=12)
#           ax.set_ylabel('Counts', fontsize=12)
#           # Set plot title using the provided parameter
#           ax.set_title(plot_title + f'   SSF: {int(fid * 100)}%', fontsize=12)
#           ax.legend()
#
#           # Save the figure
#           self.create_folder_if_not_exists(self.outerFolder)
#           outerFolder_expt = os.path.join(self.outerFolder, "ss_repeat_meas_ge")
#           self.create_folder_if_not_exists(outerFolder_expt)
#           outerFolder_expt = os.path.join(outerFolder_expt, "Q" + str(self.QubitIndex + 1))
#           self.create_folder_if_not_exists(outerFolder_expt)
#           now = datetime.datetime.now()
#           formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
#           file_name = os.path.join(outerFolder_expt,
#                                    f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}.png")
#           fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
#           plt.close(fig)
#
#
#
#
#
#       return fid, threshold, theta, ig_new, ie_new
#
#   def create_folder_if_not_exists(self, folder):
#       """Creates a folder at the given path if it doesn't already exist."""
#       if not os.path.exists(folder):
#           os.makedirs(folder)




