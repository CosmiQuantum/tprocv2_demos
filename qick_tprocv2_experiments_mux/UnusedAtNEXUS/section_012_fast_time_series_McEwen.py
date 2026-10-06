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
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        
        ##Declare generators: these are the generators needed for readout (resonator). Then do a loop over these for the various readouts needed
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        #Now we add a qubit pulse, declare its channel generator
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch[0],
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        #         self.add_loop("shotloop", cfg["steps"]) # number of total shots
        #self.add_loop("shotloop", cfg["steps"])  # Pulse / no Pulse loop

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

        #self.add_loop("shotloop", cfg["steps"])

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
                 target_tls=True, tls_gain= None, tls_detuning=0, verbose = False, logger = None, qick_verbose=True):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "FastRelEx"
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
        #raw_I = []
        #raw_Q = []
        
        #Use the conditional "target_tls" within this object to determine whether we run our N shots. If we target the TLS, 
        #we don't throw in a pi pulse before measuring.
        if self.target_tls == True:
            fastExProg = FastTLSExcitationProgram(self.experiment.soccfg,
                                                  reps=self.config['reps'],
                                                  final_delay=self.config['relax_delay'],
                                                  cfg=self.config)
            iq_list_avg = fastExProg.acquire(self.experiment.soc,
                                             soft_avgs=1,
                                             progress=True,
                                             threshold=self.experiment.readout_cfg["threshold"],
                                             angle=self.experiment.readout_cfg["ro_phase"],
                                             )

            #REL Waypoint: check dimensionality of data here
            raw_IQ = fastExProg.get_raw()
            raw_shots = fastExProg.get_shots()
            raw_I = raw_IQ[self.QubitIndex][:,:,0,0]
            raw_Q = raw_IQ[self.QubitIndex][:,:,0,1]
            states = raw_shots[self.QubitIndex][:,:,0]
            #P = iq_list_avg[self.QubitIndex][0][:,0]

            time_axis = fastExProg.get_time_axis(self.QubitIndex)
        
        else:
            #If we're not targeting a tls (for example, if we can't find one), then just do a fast relaxation only program, which does
            #include a pi pulse
            
            fastRelProg = FastRelaxationProgram(self.experiment.soccfg,
                                                reps=self.config['reps'],
                                                final_delay=self.config['relax_delay'],
                                                cfg=self.config)
            iq_list_avg = fastRelProg.acquire(self.experiment.soc,
                                          soft_avgs=1,
                                          progress=True,
                                          threshold=self.experiment.readout_cfg["threshold"],
                                          angle=self.experiment.readout_cfg["ro_phase"],
                                          )

            #REL Waypoint: check dimensionality of data here
            raw_IQ = fastRelProg.get_raw()
            raw_shots = fastRelProg.get_shots()
            raw_I = raw_IQ[self.QubitIndex][:,0,0]
            raw_Q = raw_IQ[self.QubitIndex][:,0,1]
            states = raw_shots[self.QubitIndex][:, 0]
            #P = iq_list_avg[self.QubitIndex][0][:,0]

            time_axis = fastRelProg.get_time_axis(self.QubitIndex)


        return raw_I, raw_Q, states, time_axis, self.config
        
                          
    def plot(self, raw_I, raw_Q, P, time_axis, save_figs=False):
        time_axis = np.arange(0,len(P)) #placeholder, need to understand get_time_axis output

        fig, axes = plt.subplots(3,1)

        if self.target_tls:
            fig.suptitle("TLS found, Fast Excitation Measurement at TLS frequency")
        else:
            fig.suptitle("Fast Relaxation Measurement at qubit frequency")

        x = np.arange(0,len(P))
        ax = axes[0]
        mag = np.transpose(np.sqrt(np.square(raw_I) + np.square(raw_Q)))
        ax.plot(time_axis, mag,'-bo')
        ax.set_ylabel('I,Q magnitude [a.u.]')
        ax.set_xlabel('time [us]')

        ax = axes[1]
        ax.plot(time_axis, P,'-bo')
        ax.set_ylabel('state')
        ax.set_xlabel('time [us]')

        #throw out measurements where preceding state is 1, post-selection
        P_post_process = []
        time_post_process = []
        for round in np.arange(1,len(P)):
            if P[round - 1] == 0:
                P_post_process.append(P[round])
                time_post_process.append(time_axis[round])

        ax = axes[2]
        ax.plot(time_post_process, P_post_process,'-bo')
        ax.set_ylabel('post-selection')
        ax.set_xlabel('time [us]')

        if save_figs:
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(self.outerFolder,
                                         f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex}.png")
            fig.savefig(file_name, dpi=100, bbox_inches='tight')
            plt.close(fig)




