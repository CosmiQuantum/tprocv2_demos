import datetime
import numpy as np
np.set_printoptions(threshold=1000000000000000)
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import h5py
# Assuming these are defined elsewhere and importable
from build_task import *
from build_state import *
# from expt_config import *
from expt_config import * # Change for quiet vs nexus
from system_config import QICK_experiment
import copy
import os



class ParityProgram_ef(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        gen_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_chs[0],
                         mux_freqs=cfg['res_freq_ef'],
                         mux_gains=cfg['res_gain_ef'],
                         # mux_freqs=cfg['res_freq_ge'],
                         # mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ef'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=[0, 1, 2, 3, 4, 5],
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_gauss(ch=qubit_ch, name="geramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)

        self.add_pulse(ch=qubit_ch, name="ge_pi_pulse",
                       style="arb",
                       envelope="geramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )
        self.add_gauss(ch=qubit_ch, name="eframp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)

        self.add_pulse(ch=qubit_ch, name="ef_pi2_pulse_1",
                       style="arb",
                       envelope="eframp",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_ef_amp']/2,
                       )
        self.add_pulse(ch=qubit_ch, name="ef_pi2_pulse_2",
                       style="arb",
                       envelope="eframp",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'] - 90, #(90* 180 / np.pi),
                       gain=cfg['pi_ef_amp']/2,
                       )

        self.add_loop("shotloop", cfg["steps"])  # number of total shots

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="ge_pi_pulse", t=0)  # play pulse to drive qu to e
        self.delay_auto(0.0)
        
        self.pulse(ch=self.cfg["qubit_ch"], name="ef_pi2_pulse_1", t=0)  # play pi/2 pulse 
        self.delay_auto(cfg["wait_time_ef"])  # wait_time after: the idle-time
        self.pulse(ch=self.cfg["qubit_ch"], name="ef_pi2_pulse_2", t=0)  # play pi/2 pulse
        self.delay_auto(0.0)
        # self.pulse(ch=self.cfg["qubit_ch"], name="ge_pi_pulse", t=0)  # play pulse to drive qu to e
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])


class Parity_ef:
    def __init__(self, QubitIndex, num_qubits, outerFolder, round_num, save_figs=False, experiment = None):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "Parity_ef" #"Readout_Optimization"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.round_num = round_num
        self.save_figs = save_figs
        self.experiment = experiment
        self.num_qubits=num_qubits
        self.list_of_all_qubits = list_of_all_qubits

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.num_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            print(f'Q {self.QubitIndex + 1} Round {self.round_num} "Parity ef" configuration: ', self.config)

        self.q1_t1 = []
        self.q1_t1_err = []
        self.dates = []


    def runParity(self):
        # Run the single shot programs (g and e)
       

        par_ef = ParityProgram_ef(self.experiment.soccfg, reps=1, final_delay=self.config['relax_delay'],
                                    cfg=self.config)
        start_time = time.time()
        print('config', self.config)
        iq_list_f = par_ef.acquire(self.experiment.soc, soft_avgs=1, progress=False)
        timetaken = time.time() - start_time
        self.plot_results( iq_list_f, timetaken)

        # Use the fidelity calculation from SingleShot
        # fidelity, _, _, _,_ = self.hist_ssf(
        #     data=[ iq_list_f[self.QubitIndex][0].T[0], iq_list_f[self.QubitIndex][0].T[1]],
        #     cfg=self.config, plot=False)

        return iq_list_f,self.config, timetaken


    def create_folder_if_not_exists(self, folder_path):
        import os
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def plot_results(self, iq_list_f, timetaken):
        fig_quality = 100
        I = iq_list_f[self.QubitIndex][0].T[0]
        Q = iq_list_f[self.QubitIndex][0].T[1]
        ts=np.linspace(0,timetaken, len(I))





        # If we get here, the fit was successful and we can proceed with plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})

        # I subplot
        ax1.plot(ts, I, label='I', linewidth=2)
        ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.legend()

        # Q subplot
        ax2.plot(ts, Q, label='Q', linewidth=2)
        ax2.set_xlabel("time s", fontsize=20)
        ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.legend()



        # Calculate the middle of the plot area
        plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

        # Add title, centered on the plot area
        if self.config is not None:  # then its been passed to this definition, so use that
            fig.text(plot_middle, 0.98,
                     f"EF Qubit Parity Q{self.QubitIndex + 1}"  +
                     f", {self.config['steps']} avgs",
                     fontsize=24, ha='center', va='top')
        else:
            fig.text(plot_middle, 0.98,
                     f"EF Qubit Parity Q{self.QubitIndex + 1}" +
                     f", {self.config['reps']} avgs",
                     fontsize=24, ha='center', va='top')

        # Adjust spacing
        plt.tight_layout()

        # Adjust the top margin to make room for the title
        plt.subplots_adjust(top=0.93)

        ### Save figure
        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" +
                                     f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)
        return












