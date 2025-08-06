from build_task import *
from build_state import *
from expt_config import *
import matplotlib.pyplot as plt
import numpy as np
import datetime
import copy
import visdom
import logging


class StarkShift2D:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, save_figs, experiment=None, signal=None):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "stark_shift_2D"
        self.save_figs = save_figs
        self.experiment = experiment
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.number_of_qubits = number_of_qubits
        self.signal = signal

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            print(f'Q {self.QubitIndex} Stark Shift 2D configuration: ', self.config)

    def run(self, set_pos_detuning = True):

        if set_pos_detuning is True:
            self.config['detuning'] = np.abs(self.config['detuning'][self.QubitIndex])
            self.config['start_freq'] = -1 * np.abs(self.config['max_freq'])
            self.config['end_freq'] = np.abs(self.config['min_freq'])
        else:
            self.config['detuning'] = -1 * np.abs(self.config['detuning'][self.QubitIndex])
            self.config['start_freq'] = -1 * np.abs(self.config['min_freq'])
            self.config['end_freq'] = np.abs(self.config['max_freq'])

        prog = StarkShift2DProgram(self.experiment.soccfg, reps=self.config['reps'], final_delay = 0.5, cfg=self.config)
        iq_list = prog.acquire(self.experiment.soc, soft_avgs=self.exp_cfg["rounds"], progress=True) #check soft_avgs
        I = iq_list[self.QubitIndex][0,:,:,0]
        Q = iq_list[self.QubitIndex][0,:,:,1]

        qu_freq_sweep = prog.get_pulse_param('qubit_pulse', "freq", as_array=True)
        gain_sweep = prog.get_pulse_param("stark_tone", "gain", as_array=True)

        return I, Q, qu_freq_sweep, gain_sweep, self.config


    def plot(self, I, Q, qu_freq_sweep, res_gain_sweep):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        plot = axes[0]
        plot.set_box_aspect(1)
        plt.colorbar(plot.pcolormesh(qu_freq_sweep, res_gain_sweep ** 2, I, cmap="viridis"), ax=plot, shrink=0.7)
        plot.set_title("I")
        plot.set_ylabel("stark tone power [a.u.]")
        plot.set_xlabel("qubit pulse frequency [MHz]")

        plot = axes[1]
        plot.set_box_aspect(1)
        plt.colorbar(plot.pcolormesh(qu_freq_sweep, res_gain_sweep ** 2, Q, cmap='viridis'), ax=plot, shrink=0.7)
        plot.set_title("Q")
        plot.set_ylabel("stark tone power [a.u.]")
        plot.set_xlabel("qubit pulse frequency [MHz]")

        plot = axes[2]
        plot.set_box_aspect(1)
        plt.colorbar(plot.pcolormesh(qu_freq_sweep, res_gain_sweep ** 2, np.sqrt(np.square(I) + np.square(Q)), cmap='viridis'), ax=plot,
                     shrink=0.7)
        plot.set_title("magnitude")
        plot.set_ylabel("stark tone power [a.u.]")
        plot.set_xlabel("qubit pulse frequency [MHz]")

        plt.show()

        if self.save_figs:
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(self.outerFolder, f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex}.png")
            fig.savefig(file_name, dpi=100, bbox_inches='tight')
            plt.close(fig)

class StarkShift2DProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        stark_ch = cfg['qubit_ampl_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ge'],  # res of interest frequency at QubitIndex and 7
                         mux_gains=cfg['res_gain_ge'],  # readout gain, stark gain
                         mux_phases=cfg['res_phase'],  # res of interest phase repeated at QubitIndex and 7
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        # self.add_pulse(ch=res_ch, name="stark_tone",
        #                style="const",
        #                length=cfg['stark_length'],
        #                mask=cfg['stark_mask'], #only play stark tone
        #                )

        self.add_pulse(ch=res_ch, name="readout_pulse",
                       style="const",
                       length=cfg['res_length'],
                       mask=cfg['list_of_all_qubits'], #only play readout tone
                       )

        self.declare_gen(ch=stark_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_pulse(ch=stark_ch, name="stark_tone", ro_ch=ro_ch[0],
                       style="const",
                       length=cfg['stark_length'],
                       freq=cfg['qubit_freq_ge'] + cfg['detuning'],
                       phase=0,
                       gain=QickSweep1D("gain_loop",cfg["start_gain"], cfg["end_gain"]),
                       )


        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        # self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        # self.add_pulse(ch=qubit_ch, name="qubit_pulse",
        #                style="arb",
        #                envelope="ramp",
        #                freq=QickSweep1D("qubit_pulse_loop", cfg["qubit_freq_ge"] + cfg["start_freq"], cfg["qubit_freq_ge"] + cfg["end_freq"]),
        #                phase=cfg['qubit_phase'],
        #                gain=cfg['pi_amp'],
        #                )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch[0],
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=QickSweep1D("qubit_pulse_loop", cfg["qubit_freq_ge"] + cfg["start_freq"], cfg["qubit_freq_ge"] + cfg["end_freq"]),
                       phase=0,
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_loop("gain_loop", cfg["gain_steps"])
        self.add_loop("qubit_pulse_loop", cfg["qubit_pulse_steps"]) #inner loop

    def _body(self, cfg):
        self.pulse(ch=self.cfg['qubit_ampl_ch'], name="stark_tone", t=0)  # play stark tone
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse", t=cfg['qubit_pulse_delay']) #play qubit pulse with delay
        self.delay(t=cfg['stark_length'] + cfg['readout_pulse_delay']) #wait for stark tone to finish and for resonator to reach vacuum
        self.pulse(ch=cfg['res_ch'], name="readout_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class ResStarkShift2D:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, res_freq_stark, res_phase_stark, save_figs, experiment=None, signal=None, unmasking_resgain=False):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "res_stark_shift_2D"
        self.save_figs = save_figs
        self.experiment = experiment
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.number_of_qubits = number_of_qubits
        self.signal = signal

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            print(f'Q {self.QubitIndex} Stark Shift 2D configuration: ', self.config)
            self.config['res_freq_stark'] = res_freq_stark
            self.config['res_phase_stark'] = res_phase_stark
            stark_mask = np.arange(0, self.number_of_qubits + 1)
            stark_mask = np.delete(stark_mask,QubitIndex)
            if unmasking_resgain:
                self.config["stark_mask"] = [QubitIndex, 6]
            else:
                self.config['stark_mask'] = stark_mask

    def run(self):
        I = []
        Q = []
        res_gain_ge = copy.deepcopy(self.config['res_gain_ge'])
        gain_sweep = np.linspace(self.config['start_gain'], self.config['end_gain'], self.config['gain_steps'])
        for g in gain_sweep:
            gain = round(g, 3)
            self.config['stark_gain'] = np.concatenate((res_gain_ge, [gain]))  #readout pulse gain, stark tone gain
            prog = ResStarkShift2DProgram(self.experiment.soccfg, reps=self.config['reps'], final_delay = 0.5, cfg=self.config)
            iq_list = prog.acquire(self.experiment.soc, soft_avgs=self.exp_cfg["rounds"], progress=True) #check soft_avgs
            I.append(iq_list[self.QubitIndex][0,:,0])
            Q.append(iq_list[self.QubitIndex][0,:,1])

        qu_freq_sweep = prog.get_pulse_param('qubit_pulse', "freq", as_array=True)

        return I, Q, qu_freq_sweep, gain_sweep, self.config

    def plot(self, I, Q, qu_freq_sweep, res_gain_sweep):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        plot = axes[0]
        plot.set_box_aspect(1)
        plt.colorbar(plot.pcolormesh(qu_freq_sweep, res_gain_sweep ** 2, I, cmap="viridis"), ax=plot, shrink=0.7)
        plot.set_title("I [a.u.]")
        plot.set_ylabel("stark tone power [a.u.]")
        plot.set_xlabel("qubit pulse frequency [MHz]")

        plot = axes[1]
        plot.set_box_aspect(1)
        plt.colorbar(plot.pcolormesh(qu_freq_sweep, res_gain_sweep ** 2, Q, cmap='viridis'), ax=plot, shrink=0.7)
        plot.set_title("Q [a.u.]")
        plot.set_ylabel("stark tone power [a.u.]")
        plot.set_xlabel("qubit pulse frequency [MHz]")

        plot = axes[2]
        plot.set_box_aspect(1)
        plt.colorbar(plot.pcolormesh(qu_freq_sweep, res_gain_sweep ** 2, np.sqrt(np.square(I) + np.square(Q)), cmap='viridis'), ax=plot,
                     shrink=0.7)
        plot.set_title("magnitude")
        plot.set_ylabel("stark tone power [a.u.]")
        plot.set_xlabel("qubit pulse frequency [MHz]")

        plt.show()

        if self.save_figs:
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(self.outerFolder, f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex}.png")
            fig.savefig(file_name, dpi=100, bbox_inches='tight')
            plt.close(fig)

# 2D scan over resonator gain, qubit pulse frequency
class ResStarkShift2DProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        print(cfg['stark_gain'], cfg['stark_mask'])
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_stark'], # res of interest frequency at QubitIndex and 7
                         mux_gains=cfg['stark_gain'], # readout gain, stark gain
                         mux_phases=cfg['res_phase_stark'], # res of interest phase repeated at QubitIndex and 7
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="stark_tone",
                       style="const",
                       length=cfg['stark_length'],
                       mask=cfg['stark_mask'], #only play stark tone
                       )

        self.add_pulse(ch=res_ch, name="readout_pulse",
                       style="const",
                       length=cfg['res_length'],
                       mask=cfg['list_of_all_qubits'], #only play readout tone
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        # self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        # self.add_pulse(ch=qubit_ch, name="qubit_pulse",
        #                style="arb",
        #                envelope="ramp",
        #                freq=QickSweep1D("qubit_pulse_loop", cfg['qubit_freq_ge'] + cfg["start_freq"], cfg['qubit_freq_ge'] + cfg["end_freq"]),
        #                phase=cfg['qubit_phase'],
        #                gain=cfg['pi_amp'],
        #                )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch[0],
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=QickSweep1D("qubit_pulse_loop", cfg['qubit_freq_ge'] + cfg["start_freq"], cfg['qubit_freq_ge'] + cfg["end_freq"]),
                       phase=0,
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_loop("qubit_pulse_loop", cfg["qubit_pulse_steps"]) #inner loop

    def _body(self, cfg):
        self.pulse(ch=self.cfg['res_ch'], name="stark_tone", t=0)  # play stark tone
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse", t=cfg['qubit_pulse_delay']) #play qubit pulse with delay
        self.delay(t=cfg['stark_length'] + cfg['readout_pulse_delay']) #wait for stark tone to finish and for resonator to reach vacuum
        self.pulse(ch=cfg['res_ch'], name="readout_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

# TLS SPECTROSCOPY ############################################################################

class StarkShiftSpec:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, save_figs, experiment=None):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "stark_shift_spec"
        self.save_figs = save_figs
        self.experiment = experiment
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.number_of_qubits = number_of_qubits

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            print(f'Q {self.QubitIndex} Stark Shift Spec configuration: ', self.config)

    def get_gain_sweep(self):
        target_freq = np.linspace(0, self.config['max_shift'], num=self.config['steps'])
        alpha = self.config["anharmonicity"][self.qubitIndex]
        wq = self.config['qubit_freq_ge']
        detuning = self.config['detuning']
        c = self.config['duffing_constant']
        gain_sweep_pos = np.sqrt((2 * target_freq * (wq - detuning) * (alpha + wq - detuning)) / (alpha)) / c
        gain_sweep_neg = np.sqrt((2 * target_freq * (wq + detuning) * (alpha + wq + detuning)) / (alpha)) / c
        gain_sweep = np.array([gain_sweep_neg[::-1], gain_sweep_pos]).reshape([2 * self.config['steps'], 1])
        return gain_sweep

    def run_with_python_loop(self):
        I = []
        Q = []
        shots = []
        P = []

        gain_sweep = self.get_gain_sweep()

        count = 0
        for g in gain_sweep:
            if count == 150:
                self.config['detuning'] = self.config['detuning']*-1 #switch to positive detuning

            self.config['stark_gain'] = np.round(g,3)
            print(g)
            prog = StarkShiftSpectroscopyProgram(self.experiment.soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'],
                                                 cfg=self.config)

            iq_list = prog.acquire(self.experiment.soc, soft_avgs=self.config["rounds"],
                                threshold=self.experiment.readout_cfg["threshold"],
                                angle=self.experiment.readout_cfg["ro_phase"],
                                progress=True)
            raw_0 = prog.get_raw()  # I,Q data without normalizing to readout window, subtracting readout offset, or rotation/thresholding
            shots_0 = prog.get_shots()  # state assignment from built in thresholding
            I.append(raw_0[self.QubitIndex][:,0,0])
            Q.append(raw_0[self.QubitIndex][:,0,1])
            shots.append(shots_0[self.QubitIndex][:,0])
            P.append(iq_list[self.QubitIndex][:,0])
            count +=1

    def run_with_qick_sweep(self):
        self.config['detuning'] = self.config['detuning'][self.QubitIndex]

        # run with negative detuning
        #self.config['stark_gain'] = QickSweep1D("gain_loop", -1 *self.config['end_gain'], self.config['start_gain'])
        self.config['stark_gain'] = QickSweep1D("gain_loop", self.config['start_gain'], self.config['end_gain'])
        prog_neg = StarkShiftSpectroscopyProgram(self.experiment.soccfg, reps=self.config['reps'],
                                             final_delay=self.config['relax_delay'],
                                             cfg=self.config)
        iq_list = prog_neg.acquire(self.experiment.soc, soft_avgs=self.config["rounds"],
                                 threshold=self.experiment.readout_cfg["threshold"],
                                 angle=self.experiment.readout_cfg["ro_phase"],
                                 progress=True)
        raw_0 = prog_neg.get_raw()  # I,Q data without normalizing to readout window, subtracting readout offset, or rotation/thresholding
        shots_0 = prog_neg.get_shots()  # state assignment from built in thresholding

        #flip all data for continuous frequency sweep format
        I_neg = np.flip(np.transpose(raw_0[self.QubitIndex][:,:,0,0]), axis=0)
        Q_neg = np.flip(np.transpose(raw_0[self.QubitIndex][:,:,0,1]), axis=0)
        shots_neg = np.flip(np.transpose(shots_0[self.QubitIndex][:,:,0]), axis=0)
        P_neg = np.flip(np.transpose(iq_list[self.QubitIndex][:,:,0]),axis=0)
        gain_sweep_neg = np.flip(prog_neg.get_pulse_param("stark_tone", "gain", as_array=True),axis=0)

        # run with positive detuning
        self.config['detuning'] = self.config['detuning'] * -1
        self.config['stark_gain'] = QickSweep1D("gain_loop", self.config["start_gain"], self.config["end_gain"])

        prog_pos = StarkShiftSpectroscopyProgram(self.experiment.soccfg, reps=self.config['reps'],
                                             final_delay=self.config['relax_delay'],
                                             cfg=self.config)

        iq_list = prog_pos.acquire(self.experiment.soc, soft_avgs=self.config["rounds"],
                               threshold=self.experiment.readout_cfg["threshold"],
                               angle=self.experiment.readout_cfg["ro_phase"],
                               progress=True)
        raw_0 = prog_pos.get_raw()  # I,Q data without normalizing to readout window, subtracting readout offset, or rotation/thresholding
        shots_0 = prog_pos.get_shots()  # state assignment from built in thresholding

        I_pos = np.transpose(raw_0[self.QubitIndex][:, :, 0, 0])
        Q_pos = np.transpose(raw_0[self.QubitIndex][:, :, 0, 1])
        shots_pos = np.transpose(shots_0[self.QubitIndex][:, :, 0])
        P_pos = np.transpose(iq_list[self.QubitIndex][:, :, 0])
        gain_sweep_pos = prog_pos.get_pulse_param("stark_tone", "gain", as_array=True)

        I = np.concatenate((I_neg, I_pos), axis=0)
        Q = np.concatenate((Q_neg, Q_pos), axis=0)
        shots = np.concatenate((shots_neg, shots_pos), axis=0)
        P = np.concatenate((P_neg, P_pos))
        gain_sweep = np.concatenate((gain_sweep_neg, gain_sweep_pos))

        return I, Q, P,  shots, gain_sweep, self.config

    def gain_to_freq(self, gain_sweep):
        anharmonicity = self.config['anharmonicity']
        alpha = anharmonicity[self.QubitIndex]
        constants = self.config['duffing_constant']
        const = constants[self.QubitIndex]
        wq = self.config['qubit_freq_ge']
        ws = np.abs(self.config['detuning'])

        delta_wq_neg = const * (alpha * gain_sweep[0:self.config["gain_steps"]] ** 2) / (2 * (wq + ws) * (alpha + wq + ws))
        delta_wq_pos = const * (alpha * gain_sweep[self.config["gain_steps"]:] ** 2) / (2 * (wq - ws) * (alpha + wq - ws))
        delta_wq = np.concatenate((-1*delta_wq_neg, delta_wq_pos))
        return delta_wq

    def plot(self, P, gain_sweep):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].plot(gain_sweep, P)
        axes[0].set_xlabel("stark tone gain [a.u.]")
        axes[0].set_ylabel("P(MS=1)")
        axes[0].set_title(f"Qubit {self.QubitIndex}, stark length={self.config['stark_length']} us")
        axes[0].set_ylim([0.0, 1.0])

        freq_sweep = self.gain_to_freq(gain_sweep)
        axes[1].plot(freq_sweep, P)
        axes[1].set_xlabel("AC Stark shift [MHz]")
        axes[1].set_ylabel("P(MS=1)")
        axes[1].set_title(f"Qubit {self.QubitIndex}, stark length={self.config['stark_length']} us")
        axes[1].set_ylim([0.0, 1.0])

        plt.show()

        if self.save_figs:
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(self.outerFolder, f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex}.png")
            fig.savefig(file_name, dpi=100, bbox_inches='tight')
            plt.close(fig)

    def plot_shots(self, I, Q, shots, gain_sweep, gain_index=0):
        fig, axes = plt.subplots(1, 1, figsize=(3, 3))
        plot = axes
        plot.set_box_aspect(1)
        idx = np.where(shots[gain_index] == 1)[0]
        plot.scatter(I[gain_index][idx], Q[gain_index][idx], c='r', label='e')
        idx = np.where(shots[gain_index] == 0)[0]
        plot.scatter(I[gain_index][idx], Q[gain_index][idx], c='b', label='g')
        plot.set_xlabel("I")
        plot.set_ylabel("Q")
        plot.set_title(f"I,Q shots with built-in threshold at gain = {gain_sweep[gain_index]}")
        plot.legend()

        plt.show()

class StarkShiftSpectroscopyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        stark_ch = cfg['qubit_ampl_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="readout_pulse",
                               style="const",
                               length=cfg['res_length'],
                               mask=cfg["list_of_all_qubits"],
                               )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                               style="arb",
                               envelope="ramp",
                               freq=cfg['qubit_freq_ge'],
                               phase=cfg['qubit_phase'],
                               gain=cfg['pi_amp'],
                               )


        self.add_loop("gain_loop", cfg["gain_steps"])
        self.declare_gen(ch=stark_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_gauss(ch=stark_ch, name="stark_ramp", sigma=cfg['stark_sigma'], length = cfg['stark_sigma'] *2)
        self.add_pulse(ch=stark_ch, name="stark_tone",
                       style="flat_top",
                       envelope="stark_ramp",
                       freq=cfg['qubit_freq_ge'] + cfg['detuning'],
                       phase=cfg['qubit_phase'],
                       gain = cfg['stark_gain'],
                       length=cfg['stark_length'],
                       )

    def _body(self, cfg):
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse", t=0)  # play qubit pi pulse
        self.delay_auto(t=0.01, tag='wait pi')  # wait for qubit pi pulse to finish
        self.pulse(ch=self.cfg['qubit_ampl_ch'], name="stark_tone", t=0)  # play stark tone
        self.delay_auto(t=0.01, tag='wait stark')  # wait for stark tone to finish
        self.delay(t=cfg['readout_pulse_delay']) #wait for resonator to return to vacuum
        self.pulse(ch=cfg['res_ch'], name="readout_pulse", t=0)  # play readout pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])  # get readout

class ResStarkShiftSpec:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, res_freq_stark, res_phase_stark, save_figs, experiment=None):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "stark_shift_spec"
        self.save_figs = save_figs
        self.experiment = experiment
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.number_of_qubits = number_of_qubits

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            print(f'Q {self.QubitIndex} Stark Shift Spec configuration: ', self.config)
            self.config['res_freq_stark'] = res_freq_stark
            self.config['res_phase_stark'] = res_phase_stark
            stark_mask = np.arange(0, self.number_of_qubits + 1)
            stark_mask = np.delete(stark_mask,QubitIndex)
            self.config['stark_mask'] = stark_mask

    def run(self):
        I = []
        Q = []
        P = []
        shots = []

        res_gain_ge = np.array(copy.deepcopy(self.config['res_gain_ge']))
        res_gain_sweep = np.sqrt(np.linspace(self.config['start_gain'], self.config['end_gain'], self.config['gain_steps']))
        for g in res_gain_sweep:
            gain = np.array([round(g, 3)])
            self.config['res_gain_stark'] = np.concatenate((res_gain_ge, gain))
            prog = ResStarkShiftSpectroscopyProgram(self.experiment.soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'],
                                                 cfg=self.config)
            iq_list = prog.acquire(self.experiment.soc, soft_avgs=self.config["rounds"],
                                   threshold=self.experiment.readout_cfg["threshold"],
                                   angle=self.experiment.readout_cfg["ro_phase"],
                                   progress=False)

            raw_0 = prog.get_raw() # I,Q data without normalizing to readout window, subtracting readout offset, or rotation/thresholding
            shots_0 = prog.get_shots() # state assignment from built in thresholding
            I.append(raw_0[self.QubitIndex][:,0][:,0])
            Q.append(raw_0[self.QubitIndex][:,0][:,1])
            shots.append(shots_0[self.QubitIndex][:,0])
            P.append(iq_list[self.QubitIndex][:,0])

        return I, Q, P,  shots, res_gain_sweep, self.config

    def plot(self, P, res_gain_sweep):
        fig, axes = plt.subplots(1, 2, figsize=(3, 6))
        axes[0].plot(res_gain_sweep, P)
        axes[0].set_xlabel("resonator gain [a.u.]")
        axes[0].set_ylabel("P(MS=1)")
        axes[0].set_title(f"Qubit {self.QubitIndex}, stark length={self.config['stark_length']} us")
        axes[0].set_ylim([0.0, 1.0])

        axes[1].plot(res_gain_sweep ** 2 * -25, P)
        axes[1].set_xlabel("AC Stark Shift [MHz]")
        axes[1].set_ylabel("P(MS=1)")
        axes[1].set_title(f"Qubit {self.QubitIndex}, stark length={self.config['stark_length']} us")
        axes[1].set_ylim([0.0, 1.0])

        plt.show()

        if self.save_figs:
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(self.outerFolder,
                                     f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex}.png")
            fig.savefig(file_name, dpi=100, bbox_inches='tight')
            plt.close(fig)

    def plot_shots(self, I, Q, shots, res_gain_sweep, gain_index=0):
        fig, axes = plt.subplots(1, 1, figsize=(3, 3))
        plot = axes
        plot.set_box_aspect(1)
        idx = np.where(shots[gain_index] == 1)[0]
        plot.scatter(I[gain_index][idx], Q[gain_index][idx], c='r', label='e')
        idx = np.where(shots[gain_index] == 0)[0]
        plot.scatter(I[gain_index][idx], Q[gain_index][idx], c='b', label='g')
        plot.set_xlabel("I")
        plot.set_ylabel("Q")
        plot.set_title(f"I,Q shots with built-in threshold at gain = {res_gain_sweep[gain_index]}")
        plot.legend()

        plt.show()

class ResStarkShiftSpectroscopyProgram(AveragerProgramV2):
    def _initialize(self, cfg):

        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_stark'],
                         mux_gains=cfg['res_gain_stark'],
                         mux_phases=cfg['res_phase_stark'],
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="stark_tone",
                               style="const",
                               length=cfg['stark_length'],
                               mask=cfg["stark_mask"],
                               )

        self.add_pulse(ch=res_ch, name="readout_pulse",
                               style="const",
                               length=cfg['res_length'],
                               mask=cfg["list_of_all_qubits"],
                               )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                               style="arb",
                               envelope="ramp",
                               freq=cfg['qubit_freq_ge'],
                               phase=cfg['qubit_phase'],
                               gain=cfg['pi_amp'],
                               )

    def _body(self, cfg):
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse", t=0)  # play qubit pi pulse
        self.delay_auto(t=0.0, tag='wait pi')  # wait for qubit pi pulse to finish
        self.pulse(ch=self.cfg['res_ch'], name="stark_tone", t=0)  # play stark tone
        self.delay_auto(t=0.0, tag='wait stark')  # wait for stark tone to finish
        self.delay(t=cfg['readout_pulse_delay']) #wait for resonator to return to vacuum
        self.pulse(ch=cfg['res_ch'], name="readout_pulse", t=0)  # play readout pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])  # get readout
