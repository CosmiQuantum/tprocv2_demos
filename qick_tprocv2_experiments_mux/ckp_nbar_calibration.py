from scipy.optimize import curve_fit
from build_task import *
from build_state import *
from expt_config import *
from system_config import *
import matplotlib.pyplot as plt
import numpy as np
import logging

class CKPProgram_g(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ckp'],  # res of interest frequency at QubitIndex and 7
                         mux_gains=cfg['ckp_gain'],  # readout gain, stark gain
                         mux_phases=cfg['res_phase_stark'],  # res of interest phase repeated at QubitIndex and 7
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="stark_tone",
                       style="const",
                       length=cfg['ckp_length'],
                       mask=cfg['ckp_mask'],  # only play stark tone
                       )

        self.add_pulse(ch=res_ch, name="readout_pulse",
                       style="const",
                       length=cfg['res_length'],
                       mask=cfg['list_of_all_qubits'],  # only play readout tone
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch[0],
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=QickSweep1D("qubit_pulse_loop", cfg['qubit_freq_ge'] + cfg["start_freq"],
                                        cfg['qubit_freq_ge'] + cfg["end_freq"]),
                       phase=0,
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_loop("qubit_pulse_loop", cfg["qubit_pulse_steps"])  # inner loop

    def _body(self, cfg):
        self.pulse(ch=self.cfg['res_ch'], name="stark_tone", t=0)  # play stark tone
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse", t=cfg['qubit_pulse_delay'])  # play qubit pulse with delay
        self.delay(t=cfg['ckp_length'] + cfg[
            'readout_pulse_delay'])  # wait for stark tone to finish and for resonator to reach vacuum
        self.pulse(ch=cfg['res_ch'], name="readout_pulse", t=cfg['qubit_pulse_delay']) #wait for resonator to reach vacc again
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class CKPProgram_e(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ckp'],  # res of interest frequency at QubitIndex and 7
                         mux_gains=cfg['ckp_gain'],  # readout gain, stark gain
                         mux_phases=cfg['res_phase_ckp'],  # res of interest phase repeated at QubitIndex and 7
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="stark_tone",
                       style="const",
                       length=cfg['ckp_length'],
                       mask=cfg['ckp_mask'],  # only play stark tone
                       )

        self.add_pulse(ch=res_ch, name="readout_pulse",
                       style="const",
                       length=cfg['res_length'],
                       mask=cfg['list_of_all_qubits'],  # only play readout tone
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch[0],
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=QickSweep1D("qubit_pulse_loop", cfg['qubit_freq_ge'] + cfg["start_freq"],
                                        cfg['qubit_freq_ge'] + cfg["end_freq"]),
                       phase=0,
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_pulse(ch=qubit_ch, name="pi_pulse",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        self.add_loop("qubit_pulse_loop", cfg["qubit_pulse_steps"])  # inner loop

    def _body(self, cfg):
        self.pulse(ch=cfg['qubit_ch'], name="pi_pulse") # put qubit in e
        self.delay_auto()
        self.pulse(ch=self.cfg['res_ch'], name="stark_tone", t=0)  # play stark tone
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse", t=cfg['qubit_pulse_delay'])  # play qubit pulse with delay
        self.delay(t=cfg['ckp_length'] + cfg[
            'readout_pulse_delay'])  # wait for stark tone to finish and for resonator to reach vacuum
        self.pulse(ch=cfg['res_ch'], name="readout_pulse",
                   t=cfg['qubit_pulse_delay'])  # wait for resonator to reach vacc again
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class CKPMeasurement:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, signal, save_figs,res_freq_ckp, res_phase_ckp, experiment = None,
                 fit_data = None, verbose = False, logger = None, qick_verbose=True):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.fit_data = fit_data
        self.expt_name = "ckp_nbar_calibration"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.signal = signal
        self.number_of_qubits = number_of_qubits
        self.save_figs = save_figs
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            self.config['res_freq_ckp'] = res_freq_ckp
            self.config['res_phase_ckp'] = res_phase_ckp
            stark_mask = np.arange(0, self.number_of_qubits + 1)
            stark_mask = np.delete(stark_mask, QubitIndex)
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} CKP configuration: ', self.config)
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} CKP configuration:{self.config}')

    def run(self):
        now = datetime.datetime.now()

        gain_sweep = np.linspace(self.config["start_gain"], self.config["end_gain"],num=self.config["gain_steps"])
        res_freq_sweep = np.linspace(self.config["res_freq_start"], self.config["res_freq_stop"], num=self.config["res_freq_steps"])

        I_g = []
        Q_g = []
        I_e = []
        Q_e = []

        for g in gain_sweep:
            for f in res_freq_sweep:
                self.config['ckp_gain'] = np.round(g,3)
                self.config['res_freq_ckp'][-1] = np.round(f, 6) #last channel pulse
                ckp_g = CKPProgram_g(self.experiment.soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'],
                             cfg=self.config)
                ckp_e = CKPProgram_e(self.experiment.soccfg, reps=self.config['reps'],
                                     final_delay=self.config['relax_delay'],
                                     cfg=self.config)

                iq_list_g = ckp_g.acquire(self.experiment.soc, soft_avgs=self.config['rounds'], progress=self.qick_verbose)
                i0_g = iq_list_g[self.QubitIndex][0, :, 0]
                q0_g = iq_list_g[self.QubitIndex][0, :, 1]
                I_g.append(i0_g)
                Q_g.append(q0_g)

                iq_list_e = ckp_e.acquire(self.experiment.soc, soft_avgs=self.config['rounds'], progress=self.qick_verbose)
                i0_e = iq_list_e[self.QubitIndex][0, :, 0]
                q0_e = iq_list_e[self.QubitIndex][0, :, 1]
                I_e.append(i0_e)
                Q_e.append(q0_e)

                qu_freq_sweep = ckp_g.get_pulse_param("qubit_pulse","freq", as_array=True)

        return I_g, Q_g, I_e, Q_e, qu_freq_sweep, gain_sweep, res_freq_sweep, self.config

    def set_res_gain_ge(self, QUBIT_INDEX, num_qubits=6):
        """Sets the gain for the selected qubit to 1, others to 0."""
        res_gain_ge = [0] * num_qubits  # Initialize all gains to 0
        if 0 <= QUBIT_INDEX < num_qubits:  # makes sure you are within the range of options
            res_gain_ge[QUBIT_INDEX] = 1  # Set the gain for the selected qubit
        return res_gain_ge

    def exponential(self, x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)