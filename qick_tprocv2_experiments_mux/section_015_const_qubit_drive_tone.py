from build_task import *
from build_state import *
from expt_config import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import datetime
import time
import copy
import logging

class QubitToneSpectrumAnalyzer:
    def __init__(self, QubitIndex, number_of_qubits,  outerFolder,  round_num, signal, save_figs, experiment = None,
                 live_plot = None, verbose = False, logger = None, qick_verbose=True, increase_reps = False, increase_rounds = False,
                 increase_reps_to = 500, increase_rounds_to = 2, plot_fit=True, fit_data=True, unmasking_resgain = False,
                 qubit_pulse_mode="periodic", qubit_sa_hold_time=100.0):

        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.plot_fit=plot_fit
        self.fit_data = fit_data
        self.increase_rounds = increase_rounds
        self.increase_rounds_to = increase_rounds_to
        self.expt_name = "qubit_spec_ge"
        self.signal = signal
        self.save_figs = save_figs
        self.experiment = experiment
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.number_of_qubits = number_of_qubits
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")
        self.increase_reps = increase_reps
        self.increase_reps_to = increase_reps_to

        self.qubit_pulse_mode = qubit_pulse_mode
        self.qubit_sa_hold_time = qubit_sa_hold_time

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.live_plot = live_plot
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} configuration: ', self.config)
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} configuration: {self.config}')

    def run(self):
        qubit_pulse_mode = self.qubit_pulse_mode
        qubit_sa_hold_time = self.qubit_sa_hold_time

        print("\nRunning qubit-drive spectrum analyzer tone:")
        print(f"  Qubit: Q{self.QubitIndex + 1}")
        print(f"  qubit_ch: {self.config['qubit_ch']}")
        print(f"  qubit_freq_ge: {self.config['qubit_freq_ge']} MHz")
        print(f"  qubit_gain_ge: {self.config['qubit_gain_ge']}")
        print(f"  qubit_length_ge: {self.config['qubit_length_ge']} us")
        print(f"  reps: {self.config['reps']}")
        print(f"  rounds: {self.config['rounds']}")
        print(f"  qubit_pulse_mode: {self.qubit_pulse_mode}")
        print(f"  qubit_sa_hold_time: {self.qubit_sa_hold_time} us")

        class QubitToneSpectrumAnalyzerProgram(AveragerProgramV2):
            def _initialize(self, cfg):
                ro_ch = cfg['ro_ch']
                qubit_ch = cfg['qubit_ch']

                self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
                self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch[0],
                               style="const",
                               length=cfg['qubit_length_ge'],
                               freq=cfg['qubit_freq_ge'],
                               phase=0,
                               gain=cfg['qubit_gain_ge'],
                               mode=qubit_pulse_mode)

            def _body(self, cfg):
                self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)
                self.delay(qubit_sa_hold_time)

        prog = QubitToneSpectrumAnalyzerProgram(
            self.experiment.soccfg,
            reps=self.config['reps'],
            final_delay=0.0,
            cfg=self.config)

        prog.acquire(
            self.experiment.soc,
            soft_avgs=1,
            progress=self.qick_verbose)