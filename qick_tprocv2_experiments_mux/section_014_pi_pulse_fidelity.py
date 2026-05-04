from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from build_task import *
from build_state import *
from expt_config import *
import copy
import logging

class RepeatedPiPulseProgram(AveragerProgramV2):
    """
    Apply N repeated ge pi pulses, then read out.
    Used for quick qubit-drive pulse quality / repeated-pulse contrast check.

    This experiment answers the question:
    If I apply my calibrated pi pulse over and over, does it behave like a perfect pi rotation each time?

    Expected Result:
    N = 0 → ground-like
    N = 1 → excited-like
    N = 2 → ground-like
    N = 3 → excited-like
    
    We can later plot the driven excited-state population (Pe) vs N, where we expect N = 0, 2, 4, 6,... to have low driven-Pe 
    and N = 1, 3, 5, 7,... to have high driven-Pe. 
    Do NOT get this driven-Pe confused with thermal Pe, these are different!
    If the pi-pulse is good, we expect a clean alternating high driven-Pe and low driven-Pe pattern.
    """

    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        gen_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # Readout generator
        self.declare_gen(
            ch=gen_ch,
            nqz=cfg['nqz_res'],
            ro_ch=ro_chs[0],
            mux_freqs=cfg['res_freq_ge'],
            mux_gains=cfg['res_gain_ge'],
            mux_phases=cfg['res_phase'],
            mixer_freq=cfg['mixer_freq']
        )

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(
                ch=ch,
                length=cfg['res_length'],
                freq=f,
                phase=ph,
                gen_ch=gen_ch
            )

        self.add_pulse(
            ch=gen_ch,
            name="res_pulse",
            style="const",
            length=cfg["res_length"],
            mask=cfg["list_of_all_qubits"],
        )

        # Qubit drive generator
        self.declare_gen(
            ch=qubit_ch,
            nqz=cfg['nqz_qubit'],
            mixer_freq=cfg['qubit_mixer_freq']
        )

        self.add_gauss(
            ch=qubit_ch,
            name="ramp",
            sigma=cfg['sigma'],
            length=cfg['sigma'] * 4,
            even_length=False
        )

        self.add_pulse(
            ch=qubit_ch,
            name="pi_pulse",
            style="arb",
            envelope="ramp",
            freq=cfg['qubit_freq_ge'],
            phase=cfg['qubit_phase'],
            gain=cfg['pi_amp'],   # freshly extracted pi gain
        )

        self.add_loop("shotloop", cfg["steps"])

    def _body(self, cfg):
        n_pi_pulses = cfg.get("n_pi_pulses", 0)

        for _ in range(n_pi_pulses):
            self.pulse(ch=cfg["qubit_ch"], name="pi_pulse", t=0)
            self.delay_auto(0.0)

        self.delay_auto(0.01)

        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class RepeatedPiPulseFidelity:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, outerFolder_save_plots,
                 round_num, experiment=None, save_figs=False, verbose=False, logger=None):

        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.outerFolder_save_plots = outerFolder_save_plots
        self.round_num = round_num
        self.experiment = experiment
        self.save_figs = save_figs
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        self.expt_name = "Pi_Pulse_Fid"
        self.Qubit = 'Q' + str(self.QubitIndex)

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}

    def run(self, n_pi_list=None):
        if n_pi_list is None:
            n_pi_list = np.arange(0, 41, 1)

        I_data = []
        Q_data = []

        for n_pi in n_pi_list:
            cfg = copy.deepcopy(self.config)
            cfg["n_pi_pulses"] = int(n_pi)

            prog = RepeatedPiPulseProgram(
                self.experiment.soccfg,
                reps=1,
                final_delay=cfg['relax_delay'],
                cfg=cfg
            )

            iq_list = prog.acquire(
                self.experiment.soc,
                soft_avgs=1,
                progress=False
            )

            I = iq_list[self.QubitIndex][0].T[0]
            Q = iq_list[self.QubitIndex][0].T[1]

            I_data.append(I)
            Q_data.append(Q)

        return {
            "n_pi_list": np.array(n_pi_list),
            "I": I_data,
            "Q": Q_data,
            "config": self.config}

    def plot_repeated_pi_population(self, n_pi_list, driven_pe_list, driven_pe_err=None, title=None, save_path=None):
        """
        Simple plot for repeated pi-pulse measurement. This function plots the driven excited-state population (Pe) vs pulse count (N),
        where we expect N = 0, 2, 4, 6,... to have low driven-Pe and N = 1, 3, 5, 7,... to have high driven-Pe.
        Do NOT get this driven-Pe confused with thermal Pe, these are different!
        If the pi-pulse is good, we expect a clean alternating high driven-Pe and low driven-Pe pattern.

        n_pi_list: array of pulse counts, e.g. [0, 1, 2, ..., 40]
        driven_pe_list: extracted DRIVEN excited-state population for each N
        driven_pe_err: optional uncertainty on driven-Pe
        """
        n_pi_list = np.asarray(n_pi_list)
        driven_pe_list = np.asarray(driven_pe_list)

        even_mask = (n_pi_list % 2 == 0)
        odd_mask = ~even_mask

        fig, ax = plt.subplots(figsize=(8, 5))

        if driven_pe_err is None:
            ax.scatter(n_pi_list[even_mask], driven_pe_list[even_mask], label="Even N: should be |g⟩", marker="o")
            ax.scatter(n_pi_list[odd_mask], driven_pe_list[odd_mask], label="Odd N: should be |e⟩", marker="s")
        else:
            driven_pe_err = np.asarray(driven_pe_err)
            ax.errorbar(n_pi_list[even_mask], driven_pe_list[even_mask],
                        yerr=driven_pe_err[even_mask], fmt="o", capsize=3,
                        label="Even N: should be |g⟩")
            ax.errorbar(n_pi_list[odd_mask], driven_pe_list[odd_mask],
                        yerr=driven_pe_err[odd_mask], fmt="s", capsize=3,
                        label="Odd N: should be |e⟩")

        ax.plot(n_pi_list, driven_pe_list, alpha=0.4)

        ax.axhline(0, linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(1, linestyle="--", linewidth=1, alpha=0.5)

        ax.set_xlabel("Number of repeated $X_\\pi$ pulses, N")
        ax.set_ylabel("Excited-state population, $P_e$")
        ax.set_ylim(-0.05, 1.05)

        if title is None:
            title = "Repeated $X_\\pi$ Pulse Check"
        ax.set_title(title)

        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()