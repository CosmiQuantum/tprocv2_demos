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

        self.declare_gen(ch=gen_ch,nqz=cfg['nqz_res'],ro_ch=ro_chs[0],mux_freqs=cfg['res_freq_ge'],mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],mixer_freq=cfg['mixer_freq'])

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch,length=cfg['res_length'],freq=f,phase=ph,gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch,name="res_pulse",style="const",length=cfg["res_length"],mask=cfg["list_of_all_qubits"])

        self.declare_gen(ch=qubit_ch,nqz=cfg['nqz_qubit'],mixer_freq=cfg['qubit_mixer_freq'])
        self.add_gauss(ch=qubit_ch,name="ramp",sigma=cfg['sigma'],length=cfg['sigma'] * 4,even_length=False)
        self.add_pulse(ch=qubit_ch,name="pi_pulse",style="arb",envelope="ramp",freq=cfg['qubit_freq_ge'],phase=cfg['qubit_phase'], gain=cfg['pi_amp'])

        self.add_loop("shotloop", cfg["shots_per_pulse_count"]) # This loop collects many single-shot IQ points for one fixed N

    def _body(self, cfg):
        num_pi_pulses_this_point = cfg["n_pi_pulses"]

        # Apply the selected number of repeated pi pulses for this N point
        for _ in range(num_pi_pulses_this_point):
            self.pulse(ch=cfg["qubit_ch"], name="pi_pulse", t=0)
            self.delay_auto(0.0) # spacing between repeated pi pulses

        # self.delay_auto(0.0) # spacing between the last pi pulse and the readout pulse
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0) # readout pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class RepeatedPiPulseFidelity:
    """
    Explanation of the different loops:
    1. run() loop
        Chooses N = 0, 1, 2, ..., 40

    2. _body() loop
       Plays N pi pulses and reads out, to produce one shot

    3. shotloop
       Repeats the loop in _body a chosen number of times to collect statistics (various shots for each N)
    """
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
        # n_pi_list = all N values you sweep
        # n_pi_pulses tells the program how many pi pulses to apply for that one shot set.
        if n_pi_list is None:
            n_pi_list = np.arange(
                self.config["min_pi_pulses"],
                self.config["max_pi_pulses"] + self.config["pi_pulse_step"],
                self.config["pi_pulse_step"])

        I_data = []
        Q_data = []

        for n_pi in n_pi_list:
            cfg = copy.deepcopy(self.config)
            cfg["n_pi_pulses"] = int(n_pi)

            prog = RepeatedPiPulseProgram(self.experiment.soccfg,reps=1,final_delay=cfg['relax_delay'], cfg=cfg)

            iq_list = prog.acquire(self.experiment.soc, soft_avgs=cfg["py_avg"], progress=False)

            I = iq_list[self.QubitIndex][0].T[0]
            Q = iq_list[self.QubitIndex][0].T[1]

            I_data.append(I)
            Q_data.append(Q)

        return {
            "n_pi_list": np.array(n_pi_list),
            "I": I_data,
            "Q": Q_data,
            "config": self.config}

    def plot_repeated_pi_population(self, n_pi_list, driven_pe_list, driven_pe_err=None,
                                    title=None, save_path=None):
        """
        Plots driven Pe vs number of repeated pi pulses and estimates a rough
        per-pulse fidelity from the even/odd contrast decay.

        This is NOT formal RB fidelity. It is a repeated-pulse contrast fidelity estimate.
        """

        n_pi_list = np.asarray(n_pi_list)
        driven_pe_list = np.asarray(driven_pe_list)

        if driven_pe_err is not None:
            driven_pe_err = np.asarray(driven_pe_err)

        even_mask = (n_pi_list % 2 == 0)
        odd_mask = (n_pi_list % 2 == 1)

        # ---------------- Plot driven Pe vs N ----------------
        fig, ax = plt.subplots(figsize=(8, 5))

        if driven_pe_err is None:
            ax.scatter(n_pi_list[even_mask], driven_pe_list[even_mask],
                       label="Even N: should be |g⟩", marker="o")
            ax.scatter(n_pi_list[odd_mask], driven_pe_list[odd_mask],
                       label="Odd N: should be |e⟩", marker="s")
        else:
            ax.errorbar(n_pi_list[even_mask], driven_pe_list[even_mask],
                        yerr=driven_pe_err[even_mask], fmt="o", capsize=3,
                        label="Even N: should be |g⟩")
            ax.errorbar(n_pi_list[odd_mask], driven_pe_list[odd_mask],
                        yerr=driven_pe_err[odd_mask], fmt="s", capsize=3,
                        label="Odd N: should be |e⟩")

        ax.plot(n_pi_list, driven_pe_list, alpha=0.4)

        ax.axhline(0, linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(1, linestyle="--", linewidth=1, alpha=0.5)

        ax.set_xlabel("Number of repeated $\pi$ pulses, N")
        ax.set_ylabel("Driven excited-state population, $P_e$")
        ax.set_ylim(-0.05, 1.05)

        # ---------------- Calculate contrast decay ----------------
        contrast_N = []
        contrast_vals = []

        for odd_N in n_pi_list[odd_mask]:
            even_N = odd_N + 1

            if even_N in n_pi_list:
                odd_idx = np.where(n_pi_list == odd_N)[0][0]
                even_idx = np.where(n_pi_list == even_N)[0][0]

                contrast = driven_pe_list[odd_idx] - driven_pe_list[even_idx]

                contrast_N.append(odd_N)
                contrast_vals.append(contrast)

        contrast_N = np.asarray(contrast_N)
        contrast_vals = np.asarray(contrast_vals)

        pulse_fidelity = None
        pulse_error = None
        Nd = None

        def exp_decay(N, C0, Nd):
            return C0 * np.exp(-N / Nd)

        try:
            valid = np.isfinite(contrast_vals) & (contrast_vals > 0)

            if np.sum(valid) >= 3:
                popt, pcov = curve_fit(
                    exp_decay,
                    contrast_N[valid],
                    contrast_vals[valid],
                    p0=[contrast_vals[valid][0], 100],
                    bounds=([0, 1e-6], [2, np.inf]))

                C0, Nd = popt

                pulse_error = 1 / (2 * Nd)
                pulse_fidelity = 1 - pulse_error

                N_fit = np.linspace(np.min(contrast_N[valid]),
                                    np.max(contrast_N[valid]), 300)
                C_fit = exp_decay(N_fit, C0, Nd)

                # Optional: plot contrast decay on same axis centered around 0.5 for visual guide
                # This is just a visual guide, not the raw Pe data.
                ax.plot(N_fit, 0.5 + 0.5 * C_fit, linestyle="--",
                        label="Contrast decay guide")
                ax.plot(N_fit, 0.5 - 0.5 * C_fit, linestyle="--")

                text = (
                    f"$N_d$ = {Nd:.1f} pulses\n"
                    f"$r_{{pulse}} \\approx$ {pulse_error:.3e}\n"
                    f"$F_{{pulse}} \\approx$ {pulse_fidelity * 100:.2f}%")

                ax.text(
                    0.03, 0.05, text,
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

        except Exception as e:
            print(f"Could not fit contrast decay: {e}")

        if title is None:
            title = "Repeated $\pi$ Pulse Check"
        ax.set_title(title)

        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

        return {
            "contrast_N": contrast_N,
            "contrast_vals": contrast_vals,
            "Nd": Nd,
            "pulse_error": pulse_error,
            "pulse_fidelity": pulse_fidelity
        }