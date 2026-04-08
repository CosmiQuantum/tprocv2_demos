import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import datetime
from build_task import *
from build_state import *
# from expt_config import *
from expt_config import *
import copy
from iminuit import Minuit
import time
import logging
import visdom

class EF_AmplitudeRabiExperiment:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, signal, save_shots=False, save_figs = True, experiment = None,
                 live_plot = None, increase_qubit_reps = False, qubit_to_increase_reps_for = None,
                 multiply_qubit_reps_by = 0, verbose = False, logger = None, qick_verbose=True, QZE=False,
                 projective_readout_pulse_len_us=9,  time_between_projective_readout_pulses=None, expt_name = "power_rabi_ef", unmasking_resgain = False):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.expt_name = expt_name
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.live_plot = live_plot
        self.signal = signal
        self.save_figs = save_figs
        self.save_shots = save_shots
        self.experiment = experiment
        self.verbose = verbose
        self.QZE = QZE
        self.projective_readout_pulse_len_us = projective_readout_pulse_len_us
        self.time_between_projective_readout_pulses = time_between_projective_readout_pulses
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

            if experiment is not None:
                self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
                self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
                self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
                if increase_qubit_reps:
                    if self.QubitIndex==qubit_to_increase_reps_for:
                        print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.config["reps"] *= multiply_qubit_reps_by
                print(f'Q {self.QubitIndex + 1} Round {self.round_num} EF Rabi configuration: ', self.config)


    def run(self, thresholding=False, use_iminuit_instead = True):
        print(self.config)

        amp_rabi = EF_AmplitudeRabiProgram(self.experiment.soccfg, reps=self.config['reps'],
                                        final_delay=self.config['relax_delay'], cfg=self.config)

        if self.live_plot:
            I, Q, gains = self.live_plotting(amp_rabi, thresholding)
        else:
            # Send the complied program that was set above to the qick hardware using soc
            # Tell how many times to repeat using the rounds function, and the definition will do that many measurements
            # and average over those
            # progress=True shows you the bar as data is being collected. maybe disable for speed in the future
            # The QICK will run the 'body' method in AmplitudeRabiProgram repeatedly for the iterations set in the
            # initalize loop when this aquire def is used
            # if thresholding:
            #     iq_list = amp_rabi.acquire(self.experiment.soc, soft_avgs=self.config["rounds"],
            #                                threshold=self.experiment.readout_cfg["threshold"],
            #                                angle=self.experiment.readout_cfg["ro_phase"], progress=self.qick_verbose)
            # else:
            #     iq_list = amp_rabi.acquire(self.experiment.soc, soft_avgs=self.config["rounds"],
            #                                progress=self.qick_verbose)
            iq_list = amp_rabi.acquire(self.experiment.soc, soft_avgs=self.config["rounds"], progress=self.qick_verbose)
            I = iq_list[self.QubitIndex][0, :, 0]
            Q = iq_list[self.QubitIndex][0, :, 1]
        # get the gains that were used so you can use to plot on the x axis
        gains = amp_rabi.get_pulse_param('qubit_pulse', "gain", as_array=True)
            # print('gains', gains)
            # print('I: ', I)
            # print('Q: ', Q)
        measurement_timestamp = (time.mktime(datetime.datetime.now().timetuple()))
        q1_fit_cosine, pi_amp = self.plot_results( I, Q, gains, config = self.config, use_iminuit_instead = use_iminuit_instead)
        # self.plot_results(I, Q, gains, config=self.config)
        return I, Q, gains, q1_fit_cosine, pi_amp, self.config, measurement_timestamp
        #return I, Q, gains, self.config

    def live_plotting(self, amp_rabi, soc):
        I = Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        assert viz.check_connection(timeout_seconds=5), "Visdom server not connected!"

        for ii in range(self.config["rounds"]):
            iq_list = amp_rabi.acquire(soc, soft_avgs=1, progress=True)
            gains = amp_rabi.get_pulse_param('qubit_pulse', "gain", as_array=True)

            this_I = iq_list[self.QubitIndex][0, :, 0]
            this_Q = iq_list[self.QubitIndex][0, :, 1]

            if I is None:  # ii == 0
                I, Q = this_I, this_Q
            else:
                I = (I * ii + this_I) / (ii + 1.0)
                Q = (Q * ii + this_Q) / (ii + 1.0)

            viz.line(X=gains, Y=I, opts=dict(height=400, width=700, title='Rabi I', showlegend=True, xlabel='expt_pts'),win='Rabi_I')
            viz.line(X=gains, Y=Q, opts=dict(height=400, width=700, title='Rabi Q', showlegend=True, xlabel='expt_pts'),win='Rabi_Q')
        return I, Q, gains

    def cosine(self, x, a, b, c, d):

        return a * np.cos(2. * np.pi * b * x - c * 2 * np.pi) + d

    def fit_cosine_iminuit(self, x, y, p0, fix_b=None, fix_c=None):
        """
        Iminuit-based cosine fit that mirrors scipy.curve_fit's output:
        returns popt and an approximate covariance matrix pcov.

        a: oscillation amplitude (what you use for populations)
        b: oscillation frequency in gain units (how many cycles per gain)
        c: phase offset (where the oscillation starts)
        d: DC offset (baseline of the readout)

        Optional: if we want to fit Q using the same b and c params we used for I
        fix_b: if not None, hold b fixed at this value
            Both I and Q are responding to the same driven Rabi oscillation.
            Fixing b says "These are two quadratures of the same rotation in the IQ plane".
        fix_c: if not None, hold c fixed at this value
            The oscillation’s phase should be a property of the qubit drive, not of which quadrature you look at.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        def chi2(a, b, c,
                 d):  # we don't have sigmas available so this is technically just the sum of squared errors, not chi2, which is fine.
            model = self.cosine(x, a, b, c, d)
            return np.sum((y - model) ** 2)

        m = Minuit(chi2, a=p0[0], b=p0[1], c=p0[2], d=p0[3])
        m.errordef = Minuit.LEAST_SQUARES  # least-squares / chi^2 objective

        # Setting some param limits
        m.limits["c"] = (-2 * np.pi, 2 * np.pi)  # phase periodic: keep it in a reasonable range

        # --- optionally fix b and/or c ---
        # We do this, for example, when we want to fit Q using the same b and c params we used for I
        if fix_b is not None:
            m.values["b"] = float(fix_b)
            m.fixed["b"] = True

        if fix_c is not None:
            m.values["c"] = float(fix_c)
            m.fixed["c"] = True
        # -------------------------------------

        # NEW: finds the correct valley. Runs Nelder-Mead simplex minimization. Note this is optional.
        # It only evaluates your objective function (chi2) at a small set of points.
        # It works by moving a geometric shape (a simplex) around parameter space until it finds a low point.
        # Makes fitting less sensitive to initial guesses
        m.simplex()

        m.migrad()  # Once you're in the correct valley, gradients are reliable and fast (find optimal params)
        m.hesse()  # Measure how wide the valley is (get errors)

        # Extract best-fit parameter values into a NumPy array
        popt = np.array([m.values["a"], m.values["b"], m.values["c"], m.values["d"]])

        # Convert Minuit's covariance object to a regular NumPy matrix
        # Our analysis code expects a NumPy array like the one from curve_fit
        cov = m.covariance
        if cov is None:
            pcov = np.full((4, 4), np.nan)
        else:
            names = ["a", "b", "c", "d"]
            pcov = np.zeros((4, 4))
            for i, ni in enumerate(names):
                for j, nj in enumerate(names):
                    pcov[i, j] = cov[ni, nj]

            # curve_fit default behavior (absolute_sigma=False) rescales covariance by chi2/(N - p)
            # If we want Minuit's errors to be comparable, we apply the same scaling here. Raw Minuit covariance is missing the noise scale
            # estimates the noise variance from the residuals
            N = x.size
            # Count how many parameters are actually free in this Minuit run
            # (fixed params reduce the number of fit dof)
            p_free = sum(not m.fixed[name] for name in ["a", "b", "c", "d"])

            ndof = N - p_free
            if ndof > 0 and np.isfinite(m.fval):
                scale = m.fval / ndof  # residual variance estimate
                pcov = pcov * scale
            else:
                pcov[:] = np.nan

        return popt, pcov

    def plot_results(self, I, Q, gains, config=None, fig_quality=100, use_iminuit_instead=True):
        """
        Updated old-style Rabi plotting function.

        Keeps:
        - original 2-panel plot
        - original title logic
        - original return values: (best_signal_fit, pi_amp)

        Adds:
        - optional iminuit fitting (the other option is curve fit)
        - fit better quadrature first, then fit the other with shared b
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            plt.rcParams.update({'font.size': 18})

            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

            # Initial guesses
            q1_a_guess_I = (np.max(I) - np.min(I)) / 2
            q1_d_guess_I = np.mean(I)
            q1_a_guess_Q = (np.max(Q) - np.min(Q)) / 2
            q1_d_guess_Q = np.mean(Q)

            q1_b_guess = 1 / gains[-1]
            q1_c_guess = 0

            q1_guess_I = [q1_a_guess_I, q1_b_guess, q1_c_guess, q1_d_guess_I]
            q1_guess_Q = [q1_a_guess_Q, q1_b_guess, q1_c_guess, q1_d_guess_Q]

            # Decide which quadrature to fit first
            span_I = abs(np.mean(I[-3:]) - np.mean(I[:3]))
            span_Q = abs(np.mean(Q[-3:]) - np.mean(Q[:3]))

            if 'I' in self.signal:
                fit_first = 'I'
            elif 'Q' in self.signal:
                fit_first = 'Q'
            elif 'None' in self.signal:
                fit_first = 'Q' if span_Q > span_I else 'I'
            else:
                print('Invalid signal passed, please do I Q or None')
                fit_first = 'I'

            # Fit first signal, then second using shared b if using iminuit
            if fit_first == 'I':
                if use_iminuit_instead:
                    q1_popt_I, q1_pcov_I = self.fit_cosine_iminuit(gains, I, q1_guess_I)
                else:
                    q1_popt_I, q1_pcov_I = curve_fit(
                        self.cosine, gains, I, maxfev=100000, p0=q1_guess_I
                    )

                q1_fit_cosine_I = self.cosine(gains, *q1_popt_I)
                b_shared = q1_popt_I[1]

                if use_iminuit_instead:
                    q1_popt_Q, q1_pcov_Q = self.fit_cosine_iminuit(
                        gains, Q, q1_guess_Q, fix_b=b_shared
                    )
                else:
                    q1_popt_Q, q1_pcov_Q = curve_fit(
                        self.cosine, gains, Q, maxfev=100000, p0=q1_guess_Q
                    )

                q1_fit_cosine_Q = self.cosine(gains, *q1_popt_Q)

            else:  # fit_first == 'Q'
                if use_iminuit_instead:
                    q1_popt_Q, q1_pcov_Q = self.fit_cosine_iminuit(gains, Q, q1_guess_Q)
                else:
                    q1_popt_Q, q1_pcov_Q = curve_fit(
                        self.cosine, gains, Q, maxfev=100000, p0=q1_guess_Q
                    )

                q1_fit_cosine_Q = self.cosine(gains, *q1_popt_Q)
                b_shared = q1_popt_Q[1]

                if use_iminuit_instead:
                    q1_popt_I, q1_pcov_I = self.fit_cosine_iminuit(
                        gains, I, q1_guess_I, fix_b=b_shared
                    )
                else:
                    q1_popt_I, q1_pcov_I = curve_fit(
                        self.cosine, gains, I, maxfev=100000, p0=q1_guess_I
                    )

                q1_fit_cosine_I = self.cosine(gains, *q1_popt_I)

            first_three_avg_I = np.mean(q1_fit_cosine_I[:3])
            last_three_avg_I = np.mean(q1_fit_cosine_I[-3:])
            first_three_avg_Q = np.mean(q1_fit_cosine_Q[:3])
            last_three_avg_Q = np.mean(q1_fit_cosine_Q[-3:])

            best_signal_fit = None
            pi_amp = None

            if 'Q' in self.signal:
                best_signal_fit = q1_fit_cosine_Q
                if last_three_avg_Q > first_three_avg_Q:
                    pi_amp = gains[np.argmax(best_signal_fit)]
                else:
                    pi_amp = gains[np.argmin(best_signal_fit)]

            elif 'I' in self.signal:
                best_signal_fit = q1_fit_cosine_I
                if last_three_avg_I > first_three_avg_I:
                    pi_amp = gains[np.argmax(best_signal_fit)]
                else:
                    pi_amp = gains[np.argmin(best_signal_fit)]

            elif 'None' in self.signal:
                if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
                    best_signal_fit = q1_fit_cosine_Q
                    if last_three_avg_Q > first_three_avg_Q:
                        pi_amp = gains[np.argmax(best_signal_fit)]
                    else:
                        pi_amp = gains[np.argmin(best_signal_fit)]
                else:
                    best_signal_fit = q1_fit_cosine_I
                    if last_three_avg_I > first_three_avg_I:
                        pi_amp = gains[np.argmax(best_signal_fit)]
                    else:
                        pi_amp = gains[np.argmin(best_signal_fit)]
            else:
                print('Invalid signal passed, please do I Q or None')

            if pi_amp is not None:
                ax1.axvline(pi_amp, linestyle='--', linewidth=2, color='black', label='pi amp')
                ax2.axvline(pi_amp, linestyle='--', linewidth=2, color='black', label='pi amp')

            # Plot fits
            ax2.plot(gains, q1_fit_cosine_Q, '-', color='red', linewidth=3, label="Fit")
            ax1.plot(gains, q1_fit_cosine_I, '-', color='red', linewidth=3, label="Fit")

            # Title logic kept from old function
            if config is not None:
                if self.QZE:
                    fig.text(
                        plot_middle, 0.98,
                        f"Rabi Q{self.QubitIndex + 1}_"
                        + f", {config['reps']}*{config['rounds']} avgs"
                        + f" pi_amp {round(pi_amp, 2)} "
                        + f"projective readout pulse length: {self.projective_readout_pulse_len_us}"
                        + f" readout pulse amp: {self.experiment.readout_cfg['res_gain_ge'][self.QubitIndex]} ",
                        fontsize=24, ha='center', va='top'
                    )
                else:
                    fig.text(
                        plot_middle, 0.98,
                        f"Rabi Q{self.QubitIndex + 1}_"
                        + f", {config['reps']}*{config['rounds']} avgs"
                        + f" pi_amp {pi_amp} ",
                        fontsize=24, ha='center', va='top'
                    )
            else:
                fig.text(
                    plot_middle, 0.98,
                    f"Rabi Q{self.QubitIndex + 1}_"
                    + f", {self.config['sigma'] * 1000} ns sigma"
                    + f" pi_amp {pi_amp} "
                    + f", {self.config['reps']}*{self.config['rounds']} avgs",
                    fontsize=24, ha='center', va='top'
                )

            # Plot raw data
            ax1.plot(gains, I, label="Gain (a.u.)", linewidth=2)
            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)

            ax2.plot(gains, Q, label="Q", linewidth=2)
            ax2.set_xlabel("Gain (a.u.)", fontsize=20)
            ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
            ax2.tick_params(axis='both', which='major', labelsize=16)

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(
                    outerFolder_expt,
                    f"R{self.round_num}_Q{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)
            return best_signal_fit, pi_amp

        except Exception as e:
            if self.verbose:
                print("Error fitting cosine:", e)
            self.logger.info(f"Error fitting cosine: {e}")
            return None, None


    def get_results(self, I, Q, gains, grab_depths = False):

        q1_a_guess_I = (np.max(I) - np.min(I)) / 2
        q1_d_guess_I = np.mean(I)
        q1_a_guess_Q = (np.max(Q) - np.min(Q)) / 2
        q1_d_guess_Q = np.mean(Q)
        q1_b_guess = 1 / gains[-1]
        q1_c_guess = 0

        q1_guess_I = [q1_a_guess_I, q1_b_guess, q1_c_guess, q1_d_guess_I]
        q1_popt_I, q1_pcov_I = curve_fit(self.cosine, gains, I, maxfev=100000, p0=q1_guess_I)
        q1_fit_cosine_I = self.cosine(gains, *q1_popt_I)

        q1_guess_Q = [q1_a_guess_Q, q1_b_guess, q1_c_guess, q1_d_guess_Q]
        q1_popt_Q, q1_pcov_Q = curve_fit(self.cosine, gains, Q, maxfev=100000, p0=q1_guess_Q)
        q1_fit_cosine_Q = self.cosine(gains, *q1_popt_Q)

        first_three_avg_I = np.mean(q1_fit_cosine_I[:3])
        last_three_avg_I = np.mean(q1_fit_cosine_I[-3:])
        first_three_avg_Q = np.mean(q1_fit_cosine_Q[:3])
        last_three_avg_Q = np.mean(q1_fit_cosine_Q[-3:])

        best_signal_fit = None
        pi_amp = None
        if 'Q' in self.signal:
            best_signal_fit = q1_fit_cosine_Q
            # figure out if you should take the min or the max value of the fit to say where pi_amp should be
            if last_three_avg_Q > first_three_avg_Q:
                pi_amp = gains[np.argmax(best_signal_fit)]
            else:
                pi_amp = gains[np.argmin(best_signal_fit)]
        if 'I' in self.signal:
            best_signal_fit = q1_fit_cosine_I
            # figure out if you should take the min or the max value of the fit to say where pi_amp should be
            if last_three_avg_I > first_three_avg_I:
                pi_amp = gains[np.argmax(best_signal_fit)]
            else:
                pi_amp = gains[np.argmin(best_signal_fit)]
        if 'None' in self.signal:
            # choose the best signal depending on which has a larger magnitude
            if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
                best_signal_fit = q1_fit_cosine_Q
                # figure out if you should take the min or the max value of the fit to say where pi_amp should be
                if last_three_avg_Q > first_three_avg_Q:
                    pi_amp = gains[np.argmax(best_signal_fit)]
                else:
                    pi_amp = gains[np.argmin(best_signal_fit)]
            else:
                best_signal_fit = q1_fit_cosine_I
                # figure out if you should take the min or the max value of the fit to say where pi_amp should be
                if last_three_avg_I > first_three_avg_I:
                    pi_amp = gains[np.argmax(best_signal_fit)]
                else:
                    pi_amp = gains[np.argmin(best_signal_fit)]
            tot_amp = [np.sqrt((ifit)**2 + (qfit)**2) for ifit,qfit in zip(q1_fit_cosine_I, q1_fit_cosine_Q)]
            depth = abs(tot_amp[np.argmin(tot_amp)] - tot_amp[np.argmax(tot_amp)])
        else:
            print('Invalid signal passed, please do I Q or None')
        if grab_depths:
            return best_signal_fit, pi_amp, depth
        else:
            return best_signal_fit, pi_amp

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)


class EF_AmplitudeRabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ef'],
                         mux_gains=cfg['res_gain_ef'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ef'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_gauss(ch=qubit_ch, name="ge_ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="pi_ge",
                       style="arb",
                       envelope="ge_ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma_ef'], length=cfg['sigma_ef'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ef'],
                       )

        self.add_loop("gainloop", cfg["steps"])

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="pi_ge", t=0)  # play ge pi pulse
        self.delay_auto(t=0.0, tag='waiting after pi')  # Wait til ge pi pulse is done before proceeding
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0) #ef pulse
        self.delay_auto(t=0.0, tag='waiting') #wait
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0) #probe pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

