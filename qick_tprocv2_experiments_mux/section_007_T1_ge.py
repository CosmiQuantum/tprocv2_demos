from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from build_task import *
from build_state import *
from expt_config import *
from system_config import *
import copy
import visdom
import logging
from iminuit import Minuit

class T1Program(AveragerProgramV2):
    def _initialize(self, cfg):

        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
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

        self.add_loop("waitloop", cfg["steps"])

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse
        self.delay_auto(cfg['wait_time'] + 0.01, tag='wait')  # wait_time after last pulse
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])


class T1Measurement:
    def __init__(self, QubitIndex, number_of_qubits,  outerFolder, round_num, signal, save_figs, experiment = None,
                 live_plot = None, fit_data = None, increase_qubit_reps = False, qubit_to_increase_reps_for = None,
                 multiply_qubit_reps_by = 0, verbose = False, logger = None, qick_verbose=True, save_shots=False,
                 set_relax_delay=False, relax_delay=1000):

        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.expt_name = "T1_ge"
        self.fit_data = fit_data
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.live_plot = live_plot
        self.signal = signal
        self.save_figs = save_figs
        self.verbose = verbose
        self.save_shots = save_shots
        self.set_relax_delay = set_relax_delay
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            if increase_qubit_reps:
                    if self.QubitIndex==qubit_to_increase_reps_for:
                        self.logger.info(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        if self.verbose: print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.config["reps"] *= multiply_qubit_reps_by
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} T1 configuration: {self.config}')
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} T1 configuration: {self.config}')
            if self.set_relax_delay:
                self.config['relax_delay'] = relax_delay
                print(f'set t1 relax delay to {relax_delay} us')

    def run(self, thresholding=False):
        now = datetime.datetime.now()
        t1 = T1Program(self.experiment.soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'], cfg=self.config)

        if self.live_plot:
            I, Q, delay_times = self.live_plotting(t1, thresholding)
        else:
            if thresholding:
                iq_list = t1.acquire(self.experiment.soc, soft_avgs=self.config['rounds'],
                                           threshold=self.experiment.readout_cfg["threshold"],
                                           angle=self.experiment.readout_cfg["ro_phase"], progress=True)
            else:
                iq_list = t1.acquire(self.experiment.soc, soft_avgs=self.config['rounds'], progress=True)


            I = iq_list[self.QubitIndex][0, :, 0]
            Q = iq_list[self.QubitIndex][0, :, 1]
            delay_times = t1.get_time_param('wait', "t", as_array=True)


        if self.fit_data:
            q1_fit_exponential, T1_err, T1_est, plot_sig = self.t1_fit(I, Q, delay_times)
        else:
            q1_fit_exponential, T1_est, T1_err = None, None, None

        if self.plot_results:
            self.plot_results( I, Q, delay_times, now)

        if self.save_shots:
            raw_0 = t1.get_raw()  # I,Q data without normalizing to readout window, subtracting readout offset, or rotation/thresholding
            Ishots = raw_0[self.QubitIndex][:, :, 0, 0]
            Qshots = raw_0[self.QubitIndex][:, :, 0, 1]
            return T1_est, T1_err, Ishots, Qshots, delay_times, q1_fit_exponential, self.config

        else:
            return  T1_est, T1_err, I, Q, delay_times, q1_fit_exponential, self.config

    def live_plotting(self, t1, thresholding):
        I = Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        if not viz.check_connection(timeout_seconds=5):
            raise RuntimeError("Visdom server not connected!")
        for ii in range(self.config["rounds"]):
            #iq_list = t1.acquire(self.experiment.soc, soft_avgs=1, progress=True)
            if thresholding:
                iq_list = t1.acquire(self.experiment.soc, soft_avgs=1,
                                           threshold=self.experiment.readout_cfg["threshold"],
                                           angle=self.experiment.readout_cfg["ro_phase"], progress=True)
            else:
                iq_list = t1.acquire(self.experiment.soc, soft_avgs=1, progress=True)
            delay_times = t1.get_time_param('wait', "t", as_array=True)

            this_I = iq_list[self.QubitIndex][0, :, 0]
            this_Q = iq_list[self.QubitIndex][0, :, 1]

            if I is None:  # ii == 0
                I, Q = this_I, this_Q
            else:
                I = (I * ii + this_I) / (ii + 1.0)
                Q = (Q * ii + this_Q) / (ii + 1.0)

            viz.line(X=delay_times, Y=I, opts=dict(height=400, width=700, title='T1 I', showlegend=True, xlabel='expt_pts'),win='T1_I')
            viz.line(X=delay_times, Y=Q, opts=dict(height=400, width=700, title='T1 Q', showlegend=True, xlabel='expt_pts'),win='T1_Q')
        return I, Q, delay_times

    def exponential(self, x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def exponential(self, x, a, b, c, d):
        return a * np.exp(- (x - b) / c) + d

    def t1_fit(self, I, Q, delay_times):
        if 'I' in self.signal:
            signal = I
            plot_sig = 'I'
        elif 'Q' in self.signal:
            signal = Q
            plot_sig = 'Q'
        else:
            if abs(I[-1] - I[0]) > abs(Q[-1] - Q[0]):
                signal = I
                plot_sig = 'I'
            else:
                signal = Q
                plot_sig = 'Q'

        # Initial guess for parameters
        q1_a_guess = np.max(signal) - np.min(signal)  # Initial guess for amplitude (a)
        q1_b_guess = 0  # Initial guess for time shift (b)
        q1_c_guess = (delay_times[-1] - delay_times[0]) / 5  # Initial guess for decay constant (T1)
        q1_d_guess = np.min(signal)  # Initial guess for baseline (d)

        # Form the guess array
        q1_guess = [q1_a_guess, q1_b_guess, q1_c_guess, q1_d_guess]

        # Define bounds to constrain T1 (c) to be positive, but allow amplitude (a) to be negative
        lower_bounds = [-np.inf, -np.inf, 0, -np.inf]  # Amplitude (a) can be negative/positive, but T1 (c) > 0
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]  # No upper bound on parameters

        # Perform the fit using the 'trf' method with bounds
        q1_popt, q1_pcov = curve_fit(self.exponential, delay_times, signal,
                                     p0=q1_guess, bounds=(lower_bounds, upper_bounds),
                                     method='trf', maxfev=10000)

        # Generate the fitted exponential curve
        q1_fit_exponential = self.exponential(delay_times, *q1_popt)

        # Extract T1 and its error
        T1_est = q1_popt[2]  # Decay constant T1
        T1_err = np.sqrt(q1_pcov[2][2]) if q1_pcov[2][2] >= 0 else float('inf')  # Ensure error is valid

        return q1_fit_exponential, T1_err, T1_est, plot_sig

    def t1_fit_iminuit(self, I, Q, delay_times, y_errs=None):
        """
        Fits T1 curve using a 3-parameter exponential with a chi^2 or least-squares minimizer depending on whether you provide
        the errs of each point in the curve or not (iminuit).
        (3 parameters, no time shift):
            y(t) = d + a * (1 - exp(-t / c))

            a: amplitude (positive or negative)
            c: T1 (>0)
            d: baseline
        Migrad gives you best-fit parameters.
        Hesse tells you how uncertain they are.
        """
        # ---------------- choose signal (same logic we were using before) ----------------
        I = np.asarray(I, float)
        Q = np.asarray(Q, float)
        t = np.asarray(delay_times, float)

        if 'I' in self.signal:
            signal = I
            plot_sig = 'I'
        elif 'Q' in self.signal:
            signal = Q
            plot_sig = 'Q'
        else:
            # auto-pick whichever has bigger diff
            if abs(I[-1] - I[0]) > abs(Q[-1] - Q[0]):
                signal = I
                plot_sig = 'I'
            else:
                signal = Q
                plot_sig = 'Q'

        # sort by time just in case
        order = np.argsort(t)
        t = t[order]
        signal = signal[order]
        if y_errs is not None:
            sigma = np.asarray(y_errs, float)[order]

        # ---------------- new, simpler 3-parameter model ----------------
        def t1_model(tvals, a, c, d):
            return d + a * (1.0 - np.exp(-tvals / c))

        # ---------------- initial guesses ----------------
        sig_min = float(np.min(signal))
        sig_max = float(np.max(signal))

        d_guess = float(signal[0])  # value at t=0. Based on the new function we are using (t1_model above), d is just the starting value
        a_guess = float(sig_max - sig_min)  # amplitude (can be +/-), this sign decides if the data rises or decays
        if a_guess == 0.0: # if the signal is extremely flat or constant
            a_guess = float(np.ptp(signal) or 1.0) # peak-to-peak amplitude. If that number is zero, uses 1 instead.

        span = max(float(t[-1] - t[0]), 1e-3)
        c_guess = max(span / 3.0, 1e-3)  # T1 ~ span/3, and we set a floor of 1e-3 to make sure T>0

        # ---------------- chi^2 function ----------------
        if y_errs is not None:
            # Handle zero or negative uncertainties
            if np.any(sigma > 0): # Look at only the positive sigmas, and use their median.
                med_pos = np.median(sigma[sigma > 0])
                sigma = np.where(sigma <= 0, med_pos, sigma) # replaces zeros, negative sigmas, NaNs
            else:
                raise ValueError("y_errs must contain at least one positive value if y_errs is not set to None.")

            def chi2(a, c, d):
                model = t1_model(t, a, c, d)
                r = (signal - model) / sigma
                return np.sum(r * r)

            minimizer_func = chi2

        else:
            def lsquares(a, c, d):
                model = t1_model(t, a, c, d)
                r = signal - model
                return np.sum(r * r)

            minimizer_func = lsquares

        # ---------------- run Minuit ----------------
        m = Minuit(
            minimizer_func,
            a=a_guess,
            c=c_guess,
            d=d_guess,
        )

        m.errordef = Minuit.LEAST_SQUARES # used for chi^2 or least squares method, which we are using

        # limits: we enforce c>0 (for a positive T1 value) and limit it to reasonable values relative to the time window
        T1_min = max(span / 50.0, 1e-3)
        T1_max = span * 20.0

        m.limits["c"] = (T1_min, T1_max)

        # run minimization
        m.migrad()

        # try one more time if we get invalid results
        if not m.valid: # If not valid, we try again with opposite a
            m.values["a"] = -a_guess
            m.migrad()

        m.hesse() # computes the covariance matrix after minimization

        # ---------------- we extract the results ----------------
        a_fit = m.values["a"]
        c_fit = m.values["c"]  # T1
        d_fit = m.values["d"]

        # ---------------------- T1 uncertainty from covariance matrix ----------------------
        cov = m.covariance
        T1_err = np.nan

        if cov is not None:
            try:
                var_c = cov["c", "c"]  # Var(T1) = C[c,c]
            except Exception:
                # Fallback if covariance behaves like a NumPy array
                params = list(m.parameters)
                if "c" in params:
                    idx_c = params.index("c")
                    var_c = cov[idx_c, idx_c]
                else:
                    var_c = None

            # --- Apply residual scaling if fit was unweighted ---
            if var_c is not None and var_c >= 0:
                if y_errs is None:
                    N = len(t)
                    p = 3  # a, c (T1), d
                    ndof = N - p
                    if ndof > 0 and np.isfinite(m.fval):
                        scale = m.fval / ndof  # residual variance estimate
                        var_c *= scale

                T1_err = float(np.sqrt(var_c))

        T1_est = float(c_fit) # out T1 result

        # compute fit and unsort back
        fit_sorted = t1_model(t, a_fit, c_fit, d_fit) # since we sorted the data at the beginning just in case it was out of order, the fit is based on sorted data
        inv_order = np.argsort(order)
        fit_curve = fit_sorted[inv_order] # putting it back to the original order (should be sorted nonetheless, but this is done to be 100% consistent w original order)

        return fit_curve, T1_err, T1_est, plot_sig

    def plot_results(self, I, Q, delay_times, date, I_errs = None, Q_errs = None, config = None, fig_quality =100, iminuit_fit_instead = False):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})

        # Calculate the middle of the plot area
        plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

        y_err_for_fit = None # used to choose which y_err to pass into iminuit (I or Q per-point errors)

        if self.fit_data:
            if iminuit_fit_instead:
                if 'I' in self.signal and I_errs is not None:
                    y_err_for_fit = I_errs
                elif 'Q' in self.signal and Q_errs is not None:
                    y_err_for_fit = Q_errs
                # else: no matching errs, leave as None

                q1_fit_exponential, T1_err, T1_est, plot_sig = self.t1_fit_iminuit(I, Q, delay_times, y_err_for_fit)
            else:
                q1_fit_exponential, T1_err, T1_est, plot_sig = self.t1_fit(I, Q, delay_times)

            if 'I' in plot_sig:
                ax1.plot(delay_times, q1_fit_exponential, '-', color='red', linewidth=3, label="Fit")
            else:
                ax2.plot(delay_times, q1_fit_exponential, '-', color='red', linewidth=3, label="Fit")

            # Add title, centered on the plot area
            if config is not None:
                fig.text(plot_middle, 0.98,
                         f"Q{self.QubitIndex + 1} " + f"T1={T1_est:.2f} +/- {T1_err:.2f} us" + f", {float(config['reps'])}*{float(config['rounds'])} avgs,",
                         fontsize=24, ha='center',
                         va='top')  # , pi gain %.2f" % float(config['pi_amp']) + f", {float(config['sigma']) * 1000} ns sigma
            else:
                fig.text(plot_middle, 0.98,
                         f"T1 Q{self.QubitIndex + 1}, T1={T1_est:.2f} +/- {T1_err:.2f}us",
                         fontsize=24, ha='center', va='top')

        else:
            if config is not None:
                fig.text(plot_middle, 0.98,
                         f"T1 Q{self.QubitIndex + 1}" + f", {float(config['reps'])}*{float(config['rounds'])} avgs,",
                         fontsize=24, ha='center',
                         va='top')  # , pi gain %.2f" % float(config['pi_amp']) + f", {float(config['sigma']) * 1000} ns sigma"   you can put this back once you save configs properly for when replotting
            else:
                fig.text(plot_middle, 0.98,
                         f"T1 Q{self.QubitIndex + 1}",
                         fontsize=24, ha='center', va='top')
            q1_fit_exponential = None
            T1_est = None
            T1_err = None

        # I subplot
        if I_errs is not None:
            ax1.errorbar(
                delay_times,
                I,
                yerr=I_errs,
                fmt='o-',
                linewidth=2,
                capsize=3,
                label="I")
        else:
            ax1.plot(delay_times, I, label="I", linewidth=2)

        ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        # ax1.axvline(freq_q, color='orange', linestyle='--', linewidth=2)

        # Q subplot
        if Q_errs is not None:
            ax2.errorbar(
                delay_times,
                Q,
                yerr=Q_errs,
                fmt='o-',
                linewidth=2,
                capsize=3,
                label="Q"
            )
        else:
            ax2.plot(delay_times, Q, label="Q", linewidth=2)

        ax2.set_xlabel("Delay time (us)", fontsize=20)
        ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        # ax2.axvline(freq_q, color='orange', linestyle='--', linewidth=2)

        # Adjust spacing
        plt.tight_layout()

        # Adjust the top margin to make room for the title
        plt.subplots_adjust(top=0.93)
        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')  # , facecolor='white'
        plt.close(fig)

        return I, Q, delay_times, q1_fit_exponential, T1_err, T1_est, plot_sig
