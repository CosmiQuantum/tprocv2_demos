from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from build_task import *
from build_state import *
from expt_config import *
from system_config import *
import copy
from iminuit import Minuit
import time
import visdom
import logging

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
                 multiply_qubit_reps_by = 0, verbose = False, logger = None, qick_verbose=True, save_shots=True,
                 set_relax_delay=False, relax_delay=1000, unmasking_resgain = False, adjust_reps_to = None,
                 reduce_rlx_delay = False, reduce_rlx_delay_to = 1000):

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
        self.reduce_rlx_delay = reduce_rlx_delay
        self.reduce_rlx_delay_to = reduce_rlx_delay_to
        self.live_plot = live_plot
        self.signal = signal
        self.save_figs = save_figs
        self.verbose = verbose
        self.save_shots = save_shots
        self.set_relax_delay = set_relax_delay
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]
        if adjust_reps_to is not None:
            self.exp_cfg["reps"] = adjust_reps_to
        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            if increase_qubit_reps:
                    if self.QubitIndex==qubit_to_increase_reps_for:
                        self.logger.info(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        if self.verbose: print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.config["reps"] *= multiply_qubit_reps_by

            if reduce_rlx_delay:
                print(f"Reducing relax_delay for {self.QubitIndex + 1} to {reduce_rlx_delay_to}.")
                self.config["relax_delay"] = reduce_rlx_delay_to
                self.logger.info(f"Reducing relax_delay for {self.QubitIndex + 1} to {reduce_rlx_delay_to}")

            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} T1 configuration: {self.config}')
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} T1 configuration: {self.config}')
            if self.set_relax_delay:
                self.config['relax_delay'] = relax_delay
                print(f'set t1 relax delay to {relax_delay} us')

    def run(self, thresholding=False, use_iminuit_instead = True, active_reset = False):
        if active_reset:

            t1 = T1Program_active_reset(self.experiment.soccfg, reps=self.config['reps'],final_delay=0.01, cfg=self.config)

            if thresholding:
                iq_list = t1.acquire(self.experiment.soc, soft_avgs=self.config['rounds'],
                                     threshold=self.experiment.readout_cfg["threshold"],
                                     angle=self.experiment.readout_cfg["ro_phase"], progress=True)
            else:
                iq_list = t1.acquire(self.experiment.soc, soft_avgs=self.config['rounds'], progress=True)

                iq_q = np.asarray(iq_list[self.QubitIndex])
                print("Processed IQ shape:", iq_q.shape)
                print("Number of T1 points:", self.config["steps"])

                I = iq_q[-1, :, 0]
                Q = iq_q[-1, :, 1]
                delay_times = t1.get_time_param('wait', "t", as_array=True)

                # # plots all of the stored indices throughout the active reset pipeline
                # fig, ax = plt.subplots(figsize=(10, 6))
                # for read_idx in range(iq_q.shape[0]):
                #     ax.plot(delay_times, iq_q[read_idx, :, 0], label=f"Readout {read_idx}")
                # ax.set_xlabel("Delay time (us)")
                # ax.set_ylabel("I (a.u.)")
                # ax.legend()
                # plt.tight_layout()
                # debug_file = os.path.join(self.outerFolder, f"active_reset_T1_all_readouts_Q{self.QubitIndex + 1}.png")
                # fig.savefig(debug_file, dpi=150, bbox_inches="tight")
                # plt.close(fig)

            measurement_timestamp = time.mktime(datetime.datetime.now().timetuple())

            if self.fit_data:
                if use_iminuit_instead:
                    q1_fit_exponential, T1_err, T1_est, fit_info = self.t1_fit_iminuit(I, Q, delay_times)
                    plot_sig = fit_info["plot_sig"]
                else:
                    q1_fit_exponential, T1_err, T1_est, plot_sig = self.t1_fit(I, Q, delay_times)
            else:
                q1_fit_exponential, T1_est, T1_err = None, None, None

            if self.plot_results:
                self.plot_results_active_reset(I, Q, delay_times)

            if self.save_shots:
                raw_0 = t1.get_raw()
                raw_q = np.asarray(raw_0[self.QubitIndex])

                ###########################################################
                # Diagnostic plot to see threshold used with all shots for one relax delay step
                # This must be exactly the same numerical threshold passed to read_and_jump().
                ro_ch_this = self.config["ro_ch"][0]
                res_length_cycles = self.experiment.soccfg.us2cycles(us=self.config["res_length"], ro_ch=ro_ch_this)
                threshold_raw = int(round(self.config["threshold"] * res_length_cycles))
                diagnostic_folder = os.path.join(self.outerFolder, "T1_ge_active_reset","decision_threshold_diagnostics")
                self.plot_first_delay_reset_decision_shots(
                    raw_q=raw_q,
                    decision_threshold=threshold_raw,
                    delay_times=delay_times,
                    delay_index=1,
                    decision_readout_index=0,
                    save_folder=diagnostic_folder,
                    show_plot=False,
                    print_summary=True)

                ##########################################################
                Ishots = raw_q[:, :, -1, 0]
                Qshots = raw_q[:, :, -1, 1]
                return T1_est, T1_err, I, Q, Ishots, Qshots, delay_times, q1_fit_exponential, self.config, measurement_timestamp
            else:
                return T1_est, T1_err, I, Q, None, None, delay_times, q1_fit_exponential, self.config, measurement_timestamp

        else: # standard non-active reset code
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

            measurement_timestamp = (time.mktime(datetime.datetime.now().timetuple()))

            if self.fit_data:
                if use_iminuit_instead:
                    q1_fit_exponential, T1_err, T1_est, fit_info = self.t1_fit_iminuit(I, Q, delay_times)
                    plot_sig = fit_info["plot_sig"]
                else:
                    q1_fit_exponential, T1_err, T1_est, plot_sig = self.t1_fit(I, Q, delay_times) # curve fit
            else:
                q1_fit_exponential, T1_est, T1_err = None, None, None

            if self.plot_results:
                self.plot_results( I, Q, delay_times)

            if self.save_shots:
                raw_0 = t1.get_raw()  # I,Q data without normalizing to readout window, subtracting readout offset, or rotation/thresholding
                Ishots = raw_0[self.QubitIndex][:, :, 0, 0]
                Qshots = raw_0[self.QubitIndex][:, :, 0, 1]
                return T1_est, T1_err, I, Q, Ishots, Qshots, delay_times, q1_fit_exponential, self.config, measurement_timestamp

            else:
                return  T1_est, T1_err, I, Q, None, None, delay_times, q1_fit_exponential, self.config, measurement_timestamp

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

    def add_and_plot_active_reset_comparison(self, comparison_runs, label, I, Q, delay_times, T1_est, T1_err,
                                             fit, save_folder, QubitIndex=None, signal=None,
                                             filename_tag="active_reset_T1_comparison", ylim=None, verbose=False):
        if QubitIndex is None:
            QubitIndex = self.QubitIndex

        comparison_runs.append({
            "label": label,
            "I": np.asarray(I, dtype=float),
            "Q": np.asarray(Q, dtype=float),
            "delay_times": np.asarray(delay_times, dtype=float),
            "fit": None if fit is None else np.asarray(fit, dtype=float),
            "T1": T1_est,
            "T1_err": T1_err,
        })

        if len(comparison_runs) < 2:
            if verbose:
                print(f"Stored {label}; waiting for another T1 run before plotting.")
            return comparison_runs

        if signal is None or signal == "None":
            I0 = comparison_runs[0]["I"]
            Q0 = comparison_runs[0]["Q"]
            signal_to_plot = "I" if np.ptp(I0) >= np.ptp(Q0) else "Q"
        else:
            signal_to_plot = signal

        os.makedirs(save_folder, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))

        for run in comparison_runs:
            x = run["delay_times"]
            y = run[signal_to_plot]
            fit_curve = run["fit"]

            T1_value = run.get("T1")
            T1_error = run.get("T1_err")

            if T1_value is not None and np.isfinite(T1_value):
                if T1_error is not None and np.isfinite(T1_error):
                    curve_label = f'{run["label"]}: T1 = {T1_value:.2f} ± {T1_error:.2f} us'
                else:
                    curve_label = f'{run["label"]}: T1 = {T1_value:.2f} us'
            else:
                curve_label = run["label"]

            color = "C0" if comparison_runs.index(run) == 0 else "C2"
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, color=color, label=curve_label)
            if fit_curve is not None and fit_curve.shape == y.shape:
                ax.plot(x, fit_curve, "--", linewidth=2,color=color)

        ax.set_xlabel("Delay time (us)", fontsize=14)
        ax.set_ylabel(f"{signal_to_plot} amplitude (a.u.)", fontsize=14)
        ax.set_title(f"Q{QubitIndex + 1} T1 active-reset comparison", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.25)

        if ylim is not None:
            ax.set_ylim(ylim)

        fig.tight_layout()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(save_folder, f"Q{QubitIndex + 1}_{filename_tag}_{timestamp}.png")
        fig.savefig(file_name, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if verbose:
            print("Saved T1 active-reset comparison to:", file_name)

        return comparison_runs

    def exponential(self, x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def exponential(self, x, a, b, c, d):
        return a * np.exp(- (x - b) / c) + d

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

        d_guess = float(signal[
                            0])  # value at t=0. Based on the new function we are using (t1_model above), d is just the starting value
        a_guess = float(sig_max - sig_min)  # amplitude (can be +/-), this sign decides if the data rises or decays
        if a_guess == 0.0:  # if the signal is extremely flat or constant
            a_guess = float(np.ptp(signal) or 1.0)  # peak-to-peak amplitude. If that number is zero, uses 1 instead.

        span = max(float(t[-1] - t[0]), 1e-3)
        c_guess = max(span / 3.0, 1e-3)  # T1 ~ span/3, and we set a floor of 1e-3 to make sure T>0

        # ---------------- chi^2 function ----------------
        if y_errs is not None:
            # Handle zero or negative uncertainties
            if np.any(sigma > 0):  # Look at only the positive sigmas, and use their median.
                med_pos = np.median(sigma[sigma > 0])
                sigma = np.where(sigma <= 0, med_pos, sigma)  # replaces zeros, negative sigmas, NaNs
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

        m.errordef = Minuit.LEAST_SQUARES  # used for chi^2 or least squares method, which we are using

        # limits: we enforce c>0 (for a positive T1 value) and limit it to reasonable values relative to the time window
        T1_min = max(span / 50.0, 1e-3)
        T1_max = span * 20.0

        m.limits["c"] = (T1_min, T1_max)

        # run minimization: first simplex, then migrad
        # m.simplex() # if fitting is failing often, uncomment this, it can help
        m.migrad()

        # try one more time if we get invalid results
        if not m.valid:  # If not valid, we try again with opposite a
            m.values["a"] = -a_guess
            m.migrad()

        m.hesse()  # computes the covariance matrix after minimization

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

        T1_est = float(c_fit)  # out T1 result

        # compute fit and unsort back
        fit_sorted = t1_model(t, a_fit, c_fit,
                              d_fit)  # since we sorted the data at the beginning just in case it was out of order, the fit is based on sorted data
        inv_order = np.argsort(order)
        fit_curve = fit_sorted[
            inv_order]  # putting it back to the original order (should be sorted nonetheless, but this is done to be 100% consistent w original order)

        # For quality cuts (BIC score calc)
        n = len(t)
        k = 3  # a, c, d
        if y_errs is not None:
            # objective = chi^2
            bic_score = m.fval + k * np.log(n)
        else:
            # objective = RSS
            rss = m.fval
            bic_score = n * np.log(rss / n) + k * np.log(n)

        fit_info = {
            "t": t.copy(),
            "plot_sig": plot_sig,  # type (I or Q)
            "signal": signal.copy(),  # actual signal data
            "sigma": sigma.copy() if y_errs is not None else None,
            "a_fit": float(a_fit),
            "c_fit": float(c_fit),
            "d_fit": float(d_fit),
            "minimization_obj": float(m.fval),
            "bic_score": float(bic_score),
            "fit_valid": bool(m.valid),
            "n_points": int(n),
        }

        return fit_curve, T1_err, T1_est, fit_info

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

    def plot_results(self, I, Q, delay_times, config = None, fig_quality =100):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})

        # Calculate the middle of the plot area
        plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2


        if self.fit_data:
            #q1_fit_exponential, T1_err, T1_est, plot_sig = self.t1_fit(I, Q, delay_times)
            q1_fit_exponential, T1_err, T1_est, fit_info = self.t1_fit_iminuit(I, Q, delay_times)
            plot_sig = fit_info["plot_sig"]

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
                         f"T1 Q{self.QubitIndex + 1} " + f"T1={T1_est:.2f} +/- {T1_err:.2f} us" + f", {self.config['reps']}*{self.config['rounds']} avgs,",
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
        ax1.plot(delay_times, I, label="Gain (a.u.)", linewidth=2)
        ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        # ax1.axvline(freq_q, color='orange', linestyle='--', linewidth=2)

        # Q subplot
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

    def plot_results_active_reset(self, I, Q, delay_times, config = None, fig_quality =100):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})

        # Calculate the middle of the plot area
        plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2


        if self.fit_data:
            q1_fit_exponential, T1_err, T1_est, plot_sig = self.t1_fit(I, Q, delay_times)

            if 'I' in plot_sig:
                ax1.plot(delay_times, q1_fit_exponential, '-', color='red', linewidth=3, label="Fit")
            else:
                ax2.plot(delay_times, q1_fit_exponential, '-', color='red', linewidth=3, label="Fit")

            # Add title, centered on the plot area
            if config is not None:
                fig.text(plot_middle, 0.98,
                         f"Q{self.QubitIndex + 1} " + f"T1={T1_est:.2f} us" + f", {float(config['reps'])}*{float(config['rounds'])} avgs,",
                         fontsize=24, ha='center',
                         va='top')  # , pi gain %.2f" % float(config['pi_amp']) + f", {float(config['sigma']) * 1000} ns sigma
            else:
                fig.text(plot_middle, 0.98,
                         f"T1 Q{self.QubitIndex + 1}, T1 %.2f us" % T1_est + f", {self.config['reps']}*{self.config['rounds']} avgs,",
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
        ax1.plot(delay_times, I, label="Gain (a.u.)", linewidth=2)
        ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        # ax1.axvline(freq_q, color='orange', linestyle='--', linewidth=2)

        # Q subplot
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
            outerFolder_expt = os.path.join(self.outerFolder, "T1_ge_active_reset")
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt, f"Reset_R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')  # , facecolor='white'
        plt.close(fig)

    def plot_first_delay_reset_decision_shots(
            self,
            raw_q,
            decision_threshold,
            delay_times=None,
            delay_index=0,
            decision_readout_index=0,
            save_folder=None,
            filename_tag="T1_first_delay_reset_decision",
            show_plot=False,
            print_summary=True):
        """
        THIS ASSUMES A VERIFICATION BLOCK  IN THE ACTIVE RESET CODE.

        Plot the active-reset decision shots at one T1 delay point together
        with the exact raw-unit threshold passed to read_and_jump().

        Expected raw_q shape:
            (n_delay_points, reps, n_readouts, 2)

        Indices:
            raw_q[delay_index, :, decision_readout_index, 0] -> raw I shots
            raw_q[delay_index, :, decision_readout_index, 1] -> raw Q shots

        For the current active-reset sequence:
            readout index 0  -> first decision readout
            readout index 1  -> first verification readout
            readout index 2  -> second decision readout
            readout index 3  -> second verification readout
            ...
            readout index -1 -> final T1 readout

        The current FPGA decision is:
            I < threshold  -> ground-like, skip correction pi
            I >= threshold -> excited-like, apply correction pi
        """

        raw_q = np.asarray(raw_q)
        threshold_raw = int(decision_threshold)

        if raw_q.ndim != 4:
            raise ValueError(
                "Expected raw_q to have four dimensions "
                "(delay, repetition, readout, IQ), but got "
                f"shape {raw_q.shape}."
            )

        if raw_q.shape[-1] != 2:
            raise ValueError(
                f"Expected the final raw_q dimension to contain I and Q, "
                f"but got shape {raw_q.shape}."
            )

        n_delays, n_reps, n_readouts, _ = raw_q.shape

        if not 0 <= delay_index < n_delays:
            raise IndexError(
                f"delay_index={delay_index} is invalid. "
                f"Available delay indices are 0 through {n_delays - 1}."
            )

        # Allow Python-style negative readout indices.
        resolved_readout_index = decision_readout_index
        if resolved_readout_index < 0:
            resolved_readout_index += n_readouts

        if not 0 <= resolved_readout_index < n_readouts:
            raise IndexError(
                f"decision_readout_index={decision_readout_index} is invalid. "
                f"There are {n_readouts} stored readouts."
            )

        # These are the unnormalized integrated values returned by get_raw().
        I_decision = np.asarray(
            raw_q[delay_index, :, resolved_readout_index, 0],
            dtype=float
        ).ravel()

        Q_decision = np.asarray(
            raw_q[delay_index, :, resolved_readout_index, 1],
            dtype=float
        ).ravel()

        finite = np.isfinite(I_decision) & np.isfinite(Q_decision)
        I_decision = I_decision[finite]
        Q_decision = Q_decision[finite]

        if I_decision.size == 0:
            raise ValueError(
                "No finite decision shots were found at the requested indices."
            )

        # Match the FPGA condition exactly:
        # read_and_jump(... test='<') means I < threshold skips the pi pulse.
        ground_like = I_decision < threshold_raw
        correction_needed = ~ground_like

        n_ground_like = int(np.count_nonzero(ground_like))
        n_correction = int(np.count_nonzero(correction_needed))
        n_total = int(I_decision.size)

        ground_fraction = n_ground_like / n_total
        correction_fraction = n_correction / n_total

        if delay_times is not None:
            delay_times = np.asarray(delay_times, dtype=float)
            delay_value = float(delay_times[delay_index])
            delay_text = f"{delay_value:.4g} us"
        else:
            delay_value = None
            delay_text = f"index {delay_index}"

        if print_summary:
            print("\n--- Active-reset decision-shot diagnostic ---")
            print("raw_q shape:", raw_q.shape)
            print("T1 delay:", delay_text)
            print("Decision readout index:", decision_readout_index)
            print("Resolved readout index:", resolved_readout_index)
            print("Threshold passed to read_and_jump:", threshold_raw)
            print("Minimum raw I:", np.min(I_decision))
            print("Maximum raw I:", np.max(I_decision))
            print("Median raw I:", np.median(I_decision))
            print(
                f"Ground-like, I < threshold: "
                f"{n_ground_like}/{n_total} = {ground_fraction:.4f}"
            )
            print(
                f"Correction needed, I >= threshold: "
                f"{n_correction}/{n_total} = {correction_fraction:.4f}"
            )

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        # ---------------- Raw-I histogram ----------------
        ax_hist = axes[0]

        ax_hist.hist(
            I_decision,
            bins=75,
            alpha=0.75,
            label="Decision-readout shots"
        )

        ax_hist.axvline(
            threshold_raw,
            linestyle="--",
            linewidth=2.5,
            label=f"FPGA threshold = {threshold_raw}"
        )

        ax_hist.set_xlabel("Raw integrated I")
        ax_hist.set_ylabel("Shot count")
        ax_hist.set_title("Raw-I decision distribution")
        ax_hist.legend()
        ax_hist.grid(alpha=0.25)

        # ---------------- Raw IQ scatter ----------------
        ax_iq = axes[1]

        ax_iq.scatter(
            I_decision[ground_like],
            Q_decision[ground_like],
            s=10,
            alpha=0.45,
            label=f"Skip pi: I < {threshold_raw}"
        )

        ax_iq.scatter(
            I_decision[correction_needed],
            Q_decision[correction_needed],
            s=10,
            alpha=0.45,
            label=f"Apply pi: I >= {threshold_raw}"
        )

        ax_iq.axvline(
            threshold_raw,
            linestyle="--",
            linewidth=2.5,
            label="Decision boundary"
        )

        ax_iq.set_xlabel("Raw integrated I")
        ax_iq.set_ylabel("Raw integrated Q")
        ax_iq.set_title("Decision shots in the raw IQ frame")
        ax_iq.legend(fontsize=9)
        ax_iq.grid(alpha=0.25)

        fig.suptitle(
            f"Q{self.QubitIndex + 1} active-reset decision at "
            f"T1 delay {delay_text}\n"
            f"Correction requested for {100 * correction_fraction:.1f}% of shots",
            fontsize=14
        )

        fig.tight_layout(rect=[0, 0, 1, 0.91])

        file_name = None

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(
                save_folder,
                f"Q{self.QubitIndex + 1}_{filename_tag}_{timestamp}.png"
            )

            fig.savefig(file_name, dpi=150, bbox_inches="tight")

            if print_summary:
                print("Saved decision diagnostic to:", file_name)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return {
            "I_decision_raw": I_decision,
            "Q_decision_raw": Q_decision,
            "threshold_raw": threshold_raw,
            "delay_index": delay_index,
            "delay_value_us": delay_value,
            "decision_readout_index": resolved_readout_index,
            "ground_like_mask": ground_like,
            "correction_needed_mask": correction_needed,
            "ground_like_fraction": ground_fraction,
            "correction_needed_fraction": correction_fraction,
            "file_name": file_name,
        }


class T1Program_active_reset(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ge'], mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'], mixer_freq=cfg['mixer_freq'])

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="res_pulse", style="const",
                       length=cfg["res_length"], mask=cfg["list_of_all_qubits"])

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'],
                       length=cfg['sigma'] * 4, even_length=False)

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", style="arb", envelope="ramp",
                       freq=cfg['qubit_freq_ge'], phase=cfg['qubit_phase'], gain=cfg['pi_amp'])

        self.add_loop("waitloop", cfg["steps"])

    def _active_reset_block(self, cfg, label_addition=""):
        n_resets = cfg.get("n_resets", 3)
        ro_ch_this = cfg["ro_ch"][0]
        res_length_cycles = self.soccfg.us2cycles(us=cfg["res_length"], ro_ch=ro_ch_this)
        threshold_raw = int(round(cfg["threshold"] * res_length_cycles))
        delay1_act_reset = cfg.get("delay1_act_reset", 6.0)
        delay2_act_reset = cfg.get("delay2_act_reset", 6.0)

        for i in range(n_resets):
            skip_label = f"skip_reset_{label_addition}_{i}"
            self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
            self.trigger(ros=cfg["ro_ch"], pins=[0], t=cfg["trig_time"])
            self.wait_auto(0.0, gens=True, ros=True)
            self.resync()
            self.delay_auto(t=0.0)
            self.read_and_jump(ro_ch=ro_ch_this, component="I", threshold=threshold_raw, test="<", label=skip_label)
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)
            self.label(skip_label)
            self.delay_auto(t=delay1_act_reset)

        self.delay_auto(t=delay2_act_reset)

    def _body(self, cfg):
        # Reset the unknown state left from the previous repetition.
        self._active_reset_block(cfg, label_addition="pre")

        # Standard T1 sequence.
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)
        self.delay_auto(cfg["wait_time"] + 0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=cfg["ro_ch"], pins=[0], t=cfg["trig_time"])

