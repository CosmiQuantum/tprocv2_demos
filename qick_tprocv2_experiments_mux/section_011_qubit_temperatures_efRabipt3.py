import matplotlib.pyplot as plt
import numpy as np
import visdom
from scipy.optimize import curve_fit
import datetime
from build_task import *
from build_state import *
# from expt_config import *
from expt_config import *
import time
from iminuit import Minuit
import copy
# import visdom
from scipy.signal import argrelextrema

class Temps_EFAmpRabiExperiment:
    def __init__(self, QubitIndex, number_of_qubits, list_of_all_qubits,  outerFolder, round_num, signal, save_figs, experiment = None, live_plot = None,
                 increase_qubit_reps = False, increase_qubit_reps_to = 400, increase_qubit_reps2 = False,
                 increase_qubit_reps2_to = 1000, unmasking_resgain = False):

        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.expt_name = "power_rabi_ef"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.live_plot = live_plot
        self.signal = signal
        self.save_figs = save_figs
        self.experiment = experiment
        self.list_of_all_qubits = list_of_all_qubits

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            if increase_qubit_reps:
                    print(f"Increasing reps for {self.QubitIndex + 1} to {increase_qubit_reps_to}")
                    self.config["reps"] = increase_qubit_reps_to
            if increase_qubit_reps2:
                    print(f"Increasing reps for {self.QubitIndex + 1} to {increase_qubit_reps2_to}.")
                    self.config["reps2"] = increase_qubit_reps2_to
            print(f'Q {self.QubitIndex + 1} Round {self.round_num} EF Rabi configuration: ', self.config)


    def run(self, soccfg, soc, use_iminuit_instead = True):
        print(self.config)
        # --- Pe sequence ---
        amp_rabi1 = AmplitudeRabiProgram1(soccfg, reps=self.config['reps2'], final_delay=self.config['relax_delay'], cfg=self.config)
        if self.live_plot:
            I1, Q1, gains1 = self.live_plotting(amp_rabi1, soc)
        else:
            iq_list1 = amp_rabi1.acquire(soc, soft_avgs=self.config["rounds"], progress=True)
            I1 = iq_list1[self.QubitIndex][0, :, 0]
            Q1 = iq_list1[self.QubitIndex][0, :, 1]
            gains1 = amp_rabi1.get_pulse_param('qubit_pulse', "gain", as_array=True)
        A_amp_IQ1, A_amp_IQ_err1, fit_params1 = self.plot_results( I1, Q1, gains1, config = self.config, use_iminuit_instead = use_iminuit_instead)
        fit_IQ_1 = fit_params1["fit_IQ"]

        # --- Pg sequence ---
        amp_rabi2 = AmplitudeRabiProgram2(soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'], cfg=self.config)
        if self.live_plot:
            I2, Q2, gains2 = self.live_plotting(amp_rabi2, soc)
        else:
            iq_list2 = amp_rabi2.acquire(soc, soft_avgs=self.config["rounds"], progress=True)
            I2 = iq_list2[self.QubitIndex][0, :, 0]
            Q2 = iq_list2[self.QubitIndex][0, :, 1]
            gains2 = amp_rabi2.get_pulse_param('qubit_pulse', "gain", as_array=True)

        A_amp_IQ2, A_amp_IQ_err2, fit_params2 = self.plot_results(I2, Q2, gains2, config=self.config, use_iminuit_instead = use_iminuit_instead)
        fit_IQ_2 = fit_params2["fit_IQ"]

        measurement_timestamp = (time.mktime(datetime.datetime.now().timetuple()))

        return I1, Q1, gains1, I2, Q2, gains2, A_amp_IQ1, A_amp_IQ2, A_amp_IQ_err1, A_amp_IQ_err2, fit_IQ_1, fit_IQ_2, self.config, measurement_timestamp


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

    def canonicalize_cos_params(self, popt):
        """
        Enforce A >= 0 and confine phase to [-pi, pi].
        Model: y = A*cos(b*x + c) + d
        Identity: A*cos(theta) == (-A)*cos(theta + pi)
        """
        popt = np.array(popt, dtype=float).copy()
        if popt[0] < 0:  # if A < 0
            popt[0] *= -1  # flip amplitude
            popt[2] += np.pi  # shift phase by pi
        # maps phase into [-pi, pi) to keep it stable / comparable across scans and ensure unique sol
        popt[2] = (popt[2] + np.pi) % (2 * np.pi) - np.pi
        return popt

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

    def plot_results(self, I, Q, gains, config=None, fig_quality=200, use_iminuit_instead=True, filename_ext="",
                     rotate_using_ssf=False, ssf_angle=None):
        """
        iminuit: default
        Figures out which signal is best (I or Q), fits that one first, then uses the found oscillation frequency in the first
        fit to fit the other signal component. Optional: can fix the phase offset too.

        Curve fit:
        Same thing but the option to fix the oscillation frequency is not included. Neither is the option to fix the phase offset.

        Calculates the rabi amplitude as sqrt(A_I**2 + A_Q**2)
        """
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            plt.rcParams.update({'font.size': 18})

            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

            # Initial seeds (guesses)
            q1_a_guess_I = (np.max(I) - np.min(I)) / 2
            q1_d_guess_I = np.mean(I)
            q1_a_guess_Q = (np.max(Q) - np.min(Q)) / 2
            q1_d_guess_Q = np.mean(Q)

            q1_b_guess = 1 / gains[-1]
            q1_c_guess = 0  # another option is np.pi/2

            # Initial guesses for I curve
            q1_guess_I = [q1_a_guess_I, q1_b_guess, q1_c_guess, q1_d_guess_I]

            # Initial guesses for Q curve
            q1_guess_Q = [q1_a_guess_Q, q1_b_guess, q1_c_guess, q1_d_guess_Q]

            # ---------------- Minimal tweak: decide which quadrature to fit first (best signal) ----------------
            # Use the same metric we already rely on later: how much the signal changes from start to end.
            span_I = abs(np.mean(I[-3:]) - np.mean(I[:3]))
            span_Q = abs(np.mean(Q[-3:]) - np.mean(Q[:3]))

            # If user forces I or Q, respect that; if 'None', choose the larger-span quadrature.
            if 'I' in self.signal:
                fit_first = 'I'
            elif 'Q' in self.signal:
                fit_first = 'Q'
            elif 'None' in self.signal:
                fit_first = 'Q' if span_Q > span_I else 'I'
            else:
                print('Invalid signal passed, please do I Q or None')
                fit_first = 'I'
            # ---------------------------------------------------------------------------------------------------

            if fit_first == 'I':
                if use_iminuit_instead:
                    popt_I, pcov_I = self.fit_cosine_iminuit(gains, I, q1_guess_I)
                else:  # NOTE; I HAVE NOT IMPLEMENTED SHARED USE OF b or c FOR CURVEFIT
                    popt_I, pcov_I = curve_fit(self.cosine, gains, I, maxfev=100000, p0=q1_guess_I)

                popt_I = self.canonicalize_cos_params(popt_I)  # new
                fit_cosine_I = self.cosine(gains, *popt_I)

                # Extract shared b,c from I fit
                b_shared = popt_I[1]  # oscillation frequency
                # c_shared = popt_I[2]  # phase

                if use_iminuit_instead:
                    # Fit Q but lock b,c to the I-fit values
                    popt_Q, pcov_Q = self.fit_cosine_iminuit(gains, Q, q1_guess_Q, fix_b=b_shared)
                else:  # NOTE; I HAVE NOT IMPLEMENTED SHARED USE OF b or c FOR CURVEFIT
                    popt_Q, pcov_Q = curve_fit(self.cosine, gains, Q, maxfev=100000, p0=q1_guess_Q)

                popt_Q = self.canonicalize_cos_params(popt_Q)  # new
                fit_cosine_Q = self.cosine(gains, *popt_Q)

            else:  # fit_first == 'Q'
                if use_iminuit_instead:
                    popt_Q, pcov_Q = self.fit_cosine_iminuit(gains, Q, q1_guess_Q)
                else:  # NOTE; I HAVE NOT IMPLEMENTED SHARED USE OF b or c FOR CURVEFIT
                    popt_Q, pcov_Q = curve_fit(self.cosine, gains, Q, maxfev=100000, p0=q1_guess_Q)

                popt_Q = self.canonicalize_cos_params(popt_Q)  # new
                fit_cosine_Q = self.cosine(gains, *popt_Q)

                # Extract shared b,c from Q fit
                b_shared = popt_Q[1]  # oscillation frequency
                # c_shared = popt_Q[2]  # phase

                if use_iminuit_instead:
                    # Fit I but lock b,c to the Q-fit values
                    popt_I, pcov_I = self.fit_cosine_iminuit(gains, I, q1_guess_I, fix_b=b_shared)
                else:  # NOTE; I HAVE NOT IMPLEMENTED SHARED USE OF b AND c FOR CURVEFIT
                    popt_I, pcov_I = curve_fit(self.cosine, gains, I, maxfev=100000, p0=q1_guess_I)

                popt_I = self.canonicalize_cos_params(popt_I)  # new
                fit_cosine_I = self.cosine(gains, *popt_I)

            ax2.plot(gains, fit_cosine_Q, '-', color='red', linewidth=3, label="Fit")
            ax1.plot(gains, fit_cosine_I, '-', color='red', linewidth=3, label="Fit")

            # print(len(gains))
            ax1.plot(gains, I, label="Gain (a.u.)", linewidth=2)
            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)

            ax2.plot(gains, Q, label="Q", linewidth=2)
            ax2.set_xlabel("Gain (a.u.)", fontsize=20)
            ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
            ax2.tick_params(axis='both', which='major', labelsize=16)

            # ---------------------------------------------------------------------------------
            # --- Compute magnitude data from I and Q ---
            magnitude_data = np.sqrt(np.array(I) ** 2 + np.array(Q) ** 2)

            # --- Fit the amplitude data with the cosine function ---
            # Define initial guesses based on the magnitude_data characteristics.
            a_guess_mag = (np.max(magnitude_data) - np.min(magnitude_data)) / 2
            d_guess_mag = np.mean(magnitude_data)
            b_guess_mag = 1 / gains[-1]
            c_guess_mag = 0

            mag_guess = [a_guess_mag, b_guess_mag, c_guess_mag, d_guess_mag]
            if use_iminuit_instead:
                mag_popt, mag_pcov = self.fit_cosine_iminuit(gains, magnitude_data, mag_guess)
            else:
                mag_popt, mag_pcov = curve_fit(self.cosine, gains, magnitude_data, maxfev=100000, p0=mag_guess)

            mag_popt = self.canonicalize_cos_params(mag_popt)  # New
            magnitude_fit = self.cosine(gains, *mag_popt)

            # Useful for tests:
            # this gives the distance from the origin of the fitted IQ point at each gain
            # it is derived from the I and Q fits, but it is NOT the same as our final returned amplitude
            # it is more like a magnitude curve constructed from the fits rather than from the raw data
            fit_IQ = np.sqrt(fit_cosine_I ** 2 + fit_cosine_Q ** 2)  # point-by-point magnitude of the fitted IQ vector

            # ------------------------------- compute amplitude curve from the I and Q FITS --------------------------------------
            A_I = popt_I[0]  # I-curve amplitude
            A_Q = popt_Q[0]  # Q-curve amplitude
            A_amp_IQ = np.sqrt(A_I ** 2 + A_Q ** 2)  # Combined IQ Amplitude from the amplitudes of the I and Q fits
            sigma_A_I = np.sqrt(np.diag(pcov_I))[0]  # I-curve amplitude error
            sigma_A_Q = np.sqrt(np.diag(pcov_Q))[0]  # Q-curve amplitude error
            A_amp_IQ_err = np.sqrt((A_I / A_amp_IQ) ** 2 * sigma_A_I ** 2 + (A_Q / A_amp_IQ) ** 2 * sigma_A_Q ** 2)  # Combined IQ Amplitude err, no covariance term
            # ------------------------------------------------------------------------------------------------------------------------

            ax1.legend([f"A={A_I:.4f}+/-{sigma_A_I:.4f}"], loc='best')
            ax2.legend([f"A={A_Q:.4f}+/-{sigma_A_Q:.4f}"], loc='best')

            # --- Extract the amplitude parameter A directly: the amplitude of the cosine fit to the magnitude data ---
            # THIS IS THE WAY WE PREVIOUSLY DID IT WHICH WAS WRONG, here for comparison
            # THIS DEF OF AMPLITUDE MEASURES DISTANCE FROM THE ORIGIN, NOT THE AMPLITUDE OF THE RABI OSCILLATION THAT WE NEED for QTEMPS
            # A_amplitude = mag_popt[0]
            # amp_perr = np.sqrt(np.diag(mag_pcov))
            # A_amplitude_err = amp_perr[0]

            ###################### Geerlings-style TEST: rotate+project IQ onto a fixed axis #########################3
            if rotate_using_ssf and ssf_angle is not None:
                # Choose a single projection angle and build S = I cos(theta) + Q sin(theta)
                # This makes the cosine amplitude of S equal to sqrt(A_I^2 + A_Q^2) if the phases are consistent.

                # if you don't have an ssf angle you could maybe use this:
                # alpha = np.arctan2(A_Q, A_I)  # angle of the oscillation vector in IQ from the data

                # Build the combined 1D signal from RAW data
                S_data = np.cos(ssf_angle) * np.asarray(I) + np.sin(ssf_angle) * np.asarray(Q)

                # Fit S_data to the SAME cosine model we already use
                a_guess_S = (np.max(S_data) - np.min(S_data)) / 2
                d_guess_S = np.mean(S_data)
                b_guess_S = b_shared if 'b_shared' in locals() else (1 / gains[-1])  # reuse shared b if available
                c_guess_S = 0.0

                S_guess = [a_guess_S, b_guess_S, c_guess_S, d_guess_S]  # array of initial guesses to feed iminuit

                if use_iminuit_instead:
                    popt_S, pcov_S = self.fit_cosine_iminuit(gains, S_data, S_guess, fix_b=b_guess_S)
                else:
                    popt_S, pcov_S = curve_fit(self.cosine, gains, S_data, maxfev=100000, p0=S_guess)

                popt_S = self.canonicalize_cos_params(popt_S)
                fit_cosine_S = self.cosine(gains, *popt_S)

                A_S = popt_S[0]
                sigma_A_S = np.sqrt(np.diag(pcov_S))[0] if pcov_S is not None else np.nan

                # Plot it as an extra "Geerlings-style" curve (optional)
                ax3.plot(gains, S_data, '-', linewidth=2, label=f"S = Icos(theta)+Qsin(theta) (theta={ssf_angle:.3f})")
                ax3.plot(gains, fit_cosine_S, '-', linewidth=3, label=f"Fit to S, A_S={A_S:.4f}")

                # print(f"[Geerlings-style] alpha={alpha:.6f} rad, A_S={A_S:.6f} +/- {sigma_A_S:.6f}")
            ############################################################################################
            else:
                # --- Plot amplitude (magnitude) data and its cosine fit on the third subplot ---
                ax3.plot(gains, magnitude_data, '-', label="Magnitude Data", linewidth=2)
                # ax3.plot(gains, magnitude_fit, '-', color='green', linewidth=3, label=f"Fit to Magnitude Data")

                # curve made from the fits of the I + Q data
                ax3.plot(gains, fit_IQ, '-', color='orange', linewidth=3, label=f"sqrt(I_fit**2 + Q_fit**2)")

            if config is not None:
                fig.text(plot_middle, 0.90,
                         f"e-f RPM Q{self.QubitIndex + 1}: " + f", Pg: {config['reps']}*{config['rounds']} avgs, Pe: {config['reps2']}*{config['rounds']} avgs, A=sqrt(A_I**2 + A_Q**2)={A_amp_IQ:.4f}+/-{A_amp_IQ_err:.4f}",
                         fontsize=18, ha='center',
                         va='top')  # f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back
            else:
                fig.text(plot_middle, 0.90,
                         f"e-f RPM Q{self.QubitIndex + 1}: A=sqrt(A_I**2 + A_Q**2)={A_amp_IQ:.4f}+/-{A_amp_IQ_err:.4f}",
                         fontsize=18, ha='center', va='top')

            ax3.set_xlabel("Gain (a.u.)", fontsize=20)
            if rotate_using_ssf and ssf_angle is not None:
                ax3.set_ylabel("S(theta) (a.u.)", fontsize=20)
            else:
                ax3.set_ylabel("Magnitude (a.u.)", fontsize=20)
            ax3.tick_params(axis='both', which='major', labelsize=16)
            ax3.legend(loc='best')
            # ------------------------------------------------------------------------------
            if self.save_figs:
                # today_date = datetime.datetime.now().strftime("%Y-%m-%d")
                # dated_folder_name = f"made_on_{today_date}"
                # outerFolder_expt = os.path.join(self.outerFolder, dated_folder_name)
                # outerFolder_expt = os.path.join(self.outerFolder, "q_temperatures", dated_folder_name)
                outerFolder_expt = os.path.join(self.outerFolder, "q_temperatures")
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y%m%d%H%M%S")
                file_name = os.path.join(outerFolder_expt,f"{filename_ext}Q{self.QubitIndex + 1}_" + f"Qtemps_RPM_" + f"{formatted_datetime}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
                # print('Plots saved to this folder:',outerFolder_expt)
            plt.close(fig)

            fit_params = {"fit_IQ": fit_IQ,
                          "I_fit": fit_cosine_I,
                          "Q_fit": fit_cosine_Q,
                          "popt_I": popt_I,
                          "pcov_I": pcov_I,
                          "popt_Q": popt_Q,
                          "pcov_Q": pcov_Q,
                          "A_I": A_I,
                          "sigma_A_I": sigma_A_I,
                          "A_Q": A_Q,
                          "sigma_A_Q": sigma_A_Q}

            if rotate_using_ssf and ssf_angle is not None:
                return A_S, sigma_A_S, fit_params
            else:
                return A_amp_IQ, A_amp_IQ_err, fit_params

        except Exception as e:
            print("Error fitting cosine:", e)
            # Return None if the fit didn't work
            return None, None, None

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


class AmplitudeRabiProgram1(AveragerProgramV2):
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

    def _body(self, cfg): #this gives A_e
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # e-f pulse
        self.delay_auto(t=0.0, tag='waiting')  # wait

        self.pulse(ch=self.cfg["qubit_ch"], name="pi_ge", t=0)  # play g-e pi pulse
        self.delay_auto(t=0.0, tag='waiting after pi')  # Wait til ge pi pulse is done before proceeding

        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # probe pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class AmplitudeRabiProgram2(AveragerProgramV2):
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

        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma_ef'], length=cfg['sigma_ef'] * 4,
                       even_length=False)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ef'],
                       )

        self.add_loop("gainloop", cfg["steps"])

    def _body(self, cfg): # this gives A_g
        self.pulse(ch=self.cfg["qubit_ch"], name="pi_ge", t=0)  # play g-e pi pulse
        self.delay_auto(t=0.0, tag='waiting after pi')  # Wait til g-e pi pulse is done before proceeding

        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # e-f pulse
        self.delay_auto(t=0.0, tag='waiting')  # wait

        self.pulse(ch=self.cfg["qubit_ch"], name="pi_ge", t=0)  # play g-e pi pulse
        self.delay_auto(t=0.0, tag='2nd waiting after pi')  # Wait til g-e pi pulse is done before proceeding

        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # probe pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])