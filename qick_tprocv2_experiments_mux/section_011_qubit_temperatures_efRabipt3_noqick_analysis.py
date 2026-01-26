import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import datetime
# from build_task import *
# from build_state import *
# from build_state_noqick import *
# from expt_config import *
from expt_config import *
import copy
from iminuit import Minuit
# import visdom
from scipy.signal import argrelextrema
import os

class Temps_EFAmpRabiExperiment:
    def __init__(self, QubitIndex, number_of_qubits, list_of_all_qubits,  outerFolder, round_num, signal, save_figs, experiment = None, live_plot = None,
                 increase_qubit_reps = False, qubit_to_increase_reps_for = None, multiply_qubit_reps_by = 0):
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
        # if experiment is not None:
        #     self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
        #     self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
        #     self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
        #     if increase_qubit_reps:
        #             if self.QubitIndex==qubit_to_increase_reps_for:
        #                 print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
        #                 self.config["reps"] *= multiply_qubit_reps_by
        #     print(f'Q {self.QubitIndex + 1} Round {self.round_num} EF Rabi configuration: ', self.config)

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

        def chi2(a, b, c,d):  # we don't have sigmas available so this is technically just the sum of squared errors, not chi2, which is fine.
            model = self.cosine(x, a, b, c, d)
            return np.sum((y - model) ** 2)

        m = Minuit(chi2, a=p0[0], b=p0[1], c=p0[2], d=p0[3])
        m.errordef = Minuit.LEAST_SQUARES  # least-squares / chi^2 objective

        # --- NEW: optionally fix b and/or c ---
        # We do this, for example, when we want to fit Q using the same b and c params we used for I
        if fix_b is not None:
            m.values["b"] = float(fix_b)
            m.fixed["b"] = True

        if fix_c is not None:
            m.values["c"] = float(fix_c)
            m.fixed["c"] = True
        # -------------------------------------

        m.migrad()
        m.hesse()

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
            p = len(names)
            ndof = N - p
            if ndof > 0 and np.isfinite(m.fval):
                scale = m.fval / ndof
                pcov = pcov * scale

        return popt, pcov

    def plot_results(self, I, Q, gains, config = None, fig_quality = 200, use_iminuit_instead = False, filename_ext = ""):
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            plt.rcParams.update({'font.size': 18})

            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

            q1_a_guess_I = (np.max(I) - np.min(I)) / 2
            q1_d_guess_I = np.mean(I)
            q1_a_guess_Q = (np.max(Q) - np.min(Q)) / 2
            q1_d_guess_Q = np.mean(Q)
            q1_b_guess = 1 / gains[-1]
            q1_c_guess = 0 # np.pi/2 # you can try 0 too

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
                else: # NOTE; I HAVE NOT IMPLEMENTED SHARED USE OF b or c FOR CURVEFIT
                    popt_I, pcov_I = curve_fit(self.cosine, gains, I, maxfev=100000, p0=q1_guess_I)
                fit_cosine_I = self.cosine(gains, *popt_I)

                # Extract shared b,c from I fit
                b_shared = popt_I[1]  # oscillation frequency
                # c_shared = popt_I[2]  # phase offset

                if use_iminuit_instead:
                    # Fit Q but lock b,c to the I-fit values
                    popt_Q, pcov_Q = self.fit_cosine_iminuit(gains, Q, q1_guess_Q, fix_b=b_shared)
                else:  # NOTE; I HAVE NOT IMPLEMENTED SHARED USE OF b or c FOR CURVEFIT
                    popt_Q, pcov_Q = curve_fit(self.cosine, gains, Q, maxfev=100000, p0=q1_guess_Q)
                fit_cosine_Q = self.cosine(gains, *popt_Q)

            else:  # fit_first == 'Q'
                if use_iminuit_instead:
                    popt_Q, pcov_Q = self.fit_cosine_iminuit(gains, Q, q1_guess_Q)
                else: # NOTE; I HAVE NOT IMPLEMENTED SHARED USE OF b or c FOR CURVEFIT
                    popt_Q, pcov_Q = curve_fit(self.cosine, gains, Q, maxfev=100000, p0=q1_guess_Q)
                fit_cosine_Q = self.cosine(gains, *popt_Q)

                # Extract shared b,c from Q fit
                b_shared = popt_Q[1]  # oscillation frequency
                # c_shared = popt_Q[2]  # phase offset

                if use_iminuit_instead:
                    # Fit I but lock b,c to the Q-fit values
                    popt_I, pcov_I = self.fit_cosine_iminuit(gains, I, q1_guess_I, fix_b=b_shared)
                else:  # NOTE; I HAVE NOT IMPLEMENTED SHARED USE OF b AND c FOR CURVEFIT
                    popt_I, pcov_I = curve_fit(self.cosine, gains, I, maxfev=100000, p0=q1_guess_I)
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

            #---------------------------------------------------------------------------------
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
            magnitude_fit = self.cosine(gains, *mag_popt)

            # ------------------------------- compute amplitude curve from the I and Q FITS instead of the data --------------------------------------
            amp_fit_IQ = np.sqrt(fit_cosine_I ** 2 + fit_cosine_Q ** 2) # this constructs the point-by-point magnitude of the fitted IQ vector
            A_I = popt_I[0] # I-curve amplitude
            A_Q = popt_Q[0] # Q-curve amplitude
            A_amp_IQ = np.sqrt(A_I ** 2 + A_Q ** 2) # Combined IQ Amplitude from the amplitudes of the I and Q fits
            sigma_A_I = np.sqrt(np.diag(pcov_I))[0] # I-curve amplitude error
            sigma_A_Q = np.sqrt(np.diag(pcov_Q))[0] # Q-curve amplitude error
            A_amp_IQ_err = np.sqrt((A_I / A_amp_IQ) ** 2 * sigma_A_I ** 2 +(A_Q / A_amp_IQ) ** 2 * sigma_A_Q ** 2) # Combined IQ Amplitude err
            # ------------------------------------------------------------------------------------------------------------------------

            ax1.legend([f"A={A_I:.4f}+/-{sigma_A_I:.4f}"], loc='best')
            ax2.legend([f"A={A_Q:.4f}+/-{sigma_A_Q:.4f}"], loc='best')

            # --- Extract the amplitude parameter A directly: the amplitude of the cosine fit to the magnitude data ---
            # THIS IS THE WAY WE PREVIOUSLY DID IT WHICH WAS WRONG, here for comparison
            # THIS DEF OF AMPLITUDE MEASURES DISTANCE FROM THE ORIGIN, NOT THE AMPLITUDE OF THE RABI OSCILLATION THAT WE NEED for QTEMPS
            # A_amplitude = mag_popt[0]
            # amp_perr = np.sqrt(np.diag(mag_pcov))
            # A_amplitude_err = amp_perr[0]

            if config is not None:
                fig.text(plot_middle, 0.98,
                         f"e-f RPM Q{self.QubitIndex + 1}: "  + f", Pg: {config['reps']}*{config['rounds']} avgs, Pe: {config['reps2']}*{config['rounds']} avgs, A=sqrt(A_I**2 + A_Q**2)={A_amp_IQ:.4f}+/-{A_amp_IQ_err:.4f}",
                         fontsize=18, ha='center', va='top') #f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back
            else:
                fig.text(plot_middle, 0.98,
                         f"e-f RPM Q{self.QubitIndex + 1}: A=sqrt(A_I**2 + A_Q**2)={A_amp_IQ:.4f}+/-{A_amp_IQ_err:.4f}",
                         fontsize=18, ha='center', va='top')

            # --- Plot amplitude (magnitude) data and its cosine fit on the third subplot ---
            ax3.plot(gains, magnitude_data, '-', label="Amp Data", linewidth=2)
            ax3.plot(gains, magnitude_fit, '-', color='green', linewidth=3, label=f"Fit to Magnitude Data")

            # Test: curve made from the fits of the I + Q data
            ax3.plot(gains, amp_fit_IQ, '-', color='orange', linewidth=3, label=f"sqrt(I_fit**2 + Q_fit**2)") # for a test

            ax3.set_xlabel("Gain (a.u.)", fontsize=20)
            ax3.set_ylabel("Amplitude (a.u.)", fontsize=20)
            ax3.tick_params(axis='both', which='major', labelsize=16)
            ax3.legend(loc='best')

            #------------------------------------------------------------------------------------------------
            if self.save_figs:
                today_date = datetime.datetime.now().strftime("%Y-%m-%d")
                dated_folder_name = f"made_on_{today_date}"
                # outerFolder_expt = os.path.join(self.outerFolder, "q_temperatures", dated_folder_name)
                outerFolder_expt = os.path.join(self.outerFolder, dated_folder_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y%m%d%H%M%S")
                file_name = os.path.join(outerFolder_expt, f"{filename_ext}Q{self.QubitIndex + 1}_" + f"Qtemps_RPM_" + f"{formatted_datetime}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
                # print('Plots saved to this folder:',outerFolder_expt)
            plt.close(fig)

            fit_params = {"amp_fit_IQ": amp_fit_IQ,
                        "I_fit": fit_cosine_I,
                        "Q_fit": fit_cosine_Q,
                        "popt_I": popt_I,
                        "pcov_I": pcov_I,
                        "popt_Q": popt_Q,
                        "pcov_Q": pcov_Q}

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
        popt_I, pcov_I = curve_fit(self.cosine, gains, I, maxfev=100000, p0=q1_guess_I)
        fit_cosine_I = self.cosine(gains, *popt_I)

        q1_guess_Q = [q1_a_guess_Q, q1_b_guess, q1_c_guess, q1_d_guess_Q]
        popt_Q, pcov_Q = curve_fit(self.cosine, gains, Q, maxfev=100000, p0=q1_guess_Q)
        fit_cosine_Q = self.cosine(gains, *popt_Q)

        first_three_avg_I = np.mean(fit_cosine_I[:3])
        last_three_avg_I = np.mean(fit_cosine_I[-3:])
        first_three_avg_Q = np.mean(fit_cosine_Q[:3])
        last_three_avg_Q = np.mean(fit_cosine_Q[-3:])

        best_signal_fit = None
        pi_amp = None
        if 'Q' in self.signal:
            best_signal_fit = fit_cosine_Q
            # figure out if you should take the min or the max value of the fit to say where pi_amp should be
            if last_three_avg_Q > first_three_avg_Q:
                pi_amp = gains[np.argmax(best_signal_fit)]
            else:
                pi_amp = gains[np.argmin(best_signal_fit)]
        if 'I' in self.signal:
            best_signal_fit = fit_cosine_I
            # figure out if you should take the min or the max value of the fit to say where pi_amp should be
            if last_three_avg_I > first_three_avg_I:
                pi_amp = gains[np.argmax(best_signal_fit)]
            else:
                pi_amp = gains[np.argmin(best_signal_fit)]
        if 'None' in self.signal:
            # choose the best signal depending on which has a larger magnitude
            if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
                best_signal_fit = fit_cosine_Q
                # figure out if you should take the min or the max value of the fit to say where pi_amp should be
                if last_three_avg_Q > first_three_avg_Q:
                    pi_amp = gains[np.argmax(best_signal_fit)]
                else:
                    pi_amp = gains[np.argmin(best_signal_fit)]
            else:
                best_signal_fit = fit_cosine_I
                # figure out if you should take the min or the max value of the fit to say where pi_amp should be
                if last_three_avg_I > first_three_avg_I:
                    pi_amp = gains[np.argmax(best_signal_fit)]
                else:
                    pi_amp = gains[np.argmin(best_signal_fit)]
            tot_amp = [np.sqrt((ifit)**2 + (qfit)**2) for ifit,qfit in zip(fit_cosine_I, fit_cosine_Q)]
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
