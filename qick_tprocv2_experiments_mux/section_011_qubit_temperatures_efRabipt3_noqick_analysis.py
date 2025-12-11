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

    def fit_cosine_iminuit(self, x, y, p0):
        """
        Iminuit-based cosine fit that mirrors scipy.curve_fit's output:
        returns popt and an approximate covariance matrix pcov.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        def chi2(a, b, c, d): # we don't have sigmas available so this is technically just the least squares part, which is fine.
            model = self.cosine(x, a, b, c, d)
            return np.sum((y - model) ** 2)

        m = Minuit(chi2, a=p0[0], b=p0[1], c=p0[2], d=p0[3])
        m.errordef = Minuit.LEAST_SQUARES  # least-squares / chi^2 objective

        m.migrad()
        m.hesse()

        # Extract best-fit parameter values into a NumPy array
        popt = np.array([m.values["a"], m.values["b"], m.values["c"], m.values["d"]])

        # Convert Minuit’s covariance object to a regular NumPy matrix
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

        return popt, pcov

    def plot_results(self, I, Q, gains, config = None, fig_quality = 200, use_iminuit_instead = False):
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            plt.rcParams.update({'font.size': 18})

            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

            q1_a_guess_I = (np.max(I) - np.min(I)) / 2
            q1_d_guess_I = np.mean(I)
            q1_a_guess_Q = (np.max(Q) - np.min(Q)) / 2
            q1_d_guess_Q = np.mean(Q)
            q1_b_guess = 1 / gains[-1]
            q1_c_guess = 0

            q1_guess_I = [q1_a_guess_I, q1_b_guess, q1_c_guess, q1_d_guess_I]
            if use_iminuit_instead:
                q1_popt_I, q1_pcov_I = self.fit_cosine_iminuit(gains, I, q1_guess_I)
            else:
                q1_popt_I, q1_pcov_I = curve_fit(self.cosine, gains, I, maxfev=100000, p0=q1_guess_I)
            q1_fit_cosine_I = self.cosine(gains, *q1_popt_I)

            q1_guess_Q = [q1_a_guess_Q, q1_b_guess, q1_c_guess, q1_d_guess_Q]
            if use_iminuit_instead:
                q1_popt_Q, q1_pcov_Q = self.fit_cosine_iminuit(gains, Q, q1_guess_Q)
            else:
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
            else:
                print('Invalid signal passed, please do I Q or None')


            ax2.plot(gains, q1_fit_cosine_Q, '-', color='red', linewidth=3, label="Fit")
            ax1.plot(gains, q1_fit_cosine_I, '-', color='red', linewidth=3, label="Fit")

            # print(len(gains))
            ax1.plot(gains, I, label="Gain (a.u.)", linewidth=2)
            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)

            ax2.plot(gains, Q, label="Q", linewidth=2)
            ax2.set_xlabel("Gain (a.u.)", fontsize=20)
            ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
            ax2.tick_params(axis='both', which='major', labelsize=16)

            #---------------------------------------------------------------------------------
            # --- Compute amplitude data from I and Q ---
            amplitude_data = np.sqrt(np.array(I) ** 2 + np.array(Q) ** 2)

            # --- Fit the amplitude data with the cosine function ---
            # Define initial guesses based on the amplitude_data characteristics.
            a_guess_amp = (np.max(amplitude_data) - np.min(amplitude_data)) / 2
            d_guess_amp = np.mean(amplitude_data)
            b_guess_amp = 1 / gains[-1]
            c_guess_amp = 0

            amp_guess = [a_guess_amp, b_guess_amp, c_guess_amp, d_guess_amp]
            if use_iminuit_instead:
                amp_popt, amp_pcov = self.fit_cosine_iminuit(gains, amplitude_data, amp_guess)
            else:
                amp_popt, amp_pcov = curve_fit(self.cosine, gains, amplitude_data, maxfev=100000, p0=amp_guess)
            amplitude_fit = self.cosine(gains, *amp_popt)

            # --- Compute amplitude curve from the I and Q FITS instead of the data, this is for a test ---
            amp_fit = np.sqrt(q1_fit_cosine_I ** 2 + q1_fit_cosine_Q ** 2)

            # --- Extract the amplitude parameter A directly ---
            A_amplitude = amp_popt[0]
            # print("Amplitude parameter A from cosine fit:", A_amplitude)
            amp_perr = np.sqrt(np.diag(amp_pcov))
            A_amplitude_err = amp_perr[0]
            #print('Amplitude error (std): ', A_amplitude_err)

            if config is not None:
                fig.text(plot_middle, 0.98,
                         f"e-f RPM Q{self.QubitIndex + 1}: {pi_amp:.2f} (a.u.),  A={A_amplitude}"  + f", Pg: {config['reps']}*{config['rounds']} avgs, Pe: {config['reps2']}*{config['rounds']} avgs",
                         fontsize=18, ha='center', va='top') #f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back
            else:
                fig.text(plot_middle, 0.98,
                         f"e-f RPM Q{self.QubitIndex + 1}: {pi_amp:.2f} (a.u.), A={A_amplitude}",
                         fontsize=18, ha='center', va='top')

            # --- Compute R-squared to evaluate goodness of amplitude fit ---
            ss_res = np.sum((amplitude_data - amplitude_fit) ** 2) #Residual sum of squares
            ss_tot = np.sum((amplitude_data - np.mean(amplitude_data)) ** 2) # Total sum of squares (tot variance, how much the raw data varies around its mean)
            R2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

            # --- Plot amplitude data and its cosine fit on the third subplot ---
            ax3.plot(gains, amplitude_data, '-', label="Amp Data", linewidth=2)
            ax3.plot(gains, amplitude_fit, '-', color='green', linewidth=3, label="Cos Fit to Amp Data")

            ax3.plot(gains, amp_fit, '-', color='orange', linewidth=3, label="Amp Fit from I and Q Fits") # for a test

            ax3.set_xlabel("Gain (a.u.)", fontsize=20)
            ax3.set_ylabel("Amplitude (a.u.)", fontsize=20)
            ax3.tick_params(axis='both', which='major', labelsize=16)
            ax3.legend(loc='best')

            #------------------------------------------------------------------------------------------------
            if self.save_figs:
                today_date = datetime.datetime.now().strftime("%Y-%m-%d")
                dated_folder_name = f"made_on_{today_date}"
                outerFolder_expt = os.path.join(self.outerFolder, "q_temperatures", dated_folder_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y%m%d%H%M%S")
                file_name = os.path.join(outerFolder_expt, f"Q{self.QubitIndex + 1}_" + f"Qtemps_RPM_" + f"{formatted_datetime}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
                # print('Plots saved to this folder:',outerFolder_expt)
            plt.close(fig)
            return best_signal_fit, pi_amp, A_amplitude, A_amplitude_err, amplitude_fit, R2

        except Exception as e:
            print("Error fitting cosine:", e)
            # Return None if the fit didn't work
            return None, None, None, None, None, None


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
