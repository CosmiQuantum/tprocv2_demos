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

    def plot_results(self, I, Q, gains, config = None, fig_quality = 200):
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
            else:
                print('Invalid signal passed, please do I Q or None')


            ax2.plot(gains, q1_fit_cosine_Q, '-', color='red', linewidth=3, label="Fit")
            ax1.plot(gains, q1_fit_cosine_I, '-', color='red', linewidth=3, label="Fit")

            if config is not None:
                fig.text(plot_middle, 0.98,
                         f"e-f Rabi Q{self.QubitIndex + 1}: {pi_amp:.4f} (a.u.) _"  + f", {config['reps']}*{config['rounds']} avgs",
                         fontsize=24, ha='center', va='top') #f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back
            else:
                fig.text(plot_middle, 0.98,
                         f"e-f Rabi Q{self.QubitIndex + 1}: {pi_amp:.4f} (a.u.)_" f", {self.config['sigma'] * 1000} ns sigma" + f", {self.config['reps']}*{self.config['rounds']} avgs",
                         fontsize=24, ha='center', va='top')
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
            amp_popt, amp_pcov = curve_fit(self.cosine, gains, amplitude_data, maxfev=100000, p0=amp_guess)
            amplitude_fit = self.cosine(gains, *amp_popt)

            # --- Extract the amplitude parameter A directly ---
            A_amplitude = amp_popt[0]
            # print("Amplitude parameter A from cosine fit:", A_amplitude)
            amp_perr = np.sqrt(np.diag(amp_pcov))
            A_amplitude_err = amp_perr[0]
            #print('Amplitude error (std): ', A_amplitude_err)

            # --- Plot amplitude data and its cosine fit on the third subplot ---
            ax3.plot(gains, amplitude_data, '-', label="Amplitude Data", linewidth=2)
            ax3.plot(gains, amplitude_fit, '-', color='green', linewidth=3, label="Amplitude Fit")
            ax3.set_xlabel("Gain (a.u.)", fontsize=20)
            ax3.set_ylabel("Amplitude (a.u.)", fontsize=20)
            ax3.tick_params(axis='both', which='major', labelsize=16)
            ax3.legend(loc='best')

            # # Optionally, annotate the amplitude subplot with the extracted A
            # ax3.axhline(y=A_amplitude, linestyle='--', color='black', label=f"A = {A_amplitude:.2f}")
            # ax3.annotate(f"{A_amplitude:.2f}",
            #              xy=(gains[-1], A_amplitude),
            #              xytext=(gains[-1], A_amplitude + 0.1 * A_amplitude),
            #              arrowprops=dict(arrowstyle="->", color="black"),
            #              fontsize=16)

            #------------------------------------------------------------------------------------------------

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            if self.save_figs:
                today_date = datetime.datetime.now().strftime("%Y-%m-%d")
                dated_folder_name = f"made_on_{today_date}"
                outerFolder_expt = os.path.join(self.outerFolder, "q_temperatures", dated_folder_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y%m%d%H%M%S")
                file_name = os.path.join(outerFolder_expt, f"Q{self.QubitIndex + 1}_" + f"Qtemps_RPM_" + f"{formatted_datetime}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
                print('Plots saved to this folder:',outerFolder_expt)
            plt.close(fig)
            return best_signal_fit, pi_amp, A_amplitude, A_amplitude_err, amplitude_fit

        except Exception as e:
            print("Error fitting cosine:", e)
            # Return None if the fit didn't work
            return None, None, None, None, None


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
