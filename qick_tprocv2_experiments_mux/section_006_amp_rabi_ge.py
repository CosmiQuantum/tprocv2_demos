import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import datetime
from build_task import *
from build_state import *
from expt_config import *
import copy
import visdom
import logging
import math
from sklearn import preprocessing
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import datetime
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


class AmplitudeRabiExperiment:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, signal, save_shots=False, save_figs = True, experiment = None,
                 live_plot = None, increase_qubit_reps = False, qubit_to_increase_reps_for = None,
                 multiply_qubit_reps_by = 0, verbose = False, logger = None, qick_verbose=True, QZE=False,
                 projective_readout_pulse_len_us=9,  time_between_projective_readout_pulses=None, expt_name = "power_rabi_ge", unmasking_resgain = False):
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
        self.time_between_projective_readout_pulses=time_between_projective_readout_pulses
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            if increase_qubit_reps:
                    if self.QubitIndex==qubit_to_increase_reps_for:
                        if self.verbose: print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.logger.info(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.config["reps"] *= multiply_qubit_reps_by
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} Rabi configuration: {self.config}')
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} Rabi configuration: ', self.config)


    def run(self, thresholding=False):
        #initialize everything and define the sequence for each loop, send to QICK hardware using the soc object
        if self.QZE:
            amp_rabi = AmplitudeRabi_QZE_Program(self.experiment.soccfg,  reps=self.config['reps'],
                                            final_delay=self.config['relax_delay'], cfg=self.config,
                                                 projective_readout_pulse_len_us = self.projective_readout_pulse_len_us,
                                                 time_between_projective_readout_pulses = self.time_between_projective_readout_pulses)
            if thresholding:
                iq_list = amp_rabi.acquire(self.experiment.soc, soft_avgs=self.config["rounds"],
                                           threshold=self.experiment.readout_cfg["threshold"],
                                           angle=self.experiment.readout_cfg["ro_phase"], progress=self.qick_verbose)
            else:
                iq_list = amp_rabi.acquire(self.experiment.soc, soft_avgs=self.config["rounds"],
                                           progress=self.qick_verbose)
        else:
            amp_rabi = AmplitudeRabiProgram(self.experiment.soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'], cfg=self.config)

            if self.live_plot:
                I, Q, gains = self.live_plotting(amp_rabi, thresholding)
            else:
                # Send the complied program that was set above to the qick hardware using soc
                # Tell how many times to repeat using the rounds function, and the definition will do that many measurements
                # and average over those
                # progress=True shows you the bar as data is being collected. maybe disable for speed in the future
                # The QICK will run the 'body' method in AmplitudeRabiProgram repeatedly for the iterations set in the
                # initalize loop when this aquire def is used
                if thresholding:
                    iq_list = amp_rabi.acquire(self.experiment.soc, soft_avgs=self.config["rounds"],
                                               threshold=self.experiment.readout_cfg["threshold"],
                                               angle=self.experiment.readout_cfg["ro_phase"], progress=self.qick_verbose)
                else:
                    iq_list = amp_rabi.acquire(self.experiment.soc, soft_avgs=self.config["rounds"], progress=self.qick_verbose)

                I = iq_list[self.QubitIndex][0][ :, 0]
                Q = iq_list[self.QubitIndex][0][ :, 1]

            #get the gains that were used so you can use to plot on the x axis
            gains = amp_rabi.get_pulse_param('qubit_pulse', "gain", as_array=True)

        q1_fit_cosine, pi_amp = self.plot_results( I, Q, gains, config = self.config)

        if self.save_shots:
            raw_0 = amp_rabi.get_raw()  # I,Q data without normalizing to readout window, subtracting readout offset, or rotation/thresholding
            Ishots = raw_0[self.QubitIndex][:, :, 0, 0]
            Qshots = raw_0[self.QubitIndex][:, :, 0, 1]
            return I, Q, Ishots, Qshots, gains, q1_fit_cosine, pi_amp, self.config

        else:
            return I, Q, gains, q1_fit_cosine, pi_amp, self.config

    def live_plotting(self, amp_rabi, thresholding):
        I = Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        if not viz.check_connection(timeout_seconds=5):
            raise RuntimeError("Visdom server not connected!")

        for ii in range(self.config["rounds"]):
            if thresholding:
                iq_list = amp_rabi.acquire(self.experiment.soc, soft_avgs=1,
                                           threshold=self.experiment.readout_cfg["threshold"],
                                           angle=self.experiment.readout_cfg["ro_phase"], progress=self.qick_verbose)
            else:
                iq_list = amp_rabi.acquire(self.experiment.soc, soft_avgs=1, progress=self.qick_verbose)
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

    def plot_results(self, I, Q, gains, config = None, fig_quality = 100):
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
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
                if self.QZE:
                    fig.text(plot_middle, 0.98,
                             f"Rabi Q{self.QubitIndex + 1}_" + f", {config['reps']}*{config['rounds']} avgs" + f' pi_amp {round(pi_amp,2)} '
                                                                                                               f'projective readout pulse length'
                                                                                                               f': {self.projective_readout_pulse_len_us}'
                                                                                                               f' readout pulse amp: '
                                                                                                               f' {self.experiment.readout_cfg["res_gain_ge"][self.QubitIndex]} ',
                             fontsize=24, ha='center',
                             va='top')  # f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back

                else:
                    fig.text(plot_middle, 0.98,
                             f"Rabi Q{self.QubitIndex + 1}_"  + f", {config['reps']}*{config['rounds']} avgs" + f' pi_amp {pi_amp} ',
                             fontsize=24, ha='center', va='top') #f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back

            else:
                fig.text(plot_middle, 0.98,
                         f"Rabi Q{self.QubitIndex + 1}_" f", {self.config['sigma'] * 1000} ns sigma" + f' pi_amp {pi_amp} '+ f", {self.config['reps']}*{self.config['rounds']} avgs",
                         fontsize=24, ha='center', va='top')

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
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_plots")
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)
            return best_signal_fit, pi_amp

        except Exception as e:
            if self.verbose: print("Error fitting cosine:", e)
            self.logger.info("Error fitting cosine: {e}")
            # Return None if the fit didn't work
            return None, None

    def plot_QZE(self, I, Q, gains, proj_pulse_gains, fig_quality=100, filter_amp_above=None, mark_w01s=True,
                 pi_line_label_left=r"$\omega_{01}$",pi_line_label_right=r"$\omega_{01\text{ Stark Shifted}}$", pi_len=0.11):

        #sort
        import numpy as np
        proj_gains_array = np.array(proj_pulse_gains)
        sorted_indices = np.argsort(proj_gains_array)

        #sort the rest
        sorted_I = [I[i] for i in sorted_indices]
        sorted_Q = [Q[i] for i in sorted_indices]
        sorted_gains = [gains[i] for i in sorted_indices]
        sorted_proj_pulse_gains = proj_gains_array[sorted_indices]

        #filter above a certain gain because of the reverse shift thing that i think is caused by inaccurate qspec due to power broadening
        if filter_amp_above is not None:
            mask = sorted_proj_pulse_gains > filter_amp_above
            sorted_I = [sorted_I[i] for i in range(len(sorted_I)) if mask[i]]
            sorted_Q = [sorted_Q[i] for i in range(len(sorted_Q)) if mask[i]]
            sorted_gains = [sorted_gains[i] for i in range(len(sorted_gains)) if mask[i]]
            sorted_proj_pulse_gains = sorted_proj_pulse_gains[mask]

        fig, ax3 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})
        cmap = plt.get_cmap('coolwarm_r')
        min_gain = np.min(sorted_proj_pulse_gains)
        max_gain = np.max(sorted_proj_pulse_gains)
        if max_gain - min_gain == 0:
            norm_gains = np.zeros_like(sorted_proj_pulse_gains)
        else:
            norm_gains = (sorted_proj_pulse_gains - min_gain) / (max_gain - min_gain)

        for idx in range(len(sorted_I)):
            magnitudes = np.abs(np.array(sorted_I[idx]) + 1j * np.array(sorted_Q[idx]))
            color = cmap(norm_gains[idx])
            ax3.plot(sorted_gains[idx], magnitudes, linewidth=2, color=color)

        ax3.set_ylabel("Magnitude (a.u.)", fontsize=14)
        ax3.set_xlabel("Qubit drive pulse width (us)", fontsize=14)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        ax3.set_title(
            "Inducing the Quantum Zeno Effect in a Transmon Qubit:\nHolding the $|1\\rangle$ State During Rabi Drive with Stark Shift Correction",
            fontsize=16)

        #add line if mark_w01s is True
        if mark_w01s:
            # mark freq cutoff
            ax3.axvline(x=pi_len, ls=":", c="black", lw=2)
            y_min, y_max = ax3.get_ylim()
            y_pos = y_min + 0.01 * (y_max - y_min)
            ax3.text(pi_len - 0.5, y_pos, pi_line_label_left,
                     ha='right', va='bottom', color='black', fontsize=12)
            ax3.text(pi_len + 0.5, y_pos, pi_line_label_right,
                     ha='left', va='bottom', color='black', fontsize=12)



        norm_obj = Normalize(vmin=min_gain, vmax=max_gain)
        sm = ScalarMappable(norm=norm_obj, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax3)
        cbar.set_label("Projection pulse amplitude (a.u)", fontsize=14)

        plt.tight_layout()

        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt,
                                 f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}")
        fig.savefig(file_name + '.png', dpi=fig_quality, bbox_inches='tight')
        fig.savefig(file_name + '.pdf', dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)
        return

    def plot_chevron_2d(self,I,Q,gains,offset_freq,fig_quality: int = 100,filter_amp_above=None,mark_w01s: bool = True,
            pi_line_label_left: str = r"$\omega_{01}$",
            pi_line_label_right: str = r"$\omega_{01\text{ Stark Shifted}}$", pi_len=0.11, reg_qfreq=None, z_limit=None
    ):

        offset_freq_array = np.array(offset_freq)
        order = np.argsort(offset_freq_array)

        I_sorted = [I[i] for i in order]
        Q_sorted = [Q[i] for i in order]
        x_sorted = [gains[i] for i in order]  # x ≡ drive-pulse width
        y_sorted = offset_freq_array[order]  # y ≡ proj-pulse gain

        if filter_amp_above is not None:
            keep_mask = y_sorted > filter_amp_above
            I_sorted = [I_sorted[i] for i in range(len(I_sorted)) if keep_mask[i]]
            Q_sorted = [Q_sorted[i] for i in range(len(Q_sorted)) if keep_mask[i]]
            x_sorted = [x_sorted[i] for i in range(len(x_sorted)) if keep_mask[i]]
            y_sorted = y_sorted[keep_mask]

        x_vals = np.asarray(x_sorted[0])
        n_x, n_y = len(x_vals), len(y_sorted)
        mag_matrix = np.empty((n_y, n_x))

        for row, (I_trace, Q_trace) in enumerate(zip(I_sorted, Q_sorted)):
            magnitudes = np.abs(np.array(I_trace) + 1j * np.array(Q_trace))
            mag_matrix[row] = magnitudes

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.get_cmap('viridis')
        im = ax.imshow(
            mag_matrix,
            origin="lower",
            aspect="auto",
            extent=[x_vals[0], x_vals[-1], y_sorted[0], y_sorted[-1]],
            cmap=cmap,
            interpolation="nearest"
        )
        if z_limit is not None: im.set_clim(vmin=0, vmax=z_limit)  # same z limit for all plots for easy comparison
        # Axes & labels
        ax.set_xlabel("Qubit drive-pulse width (us)", fontsize=14)
        ax.set_ylabel("Frequency used to drive qubit (MHz)", fontsize=14)
        ax.set_title(
            "Rabi Chevron in same format as QZE experiment but with no resonator pulse",
            fontsize=16
        )
        ax.tick_params(axis='both', which='major', labelsize=12)

        if mark_w01s:
            ax.axhline(y=reg_qfreq, ls=":", c="white", lw=2)
            ax.axvline(x=pi_len, ls=":", c="white", lw=2)
            ax.text(pi_len - 0.5, y_sorted[0], pi_line_label_left,
                    ha='right', va='bottom', color='white', fontsize=12)
            ax.text(pi_len + 0.5, y_sorted[0], pi_line_label_right,
                    ha='left', va='bottom', color='white', fontsize=12)

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Signal Magnitude (Hotter corresponds to |1>)", fontsize=14)

        plt.tight_layout()

        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        fname_base = os.path.join(
            outerFolder_expt,
            f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{timestamp}_{self.expt_name}_q{self.QubitIndex + 1}_heatmap"
        )
        fig.savefig(fname_base + ".png", dpi=fig_quality, bbox_inches="tight")
        fig.savefig(fname_base + ".pdf", dpi=fig_quality, bbox_inches="tight")
        plt.close(fig)

        return

    def plot_QZE_2d(self,I,Q,gains,proj_pulse_gains,fig_quality: int = 100,filter_amp_above=None,mark_w01s: bool = True,
            pi_line_label_left: str = r"$\omega_{01}$",
            pi_line_label_right: str = r"$\omega_{01\text{ Stark Shifted}}$", pi_len=0.11, z_limit=None,z_lower_limit=None
    ):

        proj_gains_array = np.array(proj_pulse_gains)
        order = np.argsort(proj_gains_array)

        I_sorted = [I[i] for i in order]
        Q_sorted = [Q[i] for i in order]
        x_sorted = [gains[i] for i in order]  # x ≡ drive-pulse width
        y_sorted = proj_gains_array[order]  # y ≡ proj-pulse gain

        if filter_amp_above is not None:
            keep_mask = y_sorted > filter_amp_above
            I_sorted = [I_sorted[i] for i in range(len(I_sorted)) if keep_mask[i]]
            Q_sorted = [Q_sorted[i] for i in range(len(Q_sorted)) if keep_mask[i]]
            x_sorted = [x_sorted[i] for i in range(len(x_sorted)) if keep_mask[i]]
            y_sorted = y_sorted[keep_mask]

        x_vals = np.asarray(x_sorted[0])
        n_x, n_y = len(x_vals), len(y_sorted)
        mag_matrix = np.empty((n_y, n_x))

        for row, (I_trace, Q_trace) in enumerate(zip(I_sorted, Q_sorted)):
            magnitudes = np.abs(np.array(I_trace) + 1j * np.array(Q_trace))
            mag_matrix[row] = magnitudes

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.get_cmap('viridis')
        im = ax.imshow(
            mag_matrix,
            origin="lower",
            aspect="auto",
            extent=[x_vals[0], x_vals[-1], y_sorted[0], y_sorted[-1]],
            cmap=cmap,
            interpolation="nearest"
        )
        if z_limit is not None: im.set_clim(vmin=0, vmax=z_limit) #same z limit for all plots for easy comparison
        if z_lower_limit is not None: im.set_clim(vmin=z_lower_limit, vmax=z_limit)  # same z limit for all plots for easy comparison
        # Axes & labels
        ax.set_xlabel("Qubit drive-pulse width (us)", fontsize=14)
        ax.set_ylabel("Projection-pulse amplitude (a.u.)", fontsize=14)
        ax.set_title(
            "Inducing the Quantum Zeno Effect in a Transmon Qubit\n"
            "Heat-map of  Rabi oscillations",
            fontsize=16
        )
        ax.tick_params(axis='both', which='major', labelsize=12)

        if mark_w01s:
            ax.axvline(x=pi_len, ls=":", c="white", lw=2)
            ax.text(pi_len - 0.5, y_sorted[0], pi_line_label_left,
                    ha='right', va='bottom', color='white', fontsize=12)
            ax.text(pi_len + 0.5, y_sorted[0], pi_line_label_right,
                    ha='left', va='bottom', color='white', fontsize=12)

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Signal Magnitude (Hotter corresponds to |1>)", fontsize=14)

        plt.tight_layout()

        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        fname_base = os.path.join(
            outerFolder_expt,
            f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{timestamp}_{self.expt_name}_q{self.QubitIndex + 1}_heatmap"
        )
        fig.savefig(fname_base + ".png", dpi=fig_quality, bbox_inches="tight")
        fig.savefig(fname_base + ".pdf", dpi=fig_quality, bbox_inches="tight")
        plt.close(fig)

        return

    def plot_QZE_basic(self, I, Q, gains, proj_pulse_gains, fig_quality=100, filter_amp_above=None, mark_w01s=True,
                       pi_line_label_left=r"$\omega_{01}$", pi_line_label_right=r"$\omega_{01\text{ Stark Shifted}}$"):

        proj_gains_array = np.array(proj_pulse_gains)
        sorted_indices = np.argsort(proj_gains_array)
        sorted_I = [I[i] for i in sorted_indices]
        sorted_Q = [Q[i] for i in sorted_indices]
        sorted_gains = [gains[i] for i in sorted_indices]
        sorted_proj_pulse_gains = proj_gains_array[sorted_indices]
        if filter_amp_above is not None:
            mask = sorted_proj_pulse_gains > filter_amp_above
            sorted_I = [sorted_I[i] for i in range(len(sorted_I)) if mask[i]]
            sorted_Q = [sorted_Q[i] for i in range(len(sorted_Q)) if mask[i]]
            sorted_gains = [sorted_gains[i] for i in range(len(sorted_gains)) if mask[i]]
            sorted_proj_pulse_gains = sorted_proj_pulse_gains[mask]
        fig, ax3 = plt.subplots(1, 1, figsize=(15, 4), sharex=True)
        plt.rcParams.update({'font.size': 18})
        for idx in range(len(sorted_I)):
            magnitudes = np.abs(np.array(sorted_I[idx]) + 1j * np.array(sorted_Q[idx]))
            ax3.plot(sorted_gains[idx], magnitudes, linewidth=2, color='blue')
        ax3.set_ylabel("Magnitude (a.u.)", fontsize=14)
        ax3.set_xlabel("Qubit drive pulse width (us)", fontsize=14)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        ax3.set_title("Inducing the Quantum Zeno Effect in a Transmon Qubit at various time intervals", fontsize=16)
        if mark_w01s:
            ax3.axvline(x=0.11, linestyle=':', color='black', linewidth=2)
            y_min, y_max = ax3.get_ylim()
            y_pos = y_min + 0.02 * (y_max - y_min)
            ax3.text(0.11 - 0.015, y_pos, pi_line_label_left, fontsize=14, verticalalignment='center',
                     horizontalalignment='right')
            ax3.text(0.11 + 0.015, y_pos, pi_line_label_right, fontsize=14, verticalalignment='center',
                     horizontalalignment='left')
        plt.tight_layout()
        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt,
                                 f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}")
        fig.savefig(file_name + 'basic.png', dpi=fig_quality, bbox_inches='tight')
        fig.savefig(file_name + 'basic.pdf', dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)
        return

    def plot_QZE_detuned_amp(self, I, Q, gains, proj_pulse_gains, fig_quality=100, filter_amp_above=None,
                             mark_w01s=True, frequency_difference=None, fwhm_w01 = None, fwhm_w01_starked=None,
                             pi_line_label_left=r"$\omega_{01}$",pi_line_label_right=r"$\omega_{01\text{ Stark Shifted}}$",
                             pi_len=None, log_y=True,window=9):

        #sort
        proj_gains_array = np.array(proj_pulse_gains)
        sorted_indices = np.argsort(proj_gains_array)

        #sort everything else
        sorted_I = [I[i] for i in sorted_indices]
        sorted_Q = [Q[i] for i in sorted_indices]
        sorted_gains = [gains[i] for i in sorted_indices]
        sorted_proj_pulse_gains = proj_gains_array[sorted_indices]

        #keep sorting
        if frequency_difference is not None:
            frequency_difference = np.array(frequency_difference)
            sorted_frequency_difference = frequency_difference[sorted_indices]

        # also sort fwhm lists if provided
        if fwhm_w01 is not None and fwhm_w01_starked is not None:
            fwhm_w01 = np.array(fwhm_w01)
            fwhm_w01_starked = np.array(fwhm_w01_starked)

            sorted_fwhm_w01 = fwhm_w01[sorted_indices]
            sorted_fwhm_w01_starked = fwhm_w01_starked[sorted_indices]

        #filter above bad qspec freq fits due to power broadening that makes stark shift correction not accurate
        if filter_amp_above is not None:
            mask = sorted_proj_pulse_gains > filter_amp_above
            sorted_I = [sorted_I[i] for i in range(len(sorted_I)) if mask[i]]
            sorted_Q = [sorted_Q[i] for i in range(len(sorted_Q)) if mask[i]]
            sorted_gains = [sorted_gains[i] for i in range(len(sorted_gains)) if mask[i]]
            sorted_proj_pulse_gains = sorted_proj_pulse_gains[mask]
            if frequency_difference is not None:
                sorted_frequency_difference = sorted_frequency_difference[mask]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))
        plt.rcParams.update({'font.size': 18})

        cmap = plt.get_cmap('coolwarm_r')
        min_gain = np.min(sorted_proj_pulse_gains) if len(sorted_proj_pulse_gains) > 0 else 0
        max_gain = np.max(sorted_proj_pulse_gains) if len(sorted_proj_pulse_gains) > 0 else 1
        if max_gain - min_gain == 0:
            norm_gains = np.zeros_like(sorted_proj_pulse_gains)
        else:
            norm_gains = (sorted_proj_pulse_gains - min_gain) / (max_gain - min_gain)

        #use this to fit
        def sine_func(x, A, B, C, D):
            return A * np.sin(B * (x - C)) + D

        proj_amp_list = []
        pi_time_list = []
        pi_time_err_list = []



        for idx in range(len(sorted_I)):
            I_arr = np.array(sorted_I[idx])
            Q_arr = np.array(sorted_Q[idx])
            magnitudes = np.abs(I_arr + 1j * Q_arr)

            #confine to [0.11, 0.4]
            x_data = np.array(sorted_gains[idx])
            mask_x = (x_data >= pi_len) & (x_data <= 10)
            x_filtered = x_data[mask_x]
            y_filtered = magnitudes[mask_x]

            #plot actual data
            color = cmap(norm_gains[idx]) if len(norm_gains) > 0 else 'blue'
            if idx == 0:
                ax1.plot(x_filtered, y_filtered, 'o', color=color,
                         label='Data (first set shown in legend)')
            else:
                ax1.plot(x_filtered, y_filtered, 'o', color=color)

            # rolling avg value, adjust as needed
            if sorted_proj_pulse_gains[idx]>0.9:
                rolling_window=window
            elif sorted_proj_pulse_gains[idx]>0.4:
                rolling_window=window
            else:
                rolling_window=window
            #rolling avg
            y_rolled = [
                np.mean(y_filtered[i: i + rolling_window])
                for i in range(len(y_filtered) - rolling_window + 1)
            ]
            x_rolled = [
                x_filtered[i + rolling_window // 2]
                for i in range(len(x_filtered) - rolling_window + 1)
            ]
            y_rolled = np.array(y_rolled)
            x_rolled = np.array(x_rolled)

            rolled_minima, _ = find_peaks(-y_rolled)  #find first min of rolled avg
            rolled_maxima, _ = find_peaks(y_rolled)  #find first max of rolled avg

            if len(rolled_minima) == 0 or len(rolled_maxima) == 0:
                continue

            i_min_rolled = rolled_minima[0]
            valid_rolled_maxima = [m for m in rolled_maxima if m > i_min_rolled]
            if not valid_rolled_maxima:
                continue
            i_max_rolled = valid_rolled_maxima[0]

            #now that we found min and max of smoothed data, find corresponding vals in orig data
            center_offset = rolling_window // 2
            i_min_original = i_min_rolled + center_offset
            i_max_original = i_max_rolled + center_offset

            i_min_original = i_min_original - 3
            i_max_original = i_max_original + 3

            i_min_original = max(i_min_original, 0)
            i_max_original = min(i_max_original, len(x_filtered) - 1)
            #fit to this now that weve isolated the nice first rise
            x_fit_data = x_filtered[i_min_original: i_max_original + 1]
            y_fit_data = y_filtered[i_min_original: i_max_original + 1]
            #try fitting, dont break things if it fails
            try:
                A_guess = (np.max(y_fit_data) - np.min(y_fit_data)) / 2.0
                D_guess = np.mean(y_fit_data)
                span = x_fit_data[-1] - x_fit_data[0]
                if span <= 0:
                    continue
                B_guess = 2 * np.pi / span
                C_guess = 0.5 * (x_fit_data[0] + x_fit_data[-1])

                popt, pcov = curve_fit(sine_func, x_fit_data, y_fit_data,
                                       p0=[A_guess, B_guess, C_guess, D_guess])
                #get standard err
                perr = np.sqrt(np.diag(pcov))

                A_fit, B_fit, C_fit, D_fit = popt
                A_err, B_err, C_err, D_err = perr

                #make a nice smooth curve and plot
                x_dense = np.linspace(x_fit_data[0], x_fit_data[-1], 300)
                y_dense_fit = sine_func(x_dense, *popt)

                #get min and max for pi time
                idx_min = np.argmin(y_dense_fit)
                idx_max = np.argmax(y_dense_fit)
                x_min = x_dense[idx_min]
                x_max = x_dense[idx_max]

                ax1.plot(x_dense, y_dense_fit, '-', color=color, linewidth=2)
                #find what pi time is
                pi_time = np.abs(x_max - x_min)
                #propagate err as normal
                pi_time_err = (np.pi / (B_fit ** 2)) * B_err

                proj_amp_list.append(sorted_proj_pulse_gains[idx])
                pi_time_list.append(pi_time)
                pi_time_err_list.append(pi_time_err)

            except Exception as e:
                print(f"Fit failed for index {idx}: {e}")
                continue

        ax1.set_ylabel("Magnitude (a.u.)", fontsize=14)
        ax1.set_xlabel("Qubit drive pulse width (us)", fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.set_title("First Rabi fringes with Curve Fit", fontsize=16)

        if mark_w01s:
            ax1.axvline(x=pi_len, linestyle=':', color='black', linewidth=2)
            y_min, y_max = ax1.get_ylim()
            y_pos = y_min + 0.02 * (y_max - y_min)
            ax1.text(pi_len - 0.015, y_pos, pi_line_label_left, fontsize=14,
                     verticalalignment='center', horizontalalignment='right')
            ax1.text(pi_len + 0.015, y_pos, pi_line_label_right, fontsize=14,
                     verticalalignment='center', horizontalalignment='left')

        norm_obj = Normalize(vmin=min_gain, vmax=max_gain)
        sm = ScalarMappable(norm=norm_obj, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1)
        cbar.set_label("Projection pulse amplitude (a.u.)", fontsize=14)

        #bottom plot for theory test
        if proj_amp_list and pi_time_list:
            proj_amp_array = np.array(proj_amp_list)
            pi_time_array = np.array(pi_time_list)
            pi_time_err_array = np.array(pi_time_err_list)

            sort_idx = np.argsort(proj_amp_array)
            proj_amp_array = proj_amp_array[sort_idx]
            pi_time_array = pi_time_array[sort_idx]
            pi_time_err_array = pi_time_err_array[sort_idx]

            #plot w errs propagated from fit
            ax2.errorbar(
                proj_amp_array, pi_time_array,
                yerr=pi_time_err_array,
                fmt='o', color='black',
                ecolor='black', elinewidth=1.5, capsize=3,
                label=r"Fitted $\pi$ Pulse Time"
            )

            #do the theory part and find expected vals from naghiloo thesis
            # calculate theory and propagate errors if frequency_difference and fwhm lists are provided

            # constant found from experiment
            A_const = np.pi / pi_len*(1e-6)

            # convert frequency difference from MHz to rad/s
            factor = 1e6 * 2 * np.pi
            delta_d_theory = sorted_frequency_difference * factor


            # compute omega_r and the expected pi times (in microseconds)
            omega_r = np.sqrt(A_const ** 2 + delta_d_theory ** 2) #frequency diference gets larger so omega r gets larger too with proj pulse gain
            expected_pi_times = (np.pi / omega_r) * 1e6  # convert to us


            # plot theory and error band
            ax2.plot(sorted_proj_pulse_gains, expected_pi_times, ':',
                     color='darkgreen', linewidth=2, label=r"Expected $\pi$ pulse time from frequency detuning")

            if fwhm_w01 is not None and fwhm_w01_starked is not None:
                # # compute uncertainty in frequency difference
                print(sorted_fwhm_w01)
                print(sorted_fwhm_w01_starked)
                #convert fwhm to sigma using https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/
                freq_diff_err = np.sqrt(((sorted_fwhm_w01/2.3548)* 1e6) ** 2 + ((sorted_fwhm_w01_starked/2.3548)* 1e6) ** 2)
                sigma_delta = 2*np.pi * freq_diff_err
                sigma_omega = (abs(delta_d_theory) / omega_r) * sigma_delta
                sigma_tpi = (np.pi * 1e6 / omega_r ** 2) * sigma_omega  # us

                ax2.fill_between(sorted_proj_pulse_gains,
                                 expected_pi_times - sigma_tpi,
                                 expected_pi_times + sigma_tpi,
                                 color='lightgreen', alpha=0.3)

            ax2.set_xlabel("Projection pulse amplitude (a.u.)", fontsize=14)
            ax2.set_ylabel(r"$\Pi$ pulse time ($\mu$ s)", fontsize=14)
            ax2.set_title(r"$\pi$ Pulse Time vs Projection Pulse Amplitude", fontsize=16)
            ax2.tick_params(axis='both', which='major', labelsize=16)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, r"No valid $\pi$ pulse time data extracted", transform=ax2.transAxes,
                     fontsize=16, ha='center')
        if log_y:
            ax2.set_yscale('symlog', linthresh=5, linscale=1)
        plt.tight_layout()
        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        import datetime
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(
            outerFolder_expt,
            f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}"
        )
        fig.savefig(file_name + 'detuned_amp.png', dpi=fig_quality, bbox_inches='tight')
        fig.savefig(file_name + 'detuned_amp.pdf', dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)

    def plot_QZE_detuning(self, I, Q, gains, proj_pulse_gains, fig_quality=100, filter_amp_above=None,
                             mark_w01s=True, frequency_difference=None, fwhm_w01 = None, fwhm_w01_starked=None,
                          pi_line_label_left=r"$\omega_{01}$",pi_line_label_right=r"$\omega_{01\text{ Stark Shifted}}$",pi_len=None,window=9):

        #sort
        proj_gains_array = np.array(proj_pulse_gains)
        sorted_indices = np.argsort(proj_gains_array)

        #sort everything else
        sorted_I = [I[i] for i in sorted_indices]
        sorted_Q = [Q[i] for i in sorted_indices]
        sorted_gains = [gains[i] for i in sorted_indices]
        sorted_proj_pulse_gains = proj_gains_array[sorted_indices]

        #keep sorting
        if frequency_difference is not None:
            frequency_difference = np.array(frequency_difference)
            sorted_frequency_difference = frequency_difference[sorted_indices]

        # also sort fwhm lists if provided
        if fwhm_w01 is not None and fwhm_w01_starked is not None:
            fwhm_w01 = np.array(fwhm_w01)
            fwhm_w01_starked = np.array(fwhm_w01_starked)

            sorted_fwhm_w01 = fwhm_w01[sorted_indices]
            sorted_fwhm_w01_starked = fwhm_w01_starked[sorted_indices]

        #filter above bad qspec freq fits due to power broadening that makes stark shift correction not accurate
        if filter_amp_above is not None:
            mask = sorted_proj_pulse_gains > filter_amp_above
            sorted_I = [sorted_I[i] for i in range(len(sorted_I)) if mask[i]]
            sorted_Q = [sorted_Q[i] for i in range(len(sorted_Q)) if mask[i]]
            sorted_gains = [sorted_gains[i] for i in range(len(sorted_gains)) if mask[i]]
            sorted_proj_pulse_gains = sorted_proj_pulse_gains[mask]
            if frequency_difference is not None:
                sorted_frequency_difference = sorted_frequency_difference[mask]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))
        plt.rcParams.update({'font.size': 18})

        cmap = plt.get_cmap('coolwarm_r')
        min_gain = np.min(sorted_proj_pulse_gains) if len(sorted_proj_pulse_gains) > 0 else 0
        max_gain = np.max(sorted_proj_pulse_gains) if len(sorted_proj_pulse_gains) > 0 else 1
        if max_gain - min_gain == 0:
            norm_gains = np.zeros_like(sorted_proj_pulse_gains)
        else:
            norm_gains = (sorted_proj_pulse_gains - min_gain) / (max_gain - min_gain)

        #use this to fit
        def sine_func(x, A, B, C, D):
            return A * np.sin(B * (x - C)) + D

        proj_amp_list = []
        pi_time_list = []
        pi_time_err_list = []



        for idx in range(len(sorted_I)):
            I_arr = np.array(sorted_I[idx])
            Q_arr = np.array(sorted_Q[idx])
            magnitudes = np.abs(I_arr + 1j * Q_arr)

            #confine to [0.11, 0.4]
            x_data = np.array(sorted_gains[idx])
            mask_x = (x_data >= pi_len) & (x_data <= 10)
            x_filtered = x_data[mask_x]
            y_filtered = magnitudes[mask_x]

            #plot actual data
            color = cmap(norm_gains[idx]) if len(norm_gains) > 0 else 'blue'
            if idx == 0:
                ax1.plot(x_filtered, y_filtered, 'o', color=color,
                         label='Data (first set shown in legend)')
            else:
                ax1.plot(x_filtered, y_filtered, 'o', color=color)

            # rolling avg value, adjust as needed
            if sorted_proj_pulse_gains[idx]>0.9:
                rolling_window=window
            elif sorted_proj_pulse_gains[idx]>0.4:
                rolling_window=window
            else:
                rolling_window=window
            #rolling avg
            y_rolled = [
                np.mean(y_filtered[i: i + rolling_window])
                for i in range(len(y_filtered) - rolling_window + 1)
            ]
            x_rolled = [
                x_filtered[i + rolling_window // 2]
                for i in range(len(x_filtered) - rolling_window + 1)
            ]
            y_rolled = np.array(y_rolled)
            x_rolled = np.array(x_rolled)

            rolled_minima, _ = find_peaks(-y_rolled)  #find first min of rolled avg
            rolled_maxima, _ = find_peaks(y_rolled)  #find first max of rolled avg

            if len(rolled_minima) == 0 or len(rolled_maxima) == 0:
                continue

            i_min_rolled = rolled_minima[0]
            valid_rolled_maxima = [m for m in rolled_maxima if m > i_min_rolled]
            if not valid_rolled_maxima:
                continue
            i_max_rolled = valid_rolled_maxima[0]

            #now that we found min and max of smoothed data, find corresponding vals in orig data
            center_offset = rolling_window // 2
            i_min_original = i_min_rolled + center_offset
            i_max_original = i_max_rolled + center_offset

            i_min_original = i_min_original - 3
            i_max_original = i_max_original + 3

            i_min_original = max(i_min_original, 0)
            i_max_original = min(i_max_original, len(x_filtered) - 1)
            #fit to this now that weve isolated the nice first rise
            x_fit_data = x_filtered[i_min_original: i_max_original + 1]
            y_fit_data = y_filtered[i_min_original: i_max_original + 1]
            #try fitting, dont break things if it fails
            try:
                A_guess = (np.max(y_fit_data) - np.min(y_fit_data)) / 2.0
                D_guess = np.mean(y_fit_data)
                span = x_fit_data[-1] - x_fit_data[0]
                if span <= 0:
                    continue
                B_guess = 2 * np.pi / span
                C_guess = 0.5 * (x_fit_data[0] + x_fit_data[-1])

                popt, pcov = curve_fit(sine_func, x_fit_data, y_fit_data,
                                       p0=[A_guess, B_guess, C_guess, D_guess])
                #get standard err
                perr = np.sqrt(np.diag(pcov))

                A_fit, B_fit, C_fit, D_fit = popt
                A_err, B_err, C_err, D_err = perr

                #make a nice smooth curve and plot
                x_dense = np.linspace(x_fit_data[0], x_fit_data[-1], 300)
                y_dense_fit = sine_func(x_dense, *popt)

                #get min and max for pi time
                idx_min = np.argmin(y_dense_fit)
                idx_max = np.argmax(y_dense_fit)
                x_min = x_dense[idx_min]
                x_max = x_dense[idx_max]

                ax1.plot(x_dense, y_dense_fit, '-', color=color, linewidth=2)
                #find what pi time is
                pi_time = np.abs(x_max - x_min)
                #propagate err as normal
                pi_time_err = (np.pi / (B_fit ** 2)) * B_err

                proj_amp_list.append(sorted_proj_pulse_gains[idx])
                pi_time_list.append(pi_time)
                pi_time_err_list.append(pi_time_err)

            except Exception as e:
                print(f"Fit failed for index {idx}: {e}")
                continue

        ax1.set_ylabel("Magnitude (a.u.)", fontsize=14)
        ax1.set_xlabel("Qubit drive pulse width (us)", fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.set_title("First Rabi fringes with Curve Fit", fontsize=16)

        if mark_w01s:
            ax1.axvline(x=pi_len, linestyle=':', color='black', linewidth=2)
            y_min, y_max = ax1.get_ylim()
            y_pos = y_min + 0.02 * (y_max - y_min)
            ax1.text(pi_len - 0.015, y_pos, pi_line_label_left, fontsize=14,
                     verticalalignment='center', horizontalalignment='right')
            ax1.text(pi_len + 0.015, y_pos, pi_line_label_left, fontsize=14,
                     verticalalignment='center', horizontalalignment='left')

        norm_obj = Normalize(vmin=min_gain, vmax=max_gain)
        sm = ScalarMappable(norm=norm_obj, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1)
        cbar.set_label("Projection pulse amplitude (a.u.)", fontsize=14)

        #bottom plot for theory test
        if proj_amp_list and pi_time_list:
            proj_amp_array = np.array(proj_amp_list)
            pi_time_array = np.array(pi_time_list)
            pi_time_err_array = np.array(pi_time_err_list)

            sort_idx = np.argsort(proj_amp_array)
            proj_amp_array = proj_amp_array[sort_idx]
            pi_time_array = pi_time_array[sort_idx]
            pi_time_err_array = pi_time_err_array[sort_idx]

            # Constant from theory: given in rad/s.
            A_const = np.pi / pi_len*(1e-6)

            # Convert frequency difference from MHz to rad/s.
            # Here 'sorted_frequency_difference' is assumed to be sorted correspondingly to your proj_amp or pulse gains.
            factor = 1e6 * 2 * np.pi
            delta_d_theory = sorted_frequency_difference * factor


            # --- Inferred delta_d from measured pi_time ---

            # Inversion of the π-pulse time equation:
            #   T_pi (in us) = (pi/omega_r)*1e6  --> omega_r = (pi*1e6)/T_pi.
            #   Then, delta_d (measured) = sqrt(omega_r^2 - A_const^2).
            C = np.pi * 1e6  # factor to convert time into angular frequency units

            # Calculate inferred delta_d from the measured pi times.
            delta_d_measured = np.sqrt((C / pi_time_array) ** 2 - A_const ** 2)


            # --- Plotting the results: compare theoretical and measured δ_d ---

            #fig, ax = plt.subplots(figsize=(8, 6))

            # Plot theory delta_d (from frequency_difference).
            # Replace 'sorted_proj_pulse_gains' with the correct sorted x-axis values if they differ from proj_amp_array.
            ax2.plot(sorted_proj_pulse_gains, delta_d_theory, linestyle=':', color='darkgreen', linewidth=2,
                    label=r"$\Delta_d$ Prediction from fitted $\pi$ time found")

            if fwhm_w01 is not None and fwhm_w01_starked is not None:
                # Compute uncertainty in frequency difference by combining the two FWHM uncertainties.
                # convert fwhm to sigma using https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/
                freq_diff_err = np.sqrt(
                    ((sorted_fwhm_w01 / 2.3548) ) ** 2 + ((sorted_fwhm_w01_starked / 2.3548) ) ** 2)

                delta_d_theory_err = freq_diff_err * factor

                ax2.fill_between(sorted_proj_pulse_gains,
                                delta_d_theory - delta_d_theory_err,
                                delta_d_theory + delta_d_theory_err,
                                color='lightgreen', alpha=0.3, label='Uncertainty from QSpec FWHM')

                # Error propagation:
                # Derivative: d/dT [sqrt((C/T)^2 - A_const^2)] = - (C^2)/(T^3 * sqrt((C/T)^2 - A_const^2))
                delta_d_measured_err = ((C ** 2) / (
                        pi_time_array ** 3 * np.sqrt((C / pi_time_array) ** 2 - A_const ** 2))) * pi_time_err_array

                # --- Plot inferred (measured) δ₍d₎ from the π-pulse time inversion ---
                ax2.errorbar(proj_amp_array, delta_d_measured, yerr=delta_d_measured_err,
                             fmt='s', color='blue', ecolor='blue', elinewidth=1.5, capsize=3,
                             label=r"Measured $\Delta_d$ (from pi time)")
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / (2 * np.pi * 1e6):.2f}'))

            ax2.set_xlabel('Projection Pulse Amplitude')
            ax2.set_ylabel(r"$\Delta_d$ (MHz)")
            ax2.legend()
            ax2.set_title(r"Actual $\Delta_d$ from QSpec vs $\Delta_d$ found from fit to Rabi")
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "No valid pi pulse time data extracted", transform=ax2.transAxes,
                     fontsize=16, ha='center')

        plt.tight_layout()
        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        import datetime
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(
            outerFolder_expt,
            f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}"
        )
        fig.savefig(file_name + 'detuning.png', dpi=fig_quality, bbox_inches='tight')
        fig.savefig(file_name + 'detuning.pdf', dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)

    def plot_QZE_old_format(self, I, Q, gains, proj_pulse_gains, fig_quality=100):

        # Create the figure with two subplots for I and Q
        fig, ax3 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})

        for idx in range(len(I)):
            magnitudes = np.abs(np.array(I[idx]) + 1j * np.array(Q[idx]))
            ax3.plot(gains[idx], magnitudes, label=f"Proj Pulse Gain {proj_pulse_gains[idx]}", linewidth=2)

        ax3.set_ylabel("Magnitude (a.u.)", fontsize=20)
        ax3.set_xlabel("Qubit drive pulse width (us)", fontsize=20)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        ax3.legend(fontsize=10)

        plt.tight_layout()

        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt,
                                 f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}.png")
        fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)
        return

    def plot_QZE_w_systematic_subtraction(self, I, Q, gains, proj_pulse_gains, I_systematics, Q_systematics,
                                              gains_systematics, proj_pulse_gains_systematics,
                                              fig_quality):

        fig, ax3 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})
        plot_middle = (ax3.get_position().x0 + ax3.get_position().x1) / 2

        for idx in range(len(I)):
            magnitudes_qze = np.abs(np.array(I[idx]) + 1j * np.array(Q[idx]))
            magnitudes_systematics = np.abs(np.array(I_systematics[idx]) + 1j * np.array(Q_systematics[idx]))

            # x = np.array(gains[idx])
            # y = np.array(magnitudes_systematics)
            # slope, intercept = np.polyfit(x, y, 1)
            # fitted_line = slope * x + intercept

            magnitudes = [val - syste for val, syste in zip(magnitudes_qze, magnitudes_systematics)]

            ax3.plot(gains[idx], magnitudes, label=f"Proj Pulse Gain {proj_pulse_gains[idx]}", linewidth=2)

        ax3.set_ylabel("Magnitude (a.u.)", fontsize=20)
        ax3.set_xlabel("Qubit drive pulse width (us)", fontsize=20)
        ax3.tick_params(axis='both', which='major', labelsize=16)

        ax3.legend(fontsize=10)

        plt.tight_layout()

        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt,
                                 f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}.png")
        fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)
        return

    def plot_QZE_w_systematic_subtraction_fit(self, I, Q, gains, proj_pulse_gains, I_systematics, Q_systematics,
                                          gains_systematics, proj_pulse_gains_systematics,
                                          fig_quality):

        fig, ax3 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})
        for idx in range(len(I) ):
            try:
                magnitudes_qze = np.abs(np.array(I[idx]) + 1j * np.array(Q[idx]))
                magnitudes_systematics = np.abs(np.array(I_systematics[idx]) + 1j * np.array(Q_systematics[idx]))

                # x = np.array(gains[idx])
                # y = np.array(magnitudes_systematics)
                # slope, intercept = np.polyfit(x, y, 1)
                # fitted_line = slope * x + intercept

                magnitudes = [val - syste for val, syste in zip(magnitudes_qze, magnitudes_systematics)]


                fit_curve, t2r_est, t2r_err = self.t2_fit(gains[idx], magnitudes, verbose=False,
                                                                    guess=None,
                                                                    plot=False)
            except:
                continue
            ax3.plot(gains[idx], fit_curve, label=f"Proj Pulse Gain {proj_pulse_gains[idx]}", linewidth=2)
        ax3.set_ylabel("Magnitude (a.u.)", fontsize=20)
        ax3.set_xlabel("Qubit drive pulse width (us)", fontsize=20)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        ax3.legend(fontsize=10)
        plt.tight_layout()
        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt,
                                 f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}.png")
        fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)
        return

    def t2_fit(self, x_data, magnitudes, verbose = False, guess=None, plot=False):
        #fitting code adapted from https://github.com/qua-platform/py-qua-tools/blob/37c741ade5a8f91888419c6fd23fd34e14372b06/qualang_tools/plot/fitting.py



        # Normalizing the vectors

        xn = preprocessing.normalize([x_data], return_norm=True)
        yn = preprocessing.normalize([magnitudes], return_norm=True)
        x = xn[0][0]
        y = yn[0][0]
        x_normal = xn[1][0]
        y_normal = yn[1][0]

        # Compute the FFT for guessing the frequency
        fft = np.fft.fft(y)
        f = np.fft.fftfreq(len(x))
        # Take the positive part only
        fft = fft[1: len(f) // 2]
        f = f[1: len(f) // 2]
        # Remove the DC peak if there is one
        if (np.abs(fft)[1:] - np.abs(fft)[:-1] > 0).any():
            first_read_data_ind = np.where(np.abs(fft)[1:] - np.abs(fft)[:-1] > 0)[0][0]  # away from the DC peak
            fft = fft[first_read_data_ind:]
            f = f[first_read_data_ind:]

        # Finding a guess for the frequency
        out_freq = f[np.argmax(np.abs(fft))]
        guess_freq = out_freq / (x[1] - x[0])

        # The period is 1 / guess_freq --> number of oscillations --> peaks decay to get guess_T2
        period = int(np.ceil(1 / out_freq))
        peaks = (
                np.array([np.std(y[i * period: (i + 1) * period]) for i in range(round(len(y) / period))]) * np.sqrt(
            2) * 2
        )

        # Finding a guess for the decay (slope of log(peaks))
        if len(peaks) > 1:
            guess_T2 = -1 / ((np.log(peaks)[-1] - np.log(peaks)[0]) / (period * (len(peaks) - 1))) * (x[1] - x[0])
        else:
            guess_T2 = 100 / x_normal

        # Finding a guess for the offsets
        initial_offset = np.mean(y[:period])
        final_offset = np.mean(y[-period:])

        # Finding a guess for the phase
        guess_phase = np.angle(fft[np.argmax(np.abs(fft))]) - guess_freq * 2 * np.pi * x[0]

        # Check user guess
        if guess is not None:
            for key in guess.keys():
                if key == "f":
                    guess_freq = float(guess[key]) * x_normal
                elif key == "phase":
                    guess_phase = float(guess[key])
                elif key == "T2":
                    guess_T2 = float(guess[key]) * x_normal
                elif key == "amp":
                    peaks[0] = float(guess[key]) / y_normal
                elif key == "initial_offset":
                    initial_offset = float(guess[key]) / y_normal
                elif key == "final_offset":
                    final_offset = float(guess[key]) / y_normal
                else:
                    raise Exception(
                        f"The key '{key}' specified in 'guess' does not match a fitting parameters for this function."
                    )

        # Print the initial guess if verbose=True
        if verbose:
            print(
                f"Initial guess:\n"
                f" f = {guess_freq / x_normal:.3f}, \n"
                f" phase = {guess_phase:.3f}, \n"
                f" T2 = {guess_T2 * x_normal:.3f}, \n"
                f" amp = {peaks[0] * y_normal:.3f}, \n"
                f" initial offset = {initial_offset * y_normal:.3f}, \n"
                f" final_offset = {final_offset * y_normal:.3f}"
            )

        # Fitting function
        def func(x_var, a0, a1, a2, a3, a4, a5):
            return final_offset * a4 * (1 - np.exp(-x_var / (guess_T2 * a1))) + peaks[0] / 2 * a2 * (
                    np.exp(-x_var / (guess_T2 * a1))
                    * (a5 * initial_offset / peaks[0] * 2 + np.cos(2 * np.pi * a0 * guess_freq * x + a3))
            )

        def fit_type(x_var, a):
            return func(x_var, a[0], a[1], a[2], a[3], a[4], a[5])

        popt, pcov = optimize.curve_fit(
            func,
            x,
            y,
            p0=[1, 1, 1, guess_phase, 1, 1],
        )

        perr = np.sqrt(np.diag(pcov))

        # Output the fitting function and its parameters
        out = {
            "fit_func": lambda x_var: fit_type(x_var / x_normal, popt) * y_normal,
            "f": [popt[0] * guess_freq / x_normal, perr[0] * guess_freq / x_normal],
            "phase": [popt[3] % (2 * np.pi), perr[3] % (2 * np.pi)],
            "T2": [(guess_T2 * popt[1]) * x_normal, perr[1] * guess_T2 * x_normal],
            "amp": [peaks[0] * popt[2] * y_normal, perr[2] * peaks[0] * y_normal],
            "initial_offset": [
                popt[5] * initial_offset * y_normal,
                perr[5] * initial_offset * y_normal,
            ],
            "final_offset": [
                final_offset * popt[4] * y_normal,
                perr[4] * final_offset * y_normal,
            ],
        }
        # Print the fitting results if verbose=True
        if verbose:
            print(
                f"Fitting results:\n"
                f" f = {out['f'][0] * 1000:.3f} +/- {out['f'][1] * 1000:.3f} MHz, \n"
                f" phase = {out['phase'][0]:.3f} +/- {out['phase'][1]:.3f} rad, \n"
                f" T2 = {out['T2'][0]:.2f} +/- {out['T2'][1]:.3f} ns, \n"
                f" amp = {out['amp'][0]:.2f} +/- {out['amp'][1]:.3f} a.u., \n"
                f" initial offset = {out['initial_offset'][0]:.2f} +/- {out['initial_offset'][1]:.3f}, \n"
                f" final_offset = {out['final_offset'][0]:.2f} +/- {out['final_offset'][1]:.3f} a.u."
            )
        # Plot the data and the fitting function if plot=True
        if plot:
            plt.plot(x_data, fit_type(x, popt) * y_normal)
            plt.plot(
                x_data,
                magnitudes,
                ".",
                label=f"T2  = {out['T2'][0]:.1f} +/- {out['T2'][1]:.1f}ns \n f = {out['f'][0] * 1000:.3f} +/- {out['f'][1] * 1000:.3f} MHz",
            )
            plt.legend(loc="upper right")
        t2r_est = out['T2'][0] #in ns
        t2r_err = out['T2'][1] #in ns
        return fit_type(x, popt) * y_normal, t2r_est, t2r_err

    def plot_QZE_fit(self, I, Q, gains, proj_pulse_gains, fig_quality=100):
        fig, ax3 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})
        for idx in range(len(I) ):
            try:
                magnitudes = np.abs(np.array(I[idx]) + 1j * np.array(Q[idx]))
                fit_curve, t2r_est, t2r_err = self.t2_fit(gains[idx], magnitudes,  verbose=False, guess=None,
                                                                    plot=False)
            except:
                continue
            ax3.plot(gains[idx], fit_curve, label=f"Proj Pulse Gain {proj_pulse_gains[idx]}", linewidth=2)
        ax3.set_ylabel("Magnitude (a.u.)", fontsize=20)
        ax3.set_xlabel("Qubit drive pulse width (us)", fontsize=20)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        ax3.legend(fontsize=10)
        plt.tight_layout()
        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt,
                                 f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}.png")
        fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)
        return

    def roll(self, data: np.ndarray) -> np.ndarray:

        kernel = np.ones(5) / 5
        smoothed = np.convolve(data, kernel, mode='valid')

        # Preserve the original array's shape by padding the edges
        pad_size = (len(data) - len(smoothed)) // 2
        return np.concatenate((data[:pad_size], smoothed, data[-pad_size:]))

    def get_results(self, I, Q, gains, grab_depths = False, rolling_avg=False):
        if rolling_avg:
            I = self.roll(I)
            Q = self.roll(Q)

            first_three_avg_I = np.mean(I[:3])
            last_three_avg_I = np.mean(I[-3:])
            first_three_avg_Q = np.mean(Q[:3])
            last_three_avg_Q = np.mean(Q[-3:])
            if 'Q' in self.signal:
                best_signal = Q
                # figure out if you should take the min or the max value of the fit to say where pi_amp should be
                if last_three_avg_Q > first_three_avg_Q:
                    pi_amp = gains[np.argmax(best_signal)]
                else:
                    pi_amp = gains[np.argmin(best_signal)]
            if 'I' in self.signal:
                best_signal = I
                # figure out if you should take the min or the max value of the fit to say where pi_amp should be
                if last_three_avg_I > first_three_avg_I:
                    pi_amp = gains[np.argmax(best_signal)]
                else:
                    pi_amp = gains[np.argmin(best_signal)]
            if 'None' in self.signal:
                # choose the best signal depending on which has a larger magnitude
                if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
                    best_signal = Q
                    # figure out if you should take the min or the max value of the fit to say where pi_amp should be
                    if last_three_avg_Q > first_three_avg_Q:
                        pi_amp = gains[np.argmax(best_signal)]
                    else:
                        pi_amp = gains[np.argmin(best_signal)]
                else:
                    best_signal = I
                    # figure out if you should take the min or the max value of the fit to say where pi_amp should be
                    if last_three_avg_I > first_three_avg_I:
                        pi_amp = gains[np.argmax(best_signal)]
                    else:
                        pi_amp = gains[np.argmin(best_signal)]
                tot_amp = [np.sqrt((ifit) ** 2 + (qfit) ** 2) for ifit, qfit in zip(I, Q)]
                depth = abs(tot_amp[np.argmin(tot_amp)] - tot_amp[np.argmax(tot_amp)])
            else:
                print('Invalid signal passed, please do I Q or None')
            return best_signal, pi_amp

        else:
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


class AmplitudeRabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # Define a generator for the readout pulses with the gains, phases, and mixer/mux frequencies
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            # We have many qubits and many readout channels, so go through all of them and declare a readout for each
            # of them to tell the system how long the readout pulse is and qhat its freq and phase should be
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)
        # Configure the hardware to set this sort of pulse that we can trigger later
        # This has a rectangle pulse becuase style="const"
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )
        # Tell the system via another generator how to set up the qubit drive pulse
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        # Add a gaussian envolope for the pulse shape with wdith sigma and total length 4sigma
        # print("cfg['sigma']", cfg['sigma'])
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        # Add a pulse configuration to store in the hardware so you can just trigger it later on
        # Tell it to shape the pulse with the gaussian pulse we just defined as 'ramp'. then set the feq/phase/gain
        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )
        self.add_pulse(ch=qubit_ch, name="pi_pulse",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )
        # Make a loop that interates over different pulse amplitudes/gains, this wil be used for the qubit pump, and rabi later
        self.add_loop("gainloop", cfg["steps"])

    def _body(self, cfg):
        # Here we define a sequence of operations that we will use for each iteration of the loop
        # Drive the qubit:
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)
        # Delay
        self.delay_auto(t=0.0, tag='waiting')
        # Readout pulse to look at qubit state
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        # Trigger the readout channels to start collecting the data
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

        ################ Active Reset #################################
        # Wait for readout to be completed
        # self.wait_auto(cfg['res_length']  + 0.2)
        # self.delay_auto(cfg['res_length'] + 0.2)
        # self.label("Readout and check conditions")
        # # n = n + 1
        # self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        # self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])
        #
        # # # Wait for readout to be completed
        # self.wait_auto(cfg['res_length']  + 0.2)
        # self.delay_auto(cfg['res_length'] + 0.2)
        # #
        # # # Read from ro_ch buffer???
        # # # print("cfg['ro_ch'][0])", cfg['ro_ch'][0])
        # # self.read_input(ro_ch=cfg['ro_ch'][0])
        # # self.write_dmem(addr=0, src='s_port_l')
        # # self.write_dmem(addr=1, src='s_port_h')
        # #
        # # # if whatever is read from ro_ch is greater or equal to threshold 1, skip to label('skip everything'))
        # self.read_and_jump(ro_ch=cfg['ro_ch'][0],
        #                    component='I',
        #                    threshold=cfg['edge_of_e_state_threshold'],
        #                    test=">=", label='skip everything')
        #
        # # if whatever is read from ro_ch is greater or equal to threshold 2 (between_g_and_e), go back to label("Readout and check conditions")
        # # self.read_and_jump(ro_ch=cfg['ro_ch'][0],
        # #                    component='I',
        # #                    threshold=cfg['edge_of_e_state_threshold'],
        # #                    test=">=", label="Readout and check conditions")
        # #
        # # # print('playing pi in active to move e to g')
        # # # Play a pi pulse if whatever is read from ro_ch is lesser than both thresholds 1 and 2
        # self.pulse(ch=self.cfg["qubit_ch"], name="pi_pulse", t=0)  # play pulse pi
        # # self.delay_auto()#(self.cfg['sigma'] * 4)  # ????
        # self.jump("Readout and check conditions")
        # self.label('skip everything')

class AmplitudeRabi_QZE_Program(AveragerProgramV2):
    def __init__(self, soccfg, reps, final_delay, final_wait=0, initial_delay=1.0,
                 reps_innermost=False, before_reps=None, after_reps=None, cfg=None,
                 projective_readout_pulse_len_us=None,
                 time_between_projective_readout_pulses = None):

        self.projective_readout_pulse_len_us = projective_readout_pulse_len_us
        self.time_between_projective_readout_pulses = time_between_projective_readout_pulses

        super().__init__(soccfg, reps, final_delay, final_wait, initial_delay,
                         reps_innermost, before_reps, after_reps, cfg)


    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # Configure the resonator (readout) generator
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        # Instead of one long readout pulse, define two pulses:
        # 1. A short projection pulse for the QZE (e.g., 9 ns)
        # 2. A final readout pulse (kept at your original res_length, e.g., 9 us)
        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=self.projective_readout_pulse_len_us,  # new parameter, e.g., 9 ns
                       mask=cfg["list_of_all_qubits"],
                       )
        self.add_pulse(ch=res_ch, name="final_res_pulse",
                       style="const",
                       length=cfg["res_length"],  # final readout pulse remains as originally configured
                       mask=cfg["list_of_all_qubits"],
                       )

        # Set up the qubit drive generator
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4 , even_length=False)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )
        # Loop over different qubit drive pulse amplitudes (for Rabi x axis)
        self.add_loop("gainloop", cfg["steps"])

    def _body(self, cfg):
        # Starting the qubit drive pulse.
        # For the quantum Zeno effect, make sure this pulse lasts long enough to cover all readout pulses
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)

        # Insert a series of short projection pulses during the Rabi drive.
        # cfg["n_proj"] defines the number of projection pulses.
        # cfg["T_proj"] defines the period between these pulses (e.g., on the order of 10s of ns).
        tot_projective_pulses = math.floor(cfg["res_length"]/self.projective_readout_pulse_len_us)
        for i in range(tot_projective_pulses): #divide 9us, normal rabi readout len, by projective len and round down
            t_proj = i * self.time_between_projective_readout_pulses
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=t_proj)

        # After all the projection pulses, add a final readout pulse to capture the qubit state.
        t_final = tot_projective_pulses * self.time_between_projective_readout_pulses
        self.pulse(ch=cfg['res_ch'], name="final_res_pulse", t=t_final)

        # Trigger the readout channels to collect the data.
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

