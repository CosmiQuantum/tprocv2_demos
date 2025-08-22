from build_task import *
from build_state import *
from expt_config import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import datetime
import copy
import visdom
import logging

class QubitSpectroscopy:
    def __init__(self, QubitIndex, number_of_qubits,  outerFolder,  round_num, signal, save_figs, experiment = None,
                 live_plot = None, verbose = False, logger = None, qick_verbose=True, increase_reps = False,
                 increase_reps_to = 500, plot_fit=True, zeno_stark=False, zeno_stark_pulse_gain=None,
                 ext_q_spec=False, high_gain_q_spec=False, fit_data=True, unmasking_resgain = False):

        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.plot_fit=plot_fit
        self.zeno_stark = zeno_stark
        self.zeno_stark_pulse_gain = zeno_stark_pulse_gain
        self.ext_q_spec = ext_q_spec
        self.fit_data = fit_data
        self.high_gain_q_spec = high_gain_q_spec
        if self.zeno_stark:
            self.expt_name = "qubit_spec_ge_zeno_stark"
        elif self.ext_q_spec:
            self.expt_name = "qubit_spec_ge_extended"
        elif self.high_gain_q_spec:
            self.expt_name = "qubit_spec_ge_high_gain"
        else:
            self.expt_name = "qubit_spec_ge"
        self.signal = signal
        self.save_figs = save_figs
        self.experiment = experiment
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.number_of_qubits = number_of_qubits
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")
        self.increase_reps = increase_reps
        self.increase_reps_to = increase_reps_to

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

        if experiment is not None:
            if self.zeno_stark:
                qze_mask = np.arange(0, self.number_of_qubits + 1)
                qze_mask = np.delete(qze_mask, QubitIndex)
                self.exp_cfg['qze_mask'] = qze_mask
                self.experiment.readout_cfg['res_gain_qze'] = [self.experiment.readout_cfg['res_gain_ge'][QubitIndex],
                                                               0, 0, 0, 0, 0, self.zeno_stark_pulse_gain]
                self.experiment.readout_cfg['res_freq_qze'] = self.experiment.readout_cfg['res_freq_ge']
                self.experiment.readout_cfg['res_phase_qze'] = self.experiment.readout_cfg['res_phase']
                if len(self.experiment.readout_cfg['res_freq_qze']) < 7:  # otherise it keeps appending
                    self.experiment.readout_cfg['res_freq_qze'].append(
                        experiment.readout_cfg['res_freq_qze'][self.QubitIndex])
                    self.experiment.readout_cfg['res_phase_qze'].append(
                        experiment.readout_cfg['res_phase_qze'][self.QubitIndex])

            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            print(self.q_config)
            self.live_plot = live_plot
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            print(expt_cfg)
            print(self.exp_cfg)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            print(self.config)
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} Qubit Spec configuration: ', self.config)
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} Qubit Spec configuration: {self.config}')

    def run(self,return_fwhm=False):

        if self.increase_reps:
            self.config['reps'] = self.increase_reps_to

        qspec = PulseProbeSpectroscopyProgram(self.experiment.soccfg, reps=self.config['reps'], final_delay=0.5, cfg=self.config)

        # iq_lists= []
        if self.live_plot:
            I, Q, freqs = self.live_plotting(qspec)
        else:
            iq_list = qspec.acquire(self.experiment.soc, soft_avgs=self.exp_cfg["rounds"], progress=self.qick_verbose)
            I = iq_list[self.QubitIndex][0, :, 0]
            Q = iq_list[self.QubitIndex][0, :, 1]
            freqs = qspec.get_pulse_param('qubit_pulse', "freq", as_array=True)
            self.plot_results(I, Q, freqs, config=self.config,
                              return_fwhm=return_fwhm)

        if self.fit_data:
            if return_fwhm:
                largest_amp_curve_mean, I_fit, Q_fit, fwhm = self.plot_results(I, Q, freqs, config=self.config,
                                                                           return_fwhm=return_fwhm)
                return I, Q, freqs, I_fit, Q_fit, largest_amp_curve_mean, self.config, fwhm
            else:
                largest_amp_curve_mean, I_fit, Q_fit = self.plot_results(I, Q, freqs, config=self.config,
                                                                           return_fwhm=return_fwhm)
                return I, Q, freqs, I_fit, Q_fit, largest_amp_curve_mean, self.config
        else:
            return I, Q, freqs, None, None, None, self.config
        # return I, Q, freqs, None, None, None, self.config

    def run_with_stark_tone(self, wait_for_res_ring_up=False):

        if self.increase_reps:
            self.config['reps'] = self.increase_reps_to
        if wait_for_res_ring_up:
            # gain_to_print = self.config['qubit_gain_ge']
            # len_to_print = self.config['qubit_length_ge']
            # zeno_ras_gain=self.config['res_gain_qze']
            # print(f'qspec for starked freq using qubit pulse gain of {gain_to_print} and pulse length of {len_to_print}, zeno pulse gain {zeno_ras_gain}')
            # print(self.config)
            qspec = PulseProbeSpectroscopyProgram_WithStark_WaitForRingUp(self.experiment.soccfg, reps=self.config['reps'] * 2,
                                                             final_delay=0.5, cfg=self.config)
        else:
            qspec = PulseProbeSpectroscopyProgram_WithStark(self.experiment.soccfg, reps=self.config['reps']*2, final_delay=0.5, cfg=self.config)

        iq_list = qspec.acquire(self.experiment.soc, soft_avgs=self.exp_cfg["rounds"],)
        I = iq_list[self.QubitIndex][0, :, 0]
        Q = iq_list[self.QubitIndex][0, :, 1]
        freqs = qspec.get_pulse_param('qubit_pulse', "freq", as_array=True)

        largest_amp_curve_mean, I_fit, Q_fit, fwhm = self.plot_results(I, Q, freqs, config = self.config, sigma_guess = 10, return_fwhm=True)
        return I, Q, freqs, I_fit, Q_fit, largest_amp_curve_mean, self.config, fwhm

    def live_plotting(self, qspec):
        I = Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        if not viz.check_connection(timeout_seconds=5):
            raise RuntimeError("Visdom server not connected!")
        viz.close(win=None)  # close previous plots
        for ii in range(self.config["rounds"]):
            iq_list = qspec.acquire(self.experiment.soc, soft_avgs=1, progress=self.qick_verbose)
            freqs = qspec.get_pulse_param('qubit_pulse', "freq", as_array=True)

            this_I = iq_list[self.QubitIndex][0, :, 0]
            this_Q = iq_list[self.QubitIndex][0, :, 1]

            if I is None:  # ii == 0
                I, Q = this_I, this_Q
            else:
                I = (I * ii + this_I) / (ii + 1.0)
                Q = (Q * ii + this_Q) / (ii + 1.0)

            viz.line(X=freqs, Y=I, opts=dict(height=400, width=700, title='Qubit Spectroscopy I', showlegend=True, xlabel='expt_pts'),win='QSpec_I')
            viz.line(X=freqs, Y=Q, opts=dict(height=400, width=700, title='Qubit Spectroscopy Q', showlegend=True, xlabel='expt_pts'),win='QSpec_Q')
        return I, Q, freqs

    def plot_results(self, I, Q, freqs, config=None, fig_quality=100, sigma_guess=1, return_fwhm=False):
        freqs = np.array(freqs)
        freq_q = freqs[np.argmax(I)]

        mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err = self.fit_lorenzian(I, Q, freqs,
                                                                                                          freq_q,sigma_guess)

        # Check if the returned values are all None
        if (mean_I is None and mean_Q is None and I_fit is None and Q_fit is None
                and largest_amp_curve_mean is None and largest_amp_curve_fwhm is None):
            # If so, return None for the values in this definition as well
            return None, None, None

        # If we get here, the fit was successful and we can proceed with plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})

        # I subplot
        ax1.plot(freqs, I, label='I', linewidth=2)
        ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.legend()

        # Q subplot
        ax2.plot(freqs, Q, label='Q', linewidth=2)
        ax2.set_xlabel("Qubit Frequency (MHz)", fontsize=20)
        ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.legend()
        # Plot the fits
        if self.plot_fit:
            ax1.plot(freqs, I_fit, 'r--', label='Lorentzian Fit')
            ax1.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

            ax2.plot(freqs, Q_fit, 'r--', label='Lorentzian Fit')
            ax2.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

        # Calculate the middle of the plot area
        plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

        if self.plot_fit:
            # Add title, centered on the plot area
            if config is not None:  # then its been passed to this definition, so use that
                fig.text(plot_middle, 0.98,
                         f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                         f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                         f", {config['reps']}*{config['rounds']} avgs",
                         fontsize=24, ha='center', va='top')
            else:
                fig.text(plot_middle, 0.98,
                         f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                         f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                         f", {self.config['reps']}*{self.config['rounds']} avgs",
                         fontsize=24, ha='center', va='top')
        else:
            # Add title, centered on the plot area
            if config is not None:  # then its been passed to this definition, so use that
                fig.text(plot_middle, 0.98,
                         f"Qubit Spectroscopy Q{self.QubitIndex + 1}" +
                         f", {config['reps']}*{config['rounds']} avgs",
                         fontsize=24, ha='center', va='top')
            else:
                fig.text(plot_middle, 0.98,
                         f"Qubit Spectroscopy Q{self.QubitIndex + 1}",
                         fontsize=24, ha='center', va='top')


                # Adjust spacing
        plt.tight_layout()

        # Adjust the top margin to make room for the title
        plt.subplots_adjust(top=0.93)

        ### Save figure
        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_plots")
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" +
                                     f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)
        if return_fwhm:
            return largest_amp_curve_mean, I_fit, Q_fit, largest_amp_curve_fwhm
        else:
            return largest_amp_curve_mean, I_fit, Q_fit

    def get_results(self, I, Q, freqs):
        freqs = np.array(freqs)
        freq_q = freqs[np.argmax(I)]

        mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err = self.fit_lorenzian(I, Q, freqs, freq_q)

        return largest_amp_curve_mean, I_fit, Q_fit, qspec_fit_err

    def get_results_Two_peaks(self, I, Q, freqs):
        freqs = np.array(freqs)
        freq_q = freqs[np.argmax(I)]

        I_fit, Q_fit, fit_err_I, fit_err_Q, mean_I_1, mean_I_2, mean_Q_1, mean_Q_2, sigma_I_1, sigma_I_2,  sigma_Q_1, sigma_Q_2, height_I_1, height_I_2, height_Q_1, height_Q_2, base_I_1, base_I_2, base_Q_1, base_Q_2 = self.fit_lorenzian_two_peaks(I, Q, freqs)

        return  mean_I_1, mean_I_2, mean_Q_1, mean_Q_2


    def lorentzian(self, f, f0, gamma, A, B):

        return A * gamma ** 2 / ((f - f0) ** 2 + gamma ** 2) + B

    def Two_lorentzians_Q(self, f, f0_1, gamma_1, A_1, B_1, f0_2, gamma_2, A_2, B_2):

        return -1*(A_1 * gamma_1 ** 2 / ((f - f0_1) ** 2 + gamma_1 ** 2) + B_1) - (A_2 * gamma_2 ** 2 / ((f - f0_2) ** 2 + gamma_2 ** 2) + B_2)

    def Two_lorentzians_I(self, f, f0_1, gamma_1, A_1, B_1, f0_2, gamma_2, A_2, B_2):

        return (A_1 * gamma_1 ** 2 / ((f - f0_1) ** 2 + gamma_1 ** 2) + B_1) + (A_2 * gamma_2 ** 2 / ((f - f0_2) ** 2 + gamma_2 ** 2) + B_2)

    def max_offset_difference_with_x(self, x_values, y_values, offset):
        max_average_difference = -1
        corresponding_x = None

        # average all 3 to avoid noise spikes
        for i in range(len(y_values) - 2):
            # group 3 vals
            y_triplet = y_values[i:i + 3]

            # avg differences for these 3 vals
            average_difference = sum(abs(y - offset) for y in y_triplet) / 3

            # see if this is the highest difference yet
            if average_difference > max_average_difference:
                max_average_difference = average_difference
                # x value for the middle y value in the 3 vals
                corresponding_x = x_values[i + 1]

        return corresponding_x, max_average_difference

    def fit_lorenzian(self, I, Q, freqs, freq_q, sigma_guess = 1):
        try:
            # Initial guesses for I and Q
            initial_guess_I = [freq_q, sigma_guess, np.max(I), np.min(I)]
            initial_guess_Q = [freq_q, sigma_guess, np.max(Q), np.min(Q)]

            # First round of fits (to get rough estimates)
            params_I, _ = curve_fit(self.lorentzian, freqs, I, p0=initial_guess_I)
            params_Q, _ = curve_fit(self.lorentzian, freqs, Q, p0=initial_guess_Q)

            # Use these fits to refine guesses
            x_max_diff_I, max_diff_I = self.max_offset_difference_with_x(freqs, I, params_I[3])
            x_max_diff_Q, max_diff_Q = self.max_offset_difference_with_x(freqs, Q, params_Q[3])
            initial_guess_I = [x_max_diff_I, sigma_guess, np.max(I), np.min(I)]
            initial_guess_Q = [x_max_diff_Q, sigma_guess, np.max(Q), np.min(Q)]

            # Second (refined) round of fits, this time capturing the covariance matrices
            params_I, cov_I = curve_fit(self.lorentzian, freqs, I, p0=initial_guess_I)
            params_Q, cov_Q = curve_fit(self.lorentzian, freqs, Q, p0=initial_guess_Q)

            # Create the fitted curves
            I_fit = self.lorentzian(freqs, *params_I)
            Q_fit = self.lorentzian(freqs, *params_Q)

            # Calculate errors from the covariance matrices
            fit_err_I = np.sqrt(np.diag(cov_I))
            fit_err_Q = np.sqrt(np.diag(cov_Q))

            # Extract fitted means and FWHM (assuming params[0] is the mean and params[1] relates to the width)
            mean_I = params_I[0]
            mean_Q = params_Q[0]
            fwhm_I = 2 * params_I[1]
            fwhm_Q = 2 * params_Q[1]

            # Calculate the amplitude differences from the fitted curves
            amp_I_fit = abs(np.max(I_fit) - np.min(I_fit))
            amp_Q_fit = abs(np.max(Q_fit) - np.min(Q_fit))

            # Choose which curve to use based on the input signal indicator
            if 'None' in self.signal or self.signal is None:
                if amp_I_fit > amp_Q_fit:
                    largest_amp_curve_mean = mean_I
                    largest_amp_curve_fwhm = fwhm_I
                    # error on the Q fit's center frequency (first parameter):
                    qspec_fit_err = fit_err_I[0]
                else:
                    largest_amp_curve_mean = mean_Q
                    largest_amp_curve_fwhm = fwhm_Q
                    qspec_fit_err = fit_err_Q[0]
            elif 'I' in self.signal:
                largest_amp_curve_mean = mean_I
                largest_amp_curve_fwhm = fwhm_I
                qspec_fit_err = fit_err_I[0]
            elif 'Q' in self.signal:
                largest_amp_curve_mean = mean_Q
                largest_amp_curve_fwhm = fwhm_Q
                qspec_fit_err = fit_err_Q[0]
            else:
                print('Invalid signal passed, please choose "I", "Q", or "None".')
                return None

        except Exception as e:
            if self.verbose: print("Error during Lorentzian fit:", e)
            self.logger.info(f'Error during Lorentzian fit: {e}')
            # Return all desired results including the error on the Q fit
        return mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err

    def fit_lorenzian_two_peaks(self, I, Q, freqs ):
        try:
            # Initial guesses for I and Q
            mean_guess_1 = 3450.75
            mean_guess_2 = 3450.86
            sigma_guess_1 = 0.08
            sigma_guess_2 = 0.08
            base1 = 71.5
            base2 = 71.5
            height1 = 4.5
            height2 = 4
            initial_guess_I = [freqs, mean_guess_1, sigma_guess_1, base1, height1, mean_guess_2, sigma_guess_2, base2, height2]
            initial_guess_Q = [freqs, mean_guess_1, sigma_guess_1, base1, height1, mean_guess_2, sigma_guess_2, base2, height2]

            # First round of fits (to get rough estimates)
            # params_I, _ = curve_fit(self.Two_lorentzians_I, freqs, I, p0=initial_guess_I)
            # params_Q, _ = curve_fit(self.Two_lorentzians_Q, freqs, Q, p0=initial_guess_Q)
            #
            # # Use these fits to refine guesses
            # x_max_diff_I, max_diff_I = self.max_offset_difference_with_x(freqs, I, params_I[3])
            # x_max_diff_Q, max_diff_Q = self.max_offset_difference_with_x(freqs, Q, params_Q[3])
            # initial_guess_I = [x_max_diff_I, sigma_guess, np.max(I), np.min(I)]
            # initial_guess_Q = [x_max_diff_Q, sigma_guess, np.max(Q), np.min(Q)]

            # Second (refined) round of fits, this time capturing the covariance matrices
            params_I, cov_I = curve_fit(self.Two_lorentzians_I, freqs, I, p0=initial_guess_I)
            params_Q, cov_Q = curve_fit(self.Two_lorentzians_Q, freqs, Q, p0=initial_guess_Q)

            # Create the fitted curves
            I_fit = self.Two_lorentzians_I(freqs, *params_I)
            Q_fit = self.Two_lorentzians_Q(freqs, *params_Q)

            # Calculate errors from the covariance matrices
            fit_err_I = np.sqrt(np.diag(cov_I))
            fit_err_Q = np.sqrt(np.diag(cov_Q))

            # Extract fitted means and FWHM (assuming params[0] is the mean and params[1] relates to the width)
            mean_I_1 = params_I[0]
            mean_I_2 = params_I[4]
            mean_Q_1 = params_Q[0]
            mean_Q_2 = params_Q[4]
            sigma_I_1= 2*params_I[1]
            sigma_I_2 = 2*params_I[5]
            sigma_Q_1 = 2*params_Q[1]
            sigma_Q_2 = 2*params_Q[5]
            height_I_1 = params_I[3]
            height_I_2 = params_I[7]
            height_Q_1 = params_Q[3]
            height_Q_2 = params_Q[7]
            base_I_1= params_I[2]
            base_I_2 = params_I[6]
            base_Q_1 = params_Q[2]
            base_Q_2 = params_Q[6]
            # fwhm_I_1 = 2 * params_I[1]
            # fwhm_I_2 = 2 * params_I[1]
            # fwhm_Q = 2 * params_Q[1]
            # fwhm_Q = 2 * params_Q[1]

            # Calculate the amplitude differences from the fitted curves
            amp_I_fit = abs(np.max(I_fit) - np.min(I_fit))
            amp_Q_fit = abs(np.max(Q_fit) - np.min(Q_fit))

            # Choose which curve to use based on the input signal indicator
            # if 'None' in self.signal or self.signal is None:
            #     if amp_I_fit > amp_Q_fit:
            #         largest_amp_curve_mean = mean_I
            #         largest_amp_curve_fwhm = fwhm_I
            #         # error on the Q fit's center frequency (first parameter):
            #         qspec_fit_err = fit_err_I[0]
            #     else:
            #         largest_amp_curve_mean = mean_Q
            #         largest_amp_curve_fwhm = fwhm_Q
            #         qspec_fit_err = fit_err_Q[0]
            # elif 'I' in self.signal:
            #     largest_amp_curve_mean = mean_I
            #     largest_amp_curve_fwhm = fwhm_I
            #     qspec_fit_err = fit_err_I[0]
            # elif 'Q' in self.signal:
            #     largest_amp_curve_mean = mean_Q
            #     largest_amp_curve_fwhm = fwhm_Q
            #     qspec_fit_err = fit_err_Q[0]
            # else:
            #     print('Invalid signal passed, please choose "I", "Q", or "None".')
            #     return None

            # Return all desired results including the error on the Q fit
        except Exception as e:
            if self.verbose: print("Error during Lorentzian fit:", e)
            self.logger.info(f'Error during Lorentzian fit: {e}')
        # return None, None, None, None, None, None, None
        return I_fit, Q_fit, fit_err_I, fit_err_Q, mean_I_1, mean_I_2, mean_Q_1, mean_Q_2, sigma_I_1, sigma_I_2,  sigma_Q_1, sigma_Q_2, height_I_1, height_I_2, height_Q_1, height_Q_2, base_I_1, base_I_2, base_Q_1, base_Q_2

        # except Exception as e:
        #     if self.verbose: print("Error during Lorentzian fit:", e)
        #     self.logger.info(f'Error during Lorentzian fit: {e}')
        #     return None, None,None,None,None,None,None

    def create_folder_if_not_exists(self, folder_path):
        import os
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


class PulseProbeSpectroscopyProgram(AveragerProgramV2):
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
        print(cfg["list_of_all_qubits"])
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch[0],
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=cfg['qubit_freq_ge'],
                       phase=0,
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse
        self.delay_auto(t=0.01, tag='waiting')  # Wait til qubit pulse is done before proceeding
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class PulseProbeSpectroscopyProgram_WithStark(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_qze'],
                         mux_gains=cfg['res_gain_qze'],  # has 7 values not just 6, extra one for the zeno/stark
                         mux_phases=cfg['res_phase_qze'],
                         mixer_freq=cfg['mixer_freq'])

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch[0],
                       style="const",
                       length=cfg['qubit_length_ge']-0.11,#
                       freq=cfg['qubit_freq_ge'],
                       phase=0,
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'] - 0.11,#
                       mask=cfg['qze_mask'],
                       )

        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse
        self.delay_auto(t=0, tag='waiting')  # Wait til qubit pulse is done before proceeding
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class PulseProbeSpectroscopyProgram_WithStark_WaitForRingUp(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_qze'],
                         mux_gains=cfg['res_gain_qze'],  # has 7 values not just 6, extra one for the zeno/stark
                         mux_phases=cfg['res_phase_qze'],
                         mixer_freq=cfg['mixer_freq'])

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch[0],
                       style="const",
                       length=cfg['qubit_length_ge'] - cfg['qubit_pi_len'],  #
                       freq=cfg['qubit_freq_ge'],
                       phase=0,
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'] - cfg['qubit_pi_len'] + cfg['res_ring_up_time'],  #add ring up time, 2us
                       mask=cfg['qze_mask'],
                       )

        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=cfg['res_ring_up_time'])  # play probe pulse after res ring up to get saturated resonator stark/zeno tone
        self.delay_auto(t=0.0, tag='wait')  # wait for stark tone to finish
        self.delay(t=cfg['res_ring_up_time']) #wait for ring down
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0) #ring down time, then res readout pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class QZEStyleResStarkShift2D:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, res_freq_stark, res_phase_stark, save_figs,
                 experiment=None, signal=None,zeno_stark_pulse_gain=None):

        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "qubit_spec_ge_zeno_stark"
        self.save_figs = save_figs
        self.experiment = experiment
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.number_of_qubits = number_of_qubits
        self.signal = signal
        self.zeno_stark_pulse_gain = zeno_stark_pulse_gain

        if experiment is not None:
            qze_mask = np.arange(0, self.number_of_qubits + 1)
            qze_mask = np.delete(qze_mask, QubitIndex)
            self.exp_cfg['qze_mask'] = qze_mask
            self.experiment.readout_cfg['res_gain_qze'] = [0, 0, 0, 0, 0, 0, self.zeno_stark_pulse_gain]
            self.experiment.readout_cfg['res_gain_qze'][QubitIndex] = self.experiment.readout_cfg['res_gain_ge'][
                QubitIndex]

            self.experiment.readout_cfg['res_freq_qze'] = self.experiment.readout_cfg['res_freq_ge']
            self.experiment.readout_cfg['res_phase_qze'] = self.experiment.readout_cfg['res_phase']
            if len(self.experiment.readout_cfg['res_freq_qze']) < 7:  # otherise it keeps appending
                self.experiment.readout_cfg['res_freq_qze'].append(
                    experiment.readout_cfg['res_freq_qze'][self.QubitIndex])
                self.experiment.readout_cfg['res_phase_qze'].append(
                    experiment.readout_cfg['res_phase_qze'][self.QubitIndex])

            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            print(f'Q {self.QubitIndex} Stark Shift 2D configuration: ', self.config)
            self.config['res_freq_stark'] = res_freq_stark
            self.config['res_phase_stark'] = res_phase_stark
            stark_mask = np.arange(0, self.number_of_qubits + 1)
            stark_mask = np.delete(stark_mask,QubitIndex)
            self.config['stark_mask'] = stark_mask
            res_gain_ge = copy.deepcopy(self.config['res_gain_ge'])
            self.config['stark_gain'] = np.concatenate(
                (res_gain_ge, [zeno_stark_pulse_gain]))  # readout pulse gain, stark tone gain

    def run(self, length=None):
        if length:
            self.config['qubit_length_ge'] = length

        self.config['reps'] = self.config['reps']
        prog = QZEStyleStarkedFreq(self.experiment.soccfg, reps=self.config['reps'], final_delay = 0.5, cfg=self.config)

        iq_list = prog.acquire(self.experiment.soc, soft_avgs=self.exp_cfg["rounds"], progress=True)
        I = iq_list[self.QubitIndex][0, :, 0]
        Q = iq_list[self.QubitIndex][0, :, 1]

        qu_freq_sweep = prog.get_pulse_param('qubit_pulse', "freq", as_array=True)
        starked_freq, fit, fit_err, fwhm = self.plot_results(I, Q, qu_freq_sweep, config = self.config, sigma_guess = 1, return_fwhm=True)
        return I, Q, qu_freq_sweep, starked_freq,fwhm, self.config

    def plot(self, I, Q, qu_freq_sweep, res_gain_sweep):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        plot = axes[0]
        plot.set_box_aspect(1)
        plt.colorbar(plot.pcolormesh(qu_freq_sweep, res_gain_sweep ** 2, I, cmap="viridis"), ax=plot, shrink=0.7)
        plot.set_title("I [a.u.]")
        plot.set_ylabel("stark tone power [a.u.]")
        plot.set_xlabel("qubit pulse frequency [MHz]")

        plot = axes[1]
        plot.set_box_aspect(1)
        plt.colorbar(plot.pcolormesh(qu_freq_sweep, res_gain_sweep ** 2, Q, cmap='viridis'), ax=plot, shrink=0.7)
        plot.set_title("Q [a.u.]")
        plot.set_ylabel("stark tone power [a.u.]")
        plot.set_xlabel("qubit pulse frequency [MHz]")

        plot = axes[2]
        plot.set_box_aspect(1)
        plt.colorbar(plot.pcolormesh(qu_freq_sweep, res_gain_sweep ** 2, np.sqrt(np.square(I) + np.square(Q)), cmap='viridis'), ax=plot,
                     shrink=0.7)
        plot.set_title("magnitude")
        plot.set_ylabel("stark tone power [a.u.]")
        plot.set_xlabel("qubit pulse frequency [MHz]")

        plt.show()

        if self.save_figs:
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(self.outerFolder, f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex}.png")
            fig.savefig(file_name, dpi=100, bbox_inches='tight')
            plt.close(fig)

    def plot_results(self, I, Q, freqs, config=None, fig_quality=100, sigma_guess=1, return_fwhm=False):
        freqs = np.array(freqs)
        mag = np.sqrt(np.square(I) + np.square(Q))
        freq_q = freqs[np.argmax(mag)]
        # plt.figure()
        # plt.plot(freqs,mag)
        # plt.show()


        mean, fit, fit_err, fwhm = self.fit_lorenzian(mag, freqs,freq_q,sigma_guess)


        # Check if the returned values are all None
        if (mean is None and fit is None ):
            # If so, return None for the values in this definition as well
            return None, None, None, None

        # If we get here, the fit was successful and we can proceed with plotting
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})

        # I subplot
        ax1.plot(freqs, I, label='Magnitude', linewidth=2)
        ax1.set_ylabel("Magnitude Amplitude (a.u.)", fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.legend()


        ax1.plot(freqs, fit, 'r--', label='Lorentzian Fit')
        ax1.axvline(mean, color='orange', linestyle='--', linewidth=2)

        # Calculate the middle of the plot area
        plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2


        # Add title, centered on the plot area
        if config is not None:  # then its been passed to this definition, so use that
            fig.text(plot_middle, 0.98,
                     f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % mean +
                     f", {config['reps']}*{config['rounds']} avgs",
                     fontsize=24, ha='center', va='top')
        else:
            fig.text(plot_middle, 0.98,
                     f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % mean  +
                     f", {self.config['reps']}*{self.config['rounds']} avgs",
                     fontsize=24, ha='center', va='top')
        plt.tight_layout()

        # Adjust the top margin to make room for the title
        plt.subplots_adjust(top=0.93)

        ### Save figure
        if self.save_figs:

            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(self.outerFolder,  f"Q_{self.QubitIndex + 1}_" +
                                     f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)
        return mean, fit, fit_err, fwhm

    def lorentzian(self, f, f0, gamma, A, B):

        return A * gamma ** 2 / ((f - f0) ** 2 + gamma ** 2) + B

    def max_offset_difference_with_x(self, x_values, y_values, offset):
        max_average_difference = -1
        corresponding_x = None

        # average all 3 to avoid noise spikes
        for i in range(len(y_values) - 2):
            # group 3 vals
            y_triplet = y_values[i:i + 3]

            # avg differences for these 3 vals
            average_difference = sum(abs(y - offset) for y in y_triplet) / 3

            # see if this is the highest difference yet
            if average_difference > max_average_difference:
                max_average_difference = average_difference
                # x value for the middle y value in the 3 vals
                corresponding_x = x_values[i + 1]

        return corresponding_x, max_average_difference

    def fit_lorenzian(self, mag, freqs, freq_q, sigma_guess = 1):
        try:
            # Initial guesses for I and Q
            initial_guess = [freq_q, sigma_guess, np.max(mag), np.min(mag)]

            # First round of fits (to get rough estimates)
            params, cov = curve_fit(self.lorentzian, freqs, mag, p0=initial_guess) #p0=initial_guess


            # Create the fitted curves
            fit = self.lorentzian(freqs, *params)

            # Calculate errors from the covariance matrices
            fit_err = np.sqrt(np.diag(cov))

            mean = params[0]
            fwhm = 2 * params[1]

            # Return all desired results including the error on the Q fit
            return mean, fit, fit_err, fwhm

        except Exception as e:
            return None, None,None,None

class QZEStyleStarkedFreq(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_qze'],
                         mux_gains=cfg['res_gain_qze'],  # has 7 values not just 6, extra one for the zeno
                         mux_phases=cfg['res_phase_qze'],
                         mixer_freq=cfg['mixer_freq'])
        # readout on each channel with the sampling frequency and length of readout (basically open the window in qick readout)
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph,
                                 gen_ch=res_ch)  # length=readout length at end

        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=cfg['qubit_length_ge']  + cfg['res_ring_up_time']- cfg['qubit_pi_len'],
                       mask=cfg["qze_mask"],
                       )

        self.add_pulse(ch=res_ch, name="readout_pulse",
                       style="const",
                       length=cfg['res_length'],
                       mask=cfg['list_of_all_qubits'], #only play readout tone
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch[0],  # for before we hit pi pulse len
                       style="const",
                       length=cfg['qubit_length_ge']  - cfg['qubit_pi_len'],
                       freq=cfg['qubit_freq_ge'],  # [0] # only should be one value,
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])

        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        self.pulse(ch=self.cfg['res_ch'], name="proj_pulse", t=0)  # play stark/zeno tone
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse", t=cfg['res_ring_up_time']) #play qubit pulse with delay
        self.delay_auto(t=0.5,tag='waiting') #cfg['res_ring_up_time']
        self.pulse(ch=cfg['res_ch'], name="readout_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class ResStarkShift2DAdapted:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, res_freq_stark, res_phase_stark, save_figs, experiment=None, signal=None):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "res_stark_shift_2D"
        self.save_figs = save_figs
        self.experiment = experiment
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.number_of_qubits = number_of_qubits
        self.signal = signal

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            print(f'Q {self.QubitIndex} Stark Shift 2D configuration: ', self.config)
            self.config['res_freq_stark'] = res_freq_stark
            self.config['res_phase_stark'] = res_phase_stark
            stark_mask = np.arange(0, self.number_of_qubits + 1)
            stark_mask = np.delete(stark_mask,QubitIndex)
            self.config['stark_mask'] = stark_mask

    def run(self):
        I = []
        Q = []
        res_gain_ge = copy.deepcopy(self.config['res_gain_ge'])
        gain_sweep = np.linspace(0.3, 1, 3)
        for g in gain_sweep:
            gain = round(g, 3)
            self.config['stark_gain'] = np.concatenate((res_gain_ge, [gain]))  #readout pulse gain, stark tone gain
            prog = ResStarkShift2DProgram(self.experiment.soccfg, reps=self.config['reps'], final_delay = 0.5, cfg=self.config)
            iq_list = prog.acquire(self.experiment.soc, soft_avgs=self.exp_cfg["rounds"], progress=True) #check soft_avgs
            I.append(iq_list[self.QubitIndex][0,:,0])
            Q.append(iq_list[self.QubitIndex][0,:,1])

        qu_freq_sweep = prog.get_pulse_param('qubit_pulse', "freq", as_array=True)
        self.plot( I, Q, qu_freq_sweep, gain_sweep)
        return I, Q, qu_freq_sweep, gain_sweep, self.config

    def plot(self, I, Q, qu_freq_sweep, res_gain_sweep):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        plot = axes[0]
        plot.set_box_aspect(1)
        plt.colorbar(plot.pcolormesh(qu_freq_sweep, res_gain_sweep , I, cmap="viridis"), ax=plot, shrink=0.7)
        plot.set_title("I [a.u.]")
        plot.set_ylabel("stark tone power [a.u.]")
        plot.set_xlabel("qubit pulse frequency [MHz]")

        plot = axes[1]
        plot.set_box_aspect(1)
        plt.colorbar(plot.pcolormesh(qu_freq_sweep, res_gain_sweep , Q, cmap='viridis'), ax=plot, shrink=0.7)
        plot.set_title("Q [a.u.]")
        plot.set_ylabel("stark tone power [a.u.]")
        plot.set_xlabel("qubit pulse frequency [MHz]")

        plot = axes[2]
        plot.set_box_aspect(1)
        plt.colorbar(plot.pcolormesh(qu_freq_sweep, res_gain_sweep , np.sqrt(np.square(I) + np.square(Q)), cmap='viridis'), ax=plot,
                     shrink=0.7)
        plot.set_title("magnitude")
        plot.set_ylabel("stark tone power [a.u.]")
        plot.set_xlabel("qubit pulse frequency [MHz]")

        plt.show()

        if self.save_figs:
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(self.outerFolder, f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex}.png")
            fig.savefig(file_name, dpi=100, bbox_inches='tight')
            plt.close(fig)

# 2D scan over resonator gain, qubit pulse frequency
class ResStarkShift2DProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_stark'], # res of interest frequency at QubitIndex and 7
                         mux_gains=cfg['stark_gain'], # readout gain, stark gain
                         mux_phases=cfg['res_phase_stark'], # res of interest phase repeated at QubitIndex and 7
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="stark_tone",
                       style="const",
                       length=cfg['stark_length'],
                       mask=cfg['stark_mask'], #only play stark tone
                       )

        self.add_pulse(ch=res_ch, name="readout_pulse",
                       style="const",
                       length=cfg['res_length'],
                       mask=cfg['list_of_all_qubits'], #only play readout tone
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        # self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        # self.add_pulse(ch=qubit_ch, name="qubit_pulse",
        #                style="arb",
        #                envelope="ramp",
        #                freq=QickSweep1D("qubit_pulse_loop", cfg['qubit_freq_ge'] + cfg["start_freq"], cfg['qubit_freq_ge'] + cfg["end_freq"]),
        #                phase=cfg['qubit_phase'],
        #                gain=cfg['pi_amp'],
        #                )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch[0],
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=QickSweep1D("qubit_pulse_loop", cfg['qubit_freq_ge'] + cfg["start_freq"], cfg['qubit_freq_ge'] + cfg["end_freq"]),
                       phase=0,
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_loop("qubit_pulse_loop", cfg["qubit_pulse_steps"]) #inner loop

    def _body(self, cfg):
        self.pulse(ch=self.cfg['res_ch'], name="stark_tone", t=0)  # play stark tone
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse", t=cfg['qubit_pulse_delay']) #play qubit pulse with delay
        self.delay(t=cfg['stark_length'] + cfg['readout_pulse_delay']) #wait for stark tone to finish and for resonator to reach vacuum
        self.pulse(ch=cfg['res_ch'], name="readout_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])
