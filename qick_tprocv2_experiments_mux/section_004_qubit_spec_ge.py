from build_task import *
from build_state import *
# from expt_config import *
from expt_config_nexus import * # Change for quiet vs nexus
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import datetime
import copy
import visdom

class QubitSpectroscopy:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder,  round_num, signal, save_figs, experiment = None, live_plot = None):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "qubit_spec_ge"
        self.signal = signal
        self.save_figs = save_figs
        self.experiment = experiment
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num

        self.number_of_qubits = number_of_qubits
        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.live_plot = live_plot
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}

            print(f'Q {self.QubitIndex + 1} Round {self.round_num} Qubit Spec configuration: ', self.config)

    def run(self, soccfg, soc):
        qspec = PulseProbeSpectroscopyProgram(soccfg, reps=self.config['reps'], final_delay=0.5, cfg=self.config)

        # iq_lists= []
        if self.live_plot:
            I, Q, freqs = self.live_plotting(qspec, soc)
        else:
            iq_list = qspec.acquire(soc, soft_avgs=self.exp_cfg["rounds"], progress=True)
            I = iq_list[self.QubitIndex][0, :, 0]
            Q = iq_list[self.QubitIndex][0, :, 1]
            freqs = qspec.get_pulse_param('qubit_pulse', "freq", as_array=True)

        largest_amp_curve_mean, I_fit, Q_fit = self.plot_results(I, Q, freqs, config = self.config)
        return I, Q, freqs, I_fit, Q_fit, largest_amp_curve_mean

    def live_plotting(self, qspec, soc):
        I = Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        assert viz.check_connection(timeout_seconds=5), "Visdom server not connected!"
        viz.close(win=None)  # close previous plots
        for ii in range(self.config["rounds"]):
            iq_list = qspec.acquire(soc, soft_avgs=1, progress=True)
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

    def plot_results(self, I, Q, freqs, config=None, fig_quality=100):
        freqs = np.array(freqs)
        freq_q = freqs[np.argmax(I)]

        mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err = self.fit_lorenzian(I, Q, freqs,
                                                                                                          freq_q)

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
        ax1.plot(freqs, I_fit, 'r--', label='Lorentzian Fit')
        ax1.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

        ax2.plot(freqs, Q_fit, 'r--', label='Lorentzian Fit')
        ax2.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

        # Calculate the middle of the plot area
        plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

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

        # Adjust spacing
        plt.tight_layout()

        # Adjust the top margin to make room for the title
        plt.subplots_adjust(top=0.93)

        ### Save figure
        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" +
                                     f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)
        return largest_amp_curve_mean, I_fit, Q_fit

    def get_results(self, I, Q, freqs):
        freqs = np.array(freqs)
        freq_q = freqs[np.argmax(I)]

        mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err = self.fit_lorenzian(I, Q, freqs, freq_q)

        return largest_amp_curve_mean, I_fit, Q_fit, qspec_fit_err


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

    def fit_lorenzian(self, I, Q, freqs, freq_q):
        try:
            # Initial guesses for I and Q
            initial_guess_I = [freq_q, 1, np.max(I), np.min(I)]
            initial_guess_Q = [freq_q, 1, np.max(Q), np.min(Q)]

            # First round of fits (to get rough estimates)
            params_I, _ = curve_fit(self.lorentzian, freqs, I, p0=initial_guess_I)
            params_Q, _ = curve_fit(self.lorentzian, freqs, Q, p0=initial_guess_Q)

            # Use these fits to refine guesses
            x_max_diff_I, max_diff_I = self.max_offset_difference_with_x(freqs, I, params_I[3])
            x_max_diff_Q, max_diff_Q = self.max_offset_difference_with_x(freqs, Q, params_Q[3])
            initial_guess_I = [x_max_diff_I, 1, np.max(I), np.min(I)]
            initial_guess_Q = [x_max_diff_Q, 1, np.max(Q), np.min(Q)]

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
            if 'None' in self.signal:
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

            # Return all desired results including the error on the Q fit
            return mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err

        except Exception as e:
            print("Error during Lorentzian fit:", e)
            return None, None,None,None,None,None,None

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

