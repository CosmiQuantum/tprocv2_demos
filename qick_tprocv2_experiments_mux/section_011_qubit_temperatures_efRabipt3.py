import matplotlib.pyplot as plt
import numpy as np
import visdom
from scipy.optimize import curve_fit
import datetime
from build_task import *
from build_state import *
# from expt_config import *
from expt_config import *
import copy
# import visdom
from scipy.signal import argrelextrema

class Temps_EFAmpRabiExperiment:
    def __init__(self, QubitIndex, number_of_qubits, list_of_all_qubits,  outerFolder, round_num, signal, save_figs, experiment = None, live_plot = None,
                 increase_qubit_reps = False, qubit_to_increase_reps_for = None, multiply_qubit_reps_by = 0, unmasking_resgain = False):
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
                    if self.QubitIndex==qubit_to_increase_reps_for:
                        print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.config["reps"] *= multiply_qubit_reps_by
            print(f'Q {self.QubitIndex + 1} Round {self.round_num} EF Rabi configuration: ', self.config)


    def run(self, soccfg, soc):
        print(self.config)
        amp_rabi1 = AmplitudeRabiProgram1(soccfg, reps=self.config['reps2'], final_delay=self.config['relax_delay'], cfg=self.config)
        if self.live_plot:
            I1, Q1, gains1 = self.live_plotting(amp_rabi1, soc)
        else:
            iq_list1 = amp_rabi1.acquire(soc, soft_avgs=self.config["rounds"], progress=True)
            I1 = iq_list1[self.QubitIndex][0, :, 0]
            Q1 = iq_list1[self.QubitIndex][0, :, 1]
            gains1 = amp_rabi1.get_pulse_param('qubit_pulse', "gain", as_array=True)
        q1_fit_cosine1, pi_amp1, A_amplitude1, amp_fit1 = self.plot_results( I1, Q1, gains1, config = self.config)


        amp_rabi2 = AmplitudeRabiProgram2(soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'],
                                          cfg=self.config)
        if self.live_plot:
            I2, Q2, gains2 = self.live_plotting(amp_rabi2, soc)
        else:
            iq_list2 = amp_rabi2.acquire(soc, soft_avgs=self.config["rounds"], progress=True)
            I2 = iq_list2[self.QubitIndex][0, :, 0]
            Q2 = iq_list2[self.QubitIndex][0, :, 1]
            gains2 = amp_rabi2.get_pulse_param('qubit_pulse', "gain", as_array=True)
        q1_fit_cosine2, pi_amp2, A_amplitude2, amp_fit2 = self.plot_results(I2, Q2, gains2, config=self.config)


        return I1, Q1, gains1, q1_fit_cosine1, pi_amp1, A_amplitude1, amp_fit1, I2, Q2, gains2, q1_fit_cosine2, pi_amp2, A_amplitude2, amp_fit2, self.config


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
            amp_fit = self.cosine(gains, *amp_popt)

            # --- Extract the amplitude parameter A directly ---
            # A_amplitude = abs(amp_popt[0])
            A_amplitude = amp_popt[0]
            # print("Amplitude parameter A from cosine fit:", A_amplitude)


            # --- Plot amplitude data and its cosine fit on the third subplot ---
            ax3.plot(gains, amplitude_data, '-', label="Amplitude Data", linewidth=2)
            ax3.plot(gains, amp_fit, '-', color='green', linewidth=3, label="Amplitude Fit")
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
                outerFolder_expt = os.path.join(self.outerFolder, "q_temperatures_plots")
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"Qtemps_q{self.QubitIndex + 1}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)
            return best_signal_fit, pi_amp, A_amplitude, amp_fit

        except Exception as e:
            print("Error fitting cosine:", e)
            # Return None if the fit didn't work
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