from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from build_task import *
from build_state import *
from expt_config import *
from system_config import *
import copy
import visdom
import logging

class FG_T1Program(AveragerProgramV2):
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

        #normal w01 pulse
        self.add_gauss(ch=qubit_ch, name="ge_ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="ge_qubit_pulse",
                       style="arb",
                       envelope="ge_ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )
        #w12 pulse now
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma_ef'], length=cfg['sigma_ef'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="ef_qubit_pulse",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_ef_amp'],
                       )

        ef_freq=cfg['qubit_freq_ef']
        ge_freq = cfg['qubit_freq_ge']
        print(f'Used in f-g T1 drive: ge qubit freq={str(ge_freq)}  ef qubit freq={str(ef_freq)}')

        self.add_loop("waitloop", cfg["steps"])

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="ge_qubit_pulse", t=0)  # play ge drive pulse
        self.delay_auto(t=0.01, tag='waiting after ge drive')  # Wait a small time after ge drive pulse is complete
        self.pulse(ch=self.cfg["qubit_ch"], name="ef_qubit_pulse", t=0)  # f-e drive pulse
        self.delay_auto(t=cfg['wait_time'] + 0.01, tag='wait')  # wait some amount of delay_time after driving to f, then readout
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  #resonator probe pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class FE_T1Program(AveragerProgramV2):
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

        #normal w01 pulse
        self.add_gauss(ch=qubit_ch, name="ge_ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="ge_qubit_pulse",
                       style="arb",
                       envelope="ge_ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )
        #w12 pulse now
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma_ef'], length=cfg['sigma_ef'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="ef_qubit_pulse",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_ef_amp'],
                       )

        ef_freq=cfg['qubit_freq_ef']
        ge_freq = cfg['qubit_freq_ge']
        print(f'Used in f-e T1 drive: ge qubit freq={str(ge_freq)}  ef qubit freq={str(ef_freq)}')

        self.add_loop("waitloop", cfg["steps"])

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="ge_qubit_pulse", t=0)  # play ge drive pulse
        self.delay_auto(t=0.01, tag='waiting after ge drive')  # Wait a small time after ge drive pulse is complete
        self.pulse(ch=self.cfg["qubit_ch"], name="ef_qubit_pulse", t=0)  # f-e drive pulse
        self.delay_auto(t=cfg['wait_time'] + 0.01, tag='wait')  # wait some amount of delay_time after driving to f
        self.pulse(ch=self.cfg["qubit_ch"], name="ge_qubit_pulse", t=0)  # play ge drive pulse to flip e to g
        self.delay_auto(t=0.01, tag='waiting after eg drive')  # Wait a small time after ge drive pulse is complete then readout
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  #resonator probe pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])


class EF_T1Measurement:
    def __init__(self, QubitIndex, number_of_qubits,  outerFolder, round_num, signal, save_figs, experiment = None,
                 live_plot = None, fit_data = None, increase_qubit_reps = False, qubit_to_increase_reps_for = None,
                 multiply_qubit_reps_by = 0, verbose = False, logger = None, qick_verbose=True, expt_name='ef_T1'):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.expt_name = expt_name
        self.fit_data = fit_data
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.live_plot = live_plot
        self.signal = signal
        self.save_figs = save_figs
        self.verbose = verbose
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
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} {expt_name} configuration: {self.config}')
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} T1 configuration: {self.config}')

    def run(self, thresholding=False):
        now = datetime.datetime.now()
        t1=''
        if 'fg' in self.expt_name:
            t1 = FG_T1Program(self.experiment.soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'], cfg=self.config)
        elif 'fe' in self.expt_name:
            t1 = FE_T1Program(self.experiment.soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'], cfg=self.config)
        else:
            print('Please pass fg or fe in expt_name, or reconfigure')

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

    def plot_results(self, I, Q, delay_times, now, config = None, fig_quality =100):
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
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')  # , facecolor='white'
        plt.close(fig)


