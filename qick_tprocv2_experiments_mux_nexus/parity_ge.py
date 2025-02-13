
from scipy.signal import welch

from build_task import *
from build_state import *
from expt_config import *
from system_config import *

import visdom

import matplotlib.pyplot as plt

import numpy as np

from NetDrivers import E36300






class ParityMeasurement:
    def __init__(self, QubitIndex, outerFolder,  experiment=None):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        #self.fit_data = fit_data
        self.expt_name = "parity_ge"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        #self.round_num = round_num
        #self.signal = signal
        #self.save_figs = save_figs
        #self.live_plot = live_plot
        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}

    # def PSDandPSDfit(Iamp, Qamp, timetaken, steps):
    #
    #     t = np.linspace(0, timetaken, int(steps))
    #     # Apply Welch's method to estimate the PSD
    #     fs = 1 / (timetaken / steps)  # sampling freq=1/time_step
    #     print('Iamp len',len(Iamp))
    #     frequencies, I_psd_values = welch(Iamp, fs, nperseg=1024)
    #     frequencies, Q_psd_values = welch(Qamp, fs, nperseg=1024)
    #
    #     return frequencies, I_psd_values, Q_psd_values


    def run(self, soccfg, soc, v):
        # Bias_PS_ip = ['192.168.0.44', '192.168.0.44', '192.168.0.44',
        #               '192.168.0.41']  # IP address of bias PS (qubits 1-3 are the same PS)
        # Bias_ch = [1, 2, 3, 1]  # Channel number of qubit 1-4 on associated PS
        # qubit_index = int(self.QubitIndex)
        #
        # BiasPS = E36300(Bias_PS_ip[qubit_index], server_port=5025)
        #
        # BiasPS.setVoltage(0, Bias_ch[qubit_index])
        # BiasPS.enable(Bias_ch[qubit_index])
        #
        # BiasPS.setVoltage(v, Bias_ch[qubit_index])
        # time.sleep(8)
        now = datetime.datetime.now()

        parity = ParityProgram(soccfg, reps=1, final_delay=self.exp_cfg['relax_delay'], cfg=self.config)
        start_time = time.time()

        iq_list = parity.acquire(soc, soft_avgs=1, progress=True)
        I = iq_list[0][0].T[0]
        Q = iq_list[0][0].T[1]
        #print('I[:5]',I[:5])
        timetaken = time.time() - start_time
        #BiasPS.setVoltage(0, Bias_ch[qubit_index])

        #frequencies, I_psd_value, Q_psd_value = self.PSDandPSDfit(I, Q, timetaken)
        #self.plot_results(self, I, Q, fig_quality=100)


        return I, Q, timetaken  # ,t2r_est, t2r_err , delay_times, fit



    # def live_plotting(self, ramsey, soc):
    #     I = Q = expt_mags = expt_phases = expt_pop = None
    #     viz = visdom.Visdom()
    #     assert viz.check_connection(timeout_seconds=5), "Visdom server not connected!"
    #
    #     for ii in range(self.config["rounds"]):
    #         iq_list = ramsey.acquire(soc, soft_avgs=1, progress=True)
    #         delay_times = ramsey.get_time_param('wait', "t", as_array=True)
    #
    #         this_I = iq_list[self.QubitIndex][0, :, 0]
    #         this_Q = iq_list[self.QubitIndex][0, :, 1]
    #
    #         if I is None:  # ii == 0
    #             I, Q = this_I, this_Q
    #         else:
    #             I = (I * ii + this_I) / (ii + 1.0)
    #             Q = (Q * ii + this_Q) / (ii + 1.0)
    #
    #         viz.line(X=delay_times, Y=I,
    #                  opts=dict(height=400, width=700, title='T2 Ramsey I', showlegend=True, xlabel='expt_pts'),
    #                  win='T2R_I')
    #         viz.line(X=delay_times, Y=Q,
    #                  opts=dict(height=400, width=700, title='T2 Ramsey Q', showlegend=True, xlabel='expt_pts'),
    #                  win='T2R_Q')
    #     return I, Q, delay_times

    def set_res_gain_ge(self, QUBIT_INDEX, num_qubits=6):
        """Sets the gain for the selected qubit to 1, others to 0."""
        res_gain_ge = [0] * num_qubits  # Initialize all gains to 0
        if 0 <= QUBIT_INDEX < num_qubits:  # makes sure you are within the range of options
            res_gain_ge[QUBIT_INDEX] = 1  # Set the gain for the selected qubit
        return res_gain_ge

    def exponential(self, x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def plot_results(self, I, Q, timetaken, fig_quality=100):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})
        ti=np.linspace(0,timetaken, len(I))


        # I subplot
        ax1.plot(ti, I, label="Gain (a.u.)", linewidth=2)
        ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        # ax1.axvline(freq_q, color='orange', linestyle='--', linewidth=2)

        # Q subplot
        ax2.plot(ti, Q, label="Q", linewidth=2)
        ax2.set_xlabel("time s", fontsize=20)
        ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        # ax2.axvline(freq_q, color='orange', linestyle='--', linewidth=2)

        # ax3.plot(frequencies, I_psd_value)
        # ax3.set_xlabel("frequencies", fontsize=20)
        # ax3.set_ylabel("I PSD (amplitude**2)/Hz", fontsize=20)
        # ax3.set_title('PSD')
        #
        # ax4.plot(frequencies, Q_psd_value)
        # ax4.set_xlabel("frequencies", fontsize=20)
        # ax4.set_ylabel("Q PSD (amplitude**2)/Hz", fontsize=20)
        # ax4.set_title('PSD')

        # Adjust spacing
        plt.tight_layout()

        # Adjust the top margin to make room for the title
        plt.subplots_adjust(top=0.93)
        #if self.save_figs:
        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt,
                                  f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
        fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')  # , facecolor='white'
        plt.close(fig)


class ParityProgram(AveragerProgramV2):
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
                       mask=[0, 1, 2, 3],
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse1",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'] / 2,
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse2",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg[
                                 'qubit_phase'] + 90,  # current phase + 90   #+ cfg['wait_time']*360*cfg['ramsey_freq'],
                       gain = cfg['pi_amp'] / 2,
        )

        self.add_loop("waitloop", cfg["steps"])

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse1", t=0)  # play probe pulse
        self.delay_auto(cfg['wait_time'] , tag='wait')  # wait_time after last pulse
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse2", t=0)  # play probe pulse
        self.delay_auto(0.01)  # wait_time after last pulse
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])