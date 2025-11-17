import matplotlib.pyplot as plt
from qick.asm_v2 import AveragerProgramV2
from tqdm import tqdm
from build_state import *
from expt_config import *
import copy
import datetime
import time
from windfreak import SynthHD

class SingleToneSpectroscopyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        res_ch = cfg['res_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_chs[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="mymux",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )

    def _body(self, cfg):
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'], ddr4=True)
        self.pulse(ch=cfg['res_ch'], name="mymux", t=0)
    # def _initialize(self, cfg):
    #     ro_chs = cfg['ro_ch']
    #     gen_ch = cfg['qubit_ampl_ch']
    #     qubit_ch = cfg['qubit_ch']
    #
    #     t = np.linspace(0, cfg['rmeas_length_ge'], int((cfg['rmeas_length_ge']) / 0.026))
    #     QubitIndex = int(1)
    #     L_total = cfg['rmeas_length_ge']  # [QubitIndex]
    #     res_sigma = cfg['rmeas_length_ge'] / 4  # [QubitIndex] / 4
    #     alpha = 1.0
    #     Ltwo = 0.05  # 0.050
    #     # print('before')
    #     idata = 0.5 * (self.two_step_pulse(t, L_total, res_sigma, alpha, Ltwo)) * self.soccfg.get_maxv(gen_ch)
    #     # print('idata',idata)
    #     qdata = np.zeros(len(idata))
    #
    #     self.add_envelope(ch=gen_ch, name="res_envelope", idata=idata, qdata=qdata)
    #
    #     self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_chs[0], mixer_freq=cfg['qubit_mixer_freq2'])
    #
    #     for ch, f, ph in zip(cfg['ro_ch'], cfg['rqubit_freq_ge'], cfg['rqubit_phase']):
    #         self.declare_readout(ch=ch, length=cfg['rmeas_length_ge'], freq=f, phase=ph, gen_ch=gen_ch)
    #
    #     self.add_pulse(
    #         ch=gen_ch, name="res_pulse",
    #         style="arb",
    #         envelope="res_envelope",
    #         # length=cfg["qubit_length_ge"],
    #         freq=cfg["meas_freq_ge"],
    #         gain=cfg["meas_gain_ge"],
    #         phase=cfg["meas_phase"],
    #         # mask=cfg["list_of_all_qubits"],
    #     )
    #
    #     self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
    #
    #     self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
    #
    #     self.add_pulse(ch=qubit_ch, name="ge_pi_pulse",
    #                    style="arb",
    #                    envelope="ramp",
    #                    freq=cfg['qubit_freq_ge'],
    #                    phase=cfg['qubit_phase'],
    #                    gain=cfg['pi_amp'],
    #                    )
    #
    #     self.add_loop("shotloop", cfg["steps"])  # number of total shots
    #
    # def _body(self, cfg):
    #     self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'], ddr4=True)
    #     self.pulse(ch=cfg['qubit_ampl_ch'], name="res_pulse", t=0)
    # #     # self.delay_auto(0.01)
    # #     # self.pulse(ch=cfg['qubit_ampl_ch'], name="res_pulse", t=0)  # play probe pulse
    # #     # self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])


class PunchOut:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, experiment, unmasking_resgain=False):
        self.outerFolder = outerFolder
        self.expt_name = "res_spec"
        self.number_of_qubits = number_of_qubits
        self.experiment = experiment
        self.Qubit = 'Q' + str(1)
        self.QubitIndex = QubitIndex
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [self.QubitIndex]

        self.q_config = all_qubit_state(experiment, self.number_of_qubits)
        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
        print(f'Punch Out configuration: ', self.config)

    def run(self, soccfg, soc, start_gain, stop_gain, num_points, DAC_att, ADC_att, plot_Center_shift = True, plot_res_sweeps = True):
        fpts = self.exp_cfg["start"] + self.exp_cfg["step_size"] * np.arange(self.exp_cfg["steps"])
        fcenter = self.config['res_freq_ge']

        resonance_vals, power_sweep, frequency_sweeps = self.sweep_power(soccfg, soc, fpts, fcenter, start_gain, stop_gain, num_points)

        if plot_Center_shift:
            self.plot_center_shift(resonance_vals, power_sweep, DAC_att, ADC_att)

        if plot_res_sweeps:
            self.plot_res_sweeps(fpts, fcenter, frequency_sweeps, power_sweep, DAC_att, ADC_att,)

        return

    def sweep_power(self, soccfg, soc, fpts, fcenter, start_gain, stop_gain, num_points):
        power_sweep = np.linspace(start_gain, stop_gain, num_points)

        resonance_vals = []
        frequency_sweeps = []
        # print("self.config['res_freq_ge']",self.config['res_freq_ge'])
        for p in power_sweep:
            power = round(p, 3)
            self.config['res_gain_ge'] = [power for i in range(0, self.number_of_qubits)]
            amps = np.zeros((len(fcenter), len(fpts)))
            for index, f in enumerate(tqdm(fpts)):
                self.config["res_freq_ge"] = fcenter + f
                prog = SingleToneSpectroscopyProgram(soccfg, reps=self.exp_cfg["reps"], final_delay=0.5,
                                                     cfg=self.config)
                iq_list = prog.acquire(soc, soft_avgs=self.exp_cfg["rounds"], progress=False)
                for i in range(len(self.config['res_freq_ge'])):
                    # print("np.abs(iq_list[i][:,0])",np.abs(iq_list[i][:,0]))
                    amps[i][index] = np.abs(iq_list[i][:, 0] + 1j * iq_list[i][:, 1])
            amps = np.array(amps)
            frequency_sweeps.append(amps)

            freq_res = []
            for i in range(self.number_of_qubits):
                freq_res.append(round(float(fpts[np.argmin(amps[i])] + fcenter[i]), 3))
            resonance_vals.append(freq_res)
        return resonance_vals, power_sweep, frequency_sweeps

    def plot_center_shift(self, resonance_vals, power_sweep,DAC_att, ADC_att ):
        plt.figure(figsize=(12, 8))

        # Set larger font sizes
        plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })

        for i in range(self.number_of_qubits):
            plt.subplot(2, 3, i + 1)
            plt.plot(power_sweep, [six_resonance_vals[i] for six_resonance_vals in resonance_vals], '-', linewidth=1.5)

            plt.xlabel("Probe Gain", fontweight='normal')
            plt.ylabel("Freq (MHz)", fontweight='normal')
            plt.title(f"Resonator {i + 1}", pad=10)

        # Add a main title to the figure
        plt.suptitle(f"Frequency vs Probe Gain, _DAC_Att_{DAC_att}, ADC_ATT_{ADC_att}", fontsize=24, y=0.95)

        plt.tight_layout(pad=2.0)

        outerFolder_expt = os.path.join(self.outerFolder, 'punch_out')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt, f"{formatted_datetime}_punch_out_center_shift_DAC_Att_{DAC_att}_ADC_ATT_{ADC_att}.png")
        plt.savefig(file_name, dpi=300)
        plt.close()
        return

    def plot_res_sweeps(self, fpts, fcenter, frequency_sweeps, power_sweep, DAC_att, ADC_att):
        plt.figure(figsize=(12, 8))

        # Set larger font sizes
        plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })
        for power_index in range(len(power_sweep)):
            for i in range(self.number_of_qubits):
                plt.subplot(2, 3, i + 1)
                plt.plot(fpts + fcenter[i], frequency_sweeps[power_index][i] + 10*power_index, '-', linewidth=1.5,
                         label=round(power_sweep[power_index], 3))

                plt.xlabel("Frequency (MHz)", fontweight='normal')
                plt.ylabel("Amplitude (a.u)", fontweight='normal')
                plt.title(f"Resonator {i + 1}", pad=10)
                plt.legend(loc='upper left', fontsize='6', title='Gain')

        # Add a main title to the figure
        plt.suptitle(f"Resonance At Various Probe Gains DAC_Att_{DAC_att}, ADC_ATT_{ADC_att}", fontsize=24, y=0.95)

        plt.tight_layout(pad=2.0)
        outerFolder_expt = os.path.join(self.outerFolder, "punch_out")
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt, f"{formatted_datetime}_punch_out_res_sweep_DAC_Att_{DAC_att}_ADC_ATT_{ADC_att}.png")
        plt.savefig(file_name, dpi=300)
        plt.close()
        return


class TWPAConsistency:
    def __init__(self, outerFolder, experiment):
        self.outerFolder = outerFolder
        self.expt_name = "res_spec"

        self.experiment = experiment
        self.Qubit = 'Q' + str(1)
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.q_config = all_qubit_state(experiment)
        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
        print(f'TWPA Consistency configuration: ', self.config)

    def run(self, soccfg, soc, pump_power, pump_freq, num_points, plot_res_sweeps=True, plot_gains=True):
        fpts = (np.linspace(self.exp_cfg["start"], self.exp_cfg["stop"], self.exp_cfg["steps"]))

        resonance_vals, frequency_sweeps, gains = self.repeat_TWPA(soccfg, soc, fpts, pump_power, pump_freq, num_points)

        if plot_res_sweeps:
            self.plot_res_sweeps(fpts, frequency_sweeps, pump_power, pump_freq, num_points)

        if plot_gains:
            self.plot_TWPA_gains(pump_power, pump_freq, num_points, gains)

        return

    def repeat_TWPA(self, soccfg, soc, fpts, pump_power, pump_freq, num_points):
        #power_sweep = np.linspace(start_power, stop_power, num_points)

        resonance_vals = []
        frequency_sweeps = []
        gains = np.zeros((len(self.config['res_freq_ge']), num_points))

        synth = SynthHD('/dev/ttyACM0')
        synth[0].frequency = pump_freq
        synth[0].power = pump_power
        synth[0].enable = True
        time.sleep(2)

        for n in range(num_points):
            amps = np.zeros((len(self.config['res_freq_ge']), len(fpts)))
            for index, f in enumerate(tqdm(fpts)):
                self.config["res_freq_ge"] = f
                prog = SingleToneSpectroscopyProgram(soccfg, reps=self.exp_cfg["reps"], final_delay=0.5,
                                                     cfg=self.config)
                iq_list = prog.acquire(soc, soft_avgs=self.exp_cfg["rounds"], progress=False)
                for i in range(len(self.config['res_freq_ge'])):
                    amps[i][index] = np.abs(iq_list[i][:, 0] + 1j * iq_list[i][:, 1])
                    gains[i][n] = np.max(amps[i]) - np.min(amps[i])
            amps = np.array(amps)
            frequency_sweeps.append(amps)

            freq_res = []
            for i in range(len(self.config['res_freq_ge'])):
                freq_res.append(fpts[np.argmin(amps[i])])
            resonance_vals.append(freq_res)
        synth[0].enable = False

        return resonance_vals, frequency_sweeps, gains

    def plot_res_sweeps(self, fpts, frequency_sweeps, pump_power, pump_freq, num_points):
        plt.figure(figsize=(12, 8))

        # Set larger font sizes
        plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })
        for meas_index in range(num_points):
            for i in range(len(self.config['res_freq_ge'])):
                plt.subplot(2, 2, i + 1)
                plt.plot(fpts.T[i], frequency_sweeps[meas_index][i], '-', linewidth=1.5,
                         label=(meas_index+1))

                plt.xlabel("Frequency (MHz)", fontweight='normal')
                plt.ylabel("Amplitude (a.u)", fontweight='normal')
                plt.title(f"Resonator {i + 1}", pad=10)
                plt.legend(loc='upper left', fontsize='6', title='Meas Num')

        # Add a main title to the figure
        plt.suptitle(f"Resonance With TWPA: {pump_freq/1e9} GHz, {pump_power} dBm", fontsize=24, y=0.95)

        plt.tight_layout(pad=2.0)
        outerFolder_expt = os.path.join(self.outerFolder, 'TWPA_opt')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt, f"{formatted_datetime}_TWPAcons_res_sweep.png")
        plt.savefig(file_name, dpi=300)
        plt.close()
        return

    def plot_TWPA_gains(self, pump_power, pump_freq, num_points, gains):
        plt.figure(figsize=(12, 8))

        # Set larger font sizes
        plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })
        for i in range(len(self.config['res_freq_ge'])):
            plt.subplot(2, 2, i + 1)
            plt.scatter(range(1, num_points+1), gains[i])

            plt.xlabel("Measurement Number", fontweight='normal')
            plt.ylabel("Peak height (a.u)", fontweight='normal')
            plt.title(f"Resonator {i + 1}", pad=10)

        # Add a main title to the figure
        plt.suptitle(f"Amplitude Heights With TWPA: {pump_freq/1e9} GHz, {pump_power} dBm", fontsize=24, y=0.95)

        plt.tight_layout(pad=2.0)
        outerFolder_expt = os.path.join(self.outerFolder, 'TWPA_opt')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt, f"{formatted_datetime}_TWPAcons_heights.png")
        plt.savefig(file_name, dpi=300)
        plt.close()
        print(f"For TWPA at {pump_freq/1e9} GHz, {pump_power} dBm:")
        for q in range(len(self.config['res_freq_ge'])):
            max_gain = np.max(gains[q])
            min_gain = np.min(gains[q])
            print(f"Q{q + 1} Max Height {max_gain}")
            print(f"Q{q + 1} Min Height {min_gain}")
        return