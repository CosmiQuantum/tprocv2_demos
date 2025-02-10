from backup.tprocv2_demos.qick_tprocv2_experiments_mux_nexus.ReadoutOpt_gain_freqs import QubitIndex
from build_task import *
from build_state import *
from expt_config import *
import matplotlib.pyplot as plt
import numpy as np
import csv
import datetime
import time

from NetDrivers import E36300


class TomographyMeasurement:
    def __init__(self, QubitIndex, outerFolder, experiment):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "Tomography"
        self.experiment = experiment
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.q_config = all_qubit_state(self.experiment)
        self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}

        print(f'Q {self.QubitIndex + 1} Tomography configuration: ', self.config)

    def run_tomography(self, soccfg, soc, start_volt, stop_volt, volt_pts, plot=True, save=False):

        vsweep = np.linspace(start_volt, stop_volt, volt_pts, endpoint=True)
        I_list, Q_list, amps = self.bias_sweep(soccfg, soc, vsweep)

        if plot:
            self.plot_tomography(vsweep, I_list, Q_list)

        if save:
            self.save_arrays(vsweep, I_list, Q_list, amps)

        return

    def bias_sweep(self, soccfg, soc, vsweep):
        Bias_PS_ip = ['192.168.0.44', '192.168.0.44', '192.168.0.44',
                      '192.168.0.41']  # IP address of bias PS (qubits 1-3 are the same PS)
        Bias_ch = [1, 2, 3, 1]  # Channel number of qubit 1-4 on associated PS
        qubit_index = int(self.QubitIndex)

        BiasPS = E36300(Bias_PS_ip[qubit_index], server_port=5025)

        BiasPS.setVoltage(0, Bias_ch[qubit_index])
        BiasPS.enable(Bias_ch[qubit_index])

        #prepare signal arrays
        I_arr = []
        Q_arr = []
        amps_arr = []

        for index, v in enumerate(vsweep):
            BiasPS.setVoltage(v, Bias_ch[qubit_index])
            time.sleep(8)

            tomography = TomographyProgram(soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'], cfg=self.config)
            iq_list = tomography.acquire(soc, soft_avgs=self.config['rounds'], progress=True)
            I = iq_list[self.QubitIndex][0, :, 0]
            Q = iq_list[self.QubitIndex][0, :, 1]
            amps = np.sqrt(np.abs(I + 1j *Q))
            print(I)
            I_arr.append(I)
            Q_arr.append(Q)
            amps_arr.append(amps)
        BiasPS.disable(Bias_ch[qubit_index])
        print(I_arr)

        return I_arr, Q_arr, amps_arr

    def save_arrays(self, vsweep, I_arr, Q_arr, amps_arr):
        outerFolder_expt = os.path.join(self.outerFolder, 'tomography')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")

        file_name_vsweep = os.path.join(outerFolder_expt,
                                      f"{formatted_datetime}_Tomography_Q{self.QubitIndex + 1}_vsweep")
        file_name_Iarr = os.path.join(outerFolder_expt,
                                      f"{formatted_datetime}_Tomography_Q{self.QubitIndex + 1}_Iarr")
        file_name_Qarr = os.path.join(outerFolder_expt,
                                      f"{formatted_datetime}_Tomography_Q{self.QubitIndex + 1}_Qarr")
        file_name_Amparr = os.path.join(outerFolder_expt,
                                        f"{formatted_datetime}_Tomography_Q{self.QubitIndex + 1}_Amparr")
        with open(f"{file_name_vsweep}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(vsweep)
        with open(f"{file_name_Iarr}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(I_arr)
        with open(f"{file_name_Qarr}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(Q_arr)
        with open(f"{file_name_Amparr}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(amps_arr)

        return

    def plot_tomography(self, vsweep, I_arr, Q_arr):

        plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex='all')
        ax1.set_ylable("I Amplitude (a.u.)", fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.plot(vsweep, I_arr, linewidth=2)   # I_arr might be the wrong shape!

        ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
        ax2.set_xlabel("Applied Voltage Bias (mV)", fontsize=20)
        ax2.plot(vsweep, Q_arr, linewidth=2)   # Q_arr might be the wrong shape!

        fig.suptitle(f"Charge Tomography Q{self,QubitIndex + 1}", fontsize=24)
        plt.tight_layout()

        plt.subplots_adjust(top=0.93)

        outerFolder_expt = os.path.join(self.outerFolder, 'tomography')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-$M-%S")
        file_name = os.path.join(outerFolder_expt, f"{formatted_datetime}_Tomography_Q{self.QubitIndex + 1}.png")
        fig.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return


class TomographyProgram(AveragerProgramV2):
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
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse1",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['tomography_ge'] ,
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'] / 2,
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse2",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['tomography_ge'],
                       phase=cfg['qubit_phase'],  # + cfg['wait_time']*360*cfg['ramsey_freq'], # current phase + time * 2pi * ramsey freq #how to do this for tomography?
                       gain=cfg['pi_amp'] / 2,
                      )


    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse1", t=0)  # play probe pulse
        self.delay_auto(cfg['wait_time'])  # wait_time after last pulse
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse2", t=0)  # play probe pulse
        self.delay_auto(0.01)  # wait_time after last pulse
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])


## drive at center freq and set qubit phase to 0 (for both pulses)

## look at how it's done in section_005_single_shot_ge,