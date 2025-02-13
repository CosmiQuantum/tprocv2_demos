from build_task import *
from build_state import *
from expt_config import *
import matplotlib.pyplot as plt
import numpy as np
import csv
import datetime
import time

from NetDrivers import E36300

class AllQubitTomographyMeasurement:
    def __init__(self, outerFolder, experiment):
        self.outerFolder = outerFolder
        self.Q13_BiasPS = E36300('192.168.0.44', server_port=5025)
        self.Q4_BiasPS = E36300('192.168.0.41', server_port=5025)
        self.q1_expt_name = "tomography_ge_q1"
        self.q2_expt_name = "tomography_ge_q2"
        self.q3_expt_name = "tomography_ge_q3"
        self.q4_expt_name = "tomography_ge_q4"
        self.experiment = experiment
        self.q1_exp_cfg = expt_cfg[self.q1_expt_name]
        self.q2_exp_cfg = expt_cfg[self.q2_expt_name]
        self.q3_exp_cfg = expt_cfg[self.q3_expt_name]
        self.q4_exp_cfg = expt_cfg[self.q4_expt_name]
        self.q_config = all_qubit_state(self.experiment)
        self.q1_exp_cfg = add_qubit_experiment(expt_cfg, self.q1_expt_name, 0)
        self.q1_config = {**self.q_config['Q0'], **self.q1_exp_cfg}
        self.q2_exp_cfg = add_qubit_experiment(expt_cfg, self.q2_expt_name, 1)
        self.q2_config = {**self.q_config['Q1'], **self.q2_exp_cfg}
        self.q3_exp_cfg = add_qubit_experiment(expt_cfg, self.q3_expt_name, 2)
        self.q3_config = {**self.q_config['Q2'], **self.q3_exp_cfg}
        self.q4_exp_cfg = add_qubit_experiment(expt_cfg, self.q4_expt_name, 3)
        self.q4_config = {**self.q_config['Q3'], **self.q4_exp_cfg}

        print(f'Q1 Tomography configuration: ', self.q1_config)
        print(f'Q2 Tomography configuration: ', self.q2_config)
        print(f'Q3 Tomography configuration: ', self.q3_config)
        print(f'Q4 Tomography configuration: ', self.q4_config)

    def allq_run_tomography(self, soccfg, soc, start_volt, stop_volt, volt_pts, rounds, plot=True, save=True):

        vsweep = np.linspace(start_volt, stop_volt, volt_pts, endpoint=True)
        self.set_up_PS()
        self.save_metadata(vsweep, rounds)
        self.bias_sweep(soccfg, soc, vsweep, rounds, plot_data = plot, save_data = save)

        return

    def set_up_PS(self):
        # Bias_PS_ip = ['192.168.0.44', '192.168.0.44', '192.168.0.44', '192.168.0.41']  # IP address of bias PS (qubits 1-3 are the same PS)
        # Bias_ch = [1, 2, 3, 1]  # Channel number of qubit 1-4 on associated PS

        self.Q13_BiasPS.setVoltage(0, 1)
        self.Q13_BiasPS.enable(1)
        self.Q13_BiasPS.setVoltage(0, 2)
        self.Q13_BiasPS.enable(2)
        self.Q13_BiasPS.setVoltage(0, 3)
        self.Q13_BiasPS.enable(3)
        self.Q4_BiasPS.setVoltage(0, 1)
        self.Q4_BiasPS.enable(1)
        return

    def bias_sweep(self, soccfg, soc, vsweep, total_rounds, plot_data=False, save_data=True):
        #overall for loop for total # of rounds:
        for r in total_rounds:
            ## so round_num starts at 1 not 0
            round_num = r + 1

            ## prepare signal arrays
            Q1_Iarr = []
            Q1_Qarr = []
            Q2_Iarr = []
            Q2_Qarr = []
            Q3_Iarr = []
            Q3_Qarr = []
            Q4_Iarr = []
            Q4_Qarr = []

            ## sweep voltage and take data on all 4 qubits
            for index, v in enumerate(vsweep):
                self.Q13_BiasPS.setVoltage(v, 1)
                self.Q13_BiasPS.setVoltage(v, 2)
                self.Q13_BiasPS.setVoltage(v, 3)
                self.Q4_BiasPS.setVoltage(v, 1)

                ## Q1
                q1_tomography = TomographyProgram(soccfg, reps=self.q1_config['reps'], final_delay=self.q1_config['relax_delay'],
                                                  cfg=self.q1_config)
                q1_iq_list = q1_tomography.acquire(soc, soft_avgs=self.q1_config['rounds'], progress=True)
                q1I = q1_iq_list[0][0, 0]
                q1Q = q1_iq_list[0][0, 1]
                Q1_Iarr.append(q1I)
                Q1_Qarr.append(q1Q)

                ## Q2
                q2_tomography = TomographyProgram(soccfg, reps=self.q2_config['reps'], final_delay=self.q2_config['relax_delay'],
                                                  cfg=self.q2_config)
                q2_iq_list = q2_tomography.acquire(soc, soft_avgs=self.q2_config['rounds'], progress=True)
                q2I = q2_iq_list[1][0, 0]
                q2Q = q2_iq_list[1][0, 1]
                Q2_Iarr.append(q2I)
                Q2_Qarr.append(q2Q)

                ## Q3
                q3_tomography = TomographyProgram(soccfg, reps=self.q3_config['reps'], final_delay=self.q3_config['relax_delay'],
                                                  cfg=self.q3_config)
                q3_iq_list = q3_tomography.acquire(soc, soft_avgs=self.q3_config['rounds'], progress=True)
                # print(np.shape(iq_list))
                q3I = q3_iq_list[2][0, 0]
                q3Q = q3_iq_list[2][0, 1]
                Q3_Iarr.append(q3I)
                Q3_Qarr.append(q3Q)

                ## Q4
                q4_tomography = TomographyProgram(soccfg, reps=self.q4_config['reps'], final_delay=self.q4_config['relax_delay'],
                                                  cfg=self.q4_config)
                q4_iq_list = q4_tomography.acquire(soc, soft_avgs=self.q4_config['rounds'], progress=True)
                q4I = q4_iq_list[3][0, 0]
                q4Q = q4_iq_list[3][0, 1]
                Q4_Iarr.append(q4I)
                Q4_Qarr.append(q4Q)

            self.Q13_BiasPS.setVoltage(0, 1)
            self.Q13_BiasPS.setVoltage(0, 2)
            self.Q13_BiasPS.setVoltage(0,3)
            self.Q4_BiasPS.setVoltage(0,1)

            ## put all the data together
            all_data = np.array([Q1_Iarr, Q1_Qarr, Q2_Iarr, Q2_Qarr, Q3_Iarr, Q3_Qarr, Q4_Iarr, Q4_Qarr])

            ## get time to give to save and plot funcs so they have the same timestamp
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")

            ## save and plot
            if save_data:
                self.save_all_tomography(all_data, round_num, formatted_datetime)
            if plot_data:
                self.plot_all_tomography(vsweep, all_data, round_num, formatted_datetime)

            return

    def save_metadata(self, vsweep, total_rounds):
        outerFolder_expt = os.path.join(self.outerFolder, 'repeated_tomography')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt, f"Tomography_Metadata_AllQs_{formatted_datetime}")
        np.savez(f"{file_name}", q1_cfg = self.q1_config, q2_cfg = self.q2_config, q3_cfg = self.q3_config, q4_cfg = self.q4_config,
                 vsweep = vsweep, tot_rounds = total_rounds)
        return

    def save_all_tomography(self, alldata, round_num, formatted_datetime):
        outerFolder_expt = os.path.join(self.outerFolder, 'repeated_tomography')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        file_name = os.path.join(outerFolder_expt, f"Tomography_AllQs_R{round_num}_{formatted_datetime}")
        np.savez(f"{file_name}", all_xi_xq=alldata)

        ## Data for all qubits is saved together in one array of the form: [q1i, q1q, q2i, q2q, q3i, q3q, q4i, q4q]
        ## where each of these is a 1d array - see how to access in plot_all_tomography function

        return

    def plot_all_tomography(self, vsweep, alldata, round_num, formatted_datetime):
        plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })

        for q in range(0, 4):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex='all')
            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=16)
            ax1.tick_params(axis='both', which='major', labelsize=14)
            ax1.plot(vsweep * 1000, alldata[q*2], linewidth=2)  # I_arr might be the wrong shape!

            ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=16)
            ax2.set_xlabel("Applied Voltage Bias (mV)", fontsize=16)
            ax2.plot(vsweep * 1000, alldata[q*2 + 1], linewidth=2)  # Q_arr might be the wrong shape!

            fig.suptitle(
                f"Charge Tomography Q{q+1}, Round {round_num}", fontsize=20)
            plt.tight_layout()

            plt.subplots_adjust(top=0.9)

            outerFolder_expt = os.path.join(self.outerFolder, 'repeated_tomography')
            self.experiment.create_folder_if_not_exists(outerFolder_expt)
            file_name = os.path.join(outerFolder_expt, f"Tomography_Q{q+1}_R{round_num}_{formatted_datetime}_.png")
            fig.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close(fig)
        return


class TomographyMeasurement:
    def __init__(self, QubitIndex, outerFolder, experiment):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "tomography_ge"
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
            #time.sleep(2)

            tomography = TomographyProgram(soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'], cfg=self.config)
            iq_list = tomography.acquire(soc, soft_avgs=self.config['rounds'], progress=True)
            #print(np.shape(iq_list))
            I = iq_list[self.QubitIndex][0, 0]
            Q = iq_list[self.QubitIndex][0, 1]
            amps = np.sqrt(np.abs(I + 1j *Q))
            #print(I)
            I_arr.append(I)
            Q_arr.append(Q)
            amps_arr.append(amps)
        #BiasPS.disable(Bias_ch[qubit_index])
        BiasPS.setVoltage(0, Bias_ch[qubit_index])
        #print(I_arr)

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
        ax1.set_ylabel("I Amplitude (a.u.)", fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.plot(vsweep*1000, I_arr, linewidth=2)   # I_arr might be the wrong shape!

        ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=16)
        ax2.set_xlabel("Applied Voltage Bias (mV)", fontsize=16)
        ax2.plot(vsweep*1000, Q_arr, linewidth=2)   # Q_arr might be the wrong shape!

        fig.suptitle(f"Charge Tomography Q{self.QubitIndex + 1} \n Wait time: {round(self.exp_cfg['wait_time'], 3)} us, qfreq = 4574.53 MHz, 50 pt", fontsize=20)
        plt.tight_layout()

        plt.subplots_adjust(top=0.9)

        outerFolder_expt = os.path.join(self.outerFolder, 'tomography')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
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
                       mask=[0, 1, 2, 3],
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
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
                       phase=cfg['qubit_phase'],  # + cfg['wait_time']*360*cfg['ramsey_freq'], # current phase + time * 2pi * ramsey freq #how to do this for tomography?
                       gain=cfg['pi_amp'] / 2,
                      )

        #self.add_loop("loop", cfg["steps"])  # number of times; should be 1


    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse1", t=0)  # play probe pulse
        self.delay_auto(cfg['wait_time'])  # wait_time after last pulse
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse2", t=0)  # play probe pulse
        self.delay_auto(0.01)  # wait_time after last pulse
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])


## drive at center freq and set qubit phase to 0 (for both pulses)

## look at how it's done in section_005_single_shot_ge,