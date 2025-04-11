from build_task import *
from build_state import *
from expt_config import *
import matplotlib.pyplot as plt
import numpy as np
import csv
import datetime
#import time

from NetDrivers import E36300

class AllQubitParityMeasurement:
    def __init__(self, outerFolder, number_of_qubits, experiment,  fids , ssf_I_g, ssf_I_e, ssf_Q_g, ssf_Q_e ):
        self.outerFolder = outerFolder
        self.Q13_BiasPS = E36300('192.168.0.44', server_port=5025)
        self.Q4_BiasPS = E36300('192.168.0.41', server_port=5025)
        self.q1_expt_name = "Parity_ge_q1"
        self.q2_expt_name = "Parity_ge_q2"
        self.q3_expt_name = "Parity_ge_q3"
        self.q4_expt_name = "Parity_ge_q4"
        self.experiment = experiment
        self.number_of_qubits = number_of_qubits
        self.q1_exp_cfg = expt_cfg[self.q1_expt_name]
        self.q2_exp_cfg = expt_cfg[self.q2_expt_name]
        self.q3_exp_cfg = expt_cfg[self.q3_expt_name]
        self.q4_exp_cfg = expt_cfg[self.q4_expt_name]
        self.q_config = all_qubit_state(self.experiment,  self.number_of_qubits)
        self.q1_exp_cfg = add_qubit_experiment(expt_cfg, self.q1_expt_name, 0)
        self.q1_config = {**self.q_config['Q0'], **self.q1_exp_cfg}
        self.q2_exp_cfg = add_qubit_experiment(expt_cfg, self.q2_expt_name, 1)
        self.q2_config = {**self.q_config['Q1'], **self.q2_exp_cfg}
        self.q3_exp_cfg = add_qubit_experiment(expt_cfg, self.q3_expt_name, 2)
        self.q3_config = {**self.q_config['Q2'], **self.q3_exp_cfg}
        self.q4_exp_cfg = add_qubit_experiment(expt_cfg, self.q4_expt_name, 3)
        self.q4_config = {**self.q_config['Q3'], **self.q4_exp_cfg}
        self.fids=fids
        self.ssf_I_g = ssf_I_g
        self.ssf_I_e = ssf_I_e
        self.ssf_Q_g = ssf_Q_g
        self.ssf_Q_e = ssf_Q_e


        print(f'Q1 Parity configuration: ', self.q1_config)
        print(f'Q2 Parity configuration: ', self.q2_config)
        print(f'Q3 Parity configuration: ', self.q3_config)
        print(f'Q4 Parity configuration: ', self.q4_config)

    def allq_run_Parity(self, soccfg, soc, start_volt, stop_volt, volt_pts, rounds, plot=True, save=True):

        vsweep = np.linspace(start_volt, stop_volt, volt_pts, endpoint=True)
        self.set_up_PS()
        start_datetime = self.save_metadata(vsweep, rounds)
        self.bias_sweep(soccfg, soc, vsweep, rounds, start_datetime, plot_data = plot, save_data = save)

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

    def bias_sweep(self, soccfg, soc, vsweep, total_rounds, start_time, plot_data=False, save_data=True):
        #overall for loop for total # of rounds:
        for r in range(total_rounds):
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

            self.Q13_BiasPS.setVoltage(0.108, 1)
            self.Q13_BiasPS.setVoltage(0.093, 2)
            self.Q13_BiasPS.setVoltage(0.062, 3)
            self.Q4_BiasPS.setVoltage(0.083, 1)
            ## sweep voltage and take data on all 4 qubits
            start_scan_time = time.time()
            for index, v in enumerate(vsweep):
                print(f'v index={index}')

                ## Q1
                q1_Parity = ParityProgram(soccfg, reps=self.q1_config['reps'], final_delay=self.q1_config['relax_delay'],
                                                  cfg=self.q1_config)
                q1_iq_list = q1_Parity.acquire(soc, soft_avgs=self.q1_config['rounds'], progress=True)
                q1I = q1_iq_list[0][0, 0]
                q1Q = q1_iq_list[0][0, 1]
                Q1_Iarr.append(q1I)
                Q1_Qarr.append(q1Q)

                ## Q2
                q2_Parity = ParityProgram(soccfg, reps=self.q2_config['reps'], final_delay=self.q2_config['relax_delay'],
                                                  cfg=self.q2_config)
                q2_iq_list = q2_Parity.acquire(soc, soft_avgs=self.q2_config['rounds'], progress=True)
                q2I = q2_iq_list[1][0, 0]
                q2Q = q2_iq_list[1][0, 1]
                Q2_Iarr.append(q2I)
                Q2_Qarr.append(q2Q)

                ## Q3
                q3_Parity = ParityProgram(soccfg, reps=self.q3_config['reps'], final_delay=self.q3_config['relax_delay'],
                                                  cfg=self.q3_config)
                q3_iq_list = q3_Parity.acquire(soc, soft_avgs=self.q3_config['rounds'], progress=True)
                # print(np.shape(iq_list))
                q3I = q3_iq_list[2][0, 0]
                q3Q = q3_iq_list[2][0, 1]
                Q3_Iarr.append(q3I)
                Q3_Qarr.append(q3Q)

                ## Q4
                q4_Parity = ParityProgram(soccfg, reps=self.q4_config['reps'], final_delay=self.q4_config['relax_delay'],
                                                  cfg=self.q4_config)
                q4_iq_list = q4_Parity.acquire(soc, soft_avgs=self.q4_config['rounds'], progress=True)
                q4I = q4_iq_list[3][0, 0]
                q4Q = q4_iq_list[3][0, 1]
                Q4_Iarr.append(q4I)
                Q4_Qarr.append(q4Q)
            end_time = time.time()
            elapsed_time = end_time - start_scan_time
            print('elapsed_time', elapsed_time)
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
                self.save_all_Parity(all_data, round_num, formatted_datetime, start_time, elapsed_time)
            if plot_data:
                self.plot_all_Parity(vsweep, all_data, round_num, formatted_datetime, start_time)

        return

    def save_metadata(self, vsweep, total_rounds):
        now = datetime.datetime.now()
        start_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        outerFolder_expt = os.path.join(self.outerFolder, f'repeated_Parity_{start_datetime}')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        file_name = os.path.join(outerFolder_expt, f"Parity_Metadata_AllQs_{start_datetime}")
        np.savez(f"{file_name}", q1_cfg = self.q1_config, q2_cfg = self.q2_config, q3_cfg = self.q3_config, q4_cfg = self.q4_config,
                 vsweep = vsweep, tot_rounds = total_rounds , fids=self.fids, ssf_Ig=self.ssf_I_g  , ssf_Ie=self.ssf_I_e  , ssf_Qg=self.ssf_Q_g , ssf_Qe=self.ssf_Q_e  )
        return start_datetime

    def save_all_Parity(self, alldata, round_num, formatted_datetime, start_datetime,  elapsed_time):
        outerFolder_expt = os.path.join(self.outerFolder, f'repeated_Parity_{start_datetime}')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        file_name = os.path.join(outerFolder_expt, f"Parity_AllQs_R{round_num}_{formatted_datetime}")
        np.savez(f"{file_name}", all_xi_xq=alldata, elapsed_time=elapsed_time)

        ## Data for all qubits is saved together in one array of the form: [q1i, q1q, q2i, q2q, q3i, q3q, q4i, q4q]
        ## where each of these is a 1d array - see how to access in plot_all_Parity function

        return

    def plot_all_Parity(self, vsweep, alldata, round_num, formatted_datetime, start_datetime):
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
                f"Charge Parity Q{q+1}, Round {round_num}", fontsize=20)
            plt.tight_layout()

            plt.subplots_adjust(top=0.9)

            outerFolder_expt = os.path.join(self.outerFolder, f'repeated_Parity_{start_datetime}')
            self.experiment.create_folder_if_not_exists(outerFolder_expt)
            file_name = os.path.join(outerFolder_expt, f"Parity_Q{q+1}_R{round_num}_{formatted_datetime}_.png")
            fig.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close(fig)
        return


class ParityMeasurement:
    def __init__(self, QubitIndex, outerFolder, number_of_qubits, experiment , fids, ssf_I_g, ssf_I_e, ssf_Q_g,
                                           ssf_Q_e):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "Parity_ge_q" + str(1+QubitIndex)  #"Parity_ge"
        self.experiment = experiment
        self.number_of_qubits = number_of_qubits
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.q_config = all_qubit_state(self.experiment, self.number_of_qubits )
        self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
        self.fids = fids
        self.ssf_I_g = ssf_I_g
        self.ssf_I_e = ssf_I_e
        self.ssf_Q_g = ssf_Q_g
        self.ssf_Q_e = ssf_Q_e
        # self.fids=fids
        print(f'Q {self.QubitIndex + 1} Parity configuration: ', self.config)

    def run_Parity(self, soccfg, soc, start_volt, stop_volt, volt_pts):
        now = datetime.datetime.now()
        vsweep = np.linspace(start_volt, stop_volt, volt_pts, endpoint=True)
        I_list, Q_list, amps , timeTaken= self.bias_sweep(soccfg, soc, vsweep)
        outerFolder_expt = os.path.join(self.outerFolder, 'Parity')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)

        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        name=os.path.join(outerFolder_expt,
                                      f"{formatted_datetime}_Parity_Q{self.QubitIndex + 1}_parity_data")
        np.savez(name,  I_list=I_list, Q_list=Q_list, amps=amps, q_cfg = self.q_config,     vsweep = vsweep, timeTaken=timeTaken , fids=self.fids, ssf_Ig=self.ssf_I_g  , ssf_Ie=self.ssf_I_e  , ssf_Qg=self.ssf_Q_g , ssf_Qe=self.ssf_Q_e )
        save = False
        plot = True,
        if plot:
            self.plot_Parity(vsweep, I_list, Q_list, timeTaken)

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
            print(f'index={index}')
            # BiasPS.setVoltage(v, Bias_ch[qubit_index])
            # time.sleep(2)
            startT=time.time()
            Parity = ParityProgram(soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'], cfg=self.config)
            iq_list = Parity.acquire(soc, soft_avgs=self.config['rounds'], progress=True)
            #print(np.shape(iq_list))
            I = iq_list[self.QubitIndex][0, 0]
            Q = iq_list[self.QubitIndex][0, 1]
            amps = np.sqrt(np.abs(I + 1j *Q))
            #print(I)
            I_arr.append(I)
            Q_arr.append(Q)
            amps_arr.append(amps)
            endT=time.time()
            timeTaken=endT-startT
        BiasPS.disable(Bias_ch[qubit_index])
        BiasPS.setVoltage(0, Bias_ch[qubit_index])
        #print(I_arr)

        return I_arr, Q_arr, amps_arr, timeTaken

    def save_arrays(self, vsweep, I_arr, Q_arr, amps_arr):
        outerFolder_expt = os.path.join(self.outerFolder, 'Parity')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")

        file_name_vsweep = os.path.join(outerFolder_expt,
                                      f"{formatted_datetime}_Parity_Q{self.QubitIndex + 1}_vsweep")
        file_name_Iarr = os.path.join(outerFolder_expt,
                                      f"{formatted_datetime}_Parity_Q{self.QubitIndex + 1}_Iarr")
        file_name_Qarr = os.path.join(outerFolder_expt,
                                      f"{formatted_datetime}_Parity_Q{self.QubitIndex + 1}_Qarr")
        file_name_Amparr = os.path.join(outerFolder_expt,
                                        f"{formatted_datetime}_Parity_Q{self.QubitIndex + 1}_Amparr")
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

    def plot_Parity(self, vsweep, I_arr, Q_arr, timeTaken):

        plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })
        t=np.linspace(0,timeTaken*len(vsweep), len(vsweep))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex='all')
        ax1.set_ylabel("I Amplitude (a.u.)", fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.plot(t, I_arr, linewidth=2)   # I_arr might be the wrong shape!

        ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=16)
        ax2.set_xlabel("time (s)", fontsize=16)
        ax2.plot(t, Q_arr, linewidth=2)   # Q_arr might be the wrong shape!

        fig.suptitle(f"Charge Parity Q{self.QubitIndex + 1} \n Wait time: {round(self.exp_cfg['wait_time'], 3)} us, qfreq = 4574.53 MHz, 50 pt", fontsize=20)
        plt.tight_layout()

        plt.subplots_adjust(top=0.9)

        outerFolder_expt = os.path.join(self.outerFolder, 'Parity')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt, f"{formatted_datetime}_Parity_Q{self.QubitIndex + 1}.png")
        fig.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return

############################################################################################################################

class EFParityMeasurement:
    def __init__(self, QubitIndex, outerFolder, number_of_qubits, experiment ,  fids):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "Parity_ef_q" + str(1+QubitIndex)  #"Parity_ge"
        self.experiment = experiment
        self.number_of_qubits = number_of_qubits
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.q_config = all_qubit_state(self.experiment, self.number_of_qubits )
        self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
        # self.fids=fids
        print(f'Q {self.QubitIndex + 1} Parity configuration: ', self.config)

    def run_Parity(self, soccfg, soc, start_volt, stop_volt, volt_pts, plot=True, save=False):
        now = datetime.datetime.now()
        vsweep = np.linspace(start_volt, stop_volt, volt_pts, endpoint=True)
        I_list, Q_list, amps , timeTaken= self.bias_sweep(soccfg, soc, vsweep)
        outerFolder_expt = os.path.join(self.outerFolder, 'Parity')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)

        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        name=os.path.join(outerFolder_expt,
                                      f"{formatted_datetime}_Parity_Q{self.QubitIndex + 1}_parity_data")
        np.savez(name,  I_list=I_list, Q_list=Q_list, amps=amps, q_cfg = self.q_config,     vsweep = vsweep, timeTaken=timeTaken)
        if plot:
            self.plot_Parity(vsweep, I_list, Q_list, timeTaken)

        if save:
            self.save_arrays(vsweep, I_list, Q_list, amps)

        return

    def bias_sweep(self, soccfg, soc, vsweep):
        Bias_PS_ip = ['192.168.0.44', '192.168.0.44', '192.168.0.44',
                      '192.168.0.41']  # IP address of bias PS (qubits 1-3 are the same PS)
        # Bias_ch = [1, 2, 3, 1]  # Channel number of qubit 1-4 on associated PS
        # qubit_index = int(self.QubitIndex)
        #
        # BiasPS = E36300(Bias_PS_ip[qubit_index], server_port=5025)
        #
        # BiasPS.setVoltage(0, Bias_ch[qubit_index])
        # BiasPS.enable(Bias_ch[qubit_index])

        #prepare signal arrays
        I_arr = []
        Q_arr = []
        amps_arr = []

        for index, v in enumerate(vsweep):
            print(f'index={index}')
            # BiasPS.setVoltage(v, Bias_ch[qubit_index])
            # time.sleep(2)
            startT=time.time()
            EFParity = EFParityProgram(soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'], cfg=self.config)
            iq_list = EFParity.acquire(soc, soft_avgs=self.config['rounds'], progress=True)
            #print(np.shape(iq_list))
            I = iq_list[self.QubitIndex][0, 0]
            Q = iq_list[self.QubitIndex][0, 1]
            amps = np.sqrt(np.abs(I + 1j *Q))
            #print(I)
            I_arr.append(I)
            Q_arr.append(Q)
            amps_arr.append(amps)
            endT=time.time()
            timeTaken=endT-startT
        # BiasPS.disable(Bias_ch[qubit_index])
        # BiasPS.setVoltage(0, Bias_ch[qubit_index])
        #print(I_arr)

        return I_arr, Q_arr, amps_arr, timeTaken

    def save_arrays(self, vsweep, I_arr, Q_arr, amps_arr):
        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")

        file_name_vsweep = os.path.join(outerFolder_expt,
                                      f"{formatted_datetime}_Parity_Q{self.QubitIndex + 1}_vsweep")
        file_name_Iarr = os.path.join(outerFolder_expt,
                                      f"{formatted_datetime}_Parity_Q{self.QubitIndex + 1}_Iarr")
        file_name_Qarr = os.path.join(outerFolder_expt,
                                      f"{formatted_datetime}_Parity_Q{self.QubitIndex + 1}_Qarr")
        file_name_Amparr = os.path.join(outerFolder_expt,
                                        f"{formatted_datetime}_Parity_Q{self.QubitIndex + 1}_Amparr")
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

    def plot_Parity(self, vsweep, I_arr, Q_arr, timeTaken):

        plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })
        t=np.linspace(0,timeTaken*len(vsweep), len(vsweep))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex='all')
        ax1.set_ylabel("I Amplitude (a.u.)", fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.plot(t, I_arr, linewidth=2)   # I_arr might be the wrong shape!

        ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=16)
        ax2.set_xlabel("time (s)", fontsize=16)
        ax2.plot(t, Q_arr, linewidth=2)   # Q_arr might be the wrong shape!

        fig.suptitle(f"Charge Parity Q{self.QubitIndex + 1} \n Wait time: {round(self.exp_cfg['wait_time'], 3)} us, qfreq = 4574.53 MHz, 50 pt", fontsize=20)
        plt.tight_layout()

        plt.subplots_adjust(top=0.9)

        outerFolder_expt = os.path.join(self.outerFolder, 'Parity')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt, f"{formatted_datetime}_Parity_Q{self.QubitIndex + 1}.png")
        fig.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return


#################################################################################################################################

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
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse1",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'] ,
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse2",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'] -90,  #cfg['wait_time']*90*cfg['freqdisp'], # current phase + time * 2pi * ramsey freq #how to do this for Parity?
                       gain=cfg['pi_amp'] ,
                      )

        #self.add_loop("loop", cfg["steps"])  # number of times; should be 1


    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse1", t=0)  # play probe pulse
        self.delay_auto(cfg['wait_time'])  # wait_time after last pulse
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse2", t=0)  # play probe pulse
        self.delay_auto(0.01)  # wait_time after last pulse
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

######################################################################
class EFParityProgram(AveragerProgramV2):
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

        self.add_gauss(ch=qubit_ch, name="geramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="pi_ge",
                       style="arb",
                       envelope="geramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        self.add_gauss(ch=qubit_ch, name="eframp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse1",
                       style="arb",
                       envelope="eframp",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_ef_amp'] / 2,
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse2",
                       style="arb",
                       envelope="eframp",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'] -90,  #cfg['wait_time']*90*cfg['freqdisp'], # current phase + time * 2pi * ramsey freq #how to do this for Parity?
                       gain=cfg['pi_ef_amp'] / 2,
                      )

        #self.add_loop("loop", cfg["steps"])  # number of times; should be 1


    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="pi_ge", t=0)  # play ge pi pulse
        self.delay_auto(t=0, tag='waiting after pi')  # Wait til qubit pulse is done before proceeding

        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse1", t=0)  # play probe pulse
        self.delay_auto(cfg['wait_time'])  # wait_time after last pulse
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse2", t=0)  # play probe pulse

        self.pulse(ch=self.cfg["qubit_ch"], name="pi_ge", t=0)  # play ge pi pulse
        self.delay_auto(t=0, tag='waiting after pi')  # Wait til qubit pulse is done before proceeding

        self.delay_auto(0.01)  # wait_time after last pulse
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

## drive at center freq and set qubit phase to 0 (for both pulses)

## look at how it's done in section_005_single_shot_ge,