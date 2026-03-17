import os.path

from build_task import *
from build_state import *
from expt_config import *

import datetime
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json

from NetDrivers import Keithley2400

class AllQubitTomographyMeasurement:
    def __init__(self, outerFolder, experiment, num_qubits, res_len, freq_offset, measure_qubits=[1, 2, 3, 4], unmasking_resgain = True, progress=True):
        self.outerFolder = outerFolder
        ### stuff for bias PS
        self.progress = progress
        self.error_log = os.path.join(self.outerFolder, "errors.log") ##Error log for qick errors (and volt errors if we use them)
        self.time_log = os.path.join(self.outerFolder, "time.log") ## Log for rnd times - maybe don't need
        self.measure_qubits = measure_qubits
        self.n_qubits = len(measure_qubits)

        self.q1_expt_name = "tomography_ge_q1"
        self.q2_expt_name = "tomography_ge_q2"
        self.q3_expt_name = "tomography_ge_q3"
        self.q4_expt_name = "tomography_ge_q4"
        self.experiment = experiment
        self.q1_exp_cfg = expt_cfg[self.q1_expt_name]
        self.q2_exp_cfg = expt_cfg[self.q2_expt_name]
        self.q3_exp_cfg = expt_cfg[self.q3_expt_name]
        self.q4_exp_cfg = expt_cfg[self.q4_expt_name]

        if unmasking_resgain:
            self.q1_exp_cfg["list_of_all_qubits"] = 0
            self.q2_exp_cfg["list_of_all_qubits"] = 1
            self.q3_exp_cfg["list_of_all_qubits"] = 2
            self.q4_exp_cfg["list_of_all_qubits"] = 3

        self.q_config = all_qubit_state(self.experiment, num_qubits)
        self.q1_exp_cfg = add_qubit_experiment(expt_cfg, self.q1_expt_name, 0)
        self.q1_config = {**self.q_config['Q0'], **self.q1_exp_cfg}
        self.q2_exp_cfg = add_qubit_experiment(expt_cfg, self.q2_expt_name, 1)
        self.q2_config = {**self.q_config['Q1'], **self.q2_exp_cfg}
        self.q3_exp_cfg = add_qubit_experiment(expt_cfg, self.q3_expt_name, 2)
        self.q3_config = {**self.q_config['Q2'], **self.q3_exp_cfg}
        self.q4_exp_cfg = add_qubit_experiment(expt_cfg, self.q4_expt_name, 3)
        self.q4_config = {**self.q_config['Q3'], **self.q4_exp_cfg}

        self.q1_config['res_length'] = res_len[0]
        self.q2_config['res_length'] = res_len[1]
        self.q3_config['res_length'] = res_len[2]
        self.q4_config['res_length'] = res_len[3]

        good_res_freq_list = [base + offset for base, offset in zip(self.q1_config['res_freq_ge'], freq_offset)]

        self.q1_config['res_freq_ge'] = good_res_freq_list
        self.q2_config['res_freq_ge'] = good_res_freq_list
        self.q3_config['res_freq_ge'] = good_res_freq_list
        self.q4_config['res_freq_ge'] = good_res_freq_list

        print(f'Q1 Tomography configuration: ', self.q1_config)
        print(f'Q2 Tomography configuration: ', self.q2_config)
        print(f'Q3 Tomography configuration: ', self.q3_config)
        print(f'Q4 Tomography configuration: ', self.q4_config)

        ## h5 stuff
        self.rows_written = 0
        self.qubit_configs = {
            1: self.q1_config,
            2: self.q2_config,
            3: self.q3_config,
            4: self.q4_config,
        }

    def run_tomography(self, soccfg, soc, start_volt, stop_volt, volt_pts, rounds, plot=False, plot_together=False, save=True):
        vsweep = np.linspace(start_volt, stop_volt, volt_pts, endpoint = True)
        vsweep = np.round(vsweep, 3)

        if save:
            file_timestamp = self.create_h5_file(vsweep, rounds)
            self.h5_file = h5py.File(self.h5_path, "a")

        bias_source = self.init_bias_source()
        try:
            self.bias_sweep(soccfg, soc, bias_source, vsweep, rounds, plot_data = plot, plot_together = plot_together, save_data = save)
        except KeyboardInterrupt:
            print(f"User stopped run - data saved through last completed round (R{self.rows_written})")
            raise
        finally:
            if save:
                self.h5_file.flush()
                self.h5_file.close()
                self.h5_file = None
            bias_source.setSourceVoltage(0)
            bias_source.setOutputState(enable=False)
        if save:
            return file_timestamp

    def create_h5_file(self, vsweep, rounds):
        folder_data = os.path.join(self.outerFolder, 'study_data')
        folder_plots = os.path.join(self.outerFolder, 'documentation')
        self.experiment.create_folder_if_not_exists(folder_data)
        self.experiment.create_folder_if_not_exists(folder_plots)

        now = datetime.datetime.now()
        self.file_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        self.qubits_meas = "".join(str(q) for q in self.measure_qubits)
        self.h5_path = os.path.join(folder_data, f"Tomography_Qs{self.qubits_meas}_{self.file_timestamp}.h5")

        npts = len(vsweep)
        configs = {
            qid: self.qubit_configs[qid]
            for qid in self.measure_qubits
        }

        with h5py.File(self.h5_path, "w") as f:

            # Save static metadata
            f.create_dataset("vsweep", data = vsweep)
            f.create_dataset("configs", data = np.string_(json.dumps(configs)))
            f.create_dataset("total_rounds", data = rounds)
            f.create_dataset(
                "qdata",
                shape = (rounds, self.n_qubits, 2, npts),
                maxshape = (rounds, self.n_qubits, 2, npts),
                chunks = (1, self.n_qubits, 2, npts),
                dtype = float
            )

            f.create_dataset(
                "voltage_check",
                shape = (rounds, self.n_qubits, npts),
                maxshape = (rounds, self.n_qubits, npts),
                chunks = (1, self.n_qubits, npts),
                dtype= 'i1',
                fillvalue = -2
            )

            f.create_dataset(
                "timestamps",
                shape=(rounds,),
                maxshape=(rounds,),
                chunks = (1,),
                dtype = h5py.string_dtype("ascii")
                )

            f.create_dataset(
                "round_duration_sec",
                shape = (rounds, ),
                maxshape = (rounds,),
                chunks=(1,),
                dtype=float
            )

            f.create_dataset("qubits", data = np.array(self.measure_qubits, dtype=np.int32))

            f.attrs['rows_written'] = 0
            f.attrs['n_qubits'] = self.n_qubits

        self.rows_written = 0

        print(f"Created HDF5 file: {self.h5_path}")

    def append_round_to_h5(self, qdata, volt_flags, timestamp, rnd_time):
        """
        qdata: shape (n_qubits, 2, npts)
        timestamps: "YYYY-MM-DD_HH-MM-SS"
        """
        if self.h5_file is None:
            raise RuntimeError("HDF5 file not open")
        f = self.h5_file
        row = self.rows_written

        f["qdata"][row,:,:,:] = qdata
        f["voltage_check"][row, :,:] = volt_flags
        f["timestamps"][row] = timestamp
        f["round_duration_sec"][row] = rnd_time

        self.rows_written += 1
        f.attrs["rows_written"] = self.rows_written

    def init_bias_source(self):
        bias_source = Keithley2400(server_ip = "192.168.0.45", server_port = 4001)
        bias_source.clearErrors()
        bias_source.reset()
        bias_source.initializeVoltageSource(vrange=0.2, current_limit=2e-2, enable_output=False)
        return bias_source

    def log_voltage_error(self):
        print('hi world')
        #idk if this makes sense anymore with new supply, figure out

    # def moving_avg_check(self, scan_avgs, avg_window, thresholds):
    #     if len(scan_avgs) < avg_window:
    #         avg = np.average(scan_avgs)
    #     else:
    #         avg = np.average(scan_avgs[-(avg_window):])
    #     if thresholds[0] < avg or thresholds[1] > avg:
    #         print("Data average is outside of threshold. Stopping to get new parameters from RR.")
    #         return 1
    #     else:
    #         return 0

    def bias_sweep(self, soccfg, soc, bias_source, vsweep, rounds, plot_data=False, plot_together = False, save_data=True):
        n_pts = len(vsweep)

        bias_source.setSourceVoltage(0)
        bias_source.setOutputState(enable=True)

        q1_tomography = TomographyProgram(soccfg, reps=self.q1_config['reps'], final_delay=self.q1_config['relax_delay'],
                                          cfg = self.q1_config)
        q2_tomography = TomographyProgram(soccfg, reps=self.q2_config['reps'], final_delay=self.q2_config['relax_delay'],
                                          cfg=self.q2_config)
        q3_tomography = TomographyProgram(soccfg, reps=self.q3_config['reps'], final_delay=self.q3_config['relax_delay'],
                                          cfg=self.q3_config)
        q4_tomography = TomographyProgram(soccfg, reps=self.q4_config['reps'], final_delay=self.q4_config['relax_delay'],
                                          cfg=self.q4_config)

        #scan_averages = np.empty_like((self.n_qubits, ))

        for round_num in range(rounds):
            print(f"Round {round_num}")

            ## get round timestamp
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")

            ## prepare signal and saving arrays
            I_arr = np.zeros((self.n_qubits, n_pts))
            Q_arr = np.zeros((self.n_qubits, n_pts))
            volt_flags = np.zeros((self.n_qubits, n_pts), dtype = np.int8)

            for index, v in enumerate(vsweep):
                try:
                    bias_source.setSourceVoltage(v)
                    #time.sleep(2)
                    bias_source.measureVoltage()
                    #print('voltage set')
                except Exception as e:
                    print(f"Couldn't bias qubits: {e}")

                for qi, qid in enumerate(self.measure_qubits):  #qi: index of qubit in measurement list, qid: physical qubit "name" in measurment list
                    try:
                        if qid == 1:
                            q1_iq_list = q1_tomography.acquire(soc, soft_avgs = self.q1_config['rounds'], progress=self.progress)
                            I = q1_iq_list[0][0,0]
                            Q = q1_iq_list[0][0,1]
                        if qid == 2:
                            q2_iq_list = q2_tomography.acquire(soc, soft_avgs=self.q2_config['rounds'], progress=self.progress)
                            I = q2_iq_list[1][0, 0]
                            Q = q2_iq_list[1][0, 1]
                        if qid == 3:
                            q3_iq_list = q3_tomography.acquire(soc, soft_avgs=self.q3_config['rounds'], progress=self.progress)
                            I = q3_iq_list[2][0, 0]
                            Q = q3_iq_list[2][0, 1]
                        if qid == 4:
                            q4_iq_list = q4_tomography.acquire(soc, soft_avgs=self.q4_config['rounds'], progress=self.progress)
                            I = q4_iq_list[3][0, 0]
                            Q = q4_iq_list[3][0, 1]
                        I_arr[qi, index] = I
                        Q_arr[qi, index] = Q

                    except Exception as e:
                        with open(self.error_log, "a") as f:
                            f.write(
                                f"{formatted_datetime} - Q{qid} - Round {round_num} - Voltage pt {v} - Error during tomography acquisition: {e}\n")
                        print(f"[Error] Round {round_num} Q{qid} - Exception during tomography: {e}")

                        I_arr[qi, index] = np.nan
                        Q_arr[qi, index] = np.nan

            ## put all round data together
            q_data = np.stack([I_arr, Q_arr], axis=1)

            ## Get total round time
            rnd_end = datetime.datetime.now()
            rnd_time = (rnd_end - now).total_seconds()

            ## Save data
            if save_data:
                self.append_round_to_h5(q_data, volt_flags, formatted_datetime, rnd_time)

                # Flush every 10 rounds
                if (round_num + 1) % 10 == 0:
                    self.h5_file.flush()

            # with open(self.time_log, "a") as f:
            #     f.write(f"{formatted_datetime} | Round {round_num} | Duration {rnd_time:.2f} s\n")

            ## Plot data
            if plot_data:
                self.plot_all_tomography(vsweep, q_data, volt_flags, round_num, formatted_datetime, plot_together)

        bias_source.setSourceVoltage(0)
        bias_source.setOutputState(enable=False)
        return

    def plot_all_tomography(self, vsweep, qdata, volt_flags, round_num, formatted_datetime, plot_together = False):
        ### FIX PLOTTING
        plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })
        if plot_together:
            fig, (ax1, ax2) = plt.subplot(2, 1, figzise=(10,8), sharex='all')
            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=16)
            ax1.tick_params(axis='both', which='major', labelsize=14)
            ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=16)
            ax2.set_xlabel("Applied Voltage Bias (mV)", fontsize=16)
            for qi, qid in enumerate(self.measure_qubits):
                ax1.plot(vsweep * 1000, qdata[qi, 0, :], label = f"Q{qid}")
                ax2.plot(vsweep * 1000, qdata[qi, 1, :], label = f"Q{qid}")

            fig.suptitle(f"Charge Tomography Q{self.qubits_meas}, Round {round_num} \n {formatted_datetime}", fontsize=20)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)

            folder_plots = os.path.join(self.outerFolder, 'documentation')
            self.experiment.create_folder_if_not_exists(folder_plots)
            file_name = os.path.join(folder_plots, f"Tomography_Q{self.qubits_meas}_R{round_num}_{formatted_datetime}.png")
            fig.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            for qi, qid in enumerate(self.measure_qubits):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex='all')
                ax1.plot(vsweep * 1000, qdata[qi, 0, :])
                ax1.set_ylabel("I Amplitude (a.u.)", fontsize=16)
                ax1.tick_params(axis='both', which='major', labelsize=14)
                ax2.plot(vsweep * 1000, qdata[qi, 1, :])
                ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=16)
                ax2.set_xlabel("Applied Voltage Bias (mV)", fontsize=16)

                fig.suptitle(f"Charge Tomography Q{qid}, Round {round_num} \n {formatted_datetime}", fontsize=20)
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)

                folder_plots = os.path.join(self.outerFolder, 'documentation')
                self.experiment.create_folder_if_not_exists(folder_plots)
                file_name = os.path.join(folder_plots, f"Tomography_Q{qid}_R{round_num}_{formatted_datetime}.png")
                fig.savefig(file_name, dpi=300, bbox_inches='tight')
                plt.close(fig)
        return

class TomographyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        #print(cfg['res_length'])

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