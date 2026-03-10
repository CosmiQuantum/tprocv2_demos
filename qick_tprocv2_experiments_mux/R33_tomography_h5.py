import os.path

from IPython.core.pylabtools import figsize

from build_task import *
from build_state import *
from expt_config import *
import matplotlib.pyplot as plt
import numpy as np
import h5py
import json
import datetime
import time
import threading
import queue

from NetDrivers import E36300
from NetDrivers import Keithley2400

class VoltageLogger(threading.Thread):
    def __init__(self, log_func):
        super().__init__(daemon=True)
        self.log_func = log_func
        self.q = queue.Queue()
        self.results = queue.Queue()
        self.start()

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            try:
                result_queue = item[-1]
                flag = self.log_func(*item[:-1])
                result_queue.put(flag)
            except Exception as e:
                print(f"[VoltageLogger] Error: {e}")
                result_queue.put(-1)
            finally:
                self.q.task_done()

    def submit(self, *args):
        self.q.put(args)

    def stop(self):
        self.q.put(None)

class TomographyMeasurement:
    def __init__(self, QubitIndex, outerFolder, experiment, num_qubits, res_len, freq_offset, unmasking_resgain=False, progress=True):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.progress = progress
        self.error_log = os.path.join(self.outerFolder, "errors.log")
        self.expt_name = "tomography_ge"
        self.experiment = experiment
        self.Qubit = 'Q' + str(self.QubitIndex)
        #print(self.Qubit)
        self.exp_cfg = expt_cfg[self.expt_name]
        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [self.QubitIndex]

        self.q_config = all_qubit_state(self.experiment, num_qubits)
        self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}

        self.config['res_length'] = res_len[self.QubitIndex]
        good_res_freq_list = [base + offset for base, offset in zip(self.config['res_freq_ge'], freq_offset)]
        self.config['res_freq_ge'] = good_res_freq_list

        print(f'Q {self.QubitIndex + 1} Tomography configuration: ', self.config)

        #HDF5 info
        self.h5_path = None
        self.block_size = 10000
        self.current_max_rows = 0
        self.rows_written = 0

    def run_tomography(self, soccfg, soc, start_volt, stop_volt, volt_pts, rounds, plot=True, save=False):

        vsweep = np.linspace(start_volt, stop_volt, volt_pts, endpoint=True)
        vsweep = np.round(vsweep, 3) #rounds to 3 decimenal places to match the voltage supply output

        # Create saving file (metadata + data in H5 file)
        if save:
            self.create_h5_file(vsweep, rounds)

        #self.save_metadata(vsweep, rounds)
        # Run tomography with saving
        try:
            self.bias_sweep(soccfg, soc, vsweep, rounds, plot_data=plot, save_data=save)
        finally:
            self.truncate_h5_file()

        return

    def create_h5_file(self, vsweep, rounds):
        folder_data = os.path.join(self.outerFolder, 'study_data')
        folder_plots = os.path.join(self.outerFolder, 'documentation')
        self.experiment.create_folder_if_not_exists(folder_data)
        self.experiment.create_folder_if_not_exists(folder_plots)

        now = datetime.datetime.now()
        self.file_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        self.h5_path = os.path.join(folder_data, f"Tomography_Q{self.QubitIndex+1}_{self.file_timestamp}.h5")

        # Initial block of rows
        initial_rows = min(self.block_size, rounds)
        npts = len(vsweep)

        # Create appendable h5 file
        with h5py.File(self.h5_path, 'w') as f:

            # Save static metadata
            f.create_dataset("vsweep", data = vsweep)
            f.create_dataset("config", data = np.string_(json.dumps(self.config)))
            f.create_dataset("total_rounds", data = rounds)


            # Appendable dataset for qdata, shape (rounds, 2, npts)
            f.create_dataset(
                "qdata",
                shape = (initial_rows, 2, npts),
                maxshape = (None, 2, npts),
                chunks = (1, 2, npts),
                dtype = float
            )

            # Appendable dataset to flag rows with bad voltage settings
            f.create_dataset(
                "voltage_check",
                shape = (initial_rows, npts),
                maxshape=(None, npts),
                chunks=(1, npts),
                dtype='i1',     # small int: 0, 1
                fillvalue= -2   # uninitialized value
            )

            # Appendable dataset for timestamps
            dt = h5py.string_dtype(encoding='ascii')
            f.create_dataset(
                "timestamps",
                shape = (initial_rows,),
                maxshape = (None,),
                chunks = (1,),
                dtype = dt
            )
            # Track rows written
            f.attrs['rows_written'] = 0
        self.current_max_rows = initial_rows
        self.rows_written = 0

        print(f"Created HDF5 file: {self.h5_path}")
        return

    def append_round_to_h5(self, qdata, volt_flags, timestamp):
        """
        qdata: shape (2, npts)
        timestamps : "YYYY-MM-DD_HH-MM-SS"
        """
        with h5py.File(self.h5_path, "a") as f:
            ds = f["qdata"]
            vs = f["voltage_check"]
            ts = f["timestamps"]
            # Grow dataset if needed
            if self.rows_written >= self.current_max_rows:
                # Get another block
                new_max = self.current_max_rows + self.block_size
                ds.resize((new_max, 2, qdata.shape[1]))
                vs.resize((new_max, vs.shape[1]))
                ts.resize((new_max,))
                self.current_max_rows = new_max
                print(f"New allocated block, total rows now: {new_max}")

            # Or just append data
            ds[self.rows_written, :, :] = qdata
            vs[self.rows_written, :] = volt_flags
            ts[self.rows_written] = timestamp

            self.rows_written += 1
            f.attrs['rows_written'] = self.rows_written
        return

    def truncate_h5_file(self):
        if self.rows_written == self.current_max_rows:
            return
        with h5py.File(self.h5_path, "a") as f:
            ds = f["qdata"]
            vs = f["voltage_check"]
            ts = f["timestamps"]
            ds.resize((self.rows_written, 2, ds.shape[2]))
            vs.resize((self.rows_written, vs.shape[1]))
            ts.resize((self.rows_written,))
            print(f"File truncated to {self.rows_written} rounds")
            f.attrs["rows_written"] = self.rows_written
        return

    def log_voltage_error(self, BiasPS, voltage, set_voltage, round_num, timestamp):
        ## Check voltage setting. If doesn't match, log.
        try:
            #actual_raw = BiasPS.getVoltage(ch)
            #print(f"raw {set_voltage}")
            #If it's a list, extract
            if isinstance(set_voltage, list) or isinstance(set_voltage, tuple):
                set_voltage = set_voltage[0]
                #print(f"first index {set_voltage}")

            # conversion to float, check if not a value
            try:
                actual = float(set_voltage)
                #print(f"float {actual}")
            except Exception:
                with open(self.error_log, "a") as f:
                    f.write(f"{timestamp} - Round {round_num} - non-numeric readback: '{set_voltage}' for {voltage} V setpoint \n")
                print(f"[Error] Round {round_num} - non-numeric readback: '{set_voltage}'")
                return 0    # bad readback

            # check if it's a different value
            if not np.isclose(actual, voltage, atol=1e-4):
                with open(self.error_log, "a") as f:
                    f.write(f"{timestamp} - Round {round_num} - voltage mismatch: set = {voltage} V, readback = {actual} V\n")
                print(f"[Error] Round {round_num} - voltage mismatch: set = {voltage} V, readback = {actual} V")
                return 0 # wrong readback

            return 1 # voltage ok
        # other errors: communication, etc
        except Exception as e:
            with open(self.error_log, "a") as f:
                f.write(f"{timestamp} - Round {round_num} - exception during voltage readback for {voltage} V setting: {e}\n")
            print(f"[Error] Round {round_num} - exception during voltage readback: {e}")
            return -1 #other exception

    def init_bias_source_Keithley(self):
        bias_source = Keithley2400(server_ip = "192.168.0.45", server_port = 4001)
        bias_source.clearErrors()
        bias_source.reset()
        bias_source.initializeVoltageSource(vrange=0.2, current_limit=2e-2, enable_output=False)
        return bias_source

    def bias_sweep(self, soccfg, soc, vsweep, rounds, plot_data=False, save_data=True):
        # Bias_PS_ip = ['192.168.0.44', '192.168.0.44', '192.168.0.44',
        #               '192.168.0.41']  # IP address of bias PS (qubits 1-3 are the same PS)
        # Bias_ch = [1, 2, 3, 1]  # Channel number of qubit 1-4 on associated PS
        # qubit_index = int(self.QubitIndex)
        #
        # voltage_logger = VoltageLogger(self.log_voltage_error)
        #
        # BiasPS = E36300(Bias_PS_ip[qubit_index], server_port=5025)
        #
        # set_v = BiasPS.setVoltage(0, Bias_ch[qubit_index])
        # BiasPS.enable(Bias_ch[qubit_index])

        bias_source = self.init_bias_source_Keithley()
        bias_source.setSourceVoltage(0)
        bias_source.setOutputState(enable=True)

        point_duration = 3.0 #s

        for round_num in range(rounds):
            print(f"Round {round_num}")
            #start_time = time.time()

            ## get round timestamp
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")

            #prepare signal and saving arrays
            I_arr = []
            Q_arr = []
            volt_flags = []

            # create tomography object
            tomography = TomographyProgram(soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'], cfg=self.config)

            for index, v in enumerate(vsweep):
                #start_time = time.time()
                point_start = time.time()
                # initialize flag
                flag = -10

                result_queue = queue.Queue()

                # try:
                #     set_v = BiasPS.setVoltage(v, Bias_ch[qubit_index])
                #     #time.sleep(1)
                #     voltage_logger.submit(BiasPS, v, set_v, round_num, formatted_datetime, result_queue)
                #
                #     # flag = -3
                #     # flag = self.log_voltage_error(BiasPS, v, set_v, round_num, formatted_datetime)
                # except Exception as e:
                #     with open(self.error_log, "a") as f:
                #         f.write(
                #             f"{formatted_datetime} - Round {round_num} - exception during {v} V setpoint: {e} \n")
                #     print(f"Couldn't bias the qubit: {e}")
                #     result_queue.put(-1)

                try:
                    bias_source.setSourceVoltage(v)
                    # time.sleep(2)
                    print(bias_source.measureVoltage())
                    # print('voltage set')
                except Exception as e:
                    with open(self.error_log, "a") as f:
                        f.write(
                                f"{formatted_datetime} - Round {round_num} - exception during {v} V setpoint: {e} \n")
                    print(f"Couldn't bias qubits: {e}")
                    result_queue.put(-1)

                    #flag = -1
                #flag = result_queue.get()
                #volt_flags.append(flag)

                try:
                    iq_list = tomography.acquire(soc, soft_avgs=self.config['rounds'], progress=self.progress)

                    I = iq_list[self.QubitIndex][0, 0]
                    Q = iq_list[self.QubitIndex][0, 1]

                except Exception as e:
                    with open(self.error_log, "a") as f:
                        f.write(f"{formatted_datetime} - Round {round_num} - Voltage pt {v} - Error during tomography acquisition: {e}\n")
                    print(f"[Error] Round {round_num} - Exception during tomography: {e}")

                    I = np.nan
                    Q = np.nan

                # check the voltage and then append the flag
                # if flag == -10:
                #     flag = self.log_voltage_error(BiasPS, v, Bias_ch[qubit_index], round_num, formatted_datetime)
                # else:
                #     flag = flag_e
                # volt_flags.append(flag)

                try:
                    flag = result_queue.get(timeout=0.1)
                except queue.Empty:
                    flag = -2 #logging timed out or still running
                volt_flags.append(flag)
                I_arr.append(I)
                Q_arr.append(Q)

                elapsed = time.time() - point_start
                #print(elapsed)
                remaining = point_duration - elapsed

                if remaining > 0:
                    time.sleep(remaining)

                # end_time = time.time()
                # round_time = end_time-start_time
                # print(f"[Info] -  Voltage completed in {round_time} sec")

            #BiasPS.disable(Bias_ch[qubit_index])
            #BiasPS.setVoltage(0, Bias_ch[qubit_index])

            ## put all the data together
            q_data = np.array([I_arr, Q_arr])

            # Save data
            if save_data:
                self.append_round_to_h5(q_data, volt_flags, formatted_datetime)

                # Flush every 1,000 rounds
                if (round_num + 1) % 1000 == 0:
                    with h5py.File(self.h5_path, "a") as f:
                        f.flush()
                    print(f"[Checkpoint] Saved through round {round_num}")

            ## Plot
            if plot_data:
                self.plot_tomography(vsweep, q_data, volt_flags, round_num, formatted_datetime)

            # end_time = time.time()
            # round_time = end_time-start_time
            # print(f"[Info] - Round {round_num} completed in {round_time} sec")
        bias_source.setSourceVoltage(0)
        bias_source.setOutputState(enable=False)
        return


    def plot_tomography(self, vsweep, qdata, volt_flags, round_num, formatted_datetime):
        plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })

        volt_flags = np.array(volt_flags)
        round_bad = np.any(volt_flags <= 0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex='all')
        ax1.set_ylabel("I Amplitude (a.u.)", fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=16)
        ax2.set_xlabel("Applied Voltage Bias (mV)", fontsize=16)

        if round_bad:
            ax1.plot(vsweep * 1000, qdata[0], color = 'r', linewidth=2)
            ax2.plot(vsweep * 1000, qdata[1], color = 'r', linewidth=2)
        else:
            ax1.plot(vsweep * 1000, qdata[0], linewidth=2)  # I_arr might be the wrong shape!
            ax2.plot(vsweep * 1000, qdata[1], linewidth=2)  # Q_arr might be the wrong shape!

        fig.suptitle(
            f"Charge Tomography Q{self.QubitIndex+1}, Round {round_num}", fontsize=20)
        plt.tight_layout()

        plt.subplots_adjust(top=0.9)

        folder_plots = os.path.join(self.outerFolder, 'documentation')
        self.experiment.create_folder_if_not_exists(folder_plots)
        file_name = os.path.join(folder_plots, f"Tomography_Q{self.QubitIndex+1}_R{round_num}_{formatted_datetime}_.png")
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


## drive at center freq and set qubit phase to 0 (for both pulses)

## look at how it's done in section_005_single_shot_ge,