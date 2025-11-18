import multiprocessing

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from qick.asm_v2 import AveragerProgramV2
from tqdm import tqdm
from build_state import *
from expt_config import *
import datetime
import logging
import pandas as pd
from scipy.signal import find_peaks
from multiprocessing import Pool, Process, cpu_count
import time

class SingleResonatorTimestreamWithDrive(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['dynro_ch'][0]
        gen_ch = cfg['gen_ch']
        drive_ch = cfg['drive_ch'] ## channel added to send pulse to drive resonator

        # play individual resonator pulse from MUX DAC
        self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch,
                         mux_freqs=[cfg['sensor_res_freq']],
                         mux_gains=cfg['sensor_gain'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])

        print(f"sensor resonator freq: {cfg['sensor_res_freq']}")
        print(f"drive resonator freq: {cfg['drive_res_freq']}")
        print(f"drive resonator gain: {cfg['drive_gain']}")
        print(f"sensor resonator gain: {cfg['sensor_gain']}")


        ## length of the rep

        readout_length =  cfg['pre_drive_delay'] + (cfg['drive_length'] * cfg['drive_reps']) + cfg['post_drive_delay']
        print(f"readout length: {readout_length} us")
        self.add_pulse(ch=gen_ch, name="mux_pulse",
                       style="const",
                       length=readout_length,
                       mask=cfg["res_mask"], #for a single resonator, this will always be [0] and is set in system_config
                       )

        ## set up drive pulse from fullspeed DAC
        self.declare_gen(ch=drive_ch, nqz=cfg['nqz_res']) #,mixer_freq=cfg['mixer_freq'])
        self.add_pulse(ch=drive_ch, name="drive_pulse",
                               style="const",
                               freq=cfg['drive_res_freq'],
                               length=cfg["drive_length"],
                               phase=cfg['res_phase'][0],
                               gain=cfg['drive_gain'],
                               )

        # dynamic readout on one channel

        self.declare_readout(ch=ro_ch, length=readout_length)
        self.add_readoutconfig(ch=ro_ch, name="ro",
                               freq=cfg['sensor_res_freq'], #entry res_freq
                               gen_ch=gen_ch,
                               outsel='product')
        self.send_readoutconfig(ch=ro_ch, name="ro", t=0)
        #self.delay_auto() #resets clock to zero

        ## trigger DDR4 buffer, data collection begins now
        if cfg['DDR4'] is True:
            #self.trigger(ros=cfg['dynro_ch'], pins=[0], t=cfg['trig_time'], ddr4=True)
            self.trigger(ros=cfg['dynro_ch'], pins=[0], t=0, ddr4=True)
            print("open DDR4 buffer")

    def _body(self, cfg):
        # send readout resonator pulse
        #self.delay_auto() #resets clock to 0
        self.pulse(ch=cfg['gen_ch'], name="mux_pulse", t=0)

        if cfg['DDR4'] is False:
            self.trigger(ros=cfg['dynro_ch'], pins=[0], t=cfg['trig_time'])

        # send drive resonator pulse
        #self.pulse(ch=cfg['drive_ch'], name="drive_pulse", t=cfg['pre_drive_delay'])
        for r in np.arange(0,cfg["drive_reps"]):
            self.pulse(ch=cfg['drive_ch'], name="drive_pulse", t=cfg['pre_drive_delay'] + (r * cfg['drive_length']))

        #self.delay(t=cfg['pre_drive_delay'])
        #self.delay(t=cfg['drive_length'])
        #self.delay(t=cfg['post_drive_delay'])


class ResonatorDrive:
    def __init__(self, DriveResonatorIndex, SensorResonatorIndex, number_of_resonators, studyDocumentationFolder, dataSetFolder, round_num, save_figs=False, experiment=None,
                 verbose=False, logger=None, drive_gain=None):
        self.DriveResonatorIndex = DriveResonatorIndex
        self.SensorResonatorIndex = SensorResonatorIndex
        self.round_num = round_num
        self.Resonator = "R0" #+ str(self.ResonatorIndex)
        self.number_of_resonators = number_of_resonators
        self.studyDocumentationFolder = studyDocumentationFolder
        self.dataSetFolder = dataSetFolder
        self.drive_gain = drive_gain

        if self.number_of_resonators == 1:
            self.expt_name = "single_res_timestream_drive"
        else:
            self.expt_name = "multi_res_timestream_drive"
        self.round_num = round_num
        self.save_figs = save_figs
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if experiment is not None:
            self.r_config = all_resonator_state(experiment, self.number_of_resonators)
            self.config = {**self.r_config[self.Resonator], **self.exp_cfg}
            self.logger.info(f'R {self.DriveResonatorIndex} Round {self.round_num} Res Spec configuration: {self.config}')
            if self.verbose: print(f'R {self.DriveResonatorIndex} Round {self.round_num} Res Spec configuration: ',
                                   self.config)
            if self.drive_gain is not None:
                self.config['drive_gain'] = self.drive_gain
                print(self.config['drive_gain'])

    ## define data saving functions
    def save_pandas(self, iq_ddr4, data_path, idx, multi_file=False):
                ### store data in a series object
                series_I = pd.Series(iq_ddr4[:, 0])
                series_Q = pd.Series(iq_ddr4[:, 1])

                if multi_file is False:
                    ## create one file and append batches of data
                    file_path = os.path.join(data_path,
                                             f"timestream_D{self.DriveResonatorIndex}_drive_gain_{self.drive_gain}_S{self.SensorResonatorIndex}.h5")
                    print(f"saving batch {idx} to {file_path}")

                    series_I.to_hdf(file_path, key="I", mode='a', format='fixed')
                    series_Q.to_hdf(file_path, key="Q", mode='a', format='fixed')

                else:
                    ### save batches of data in individual files labeled with their index
                    file_path = os.path.join(data_path,
                                             f"timestream_R{self.DriveResonatorIndex}_r{self.round_num}_i{idx}_{formatted_datetime}.h5")
                    print(f"saving batch {idx} to {file_path}")
                    series_I.to_hdf(file_path, key="I", mode='w', format='fixed')
                    series_Q.to_hdf(file_path, key="Q", mode='w', format='fixed')

    def multi_save(self, queue, iq_ddr4, data_path, idx):
                self.save_pandas(iq_ddr4, data_path, idx, multi_file=True)


    def run(self, threading=False, multi_file = False):

        if self.expt_name == "single_res_timestream_drive":
            self.config["sensor_res_freq"] = self.config["res_freq"][self.SensorResonatorIndex[0]] + self.config['offset']
            self.config["res_mask"] = [0]
            self.config["mixer_freq"] = self.config["sensor_res_freq"] + 300
            self.config["drive_res_freq"] = self.config["res_freq"][self.DriveResonatorIndex[0]]

            prog = SingleResonatorTimestreamWithDrive(self.experiment.soccfg, reps=self.config["reps"], final_delay=self.config["relax_delay"],
                                             cfg=self.config)


        elif self.expt_name == "multi_res_timestream":
            self.config['dynro_time'] = self.config['pulse_length']/len(self.config['res_idx'])
            prog = MultiResonatorTimestream(self.experiment.soccfg, reps=self.config["reps"],
                                             final_delay=self.config["relax_delay"],
                                             cfg=self.config)

        if self.config['DDR4'] is False:
            iq_list = prog.acquire_decimated(self.experiment.soc, rounds=self.config['rounds'])
            t = prog.get_time_axis(ro_index = self.config['dynro_ch'][0])
            timestep = t[1] - t[0]
            self.plot_timestream(iq_list[0], t)

        elif self.config['DDR4'] is True:
            #set number of transfers
            rep_length = self.config['pre_drive_delay'] + (self.config["drive_reps"] * self.config['drive_length']) + self.config['post_drive_delay'] + self.config['relax_delay']
            total_transfers = int(self.config['reps']*np.ceil(rep_length/(self.config['period'] * 128))) + 1000 + int(self.config['buffer_length']/self.config['period'])

            #arm DDR4 buffer
            self.experiment.soc.arm_ddr4(ch=self.config['dynro_ch'][0], nt=total_transfers)

            #run program
            prog.run_rounds(self.experiment.soc)

            ## set up data saving path
            formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            data_path = os.path.join(self.dataSetFolder,"study_data", "Data_h5", "timestream_data")
            self.experiment.create_folder_if_not_exists(data_path)

            ### transfer a few samples from the buffer to find the timestep and record
            iq_ddr4 = self.experiment.soc.get_ddr4(nt=10,start=int(401 + 128))
            t = prog.get_time_axis_ddr4(self.config['dynro_ch'][0], iq_ddr4)
            timestep = t[1] - t[0]
            start_time = time.time() ## marker to calculate data saving time

            if threading is False:
                ## save data without multi-processing
                #for idx in np.arange(0, int(np.floor(total_transfers / self.config['num_transfers']))):
                for idx in np.arange(0, 1):
                    #iq_ddr4 = self.experiment.soc.get_ddr4(nt=int(self.config['num_transfers']),
                                                       #start=int(401 + 128 * idx * self.config['num_transfers']))
                    iq_ddr4 = self.experiment.soc.get_ddr4(nt=total_transfers)
                    t = prog.get_time_axis_ddr4(self.config['dynro_ch'][0], iq_ddr4)



                    #self.stack_reps(iq_ddr4, t)

                    #self.plot_timestream(iq_ddr4, t)
                    ##

                    self.save_pandas(iq_ddr4, data_path, idx, multi_file=multi_file)

            else:
                ## speed up writing data with multi-processing
                #create a queue to store data
                queue = multiprocessing.Queue()
                processes=[]

                num_process = 8 #how many concurrent processes to allow

                for idx in np.arange(0,int(np.floor(total_transfers/self.config['num_transfers']))):
                    iq_ddr4 = self.experiment.soc.get_ddr4(nt=int(self.config['num_transfers']),
                                                       start=int(401 + 128 * idx * self.config['num_transfers']))

                    p = Process(target=self.multi_save, args=(queue, iq_ddr4, data_path, idx))
                    processes.append(p)

                    while len(multiprocessing.active_children()) > num_process:
                        time.sleep(0.10)

                    p.start()

                print(f"waiting for all processes {processes} to finish")
                for p in processes:
                    p.join()
                print("all processes complete")

            end_time = time.time()
            print(f"total data saving time: {end_time - start_time}")

        return iq_ddr4, t, timestep, self.config


    def stack_reps(self, iq_ddr4, t):
        #calculate number of timesteps in one rep
        time_per_rep = self.config['relax_delay'] + (self.config['drive_reps'] * self.config['drive_length']) + self.config['post_drive_delay'] + self.config['pre_drive_delay']
        timestep = t[1] - t[0]
        steps_per_rep = int(time_per_rep/timestep)
        # reshape iq_ddr4 by number of reps

        start = int(200/timestep) ##cut out first 200 us of data due to delay of opening DDR4 buffer
        end = (self.config['reps'] * steps_per_rep) + start

        I = iq_ddr4[start:end,0].reshape((self.config['reps'], -1))
        Q = iq_ddr4[start:end,1].reshape((self.config['reps'], -1))
        Iavg = np.average(I, axis=0)
        Qavg = np.average(Q, axis=0)
        t_rep = t[0:len(Iavg)]

        self.plot_timestream(np.transpose(np.vstack((Iavg, Qavg))), t_rep)


    def find_pulses(self, signal, t, batch_idx):
        sigma = np.std(signal)
        mean = np.mean(signal)
        #peaks = np.sort(np.concatenate((np.transpose(np.where(signal > (mean + self.config['sigma_factor']*sigma))),
                                        #np.transpose(np.where(signal < (mean - self.config['sigma_factor']*sigma))))))
        peaks, properties = find_peaks(signal, height=self.config['sigma_factor']*sigma+mean)
        #peaks = np.unique(np.round(peaks, decimals=-2)).astype(int)
        pulse_cuts = []
        t_cuts = []
        pulse_id = 0
        for peak in peaks:
            t_cut, signal_cut = self.cut_pulse(signal, t, peak)
            #if self.save_figs:
                #self.plot_pulse(t_cut, signal_cut, sigma, mean, pulse_id, batch_idx)
            pulse_cuts.append(signal_cut)
            t_cuts.append(t_cut)
            pulse_id=pulse_id+1

        return pulse_cuts, t_cuts

    def plot_pulse(self, t, pulse, sigma, mean, pulse_id, batch_id):
        fig, ax = plt.subplots()
        ax.plot(t, pulse,c='k',marker='o')
        ax.plot([t[0], t[len(t)-1]],[mean, mean],c='r',linestyle=':')
        ax.plot([t[0], t[len(t)-1]], [mean+self.config['sigma_factor']*sigma, mean+self.config['sigma_factor']*sigma],
                c='b', linestyle=':')
        ax.plot([t[0], t[len(t)-1]], [mean-self.config['sigma_factor']*sigma, mean-self.config['sigma_factor']*sigma],
                c='b',linestyle=':')
        ax.set_xlabel('time [us]')
        ax.set_ylabel('amplitude [a.u.]')
        ax.set_title(f'R{self.ResonatorIndex} pulse? {pulse_id} in batch {batch_id}')

        if self.save_figs:
            now = datetime.datetime.now() 
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(self.studyDocumentationFolder,
                                    f"R{self.ResonatorIndex}_{self.expt_name}_{formatted_datetime}_pulse_{pulse_id}_batch_{batch_id}")
            plt.savefig(f"{filename}.png")


    def cut_pulse(self, signal, t, peak):
        start_idx = np.max((0,int(np.ceil(peak - self.config['window']/(self.config['chunk_size']*self.config['period'])))))
        end_idx = np.min((int(np.ceil(peak + self.config['window']/(self.config['chunk_size']*self.config['period']))),len(signal)-1))
        t_window = t[start_idx:end_idx]
        signal_window = signal[start_idx:end_idx]

        return t_window, signal_window


    def average_timestream(self, iq_ddr4, t):

        chunk_size = self.config['chunk_size']

        # calculate number of timesteps in one rep
        time_per_rep = self.config['relax_delay'] + (self.config['drive_reps'] * self.config['drive_length']) + \
                       self.config['post_drive_delay'] + self.config['pre_drive_delay']
        timestep = t[1] - t[0]
        steps_per_rep = int(time_per_rep / timestep)
        # reshape iq_ddr4 by number of reps

        start = int(self.config['buffer_length']/timestep)  ##cut out first 200 us of data due to delay of opening DDR4 buffer
        end = (self.config['reps'] * steps_per_rep) + start
        I = iq_ddr4[start:end, 0]
        Q = iq_ddr4[start:end, 1]
        t2 = t[start:end]
        amps = np.sqrt(I**2 + Q**2)

        block_I = []
        block_Q = []
        block_amps = []
        block_t = []

        for r in np.arange(0,self.config['reps']):
            I_rep = I[(r*steps_per_rep):((r+1)*steps_per_rep)]
            temp_block_I =[np.mean(I_rep[i:i + chunk_size]) for i in range(0, len(I_rep), chunk_size)]
            block_I.append(temp_block_I)

            Q_rep = Q[(r*steps_per_rep):((r+1)*steps_per_rep)]
            temp_block_Q = [np.mean(Q_rep[i:i + chunk_size]) for i in range(0, len(Q_rep), chunk_size)]
            block_Q.append(temp_block_Q)

            amps_rep = amps[(r*steps_per_rep):((r+1)*steps_per_rep)]
            temp_block_amps = [np.mean(amps_rep[i:i + chunk_size]) for i in range(0, len(amps_rep), chunk_size)]
            block_amps.append(temp_block_amps)

            t_rep = t2[(r*steps_per_rep):((r+1)*steps_per_rep)]
            temp_block_t = [np.mean(t_rep[i:i + chunk_size]) for i in range(0, len(t_rep), chunk_size)]
            block_t.append(temp_block_t)


        # block_I = [np.mean(I[i:i + chunk_size]) for i in range(0, len(I), chunk_size)]
        # block_Q = [np.mean(Q[i:i + chunk_size]) for i in range(0, len(Q), chunk_size)]
        # block_amps = [np.mean(amps[i:i+chunk_size]) for i in range(0, len(amps), chunk_size)]
        # block_t = [np.mean(t2[i:i + chunk_size]) for i in range(0, len(t2), chunk_size)]
        amp_block_IQ = np.sqrt(np.square(block_I) + np.square(block_Q))

        return np.array(block_I).flatten(), np.array(block_Q).flatten(), np.array(block_amps).flatten(), np.array(amp_block_IQ).flatten(), np.array(block_t).flatten()

    def plot_timestreamv2(self, block_I, block_Q, block_amps, amp_block_IQ, block_t, fig, ax):

        # fig, ax = plt.subplots(3,1)
        # ax[0].plot(block_t, block_I, c='b',label=f'averaged I, chunk={self.config["chunk_size"]})')
        # ax[0].plot(block_t, block_Q, c='r', label=f'averaged Q, chunk={self.config["chunk_size"]})')
        # ax[0].set_ylabel('I/Q [a.u.]')
        # ax[0].set_xlabel('time [us]')
        # ax[0].legend()
        #
        # ax[1].plot(block_t, block_amps, c='b',label=f'averaged amplitude, chunk={self.config["chunk_size"]})')
        # ax[1].set_ylabel('amplitude [a.u.]')
        # ax[1].set_xlabel('time [us]')
        # ax[1].legend()
        #
        # ax[2].plot(block_t, amp_block_IQ, c='g',label=f'amplitude of averaged I,Q, chunk={self.config["chunk_size"]})')
        # ax[2].set_ylabel('amplitude [a.u.]')
        # ax[2].set_xlabel('time [us]')
        # ax[2].legend()

        # ax.plot(block_t, block_amps, c='b',label=f'averaged amplitude, chunk={self.config["chunk_size"]})')
        # ax.set_ylabel('amplitude [a.u.]')
        # ax.set_xlabel('time [us]')
        # ax.set_title(f"Drive Resonator R{self.DriveResonatorIndex[0]}")
        # #ax.legend()

        #ax2 = ax.twinx()
        ax.plot(block_t, amp_block_IQ,label=f"drive_gain={self.config['drive_gain']}")
        ax.set_ylabel('amplitude [a.u.]')
        ax.set_xlabel('time [us]')
        ax.set_title(f"Drive Resonator R{self.DriveResonatorIndex[0]}")
        #ax.set_ylim([0,0.1])
        #ax.set_xlim([0,200])


    def stack_repsv2(self, block_I, block_Q, block_amps, amp_block_IQ, block_t):    ## stack reps

        I_stack = np.average(block_I.reshape((self.config['reps'], -1)),axis=0)
        Q_stack = np.average(block_Q.reshape((self.config['reps'], -1)),axis=0)
        amp_stack = np.average(block_amps.reshape((self.config['reps'], -1)),axis=0)
        amp_block_IQ_stack = np.average(amp_block_IQ.reshape((self.config['reps'], -1)),axis=0)
        block_t_stack = block_t.reshape((self.config['reps'], -1))[0] - self.config['buffer_length']

        return I_stack, Q_stack, amp_stack, amp_block_IQ_stack, block_t_stack


    def plot_timestream(self, iq_ddr4, t):

        amps = np.sqrt(np.square(iq_ddr4[:,0]) + np.square(iq_ddr4[:,1]))
        #amps = np.abs(iq_ddr4[:,0] + 1j*iq_ddr4[:,1])
        ## convert to complex number, get phase, magnitude data
        iq_complex = iq_ddr4[:,0] + 1j * iq_ddr4[:,1]  ## FOR QICK I,Q DATA
        phase = np.unwrap(np.angle(iq_complex))  # radians
        linear_magnitude = np.abs(iq_complex)
        magnitude = np.log10(linear_magnitude) * 20  # dB scale


        series_I = pd.Series(iq_ddr4[:,0])
        series_Q = pd.Series(iq_ddr4[:,1])
        series_amps = pd.Series(amps)
        chunk_size = self.config['chunk_size']

        rolling_I = series_I.rolling(window=chunk_size).mean()
        # block_I = [series_I[i:i + chunk_size].mean() for i in range(0, len(series_I), chunk_size)]
        block_I = [np.mean(iq_ddr4[i:i + chunk_size,0]) for i in range(0, len(iq_ddr4[:,0]), chunk_size)]

        rolling_Q = series_Q.rolling(window=chunk_size).mean()
        # block_Q = [series_Q[i:i + chunk_size].mean() for in range(0, len(series_Q), chunk_size)]
        block_Q = [np.mean(iq_ddr4[i:i + chunk_size, 1]) for i in range(0, len(iq_ddr4[:,1]), chunk_size)]

        rolling_amps = series_amps.rolling(window=chunk_size).mean()
        block_amps = [series_amps[i:i + chunk_size].mean() for i in range(0, len(series_amps), chunk_size)]
        block_amps2 = np.sqrt(np.square(block_I) + np.square(block_Q))
        #block_amps2 = np.sqrt(np.square(rolling_I) + np.square(rolling_Q))

        series_t = pd.Series(t)
        block_t = [series_t[i:i + chunk_size].mean() for i in range(0, len(series_t), chunk_size)]

        ### plot data within one chunk
        chunk_idx = 50
        start_idx = chunk_idx*chunk_size
        end_idx = start_idx + chunk_size

        fig, ax = plt.subplots(4,1)
        ax[0].plot(t[start_idx:end_idx], iq_ddr4[start_idx:end_idx,0], c='b',label='raw I')
        #ax[0].plot(t[start_idx:end_idx], np.abs(iq_ddr4[start_idx:end_idx, 0]), c='k', label='|raw I|')
        ax[0].plot([t[start_idx], t[end_idx]], [np.mean(iq_ddr4[start_idx:end_idx,0]), np.mean(iq_ddr4[start_idx:end_idx,0])], label='average')
        ax[0].plot([t[start_idx], t[end_idx]], [block_I[chunk_idx], block_I[chunk_idx]],
                   label='chunk value for raw I')
        ax[0].set_xlabel('time [us]')
        ax[0].set_ylabel('I [a.u.]')
        ax[0].legend(bbox_to_anchor=(1, 1), loc='upper right')

        ax[1].plot(t[start_idx:end_idx], iq_ddr4[start_idx:end_idx, 1], c='b', label='raw Q')
        #ax[1].plot(t[start_idx:end_idx], np.abs(iq_ddr4[start_idx:end_idx, 1]), c='k', label='|raw Q|')
        ax[1].plot([t[start_idx], t[end_idx]], [np.mean(iq_ddr4[start_idx:end_idx,1]), np.mean(iq_ddr4[start_idx:end_idx,1])], label='average')
        ax[1].plot([t[start_idx], t[end_idx]], [block_Q[chunk_idx], block_Q[chunk_idx]],
                   label='chunk value for raw Q')
        ax[1].set_xlabel('time [us]')
        ax[1].set_ylabel('Q [a.u.]')
        ax[1].legend(bbox_to_anchor=(1, 1), loc='upper right')

        ax[2].plot(t[start_idx:end_idx], amps[start_idx:end_idx], c='b', label='raw amplitude')
        ax[2].plot([t[start_idx], t[end_idx]], [np.mean(amps[start_idx:end_idx]), np.mean(amps[start_idx:end_idx])], label='average')
        ax[2].plot([t[start_idx], t[end_idx]], [block_amps[chunk_idx], block_amps[chunk_idx]],label='chunk value for raw amplitude')
        ax[2].plot([t[start_idx], t[end_idx]], [block_amps2[chunk_idx], block_amps2[chunk_idx]],
                   label='chunk value for amplitude from chunked I,Q')
        ax[2].set_xlabel('time [us]')
        ax[2].set_ylabel('amplitude [a.u.]')
        ax[2].legend(bbox_to_anchor=(1, 1), loc='upper right')

        ax[3].plot(t[start_idx:end_idx], magnitude[start_idx:end_idx], c='b', label='magnitude')

        ax[3].set_xlabel('time [us]')
        ax[3].set_ylabel('magnitude [a.u.]')
        ax[3].legend(bbox_to_anchor=(1, 1), loc='upper right')

        ax4 = ax[3].twinx()
        ax4.plot(t[start_idx:end_idx], phase[start_idx:end_idx], c='r', label='phase')
        ax4.set_ylabel('phase [rad]')


        fig, ax = plt.subplots(4, 1)
        #ax[0].scatter(t, iq_ddr4[:,0],label='I',s=1,c='k')
        ax[0].plot(t, rolling_I, label='rolling average')
        ax[0].plot(block_t, block_I, label=f'block size {chunk_size}')
        ax[0].legend()
        ax[0].set_xlabel('time [us]')
        ax[0].set_ylabel('I [a.u.]')

        #ax[1].scatter(t, iq_ddr4[:,1],label='Q',s=1,c='k')
        ax[1].plot(t, rolling_Q, label='rolling average')
        ax[1].plot(block_t, block_Q, label=f'block size {chunk_size}')
        ax[1].legend()
        ax[1].set_xlabel('time [us]')
        ax[1].set_ylabel('Q [a.u.]')

        #ax[2].scatter(t, amps,label='amplitude',s=1,c='k')
        mean = np.mean(rolling_amps)
        sigma = np.std(rolling_amps)
        ax[2].plot(t, rolling_amps, label='rolling average')
        ax[2].plot(block_t, block_amps, label=f'block size {chunk_size}')
        #ax[2].plot(block_t, block_amps2, label=f'block version 2', zorder=-100)
        ax[2].plot([t[0], t[len(t)-1]],[mean, mean],c='r',linestyle=':')
        # ax[2].plot([t[0], t[len(t)-1]], [mean+self.config['sigma_factor']*sigma, mean+self.config['sigma_factor']*sigma],
        #         c='b', linestyle=':')
        # ax[2].plot([t[0], t[len(t)-1]], [mean-self.config['sigma_factor']*sigma, mean-self.config['sigma_factor']*sigma],
        #         c='b',linestyle=':')
        ax[2].legend()
        ax[2].set_xlabel('time [us]')
        ax[2].set_ylabel('amplitude [a.u.]')

        ax[3].plot(block_t, block_amps2, label=f'block version 2', zorder=-100)
        ax[3].legend()
        ax[3].set_xlabel('time [us]')
        ax[3].set_ylabel('amplitude [a.u.]')


        plt.show()

