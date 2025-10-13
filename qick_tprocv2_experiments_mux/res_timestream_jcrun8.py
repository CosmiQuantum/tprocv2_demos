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

class SingleResonatorTimestream(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['dynro_ch'][0]
        gen_ch = cfg['gen_ch']

        # play individual resonator pulse from MUX DAC
        self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch,
                         mux_freqs=cfg['this_res_freq'],
                         mux_gains=cfg['res_gain'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])

        self.add_pulse(ch=gen_ch, name="mux_pulse",
                       style="const",
                       length=cfg["pulse_length"],
                       mask=cfg["res_mask"], #for a single resonator, this will always be [0] and is set in system_config
                       )

        # dynamic readout on one channel
        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])
        self.add_readoutconfig(ch=ro_ch, name="ro",
                               freq=cfg['this_res_freq'][0], #entry res_freq
                               gen_ch=gen_ch,
                               outsel='product')
        self.send_readoutconfig(ch=ro_ch, name="ro", t=0)

        ## trigger DDR4 buffer, data collection begins now
        #self.trigger(t=cfg['trig_time'], ddr4=True)
        self.trigger(ros=cfg['dynro_ch'], pins=[0], t=cfg['trig_time'], ddr4=True)
        self.pulse(ch=cfg['gen_ch'], name="mux_pulse", t=0)

    def _body(self, cfg):
        ## body of program cycles through sending pulse and continuously reading out on one channel
        a=1
        #self.trigger(ros=cfg['dynro_ch'], pins=[0], t=cfg['trig_time'],ddr4=True)

class MultiResonatorTimestream(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['dynro_ch'][0]
        gen_ch = cfg['gen_ch']

        num_res = len(cfg['res_idx'])

        # set up resonator pulses from MUX DAC
        self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch,
                         mux_freqs= [cfg['res_freq'][i] for i in cfg['res_idx']],
                         mux_gains=cfg['res_gain'] * num_res,
                         mux_phases=cfg['res_phase'] * num_res,
                         mixer_freq=cfg['mixer_freq'])

        self.add_pulse(ch=gen_ch, name="mux_pulse",
                       style="const",
                       length=cfg["pulse_length"],
                       mask=cfg["res_idx"],
                       )

        # set up dynamic readout channels
        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])

        for r in cfg['res_idx']:
            ro_config_name = f"dynro{r}"
            self.add_readoutconfig(ch=ro_ch, name=ro_config_name,
                               freq=cfg['res_freq'][r],  # entry res_freq
                               gen_ch=gen_ch,
                               outsel='product')

        ## trigger DDR4 buffer, data collection begins now
        self.trigger(t=cfg['trig_time'], ddr4=True)

        ## send multiplexed pulse
        self.pulse(ch=cfg['gen_ch'], name="mux_pulse", t=0.0)

    def _body(self, cfg):
        ## body of program cycles readout channels until mux pulse ends.
        t=0
        while t < cfg['pulse_length']:
            for r in cfg['res_idx']:
                ro_config_name = f"dynro{r}"
                self.send_readoutconfig(ch=cfg['dynro_ch'][0], name=ro_config_name, t=t)
                t = t + cfg['dynro_time']

class SingleResonatorTimestreamWithDrive(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['dynro_ch'][0]
        gen_ch = cfg['gen_ch']
        drive_ch = cfg['drive_ch'] ## channel added to send pulse to drive resonator

        # play individual resonator pulse from MUX DAC
        self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch,
                         mux_freqs=cfg['this_res_freq'],
                         mux_gains=cfg['res_gain'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])

        self.add_pulse(ch=gen_ch, name="mux_pulse",
                       style="const",
                       length=cfg["pulse_length"],
                       mask=cfg["res_mask"], #for a single resonator, this will always be [0] and is set in system_config
                       )

        ## set up drive pulse from nonMUX DAC
        self.declare_gen(ch=drive_ch, nqz=cfg['nqz_res'], mixer_freq=cfg['mixer_freq'])
        self.add_pulse(ch=drive_ch, name="drive_pulse",
                               style="const",
                               freq=cfg['drive_res_freq'],
                               length=cfg["drive_length"],
                               phase=cfg['res_phase'],
                               gain=cfg['drive_gain'],
                               )


        # dynamic readout on one channel
        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])
        self.add_readoutconfig(ch=ro_ch, name="ro",
                               freq=cfg['this_res_freq'][0], #entry res_freq
                               gen_ch=gen_ch,
                               outsel='product')
        self.send_readoutconfig(ch=ro_ch, name="ro", t=0)

        ## trigger DDR4 buffer, data collection begins now
        #self.trigger(t=cfg['trig_time'], ddr4=True)
        self.trigger(ros=cfg['dynro_ch'], pins=[0], t=cfg['trig_time'], ddr4=True)

        # send drive resonator pulse
        self.pulse(ch=cfg['drive_ch'],name="drive_pulse",t=0)

        # send readout resonator pulse
        self.pulse(ch=cfg['gen_ch'], name="mux_pulse", t=0)

class ResonatorTimestream:
    def __init__(self, ResonatorIndex, number_of_resonators, studyDocumentationFolder, dataSetFolder, round_num, drive=False, driveIndex = 0, save_figs=False, experiment=None,
                 verbose=False, logger=None):
        self.ResonatorIndex = ResonatorIndex
        self.round_num = round_num
        self.Resonator = "R0" #+ str(self.ResonatorIndex)
        self.number_of_resonators = number_of_resonators
        self.studyDocumentationFolder = studyDocumentationFolder
        self.dataSetFolder = dataSetFolder
        self.drive = drive ## use drive resonator
        self.driveIndex = driveIndex
        if self.number_of_resonators == 1:
            self.expt_name = "single_res_timestream"
            if self.drive is True:
                self.expt_name = "single_res_timestream_drive"
        else:
            self.expt_name = "multi_res_timestream"
        self.round_num = round_num
        self.save_figs = save_figs
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if experiment is not None:
            self.r_config = all_resonator_state(experiment, self.number_of_resonators)
            self.config = {**self.r_config[self.Resonator], **self.exp_cfg}
            self.logger.info(f'R {self.ResonatorIndex} Round {self.round_num} Res Spec configuration: {self.config}')
            if self.verbose: print(f'R {self.ResonatorIndex} Round {self.round_num} Res Spec configuration: ',
                                   self.config)

    def run(self, threading=False, multi_file = False):

        #set number of transfers
        total_transfers = int(self.config['reps']*np.ceil(self.config['pulse_length']/(self.config['period'] * 128)))

        #arm DDR4 buffer
        self.experiment.soc.arm_ddr4(ch=self.config['dynro_ch'][0], nt=total_transfers)

        if self.expt_name == "single_res_timestream":
            self.config["this_res_freq"] = [self.config["res_freq"][self.ResonatorIndex] + self.config['offset']]
            self.config["res_mask"] = [0]
            self.config["mixer_freq"] = self.config["this_res_freq"][0] + 300

            prog = SingleResonatorTimestream(self.experiment.soccfg, reps=self.config["reps"], final_delay=self.config["relax_delay"],
                                             cfg=self.config)
        elif self.expt_name == "single_res_timestream_drive":
            self.config["this_res_freq"] = [self.config["res_freq"][self.ResonatorIndex] + self.config['offset']]
            self.config["res_mask"] = [0]
            self.config["mixer_freq"] = self.config["this_res_freq"][0] + 300
            self.config["drive_res_freq"] = self.config["res_freq"][self.driveIndex]

            prog = SingleResonatorTimestreamWithDrive(self.experiment.soccfg, reps=self.config["reps"], final_delay=self.config["relax_delay"], cfg=self.config)

        elif self.expt_name == "multi_res_timestream":
            self.config['dynro_time'] = self.config['pulse_length']/len(self.config['res_idx'])
            prog = MultiResonatorTimestream(self.experiment.soccfg, reps=self.config["reps"],
                                             final_delay=self.config["relax_delay"],
                                             cfg=self.config)
        #run program
        prog.run_rounds(self.experiment.soc)

        ## define data saving functions
        def save_pandas(iq_ddr4, data_path, idx, multi_file = False):
            ### store data in a series object
            series_I = pd.Series(iq_ddr4[:,0])
            series_Q = pd.Series(iq_ddr4[:,1])

            if multi_file is False:
                ## create one file and append batches of data
                file_path = os.join(data_path, f"timestream_{self.round_num}_{formatted_datetime}.h5")

                series_I.to_hdf(file_path, key="I",mode='a',format='fixed')
                series_Q.to_hdf(file_path, key="Q",mode='a',format='fixed')

            else:
                ### save batches of data in individual files labeled with their index
                file_path = os.join(data_path, f"timestream_{self.round_num}_{formatted_datetime}_{idx}.h5")

                series_I.to_hdf(file_path, key="I", mode='w', format='fixed')
                series_Q.to_hdf(file_path, key="Q", mode='w', format='fixed')

        #def save_ascii():

        def multi_save(queue, iq_ddr4, data_path, idx):
            save_pandas(iq_ddr4, data_path, idx, multi_file=True)

        ## set up data saving path
        formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data_path = os.join(self.dataSetFolder, "Data_h5", "timestream_run8")

        ### transfer a few samples from the buffer to find the timestep and record
        iq_ddr4 = self.experiment.soc.get_ddr4(nt=10,start=int(401 + 128))
        t = prog.get_time_axis_ddr4(self.config['dynro_ch'][0], iq_ddr4)
        timestep = t[1] - t[0]

        start_time = time.time() ## marker to calculate data saving time
        if threading is False:
            ## save data without multi-processing
            for idx in np.arange(0, int(np.floor(total_transfers / self.config['num_transfers']))):
                iq_ddr4 = self.experiment.soc.get_ddr4(nt=int(self.config['num_transfers']),
                                                       start=int(401 + 128 * idx * self.config['num_transfers']))

                save_pandas(iq_ddr4, data_path, idx, multi_file=multi_file)

        else:
            ## speed up writing data with multi-processing
            #create a queue to store data
            queue = multiprocessing.Queue()
            processes=[]

            num_process = 8 #how many concurrent processes to allow

            for idx in np.arange(0,int(np.floor(total_transfers/self.config['num_transfers']))):
                iq_ddr4 = self.experiment.soc.get_ddr4(nt=int(self.config['num_transfers']),
                                                       start=int(401 + 128 * idx * self.config['num_transfers']))

                p = Process(target=multi_save, args=(queue, iq_ddr4, idx))
                processes.append(p)

                while len(multiprocessing.active_children()) > num_process:
                    time.sleep(0.10)

                p.start()

            print(f"waiting for all processes {processes} to finish")
            for p in processes:
                p.join()
            print("all processes complete")

        return timestep, self.config


    def plot_accumulated_histogram(self, counts, hmin=0.5, hmax=4.0, bins=250):
        print("inside plot_accumulated_histogram function")
        cts, bin_edges = np.histogram([4], range=(hmin, hmax), bins=bins)

        fig, ax = plt.subplots()
        dummy_data = bin_edges[:-1]
        ax.hist(dummy_data, bins=bin_edges, weights=counts)
        ax.set_xlabel('amplitude [a.u.]')
        ax.set_yscale('log')
        ax.set_title(f'R{self.ResonatorIndex} {np.sum(counts) * self.config["chunk_size"] * self.config["period"] * 1./np.power(10,6)} seconds')
        print("made figure")

        if self.save_figs:
            print("attempting to save figure")
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(self.studyDocumentationFolder,
                                    f"R{self.ResonatorIndex}_{self.expt_name}_histogram_{formatted_datetime}")
            plt.savefig(f"{filename}.png")
            print("saved figure")

        return bin_edges

    def plot_histogram(self, block_amps):
        fig,ax = plt.subplots()
        counts, bin_edges, patches = ax.hist(block_amps, bins=200)
        ax.set_xlabel('amplitude [a.u.]')
        mean = np.mean(block_amps)
        sigma = np.std(block_amps)
        ax.plot([mean, mean], [0,counts.max()], linestyle=':',color='r')
        ax.plot([mean+sigma*self.config['sigma_factor'], mean+sigma*self.config['sigma_factor']], [0, counts.max()], linestyle=':', color='k')
        ax.plot([mean-sigma*self.config['sigma_factor'], mean-sigma*self.config['sigma_factor']], [0, counts.max()], linestyle=':', color='k')
        ax.set_yscale('log')

        if self.save_figs:
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(self.studyDocumentationFolder,
                                    f"R{self.ResonatorIndex}_{self.expt_name}_histogram_{formatted_datetime}")
            plt.savefig(f"{filename}.png")


    def get_histogram(self, block_amps, hmin=0.5, hmax=4.0, bins=250):
        counts, bin_edges = np.histogram(block_amps, range=(hmin,hmax),bins=bins)
        return counts

    def trim_data(self, iq_ddr4, t):
        idx = int(np.floor(self.config['trim_buffer']/(self.config['chunk_size']*self.config['period'])))
        iq_ddr4_trim = iq_ddr4[idx:]
        t_trim = t[idx:]

        return iq_ddr4_trim, t_trim

    def average_chunk(self, iq_ddr4, t):
        amps = np.abs(iq_ddr4[:, 0] + 1j * iq_ddr4[:, 1])

        series_t = pd.Series(t)
        series_amps = pd.Series(amps)
        chunk_size = self.config['chunk_size']

        block_amps = [series_amps[i:i + chunk_size].mean() for i in range(0, len(series_amps), chunk_size)]
        block_t = [series_t[i:i + chunk_size].mean() for i in range(0, len(series_t), chunk_size)]

        return block_amps, block_t

    def average_rolling(self, iq_ddr4):
        series_I = pd.Series(iq_ddr4[:, 0])
        series_Q = pd.Series(iq_ddr4[:, 1])
        chunk_size = self.config['chunk_size']

        rolling_I = series_I.rolling(window=chunk_size).mean()
        rolling_Q = series_Q.rolling(window=chunk_size).mean()

        return rolling_I, rolling_Q

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

    def plot_timestream(self, iq_ddr4, t):
        fig, ax = plt.subplots(3,1)

        amps = np.abs(iq_ddr4[:,0] + 1j*iq_ddr4[:,1])

        series_I = pd.Series(iq_ddr4[:,0])
        series_Q = pd.Series(iq_ddr4[:,1])
        series_amps = pd.Series(amps)
        chunk_size = self.config['chunk_size']

        rolling_I = series_I.rolling(window=chunk_size).mean()
        block_I = [series_I[i:i + chunk_size].mean() for i in range(0, len(series_I), chunk_size)]

        rolling_Q = series_Q.rolling(window=chunk_size).mean()
        block_Q = [series_Q[i:i + chunk_size].mean() for i in range(0, len(series_Q), chunk_size)]

        rolling_amps = series_amps.rolling(window=chunk_size).mean()
        block_amps = [series_amps[i:i + chunk_size].mean() for i in range(0, len(series_amps), chunk_size)]

        series_t = pd.Series(t)
        block_t = [series_t[i:i + chunk_size].mean() for i in range(0, len(series_t), chunk_size)]

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
        ax[2].plot([t[0], t[len(t)-1]],[mean, mean],c='r',linestyle=':')
        ax[2].plot([t[0], t[len(t)-1]], [mean+self.config['sigma_factor']*sigma, mean+self.config['sigma_factor']*sigma],
                c='b', linestyle=':')
        ax[2].plot([t[0], t[len(t)-1]], [mean-self.config['sigma_factor']*sigma, mean-self.config['sigma_factor']*sigma],
                c='b',linestyle=':')
        ax[2].legend()
        ax[2].set_xlabel('time [us]')
        ax[2].set_ylabel('amplitude [a.u.]')

        plt.show()

