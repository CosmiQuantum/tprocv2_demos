from qick.asm_v2 import AveragerProgramV2
import matplotlib.pyplot as plt
from build_state import *
from expt_config import *
from system_config import *
import numpy as np

class TOFExperiment:
    def __init__(self, QubitIndex,  outerFolder, experiment, round_num = 1, save_figs = True, title = False, qick_verbose=True, unmasking_resgain = False):
        # every time a class instance is created, these definitions are set
        self.expt_name = "tof"
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.Qubit = 'Q' + str(QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.experiment = experiment
        self.save_figs = save_figs
        self.title = title
        self.qick_verbose=qick_verbose

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

        self.q_config = all_qubit_state(self.experiment,6)
        self.round_num = round_num
        if 'All' in self.Qubit:
            self.config = {**self.q_config['Q0'], **self.exp_cfg}
            print(f'Q {self.QubitIndex} Round {round_num} TOF configuration: ', self.config)
        else:
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            print(f'Q {self.QubitIndex + 1} Round {round_num} TOF configuration: ',self.config)


    def run(self):
        class MuxProgram(AveragerProgramV2):
            # def _initialize(self, cfg):
            #     ro_chs = cfg['ro_ch']
            #     gen_ch = cfg['res_ch']
            #     stark_ch = cfg['res_ch2']
            #
            #     self.declare_gen(
            #         ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_chs[0],
            #         mux_freqs=[f+1 for f in cfg['res_freq_ge']],
            #         mux_gains= cfg['res_gain_ge'], #[1,0,0,0,0,0],#cfg['res_gain_ge'], #[1,0,0,0,0,0]
            #         mux_phases=cfg['res_phase'],
            #         mixer_freq=cfg['mixer_freq']
            #     )
            #     print('before')
            #     self.declare_gen(ch=stark_ch, nqz=cfg['nqz_res'], mixer_freq=cfg['mixer_freq'])
            #     self.add_pulse(ch=stark_ch, name="res_tone", ro_ch=ro_chs[0],
            #                    style="const",
            #                    length=cfg['res_length'],
            #                    freq=cfg['res_freq_ge'] ,
            #                    phase=cfg['res_phase'],
            #                    gain=cfg["res_gain_ge"],
            #                    )
            #     print('before')
            #
            #     for ch, f, ph in zip(cfg['ro_ch'], [f+1 for f in cfg['res_freq_ge']], cfg['ro_phase']):
            #         self.declare_readout(
            #             ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=gen_ch
            #         )
            #     print('after')
            #
            #     # self.add_pulse(
            #     #     ch=gen_ch, name="mymux",
            #     #     style="const",
            #     #
            #     #     freq=cfg["res_freq_ge"],
            #     #     phase=cfg["res_phase"],
            #     #     gain=cfg["res_gain_ge"],
            #     #     length=cfg["res_lengths"],
            #     #     # mask=cfg["list_of_all_qubits"],
            #     # )
            #     print('after2')
            #     # self.add_pulse(ch=gen_ch, name="mymux", ro_ch=ro_chs[0],
            #     #                style="const",
            #     #                freq=cfg['res_freq_ge'],
            #     #                length=cfg['res_length'],
            #     #                phase=0,
            #     #                gain=cfg['res_gain_ge'],
            #     #                )
            #
            #     # self.add_pulse(ch=gen_ch, name="res_pulse", ro_ch=ro_chs,
            #     #                style="const",
            #     #                length=cfg['res_length'],
            #     #                freq=cfg['res_freq_ge'],
            #     #                phase=cfg['res_phase'],
            #     #                gain=cfg['res_gain_ge'],
            #     #                )
            #
            #
            #
            # def _body(self, cfg):
            #     self.trigger(ros=self.ro_chs, pins=[0], t=0, ddr4=True)
            #     self.pulse(ch=self.stark_ch, name="mymux", t=0)
            # class StarkShift2DProgram(AveragerProgramV2):
            def _initialize(self, cfg):
                ro_ch = cfg['ro_ch']
                res_ch = cfg['res_ch']
                qubit_ch = cfg['qubit_ch']
                stark_ch = cfg['qubit_ampl_ch']

                self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                                 mux_freqs=cfg['res_freq_ge'],  # res of interest frequency at QubitIndex and 7
                                 mux_gains=cfg['res_gain_ge'],  # readout gain, stark gain
                                 mux_phases=cfg['res_phase'],  # res of interest phase repeated at QubitIndex and 7
                                 mixer_freq=cfg['mixer_freq'])
                for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
                    self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)

                # self.add_pulse(ch=res_ch, name="stark_tone",
                #                style="const",
                #                length=cfg['stark_length'],
                #                mask=cfg['stark_mask'], #only play stark tone
                #                )

                self.add_pulse(ch=res_ch, name="readout_pulse",
                               style="const",
                               length=cfg['res_length'],
                               mask=cfg['list_of_all_qubits'],  # only play readout tone
                               )

                self.declare_gen(ch=stark_ch, nqz=cfg['nqz_res'], mixer_freq=cfg['mixer_freq'])
                self.add_pulse(ch=stark_ch, name="stark_tone", ro_ch=ro_ch[0],
                               style="const",
                               length=cfg['res_length'],
                               freq=cfg['res_freq_ge'][1],# + cfg['detuning'],
                               phase=cfg['res_phase'][1],
                               gain=cfg['res_gain_ge'][1],
                               )

                # self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
                # self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
                # self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                #                style="arb",
                #                envelope="ramp",
                #                freq=QickSweep1D("qubit_pulse_loop", cfg["qubit_freq_ge"] + cfg["start_freq"], cfg["qubit_freq_ge"] + cfg["end_freq"]),
                #                phase=cfg['qubit_phase'],
                #                gain=cfg['pi_amp'],
                #                )

                # self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch[0],
                #                style="const",
                #                length=cfg['qubit_length_ge'],
                #                freq=QickSweep1D("qubit_pulse_loop", cfg["qubit_freq_ge"] + cfg["start_freq"],
                #                                 cfg["qubit_freq_ge"] + cfg["end_freq"]),
                #                phase=0,
                #                gain=cfg['qubit_gain_ge'],
                #                )

                # self.add_loop("gain_loop", cfg["gain_steps"])
                # self.add_loop("qubit_pulse_loop", cfg["qubit_pulse_steps"])  # inner loop

            def _body(self, cfg):
                # self.pulse(ch=self.cfg['qubit_ampl_ch'], name="stark_tone", t=0)  # play stark tone
                # self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse",
                #            t=cfg['qubit_pulse_delay'])  # play qubit pulse with delay
                # self.delay(t=cfg['stark_length'] + cfg[
                #     'readout_pulse_delay'])  # wait for stark tone to finish and for resonator to reach vacuum
                self.pulse(ch=self.cfg['qubit_ampl_ch'], name="stark_tone", t=0)
                self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

        prog = MuxProgram(self.experiment.soccfg, reps=1, final_delay=0.5, cfg=self.config)
        iq_list = prog.acquire_decimated(self.experiment.soc, soft_avgs=self.config['soft_avgs'])
        if self.save_figs:
            (average_y_mag_values_last, average_y_mag_values_mid, average_y_mag_values_oct, DAC_attenuator1, DAC_attenuator2, ADC_attenuator) = self.plot_results(prog, iq_list)
        else:
            (average_y_mag_values_last, average_y_mag_values_mid, average_y_mag_values_oct, DAC_attenuator1, DAC_attenuator2, ADC_attenuator) = None, None, None, None, None, None,

        return (average_y_mag_values_last, average_y_mag_values_mid, average_y_mag_values_oct, DAC_attenuator1, DAC_attenuator2, ADC_attenuator)


    def plot_results(self, prog, iq_list):
        t = prog.get_time_axis(ro_index=0)
        fig, axes = plt.subplots(len(self.config['ro_ch']), 1, figsize=(12, 12))
        phase_offsets=[]
        average_y_mag_values_mid = []
        average_y_I_values_mid = []
        average_y_Q_values_mid = []
        average_y_mag_values_oct = []
        average_y_I_values_oct = []
        average_y_Q_values_oct = []
        average_y_mag_values_last = []
        average_y_I_values_last = []
        average_y_Q_values_last = []
        for i, ch in enumerate(self.config['ro_ch']):
            plot = axes[i]
            plot.plot(t, iq_list[i][:, 0], label="I value")
            print(f'channel {i+1} res_length',self.config['res_length'], 'len(iq_list[i][:, 0])', len(iq_list[i][:, 0]))
            print('sum', sum(iq_list[i][:,0]))
            print('avg', np.mean(iq_list[i][:, 0]))
            plot.plot(t, iq_list[i][:, 1], label="Q value")
            magnitude = np.abs(iq_list[i].dot([1, 1j]))
            plot.plot(t, magnitude, label="magnitude")
            plot.legend()
            plot.set_ylabel("a.u.")
            plot.set_xlabel("us")
            plot.axvline(0.75, c='r')

            phase_offset = np.angle(iq_list[i].dot([1, 1j]).sum(), deg=True)
            # print("measured phase %f degrees" % (phase_offset))
            phase_offsets.append(phase_offset)


            # Find indices of the middle three x-values
            mid_index = len(t) // 2
            indices_mid = [mid_index - 15, mid_index, mid_index + 15] #average 7 values

            one_eighth_index = len(t) // 15
            indices_oct = [one_eighth_index - 1, one_eighth_index, one_eighth_index + 1]

            indices_last = slice(-15, None)  # this will grab the last 7 elements

            # Calculate average y-values for I, Q, and magnitude
            avg_i_mid = np.mean(iq_list[i][indices_mid, 0])
            avg_q_mid = np.mean(iq_list[i][indices_mid, 1])
            avg_mag_mid = np.mean(magnitude[indices_mid])

            # Calculate average y-values for I, Q, and magnitude
            avg_i_oct = np.mean(iq_list[i][indices_oct, 0])
            avg_q_oct = np.mean(iq_list[i][indices_oct, 1])
            avg_mag_oct = np.mean(magnitude[indices_oct])

            # Calculate average y-values for I, Q, and magnitude
            avg_i_last = np.mean(iq_list[i][indices_last, 0])
            avg_q_last = np.mean(iq_list[i][indices_last, 1])
            avg_mag_last = np.mean(magnitude[indices_last])

            # Append the average magnitude to the list, you can change this to average I or Q.
            average_y_mag_values_mid.append(avg_mag_mid)
            average_y_I_values_mid.append(avg_i_mid)
            average_y_Q_values_mid.append(avg_q_mid)

            average_y_mag_values_oct.append(avg_mag_oct)
            average_y_I_values_oct.append(avg_i_oct)
            average_y_Q_values_oct.append(avg_q_oct)

            average_y_mag_values_last.append(avg_mag_last)
            average_y_I_values_last.append(avg_i_last)
            average_y_Q_values_last.append(avg_q_last)
        if self.title:
            plt.suptitle(f"TOF DAC_Att_1:{self.experiment.DAC_attenuator1} DAC_Att_2:{self.experiment.DAC_attenuator2} ADC_Att:{self.experiment.ADC_attenuator}", fontsize=24, y=0.95)


        # Save
        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
            self.experiment.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            if 'All' in self.Qubit:
                file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}" + f"Q_{self.QubitIndex}" + f"{formatted_datetime}_" + self.expt_name + ".png")
            else:
                file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}" + f"Q_{self.QubitIndex+1}" + f"{formatted_datetime}_" + self.expt_name + ".png")
            plt.savefig(file_name, dpi=50)
            # plt.show()
            # plt.close(fig)

        return average_y_mag_values_last, average_y_mag_values_mid, average_y_mag_values_oct, self.experiment.DAC_attenuator1, self.experiment.DAC_attenuator2, self.experiment.ADC_attenuator



