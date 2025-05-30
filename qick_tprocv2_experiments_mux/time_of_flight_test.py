from copy import deepcopy
from section_004_qubit_spec_ge import QubitSpectroscopy
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import datetime
from build_task import *
from build_state import *
from expt_config import *
import copy
import visdom
import logging
import math
from section_004_qubit_spec_ge import QZEStyleResStarkShift2D
import copy
class TOF:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, signal, save_figs, experiment = None,
                 live_plot = None, increase_qubit_reps = False, qubit_to_increase_reps_for = None,
                 multiply_qubit_reps_by = 0, verbose = False, logger = None, qick_verbose=True, QZE=False,
                 projective_readout_pulse_len_us=9,  time_between_projective_readout_pulses=None, zeno_pulse_gain=None,
                 bare_pi_len_qze_experiment=False):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.fit_data = True
        self.expt_name = "Dephasing_ge_with_ef_noise"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.signal = signal
        self.save_figs = save_figs
        self.live_plot = live_plot
        self.number_of_qubits = number_of_qubits
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            if increase_qubit_reps:
                if self.QubitIndex == qubit_to_increase_reps_for:
                    if self.verbose: print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                    self.logger.info(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                    self.config["reps"] *= multiply_qubit_reps_by
                    # self.config['ramsey_freq'] = 2 * self.config['ramsey_freq']
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} T2E configuration: ', self.config)
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} T2E configuration: {self.config}')

    def old_tof_program(self):
        prog = OscilliscopeQZEProgram(self.experiment.soccfg, reps=1, final_delay=0.5, cfg=self.config)
        iq_list = prog.acquire_decimated(self.experiment.soc, soft_avgs=self.config['rounds'])

        I = iq_list[self.QubitIndex][:, 0]
        Q = iq_list[self.QubitIndex][:, 1]
        t = prog.get_time_axis(ro_index=0)
        magnitude = np.abs(iq_list[0].dot([1, 1j]))

        plt.figure()
        plt.plot(t, I, label="I value")
        plt.plot(t, Q, label="Q value")
        plt.plot(t, magnitude, label="magnitude")
        plt.ylabel("a.u.")
        plt.xlabel("us")
        plt.show()

    def run_oscilliscope_simple(self, thresholding=False):

        prog = OscilliscopeExampleProgram(self.experiment.soccfg, reps=1, final_delay=0.1, cfg=self.config)
        iq_list = prog.acquire_decimated(self.experiment.soc, soft_avgs=self.config['rounds'])


        I = iq_list[self.QubitIndex][:, 0]
        Q = iq_list[self.QubitIndex][:, 1]

        t = prog.get_time_axis(ro_index=0)

        plt.plot(t, I, label="I value")
        plt.plot(t, Q, label="Q value")
        plt.plot(t, np.abs(iq_list[0].dot([1, 1j])), label="magnitude")
        plt.legend()
        plt.ylabel("a.u.")
        plt.xlabel("us")
        plt.show()

    def run_oscilliscope(self, thresholding=False):
        #self.config['noise_pulse_gain']=0

        #self.config['res_freq_ge']=self.config['res_freq_ge'][2]
        from section_014_dynamical_decoupling_ge import DephasingProgram
        # prog = DephasingWithEFNoiseProgramTOF(self.experiment.soccfg, reps=1,  final_delay=self.config['relax_delay'],
        #                  cfg=self.config)

        # prog = DephasingWithEFNoiseProgramTOF(self.experiment.soccfg, reps=self.config['reps'],
        #                         final_delay=self.config['relax_delay'],
        #                         cfg=self.config)
        prog = DephasingProgram(self.experiment.soccfg, reps=self.config['reps'],
                                final_delay=self.config['relax_delay'],
                                cfg=self.config)
        iq_list = prog.acquire(self.experiment.soc, soft_avgs=self.config['rounds'], progress=self.qick_verbose)
        I = iq_list[self.QubitIndex][0, :, 0]
        Q = iq_list[self.QubitIndex][0, :, 1]

        delay_times = 0
        for dephasing_round in range(self.config["dephasing_rounds_plus_1"] - 1):
            delay_times_n = prog.get_time_param('wait1' + str(dephasing_round), "t", as_array=True)
            delay_times = delay_times + delay_times_n
        delay_times2 = prog.get_time_param('wait2', "t", as_array=True)
        delay_times = delay_times + delay_times2


        plt.plot(delay_times, I, label="I value")
        plt.plot(delay_times, Q, label="Q value")
        #plt.plot(delay_times, np.abs(iq_list[0].dot([1, 1j])), label="magnitude")
        plt.legend()
        plt.ylabel("a.u.")
        plt.xlabel("us")
        plt.show()



    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

class OscilliscopeExampleProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        gen_ch = cfg['res_ch']

        self.declare_gen(
            ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_chs[0],
            mux_freqs=cfg['res_freq_qze'],
            mux_gains=[1, 0, 0, 0, 0, 0, 0],  # need to ramp it up here to see a clean signal
            mux_phases=cfg['res_phase'],
            mixer_freq=cfg['mixer_freq']
        )
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(
                ch=ch, length=10, freq=f, phase=ph, gen_ch=gen_ch
            )

        self.add_pulse(
            ch=gen_ch, name="mymux",
            style="const",
            length=cfg["res_length"],
            mask=cfg["list_of_all_qubits"],
        )

        self.add_pulse(ch=gen_ch, name="mygaus",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )

    def _body(self, cfg):
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=0, ddr4=True)
        self.pulse(ch=cfg['res_ch'], name="mymux", t=0)
        self.delay_auto(t=3, tag='waiting')
        self.pulse(ch=cfg['res_ch'], name="mygaus", t=0)


class DephasingWithEFNoiseProgramTOF(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        noise_ch = cfg['qubit_ampl_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(
                ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch
            )
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)

        self.declare_gen(ch=noise_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_pulse(ch=qubit_ch, name="noise_pulse",
                       style="const",
                       length=cfg["noise_pulse_len"],
                       freq=cfg['res_freq_ge'][2],
                       phase=cfg['qubit_phase'],
                       gain=cfg['noise_pulse_gain'],
                       mode='periodic'
                       )
        self.add_pulse(ch=qubit_ch, name="stop_periodic_pulse",
                       style="const",
                       length=0.01,
                       freq=cfg['res_freq_ge'][2],
                       phase=cfg['qubit_phase'],
                       gain=0
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse1",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['res_freq_ge'][2],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'] / 2,
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse_pi",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['res_freq_ge'][2],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse2",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['res_freq_ge'][2],
                       phase=cfg['qubit_phase'] + cfg['wait_time'] * 360 * cfg['ramsey_freq'],
                       # current phase + time * 2pi * ramsey freq
                       gain=cfg['pi_amp'] / 2,
                       )

        self.add_loop("waitloop", cfg["steps"])

    def _body(self, cfg):

        self.pulse(ch=self.cfg["qubit_ampl_ch"], name="noise_pulse", t=3)  # play noise pulse

        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse1",
                   t=self.cfg['noise_pulse_ramp_time']+3)  # play probe pulse after some ramp up time for noise pulse

        for dephasing_round in range(self.cfg[
                                         "dephasing_rounds_plus_1"] - 1):  # cant use delay auto because that will wait for the noise pulse to be done which it shouldnt

            self.delay_auto((cfg['wait_time'] / self.cfg["dephasing_rounds_plus_1"]) + 0.01,
                            tag='wait1' + str(dephasing_round))
            self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse_pi", t=0)  # play pulse

        self.delay_auto((cfg['wait_time'] / self.cfg["dephasing_rounds_plus_1"]) + 0.01,
                        tag='wait2')  # wait_time after last pulse (wait / 2)
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse2", t=0)
        self.delay_auto()
        self.pulse(ch=self.cfg["qubit_ampl_ch"], name="stop_periodic_pulse", t=0)
        self.delay_auto(0.01)  # wait_time after last pulse
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class OscilliscopeQZEProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # generator for the readout pulses
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], [f+10 for f in cfg['res_freq_ge']], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=10, freq=f, phase=ph, gen_ch=res_ch) #length=cfg['res_length']

        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=9,  # 9us as usual  cfg["res_length"]
                       mask=cfg["list_of_all_qubits"])
        # projection pulse (the short pulse used for projective measurement,9 ns)
        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=0.05, #0.007
                       mask=cfg["list_of_all_qubits"],
                       )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_res'], mixer_freq=cfg['mixer_freq'])
        # total_drive_length = 0.1 + 0.6

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['res_freq_ge'][0], #only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'] * 20) #ramp it up here

    def _body(self, cfg):

        # drive the qubit on the qubit channel (list with length 6)
        #self.pulse(ch=cfg["res_ch"], name="qubit_pulse", t=0) #why do i need to send this on the res _ch?

        # length of the qubit pulse (us)
        Tdrive = cfg['qubit_length_ge']  #1.5 #start in congfig is set to 0.1 so i always start at 0.1 and end at 0.7, so qubit should be in first excited state

        # now we have started pulse at 0.1us
        # each pulse is 9 ns long with a 2 ns gap between pulses
        # schendule pulses as long as the entire pulse fits within the qubit drive pulse time

        t_pulse = 0  # start at 0 (qubits in first excited state because config starts at 0.1us qubit pulse)
        while t_pulse + 0.05 <= Tdrive:  # as long as we are below the qubit drive pulse time for the next short res pulse
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=t_pulse)  # schedule this pulse
            t_pulse += 0.2  # 9 ns pulse + 2 ns wait = 11 ns per cycle


        #self.delay_auto(t=0.5, tag='waiting')  # auto wait for those pulses to be done
        # # immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=Tdrive+0.5)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class OscilliscopeExampleProgram(AveragerProgramV2):
    def _initialize(self, cfg):

        ro_chs = cfg['ro_ch']
        gen_ch = cfg['res_ch']

        self.declare_gen(
            ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_chs[0],
            mux_freqs=cfg['res_freq_ge'],
            mux_gains=[0,0,1,0,0,0], #need to ramp it up here to see a clean signal
            mux_phases=cfg['res_phase'],
            mixer_freq=cfg['mixer_freq']
        )
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(
                ch=ch, length=10, freq=f, phase=ph, gen_ch=gen_ch
            )

        self.add_pulse(
            ch=gen_ch, name="mymux",
            style="const",
            length=cfg["res_length"],
            mask=cfg["list_of_all_qubits"],
        )


        self.add_pulse(ch=gen_ch, name="mygaus",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )

    def _body(self, cfg):

        self.trigger(ros=cfg['ro_ch'], pins=[0], t=0, ddr4=True)
        self.pulse(ch=cfg['res_ch'], name="mymux", t=0)
        self.delay_auto(t=3, tag='waiting')
        self.pulse(ch=cfg['res_ch'], name="mygaus", t=0)

from system_config import QICK_experiment
folder='run6b/6transmon/QZE/dephasing_from_higher_energy_levels/TOF_test'
experiment = QICK_experiment(folder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10,
                                 fridge=FRIDGE)
updated_qubit_gain = 0.05 # lets do a low gain to start so I dont have a broad linewidth for the qubit
QubitIndex=2
res_leng_vals = [4, 14, 6, 10, 6, 7]
res_gain = [1,1,1,0.7,0.8,0.6]
freq_offsets = [0.1, -0.25, -0.2, 0.2, -0.1, -0.1]
experiment.qubit_cfg['qubit_gain_ge'][
    QubitIndex] = updated_qubit_gain

# Mask out all other resonators except this one
res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
experiment.readout_cfg['res_gain_ge'] = res_gains

res_gains_ef = experiment.mask_gain_res(QubitIndex, IndexGain=experiment.readout_cfg['res_gain_ef'][QubitIndex], num_qubits=tot_num_of_qubits)
experiment.readout_cfg['res_gain_ef'] = res_gains_ef

res_gains_fh = experiment.mask_gain_res(QubitIndex, IndexGain=experiment.readout_cfg['res_gain_fh'][QubitIndex],
                                        num_qubits=tot_num_of_qubits)
experiment.readout_cfg['res_gain_fh'] = res_gains_fh

experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

################################################# g-e Res spec ####################################################
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, folder, 0, True,
                                 experiment)
res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
experiment.readout_cfg['res_freq_ge'] = res_freqs

# incorporating offset (if you don't want to, then set all values inside freq_offsets to zero)
offset = freq_offsets[QubitIndex]
offset_res_freqs = [r + offset for r in res_freqs]
experiment.readout_cfg['res_freq_ge'] = offset_res_freqs


del res_spec
tof = TOF(QubitIndex=3, number_of_qubits=6, outerFolder=folder, experiment=experiment, round_num=0, signal='None', save_figs=True)

tof.run_oscilliscope_simple(thresholding=False)  # thres# holding not implemented
