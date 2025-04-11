from qick import *
from qick.pyro import make_proxy

# for now, all the tProc v2 classes need to be individually imported (can't use qick.*)

# the main program class
from qick.asm_v2 import AveragerProgramV2
# for defining sweeps
from qick.asm_v2 import QickSpan, QickSweep1D

import json
import datetime
import ast

# Used for live plotting, need to run "python -m visdom.server" in the terminal and open the IP address in browser
import visdom
import numpy as np
from build_task import *
from build_state import *
from system_config import *
import os
import h5py
from qualang_tools.plot import Fit
import pprint as pp
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from section_005_single_shot_gef import SingleShotProgram_g, SingleShotProgram_e, SingleShotProgram_f

class QubitTemperatureProgram(AveragerProgramV2):
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

        self.add_gauss(ch=qubit_ch, name="ge_ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=True)
        self.add_pulse(ch=qubit_ch, name="pi_ge",
                       style="arb",
                       envelope="ge_ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )


        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ef'],
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ef'],
                       )

    def _body(self, cfg):
        # self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        # self.pulse(ch=cfg['qubit_ch'], name="pi_ge", t=0)
        # self.delay_auto(0.0)
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse
        self.delay_auto(0.0)
        self.pulse(ch=cfg['qubit_ch'], name="pi_ge", t=0)
        self.delay_auto(0.0)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])


class QubitTemperatureRefProgram(AveragerProgramV2):
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

        self.add_gauss(ch=qubit_ch, name="ge_ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=True)
        self.add_pulse(ch=qubit_ch, name="pi_ge",
                       style="arb",
                       envelope="ge_ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ef'],
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ef'],
                       )

    def _body(self, cfg):
        #self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=cfg['qubit_ch'], name="pi_ge", t=0)
        self.delay_auto(0.0)
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse
        self.delay_auto(0.0)
        self.pulse(ch=cfg['qubit_ch'], name="pi_ge", t=0)
        self.delay_auto(0.0)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class EFRabiQubitTempsExperiment:
    def __init__(self, QubitIndex, number_of_qubits, experiment, outerFolder_qtemps, round_num):
        self.round_num = round_num
        self.outerFolder_qtemps = outerFolder_qtemps
        self.experiment = experiment
        self.number_of_qubits = number_of_qubits
        self.expt_name = "qubit_temp"
        self.QubitIndex = QubitIndex
        self.Qubit = 'Q' + str(self.QubitIndex)

    def run(self, soccfg, soc, expt_cfg, data_path, IS_VISDOM=False, save_data=False, SS=False):
        ss_config = expt_cfg["IQ_plot"]

        self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
        self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}

        py_avg = self.config['py_avg']
        expt_pts = [self.config["start"][0] + ii * self.config["step"] for ii in range(self.config["expts"][0])]

        # for live plotting
        IS_VISDOM = False
        if IS_VISDOM:
            expt_I = expt_Q = expt_mags = expt_phases = expt_pop = None
            expt_I_ref = expt_Q_ref = expt_mags_ref = expt_phases_ref = expt_pop_ref = None
            viz = visdom.Visdom()
            assert viz.check_connection(timeout_seconds=5), "Visdom server not connected!"
            viz.close(win=None)  # close previous plots
            win1 = viz.line(X=np.arange(0, 1), Y=np.arange(0, 1),
                            opts=dict(height=400, width=700, title='Length Rabi', showlegend=True, xlabel='expt_pts'))

        results_tot = []
        results_tot_ref = []

        #For first iteration
        expt_I = None
        expt_Q = None
        expt_I_ref = None
        expt_Q_ref = None

        for ii in range(py_avg):
            results = []
            pulse_lengths = []
            results_ref = []
            for length in expt_pts:
                self.config["qubit_length_ef"] = length

                rabi = QubitTemperatureProgram(soccfg,
                                               reps=self.config['reps'],
                                               final_delay=self.config['relax_delay'],
                                               cfg=self.config)
                data = rabi.acquire(soc, soft_avgs=1, progress=False)

                results.append(data[0][0])
                #pulse_lengths.append(rabi.get_pulse_param('qubit_pulse', 'length', as_array=True))

                rabi_ref = QubitTemperatureRefProgram(soccfg,
                                                      reps=self.config['reps'],
                                                      final_delay=self.config['relax_delay'],
                                                      cfg=self.config)
                data_ref = rabi_ref.acquire(soc, soft_avgs=1, progress=False)

                results_ref.append(data_ref[0][0])
                # wait_times.append(rabi.get_time_param('waiting', 't', as_array=True))

            iq_list = np.array(results).T
            # what is the correct shape/index?
            this_I = (iq_list[0])
            this_Q = (iq_list[1])

            if expt_I is None:  # ii == 0
                expt_I, expt_Q = this_I, this_Q
            else:
                expt_I = (expt_I * ii + this_I) / (ii + 1.0)
                expt_Q = (expt_Q * ii + this_Q) / (ii + 1.0)

            expt_mags = np.abs(expt_I + 1j * expt_Q)  # magnitude
            expt_phases = np.angle(expt_I + 1j * expt_Q)  # phase

            iq_list_ref = np.array(results_ref).T
            # what is the correct shape/index?
            this_I_ref = (iq_list_ref[0])
            this_Q_ref = (iq_list_ref[1])

            if expt_I_ref is None:  # ii == 0
                expt_I_ref, expt_Q_ref = this_I_ref, this_Q_ref
            else:
                expt_I_ref = (expt_I_ref * ii + this_I_ref) / (ii + 1.0)
                expt_Q_ref = (expt_Q_ref * ii + this_Q_ref) / (ii + 1.0)

            expt_mags_ref = np.abs(expt_I_ref + 1j * expt_Q_ref)  # magnitude
            expt_phases_ref = np.angle(expt_I_ref + 1j * expt_Q_ref)  # phase

            if IS_VISDOM:
                viz.line(X=pulse_lengths, Y=expt_mags_ref, win=win1, name='ref')
                viz.line(X=pulse_lengths, Y=expt_mags, win=win1, name='I', update='append')
        amps = expt_mags
        amps_ref = expt_mags_ref

        # # ### Fit ###
        fit = Fit()
        # Choose the suitable fitting function
        fit_result = fit.rabi(expt_pts, amps)

        fit_result = {
                "f": fit_result['f'],
                "phase": fit_result['phase'],
                "T": fit_result['T'],
                "amp": fit_result['amp'],
                "offset": fit_result['offset']
            }

        fit_result_ref = fit.rabi(expt_pts, amps_ref)

        fit_result_ref = {
                "f": fit_result_ref['f'],
                "phase": fit_result_ref['phase'],
                "T": fit_result_ref['T'],
                "amp": fit_result_ref['amp'],
                "offset": fit_result_ref['offset']
            }

        pp.pprint(fit_result_ref)

        time_stamp = time.mktime(datetime.datetime.now().timetuple())

        if SS == True:
            print('performing single shot for g-e-f calibration')

            ssp_g = SingleShotProgram_g(soccfg, reps=1, final_delay=ss_config['relax_delay'], cfg=ss_config)
            iq_list_g = ssp_g.acquire(soc, soft_avgs=1, progress=True)

            ssp_e = SingleShotProgram_e(soccfg, reps=1, final_delay=ss_config['relax_delay'], cfg=ss_config)
            iq_list_e = ssp_e.acquire(soc, soft_avgs=1, progress=True)

            ssp_f = SingleShotProgram_f(soccfg, reps=1, final_delay=ss_config['relax_delay'], cfg=ss_config)
            iq_list_f = ssp_f.acquire(soc, soft_avgs=1, progress=True)

            I_g = iq_list_g[0][0].T[0]
            Q_g = iq_list_g[0][0].T[1]
            I_e = iq_list_e[0][0].T[0]
            Q_e = iq_list_e[0][0].T[1]
            I_f = iq_list_f[0][0].T[0]
            Q_f = iq_list_f[0][0].T[1]

        # Save data to an H5 file
        if save_data:
            if data_path is None:
                raise ValueError("data_path must be provided when save_data is True")

            # Generate a unique filename with a timestamp
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            outerFolder_saveh5s = os.path.join(self.outerFolder_qtemps, "Data_h5")
            if not os.path.exists(outerFolder_saveh5s): os.makedirs(outerFolder_saveh5s)

            h5_filename = os.path.join(outerFolder_saveh5s, f"Q{self.QubitIndex+1}_temperature_efRabi_round{self.round_num}_{timestamp}.h5")
            with h5py.File(h5_filename, 'w') as f:
                f.create_dataset('QubitIndex', data=self.QubitIndex)
                f.create_dataset('time_stamp', data=time_stamp)
                f.create_dataset('lengths', data=expt_pts)
                f.create_dataset('amps', data=amps)
                f.create_dataset('amps_ref', data=amps_ref)
                f.create_dataset('avgi', data=expt_I)
                f.create_dataset('avgq', data=expt_Q)
                f.create_dataset('avgi_ref', data=expt_I_ref)
                f.create_dataset('avgq_ref', data=expt_Q_ref)
                f.attrs['config'] = json.dumps(self.config)
                f.attrs['fit_result'] = json.dumps(fit_result)
                f.attrs['fit_result_ref'] = json.dumps(fit_result_ref)

                if SS:
                    f.create_dataset('I_g', data=I_g)
                    f.create_dataset('Q_g', data=Q_g)
                    f.create_dataset('I_e', data=I_e)
                    f.create_dataset('Q_e', data=Q_e)
                    f.create_dataset('I_f', data=I_f)
                    f.create_dataset('Q_f', data=Q_f)
                    f.attrs['ss_config'] = json.dumps(ss_config)
            print("Data saved to:", self.outerFolder_qtemps)

        return fit_result, fit_result_ref