
import datetime
import numpy as np
np.set_printoptions(threshold=1000000000000000)
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import h5py
# Assuming these are defined elsewhere and importable
from build_task import *
from build_state import *
from expt_config import *
from system_config import QICK_experiment
import copy
import os

# Both g and e during the same experiment.
class GEF_SingleShotProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_chs']
        gen_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=gen_ch, nqz=cfg['nqz'], ro_ch=ro_chs[0],
                         mux_freqs=cfg['f_res'],
                         mux_gains=cfg['res_gain'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_chs'], cfg['f_res'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_len'], freq=f, phase=ph, gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_len"],
                       mask=[0, 1, 2, 3],
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_chs[0],
                       style="arb",
                       envelope="ramp",
                       freq=cfg['f_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_gain'],
                       )

        #         self.add_loop("shotloop", cfg["steps"]) # number of total shots
        self.add_loop("gainloop", cfg["expts"])  # Pulse / no Pulse loop

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play pulse
        self.delay_auto(0.01)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=cfg['ro_chs'], pins=[0], t=cfg['trig_time'])


# Separate g and e per each experiment defined.
class SingleShotProgram_g(AveragerProgramV2):
    def _initialize(self, cfg):

        ro_chs = cfg['ro_ch']
        gen_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        print(cfg["res_freq_ge"], cfg["res_gain_ge"])
        self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_chs[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=[0, 1, 2, 3],
                       )

        self.add_loop("shotloop", cfg["steps"])  # number of total shots

    def _body(self, cfg):
        self.delay_auto(0.01)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])
        # relax delay ...


class SingleShotProgram_e(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        gen_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_chs[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=[0, 1, 2, 3],
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        self.add_loop("shotloop", cfg["steps"])  # number of total shots

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play pulse
        self.delay_auto(0.0)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class SingleShotProgram_f(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        gen_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_chs[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=[0, 1, 2, 3],
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_gauss(ch=qubit_ch, name="geramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)

        self.add_pulse(ch=qubit_ch, name="ge_pi_pulse",
                       style="arb",
                       envelope="geramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )
        self.add_gauss(ch=qubit_ch, name="eframp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)

        self.add_pulse(ch=qubit_ch, name="ef_pi_pulse",
                       style="arb",
                       envelope="eframp",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_ef_amp'],
                       )

        self.add_loop("shotloop", cfg["steps"])  # number of total shots

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="ge_pi_pulse", t=0)  # play pulse
        self.delay_auto(0.0)
        self.pulse(ch=self.cfg["qubit_ch"], name="ef_pi_pulse", t=0)  # play pulse
        self.delay_auto(0.0)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])


class SingleShot:
    def __init__(self, QubitIndex, num_qubits, list_of_all_qubits, outerFolder, round_num, save_figs=False, experiment = None):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "Readout_Optimization"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.round_num = round_num
        self.save_figs = save_figs
        self.experiment = experiment
        self.num_qubits=num_qubits
        self.list_of_all_qubits = list_of_all_qubits

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.num_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            print(f'Q {self.QubitIndex + 1} Round {self.round_num} Single Shot configuration: ', self.config)

        self.q1_t1 = []
        self.q1_t1_err = []
        self.dates = []


    def fidelity_test(self, soccfg, soc):
        # Run the single shot programs (g and e)
        ssp_g = SingleShotProgram_g(soccfg, reps=1, final_delay=self.config['relax_delay'],
                                    cfg=self.config)
        iq_list_g = ssp_g.acquire(soc, soft_avgs=1, progress=False)

        ssp_e = SingleShotProgram_e(soccfg, reps=1, final_delay=self.config['relax_delay'],
                                    cfg=self.config)
        iq_list_e = ssp_e.acquire(soc, soft_avgs=1, progress=False)

        ssp_f = SingleShotProgram_f(soccfg, reps=1, final_delay=self.config['relax_delay'],
                                    cfg=self.config)
        iq_list_f = ssp_f.acquire(soc, soft_avgs=1, progress=False)

        # Use the fidelity calculation from SingleShot
        fidelity, _, _, _,_ = self.hist_ssf(
            data=[iq_list_g[self.QubitIndex][0].T[0], iq_list_g[self.QubitIndex][0].T[1],
                  iq_list_e[self.QubitIndex][0].T[0], iq_list_e[self.QubitIndex][0].T[1],
                  iq_list_f[self.QubitIndex][0].T[0], iq_list_f[self.QubitIndex][0].T[1]],
            cfg=self.config, plot=False)

        return fidelity

    def run(self, soccfg, soc):
        ssp_g = SingleShotProgram_g(soccfg, reps=1, final_delay=self.config['relax_delay'], cfg=self.config)
        iq_list_g = ssp_g.acquire(soc, soft_avgs=1, progress=True)

        ssp_e = SingleShotProgram_e(soccfg, reps=1, final_delay=self.config['relax_delay'], cfg=self.config)
        iq_list_e = ssp_e.acquire(soc, soft_avgs=1, progress=True)

        ssp_f = SingleShotProgram_f(soccfg, reps=1, final_delay=self.config['relax_delay'], cfg=self.config)
        iq_list_f = ssp_f.acquire(soc, soft_avgs=1, progress=True)

        fid, angle = self.plot_results(iq_list_g, iq_list_e, iq_list_f, self.QubitIndex)
        return fid, angle, iq_list_g, iq_list_e, iq_list_f

    def plot_results(self, iq_list_g, iq_list_e, iq_list_f, QubitIndex,  fig_quality=100):
        I_g = iq_list_g[QubitIndex][0].T[0]
        Q_g = iq_list_g[QubitIndex][0].T[1]
        I_e = iq_list_e[QubitIndex][0].T[0]
        Q_e = iq_list_e[QubitIndex][0].T[1]
        I_f = iq_list_f[QubitIndex][0].T[0]
        Q_f = iq_list_f[QubitIndex][0].T[1]
        print(QubitIndex)

        fid, threshold, angle, ig_new, ie_new = self.hist_ssf(data=[I_g, Q_g, I_e, Q_e, I_f, Q_f], cfg=self.config, plot=self.save_figs,  fig_quality=fig_quality)
        print('Optimal fidelity after rotation = %.3f' % fid)
        print('Optimal angle after rotation = %f' % angle)
        print(self.config)

        return fid, angle

    def hist_ssf(self, data=None, cfg=None, plot=True,  fig_quality = 100):

        ig = data[0]
        qg = data[1]
        ie = data[2]
        qe = data[3]
        i_f = data[4]
        qf = data[5]

        numbins = round(math.sqrt(float(cfg["steps"])))

        xg, yg = np.median(ig), np.median(qg)
        xe, ye = np.median(ie), np.median(qe)
        xf, yf = np.median(i_f), np.median(qf)

        if plot == True:
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 4))
            fig.tight_layout()

            axs[0][0].scatter(ig, qg, label='g', color='b', marker='*')
            axs[0][0].scatter(ie, qe, label='e', color='r', marker='*')
            axs[1][0].scatter(ig, qg, label='g', color='b', marker='*')
            axs[1][0].scatter(i_f, qf, label='f', color='r', marker='*')
            axs[0][0].scatter(xg, yg, color='k', marker='o')
            axs[0][0].scatter(xe, ye, color='k', marker='o')
            axs[1][0].scatter(xg, yg, color='k', marker='o')
            axs[1][0].scatter(xf, yf, color='k', marker='o')
            axs[0][0].set_xlabel('I (a.u.)')
            axs[0][0].set_ylabel('Q (a.u.)')
            axs[0][0].legend(loc='upper right')
            axs[0][0].set_title('Unrotated')
            axs[0].axis('equal')
            axs[1][0].set_xlabel('I (a.u.)')
            axs[1][0].set_ylabel('Q (a.u.)')
            axs[1][0].legend(loc='upper right')
            axs[1][0].set_title('Unrotated')
        """Compute the rotation angle"""
        theta_ge = -np.arctan2((ye - yg), (xe - xg))
        theta_gf = -np.arctan2((yf - yg), (xf - xg))
        """Rotate the IQ data"""
        ig_new_we = ig * np.cos(theta_ge) - qg * np.sin(theta_ge)
        qg_new_we = ig * np.sin(theta_ge) + qg * np.cos(theta_ge)
        ie_new = ie * np.cos(theta_ge) - qe * np.sin(theta_ge)
        qe_new = ie * np.sin(theta_ge) + qe * np.cos(theta_ge)

        ig_new_wf = ig * np.cos(theta_gf) - qg * np.sin(theta_gf)
        qg_new_wf = ig * np.sin(theta_gf) + qg * np.cos(theta_gf)
        if_new = i_f * np.cos(theta_gf) - qf * np.sin(theta_gf)
        qf_new = i_f * np.sin(theta_gf) + qf * np.cos(theta_gf)

        """New means of each blob"""
        xg_we, yg_we = np.median(ig_new_we), np.median(qg_new_we)
        xg_wf, yg_wf = np.median(ig_new_wf), np.median(qg_new_wf)
        xe, ye = np.median(ie_new), np.median(qe_new)
        xf, yf = np.median(if_new), np.median(qf_new)

        # print(xg, xe)
        #xlims = [xg - ran, xg + ran]
        xlims_ge = [np.min(ig_new_we), np.max(ie_new)]
        xlims_ef = [np.min(ig_new_wf), np.max(if_new)]

        if plot == True:
            axs[0][1].scatter(ig_new_we, qg_new_we, label='g', color='b', marker='*')
            axs[0][1].scatter(ie_new, qe_new, label='e', color='r', marker='*')
            axs[0][1].scatter(xg, yg, color='k', marker='o')
            axs[0][1].scatter(xe, ye, color='k', marker='o')
            axs[0][1].set_xlabel('I (a.u.)')
            axs[0][1].legend(loc='lower right')
            axs[0][1].set_title(f'Rotated Theta:{round(theta_ge, 5)}')
            axs[0][1].axis('equal')

            axs[3].scatter(ig_new_wf, qg_new_wf, label='g', color='b', marker='*')
            axs[3].scatter(if_new, qf_new, label='f', color='r', marker='*')
            axs[3].scatter(xg, yg, color='k', marker='o')
            axs[3].scatter(xe, ye, color='k', marker='o')
            axs[3].set_xlabel('I (a.u.)')
            axs[2].legend(loc='lower right')
            axs[2].set_title(f'Rotated Theta:{round(theta_gf, 5)}')
            axs[2].axis('equal')

            """X and Y ranges for histogram"""
            ng, binsg, pg = axs[3].hist(ig_new_we, bins=numbins, range=xlims_ge, color='b', label='g', alpha=0.5)
            ne, binse, pe = axs[3].hist(ie_new, bins=numbins, range=xlims_ge, color='r', label='e', alpha=0.5)

            axs[2].set_xlabel('I(a.u.)')
        else:
            ng, binsg = np.histogram(ig_new_we, bins=numbins, range=xlims_ge)
            ne, binse = np.histogram(ie_new, bins=numbins, range=xlims_ge)

        """Compute the fidelity using overlap of the histograms"""
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
        tind = contrast.argmax()
        threshold = binsg[tind]
        fid = contrast[tind]
        #axs[2].set_title(f"Fidelity = {fid * 100:.2f}%")


        if plot == True:
            outerFolder_expt = os.path.join(self.outerFolder, "ss_repeat_meas")
            self.create_folder_if_not_exists(outerFolder_expt)
            outerFolder_expt = os.path.join(outerFolder_expt, "Q" + str(self.QubitIndex + 1))
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt,
                                     f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")

            axs[2].set_title(f"Fidelity = {fid * 100:.2f}%")
            fig.savefig(file_name,  dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)

        return fid, threshold, theta, ig_new, ie_new

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)


class GainFrequencySweep:
    def __init__(self,qubit_index, experiment, optimal_lengths=None, output_folder="/default/path/"):
        self.qubit_index = qubit_index
        self.output_folder = output_folder
        self.expt_name = "Readout_Optimization"
        self.Qubit = 'Q' + str(self.qubit_index)
        self.optimal_lengths = optimal_lengths

        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.q_config = all_qubit_state(self.experiment)
        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}

    def set_res_gain_ge(self, QUBIT_INDEX, set_gain, num_qubits=6):
        """Sets the gain for the selected qubit to 1, others to 0."""
        res_gain_ge = [0] * num_qubits  # Initialize all gains to 0
        if 0 <= QUBIT_INDEX < num_qubits:  # makes sure you are within the range of options
            res_gain_ge[QUBIT_INDEX] = set_gain  # Set the gain for the selected qubit
        return res_gain_ge

    def run_sweep(self, freq_range, gain_range, freq_steps, gain_steps):
        freq_step_size = (freq_range[1] - freq_range[0]) / freq_steps
        gain_step_size = (gain_range[1] - gain_range[0]) / gain_steps
        results = []

        # Use the optimal readout length for the current qubit
        readout_length = self.optimal_lengths[self.qubit_index]
        print('readout_length for this qubit: ',readout_length)
        for freq_step in range(freq_steps):
            print('On iter', freq_step, 'out of ', freq_steps)
            freq = freq_range[0] + freq_step * freq_step_size
            #print('Running for res_freq: ', freq, '...')
            fid_results = []
            for gain_step in range(gain_steps):
                print('On iter', freq_step, 'out of ', freq_steps)
                #experiment = QICK_experiment(self.output_folder)
                #experiment = QICK_experiment(self.output_folder, DAC_attenuator1=10, DAC_attenuator2=5, ADC_attenuator=10)
                fresh_experiment = copy.deepcopy(self.experiment)
                gain = gain_range[0] + gain_step * gain_step_size
                print('gain', gain)

                # Update config with current gain and frequency values
                fresh_experiment.readout_cfg['res_freq_ge'][self.qubit_index]= freq
                fresh_experiment.readout_cfg['res_length'] = readout_length  # Set the optimal readout length for the qubit

                res_gains = fresh_experiment.mask_gain_res(self.qubit_index, gain)
                fresh_experiment.readout_cfg['res_gain_ge'] = res_gains

                # Initialize SingleShot instance for fidelity calculation
                round_num = 0
                save_figs = False
                single_shot = SingleShot(self.qubit_index,  self.output_folder, round_num, save_figs, fresh_experiment) #SingleShot(self.qubit_index, self.output_folder, fresh_experiment, round_num=0, save_figs = False)
                fidelity = single_shot.fidelity_test(fresh_experiment.soccfg, fresh_experiment.soc)
                fid_results.append(fidelity)
                del fresh_experiment
                del single_shot

            results.append(fid_results)

        return results

