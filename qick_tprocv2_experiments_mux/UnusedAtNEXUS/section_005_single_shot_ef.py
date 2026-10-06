import datetime
import numpy as np

np.set_printoptions(threshold=1000000000000000)
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import h5py
import logging
# Assuming these are defined elsewhere and importable
from build_task import *
from build_state import *
from expt_config import *
from system_config import QICK_experiment
import copy
import os
from matplotlib.patches import Circle
from section_005_single_shot_ge import SingleShot


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
                       mask=[0, 1, 2, 3, 4, 5],
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=True)
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

        self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_chs[0],
                         mux_freqs=cfg['res_freq_ef'],
                         mux_gains=cfg['res_gain_ef'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ef'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"]  # [0, 1, 2, 3, 4, 5],
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
                         mux_freqs=cfg['res_freq_ef'],
                         mux_gains=cfg['res_gain_ef'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ef'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"]  # [0, 1, 2, 3, 4, 5],
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
                         mux_freqs=cfg['res_freq_ef'],
                         mux_gains=cfg['res_gain_ef'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ef'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"]  # [0, 1, 2, 3, 4, 5],
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
        self.add_gauss(ch=qubit_ch, name="eframp", sigma=cfg['sigma_ef'], length=cfg['sigma_ef'] * 4, even_length=False)

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


class SingleShotProgram_h(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        gen_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_chs[0],
                         mux_freqs=cfg['res_freq_fh'],
                         mux_gains=cfg['res_gain_fh'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_fh'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=gen_ch)

        self.add_pulse(ch=gen_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"]  # [0, 1, 2, 3, 4, 5],
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
        self.add_gauss(ch=qubit_ch, name="eframp", sigma=cfg['sigma_ef'], length=cfg['sigma_ef'] * 4, even_length=False)

        self.add_pulse(ch=qubit_ch, name="ef_pi_pulse",
                       style="arb",
                       envelope="eframp",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_ef_amp'],
                       )

        self.add_gauss(ch=qubit_ch, name="fhramp", sigma=cfg['sigma_fh'], length=cfg['sigma_fh'] * 4, even_length=False)

        self.add_pulse(ch=qubit_ch, name="fh_pi_pulse",
                       style="arb",
                       envelope="fhramp",
                       freq=cfg['qubit_freq_fh'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_fh_amp'],
                       )

        self.add_loop("shotloop", cfg["steps"])  # number of total shots

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="ge_pi_pulse", t=0)  # play pulse
        self.delay_auto(0.0)
        self.pulse(ch=self.cfg["qubit_ch"], name="ef_pi_pulse", t=0)  # play pulse
        self.delay_auto(0.0)
        self.pulse(ch=self.cfg["qubit_ch"], name="fh_pi_pulse", t=0)  # play pulse
        self.delay_auto(0.0)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])


class SingleShot_ef:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, save_figs=False, experiment=None,
                 verbose=False, logger=None, qick_verbose=True, unmasking_resgain=False):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "Readout_Optimization"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.round_num = round_num
        self.save_figs = save_figs
        self.experiment = experiment
        self.number_of_qubits = number_of_qubits
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")
        self.exp_cfg = expt_cfg[self.expt_name]

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} EF Single Shot configuration: ',
                                   self.config)
            self.logger.info(
                f'Q {self.QubitIndex + 1} Round {self.round_num} EF Single Shot configuration: {self.config}')

        self.q1_t1 = []
        self.q1_t1_err = []
        self.dates = []

    def fidelity_test(self, outerfolder_plots):
        # print('output [plots folder path from fidelity_test', outerfolder_plots)
        # Run the single shot programs (g and e)
        # ssp_g = SingleShotProgram_g(self.experiment.soccfg, reps=1, final_delay=self.config['relax_delay'], cfg=self.config)
        # iq_list_g = ssp_g.acquire(self.experiment.soccfg, soft_avgs=1, progress=True)
        print('config', self.config)
        ssp_e = SingleShotProgram_e(self.experiment.soccfg, reps=1, final_delay=self.config['relax_delay'],
                                    cfg=self.config)
        iq_list_e = ssp_e.acquire(self.experiment.soc, soft_avgs=1, progress=True)

        ssp_f = SingleShotProgram_f(self.experiment.soccfg, reps=1, final_delay=self.config['relax_delay'],
                                    cfg=self.config)
        iq_list_f = ssp_f.acquire(self.experiment.soc, soft_avgs=1, progress=True)

        # Use the fidelity calculation from SingleShotGE
        # fid, theta_ef, ie_new, qe_new, if_new, qf_new, threshold_ef = self.hist_ssf(self.config["res_freq_ef"][self.QubitIndex],  self.config["res_gain_ef"][self.QubitIndex], outerfolder_plots,
        #     data=[iq_list_e[self.QubitIndex][0].T[0], iq_list_e[self.QubitIndex][0].T[1],
        #           iq_list_f[self.QubitIndex][0].T[0], iq_list_f[self.QubitIndex][0].T[1]],
        #     cfg=self.config, plot=True)
        fid, theta_ef, ie_new, qe_new, if_new, qf_new, threshold_ef = self.hist_ssf(
            outerfolder_plots,
            data=[iq_list_e[self.QubitIndex][0].T[0], iq_list_e[self.QubitIndex][0].T[1],
                  iq_list_f[self.QubitIndex][0].T[0], iq_list_f[self.QubitIndex][0].T[1]],
            cfg=self.config, plot=True)

        return fid, theta_ef, ie_new, qe_new, if_new, qf_new

    def run(self):
        ssp_g = SingleShotProgram_g(self.experiment.soccfg, reps=1, final_delay=self.config['relax_delay'],
                                    cfg=self.config)
        iq_list_g = ssp_g.acquire(self.experiment.soc, soft_avgs=1, progress=True)

        ssp_e = SingleShotProgram_e(self.experiment.soccfg, reps=1, final_delay=self.config['relax_delay'],
                                    cfg=self.config)
        iq_list_e = ssp_e.acquire(self.experiment.soc, soft_avgs=1, progress=True)

        ssp_f = SingleShotProgram_f(self.experiment.soccfg, reps=1, final_delay=self.config['relax_delay'],
                                    cfg=self.config)
        iq_list_f = ssp_f.acquire(self.experiment.soc, soft_avgs=1, progress=True)

        ssp_h = SingleShotProgram_h(self.experiment.soccfg, reps=1, final_delay=self.config['relax_delay'],
                                    cfg=self.config)
        iq_list_h = ssp_h.acquire(self.experiment.soc, soft_avgs=1, progress=True)

        fid, fid_fh, theta_ef, theta_fh, ie_new, qe_new, if_new, qf_new, threshold_ef, ih_new, qh_new, threshold_fh = self.plot_results(self.outerFolder, iq_list_g, iq_list_e,
                                                                        iq_list_f, iq_list_h, self.QubitIndex)
        # ie_new, qe_new, if_new, qf_new, theta_ef, threshold_ef = self.plot_results( iq_list_e,iq_list_f,self.QubitIndex)
        # return fid, angle, iq_list_g, iq_list_e, iq_list_f
        return iq_list_e, iq_list_f, ie_new, if_new, theta_ef, theta_fh,  threshold_ef, threshold_fh , self.config, fid, fid_fh

    def plot_results(self, outerfolder_plots, iq_list_g, iq_list_e, iq_list_f, iq_list_h, QubitIndex, fig_quality=100):
        I_g = iq_list_g[QubitIndex][0].T[0]
        Q_g = iq_list_g[QubitIndex][0].T[1]
        I_e = iq_list_e[QubitIndex][0].T[0]
        Q_e = iq_list_e[QubitIndex][0].T[1]
        I_f = iq_list_f[QubitIndex][0].T[0]
        Q_f = iq_list_f[QubitIndex][0].T[1]
        I_h = iq_list_h[QubitIndex][0].T[0]
        Q_h = iq_list_h[QubitIndex][0].T[1]
        print(QubitIndex)
        # fid, theta_ef, ie_new, qe_new, if_new, qf_new, threshold_ef

        # fid, theta_ef,  ie_new, qe_new, if_new, qf_new, threshold_ef  = self.hist_ssf(outerfolder_plots, data=[I_g, Q_g, I_e, Q_e, I_f, Q_f, I_h, Q_h], cfg=self.config, plot=self.save_figs,  fig_quality=fig_quality)

        fid, fid_fh, theta_ef, theta_fh, ie_new, qe_new, if_new, qf_new, threshold_ef, ih_new, qh_new, threshold_fh = self.hist_ssf(
            outerfolder_plots, data=[I_g, Q_g, I_e, Q_e, I_f, Q_f, I_h, Q_h], cfg=self.config, plot=self.save_figs,
            fig_quality=fig_quality)

        # fid, theta_ef, ig_new, ie_new, qe_new, if_new, qf_new, threshold_ef = self.hist_ssf(outerfolder_plots, data=[I_e, Q_e, I_f, Q_f], cfg=self.config, plot=self.save_figs,  fig_quality=fig_quality)

        # print('Optimal fidelity after rotation = %.3f' % fid)
        # print('Optimal angle after rotation = %f' % angle)
        # print(self.config)

        # return fid, angle
        #     return ie_new, qe_new, if_new, qf_new, theta_ef, threshold_ef

        # return ie_new,  if_new,  theta_ef, threshold_ef, fid
        return fid, fid_fh, theta_ef, theta_fh, ie_new, qe_new, if_new, qf_new, threshold_ef, ih_new, qh_new, threshold_fh


    # def hist_ssf(self, freq, gain, outerfolder_plots, data=None, cfg=None, plot=True, fig_quality=100):
    def hist_ssf(self, outerfolder_plots, data=None, cfg=None, plot=True, fig_quality=100):
        ig = data[0]
        qg = data[1]
        ie = data[2]
        qe = data[3]
        i_f = data[4]
        qf = data[5]
        ih = data[6]
        qh = data[7]

        numbins = round(math.sqrt(float(cfg["steps"])))

        xg, yg = np.median(ig), np.median(qg)
        xe, ye = np.median(ie), np.median(qe)
        xf, yf = np.median(i_f), np.median(qf)
        xh, yh = np.median(ih), np.median(qh)

        if plot == True:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
            fig.tight_layout()

            # axs[0].scatter(ig, qg, label='g', color='b', marker='*', alpha=0.3)
            # axs[0].scatter(ie, qe, label='e', color='r', marker='*', alpha=0.3)
            axs[0].scatter(i_f, qf, label='f', color='g', marker='*', alpha=0.3)
            axs[0].scatter(ih, qh, label='h', color='y', marker='*', alpha=0.3)

            # axs[0].scatter(xg, yg, color='k', marker='o')
            # axs[0].scatter(xe, ye, color='k', marker='o')
            axs[0].scatter(xf, yf, color='k', marker='o')
            axs[0].scatter(xh, yh, color='k', marker='o')

            axs[0].set_xlabel('I (a.u.)')
            axs[0].set_ylabel('Q (a.u.)')
            axs[0].legend(loc='upper right')
            axs[0].set_title('Unrotated')
            axs[0].axis('equal')
            axs[0].set_xlabel('I (a.u.)')
            axs[0].set_ylabel('Q (a.u.)')
            axs[0].legend(loc='upper right')


        """Compute the rotation angle"""
        # Will use the same angle to rotate all datasets (g, e, and f), to ensure that all states are rotated into a common frame for direct comparison.
        theta_ef = -np.arctan2((yf - ye), (xf - xe))
        theta_gf = -np.arctan2((yf - yg), (xf - xg))
        theta_fh = -np.arctan2((yh - yf), (xh - xf)) + (np.pi/8)
        # theta_fh = np.arctan2((yf - yh), (xf - xh))

        """Rotate the IQ data"""
        # ig_new = ig * np.cos(theta_ge) - qg * np.sin(theta_ge)
        # qg_new = ig * np.sin(theta_ge) + qg * np.cos(theta_ge)
        ie_new = ie * np.cos(theta_ef) - qe * np.sin(theta_ef)
        qe_new = ie * np.sin(theta_ef) + qe * np.cos(theta_ef)

        if_new = i_f * np.cos(theta_ef) - qf * np.sin(theta_ef)
        qf_new = i_f * np.sin(theta_ef) + qf * np.cos(theta_ef)

        ih_new = ih * np.cos(theta_fh) - qh * np.sin(theta_fh)
        qh_new = ih * np.sin(theta_fh) + qh * np.cos(theta_fh)

        """New means of each blob"""
        # xg, yg = np.median(ig_new), np.median(qg_new)
        xe, ye = np.median(ie_new), np.median(qe_new)
        xf, yf = np.median(if_new), np.median(qf_new)
        xh, yh = np.median(ih_new), np.median(qh_new)

        # print(xg, xe)
        # xlims = [xg - ran, xg + ran]
        xlims = [np.min(if_new), np.max(ih_new)]

        if plot == True:
            # axs[1].scatter(ig_new, qg_new, label='g', color='b', marker='*')
            # axs[1].scatter(ie_new, qe_new, label='e', color='r', marker='*', alpha=0.3)
            axs[1].scatter(if_new, qf_new, label='f', color='g', marker='*', alpha=0.3)
            axs[1].scatter(ih_new, qh_new, label='h', color='y', marker='*', alpha=0.3)
            # axs[1].scatter(xg, yg, color='k', marker='o')
            # axs[1].scatter(xe, ye, color='k', marker='o')
            axs[1].scatter(xf, yf, color='k', marker='o')
            axs[1].scatter(xh, yh, color='k', marker='o')
            axs[1].set_xlabel('I (a.u.)')
            axs[1].legend(loc='lower right')
            axs[1].set_title(f'Rotated Theta:{round(theta_fh, 5)}')
            axs[1].axis('equal')

            # --- Draw a circle around the f-state centroid ---
            # # Compute distances of each f-point from its centroid:
            # distances_f = np.sqrt((if_new - xf) ** 2 + (qf_new - yf) ** 2)
            # # For example, use the median distance as the circle's radius:
            # radius_f = np.median(distances_f)
            # # Create and add the circle to the rotated plot
            # circle_f = Circle((xf, yf), radius=radius_f, edgecolor='black', facecolor='none', linestyle='--')
            # axs[1].add_patch(circle_f)
            # print("Circle center:", circle_f.get_center())
            # print("Circle radius:", circle_f.get_radius())
            # -----------------------------------------------------

            """X and Y ranges for histogram"""
            # ng, binsg, pg = axs[2].hist(ig_new, bins=numbins, range=xlims, color='b', label='g', alpha=0.5)
            ne, binse, pe = axs[2].hist(ie_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.3)
            nf, binsf, pf = axs[2].hist(if_new, bins=numbins, range=xlims, color='g', label='f', alpha=0.3)
            nh, binsh, ph = axs[2].hist(ih_new, bins=numbins, range=xlims, color='y', label='h', alpha=0.3)

            axs[2].set_xlabel('I(a.u.)')
        else:
            # ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
            ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)
            nf, binsf = np.histogram(if_new, bins=numbins, range=xlims)
            nh, binsh = np.histogram(ih_new, bins=numbins, range=xlims)

        """Compute the fidelity using overlap of the histograms"""
        contrast = np.abs(((np.cumsum(ne) - np.cumsum(nf)) / (0.5 * ne.sum() + 0.5 * nf.sum())))
        tind = contrast.argmax()
        threshold_ef = binse[tind]
        fid = contrast[tind]

        contrast_fh = np.abs(((np.cumsum(nf) - np.cumsum(nh)) / (0.5 * nf.sum() + 0.5 * nh.sum())))
        tind_fh = contrast_fh.argmax()
        threshold_fh = binsf[tind_fh]
        fid_fh = contrast_fh[tind_fh]

        if plot == True:
            outerFolder_expt = os.path.join(outerfolder_plots, "ss_repeat_meas_efh")
            self.create_folder_if_not_exists(outerFolder_expt)
            outerFolder_expt = os.path.join(outerFolder_expt, "Q" + str(self.QubitIndex + 1))
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt,
                                     f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")

            axs[2].set_title(f"EF Fidelity = {fid * 100:.2f}% \n FH Fidelity = {fid_fh * 100:.2f}%")  # , freq={freq}MHz, gain={gain}")
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)

            # return fid, threshold_ge, theta, ig_new, ie_new, if_new

            # return fid, theta_ef, ie_new, qe_new, if_new, qf_new, threshold_ef
            return fid, fid_fh, theta_ef, theta_fh, ie_new, qe_new, if_new, qf_new, threshold_ef, ih_new, qh_new, threshold_fh


    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)


class GainFrequencySweep:
    def __init__(self, qubit_index, number_of_qubits, n_loops, list_of_all_qubits, experiment, save_figs=False,
                 optimal_lengths=None, output_folder="/default/path/", unmasking_resgain=False):

        self.n_loops = n_loops
        self.num_qubits = number_of_qubits
        self.n_loops = n_loops
        self.qubit_index = qubit_index
        self.list_of_all_qubits = list_of_all_qubits
        self.output_folder = output_folder
        self.expt_name = "Readout_Optimization"
        self.Qubit = 'Q' + str(self.qubit_index)
        self.optimal_lengths = optimal_lengths
        self.number_of_qubits = number_of_qubits
        self.save_figs = save_figs
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.unmasking_resgain = unmasking_resgain

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [qubit_index]

        self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}

    def set_res_gain_ge(self, QUBIT_INDEX, set_gain, num_qubits=6):
        """Sets the gain for the selected qubit to 1, others to 0."""
        res_gain_ge = [0] * num_qubits  # Initialize all gains to 0
        if 0 <= QUBIT_INDEX < num_qubits:  # makes sure you are within the range of options
            res_gain_ge[QUBIT_INDEX] = set_gain  # Set the gain for the selected qubit
        return res_gain_ge

    def run_sweep(self, outerfolder_plots, freq_range, gain_range, freq_steps, gain_steps, fid_states):
        print('output [plots folder path from run_Sweep', outerfolder_plots)
        freq_step_size = (freq_range[1] - freq_range[0]) / freq_steps
        gain_step_size = (gain_range[1] - gain_range[0]) / gain_steps
        results = []

        # Use the optimal readout length for the current qubit
        readout_length = self.optimal_lengths[self.qubit_index]
        print('readout_length for this qubit: ', readout_length)
        for freq_step in range(freq_steps):
            freq = freq_range[0] + freq_step * freq_step_size

            fid_results = []
            for gain_step in range(gain_steps):

                fresh_experiment = copy.deepcopy(self.experiment)
                gain = gain_range[0] + gain_step * gain_step_size
                # print('EF Readout Gain-Freq Optimimization: Running for res_gain: ', gain, '...')
                print('EF Readout Gain-Freq Optimimization: Running for res_freq: ', freq, 'gain', gain)
                # Update config with current gain and frequency values
                fresh_experiment.readout_cfg['res_freq_ef'][self.qubit_index] = freq
                fresh_experiment.readout_cfg[
                    'res_length'] = readout_length  # Set the optimal readout length for the qubit

                res_gains = fresh_experiment.mask_gain_res(self.qubit_index, gain, num_qubits=tot_num_of_qubits)
                fresh_experiment.readout_cfg['res_gain_ef'] = res_gains

                # Initialize SingleShotGE instance for fidelity calculation
                round_num = 0
                # save_figs = False

                fids = []
                for k in range(self.n_loops):  # loops for each read out length
                    # ------------------------Single Shot-------------------------
                    # Initialize experiment for each loop iteration
                    experiment = copy.deepcopy(fresh_experiment)
                    # Set specific configuration values for each iteration

                    # res_gains = experiment.set_gain_filter_ge(QubitIndex, gain)  # Set gain for current qubit only
                    res_gains = experiment.mask_gain_res(self.qubit_index, IndexGain=gain)
                    experiment.readout_cfg['res_gain_ef'] = res_gains

                    if fid_states == 'ge':
                        single_shot = SingleShot(self.qubit_index, self.number_of_qubits, self.output_folder, round_num,
                                                 self.save_figs, fresh_experiment,
                                                 unmasking_resgain=self.unmasking_resgain)
                        fidelity = single_shot.run(fresh_experiment.soccfg, fresh_experiment.soc)
                        # ss = SingleShotGE(self.QubitIndex, number_of_qubits, outerFolder, j, self.save_figs, experiment,
                        #                 unmasking_resgain=unmask)  # updated way
                        fid, angle, iq_list_g, iq_list_e = ss.run()
                        fids.append(fid)
                    elif fid_states == 'ef':

                        ss = SingleShot_ef(self.qubit_index, self.number_of_qubits, self.output_folder, round_num,
                                           self.save_figs, fresh_experiment, unmasking_resgain=self.unmasking_resgain)
                        iq_list_e, iq_list_f, ie_new, if_new, theta_ef, threshold_ef, ef_ssf_cfg, fid = ss.run()
                        fids.append(fid)

                    elif fid_states == 'fh':

                        ss = SingleShot_ef(self.qubit_index, self.number_of_qubits, self.output_folder, round_num,
                                           self.save_figs, fresh_experiment, unmasking_resgain=self.unmasking_resgain)
                        fid, fid_fh, theta_ef, theta_fh, ie_new, qe_new, if_new, qf_new, threshold_ef, ih_new, qh_new, threshold_fh = ss.run()
                        # iq_list_e, iq_list_f, ie_new, if_new,  theta_ef, threshold_ef, self.config, fid
                        # ss = SingleShot_ef(self.QubitIndex, number_of_qubits, outerFolder, j, self.save_figs, experiment,
                        #                    unmasking_resgain=unmask)  # updated way
                        # fid, angle, i_list_e, q_list_e, i_list_f, q_list_f = ss.fidelity_test()
                        fids.append(fid_fh)
                fid_results.append(np.mean(fids))

            results.append(fid_results)


        return results