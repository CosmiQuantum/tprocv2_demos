import datetime
import numpy as np
import logging
from section_005_single_shot_ge import SingleShot

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
from section_005_single_shot_ge import SingleShotProgram_e


# Both g and e during the same experiment.


class Active_Reset(AveragerProgramV2):
    # def __init__(self, soccfg, cfg):
    #     super().__init__(soccfg, cfg)

    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        gen_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        # self.q_index=q_index
        # self.r_thresh1 = 6

        # self.add_reg('thresh1',init = cfg["threshold1"])# * cfg["readout_length"])
        #
        #
        #
        # self.add_reg('thresh2',init = cfg["threshold2"])

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
                       mask=cfg["list_of_all_qubits"],
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
        # self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        # self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])
        ################ Active Reset #################################
        # n = 0
        ################ Active Reset #################################
        self.label("Readout and check conditions")
        # n = n + 1
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

        # Wait for readout to be completed
        self.wait_auto(cfg['res_length'] + 0.2)
        self.delay_auto(cfg['res_length'] + 0.2)

        # Read from ro_ch buffer???
        # print("cfg['ro_ch'][0])", cfg['ro_ch'][0])
        self.read_input(ro_ch=cfg['ro_ch'][0])
        self.write_dmem(addr=0, src='s_port_l')
        self.write_dmem(addr=1, src='s_port_h')

        # if whatever is read from ro_ch is greater or equal to threshold 1, skip to label('skip everything'))
        self.read_and_jump(ro_ch=cfg['ro_ch'][0],
                           component='I',
                           threshold=cfg['qubit_is_in_g_threshold'],
                           test=">=", label='skip everything')

        # if whatever is read from ro_ch is greater or equal to threshold 2 (between_g_and_e), go back to label("Readout and check conditions")
        self.read_and_jump(ro_ch=cfg['ro_ch'][0],
                           component='I',
                           threshold=cfg['edge_of_e_state_threshold'],
                           test=">=", label="Readout and check conditions")

        # print('playing pi in active to move e to g')
        # Play a pi pulse if whatever is read from ro_ch is lesser than both thresholds 1 and 2
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play pulse pi
        self.delay_auto(self.cfg['sigma'] * 4) # ????
        self.jump("Readout and check conditions")
        self.label('skip everything')

        # print('n=', n)
        # print('passed. Moving on')


class Active_Reset_test:
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
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} Active Reset ',
                                   self.config)
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} Active Reset: {self.config}')

        self.q1_t1 = []
        self.q1_t1_err = []
        self.dates = []

    def run(self, ):
        # Run the single shot programs (g and e)
        # self.config['relax_delay']=1
        act = Active_Reset(self.experiment.soccfg, reps=1, final_delay=0.0,
                           cfg=self.config)
        iq_act = act.acquire(self.experiment.soc, soft_avgs=1, progress=True)

        # print('iq_act', iq_act)
        print("feedback readout:", self.experiment.soc.read_mem(2, 'dmem'))
        # print('resL to cycles',self.experiment.soc.us2cycles(self.config['res_length'], ro_ch=self.config['ro_ch'][0]))
        # act_idata = iq_act[self.QubitIndex][-1].T[0]
        # act_qdata = iq_act[self.QubitIndex][-1].T[1]
        shots = act.get_raw()
        print('shots',shots)
        act_idata = shots[self.QubitIndex][:, :, 0, 0]
        act_qdata = shots[self.QubitIndex][:, :, 0, 1]
        print('act_idata from get_raw()',act_idata)
        print('act_qdata from get_raw()', act_qdata)
        # self.config['relax_delay'] = 1
        ssp_e = SingleShotProgram_e(self.experiment.soccfg, reps=1, final_delay=self.config['relax_delay'],
                                    cfg=self.config)
        iq_list_e = ssp_e.acquire(self.experiment.soc, soft_avgs=1, progress=True)
        ss_shots = ssp_e.get_raw()


        no_act_idata = ss_shots[self.QubitIndex][:, :, 0, 0]
        no_act_qdata = ss_shots[self.QubitIndex][:, :, 0, 1]
        self.plot_results(act_idata[0], act_qdata[0], no_act_idata[0], no_act_qdata[0], self.QubitIndex)

        return act_idata, act_qdata, no_act_idata, no_act_qdata, self.config


    # def run(self):
    #     ssp_g = SingleShotProgram_g(self.experiment.soccfg, reps=1, final_delay=self.config['relax_delay'], cfg=self.config)
    #     iq_list_g = ssp_g.acquire(self.experiment.soc, soft_avgs=1, progress=True)

    #     ssp_e = SingleShotProgram_e(self.experiment.soccfg, reps=1, final_delay=self.config['relax_delay'], cfg=self.config)
    #     iq_list_e = ssp_e.acquire(self.experiment.soc, soft_avgs=1, progress=True)

    #     fid, angle = self.plot_results(iq_list_g, iq_list_e, self.QubitIndex)
    #     return fid, angle, iq_list_g, iq_list_e, self.config
    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)


    def plot_results(self, act_idata, act_qdata, no_act_idata, no_act_qdata, QubitIndex, fig_quality=100):
        qe = act_qdata
        ie = act_idata
        qg = no_act_qdata
        ig = no_act_idata
        xg, yg = np.median(ig), np.median(qg)
        xe, ye = np.median(ie), np.median(qe)
        # act_x, act_y = np.median(act_idata), np.median(act_qdata)
        # no_act_x, no_act_y = np.median(no_act_idata), np.median(no_act_qdata)

        # if plot == True:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
        fig.tight_layout()

        axs[0].scatter(ie, qe, label='pi-pulse then active reset', color='b', marker='*', alpha=0.1)

        axs[0].scatter(xe, ye, color='k', marker='o')

        axs[0].scatter(ig, qg, label='pi-pulse only', color='r', marker='*', alpha=0.1)

        axs[0].scatter(xg, yg, color='k', marker='o')

        axs[0].set_xlabel('I (a.u.)')
        axs[0].set_ylabel('Q (a.u.)')
        axs[0].legend(loc='upper right')
        axs[0].set_title('Unrotated Active reset data\n and Rotated Pi pulse only Data')
        axs[0].axis('equal')

        theta = -np.arctan2((ye - yg), (xe - xg))

        """Rotate the IQ data"""
        ig_new = ig * np.cos(theta) - qg * np.sin(theta)
        qg_new = ig * np.sin(theta) + qg * np.cos(theta)
        ie_new = ie * np.cos(theta) - qe * np.sin(theta)
        qe_new = ie * np.sin(theta) + qe * np.cos(theta)

        """New means of each blob"""
        xg, yg = np.median(ig_new), np.median(qg_new)
        xe, ye = np.median(ie_new), np.median(qe_new)

        # theta= -np.arctan2((act_y - no_act_y), (act_x - no_act_x))
        # no_act_idata_new = no_act_idata * np.cos(theta) - no_act_qdata * np.sin(theta)
        # no_act_qdata_new = no_act_idata * np.sin(theta) + no_act_qdata * np.cos(theta)
        # act_idata_new = act_idata * np.cos(theta) - act_qdata * np.sin(theta)
        # act_qdata_new = act_idata * np.sin(theta) + act_qdata * np.cos(theta)
        # no_act_x_new = no_act_x * np.cos(theta) - no_act_y * np.sin(theta)
        # no_act_y_new = no_act_x * np.sin(theta) + no_act_y * np.cos(theta)
        # act_x_new = act_x * np.cos(theta) - act_y * np.sin(theta)
        # act_y_new = act_x * np.sin(theta) + act_y * np.cos(theta)

        axs[1].scatter(ie_new, qe_new, label='pi-pulse then active reset', color='b', marker='*', alpha=0.1)

        axs[1].scatter(xe, ye, color='k', marker='o')

        axs[1].scatter(ig_new, qg_new, label='pi-pulse only', color='r', marker='*', alpha=0.1)

        axs[1].scatter(xg, yg, color='k', marker='o')

        axs[1].set_xlabel('I (a.u.)')
        axs[1].set_ylabel('Q (a.u.)')
        axs[1].legend(loc='upper right')
        axs[1].set_title('Rotated Active reset data\n and Rotated Pi pulse only Data')
        axs[1].axis('equal')

        xlims = [np.min(ig_new), np.max(ie_new)]
        numbins = round(math.sqrt(float(self.config["steps"])))

        """X and Y ranges for histogram"""
        print('len(ig_new)',len(ig_new))
        print('ig_new[0]',ig_new[0])
        ng, binsg, pg = axs[2].hist(ig_new, bins=numbins, range=xlims, color='r', label='pi pulse only', alpha=0.1)
        ne, binse, pe = axs[2].hist(ie_new, bins=numbins, range=xlims, color='b', label='pi pulse then active reset',
                                    alpha=0.1)
        axs[2].set_xlabel('Rotated Idata (a.u.)')
        axs[2].set_ylabel('Counts')
        axs[2].legend(loc='upper right')
        axs[2].set_title('Rotated Active reset data\n and Rotated Pi pulse only Data')
        axs[2].axis('equal')

        self.create_folder_if_not_exists(self.outerFolder)
        outerFolder_expt = os.path.join(self.outerFolder, "Active_reset")
        # self.create_folder_if_not_exists(outerFolder_expt)
        outerFolder_expt = os.path.join(outerFolder_expt, "Q" + str(self.QubitIndex + 1))
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt,
                                 f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")

        fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)

        return

    #     def hist_ssf(self, data=None, cfg=None, plot=True,  fig_quality = 100):

    #         ig = data[0]
    #         qg = data[1]
    #         ie = data[2]
    #         qe = data[3]

    #         numbins = round(math.sqrt(float(cfg["steps"])))

    #         xg, yg = np.median(ig), np.median(qg)
    #         xe, ye = np.median(ie), np.median(qe)

    #         if plot == True:
    #             fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    #             fig.tight_layout()

    #             axs[0].scatter(ig, qg, label='g', color='b', marker='*')
    #             axs[0].scatter(ie, qe, label='e', color='r', marker='*')
    #             axs[0].scatter(xg, yg, color='k', marker='o')
    #             axs[0].scatter(xe, ye, color='k', marker='o')
    #             axs[0].set_xlabel('I (a.u.)')
    #             axs[0].set_ylabel('Q (a.u.)')
    #             axs[0].legend(loc='upper right')
    #             axs[0].set_title('Unrotated')
    #             axs[0].axis('equal')
    #         """Compute the rotation angle"""
    #         theta = -np.arctan2((ye - yg), (xe - xg))
    #         """Rotate the IQ data"""
    #         ig_new = ig * np.cos(theta) - qg * np.sin(theta)
    #         qg_new = ig * np.sin(theta) + qg * np.cos(theta)
    #         ie_new = ie * np.cos(theta) - qe * np.sin(theta)
    #         qe_new = ie * np.sin(theta) + qe * np.cos(theta)

    #         """New means of each blob"""
    #         xg, yg = np.median(ig_new), np.median(qg_new)
    #         xe, ye = np.median(ie_new), np.median(qe_new)

    #         # print(xg, xe)
    #         #xlims = [xg - ran, xg + ran]
    #         xlims = [np.min(ig_new), np.max(ie_new)]

    #         if plot == True:
    #             axs[1].scatter(ig_new, qg_new, label='g', color='b', marker='*')
    #             axs[1].scatter(ie_new, qe_new, label='e', color='r', marker='*')
    #             axs[1].scatter(xg, yg, color='k', marker='o')
    #             axs[1].scatter(xe, ye, color='k', marker='o')
    #             axs[1].set_xlabel('I (a.u.)')
    #             axs[1].legend(loc='lower right')
    #             axs[1].set_title(f'Rotated Theta:{round(theta, 5)}')
    #             axs[1].axis('equal')

    #             """X and Y ranges for histogram"""
    #             ng, binsg, pg = axs[2].hist(ig_new, bins=numbins, range=xlims, color='b', label='g', alpha=0.5)
    #             ne, binse, pe = axs[2].hist(ie_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.5)

    #             axs[2].set_xlabel('I(a.u.)')
    #         else:
    #             ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
    #             ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)

    #         """Compute the fidelity using overlap of the histograms"""
    #         contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
    #         tind = contrast.argmax()
    #         threshold = binsg[tind]
    #         fid = contrast[tind]
    #         #axs[2].set_title(f"Fidelity = {fid * 100:.2f}%")


    #         if plot == True:
    #             self.create_folder_if_not_exists(self.outerFolder)
    #             outerFolder_expt = os.path.join(self.outerFolder, "ss_repeat_meas_ge")
    #             self.create_folder_if_not_exists(outerFolder_expt)
    #             outerFolder_expt = os.path.join(outerFolder_expt, "Q" + str(self.QubitIndex + 1))
    #             self.create_folder_if_not_exists(outerFolder_expt)
    #             now = datetime.datetime.now()
    #             formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
    #             file_name = os.path.join(outerFolder_expt,
    #                                      f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")

    #             axs[2].set_title(f"Fidelity = {fid * 100:.2f}%")
    #             fig.savefig(file_name,  dpi=fig_quality, bbox_inches='tight')
    #             plt.close(fig)

    #         return fid, threshold, theta, ig_new, ie_new

    #     def only_hist_ssf(self, data=None, cfg=None, plot=True, fig_quality=100, plot_title="Run 3"):
    #         import math
    #         import numpy as np
    #         import matplotlib.pyplot as plt
    #         import os
    #         import datetime

    #         # Unpack IQ data
    #         ig = data[0]
    #         qg = data[1]
    #         ie = data[2]
    #         qe = data[3]

    #         # Determine number of bins for the histogram
    #         numbins = round(math.sqrt(float(cfg["steps"])))

    #         # Compute medians (used for rotation angle calculation)
    #         xg, yg = np.median(ig), np.median(qg)
    #         xe, ye = np.median(ie), np.median(qe)

    #         # Compute rotation angle
    #         theta = -np.arctan2((ye - yg), (xe - xg))

    #         # Rotate the IQ data
    #         ig_new = ig * np.cos(theta) - qg * np.sin(theta)
    #         qg_new = ig * np.sin(theta) + qg * np.cos(theta)
    #         ie_new = ie * np.cos(theta) - qe * np.sin(theta)
    #         qe_new = ie * np.sin(theta) + qe * np.cos(theta)

    #         # New medians after rotation (not used further in plotting)
    #         xg, yg = np.median(ig_new), np.median(qg_new)
    #         xe, ye = np.median(ie_new), np.median(qe_new)

    #         # Define histogram range from the rotated ground state to the excited state
    #         xlims = [np.min(ig_new), np.max(ie_new)]
    #         ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
    #         ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)
    #         # Compute the fidelity using the overlap of the histograms
    #         contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) /
    #                            (0.5 * ng.sum() + 0.5 * ne.sum())))
    #         tind = contrast.argmax()
    #         threshold = binsg[tind]
    #         fid = contrast[tind]
    #         if plot:
    #             # Create figure and axis for the histogram
    #             fig, ax = plt.subplots(figsize=(8, 6))

    #             # Plot histogram for ground state and first excited state with updated labels
    #             ng, binsg, _ = ax.hist(ig_new, bins=numbins, range=xlims, color='b',
    #                                    label='Ground', alpha=0.5)
    #             ne, binse, _ = ax.hist(ie_new, bins=numbins, range=xlims, color='r',
    #                                    label='First Excited State', alpha=0.5)

    #             # Set axis labels with 12-point font
    #             ax.set_xlabel('I (a.u.)', fontsize=12)
    #             ax.set_ylabel('Counts', fontsize=12)
    #             # Set plot title using the provided parameter
    #             ax.set_title(plot_title + f'   SSF: {int(fid * 100)}%', fontsize=12)
    #             ax.legend()

    #             # Save the figure
    #             self.create_folder_if_not_exists(self.outerFolder)
    #             outerFolder_expt = os.path.join(self.outerFolder, "ss_repeat_meas_ge")
    #             self.create_folder_if_not_exists(outerFolder_expt)
    #             outerFolder_expt = os.path.join(outerFolder_expt, "Q" + str(self.QubitIndex + 1))
    #             self.create_folder_if_not_exists(outerFolder_expt)
    #             now = datetime.datetime.now()
    #             formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
    #             file_name = os.path.join(outerFolder_expt,
    #                                      f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}.png")
    #             fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
    #             plt.close(fig)


    #         return fid, threshold, theta, ig_new, ie_new

    #     def create_folder_if_not_exists(self, folder):
    #         """Creates a folder at the given path if it doesn't already exist."""
    #         if not os.path.exists(folder):
    #             os.makedirs(folder)


# class GainFrequencySweep:
#     def __init__(self,qubit_index, number_of_qubits, list_of_all_qubits, experiment, optimal_lengths=None, output_folder="/default/path/", unmasking_resgain = False):
#         self.qubit_index = qubit_index
#         self.list_of_all_qubits = list_of_all_qubits
#         self.output_folder = output_folder
#         self.expt_name = "Readout_Optimization"
#         self.Qubit = 'Q' + str(self.qubit_index)
#         self.optimal_lengths = optimal_lengths
#         self.number_of_qubits = number_of_qubits

#         self.experiment = experiment
#         self.exp_cfg = expt_cfg[self.expt_name]
#         self.unmasking_resgain = unmasking_resgain

#         if unmasking_resgain:
#             self.exp_cfg["list_of_all_qubits"] = [qubit_index]

#         self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
#         self.config = {**self.q_config[self.Qubit], **self.exp_cfg}

#     def set_res_gain_ge(self, QUBIT_INDEX, set_gain, num_qubits=6):
#         """Sets the gain for the selected qubit to 1, others to 0."""
#         res_gain_ge = [0] * num_qubits  # Initialize all gains to 0
#         if 0 <= QUBIT_INDEX < num_qubits:  # makes sure you are within the range of options
#             res_gain_ge[QUBIT_INDEX] = set_gain  # Set the gain for the selected qubit
#         return res_gain_ge

#     def run_sweep(self, freq_range, gain_range, freq_steps, gain_steps):
#         freq_step_size = (freq_range[1] - freq_range[0]) / freq_steps
#         gain_step_size = (gain_range[1] - gain_range[0]) / gain_steps
#         results = []

#         # Use the optimal readout length for the current qubit
#         readout_length = self.optimal_lengths[self.qubit_index]
#         for freq_step in range(freq_steps):
#             freq = freq_range[0] + freq_step * freq_step_size
#             #print('Running for res_freq: ', freq, '...')
#             fid_results = []
#             for gain_step in range(gain_steps):
#                 #experiment = QICK_experiment(self.output_folder)
#                 #experiment = QICK_experiment(self.output_folder, DAC_attenuator1=10, DAC_attenuator2=5, ADC_attenuator=10)
#                 fresh_experiment = copy.deepcopy(self.experiment)
#                 gain = gain_range[0] + gain_step * gain_step_size


#                 # Update config with current gain and frequency values
#                 fresh_experiment.readout_cfg['res_freq_ge'][self.qubit_index]= freq
#                 fresh_experiment.readout_cfg['res_length'] = readout_length  # Set the optimal readout length for the qubit

#                 res_gains = fresh_experiment.mask_gain_res(self.qubit_index, gain, num_qubits=tot_num_of_qubits)
#                 fresh_experiment.readout_cfg['res_gain_ge'] = res_gains

#                 # Initialize SingleShotGE instance for fidelity calculation
#                 round_num = 0
#                 save_figs = False
#                 single_shot = SingleShotGE(self.qubit_index, self.number_of_qubits,  self.output_folder, round_num, save_figs, fresh_experiment, unmasking_resgain = self.unmasking_resgain)
#                 fidelity = single_shot.fidelity_test(fresh_experiment.soccfg, fresh_experiment.soc)
#                 fid_results.append(fidelity)
#                 del fresh_experiment
#                 del single_shot

#             results.append(fid_results)

#         return results