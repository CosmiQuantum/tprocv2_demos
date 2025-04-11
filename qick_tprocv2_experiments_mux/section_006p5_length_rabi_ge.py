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

class LengthRabiExperiment:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, signal, save_figs, experiment = None,
                 live_plot = None, increase_qubit_reps = False, qubit_to_increase_reps_for = None,
                 multiply_qubit_reps_by = 0, verbose = False, logger = None, qick_verbose=True, QZE=False,
                 projective_readout_pulse_len_us=9,  time_between_projective_readout_pulses=None, zeno_pulse_gain=None):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder

        self.QZE = QZE
        if self.QZE:
            self.expt_name = "length_rabi_ge_qze"
            self.Qubit = 'Q' + str(self.QubitIndex)
            self.exp_cfg = expt_cfg[self.expt_name]
            self.round_num = round_num
            self.live_plot = live_plot
            self.signal = signal
            self.save_figs = save_figs
            self.experiment = experiment
            self.verbose = verbose
            self.zeno_pulse_gain = zeno_pulse_gain

            self.projective_readout_pulse_len_us = projective_readout_pulse_len_us
            self.time_between_projective_readout_pulses=time_between_projective_readout_pulses
            self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")
            qze_mask = np.arange(0, self.number_of_qubits + 1)
            qze_mask = np.delete(qze_mask, QubitIndex)
            self.exp_cfg['qze_mask'] = qze_mask

            self.experiment.readout_cfg['res_gain_qze'] = [self.experiment.readout_cfg['res_gain_ge'][QubitIndex],0,0,0,0,0,self.zeno_pulse_gain]
            self.experiment.readout_cfg['res_freq_qze'] = self.experiment.readout_cfg['res_freq_ge']
            self.experiment.readout_cfg['res_phase_qze'] = self.experiment.readout_cfg['res_phase']
            if len(self.experiment.readout_cfg['res_freq_qze']) <7: #otherise it keeps appending
                self.experiment.readout_cfg['res_freq_qze'].append(experiment.readout_cfg['res_freq_qze'][self.QubitIndex])
                self.experiment.readout_cfg['res_phase_qze'].append(experiment.readout_cfg['res_phase_qze'][self.QubitIndex])

        else:
            self.expt_name = "length_rabi_ge"
            self.Qubit = 'Q' + str(self.QubitIndex)
            self.exp_cfg = expt_cfg[self.expt_name]
            self.round_num = round_num
            self.live_plot = live_plot
            self.signal = signal
            self.save_figs = save_figs
            self.experiment = experiment
            self.verbose = verbose
        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            if increase_qubit_reps:
                    if self.QubitIndex==qubit_to_increase_reps_for:
                        if self.verbose: print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.logger.info(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.config["reps"] *= multiply_qubit_reps_by
            #self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} Rabi configuration: {self.config}')
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} Rabi configuration: ', self.config)

    def run(self, thresholding=False, constant_zeno_pulse=False):
        amp_rabi = LengthRabiProgram(
            self.experiment.soccfg,
            reps=self.config['reps'],
            final_delay=self.config['relax_delay'],
            cfg=self.config
        )
        iq_list = amp_rabi.acquire(
            self.experiment.soc,
            soft_avgs=self.config["rounds"],
            progress=self.qick_verbose
        )

        I = iq_list[self.QubitIndex][0, :, 0]
        Q = iq_list[self.QubitIndex][0, :, 1]

        #get the lens that were used so you can use to plot on the x axis
        lengths = amp_rabi.get_pulse_param('qubit_pulse', "length", as_array=True)
        #lens = amp_rabi.get_pulse_param('qubit_pulse', "gain", as_array=True)

        q1_fit_cosine, pi_len = self.plot_results( I, Q, lengths, config = self.config)
        return I, Q, lengths, q1_fit_cosine, pi_len, self.config

    def run_QZE(self,constant_zeno_pulse=False,adapt_qubit_freq=False, wait_for_res_ring_up=False, exp=None):
        qubit_length_ge_loop = np.linspace(self.config['start'], self.config['stop'], self.config['steps'])
        lengths = []
        I = []
        Q = []
        Magnitude =[]
        for row_idx, length in enumerate(qubit_length_ge_loop):
            updated_config = deepcopy(self.config)
            updated_config['qubit_length_ge'] = length
            print('updated_config[qubit_length_ge]: ', round(updated_config['qubit_length_ge'],4), ' updated_config[res_gain_qze]: ',[round(float(n),4) for n in updated_config['res_gain_qze']])

            if constant_zeno_pulse:
                if adapt_qubit_freq:
                    if updated_config['qubit_length_ge'] > 0.11:
                        exp_spec=deepcopy(exp)
                        exp_spec.qubit_cfg['qubit_length_ge'] = length #update the length of the qubit pulse drive, this is used for zeno/stark pulse inside of qspec
                        q_spec = QubitSpectroscopy(self.QubitIndex, tot_num_of_qubits, "/data/QICK_data/run6/6transmon/QZE/QZE_measurement/Documentation/", 0,
                                                   'None', save_figs=True, experiment=exp_spec,
                                                   live_plot=False, verbose=False,
                                                   qick_verbose=True, zeno_stark=True, zeno_stark_pulse_gain=self.zeno_pulse_gain) #update the zeno gain inside the qspec class when redefining the lists for a 7th channel

                        (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit,
                         qubit_freq, sys_config_qspec) = q_spec.run_with_stark_tone(length, wait_for_res_ring_up=wait_for_res_ring_up)
                        del exp_spec
                        updated_config['qubit_freq_ge_starked'][self.QubitIndex] = qubit_freq
                        print(qubit_freq)
                        print('-------------------------')

                if wait_for_res_ring_up:
                    amp_rabi = QZE_constant_pulse_RabiProgram_WaitForResRingUp(
                        self.experiment.soccfg,
                        reps=updated_config['reps'],
                        final_delay=updated_config['relax_delay'],
                        cfg=updated_config
                    )
                else:
                    amp_rabi = QZE_constant_pulse_RabiProgram(
                        self.experiment.soccfg,
                        reps=updated_config['reps'],
                        final_delay=updated_config['relax_delay'],
                        cfg=updated_config
                    )
            else:
                amp_rabi = QZERabiProgram(
                    self.experiment.soccfg,
                    reps=updated_config['reps'],
                    final_delay=updated_config['relax_delay'],
                    cfg=updated_config
                )
            iq_list = amp_rabi.acquire(
                self.experiment.soc,
                soft_avgs=updated_config["rounds"],
                progress=self.qick_verbose
            )

            I.append(iq_list[self.QubitIndex][:, 0][0]) #just a 2d list here becasue we arent doing a qick loop
            Q.append(iq_list[self.QubitIndex][:, 1][0]) #just a 2d list here becasue we arent doing a qick loop
            Magnitude.append(np.abs(iq_list[0].dot([1, 1j]))[0])
            #now I Q and mag should all by a list with a single float in them

            # get the lengs that were used so you can use to plot on the x axis
            lengths.append(amp_rabi.get_pulse_param('qubit_pulse', "length")) #should be just a float by default

            del updated_config


        I=np.asarray(I)
        Q = np.asarray(Q)
        lengths=np.asarray(lengths)
        Magnitude = np.asarray(Magnitude)
        q1_fit_cosine, pi_len = self.plot_results(I, Q, lengths, config=self.config)
        return I, Q, Magnitude, lengths, q1_fit_cosine, pi_len, self.config

    def run_QZE_one_starked_qfreq(self,constant_zeno_pulse=False,adapt_qubit_freq=False, wait_for_res_ring_up=False,
                                  exp=None,optimizationFolder=None, hold_ground=False,three_pulse_binary=False):
        exp_spec = deepcopy(exp)
        exp_spec.qubit_cfg[
            'qubit_length_ge'] = 0.2   #1us because why not, it shouldnt matter that much what is chosen here
        exp_spec.qubit_cfg['qubit_gain_ge'][
            self.QubitIndex] = 0.13  # turn it down for this qspec finding to lower err bars and minimize broadening

        q_spec = QubitSpectroscopy(self.QubitIndex, tot_num_of_qubits,
                                   "/data/QICK_data/run6/6transmon/QZE/QZE_measurement/Documentation/", 0,
                                   'None', save_figs=True, experiment=exp_spec,
                                   live_plot=False, verbose=False,
                                   qick_verbose=True, zeno_stark=True,
                                   zeno_stark_pulse_gain=self.zeno_pulse_gain)  # update the zeno gain inside the qspec class when redefining the lists for a 7th channel

        (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit,
         qubit_freq, sys_config_qspec, fwhm) = q_spec.run_with_stark_tone(0.2, wait_for_res_ring_up=wait_for_res_ring_up)
        ######################################## save qspec data #####################################
        def create_data_dict(keys, save_r, qs):
            return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}
        qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num', 'Recycled QFreq',
                      'Exp Config', 'Syst Config']
        qspec_data = create_data_dict(qspec_keys, 1, list_of_all_qubits)
        qspec_data[self.QubitIndex]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
        qspec_data[self.QubitIndex]['I'][0] = qspec_I
        qspec_data[self.QubitIndex]['Q'][0] = qspec_Q
        qspec_data[self.QubitIndex]['Frequencies'][0] = qspec_freqs
        qspec_data[self.QubitIndex]['I Fit'][0] = qspec_I_fit
        qspec_data[self.QubitIndex]['Q Fit'][0] = qspec_Q_fit
        qspec_data[self.QubitIndex]['Round Num'][0] = 0
        qspec_data[self.QubitIndex]['Batch Num'][0] = 0
        qspec_data[self.QubitIndex]['Recycled QFreq'][0] = False  # no rr so no recycling here
        qspec_data[self.QubitIndex]['Exp Config'][0] = expt_cfg
        qspec_data[self.QubitIndex]['Syst Config'][0] = sys_config_qspec
        from section_008_save_data_to_h5 import Data_H5
        saver_qspec = Data_H5(optimizationFolder, qspec_data, 0, 1)
        saver_qspec.save_to_h5('QSpec_starked')
        del saver_qspec
        del qspec_data


        del exp_spec

        self.config['qubit_freq_ge_starked'][self.QubitIndex] = qubit_freq #use for all lengths on x axis

        self.config['fwhm_w01_starked'] = fwhm
        print(qubit_freq)
        print('-------------------------')

        qubit_length_ge_loop = np.linspace(self.config['start'], self.config['stop'], self.config['steps'])
        lengths = []
        I = []
        Q = []
        Magnitude =[]

        for row_idx, length in enumerate(qubit_length_ge_loop):
            updated_config = deepcopy(self.config)
            updated_config['qubit_length_ge'] = length
            print('updated_config[qubit_gain_ge]: ', round(updated_config['qubit_gain_ge'],4), 'updated_config[qubit_length_ge]: ', round(updated_config['qubit_length_ge'],4), ' updated_config[res_gain_qze]: ',[round(float(n),4) for n in updated_config['res_gain_qze']])

            if constant_zeno_pulse:
            #
            #     if wait_for_res_ring_up:
            #         amp_rabi = QZE_constant_pulse_RabiProgram_WaitForResRingUp(
            #             self.experiment.soccfg,
            #             reps=updated_config['reps'],
            #             final_delay=updated_config['relax_delay'],
            #             cfg=updated_config
            #         )
            #     else:
                if adapt_qubit_freq:
                    if hold_ground:
                        if three_pulse_binary:
                            amp_rabi = QZE_constant_pulse_3pulse_RabiProgram(
                                self.experiment.soccfg,
                                reps=updated_config['reps'],
                                final_delay=updated_config['relax_delay'],
                                cfg=updated_config
                            )
                        else:
                            amp_rabi = QZE_constant_pulse_gnd_RabiProgram(
                                self.experiment.soccfg,
                                reps=updated_config['reps'],
                                final_delay=updated_config['relax_delay'],
                                cfg=updated_config
                            )
                    else:
                        amp_rabi = QZE_constant_pulse_RabiProgram(
                            self.experiment.soccfg,
                            reps=updated_config['reps'],
                            final_delay=updated_config['relax_delay'],
                            cfg=updated_config
                        )


                else:
                    amp_rabi = QZE_constant_pulse_RabiProgram_unstarked_freq(
                        self.experiment.soccfg,
                        reps=updated_config['reps'],
                        final_delay=updated_config['relax_delay'],
                        cfg=updated_config
                    )
            else:

                amp_rabi = QZERabiProgram(
                    self.experiment.soccfg,
                    reps=updated_config['reps'],
                    final_delay=updated_config['relax_delay'],
                    cfg=updated_config
                )
            iq_list = amp_rabi.acquire(
                self.experiment.soc,
                soft_avgs=updated_config["rounds"],
                progress=self.qick_verbose
            )

            I.append(iq_list[self.QubitIndex][:, 0][0]) #just a 2d list here becasue we arent doing a qick loop
            Q.append(iq_list[self.QubitIndex][:, 1][0]) #just a 2d list here becasue we arent doing a qick loop
            Magnitude.append(np.abs(iq_list[0].dot([1, 1j]))[0])
            #now I Q and mag should all by a list with a single float in them

            # get the lengs that were used so you can use to plot on the x axis
            lengths.append(amp_rabi.get_pulse_param('qubit_pulse', "length")) #should be just a float by default

            del updated_config


        I=np.asarray(I)
        Q = np.asarray(Q)
        lengths=np.asarray(lengths)
        Magnitude = np.asarray(Magnitude)
        q1_fit_cosine, pi_len = self.plot_results(I, Q, lengths, config=self.config)
        return I, Q, Magnitude, lengths, q1_fit_cosine, pi_len, self.config

    def run_oscilliscope_simple(self, thresholding=False):

        prog = OscilliscopeExampleProgram(self.experiment.soccfg, reps=1, final_delay=0.1, cfg=self.config)
        iq_list = prog.acquire_decimated(self.experiment.soc, soft_avgs=self.config['soft_avgs'])


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

    def run_oscilliscope_zeno(self, thresholding=False):
        qubit_length_ge_loop = np.linspace(self.config['start'], self.config['stop'], self.config['steps'])
        zeno_gain_ge_loop = np.linspace(0.2, 1, self.config['steps'])

        num_rows = len(qubit_length_ge_loop)
        num_cols = len(zeno_gain_ge_loop)

        fig, axs = plt.subplots(num_rows, num_cols, sharex='col', figsize=(10 * num_cols, 3 * num_rows))

        if num_rows == 1 and num_cols == 1:
            axs = np.array([[axs]])
        elif num_rows == 1:
            axs = np.array([axs])
        elif num_cols == 1:
            axs = np.array([[ax] for ax in axs])

        for col_idx, zeno_gain in enumerate(zeno_gain_ge_loop):
            for row_idx, length in enumerate(qubit_length_ge_loop):
                updated_config = deepcopy(self.config)
                updated_config['qubit_length_ge'] = length
                updated_config['res_gain_qze'][-1] = zeno_gain

                prog = OscilliscopeQZEProgram(self.experiment.soccfg, reps=1, final_delay=0.5, cfg=updated_config)
                iq_list = prog.acquire_decimated(self.experiment.soc, soft_avgs=updated_config['soft_avgs'])

                I = iq_list[self.QubitIndex][:, 0]
                Q = iq_list[self.QubitIndex][:, 1]
                t = prog.get_time_axis(ro_index=0)
                magnitude = np.abs(iq_list[0].dot([1, 1j]))

                ax = axs[row_idx, col_idx]
                ax.plot(t, I, label="I value")
                ax.plot(t, Q, label="Q value")
                ax.plot(t, magnitude, label="magnitude")
                ax.set_title(f"Zeno gain: {round(zeno_gain,6)}, Qubit drive length: {round(length, 6)}")
                ax.set_ylabel("a.u.")
                if row_idx == num_rows - 1:
                    ax.set_xlabel("us")
                else:
                    ax.set_xlabel("")

                if row_idx == 0 and col_idx == num_cols - 1:
                    ax.legend(loc='upper right', prop={'size': 12})

        plt.tight_layout()

        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt,
                                 f"Q{self.QubitIndex + 1}_{formatted_datetime}_pulses.png")
        fig.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.show()



    def live_plotting(self, amp_rabi, thresholding):
        I = Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        if not viz.check_connection(timeout_seconds=5):
            raise RuntimeError("Visdom server not connected!")

        for ii in range(self.config["rounds"]):
            if thresholding:
                iq_list = amp_rabi.acquire(self.experiment.soc, soft_avgs=1,
                                           threshold=self.experiment.readout_cfg["threshold"],
                                           angle=self.experiment.readout_cfg["ro_phase"], progress=self.qick_verbose)
            else:
                iq_list = amp_rabi.acquire(self.experiment.soc, soft_avgs=1, progress=self.qick_verbose)
            lens = amp_rabi.get_pulse_param('qubit_pulse', "gain", as_array=True)

            this_I = iq_list[self.QubitIndex][0, :, 0]
            this_Q = iq_list[self.QubitIndex][0, :, 1]

            if I is None:  # ii == 0
                I, Q = this_I, this_Q
            else:
                I = (I * ii + this_I) / (ii + 1.0)
                Q = (Q * ii + this_Q) / (ii + 1.0)

            viz.line(X=lens, Y=I, opts=dict(height=400, width=700, title='Rabi I', showlegend=True, xlabel='expt_pts'),win='Rabi_I')
            viz.line(X=lens, Y=Q, opts=dict(height=400, width=700, title='Rabi Q', showlegend=True, xlabel='expt_pts'),win='Rabi_Q')
        return I, Q, lens

    def cosine(self, x, a, b, c, d):

        return a * np.cos(2. * np.pi * b * x - c * 2 * np.pi) + d

    def plot_results(self, I, Q, lens, config = None, fig_quality = 100, showfig=False):
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            plt.rcParams.update({'font.size': 18})

            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

            q1_a_guess_I = (np.max(I) - np.min(I)) / 2
            q1_d_guess_I = np.mean(I)
            q1_a_guess_Q = (np.max(Q) - np.min(Q)) / 2
            q1_d_guess_Q = np.mean(Q)
            q1_b_guess = 1 / lens[-1]
            q1_c_guess = 0

            q1_guess_I = [q1_a_guess_I, q1_b_guess, q1_c_guess, q1_d_guess_I]
            q1_popt_I, q1_pcov_I = curve_fit(self.cosine, lens, I, maxfev=100000, p0=q1_guess_I)
            q1_fit_cosine_I = self.cosine(lens, *q1_popt_I)

            q1_guess_Q = [q1_a_guess_Q, q1_b_guess, q1_c_guess, q1_d_guess_Q]
            q1_popt_Q, q1_pcov_Q = curve_fit(self.cosine, lens, Q, maxfev=100000, p0=q1_guess_Q)
            q1_fit_cosine_Q = self.cosine(lens, *q1_popt_Q)

            first_three_avg_I = np.mean(q1_fit_cosine_I[:3])
            last_three_avg_I = np.mean(q1_fit_cosine_I[-3:])
            first_three_avg_Q = np.mean(q1_fit_cosine_Q[:3])
            last_three_avg_Q = np.mean(q1_fit_cosine_Q[-3:])

            best_signal_fit = None
            pi_len = None
            if 'Q' in self.signal:
                best_signal_fit = q1_fit_cosine_Q
                # figure out if you should take the min or the max value of the fit to say where pi_len should be
                if last_three_avg_Q > first_three_avg_Q:
                    pi_len = lens[np.argmax(best_signal_fit)]
                else:
                    pi_len = lens[np.argmin(best_signal_fit)]
            if 'I' in self.signal:
                best_signal_fit = q1_fit_cosine_I
                # figure out if you should take the min or the max value of the fit to say where pi_len should be
                if last_three_avg_I > first_three_avg_I:
                    pi_len = lens[np.argmax(best_signal_fit)]
                else:
                    pi_len = lens[np.argmin(best_signal_fit)]
            if 'None' in self.signal:
                # choose the best signal depending on which has a larger magnitude
                if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
                    best_signal_fit = q1_fit_cosine_Q
                    # figure out if you should take the min or the max value of the fit to say where pi_len should be
                    if last_three_avg_Q > first_three_avg_Q:
                        pi_len = lens[np.argmax(best_signal_fit)]
                    else:
                        pi_len = lens[np.argmin(best_signal_fit)]
                else:
                    best_signal_fit = q1_fit_cosine_I
                    # figure out if you should take the min or the max value of the fit to say where pi_len should be
                    if last_three_avg_I > first_three_avg_I:
                        pi_len = lens[np.argmax(best_signal_fit)]
                    else:
                        pi_len = lens[np.argmin(best_signal_fit)]
            else:
                print('Invalid signal passed, please do I Q or None')


            ax2.plot(lens, q1_fit_cosine_Q, '-', color='red', linewidth=3, label="Fit")
            ax1.plot(lens, q1_fit_cosine_I, '-', color='red', linewidth=3, label="Fit")

            if config is not None:
                if self.QZE:
                    fig.text(plot_middle, 0.98,
                             f"Rabi Q{self.QubitIndex + 1}_" + f", {config['reps']}*{config['rounds']} avgs" 
                                                                                                               f'zenopulse gain'
                                                                                                               f': {self.experiment.readout_cfg["res_gain_ge"][self.QubitIndex]}'
                                                                                                               f' readout pulse amp: '
                                                                                                               f' {self.experiment.readout_cfg["res_gain_ge"][self.QubitIndex]} ',
                             fontsize=24, ha='center',
                             va='top')  # f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back

                else:
                    fig.text(plot_middle, 0.98,
                             f"Rabi Q{self.QubitIndex + 1}_"  + f", {config['reps']}*{config['rounds']} avgs",
                             fontsize=24, ha='center', va='top') #f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back

            else:
                fig.text(plot_middle, 0.98,
                         f"Rabi Q{self.QubitIndex + 1}_" f", {self.config['sigma'] * 1000} ns sigma" + f", {self.config['reps']}*{self.config['rounds']} avgs",
                         fontsize=24, ha='center', va='top')

            ax1.plot(lens, I, label="Qubit drive pulse length (us)", linewidth=2)
            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)

            ax2.plot(lens, Q, label="Q", linewidth=2)
            ax2.set_xlabel("Qubit drive pulse length (us)", fontsize=20)
            ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
            ax2.tick_params(axis='both', which='major', labelsize=16)

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            if showfig:
                plt.show()
            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)
            return best_signal_fit, pi_len

        except Exception as e:
            if self.verbose: print("Error fitting cosine:", e)
            self.logger.info("Error fitting cosine: {e}")
            # Return None if the fit didn't work
            return None, None

    def plot_QZE(self, I, Q, lens, config=None, fig_quality=100):
        try:
            # Create the figure with two subplots for I and Q
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            plt.rcParams.update({'font.size': 18})
            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

            best_signal_fits = []
            pi_lens = []

            # Loop over each measurement in the list(s)
            for idx in range(len(I)):
                current_I = I[idx]
                current_Q = Q[idx]
                current_lens = lens[idx]

                # Calculate initial guesses from the current measurement
                q1_a_guess_I = (np.max(current_I) - np.min(current_I)) / 2
                q1_d_guess_I = np.mean(current_I)
                q1_a_guess_Q = (np.max(current_Q) - np.min(current_Q)) / 2
                q1_d_guess_Q = np.mean(current_Q)
                q1_b_guess = 1 / current_lens[-1]
                q1_c_guess = 0

                # Fit for I
                q1_guess_I = [q1_a_guess_I, q1_b_guess, q1_c_guess, q1_d_guess_I]
                q1_popt_I, q1_pcov_I = curve_fit(self.cosine, current_lens, current_I,
                                                 maxfev=100000, p0=q1_guess_I)
                q1_fit_cosine_I = self.cosine(current_lens, *q1_popt_I)

                # Fit for Q
                q1_guess_Q = [q1_a_guess_Q, q1_b_guess, q1_c_guess, q1_d_guess_Q]
                q1_popt_Q, q1_pcov_Q = curve_fit(self.cosine, current_lens, current_Q,
                                                 maxfev=100000, p0=q1_guess_Q)
                q1_fit_cosine_Q = self.cosine(current_lens, *q1_popt_Q)

                # Calculate average values from the fits for deciding the best signal
                first_three_avg_I = np.mean(q1_fit_cosine_I[:3])
                last_three_avg_I = np.mean(q1_fit_cosine_I[-3:])
                first_three_avg_Q = np.mean(q1_fit_cosine_Q[:3])
                last_three_avg_Q = np.mean(q1_fit_cosine_Q[-3:])

                best_signal_fit = None
                pi_len = None

                # Determine which signal to use based on self.signal
                if 'Q' in self.signal:
                    best_signal_fit = q1_fit_cosine_Q
                    if last_three_avg_Q > first_three_avg_Q:
                        pi_len = current_lens[np.argmax(best_signal_fit)]
                    else:
                        pi_len = current_lens[np.argmin(best_signal_fit)]
                elif 'I' in self.signal:
                    best_signal_fit = q1_fit_cosine_I
                    if last_three_avg_I > first_three_avg_I:
                        pi_len = current_lens[np.argmax(best_signal_fit)]
                    else:
                        pi_len = current_lens[np.argmin(best_signal_fit)]
                elif 'None' in self.signal:
                    if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
                        best_signal_fit = q1_fit_cosine_Q
                        if last_three_avg_Q > first_three_avg_Q:
                            pi_len = current_lens[np.argmax(best_signal_fit)]
                        else:
                            pi_len = current_lens[np.argmin(best_signal_fit)]
                    else:
                        best_signal_fit = q1_fit_cosine_I
                        if last_three_avg_I > first_three_avg_I:
                            pi_len = current_lens[np.argmax(best_signal_fit)]
                        else:
                            pi_len = current_lens[np.argmin(best_signal_fit)]
                else:
                    print('Invalid signal passed, please do I, Q, or None')

                best_signal_fits.append(best_signal_fit)
                pi_lens.append(pi_len)

                # Plot the fits and original data for this measurement.
                # Labels include the measurement index so each can be distinguished.
                ax1.plot(current_lens, q1_fit_cosine_I, '-', linewidth=3, label=f"Fit I {idx + 1}")
                ax2.plot(current_lens, q1_fit_cosine_Q, '-', linewidth=3, label=f"Fit Q {idx + 1}")
                ax1.plot(current_lens, current_I, label=f"I Data {idx + 1}", linewidth=2)
                ax2.plot(current_lens, current_Q, label=f"Q Data {idx + 1}", linewidth=2)

            # Use the last measurement's pi_len for the title/annotation (you can change this logic as needed)
            last_pi_len = pi_lens[-1] if pi_lens else None
            if config is not None:
                if self.QZE:
                    fig.text(plot_middle, 0.98,
                             f"Rabi Q{self.QubitIndex + 1}_, {config['reps']}*{config['rounds']} avgs, pi_len {round(last_pi_len, 2)} "
                             f"projective readout pulse length: {self.projective_readout_pulse_len_us} "
                             f"readout pulse amp: {self.experiment.readout_cfg['res_gain_ge'][self.QubitIndex]}",
                             fontsize=24, ha='center', va='top')
                else:
                    fig.text(plot_middle, 0.98,
                             f"Rabi Q{self.QubitIndex + 1}_, {config['reps']}*{config['rounds']} avgs, pi_len {last_pi_len}",
                             fontsize=24, ha='center', va='top')
            else:
                fig.text(plot_middle, 0.98,
                         f"Rabi Q{self.QubitIndex + 1}_, {self.config['sigma'] * 1000} ns sigma, pi_len {last_pi_len}, {self.config['reps']}*{self.config['rounds']} avgs",
                         fontsize=24, ha='center', va='top')

            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
            ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
            ax2.set_xlabel("Gain (a.u.)", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)
            ax2.tick_params(axis='both', which='major', labelsize=16)
            ax1.legend()
            ax2.legend()

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(outerFolder_expt,
                                         f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)
            return best_signal_fits, pi_lens

        except Exception as e:
            if self.verbose:
                print("Error fitting cosine:", e)
            self.logger.info(f"Error fitting cosine: {e}")
            return None, None

    def roll(self, data: np.ndarray) -> np.ndarray:

        kernel = np.ones(5) / 5
        smoothed = np.convolve(data, kernel, mode='valid')

        # Preserve the original array's shape by padding the edges
        pad_size = (len(data) - len(smoothed)) // 2
        return np.concatenate((data[:pad_size], smoothed, data[-pad_size:]))

    def get_results(self, I, Q, lens, grab_depths = False, rolling_avg=False):
        if rolling_avg:
            I = self.roll(I)
            Q = self.roll(Q)

            first_three_avg_I = np.mean(I[:3])
            last_three_avg_I = np.mean(I[-3:])
            first_three_avg_Q = np.mean(Q[:3])
            last_three_avg_Q = np.mean(Q[-3:])
            if 'Q' in self.signal:
                best_signal = Q
                # figure out if you should take the min or the max value of the fit to say where pi_len should be
                if last_three_avg_Q > first_three_avg_Q:
                    pi_len = lens[np.argmax(best_signal)]
                else:
                    pi_len = lens[np.argmin(best_signal)]
            if 'I' in self.signal:
                best_signal = I
                # figure out if you should take the min or the max value of the fit to say where pi_len should be
                if last_three_avg_I > first_three_avg_I:
                    pi_len = lens[np.argmax(best_signal)]
                else:
                    pi_len = lens[np.argmin(best_signal)]
            if 'None' in self.signal:
                # choose the best signal depending on which has a larger magnitude
                if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
                    best_signal = Q
                    # figure out if you should take the min or the max value of the fit to say where pi_len should be
                    if last_three_avg_Q > first_three_avg_Q:
                        pi_len = lens[np.argmax(best_signal)]
                    else:
                        pi_len = lens[np.argmin(best_signal)]
                else:
                    best_signal = I
                    # figure out if you should take the min or the max value of the fit to say where pi_len should be
                    if last_three_avg_I > first_three_avg_I:
                        pi_len = lens[np.argmax(best_signal)]
                    else:
                        pi_len = lens[np.argmin(best_signal)]
                tot_amp = [np.sqrt((ifit) ** 2 + (qfit) ** 2) for ifit, qfit in zip(I, Q)]
                depth = abs(tot_amp[np.argmin(tot_amp)] - tot_amp[np.argmax(tot_amp)])
            else:
                print('Invalid signal passed, please do I Q or None')
            return best_signal, pi_len

        else:
            q1_a_guess_I = (np.max(I) - np.min(I)) / 2
            q1_d_guess_I = np.mean(I)
            q1_a_guess_Q = (np.max(Q) - np.min(Q)) / 2
            q1_d_guess_Q = np.mean(Q)
            q1_b_guess = 1 / lens[-1]
            q1_c_guess = 0

            q1_guess_I = [q1_a_guess_I, q1_b_guess, q1_c_guess, q1_d_guess_I]
            q1_popt_I, q1_pcov_I = curve_fit(self.cosine, lens, I, maxfev=100000, p0=q1_guess_I)
            q1_fit_cosine_I = self.cosine(lens, *q1_popt_I)

            q1_guess_Q = [q1_a_guess_Q, q1_b_guess, q1_c_guess, q1_d_guess_Q]
            q1_popt_Q, q1_pcov_Q = curve_fit(self.cosine, lens, Q, maxfev=100000, p0=q1_guess_Q)
            q1_fit_cosine_Q = self.cosine(lens, *q1_popt_Q)

            first_three_avg_I = np.mean(q1_fit_cosine_I[:3])
            last_three_avg_I = np.mean(q1_fit_cosine_I[-3:])
            first_three_avg_Q = np.mean(q1_fit_cosine_Q[:3])
            last_three_avg_Q = np.mean(q1_fit_cosine_Q[-3:])

            best_signal_fit = None
            pi_len = None
            if 'Q' in self.signal:
                best_signal_fit = q1_fit_cosine_Q
                # figure out if you should take the min or the max value of the fit to say where pi_len should be
                if last_three_avg_Q > first_three_avg_Q:
                    pi_len = lens[np.argmax(best_signal_fit)]
                else:
                    pi_len = lens[np.argmin(best_signal_fit)]
            if 'I' in self.signal:
                best_signal_fit = q1_fit_cosine_I
                # figure out if you should take the min or the max value of the fit to say where pi_len should be
                if last_three_avg_I > first_three_avg_I:
                    pi_len = lens[np.argmax(best_signal_fit)]
                else:
                    pi_len = lens[np.argmin(best_signal_fit)]
            if 'None' in self.signal:
                # choose the best signal depending on which has a larger magnitude
                if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
                    best_signal_fit = q1_fit_cosine_Q
                    # figure out if you should take the min or the max value of the fit to say where pi_len should be
                    if last_three_avg_Q > first_three_avg_Q:
                        pi_len = lens[np.argmax(best_signal_fit)]
                    else:
                        pi_len = lens[np.argmin(best_signal_fit)]
                else:
                    best_signal_fit = q1_fit_cosine_I
                    # figure out if you should take the min or the max value of the fit to say where pi_len should be
                    if last_three_avg_I > first_three_avg_I:
                        pi_len = lens[np.argmax(best_signal_fit)]
                    else:
                        pi_len = lens[np.argmin(best_signal_fit)]
                tot_amp = [np.sqrt((ifit)**2 + (qfit)**2) for ifit,qfit in zip(q1_fit_cosine_I, q1_fit_cosine_Q)]
                depth = abs(tot_amp[np.argmin(tot_amp)] - tot_amp[np.argmax(tot_amp)])
            else:
                print('Invalid signal passed, please do I Q or None')
            if grab_depths:
                return best_signal_fit, pi_len, depth
            else:
                return best_signal_fit, pi_len

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)


class LengthRabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # Define a generator for the readout pulses with the lens, phases, and mixer/mux frequencies
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            # We have many qubits and many readout channels, so go through all of them and declare a readout for each
            # of them to tell the system how long the readout pulse is and qhat its freq and phase should be
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)
        # Configure the hardware to set this sort of pulse that we can trigger later
        # This has a rectangle pulse becuase style="const"
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )
        # Tell the system via another generator how to set up the qubit drive pulse
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])
        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )
        self.add_loop("lenloop", cfg["steps"])

    def _body(self, cfg):
        #self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse

        self.delay_auto(t=0, tag='waiting')

        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class QZE_gaus_pulse_RabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_qze'],
                         mux_gains=cfg['res_gain_qze'],  # has 7 values not just 6, extra one for the zeno
                         mux_phases=cfg['res_phase_qze'],
                         mixer_freq=cfg['mixer_freq'])
        #readout on each channel with the sampling frequency and length of readout (basically open the window in qick readout)
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)  # length=readout length at end


        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       mask=cfg["list_of_all_qubits"])

        # projection pulse (the short pulse used for projective measurement,7 ns)
        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=cfg["zeno_pulse_width"],
                       mask=cfg["qze_mask"],
                       )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)

        #length of the qubit pulse (us)
        Tdrive = cfg[
            'qubit_length_ge']  #start in congfig is set to 0.1 so i always start at 0.1 and end at 0.7, so qubit should be in first excited state

        #now we have started pulse at 0.1us
        #each pulse is 9 ns long with a 2 ns gap between pulses
        #schendule pulses as long as the entire pulse fits within the qubit drive pulse time
        if Tdrive>0.11:
            t_pulse = 0.11 #start at 0 (qubits in first excited state because config starts at 0.1us qubit pulse)
            while t_pulse + cfg["zeno_pulse_width"] <= Tdrive: # as long as we are below the qubit drive pulse time for the next short res pulse
                self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=t_pulse) #schedule this pulse
                t_pulse += cfg["zeno_pulse_period"]  # 9 ns pulse + 2 ns wait = 11 ns per cycle

        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class QZERabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_qze'],
                         mux_gains=cfg['res_gain_qze'],  # has 7 values not just 6, extra one for the zeno
                         mux_phases=cfg['res_phase_qze'],
                         mixer_freq=cfg['mixer_freq'])
        #readout on each channel with the sampling frequency and length of readout (basically open the window in qick readout)
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)  # length=readout length at end


        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       mask=cfg["list_of_all_qubits"])

        # projection pulse (the short pulse used for projective measurement,7 ns)
        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=cfg["zeno_pulse_width"],
                       mask=cfg["qze_mask"],
                       )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)

        #length of the qubit pulse (us)
        Tdrive = cfg[
            'qubit_length_ge']  #start in congfig is set to 0.1 so i always start at 0.1 and end at 0.7, so qubit should be in first excited state

        #now we have started pulse at 0.1us
        #each pulse is 9 ns long with a 2 ns gap between pulses
        #schendule pulses as long as the entire pulse fits within the qubit drive pulse time
        if Tdrive>0.11:
            t_pulse = 0.11 #start at 0 (qubits in first excited state because config starts at 0.1us qubit pulse)
            while t_pulse + cfg["zeno_pulse_width"] <= Tdrive: # as long as we are below the qubit drive pulse time for the next short res pulse
                self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=t_pulse) #schedule this pulse
                t_pulse += cfg["zeno_pulse_period"]  # 9 ns pulse + 2 ns wait = 11 ns per cycle

        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class QZE_constant_pulse_RabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_qze'],
                         mux_gains=cfg['res_gain_qze'],  # has 7 values not just 6, extra one for the zeno
                         mux_phases=cfg['res_phase_qze'],
                         mixer_freq=cfg['mixer_freq'])
        #readout on each channel with the sampling frequency and length of readout (basically open the window in qick readout)
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)  # length=readout length at end


        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       mask=cfg["list_of_all_qubits"])

        # projection pulse
        if cfg['qubit_length_ge'] > 0.11:
            self.add_pulse(ch=res_ch, name="proj_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11+ 3,
                           mask=cfg["qze_mask"],
                           )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", #for before we hit pi pulse len
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        self.add_pulse(ch=qubit_ch, name="qubit_pulse_pi_len",  #after pi pulse len, this is the normal non zeno freq drive
                       style="const",
                       length=0.11,  # total_drive_length,
                       freq=cfg['qubit_freq_ge'],  # [0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        print('starked: ',cfg['qubit_freq_ge_starked'][0], ' qfreq: ', cfg['qubit_freq_ge'])

        if cfg['qubit_length_ge'] > 0.11:
            self.add_pulse(ch=qubit_ch, name="starked_qubit_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11,  # total_drive_length,
                           freq=cfg['qubit_freq_ge_starked'][0],  # [0] # only should be one value
                           phase=cfg['qubit_phase'],
                           gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        if cfg['qubit_length_ge'] <= 0.11:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  #just drive without zeno/stark
        else:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse_pi_len", t=0) #regular w01 pi pulse first, will last 0.11us
            self.delay_auto(t=0, tag='waiting_pi') #wait to finish
            self.pulse(ch=cfg["qubit_ch"], name="starked_qubit_pulse", t=3) #play starked freq qubit drive for rest of qubit pulse len,, start 3 us after ring up
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0) #play zeno/stark tone in resonator
        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class QZE_constant_pulse_gnd_RabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_qze'],
                         mux_gains=cfg['res_gain_qze'],  # has 7 values not just 6, extra one for the zeno
                         mux_phases=cfg['res_phase_qze'],
                         mixer_freq=cfg['mixer_freq'])
        #readout on each channel with the sampling frequency and length of readout (basically open the window in qick readout)
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)  # length=readout length at end


        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       mask=cfg["list_of_all_qubits"])

        # projection pulse
        if cfg['qubit_length_ge'] > 0.11*3:
            self.add_pulse(ch=res_ch, name="proj_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11*2+ 3,
                           mask=cfg["qze_mask"],
                           )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", #for before we hit pi pulse len
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        self.add_pulse(ch=qubit_ch, name="qubit_pulse_pi_len",  #after pi pulse len, this is the normal non zeno freq drive
                       style="const",
                       length=0.11*3,  # total_drive_length,
                       freq=cfg['qubit_freq_ge'],  # [0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        print('starked: ',cfg['qubit_freq_ge_starked'][0], ' qfreq: ', cfg['qubit_freq_ge'])

        if cfg['qubit_length_ge'] > 0.11*3+0.01:
            self.add_pulse(ch=qubit_ch, name="starked_qubit_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11*3,  # total_drive_length,
                           freq=cfg['qubit_freq_ge_starked'][0],  # [0] # only should be one value
                           phase=cfg['qubit_phase'],
                           gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        if cfg['qubit_length_ge'] <= 0.11*3+0.01:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  #just drive without zeno/stark
        else:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse_pi_len", t=0) #regular w01 pi pulse first, will last 0.11us
            self.delay_auto(t=0, tag='waiting_pi') #wait to finish
            self.pulse(ch=cfg["qubit_ch"], name="starked_qubit_pulse", t=3) #play starked freq qubit drive for rest of qubit pulse len,, start 3 us after ring up
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0) #play zeno/stark tone in resonator
        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])


class QZE_constant_pulse_3pulse_RabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_qze'],
                         mux_gains=cfg['res_gain_qze'],  # has 7 values not just 6, extra one for the zeno
                         mux_phases=cfg['res_phase_qze'],
                         mixer_freq=cfg['mixer_freq'])
        #readout on each channel with the sampling frequency and length of readout (basically open the window in qick readout)
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)  # length=readout length at end


        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       mask=cfg["list_of_all_qubits"])

        # projection pulse
        if cfg['qubit_length_ge'] > 0.11*3:
            self.add_pulse(ch=res_ch, name="proj_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11*2+ 3,
                           mask=cfg["qze_mask"],
                           )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", #for before we hit pi pulse len
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        self.add_pulse(ch=qubit_ch, name="qubit_pulse_pi_len",  #after pi pulse len, this is the normal non zeno freq drive
                       style="const",
                       length=0.11*3,  # total_drive_length,
                       freq=cfg['qubit_freq_ge'],  # [0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        print('starked: ',cfg['qubit_freq_ge_starked'][0], ' qfreq: ', cfg['qubit_freq_ge'])

        if cfg['qubit_length_ge'] > 0.11*3+0.01:
            self.add_pulse(ch=qubit_ch, name="starked_qubit_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11*3,  # total_drive_length,
                           freq=cfg['qubit_freq_ge_starked'][0],  # [0] # only should be one value
                           phase=cfg['qubit_phase'],
                           gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        if cfg['qubit_length_ge'] <= 0.11*3+0.01:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  #just drive without zeno/stark
        elif 1.11 > cfg['qubit_length_ge'] > 0.11*3+0.01:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse_pi_len", t=0) #regular w01 pi pulse first, will last 0.11us
            self.delay_auto(t=0, tag='waiting_pi') #wait to finish
            self.pulse(ch=cfg["qubit_ch"], name="starked_qubit_pulse", t=3) #play starked freq qubit drive for rest of qubit pulse len,, start 3 us after ring up
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0) #play zeno/stark tone in resonator
        elif 2.2 > cfg['qubit_length_ge'] >= 0.11*3+0.01 + 1.11:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  #just drive without zeno/stark
        elif 3.19 > cfg['qubit_length_ge'] >= 0.11*3+0.01 + 2.2: #odd so should be in gnd here
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse_pi_len",
                       t=0)  # regular w01 pi pulse first, will last 0.11us
            self.delay_auto(t=0, tag='waiting_pi')  # wait to finish
            self.pulse(ch=cfg["qubit_ch"], name="starked_qubit_pulse",
                       t=3)  # play starked freq qubit drive for rest of qubit pulse len,, start 3 us after ring up
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0)  # play zeno/stark tone in resonator
        elif 3.96 > cfg['qubit_length_ge'] >= 0.11*3+0.01 + 3.19:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  #just drive without zeno/stark
        elif 5.06 > cfg['qubit_length_ge'] >= 0.11*3+0.01 + 3.96:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse_pi_len",
                       t=0)  # regular w01 pi pulse first, will last 0.11us
            self.delay_auto(t=0, tag='waiting_pi')  # wait to finish
            self.pulse(ch=cfg["qubit_ch"], name="starked_qubit_pulse",
                       t=3)  # play starked freq qubit drive for rest of qubit pulse len,, start 3 us after ring up
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0)  # play zeno/stark tone in resonator
        else:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)

        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])


class QZE_constant_pulse_RabiProgram_unstarked_freq(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_qze'],
                         mux_gains=cfg['res_gain_qze'],  # has 7 values not just 6, extra one for the zeno
                         mux_phases=cfg['res_phase_qze'],
                         mixer_freq=cfg['mixer_freq'])
        #readout on each channel with the sampling frequency and length of readout (basically open the window in qick readout)
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)  # length=readout length at end


        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       mask=cfg["list_of_all_qubits"])

        # projection pulse
        if cfg['qubit_length_ge'] > 0.11:
            self.add_pulse(ch=res_ch, name="proj_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11,
                           mask=cfg["qze_mask"],
                           )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here


        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  # just drive without zeno/stark
        if cfg['qubit_length_ge'] > 0.11:
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0.11) #play zeno/stark tone in resonator
        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])

class QZE_constant_pulse_RabiProgram_WaitForResRingUp(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_qze'],
                         mux_gains=cfg['res_gain_qze'],  # has 7 values not just 6, extra one for the zeno
                         mux_phases=cfg['res_phase_qze'],
                         mixer_freq=cfg['mixer_freq'])
        #readout on each channel with the sampling frequency and length of readout (basically open the window in qick readout)
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)  # length=readout length at end


        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       mask=cfg["list_of_all_qubits"])

        # projection pulse
        if cfg['qubit_length_ge'] > 0.11:
            self.add_pulse(ch=res_ch, name="proj_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11 + 3, #plus res ring up time
                           mask=cfg["qze_mask"],
                           )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", #for before we hit pi pulse len
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        self.add_pulse(ch=qubit_ch, name="qubit_pulse_pi_len",  #after pi pulse len, this is the normal non zeno freq drive
                       style="const",
                       length=0.11,  # total_drive_length,
                       freq=cfg['qubit_freq_ge'],  # [0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here
        if cfg['qubit_length_ge'] > 0.11:
            self.add_pulse(ch=qubit_ch, name="starked_qubit_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11,  # total_drive_length,
                           freq=cfg['qubit_freq_ge_starked'][0],  # [0] # only should be one value
                           phase=cfg['qubit_phase'],
                           gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        if cfg['qubit_length_ge'] <= 0.11:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  #just drive without zeno/stark
        else:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse_pi_len", t=0) #regular w01 pi pulse first, will last 0.11us
            self.delay_auto(t=0, tag='waiting_pi') #wait to finish
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0) #play zeno/stark tone in resonator immediately, it is the qubit length plus the ring up time
            self.pulse(ch=cfg["qubit_ch"], name="starked_qubit_pulse", t=3)  # play starked freq qubit drive after resonator has rung up, wil hold until end of zeno drive

        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])


class QZERabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=cfg['res_freq_qze'],
                         mux_gains=cfg['res_gain_qze'],  # has 7 values not just 6, extra one for the zeno
                         mux_phases=cfg['res_phase_qze'],
                         mixer_freq=cfg['mixer_freq'])
        #readout on each channel with the sampling frequency and length of readout (basically open the window in qick readout)
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)  # length=readout length at end


        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       mask=cfg["list_of_all_qubits"])

        # projection pulse (the short pulse used for projective measurement,7 ns)
        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=cfg["zeno_pulse_width"],
                       mask=cfg["qze_mask"],
                       )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qubit_mixer_freq'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)

        #length of the qubit pulse (us)
        Tdrive = cfg[
            'qubit_length_ge']  #start in congfig is set to 0.1 so i always start at 0.1 and end at 0.7, so qubit should be in first excited state

        #now we have started pulse at 0.1us
        #each pulse is 9 ns long with a 2 ns gap between pulses
        #schendule pulses as long as the entire pulse fits within the qubit drive pulse time
        if Tdrive>0.11:
            t_pulse = 0.11 #start at 0 (qubits in first excited state because config starts at 0.1us qubit pulse)
            while t_pulse + cfg["zeno_pulse_width"] <= Tdrive: # as long as we are below the qubit drive pulse time for the next short res pulse
                self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=t_pulse) #schedule this pulse
                t_pulse += cfg["zeno_pulse_period"]  # 9 ns pulse + 2 ns wait = 11 ns per cycle

        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'])


class OscilliscopeQZEProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # generator for the readout pulses
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch[0],
                         mux_freqs=[f+10 for f in cfg['res_freq_qze']],
                         mux_gains=cfg['res_gain_qze'], #has 7 values not just 6, extra one for the zeno
                         mux_phases=cfg['res_phase_qze'],
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
                       mask=cfg["qze_mask"],
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
            mux_freqs=cfg['res_freq_qze'],
            mux_gains=[1,0,0,0,0,0,0], #need to ramp it up here to see a clean signal
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