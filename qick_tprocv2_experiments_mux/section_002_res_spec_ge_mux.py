import matplotlib.pyplot as plt
from qick.asm_v2 import AveragerProgramV2
from tqdm import tqdm
from build_state import *
from expt_config import *
import copy
import datetime
import logging
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


class SingleToneSpectroscopyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        res_ch = cfg['res_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_chs[0],
                         mux_freqs=cfg['res_freq_ge'],
                         mux_gains=cfg['res_gain_ge'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])
        
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(ch=ch, length=cfg['res_length'], freq=f, phase=ph, gen_ch=res_ch)


        self.add_pulse(ch=res_ch, name="mymux",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )

    def _body(self, cfg):
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'], ddr4=True)
        self.pulse(ch=cfg['res_ch'], name="mymux", t=0)

class ResonanceSpectroscopy:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, save_figs, experiment = None,
                 verbose = False, logger = None, qick_verbose=True, unmasking_resgain = False):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.expt_name = "res_spec"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.round_num = round_num
        self.save_figs = save_figs
        self.experiment = experiment

        self.exp_cfg = expt_cfg[self.expt_name]
        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if experiment is not None:
            self.q_config = all_qubit_state(experiment, self.number_of_qubits)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} Res Spec configuration: {self.config}')
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} Res Spec configuration: ', self.config)

    def run(self, plotIQ=False):
        fpts = self.exp_cfg["start"] + self.exp_cfg["step_size"] * np.arange(self.exp_cfg["steps"])
        fcenter = self.config['res_freq_ge']
        amps = np.zeros((len(fcenter), len(fpts)))
        Iarr = np.zeros((len(fcenter), len(fpts)))
        Qarr = np.zeros((len(fcenter), len(fpts)))
        #filtered_amps = np.zeros((len(fcenter), len(fpts)))

        for index, f in enumerate(tqdm(fpts)):
            self.config["res_freq_ge"] = fcenter + f
            prog = SingleToneSpectroscopyProgram(self.experiment.soccfg, reps=self.exp_cfg["reps"], final_delay=self.config["relax_delay"], cfg=self.config)
            iq_list = prog.acquire(self.experiment.soc, soft_avgs=self.exp_cfg["rounds"], progress=self.qick_verbose)
            #print(f'freq {f}, {iq_list[0]}')
            for i in range(len(self.config['res_freq_ge'])):
                #amps[i][index]= iq_list[i][:,0]
                Iarr[i][index] = iq_list[i][0, 0]
                Qarr[i][index] = iq_list[i][0, 1]
                amps[i][index] = np.abs(iq_list[i][:, 0] + 1j * iq_list[i][:, 1])
        amps = np.array(amps)
        Iarr = np.array(Iarr)
        Qarr = np.array(Qarr)
        filtered_amps = np.array(savgol_filter(amps, window_length=21, polyorder=3))
        res_freqs = self.plot_results(fpts, fcenter, Iarr, Qarr, amps, filtered_amps, plot_IQ = plotIQ) #return freqs from plotting loop so we can use to update experiment

        return res_freqs, fpts, fcenter, Iarr, Qarr, amps, self.config

    def plot_results(self, fpts, fcenter, Iarr, Qarr, amps, filtered_amps, plot_IQ = False, reloaded_config = None, fig_quality = 100):
        res_freqs = []
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
        })
        for i in range(self.number_of_qubits):
            freq_r = fpts[np.argmin(filtered_amps[i])] + fcenter[i]
            res_freqs.append(freq_r)

        if plot_IQ:
            plt.figure(figsize=(10, 10))
            for i in range(self.number_of_qubits):
                plt.subplot(2, 2, i + 1)
                plt.plot(Iarr[i], Qarr[i], '.')
                plt.xlabel("I")
                plt.ylabel("Q")

                if i == self.QubitIndex:
                    plt.title(f"Res {i + 1}, {res_freqs[i]:.3f} MHz", pad=10)
                else:
                    plt.title(f"Res {i + 1}", pad=10)

            if self.experiment is not None:
                plt.suptitle(f"MUXed resonator spectroscopy IQ circle {self.config['reps']}*{self.config['rounds']} avgs",
                             fontsize=24, y=0.95)
            else:
                plt.suptitle(
                    f"MUXed resonator spectroscopy IQ circle {reloaded_config['reps']}*{reloaded_config['rounds']} avgs",
                    fontsize=24, y=0.95)
            plt.tight_layout(pad=2.0)

            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_ge_plots")
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(outerFolder_expt,
                                         f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + '_IQcirc')
                plt.savefig(file_name + ".png", dpi=fig_quality)
                # plt.savefig(file_name + ".pdf", dpi=fig_quality)
            plt.close()

        plt.figure(figsize=(12, 8))
        for i in range(self.number_of_qubits):
            plt.subplot(2, 2, i + 1)
            # Plot raw and filtered data on the same plot
            plt.plot([f + fcenter[i] for f in fpts], amps[i], '-', linewidth=1.5, label = 'Raw')
            plt.plot([f + fcenter[i] for f in fpts], filtered_amps[i], '-', linewidth=1.5, alpha = 0.7, label = 'Smoothed')
            if i == self.QubitIndex:
                ##### Uncomment to debug, leave commented if res spec measurment plots needed
                # freqs = [f + fcenter[i] for f in fpts]
                # print('freqs: ', freqs)
                # print('amps: ', amps[i])
                # mean, fit, fwhm, error = self.fit_lorentzian(amps[i], freqs, freq_r[-1])
                # print('mean', mean)
                # print(fwhm)
                # print(fit)
                # plt.plot([f + fcenter[i] for f in fpts], fit, 'r--', label = 'Lor Fit')
                plt.axvline(res_freqs[i], linestyle='--', color='orange', linewidth=1.5)
                plt.title(f"Resonator {i + 1} {res_freqs[i]:.3f} MHz", pad=10) #, fwhm: {fwhm:.2f} MHz
            else:
                plt.title(f"Resonator {i + 1}", pad=10)
            #plt.legend()
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Amplitude (a.u.)")

            plt.ylim(plt.ylim()[0] - 0.05 * (plt.ylim()[1] - plt.ylim()[0]), plt.ylim()[1])

        if self.experiment is not None:
            plt.suptitle(f"MUXed resonator spectroscopy {self.config['reps']}*{self.config['rounds']} avgs", fontsize=24, y=0.95)
        else:
            plt.suptitle(f"MUXed resonator spectroscopy {reloaded_config ['reps']}*{reloaded_config ['rounds']} avgs",
                         fontsize=24, y=0.95)
        plt.tight_layout(pad=2.0)

        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_ge_plots")
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name)
            plt.savefig(file_name + ".png", dpi=fig_quality)
            #plt.savefig(file_name + ".pdf", dpi=fig_quality)
        plt.close()

        res_freqs = [round(x, 5) for x in res_freqs]
        return res_freqs

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def get_results(self, fpts, fcenter, amps):
        res_freqs = []

        for i in range(self.number_of_qubits):
            freq_r = fpts[np.argmin(amps[i])] + fcenter[i]
            res_freqs.append(freq_r)

        res_freqs = [round(x, 7) for x in res_freqs]
        return res_freqs

    def lorentzian(self, f, f0, gamma, A, B):
        return A * gamma ** 2 / ((f-f0) ** 2 + gamma **2) + B

    def max_offset_difference_with_x(self, x_values, y_values, offset):
        # Taken from section 004 qubit spec ge
        max_average_difference = -1
        corresponding_x = None

        # averaging all 3 to avoid noise spikes
        for i in range(len(y_values) - 2):
            # group 3 vals
            y_triplet = y_values[i:i + 3]

            # avg differences for these 3
            average_difference = sum(abs(y - offset) for y in y_triplet) / 3

            # see if this is highest
            if average_difference > max_average_difference:
                max_average_difference = average_difference
                # x value of middle y value in the 3 vals
                corresponding_x = x_values[i + 1]
        return corresponding_x, max_average_difference

    def fit_lorentzian(self, amps, freqs, freq_r, sigma_guess=1):
        #Adapted from qubit spec ge fitting function
        print('freqs: ', freqs)
        print('amps: ', amps)
        try:
            initial_guess = [freq_r, sigma_guess, np.max(amps), np.min(amps)]

            # First round fits to get rough estimates
            params1, cov1 = curve_fit(self.lorentzian, freqs, amps, p0=initial_guess)
            print("DEBUG params1: ", params1, "type: ", type(params1))

            #Refine guess
            x_max_diff, max_diff = self.max_offset_difference_with_x(freqs, amps, params1[3])
            initial_guess = [x_max_diff, sigma_guess, np.max(amps), np.min(amps)]

            # Second round of fits, getting covariance matrices
            params, cov = curve_fit(self.lorentzian, freqs, amps, p0=initial_guess)

            # Create the fitted curves
            amp_fit = self.lorentzian(freqs, *params)

            # Calculate errrors from the covariance matrices
            fit_err = np.sqrt(np.diag(cov))

            # Extract fitted means and FWHM (assuming params[0] is mean and params[1] relates to the width
            mean = params[0]
            fwhm = 2 * params[1]

            # Return all desired results including error
            return mean, amp_fit, fwhm, fit_err

        except Exception as e:
            if self.verbose: print("Error during res Lorentzian fit:", e)
            self.logger.info(f'Error during Lorentzian fit: {e}')
            return None, None, None, None

class PostProcessResonanceSpectroscopy:
    def __init__(self, QubitIndex,  outerFolder, round_num, save_figs, experiment = None):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "res_spec"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.round_num = round_num
        self.save_figs = save_figs
        self.experiment = experiment




