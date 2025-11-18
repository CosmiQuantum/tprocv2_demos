import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from qick.asm_v2 import AveragerProgramV2
from tqdm import tqdm
from build_state import *
from expt_config import *
import datetime
import logging
from scipy.optimize import curve_fit

import jcresonators.resonator as jcresonator

class SingleToneSpectroscopyProgram(AveragerProgramV2):
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
                       mask=cfg["res_mask"],
                       )

        # dynamic readout
        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])
        self.add_readoutconfig(ch=ro_ch, name="ro",
                               freq=cfg['this_res_freq'][0], #entry res_freq
                               gen_ch=gen_ch,
                               outsel='product')
        self.send_readoutconfig(ch=ro_ch, name="ro", t=0)

    def _body(self, cfg):
        self.delay_auto()
        self.pulse(ch=cfg['gen_ch'], name="mux_pulse", t=0.0)
        self.trigger(ros=cfg['dynro_ch'], pins=[0], t=cfg['trig_time'],ddr4=True)

class SingleToneSpectroscopyFSGENoverlayProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['dynro_ch'][0]
        gen_ch = cfg['gen_ch']
        drive_ch = cfg['drive_ch']

        # play individual resonator pulse from MUX DAC
        self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch,
                         mux_freqs=cfg['this_res_freq'],
                         mux_gains=[0.5],#cfg['res_gain'],
                         mux_phases=cfg['res_phase'],
                         mixer_freq=cfg['mixer_freq'])

        self.add_pulse(ch=gen_ch, name="mux_pulse",
                       style="const",
                       length=cfg["pulse_length"],
                       mask=cfg["res_mask"],
                       )

        self.declare_gen(ch=drive_ch, nqz=cfg['nqz_res'])

        self.add_pulse(ch=drive_ch, name="fsgen_pulse",
                       style="const",
                       freq=cfg['this_res_freq'][0],
                       phase=cfg['res_phase'][0],
                       gain=cfg['res_gain'][0],
                       length=cfg["pulse_length"],
                       )

        # dynamic readout
        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])
        self.add_readoutconfig(ch=ro_ch, name="ro",
                               freq=cfg['this_res_freq'][0], #entry res_freq
                               gen_ch=gen_ch,
                               outsel='product')
        self.send_readoutconfig(ch=ro_ch, name="ro", t=0)

    def _body(self, cfg):
        self.delay_auto()
        self.pulse(ch=cfg['gen_ch'], name="mux_pulse", t=0.0)
        self.pulse(ch=cfg['drive_ch'], name="fsgen_pulse", t=0.0)
        self.trigger(ros=cfg['dynro_ch'], pins=[0], t=cfg['trig_time'],ddr4=True)

class SingleToneSpectroscopyFSGENProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['dynro_ch'][0]
        gen_ch = cfg['drive_ch']

        # play individual resonator pulse from FSGEN
        self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'])

        self.add_pulse(ch=gen_ch, name="fsgen_pulse",
                       style="const",
                       freq = cfg['this_res_freq'][0],
                       phase = cfg['res_phase'][0],
                       gain = cfg['res_gain'][0],
                       length=cfg["pulse_length"],
                       )

        # dynamic readout
        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])
        self.add_readoutconfig(ch=ro_ch, name="ro",
                               freq=cfg['this_res_freq'][0], #entry res_freq
                               gen_ch=gen_ch,
                               outsel='product')
        self.send_readoutconfig(ch=ro_ch, name="ro", t=0)

    def _body(self, cfg):
        self.delay_auto()
        self.pulse(ch=cfg['drive_ch'], name="fsgen_pulse", t=0.0)
        self.trigger(ros=cfg['dynro_ch'], pins=[0], t=cfg['trig_time'],ddr4=True)


class ResonanceSpectroscopy:
    def __init__(self, ResonatorIndex, number_of_resonators, studyDocumentationFolder, round_num, save_figs=False, experiment=None,
                 verbose=False, logger=None, dac='mux', gain=None):
        self.ResonatorIndex = ResonatorIndex
        self.Resonator = "R0" #+ str(self.ResonatorIndex)
        self.number_of_resonators = number_of_resonators
        self.studyDocumentationFolder = studyDocumentationFolder
        self.expt_name = "res_spec_jcrun7"
        self.round_num = round_num
        self.save_figs = save_figs
        self.experiment = experiment
        self.dac = dac
        self.exp_cfg = expt_cfg[self.expt_name]
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if experiment is not None:
            self.r_config = all_resonator_state(experiment, self.number_of_resonators)
            self.config = {**self.r_config[self.Resonator], **self.exp_cfg}
            self.logger.info(f'R {self.ResonatorIndex} Round {self.round_num} Res Spec configuration: {self.config}')
            if self.verbose: print(f'R {self.ResonatorIndex} Round {self.round_num} Res Spec configuration: ',
                                   self.config)
            if gain is not None:
                self.config['gain'] = gain

    def run(self):
        self.config["this_res_freq"] = [self.config["res_freq"][self.ResonatorIndex]]
        self.config["res_mask"] = [0]
        self.config["mixer_freq"] = self.config["this_res_freq"][0] + 300

        half_fpts = np.power(np.array(np.linspace(0, np.power(self.config['span']/2,1/3), num=int(np.floor(self.config['steps']/2)))),3)
        fpts = np.concatenate((-1*np.flip(half_fpts), half_fpts))
        #fpts = np.linspace(-self.config['span'] / 2, self.config['span'] / 2, num=self.config['steps'])
        freq_sweep = self.config['this_res_freq'] + fpts
        #gain_sweep = [self.config["gain"]]
        #gain_sweep = np.linspace(self.config["gain_start"], 1.0, num=self.config["gain_steps"])

        gain_sweep = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0] #hard coded in for now.

        I = np.zeros((len(gain_sweep),len(fpts)))
        Q = np.zeros((len(gain_sweep),len(fpts)))

        for j, g in enumerate(tqdm(gain_sweep)):
            self.config["res_gain"] = [np.round(g,3)]

            for i, f in enumerate(tqdm(freq_sweep)):
                # updates all multiplexed resonator frequency sweeps
                self.config["this_res_freq"]= [f]
                if self.dac == 'mux':
                    prog = SingleToneSpectroscopyProgram(self.experiment.soccfg, reps=self.config["reps"], final_delay=self.config["relax_delay"],
                                                 cfg=self.config)

                elif self.dac == 'fsgen':
                    prog = SingleToneSpectroscopyFSGENProgram(self.experiment.soccfg, reps=self.config["reps"], final_delay=self.config["relax_delay"],
                                                 cfg=self.config)

                iq_list = prog.acquire(self.experiment.soc, soft_avgs=self.config["rounds"],progress=False)
                I[j][i] = iq_list[0][:,0]
                Q[j][i] = iq_list[0][:,1]

        return freq_sweep, I, Q, gain_sweep, self.config

    def DCM_fit(self, freq_sweep, I, Q, gain_sweep):

        ##set up plot
        if self.save_figs:
            fig = plt.figure(figsize=(12,6))
            gs = GridSpec(2, 2, width_ratios=[1.75, 2], hspace=0.4, wspace=0.4)
            fig.suptitle(f"R{self.ResonatorIndex} Round {self.round_num}")
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1,0])
            ax3 = fig.add_subplot(gs[:, 1])

        phase_corrected = np.zeros_like(I)
        amp_corrected = np.zeros_like(I)
        phase_delay = np.zeros(len(gain_sweep))
        Ql = np.zeros(len(gain_sweep))
        Qi = np.zeros(len(gain_sweep))
        Qc = np.zeros(len(gain_sweep))
        fR = np.zeros(len(gain_sweep))

        for k, g in enumerate(tqdm(gain_sweep)):
            #get complex number I + jQ, extract amplitude and phase
            iq_complex = I[k] + 1j * Q[k]
            amp = np.abs(iq_complex)
            phase = np.unwrap(np.angle(iq_complex))/(2*np.pi)

            ## find phase delay and correct
            a = np.vstack([freq_sweep, np.ones_like(freq_sweep)]).T
            phase_delay[k] = np.linalg.lstsq(a, phase, rcond=None)[0][0] #in us
            iq_rotated = iq_complex * np.exp(-1j * freq_sweep * 2 * np.pi * phase_delay[k])
            phase_corrected[k] = np.unwrap(np.angle(iq_rotated)) #radians

            #convert magnitude to 20 log scale
            #amp_corrected[k] = 20 * np.log10(np.abs(iq_rotated))
            amp_corrected[k] = np.real(20*np.log10(iq_rotated/np.max(np.abs(iq_rotated))))

            data = np.transpose(np.array([freq_sweep, amp_corrected[k], phase_corrected[k]]))
            this_res = jcresonator.Resonator(data=data, preprocess_method='linear', normalize=10, fscale=1000000)
            this_res.fit_method(method='DCM', MC_iteration=100, MC_rounds=100)
            x, y, xfit, yfit, parameters = this_res.fit()
            Ql[k] = parameters[0]
            Qi[k] = parameters[1]
            Qc[k] = parameters[2]
            fR[k] = parameters[5]

            if self.save_figs:
                plt.rcParams.update({
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                })

                #ax1.plot(freq_sweep, amp_corrected[k], label=f"gain={np.round(g,3)}")
                ax1.scatter(x, 20 * np.log10(np.real(y)), label=f"gain={np.round(g, 3)}", s=15)
                ax1.plot(xfit, 20 * np.log10(np.real(yfit)), color='k', linestyle=':',label='DCM fit')
                ax1.legend()
                ax1.set_xlabel("Probe Frequency [MHz]")
                ax1.set_ylabel('normalized 20 log10(amp)')

                #ax2.plot(freq_sweep, phase_corrected[k], label=f"gain={np.round(g,3)}")
                ax2.scatter(x, np.imag(y), label=f"gain={np.round(g,3)}",s=15)
                ax2.plot(xfit, np.imag(yfit), color='k', linestyle=':', label='DCM fit')
                ax2.legend()
                ax2.set_xlabel("Probe Frequency [MHz]")
                ax2.set_ylabel('normalized Phase [rad]')

                ax3.scatter(np.real(y), np.imag(y), label=f"gain={np.round(g,3)}", s=15)
                ax3.plot(np.real(yfit), np.imag(yfit), color='k',linewidth=1, label="DCM fit")
                ax3.set_xlabel("20 log10(amp)")
                ax3.set_ylabel("phase")
                ax3.legend()

        if self.save_figs:
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(self.studyDocumentationFolder,f"R{self.ResonatorIndex}_{self.expt_name}_{formatted_datetime}")
            plt.savefig(f"{filename}.png")


        return fR, Ql, Qi, Qc

    def plot_raw(self, freq_sweep, I, Q):
            iq_complex = I[0] + 1j * Q[0]
            fig,ax = plt.subplots(2,1)
            fig.suptitle(f'R{self.ResonatorIndex} resonator spectroscopy')

            ax[0].plot(freq_sweep, I[0], color='b')
            ax[0].set_xlabel('Probe Frequency [MHz]')
            ax[0].set_ylabel('I [a.u.]')
            ax2 = ax[0].twinx()
            ax2.plot(freq_sweep, Q[0], color='r')
            ax2.set_ylabel('Q [a.u.]')

            ax[1].plot(freq_sweep, np.abs(iq_complex), color='b')
            ax[1].set_xlabel('Probe Frequency [MHz]')
            ax[1].set_ylabel('amplitude')

            if self.save_figs:
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                filename = os.path.join(self.studyDocumentationFolder,
                                        f"R{self.ResonatorIndex}_{self.expt_name}_{formatted_datetime}_raw")
                plt.savefig(f"{filename}.png")

    def plot_fsgen(self, freq_sweep, I, Q, plot=False):
        print(np.shape(I))
        amp = np.sqrt(np.square(I[0]) + np.square(Q[0]))
        print(np.shape(amp))
        freq_r = freq_sweep[np.argmin(amp)]
        amp_fit, fit_freq, fwhm, fit_err = self.fit_lorenzian(-1*amp, freq_sweep, freq_r, 0.1)

        if plot:
            fig = plt.figure()
            plt.scatter(freq_sweep, amp, color='k', label='data')
            plt.plot(freq_sweep, -1*amp_fit, color='b',label='Lorentzian fit')
            plt.plot([fit_freq, fit_freq], [np.min(amp)*0.9, np.max(amp)*1.1], 'r--',label='fit frequency')
            plt.xlabel('Probe Frequency [MHz]')
            plt.ylabel('Amplitude [a.u.]')
            plt.legend()
            plt.title(f'Drive Resonator R{self.ResonatorIndex} from FSGEN channel')

        return amp_fit, fit_freq, fwhm, fit_err

    def lorentzian(self, f, f0, gamma, A, B):

        return A * gamma ** 2 / ((f - f0) ** 2 + gamma ** 2) + B

    def max_offset_difference_with_x(self, x_values, y_values, offset):
        max_average_difference = -1
        corresponding_x = None

        # average all 3 to avoid noise spikes
        for i in range(len(y_values) - 2):
            # group 3 vals
            y_triplet = y_values[i:i + 3]

            # avg differences for these 3 vals
            average_difference = sum(abs(y - offset) for y in y_triplet) / 3

            # see if this is the highest difference yet
            if average_difference > max_average_difference:
                max_average_difference = average_difference
                # x value for the middle y value in the 3 vals
                corresponding_x = x_values[i + 1]

        return corresponding_x, max_average_difference

    def fit_lorenzian(self, amp, freqs, freq_q, sigma_guess = 0.1):
            initial_guess = [freq_q, sigma_guess, np.max(amp), np.min(amp)]

            # First round of fits (to get rough estimates)
            params, _ = curve_fit(self.lorentzian, freqs, amp, p0=initial_guess)

            # Use these fits to refine guesses
            x_max_diff, max_diff = self.max_offset_difference_with_x(freqs, amp, params[3])
            initial_guess = [x_max_diff, sigma_guess, np.max(amp), np.min(amp)]

            # Second (refined) round of fits, this time capturing the covariance matrices
            params, cov = curve_fit(self.lorentzian, freqs, amp, p0=initial_guess)

            # Create the fitted curves
            amp_fit = self.lorentzian(freqs, *params)


            # Calculate errors from the covariance matrices
            fit_err = np.sqrt(np.diag(cov))


            # Extract fitted means and FWHM (assuming params[0] is the mean and params[1] relates to the width)
            mean = params[0]
            fwhm = 2 * params[1]


            # Return all desired results including the error on the Q fit
            return amp_fit, mean, fwhm, fit_err



















