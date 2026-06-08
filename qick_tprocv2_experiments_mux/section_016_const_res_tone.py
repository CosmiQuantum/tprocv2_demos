from qick.asm_v2 import AveragerProgramV2
import matplotlib.pyplot as plt
from build_state import *
from expt_config import *
from system_config import *

class ResToneSpectrumAnalyzer:
    def __init__(
        self,
        QubitIndex,
        outerFolder,
        experiment,
        round_num=1,
        save_figs=True,
        title=False,
        qick_verbose=True,
        unmasking_resgain=False,
            
        # Spectrum analyzer inputs
        res_pulse_mode="periodic",
        sa_hold_time=100.0,
        tof_freq_offset_MHz=1.0): # 1 in typical ToF
        
        # Keep same expt_name so other scripts do not need changes
        self.expt_name = "tof"
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.Qubit = 'Q' + str(QubitIndex)

        # Use copy so we do not modify global expt_cfg["tof"]
        self.exp_cfg = expt_cfg[self.expt_name].copy()

        self.experiment = experiment
        self.save_figs = save_figs
        self.title = title
        self.qick_verbose = qick_verbose

        # SA-specific inputs, kept outside config
        self.res_pulse_mode = res_pulse_mode
        self.sa_hold_time = sa_hold_time
        self.tof_freq_offset_MHz = tof_freq_offset_MHz

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

        self.q_config = all_qubit_state(self.experiment, 6)
        self.round_num = round_num

        if 'All' in self.Qubit:
            self.config = {**self.q_config['Q0'], **self.exp_cfg}
            print(f'Q {self.QubitIndex} Round {round_num} TOF SA configuration: ', self.config)
        else:
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            print(f'Q {self.QubitIndex + 1} Round {round_num} TOF SA configuration: ', self.config)

    def run(self):
        # Pull class inputs into local variables so the nested QICK program can use them
        res_pulse_mode = self.res_pulse_mode
        sa_hold_time = self.sa_hold_time
        tof_freq_offset_MHz = self.tof_freq_offset_MHz

        class ResToneSpectrumAnalyzerProgram(AveragerProgramV2):
            def _initialize(self, cfg):
                ro_chs = cfg['ro_ch']
                gen_ch = cfg['res_ch']

                # Original TOF used f + 1 MHz.
                # This is now controlled by the class input tof_freq_offset_MHz.
                mux_freqs = [f + tof_freq_offset_MHz for f in cfg['res_freq_ge']]

                self.declare_gen(
                    ch=gen_ch,
                    nqz=cfg['nqz_res'],
                    ro_ch=ro_chs[0],
                    mux_freqs=mux_freqs,
                    mux_gains=cfg['res_gain_ge'],
                    mux_phases=cfg['res_phase'],
                    mixer_freq=cfg['mixer_freq']
                )

                # Keep readout declarations for consistency with TOF,
                # even though we do not acquire ADC data in this SA version.
                for ch, f, ph in zip(cfg['ro_ch'], mux_freqs, cfg['ro_phase']):
                    self.declare_readout(
                        ch=ch,
                        length=cfg['res_length'],
                        freq=f,
                        phase=ph,
                        gen_ch=gen_ch
                    )

                self.add_pulse(
                    ch=gen_ch,
                    name="res_pulse",
                    style="const",
                    length=cfg["res_length"],
                    mask=cfg["list_of_all_qubits"],
                    mode=res_pulse_mode,
                )

                # Start the periodic resonator tone here.
                # This was the commented line your colleague mentioned.
                self.pulse(ch=gen_ch, name="res_pulse", t=0)

            def _body(self, cfg):
                # No ADC trigger and no pulse call here.
                # The periodic pulse was started in _initialize().
                self.delay(sa_hold_time)

        print("\nRunning TOF in spectrum-analyzer mode:")
        print(f"  Qubit: Q{self.QubitIndex + 1}")
        print(f"  res_ch: {self.config['res_ch']}")
        print(f"  res_freq_ge: {self.config['res_freq_ge']}")
        print(f"  res_gain_ge: {self.config['res_gain_ge']}")
        print(f"  res_length: {self.config['res_length']} us")
        print(f"  reps: {self.config['reps']}")
        print(f"  soft_avgs: {self.config['soft_avgs']}")
        print(f"  relax_delay: {self.config['relax_delay']} us")
        print(f"  res_pulse_mode: {self.res_pulse_mode}")
        print(f"  sa_hold_time: {self.sa_hold_time} us")
        print(f"  tof_freq_offset_MHz: {self.tof_freq_offset_MHz} MHz")

        prog = ResToneSpectrumAnalyzerProgram(
            self.experiment.soccfg,
            reps=self.config["reps"],
            final_delay=self.config["relax_delay"],
            cfg=self.config
        )

        # Run program so spectrum analyzer can measure RF output.
        # No ADC decimated data needed.
        prog.acquire(
            self.experiment.soc,
            soft_avgs=self.config["soft_avgs"],
            progress=self.qick_verbose
        )