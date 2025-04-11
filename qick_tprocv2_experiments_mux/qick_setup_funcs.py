import numpy as np
import json

from build_task import *
from build_state import *
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from system_config import QICK_experiment

# Create an instance of your configuration with fridge set to "QUIET"
outerFolder = os.path.join("/data/QICK_data/run6/6transmon/", str(datetime.date.today()))
experiment = QICK_experiment(outerFolder, DAC_attenuator1 = 5, DAC_attenuator2 = 10, ADC_attenuator = 10, fridge=FRIDGE)

# Build a dictionary with all the configuration entries.
# This includes the dictionaries already defined in qick_cfg, as well as other parameters.
configs = {
    'DATA_PATH': experiment.outerFolder,
    'config': experiment.hw_cfg,         # Hardware configuration dictionary
    'expt_cfg': expt_cfg,                    # Add your experiment-specific configuration if needed
    'readout_cfg': experiment.readout_cfg, # Readout configuration dictionary
    'qubit_cfg': experiment.qubit_cfg,     # Qubit configuration dictionary
    'QICK_parameters': {               # Other parameters from the QICK_experiment instance
        'DAC_attenuator1': experiment.DAC_attenuator1,
        'DAC_attenuator2': experiment.DAC_attenuator2,
        'ADC_attenuator': experiment.ADC_attenuator,
        'FSGEN_CH': experiment.FSGEN_CH,
        'MIXMUXGEN_CH': experiment.MIXMUXGEN_CH,
        'MUXRO_CH': experiment.MUXRO_CH,
    }
}

# Write the complete configuration to a JSON file
with open('system_config.json', 'w') as f:
    json.dump(configs, f, indent=4)

print("Configuration successfully written to system_config.json")

def initialize_configs(expt_name):
    '''
    initialize_configs initializes the experiment configurations based on if we are doing mux or not.

    expt_name: a string name for the experiment

    returns:
        config: the full configuration dictionary for this experiment
        expt_name: the name of the experiment (now with '_mux' if applicable)
    '''

    with open('system_config.json', 'r') as f:
        configs = json.load(f)

    system_config = configs['config']
    expt_cfg = configs['expt_cfg']

    QUBIT_INDEX = system_config['qubit_index']
    MUX = system_config['MUX']
    readout_cfg = system_config['readout_cfg']

    # add the parameters for the specific experiment we want to do
    exp_cfg = add_qubit_experiment(expt_cfg, expt_name, QUBIT_INDEX)

    if MUX == 'True':  # MUX experiments
        if expt_name == 'res_spec_ge' or expt_name == 'res_spec_ef':  # special cases since we sweep ro freq
            exp_cfg = expt_cfg['res_spec_mux']

        hw_cfg_cpy = add_qubit_channel(system_config, QUBIT_INDEX)
        qubit_cfg_cpy = add_qubit_cfg(system_config, QUBIT_INDEX)

        config = {**hw_cfg_cpy, **qubit_cfg_cpy, **readout_cfg, **exp_cfg}

        expt_name = expt_name + '_mux'
    else:  # non-MUX experiments
        q_config = all_qubit_state(system_config)
        config = {**q_config['Q' + str(QUBIT_INDEX)], **exp_cfg}

    return config, expt_name


def declare_gen_ch(expt_func, cfg, ch, usage='res', suffix=''):
    '''
    declare_gen_ch declares a generator channel properly depending on the configurations

    expt_func: is a variable for "self" (i.e. the experiment function) in the code
    cfg: The configuration file so that you can pull values for freqs and gains
    ch: The channel index that you want to declare
    usage: 'res' or 'qubit'
    pulse_style: type of pulse you want to send 'const', 'arb', 'flat_top'
    suffix: do the parameters have a suffix? ('', '_ge', or '_ef')

    returns:
        nothing...
    '''

    if ch in fast_gen_chs:  # fast gen channels
        expt_func.declare_gen(ch=ch, nqz=cfg['nqz_' + usage])
    elif ch in mux_gen_chs:  # MUX channel
        expt_func.declare_gen(ch=ch, nqz=cfg['nqz_' + usage],
                              ro_ch=cfg['mux_ro_chs'][0],
                              mux_freqs=cfg[usage + '_freq' + suffix],
                              mux_gains=cfg[usage + '_gain' + suffix],
                              mux_phases=cfg[usage + '_phase'],
                              mixer_freq=cfg['mixer_freq'])
    elif ch in inter_gen_chs:  # interpolated channels
        expt_func.declare_gen(ch=ch, nqz=cfg['nqz_' + usage],
                              ro_ch=cfg['ro_ch'],
                              mixer_freq=cfg['mixer_freq'])


def declare_pulse(expt_func, cfg, ch, usage='res', pulse_style='const', pulse_name='pulse', suffix=''):
    '''
    declare_pulse declares a pulse to be sent properly depending on the configurations

    expt_func: is a variable for "self" (i.e. the experiment function) in the code
    cfg: The configuration file so that you can pull values for freqs and gains
    ch: The channel index that you want to declare
    usage: 'res' or 'qubit'
    pulse_style: type of pulse you want to send 'const', 'arb', 'flat_top' ('gauss' makes a gaussian pulse for us)
    pulse_name: name of the pulse we will send ('ramsey1' and 'ramsey2' are special)
    suffix: do the parameters have a suffix? ('', '_ge', or '_ef')

    returns:
        nothing...
    '''
    if pulse_name == 'ramsey1':
        expt_func.add_gauss(ch=ch, name="ramp" + suffix, sigma=cfg['sigma' + suffix], length=cfg['sigma' + suffix] * 5,
                            even_length=True)
        expt_func.add_pulse(ch=ch, name=pulse_name, ro_ch=cfg['ro_ch'],
                            style="arb",
                            envelope="ramp" + suffix,
                            freq=cfg[usage + '_freq' + suffix],
                            phase=cfg[usage + '_phase'],
                            gain=cfg[usage + '_gain' + suffix] / 2,
                            )
    elif pulse_name == 'ramsey2':
        expt_func.add_gauss(ch=ch, name="ramp" + suffix, sigma=cfg['sigma' + suffix], length=cfg['sigma' + suffix] * 5,
                            even_length=True)
        expt_func.add_pulse(ch=ch, name=pulse_name, ro_ch=cfg['ro_ch'],
                            style="arb",
                            envelope="ramp" + suffix,
                            freq=cfg[usage + '_freq' + suffix],
                            phase=cfg[usage + '_phase'] + cfg['wait_time'] * 360 * cfg['ramsey_freq'],
                            gain=cfg[usage + '_gain' + suffix] / 2,
                            )
    else:
        if pulse_style == 'gauss':  # for a gaussian pulse
            if (ch in fast_gen_chs) or (ch in inter_gen_chs):  # fast gen channels or interpolated channels
                expt_func.add_gauss(ch=ch, name="ramp" + suffix, sigma=cfg['sigma' + suffix],
                                    length=cfg['sigma' + suffix] * 5, even_length=True)
                expt_func.add_pulse(ch=ch, name=pulse_name, ro_ch=cfg['ro_ch'],
                                    style="arb",
                                    envelope="ramp" + suffix,
                                    freq=cfg[usage + '_freq' + suffix],
                                    phase=cfg[usage + '_phase'],
                                    gain=cfg[usage + '_gain' + suffix],
                                    )
            elif ch in mux_gen_chs:  # MUX channel
                expt_func.add_gauss(ch=ch, name="ramp" + suffix, sigma=cfg['sigma' + suffix],
                                    length=cfg['sigma' + suffix] * 5, even_length=True)
                expt_func.add_pulse(ch=ch, name=pulse_name,
                                    style="arb",
                                    envelope="ramp" + suffix,
                                    mask=range(len(cfg[usage + '_freq' + suffix])),
                                    )
        else:  # for other styles of pulses
            if (ch in fast_gen_chs) or (ch in inter_gen_chs):  # fast gen channels or interpolated channels
                expt_func.add_pulse(ch=ch, name=pulse_name, ro_ch=cfg['ro_ch'],
                                    style=pulse_style,
                                    length=cfg[usage + '_length' + suffix],
                                    freq=cfg[usage + '_freq' + suffix],
                                    phase=cfg[usage + '_phase'],
                                    gain=cfg[usage + '_gain' + suffix],
                                    )
            elif ch in mux_gen_chs:  # MUX channel
                expt_func.add_pulse(ch=ch, name=pulse_name,
                                    style=pulse_style,
                                    length=cfg[usage + '_length' + suffix],
                                    mask=range(len(cfg[usage + '_freq' + suffix])),
                                    )


def initialize_ro_chs(expt_func, cfg, suffix=''):
    '''
    initialize_ro_chs initializes the readout signal generator and readout ADC channels properly depending on the configurations

    expt_func: is a variable for "self" (i.e. the experiment function) in the code
    cfg: The configuration file so that you can pull values for freqs and gains
    suffix: do the parameters have a suffix? ('', '_ge', or '_ef')

    returns:
        nothing...
    '''

    if MUX == 'True':  # for MUX readout
        ro_chs = cfg['mux_ro_chs']
        res_ch = cfg['mux_ch']

        # declare a MUX gen channel
        declare_gen_ch(expt_func, cfg, res_ch, usage='res', suffix=suffix)

        for ch, f, ph in zip(ro_chs, cfg['res_freq' + suffix], cfg['res_phase']):
            expt_func.declare_readout(ch=ch, length=cfg['ro_length'], freq=f, phase=ph, gen_ch=res_ch)

        # declare a pulse for the MUX channel
        declare_pulse(expt_func, cfg, res_ch, pulse_name='res_pulse', suffix=suffix)

    else:  # for non-MUX readout
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']

        # declare the proper readout channel using
        declare_gen_ch(expt_func, cfg, res_ch, usage='res', suffix=suffix)

        expt_func.declare_readout(ch=ro_ch, length=cfg['ro_length'])
        expt_func.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['res_freq' + suffix], gen_ch=res_ch)

        # declare a pulse for the non-MUX channel
        declare_pulse(expt_func, cfg, res_ch, pulse_name='res_pulse', suffix=suffix)


def readout(expt_func, cfg):
    '''
    readout function sends a readout pulse and triggers the adc depending on the configurations

    expt_func: is a variable for "self" (i.e. the experiment function) in the code
    cfg: The configuration file so that you can pull values for freqs and gains

    returns:
        nothing...

    Note: we may want to rewrite this to not depend on the 'MUX' setting but rather the readout channel
          that is input. For now this should work fine, but if firmware changes or other types of
          channels are used, then we want to adjust for this.
    '''

    if MUX == 'True':  # for MUX readout
        expt_func.pulse(ch=cfg['mux_ch'], name="res_pulse", t=0)
        expt_func.trigger(ros=cfg['mux_ro_chs'], pins=[0], t=cfg['trig_time'])

    else:  # for non-MUX readout
        # if non-MUX then send readout configurations - useful when freq sweeping
        expt_func.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        expt_func.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        expt_func.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


# NOT CURRENTLY USEFUL -> this makes a QickParam for every experiment.
# In the future we could change this to make a large coniguration file
# that contains all experiments with full configurations in different dictionaries
def SingleQubitAllExperimentParameter():
    '''
    SingleQubitAllExperimentParameter initializes all experiment configurations for a single qubit.

    returns:
        updated_cfgs: a dictionary of all experiment configurations for a single qubit
    '''
    # all experiment names possible
    var = ["tof", "res_spec_ge", "qubit_spec_ge", "time_rabi_ge", "power_rabi_ge", "qubit_temp",
           "Ramsey_ge", "SpinEcho_ge", "T1_ge", "Ramsey_ef", "res_spec_ef", "qubit_spec_ef", "power_rabi_ef",
           "Readout_Optimization_ge", "Readout_Optimization_gef"]
    updated_cfgs = {}
    for expt_name in var:
        updated_cfgs[expt_name] = add_single_qubit_experiment(expt_cfg, expt_name, QUBIT_INDEX)

    return updated_cfgs