import sys
import os
from tprocv2_demos.qick_tprocv2_experiments_mux.section_003_punch_out_ge_mux import Repeat_Punchout
from tprocv2_demos.qick_tprocv2_experiments_mux.section_003_punch_out_ge_mux import PunchOut_MUX


sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/")) # for NEXUS
from system_config import QICK_experiment
#from section_003_punch_out_ge_mux import PunchOut
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time


def save_res_data(Q, round_num, timestamp, data, outerFolder_data):
    save_folder = os.path.join(outerFolder_data, f"Res{Q + 1}_data")
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(save_folder, f"R{round_num}_Res{Q + 1}_data_{timestamp}.npz")
    np.savez(file_path,
             round_number=round_num,
             timestamp=timestamp,
             fpts=data["fpts"],
             fcenter=data["fcenter"],
             power_sweep=data["power_sweep"],
             amps=data["amps"],
             amps_smoothed=data["amps_smoothed"],
             res_raw=data["resonance_raw"],
             res_smooth=data["resonance_smooth"])
    return

def plot_round(round_num, round_data, round_timestamp, outerFolder, save = True):
    plt.figure(figsize=(12,10))

    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 14,  # Base font size
        'axes.titlesize': 18,  # Title font size
        'axes.labelsize': 16,  # Axis label font size
        'xtick.labelsize': 14,  # X-axis tick label size
        'ytick.labelsize': 14,  # Y-axis tick label size
        'legend.fontsize': 14,  # Legend font size
    })

    for idx, Q in enumerate(sorted(round_data.keys())):
        data = round_data[Q]

        plt.subplot(2, 2, idx + 1)

        f = data["fpts"] + data["fcenter"]
        power_sweep = data["power_sweep"]
        amps = data["amps"]

        for power_index in range(len(power_sweep)):
            offset = (max(amps[0]) - min(amps[0])) / 2
            plt.plot(f, amps[power_index] + offset * power_index,
                     '-', linewidth=1.5, label=round(power_sweep[power_index], 3))
        plt.title(f"Res {Q+1}")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude (a.u.)")
        plt.legend(loc='upper left', fontsize='6', title='Gain')

    plt.suptitle(f"Punchout - Round {round_num}", fontsize=20, y=0.95)
    plt.tight_layout(pad=2.0)
    if save:
        file_path = os.path.join(outerFolder, f"round{round_num}_allres_{round_timestamp}.png")
        plt.savefig(file_path, dpi = 300)
        plt.close()
    else:
        plt.show()
    return

def centerplot_round(round_num, round_data, round_timestamp, outerFolder, save = True):
    plt.figure(figsize=(12,10))

    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 14,  # Base font size
        'axes.titlesize': 18,  # Title font size
        'axes.labelsize': 16,  # Axis label font size
        'xtick.labelsize': 14,  # X-axis tick label size
        'ytick.labelsize': 14,  # Y-axis tick label size
        'legend.fontsize': 14,  # Legend font size
    })

    for idx, Q in enumerate(sorted(round_data.keys())):
        data = round_data[Q]

        plt.subplot(2, 2, idx + 1)

        power = data["power_sweep"]
        res_raw = data["resonance_raw"]
        res_smooth = data["resonance_smooth"]
        plt.plot(power, res_raw, '-', linewidth=1.5, label="Raw")
        plt.plot(power, res_smooth, '-', linewidth=1.5, label="Smoothed")

        plt.xlabel("Probe Gain")
        plt.ylabel("Frequency (MHz)")
        plt.title(f"Res {Q+1}")
        plt.legend()
        plt.legend(loc='upper left', fontsize='6', title='Gain')

    plt.suptitle(f"Round {round_num} Frequency vs Probe Gain", fontsize=20, y=0.95)
    plt.tight_layout(pad=2.0)
    if save:
        file_path = os.path.join(outerFolder, f"round{round_num}_allres_centershift_{round_timestamp}.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
    else:
        plt.show()
    return

def sweep2d_round(round_num, round_data, round_timestamp, outerFolder, plot_smooth =False, save=True):
    plt.figure(figsize=(12, 10))

    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 14,  # Base font size
        'axes.titlesize': 18,  # Title font size
        'axes.labelsize': 16,  # Axis label font size
        'xtick.labelsize': 14,  # X-axis tick label size
        'ytick.labelsize': 14,  # Y-axis tick label size
        'legend.fontsize': 14,  # Legend font size
    })

    for idx, Q in enumerate(sorted(round_data.keys())):
        data = round_data[Q]

        plt.subplot(2, 2, idx + 1)

        fpts = data["fpts"]
        fcenter = data["fcenter"]
        power = data["power_sweep"]
        amps = data["amps"]

        mesh = plt.pcolormesh(fpts + fcenter, power, amps, shading="auto")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Gain")
        plt.title(f"Res {Q+1}")

        cbar = plt.colorbar(mesh)
        cbar.set_label("Amplitude (a.u.)")

    plt.suptitle(f"Round {round_num} Resonance vs Gain", fontsize=20, y=0.95)
    plt.tight_layout(pad=2.0)
    if save:
        file_path = os.path.join(outerFolder, f"round{round_num}_allres_2dsweep_{round_timestamp}.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
    else:
        plt.show()

    if plot_smooth:
        plt.figure(figsize=(12, 10))
        for idx, Q in enumerate(sorted(round_data.keys())):
            data = round_data[Q]

            plt.subplot(2, 2, idx + 1)

            fpts = data["fpts"]
            fcenter = data["fcenter"]
            power = data["power_sweep"]
            amps = data["amps_smoothed"]

            mesh = plt.pcolormesh(fpts + fcenter, power, amps, shading="auto")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Gain")
            plt.title(f"Res {Q + 1}")

            cbar = plt.colorbar(mesh)
            cbar.set_label("Amplitude (a.u.)")

        plt.suptitle(f"Round {round_num} Smoothed Resonance vs Gain", fontsize=20, y=0.95)
        plt.tight_layout(pad=2.0)
        if save:
            file_path = os.path.join(outerFolder, f"round{round_num}_allres_2dsweep_smoothed_{round_timestamp}.png")
            plt.savefig(file_path, dpi=300)
            plt.close()
        else:
            plt.show()
    return

number_of_qubits = 4
DAC_att_1=10
DAC_att_2=15
DAC_att=DAC_att_1+DAC_att_2
ADC_att=17

run = 'run37'
study = 'Initial Checkout' #'Initial Checkout' # 'Punchout Study'
substudy = 'Punchout_Repeated_AllQ' #'Punchout' #'Punchout_Repeated_Q4'
outerFolder = os.path.join(f"/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
outerFolder_plots = outerFolder + "/documentation/"
outerFolder_moreplots = outerFolder_plots + "/other_plots/"
os.makedirs(outerFolder_moreplots, exist_ok = True)
outerFolder_data = outerFolder + "/study_data/"

from expt_config import FRIDGE
experiment = QICK_experiment(outerFolder_plots, DAC_attenuator1 = DAC_att_1, DAC_attenuator2 = DAC_att_2, qubit_DAC_attenuator1 = 5 , qubit_DAC_attenuator2 = 4 ,ADC_attenuator = ADC_att, fridge=FRIDGE)
qubits_to_meas = [0, 1, 2, 3] #[0, 1, 2, 3]
Unmask = True #True is single, False is muxed

substudy_txt_notes = ('All Q after opt to look at noise') #('All Qs at optimal lengths from r1 optimization, 90 sec wait between scans. Lets try lower gains, short repeat' ) #Q1 5.75uss ro len (r2 res length optimal)')#('All Qs around 0.4-0.5 to get cutoff, 4us res len, TWPA on at -11.6dB, 7.807 GHz, all warm amps 6V')
file_path = os.path.join(outerFolder_plots, 'sub_study_notes.txt')
with open(file_path, "w", encoding="utf-8") as file:
    file.write(substudy_txt_notes)

res_len = [4.5, 4.75, 5.5, 5.75] #[6, 5, 6.25, 4.75] #[5.75, 5, 6.25, 4.75]

start_gain, stop_gain, num_points = 0.15, 0.5, 8 #0.15, 0.6, 10 #0.1, 0.8, 5

total_time = 90 #min
start_time = time.time()
mux = False

round_num = 0
while time.time() < (start_time + total_time*60):
    round_num += 1

    if mux:
        punch_out = PunchOut_MUX(round_num, qubits_to_meas, number_of_qubits, outerFolder, experiment, unmasking_resgain=True)
        punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points,
                      save_data = True, plot_Center_shift = True, plot_res_sweeps = True, plot_2d = True)
        time.sleep(10)
    else:
        round_data = {}
        round_timestamp = datetime.datetime.now()
        formatted_round_timestamp = round_timestamp.strftime("%Y-%m-%d_%H-%M-%S")

        for i, Q in enumerate(qubits_to_meas):
            timestamp = datetime.datetime.now()
            formatted_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

            experiment.readout_cfg['res_length'] = res_len[i]

            punch_out = Repeat_Punchout(Q, number_of_qubits, experiment, Unmask)
            data = punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points)
            round_data[Q] = data

            save_res_data(Q, round_num, formatted_timestamp, data, outerFolder_data)

            del punch_out
            #time.sleep(90)

        plot_round(round_num, round_data, formatted_round_timestamp, outerFolder_plots, save = True)
        #centerplot_round(round_num, round_data, round_timestamp, outerFolder_moreplots, save = True)
        #sweep2d_round(round_num, round_data, round_timestamp, outerFolder_moreplots, plot_smooth = True, save = True)

#del punch_out
### Used with old punchout class with all the plots
# while time.time() < (start_time + total_time*60):
#     for Q in qubits:
#/
#         punch_out = PunchOut(Q, number_of_qubits, outerfolder_plots, experiment, Unmask)
#         start_gain, stop_gain, num_points = 0.1, 0.8, 5
#         punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points, DAC_att, ADC_att,
#                       plot_Center_shift=True, plot_res_sweeps=True, plot_2d=True)
#         del punch_out
#         time.sleep(60)
# del experiment