from analysis_helper import *

#########################################################################################

QubitIndex = 4
#data_dir = "/data/QICK_data/run6/6transmon/TLS_Comprehensive_Study/test_0" #update based on file transfer location from Ryan
#dataset = "2025-04-14_23-03-39"
data_dir ="/data/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy1"
dataset = "2025-04-15_21-24-46"
set_idx = 0
analysis_flags = {"temp_ef": False, "data_block": True, "data_slice": False}

opt_dir = os.path.join(data_dir, dataset, "optimization/Data_h5")
study_dir = os.path.join(data_dir, dataset, "study_data/Data_h5")
save_dir = os.path.join(data_dir, dataset, f"documentation")
create_folder_if_not_exists(save_dir)

################################## Generate Data Block Report ###################################
if analysis_flags["data_block"]:

    fig1, ax1 = plt.subplots(3,1, layout="constrained")
    fig1.suptitle(f"qubit {QubitIndex + 1} dataset: {dataset}",fontsize=14)

    ################################### QSpec data ##################################################
    qspec_dates, qspec_freqs, qspec_freq_errs = get_qspec_data(study_dir, QubitIndex)

    plot = ax1[0]
    plot.errorbar(qspec_dates, qspec_freqs, yerr=qspec_freq_errs, fmt='o')
    plot.tick_params(axis='x', labelrotation=45)
    plot.set_ylabel('qspec_ge frequency [MHz]')

    ################################### T1 data ######################################################
    t1_dates, t1s, t1_errs = get_t1_data(study_dir, QubitIndex)

    plot = ax1[1]
    plot.errorbar(t1_dates, t1s, yerr=t1_errs, fmt='o')
    plot.tick_params(axis='x', labelrotation=45)
    plot.set_ylabel('T1_ge [us]')

    ############################## SSF data ##########################################################
    ss_dates, ss_fid, ss_angle = get_ssf_data(study_dir, QubitIndex)

    plot = ax1[2]
    plot.scatter(ss_dates, np.array(ss_fid) * 100)
    plot.tick_params(axis='x', labelrotation=45)
    plot.set_ylabel('SSF_ge [%]')

    plt.show()

    ############################## high gain qspec ##############################################
    fig2, ax2 = plt.subplots(3, 1)


################################## Generate Data Slice Report ###################################

if analysis_flags["data_slice"]:
    time_idx = 0
    fig4, ax4 = plt.subplots(2,3)
    fig4.suptitle(f"qubit {QubitIndex} dataset: {dataset} at round {time_idx}")
    row = 0
    col = 0

 ## --------- Qspec ------------
    signal = 'I'
    qspec_date, qspec_probe_freqs, I_or_Q = get_qspec_data_at_time_t(study_dir, QubitIndex, signal=signal)
    hg_qspec_date, hg_qspec_probe_freqs, hg_I_or_Q = get_qspec_data_at_time_t(study_dir, QubitIndex, signal=signal, expt_name='high_gain_qspec_ge')


    plot = ax4[row][col]
    plot.plot(qspec_probe_freqs, I_or_Q, label="gain=0.15 a.u.")
    plot.plot(hg_qspec_probe_freqs, hg_I_or_Q, label="gain=1.0 a.u.")
    plot.set_xlabel('qubit probe frequency [MHz]')
    plot.set_ylabel(f'{signal} [a.u.]')
    plot.legend()
    plot.set_title(f'Low gain Qspec: {qspec_date}. High gain Qspec: {hg_qspec_date}')

    col += 1

 ## --------- T1 ---------------
    delay_times, t1_date, q1_fit_exponential, T1_err, T1_est, plot_sig, I_or_Q = get_t1_data_at_time_t(study_dir, QubitIndex, time_idx = time_idx)

    plot = ax4[row][col]
    plot.plot(delay_times, I_or_Q, label="data")
    plot.plot(delay_times, q1_fit_exponential, label="exponential fit")
    plot.set_xlabel('delay time [us]')
    plot.set_ylabel(f'{plot_sig} [a.u.]')
    plot.set_title(f'T1: {np.round(T1_est,2)} +/- {np.round(T1_err,2)} us. {t1_date}')

    plt.show()
