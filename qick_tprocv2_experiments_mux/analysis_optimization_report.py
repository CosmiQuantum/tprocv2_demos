from analysis_helper import *

def optimization_report_ge(substudy_dir, dataset, QubitIndex):

    plt.rcParams['font.size'] = 10
    fig_opt, ax_opt = plt.subplots(2, 4, layout="constrained", figsize=[14,5])
    fig_opt.suptitle(f"g-e optimization for qubit {QubitIndex+1} dataset: {dataset}", fontsize=14)
    row = 0
    col = 0
    data_path = os.path.join(substudy_dir, dataset, "optimization/Data_h5/")

    ### res spec ###
    round_num = 0
    rspec_dates, rspec_freqs, rspec_probe_freqs, rspec_amp0, rspec_freq0, rspec_fit0, num_rounds = get_rspec_data(data_path, QubitIndex, round_num=round_num)

    # plot example res spec at a given round number
    plot = ax_opt[row][col]
    idx = np.arange(0, len(rspec_probe_freqs)) + QubitIndex * 100
    amps = np.array(rspec_amp0)[idx]
    adj_rspec_probe_freqs = np.array(rspec_probe_freqs) + rspec_freq0[QubitIndex]

    plot.plot(adj_rspec_probe_freqs, amps, label='data')
    plot.plot([rspec_fit0[QubitIndex], rspec_fit0[QubitIndex]], [np.min(amps), np.max(amps)], 'r:', label='center freq')
    plot.set_xlabel('resonator probe frequency [MHz]',fontsize=10)
    plot.set_ylabel('I,Q magnitude [a.u.]',fontsize=10)
    plot.set_title(f'res_ge {np.round(rspec_fit0[QubitIndex],2)} MHz, round {round_num + 1} of {num_rounds}',fontsize=10)
    plot.legend(fontsize=10)

    rspec_times = []
    for date in rspec_dates:
        rspec_times.append((date - rspec_dates[0]).total_seconds())

    # plot timestream of res spec rounds
    col = col+1
    plot = ax_opt[row][col]
    plot.scatter(rspec_times, rspec_freqs)
    plot.set_ylabel('res_ge frequency [MHz]',fontsize=10)
    plot.set_xlabel('time [s]',fontsize=10)

    col = col + 1

    ############################### QSpec #######################
    max_signal, I_or_Q, qspec_probe_freqs, qspec_freq, qspec_fwhm, qspec_fit, qspec_fit_err = get_opt_qspec_data(
        data_path, QubitIndex)

    plot = ax_opt[row, col]
    plot.plot(qspec_probe_freqs, I_or_Q)
    plot.plot(qspec_probe_freqs, qspec_fit, label='Lorentzian')
    plot.plot([qspec_freq, qspec_freq], [np.min(I_or_Q), np.max(I_or_Q)], 'r:')
    plot.legend(loc='best',fontsize=10)
    plot.set_xlabel('qubit probe frequency [MHz]',fontsize=10)
    plot.set_ylabel(max_signal)
    plot.set_title(f'qspec_ge: {np.round(qspec_freq, 2)} +/- {np.round(qspec_fit_err, 2)} MHz, fwhm: {np.round(qspec_fwhm, 2)} MHz',fontsize=10)
    col = col + 1

    
   # qspec_probe_freqs, mag, qspec_date = get_ext_qspec_data(data_path, QubitIndex)
   # plot = ax_opt[row, col]
   # plot.plot(qspec_probe_freqs, mag)
   # plot.set_xlabel('qubit probe frequency [MHz]',fontsize=10)
   # plot.set_ylabel('I,Q magnitude [a.u.]',fontsize=10)
   # plot.set_title('extended_qspec_ge',fontsize=10)

    ################ Rabi ##########################
    col = 0
    row = 1

    max_signal, I_or_Q, rabi_gains, pi_amp, rabi_fit = get_opt_rabi_data(data_path, QubitIndex)

    plot = ax_opt[row, col]
    plot.plot(rabi_gains, I_or_Q, label='data')
    plot.plot(rabi_gains, rabi_fit, label='cosine')
    plot.legend(fontsize=10)
    plot.set_xlabel('gain [a.u.]',fontsize=10)
    plot.set_ylabel(max_signal,fontsize=10)
    plot.set_title(f'rabi_ge: {np.round(pi_amp, 4)} a.u. pi amp', fontsize=10)
    col = col + 1

    ######################## offset res freq data ############
    res_freq_steps, mean_fids = get_opt_offset_data(data_path, QubitIndex)

    plot = ax_opt[row, col]
    plot.plot(res_freq_steps, np.array(mean_fids) * 100,marker='o',linestyle='-')
    plot.set_xlabel('resonator frequency offset [MHz]',fontsize=10)
    plot.set_ylabel('avg fidelity [%]',fontsize=10)
    col = col+1

    ######################## ssf data #######################
    ss_dates, ss_fid, ss_angle, I_g, Q_g, I_e, Q_e, num_rounds = get_opt_ssf_data(data_path, QubitIndex)
    hist_ssf([I_g, Q_g, I_e, Q_e], ax_opt, row, col, numbins=100)

    ###############save data#######################
    plt.show(block=False)
    png_name = f'{dataset}_Q{QubitIndex+1}_optimization_report_ge.png'
    png_path = os.path.join(substudy_dir, dataset, 'documentation', png_name)
    fig_opt.savefig(png_path)

    plt.close(fig_opt)

def optimization_report_ef(substudy_dir, dataset, QubitIndex):

    plt.rcParams['font.size'] = 10
    fig_opt, ax_opt = plt.subplots(1, 4, layout="constrained", figsize=[13, 2.5])
    fig_opt.suptitle(f"e-f optimization for qubit {QubitIndex + 1} dataset: {dataset}", fontsize=14)
    data_path = os.path.join(substudy_dir, dataset, "optimization/Data_h5/")

    ### e-f res spec ###

    rspec_dates, rspec_freqs, rspec_probe_freqs, rspec_amp0, rspec_freq0, rspec_fit0, num_rounds = get_rspec_data(
        data_path, QubitIndex, expt_name='res_ef')

    # plot example res spec at a given round number
    plot = ax_opt[0]
    idx = np.arange(0, len(rspec_probe_freqs)) + QubitIndex * 100
    amps = np.array(rspec_amp0)[idx]
    adj_rspec_probe_freqs = np.array(rspec_probe_freqs) + rspec_freq0[QubitIndex]

    plot.plot(adj_rspec_probe_freqs, amps, label='data')
    plot.plot([rspec_fit0[QubitIndex], rspec_fit0[QubitIndex]], [np.min(amps), np.max(amps)], 'r:', label='center freq')
    plot.set_xlabel('resonator probe frequency [MHz]',fontsize=10)
    plot.set_ylabel('I,Q magnitude [a.u.]',fontsize=10)
    plot.set_title(f'res_ef {np.round(rspec_fit0[QubitIndex],2)} MHz, round {num_rounds} of {num_rounds}', fontsize=10)
    plot.legend(fontsize=10)


    ############################### e-f QSpec #######################
    max_signal, I_or_Q, qspec_probe_freqs, qspec_freq, qspec_fwhm, qspec_fit, qspec_fit_err = get_opt_qspec_data(
        data_path, QubitIndex, expt_name='qspec_ef')

    plot = ax_opt[1]
    plot.plot(qspec_probe_freqs, I_or_Q)
    plot.plot(qspec_probe_freqs, qspec_fit, label='Lorentzian')
    plot.plot([qspec_freq, qspec_freq], [np.min(I_or_Q), np.max(I_or_Q)], 'r:')
    plot.legend(loc='best',fontsize=10)
    plot.set_xlabel('qubit probe frequency [MHz]',fontsize=10)
    plot.set_ylabel(max_signal,fontsize=10)
    plot.set_title(
        f'qspec_ef: {np.round(qspec_freq, 2)} +/- {np.round(qspec_fit_err, 2)} MHz, fwhm: {np.round(qspec_fwhm, 2)} MHz',
        fontsize=10)

    ################ rabi ##########################

    gains1, mag1, best_signal_fit1, pi_amp1, gains2, mag2, best_signal_fit2, pi_amp2 = get_ef_rabi_data(data_path, QubitIndex)

    plot = ax_opt[2]
    plot.plot(gains1, mag1, label='data')
    plot.plot(gains1, best_signal_fit1, label='cosine')
    plot.legend(fontsize=10)
    plot.set_xlabel('gain [a.u.]',fontsize=10)
    plot.set_ylabel('I,Q magnitude',fontsize=10)
    plot.set_title(f'rabi_ef 1: {np.round(pi_amp1, 4)} a.u. pi amp', fontsize=10)

    plot = ax_opt[3]
    plot.plot(gains2, mag2, label='data')
    plot.plot(gains2, best_signal_fit2, label='cosine')
    plot.legend(fontsize=10)
    plot.set_xlabel('gain [a.u.]',fontsize=10)
    plot.set_ylabel('I,Q magnitude',fontsize=10)
    plot.set_title(f'rabi_ef 2: {np.round(pi_amp2, 4)} a.u. pi amp', fontsize=10)

    # save data
    plt.show(block=False)
    png_name = f'{dataset}_Q{QubitIndex + 1}_optimization_report_ef.png'
    png_path = os.path.join(substudy_dir, dataset, 'documentation', png_name)
    fig_opt.savefig(png_path, dpi=200)

    plt.close(fig_opt)

optimization_report_ge("/data/QICK_data/run6/6transmon/TLS_Comprehensive_Study/test_0/", "2025-04-14_23-03-39", 0)
