
from analysis_helper_TLS_comprehensive import *



############### set values here ###################
QubitIndex = 0
data_dir = "/Users/joycecs/PycharmProjects/PythonProject/.venv/QUIET/QUIET_data/RR_comprehensive_TLS/" #update based on file transfer location from Ryan
dataset = '2025-04-15_21-24-46'
analysis_flags = {"timestream": True, "slice": False}
theta = 20 * np.pi/180
threshold = -1100
selected_round = 0

res_stark_constant = [-17, 0, 0, 0, -25, 0] # from res stark 2D map
duffing_constant = [1, 1, 1, 1, 1, 1] # from stark 2D map at fixed detunings

if analysis_flags["timestream"]:

    fig, ax = plt.subplots(3,2, layout='constrained')
    fig.suptitle(f'dataset {dataset} qubit {QubitIndex + 1} timestream')

    ##### qspec data ####
    qspec_ge = qspec(data_dir, dataset, QubitIndex)
    qspec_dates, n, qspec_probe_freqs, I, Q = qspec_ge.load_all()
    #qspec_ge.get_qspec_freq_in_round(qspec_probe_freqs, I, Q, 0, n, plot=True)
    qspec_freqs, qspec_errs, qspec_fwhms = qspec_ge.get_all_qspec_freq(qspec_probe_freqs, I, Q, n)

    plot = ax[0][0]
    plot.errorbar(qspec_dates, qspec_freqs, qspec_errs, fmt='o')
    plot.tick_params(axis='x', labelrotation=45)
    plot.set_ylabel('qspec frequency [MHz]')

    ##### t1 data #####
    t1_ge = t1(data_dir, dataset, QubitIndex, theta, threshold)
    t1_dates, n, delay_times, steps, reps, I_shots, Q_shots = t1_ge.load_all()
    #t1_ge.plot_shots(I_shots, Q_shots, delay_times, n, idx=25)
    p_excited = t1_ge.process_shots(I_shots, Q_shots, n, steps)
    #t1_ge.get_t1_in_round(delay_times, p_excited, n, 0, plot = True)
    t1s, t1_errs = t1_ge.get_all_t1(delay_times, p_excited, n)

    plot = ax[2][0]
    plot.errorbar(t1_dates, t1s, t1_errs, fmt='o')
    plot.tick_params(axis='x', labelrotation=45)
    plot.set_ylabel('t1_ge [us]')

    ##### high gain qspec #####
    hgqspec = qspec(data_dir, dataset, QubitIndex, expt_name="high_gain_qspec_ge")
    hgqspec_dates, n, hgqspec_probe_freqs, I, Q = hgqspec.load_all()
    # qspec_ge.get_qspec_freq_in_round(qspec_probe_freqs, I, Q, 0, n, plot=True)
    plot = ax[0][1]
    cbar = plt.colorbar(plot.pcolormesh(hgqspec_dates, hgqspec_probe_freqs , np.transpose(I), #np.transpose(np.sqrt(np.square(I)+np.square(Q))),
                                        shading="nearest", cmap="viridis"), ax=plot)
    cbar.set_label("I [a.u.]")
    plot.set_ylabel('qubit probe frequency [MHz]')
    plot.set_title('high gain qspec_ge')

    ##### res stark spec ######
    rstark = resstarkspec(data_dir, dataset, QubitIndex, res_stark_constant[QubitIndex], theta, threshold)
    rstark_dates, n, gains, steps, reps, I_shots, Q_shots = rstark.load_all()
    p_excited = rstark.process_shots(I_shots, Q_shots, n, steps)
    #rstark.plot_shots(I_shots, Q_shots, gains, n, idx=25)
    #rstark.get_p_excited_in_round(gains, p_excited, n, 0, plot = True)
    rstark_freqs = rstark.gain2freq(gains)


    plot = ax[1][1]
    cbar = plt.colorbar(plot.pcolormesh(rstark_dates, rstark_freqs, np.transpose(p_excited),
                                        shading="nearest", cmap="viridis"), ax=plot)
    cbar.set_label("P(MS=1)")
    plot.set_ylabel('stark shift [MHz]')
    plot.set_title('stark tone @ resonator frequency')

    #### stark spec at fixed detuning #####
    stark = starkspec(data_dir, dataset, QubitIndex, duffing_constant[QubitIndex], theta, threshold)
    stark_dates, n, gains, steps, reps, I_shots, Q_shots = stark.load_all()
    p_excited = stark.process_shots(I_shots, Q_shots, n, steps)
    #stark.get_p_excited_in_round(gains, p_excited, n, 0, plot = True)
    #stark_freqs = stark.gain2freq(gains)


    plot = ax[2][1]
    cbar = plt.colorbar(plot.pcolormesh(stark_dates, gains, np.transpose(p_excited),
                                        shading="nearest", cmap="viridis"), ax=plot)
    cbar.set_label("P(MS=1)")
    plot.set_ylabel('stark shift [MHz]')
    plot.set_title('stark tone @ fixed detuning from qubit frequency')
    plt.show()

if analysis_flags['slice']:
    fig, ax = plt.subplots(2, 2, layout='constrained')
    fig.suptitle(f'dataset {dataset} qubit {QubitIndex + 1} round {selected_round + 1}')

    ##### qspec data ####
    qspec_ge = qspec(data_dir, dataset, QubitIndex)
    qspec_dates, n, qspec_probe_freqs, qspec_I, qspec_Q = qspec_ge.load_all()
    qspec_mag = np.sqrt(np.square(qspec_I)+np.square(qspec_Q))
    qfreq, qfreq_err, fwhm, qspec_fit = qspec_ge.get_qspec_freq_in_round(qspec_probe_freqs, qspec_I, qspec_Q, selected_round, n, plot=False)

    ##### high gain qspec #####
    hgqspec = qspec(data_dir, dataset, QubitIndex, expt_name="high_gain_qspec_ge")
    hgqspec_dates, n, hgqspec_probe_freqs, hgqspec_I, hgqspec_Q = hgqspec.load_all()
    hgqspec_mag = np.sqrt(np.square(hgqspec_I) + np.square(hgqspec_Q))

    plot = ax[0][0]
    plot.plot(qspec_probe_freqs, qspec_mag[selected_round], label='gain=0.15')
    plot.plot(qspec_probe_freqs, qspec_fit)
    plot.plot(hgqspec_probe_freqs, hgqspec_mag[selected_round], label='gain=1.0')
    plot.legend()
    plot.set_xlabel('qubit probe frequency [MHz]')
    plot.set_ylabel('I,Q magnitude [a.u.]')

    ##### t1 data #####
    t1_ge = t1(data_dir, dataset, QubitIndex, theta, threshold)
    t1_dates, n, delay_times, steps, reps, I_shots, Q_shots = t1_ge.load_all()
    # t1_ge.plot_shots(I_shots, Q_shots, delay_times, n, idx=25)
    p_excited = t1_ge.process_shots(I_shots, Q_shots, n, steps)
    q1_fit_exponential, T1_err, T1_est = t1_ge.get_t1_in_round(delay_times, p_excited, n, selected_round, plot = False)

    plot = ax[0][1]
    plot.plot(delay_times, p_excited[selected_round], label='data')
    plot.plot(delay_times, q1_fit_exponential, label='exponential')
    plot.set_title(f'T1 = {T1_est:.2f} +/- {T1_err:.2f} us')
    plot.set_ylabel('P(e)')
    plot.set_xlabel('delay time [us]')

    # ##### res stark spec ######
    rstark = resstarkspec(data_dir, dataset, QubitIndex, res_stark_constant[QubitIndex], theta, threshold)
    rstark_dates, n, rstark_gains, steps, reps, rstark_I_shots, rstark_Q_shots = rstark.load_all()
    rstark_p_excited = rstark.process_shots(I_shots, Q_shots, n, steps)
    # rstark.plot_shots(I_shots, Q_shots, gains, n, idx=25)
    rstark_p_excited_in_round = rstark.get_p_excited_in_round(rstark_gains, rstark_p_excited, n, selected_round, plot = False)
    rstark_freqs = rstark.gain2freq(rstark_gains)

    #### stark spec at fixed detuning #####
    stark = starkspec(data_dir, dataset, QubitIndex, duffing_constant[QubitIndex], theta, threshold)
    stark_dates, n, gains, steps, reps, I_shots, Q_shots = stark.load_all()
    starkp_excited = stark.process_shots(I_shots, Q_shots, n, steps)
    starkp_excited_in_round = stark.get_p_excited_in_round(gains, starkp_excited, n, selected_round, plot = False)
    # stark_freqs = stark.gain2freq(gains)

    plot = ax[1][0]
    plot.plot(rstark_freqs, rstark_p_excited_in_round, label='stark tone @ resonator frequency')
    #plot.plot(gains * 10, starkp_excited_in_round, label='stark tone @ fixed detuning')
    plot.set_ylabel('stark shift [MHz]')
    plot.set_title('stark tone @ resonator frequency')

    plt.show()




